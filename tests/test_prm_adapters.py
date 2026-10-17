"""CPU-only real-tokenizer snapshots and tiny final-norm/head checks.

Set PRM_TOKENIZER_DIR to folders named HF_id.replace('/', '__'), or pre-cache
these tokenizers with Hugging Face. No model weights are loaded by this suite.
"""
import json
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
from spectral_utils.prm_adapters import ADAPTERS, PRMAdapter, get_adapter, remove_step_prefix

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = json.loads((ROOT/'tests/fixtures/prm_template_snapshots.json').read_text())


class Templates(unittest.TestCase):
    def test_all_real_tokenizers(self):
        for name, cls in ADAPTERS.items():
            directory = os.environ.get('PRM_TOKENIZER_DIR')
            # PURE's legacy id redirects to this canonical tokenizer directory.
            mid = cls.model_id.replace('Qwen2.5-Math-7B-PRM800K','PURE-PRM-7B')
            location = str(Path(directory)/mid.replace('/','__')) if directory else cls.model_id
            tok = AutoTokenizer.from_pretrained(location, local_files_only=True)
            adapter = cls()
            for i, fixture in enumerate(SNAPSHOTS['fixtures']):
                with self.subTest(adapter=name, fixture=i):
                    batch = adapter.build(tok, fixture['question'], fixture['steps'])
                    snapshot = SNAPSHOTS['snapshots'][name][i]
                    self.assertEqual(batch['input_ids'], snapshot['input_ids'])
                    self.assertEqual(tok.decode(batch['input_ids']), snapshot['decoded_input'])
                    self.assertEqual(len(batch['score_positions']), len(fixture['steps']))
                    self.assertEqual(batch['n_step_text_mismatch'], 0)
                    used = [p for seg in batch['step_token_positions'] for p in seg]
                    self.assertEqual(len(used), len(set(used)))
                    self.assertFalse(set(used)&set(batch['score_positions']))
                    for positions, cleaned in zip(batch['step_token_positions'], batch['cleaned_steps']):
                        decoded = tok.decode([batch['input_ids'][p] for p in positions])
                        # BPE may merge the immediately preceding template space
                        # with the first content token. No non-whitespace template
                        # text is admitted by _mapped; decode equality is modulo it.
                        self.assertEqual(decoded.strip(), cleaned.strip())
                        self.assertNotIn('ки', decoded)
                        self.assertNotIn('<extra_0>', decoded)
                    if name.startswith('skywork'):
                        self.assertNotIn(batch['terminal_score_position'], used)
                        self.assertIn('blocked_reason', batch)
                    if name == 'rlhflow_prm':
                        self.assertFalse(any(batch['input_ids'][p] == 10 for p in used))
                    if name == 'math_shepherd':
                        self.assertFalse(any('Step' in tok.decode([batch['input_ids'][p] for p in seg])
                                             for seg in batch['step_token_positions']))

    def test_literal_upstream_cleanup(self):
        self.assertEqual(remove_step_prefix(' Step 3: x '), 'Step 3: x')
        self.assertEqual(remove_step_prefix('Step 3. x'), 'x')
        self.assertEqual(remove_step_prefix('3. x'), 'x')

    def test_blocked_models_fail_before_loading(self):
        for name in ['skywork_prm', 'skywork_prm_1_5b']:
            with self.assertRaisesRegex(ValueError, 'n\\+1'):
                get_adapter(name).load()
        with self.assertRaisesRegex(ValueError, 'blocked'):
            get_adapter('reasoneval')


class Heads(unittest.TestCase):
    def test_projection_uses_final_normalized_hidden(self):
        torch.manual_seed(16)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        backbone = Qwen2Model(config).eval()
        observed = {}
        handle = backbone.norm.register_forward_hook(lambda _, inputs, output:
                    observed.update(pre=inputs[0].detach(), final=output.detach()))
        with torch.no_grad(): hidden = backbone(torch.tensor([[1, 2, 3, 4]]), use_cache=False)[0]
        handle.remove()
        self.assertTrue(torch.equal(hidden, observed['final']))
        self.assertFalse(torch.equal(hidden, observed['pre']))
        W = torch.randn(32, 16)
        for source in ['own_unused', 'borrowed_base:tiny-random-base', 'own_used']:
            with self.subTest(source=source):
                adapter = PRMAdapter(); adapter.bind_head(W, source)
                projected = adapter.vocab_logits(hidden, [0, 2])
                torch.testing.assert_close(projected, observed['final'][0, [0, 2]] @ W.T, rtol=0, atol=0)
                self.assertEqual(adapter.head_meta['lm_head_source'], source)
                self.assertTrue(adapter.head_meta['hidden_after_final_norm'])

    def test_borrowed_tied_embedding_inventory_from_tiny_checkpoint(self):
        import tempfile
        from transformers import Qwen2ForCausalLM, Qwen2ForTokenClassification
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             tie_word_embeddings=True)
        base = Qwen2ForCausalLM(config).eval()
        prm = Qwen2ForTokenClassification(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            base.save_pretrained(directory)
            adapter = PRMAdapter(); adapter.name = 'tiny'; adapter.base_id = directory
            adapter.head_source = 'borrowed_base:'+directory; adapter.backbone = prm.model
            adapter._inventory(prm, None)
            with torch.no_grad(): h = prm.model(torch.tensor([[1,2,3]]),use_cache=False)[0]
            torch.testing.assert_close(adapter.vocab_logits(h,[0,1]),
                                        h[0,[0,1]] @ base.model.embed_tokens.weight.detach().T, rtol=0,atol=0)
            self.assertEqual(adapter.head_meta['lm_head_vs_base_max_abs_diff'],0.)

    def test_reward_rules_match_official_heads(self):
        from types import SimpleNamespace
        torch.manual_seed(4)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        from transformers import Qwen2ForCausalLM
        model = Qwen2ForCausalLM(config).eval()
        x = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            hidden = model.model(x, use_cache=False)[0]
            logits = model(x, use_cache=False).logits
        batch = dict(score_positions=[1, 3], candidate_tokens=[10, 11])
        for name in ['rlhflow_prm', 'math_shepherd']:
            adapter = get_adapter(name); adapter.bind_head(model.lm_head.weight.detach(), 'own_used')
            reward = adapter.rewards(dict(hidden=hidden), batch)
            expected = (logits[0, [1,3]].softmax(-1)[:, 10] if name == 'rlhflow_prm'
                        else logits[0, [1,3]][:, [10,11]].softmax(-1)[:, 0])
            np.testing.assert_allclose(reward, expected.detach().numpy(), atol=1e-7, rtol=0)
        scores = torch.tensor([[[0., 0.], [1., 2.], [3., 4.], [0., 3.]]])
        for name in ['qwen_prm', 'pure_prm']:
            adapter = get_adapter(name)
            reward = adapter.rewards(dict(logits=scores), batch)
            p = scores[0, [1,3]].softmax(-1)
            expected = p[:,1] if name == 'qwen_prm' else ((p[:,1]-p[:,0]).float()+1)/2
            np.testing.assert_array_equal(reward, expected.numpy())

    def test_native_thresholds(self):
        self.assertEqual(get_adapter('qwen_prm').native_labels([.5,.5001]), [0,1])
        self.assertEqual(get_adapter('pure_prm').native_labels([.5,.5001]), [0,1])
        self.assertEqual(get_adapter('skywork_prm').native_labels([.9,.84,.9]), [1,0,1])


if __name__ == '__main__':
    unittest.main()
