"""Pinned PRMBench PRM protocols and same-backbone vocabulary projections.

No base-model forward is ever performed. Borrowed weights are read from the local
HF cache and projected onto the PRM backbone's final-normalized output.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from .prm_scorer import PRM_SYSTEM_PROMPT, load_prm_model, make_step_rewards

OFFICIAL_QWEN_SYSTEM_PROMPT = 'Please reason step by step, and put your final answer within \x08oxed{}.'

OFFICIAL_WRAPPER_COMMIT = 'a1ae4eab803e14c4c6316bf89d1679b0486faa8c'
SKYWORK_COMMIT = '719b56b17447405e0f10e6c0360a581cf4ffa9c1'


def remove_step_prefix(text):
    # Verbatim mr_eval/utils/model_utils.py. A colon prefix is intentionally retained.
    text = text.strip()
    return re.sub(r"^(Step\s*\d+\.\s*|\d+\.\s*)", "", text, flags=re.IGNORECASE)


def load_head_weight(model_id, name='lm_head.weight'):
    """Read one existing tensor, never instantiate/forward a standalone base LM."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    directory = Path(model_id)
    if not directory.is_dir():
        directory = Path(snapshot_download(model_id, local_files_only=True))
    index = directory / 'model.safetensors.index.json'
    shard = json.loads(index.read_text())['weight_map'][name] if index.exists() else 'model.safetensors'
    with safe_open(str(directory / shard), framework='pt') as f:
        return f.get_tensor(name)


def _mapped(tok, text, ranges, *, add_special_tokens=True):
    """Select only tokens overlapping content; never separators or injected tags.

    Fast-tokenizer offsets are measured on the official input, rather than by
    independently tokenizing steps (which changes BPE merges at boundaries).
    """
    encoded = tok(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    ids = list(encoded['input_ids']); offsets = encoded['offset_mapping']
    for lo, hi in ranges:
        if hi <= lo: continue
        for a, b in offsets:
            if b <= a or a >= hi or b <= lo: continue
            if (a < lo and text[a:lo].strip()) or (b > hi and text[hi:b].strip()):
                raise ValueError('content token merges non-whitespace template text')
    positions = [[i for i, (a, b) in enumerate(offsets) if b > a and a < hi and b > lo]
                 if hi > lo else [] for lo, hi in ranges]
    return ids, positions


def _batch(tok, text, ids, positions, scores, cleaned, **extra):
    if len(scores) != len(cleaned) or len(positions) != len(cleaned):
        raise ValueError(f'protocol alignment: {len(scores)} scores for {len(cleaned)} steps')
    used = [p for seg in positions for p in seg]
    if len(used) != len(set(used)) or any(p <= 0 for p in used):
        raise ValueError('step tokens overlap or lack a causal predecessor')
    if set(used) & set(scores):
        raise ValueError('step tokens overlap reward/template positions')
    decoded = [tok.decode([ids[p] for p in seg]) for seg in positions]
    return dict(input_ids=ids, step_token_positions=positions, score_positions=scores,
                input_text=text, cleaned_steps=cleaned,
                n_step_text_mismatch=sum((a != b if extra.get('protocol') == 'legacy_raw_steps' else a.strip() != b.strip())
                                         for a, b in zip(decoded, cleaned)),
                n_boundary_whitespace=sum(a != b and a.strip() == b.strip() for a, b in zip(decoded, cleaned)), **extra)


class PRMAdapter:
    model_id = None
    base_id = None
    head_source = 'own_used'
    leaderboard = None
    wrapper = None

    def bind_head(self, weight, source=None):
        self.head_weight = weight
        self.head_meta = dict(lm_head_source=source or self.head_source,
                              lm_head_vs_base_max_abs_diff=None, hidden_after_final_norm=True)

    def vocab_logits(self, hidden, positions):
        import torch
        h = hidden[0] if hidden.ndim == 3 else hidden
        p = torch.as_tensor(positions, device=h.device, dtype=torch.long)
        # Project on the head's device (important for device_map='auto').
        return h[p].to(self.head_weight.device, dtype=self.head_weight.dtype) @ self.head_weight.T

    def native_labels(self, rewards):
        return [int(v > .5) for v in rewards]

    def _inventory(self, model, tok):
        import torch
        source = self.head_source
        if source.startswith('borrowed_base:'):
            base_cfg = json.loads(_local_config(self.base_id).read_text())
            name = 'model.embed_tokens.weight' if base_cfg.get('tie_word_embeddings') else 'lm_head.weight'
            W = load_head_weight(self.base_id, name)
        elif source == 'own_unused' and self.wrapper == 'qwen_prm.py':
            W = load_head_weight(self.loaded_id)
        else:
            parent = getattr(model, 'pretrained_model', model)
            W = parent.get_output_embeddings().weight.detach()
        device = next(self.backbone.parameters()).device
        self.bind_head(W.to(device, dtype=torch.bfloat16 if self.wrapper == 'qwen_prm.py' else W.dtype), source)
        try:
            base_cfg = json.loads(_local_config(self.base_id).read_text())
            name = 'model.embed_tokens.weight' if base_cfg.get('tie_word_embeddings') else 'lm_head.weight'
            base = load_head_weight(self.base_id, name).to(self.head_weight.device, dtype=self.head_weight.dtype)
            if base.shape != self.head_weight.shape: raise ValueError('base vocabulary/head shape mismatch')
            self.head_meta['lm_head_vs_base_max_abs_diff'] = max(
                float((self.head_weight[i:i+1024].float() - base[i:i+1024].float()).abs().max())
                for i in range(0, len(base), 1024))
            del base
        except Exception as exc:
            self.head_meta['base_head_compare_error'] = str(exc)[:300]
        if self.head_weight.shape != (model.config.vocab_size, model.config.hidden_size):
            raise ValueError('vocabulary head dimensions do not match the PRM')
        self.head_meta.update(adapter=self.name, official_wrapper_commit=OFFICIAL_WRAPPER_COMMIT,
                              base_model=self.base_id, model_revision=getattr(model.config, '_commit_hash', None),
                              scored_positions='step content only; logits at absolute token position minus one')
        self.model = model
        return model, tok, self.head_meta

    def forward(self, x, batch):
        """Exactly one backbone forward; output[0] is after its final norm."""
        h = self.backbone(input_ids=x, use_cache=False)[0]
        scores = self.classifier(h) if hasattr(self, 'classifier') else None
        return h, dict(hidden=h, logits=scores)


def _local_config(model_id):
    from huggingface_hub import hf_hub_download
    p = Path(model_id) / 'config.json'
    return p if p.exists() else Path(hf_hub_download(model_id, 'config.json', local_files_only=True))


class QwenPRM(PRMAdapter):
    name = 'qwen_prm'
    model_id = 'Qwen/Qwen2.5-Math-PRM-7B'
    base_id = 'Qwen/Qwen2.5-Math-7B-Instruct'
    head_source = 'own_unused'
    wrapper = 'qwen_prm.py'
    leaderboard = 65.5
    clean = False  # historical driver T1: retain raw steps byte-for-byte

    def build(self, tok, question, steps):
        cleaned = [remove_step_prefix(s) for s in steps] if self.clean else list(steps)
        response = '<extra_0>'.join(cleaned) + '<extra_0>'
        messages = [dict(role='system', content=OFFICIAL_QWEN_SYSTEM_PROMPT if self.clean else PRM_SYSTEM_PROMPT), dict(role='user', content=question),
                    dict(role='assistant', content=response)]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        cut = text.rindex(response); ids = tok.encode(text); pre = tok.encode(text[:cut])
        if ids[:len(pre)] != pre: raise ValueError('prefix tokenization differs inside the full conversation')
        sep = tok.encode('<extra_0>')[0]
        seps = [p for p in range(len(pre), len(ids)) if ids[p] == sep]
        start = len(pre); positions = []
        for p in seps:
            positions.append(list(range(start, p))); start = p + 1
        return _batch(tok, text, ids, positions, seps, cleaned,
                      protocol='legacy_raw_steps' if not self.clean else 'official_remove_step_prefix')

    def load(self, model_id=None):
        self.loaded_id = model_id or self.model_id
        if '72B' in self.loaded_id: self.base_id = 'Qwen/Qwen2.5-Math-72B-Instruct'
        model, tok = load_prm_model(self.loaded_id)
        self.backbone, self.classifier = model.model, model.score
        return self._inventory(model, tok)

    def rewards(self, outputs, batch):
        import torch
        mask = torch.zeros(outputs['logits'].shape[:2], dtype=torch.bool, device=outputs['logits'].device)
        mask[0, batch['score_positions']] = True
        return make_step_rewards(outputs['logits'], mask)[0]


class QwenPRM72B(QwenPRM):
    name = 'qwen_prm_72b'
    model_id = 'Qwen/Qwen2.5-Math-PRM-72B'
    base_id = 'Qwen/Qwen2.5-Math-72B-Instruct'
    leaderboard = 68.2
    clean = True


class SkyworkPRM(PRMAdapter):
    blocked_reason = 'Upstream prepare_input produces n+1 rewards for n steps; blocked by user pending protocol clarification'
    name = 'skywork_prm'
    model_id = 'Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B'
    base_id = 'Qwen/Qwen2.5-Math-7B-Instruct'
    head_source = 'own_unused'
    wrapper = 'skywork_prm.py'
    leaderboard = 65.1

    def build(self, tok, question, steps):
        from third_party.skywork_prm.model_utils.io_utils import prepare_input
        cleaned = [remove_step_prefix(s).strip() for s in steps]
        response = ''.join(f'Step {i+1}: {s} ки\n' for i, s in enumerate(cleaned))
        ids, _, flags = prepare_input(question, response, tok, 'ки')
        scores = [i for i, f in enumerate(flags) if f]
        # Upstream includes one terminal sentinel reward. It is not a reasoning step.
        if len(scores) != len(steps) + 1: raise ValueError('unexpected Skywork terminal sentinel protocol')
        # Preserve upstream ids exactly; select content within independently encoded segments.
        prompt = tok.bos_token + question + '\n'; pos = len(tok.encode(prompt)); positions = []
        for i, segment in enumerate(response.split('ки')[:-1]):
            prefix = ('' if i == 0 else '\n') + f'Step {i+1}: '
            lo = len(prefix); hi = lo + len(cleaned[i])
            segment_ids, local = _mapped(tok, segment, [(lo, hi)])
            positions.append([pos + p for p in local[0]])
            if ids[pos:pos+len(segment_ids)] != segment_ids: raise ValueError('Skywork segment tokenization mismatch')
            pos += len(segment_ids) + 1
        return _batch(tok, prompt + response + 'ки', ids, positions, scores[:-1], cleaned, blocked_reason=self.blocked_reason,
                      terminal_score_position=scores[-1], reward_flags=flags)

    def load(self, model_id=None):
        raise ValueError(self.blocked_reason)

    def rewards(self, outputs, batch):
        raise ValueError(self.blocked_reason)

    def native_labels(self, rewards):
        changes = rewards[:1] + [rewards[i] - rewards[i-1] for i in range(1, len(rewards))]
        return [int(v > -.05) for v in changes]


class SkyworkPRM15B(SkyworkPRM):
    name = 'skywork_prm_1_5b'
    model_id = 'Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B'
    base_id = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
    leaderboard = 61.1


class CausalPRM(PRMAdapter):
    def load(self, model_id=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.loaded_id = model_id or self.model_id
        tok = AutoTokenizer.from_pretrained(self.loaded_id, trust_remote_code=True)
        # These official causal-LM wrappers leave dtype at the HF default (fp32).
        model, loading = AutoModelForCausalLM.from_pretrained(self.loaded_id, device_map='auto',
                                                             output_loading_info=True)
        model.eval()
        if loading.get('missing_keys') or loading.get('mismatched_keys'):
            raise ValueError('incomplete causal PRM checkpoint: '+str(loading))
        self.backbone = model.model
        return self._inventory(model, tok)


class RLHFlowPRM(CausalPRM):
    name = 'rlhflow_prm'
    model_id = 'RLHFlow/Llama3.1-8B-PRM-Mistral-Data'
    base_id = 'meta-llama/Llama-3.1-8B-Instruct'
    wrapper = 'llama3_1_8b_prm.py'
    leaderboard = 54.4
    pattern = [128006, 78191, 128007, 271, 10, 128009]

    def build(self, tok, question, steps):
        cleaned = [remove_step_prefix(s) for s in steps]; messages = []
        for i, s in enumerate(cleaned):
            messages += [dict(role='user', content=question + ' ' + s if i == 0 else s),
                         dict(role='assistant', content='+')]
        text = tok.apply_chat_template(messages, tokenize=False)
        ids = list(tok.apply_chat_template(messages, tokenize=True))
        # Mark content in the same chat template to locate ranges, including empty turns.
        marked = [dict(m) for m in messages]
        for i in range(len(steps)): marked[2*i]['content'] = f'PRM_CONTENT_{i}_END'
        skeleton = tok.apply_chat_template(marked, tokenize=False)
        ranges = []
        for i, s in enumerate(cleaned):
            marker = f'PRM_CONTENT_{i}_END'; at = skeleton.index(marker)
            content = question + ' ' + s if i == 0 else s
            # Llama's template strips user content.
            content = content.strip(); end = at + len(content)
            ranges.append((end-len(s), end) if s else (end, end))
            skeleton = skeleton.replace(marker, content, 1)
        if skeleton != text: raise ValueError('RLHFlow chat template range reconstruction mismatch')
        mapped_ids, positions = _mapped(tok, text, ranges, add_special_tokens=False)
        if mapped_ids != ids: raise ValueError('RLHFlow chat tokenization mismatch')
        scores = [i+3 for i in range(len(ids)-len(self.pattern)+1) if ids[i:i+len(self.pattern)] == self.pattern]
        return _batch(tok, text, ids, positions, scores, cleaned)

    def rewards(self, outputs, batch):
        logits = self.vocab_logits(outputs['hidden'], batch['score_positions'])
        # Upstream softmax on CPU across the FULL vocabulary, then token 10.
        return logits.cpu().float().softmax(-1)[:, 10].tolist()


class PurePRM(PRMAdapter):
    name = 'pure_prm'
    model_id = 'jinachris/Qwen2.5-Math-7B-PRM800K'  # resolves to PURE-PRM-7B, same revision
    base_id = 'Qwen/Qwen2.5-Math-7B'
    head_source = 'borrowed_base:Qwen/Qwen2.5-Math-7B'
    wrapper = 'pure_prm.py'
    leaderboard = 65.3

    def build(self, tok, question, steps):
        ids = tok.encode(question, add_special_tokens=False); positions = []; scores = []
        sep = tok.encode('\n\n', add_special_tokens=False)
        for s in steps:  # upstream PURE does NOT remove_step_prefix
            seg = tok.encode(s, add_special_tokens=False)
            positions.append(list(range(len(ids), len(ids)+len(seg))))
            ids += seg + sep; scores.append(len(ids)-1)
        text = question + ''.join(s+'\n\n' for s in steps)
        return _batch(tok, text, ids, positions, scores, list(steps))

    def load(self, model_id=None):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        import torch
        self.loaded_id = model_id or self.model_id
        # Fail before loading PRM weights if the required borrowed head is absent.
        _local_config(self.base_id); load_head_weight(self.base_id)
        tok = AutoTokenizer.from_pretrained(self.loaded_id, trust_remote_code=True)
        model, loading = AutoModelForTokenClassification.from_pretrained(self.loaded_id, torch_dtype=torch.bfloat16,
                                                               device_map='auto', trust_remote_code=True,
                                                               output_loading_info=True)
        model.eval()
        if loading.get('missing_keys') or loading.get('mismatched_keys'):
            raise ValueError('incomplete PURE checkpoint: '+str(loading))
        self.backbone, self.classifier = model.model, model.score
        return self._inventory(model, tok)

    def rewards(self, outputs, batch):
        p = outputs['logits'][0, batch['score_positions']].softmax(-1)
        r = p[:, 1] - p[:, 0]
        # Rescale in float32 so bf16 rounding cannot move the native r>0 boundary.
        return ((r.float() + 1) / 2).cpu().tolist()


class MathShepherdPRM(CausalPRM):
    name = 'math_shepherd'
    model_id = 'peiyi9979/math-shepherd-mistral-7b-prm'
    base_id = 'mistralai/Mistral-7B-v0.1'
    wrapper = 'math_shepherd.py'
    leaderboard = 47.0

    def build(self, tok, question, steps):
        text = question + ' '; ranges = []; cleaned = []
        for i, step in enumerate(steps):
            s = step.strip()
            if s.startswith('Step') or s.startswith('step'):
                match = re.match(r'(?:Step|step)\s*\d+[.:]\s*', s)
                prefix = s[:match.end()] if match else ''
                content = s[len(prefix):]
            else: prefix, content = f'Step {i+1}: ', s
            text += prefix; lo = len(text); text += content
            ranges.append((lo, len(text))); cleaned.append(content); text += ' ки\n'
        ids, positions = _mapped(tok, text, ranges)
        tag = tok.encode('ки')[-1]
        scores = [i for i, t in enumerate(ids) if t == tag]
        candidates = tok.encode('+ -')[1:]
        if len(candidates) != 2: raise ValueError('MathShepherd candidate protocol changed')
        return _batch(tok, text, ids, positions, scores, cleaned, candidate_tokens=candidates)

    def rewards(self, outputs, batch):
        logits = self.vocab_logits(outputs['hidden'], batch['score_positions'])
        return logits.float()[:, batch['candidate_tokens']].softmax(-1)[:, 0].cpu().tolist()


ADAPTERS = {a.name: a for a in [QwenPRM, QwenPRM72B, SkyworkPRM, SkyworkPRM15B,
                              RLHFlowPRM, PurePRM, MathShepherdPRM]}


def get_adapter(name):
    if name == 'reasoneval':
        raise ValueError('ReasonEval optional adapter blocked: borrowed WizardMath head and redundancy-head protocol not validated')
    return ADAPTERS[name]()
