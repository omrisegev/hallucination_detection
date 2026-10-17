"""CPU smoke for cluster/run_prmbench_step_voters.py (no weights, no dataset download).

Checks, on PRMBench-shaped fixture rows:
  1. uprm_judge end-to-end with a tiny random Qwen2 model and a real Qwen2.5 tokenizer:
     score_candidates runs, per-step P(err) recovery equals the module's own log p- values,
     prediction == localize_first_error.
  2. critic path: the ProcessBench prompt is built from PRMBench fields, the boxed index parses.
  3. driver lifecycle: shard split, resume skips done rows, SIGTERM -> checkpoint + exit 85,
     gold first_error is 0-based.

    PRM_SMOKE_TOKENIZER=/private/tmp/prm-tokenizers/Qwen__Qwen2.5-Math-PRM-7B \
        .venv/bin/python scripts/smoke_prmbench_step_voters.py
"""
import math
import os
import pickle
import sys
import tempfile

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'cluster')]
import run_prmbench_step_voters as D  # noqa: E402
from spectral_utils import uprm_baseline as U  # noqa: E402

ROWS = [
    dict(idx='confidence_0', source_idx=0, classification='confidence', category='Sensitivity',
         question='What is 2+3*4?', steps=['Multiply first: 3*4 = 12.', 'Add: 2 + 12 = 15.',
                                            'So the answer is 15.'], error_steps=[2]),
    dict(idx='correct_1', source_idx=1, classification='correct', category='correct',
         question='Solve x+1=3.', steps=['Subtract 1 from both sides: x = 2.', 'Answer: 2.'],
         error_steps=[]),
    dict(idx='redundency_2', source_idx=2, classification='redundency', category='Simplicity',
         question='Compute 10/2.', steps=['Step 1: 10/2 = 5.', 'Check: 5*2 = 10.', '', 'Answer 5.'],
         error_steps=[1, 4]),
]


def tiny_model(tok):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    m = Qwen2ForCausalLM(cfg).eval()
    return m


def check_uprm(tok, mdl):

    for row in ROWS:
        scores = U.score_candidates(mdl, tok, row['question'], row['steps'])
        rec = D.score_row('uprm_judge', mdl, tok, row, None)
        assert rec['failed'] is None and rec['recovery_error'] is None, rec
        assert rec['prediction'] == U.localize_first_error(scores)
        # recompute S(j) from the recovered per-step values and require exact agreement
        cum, rebuilt = 0.0, {}
        for j, lpn in enumerate(rec['step_logp_err'], 1):
            rebuilt[j] = lpn + cum
            cum += math.log1p(-math.exp(lpn))
        rebuilt[len(row['steps']) + 1] = cum
        assert all(abs(rebuilt[k] - scores[k]) < 1e-6 for k in scores), (rebuilt, scores)
        assert all(lp <= 0 for lp in rec['step_logp_err'])
    print('uprm_judge: OK (scores, per-step recovery, prediction) on', len(ROWS), 'rows')


def check_critic():
    from spectral_utils.processbench import critic_prompt, extract_critic_prediction
    p = critic_prompt({'problem': ROWS[0]['question'], 'steps': ROWS[0]['steps']})
    assert ROWS[0]['question'] in p and '<paragraph_2>' in p and '<paragraph_3>' not in p
    assert extract_critic_prediction('... the earliest error is \\boxed{1}') == 1
    assert extract_critic_prediction('all fine \\boxed{-1}') == -1

    def fake_generate(mdl, tok, msg, **kw):
        assert kw['temperature'] == 0.0 and kw['system_message'] is None and kw['max_new_tokens'] == 8192
        return dict(full_text='\\boxed{1}', token_entropies=[0.1], gen_token_ids=[1, 2])

    import spectral_utils
    real = spectral_utils.generate_full
    spectral_utils.generate_full = fake_generate
    try:
        cfg = type('C', (), dict(max_new=8192, logprob_top_k=50))
        rec = D.score_row('critic', None, None, ROWS[0], cfg)
    finally:
        spectral_utils.generate_full = real
    assert rec['prediction'] == 1 and rec['truncated'] is False
    print('critic: OK (prompt from PRMBench fields, parsing, greedy/8192/no system)')


def check_lifecycle(tok, mdl):
    D.load_prmbench = lambda **kw: ([dict(r) for r in ROWS], {'fixture': True})
    import spectral_utils
    spectral_utils.load_model = lambda *a, **k: (mdl, tok)
    with tempfile.TemporaryDirectory() as out:
        def main(*extra):
            sys.argv = ['x', '--kind', 'uprm_judge', '--model', 'tiny', '--out', out, *extra]
            D.STOP['flag'] = False
            try:
                D.main()
                return 0
            except SystemExit as e:
                return e.code

        # SIGTERM before the first row -> checkpoint and exit 85
        real_run = D.run
        D.run = lambda *a: (D.STOP.__setitem__('flag', True), real_run(*a))[1]
        assert main('--n-shards', '2', '--shard', '0') == D.EXIT_INCOMPLETE
        D.run = real_run
        # resume completes shard 0 = rows 0 and 2
        assert main('--n-shards', '2', '--shard', '0') == 0
        c = pickle.load(open(os.path.join(out, 'prmbench_uprm_judge_shard0of2.pkl'), 'rb'))
        assert sorted(c) == ['confidence_0', 'redundency_2'], sorted(c)
        assert c['confidence_0']['first_error'] == 1 and c['redundency_2']['first_error'] == 0
        assert main() == 0
        c = pickle.load(open(os.path.join(out, 'prmbench_uprm_judge.pkl'), 'rb'))
        assert c['correct_1']['first_error'] == -1 and len(c) == 3
        assert os.path.exists(os.path.join(out, 'manifest_uprm_judge.json'))
    print('lifecycle: OK (SIGTERM exit 85, resume, sharding, 0-based gold, manifest)')


def main():
    tok = AutoTokenizer.from_pretrained(os.environ.get(
        'PRM_SMOKE_TOKENIZER', '/private/tmp/prm-tokenizers/Qwen__Qwen2.5-Math-PRM-7B'))
    mdl = tiny_model(tok)
    check_uprm(tok, mdl)
    check_critic()
    check_lifecycle(tok, mdl)
    print('ALL SMOKE CHECKS PASSED')


if __name__ == '__main__':
    main()
