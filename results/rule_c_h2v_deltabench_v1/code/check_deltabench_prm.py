"""Sanity check of deltabench_official.py with the stored PRM scores (PRM mode, ext cells) vs the paper numbers.
Paper (arXiv 2502.19361): mu-sigma macro-F1 Qwen2.5-Math-PRM-7B 29.22, 72B 26.38, RLHFlow-Deepseek 12.02; 5% quantile
7B 46.48, 72B 40.44; HitRate@1 7B 49.15, 72B 41.13. Our RLHFlow cell is RLHFlow-PRM-Mistral-8B (not Deepseek)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import deltabench_official as O
SC, GOLD, JSONL, OUT = sys.argv[1:5]
T = O.official_targets(JSONL)
TE = O.official_targets(JSONL, error_only=True)
G = {g['uid']: g for g in json.load(open(GOLD))}
res = {}
for prm, cell in (('qwen_prm_7b', 'qwen_prm_deltabench_ext'), ('qwen_prm_72b', 'qwen_prm_72b_deltabench_ext'),
                  ('pure_prm', 'pure_prm_deltabench_ext'), ('rlhflow_prm_mistral8b', 'rlhflow_prm_deltabench'),
                  ('math_shepherd', 'math_shepherd_deltabench')):
    S = np.load(Path(SC) / f'SCORES_prm_{cell}.npz'); u = S['uids'].astype(str); n = S['nonempty_lengths'].astype(int)
    assert S['nonempty'].all()
    off = np.r_[0, np.cumsum(n)]
    tg = [T[x] for x in u]
    our = [set((np.flatnonzero(~np.asarray(G[x]['correct'], bool)) + 1).tolist()) for x in u]
    agree = float(np.mean([set(a) == b for a, b in zip(tg, our)]))
    nsec_ok = float(np.mean([len(G[x]['correct']) == k for x, k in zip(u, n)]))
    r = 1. - S['score__PRMPROB'].astype(float)   # stored PRMPROB is risk-high (corr with native_valid -0.83): reward = 1 - PRMPROB
    y = np.concatenate([[j + 1 in set(t) for j in range(k)] for t, k in zip(tg, n)])
    orient = float(np.corrcoef(r, y)[0, 1])
    row = dict(n_answers=int(len(u)), n_sections=int(off[-1]), target_equals_our_gold_share=agree, nsections_match_share=nsec_ok,
               corr_reward_vs_error=orient)
    row['musigma'] = O.summarize(O.per_sample(O.sets_from_flags(O.musigma_flags(r, False), off), tg))
    row['q05'] = O.summarize(O.per_sample(O.sets_from_flags(O.quantile_flags(r, False), off), tg))
    row['hit_at_1'] = O.hit_at_1(r, False, off, tg)
    row['musigma_target_error_only'] = O.summarize(O.per_sample(O.sets_from_flags(O.musigma_flags(r, False), off), [TE[x] for x in u]))
    res[prm] = row
    print(prm, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if not isinstance(v, dict)}),
          'musigma F1', round(row['musigma']['f1_macro'], 2), 'q05 F1', round(row['q05']['f1_macro'], 2), 'n_scored', row['musigma']['n_scored'], 'musigma(err-only target)', round(row['musigma_target_error_only']['f1_macro'],2), row['musigma_target_error_only']['n_scored'])
res['paper'] = dict(musigma={'qwen_prm_7b': 29.22, 'qwen_prm_72b': 26.38, 'rlhflow_deepseek': 12.02}, q05={'qwen_prm_7b': 46.48, 'qwen_prm_72b': 40.44},
                    hit_at_1={'qwen_prm_7b': 49.15, 'qwen_prm_72b': 41.13})
json.dump(res, open(OUT, 'w'), indent=1)
