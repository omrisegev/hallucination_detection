"""DeltaBench official section F1 for the arms that need no new telemetry (CPU, stored scores only):
method of record s0 (TELE_N16 standalone Qwen2.5-Math-7B-Instruct ext cell = R0 + L-SML on 16 channels) as GAP flags and
under mu-sigma (risk-high: s0 > mu + sigma); PRMs (PRM mode) under mu-sigma, plus the fixed paper threshold for the 7B.
Targets: official script union (unuseful + error) and error-only. Group bootstrap by question (2,000 draws) for C-free
contrasts is done in the final evaluation; here point values only."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import deltabench_official as O
SC, JSONL, OUT, HDR = sys.argv[1:5]
sys.path.insert(0, HDR + '/results/ssl_one_experiment_v1'); import ssl_lib as L  # GAP of the method of record
T = O.official_targets(JSONL); TE = O.official_targets(JSONL, error_only=True)
res = {}
def ev(flags, off, u):
    return {'union': O.summarize(O.per_sample(O.sets_from_flags(flags, off), [T[x] for x in u])),
            'error_only': O.summarize(O.per_sample(O.sets_from_flags(flags, off), [TE[x] for x in u]))}
S = np.load(Path(SC) / 'SCORES_sa_qwen_prm_deltabench_ext.npz'); u = S['uids'].astype(str); off = np.r_[0, np.cumsum(S['nonempty_lengths'])]
s0 = S['score__TELE_N16'].astype(float)
res['MoR_s0_GAP'] = ev(L.gap_flags(s0, off), off, u)
res['MoR_s0_musigma'] = ev(O.musigma_flags(s0, True), off, u)
res['MoR_s0_hit1'] = O.hit_at_1(s0, True, off, [T[x] for x in u])
for prm, cell in (('qwen_prm_7b', 'qwen_prm_deltabench_ext'), ('qwen_prm_72b', 'qwen_prm_72b_deltabench_ext'), ('pure_prm', 'pure_prm_deltabench_ext'), ('rlhflow_prm', 'rlhflow_prm_deltabench')):
    P = np.load(Path(SC) / f'SCORES_prm_{cell}.npz'); assert (P['uids'].astype(str) == u).all()
    r = 1. - P['score__PRMPROB'].astype(float)
    res[f'PRM_{prm}_musigma'] = ev(O.musigma_flags(r, False), off, u)
    res[f'PRM_{prm}_GAP'] = ev(L.gap_flags(-r, off), off, u)
    if prm in ('qwen_prm_7b', 'qwen_prm_72b'):
        res[f'PRM_{prm}_paper_fixed_threshold'] = ev(r < {'qwen_prm_7b': .2168, 'qwen_prm_72b': .2119}[prm], off, u)
json.dump(res, open(OUT, 'w'), indent=1)
for k, v in res.items():
    print(k, v if not isinstance(v, dict) else {t: round(m['f1_macro'], 2) for t, m in v.items()})
