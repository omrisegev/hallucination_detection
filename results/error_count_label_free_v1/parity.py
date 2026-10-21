"""Parity gate: reproduce published numbers exactly before any new result.
Telemetry mask1 N16 Q80: docs/reviews/thinkfix_rescore_review_20261011/evidence/A/metrics.csv (q80_gt_macro_f1).
Boosting Qwen-7B: results/prm_boosting_bank_sweep_v1/SWEEP_RESULTS_qwen_prm_7b.csv (Q80 / ORACLE rows).
"""
import csv, json
import numpy as np
from common_ec import *  # noqa
import common as pc  # results/prmbench_paper_metric_parity_v1/common.py

meta = load_meta()
bk = load_tele_bank('N16'); off = bk['offsets'].astype(np.int64); ids = bk['idx']
y, cnt, nonc, cls, cat = labels(ids, off, meta)
rep = {r['scorer']: float(r['q80_gt_macro_f1']) for r in csv.DictReader(
    open(ROOT / 'docs/reviews/thinkfix_rescore_review_20261011/evidence/A/metrics.csv'))}
rows = []
for s in ['LSML_CONT', 'ALL_EQ_ORIENTED', 'MEAN_MOM_ga', 'IU_PCR']:
    v = bk['S__' + s]
    ps = prmscore(y, v > np.quantile(v, .8), np.repeat(nonc, np.diff(off)))
    rows.append(dict(line='tele_mask1_N16', arm=s, rule='Q80', local=ps / 100, reported=rep[s], diff=ps / 100 - rep[s]))

# boosting: fresh refit from exported channels AND the cached ARM_SCORES; both must match the sweep
P = pc.load_population()
assert np.array_equal(P['off'], off) and np.array_equal(P['ids'], ids), 'population order differs'
assert np.array_equal(P['y'], y), 'label mismatch between loaders'
S_fresh, _ = pc.build_scores(P)
S_cache = pc.get_scores(P)
sweep = {(r['arm'], r['rule']): float(r['prmscore']) for r in csv.DictReader(open(pc.SWEEP_CSV))}
for a in ['PRM_native', 'CHAIN_N16', 'LSML_CONT']:
    d = float(np.max(np.abs(S_fresh[a] - S_cache[a])))
    f = pc.flags_for(a, S_fresh[a], off)
    ps = prmscore(y, f, np.repeat(nonc, np.diff(off)))
    rows.append(dict(line='boost_qwen7b', arm=a, rule='Q80' if a != 'PRM_native' else 'native', local=ps,
                     reported=sweep[(a, 'Q80')], diff=ps - sweep[(a, 'Q80')], refit_vs_cache_maxabs=d))
    if (a, 'ORACLE') in sweep:
        fo = topk_flags(S_fresh[a], off, cnt)
        po = prmscore(y, fo, np.repeat(nonc, np.diff(off)))
        rows.append(dict(line='boost_qwen7b', arm=a, rule='ORACLE', local=po, reported=sweep[(a, 'ORACLE')],
                         diff=po - sweep[(a, 'ORACLE')]))
np.savez_compressed(OUT / 'BOOST_SCORES.npz', **S_fresh)
for r in rows:
    print(r)
with open(OUT / 'PARITY.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
assert all(abs(r['diff']) < 1e-9 for r in rows), 'PARITY FAIL'
print('PARITY PASS')
