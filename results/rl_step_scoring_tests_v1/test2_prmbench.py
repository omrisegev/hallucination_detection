"""Test 2 (PRMBench): first-error localization on answers that contain >= 1 error step (Step-DPO use case: the
answer is known to be wrong, so every method names one step). Metric = exact first-error accuracy (%), over all
erroneous answers and per PRMBench category (id prefix). Inputs: cells/PRMB_<prm>.npz from export_prmbench_sa.py
(standalone base-model cells). Same readouts as test2_pb.py. Labels used only for scoring.  python -I test2_prmbench.py"""
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test2_pb import gap_first, argmax_pred, prm_native, segs  # noqa: E402  (no PB code runs on import? see guard)
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
B = 2000
rng = np.random.default_rng(0); rows = []; HITS = {}
for prm in PRMS:
    d = np.load(HERE / 'cells' / f'PRMB_{prm}.npz'); ids = d['ids']; off = d['off']; err = d['err']
    first = np.array([int(np.flatnonzero(err[a:b])[0]) if err[a:b].any() else -1 for a, b in segs(off)])
    E = np.flatnonzero(first >= 0)
    cat = np.array([str(u).split('|')[0].split('-')[0] for u in ids])
    arms = {'TELE16 standalone + GAP (earliest flag)': gap_first(d['arm__TELE_N16'], off),
            'TELE16 standalone, argmax step': argmax_pred(d['arm__TELE_N16'], off),
            'Always step 0': np.zeros(len(ids), int)}
    if 'arm__PRMPROB' in d.files:
        arms['PRM native (first p_err>.5, else argmax)'] = prm_native(d['arm__PRMPROB'], off)
    if 'arm__POS_ONLY' in d.files:
        arms['Position only + GAP (earliest flag)'] = gap_first(d['arm__POS_ONLY'], off)
    for name, pred in arms.items():
        hit = (pred == first).astype(float); HITS[name] = hit
        bs = [100 * hit[rng.choice(E, len(E))].mean() for _ in range(B)]
        r = dict(bench='prmbench', prm=prm, arm=name, n_err=len(E), first_error_acc=100 * hit[E].mean(),
                 lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
        for c in sorted(set(cat[E])):
            r[f'acc_{c}'] = 100 * hit[E][cat[E] == c].mean()
        rows.append(r); print(f'{prm:12s} {name:46s} {r["first_error_acc"]:6.2f} [{r["lo"]:.2f},{r["hi"]:.2f}] n={len(E)}', flush=True)
    ref = 'PRM native (first p_err>.5, else argmax)'
    if ref in HITS:
        dl = HITS['TELE16 standalone + GAP (earliest flag)'] - HITS[ref]
        bs = [100 * dl[rng.choice(E, len(E))].mean() for _ in range(B)]
        rows.append(dict(bench='prmbench', prm=prm, arm=f'DIFF TELE16+GAP minus {ref}', n_err=len(E),
                         first_error_acc=100 * dl[E].mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
        print(f'   diff vs PRM {rows[-1]["first_error_acc"]:+.2f} [{rows[-1]["lo"]:+.2f},{rows[-1]["hi"]:+.2f}]')
with open(HERE / 'TEST2_PRMBENCH.csv', 'w', newline='') as f:
    keys = list(dict.fromkeys(k for r in rows for k in r)); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
