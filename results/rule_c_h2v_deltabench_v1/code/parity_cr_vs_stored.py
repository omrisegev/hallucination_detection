"""Label-free parity: container re-run CR vs the stored C cell, 16 raw TELE16 step channels (hard top10 + 5 step).
    python parity_cr_vs_stored.py STEPS_DIR BENCH OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
d, bench, outp = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
A, B = np.load(d / f'STEPS_{bench}_CR.npz'), np.load(d / f'STEPS_{bench}_C.npz')
assert (A['uids'] == B['uids']).all() and (A['offsets'] == B['offsets']).all()
x = np.column_stack([A['stats'][:, :, 0], A['step5']]).astype(float); y = np.column_stack([B['stats'][:, :, 0], B['step5']]).astype(float)
names = [str(n) for n in A['tok_names']] + [str(n) for n in A['step_names']]
R = {}
for j, n in enumerate(names):
    m = np.isfinite(x[:, j]) & np.isfinite(y[:, j])
    R[n] = dict(spearman=float(spearmanr(x[m, j], y[m, j])[0]), max_abs_diff=float(np.abs(x[m, j] - y[m, j]).max()), exact_frac=float(np.mean(x[m, j] == y[m, j])))
out = dict(bench=bench, answers=int(len(A['uids'])), steps=int(len(x)), channels=R, spearman_min=min(v['spearman'] for v in R.values()))
json.dump(out, open(outp, 'w'), indent=1); print(bench, 'answers', out['answers'], 'steps', out['steps'], {k: round(v['spearman'], 4) for k, v in R.items()})
