"""Compare score arrays of a new SCORES npz against a frozen one on their shared keys (label-free).

    python parity_npz.py <new.npz> <frozen.npz> [--require KEY ...] [--alias NEW=FROZEN ...] [--out report.json]

Prints one line per shared key: max |new - frozen| (or EXACT for non-float arrays). Exit 1 if any --require key
is missing on either side or differs by more than 1e-9; other keys are reported only.
"""
import argparse, json, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('new'); ap.add_argument('frozen')
ap.add_argument('--require', nargs='*', default=[])
ap.add_argument('--alias', nargs='*', default=[])
ap.add_argument('--out', default=None)
A = ap.parse_args()
N, F = np.load(A.new), np.load(A.frozen)
alias = dict(a.split('=', 1) for a in A.alias)
pairs = {k: k for k in N.files if k in F.files}
pairs.update({k: v for k, v in alias.items() if k in N.files and v in F.files})
rep, bad = {}, []
for k, fk in sorted(pairs.items()):
    a, b = N[k], F[fk]
    if a.shape != b.shape:
        rep[k] = {'frozen_key': fk, 'status': f'SHAPE {a.shape} vs {b.shape}'}
    elif a.dtype.kind in 'fc' or b.dtype.kind in 'fc':
        d = float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.
        nan_ok = bool(np.array_equal(np.isnan(a.astype(float)), np.isnan(b.astype(float))))
        rep[k] = {'frozen_key': fk, 'max_abs_diff': d, 'nan_pattern_equal': nan_ok,
                  'status': 'PASS' if d <= 1e-9 and nan_ok else 'DIFF'}
    else:
        rep[k] = {'frozen_key': fk, 'status': 'PASS' if np.array_equal(a, b) else 'DIFF'}
    print(f"{k:28s} <- {fk:28s} {rep[k]['status']} {rep[k].get('max_abs_diff', '')}")
for k in A.require:
    if k not in rep or rep[k]['status'] != 'PASS':
        bad.append(k)
summary = {'new': A.new, 'frozen': A.frozen, 'required': A.require, 'failed_required': bad,
           'only_new': sorted(set(N.files) - set(pairs)), 'only_frozen': sorted(set(F.files) - set(pairs.values())),
           'keys': rep}
if A.out:
    open(A.out, 'w').write(json.dumps(summary, indent=1))
print('PARITY', 'FAIL' if bad else 'PASS', 'failed_required=', bad)
sys.exit(1 if bad else 0)
