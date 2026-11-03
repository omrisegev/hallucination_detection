"""Reproduce Z17 and CAT34 from INPUTS_<tag>.npz alone (no AIRCC, no telemetry pickle).

Needs the repository at commit f667f6679 or later (spectral_utils.fusion_utils.lsml_continuous,
spectral_utils.prmbench.prmbench_evaluate). Prints within-AUC and PRMScore for Z17 and CAT34 and,
if RESULTS.csv is given, checks them against the AIRCC run.

    python reproduce_from_bundle.py <INPUTS_tag.npz> <repo_root> [RESULTS.csv]
"""
import csv, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def az(v, o):
    out = np.zeros(len(v))
    for a, b in zip(o[:-1], o[1:]):
        x = v[a:b]; sd = x.std()
        if sd > 1e-12: out[a:b] = (x - x.mean()) / sd
    return out


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    path, root = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(root))
    from spectral_utils.fusion_utils import lsml_continuous
    from spectral_utils.prmbench import prmbench_evaluate
    B = np.load(path); o = B['offsets']; X = B['X34']; y = B['step_error']; ids = B['row_ids'].astype(str)
    eo, ef = B['error_steps_offsets'], B['error_steps_flat']
    meta = [{'idx': i, 'classification': str(c), 'error_steps': ef[eo[j]:eo[j + 1]].tolist()}
            for j, (i, c) in enumerate(zip(ids, B['classification']))]
    got = {}
    for arm, V in (('Z17', X[:, :17]), ('CAT34', X)):
        s, _ = lsml_continuous(*[V[:, j] for j in range(V.shape[1])]); s = np.asarray(s, float)
        if np.corrcoef(s, V[:, 0])[0, 1] < 0: s = -s
        z = az(s, o); flags = z > np.quantile(z, .8)          # Q80 rule
        pred = [{'idx': i, 'labels': (~flags[a:b]).astype(int).tolist()} for i, a, b in zip(ids, o[:-1], o[1:])]
        t = prmbench_evaluate(pred, meta)['total']
        w = np.nanmean([auc(y[a:b], s[a:b]) for a, b in zip(o[:-1], o[1:])])
        got[arm] = (w, 50 * (t['f1'] + t['negative_f1'])); print(path.name, arm, f'within-AUC {w:.6f}', f'PRMScore {got[arm][1]:.6f}')
    if len(sys.argv) > 3:
        tag = path.stem.replace('INPUTS_', '')
        ref = {r['arm']: r for r in csv.DictReader(open(sys.argv[3])) if r['model'] == tag}
        for arm, (w, p) in got.items():
            ok = abs(w - float(ref[arm]['within_auc'])) < 1e-9 and abs(p - float(ref[arm]['prmscore'])) < 1e-9
            print(tag, arm, 'MATCH' if ok else 'MISMATCH', ref[arm]['within_auc'], ref[arm]['prmscore'])


if __name__ == '__main__':
    main()
