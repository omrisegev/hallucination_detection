"""Label-free check of the SML/L-SML structural assumption on a bank partition.

Under conditional independence between groups given the hidden truth, the
correlation between channels in DIFFERENT groups is rank one: R_ij = v_i v_j.
We fit v on those cross-group entries only and report the share of their
squared mass left unexplained. 0 = assumption holds exactly. No labels read.
"""
import json
from pathlib import Path
import numpy as np

def rank1_residual(R, groups, iters=500):
    g = np.asarray(groups); M = g[:, None] != g[None, :]
    w, U = np.linalg.eigh(np.where(M, R, 0)); v = U[:, -1] * np.sqrt(max(w[-1], 1e-9))
    for _ in range(iters):  # alternating LS on masked entries
        v = (np.where(M, R, 0) @ v) / np.maximum((M * (v[None, :] ** 2)).sum(1), 1e-12)
    E = np.where(M, R - np.outer(v, v), 0)
    return float((E ** 2).sum() / (np.where(M, R, 0) ** 2).sum()), v

if __name__ == '__main__':
    import ast
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    ROOT = Path(__file__).resolve().parents[2]
    z = np.load(ROOT / 'results/advisor_completion_2026-10-11/inputs/bank_V_N56.npz')
    cols = [str(c) for c in z['columns']]; R = np.corrcoef(z['V'], rowvar=False)
    src = (ROOT / 'scripts/run_advisor_v11_lr.py').read_text()
    fam = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign)
               and any(getattr(t, 'id', None) == 'FAMILIES' for t in n.targets))
    fam_of = {c: f for f, cs in fam.items() for c in cs}
    L = linkage(squareform(1 - np.abs(R) - np.diag(np.diag(1 - np.abs(R))), checks=False), 'average')
    out = {'families_F1_F7': rank1_residual(R, [fam_of[c] for c in cols])[0],
           'singletons': rank1_residual(R, list(range(len(cols))))[0]}
    for thr in (.2, .5):
        out[f'clusters_|r|>={1-thr:.1f}'] = rank1_residual(R, fcluster(L, t=thr, criterion='distance'))[0]
    out = {k: round(v, 3) for k, v in out.items()}; out['labels_read'] = False
    (Path(__file__).with_name('RANK1_N56.json')).write_text(json.dumps(out, indent=1)); print(out)
