"""lsml_merge_step_v1 helpers (label-free): the absorption merge added after L-SML's grouping, and the band selection
of channels by their Dawid-Skene balanced-accuracy estimate.  Protocol: results/lsml_merge_step_v1/PROTOCOL.json.

absorb_merge: for two groups A, B of a partition, rho = lambda_2(R[A u B]) / min(lambda_1(R[A]), lambda_1(R[B])), with R the
correlation matrix of the fitting rows.  Two groups with no correlation between them give rho = 1 whatever their sizes (the
weaker group keeps its own direction inside the union); two halves of one common factor give rho well below 1 (the weaker
half is absorbed into one direction).  The pair with the smallest rho is merged while rho < thr, and never below min_groups
groups (SML over 2 virtual classifiers is not identifiable).  With exactly 3 groups L-SML's own small-m guard replaces
the cross-group eigen-solve by equal weights over SD-standardized group scores (likewise inside a group of 3).
"""
import numpy as np


def canon(g) -> np.ndarray:
    """Labels 0..G-1 in sorted order of the input labels."""
    return np.unique(np.asarray(g, int), return_inverse=True)[1]


def top_eigs(C) -> tuple[float, float]:
    w = np.sort(np.linalg.eigvalsh(np.asarray(C, float)))[::-1]
    return float(w[0]), (float(w[1]) if len(w) > 1 else 0.0)


def absorb_merge(R, groups, thr: float = 0.5, min_groups: int = 3) -> tuple[np.ndarray, list]:
    """Returns (merged labels 0..G'-1, merge log).  Each log entry is a merge {'a','b','rho','size'} (labels of the
    partition current at that step) or the final stop {'stop': 'rho' | 'min_groups', 'rho': smallest rho left}."""
    R = np.asarray(R, float); g = canon(groups); seq = []
    if R.shape != (len(g), len(g)) or not np.isfinite(R).all():
        raise ValueError('correlation matrix missing, mis-sized or non-finite')
    while g.max() + 1 >= 2:
        G = int(g.max()) + 1
        lam1 = [top_eigs(R[np.ix_(g == a, g == a)])[0] for a in range(G)]
        best = None
        for a in range(G):
            for b in range(a + 1, G):
                idx = np.flatnonzero((g == a) | (g == b))
                rho = top_eigs(R[np.ix_(idx, idx)])[1] / min(lam1[a], lam1[b])
                if best is None or rho < best[0]:
                    best = (rho, a, b, len(idx))
        if best[0] >= thr:
            seq.append({'stop': 'rho', 'rho': best[0]}); break
        if G - 1 < min_groups:
            seq.append({'stop': 'min_groups', 'rho': best[0]}); break
        seq.append({'a': best[1], 'b': best[2], 'rho': best[0], 'size': best[3]})
        g[g == best[2]] = best[1]; g = canon(g)
    return g, seq


def band_select(pi_hat, lo: float = 0.45, hi: float = 0.55) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(keep: pi_hat > hi, flip: pi_hat < lo, drop: lo <= pi_hat <= hi) as index arrays."""
    pi = np.asarray(pi_hat, float)
    if not np.isfinite(pi).all():
        raise ValueError('non-finite estimate')
    return np.flatnonzero(pi > hi), np.flatnonzero(pi < lo), np.flatnonzero((pi >= lo) & (pi <= hi))


def dependence_split(C, g) -> dict:
    """Mean and max |off-diagonal| of a (class-conditional) correlation matrix over pairs in different groups and in the
    same group of partition g (diagnosis only)."""
    C = np.asarray(C, float); g = np.asarray(g); off = ~np.eye(len(C), dtype=bool); same = g[:, None] == g[None, :]
    out = {}
    for nm, sel in (('between', off & ~same), ('within', off & same)):
        v = np.abs(C[sel]); v = v[np.isfinite(v)]
        out[nm] = {'pairs': int(len(v) // 2), 'mean_abs': float(v.mean()) if len(v) else None, 'max_abs': float(v.max()) if len(v) else None}
    return out
