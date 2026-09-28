"""Stage B of expectation_realization_v1: a decision rule built from the binary classifier properties.
Frozen protocol: results/expectation_realization_v1/PROTOCOL_STAGE_B.json.

Label-free chain: random-tie top-20% marks -> Dawid-Skene filter (keep pi_hat > 0.5) -> L-SML partition of
the survivors on their marks -> one continuous score and one binary representative per group -> Dawid-Skene
on the representatives -> maximum-likelihood vote weights (Parisi et al. 2014) applied to the continuous
group scores.  For votes f_i in {-1, +1}:
    log P(f | Y=+1) / P(f | Y=-1) = sum_i f_i log(alpha_i) / 2 + const,   alpha_i = psi_i eta_i / ((1-psi_i)(1-eta_i))
so the vote weight is proportional to log(alpha_i), which is positive exactly when psi_i + eta_i > 1.
"""
from __future__ import annotations

import numpy as np


def random_tie_marks(V: np.ndarray, off: np.ndarray, frac: float, key: np.ndarray) -> np.ndarray:
    """+1 for the top ceil(frac n) steps of each answer and column, -1 otherwise.  Order: value descending,
    exact ties by `key` ascending (a uniform random key gives uniform tie-breaking, not positional)."""
    V = np.asarray(V, float); key = np.asarray(key, float)
    if V.shape != key.shape:
        raise ValueError('value and key shapes differ')
    out = -np.ones(V.shape, np.int8)
    for a, b in zip(off[:-1], off[1:]):
        blk = V[a:b]; n = b - a; k = max(1, int(np.ceil(frac * n)))
        p = np.argsort(key[a:b], axis=0, kind='stable')                     # random order first
        o = np.argsort(-np.take_along_axis(blk, p, axis=0), axis=0, kind='stable')   # then value, stable = key order within ties
        top = np.take_along_axis(p, o, axis=0)[:k]
        m = np.zeros(blk.shape, bool); np.put_along_axis(m, top, True, axis=0)
        out[a:b][m] = 1
    return out


def answer_center(T: np.ndarray, off: np.ndarray) -> np.ndarray:
    ns = np.diff(off)
    return T - np.repeat(np.add.reduceat(T, off[:-1], axis=0) / ns[:, None], ns, axis=0)


def mle_weights(psi, eta, eps: float = 1e-4) -> np.ndarray:
    """max(0, log(psi eta / ((1-psi)(1-eta)))), psi and eta clipped to [eps, 1-eps]."""
    p = np.clip(np.asarray(psi, float), eps, 1 - eps); e = np.clip(np.asarray(eta, float), eps, 1 - eps)
    return np.maximum(0.0, np.log(p) + np.log(e) - np.log1p(-p) - np.log1p(-e))


def group_scores(values: np.ndarray, off: np.ndarray, groups: np.ndarray, standardize) -> np.ndarray:
    """Column g = answer re-standardized mean of the columns with label g (labels 0..G-1)."""
    groups = np.asarray(groups, int); G = int(groups.max()) + 1
    if not np.array_equal(np.unique(groups), np.arange(G)):
        raise ValueError('group labels must be 0..G-1')
    return standardize(np.column_stack([values[:, groups == g].mean(1) for g in range(G)]), off)


def position_profile(V: np.ndarray, off: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Label-free mean of every column at (answer step count, step index), estimated on `rows`, returned for
    every step (keys never seen in `rows` get 0)."""
    ns = np.diff(off); aid = np.repeat(np.arange(len(ns)), ns); pos = np.arange(int(off[-1])) - off[aid]
    key = ns[aid].astype(np.int64) * 100_000 + pos
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv[rows], minlength=len(uk)).astype(float)
    prof = np.zeros((len(uk), V.shape[1]))
    for j in range(V.shape[1]):
        prof[:, j] = np.divide(np.bincount(inv[rows], weights=V[rows, j], minlength=len(uk)), cnt, out=np.zeros(len(uk)), where=cnt > 0)
    return prof[inv]


def canonical(labels) -> tuple:
    """Partition as a relabeling-invariant tuple (first-appearance relabeling)."""
    m = {}; return tuple(m.setdefault(int(v), len(m)) for v in labels)


def ari(a, b) -> float:
    """Adjusted Rand index of two labelings."""
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    _, ai = np.unique(a, return_inverse=True); _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1)); np.add.at(C, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2
    s = c2(C).sum(); sa = c2(C.sum(1)).sum(); sb = c2(C.sum(0)).sum(); e = sa * sb / c2(n); mx = (sa + sb) / 2
    return 1.0 if mx == e else float((s - e) / (mx - e))


def group_bar(est: dict, tru: dict, *, margin: float = 0.02, prev_tol: float = 0.05, rho_min: float = 0.80) -> dict:
    from scipy.stats import spearmanr
    pe = np.asarray(est['pi'], float); pt = np.asarray(tru['pi'], float); G = len(pt)
    decided = np.abs(pt - 0.5) > margin
    side_wrong = int((np.sign(pe[decided] - 0.5) != np.sign(pt[decided] - 0.5)).sum())
    if G >= 4:
        rho = float(spearmanr(pe, pt).statistic); rank_ok = bool(rho >= rho_min)
    else:
        rho = float(spearmanr(pe, pt).statistic) if G == 3 else float('nan'); rank_ok = bool(G == 3 and np.array_equal(np.argsort(pe), np.argsort(pt)))
    prev_err = float(est['prevalence'] - tru['prevalence'])
    res = {'groups': G, 'spearman_pi': rho, 'rank_ok': rank_ok, 'side_checked': int(decided.sum()), 'side_wrong': side_wrong, 'side_ok': side_wrong == 0,
           'prevalence_error': prev_err, 'prevalence_ok': abs(prev_err) <= prev_tol,
           'mae_psi': float(np.mean(np.abs(np.asarray(est['psi']) - tru['psi']))), 'mae_eta': float(np.mean(np.abs(np.asarray(est['eta']) - tru['eta'])))}
    res['passes'] = bool(res['rank_ok'] and res['side_ok'] and res['prevalence_ok'])
    return res
