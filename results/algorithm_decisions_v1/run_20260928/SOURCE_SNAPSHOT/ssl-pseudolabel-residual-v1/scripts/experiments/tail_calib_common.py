"""Helpers of tail_threshold_calibration_v1 (plan: why step-level tail-mark L-SML finds K=2, and
per-family calibration of the tail threshold).  calfix_common is imported unchanged: its hash is
recorded in TRANSFER_LOCK_V1.

- lsml_fit_scaled: calfix_common.lsml_fit plus (a) pooled column standardization of the fitting
  matrix, which is the documented input contract of lsml_continuous ("z-scored continuous arrays"),
  and (b) the loading_scale of the K criterion.  standardize=False, loading_scale='unit' is lsml_fit.
- value_marks: FUSE-style value cut (z > tau, tau = fit-row quantile giving mark rate q), centred
  within answer; mixed_marks: one rank-cut fraction per column.
- groups_from_R / residual_at / rel_residual: the lsml_continuous group search driven by a
  correlation matrix (so candidate threshold configurations can be scored from one Gram matrix).
- fit_prmscore: PRMScore of a score vector on the FIT folds' PRMBench labels, with the P1 threshold
  from the calibration fold; the evaluation fold's labels are masked (NaN) and never read.
"""
from __future__ import annotations

import importlib

import numpy as np
from scipy.stats import spearmanr

from calfix_common import EPS, load_extgen

GRID = (.05, .10, .15, .20, .30, .40, .50)


def fu():
    load_extgen()
    return importlib.import_module('extgen._bank11.fusion_utils')


def col_standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(0); sd = X.std(0)
    return np.divide(X - mu, sd, out=np.zeros_like(X, dtype=float), where=sd > EPS), mu, sd


def lsml_fit_scaled(X: np.ndarray, anchor: int = 0, orient_X: np.ndarray | None = None, *, standardize: bool = True, loading_scale: str = 'unit') -> dict:
    """calfix_common.lsml_fit semantics (small_m_guard, failures raise, Spearman orientation on
    orient_X, sum|w|=1).  Weights refer to the columns of the standardized matrix when standardize."""
    fz = load_extgen(); nb = fz.numerical_backend
    x = np.asarray(X, np.float64); ox = x if orient_X is None else np.asarray(orient_X, np.float64)
    if x.ndim != 2 or x.shape[1] < 3 or len(x) < 3 * x.shape[1] or not np.isfinite(x).all():
        raise ValueError('insufficient or invalid fitting observations')
    sd = np.ones(x.shape[1])
    if standardize:
        x, _mu, sd = col_standardize(x)
        if (sd <= EPS).any():
            raise ValueError('constant fitting column')
    if ox[:, anchor].std() <= EPS:
        raise ValueError('orientation anchor is inactive')
    nb.NUMERICAL_FAILURES.clear()
    _, meta = fz.lsml_continuous(*x.T, compute_score_matrix=False, small_m_guard=True, loading_scale=loading_scale)
    if nb.NUMERICAL_FAILURES or not np.isfinite(meta['residual']):
        raise ValueError('numerical estimator failure: ' + repr(nb.NUMERICAL_FAILURES))
    w = np.zeros(x.shape[1])
    for cross, (idx, within) in zip(meta['cross_weights'], meta['group_weights']):
        w[np.asarray(idx, int)] = np.asarray(within) * cross
    rho = float(spearmanr(ox @ w, ox[:, anchor]).statistic)
    if not np.isfinite(w).all() or abs(w).sum() <= EPS or not np.isfinite(rho):
        raise ValueError('invalid weights or undefined anchor orientation')
    if rho < 0:
        w *= -1
    w /= abs(w).sum()
    return {'weights': w, 'groups': np.asarray(meta['c'], int), 'K': int(len(np.unique(meta['c']))), 'anchor_spearman': abs(rho),
            'anchor_flipped': rho < 0, 'residual': float(meta['residual']), 'small_m_guarded': [list(v) for v in meta['small_m_guarded']],
            'fit_sd': sd, 'grouping_degenerate': bool(meta['degenerate']), 'residual_gap_rel': meta['grouping_diag'].get('residual_gap_rel')}


# ------------------------------------------------------------------ marks
def answer_center(T: np.ndarray, off: np.ndarray) -> np.ndarray:
    ns = np.diff(off)
    return T - np.repeat(np.add.reduceat(T, off[:-1], axis=0) / ns[:, None], ns, axis=0)


def value_marks(V: np.ndarray, off: np.ndarray, q: float, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1[V > tau_j] with tau_j the (1-q) quantile of column j over `rows` (the fit rows), centred
    within answer.  Returns (marks over all steps, tau)."""
    tau = np.quantile(V[rows], 1 - q, axis=0)
    return answer_center((V > tau).astype(float), off), tau


def mixed_marks(marks_by_q: dict, qs) -> np.ndarray:
    """column j from the rank-cut marks at fraction qs[j]."""
    return np.column_stack([marks_by_q[q][:, j] for j, q in enumerate(qs)])


# ------------------------------------------------------------------ group search from a correlation matrix
def groups_from_R(R: np.ndarray, loading_scale: str = 'unit', K_range=None) -> tuple[int, np.ndarray, float, list]:
    """The m>4 search of detect_dependent_groups (Eq.15 score, spectral clustering per K, Eq.14
    residual argmin, strict improvement), from R instead of the data."""
    F = fu(); m = R.shape[0]; assert m > F.SMALL_M_EXACT
    s = F._score_matrix_lsml(R); curve = []; best = (None, None, float('inf'))
    for K in (K_range or range(2, min(m, 9))):
        c = F._spectral_cluster_precomputed(s, K); r = F._residual_lsml(R, c, loading_scale=loading_scale); curve.append((K, r, c))
        if r < best[2]:
            best = (K, c, r)
    return best[0], best[1], best[2], curve


def residual_at(R: np.ndarray, c, loading_scale: str = 'complete') -> float:
    return fu()._residual_lsml(R, np.asarray(c), loading_scale=loading_scale)


def rel_residual(R: np.ndarray, loading_scale: str = 'complete') -> tuple[float, int, np.ndarray]:
    """Eq.14 residual of the best latent-group model / sum of squared off-diagonal correlations."""
    K, c, r, _ = groups_from_R(R, loading_scale)
    off2 = float((R ** 2).sum() - (np.diag(R) ** 2).sum())
    return r / off2, K, c


# ------------------------------------------------------------------ nested label-using objective
def fast_answer_z(s: np.ndarray, off: np.ndarray) -> np.ndarray:
    ns = np.diff(off); mu = np.repeat(np.add.reduceat(s, off[:-1]) / ns, ns); d = s - mu
    sd = np.repeat(np.sqrt(np.add.reduceat(d * d, off[:-1]) / ns), ns)
    return np.divide(d, sd, out=np.zeros_like(d), where=sd > 1e-8)


def fit_prmscore(score_z: np.ndarray, yv_masked: np.ndarray, sel_rows: np.ndarray, cal_rows: np.ndarray) -> float:
    """PRMScore on sel_rows (fit-fold non-control PRMBench steps) with tau = q80 of the calibration
    fold's scores (P1 rule).  yv_masked: 1 valid / 0 error / NaN where labels are withheld."""
    y = yv_masked[sel_rows]
    assert not np.isnan(y).any(), 'selection touched withheld labels'
    tau = np.quantile(score_z[cal_rows], .8); p = score_z[sel_rows] < tau; y = y.astype(bool)
    tp, fp, tn, fn = np.sum(p & y), np.sum(p & ~y), np.sum(~p & ~y), np.sum(~p & y)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1); nf1 = 2 * tn / max(2 * tn + fp + fn, 1)
    return float(.5 * (f1 + nf1))


def coordinate_descent(n_cols: int, objective, grid=GRID, start=.20, sweeps: int = 2, maximize: bool = True) -> tuple[tuple, float, list]:
    """Per-column fraction search.  Start at `start` for every column; in each sweep visit columns
    in order and move a column only on a STRICT improvement (ties keep the current value; among
    strictly better values the first in grid order wins).  Stops early when a sweep moves nothing."""
    sgn = 1.0 if maximize else -1.0; cache = {}
    def f(qs):
        if qs not in cache:
            cache[qs] = float(objective(qs))
        return cache[qs]
    cur = tuple([start] * n_cols); best = f(cur); trace = [{'sweep': 0, 'qs': cur, 'value': best}]
    for sw in range(1, sweeps + 1):
        moved = False
        for j in range(n_cols):
            cand = [(f(cur[:j] + (q,) + cur[j + 1:]), q) for q in grid if q != cur[j]]
            better = [(v, q) for v, q in cand if sgn * v > sgn * best + 1e-12]
            if better:
                top = max(sgn * vv for vv, _ in better); v, q = next((vv, qq) for vv, qq in better if sgn * vv >= top - 1e-12)
                cur = cur[:j] + (q,) + cur[j + 1:]; best = v; moved = True
        trace.append({'sweep': sw, 'qs': cur, 'value': best, 'evaluations': len(cache)})
        if not moved:
            break
    return cur, best, trace
