"""Answer-local temporal probability fusion. No labels or donor-answer input."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh

from .direct_probability_fusion import zscore_columns, _orient
from .laplacian_upcr import IU_FIT_DEFAULTS, symmetric_normalized_laplacian, permute_graph
from .shrinkage_iu import target_matrix, shrink
from .upcr import upcr_fit, upcr_fit_covariance

BASE_SOLVERS = ('equal', 'iu', 'diag_lw', 'joint_lw')
METHODS = tuple(f'{r}__{m}' for r in ('current', 'lag8', 'delta') for m in BASE_SOLVERS) + (
    'shuffled_lag8__equal', 'shuffled_lag8__iu',
    'lag8__time_then_rank_iu', 'lag8__rank_then_time_iu',
    'current__chain_liu', 'current__permuted_chain_liu',
)


@dataclass
class Fit:
    score: np.ndarray
    weights: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def seed_for(uid: str) -> int:
    return int.from_bytes(hashlib.sha256(('temporal-v3:' + uid).encode()).digest()[:4], 'little')


def representation(X, kind, *, seed=0):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 17 or not len(X) or not np.isfinite(X).all():
        raise ValueError('expected finite nonempty T x 17 input')
    if kind == 'current':
        return X.copy()
    if kind == 'delta':
        return np.column_stack((X, X - X[np.maximum(np.arange(len(X))-1, 0)]))
    if kind not in ('lag8', 'shuffled_lag8'):
        raise ValueError(kind)
    history = np.column_stack([X[np.maximum(np.arange(len(X))-lag, 0)] for lag in range(1, 8)])
    if kind == 'shuffled_lag8':
        history = history[np.random.default_rng(seed).permutation(len(X))]
    return np.column_stack((X, history))


def lw_alpha_memory_bounded(Z, C, target):
    """Algebraic version of existing LW heuristic; avoid T x P x P allocation."""
    n = len(Z)
    changed = ~np.isclose(C, target)
    np.fill_diagonal(changed, False)
    if n < 3 or not changed.any():
        return 0.0
    Z2 = Z * Z
    variance_of_mean = np.maximum((Z2.T @ Z2 - n * C * C) / (n * (n-1)), 0.0)
    den = float(np.square(C-target)[changed].sum())
    return float(np.clip(variance_of_mean[changed].sum()/den, 0, 1)) if den > 0 else 0.0


def finish(Z, keep, weights, anchor, *, diagnostics=None):
    score, flipped, corr = _orient(Z @ weights, anchor)
    full = np.zeros(len(keep))
    full[keep] = weights * (-1 if flipped else 1)
    if not np.isfinite(score).all() or not np.isfinite(full).all() or np.linalg.norm(full) == 0:
        raise FloatingPointError('nonfinite/zero fusion')
    return Fit(score, full, dict(diagnostics or {}, orientation_flipped=bool(flipped),
        anchor_correlation=float(corr) if np.isfinite(corr) else None,
        active_columns=int(keep.sum()), n_tokens=len(Z)))


def fit_linear(values, anchor, method):
    if len(values) < 3:
        raise ValueError('fewer than three tokens')
    Z, keep, _, _ = zscore_columns(values)
    if Z.shape[1] < 3:
        raise ValueError('fewer than three varying columns')
    diag = {}
    if method == 'equal':
        w = np.full(Z.shape[1], 1/Z.shape[1])
    elif method == 'iu':
        fit = upcr_fit(Z.T, **dict(IU_FIT_DEFAULTS))
        w = fit.w
        diag['abstained'] = bool(fit.abstained)
    elif method in ('diag_lw', 'joint_lw'):
        C = Z.T @ Z / len(Z)
        target = target_matrix(C, np.arange(Z.shape[1]), method.split('_')[0])
        alpha = lw_alpha_memory_bounded(Z, C, target)
        fit = upcr_fit_covariance(shrink(C, target, alpha), **dict(IU_FIT_DEFAULTS))
        w = fit.w
        diag.update(alpha=alpha, abstained=bool(fit.abstained))
    else:
        raise ValueError(method)
    return finish(Z, keep, np.asarray(w), anchor, diagnostics=diag)


def hierarchical(values, anchor, *, time_first):
    Z, keep, _, _ = zscore_columns(values)
    original = np.flatnonzero(keep)
    group_ids = original % 17 if time_first else original // 17
    virtual, inner = [], []
    for group in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group)
        if len(indices) < 3:
            raise ValueError('hierarchy group has fewer than three varying columns')
        fit = upcr_fit(Z[:, indices].T, **dict(IU_FIT_DEFAULTS))
        w = np.asarray(fit.w)
        virtual.append(Z[:, indices] @ w)
        inner.append((indices, w))
    V = np.column_stack(virtual)
    Vz, vk, vm, vs = zscore_columns(V)
    if Vz.shape[1] < 3:
        raise ValueError('fewer than three varying virtual streams')
    outer = upcr_fit(Vz.T, **dict(IU_FIT_DEFAULTS))
    w = np.zeros(Z.shape[1])
    for j, coefficient in zip(np.flatnonzero(vk), outer.w):
        indices, iw = inner[j]
        w[indices] = iw * coefficient / vs[j]
    expected = Vz @ outer.w
    if not np.allclose(Z @ w, expected, atol=1e-9, rtol=1e-9):
        raise FloatingPointError('hierarchy centering/reconstruction failed')
    return finish(Z, keep, w, anchor, diagnostics={'intermediate_groups':len(inner),
        'time_first':time_first, 'reconstruction_max_error':float(np.max(np.abs(Z @ w-expected)))})


def chain_fit(values, anchor, *, permuted=False, seed=0, lambda_=0.1):
    Z, keep, _, _ = zscore_columns(values)
    if len(Z) < 3 or Z.shape[1] < 3:
        raise ValueError('insufficient chain input')
    graph = diags((np.ones(len(Z)-1), np.ones(len(Z)-1)), (-1, 1), shape=(len(Z), len(Z)), format='csr')
    if permuted:
        graph = permute_graph(graph, np.random.default_rng(seed).permutation(len(Z)))
    # Same projected solve as laplacian_iu_path, without its expensive graph
    # eigsh connectivity diagnostic (irrelevant to w and problematic for chains).
    base = upcr_fit(Z.T, **dict(IU_FIT_DEFAULTS))
    C = Z.T @ Z / len(Z)
    L = symmetric_normalized_laplacian(graph)
    _, U = eigh(C, subset_by_index=[Z.shape[1]-2, Z.shape[1]-1])
    U = U[:, ::-1]
    Q = Z @ U
    Cp = U.T @ C @ U
    Rp = Q.T @ (L @ Q) / len(Z)
    Cp, Rp = (Cp+Cp.T)/2, (Rp+Rp.T)/2
    scale = float(np.trace(Cp)/np.trace(Rp)) if np.trace(Rp) > 1e-12 else 0.0
    w = base.w.copy() if lambda_ == 0 else U @ np.linalg.solve(Cp+lambda_*scale*Rp, U.T @ base.rho_hat)
    cosine = float(w @ base.w/(np.linalg.norm(w)*np.linalg.norm(base.w)+1e-12))
    return finish(Z, keep, w, anchor, diagnostics={'lambda':lambda_, 'permuted':permuted,
        'weight_cosine_vs_iu':cosine})


def fit_all(X, anchor, *, uid, methods=METHODS):
    """Return pure fits and explicit failures, never an entropy/other-arm fallback."""
    anchor = np.asarray(anchor, float)
    if anchor.shape != (len(X),) or not np.isfinite(anchor).all():
        raise ValueError('anchor does not align with tokens')
    seed = seed_for(uid)
    representations = {}
    fits, failures, seconds = {}, {}, {}
    for name in methods:
        if name not in METHODS:
            raise ValueError(name)
        r, solver = name.split('__')
        if r not in representations:
            representations[r] = representation(X, r, seed=seed)
        values = representations[r]
        started = time.perf_counter()
        try:
            if solver in BASE_SOLVERS:
                fits[name] = fit_linear(values, anchor, solver)
            elif solver.endswith('then_rank_iu') or solver.endswith('then_time_iu'):
                fits[name] = hierarchical(values, anchor, time_first=solver.startswith('time'))
            else:
                fits[name] = chain_fit(values, anchor, permuted=solver.startswith('permuted'), seed=seed)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            failures[name] = f'{type(error).__name__}: {error}'
        seconds[name] = time.perf_counter()-started
    return fits, failures, seconds
