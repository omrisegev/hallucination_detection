"""Dawid-Skene with a position-dependent prevalence (position_prior_v1).  No labels enter.

Latent class Y per step (1 = the error class of the constant-prevalence DS fit the EM starts from); binary marks v_j
conditionally independent given Y, with psi_j = P(v_j = +1 | Y = 1) and fpr_j = P(v_j = +1 | Y = 0) as in the constant
model; the prevalence depends on the step's relative-position bin, P(Y = 1 | bin b) = pi_b.  With one bin this is the
constant model.  The EM starts from the constant fit (a nested start), so its likelihood can only rise.

The position term enters a continuous content score S in the same currency as the content: posterior log-odds
logit(pi_b) + a S, where a = (mu_1 - mu_0) / sigma_w^2 is the log-likelihood-ratio slope of S under a two-class
equal-variance Gaussian model whose class memberships are the EM posteriors.  Within an answer this ranks steps as
S + logit(pi_b) / a: no direction is assumed and nothing depends on the number of channels.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def clip(x):
    return np.clip(x, EPS, 1 - EPS)


def position_bins(off: np.ndarray, n_bins: int, order: np.ndarray | None = None) -> np.ndarray:
    """Relative-position bin of every step: floor(t * n_bins / (n - 1)), capped at n_bins - 1, in exact integer
    arithmetic (t = 0..n-1 within the answer; a one-step answer is bin 0).  `order`, if given, replaces t by a
    within-answer permutation of 0..n-1 (the order-permuted control)."""
    off = np.asarray(off, np.int64); ns = np.diff(off); S = int(off[-1])
    ans = np.repeat(np.arange(len(ns)), ns)
    t = (np.arange(S) - off[ans]) if order is None else np.asarray(order, np.int64)
    return np.minimum(t * n_bins // np.maximum(ns[ans] - 1, 1), n_bins - 1).astype(np.int64)


def fit_pds(votes: np.ndarray, bins: np.ndarray, n_bins: int, psi0, eta0, prev0: float,
            max_iter: int = 5000, tol: float = 1e-10) -> dict:
    """EM for the position-dependent-prevalence DS model from the constant fit (psi0, eta0, prev0).
    Returns pi_b per bin, psi, eta, the step posteriors q = P(Y=1 | marks, bin) and the likelihood path."""
    x = np.asarray(votes)
    if not np.isin(x, [-1, 1]).all():
        raise ValueError('binary +-1 marks required')
    bins = np.asarray(bins, np.int64)
    if bins.shape != (len(x),) or bins.min() < 0 or bins.max() >= n_bins:
        raise ValueError('bins must be 0..n_bins-1, one per row')
    b = (x > 0).astype(np.int8)
    u, inv, cnt = np.unique(np.column_stack([b, bins]), axis=0, return_inverse=True, return_counts=True)
    B = u[:, :-1].astype(float); ub = u[:, -1].astype(np.int64); w = cnt.astype(float); inv = np.asarray(inv).ravel()
    bin_rows = np.bincount(bins, minlength=n_bins)
    e1 = clip(np.asarray(psi0, float)); e0 = clip(1 - np.asarray(eta0, float)); pi = np.full(n_bins, float(clip(prev0)))
    curve = []; stable = 0; converged = False
    for it in range(max_iter + 1):
        l1 = B @ np.log(e1) + (1 - B) @ np.log1p(-e1); l0 = B @ np.log(e0) + (1 - B) @ np.log1p(-e0)
        lp1 = l1 + np.log(pi[ub]); lp0 = l0 + np.log1p(-pi[ub]); m = np.logaddexp(lp1, lp0)
        q = np.exp(lp1 - m); ll = float(w @ m); curve.append(ll)
        if len(curve) > 1:
            d = ll - curve[-2]
            if d < -1e-9 * max(1.0, abs(curve[-2])):
                raise ArithmeticError(f'EM likelihood decreased {d}')
            stable = stable + 1 if d / max(1.0, abs(curve[-2])) < tol else 0
            if stable >= 3:
                converged = True; break
        if it == max_iter:
            break
        wq = w * q; w0 = w - wq
        pi = clip(np.bincount(ub, weights=wq, minlength=n_bins) / np.maximum(np.bincount(ub, weights=w, minlength=n_bins), 1e-300))
        e1 = clip(B.T @ wq / wq.sum()); e0 = clip(B.T @ w0 / w0.sum())
    return {'pi_bins': pi, 'psi': e1, 'eta': 1 - e0, 'q': q[inv], 'loglik': curve[-1], 'loglik_start': curve[0],
            'iterations': len(curve) - 1, 'converged': converged, 'bin_rows': bin_rows,
            'oriented': bool(np.mean(e1 - e0) > 0), 'boundary_emissions': int(np.sum((e1 <= EPS) | (e1 >= 1 - EPS)) + np.sum((e0 <= EPS) | (e0 >= 1 - EPS)))}


def content_log_odds(votes: np.ndarray, psi, eta) -> np.ndarray:
    """The DS content log-likelihood ratio of each row's marks: sum_j log P(v_j | Y=1) / P(v_j | Y=0)."""
    b = (np.asarray(votes) > 0).astype(float); e1 = clip(np.asarray(psi, float)); e0 = clip(1 - np.asarray(eta, float))
    return b @ (np.log(e1) - np.log(e0)) + (1 - b) @ (np.log1p(-e1) - np.log1p(-e0))


def latent_slope(S: np.ndarray, q: np.ndarray) -> tuple[float, dict]:
    """Log-likelihood-ratio slope of a continuous score S under a two-class equal-variance Gaussian model whose class
    memberships are the posteriors q: a = (mu_1 - mu_0) / sigma_w^2."""
    S = np.asarray(S, float); q = np.asarray(q, float)
    if S.shape != q.shape:
        raise ValueError('S and q shapes differ')
    n1 = q.sum(); n0 = len(q) - n1
    mu1 = float(q @ S / n1); mu0 = float((1 - q) @ S / n0)
    var = float((q @ (S - mu1) ** 2 + (1 - q) @ (S - mu0) ** 2) / len(S))
    return (mu1 - mu0) / var, {'mu1': mu1, 'mu0': mu0, 'var_within': var, 'class1_mass': float(n1 / len(q))}
