"""Preparation for answer-local IU plus an external covariance prior.

This is an estimator seam, NOT a completed benchmark driver. The caller owns
fold exclusion, common feature coordinates, risk orientation and readout.
Alpha is external borrowing, distinct from the baseline's shrinkage strength.
"""
from __future__ import annotations

import numpy as np
from .upcr import upcr_fit_covariance
from .laplacian_upcr import IU_FIT_DEFAULTS


def checked_covariance(value, size):
    c = np.asarray(value, dtype=np.float64)
    if c.shape != (size, size) or not np.isfinite(c).all():
        raise ValueError('invalid covariance dimensions or missing values')
    tol = 1e-10 * max(1., float(np.max(np.abs(c))))
    if not np.allclose(c, c.T, rtol=0., atol=tol):
        raise ValueError('covariance must be symmetric')
    if np.linalg.eigvalsh(c).min() < -tol:
        raise ValueError('covariance must be positive semidefinite')
    return c


def borrowed_iu_weights(baseline_weights, baseline_covariance, prior_covariance,
                        alpha, *, covariance_solver=None):
    """Return weights and declared health; do not smooth token scores.

At alpha=0, return an exact copy of the caller's baseline weights without
accessing any prior or refitting. This preserves solve/subspace/full baselines.
For alpha>0, the default solver is the canonical FULL covariance IU fit.
A solve-only baseline must supply its matching covariance solver; the future
benchmark must register the choice and cannot call this a same-solver contrast
if its baseline used another solve level.
"""
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0. <= alpha <= 1.:
        raise ValueError('alpha must be finite and in [0, 1]')
    w = np.asarray(baseline_weights, dtype=np.float64)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all():
        raise ValueError('invalid baseline weights; no silent fallback')
    if alpha == 0.:
        return w.copy(), dict(alpha=alpha, source='exact_baseline', fitted=False)
    c = checked_covariance(baseline_covariance, len(w))
    prior = checked_covariance(prior_covariance, len(w))
    blended = (1. - alpha) * c + alpha * prior
    if covariance_solver is None:
        result = upcr_fit_covariance(blended, **IU_FIT_DEFAULTS)
        answer = np.asarray(result.w, dtype=np.float64)
        info = dict(solver='canonical_full_IU', abstained=bool(result.abstained),
                    simple_average=bool(result.used_simple_average),
                    components=int(result.n_components_used))
    else:
        answer, info = covariance_solver(blended)
        answer = np.asarray(answer, dtype=np.float64)
    if answer.shape != w.shape or not np.isfinite(answer).all():
        raise ValueError('covariance solver failed; no baseline substitution')
    return answer, dict(info, alpha=alpha, source='borrowed_covariance', fitted=True)
