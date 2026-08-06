"""Covariance-aware reliability estimation for two-component SU-PCR.

This module changes only the solve of the overdetermined U-PCR pair equations.
The sparse low-rank decomposition, g2 selection interval, and final PCR rule are
kept fixed so an experiment isolates the correlated-moment hypothesis.

No function in this module accepts correctness labels.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf

from .dependency_fusion import _pcr_weights
from .upcr import additive_design

__all__ = [
    "DependencyAwareRhoResult",
    "CANDIDATE_METHODS",
    "pair_product_samples",
    "pair_moment_covariance",
    "estimate_dependency_aware_rho",
]


CANDIDATE_METHODS = (
    "ols", "diag_var", "diag_mad", "gaussian_gls", "lw_gls", "hybrid_gls",
)


@dataclass
class DependencyAwareRhoResult:
    """Reliability and PCR result from one registered pair-moment solve."""

    method: str
    rho_hat: np.ndarray
    g2_hat: float
    w_pcr: np.ndarray
    pcr_eigenvalues: np.ndarray
    projection_residual: float
    pair_residual_rmse: float
    normal_condition: float
    moment_condition_raw: float
    moment_condition_regularized: float
    diagnostics: dict = field(default_factory=dict)


def pair_product_samples(F, pairs=None):
    """Return row-level samples of every off-diagonal covariance moment."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must have shape (features, samples)")
    m, n = F.shape
    if m < 3 or n < 3 or not np.isfinite(F).all():
        raise ValueError("F needs at least 3 finite features and observations")
    A, pairs = additive_design(m, pairs=pairs)
    Z = np.empty((n, len(pairs)), dtype=float)
    for k, (i, j) in enumerate(pairs):
        Z[:, k] = F[i] * F[j]
    return Z, A, pairs


def _condition_covariance(covariance, max_condition=100.0):
    """PSD-project and floor eigenvalues to make moment inversion auditable."""
    covariance = 0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    values, vectors = eigh(covariance)
    raw_hi = max(float(values[-1]), 1e-12)
    positive = values[values > raw_hi * 1e-12]
    raw_condition = float(raw_hi / positive[0]) if positive.size else float("inf")
    floor = raw_hi / float(max_condition)
    regularized_values = np.maximum(values, floor)
    regularized = (vectors * regularized_values) @ vectors.T
    return 0.5 * (regularized + regularized.T), raw_condition, float(
        regularized_values[-1] / regularized_values[0]
    ), float(floor)


def _gaussian_pair_covariance(C, pairs):
    """Covariance of pair products under a zero-mean Gaussian model."""
    C = np.asarray(C, dtype=float)
    p = len(pairs)
    covariance = np.empty((p, p), dtype=float)
    for left, (i, j) in enumerate(pairs):
        for right in range(left, p):
            k, ell = pairs[right]
            value = C[i, k] * C[j, ell] + C[i, ell] * C[j, k]
            covariance[left, right] = covariance[right, left] = value
    return covariance


def _mad_variance(Z):
    median = np.median(Z, axis=0)
    scale = 1.4826 * np.median(np.abs(Z - median), axis=0)
    fallback = np.std(Z, axis=0, ddof=1)
    scale = np.where(scale > 1e-10, scale, fallback)
    return np.maximum(scale ** 2, 1e-12)


def pair_moment_covariance(F, C, method, *, max_condition=100.0):
    """Construct the registered moment weighting matrix without labels.

    Returns a precision matrix and diagnostics.  The omitted common factor
    ``1/n`` in covariance-of-the-mean matrices cancels from every GLS solve.
    """
    if method not in CANDIDATE_METHODS:
        raise ValueError(f"unknown method {method!r}")
    Z, A, pairs = pair_product_samples(F)
    empirical_var = np.var(Z, axis=0, ddof=1)
    floor = max(float(np.median(empirical_var)) * 1e-8, 1e-12)

    if method == "ols":
        covariance = np.eye(len(pairs))
        source = "identity"
    elif method == "diag_var":
        covariance = np.diag(np.maximum(empirical_var, floor))
        source = "empirical_diagonal"
    elif method == "diag_mad":
        covariance = np.diag(np.maximum(_mad_variance(Z), floor))
        source = "robust_mad_diagonal"
    else:
        gaussian = _gaussian_pair_covariance(C, pairs)
        if method == "gaussian_gls":
            covariance = gaussian
            source = "gaussian_fourth_moment"
        else:
            lw = LedoitWolf(assume_centered=False).fit(Z)
            empirical = np.asarray(lw.covariance_, dtype=float)
            if method == "lw_gls":
                covariance = empirical
                source = "ledoit_wolf_empirical"
            else:
                # Equalize trace before the fixed blend so 50/50 refers to
                # structure rather than an accidental overall scale.
                scale = float(np.trace(empirical) / max(np.trace(gaussian), 1e-12))
                covariance = 0.5 * empirical + 0.5 * scale * gaussian
                source = "half_ledoit_wolf_half_gaussian"

    conditioned, raw_cond, reg_cond, eigen_floor = _condition_covariance(
        covariance, max_condition=max_condition,
    )
    precision = np.linalg.solve(conditioned, np.eye(conditioned.shape[0]))
    return precision, A, pairs, {
        "source": source,
        "n_pair_moments": int(len(pairs)),
        "moment_condition_raw": raw_cond,
        "moment_condition_regularized": reg_cond,
        "moment_eigenvalue_floor": eigen_floor,
        "empirical_variance_min": float(np.min(empirical_var)),
        "empirical_variance_max": float(np.max(empirical_var)),
    }


def estimate_dependency_aware_rho(
    F,
    C,
    low_rank,
    var_y,
    *,
    method="ols",
    n_components=2,
    projection_components=1,
    g2_grid=300,
    moment_max_condition=100.0,
):
    """Estimate U-PCR rho with GLS pair moments and return fixed PCR weights."""
    F = np.asarray(F, dtype=float)
    C = np.asarray(C, dtype=float)
    low_rank = np.asarray(low_rank, dtype=float)
    m = F.shape[0]
    if C.shape != (m, m) or low_rank.shape != (m, m):
        raise ValueError("F, C, and low_rank dimensions disagree")

    precision, A, pairs, diagnostics = pair_moment_covariance(
        F, C, method, max_condition=moment_max_condition,
    )
    b = np.asarray([low_rank[i, j] for i, j in pairs], dtype=float)
    normal = A.T @ precision @ A
    rhs = A.T @ precision @ b
    normal = 0.5 * (normal + normal.T)
    values = eigh(normal, eigvals_only=True)
    normal_floor = max(float(values[-1]) * 1e-12, 1e-12)
    if values[0] < normal_floor:
        normal = normal + (normal_floor - float(values[0])) * np.eye(m)
    rho0 = np.linalg.solve(normal, rhs)

    kp = min(max(1, int(projection_components)), m)
    evecs = eigh(C, subset_by_index=[m - kp, m - 1])[1][:, ::-1]
    best = (float("inf"), 0.0, rho0)
    for q in np.linspace(0.0, float(var_y), max(2, int(g2_grid))):
        rho = rho0 + 0.5 * q
        projected = evecs @ (evecs.T @ rho)
        residual = float(
            np.linalg.norm(rho - projected) / (np.linalg.norm(rho) + 1e-12)
        )
        if residual < best[0]:
            best = (residual, float(q), rho.copy())

    g2_hat, rho_hat = best[1], best[2]
    w_pcr, eigenvalues = _pcr_weights(C, rho_hat, n_components=n_components)
    pair_residual = A @ (rho_hat - 0.5 * g2_hat) - b
    normal_condition = float(np.linalg.cond(normal))
    diagnostics.update({
        "normal_condition": normal_condition,
        "normal_eigenvalue_floor": normal_floor,
    })
    return DependencyAwareRhoResult(
        method=method,
        rho_hat=rho_hat,
        g2_hat=float(g2_hat),
        w_pcr=w_pcr,
        pcr_eigenvalues=eigenvalues,
        projection_residual=float(best[0]),
        pair_residual_rmse=float(np.sqrt(np.mean(pair_residual ** 2))),
        normal_condition=normal_condition,
        moment_condition_raw=float(diagnostics["moment_condition_raw"]),
        moment_condition_regularized=float(
            diagnostics["moment_condition_regularized"]
        ),
        diagnostics=diagnostics,
    )

