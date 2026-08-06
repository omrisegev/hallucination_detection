"""Label-free stabilization operators for dependency-aware spectral fusion.

The base SDSF estimator uses a recovered low-rank-plus-sparse covariance in a
regularized linear solve.  This module adds two deliberately small, auditable
stabilizers without changing the U-PCR reliability model:

* diagonal covariance shrinkage controls uncertain dependency eigenvectors;
* bootstrap reliability shrinkage suppresses tail coordinates that are not
  reproducible under resampling.

Both operations are fitted from the feature matrix alone.  They never accept
correctness labels or an evaluation metric.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .dependency_fusion import regularized_covariance_weights, sparse_upcr_fit

__all__ = [
    "BootstrapReliability",
    "bootstrap_reliability",
    "diagonal_shrinkage",
    "stability_shrunk_weights",
]


def _symmetrize(a):
    a = np.asarray(a, dtype=float)
    return 0.5 * (a + a.T)


def diagonal_shrinkage(C, strength):
    """Shrink only off-diagonal covariance toward zero.

    ``strength=0`` returns ``C`` and ``strength=1`` returns its diagonal.  The
    target preserves each feature variance and removes only uncertain pairwise
    dependence.
    """
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    C = _symmetrize(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square")
    target = np.diag(np.diag(C))
    return _symmetrize((1.0 - strength) * C + strength * target)


@dataclass
class BootstrapReliability:
    """Bootstrap reliability estimates aligned to one full-data fit."""

    rho_samples: np.ndarray
    coordinate_mean: np.ndarray
    coordinate_sd: np.ndarray
    coordinate_snr: np.ndarray
    global_sign_flips: int
    n_successful: int
    n_requested: int


def bootstrap_reliability(F, full_fit, *, n_boot=12, seed=0, fit_kwargs=None):
    """Refit U-PCR reliability on bootstrap rows, using no labels.

    The U-PCR model has one unavoidable global sign ambiguity.  Every bootstrap
    estimate is aligned to the full-data estimate before its variability is
    measured; relative feature signs remain untouched.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must have shape (features, samples)")
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    fit_kwargs = dict(fit_kwargs or {})
    base_rho = np.asarray(full_fit.rho_hat, dtype=float)
    C = _symmetrize(full_fit.decomposition.structured_cov)
    _, Q = eigh(C)
    Q = Q[:, ::-1]
    rng = np.random.default_rng(seed)
    samples = []
    sign_flips = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, F.shape[1], size=F.shape[1])
        try:
            fitted = sparse_upcr_fit(F[:, idx], **fit_kwargs)
            rho = np.asarray(fitted.rho_hat, dtype=float)
            if not np.isfinite(rho).all() or np.linalg.norm(rho) < 1e-12:
                continue
            if float(rho @ base_rho) < 0.0:
                rho = -rho
                sign_flips += 1
            samples.append(rho)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if len(samples) < 2:
        raise RuntimeError("fewer than two successful bootstrap reliability fits")
    rho_samples = np.vstack(samples)
    coordinates = rho_samples @ Q
    mean = coordinates.mean(axis=0)
    sd = coordinates.std(axis=0, ddof=1)
    snr = np.abs(mean) / (sd + 1e-12)
    return BootstrapReliability(
        rho_samples=rho_samples,
        coordinate_mean=mean,
        coordinate_sd=sd,
        coordinate_snr=snr,
        global_sign_flips=sign_flips,
        n_successful=len(samples),
        n_requested=int(n_boot),
    )


def stability_shrunk_weights(
    full_fit,
    bootstrap,
    *,
    tau=1.0,
    preserve_components=2,
    covariance_shrinkage=0.0,
    target_condition=50.0,
):
    """Return a stabilized SDSF weight vector and transparent diagnostics.

    In the structured-covariance eigenbasis, coordinate ``j`` receives

        kappa_j = snr_j^2 / (snr_j^2 + tau^2).

    The leading U-PCR subspace is preserved.  Only reliability outside that
    subspace must earn its influence through bootstrap reproducibility.
    """
    tau = float(tau)
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    C = diagonal_shrinkage(
        full_fit.decomposition.structured_cov, covariance_shrinkage,
    )
    _, Q = eigh(C)
    Q = Q[:, ::-1]

    # Re-express the bootstrap samples in the final covariance basis.  This is
    # necessary when covariance shrinkage changes the eigenvectors.
    coords = np.asarray(bootstrap.rho_samples, dtype=float) @ Q
    coord_mean = coords.mean(axis=0)
    coord_sd = coords.std(axis=0, ddof=1)
    snr = np.abs(coord_mean) / (coord_sd + 1e-12)
    kappa = snr ** 2 / (snr ** 2 + tau ** 2 + 1e-18)
    head = min(max(0, int(preserve_components)), len(kappa))
    kappa[:head] = 1.0

    full_coordinates = Q.T @ np.asarray(full_fit.rho_hat, dtype=float)
    rho_shrunk = Q @ (kappa * full_coordinates)
    weights, solve_diag = regularized_covariance_weights(
        C, rho_shrunk, target_condition=target_condition,
    )
    return weights, {
        "tau": tau,
        "preserve_components": head,
        "covariance_shrinkage": float(covariance_shrinkage),
        "target_condition": float(target_condition),
        "tail_kappa_mean": float(np.mean(kappa[head:])) if head < len(kappa) else 1.0,
        "tail_kappa_min": float(np.min(kappa[head:])) if head < len(kappa) else 1.0,
        "rho_retained_fraction": float(
            np.linalg.norm(rho_shrunk) / (np.linalg.norm(full_fit.rho_hat) + 1e-12)
        ),
        "solve": solve_diag,
    }
