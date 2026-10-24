"""Anchored DUFS-LIU: where an external target may enter Laplacian IU-PCR.

DUFS-LIU (``laplacian_upcr``) has two places where it decides *what* it is
fusing toward, and either can follow a nuisance manifold instead of the error:

* the graph: DUFS gates choose the coordinates of the sample graph;
* the target: IU-PCR's ``rho_hat`` estimates ``Cov(f_i, Y)`` for the latent
  ``Y`` that the channels share most.

``target_anchored_laplacian`` already lets calibration labels (or a pseudo
target) set the graph gates.  This module adds the second entry point: the
same two-component projected solve as :func:`laplacian_iu_path`, with
``rho`` supplied from outside (``Cov(f_i, y)`` on calibration samples, or on
a pseudo-label such as a PRM's own flags).  The covariance ``C``, the
two-dimensional subspace ``U`` and the Laplacian roughness ``R`` stay
unlabeled and transductive.  With ``rho = rho_hat`` the solve reproduces
:func:`laplacian_iu_path` exactly.

It also holds the registered DUFS-LIU call sequence for an arbitrary feature
matrix, so experiments outside the 24-cell contract use the same recipe as
``historical_multitask_baselines.fit_registered_dufs_global``.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from .adapted_dufs import adapted_dufs_soft_gates
from .laplacian_upcr import (
    IU_FIT_DEFAULTS,
    LaplacianIUResult,
    build_graph_from_features,
    laplacian_iu_path,
    symmetric_normalized_laplacian,
)
from .upcr import upcr_fit


REGISTERED_SEEDS = (11, 23, 37)
REGISTERED_EPOCHS = 80
REGISTERED_K = 7
REGISTERED_LAMBDA = 0.1

_EPS = 1e-12


@dataclass
class RegisteredDUFSLIU:
    """One registered DUFS-LIU fit and the pieces anchored variants reuse."""

    fit: LaplacianIUResult
    gates: np.ndarray
    gate_diagnostics: dict
    graph: object


@dataclass
class ExternalTargetLIU:
    """Projected LIU solve with an externally supplied ``rho``."""

    w: np.ndarray
    rho: np.ndarray
    lambda_: float
    diagnostics: dict = field(default_factory=dict)


def pooled_zscore_features(V):
    """Return ``(F, keep)``: non-constant columns of ``V`` z-scored over all rows.

    ``V`` is samples x channels; ``F`` is features x samples, the layout every
    U-PCR entry point expects.  This is the channel preparation of
    ``monday_cont_baselines_v1/run.py:fit_iu``.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError("V must have shape (samples, channels)")
    if not np.isfinite(V).all():
        raise ValueError("V contains non-finite values")
    sd = V.std(axis=0)
    keep = sd > 0
    if keep.sum() < 3:
        raise ValueError("fewer than three non-constant channels")
    F = ((V[:, keep] - V[:, keep].mean(axis=0)) / sd[keep]).T
    return F, keep


def covariance_target(F, target, *, indices=None):
    """``rho_i = Cov(F_i, y)`` over ``indices`` only (all samples when None)."""
    F = np.asarray(F, dtype=float)
    target = np.asarray(target, dtype=float)
    if F.ndim != 2 or target.ndim != 1 or target.shape[0] != F.shape[1]:
        raise ValueError("target must provide one value per sample of F")
    if indices is None:
        indices = np.arange(F.shape[1])
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or len(indices) < 2:
        raise ValueError("at least two target samples are required")
    y = target[indices]
    if not np.isfinite(y).all():
        raise ValueError("target contains non-finite values at the used indices")
    X = F[:, indices]
    return (X - X.mean(axis=1, keepdims=True)) @ (y - y.mean()) / len(indices)


def liu_external_target(F, rho, *, graph, lambda_=REGISTERED_LAMBDA, subspace=None):
    """Two-component Laplacian IU-PCR solve with an external ``rho``.

    Same algebra as :func:`laplacian_iu_path`:
    ``w = U [U^T (C + lambda R_bar) U]^-1 U^T rho`` with ``R_bar`` trace-matched
    to ``C`` inside ``U``.  ``lambda_ = 0`` gives the plain PCR solve (no
    Laplacian).  ``U`` is the top-2 eigenvectors of ``C`` unless ``subspace``
    (features x k, full column rank) is given; it is orthonormalized first.
    """
    F = np.asarray(F, dtype=float)
    rho = np.asarray(rho, dtype=float)
    m, n = F.shape
    if rho.shape != (m,) or not np.isfinite(rho).all():
        raise ValueError("rho must be a finite vector with one entry per feature")
    lambda_ = float(lambda_)
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and nonnegative")
    C = (F @ F.T) / n
    L = symmetric_normalized_laplacian(graph)
    Z = F.T
    R = np.asarray(Z.T @ (L @ Z) / n, dtype=float)
    R = 0.5 * (R + R.T)
    if subspace is None:
        evals, U = eigh(C, subset_by_index=[m - 2, m - 1])
        U = U[:, np.argsort(evals)[::-1]]
    else:
        U, r = np.linalg.qr(np.asarray(subspace, dtype=float))
        if U.shape[0] != m or np.min(np.abs(np.diag(r))) < 1e-10:
            raise ValueError("subspace must be a full-column-rank matrix with one row per feature")
    Cp = 0.5 * (U.T @ C @ U + (U.T @ C @ U).T)
    Rp = 0.5 * (U.T @ R @ U + (U.T @ R @ U).T)
    trace_r = float(np.trace(Rp))
    scale = float(np.trace(Cp)) / trace_r if trace_r > _EPS else 0.0
    system = Cp + lambda_ * scale * Rp
    w = U @ np.linalg.solve(system, U.T @ rho)
    return ExternalTargetLIU(
        w=w,
        rho=rho.copy(),
        lambda_=lambda_,
        diagnostics={
            "projected_condition_number": float(np.linalg.cond(system)),
            "projected_roughness_scale": scale,
            "rho_in_subspace_fraction": float(
                np.linalg.norm(U.T @ rho) / (np.linalg.norm(rho) + _EPS)
            ),
        },
    )


def registered_dufs_liu_fit(F, *, seeds=REGISTERED_SEEDS, epochs=REGISTERED_EPOCHS,
                            k=REGISTERED_K, lambda_=REGISTERED_LAMBDA):
    """The registered DUFS-LIU sequence on any oriented, z-scored ``F``.

    Gates, graph and solve follow ``fit_registered_dufs_global``; labels are not
    accepted.
    """
    gates, gate_diagnostics = adapted_dufs_soft_gates(F, seeds=seeds, epochs=epochs)
    graph = build_graph_from_features(F, gates=gates, k=k)
    fit = laplacian_iu_path(F, (float(lambda_),), graph=graph,
                            baseline_kwargs=dict(IU_FIT_DEFAULTS))[float(lambda_)]
    return RegisteredDUFSLIU(fit=fit, gates=gates, gate_diagnostics=gate_diagnostics,
                             graph=graph)


def split_liu_weights(F, gates, *, rho_hat=None, k=REGISTERED_K, lambda_=REGISTERED_LAMBDA):
    """LIU weights for given graph gates without :func:`laplacian_iu_path`'s per-call diagnostics.

    Same gates -> graph -> projected solve with IU-PCR's ``rho_hat``; the weights equal
    ``laplacian_iu_path(F, (lambda_,), gates=gates, k=k)`` (tests/test_anchored_liu.py).  It skips the ARPACK
    algebraic-connectivity diagnostic, which can run for many minutes on a connected graph with ~10^5 nodes.
    Returns ``(w, graph, diagnostics)``.
    """
    from scipy.sparse.csgraph import connected_components
    base = upcr_fit(F, **IU_FIT_DEFAULTS)
    rho = base.rho_hat if rho_hat is None else rho_hat
    graph = build_graph_from_features(F, gates=gates, k=k)
    w = liu_external_target(F, rho, graph=graph, lambda_=lambda_).w
    cosine = float(np.dot(w, base.w) / (np.linalg.norm(w) * np.linalg.norm(base.w) + _EPS))
    return w, graph, {"weight_cosine_vs_iu": cosine,
                      "n_components": int(connected_components(graph, directed=False)[0])}


def orient_to(score, anchor):
    """Flip ``score`` when its Pearson correlation with ``anchor`` is negative."""
    score = np.asarray(score, dtype=float)
    r = np.corrcoef(score, np.asarray(anchor, dtype=float))[0, 1]
    flipped = bool(np.isfinite(r) and r < 0)
    return (-score if flipped else score), flipped


__all__ = [
    "ExternalTargetLIU",
    "REGISTERED_EPOCHS",
    "REGISTERED_K",
    "REGISTERED_LAMBDA",
    "REGISTERED_SEEDS",
    "RegisteredDUFSLIU",
    "covariance_target",
    "liu_external_target",
    "orient_to",
    "pooled_zscore_features",
    "registered_dufs_liu_fit",
    "split_liu_weights",
]
