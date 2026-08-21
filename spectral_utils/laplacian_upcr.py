"""Laplacian-regularized IU-PCR without feature deletion.

This module connects two existing ideas without changing the moment identity
used by U-PCR:

* IU-PCR estimates ``rho_i = Cov(f_i, Y)`` from the *ordinary* empirical
  covariance, exactly as in Tenzer et al. (2022).
* DUFS-style continuous gates define a sample-neighbourhood graph.  Its
  symmetric normalized Laplacian penalizes differences between
  degree-normalized scores at neighbouring samples.

For centered feature matrix ``Z`` (samples x features), graph Laplacian ``L``,
and ordinary IU-PCR two-component subspace ``U``, the roughness matrix is

    R = Z.T @ L @ Z / n,

because ``w.T @ R @ w = (Z @ w).T @ L @ (Z @ w) / n``.  We keep IU-PCR's
ordinary ``C`` and ``rho`` estimates and solve only the final projected system

    w_lambda = U [U.T (C + lambda R_bar) U]^-1 U.T rho.

``R_bar`` is trace-matched to ``C`` inside the two-dimensional subspace, so
``lambda`` has a stable interpretation.  At ``lambda=0`` the implementation
returns the ordinary IU-PCR weights verbatim.

The graph is symmetric and sparse.  No feature is thresholded, no labels are
accepted, and the random-walk matrix from DUFS is not used directly because it
is generally nonsymmetric.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.special import ndtr
from sklearn.neighbors import NearestNeighbors

from .upcr import UPCRResult, upcr_fit


IU_FIT_DEFAULTS = {
    "loss": "l2",
    "exclusion": False,
    "difficulty_gate": False,
    "simple_avg_fallback": False,
    "recompute_after_exclusion": False,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "n_components": 2,
    "auto_components": False,
}

_EPS = 1e-12


@dataclass
class LaplacianIUResult:
    """Result of one Laplacian-regularized IU-PCR fit."""

    w: np.ndarray
    baseline: UPCRResult
    graph: csr_matrix
    laplacian: csr_matrix
    roughness: np.ndarray
    projected_covariance: np.ndarray
    projected_roughness: np.ndarray
    projected_roughness_scaled: np.ndarray
    lambda_: float
    diagnostics: dict = field(default_factory=dict)


def _validate_features(F):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must have shape (features, samples)")
    if F.shape[0] < 3 or F.shape[1] < 3:
        raise ValueError("at least three features and samples are required")
    if not np.isfinite(F).all():
        raise ValueError("F contains non-finite values")
    return F


def self_tuning_knn_graph(samples, *, k=7):
    """Return a symmetric sparse self-tuning k-NN affinity graph.

    ``samples`` has shape ``(n_samples, n_coordinates)``.  The local scale of
    sample ``i`` is its distance to its k-th neighbour.  Directed k-NN edges are
    symmetrized by maximum affinity; the diagonal is zero.
    """
    X = np.asarray(samples, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("samples must have shape (n>=3, d)")
    if not np.isfinite(X).all():
        raise ValueError("samples contains non-finite values")
    n = X.shape[0]
    k = int(max(1, min(int(k), n - 1)))
    neighbours = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbours.fit(X)
    distances, indices = neighbours.kneighbors(X, return_distance=True)
    # Column zero is the point itself.  sigma_i is the k-th non-self distance.
    sigma = np.maximum(distances[:, -1], 1e-8)
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].reshape(-1)
    d = distances[:, 1:].reshape(-1)
    denom = sigma[rows] * sigma[cols] + _EPS
    values = np.exp(-(d ** 2) / denom)
    directed = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    graph = directed.maximum(directed.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def symmetric_normalized_laplacian(graph):
    """Return ``I - D^-1/2 W D^-1/2`` for a symmetric nonnegative graph."""
    W = csr_matrix(graph, dtype=float)
    if W.shape[0] != W.shape[1] or W.shape[0] < 3:
        raise ValueError("graph must be square with at least three samples")
    delta = W - W.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-10:
        raise ValueError("graph must be symmetric")
    if W.nnz and np.min(W.data) < -1e-12:
        raise ValueError("graph contains negative weights")
    degree = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    positive = degree > _EPS
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = diags(inv_sqrt) @ W @ diags(inv_sqrt)
    return (eye(W.shape[0], format="csr") - normalized).tocsr()


def permute_graph(graph, permutation):
    """Relabel graph nodes, preserving its spectrum and edge-weight multiset."""
    W = csr_matrix(graph)
    permutation = np.asarray(permutation, dtype=int)
    if sorted(permutation.tolist()) != list(range(W.shape[0])):
        raise ValueError("permutation must contain every node exactly once")
    return W[permutation][:, permutation].tocsr()


def graph_diagnostics(graph, laplacian=None):
    W = csr_matrix(graph, dtype=float)
    L = symmetric_normalized_laplacian(W) if laplacian is None else csr_matrix(laplacian)
    degree = np.asarray(W.sum(axis=1)).ravel()
    n_components, _ = connected_components(W, directed=False)
    algebraic = 0.0
    # A disconnected graph has repeated zero eigenvalues. ARPACK can miss one
    # member of that repeated eigenspace with k=2 and report a positive value.
    if n_components == 1 and W.shape[0] > 2:
        try:
            # ARPACK otherwise samples a random initial vector.  Algebraic
            # connectivity is diagnostic-only, but it must remain byte-stable
            # across isolated rebuilds just like every other registered field.
            v0 = np.linspace(1.0, 2.0, W.shape[0], dtype=float)
            v0 /= np.linalg.norm(v0)
            values = eigsh(
                L,
                k=2,
                which="SM",
                return_eigenvectors=False,
                tol=1e-7,
                v0=v0,
            )
            algebraic = float(np.sort(values)[1])
        except Exception:
            algebraic = float("nan")
    return {
        "n_nodes": int(W.shape[0]),
        "n_edges": int(W.nnz // 2),
        "n_components": int(n_components),
        "degree_min": float(np.min(degree)),
        "degree_mean": float(np.mean(degree)),
        "degree_max": float(np.max(degree)),
        "algebraic_connectivity": algebraic,
        "graph_symmetry_error": float(
            np.max(np.abs((W - W.T).data)) if (W - W.T).nnz else 0.0
        ),
    }


def dufs_soft_gates(F, *, seeds=(0, 1, 2), epochs=None, return_history=False):
    """Learn DUFS Eq.-7 gates and return continuous survival probabilities.

    This wraps the repository's paper-grounded DUFS implementation.  Gate
    probabilities are averaged across seeds and RMS-normalized for graph
    distances.  Only relative gates matter under self-tuning bandwidths.
    """
    F = _validate_features(F)
    import torch
    from .selectors.a2_groupfs import (
        BATCH,
        EPOCHS_STAB,
        STG_SIGMA,
        _train_dufs,
    )

    torch.set_num_threads(1)
    X = torch.tensor(F.T, dtype=torch.float32)
    epochs = EPOCHS_STAB if epochs is None else int(epochs)
    probabilities = []
    histories = []
    for seed in seeds:
        trained = _train_dufs(
            X, 0.0, epochs, BATCH, int(seed), param_free=True,
            return_history=bool(return_history),
        )
        if return_history:
            mu, history = trained
            histories.append(history)
        else:
            mu = trained
        probabilities.append(ndtr(np.asarray(mu, dtype=float) / STG_SIGMA))
    per_seed = np.asarray(probabilities, dtype=float)
    raw = per_seed.mean(axis=0)
    rms = float(np.sqrt(np.mean(raw ** 2)))
    gates = raw / (rms if rms > _EPS else 1.0)
    diagnostics = {
        "raw_probabilities": raw,
        "per_seed_probabilities": per_seed,
        "mean_probability": float(raw.mean()),
        "near_zero_fraction": float(np.mean(raw < 0.05)),
        "near_one_fraction": float(np.mean(raw > 0.95)),
        "effective_feature_count": float((raw.sum() ** 2) / (np.sum(raw ** 2) + _EPS)),
        "mean_seed_std": float(np.mean(per_seed.std(axis=0))),
    }
    if return_history:
        diagnostics["training_history"] = np.asarray(histories, dtype=float)
    return gates, diagnostics


def build_graph_from_features(F, *, gates=None, k=7):
    """Build the sample graph from soft-gated feature coordinates."""
    F = _validate_features(F)
    m = F.shape[0]
    if gates is None:
        gates = np.ones(m, dtype=float)
    gates = np.asarray(gates, dtype=float)
    if gates.shape != (m,) or not np.isfinite(gates).all() or np.any(gates < 0):
        raise ValueError("gates must be a finite nonnegative vector of length m")
    return self_tuning_knn_graph(F.T * gates[None, :], k=k)


def laplacian_iu_path(F, lambdas, *, graph=None, gates=None, k=7,
                      baseline_kwargs=None):
    """Fit a path of regularization strengths while reusing one graph.

    ``graph`` may be supplied for negative controls. Otherwise it is built from
    all feature coordinates using optional continuous ``gates``. Labels are
    neither accepted nor used. The return value maps each float lambda to a
    :class:`LaplacianIUResult`.
    """
    F = _validate_features(F)
    lambdas = tuple(float(value) for value in lambdas)
    if not lambdas:
        raise ValueError("lambdas must not be empty")
    if any(not np.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError("every lambda must be finite and nonnegative")
    kwargs = dict(IU_FIT_DEFAULTS)
    if baseline_kwargs:
        kwargs.update(baseline_kwargs)
    required = {
        "exclusion": False,
        "difficulty_gate": False,
        "simple_avg_fallback": False,
        "n_components": 2,
        "auto_components": False,
    }
    incompatible = {
        key: (kwargs.get(key), expected)
        for key, expected in required.items()
        if kwargs.get(key) != expected
    }
    if incompatible:
        details = ", ".join(
            f"{key}={actual!r} (required {expected!r})"
            for key, (actual, expected) in incompatible.items()
        )
        raise ValueError(
            "Laplacian IU-PCR requires the full-pool two-component IU baseline: "
            + details
        )
    baseline = upcr_fit(F, **kwargs)

    W = build_graph_from_features(F, gates=gates, k=k) if graph is None else csr_matrix(graph)
    L = symmetric_normalized_laplacian(W)
    n = F.shape[1]
    C = (F @ F.T) / n
    Z = F.T
    R = np.asarray(Z.T @ (L @ Z) / n, dtype=float)
    R = 0.5 * (R + R.T)

    m = F.shape[0]
    evals, U = eigh(C, subset_by_index=[m - 2, m - 1])
    order = np.argsort(evals)[::-1]
    U = U[:, order]
    Cp = 0.5 * (U.T @ C @ U + (U.T @ C @ U).T)
    Rp = 0.5 * (U.T @ R @ U + (U.T @ R @ U).T)
    trace_r = float(np.trace(Rp))
    trace_c = float(np.trace(Cp))
    scale = trace_c / trace_r if trace_r > _EPS else 0.0
    Rp_scaled = scale * Rp

    eig_r = np.linalg.eigvalsh(R)
    common_diagnostics = {
        **graph_diagnostics(W, L),
        "roughness_symmetry_error": float(np.max(np.abs(R - R.T))),
        "roughness_min_eigenvalue": float(np.min(eig_r)),
        "roughness_effective_rank": float(
            (np.sum(np.maximum(eig_r, 0.0)) ** 2)
            / (np.sum(np.maximum(eig_r, 0.0) ** 2) + _EPS)
        ),
        "projected_roughness_trace": trace_r,
        "projected_roughness_scale": scale,
    }
    output = {}
    rhs = U.T @ baseline.rho_hat
    zero_equation_w = U @ np.linalg.solve(Cp, rhs)
    zero_equation_error = float(np.max(np.abs(zero_equation_w - baseline.w)))
    for lambda_ in lambdas:
        if lambda_ == 0.0:
            # Exact identity is an experimental invariant, not a tolerance claim.
            w = baseline.w.copy()
        else:
            w = U @ np.linalg.solve(Cp + lambda_ * Rp_scaled, rhs)
        projected_system = Cp + lambda_ * Rp_scaled
        diagnostics = {
            **common_diagnostics,
            "projected_condition_number": float(np.linalg.cond(projected_system)),
            "weight_cosine_vs_iu": float(
                np.dot(w, baseline.w)
                / (np.linalg.norm(w) * np.linalg.norm(baseline.w) + _EPS)
            ),
            "weight_norm": float(np.linalg.norm(w)),
            "zero_equation_weight_error": zero_equation_error,
            "score_variance": float(np.var(w @ F)),
            "score_laplacian_energy": float((w @ F) @ (L @ (w @ F)) / n),
        }
        output[lambda_] = LaplacianIUResult(
            w=w,
            baseline=baseline,
            graph=W,
            laplacian=L,
            roughness=R,
            projected_covariance=Cp,
            projected_roughness=Rp,
            projected_roughness_scaled=Rp_scaled,
            lambda_=lambda_,
            diagnostics=diagnostics,
        )
    return output


def laplacian_iu_fit(F, *, lambda_=0.0, graph=None, gates=None, k=7,
                     baseline_kwargs=None):
    """Fit one Laplacian-regularized IU-PCR configuration."""
    return laplacian_iu_path(
        F, [lambda_], graph=graph, gates=gates, k=k,
        baseline_kwargs=baseline_kwargs,
    )[float(lambda_)]


__all__ = [
    "IU_FIT_DEFAULTS",
    "LaplacianIUResult",
    "build_graph_from_features",
    "dufs_soft_gates",
    "graph_diagnostics",
    "laplacian_iu_fit",
    "laplacian_iu_path",
    "permute_graph",
    "self_tuning_knn_graph",
    "symmetric_normalized_laplacian",
]
