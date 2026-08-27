"""Family-residual sample graphs and IU-anchored Laplacian readouts.

The module deliberately separates three operations:

1. ordinary IU-PCR defines a baseline score and provenance-family
   contributions;
2. those contributions are residualized against the IU score and used only
   as graph coordinates;
3. a supplied graph acts either through the historical two-PC LIU solver or
   through a small, IU-anchored family-residual correction.

No eigenmode of the family covariance is selected here.  In particular, this
module has no eigenvalue-one target-identification rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

from .contribution_subspace import (
    ContributionSpace,
    ContributionTransform,
    fit_contribution_transform,
    iu_family_contributions,
)
from .laplacian_upcr import (
    IU_FIT_DEFAULTS,
    symmetric_normalized_laplacian,
)
from .graph_topology import (
    adaptive_knn_graph,
    extended_graph_diagnostics,
    mutual_knn_graph,
    self_safe_knn_graph,
)
from .upcr import UPCRResult, upcr_fit


EPS = 1e-12


@dataclass(frozen=True)
class FamilyResidualState:
    """Label-free IU contribution representation for one sample population."""

    baseline_fit: UPCRResult
    contribution_space: ContributionSpace
    transform: ContributionTransform
    baseline: np.ndarray
    residuals: np.ndarray
    standardized_contributions: np.ndarray
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FamilyGraph:
    """One graph plus the coordinates and scales that produced it."""

    graph: csr_matrix
    coordinates: np.ndarray
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ContributionLaplacianResult:
    """IU-anchored correction fitted in family-residual coordinates."""

    score: np.ndarray
    correction: np.ndarray
    delta: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def fit_family_residual_state(F, feature_names, *, baseline_kwargs=None):
    """Return standardized IU score, family contributions, and residuals."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] != len(feature_names):
        raise ValueError("F must have shape (features, samples)")
    if not np.isfinite(F).all():
        raise ValueError("F contains non-finite values")
    kwargs = dict(IU_FIT_DEFAULTS)
    if baseline_kwargs:
        kwargs.update(baseline_kwargs)
    fitted = upcr_fit(F, **kwargs)
    space = iu_family_contributions(F, feature_names, fitted.w)
    indices = np.arange(F.shape[1], dtype=int)
    transform = fit_contribution_transform(space, indices)
    baseline, residuals = transform.apply(
        space.baseline_score, space.contributions
    )
    standardized = (
        np.asarray(space.contributions, dtype=float)
        - transform.contribution_mean[None, :]
    ) / transform.contribution_scale[None, :]
    covariance = float(np.dot(
        baseline - np.mean(baseline),
        residuals[:, 0] - np.mean(residuals[:, 0]),
    ) / len(baseline)) if residuals.shape[1] else 0.0
    return FamilyResidualState(
        baseline_fit=fitted,
        contribution_space=space,
        transform=transform,
        baseline=np.asarray(baseline, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        standardized_contributions=np.asarray(standardized, dtype=float),
        diagnostics={
            "n_samples": int(F.shape[1]),
            "n_features": int(F.shape[0]),
            "n_families": int(residuals.shape[1]),
            "reconstruction_error": float(
                space.diagnostics["reconstruction_error"]
            ),
            "first_residual_baseline_covariance": covariance,
        },
    )


def _deterministic_pairwise_scale(block, *, max_pairs=20000, seed=1729):
    """Estimate a block's median non-zero squared pair distance cheaply."""
    X = np.asarray(block, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("block must have shape (samples>=2, coordinates)")
    if X.shape[1] == 0:
        return 1.0
    n = X.shape[0]
    total = n * (n - 1) // 2
    if total <= max_pairs:
        left, right = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(int(seed))
        left = rng.integers(0, n, size=max_pairs)
        right = rng.integers(0, n - 1, size=max_pairs)
        right = right + (right >= left)
    squared = np.sum((X[left] - X[right]) ** 2, axis=1)
    positive = squared[squared > EPS]
    if not len(positive):
        return 1.0
    return float(max(np.median(positive), EPS))


def normalized_graph_coordinates(
    dufs_coordinates,
    baseline,
    family_coordinates,
    *,
    eta,
    beta,
    scale_seed=1729,
):
    """Build block-balanced coordinates for the hybrid graph metric.

    The squared Euclidean distance in the returned matrix is

    ``(1-eta) d_DUFS + eta [beta d_b + (1-beta) d_family]``

    after each block is divided by its deterministic median non-zero squared
    pair distance.
    """
    D = np.asarray(dufs_coordinates, dtype=float)
    b = np.asarray(baseline, dtype=float).reshape(-1, 1)
    R = np.asarray(family_coordinates, dtype=float)
    if D.ndim != 2 or R.ndim != 2 or not (
        D.shape[0] == b.shape[0] == R.shape[0]
    ):
        raise ValueError("graph blocks disagree on sample count")
    eta = float(eta)
    beta = float(beta)
    if not (0.0 <= eta <= 1.0 and 0.0 <= beta <= 1.0):
        raise ValueError("eta and beta must be in [0,1]")
    scales = {
        "dufs": _deterministic_pairwise_scale(
            D, seed=scale_seed + 11
        ),
        "baseline": _deterministic_pairwise_scale(
            b, seed=scale_seed + 23
        ),
        "family": _deterministic_pairwise_scale(
            R, seed=scale_seed + 37
        ),
    }
    blocks = []
    weights = {
        "dufs": 1.0 - eta,
        "baseline": eta * beta,
        "family": eta * (1.0 - beta),
    }
    for name, values in (("dufs", D), ("baseline", b), ("family", R)):
        weight = weights[name]
        if weight > 0.0 and values.shape[1] > 0:
            blocks.append(
                np.sqrt(weight / scales[name]) * np.asarray(values, dtype=float)
            )
    if not blocks:
        raise ValueError("hybrid graph has no active coordinate block")
    coordinates = np.column_stack(blocks)
    if not np.isfinite(coordinates).all():
        raise ValueError("hybrid graph coordinates are non-finite")
    return coordinates, {"block_scales": scales, "block_weights": weights}


def graphs_from_coordinates(
    coordinates, ks, *, topology="union", tie_keys=None
):
    """Build reviewed duplicate-safe graph topologies without labels."""
    X = np.asarray(coordinates, dtype=float)
    n = X.shape[0]
    clean_ks = tuple(sorted({int(max(1, min(int(k), n - 1))) for k in ks}))
    if not clean_ks:
        raise ValueError("at least one k is required")
    if tie_keys is None:
        tie_keys = np.arange(n, dtype=float)
    tie_keys = np.asarray(tie_keys, dtype=float)
    if tie_keys.shape != (n,) or len(np.unique(tie_keys)) != n:
        raise ValueError("tie_keys must be unique and match samples")
    if topology == "union":
        return {
            k: self_safe_knn_graph(X, k=k, tie_keys=tie_keys)
            for k in clean_ks
        }
    if topology == "mutual":
        return {
            k: mutual_knn_graph(X, k=k, tie_keys=tie_keys)
            for k in clean_ks
        }
    if topology == "adaptive":
        if clean_ks != (7,):
            raise ValueError("adaptive topology is frozen at mean-k 7")
        graph, _ = adaptive_knn_graph(
            X, mean_k=7, min_k=3, max_k=25, scale_k=7,
            tie_keys=tie_keys,
        )
        return {7: graph}
    raise ValueError("topology must be union, adaptive, or mutual")


def build_family_graphs(
    F,
    gates,
    state,
    *,
    eta,
    beta,
    ks=(7,),
    family_mode="residual",
    topology="union",
    scale_seed=1729,
):
    """Build hybrid DUFS/family sample graphs for several neighbourhood sizes."""
    F = np.asarray(F, dtype=float)
    gates = np.asarray(gates, dtype=float)
    if gates.shape != (F.shape[0],):
        raise ValueError("gates must provide one value per input feature")
    if family_mode == "residual":
        family_coordinates = state.residuals
    elif family_mode == "contribution":
        family_coordinates = state.standardized_contributions
    else:
        raise ValueError("family_mode must be residual or contribution")
    coordinates, coordinate_diag = normalized_graph_coordinates(
        F.T * gates[None, :],
        state.baseline,
        family_coordinates,
        eta=eta,
        beta=beta,
        scale_seed=scale_seed,
    )
    graphs = graphs_from_coordinates(coordinates, ks, topology=topology)
    return {
        k: FamilyGraph(
            graph=graph,
            coordinates=coordinates,
            diagnostics={
                **coordinate_diag,
                **extended_graph_diagnostics(graph),
                "eta": float(eta),
                "beta": float(beta),
                "k": int(k),
                "family_mode": family_mode,
                "topology": topology,
                "n_graph_coordinates": int(coordinates.shape[1]),
            },
        )
        for k, graph in graphs.items()
    }


def contribution_laplacian_path(
    baseline,
    residuals,
    graph,
    lambdas,
    *,
    trust_caps=(None,),
):
    """Fit an IU-anchored correction in family-residual coordinates."""
    b = np.asarray(baseline, dtype=float)
    R = np.asarray(residuals, dtype=float)
    if b.ndim != 1 or R.ndim != 2 or R.shape[0] != len(b):
        raise ValueError("baseline/residual shape mismatch")
    L = symmetric_normalized_laplacian(graph)
    n, g = R.shape
    roughness = np.asarray(R.T @ (L @ R) / n, dtype=float)
    roughness = 0.5 * (roughness + roughness.T)
    cross = np.asarray(R.T @ (L @ b) / n, dtype=float)
    trace = float(np.trace(roughness))
    trace_scale = float(g / trace) if trace > EPS else 0.0
    A = trace_scale * roughness
    c = trace_scale * cross
    output = {}
    for lambda_ in tuple(float(value) for value in lambdas):
        if not np.isfinite(lambda_) or lambda_ < 0:
            raise ValueError("lambda must be finite and nonnegative")
        for cap in trust_caps:
            trust_cap = None if cap is None else float(cap)
            if trust_cap is not None and (
                not np.isfinite(trust_cap) or trust_cap <= 0
            ):
                raise ValueError("trust cap must be positive")
            if lambda_ == 0.0 or trace_scale == 0.0:
                delta = np.zeros(g, dtype=float)
                correction = np.zeros(n, dtype=float)
                cap_scale = 1.0
                score = b.copy()
            else:
                delta = -lambda_ * np.linalg.solve(
                    np.eye(g) + lambda_ * A, c
                )
                correction = R @ delta
                correction_sd = float(np.std(correction))
                cap_scale = 1.0
                if trust_cap is not None and correction_sd > trust_cap:
                    cap_scale = trust_cap / correction_sd
                    delta = cap_scale * delta
                    correction = cap_scale * correction
                score = b + correction
            key = (lambda_, trust_cap)
            output[key] = ContributionLaplacianResult(
                score=np.asarray(score, dtype=float),
                correction=np.asarray(correction, dtype=float),
                delta=np.asarray(delta, dtype=float),
                diagnostics={
                    "lambda": lambda_,
                    "trust_cap": trust_cap,
                    "trust_cap_scale": float(cap_scale),
                    "trace_scale": trace_scale,
                    "roughness_trace": trace,
                    "delta_norm": float(np.linalg.norm(delta)),
                    "correction_sd": float(np.std(correction)),
                    "baseline_correction_covariance": float(np.cov(
                        b, correction, ddof=0
                    )[0, 1]) if np.any(correction) else 0.0,
                    "score_laplacian_energy": float(
                        score @ (L @ score) / n
                    ),
                },
            )
    return output


def diffuse_score_path(baseline, graph, lambdas):
    """Return transductive score diffusion paths as a graph-quality diagnostic."""
    b = np.asarray(baseline, dtype=float)
    L = symmetric_normalized_laplacian(graph)
    output = {}
    identity = eye(len(b), format="csr")
    for lambda_ in tuple(float(value) for value in lambdas):
        if lambda_ == 0.0:
            output[lambda_] = b.copy()
        else:
            output[lambda_] = np.asarray(
                spsolve(identity + lambda_ * L, b), dtype=float
            )
    return output


__all__ = [
    "ContributionLaplacianResult",
    "FamilyGraph",
    "FamilyResidualState",
    "build_family_graphs",
    "contribution_laplacian_path",
    "diffuse_score_path",
    "fit_family_residual_state",
    "graphs_from_coordinates",
    "normalized_graph_coordinates",
]
