"""Label-free atomic roughness diagnostics for the AOG-IU-PCR premise audit.

This module never accepts correctness labels. It constructs one-dimensional
feature graphs, projects their roughness operators into the two-dimensional
IU-PCR subspace, and measures whether their effect is reproducible and aligned
with an IU-PCR score built without the graph-defining feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata

from .laplacian_upcr import (
    IU_FIT_DEFAULTS,
    self_tuning_knn_graph,
    symmetric_normalized_laplacian,
)
from .upcr import upcr_fit


EPS = 1e-12


@dataclass(frozen=True)
class IUState:
    """Reusable full-pool, two-component IU-PCR quantities."""

    F: np.ndarray
    covariance: np.ndarray
    basis: np.ndarray
    projected_covariance: np.ndarray
    rhs: np.ndarray
    baseline_weights: np.ndarray


@dataclass(frozen=True)
class AtomicQuotientGraph:
    """Order-invariant graph on the unique values of one atomic feature.

    Samples tied on the atomic feature are indistinguishable to that feature.
    They are therefore collapsed to one quotient node rather than connected by
    an arbitrary sparse k-NN tie break. ``projection.T @ score`` is the
    count-weighted group-mean score used by the graph energy.
    """

    graph: csr_matrix
    laplacian: csr_matrix
    projection: csr_matrix
    n_samples: int
    n_unique: int
    max_tie_size: int
    requested_k: int
    effective_k: int
    valid: bool

    def project_score(self, score):
        score = np.asarray(score, dtype=float)
        if score.shape != (self.n_samples,):
            raise ValueError("score length disagrees with quotient graph")
        return np.asarray(self.projection.T @ score, dtype=float).ravel()


def stable_seed(namespace: str) -> int:
    return int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:8], 16)


def path_token(graph_k: int, lambda_: float) -> str:
    value = format(float(lambda_), ".12g").replace(".", "p")
    return f"k{int(graph_k)}_lambda{value}"


def validate_features(F, names=None):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or min(F.shape) < 3:
        raise ValueError("F must have shape (at least 3 features, at least 3 samples)")
    if not np.isfinite(F).all():
        raise ValueError("F contains non-finite values")
    if names is not None and len(tuple(names)) != F.shape[0]:
        raise ValueError("feature names and F disagree")
    return F


def iu_state(F) -> IUState:
    """Fit the exact full-pool two-component IU-PCR anchor."""
    F = validate_features(F)
    baseline = upcr_fit(F, **IU_FIT_DEFAULTS)
    covariance = F @ F.T / F.shape[1]
    m = F.shape[0]
    values, basis = eigh(covariance, subset_by_index=[m - 2, m - 1])
    order = np.argsort(values)[::-1]
    basis = basis[:, order]
    projected = basis.T @ covariance @ basis
    projected = 0.5 * (projected + projected.T)
    rhs = basis.T @ baseline.rho_hat
    check = basis @ np.linalg.solve(projected, rhs)
    if not np.allclose(check, baseline.w, atol=1e-8, rtol=1e-7):
        raise RuntimeError("two-component IU-PCR reconstruction failed")
    return IUState(
        F=F,
        covariance=covariance,
        basis=basis,
        projected_covariance=projected,
        rhs=rhs,
        baseline_weights=baseline.w.copy(),
    )


def graph_operator(state: IUState, feature_index: int, *, graph_k: int):
    """Build one order-invariant atomic quotient graph and IU-PCR operator."""
    feature_index = int(feature_index)
    if not 0 <= feature_index < state.F.shape[0]:
        raise IndexError("feature index out of range")
    values = state.F[feature_index]
    unique, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    n = state.F.shape[1]
    n_unique = len(unique)
    valid = bool(n_unique >= 3)
    if valid:
        effective_k = min(int(graph_k), n_unique - 1)
        graph = self_tuning_knn_graph(unique[:, None], k=effective_k)
        laplacian = symmetric_normalized_laplacian(graph)
    else:
        effective_k = 0
        graph = csr_matrix((n_unique, n_unique), dtype=float)
        laplacian = csr_matrix((n_unique, n_unique), dtype=float)
    membership = coo_matrix(
        (np.ones(n, dtype=float), (np.arange(n), inverse)),
        shape=(n, n_unique),
    ).tocsr()
    projection = membership @ diags(1.0 / np.sqrt(counts.astype(float)))
    quotient_features = np.asarray(state.F @ projection, dtype=float)
    roughness = np.asarray(
        quotient_features @ (laplacian @ quotient_features.T) / max(n, 1),
        dtype=float,
    )
    roughness = 0.5 * (roughness + roughness.T)
    projected = state.basis.T @ roughness @ state.basis
    projected = 0.5 * (projected + projected.T)
    trace = float(np.trace(projected))
    signature = (
        projected / trace if np.isfinite(trace) and trace > EPS
        else np.eye(2, dtype=float) / 2.0
    )
    if not valid:
        signature = np.eye(2, dtype=float) / 2.0
    ambient = state.basis @ signature @ state.basis.T
    degree = np.asarray(graph.sum(axis=1)).ravel()
    components = (
        connected_components(graph, directed=False)[0] if valid else n_unique
    )
    eigenvalues = np.maximum(np.linalg.eigvalsh(signature), 0.0)
    effective_rank = float(
        np.sum(eigenvalues) ** 2 / (np.sum(eigenvalues ** 2) + EPS)
    )
    diagnostics = {
        "valid_operator": bool(valid),
        "n_unique_values": int(n_unique),
        "max_tie_size": int(np.max(counts)),
        "tied_sample_fraction": float(
            np.sum(counts[counts > 1]) / n
        ),
        "requested_graph_k": int(graph_k),
        "effective_graph_k": int(effective_k),
        "edge_mass_per_node": float(graph.sum() / max(n_unique, 1)),
        "n_edges": int(graph.nnz // 2),
        "n_components": int(components),
        "degree_p05_over_mean": float(
            np.quantile(degree, 0.05) / (np.mean(degree) + EPS)
        ) if len(degree) else 0.0,
        "projected_effective_rank": effective_rank,
        "distance_from_ridge": float(
            np.linalg.norm(signature - np.eye(2) / 2.0)
        ),
        "anisotropy": float(
            np.hypot(signature[0, 0] - signature[1, 1], 2.0 * signature[0, 1])
        ),
        "projected_trace_before_normalization": trace,
    }
    quotient = AtomicQuotientGraph(
        graph=graph,
        laplacian=laplacian,
        projection=projection.tocsr(),
        n_samples=n,
        n_unique=n_unique,
        max_tie_size=int(np.max(counts)),
        requested_k=int(graph_k),
        effective_k=int(effective_k),
        valid=valid,
    )
    return quotient, signature, ambient, diagnostics


def weights_for_signature(state: IUState, signature, lambda_: float):
    """Solve IU-PCR after adding a trace-matched two-by-two operator."""
    signature = np.asarray(signature, dtype=float)
    if signature.shape != (2, 2) or not np.isfinite(signature).all():
        raise ValueError("signature must be a finite two-by-two matrix")
    lambda_ = float(lambda_)
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda must be finite and nonnegative")
    if lambda_ == 0.0:
        return state.baseline_weights.copy()
    scale = float(np.trace(state.projected_covariance))
    system = state.projected_covariance + lambda_ * scale * signature
    return state.basis @ np.linalg.solve(system, state.rhs)


def normalized_graph_energy(score, quotient: AtomicQuotientGraph):
    score = np.asarray(score, dtype=float)
    score = score - np.mean(score)
    grouped = quotient.project_score(score)
    denominator = float(grouped @ grouped)
    if denominator <= EPS:
        return float("nan")
    return float(grouped @ (quotient.laplacian @ grouped) / denominator)


def duplicate_keep_mask(F, feature_index: int, threshold: float):
    """Remove one feature and its absolute-correlation near clones."""
    F = validate_features(F)
    centered = F - F.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    correlation = centered @ centered.T
    correlation /= norm[:, None] * norm[None, :] + EPS
    correlation = np.clip(correlation, -1.0, 1.0)
    similarity = np.abs(correlation[int(feature_index)])
    keep = similarity < float(threshold)
    keep[int(feature_index)] = False
    fallback = False
    if int(np.sum(keep)) < 3:
        keep = np.ones(F.shape[0], dtype=bool)
        keep[int(feature_index)] = False
        fallback = True
    return keep, similarity, fallback


def crossfit_alignment(
    F,
    feature_index: int,
    quotient: AtomicQuotientGraph,
    *,
    duplicate_threshold: float,
    permutation_count: int,
    namespace: str,
):
    """Compare graph energy of a duplicate-excluded IU score to permutations."""
    F = validate_features(F)
    keep, similarity, fallback = duplicate_keep_mask(
        F, feature_index, duplicate_threshold
    )
    fit = upcr_fit(F[keep], **IU_FIT_DEFAULTS)
    score = fit.w @ F[keep]
    if not quotient.valid:
        return {
            "alignment": 0.0,
            "observed_energy": 0.0,
            "permuted_energy_median": 0.0,
            "permuted_energy_mad": 0.0,
            "n_excluded": int(np.sum(~keep)),
            "max_other_abs_correlation": float(
                np.max(np.delete(similarity, int(feature_index)))
            ),
            "duplicate_fallback": bool(fallback),
        }
    observed = normalized_graph_energy(score, quotient)
    rng = np.random.default_rng(stable_seed(namespace))
    null = np.asarray([
        normalized_graph_energy(score[rng.permutation(len(score))], quotient)
        for _ in range(int(permutation_count))
    ], dtype=float)
    null_median = float(np.nanmedian(null))
    if not np.isfinite(observed) or not np.isfinite(null_median):
        alignment = 0.0
    else:
        alignment = float(
            np.clip((null_median - observed) / (abs(null_median) + EPS), -2.0, 2.0)
        )
    return {
        "alignment": alignment,
        "observed_energy": observed,
        "permuted_energy_median": null_median,
        "permuted_energy_mad": float(
            np.nanmedian(np.abs(null - null_median))
        ),
        "n_excluded": int(np.sum(~keep)),
        "max_other_abs_correlation": float(
            np.max(np.delete(similarity, int(feature_index)))
        ),
        "duplicate_fallback": bool(fallback),
    }


def rank_change(candidate, baseline):
    """Signed normalized rank change and its mean absolute magnitude."""
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline scores disagree")
    delta = (rankdata(candidate) - rankdata(baseline)) / len(candidate)
    return delta, float(np.mean(np.abs(delta)))


def _proxy_components(operator_distances, alignments, rank_sum, rank_sq_sum, actuations):
    count = len(alignments)
    operator_reproducibility = np.clip(
        1.0 - np.median(operator_distances, axis=0) / np.sqrt(2.0), 0.0, 1.0
    )
    effect_rms = np.sqrt(np.mean((rank_sum / count) ** 2, axis=1))
    total_rms = np.sqrt(rank_sq_sum / count)
    change_reproducibility = np.clip(
        effect_rms / (total_rms + EPS), 0.0, 1.0
    )
    actuation = np.median(actuations, axis=0)
    reference = float(np.median(actuation))
    relative_actuation = np.clip(
        actuation / (reference + EPS), 0.0, 1.0
    )
    alignment = np.median(alignments, axis=0)
    stability_actuation = (
        np.sqrt(operator_reproducibility * change_reproducibility)
        * relative_actuation
    )
    primary_proxy = alignment * stability_actuation
    return {
        "operator_reproducibility": operator_reproducibility,
        "rank_change_reproducibility": change_reproducibility,
        "bootstrap_actuation": actuation,
        "relative_actuation": relative_actuation,
        "bootstrap_alignment": alignment,
        "stability_actuation_proxy": stability_actuation,
        "primary_proxy": primary_proxy,
    }


def audit_cell(
    F,
    feature_names,
    *,
    cell: str,
    graph_ks=(7, 15, 30),
    primary_graph_k=15,
    lambdas=(0.3, 1.0, 3.0),
    primary_lambda=1.0,
    duplicate_threshold=0.95,
    duplicate_sensitivities=(0.90, 0.99),
    subsamples=40,
    sample_fraction=0.80,
    sample_cap=1500,
    permutation_count=16,
    convergence_checkpoints=(4, 8, 12, 20, 30, 40),
):
    """Run the complete label-free audit for one cell.

    Returns ``(score_arrays, diagnostics)``. Correctness labels are neither
    accepted nor accessible through this API.
    """
    F = validate_features(F, feature_names)
    names = tuple(str(name) for name in feature_names)
    graph_ks = tuple(int(value) for value in graph_ks)
    lambdas = tuple(float(value) for value in lambdas)
    if primary_graph_k not in graph_ks or primary_lambda not in lambdas:
        raise ValueError("primary k/lambda must be present in sensitivity paths")
    if int(subsamples) < max(convergence_checkpoints):
        raise ValueError("subsample count is smaller than a convergence checkpoint")

    state = iu_state(F)
    baseline_score = state.baseline_weights @ F
    score_arrays = {
        "feature_names": np.asarray(names, dtype=str),
        "sample_index": np.arange(F.shape[1], dtype=np.int64),
        "iu_pcr": np.asarray(baseline_score, dtype=np.float64),
    }
    full_records = [
        {"feature_index": index, "feature": name}
        for index, name in enumerate(names)
    ]
    full_ambient = {graph_k: [] for graph_k in graph_ks}
    full_valid = {graph_k: [] for graph_k in graph_ks}

    ridge_signature = np.eye(2) / 2.0
    for lambda_ in lambdas:
        weights = weights_for_signature(state, ridge_signature, lambda_)
        score_arrays[f"ridge__lambda_{lambda_:g}"] = np.asarray(
            weights @ F, dtype=np.float64
        )

    for graph_k in graph_ks:
        signatures = []
        for feature_index in range(F.shape[0]):
            quotient, signature, ambient, graph_diag = graph_operator(
                state, feature_index, graph_k=graph_k
            )
            signatures.append(signature)
            full_ambient[graph_k].append(ambient)
            full_valid[graph_k].append(bool(quotient.valid))
            full_records[feature_index][f"valid_operator__k_{graph_k}"] = bool(
                quotient.valid
            )
            full_records[feature_index][f"n_unique_values__k_{graph_k}"] = int(
                quotient.n_unique
            )
            alignment = crossfit_alignment(
                F,
                feature_index,
                quotient,
                duplicate_threshold=duplicate_threshold,
                permutation_count=permutation_count,
                namespace=f"{cell}:full:{graph_k}:{feature_index}:{duplicate_threshold}",
            )
            full_records[feature_index][f"full_alignment__k_{graph_k}"] = float(
                alignment["alignment"]
            )
            for lambda_ in lambdas:
                weights = (
                    weights_for_signature(state, signature, lambda_)
                    if quotient.valid else state.baseline_weights
                )
                key = f"atomic__k_{graph_k}__lambda_{lambda_:g}"
                if key not in score_arrays:
                    score_arrays[key] = np.empty(
                        (F.shape[0], F.shape[1]), dtype=np.float64
                    )
                score_arrays[key][feature_index] = weights @ F
                _, actuation = rank_change(
                    score_arrays[key][feature_index], baseline_score
                )
                full_records[feature_index][
                    f"full_actuation__{path_token(graph_k, lambda_)}"
                ] = actuation
            if graph_k == primary_graph_k:
                full_records[feature_index].update(graph_diag)
                full_records[feature_index].update({
                    "signature_00": float(signature[0, 0]),
                    "signature_01": float(signature[0, 1]),
                    "signature_11": float(signature[1, 1]),
                })
                full_records[feature_index].update({
                    f"full_{key}": value for key, value in alignment.items()
                })
                for threshold in duplicate_sensitivities:
                    sensitivity = crossfit_alignment(
                        F,
                        feature_index,
                        quotient,
                        duplicate_threshold=threshold,
                        permutation_count=permutation_count,
                        namespace=f"{cell}:full:{feature_index}:{threshold}",
                    )
                    token = str(threshold).replace(".", "p")
                    full_records[feature_index][
                        f"alignment_duplicate_{token}"
                    ] = sensitivity["alignment"]

        valid_signatures = [
            signature for signature, valid in zip(signatures, full_valid[graph_k])
            if valid
        ]
        mean_signature = (
            np.mean(valid_signatures, axis=0)
            if valid_signatures else np.eye(2) / 2.0
        )
        for lambda_ in lambdas:
            weights = weights_for_signature(state, mean_signature, lambda_)
            score_arrays[
                f"uniform_atomic__k_{graph_k}__lambda_{lambda_:g}"
            ] = np.asarray(weights @ F, dtype=np.float64)

    full_ambient = {
        graph_k: np.asarray(values) for graph_k, values in full_ambient.items()
    }
    centered = F - F.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    absolute_correlation = np.abs(
        (centered @ centered.T) / (norms[:, None] * norms[None, :] + EPS)
    )
    np.fill_diagonal(absolute_correlation, 0.0)
    for index, record in enumerate(full_records):
        record["duplicate_density"] = float(
            np.mean(absolute_correlation[index] >= duplicate_threshold)
        )
        distances = np.linalg.norm(
            full_ambient[primary_graph_k] - full_ambient[primary_graph_k][index],
            axis=(1, 2),
        )
        other = np.delete(distances, index)
        bandwidth = float(np.median(other)) if len(other) else 1.0
        record["operator_duplicate_density"] = float(
            np.mean(np.exp(-((other / (bandwidth + EPS)) ** 2)))
        ) if len(other) else 0.0

    m, n = F.shape
    operator_distances = {graph_k: [] for graph_k in graph_ks}
    alignments = {graph_k: [] for graph_k in graph_ks}
    actuations = {
        (graph_k, lambda_): [] for graph_k in graph_ks for lambda_ in lambdas
    }
    rank_sum = {
        (graph_k, lambda_): np.zeros((m, n), dtype=float)
        for graph_k in graph_ks for lambda_ in lambdas
    }
    rank_sq_sum = {
        (graph_k, lambda_): np.zeros(m, dtype=float)
        for graph_k in graph_ks for lambda_ in lambdas
    }
    convergence = []
    rng = np.random.default_rng(stable_seed(f"atomic-audit:{cell}"))
    sample_size = min(int(sample_cap), max(10, int(np.floor(sample_fraction * n))))
    sample_size = min(sample_size, n)
    for replicate in range(1, int(subsamples) + 1):
        if sample_size == n:
            index = np.arange(n, dtype=int)
        else:
            index = np.sort(rng.choice(n, size=sample_size, replace=False))
        local_F = F[:, index]
        local_state = iu_state(local_F)
        local_baseline_on_full = local_state.baseline_weights @ F
        for graph_k in graph_ks:
            local_operator_distance = np.empty(m, dtype=float)
            local_alignment = np.empty(m, dtype=float)
            local_actuation = {
                lambda_: np.empty(m, dtype=float) for lambda_ in lambdas
            }
            for feature_index in range(m):
                quotient, signature, ambient, _ = graph_operator(
                    local_state, feature_index, graph_k=graph_k
                )
                local_operator_distance[feature_index] = float(
                    np.linalg.norm(
                        ambient - full_ambient[graph_k][feature_index]
                    )
                )
                local_alignment[feature_index] = crossfit_alignment(
                    local_F,
                    feature_index,
                    quotient,
                    duplicate_threshold=duplicate_threshold,
                    permutation_count=permutation_count,
                    namespace=(
                        f"{cell}:subsample:{replicate}:{graph_k}:{feature_index}"
                    ),
                )["alignment"]
                for lambda_ in lambdas:
                    weights = (
                        weights_for_signature(local_state, signature, lambda_)
                        if quotient.valid else local_state.baseline_weights
                    )
                    candidate_on_full = weights @ F
                    delta, actuation = rank_change(
                        candidate_on_full, local_baseline_on_full
                    )
                    rank_sum[(graph_k, lambda_)][feature_index] += delta
                    rank_sq_sum[(graph_k, lambda_)][feature_index] += float(
                        np.mean(delta ** 2)
                    )
                    local_actuation[lambda_][feature_index] = actuation
            operator_distances[graph_k].append(local_operator_distance)
            alignments[graph_k].append(local_alignment)
            for lambda_ in lambdas:
                actuations[(graph_k, lambda_)].append(local_actuation[lambda_])
        if replicate in convergence_checkpoints:
            for graph_k in graph_ks:
                for lambda_ in lambdas:
                    partial = _proxy_components(
                        np.asarray(operator_distances[graph_k]),
                        np.asarray(alignments[graph_k]),
                        rank_sum[(graph_k, lambda_)],
                        rank_sq_sum[(graph_k, lambda_)],
                        np.asarray(actuations[(graph_k, lambda_)]),
                    )
                    for feature_index, name in enumerate(names):
                        convergence.append({
                            "replicates": replicate,
                            "graph_k": graph_k,
                            "lambda": lambda_,
                            "feature_index": feature_index,
                            "feature": name,
                            **{
                                key: float(value[feature_index])
                                for key, value in partial.items()
                            },
                        })

    for graph_k in graph_ks:
        for lambda_ in lambdas:
            components = _proxy_components(
                np.asarray(operator_distances[graph_k]),
                np.asarray(alignments[graph_k]),
                rank_sum[(graph_k, lambda_)],
                rank_sq_sum[(graph_k, lambda_)],
                np.asarray(actuations[(graph_k, lambda_)]),
            )
            token = path_token(graph_k, lambda_)
            for feature_index, record in enumerate(full_records):
                valid = bool(full_valid[graph_k][feature_index])
                for key, value in components.items():
                    record[f"{key}__{token}"] = (
                        float(value[feature_index]) if valid else 0.0
                    )

    primary_token = path_token(primary_graph_k, primary_lambda)
    for record in full_records:
        for key in (
            "operator_reproducibility",
            "rank_change_reproducibility",
            "bootstrap_actuation",
            "relative_actuation",
            "bootstrap_alignment",
            "stability_actuation_proxy",
            "primary_proxy",
        ):
            record[key] = record[f"{key}__{primary_token}"]
        record["full_alignment"] = record[f"full_alignment__k_{primary_graph_k}"]
        record["full_actuation"] = record[
            f"full_actuation__{primary_token}"
        ]

    for key, value in score_arrays.items():
        if key not in {"feature_names", "sample_index"} and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite score array: {cell}/{key}")

    diagnostics = {
        "cell": str(cell),
        "n_samples": int(n),
        "n_features": int(m),
        "realized_sample_size": int(sample_size),
        "feature_records": full_records,
        "convergence": convergence,
        "parameters": {
            "graph_ks": list(graph_ks),
            "primary_graph_k": int(primary_graph_k),
            "lambdas": list(lambdas),
            "primary_lambda": float(primary_lambda),
            "duplicate_threshold": float(duplicate_threshold),
            "duplicate_sensitivities": list(duplicate_sensitivities),
            "subsamples": int(subsamples),
            "sample_fraction": float(sample_fraction),
            "sample_cap": int(sample_cap),
            "permutation_count": int(permutation_count),
            "convergence_checkpoints": list(convergence_checkpoints),
        },
    }
    return score_arrays, diagnostics


__all__ = [
    "AtomicQuotientGraph",
    "IUState",
    "audit_cell",
    "crossfit_alignment",
    "duplicate_keep_mask",
    "graph_operator",
    "iu_state",
    "normalized_graph_energy",
    "path_token",
    "rank_change",
    "stable_seed",
    "weights_for_signature",
]
