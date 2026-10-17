"""Joint estimator and label-free diagnostics for disjoint continuous L-SML.

This module intentionally has no benchmark target, label, outcome, prevalence,
or scoring API.  It operates on donor rows and hard partitions only.  Overlap is
outside Joint L-SML v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from .dependency_fusion import regularized_covariance_weights
from .fusion_utils import _rank1_masked, _spectral_cluster_precomputed, lsml_continuous, sml_fuse_signed


EPS = 1e-12
DEFAULT_K_RANGE = (3, 4, 6, 8)


def covariance_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 3:
        raise ValueError("values must be a finite samples-by-features matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("values contain non-finite entries")
    covariance = np.cov(np.ascontiguousarray(matrix).T)
    return 0.5 * (covariance + covariance.T)


def offdiag_relative_misfit(observed: np.ndarray, fitted: np.ndarray) -> float:
    covariance = np.asarray(observed, dtype=np.float64)
    model = np.asarray(fitted, dtype=np.float64)
    if covariance.shape != model.shape or covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("observed and fitted must be equally sized square matrices")
    mask = ~np.eye(len(covariance), dtype=bool)
    return float(np.linalg.norm((covariance - model)[mask]) / max(float(np.linalg.norm(covariance[mask])), EPS))


def absolute_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(abs(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))


def score_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Spearman inputs must be equally sized finite vectors")
    if float(np.ptp(a)) <= EPS or float(np.ptp(b)) <= EPS:
        raise ValueError("Spearman inputs must be nonconstant")
    value = float(spearmanr(a, b).statistic)
    if not np.isfinite(value):
        raise RuntimeError("Spearman correlation is non-finite")
    return value


def pairwise_score_spearman(scores: Mapping[str, Sequence[float]]) -> dict[str, float]:
    names = tuple(scores)
    return {
        f"{left}__vs__{right}": score_spearman(scores[left], scores[right])
        for left, right in combinations(names, 2)
    }


def leading_offdiag_loading(
    covariance: np.ndarray,
    *,
    anchor_index: int | None = None,
    anchor_sign: int = 1,
    scale: bool = True,
) -> tuple[np.ndarray, float]:
    observed = np.asarray(covariance, dtype=np.float64)
    offdiag = observed - np.diag(np.diag(observed))
    values, vectors = eigh(offdiag)
    vector = np.asarray(vectors[:, -1], dtype=np.float64)
    if scale:
        vector *= np.sqrt(max(float(values[-1]), 0.0))
    if anchor_index is not None:
        if int(anchor_sign) not in (-1, 1):
            raise ValueError("anchor_sign must be +/-1")
        if vector[int(anchor_index)] * int(anchor_sign) < 0.0:
            vector *= -1.0
    else:
        nonzero = np.flatnonzero(np.abs(vector) > EPS)
        if nonzero.size and vector[nonzero[0]] < 0.0:
            vector *= -1.0
    return vector, float(values[-1])


def raw_orientation_cell(
    raw_standardized_active: np.ndarray,
    *,
    entropy_index: int,
    tau: float = 0.1,
) -> dict[str, Any]:
    covariance = covariance_matrix(raw_standardized_active)
    loading, eigenvalue = leading_offdiag_loading(
        covariance, anchor_index=int(entropy_index), anchor_sign=-1, scale=False
    )
    signs = np.where(loading < 0.0, -1, 1).astype(np.int64)
    absolute = np.abs(loading)
    degree = absolute * np.maximum(float(absolute.sum()) - absolute, 0.0)
    median = float(np.median(degree))
    threshold = float(tau) * median
    degree_keep = degree >= threshold
    return {
        "signs": signs,
        "loading": loading,
        "absolute_loading": absolute,
        "weighted_degree": degree,
        "median_weighted_degree": median,
        "degree_threshold": threshold,
        "degree_keep": degree_keep,
        "leading_eigenvalue": eigenvalue,
    }


def consensus_orientation_and_roster(
    cell_estimates: Sequence[Mapping[str, Any]],
    fallback_signs: Sequence[int],
    *,
    weak_loading_threshold: float = 0.01,
    minimum_sign_votes: int = 6,
    minimum_degree_cells: int = 8,
) -> dict[str, Any]:
    if len(cell_estimates) != 9:
        raise ValueError("the global contract requires exactly nine donor cells")
    sign_rows = np.asarray([row["signs"] for row in cell_estimates], dtype=np.int64)
    magnitudes = np.asarray([row["absolute_loading"] for row in cell_estimates], dtype=np.float64)
    degree_keep = np.asarray([row["degree_keep"] for row in cell_estimates], dtype=bool)
    fallback = np.asarray(fallback_signs, dtype=np.int64)
    if sign_rows.ndim != 2 or magnitudes.shape != sign_rows.shape or degree_keep.shape != sign_rows.shape:
        raise ValueError("cell orientation estimates disagree on roster")
    if fallback.shape != (sign_rows.shape[1],) or not np.isin(fallback, (-1, 1)).all():
        raise ValueError("fallback_signs disagree on roster")
    positive = np.sum(sign_rows > 0, axis=0)
    negative = np.sum(sign_rows < 0, axis=0)
    majority = np.where(negative > positive, -1, 1).astype(np.int64)
    winning_votes = np.maximum(positive, negative)
    mean_absolute = magnitudes.mean(axis=0)
    degree_pass_count = degree_keep.sum(axis=0)
    weak = mean_absolute < float(weak_loading_threshold)
    unstable = winning_votes < int(minimum_sign_votes)
    degree_rejected = degree_pass_count < int(minimum_degree_cells)
    active = ~(weak | unstable | degree_rejected)
    schema_signs = np.where(weak | unstable | degree_rejected, fallback, majority).astype(np.int64)
    return {
        "schema_signs": schema_signs,
        "majority_signs": majority,
        "positive_votes": positive,
        "negative_votes": negative,
        "winning_votes": winning_votes,
        "mean_absolute_loading": mean_absolute,
        "degree_pass_count": degree_pass_count,
        "weak": weak,
        "unstable": unstable,
        "degree_rejected": degree_rejected,
        "active": active,
        "weak_loading_threshold": float(weak_loading_threshold),
        "minimum_sign_votes": int(minimum_sign_votes),
        "minimum_degree_cells": int(minimum_degree_cells),
    }


def global_degree_roster(
    matrices: Sequence[np.ndarray],
    *,
    anchor_indices: Sequence[int],
    tau: float = 0.1,
    minimum_cells: int = 8,
) -> dict[str, Any]:
    if len(matrices) != 9 or len(anchor_indices) != 9:
        raise ValueError("global degree roster requires nine matrices and anchor indices")
    rows = []
    for matrix, anchor in zip(matrices, anchor_indices):
        covariance = covariance_matrix(matrix)
        loading, eigenvalue = leading_offdiag_loading(
            covariance, anchor_index=int(anchor), anchor_sign=1, scale=False
        )
        absolute = np.abs(loading)
        degree = absolute * np.maximum(float(absolute.sum()) - absolute, 0.0)
        median = float(np.median(degree))
        threshold = float(tau) * median
        rows.append({
            "loading": loading,
            "absolute_loading": absolute,
            "weighted_degree": degree,
            "median_weighted_degree": median,
            "degree_threshold": threshold,
            "degree_keep": degree >= threshold,
            "leading_eigenvalue": eigenvalue,
        })
    pass_count = np.sum(np.asarray([row["degree_keep"] for row in rows], dtype=bool), axis=0)
    return {
        "cell_estimates": rows,
        "degree_pass_count": pass_count,
        "active": pass_count >= int(minimum_cells),
        "tau": float(tau),
        "minimum_cells": int(minimum_cells),
    }


def coassignment_from_labels(labels: Sequence[int]) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("labels must be a one-dimensional roster")
    return (values[:, None] == values[None, :]).astype(np.float64)


def canonicalize_labels(labels: Sequence[int]) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    groups = sorted((tuple(np.flatnonzero(values == label).tolist()) for label in np.unique(values)), key=lambda row: row[0])
    output = np.empty_like(values)
    for index, group in enumerate(groups):
        output[np.asarray(group, dtype=np.int64)] = index
    return output


def residual_affinity(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(covariance, dtype=np.float64)
    try:
        loading = np.asarray(
            _rank1_masked(observed, np.eye(len(observed), dtype=bool), scale="complete"),
            dtype=np.float64,
        )
    except Exception:
        loading, _ = leading_offdiag_loading(observed)
    affinity = np.abs(observed - np.outer(loading, loading))
    np.fill_diagonal(affinity, 0.0)
    return affinity, loading


def _owner_sufficient_statistics(values: np.ndarray, owners: Sequence[int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    matrix = np.asarray(values, dtype=np.float64)
    owner = np.asarray(owners)
    unique = np.unique(owner)
    if owner.shape != (len(matrix),) or len(unique) < 3:
        raise ValueError("owners must identify at least three responses")
    stats = {
        "counts": np.asarray([np.sum(owner == key) for key in unique], dtype=np.int64),
        "sums": np.asarray([matrix[owner == key].sum(axis=0) for key in unique], dtype=np.float64),
        "crosses": np.asarray([matrix[owner == key].T @ matrix[owner == key] for key in unique], dtype=np.float64),
    }
    return unique, stats


def _covariance_from_multiplicity(stats: Mapping[str, np.ndarray], multiplicity: np.ndarray) -> np.ndarray:
    counts = np.asarray(stats["counts"], dtype=np.float64)
    sums = np.asarray(stats["sums"], dtype=np.float64)
    crosses = np.asarray(stats["crosses"], dtype=np.float64)
    mult = np.asarray(multiplicity, dtype=np.float64)
    count = float(mult @ counts)
    if count <= 1.0:
        raise ValueError("resample has fewer than two donor rows")
    total_sum = np.einsum("g,gp->p", mult, sums)
    total_cross = np.einsum("g,gpq->pq", mult, crosses)
    mean = total_sum / count
    covariance = (total_cross - count * np.outer(mean, mean)) / (count - 1.0)
    return 0.5 * (covariance + covariance.T)


def discover_loao_consensus_groups(
    donor_values: np.ndarray,
    answer_owners: Sequence[int],
    *,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    seed: int = 2026090401,
    minimum_group_size: int = 3,
    pairwise_diagnostic_cap: int = 32768,
    minimum_held_admissible_fraction: float = 1.0,
    use_minimum_ari_tiebreak: bool = False,
) -> dict[str, Any]:
    values = np.asarray(donor_values, dtype=np.float64)
    if not 0.0 < float(minimum_held_admissible_fraction) <= 1.0:
        raise ValueError("minimum_held_admissible_fraction must be in (0, 1]")
    if int(pairwise_diagnostic_cap) != pairwise_diagnostic_cap or int(pairwise_diagnostic_cap) < 1:
        raise ValueError("pairwise_diagnostic_cap must be a positive integer")
    unique, stats = _owner_sufficient_statistics(values, answer_owners)
    valid_k = tuple(sorted({int(k) for k in k_range if 3 <= int(k) < values.shape[1]}))
    if not valid_k:
        raise ValueError("K range has no valid candidate")
    candidates = []
    ones = np.ones(len(unique), dtype=np.int64)
    for k in valid_k:
        held_labels = []
        held_group_sizes = []
        coassignments = []
        for held_index in range(len(unique)):
            multiplicity = ones.copy()
            multiplicity[held_index] = 0
            covariance = _covariance_from_multiplicity(stats, multiplicity)
            affinity, _ = residual_affinity(covariance)
            labels = canonicalize_labels(
                _spectral_cluster_precomputed(affinity, k, seed=int(seed) + 1000 * k + held_index)
            )
            held_labels.append(labels)
            held_group_sizes.append(
                tuple(int(np.sum(labels == group)) for group in np.unique(labels))
            )
            coassignments.append(coassignment_from_labels(labels))
        consensus_affinity = np.mean(np.stack(coassignments, axis=0), axis=0)
        consensus = canonicalize_labels(
            _spectral_cluster_precomputed(consensus_affinity, k, seed=int(seed) + 100000 + k)
        )
        sizes = tuple(int(np.sum(consensus == group)) for group in np.unique(consensus))
        ari_to_consensus = np.asarray(
            [adjusted_rand_score(consensus, labels) for labels in held_labels], dtype=np.float64
        )
        # Pairwise LOAO ARI is diagnostic only; K selection uses ARI to the
        # consensus.  Materializing all O(n_answers^2) pairs is intractable for
        # the full PRMBench population, so large cells use a deterministic,
        # seed-bound pair sample.  Small-cell behavior remains byte-for-byte
        # identical to the original implementation.
        pair_count = len(held_labels) * (len(held_labels) - 1) // 2
        if pair_count <= int(pairwise_diagnostic_cap):
            diagnostic_pairs = list(combinations(range(len(held_labels)), 2))
            pairwise_sampling = "all_pairs"
        else:
            rng = np.random.default_rng(int(seed) + 10_000_000 + k)
            chosen: set[tuple[int, int]] = set()
            while len(chosen) < int(pairwise_diagnostic_cap):
                pair = np.sort(rng.choice(len(held_labels), size=2, replace=False))
                chosen.add((int(pair[0]), int(pair[1])))
            diagnostic_pairs = sorted(chosen)
            pairwise_sampling = "deterministic_uniform_pair_sample"
        pairwise = np.asarray(
            [adjusted_rand_score(held_labels[left], held_labels[right]) for left, right in diagnostic_pairs],
            dtype=np.float64,
        )
        held_admissible = np.asarray([
            len(row) == k and min(row) >= int(minimum_group_size)
            for row in held_group_sizes
        ], dtype=bool)
        held_admissible_fraction = float(np.mean(held_admissible))
        all_held_admissible = bool(np.all(held_admissible))
        admissible = bool(
            len(sizes) == k and min(sizes) >= int(minimum_group_size)
            and held_admissible_fraction >= float(minimum_held_admissible_fraction)
        )
        rejection_reason = None
        if not admissible:
            rejection_reason = (
                "GROUP_SIZE_LT3_IN_CONSENSUS_OR_LOAO"
                if float(minimum_held_admissible_fraction) == 1.0 else
                "GROUP_SIZE_LT3_IN_CONSENSUS_OR_TOO_MANY_LOAO_FOLDS"
            )
        candidates.append({
            "K": k,
            "labels": consensus,
            "group_sizes": sizes,
            "admissible": admissible,
            "rejection_reason": rejection_reason,
            "held_answer_group_sizes": tuple(held_group_sizes),
            "held_admissible_fraction": held_admissible_fraction,
            "all_held_admissible": all_held_admissible,
            "consensus_coassignment": coassignment_from_labels(consensus),
            "mean_loao_coassignment": consensus_affinity,
            "held_answer_labels": tuple(held_labels),
            "ari_to_consensus": ari_to_consensus,
            "pairwise_ari": pairwise,
            "pairwise_ari_population_count": int(pair_count),
            "pairwise_ari_sampling": pairwise_sampling,
            "pairwise_ari_sample_count": int(len(pairwise)),
            "median_ari": float(np.median(ari_to_consensus)),
            "mean_ari": float(np.mean(ari_to_consensus)),
            "minimum_ari": float(np.min(ari_to_consensus)),
            "exact_fraction": float(np.mean(ari_to_consensus == 1.0)),
        })
    eligible = [row for row in candidates if row["admissible"]]
    if not eligible:
        blocked_rule = (
            "max median LOAO-to-consensus ARI; mean ARI; minimum ARI; smaller K"
            if use_minimum_ari_tiebreak else
            "max median LOAO-to-consensus ARI; mean ARI; smaller K"
        )
        return {
            "status": "BLOCKED_NO_ADMISSIBLE_PARTITION",
            "candidates": candidates,
            "selection_rule": blocked_rule,
            "minimum_held_admissible_fraction": float(minimum_held_admissible_fraction),
            "n_answers": int(len(unique)),
        }
    if use_minimum_ari_tiebreak:
        selected = sorted(
            eligible,
            key=lambda row: (-row["median_ari"], -row["mean_ari"], -row["minimum_ari"], row["K"]),
        )[0]
        selection_rule = "max median LOAO-to-consensus ARI; mean ARI; minimum ARI; smaller K"
    else:
        selected = sorted(eligible, key=lambda row: (-row["median_ari"], -row["mean_ari"], row["K"]))[0]
        selection_rule = "max median LOAO-to-consensus ARI; mean ARI; smaller K"
    return {
        "status": "SELECTED",
        "K": int(selected["K"]),
        "labels": np.asarray(selected["labels"], dtype=np.int64),
        "group_sizes": tuple(selected["group_sizes"]),
        "coassignment": np.asarray(selected["consensus_coassignment"], dtype=np.float64),
        "mean_loao_coassignment": np.asarray(selected["mean_loao_coassignment"], dtype=np.float64),
        "median_ari": float(selected["median_ari"]),
        "mean_ari": float(selected["mean_ari"]),
        "minimum_ari": float(selected["minimum_ari"]),
        "held_admissible_fraction": float(selected["held_admissible_fraction"]),
        "exact_fraction": float(selected["exact_fraction"]),
        "pairwise_ari_summary": {
            "count": int(len(selected["pairwise_ari"])),
            "minimum": float(np.min(selected["pairwise_ari"])),
            "q25": float(np.quantile(selected["pairwise_ari"], 0.25)),
            "median": float(np.median(selected["pairwise_ari"])),
            "mean": float(np.mean(selected["pairwise_ari"])),
            "q75": float(np.quantile(selected["pairwise_ari"], 0.75)),
            "maximum": float(np.max(selected["pairwise_ari"])),
        },
        "candidates": candidates,
        "selection_rule": selection_rule,
        "minimum_held_admissible_fraction": float(minimum_held_admissible_fraction),
        "n_answers": int(len(unique)),
    }


def _factor_model(global_loading: np.ndarray, group_loading: np.ndarray, group_mask: np.ndarray) -> np.ndarray:
    output = np.outer(global_loading, global_loading) + group_mask * np.outer(group_loading, group_loading)
    np.fill_diagonal(output, 0.0)
    return output


def _objective(observed: np.ndarray, fitted: np.ndarray) -> float:
    left, right = np.triu_indices(len(observed), 1)
    residual = observed[left, right] - fitted[left, right]
    return float(residual @ residual)


def _complete_initializer(observed: np.ndarray) -> np.ndarray:
    try:
        loading = np.asarray(
            _rank1_masked(observed, np.eye(len(observed), dtype=bool), scale="complete"),
            dtype=np.float64,
        )
    except Exception:
        loading, _ = leading_offdiag_loading(observed)
    if loading.shape != (len(observed),) or not np.isfinite(loading).all() or np.linalg.norm(loading) <= EPS:
        loading, _ = leading_offdiag_loading(observed)
    return loading


def _orient_loading(vector: np.ndarray, anchor_index: int) -> np.ndarray:
    output = np.asarray(vector, dtype=np.float64).copy()
    if output[int(anchor_index)] < 0.0:
        output *= -1.0
    return output


@dataclass(frozen=True)
class JointStartResult:
    start: int
    global_loading: np.ndarray
    group_loading: np.ndarray
    fitted_offdiag: np.ndarray
    objective_trace: tuple[float, ...]
    model_change_trace: tuple[float, ...]
    converged: bool
    failed_monotonicity: bool
    sweeps: int


@dataclass(frozen=True)
class JointFitResult:
    global_loading: np.ndarray
    group_loading: np.ndarray
    fitted_offdiag: np.ndarray
    model_covariance: np.ndarray
    relative_offdiag_misfit: float
    objective: float
    converged: bool
    converged_starts: int
    selected_start: int
    starts: tuple[JointStartResult, ...]
    multistart_audit: Mapping[str, Any]
    jacobian_audit: Mapping[str, Any]
    diagonal_audit: Mapping[str, Any]


def _initial_loadings(observed: np.ndarray, group_mask: np.ndarray, *, start: int, seed: int, anchor_index: int) -> list[np.ndarray]:
    unknown = (group_mask > 0.0) | np.eye(len(observed), dtype=bool)
    try:
        global_loading = np.asarray(_rank1_masked(observed, unknown, scale="complete"), dtype=np.float64)
    except Exception:
        global_loading = _complete_initializer(observed)
    if not np.isfinite(global_loading).all() or np.linalg.norm(global_loading) <= EPS:
        global_loading = _complete_initializer(observed)
    global_loading = _orient_loading(global_loading, anchor_index)
    residual = observed - np.outer(global_loading, global_loading)
    group_loading = np.zeros(len(observed), dtype=np.float64)
    labels_mask = group_mask > 0.0
    unseen = set(map(int, np.flatnonzero(np.any(labels_mask, axis=1))))
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = set(map(int, np.flatnonzero(labels_mask[node]))) & unseen
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
        index = np.asarray(sorted(component), dtype=np.int64)
        block = residual[np.ix_(index, index)]
        try:
            loading = np.asarray(_rank1_masked(block, np.eye(len(index), dtype=bool), scale="complete"), dtype=np.float64)
        except Exception:
            loading, _ = leading_offdiag_loading(block)
        nonzero = np.flatnonzero(np.abs(loading) > EPS)
        if nonzero.size and loading[nonzero[0]] < 0.0:
            loading *= -1.0
        group_loading[index] = loading
    base = [global_loading, group_loading]
    if start == 0:
        return base
    rng = np.random.default_rng(int(seed) + 104729 * int(start))
    output = []
    for index, centre in enumerate(base):
        perturbation = rng.normal(size=len(observed)) if (start + index) % 2 else rng.choice(np.asarray([-1.0, 1.0]), size=len(observed))
        perturbation /= max(float(np.sqrt(np.mean(perturbation**2))), EPS)
        scale = max(float(np.sqrt(np.mean(centre**2))), 1e-3)
        output.append(centre + 0.05 * scale * perturbation)
    output[0] = _orient_loading(output[0], anchor_index)
    return output


def _fit_one_start(
    observed: np.ndarray,
    group_mask: np.ndarray,
    *,
    start: int,
    seed: int,
    anchor_index: int,
    max_sweeps: int,
    relative_tolerance: float,
    consecutive_stable_sweeps: int,
    monotonicity_tolerance: float,
) -> JointStartResult:
    loadings = _initial_loadings(observed, group_mask, start=start, seed=seed, anchor_index=anchor_index)
    masks = [np.ones_like(observed), group_mask.copy()]
    for mask in masks:
        np.fill_diagonal(mask, 0.0)
    fitted = _factor_model(loadings[0], loadings[1], group_mask)
    previous_objective = _objective(observed, fitted)
    objective_trace = [previous_objective]
    model_changes = []
    stable = 0
    converged = False
    failed_monotonicity = False
    used = 0
    for used in range(1, int(max_sweeps) + 1):
        previous_model = fitted.copy()
        for factor_index, mask in enumerate(masks):
            loading = loadings[factor_index]
            contribution = mask * np.outer(loading, loading)
            residual_without = observed - (fitted - contribution)
            for i in range(len(observed)):
                coefficients = mask[i] * loading
                coefficients[i] = 0.0
                denominator = float(coefficients @ coefficients)
                new_value = 0.0 if denominator <= EPS else float(coefficients @ residual_without[i] / denominator)
                old_value = float(loading[i])
                if new_value == old_value:
                    continue
                loading[i] = new_value
                delta = mask[i] * loading * (new_value - old_value)
                delta[i] = 0.0
                fitted[i, :] += delta
                fitted[:, i] += delta
                fitted[i, i] = 0.0
            loadings[factor_index] = loading
        left, right = np.triu_indices(len(observed), 1)
        basis = np.column_stack([
            (mask * np.outer(loading, loading))[left, right]
            for loading, mask in zip(loadings, masks)
        ])
        amplitude, _ = nnls(basis, observed[left, right])
        loadings = [loading * np.sqrt(max(float(value), 0.0)) for loading, value in zip(loadings, amplitude)]
        fitted = _factor_model(loadings[0], loadings[1], group_mask)
        objective = _objective(observed, fitted)
        change = float(np.linalg.norm(fitted - previous_model) / max(float(np.linalg.norm(previous_model)), EPS))
        relative_decrease = float((previous_objective - objective) / max(1.0, abs(previous_objective)))
        objective_trace.append(objective)
        model_changes.append(change)
        if objective > previous_objective + float(monotonicity_tolerance) * max(1.0, abs(previous_objective)):
            failed_monotonicity = True
            break
        stable = stable + 1 if abs(relative_decrease) <= float(relative_tolerance) and change <= float(relative_tolerance) else 0
        previous_objective = objective
        if stable >= int(consecutive_stable_sweeps):
            converged = True
            break
    loadings[0] = _orient_loading(loadings[0], anchor_index)
    fitted = _factor_model(loadings[0], loadings[1], group_mask)
    return JointStartResult(
        start=int(start),
        global_loading=loadings[0].copy(),
        group_loading=loadings[1].copy(),
        fitted_offdiag=fitted,
        objective_trace=tuple(map(float, objective_trace)),
        model_change_trace=tuple(map(float, model_changes)),
        converged=bool(converged and not failed_monotonicity),
        failed_monotonicity=bool(failed_monotonicity),
        sweeps=int(used),
    )


def _profiled_jacobian_audit(global_loading: np.ndarray, group_loading: np.ndarray, group_mask: np.ndarray) -> dict[str, Any]:
    size = len(global_loading)
    left, right = np.triu_indices(size, 1)
    jv = np.zeros((len(left), size), dtype=np.float64)
    ju = np.zeros((len(left), size), dtype=np.float64)
    for row, (i, j) in enumerate(zip(left, right)):
        jv[row, i], jv[row, j] = global_loading[j], global_loading[i]
        if group_mask[i, j] > 0.0:
            ju[row, i], ju[row, j] = group_loading[j], group_loading[i]
    active_u = np.linalg.norm(ju, axis=0) > EPS
    if np.any(active_u):
        nuisance = ju[:, active_u]
        nuisance_u, nuisance_s, _ = np.linalg.svd(nuisance, full_matrices=False)
        nuisance_tolerance = (
            max(nuisance.shape) * np.finfo(float).eps * max(float(nuisance_s[0]), EPS)
            if nuisance_s.size else np.inf
        )
        nuisance_rank = int(np.sum(nuisance_s > nuisance_tolerance))
        nuisance_basis = nuisance_u[:, :nuisance_rank]
        profiled = jv - nuisance_basis @ (nuisance_basis.T @ jv)
    else:
        profiled = jv
        nuisance_rank = 0
    norms = np.linalg.norm(profiled, axis=0)
    active_v = norms > EPS
    normalized = profiled[:, active_v] / norms[active_v][None, :]
    singular = np.linalg.svd(normalized, compute_uv=False) if normalized.size else np.asarray([])
    tolerance = max(normalized.shape) * np.finfo(float).eps * max(float(singular[0]), EPS) if singular.size else np.inf
    rank = int(np.sum(singular > tolerance)) if singular.size else 0
    condition = float(singular[0] / max(float(singular[-1]), EPS)) if singular.size else float("inf")
    return {
        "full_global_rank": bool(int(active_v.sum()) == size and rank == size),
        "active_global_columns": int(active_v.sum()),
        "rank": rank,
        "condition_number": condition,
        "singular_values": singular,
        "nuisance_columns": int(active_u.sum()),
        "nuisance_rank": int(nuisance_rank),
    }


def fit_joint_lsml(
    covariance: np.ndarray,
    labels: Sequence[int],
    *,
    anchor_index: int,
    seed: int = 2026090402,
    starts: int = 5,
    max_sweeps: int = 5000,
    relative_tolerance: float = 1e-10,
    consecutive_stable_sweeps: int = 5,
    monotonicity_tolerance: float = 1e-12,
) -> JointFitResult:
    observed = np.asarray(covariance, dtype=np.float64)
    partition = canonicalize_labels(labels)
    if observed.shape != (len(partition), len(partition)) or not np.isfinite(observed).all():
        raise ValueError("covariance/partition mismatch")
    sizes = [int(np.sum(partition == group)) for group in np.unique(partition)]
    if len(sizes) < 3 or min(sizes) < 3:
        raise ValueError("Joint L-SML requires K>=3 and every group size>=3")
    coassignment = coassignment_from_labels(partition)
    group_mask = coassignment.copy()
    np.fill_diagonal(group_mask, 0.0)
    start_results = tuple(
        _fit_one_start(
            observed, group_mask, start=index, seed=int(seed), anchor_index=int(anchor_index),
            max_sweeps=int(max_sweeps), relative_tolerance=float(relative_tolerance),
            consecutive_stable_sweeps=int(consecutive_stable_sweeps),
            monotonicity_tolerance=float(monotonicity_tolerance),
        )
        for index in range(int(starts))
    )
    converged = [row for row in start_results if row.converged]
    selected = min(converged or list(start_results), key=lambda row: row.objective_trace[-1])
    fitted = selected.fitted_offdiag
    component = np.outer(selected.global_loading, selected.global_loading) + coassignment * np.outer(selected.group_loading, selected.group_loading)
    diagonal_raw = np.diag(observed) - np.diag(component)
    diagonal = np.maximum(diagonal_raw, 0.0)
    model_covariance = component.copy()
    np.fill_diagonal(model_covariance, np.diag(component) + diagonal)
    comparisons = []
    for row in converged:
        comparisons.append({
            "start": int(row.start),
            "model_normalized_difference": float(np.linalg.norm(row.fitted_offdiag - fitted) / max(float(np.linalg.norm(fitted)), EPS)),
            "global_loading_cosine": absolute_cosine(row.global_loading, selected.global_loading),
        })
    multistart_pass = bool(
        len(converged) >= 4
        and all(row["model_normalized_difference"] <= 1e-5 for row in comparisons)
        and all(row["global_loading_cosine"] >= 0.999 for row in comparisons)
    )
    return JointFitResult(
        global_loading=selected.global_loading.copy(),
        group_loading=selected.group_loading.copy(),
        fitted_offdiag=fitted.copy(),
        model_covariance=model_covariance,
        relative_offdiag_misfit=offdiag_relative_misfit(observed, fitted),
        objective=_objective(observed, fitted),
        converged=bool(len(converged) >= 4),
        converged_starts=int(len(converged)),
        selected_start=int(selected.start),
        starts=start_results,
        multistart_audit={
            "status": "PASS" if multistart_pass else "BLOCKED",
            "required_converged_starts": 4,
            "converged_starts": int(len(converged)),
            "comparisons_to_selected": comparisons,
        },
        jacobian_audit=_profiled_jacobian_audit(selected.global_loading, selected.group_loading, group_mask),
        diagonal_audit={
            "raw_residual": diagonal_raw,
            "clipped_count": int(np.sum(diagonal_raw < 0.0)),
            "clipped_mass": float(-np.minimum(diagonal_raw, 0.0).sum()),
        },
    )


def hard_lsml_misfit(covariance: np.ndarray, labels: Sequence[int]) -> dict[str, Any]:
    observed = np.asarray(covariance, dtype=np.float64)
    groups = canonicalize_labels(labels)
    same = groups[:, None] == groups[None, :]
    within = np.zeros(len(observed), dtype=np.float64)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        block = observed[np.ix_(index, index)]
        within[index] = np.asarray(_rank1_masked(block, np.eye(len(index), dtype=bool), scale="complete"), dtype=np.float64)
    between = np.asarray(_rank1_masked(observed, same, scale="complete"), dtype=np.float64)
    fitted = np.where(same, np.outer(within, within), np.outer(between, between))
    np.fill_diagonal(fitted, 0.0)
    return {
        "relative_offdiag_misfit": offdiag_relative_misfit(observed, fitted),
        "objective": _objective(observed, fitted),
        "fitted_offdiag": fitted,
        "within_loading": within,
        "between_loading": between,
    }


def _anchor_orient_weight(values: np.ndarray, weight: np.ndarray, anchor_index: int) -> tuple[np.ndarray, np.ndarray, float, bool]:
    matrix = np.asarray(values, dtype=np.float64)
    output = np.asarray(weight, dtype=np.float64).copy()
    score = matrix @ output
    correlation = score_spearman(score, matrix[:, int(anchor_index)])
    flipped = bool(np.isfinite(correlation) and correlation < 0.0)
    if flipped:
        output *= -1.0
        score *= -1.0
        correlation *= -1.0
    return output, score, float(correlation), flipped


def hierarchical_joint_weights(
    values: np.ndarray,
    labels: Sequence[int],
    global_loading: Sequence[float],
    *,
    anchor_index: int,
    small_m_guard: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float64)
    groups = canonicalize_labels(labels)
    loading = np.asarray(global_loading, dtype=np.float64)
    virtual = []
    indices = []
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        indices.append(index)
        virtual.append(matrix[:, index] @ loading[index])
    virtual_matrix = np.column_stack(virtual)
    _, cross_weight = sml_fuse_signed(
        *[virtual_matrix[:, i] for i in range(virtual_matrix.shape[1])],
        small_m_guard=small_m_guard,
    )
    weight = np.zeros(matrix.shape[1], dtype=np.float64)
    for position, index in enumerate(indices):
        weight[index] = loading[index] * float(cross_weight[position])
    weight, score, anchor_rho, flipped = _anchor_orient_weight(matrix, weight, anchor_index)
    return score, weight, {
        "cross_group_weights": np.asarray(cross_weight, dtype=np.float64),
        "virtual_classifier_count": int(virtual_matrix.shape[1]),
        "anchor_spearman": anchor_rho,
        "anchor_flipped": flipped,
        "uses_covariance_inverse": False,
        "cross_small_m_guarded": bool(small_m_guard and virtual_matrix.shape[1] == 3),
    }


def continuous_lsml_weight_vector(meta: Mapping[str, Any], size: int) -> np.ndarray:
    weight = np.zeros(int(size), dtype=np.float64)
    cross = np.asarray(meta["cross_weights"], dtype=np.float64)
    for position, (indices, within) in enumerate(meta["group_weights"]):
        weight[np.asarray(indices, dtype=np.int64)] = np.asarray(within, dtype=np.float64) * float(cross[position])
    return weight


def weight_maps(
    values: np.ndarray,
    covariance: np.ndarray,
    labels: Sequence[int],
    fit: JointFitResult,
    *,
    anchor_index: int,
    target_condition: float = 1e3,
) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    hierarchical_score, hierarchical_weight, hierarchical_meta = hierarchical_joint_weights(
        matrix, labels, fit.global_loading, anchor_index=anchor_index
    )
    model_weight, model_diagnostics = regularized_covariance_weights(
        fit.model_covariance, fit.global_loading, target_condition=float(target_condition)
    )
    model_weight, model_score, model_anchor, model_flip = _anchor_orient_weight(matrix, model_weight, anchor_index)
    sample_weight, sample_diagnostics = regularized_covariance_weights(
        covariance, fit.global_loading, target_condition=float(target_condition)
    )
    sample_weight, sample_score, sample_anchor, sample_flip = _anchor_orient_weight(matrix, sample_weight, anchor_index)
    reference_score, reference_meta = lsml_continuous(
        *[matrix[:, index] for index in range(matrix.shape[1])],
        groups=canonicalize_labels(labels), compute_score_matrix=False,
    )
    reference_weight = continuous_lsml_weight_vector(reference_meta, matrix.shape[1])
    reference_weight, reference_score_oriented, reference_anchor, reference_flip = _anchor_orient_weight(
        matrix, reference_weight, anchor_index
    )
    if not np.allclose(reference_score_oriented, np.asarray(reference_score) * (-1.0 if reference_flip else 1.0), atol=1e-10, rtol=1e-10):
        raise RuntimeError("continuous L-SML weight reconstruction drift")
    scores = {
        "hierarchical_joint": hierarchical_score,
        "model_inverse_1e3": model_score,
        "sample_inverse_1e3": sample_score,
        "continuous_lsml_reference": reference_score_oriented,
    }
    weights = {
        "hierarchical_joint": hierarchical_weight,
        "model_inverse_1e3": model_weight,
        "sample_inverse_1e3": sample_weight,
        "continuous_lsml_reference": reference_weight,
    }
    return {
        "scores": scores,
        "weights": weights,
        "pairwise_score_spearman": pairwise_score_spearman(scores),
        "diagnostics": {
            "hierarchical_joint": hierarchical_meta,
            "model_inverse_1e3": {**model_diagnostics, "anchor_spearman": model_anchor, "anchor_flipped": model_flip},
            "sample_inverse_1e3": {**sample_diagnostics, "anchor_spearman": sample_anchor, "anchor_flipped": sample_flip},
            "continuous_lsml_reference": {
                "K": int(reference_meta["K"]), "anchor_spearman": reference_anchor,
                "anchor_flipped": reference_flip,
            },
        },
    }


def dispatch_alias(
    values: np.ndarray,
    labels: Sequence[int],
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray | None, Mapping[str, Any]]:
    """Explicit compatibility aliases; no estimator-equivalence claim."""
    matrix = np.asarray(values, dtype=np.float64)
    groups = canonicalize_labels(labels)
    if mode == "flat_sml" and len(np.unique(groups)) == 1:
        score, weight = sml_fuse_signed(*[matrix[:, index] for index in range(matrix.shape[1])])
        return score, weight, {"dispatch": "sml_fuse_signed", "bit_exact_alias": True}
    if mode == "two_stage_alias":
        score, meta = lsml_continuous(
            *[matrix[:, index] for index in range(matrix.shape[1])],
            groups=groups, compute_score_matrix=False,
        )
        return score, None, {"dispatch": "lsml_continuous", "bit_exact_alias": True, "meta": meta}
    raise ValueError("unsupported alias request")


# ── Joint L-SML optimization v2 wrappers ─────────────────────────────────────
# New entry points for the v2 study (docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md).
# They compose the frozen estimators above; fit_joint_lsml internals are untouched.


def _validated_gates(gates: Sequence[float], size: int) -> np.ndarray:
    vector = np.asarray(gates, dtype=np.float64)
    if vector.shape != (int(size),) or not np.isfinite(vector).all() or np.any(vector < 0.0):
        raise ValueError("gates must be a finite nonnegative vector matching the stream count")
    return vector


def effective_gates(gates: Sequence[float], lam: float, size: int) -> np.ndarray:
    """q_lambda = (1-lambda)*1 + lambda*q — lambda=0 is exactly all-ones."""
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must lie in [0, 1]")
    q = _validated_gates(gates, size)
    return (1.0 - lam) * np.ones(int(size), dtype=np.float64) + lam * q


def gated_joint_hierarchical_fit(
    values: np.ndarray,
    labels: Sequence[int],
    gates_effective: Sequence[float],
    *,
    anchor_index: int,
    seed: int,
    small_m_guard: bool = False,
) -> tuple[np.ndarray, JointFitResult, dict[str, Any]]:
    """Hook-2 congruence on the Joint derivation (v2 rows R8/R9).

    The joint factor model is fitted on the congruence-transformed covariance
    of the gated coordinates X·diag(q), the hierarchical head (virtual group
    classifiers + cross-group SML) runs entirely in those coordinates, and the
    final weight is pulled back through diag(q) so that
    values @ w == (values·diag(q)) @ w_gated. All-ones gates reproduce the
    ungated path verbatim (the lambda=0 exact-identity contract).
    """
    matrix = np.asarray(values, dtype=np.float64)
    q = _validated_gates(gates_effective, matrix.shape[1])
    gated = matrix * q[None, :]
    covariance = covariance_matrix(gated)
    fit = fit_joint_lsml(covariance, labels, anchor_index=int(anchor_index), seed=int(seed))
    _, weight_gated, meta = hierarchical_joint_weights(
        gated, labels, fit.global_loading, anchor_index=int(anchor_index),
        small_m_guard=small_m_guard,
    )
    weight = q * np.asarray(weight_gated, dtype=np.float64)
    return weight, fit, {**dict(meta), "gates_applied": bool(not np.array_equal(q, np.ones(len(q))))}


def regularized_joint_map_weights(
    fit_values: np.ndarray,
    model_covariance: np.ndarray,
    global_loading: np.ndarray,
    *,
    mode: str,
    lam: float,
    gates: Sequence[float] | None = None,
    graph: Any | None = None,
    graph_k: int = 7,
    target_condition: float = 1e3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hook-3a ('liu') / Hook-3b ('diag') regularized joint model-inverse maps.

    'liu' reproduces the DUFS-LIU mechanism verbatim on the joint model
    covariance: DUFS-gated sample kNN graph -> symmetric normalized Laplacian
    -> feature-space roughness R = Z^T L Z / n, trace-matched to the model
    covariance, added before the analytic-ridge solve. 'diag' adds a
    trace-matched per-feature Tikhonov prior diag(1/q^2) — a distinct, cheaper
    mechanism (labeled as such, never as a LIU port). lambda=0 is an exact
    identity to the ungated model-inverse map for both modes.
    """
    from .dependency_fusion import regularized_covariance_weights
    from .laplacian_upcr import build_graph_from_features, symmetric_normalized_laplacian

    lam = float(lam)
    if lam < 0.0 or not np.isfinite(lam):
        raise ValueError("lambda must be finite and nonnegative")
    covariance = np.asarray(model_covariance, dtype=np.float64)
    extra: dict[str, Any] = {"mode": str(mode), "lambda": lam}
    if lam == 0.0:
        system = covariance
    elif mode == "liu":
        matrix = np.asarray(fit_values, dtype=np.float64)
        if graph is None:
            graph = build_graph_from_features(
                matrix.T, gates=None if gates is None else _validated_gates(gates, matrix.shape[1]),
                k=int(graph_k),
            )
        laplacian = symmetric_normalized_laplacian(graph)
        roughness = np.asarray(matrix.T @ (laplacian @ matrix), dtype=np.float64) / len(matrix)
        roughness = 0.5 * (roughness + roughness.T)
        trace_r = float(np.trace(roughness))
        scale = float(np.trace(covariance)) / trace_r if trace_r > EPS else 0.0
        system = covariance + lam * scale * roughness
        extra.update({
            "roughness_trace": trace_r,
            "trace_match_scale": scale,
            "graph_k": int(graph_k),
        })
    elif mode == "diag":
        q = _validated_gates(gates, covariance.shape[0])
        floored = np.maximum(q, 1e-6)
        prior = np.diag(1.0 / floored**2)
        trace_d = float(np.trace(prior))
        scale = float(np.trace(covariance)) / trace_d if trace_d > EPS else 0.0
        system = covariance + lam * scale * prior
        extra.update({"prior_trace": trace_d, "trace_match_scale": scale})
    else:
        raise ValueError("mode must be 'liu' or 'diag'")
    weight, diagnostics = regularized_covariance_weights(
        system, np.asarray(global_loading, dtype=np.float64),
        target_condition=float(target_condition),
    )
    return np.asarray(weight, dtype=np.float64), {**dict(diagnostics), **extra}


__all__ = [
    "JointFitResult", "JointStartResult", "absolute_cosine", "canonicalize_labels",
    "coassignment_from_labels", "consensus_orientation_and_roster", "covariance_matrix",
    "discover_loao_consensus_groups", "dispatch_alias", "effective_gates", "fit_joint_lsml",
    "gated_joint_hierarchical_fit", "global_degree_roster", "hard_lsml_misfit",
    "hierarchical_joint_weights", "leading_offdiag_loading", "offdiag_relative_misfit",
    "pairwise_score_spearman", "raw_orientation_cell", "regularized_joint_map_weights",
    "score_spearman", "weight_maps",
]
