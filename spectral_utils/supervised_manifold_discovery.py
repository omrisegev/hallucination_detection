"""Supervised, nuisance-controlled metric discovery primitives.

This module is intentionally separate from DUFS and the label-free scoring
stack.  Labels are used to fit a diagonal metric on donor environments.  The
metric must then be evaluated on held environments and against a
search-matched linear-score graph.
"""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from .graph_topology import extended_graph_diagnostics, self_safe_knn_graph
from .laplacian_upcr import symmetric_normalized_laplacian


FIT_SEEDS = (17, 29, 43)
TIE_SEEDS = (101, 211, 307)
K_GRID = (3, 5, 7, 10, 15, 25)
K_SENSITIVITY = (5, 7, 10, 15)
SUPPORT_SIZES = (5, 10, 15, None)
HEALTH_LARGEST_COMPONENT_MIN = 0.90
HEALTH_ISOLATED_MAX = 0.05


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32)


def _validated_binary(target: np.ndarray) -> np.ndarray:
    values = np.asarray(target, dtype=int)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError("target must be a binary vector")
    if len(np.unique(values)) != 2:
        raise ValueError("target must contain both classes")
    return values


def _validated_length(length: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(length, dtype=float)
    if values.shape != (n,):
        raise ValueError("length must match target rows")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("length has no finite values")
    return np.where(finite, values, np.median(values[finite]))


def cubic_length_basis(length: np.ndarray) -> np.ndarray:
    """Return a bounded, standardized cubic basis of held answer length."""

    values = np.log1p(np.maximum(np.asarray(length, dtype=float), 0.0))
    center = float(np.mean(values))
    scale = float(np.std(values))
    standardized = (values - center) / max(scale, 1e-8)
    standardized = np.clip(standardized, -5.0, 5.0)
    return np.column_stack((standardized, standardized**2, standardized**3))


def cross_fitted_length_residual(
    target: np.ndarray,
    length: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
) -> np.ndarray:
    """Cross-fitted target residual after a frozen cubic ridge length model."""

    labels = _validated_binary(target)
    held_length = _validated_length(length, len(labels))
    basis = cubic_length_basis(held_length)
    class_counts = np.bincount(labels, minlength=2)
    n_splits = int(min(int(folds), int(np.min(class_counts))))
    if n_splits < 2:
        return labels.astype(float) - float(np.mean(labels))
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(seed)
    )
    prediction = np.full(len(labels), np.nan, dtype=float)
    for train, test in splitter.split(basis, labels):
        model = Ridge(alpha=1.0)
        model.fit(basis[train], labels[train])
        prediction[test] = model.predict(basis[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError("cross-fitted length prediction is non-finite")
    return labels.astype(float) - np.clip(prediction, 0.01, 0.99)


def feature_relevance(
    matrix: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Absolute feature association with target after cross-fitted length fit."""

    samples = np.asarray(matrix, dtype=float)
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("matrix must be a finite samples-by-features array")
    residual = cross_fitted_length_residual(target, length, seed=seed)
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    feature_scale = np.sqrt(np.mean(centered**2, axis=0))
    target_scale = float(np.sqrt(np.mean(residual**2)))
    covariance = np.mean(centered * residual[:, None], axis=0)
    denominator = np.maximum(feature_scale * max(target_scale, 1e-12), 1e-12)
    return np.abs(covariance / denominator)


def _bayesian_weights(count: int, *, seed: int, namespace: str) -> np.ndarray:
    if count < 1:
        raise ValueError("weight count must be positive")
    rng = np.random.default_rng(stable_seed(namespace, seed, count))
    weights = rng.exponential(scale=1.0, size=count)
    return weights / np.sum(weights)


def fit_diagonal_metric(
    cells: list[dict],
    *,
    seed: int,
    targets: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Fit one simplex-normalized relevance vector with equal-family structure."""

    if not cells:
        raise ValueError("at least one donor cell is required")
    dimensions = {np.asarray(cell["X"]).shape[1] for cell in cells}
    if len(dimensions) != 1:
        raise ValueError("donor matrices must share a feature dimension")
    families = sorted({str(cell["family"]) for cell in cells})
    family_weights = _bayesian_weights(
        len(families), seed=seed, namespace="metric-family"
    )
    aggregate = np.zeros(next(iter(dimensions)), dtype=float)
    for family_index, family in enumerate(families):
        members = [cell for cell in cells if str(cell["family"]) == family]
        cell_weights = _bayesian_weights(
            len(members), seed=seed, namespace=f"metric-cell|{family}"
        )
        family_relevance = np.zeros_like(aggregate)
        for cell_weight, cell in zip(cell_weights, members):
            identifier = str(cell["cell"])
            target = (
                np.asarray(targets[identifier], dtype=int)
                if targets is not None
                else np.asarray(cell["y"], dtype=int)
            )
            current = feature_relevance(
                cell["X"],
                target,
                cell["length"],
                seed=stable_seed("relevance", seed, identifier),
            )
            family_relevance += float(cell_weight) * current
        aggregate += float(family_weights[family_index]) * family_relevance
    aggregate = np.maximum(aggregate, 0.0)
    total = float(np.sum(aggregate))
    if total <= 1e-12:
        return np.full_like(aggregate, 1.0 / len(aggregate))
    return aggregate / total


def fit_metric_ensemble(
    cells: list[dict],
    *,
    seeds: tuple[int, ...] = FIT_SEEDS,
    targets: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    members = {
        int(seed): fit_diagonal_metric(cells, seed=int(seed), targets=targets)
        for seed in seeds
    }
    mean = np.mean(list(members.values()), axis=0)
    mean /= np.sum(mean)
    return mean, members


def support_indices(
    weights: np.ndarray,
    feature_names: tuple[str, ...],
    support_size: int | None,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(feature_names),):
        raise ValueError("weights and feature names differ")
    size = len(weights) if support_size is None else min(int(support_size), len(weights))
    names = np.asarray(feature_names, dtype=object)
    order = np.lexsort((names, -weights))
    return np.sort(order[:size])


def metric_matrix(
    matrix: np.ndarray,
    weights: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(matrix, dtype=float)
    indexes = np.asarray(support, dtype=int)
    selected_weights = np.asarray(weights, dtype=float)[indexes]
    if not len(indexes) or np.any(selected_weights < 0):
        raise ValueError("metric support/weights are invalid")
    total = float(np.sum(selected_weights))
    if total <= 1e-12:
        selected_weights = np.ones(len(indexes), dtype=float)
    else:
        selected_weights = selected_weights * len(indexes) / total
    return samples[:, indexes] * np.sqrt(selected_weights)[None, :]


def target_blind_tie_keys(n: int, *, namespace: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("tie", namespace, seed))
    keys = rng.random(int(n))
    if len(np.unique(keys)) != len(keys):
        keys = np.arange(int(n), dtype=float)
    return keys


def graph_is_healthy(diagnostics: dict) -> bool:
    return bool(
        diagnostics["largest_component_fraction"] >= HEALTH_LARGEST_COMPONENT_MIN
        and diagnostics["isolated_fraction"] <= HEALTH_ISOLATED_MAX
    )


def select_label_free_graph(
    samples: np.ndarray,
    *,
    tie_keys: np.ndarray,
    k_grid: tuple[int, ...] = K_GRID,
) -> tuple[csr_matrix | None, dict]:
    """Choose the smallest registered k satisfying graph health without labels."""

    attempts = []
    last_graph = None
    for k in k_grid:
        graph = self_safe_knn_graph(samples, k=int(k), tie_keys=tie_keys)
        diagnostics = extended_graph_diagnostics(graph)
        healthy = graph_is_healthy(diagnostics)
        attempts.append({"k": int(k), "healthy": healthy, **diagnostics})
        last_graph = graph
        if healthy:
            return graph, {
                "eligible": True,
                "selected_k": int(k),
                "attempts": attempts,
                **diagnostics,
            }
    diagnostics = attempts[-1]
    return None, {
        "eligible": False,
        "selected_k": None,
        "attempts": attempts,
        "last_k": int(k_grid[-1]),
        "last_graph_edges": int(last_graph.nnz // 2) if last_graph is not None else 0,
        **{key: value for key, value in diagnostics.items() if key not in {"k", "healthy"}},
    }


def rayleigh(graph: csr_matrix, values: np.ndarray) -> float:
    vector = np.asarray(values, dtype=float)
    vector = vector - float(np.mean(vector))
    denominator = float(vector @ vector)
    if denominator <= 1e-12:
        return float("nan")
    laplacian = symmetric_normalized_laplacian(graph)
    return float(vector @ (laplacian @ vector) / denominator)


def conditional_residual_smoothness(
    graph: csr_matrix,
    target: np.ndarray,
    length: np.ndarray,
    *,
    seed: int,
) -> float:
    """Bounded whole-search statistic; higher means smoother residual target."""

    residual = cross_fitted_length_residual(target, length, seed=seed)
    energy = rayleigh(graph, residual)
    return float(1.0 - energy) if np.isfinite(energy) else float("nan")


def deterministic_subsample(
    n: int,
    *,
    namespace: str,
    max_rows: int,
) -> np.ndarray:
    """A target-blind row subset used only by the expensive maxT reruns."""

    if n <= int(max_rows):
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(stable_seed("subsample", namespace, max_rows))
    return np.sort(rng.choice(n, size=int(max_rows), replace=False))


def balanced_cell_class_weights(cells: list[dict], targets: dict[str, np.ndarray]) -> np.ndarray:
    output = []
    for cell in cells:
        identifier = str(cell["cell"])
        labels = _validated_binary(targets[identifier])
        weights = np.zeros(len(labels), dtype=float)
        for target in (0, 1):
            indexes = labels == target
            weights[indexes] = 1.0 / max(1, int(np.sum(indexes)))
        output.append(weights)
    merged = np.concatenate(output)
    return merged * len(merged) / np.sum(merged)


def fit_balanced_logistic(
    cells: list[dict],
    *,
    weights: np.ndarray,
    support: np.ndarray,
    targets: dict[str, np.ndarray] | None = None,
    seed: int,
) -> LogisticRegression:
    target_map = {
        str(cell["cell"]): (
            np.asarray(targets[str(cell["cell"])] if targets is not None else cell["y"], dtype=int)
        )
        for cell in cells
    }
    matrices = [metric_matrix(cell["X"], weights, support) for cell in cells]
    labels = [target_map[str(cell["cell"])] for cell in cells]
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=int(seed),
    )
    model.fit(
        np.vstack(matrices),
        np.concatenate(labels),
        sample_weight=balanced_cell_class_weights(cells, target_map),
    )
    return model


def median_pairwise_cosine(vectors: list[np.ndarray]) -> float:
    values = []
    for left, right in combinations(vectors, 2):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        values.append(float(left @ right / denominator) if denominator > 1e-12 else 0.0)
    return float(np.median(values)) if values else 1.0


def median_pairwise_jaccard(supports: list[np.ndarray]) -> float:
    values = []
    for left, right in combinations(supports, 2):
        left_set, right_set = set(map(int, left)), set(map(int, right))
        union = left_set | right_set
        values.append(len(left_set & right_set) / len(union) if union else 1.0)
    return float(np.median(values)) if values else 1.0


__all__ = [
    "FIT_SEEDS",
    "HEALTH_ISOLATED_MAX",
    "HEALTH_LARGEST_COMPONENT_MIN",
    "K_GRID",
    "K_SENSITIVITY",
    "SUPPORT_SIZES",
    "TIE_SEEDS",
    "balanced_cell_class_weights",
    "conditional_residual_smoothness",
    "cross_fitted_length_residual",
    "cubic_length_basis",
    "deterministic_subsample",
    "feature_relevance",
    "fit_balanced_logistic",
    "fit_diagonal_metric",
    "fit_metric_ensemble",
    "graph_is_healthy",
    "median_pairwise_cosine",
    "median_pairwise_jaccard",
    "metric_matrix",
    "rayleigh",
    "select_label_free_graph",
    "stable_seed",
    "support_indices",
    "target_blind_tie_keys",
]
