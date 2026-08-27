"""Deterministic graph-topology controls for DUFS/LIU diagnostics.

The deployed graph remains the self-tuning symmetric union-kNN graph in
``laplacian_upcr``.  This module supplies edge-budget-matched radius, adaptive,
and truncated-diffusion alternatives for mechanism audits.  None of the
constructors accepts labels.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .laplacian_upcr import (
    graph_diagnostics,
    symmetric_normalized_laplacian,
)


_EPS = 1e-12


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("samples must have shape (n>=3, d>=1)")
    if not np.isfinite(values).all():
        raise ValueError("samples contain non-finite values")
    return values


def _tie_augmented_samples(
    samples: np.ndarray,
    tie_keys: np.ndarray | None,
    *,
    tie_jitter: float,
) -> tuple[np.ndarray, dict]:
    samples = _validate_samples(samples)
    if tie_keys is None:
        tie_keys = np.arange(len(samples), dtype=float)
    tie_keys = np.asarray(tie_keys, dtype=float)
    if tie_keys.shape != (len(samples),) or not np.isfinite(tie_keys).all():
        raise ValueError("tie_keys must be one finite value per sample")
    if len(np.unique(tie_keys)) != len(tie_keys):
        raise ValueError("tie_keys must be unique")
    centered = tie_keys - np.mean(tie_keys)
    scale = float(np.max(np.abs(centered)))
    centered = centered / (scale if scale > _EPS else 1.0)
    coordinate_scale = float(np.median(np.std(samples, axis=0)))
    if not np.isfinite(coordinate_scale) or coordinate_scale < _EPS:
        coordinate_scale = 1.0
    epsilon = float(tie_jitter) * coordinate_scale
    augmented = np.column_stack([samples, epsilon * centered])
    return augmented, {
        "tie_jitter_relative": float(tie_jitter),
        "tie_jitter_absolute": epsilon,
    }


def _knn_table(
    samples: np.ndarray,
    k: int,
    *,
    tie_keys: np.ndarray | None = None,
    tie_jitter: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, dict]:
    samples = _validate_samples(samples)
    if tie_keys is None:
        tie_keys = np.arange(len(samples), dtype=float)
    tie_keys = np.asarray(tie_keys, dtype=float)
    if tie_keys.shape != (len(samples),) or not np.isfinite(tie_keys).all():
        raise ValueError("tie_keys must be one finite value per sample")
    if len(np.unique(tie_keys)) != len(tie_keys):
        raise ValueError("tie_keys must be unique")
    k = int(max(1, min(int(k), len(samples) - 1)))
    unique_samples, inverse = np.unique(samples, axis=0, return_inverse=True)
    groups = [np.flatnonzero(inverse == group) for group in range(len(unique_samples))]
    query_k = int(min(len(unique_samples), max(2, min(32, 2 * k + 2))))
    model = NearestNeighbors(metric="euclidean").fit(unique_samples)
    while True:
        unique_distances, unique_indices = model.kneighbors(
            unique_samples, n_neighbors=query_k, return_distance=True
        )
        clean_distances = np.empty((len(samples), k), dtype=float)
        clean_indices = np.empty((len(samples), k), dtype=int)
        complete = True
        for row in range(len(samples)):
            group = int(inverse[row])
            candidate_rows = []
            candidate_distances = []
            for distance, candidate_group in zip(
                unique_distances[group], unique_indices[group]
            ):
                members = groups[int(candidate_group)]
                if int(candidate_group) == group:
                    members = members[members != row]
                if len(members):
                    candidate_rows.append(members)
                    candidate_distances.append(
                        np.full(len(members), float(distance), dtype=float)
                    )
            if not candidate_rows:
                complete = False
                break
            row_indices = np.concatenate(candidate_rows)
            row_distances = np.concatenate(candidate_distances)
            order = np.lexsort((tie_keys[row_indices], row_distances))
            if len(order) < k:
                complete = False
                break
            selected = order[:k]
            clean_distances[row] = row_distances[selected]
            clean_indices[row] = row_indices[selected]
            selected_boundary = float(row_distances[selected[-1]])
            if (
                query_k < len(unique_samples)
                and unique_distances[group, -1] <= selected_boundary + 1e-12
            ):
                complete = False
                break
        if complete:
            break
        if query_k == len(unique_samples):
            raise RuntimeError("nearest-neighbour query did not return enough non-self rows")
        query_k = int(min(len(unique_samples), max(query_k + 1, 2 * query_k)))
    counts = np.bincount(inverse)
    return clean_distances, clean_indices, {
        "tie_rule": "distance_then_target_blind_key",
        "tie_jitter_relative": 0.0,
        "tie_jitter_requested_but_not_applied": float(tie_jitter),
        "unique_sample_rows": int(len(unique_samples)),
        "max_exact_duplicate_group": int(np.max(counts)),
        "unique_candidate_k": int(query_k),
    }


def _self_tuning_weights(
    distances: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    denom = sigma[rows] * sigma[cols] + _EPS
    return np.exp(-(np.asarray(distances, dtype=float) ** 2) / denom)


def _positive_distinct_bandwidth(
    samples: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Distance to the kth strictly-positive distinct sample location.

    Duplicate multiplicity must not collapse a local bandwidth to zero.  Each
    exact coordinate is therefore counted once for scale estimation, while the
    graph itself continues to contain every sample row.
    """

    values = _validate_samples(samples)
    unique, inverse = np.unique(values, axis=0, return_inverse=True)
    if len(unique) == 1:
        return np.ones(len(values), dtype=float)
    query_k = int(min(len(unique), max(2, int(k) + 1)))
    distances = NearestNeighbors(
        n_neighbors=query_k, metric="euclidean"
    ).fit(unique).kneighbors(unique, return_distance=True)[0]
    scales = np.empty(len(unique), dtype=float)
    for row in range(len(unique)):
        positive = distances[row][distances[row] > _EPS]
        if not len(positive):
            scales[row] = 1.0
        else:
            scales[row] = positive[min(int(k), len(positive)) - 1]
    return np.maximum(scales[inverse], 1e-8)


def _symmetric_graph(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    *,
    combine: str = "maximum",
) -> csr_matrix:
    directed = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    if combine == "maximum":
        graph = directed.maximum(directed.T).tocsr()
    elif combine == "minimum":
        graph = directed.minimum(directed.T).tocsr()
    elif combine == "mean":
        graph = ((directed + directed.T) * 0.5).tocsr()
    else:
        raise ValueError(f"unknown combine rule: {combine}")
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def self_safe_knn_graph(
    samples: np.ndarray,
    *,
    k: int = 7,
    tie_keys: np.ndarray | None = None,
    tie_jitter: float = 1e-9,
) -> csr_matrix:
    """Self-tuning union-kNN with explicit index-based self removal.

    ``laplacian_upcr.self_tuning_knn_graph`` is retained unchanged for exact
    reproduction of historical scores.  This constructor is the corrected
    topology used by new graph-semantics candidates when duplicate rows make
    the first returned neighbour not necessarily the query row itself.
    """

    samples = _validate_samples(samples)
    n = len(samples)
    k = int(max(1, min(k, n - 1)))
    distances, indices, _ = _knn_table(
        samples, k, tie_keys=tie_keys, tie_jitter=tie_jitter
    )
    sigma = _positive_distinct_bandwidth(samples, k=k)
    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)
    weights = _self_tuning_weights(distances.reshape(-1), rows, cols, sigma)
    return _symmetric_graph(n, rows, cols, weights, combine="maximum")


def _unique_undirected_candidates(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    *,
    reduce: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.minimum(rows, cols).astype(np.int64, copy=False)
    hi = np.maximum(rows, cols).astype(np.int64, copy=False)
    keep = lo != hi
    lo, hi, values = lo[keep], hi[keep], np.asarray(values, dtype=float)[keep]
    keys = lo * np.int64(n) + hi
    order = np.lexsort((values, keys))
    keys, lo, hi, values = keys[order], lo[order], hi[order], values[order]
    starts = np.r_[0, 1 + np.flatnonzero(keys[1:] != keys[:-1])]
    if reduce == "minimum":
        reduced = np.minimum.reduceat(values, starts)
    elif reduce == "maximum":
        reduced = np.maximum.reduceat(values, starts)
    else:
        raise ValueError(f"unknown reduction: {reduce}")
    return lo[starts], hi[starts], reduced


def _duplicate_zero_pairs(
    samples: np.ndarray,
    tie_keys: np.ndarray,
    *,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Materialize only the target-blind first ``limit`` exact-zero pairs."""

    _, inverse = np.unique(samples, axis=0, return_inverse=True)
    groups = []
    total = 0
    for group in range(int(np.max(inverse)) + 1):
        members = np.flatnonzero(inverse == group)
        if len(members) > 1:
            members = members[np.argsort(tie_keys[members], kind="mergesort")]
            groups.append(members)
            total += len(members) * (len(members) - 1) // 2
    heap = []
    for group_index, members in enumerate(groups):
        heapq.heappush(heap, (
            float(tie_keys[members[0]]),
            float(tie_keys[members[1]]),
            group_index,
            0,
            1,
        ))
    rows = []
    cols = []
    while heap and len(rows) < min(int(limit), int(total)):
        _, _, group_index, left, right = heapq.heappop(heap)
        members = groups[group_index]
        rows.append(int(members[left]))
        cols.append(int(members[right]))
        if right + 1 < len(members):
            next_left, next_right = left, right + 1
        elif left + 2 < len(members):
            next_left, next_right = left + 1, left + 2
        else:
            continue
        heapq.heappush(heap, (
            float(tie_keys[members[next_left]]),
            float(tie_keys[members[next_right]]),
            group_index,
            next_left,
            next_right,
        ))
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int), int(total)


def _select_smallest_pairs(
    rows: np.ndarray,
    cols: np.ndarray,
    distances: np.ndarray,
    count: int,
    *,
    tie_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 1 or count > len(distances):
        raise ValueError("requested edge count is outside candidate range")
    low = np.minimum(tie_keys[rows], tie_keys[cols])
    high = np.maximum(tie_keys[rows], tie_keys[cols])
    order = np.lexsort((high, low, distances))[:count]
    return rows[order], cols[order], distances[order]


def _select_largest_pairs(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    count: int,
    *,
    tie_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 1 or count > len(weights):
        raise ValueError("requested edge count is outside candidate range")
    low = np.minimum(tie_keys[rows], tie_keys[cols])
    high = np.maximum(tie_keys[rows], tie_keys[cols])
    order = np.lexsort((high, low, -weights))[:count]
    return rows[order], cols[order], weights[order]


def radius_edge_matched_graph(
    samples: np.ndarray,
    *,
    edge_count: int,
    scale_k: int = 7,
    initial_candidate_k: int = 32,
    max_candidate_k: int = 512,
    tie_keys: np.ndarray | None = None,
    tie_jitter: float = 1e-9,
) -> tuple[csr_matrix, dict]:
    """Return the exact globally shortest-pair graph at a fixed edge budget.

    Candidate kNN tables are expanded until every row's candidate boundary is
    strictly outside the selected radius.  At that point no omitted pair can
    lie inside the radius, so the sparse construction is equivalent to an
    all-pairs radius ordering without materializing an ``n x n`` matrix.
    """

    samples = _validate_samples(samples)
    n = len(samples)
    edge_count = int(edge_count)
    if edge_count < 1 or edge_count > n * (n - 1) // 2:
        raise ValueError("invalid edge_count")
    effective_tie_keys = (
        np.arange(n, dtype=float) if tie_keys is None else np.asarray(tie_keys, dtype=float)
    )
    zero_rows, zero_cols, zero_pair_total = _duplicate_zero_pairs(
        samples, effective_tie_keys, limit=edge_count
    )
    candidate_k = int(max(scale_k, min(initial_candidate_k, n - 1)))
    if candidate_k > int(max_candidate_k):
        raise RuntimeError("initial radius candidate-k exceeds the registered memory cap")
    while True:
        distances, indices, tie_diagnostics = _knn_table(
            samples,
            candidate_k,
            tie_keys=tie_keys,
            tie_jitter=tie_jitter,
        )
        rows = np.repeat(np.arange(n), candidate_k)
        cols = indices.reshape(-1)
        flat = distances.reshape(-1)
        undirected_rows, undirected_cols, undirected_distances = (
            _unique_undirected_candidates(
                n, rows, cols, flat, reduce="minimum"
            )
        )
        if len(zero_rows):
            undirected_rows, undirected_cols, undirected_distances = (
                _unique_undirected_candidates(
                    n,
                    np.concatenate([undirected_rows, zero_rows]),
                    np.concatenate([undirected_cols, zero_cols]),
                    np.concatenate([
                        undirected_distances,
                        np.zeros(len(zero_rows), dtype=float),
                    ]),
                    reduce="minimum",
                )
            )
        if len(undirected_distances) < edge_count:
            complete = False
        else:
            selected = np.partition(undirected_distances, edge_count - 1)[edge_count - 1]
            complete = bool(
                (selected <= 1e-12 and zero_pair_total >= edge_count)
                or np.min(distances[:, -1]) > selected + 1e-12
            )
        if complete or candidate_k == n - 1:
            break
        next_k = int(min(n - 1, max(candidate_k + 1, candidate_k * 2)))
        if next_k > int(max_candidate_k):
            raise RuntimeError(
                "radius boundary not proven within the registered candidate-k memory cap"
            )
        candidate_k = next_k
    if len(undirected_distances) < edge_count:
        raise RuntimeError("candidate expansion did not produce enough pairs")
    rows, cols, chosen_distances = _select_smallest_pairs(
        undirected_rows,
        undirected_cols,
        undirected_distances,
        edge_count,
        tie_keys=effective_tie_keys,
    )
    sigma = _positive_distinct_bandwidth(samples, k=scale_k)
    weights = _self_tuning_weights(chosen_distances, rows, cols, sigma)
    graph = _symmetric_graph(n, rows, cols, weights, combine="maximum")
    if graph.nnz // 2 != edge_count:
        raise RuntimeError("radius graph failed exact edge-budget invariant")
    cutoff = float(np.max(chosen_distances))
    return graph, {
        "edge_budget": edge_count,
        "radius": cutoff,
        "candidate_k": candidate_k,
        "candidate_boundary_min": float(np.min(distances[:, -1])),
        "candidate_boundary_proven": bool(
            (cutoff <= 1e-12 and zero_pair_total >= edge_count)
            or candidate_k == n - 1
            or np.min(distances[:, -1]) > cutoff + 1e-12
        ),
        "max_candidate_k": int(max_candidate_k),
        "exact_zero_pairs_total": int(zero_pair_total),
        "exact_zero_pairs_materialized": int(len(zero_rows)),
        **tie_diagnostics,
    }


def adaptive_neighbor_counts(
    local_scale: np.ndarray,
    *,
    mean_k: int = 7,
    min_k: int = 3,
    max_k: int = 25,
    rank_power: float = 8.0,
    tie_keys: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate exact mean degree with more directed neighbours in sparse areas."""

    scale = np.asarray(local_scale, dtype=float)
    if scale.ndim != 1 or len(scale) < 3 or not np.isfinite(scale).all():
        raise ValueError("local_scale must be a finite one-dimensional array")
    if not 1 <= min_k <= mean_k <= max_k < len(scale):
        raise ValueError("k bounds must satisfy 1 <= min <= mean <= max < n")
    if tie_keys is None:
        tie_keys = np.arange(len(scale), dtype=float)
    tie_keys = np.asarray(tie_keys, dtype=float)
    if tie_keys.shape != scale.shape or len(np.unique(tie_keys)) != len(scale):
        raise ValueError("tie_keys must be unique and match local_scale")
    target_extra = int((mean_k - min_k) * len(scale))
    capacity = int(max_k - min_k)
    percentiles = (rankdata(scale, method="average") - 0.5) / len(scale)
    weights = np.maximum(percentiles, 1e-12) ** float(rank_power)
    allocation = np.zeros(len(scale), dtype=float)
    active = np.ones(len(scale), dtype=bool)
    remaining = float(target_extra)
    while remaining > 1e-9 and np.any(active):
        active_weights = weights[active]
        quota = remaining * active_weights / np.sum(active_weights)
        active_indexes = np.flatnonzero(active)
        saturated = quota >= capacity - allocation[active_indexes] - 1e-12
        if not np.any(saturated):
            allocation[active_indexes] += quota
            remaining = 0.0
            break
        saturated_indexes = active_indexes[saturated]
        additions = capacity - allocation[saturated_indexes]
        allocation[saturated_indexes] = capacity
        remaining -= float(np.sum(additions))
        active[saturated_indexes] = False
    integer = np.floor(allocation + 1e-12).astype(int)
    remainder = target_extra - int(np.sum(integer))
    if remainder:
        fractional = allocation - integer
        eligible = integer < capacity
        order = np.lexsort((tie_keys, -weights, -fractional))
        for index in order:
            if remainder == 0:
                break
            if eligible[index]:
                integer[index] += 1
                remainder -= 1
    counts = min_k + integer
    if int(np.sum(counts)) != int(mean_k * len(scale)):
        raise RuntimeError("adaptive allocation failed exact mean invariant")
    if np.min(counts) < min_k or np.max(counts) > max_k:
        raise RuntimeError("adaptive allocation violated k bounds")
    return counts.astype(int)


def adaptive_knn_graph(
    samples: np.ndarray,
    *,
    mean_k: int = 7,
    min_k: int = 3,
    max_k: int = 25,
    scale_k: int = 7,
    rank_power: float = 8.0,
    tie_keys: np.ndarray | None = None,
    tie_jitter: float = 1e-9,
) -> tuple[csr_matrix, dict]:
    samples = _validate_samples(samples)
    n = len(samples)
    max_k = int(min(max_k, n - 1))
    min_k = int(min(min_k, max_k))
    mean_k = int(min(max(mean_k, min_k), max_k))
    lookup_k = max(max_k, scale_k)
    distances, indices, tie_diagnostics = _knn_table(
        samples, lookup_k, tie_keys=tie_keys, tie_jitter=tie_jitter
    )
    sigma = _positive_distinct_bandwidth(samples, k=scale_k)
    counts = adaptive_neighbor_counts(
        sigma,
        mean_k=mean_k,
        min_k=min_k,
        max_k=max_k,
        rank_power=rank_power,
        tie_keys=tie_keys,
    )
    rows = np.concatenate([
        np.full(int(count), index, dtype=int) for index, count in enumerate(counts)
    ])
    cols = np.concatenate([
        indices[index, : int(count)] for index, count in enumerate(counts)
    ])
    selected_distances = np.concatenate([
        distances[index, : int(count)] for index, count in enumerate(counts)
    ])
    weights = _self_tuning_weights(selected_distances, rows, cols, sigma)
    graph = _symmetric_graph(n, rows, cols, weights, combine="maximum")
    rho = float(np.corrcoef(sigma, counts)[0, 1])
    return graph, {
        "directed_k_min": int(np.min(counts)),
        "directed_k_mean": float(np.mean(counts)),
        "directed_k_max": int(np.max(counts)),
        "directed_k_sum": int(np.sum(counts)),
        "local_scale_k_correlation": rho,
        "rank_power": float(rank_power),
        **tie_diagnostics,
    }


def mutual_knn_graph(
    samples: np.ndarray,
    *,
    k: int = 7,
    tie_keys: np.ndarray | None = None,
    tie_jitter: float = 1e-9,
) -> csr_matrix:
    samples = _validate_samples(samples)
    n = len(samples)
    k = int(max(1, min(k, n - 1)))
    distances, indices, _ = _knn_table(
        samples, k, tie_keys=tie_keys, tie_jitter=tie_jitter
    )
    sigma = _positive_distinct_bandwidth(samples, k=k)
    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)
    weights = _self_tuning_weights(distances.reshape(-1), rows, cols, sigma)
    return _symmetric_graph(n, rows, cols, weights, combine="minimum")


def _row_topk(
    matrix: csr_matrix,
    k: int,
    *,
    tie_keys: np.ndarray | None = None,
) -> csr_matrix:
    matrix = csr_matrix(matrix, dtype=float)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    if tie_keys is None:
        tie_keys = np.arange(matrix.shape[0], dtype=float)
    tie_keys = np.asarray(tie_keys, dtype=float)
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        values = matrix.data[start:stop]
        indexes = matrix.indices[start:stop]
        if len(values) > k:
            # Stable target-blind tie resolution.  ``argpartition`` alone can
            # select arbitrary members of a boundary tie.
            keep = np.lexsort((tie_keys[indexes], -values))[:k]
            values, indexes = values[keep], indexes[keep]
        if len(values):
            order = np.lexsort((indexes, -values))
            rows.append(np.full(len(order), row, dtype=int))
            cols.append(indexes[order])
            data.append(values[order])
    if not data:
        return csr_matrix(matrix.shape, dtype=float)
    output = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=matrix.shape,
    ).tocsr()
    output.eliminate_zeros()
    return output


def _row_normalize(matrix: csr_matrix) -> csr_matrix:
    matrix = csr_matrix(matrix, dtype=float)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sum)
    inverse[row_sum > _EPS] = 1.0 / row_sum[row_sum > _EPS]
    output = (diags(inverse) @ matrix).tocsr()
    # Sparse multiplication/row sums can differ at machine epsilon after a
    # pure row permutation because accumulation order changes.  Register a
    # fixed numerical grid before any top-k boundary decision.
    output.data = np.round(output.data, decimals=14)
    output.eliminate_zeros()
    return output


def diffusion_edge_matched_graph(
    samples: np.ndarray,
    *,
    edge_count: int,
    base_k: int = 25,
    steps: int = 2,
    row_keep: int = 25,
    tie_keys: np.ndarray | None = None,
    tie_jitter: float = 1e-9,
) -> tuple[csr_matrix, dict]:
    """Build a truncated multi-step diffusion graph at an exact edge budget."""

    samples = _validate_samples(samples)
    if steps < 1:
        raise ValueError("steps must be positive")
    base = self_safe_knn_graph(
        samples, k=base_k, tie_keys=tie_keys, tie_jitter=tie_jitter
    )
    transition = _row_normalize(base)
    propagated = transition
    for _ in range(1, int(steps)):
        product = (propagated @ transition).tocsr()
        product.data = np.round(product.data, decimals=14)
        product.eliminate_zeros()
        propagated = _row_topk(product, int(row_keep), tie_keys=tie_keys)
        propagated = _row_normalize(propagated)
    symmetric = propagated.maximum(propagated.T).tocsr()
    symmetric.setdiag(0.0)
    symmetric.eliminate_zeros()
    coo = symmetric.tocoo()
    keep = coo.row < coo.col
    rows, cols, weights = coo.row[keep], coo.col[keep], coo.data[keep]
    if len(weights) < edge_count:
        raise RuntimeError("diffusion truncation produced too few candidate edges")
    effective_tie_keys = (
        np.arange(len(samples), dtype=float)
        if tie_keys is None
        else np.asarray(tie_keys, dtype=float)
    )
    rows, cols, weights = _select_largest_pairs(
        rows,
        cols,
        weights,
        int(edge_count),
        tie_keys=effective_tie_keys,
    )
    graph = _symmetric_graph(len(samples), rows, cols, weights, combine="maximum")
    if graph.nnz // 2 != int(edge_count):
        raise RuntimeError("diffusion graph failed exact edge-budget invariant")
    return graph, {
        "edge_budget": int(edge_count),
        "base_k": int(base_k),
        "steps": int(steps),
        "row_keep": int(row_keep),
        "candidate_edges": int(len(coo.data) // 2),
        "tie_rule": "distance_then_target_blind_key",
        "tie_jitter_relative": 0.0,
        "tie_jitter_requested_but_not_applied": float(tie_jitter),
    }


def length_only_graph(
    length: np.ndarray,
    *,
    k: int = 7,
    tie_keys: np.ndarray | None = None,
    tie_jitter: float = 1e-9,
) -> csr_matrix:
    values = np.asarray(length, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("length has no finite values")
    fill = float(np.median(values[finite]))
    values = np.where(finite, values, fill)
    return self_safe_knn_graph(
        np.log1p(np.maximum(values, 0.0))[:, None],
        k=k,
        tie_keys=tie_keys,
        tie_jitter=tie_jitter,
    )


def extended_graph_diagnostics(graph: csr_matrix) -> dict:
    graph = csr_matrix(graph, dtype=float)
    base = graph_diagnostics(graph)
    weighted_degree = np.asarray(graph.sum(axis=1)).ravel()
    binary_degree = np.diff(graph.indptr).astype(float)
    squared = np.asarray(graph.multiply(graph).sum(axis=1)).ravel()
    effective = np.zeros_like(weighted_degree)
    positive = squared > _EPS
    effective[positive] = weighted_degree[positive] ** 2 / squared[positive]
    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    base.update({
        "all_edge_weights_finite": bool(np.isfinite(graph.data).all()),
        "minimum_edge_weight": float(np.min(graph.data)) if graph.nnz else 0.0,
        "isolated_fraction": float(np.mean(binary_degree == 0)),
        "largest_component_fraction": float(np.max(sizes) / len(labels)),
        "binary_degree_median": float(np.median(binary_degree)),
        "binary_degree_p90": float(np.quantile(binary_degree, 0.90)),
        "weighted_degree_median": float(np.median(weighted_degree)),
        "weighted_degree_p90": float(np.quantile(weighted_degree, 0.90)),
        "effective_neighbors_mean": float(np.mean(effective)),
        "effective_neighbors_median": float(np.median(effective)),
    })
    return base


def _validated_target_length(
    target: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(target, dtype=int)
    values = np.asarray(length, dtype=float)
    if labels.ndim != 1 or len(np.unique(labels)) != 2:
        raise ValueError("target must be a binary one-dimensional array")
    if values.shape != labels.shape:
        raise ValueError("length must match target")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("length has no finite values")
    values = np.where(finite, values, np.median(values[finite]))
    return labels, values


def unconditional_target_permutations(
    target: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(target, dtype=int)
    if labels.ndim != 1 or len(np.unique(labels)) != 2:
        raise ValueError("target must be a binary one-dimensional array")
    rng = np.random.default_rng(int(seed))
    return np.column_stack([
        rng.permutation(labels) for _ in range(int(permutations))
    ])


def exact_length_permutations(
    target: np.ndarray,
    length: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Permute labels only among rows with exactly equal held-out length."""

    labels, values = _validated_target_length(target, length)
    _, groups = np.unique(values, return_inverse=True)
    rng = np.random.default_rng(int(seed))
    draws = np.tile(labels[:, None], (1, int(permutations)))
    movable = np.zeros(len(labels), dtype=bool)
    mixed_groups = 0
    for group in np.unique(groups):
        indexes = np.flatnonzero(groups == group)
        if len(np.unique(labels[indexes])) == 2:
            movable[indexes] = True
            mixed_groups += 1
        for column in range(int(permutations)):
            draws[indexes, column] = labels[indexes][rng.permutation(len(indexes))]
    sizes = np.bincount(groups)
    movable_rows = int(np.sum(movable))
    return draws, {
        "strata": int(len(sizes)),
        "mixed_strata": int(mixed_groups),
        "movable_rows": movable_rows,
        "movable_fraction": float(movable_rows / len(labels)),
        "min_stratum_size": int(np.min(sizes)),
        "max_stratum_size": int(np.max(sizes)),
    }


def propensity_crt_permutations(
    target: np.ndarray,
    length: np.ndarray,
    *,
    permutations: int,
    seed: int,
    folds: int = 5,
) -> tuple[np.ndarray, dict]:
    """Cross-fitted flexible Bernoulli null for ``target | held length``.

    The one-dimensional propensity model is fixed and graph-independent.  OOF
    probabilities prevent the same row's target from directly fitting its own
    conditional null probability.
    """

    labels, values = _validated_target_length(target, length)
    log_length = np.log1p(np.maximum(values, 0.0))[:, None]
    class_counts = np.bincount(labels, minlength=2)
    n_splits = int(min(int(folds), np.min(class_counts)))
    if n_splits < 2:
        raise ValueError("insufficient class count for cross-fitted propensity")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    propensity = np.full(len(labels), np.nan, dtype=float)
    min_leaf = int(max(10, min(50, len(labels) // 50)))
    for fold, (train, test) in enumerate(splitter.split(log_length, labels)):
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=min_leaf,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=int(seed) + fold + 1,
        )
        model.fit(log_length[train], labels[train])
        propensity[test] = model.predict_proba(log_length[test])[:, 1]
    if not np.isfinite(propensity).all():
        raise RuntimeError("cross-fitted propensity contains non-finite values")
    propensity = np.clip(propensity, 0.01, 0.99)
    rng = np.random.default_rng(int(seed) + 1009)
    draws = np.empty((len(labels), int(permutations)), dtype=int)
    all_binary = True
    for column in range(int(permutations)):
        draw = (rng.random(len(labels)) < propensity).astype(int)
        attempts = 0
        while len(np.unique(draw)) < 2 and attempts < 100:
            draw = (rng.random(len(labels)) < propensity).astype(int)
            attempts += 1
        if len(np.unique(draw)) < 2:
            all_binary = False
        draws[:, column] = draw
    brier = float(np.mean((labels - propensity) ** 2))
    baseline = float(np.mean((labels - np.mean(labels)) ** 2))
    bin_id = np.minimum((propensity * 10).astype(int), 9)
    calibration_error = 0.0
    for bin_index in range(10):
        indexes = bin_id == bin_index
        if np.any(indexes):
            calibration_error += float(np.mean(indexes)) * abs(
                float(np.mean(labels[indexes]) - np.mean(propensity[indexes]))
            )
    return draws, {
        "folds": n_splits,
        "min_samples_leaf": min_leaf,
        "overlap_fraction": float(np.mean(
            (propensity >= 0.05) & (propensity <= 0.95)
        )),
        "brier": brier,
        "constant_brier": baseline,
        "calibration_mae": float(calibration_error),
        "propensity_auroc": float(roc_auc_score(labels, propensity)),
        "propensity_min": float(np.min(propensity)),
        "propensity_max": float(np.max(propensity)),
        "all_draws_binary": bool(all_binary),
    }


def matched_pair_permutations(
    target: np.ndarray,
    length: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Target-blind adjacent-length swaps used only as a sensitivity null."""

    labels, values = _validated_target_length(target, length)
    log_length = np.log1p(np.maximum(values, 0.0))
    order = np.argsort(log_length, kind="mergesort")
    pair_count = len(labels) // 2
    left = order[: 2 * pair_count : 2]
    right = order[1 : 2 * pair_count : 2]
    gaps = np.abs(log_length[left] - log_length[right])
    discordant = labels[left] != labels[right]
    rng = np.random.default_rng(int(seed))
    draws = np.tile(labels[:, None], (1, int(permutations)))
    for column in range(int(permutations)):
        swap = rng.integers(0, 2, size=pair_count).astype(bool)
        selected_left = left[swap]
        selected_right = right[swap]
        draws[selected_left, column] = labels[selected_right]
        draws[selected_right, column] = labels[selected_left]
    return draws, {
        "pairs": int(pair_count),
        "unpaired_rows": int(len(labels) - 2 * pair_count),
        "discordant_pairs": int(np.sum(discordant)),
        "movable_rows": int(2 * np.sum(discordant)),
        "movable_fraction": float(2 * np.sum(discordant) / len(labels)),
        "exact_tie_fraction": float(np.mean(gaps == 0.0)),
        "median_log_length_gap": float(np.median(gaps)),
        "p95_log_length_gap": float(np.quantile(gaps, 0.95)),
        "max_log_length_gap": float(np.max(gaps)),
    }


def sample_tie_diagnostics(samples: np.ndarray) -> dict:
    """Report exact duplicate-row exposure before target-blind tie breaking."""

    values = _validate_samples(samples)
    _, counts = np.unique(values, axis=0, return_counts=True)
    tied = counts[counts > 1]
    tied_rows = int(np.sum(tied)) if len(tied) else 0
    return {
        "exact_duplicate_groups": int(len(tied)),
        "exact_duplicate_rows": tied_rows,
        "exact_duplicate_row_fraction": float(tied_rows / len(values)),
        "max_exact_duplicate_group": int(np.max(tied)) if len(tied) else 1,
    }


def target_permutations(
    target: np.ndarray,
    length: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Backward-compatible wrapper: unconditional plus exact-length swaps."""

    raw = unconditional_target_permutations(
        target, permutations=permutations, seed=seed
    )
    exact, diagnostic = exact_length_permutations(
        target, length, permutations=permutations, seed=seed + 1
    )
    return raw, exact, diagnostic


def _rayleigh_columns(laplacian: csr_matrix, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    denominator = np.sum(matrix * matrix, axis=0)
    energy = np.sum(matrix * (laplacian @ matrix), axis=0)
    output = np.full(matrix.shape[1], np.nan, dtype=float)
    valid = denominator > _EPS
    output[valid] = energy[valid] / denominator[valid]
    return output


def smoothness_against_permutations(
    graph: csr_matrix,
    observed: np.ndarray,
    permutations: np.ndarray,
) -> dict:
    laplacian = symmetric_normalized_laplacian(graph)
    observed_value = float(_rayleigh_columns(laplacian, np.asarray(observed)) [0])
    null = _rayleigh_columns(laplacian, np.asarray(permutations))
    null = null[np.isfinite(null)]
    if not len(null):
        return {
            "rayleigh": observed_value,
            "null_mean": float("nan"),
            "null_sd": float("nan"),
            "effect": float("nan"),
            "z": float("nan"),
            "p_smoother": float("nan"),
        }
    mean = float(np.mean(null))
    sd = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
    return {
        "rayleigh": observed_value,
        "null_mean": mean,
        "null_sd": sd,
        "effect": float((mean - observed_value) / mean) if abs(mean) > _EPS else 0.0,
        "z": float((mean - observed_value) / sd) if sd > _EPS else 0.0,
        "p_smoother": float((1 + np.sum(null <= observed_value)) / (len(null) + 1)),
    }


def purity_against_permutations(
    graph: csr_matrix,
    observed: np.ndarray,
    permutations: np.ndarray,
    *,
    batch_size: int = 32,
) -> dict:
    graph = csr_matrix(graph, dtype=float)
    coo = graph.tocoo()
    keep = coo.row < coo.col
    rows, cols, weights = coo.row[keep], coo.col[keep], coo.data[keep]
    total = float(np.sum(weights))
    labels = np.asarray(observed)
    score = float(np.sum(weights * (labels[rows] == labels[cols])) / total)
    permutations = np.asarray(permutations)
    null_parts = []
    for start in range(0, permutations.shape[1], int(batch_size)):
        block = permutations[:, start : start + int(batch_size)]
        matches = block[rows, :] == block[cols, :]
        null_parts.append(np.sum(weights[:, None] * matches, axis=0) / total)
    null = np.concatenate(null_parts)
    mean = float(np.mean(null))
    sd = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
    return {
        "purity": score,
        "null_mean": mean,
        "null_sd": sd,
        "excess": float(score - mean),
        "z": float((score - mean) / sd) if sd > _EPS else 0.0,
        "p_purer": float((1 + np.sum(null >= score)) / (len(null) + 1)),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return output
    # A missing member must never silently shrink the predeclared family.
    if len(finite) != len(values):
        output[finite] = 1.0
        return output
    order = finite[np.argsort(values[finite], kind="mergesort")]
    adjusted_sorted = np.maximum.accumulate(
        np.asarray([(len(order) - rank) * values[index] for rank, index in enumerate(order)])
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    for index, adjusted in zip(order, adjusted_sorted):
        output[index] = adjusted
    return output


__all__ = [
    "adaptive_knn_graph",
    "adaptive_neighbor_counts",
    "diffusion_edge_matched_graph",
    "extended_graph_diagnostics",
    "holm_adjust",
    "exact_length_permutations",
    "length_only_graph",
    "matched_pair_permutations",
    "mutual_knn_graph",
    "propensity_crt_permutations",
    "purity_against_permutations",
    "radius_edge_matched_graph",
    "sample_tie_diagnostics",
    "self_safe_knn_graph",
    "smoothness_against_permutations",
    "target_permutations",
    "unconditional_target_permutations",
]
