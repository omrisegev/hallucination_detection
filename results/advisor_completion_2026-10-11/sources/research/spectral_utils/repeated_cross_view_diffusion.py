"""Repeated complementary-view alternating diffusion for IU-PCR.

The module is intentionally label-free.  It partitions feature coordinates,
builds a sample graph in each complementary view, composes their random-walk
operators, and averages the resulting two-view graphs across deterministic
partitions.  The consensus graph is then passed to the existing
Laplacian-IU-PCR solver.

The dependency-blocked schema uses only absolute rank correlation.  A block is
an anti-leakage unit, not a reliability group and not a selected feature set.
"""

from __future__ import annotations

import hashlib
from itertools import product

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr

from .cross_view_graph import centered_affinity_cka
from .laplacian_upcr import (
    graph_diagnostics,
    laplacian_iu_fit,
    laplacian_iu_path,
    permute_graph,
    self_tuning_knn_graph,
)
from .specrage_views import FEATURE_TO_VIEW, VIEW_ORDER


EPS = 1e-12


def stable_seed(namespace: str) -> int:
    return int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:8], 16)


def _validate(F, feature_names):
    values = np.asarray(F, dtype=float)
    names = tuple(str(name) for name in feature_names)
    if values.ndim != 2 or values.shape[0] != len(names):
        raise ValueError("F must have shape (features, samples)")
    if min(values.shape) < 3 or not np.isfinite(values).all():
        raise ValueError("F must be finite with at least three features and samples")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    return values, names


def _canonical_feature_order(feature_names):
    return np.asarray(sorted(range(len(feature_names)), key=lambda i: feature_names[i]))


def atomic_blocks(F, feature_names):
    """Return one deterministic block per feature."""
    _, names = _validate(F, feature_names)
    return tuple((int(index),) for index in _canonical_feature_order(names))


def dependency_blocks(F, feature_names, *, distance_threshold=0.15):
    """Complete-linkage blocks from absolute Spearman rank correlation.

    Complete linkage prevents a transitive chain of moderate correlations from
    merging into one large block.  Feature order is canonicalized by name
    before clustering so column permutation cannot change tie handling.
    """
    values, names = _validate(F, feature_names)
    threshold = float(distance_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("distance_threshold must be in (0, 1)")
    order = _canonical_feature_order(names)
    ranked = np.asarray([
        rankdata(values[index], method="average") for index in order
    ], dtype=float)
    correlation = np.corrcoef(ranked)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    distance = np.clip(1.0 - np.abs(correlation), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="complete")
    labels = fcluster(tree, t=threshold, criterion="distance")
    blocks = []
    for label in sorted(set(labels)):
        members = tuple(sorted((int(order[i]) for i in np.flatnonzero(labels == label)),
                               key=lambda i: names[i]))
        blocks.append(members)
    return tuple(sorted(blocks, key=lambda block: tuple(names[i] for i in block)))


def family_blocks(F, feature_names):
    """Return frozen provenance families as indivisible anti-leakage blocks."""
    _, names = _validate(F, feature_names)
    unknown = sorted(set(names) - set(FEATURE_TO_VIEW))
    if unknown:
        raise KeyError("unregistered feature(s): " + ", ".join(unknown))
    blocks = []
    for family in VIEW_ORDER:
        members = tuple(sorted(
            (index for index, name in enumerate(names) if FEATURE_TO_VIEW[name] == family),
            key=lambda i: names[i],
        ))
        if members:
            blocks.append(members)
    return tuple(blocks)


def _mask_digest(mask, seed):
    token = f"{int(seed)}:" + "".join(map(str, mask))
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def balanced_partitions(
    blocks,
    n_features,
    *,
    count=16,
    min_fraction=0.30,
    seed=0,
):
    """Return unique balanced complementary block assignments.

    The first block is fixed to the left side because swapping both sides does
    not change alternating diffusion.  When there are at most 16 blocks every
    assignment is enumerated.  Larger schemas use a deterministic random
    candidate pool.  The most balanced candidates are retained with a hash
    tie-breaker.
    """
    blocks = tuple(tuple(map(int, block)) for block in blocks)
    n_features = int(n_features)
    count = int(count)
    min_fraction = float(min_fraction)
    if len(blocks) < 2 or count < 1:
        raise ValueError("at least two blocks and one partition are required")
    flat = sorted(index for block in blocks for index in block)
    if flat != list(range(n_features)):
        raise ValueError("blocks must partition every feature exactly once")
    minimum = int(np.ceil(min_fraction * n_features))
    if minimum < 1 or 2 * minimum > n_features:
        raise ValueError("min_fraction leaves no feasible complementary views")

    candidates = set()
    block_count = len(blocks)
    if block_count <= 16:
        iterator = ((0,) + bits for bits in product((0, 1), repeat=block_count - 1))
        for mask in iterator:
            left_size = sum(len(block) for bit, block in zip(mask, blocks) if bit == 0)
            if min(left_size, n_features - left_size) >= minimum:
                candidates.add(tuple(mask))
    else:
        rng = np.random.default_rng(int(seed))
        attempts = max(20_000, count * 1_000)
        for _ in range(attempts):
            mask = rng.integers(0, 2, size=block_count, dtype=np.int8)
            if mask[0] == 1:
                mask = 1 - mask
            left_size = sum(
                len(block) for bit, block in zip(mask, blocks) if int(bit) == 0
            )
            if min(left_size, n_features - left_size) >= minimum:
                candidates.add(tuple(map(int, mask)))
            if len(candidates) >= max(5_000, count * 100):
                break
    if not candidates:
        raise RuntimeError("no feasible balanced block assignment")

    def key(mask):
        left_size = sum(len(block) for bit, block in zip(mask, blocks) if bit == 0)
        return (abs(n_features - 2 * left_size), _mask_digest(mask, seed))

    selected = sorted(candidates, key=key)[:count]
    partitions = []
    for mask in selected:
        left = tuple(sorted(index for bit, block in zip(mask, blocks)
                            if bit == 0 for index in block))
        right = tuple(sorted(index for bit, block in zip(mask, blocks)
                             if bit == 1 for index in block))
        partitions.append({"left": left, "right": right, "mask": tuple(mask)})
    return tuple(partitions)


def row_stochastic(graph):
    W = csr_matrix(graph, dtype=float)
    degree = np.asarray(W.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    positive = degree > EPS
    inverse[positive] = 1.0 / degree[positive]
    return (diags(inverse) @ W).tocsr()


def topk_symmetric(matrix, *, k=7):
    """Symmetrize and retain the deterministic largest positive entries/row."""
    values = csr_matrix(matrix, dtype=float)
    if values.shape[0] != values.shape[1] or values.shape[0] < 3:
        raise ValueError("matrix must be square with at least three rows")
    values = (0.5 * (values + values.T)).tocsr()
    values.setdiag(0.0)
    values.eliminate_zeros()
    if values.nnz:
        values.data = np.maximum(values.data, 0.0)
        values.eliminate_zeros()
    n = values.shape[0]
    k = int(max(1, min(int(k), n - 1)))
    rows, cols, data = [], [], []
    for row in range(n):
        start, stop = values.indptr[row], values.indptr[row + 1]
        local_columns = values.indices[start:stop]
        local_values = values.data[start:stop]
        if len(local_values) > k:
            chosen = np.lexsort((local_columns, -local_values))[:k]
            local_columns = local_columns[chosen]
            local_values = local_values[chosen]
        keep = local_values > EPS
        local_columns = local_columns[keep]
        local_values = local_values[keep]
        rows.extend([row] * len(local_values))
        cols.extend(local_columns.tolist())
        data.extend(local_values.tolist())
    directed = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    graph = directed.maximum(directed.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def alternating_pair_graph(F, left, right, *, k=7, return_direct=False):
    """Build one complementary-view alternating-diffusion graph."""
    values = np.asarray(F, dtype=float)
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    if set(left).intersection(set(right)) or len(left) + len(right) != values.shape[0]:
        raise ValueError("left and right must be a disjoint full partition")
    W_left = self_tuning_knn_graph(values[left].T, k=k)
    W_right = self_tuning_knn_graph(values[right].T, k=k)
    P_left = row_stochastic(W_left)
    P_right = row_stochastic(W_right)
    shared = 0.5 * (P_left @ P_right + P_right @ P_left)
    alternating = topk_symmetric(shared, k=k)
    if not return_direct:
        return alternating
    direct = topk_symmetric(0.5 * (P_left + P_right), k=k)
    return alternating, direct


def consensus_graph(graphs, *, k=7):
    graphs = tuple(csr_matrix(graph, dtype=float) for graph in graphs)
    if not graphs:
        raise ValueError("at least one graph is required")
    total = graphs[0].copy()
    for graph in graphs[1:]:
        if graph.shape != total.shape:
            raise ValueError("all graphs must have the same shape")
        total = total + graph
    return topk_symmetric(total * (1.0 / len(graphs)), k=k)


def edge_jaccard(left, right):
    A = csr_matrix(left).astype(bool).astype(np.int8)
    B = csr_matrix(right).astype(bool).astype(np.int8)
    intersection = A.multiply(B).nnz
    union = A.nnz + B.nnz - intersection
    return float(intersection / union) if union else 1.0


def sparse_hash(graph):
    W = csr_matrix(graph, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(W.shape).encode("utf-8"))
    digest.update(W.data.tobytes())
    digest.update(W.indices.tobytes())
    digest.update(W.indptr.tobytes())
    return digest.hexdigest()


def _schema_blocks(F, names, schema, dependency_threshold):
    if schema == "atomic_random":
        return atomic_blocks(F, names)
    if schema == "dependency_blocked":
        return dependency_blocks(
            F, names, distance_threshold=dependency_threshold
        )
    if schema == "family_blocked":
        return family_blocks(F, names)
    raise ValueError(f"unknown partition schema: {schema}")


def _score_spearman(left, right):
    value = spearmanr(np.asarray(left), np.asarray(right)).statistic
    return float(value) if np.isfinite(value) else 0.0


def fit_schema(
    F,
    feature_names,
    *,
    schema,
    seed,
    partition_count=16,
    min_fraction=0.30,
    dependency_threshold=0.15,
    k=7,
    lambdas=(0.0, 0.03, 0.1, 0.3, 1.0, 3.0),
    primary_lambda=0.1,
    prefix_counts=(),
    include_direct=False,
):
    """Fit one repeated partition schema and return label-free artifacts."""
    values, names = _validate(F, feature_names)
    blocks = _schema_blocks(values, names, schema, dependency_threshold)
    partitions = balanced_partitions(
        blocks, values.shape[0], count=partition_count,
        min_fraction=min_fraction, seed=seed,
    )
    pair_graphs, direct_graphs = [], []
    for partition in partitions:
        result = alternating_pair_graph(
            values, partition["left"], partition["right"], k=k,
            return_direct=include_direct,
        )
        if include_direct:
            pair, direct = result
            direct_graphs.append(direct)
        else:
            pair = result
        pair_graphs.append(pair)
    graph = consensus_graph(pair_graphs, k=k)
    path = laplacian_iu_path(values, lambdas, graph=graph, k=k)
    primary = path[float(primary_lambda)]
    consensus_scores = np.asarray(primary.w @ values, dtype=float)

    partition_scores = []
    for pair in pair_graphs:
        fitted = laplacian_iu_fit(
            values, lambda_=primary_lambda, graph=pair, k=k
        )
        partition_scores.append(np.asarray(fitted.w @ values, dtype=float))
    partition_scores = np.asarray(partition_scores, dtype=float)

    prefix_output = {}
    prefix_diagnostics = {}
    for count in prefix_counts:
        count = int(count)
        if count > len(pair_graphs):
            continue
        prefix_graph = consensus_graph(pair_graphs[:count], k=k)
        fitted = laplacian_iu_fit(
            values, lambda_=primary_lambda, graph=prefix_graph, k=k
        )
        score = np.asarray(fitted.w @ values, dtype=float)
        prefix_output[count] = {"graph": prefix_graph, "fit": fitted, "score": score}
        prefix_diagnostics[str(count)] = {
            "graph_sha256": sparse_hash(prefix_graph),
            "score_spearman_vs_final": _score_spearman(score, consensus_scores),
            "mean_abs_rank_change_vs_final": float(np.mean(np.abs(
                rankdata(score) - rankdata(consensus_scores)
            )) / values.shape[1]),
        }

    direct_output = None
    if include_direct:
        direct_graph = consensus_graph(direct_graphs, k=k)
        direct_fit = laplacian_iu_fit(
            values, lambda_=primary_lambda, graph=direct_graph, k=k
        )
        direct_output = {
            "graph": direct_graph,
            "fit": direct_fit,
            "score": np.asarray(direct_fit.w @ values, dtype=float),
        }

    block_names = [list(names[index] for index in block) for block in blocks]
    partition_names = [{
        "left": [names[index] for index in partition["left"]],
        "right": [names[index] for index in partition["right"]],
    } for partition in partitions]
    cka = [centered_affinity_cka(item, graph) for item in pair_graphs]
    jaccard = [edge_jaccard(item, graph) for item in pair_graphs]
    score_agreement = [
        _score_spearman(score, consensus_scores) for score in partition_scores
    ]
    baseline = np.asarray(primary.baseline.w @ values, dtype=float)
    diagnostics = {
        "schema": schema,
        "seed": int(seed),
        "k": int(k),
        "partition_count_requested": int(partition_count),
        "partition_count_used": int(len(partitions)),
        "min_fraction": float(min_fraction),
        "dependency_threshold": float(dependency_threshold),
        "block_count": int(len(blocks)),
        "block_sizes": [len(block) for block in blocks],
        "blocks": block_names,
        "partitions": partition_names,
        "left_size_min": int(min(len(item["left"]) for item in partitions)),
        "left_size_max": int(max(len(item["left"]) for item in partitions)),
        "right_size_min": int(min(len(item["right"]) for item in partitions)),
        "right_size_max": int(max(len(item["right"]) for item in partitions)),
        "graph_sha256": sparse_hash(graph),
        "graph": graph_diagnostics(graph),
        "partition_consensus_cka_median": float(np.median(cka)),
        "partition_consensus_cka_min": float(np.min(cka)),
        "partition_consensus_jaccard_median": float(np.median(jaccard)),
        "partition_score_spearman_median": float(np.median(score_agreement)),
        "partition_score_spearman_min": float(np.min(score_agreement)),
        "mean_abs_rank_change_vs_iu": float(np.mean(np.abs(
            rankdata(consensus_scores) - rankdata(baseline)
        )) / values.shape[1]),
        "primary_weight_cosine_vs_iu": float(
            primary.diagnostics["weight_cosine_vs_iu"]
        ),
        "primary_projected_condition_number": float(
            primary.diagnostics["projected_condition_number"]
        ),
        "primary_projected_roughness_trace": float(
            primary.diagnostics["projected_roughness_trace"]
        ),
        "prefixes": prefix_diagnostics,
    }
    return {
        "graph": graph,
        "path": path,
        "partition_scores": partition_scores,
        "prefixes": prefix_output,
        "direct": direct_output,
        "diagnostics": diagnostics,
    }


def lambda_token(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def fit_repeated_cross_view_paths(
    F,
    feature_names,
    *,
    cell,
    partition_count=16,
    min_fraction=0.30,
    dependency_threshold=0.15,
    primary_k=7,
    sensitivity_ks=(5, 11),
    lambdas=(0.0, 0.03, 0.1, 0.3, 1.0, 3.0),
    primary_lambda=0.1,
    prefix_counts=(4, 8, 16),
):
    """Fit all frozen schemas and controls without accepting labels."""
    values, names = _validate(F, feature_names)
    base_seed = stable_seed(f"rcv-ad:{cell}")
    schemas = {}
    for offset, schema in enumerate(
        ("atomic_random", "dependency_blocked", "family_blocked")
    ):
        schemas[schema] = fit_schema(
            values, names, schema=schema, seed=base_seed + 10_000 * offset,
            partition_count=partition_count, min_fraction=min_fraction,
            dependency_threshold=dependency_threshold, k=primary_k,
            lambdas=lambdas, primary_lambda=primary_lambda,
            prefix_counts=prefix_counts if schema == "dependency_blocked" else (),
            include_direct=schema == "dependency_blocked",
        )

    sensitivity = {}
    for k in sensitivity_ks:
        sensitivity[int(k)] = fit_schema(
            values, names, schema="dependency_blocked", seed=base_seed + 10_000,
            partition_count=partition_count, min_fraction=min_fraction,
            dependency_threshold=dependency_threshold, k=int(k),
            lambdas=(0.0, primary_lambda), primary_lambda=primary_lambda,
        )

    dependency = schemas["dependency_blocked"]
    permutation = np.random.default_rng(base_seed + 991_337).permutation(values.shape[1])
    permuted_graph = permute_graph(dependency["graph"], permutation)
    permuted_fit = laplacian_iu_fit(
        values, lambda_=primary_lambda, graph=permuted_graph, k=primary_k
    )

    baseline = dependency["path"][0.0]
    outputs = {
        "sample_index": np.arange(values.shape[1], dtype=np.int64),
        "feature_names": np.asarray(names, dtype=str),
        "iu_pcr": np.asarray(baseline.w @ values, dtype=np.float64),
    }
    for schema, result in schemas.items():
        for lambda_, fitted in result["path"].items():
            outputs[f"{schema}__lambda_{lambda_token(lambda_)}"] = np.asarray(
                fitted.w @ values, dtype=np.float64
            )
        outputs[f"{schema}__partition_scores_lambda_{lambda_token(primary_lambda)}"] = np.asarray(
            result["partition_scores"], dtype=np.float64
        )
    for count, item in dependency["prefixes"].items():
        outputs[f"dependency_blocked_t{count}__lambda_{lambda_token(primary_lambda)}"] = np.asarray(
            item["score"], dtype=np.float64
        )
    outputs[f"dependency_direct__lambda_{lambda_token(primary_lambda)}"] = np.asarray(
        dependency["direct"]["score"], dtype=np.float64
    )
    outputs[f"dependency_node_permuted__lambda_{lambda_token(primary_lambda)}"] = np.asarray(
        permuted_fit.w @ values, dtype=np.float64
    )
    for k, result in sensitivity.items():
        outputs[f"dependency_blocked_k{k}__lambda_{lambda_token(primary_lambda)}"] = np.asarray(
            result["path"][float(primary_lambda)].w @ values, dtype=np.float64
        )

    diagnostics = {
        "cell": str(cell),
        "schemas": {name: result["diagnostics"] for name, result in schemas.items()},
        "k_sensitivity": {
            str(k): result["diagnostics"] for k, result in sensitivity.items()
        },
        "node_permutation": permutation.tolist(),
        "node_permuted_graph_sha256": sparse_hash(permuted_graph),
        "node_permuted_mean_abs_rank_change_vs_iu": float(np.mean(np.abs(
            rankdata(permuted_fit.w @ values) - rankdata(outputs["iu_pcr"])
        )) / values.shape[1]),
    }
    return outputs, diagnostics


__all__ = [
    "alternating_pair_graph",
    "atomic_blocks",
    "balanced_partitions",
    "consensus_graph",
    "dependency_blocks",
    "edge_jaccard",
    "family_blocks",
    "fit_repeated_cross_view_paths",
    "fit_schema",
    "lambda_token",
    "row_stochastic",
    "sparse_hash",
    "topk_symmetric",
]
