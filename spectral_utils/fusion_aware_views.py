"""Label-free view construction for the SpecRaGE-IU factorial benchmark.

Three schemas are exposed:

``manual``
    Frozen feature-provenance groups from :mod:`specrage_views`.
``atomic``
    One feature per view.  A leave-one-cell-out micro partition supplies a
    group-balanced prior so near-duplicate atomic views do not receive extra
    total mass merely because there are several of them.
``micro``
    Leave-one-cell-out groups learned from how one-feature graphs act on the
    two-dimensional IU-PCR subspace.  No correctness label is accepted.

The raw projected matrices ``U.T @ R_j @ U`` are not compared across cells.
Their coordinates depend on eigenvector sign and, near a repeated eigenvalue,
on the chosen basis.  We instead compute pairwise Frobenius distances between
projected matrices inside each cell.  A common orthogonal change of basis is a
conjugation applied to both matrices and leaves that distance unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .laplacian_upcr import self_tuning_knn_graph, symmetric_normalized_laplacian
from .specrage_views import provenance_views


SCHEMA_VERSION = "fusion-aware-views-v1-2026-08-07"
DEFAULT_GRAPH_K = 15
DEFAULT_MAX_SAMPLES = 1500
DEFAULT_SAMPLE_REPLICATES = 4
DEFAULT_SAMPLE_FRACTION = 0.80
DEFAULT_CLUSTER_BOOTSTRAPS = 40
DEFAULT_K_VALUES = (3, 4, 5, 6, 7, 8)
EPS = 1e-12


def _stable_seed(namespace: str) -> int:
    return int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:8], 16)


def _validate_matrix(matrix, names):
    matrix = np.asarray(matrix, dtype=float)
    names = tuple(str(name) for name in names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("matrix must have shape (samples, named features)")
    if matrix.shape[0] < 10 or matrix.shape[1] < 3:
        raise ValueError("at least ten samples and three features are required")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    return matrix, names


def _subsample_indices(n, *, rng, maximum, fraction):
    size = min(int(maximum), max(10, int(np.floor(float(fraction) * n))))
    size = min(size, n)
    if size == n:
        return np.arange(n, dtype=int)
    return np.sort(rng.choice(n, size=size, replace=False))


def projected_impact_signatures(matrix, *, graph_k=DEFAULT_GRAPH_K):
    """Return trace-normalized ``U.T @ R_j @ U`` for every feature.

    ``matrix`` is sample by feature and must already use the frozen orientation
    and standardization contract.  The method is label-free.
    """
    X = np.asarray(matrix, dtype=float)
    n, m = X.shape
    F = X.T
    covariance = F @ F.T / n
    _, basis = eigh(covariance, subset_by_index=[m - 2, m - 1])
    basis = basis[:, ::-1]
    signatures = []
    graph_edges = []
    for feature_index in range(m):
        graph = self_tuning_knn_graph(X[:, [feature_index]], k=graph_k)
        laplacian = symmetric_normalized_laplacian(graph)
        roughness = np.asarray(F @ (laplacian @ F.T) / n, dtype=float)
        roughness = 0.5 * (roughness + roughness.T)
        projected = basis.T @ roughness @ basis
        projected = 0.5 * (projected + projected.T)
        trace = float(np.trace(projected))
        if not np.isfinite(trace) or trace <= EPS:
            projected = np.eye(2) / 2.0
        else:
            projected = projected / trace
        signatures.append(projected)
        graph_edges.append(int(graph.nnz))
    return np.asarray(signatures), np.asarray(graph_edges, dtype=int)


def signature_distance_matrix(signatures):
    """Pairwise basis-invariant projected-impact distances in [0, 1]."""
    signatures = np.asarray(signatures, dtype=float)
    if signatures.ndim != 3 or signatures.shape[1:] != (2, 2):
        raise ValueError("signatures must have shape (features, 2, 2)")
    difference = signatures[:, None, :, :] - signatures[None, :, :, :]
    distance = np.sqrt(np.sum(difference ** 2, axis=(2, 3))) / np.sqrt(2.0)
    distance = np.clip(0.5 * (distance + distance.T), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def cell_impact_profile(
    matrix, names, *, cell, graph_k=DEFAULT_GRAPH_K,
    max_samples=DEFAULT_MAX_SAMPLES,
    sample_replicates=DEFAULT_SAMPLE_REPLICATES,
    sample_fraction=DEFAULT_SAMPLE_FRACTION,
):
    """Compute full/subsample impact distances for one unlabeled cell."""
    matrix, names = _validate_matrix(matrix, names)
    n = matrix.shape[0]
    rng = np.random.default_rng(_stable_seed(f"impact-profile:{cell}"))
    distances, edges, sample_sizes = [], [], []
    # The first replicate is the largest deterministic sample.  Remaining
    # replicates independently probe sample-neighbourhood stability.
    fractions = [1.0] + [sample_fraction] * int(sample_replicates)
    for replicate, fraction in enumerate(fractions):
        local_rng = np.random.default_rng(
            rng.integers(0, np.iinfo(np.int32).max) + replicate
        )
        index = _subsample_indices(
            n, rng=local_rng, maximum=max_samples, fraction=fraction
        )
        signatures, graph_edges = projected_impact_signatures(
            matrix[index], graph_k=min(int(graph_k), len(index) - 1)
        )
        distances.append(signature_distance_matrix(signatures))
        edges.append(graph_edges)
        sample_sizes.append(int(len(index)))
    distances = np.asarray(distances)
    upper = np.triu_indices(len(names), k=1)
    variability = np.std(distances[:, upper[0], upper[1]], axis=0)
    return {
        "cell": str(cell),
        "names": names,
        "distances": distances,
        "mean_distance": np.mean(distances, axis=0),
        "sample_sizes": np.asarray(sample_sizes, dtype=int),
        "graph_edges": np.asarray(edges, dtype=int),
        "pair_distance_bootstrap_std_mean": float(np.mean(variability)),
        "pair_distance_bootstrap_std_p95": float(np.quantile(variability, 0.95)),
    }


def _aggregate_profiles(profiles, feature_names, *, replicate_choices=None):
    feature_names = tuple(feature_names)
    position = {name: index for index, name in enumerate(feature_names)}
    values = [[[] for _ in feature_names] for _ in feature_names]
    support = np.zeros((len(feature_names), len(feature_names)), dtype=int)
    for profile_index, profile in enumerate(profiles):
        local = {name: index for index, name in enumerate(profile["names"])}
        matrix = profile["mean_distance"]
        if replicate_choices is not None:
            matrix = profile["distances"][int(replicate_choices[profile_index])]
        present = [name for name in feature_names if name in local]
        for left_index, left in enumerate(present):
            i = position[left]
            li = local[left]
            for right in present[left_index + 1:]:
                j = position[right]
                value = float(matrix[li, local[right]])
                values[i][j].append(value)
                values[j][i].append(value)
                support[i, j] += 1
                support[j, i] += 1
    observed = [
        value for row in values for cell_values in row for value in cell_values
    ]
    fallback = float(np.quantile(observed, 0.90)) if observed else 1.0
    distance = np.zeros_like(support, dtype=float)
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if support[i, j] >= 3:
                value = float(np.median(values[i][j]))
            else:
                value = fallback
            distance[i, j] = distance[j, i] = value
    distance = np.clip(distance, 0.0, 1.0)
    return distance, support


def _cluster(distance, k):
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=int(k), criterion="maxclust") - 1
    # fcluster can return fewer groups when distances tie.  This is allowed but
    # must still leave at least two groups for SpecRaGE.
    if len(set(labels.tolist())) < 2:
        raise RuntimeError("fusion-aware clustering collapsed to one view")
    return labels.astype(int)


def _partition_penalties(labels):
    sizes = np.bincount(labels)
    singleton_fraction = float(np.sum(sizes == 1) / len(sizes))
    imbalance = float(np.std(sizes) / (np.mean(sizes) + EPS))
    return singleton_fraction, imbalance


def learn_loco_micro_partition(
    profiles, *, held_cell, held_feature_names,
    k_values=DEFAULT_K_VALUES,
    cluster_bootstraps=DEFAULT_CLUSTER_BOOTSTRAPS,
):
    """Learn a deterministic partition from every cell except ``held_cell``."""
    training = [profile for profile in profiles if profile["cell"] != held_cell]
    if len(training) < 3:
        raise ValueError("at least three non-held cells are required")
    training_names = sorted({name for profile in training for name in profile["names"]})
    held_feature_names = tuple(str(name) for name in held_feature_names)
    # A feature absent from all 23 training cells has no transferable impact
    # distance.  Do not let tied fallback distances assign it to an arbitrary
    # cluster; append it as an explicit held-only singleton below.
    feature_names = tuple(training_names)
    base_distance, support = _aggregate_profiles(training, feature_names)
    candidates = []
    rng = np.random.default_rng(_stable_seed(f"micro-partition:{held_cell}"))
    allowed_k = [
        int(k) for k in k_values if 2 <= int(k) < len(feature_names)
    ]
    if not allowed_k:
        raise ValueError("no valid micro-view cluster counts")
    # Use exactly the same bootstrap perturbations for every candidate K.  If
    # each K consumed a different part of the random stream, the model-selection
    # score would confound cluster count with bootstrap luck and could even
    # change when ``k_values`` was reordered.
    bootstrap_distances = []
    for _ in range(int(cluster_bootstraps)):
        chosen_profiles = [
            training[index]
            for index in rng.integers(0, len(training), size=len(training))
        ]
        replicate_choices = [
            int(rng.integers(0, len(profile["distances"])))
            for profile in chosen_profiles
        ]
        boot_distance, _ = _aggregate_profiles(
            chosen_profiles, feature_names, replicate_choices=replicate_choices
        )
        bootstrap_distances.append(boot_distance)
    for k in allowed_k:
        labels = _cluster(base_distance, k)
        if len(set(labels.tolist())) < 2:
            continue
        silhouette = float(silhouette_score(base_distance, labels, metric="precomputed"))
        bootstrap_ari = []
        for boot_distance in bootstrap_distances:
            boot_labels = _cluster(boot_distance, k)
            bootstrap_ari.append(adjusted_rand_score(labels, boot_labels))
        stability = float(np.mean(bootstrap_ari))
        singleton_fraction, imbalance = _partition_penalties(labels)
        score = (
            0.5 * ((silhouette + 1.0) / 2.0)
            + 0.5 * max(stability, 0.0)
            - 0.15 * singleton_fraction
            - 0.05 * imbalance
        )
        candidates.append({
            "requested_k": k,
            "actual_k": int(len(set(labels.tolist()))),
            "silhouette": silhouette,
            "bootstrap_ari": stability,
            "singleton_fraction": singleton_fraction,
            "size_imbalance": imbalance,
            "selection_score": float(score),
            "labels": labels,
        })
    if not candidates:
        raise RuntimeError("all micro-view partitions collapsed")
    chosen = max(candidates, key=lambda row: (row["selection_score"], -row["requested_k"]))
    labels = chosen["labels"]
    raw_groups = {}
    for name, label in zip(feature_names, labels):
        raw_groups.setdefault(int(label), []).append(name)
    ordered_groups = sorted(raw_groups.values(), key=lambda members: tuple(sorted(members)))
    partition = OrderedDict(
        (f"micro_{index + 1:02d}", tuple(sorted(members)))
        for index, members in enumerate(ordered_groups)
    )
    held_partition = OrderedDict()
    assigned = set()
    for group, members in partition.items():
        present = tuple(name for name in members if name in held_feature_names)
        if present:
            held_partition[group] = present
            assigned.update(present)
    for name in sorted(set(held_feature_names) - assigned):
        held_partition[f"micro_unseen_{name}"] = (name,)
    if len(held_partition) < 2:
        raise RuntimeError("held cell has fewer than two learned micro-views")
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "held_cell": held_cell,
        "training_cells": [profile["cell"] for profile in training],
        "feature_names": list(feature_names),
        "support_min_off_diagonal": int(np.min(support[np.triu_indices(len(feature_names), 1)])),
        "support_median_off_diagonal": float(np.median(
            support[np.triu_indices(len(feature_names), 1)]
        )),
        "chosen_k": int(chosen["actual_k"]),
        "chosen_requested_k": int(chosen["requested_k"]),
        "chosen_silhouette": float(chosen["silhouette"]),
        "chosen_bootstrap_ari": float(chosen["bootstrap_ari"]),
        "chosen_singleton_fraction": float(chosen["singleton_fraction"]),
        "chosen_size_imbalance": float(chosen["size_imbalance"]),
        "candidate_scores": [
            {key: value for key, value in row.items() if key != "labels"}
            for row in candidates
        ],
        "global_partition": {key: list(value) for key, value in partition.items()},
        "held_partition": {key: list(value) for key, value in held_partition.items()},
    }
    return held_partition, diagnostics


def atomic_views(matrix, names):
    matrix, names = _validate_matrix(matrix, names)
    return OrderedDict(
        (name, matrix[:, [index]]) for index, name in enumerate(names)
    )


def partition_views(matrix, names, partition):
    matrix, names = _validate_matrix(matrix, names)
    position = {name: index for index, name in enumerate(names)}
    output = OrderedDict()
    used = set()
    for group, members in partition.items():
        selected = [name for name in members if name in position]
        if not selected:
            continue
        duplicate = used.intersection(selected)
        if duplicate:
            raise ValueError(f"features appear in multiple groups: {sorted(duplicate)}")
        used.update(selected)
        output[str(group)] = matrix[:, [position[name] for name in selected]]
    if used != set(names):
        raise ValueError(f"partition does not cover features: {sorted(set(names) - used)}")
    if len(output) < 2:
        raise ValueError("partition must create at least two views")
    return output


def equal_view_prior(views):
    mass = 1.0 / len(views)
    return OrderedDict((name, mass) for name in views)


def group_balanced_atomic_prior(names, partition):
    names = tuple(names)
    present_groups = []
    for members in partition.values():
        present = [name for name in members if name in names]
        if present:
            present_groups.append(present)
    assigned = {name for members in present_groups for name in members}
    present_groups.extend([[name] for name in names if name not in assigned])
    group_mass = 1.0 / len(present_groups)
    prior = {}
    for members in present_groups:
        for name in members:
            prior[name] = group_mass / len(members)
    return OrderedDict((name, float(prior[name])) for name in names)


def build_view_schemas(matrix, names, micro_partition):
    """Return the registered views and priors for one held cell."""
    manual = provenance_views(matrix, names)
    atomic = atomic_views(matrix, names)
    micro = partition_views(matrix, names, micro_partition)
    return OrderedDict((
        ("manual", {
            "views": manual,
            "prior": equal_view_prior(manual),
        }),
        ("atomic", {
            "views": atomic,
            "prior": group_balanced_atomic_prior(names, micro_partition),
        }),
        ("micro", {
            "views": micro,
            "prior": equal_view_prior(micro),
        }),
    ))


__all__ = [
    "SCHEMA_VERSION",
    "atomic_views",
    "build_view_schemas",
    "cell_impact_profile",
    "equal_view_prior",
    "group_balanced_atomic_prior",
    "learn_loco_micro_partition",
    "partition_views",
    "projected_impact_signatures",
    "signature_distance_matrix",
]
