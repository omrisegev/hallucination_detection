"""Label-free cross-view auditing for sample-neighbourhood graphs.

This module isolates the Phase-1 mechanism in
``docs/research_notes/graybox_cross_view_manifold_proposal.md``.  A discovery
view proposes a graph.  A primitive-disjoint audit view tests whether that
graph transfers beyond its construction coordinates, and a nuisance view can
veto a graph whose transfer disappears after nuisance residualization.

No function accepts labels.  The caller may evaluate the frozen graph and
scores only after this module returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .laplacian_upcr import (
    self_tuning_knn_graph,
    symmetric_normalized_laplacian,
)


EPS = 1e-12


def standardize_columns(matrix):
    """Return a finite, column-standardized sample-by-coordinate matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("matrix must have shape (samples>=3, coordinates>=1)")
    if not np.isfinite(values).all():
        raise ValueError("matrix contains non-finite values")
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    if np.any(scale < EPS):
        raise ValueError("matrix contains a constant coordinate")
    return centered / scale


def deterministic_permutations(n, count, seed):
    """Generate synchronized node permutations for one dataset replicate."""
    if n < 3 or count < 1:
        raise ValueError("n>=3 and count>=1 are required")
    rng = np.random.default_rng(int(seed))
    return np.asarray([rng.permutation(n) for _ in range(int(count))], dtype=int)


def _group_median(values, group_ids):
    values = np.asarray(values, dtype=float)
    if group_ids is None:
        group_ids = tuple(range(values.shape[-1]))
    group_ids = tuple(group_ids)
    if len(group_ids) != values.shape[-1]:
        raise ValueError("group_ids must match the coordinate count")
    ordered = tuple(dict.fromkeys(group_ids))
    grouped = [
        np.median(values[..., np.asarray([g == key for g in group_ids])], axis=-1)
        for key in ordered
    ]
    return np.median(np.stack(grouped, axis=-1), axis=-1)


def transfer_test(graph, audit, permutations, *, group_ids=None):
    """Permutation-calibrated out-of-family graph smoothness test.

    Large positive ``statistic`` means the audit coordinates are smoother on
    the observed graph than on node-relabelled versions of the same graph.
    The empirical p-value is computed on the synchronized aggregate statistic,
    so transformed coordinates are not treated as independent replicates.
    """
    W = csr_matrix(graph, dtype=float)
    A = standardize_columns(audit)
    permutations = np.asarray(permutations, dtype=int)
    if W.shape != (A.shape[0], A.shape[0]):
        raise ValueError("graph and audit sample counts differ")
    if permutations.ndim != 2 or permutations.shape[1] != A.shape[0]:
        raise ValueError("permutations have the wrong shape")
    L = symmetric_normalized_laplacian(W)

    def energies(laplacian):
        return np.sum(A * (laplacian @ A), axis=0) / (
            np.sum(A * A, axis=0) + EPS
        )

    observed = energies(L)
    null = np.asarray([
        energies(L[permutation][:, permutation])
        for permutation in permutations
    ])
    center = np.median(null, axis=0)
    mad = np.median(np.abs(null - center[None, :]), axis=0)
    scale = 1.4826 * mad + EPS
    observed_z = (center - observed) / scale
    null_z = (center[None, :] - null) / scale[None, :]
    statistic = float(_group_median(observed_z, group_ids))
    null_statistic = np.asarray(_group_median(null_z, group_ids), dtype=float)
    p_value = float(
        (1 + np.sum(null_statistic >= statistic)) / (len(null_statistic) + 1)
    )
    coordinate_p = (1 + np.sum(null <= observed[None, :], axis=0)) / (
        len(null) + 1
    )
    return {
        "statistic": statistic,
        "p_value": p_value,
        "observed_energy": observed,
        "coordinate_z": observed_z,
        "coordinate_p": coordinate_p,
        "null_statistic": null_statistic,
    }


def _centered_inner(left, right):
    """Frobenius inner product of two double-centered sparse affinities."""
    K = csr_matrix(left, dtype=float)
    L = csr_matrix(right, dtype=float)
    if K.shape != L.shape or K.shape[0] != K.shape[1]:
        raise ValueError("affinities must be square and shape-matched")
    n = K.shape[0]
    degree_k = np.asarray(K.sum(axis=1)).ravel()
    degree_l = np.asarray(L.sum(axis=1)).ravel()
    raw = float(K.multiply(L).sum())
    cross = float(np.dot(degree_k, degree_l))
    total = float(degree_k.sum() * degree_l.sum())
    return raw - 2.0 * cross / n + total / (n * n)


def centered_affinity_cka(left, right):
    """Centered-kernel alignment of two symmetric affinity matrices."""
    numerator = _centered_inner(left, right)
    norm_left = max(_centered_inner(left, left), 0.0)
    norm_right = max(_centered_inner(right, right), 0.0)
    return float(numerator / (np.sqrt(norm_left * norm_right) + EPS))


def affinity_cka_test(graph, reference_graph, permutations):
    """One-sided permutation test for unusually high centered affinity CKA."""
    W = csr_matrix(graph, dtype=float)
    reference = csr_matrix(reference_graph, dtype=float)
    permutations = np.asarray(permutations, dtype=int)
    observed = centered_affinity_cka(W, reference)
    null = np.asarray([
        centered_affinity_cka(W[permutation][:, permutation], reference)
        for permutation in permutations
    ])
    p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    return {"cka": observed, "p_value": p_value, "null_cka": null}


def cross_fitted_nuisance_residuals(audit, nuisance, *, seed, alpha=1.0):
    """Five-fold row-cross-fitted ridge residuals; labels are never involved."""
    A = standardize_columns(audit)
    N = standardize_columns(nuisance)
    if A.shape[0] != N.shape[0]:
        raise ValueError("audit and nuisance sample counts differ")
    folds = KFold(n_splits=5, shuffle=True, random_state=int(seed))
    predicted = np.empty_like(A)
    for train, test in folds.split(N):
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(N[train], A[train])
        predicted[test] = model.predict(N[test])
    residual = A - predicted
    centered = residual - residual.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    # A nuisance basis can explain a coordinate almost exactly in a synthetic
    # known-answer world.  Keep a deterministic zero residual instead of
    # amplifying numerical dust.
    safe = np.where(scale > 1e-10, scale, 1.0)
    return centered / safe


def graph_structure(graph):
    W = csr_matrix(graph, dtype=float)
    degree = np.asarray(W.sum(axis=1)).ravel()
    n_components, labels = connected_components(W, directed=False)
    component_sizes = np.bincount(labels, minlength=n_components)
    return {
        "zero_degree_count": int(np.sum(degree <= EPS)),
        "n_components": int(n_components),
        "largest_component_fraction": float(component_sizes.max() / len(labels)),
        "degree_min": float(degree.min()),
        "degree_mean": float(degree.mean()),
        "degree_max": float(degree.max()),
    }


@dataclass
class DirectionAudit:
    accepted: bool
    graph: csr_matrix
    diagnostics: dict


def audit_direction(
    discovery,
    audit,
    nuisance,
    *,
    permutations,
    residual_seed,
    ks=(5, 7, 11),
    primary_k=7,
    p_threshold=0.025,
    z_threshold=2.0,
    stability_threshold=0.75,
    group_ids=None,
):
    """Build and audit one registered discovery-to-audit graph direction."""
    G = standardize_columns(discovery)
    A = standardize_columns(audit)
    N = standardize_columns(nuisance)
    if not (G.shape[0] == A.shape[0] == N.shape[0]):
        raise ValueError("all views must contain the same samples")
    ks = tuple(int(k) for k in ks)
    if primary_k not in ks:
        raise ValueError("primary_k must be present in ks")
    residual = cross_fitted_nuisance_residuals(
        A, N, seed=int(residual_seed), alpha=1.0
    )
    graphs = {k: self_tuning_knn_graph(G, k=k) for k in ks}
    nuisance_graphs = {k: self_tuning_knn_graph(N, k=k) for k in ks}
    per_k = {}
    for k in ks:
        transfer = transfer_test(
            graphs[k], A, permutations, group_ids=group_ids
        )
        nuisance_alignment = affinity_cka_test(
            graphs[k], nuisance_graphs[k], permutations
        )
        residual_transfer = transfer_test(
            graphs[k], residual, permutations, group_ids=group_ids
        )
        structure = graph_structure(graphs[k])
        raw_pass = bool(
            transfer["p_value"] <= p_threshold
            and transfer["statistic"] >= z_threshold
        )
        residual_disappears = bool(
            residual_transfer["p_value"] > p_threshold
            or residual_transfer["statistic"]
            <= 0.5 * max(transfer["statistic"], 0.0)
        )
        nuisance_veto = bool(
            nuisance_alignment["p_value"] <= p_threshold
            and residual_disappears
        )
        structural_pass = bool(
            structure["zero_degree_count"] == 0
            and structure["largest_component_fraction"] >= 0.95
        )
        decision = bool(raw_pass and structural_pass and not nuisance_veto)
        per_k[str(k)] = {
            "raw_pass": raw_pass,
            "decision": decision,
            "nuisance_veto": nuisance_veto,
            "structural_pass": structural_pass,
            "transfer_statistic": transfer["statistic"],
            "transfer_p": transfer["p_value"],
            "residual_transfer_statistic": residual_transfer["statistic"],
            "residual_transfer_p": residual_transfer["p_value"],
            "nuisance_cka": nuisance_alignment["cka"],
            "nuisance_p": nuisance_alignment["p_value"],
            **structure,
        }

    primary_graph = graphs[int(primary_k)]
    stability_values = [
        centered_affinity_cka(primary_graph, graphs[k])
        for k in ks if k != primary_k
    ]
    stability = float(np.median(stability_values)) if stability_values else 1.0
    decisions = [per_k[str(k)]["decision"] for k in ks]
    decision_agreement = bool(all(value == decisions[0] for value in decisions))
    accepted = bool(
        per_k[str(primary_k)]["decision"]
        and decision_agreement
        and stability >= stability_threshold
    )

    audit_rng = np.random.default_rng(int(residual_seed) + 91_337)
    shuffled_audit = A[audit_rng.permutation(len(A))]
    audit_permutation = transfer_test(
        primary_graph, shuffled_audit, permutations, group_ids=group_ids
    )
    diagnostics = {
        "accepted": accepted,
        "primary_k": int(primary_k),
        "p_threshold": float(p_threshold),
        "z_threshold": float(z_threshold),
        "stability_threshold": float(stability_threshold),
        "decision_agreement": decision_agreement,
        "stability_cka": stability,
        "audit_row_permutation_statistic": audit_permutation["statistic"],
        "audit_row_permutation_p": audit_permutation["p_value"],
        "per_k": per_k,
    }
    return DirectionAudit(accepted=accepted, graph=primary_graph, diagnostics=diagnostics)


def cross_view_consensus(
    distribution_view,
    realized_view,
    nuisance_view,
    *,
    seed,
    permutation_count=199,
    ks=(5, 7, 11),
    primary_k=7,
):
    """Return the hard-veto bidirectional graph and label-free diagnostics."""
    distribution = standardize_columns(distribution_view)
    realized = standardize_columns(realized_view)
    nuisance = standardize_columns(nuisance_view)
    n = distribution.shape[0]
    if realized.shape[0] != n or nuisance.shape[0] != n:
        raise ValueError("registered views must be row-aligned")
    permutations = deterministic_permutations(
        n, int(permutation_count), int(seed) + 73_001
    )
    forward = audit_direction(
        distribution,
        realized,
        nuisance,
        permutations=permutations,
        residual_seed=int(seed) + 11,
        ks=ks,
        primary_k=primary_k,
    )
    reverse = audit_direction(
        realized,
        distribution,
        nuisance,
        permutations=permutations,
        residual_seed=int(seed) + 29,
        ks=ks,
        primary_k=primary_k,
    )
    accepted_graphs = [
        item.graph for item in (forward, reverse) if item.accepted
    ]
    graph = None
    if accepted_graphs:
        graph = accepted_graphs[0].copy().astype(float)
        for item in accepted_graphs[1:]:
            graph = graph + item
        graph = (graph / len(accepted_graphs)).tocsr()
        graph = graph.maximum(graph.T).tocsr()
        graph.setdiag(0.0)
        graph.eliminate_zeros()
    return {
        "graph": graph,
        "accepted_count": int(len(accepted_graphs)),
        "forward": forward,
        "reverse": reverse,
        "permutations": permutations,
    }


def _normalized_affinity(graph):
    W = csr_matrix(graph, dtype=float)
    degree = np.asarray(W.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    positive = degree > EPS
    inverse[positive] = 1.0 / np.sqrt(degree[positive])
    return (diags(inverse) @ W @ diags(inverse)).tocsr()


def _top_k_symmetric(dense, k):
    values = np.asarray(dense, dtype=float)
    n = values.shape[0]
    k = int(max(1, min(int(k), n - 1)))
    np.fill_diagonal(values, 0.0)
    rows, cols, data = [], [], []
    for row in range(n):
        candidates = np.argpartition(values[row], -k)[-k:]
        candidates = candidates[values[row, candidates] > EPS]
        rows.extend([row] * len(candidates))
        cols.extend(candidates.tolist())
        data.extend(values[row, candidates].tolist())
    directed = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    graph = directed.maximum(directed.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def mmdufs_shared_graph(left_view, right_view, *, k=7):
    """mmDUFS-inspired shared operator converted to an IU-PCR affinity graph.

    The paper calls ``D^-1/2 W D^-1/2`` a normalized Laplacian and defines
    ``P_shared=LxLy+LyLx``.  Here the nonnegative symmetric product is clipped,
    top-k sparsified, and re-normalized for the IU-PCR penalty.  That final
    conversion is an explicit repository adaptation, not paper-faithful mmDUFS.
    """
    left = standardize_columns(left_view)
    right = standardize_columns(right_view)
    if left.shape[0] != right.shape[0]:
        raise ValueError("modalities must be row-aligned")
    affinity_left = _normalized_affinity(self_tuning_knn_graph(left, k=k))
    affinity_right = _normalized_affinity(self_tuning_knn_graph(right, k=k))
    shared = affinity_left @ affinity_right + affinity_right @ affinity_left
    dense = np.maximum(0.0, 0.5 * (shared.toarray() + shared.toarray().T))
    graph = _top_k_symmetric(dense, k)
    if graph.nnz == 0:
        raise ValueError("mmDUFS-inspired shared graph is degenerate")
    return graph


__all__ = [
    "DirectionAudit",
    "affinity_cka_test",
    "audit_direction",
    "centered_affinity_cka",
    "cross_fitted_nuisance_residuals",
    "cross_view_consensus",
    "deterministic_permutations",
    "graph_structure",
    "mmdufs_shared_graph",
    "standardize_columns",
    "transfer_test",
]
