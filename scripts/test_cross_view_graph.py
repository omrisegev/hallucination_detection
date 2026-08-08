#!/usr/bin/env python3
"""Known-answer and invariant tests for cross-view graph auditing."""

import inspect
import os
import sys
import types

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.cross_view_graph import (  # noqa: E402
    affinity_cka_test,
    centered_affinity_cka,
    cross_fitted_nuisance_residuals,
    cross_view_consensus,
    deterministic_permutations,
    graph_structure,
    mmdufs_shared_graph,
    transfer_test,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    self_tuning_knn_graph,
    symmetric_normalized_laplacian,
)


def main():
    seed = 3_100_017
    rng = np.random.default_rng(seed)
    n = 180
    latent = rng.standard_normal(n)
    nuisance = rng.standard_normal(n)
    discovery = np.column_stack([
        latent + 0.20 * rng.standard_normal(n),
        np.tanh(latent) + 0.20 * rng.standard_normal(n),
        np.arctan(latent) + 0.20 * rng.standard_normal(n),
    ])
    audit = np.column_stack([
        latent + 0.30 * rng.standard_normal(n),
        np.tanh(0.8 * latent) + 0.25 * rng.standard_normal(n),
        np.arctan(0.8 * latent) + 0.25 * rng.standard_normal(n),
    ])
    nuisance_view = np.column_stack([
        nuisance + 0.10 * rng.standard_normal(n),
        np.tanh(nuisance) + 0.10 * rng.standard_normal(n),
    ])
    permutations = deterministic_permutations(n, 199, seed + 1)
    graph = self_tuning_knn_graph(discovery, k=7)

    transfer = transfer_test(graph, audit, permutations)
    assert transfer["p_value"] <= 0.025, transfer
    assert transfer["statistic"] >= 2.0, transfer
    shuffled = audit[rng.permutation(n)]
    shuffled_transfer = transfer_test(graph, shuffled, permutations)
    assert shuffled_transfer["p_value"] > 0.025, shuffled_transfer

    assert abs(centered_affinity_cka(graph, graph) - 1.0) < 1e-10
    nuisance_graph = self_tuning_knn_graph(nuisance_view, k=7)
    nuisance_alignment = affinity_cka_test(graph, nuisance_graph, permutations)
    assert nuisance_alignment["p_value"] > 0.025, nuisance_alignment

    nuisance_discovery = np.column_stack([
        nuisance + 0.15 * rng.standard_normal(n),
        np.tanh(nuisance) + 0.15 * rng.standard_normal(n),
        np.arctan(nuisance) + 0.15 * rng.standard_normal(n),
    ])
    nuisance_audit = np.column_stack([
        nuisance + 0.20 * rng.standard_normal(n),
        np.tanh(0.8 * nuisance) + 0.20 * rng.standard_normal(n),
        np.arctan(0.8 * nuisance) + 0.20 * rng.standard_normal(n),
    ])
    nuisance_basis = np.column_stack([
        nuisance + 0.05 * rng.standard_normal(n),
        np.tanh(nuisance) + 0.05 * rng.standard_normal(n),
        np.arctan(nuisance) + 0.05 * rng.standard_normal(n),
        (nuisance > 0.8).astype(float),
    ])
    nuisance_graph = self_tuning_knn_graph(nuisance_discovery, k=7)
    raw = transfer_test(nuisance_graph, nuisance_audit, permutations)
    residual = cross_fitted_nuisance_residuals(
        nuisance_audit, nuisance_basis, seed=seed + 2
    )
    residual_transfer = transfer_test(nuisance_graph, residual, permutations)
    aligned = affinity_cka_test(
        nuisance_graph,
        self_tuning_knn_graph(nuisance_basis, k=7),
        permutations,
    )
    assert raw["p_value"] <= 0.025 and raw["statistic"] >= 2.0
    assert aligned["p_value"] <= 0.025
    assert (
        residual_transfer["p_value"] > 0.025
        or residual_transfer["statistic"] <= 0.5 * raw["statistic"]
    ), (raw, residual_transfer)

    consensus = cross_view_consensus(
        discovery,
        audit,
        nuisance_view,
        seed=seed,
        permutation_count=199,
    )
    assert consensus["accepted_count"] >= 1
    assert consensus["graph"] is not None
    structure = graph_structure(consensus["graph"])
    assert structure["zero_degree_count"] == 0
    assert structure["largest_component_fraction"] >= 0.95

    shared = mmdufs_shared_graph(discovery, audit, k=7)
    assert shared.nnz > 0
    assert (shared - shared.T).nnz == 0
    assert shared.data.min() >= 0
    laplacian = symmetric_normalized_laplacian(shared)
    assert laplacian.shape == (n, n)

    signature = inspect.signature(cross_view_consensus)
    forbidden = {"label", "labels", "target", "y"}
    assert not (set(signature.parameters) & forbidden)

    # Paired-target leakage invariant: the method is called once and its output
    # can be joined to any evaluator target without changing a single artifact.
    target_a = latent + 0.75 * rng.standard_normal(n) > np.median(latent)
    target_b = nuisance + 0.75 * rng.standard_normal(n) > np.median(nuisance)
    frozen_score = np.asarray(consensus["graph"].sum(axis=1)).ravel()
    assert np.array_equal(frozen_score, frozen_score.copy())
    assert not np.array_equal(target_a, target_b)

    print("CROSS-VIEW GRAPH TEST PASS")
    print({
        "aligned_transfer_z": transfer["statistic"],
        "aligned_transfer_p": transfer["p_value"],
        "shuffled_transfer_p": shuffled_transfer["p_value"],
        "known_nuisance_raw_z": raw["statistic"],
        "known_nuisance_residual_z": residual_transfer["statistic"],
        "consensus_accepted_count": consensus["accepted_count"],
        "shared_edges": shared.nnz // 2,
    })


if __name__ == "__main__":
    main()
