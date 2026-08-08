#!/usr/bin/env python3
"""Known-answer and leakage tests for SpecRaGE-LIU.

This file is intentionally executable without a test framework.  It is not to
be run until the independent Gate-A review in SPEC_SPECRAGE_LIU_V1.md passes.
"""

import inspect
import os
import sys
import types
from dataclasses import replace

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.laplacian_upcr import laplacian_iu_fit  # noqa: E402
from spectral_utils.specrage_laplacian import (             # noqa: E402
    SpecRaGEConfig,
    _spectral_rayleigh_terms,
    control_alpha,
    cross_view_agreement_targets,
    embedding_knn_affinity,
    fit_specrage_graph,
    gaussian_knn_affinity,
    graph_for_control,
    weighted_multiview_graph,
)


def synthetic(seed=41, n=96):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    first = np.column_stack([
        latent + 0.20 * rng.standard_normal(n),
        np.tanh(latent) + 0.25 * rng.standard_normal(n),
    ])
    second = np.column_stack([
        latent + 0.30 * rng.standard_normal(n),
        np.arctan(latent) + 0.25 * rng.standard_normal(n),
    ])
    fusion = np.vstack([
        latent + (0.25 + 0.08 * index) * rng.standard_normal(n)
        for index in range(8)
    ])
    fusion = (
        fusion - fusion.mean(axis=1, keepdims=True)
    ) / fusion.std(axis=1, keepdims=True)
    return {"first": first, "second": second}, fusion


def assert_symmetric_nonnegative(graph):
    assert graph.shape[0] == graph.shape[1]
    assert graph.nnz > 0
    delta = graph - graph.T
    assert not delta.nnz or np.max(np.abs(delta.data)) < 1e-12
    assert np.min(graph.data) >= 0
    assert np.isfinite(graph.data).all()


def main():
    # Leakage invariant: there is no API seam into which labels can be passed.
    signature = inspect.signature(fit_specrage_graph)
    forbidden = {"y", "label", "labels", "target", "targets", "auroc"}
    assert forbidden.isdisjoint(signature.parameters)

    views, fusion = synthetic()
    graph_a, details = gaussian_knn_affinity(views["first"], n_neighbors=7)
    graph_b, _ = gaussian_knn_affinity(views["second"], n_neighbors=7)
    assert details["n_neighbors"] == 7
    assert details["sigma"] > 0
    assert_symmetric_nonnegative(graph_a)
    assert_symmetric_nonnegative(graph_b)

    # Known-answer check for the factor in manuscript Equation (4): a symmetric
    # pair-distance sum is 2*Tr(Y.T L Y), not Tr(Y.T L Y).
    import torch
    small_y = torch.as_tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float64
    )
    affinity_one = torch.as_tensor(
        [[0.0, 1.0, 0.5], [1.0, 0.0, 0.25], [0.5, 0.25, 0.0]],
        dtype=torch.float64,
    )
    affinity_two = torch.as_tensor(
        [[0.0, 0.2, 0.8], [0.2, 0.0, 0.4], [0.8, 0.4, 0.0]],
        dtype=torch.float64,
    )
    small_alpha = torch.as_tensor(
        [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float64
    )
    observed_energy, _ = _spectral_rayleigh_terms(
        small_y, (affinity_one, affinity_two), small_alpha
    )
    traces = []
    for view, affinity in enumerate((affinity_one, affinity_two)):
        weighted_affinity = (
            affinity
            * small_alpha[:, view, None]
            * small_alpha[None, :, view]
        )
        laplacian = torch.diag(weighted_affinity.sum(dim=1)) - weighted_affinity
        traces.append(torch.trace(small_y.T @ laplacian @ small_y))
    expected_energy = 2.0 * sum(traces) / (small_y.shape[0] ** 2 * 2)
    assert torch.allclose(observed_energy, expected_energy, atol=1e-12, rtol=1e-12)

    rng = np.random.default_rng(3)
    raw = rng.uniform(size=(fusion.shape[1], 2))
    alpha = raw / raw.sum(axis=1, keepdims=True)
    base_graphs = (graph_a, graph_b)
    weighted = weighted_multiview_graph(base_graphs, alpha)
    assert_symmetric_nonnegative(weighted)
    embedding_graph, _ = embedding_knn_affinity(
        np.column_stack((np.arange(fusion.shape[1]), np.zeros(fusion.shape[1]))),
        n_neighbors=7,
    )
    assert_symmetric_nonnegative(embedding_graph)

    # Three-view agreement can identify one discordant graph without labels.
    node_permutation = np.random.default_rng(31).permutation(fusion.shape[1])
    discordant = graph_a[node_permutation][:, node_permutation]
    targets, _ = cross_view_agreement_targets(
        (graph_a, graph_b, discordant), temperature=0.08
    )
    assert targets.shape == (fusion.shape[1], 3)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert np.mean(targets[:, :2]) > np.mean(targets[:, 2])

    uniform = control_alpha(alpha, "uniform")
    global_alpha = control_alpha(alpha, "global")
    permuted = control_alpha(alpha, "permuted", seed=17)
    for control in (uniform, global_alpha, permuted):
        assert np.all(control >= 0)
        assert np.allclose(control.sum(axis=1), 1.0)
    assert np.allclose(uniform, 0.5)
    assert np.allclose(global_alpha[0], global_alpha[-1])
    assert not np.allclose(permuted, alpha)

    # Weighted graph construction is equivariant to relabelling samples.
    permutation = np.random.default_rng(19).permutation(fusion.shape[1])
    permuted_graphs = tuple(graph[permutation][:, permutation] for graph in base_graphs)
    equivariant = weighted_multiview_graph(permuted_graphs, alpha[permutation])
    expected = weighted[permutation][:, permutation]
    difference = equivariant - expected
    assert not difference.nnz or np.max(np.abs(difference.data)) < 1e-12

    config = SpecRaGEConfig(
        output_dim=2,
        n_neighbors=7,
        temperature=10.0,
        batch_size=96,
        max_epochs=8,
        min_epochs=4,
        patience=4,
        encoder_hidden=(12,),
        fusion_hidden=(12,),
    )
    result = fit_specrage_graph(views, config=config, seeds=(5, 7))
    assert result.alpha.shape == (fusion.shape[1], 2)
    assert np.all(result.alpha >= 0)
    assert np.allclose(result.alpha.sum(axis=1), 1.0, atol=1e-6)
    assert_symmetric_nonnegative(result.graph)
    assert_symmetric_nonnegative(result.embedding_graph)
    assert 0 <= result.diagnostics["alpha_entropy_normalized"] <= 1.0 + 1e-6

    # Same seed and deterministic CPU algorithms must reproduce the graph.
    repeated = fit_specrage_graph(views, config=config, seeds=(5, 7))
    assert np.array_equal(result.alpha, repeated.alpha)
    graph_delta = result.graph - repeated.graph
    assert not graph_delta.nnz or np.max(np.abs(graph_delta.data)) == 0.0

    # The headline sample graph and its control constructor must use the same
    # seed-mean-alpha ensembling operator.
    rebuilt_sample = graph_for_control(result, "sample_specific")
    graph_delta = result.graph - rebuilt_sample
    assert not graph_delta.nnz or np.max(np.abs(graph_delta.data)) == 0.0

    rebuilt_prior = graph_for_control(result, "uniform")
    direct_prior = weighted_multiview_graph(
        result.base_graphs,
        np.repeat(result.view_prior[None, :], result.alpha.shape[0], axis=0),
        view_prior=result.view_prior,
        mass_normalize=result.config.view_mass_normalization,
    )
    graph_delta = rebuilt_prior - direct_prior
    assert not graph_delta.nnz or np.max(np.abs(graph_delta.data)) == 0.0

    for mode in ("sample_specific", "uniform", "global", "permuted"):
        control_graph = graph_for_control(result, mode, seed=29)
        assert_symmetric_nonnegative(control_graph)

    uniform_config = SpecRaGEConfig(
        output_dim=2,
        n_neighbors=7,
        temperature=10.0,
        batch_size=24,
        max_epochs=2,
        min_epochs=2,
        patience=4,
        encoder_hidden=(8,),
        fusion_hidden=(8,),
        fusion_mode="uniform",
    )
    uniform_result = fit_specrage_graph(
        views, config=uniform_config, seeds=(13,)
    )
    assert np.allclose(uniform_result.alpha, 0.5)
    # 86 training samples yield three full 24-sample batches per epoch.
    assert uniform_result.seed_results[0].diagnostics["optimizer_updates"] == 6

    # CA target construction must obey the same cap as training, while alpha
    # inference and the final graph still cover every sample.
    ca_config = replace(
        config,
        max_epochs=2,
        min_epochs=2,
        patience=3,
        agreement_strength=1.0,
        edge_mass_strength=0.1,
        view_mass_normalization=True,
        fit_sample_cap=64,
    )
    ca_result = fit_specrage_graph(views, config=ca_config, seeds=(17,))
    ca_target = ca_result.diagnostics["agreement_target"]
    assert ca_target["target_scope"] == "seed_specific_unlabeled_fit_pool"
    assert ca_target["fit_sample_count_by_seed"] == [64]
    assert ca_result.alpha.shape[0] == fusion.shape[1]
    assert ca_result.seed_results[0].diagnostics["inference_sample_count"] == 96

    # SpecRaGE integration must preserve IU-PCR exactly at lambda=0.
    zero = laplacian_iu_fit(fusion, lambda_=0.0, graph=result.graph)
    assert np.array_equal(zero.w, zero.baseline.w)
    assert np.array_equal(zero.w @ fusion, zero.baseline.w @ fusion)
    regularized = laplacian_iu_fit(fusion, lambda_=0.1, graph=result.graph)
    assert np.isfinite(regularized.w).all()
    assert regularized.diagnostics["roughness_min_eigenvalue"] >= -1e-9
    assert regularized.diagnostics["projected_condition_number"] > 0

    # Artifact loader contract: exported real bundles use __labels, not __y.
    bundle_path = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
    if os.path.exists(bundle_path):
        bundle = np.load(bundle_path, allow_pickle=True)
        cells = sorted({name.rsplit("__", 1)[0] for name in bundle.files})
        assert cells
        assert all(f"{cell}__labels" in bundle.files for cell in cells)
        assert all(f"{cell}__y" not in bundle.files for cell in cells)

    print("SPECRAGE-LIU TEST PASS")
    print({
        "config": config.fingerprint,
        "epochs": result.seed_results[0].diagnostics["epochs"],
        "alpha_entropy": result.diagnostics["alpha_entropy_normalized"],
        "alpha_seed_std": result.diagnostics["alpha_seed_std_mean"],
        "edges": result.diagnostics["n_edges"],
    })


if __name__ == "__main__":
    main()
