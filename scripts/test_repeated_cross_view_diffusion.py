#!/usr/bin/env python3
"""Known-answer and invariant tests for repeated cross-view diffusion."""

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

from spectral_utils.cross_view_graph import centered_affinity_cka  # noqa: E402
from spectral_utils.laplacian_upcr import graph_diagnostics, permute_graph  # noqa: E402
from spectral_utils.repeated_cross_view_diffusion import (  # noqa: E402
    alternating_pair_graph,
    balanced_partitions,
    dependency_blocks,
    fit_repeated_cross_view_paths,
    fit_schema,
)


FEATURE_NAMES = (
    "epr", "trace_length", "spectral_entropy", "low_band_power",
    "high_band_power", "hl_ratio", "epr_spilled", "sw_var_peak_spilled",
    "epr_energy", "min_energy", "mean_top1_logprob", "logprob_margin",
)


def standardize(F):
    return (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)


def main():
    seed = 8_170_031
    rng = np.random.default_rng(seed)
    n = 220
    latent = rng.standard_normal(n)

    # Two independent views observe the same latent coordinate.
    left = np.asarray([
        np.tanh((0.7 + 0.05 * j) * latent) + 0.35 * rng.standard_normal(n)
        for j in range(6)
    ])
    right = np.asarray([
        np.arctan((0.8 + 0.05 * j) * latent) + 0.35 * rng.standard_normal(n)
        for j in range(6)
    ])
    F = standardize(np.vstack([left, right]))
    graph = alternating_pair_graph(F, range(6), range(6, 12), k=7)
    structure = graph_diagnostics(graph)
    assert structure["n_components"] == 1, structure
    assert structure["graph_symmetry_error"] < 1e-12, structure
    oracle = alternating_pair_graph(
        standardize(np.vstack([
            latent + 0.1 * rng.standard_normal(n) for _ in range(4)
        ])),
        (0, 1), (2, 3), k=7,
    )
    observed_alignment = centered_affinity_cka(graph, oracle)
    shuffled = permute_graph(graph, rng.permutation(n))
    shuffled_alignment = centered_affinity_cka(shuffled, oracle)
    assert observed_alignment > shuffled_alignment + 0.10, (
        observed_alignment, shuffled_alignment
    )

    # Complete-linkage blocks keep near-duplicate coordinates together.
    duplicate_F = F.copy()
    duplicate_F[1] = duplicate_F[0] + 0.01 * rng.standard_normal(n)
    blocks = dependency_blocks(
        duplicate_F, FEATURE_NAMES, distance_threshold=0.15
    )
    duplicate_block = next(block for block in blocks if 0 in block)
    assert 1 in duplicate_block, blocks
    partitions = balanced_partitions(
        blocks, len(FEATURE_NAMES), count=8, min_fraction=0.30, seed=seed
    )
    assert len(partitions) == 8
    for partition in partitions:
        assert len(partition["left"]) >= 4 and len(partition["right"]) >= 4
        for block in blocks:
            assert set(block) <= set(partition["left"]) or set(block) <= set(
                partition["right"]
            )

    # The full API contains no target or correctness argument and lambda zero
    # returns ordinary IU-PCR exactly.
    signature = inspect.signature(fit_repeated_cross_view_paths)
    forbidden = {"label", "labels", "target", "targets", "y"}
    assert not (set(signature.parameters) & forbidden)
    result, diagnostics = fit_repeated_cross_view_paths(
        duplicate_F,
        FEATURE_NAMES,
        cell="known_answer",
        partition_count=4,
        min_fraction=0.25,
        primary_k=5,
        sensitivity_ks=(3,),
        lambdas=(0.0, 0.1),
        primary_lambda=0.1,
        prefix_counts=(2, 4),
    )
    assert np.array_equal(
        result["dependency_blocked__lambda_0"], result["iu_pcr"]
    )
    assert diagnostics["schemas"]["dependency_blocked"]["partition_count_used"] == 4

    # Feature-column permutation cannot change the dependency-blocked graph.
    permutation = rng.permutation(len(FEATURE_NAMES))
    original = fit_schema(
        duplicate_F, FEATURE_NAMES, schema="dependency_blocked", seed=seed,
        partition_count=4, min_fraction=0.25, k=5,
        lambdas=(0.0, 0.1), primary_lambda=0.1,
    )
    reordered = fit_schema(
        duplicate_F[permutation], tuple(FEATURE_NAMES[i] for i in permutation),
        schema="dependency_blocked", seed=seed, partition_count=4,
        min_fraction=0.25, k=5, lambdas=(0.0, 0.1), primary_lambda=0.1,
    )
    assert centered_affinity_cka(original["graph"], reordered["graph"]) > 1 - 1e-10
    assert np.allclose(
        original["path"][0.1].w @ duplicate_F,
        reordered["path"][0.1].w @ duplicate_F[permutation],
        atol=1e-9,
    )

    print("REPEATED CROSS-VIEW DIFFUSION TEST PASS")
    print({
        "common_manifold_cka": observed_alignment,
        "permuted_cka": shuffled_alignment,
        "dependency_blocks": len(blocks),
        "primary_graph_cka_median": diagnostics["schemas"][
            "dependency_blocked"
        ]["partition_consensus_cka_median"],
    })


if __name__ == "__main__":
    main()
