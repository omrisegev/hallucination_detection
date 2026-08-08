#!/usr/bin/env python3
"""Known-answer tests for label-free fusion-aware view construction."""

import os
import sys

import numpy as np
from scipy.sparse import csr_matrix


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.fusion_aware_views import (  # noqa: E402
    build_view_schemas,
    learn_loco_micro_partition,
    signature_distance_matrix,
)
from spectral_utils.specrage_laplacian import (  # noqa: E402
    cross_view_agreement_targets,
    weighted_multiview_graph,
)


def profile(cell, names, distance, offset):
    distance = np.asarray(distance, dtype=float)
    replicas = np.stack([
        distance,
        np.clip(distance + offset * (np.ones_like(distance) - np.eye(len(names))), 0, 1),
    ])
    return {
        "cell": cell,
        "names": tuple(names),
        "distances": replicas,
        "mean_distance": replicas.mean(axis=0),
    }


def main():
    signatures = np.asarray([
        [[0.8, 0.1], [0.1, 0.2]],
        [[0.7, -0.05], [-0.05, 0.3]],
        [[0.2, 0.0], [0.0, 0.8]],
    ])
    angle = 0.73
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    transformed = np.asarray([rotation.T @ value @ rotation for value in signatures])
    assert np.allclose(
        signature_distance_matrix(signatures),
        signature_distance_matrix(transformed),
        atol=1e-12,
    )

    names = ("a", "b", "c", "d", "e", "f")
    distance = np.asarray([
        [0, .05, .80, .82, .90, .91],
        [.05, 0, .78, .81, .89, .90],
        [.80, .78, 0, .06, .75, .77],
        [.82, .81, .06, 0, .73, .76],
        [.90, .89, .75, .73, 0, .04],
        [.91, .90, .77, .76, .04, 0],
    ])
    profiles = [
        profile(f"cell_{index}", names, distance, 0.002 * index)
        for index in range(5)
    ]
    partition, diagnostics = learn_loco_micro_partition(
        profiles, held_cell="cell_0", held_feature_names=names,
        k_values=(3,), cluster_bootstraps=5,
    )
    assert len(partition) == 3
    assert diagnostics["chosen_k"] == 3
    assert {frozenset(value) for value in partition.values()} == {
        frozenset(("a", "b")), frozenset(("c", "d")), frozenset(("e", "f")),
    }
    forward_partition, forward_diagnostics = learn_loco_micro_partition(
        profiles, held_cell="cell_0", held_feature_names=names,
        k_values=(3, 4), cluster_bootstraps=7,
    )
    reverse_partition, reverse_diagnostics = learn_loco_micro_partition(
        profiles, held_cell="cell_0", held_feature_names=names,
        k_values=(4, 3), cluster_bootstraps=7,
    )
    assert forward_partition == reverse_partition
    forward_candidates = {
        row["requested_k"]: row for row in forward_diagnostics["candidate_scores"]
    }
    reverse_candidates = {
        row["requested_k"]: row for row in reverse_diagnostics["candidate_scores"]
    }
    assert forward_candidates == reverse_candidates
    reduced_profiles = [
        profile(f"reduced_{index}", names[:-1], distance[:-1, :-1], 0.002 * index)
        for index in range(5)
    ]
    unseen_partition, _ = learn_loco_micro_partition(
        reduced_profiles, held_cell="held_elsewhere", held_feature_names=names,
        k_values=(3,), cluster_bootstraps=3,
    )
    assert unseen_partition["micro_unseen_f"] == ("f",)

    matrix = np.random.default_rng(7).standard_normal((60, len(names)))
    # Use a local synthetic manual mapping only through atomic/micro assertions;
    # real manual views require the registered feature names.
    from spectral_utils.fusion_aware_views import (
        atomic_views, group_balanced_atomic_prior, partition_views,
    )
    atomic = atomic_views(matrix, names)
    micro = partition_views(matrix, names, partition)
    prior = group_balanced_atomic_prior(names, partition)
    assert list(atomic) == list(names)
    assert len(micro) == 3
    assert np.isclose(sum(prior.values()), 1.0)
    for members in partition.values():
        assert np.isclose(sum(prior[name] for name in members), 1.0 / 3.0)

    graph_a = csr_matrix(np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    graph_b = csr_matrix(np.asarray([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=float))
    graph_prior = np.asarray([0.8, 0.2])
    alpha = np.repeat(graph_prior[None, :], 3, axis=0)
    fused = weighted_multiview_graph(
        (graph_a, graph_b), alpha, view_prior=graph_prior, mass_normalize=True
    )
    expected = 0.8 * graph_a + 0.2 * graph_b
    assert np.allclose(fused.toarray(), expected.toarray())
    targets, _ = cross_view_agreement_targets(
        (graph_a, graph_b), view_prior=graph_prior
    )
    assert np.allclose(targets, alpha)

    # With at least three views, Equation (2) uses the registered mass of every
    # *other* view.  This known-answer check catches accidental reversion to an
    # unweighted arithmetic mean.
    graph_c = csr_matrix(np.asarray([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float))
    three_prior = np.asarray([0.6, 0.3, 0.1])
    three_graphs = (graph_a, graph_b, graph_c)
    three_targets, three_agreement = cross_view_agreement_targets(
        three_graphs, temperature=0.2, view_prior=three_prior
    )
    dense_profiles = []
    for graph in three_graphs:
        dense = graph.toarray()
        transition = dense / np.maximum(dense.sum(axis=1, keepdims=True), 1e-12)
        diffusion = 0.5 * transition + 0.5 * (transition @ transition)
        diffusion /= np.maximum(
            np.linalg.norm(diffusion, axis=1, keepdims=True), 1e-12
        )
        dense_profiles.append(diffusion)
    expected_agreement = np.zeros_like(three_agreement)
    for view in range(3):
        for other in range(3):
            if other != view:
                expected_agreement[:, view] += (
                    three_prior[other] / (1.0 - three_prior[view])
                    * np.sum(dense_profiles[view] * dense_profiles[other], axis=1)
                )
    logits = expected_agreement / 0.2 + np.log(three_prior[None, :])
    logits -= logits.max(axis=1, keepdims=True)
    expected_targets = np.exp(logits)
    expected_targets /= expected_targets.sum(axis=1, keepdims=True)
    assert np.allclose(three_agreement, expected_agreement)
    assert np.allclose(three_targets, expected_targets)
    print("FUSION-AWARE VIEW TEST PASS")


if __name__ == "__main__":
    main()
