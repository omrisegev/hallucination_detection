#!/usr/bin/env python3
"""Mechanical tests for family-residual graph LIU."""

from __future__ import annotations

import numpy as np

from spectral_utils.family_residual_graph import (
    build_family_graphs,
    contribution_laplacian_path,
    diffuse_score_path,
    fit_family_residual_state,
    graphs_from_coordinates,
    normalized_graph_coordinates,
)
from spectral_utils.laplacian_upcr import symmetric_normalized_laplacian


FEATURES = (
    "epr",
    "spectral_entropy",
    "epr_spilled",
    "epr_energy",
    "mean_top1_logprob",
    "trace_length",
)


def fixture(seed=17, n=160):
    rng = np.random.default_rng(seed)
    target = rng.standard_normal(n)
    nuisance = rng.standard_normal(n)
    rows = np.vstack([
        target + .4 * rng.standard_normal(n),
        target + nuisance + .4 * rng.standard_normal(n),
        target - nuisance + .5 * rng.standard_normal(n),
        target + .7 * nuisance + .5 * rng.standard_normal(n),
        target - .5 * nuisance + .6 * rng.standard_normal(n),
        nuisance + .3 * rng.standard_normal(n),
    ])
    rows -= rows.mean(axis=1, keepdims=True)
    rows /= rows.std(axis=1, keepdims=True)
    return rows


def test_coordinate_metric():
    F = fixture()
    state = fit_family_residual_state(F, FEATURES)
    gates = np.linspace(.5, 1.0, len(F))
    D = F.T * gates
    Z, diagnostics = normalized_graph_coordinates(
        D, state.baseline, state.residuals, eta=.4, beta=.25
    )
    assert Z.shape[0] == F.shape[1]
    assert Z.shape[1] == F.shape[0] + 1 + len(state.contribution_space.families)
    assert diagnostics["block_weights"] == {
        "dufs": .6, "baseline": .1, "family": .30000000000000004
    }
    assert np.isfinite(Z).all()


def test_graph_and_zero_identities():
    F = fixture()
    state = fit_family_residual_state(F, FEATURES)
    graphs = build_family_graphs(
        F, np.ones(len(F)), state, eta=.5, beta=.5, ks=(5, 7)
    )
    assert set(graphs) == {5, 7}
    for fitted in graphs.values():
        W = fitted.graph
        assert W.shape == (F.shape[1], F.shape[1])
        assert not (W - W.T).nnz
        L = symmetric_normalized_laplacian(W)
        assert np.min(np.linalg.eigvalsh(L.toarray())) > -1e-9
        cs = contribution_laplacian_path(
            state.baseline, state.residuals, W, (0.0, .1),
            trust_caps=(1 / len(state.contribution_space.families),),
        )
        zero = cs[(0.0, 1 / len(state.contribution_space.families))]
        assert np.array_equal(zero.score, state.baseline)
        assert np.array_equal(zero.correction, np.zeros_like(state.baseline))
        active = cs[(.1, 1 / len(state.contribution_space.families))]
        assert np.isfinite(active.score).all()
        assert active.diagnostics["correction_sd"] <= (
            1 / len(state.contribution_space.families) + 1e-12
        )
        diffusion = diffuse_score_path(state.baseline, W, (0.0, .1))
        assert np.array_equal(diffusion[0.0], state.baseline)
        assert np.isfinite(diffusion[.1]).all()


def test_residualization_and_contribution_ablation_differ():
    F = fixture()
    state = fit_family_residual_state(F, FEATURES)
    centered_b = state.baseline - np.mean(state.baseline)
    covariance = centered_b @ (
        state.residuals - state.residuals.mean(axis=0)
    ) / len(centered_b)
    assert np.max(np.abs(covariance)) < 1e-10
    residual = build_family_graphs(
        F, np.ones(len(F)), state, eta=1, beta=0, family_mode="residual"
    )[7].graph
    contribution = build_family_graphs(
        F, np.ones(len(F)), state, eta=1, beta=0,
        family_mode="contribution",
    )[7].graph
    assert (residual != contribution).nnz > 0


def test_duplicate_coordinates_remove_self_by_identity():
    coordinates = np.repeat(np.arange(20, dtype=float), 3)[:, None]
    tie_keys = np.arange(len(coordinates), dtype=float) * 17 + 3
    graph = graphs_from_coordinates(
        coordinates, (7,), tie_keys=tie_keys
    )[7]
    assert graph.diagonal().sum() == 0
    assert np.min(np.asarray(graph.sum(axis=1)).ravel()) > 0
    permutation = np.random.default_rng(91).permutation(len(coordinates))
    rebuilt = graphs_from_coordinates(
        coordinates[permutation], (7,), tie_keys=tie_keys[permutation]
    )[7]
    expected = graph[permutation][:, permutation]
    difference = rebuilt - expected
    assert not difference.nnz or np.max(np.abs(difference.data)) < 1e-12


def main():
    test_coordinate_metric()
    test_graph_and_zero_identities()
    test_residualization_and_contribution_ablation_differ()
    test_duplicate_coordinates_remove_self_by_identity()
    print("family residual graph tests: PASS")


if __name__ == "__main__":
    main()
