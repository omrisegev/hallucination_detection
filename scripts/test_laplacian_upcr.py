#!/usr/bin/env python3
"""Known-answer tests for Laplacian-regularized IU-PCR."""

import os
import sys
import types

import numpy as np
from scipy.linalg import eigh


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    laplacian_iu_fit,
    permute_graph,
    symmetric_normalized_laplacian,
)


def synthetic(seed=17, n=180, m=8):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    F = np.vstack([
        latent + (0.25 + 0.1 * index) * rng.standard_normal(n)
        for index in range(m)
    ])
    F = (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)
    return F


def main():
    F = synthetic()
    graph = build_graph_from_features(F, k=7)
    laplacian = symmetric_normalized_laplacian(graph)

    assert graph.shape == (F.shape[1], F.shape[1])
    assert graph.nnz > 0
    assert (graph - graph.T).nnz == 0
    assert np.max(np.abs((laplacian - laplacian.T).data)) < 1e-12 \
        if (laplacian - laplacian.T).nnz else True

    base = laplacian_iu_fit(F, lambda_=0.0, graph=graph)
    assert np.array_equal(base.w, base.baseline.w), "lambda=0 is not exact IU-PCR"
    assert np.array_equal(base.w @ F, base.baseline.w @ F), "lambda=0 score changed"
    # Verify the mathematical zero-penalty equation independently of the exact
    # baseline-copy branch used to make lambda=0 a bitwise invariant.
    C = F @ F.T / F.shape[1]
    values, U = eigh(C, subset_by_index=[F.shape[0] - 2, F.shape[0] - 1])
    U = U[:, np.argsort(values)[::-1]]
    equation_w = U @ np.linalg.solve(U.T @ C @ U, U.T @ base.baseline.rho_hat)
    assert np.allclose(equation_w, base.baseline.w, atol=1e-10, rtol=1e-10)
    assert base.diagnostics["zero_equation_weight_error"] < 1e-10
    assert base.diagnostics["roughness_min_eigenvalue"] >= -1e-9
    assert base.diagnostics["roughness_symmetry_error"] < 1e-12

    regularized = laplacian_iu_fit(F, lambda_=0.3, graph=graph)
    assert np.isfinite(regularized.w).all()
    assert regularized.diagnostics["projected_condition_number"] > 0
    assert regularized.diagnostics["score_laplacian_energy"] <= \
        base.diagnostics["score_laplacian_energy"] + 1e-8

    permutation = np.random.default_rng(2).permutation(F.shape[1])
    permuted = permute_graph(graph, permutation)
    evals_original = np.linalg.eigvalsh(graph.toarray())
    evals_permuted = np.linalg.eigvalsh(permuted.toarray())
    assert np.allclose(evals_original, evals_permuted, atol=1e-10)
    assert not np.allclose(graph.toarray(), permuted.toarray())

    # ARPACK may miss a repeated zero eigenvalue; the connectivity diagnostic
    # must use the known component count for a disconnected graph.
    block = graph.copy().tolil()
    midpoint = F.shape[1] // 2
    block[:midpoint, midpoint:] = 0.0
    block[midpoint:, :midpoint] = 0.0
    disconnected = laplacian_iu_fit(F, lambda_=0.0, graph=block.tocsr())
    assert disconnected.diagnostics["n_components"] >= 2
    assert disconnected.diagnostics["algebraic_connectivity"] == 0.0

    try:
        laplacian_iu_fit(F, lambda_=0.1, graph=graph,
                         baseline_kwargs={"exclusion": True})
        raise AssertionError("incompatible IU baseline did not raise")
    except ValueError:
        pass

    try:
        laplacian_iu_fit(F, lambda_=-0.1, graph=graph)
        raise AssertionError("negative lambda did not raise")
    except ValueError:
        pass

    print("LAPLACIAN U-PCR TEST PASS")
    print({
        "nodes": base.diagnostics["n_nodes"],
        "edges": base.diagnostics["n_edges"],
        "roughness_min_eigenvalue": base.diagnostics["roughness_min_eigenvalue"],
        "lambda_03_energy_ratio": (
            regularized.diagnostics["score_laplacian_energy"]
            / base.diagnostics["score_laplacian_energy"]
        ),
    })


if __name__ == "__main__":
    main()
