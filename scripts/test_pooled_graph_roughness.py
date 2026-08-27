#!/usr/bin/env python3
"""Deterministic mechanical tests for pooled graph roughness."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.pooled_graph_roughness import (  # noqa: E402
    GraphRoughnessMoment,
    align_family_matrix,
    apply_pooled_roughness,
    fit_pooled_roughness_calibration,
    graph_roughness_moment,
    pool_graph_roughness_moments,
)


def close(left, right, tolerance=1e-11):
    np.testing.assert_allclose(left, right, atol=tolerance, rtol=tolerance)


def test_moment_manual_and_derivative():
    b = np.asarray([-1.0, 0.2, 0.8, 1.4])
    R = np.asarray([
        [-1.0, 0.1], [-0.2, -1.0], [0.4, 0.7], [0.8, 0.2]
    ])
    W = csr_matrix(np.asarray([
        [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]
    ], dtype=float))
    moment = graph_roughness_moment(
        b, R, ("entropy_level", "structural"), W
    )
    assert np.min(np.linalg.eigvalsh(moment.A)) >= -1e-10
    direction = np.asarray([0.3, -0.2])
    epsilon = 1e-6
    from spectral_utils.laplacian_upcr import symmetric_normalized_laplacian
    L = symmetric_normalized_laplacian(W)

    def energy(scale):
        value = b + scale * (R @ direction)
        return float(value @ (L @ value) / len(b))

    numeric = (energy(epsilon) - energy(-epsilon)) / (2 * epsilon)
    scale = moment.diagnostics["trace_scale"]
    local = np.flatnonzero(moment.presence)
    analytic = 2 * float(moment.c[local] @ direction) / scale
    close(numeric, analytic, 1e-8)


def test_hierarchical_pool_and_missing_embedding():
    families = (
        "entropy_level", "entropy_dynamics", "sampled_token_energy",
        "partition_energy", "topk_distribution", "structural",
    )
    moments = []
    for value in (1.0, 3.0, 9.0):
        moments.append(GraphRoughnessMoment(
            A=np.eye(6) * value,
            c=np.ones(6) * value,
            presence=np.ones(6, dtype=bool),
            families=families,
        ))
    A, c, groups = pool_graph_roughness_moments(
        moments, ("a", "a", "b"), pooling="equal_group"
    )
    expected = 0.5 * ((1.0 + 3.0) / 2 + 9.0)
    close(A, np.eye(6) * expected)
    close(c, np.ones(6) * expected)
    assert groups == ("a", "b")
    aligned, presence = align_family_matrix(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        ("structural", "entropy_level"),
    )
    close(aligned[:, 0], [2.0, 4.0])
    close(aligned[:, 5], [1.0, 3.0])
    assert int(np.sum(presence)) == 2


def test_solve_and_application_scale():
    families = (
        "entropy_level", "entropy_dynamics", "sampled_token_energy",
        "partition_energy", "topk_distribution", "structural",
    )
    moment = GraphRoughnessMoment(
        A=np.diag([1, 2, 3, 4, 5, 6]).astype(float),
        c=np.asarray([1, -1, .5, .2, -.3, .7]),
        presence=np.ones(6, dtype=bool),
        families=families,
    )
    calibration = fit_pooled_roughness_calibration(
        [moment], ["source"], .3
    )
    assert calibration.diagnostics["solve_residual"] < 1e-12
    assert np.dot(calibration.c, calibration.direction) < 0
    rng = np.random.default_rng(7)
    R = rng.normal(size=(100, 6))
    b = rng.normal(size=100)
    result = apply_pooled_roughness(
        b, R, families, calibration, trust_factor=.5
    )
    close(np.std(result.correction), .5 / 6)
    identity = apply_pooled_roughness(
        b, R, families, calibration, trust_factor=0
    )
    close(identity.score, b)


def main():
    test_moment_manual_and_derivative()
    test_hierarchical_pool_and_missing_embedding()
    test_solve_and_application_scale()
    print("pooled graph roughness tests: PASS")


if __name__ == "__main__":
    main()
