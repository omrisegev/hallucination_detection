#!/usr/bin/env python3
"""Known-answer tests for target-anchored Laplacian IU-PCR."""

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

from spectral_utils.target_anchored_laplacian import (  # noqa: E402
    fixed_logistic_scores,
    ordinary_u2_coordinates,
    positive_correlation_gates,
    projected_ridge_fit,
    pseudo_anchor_laplacian_fit,
    target_anchored_laplacian_fit,
)


def zscore_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    return centered / centered.std(axis=1, keepdims=True)


def synthetic(seed=31, n=240):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    labels = (latent + 0.35 * rng.standard_normal(n) > 0).astype(int)
    rows = [latent + 0.2 * rng.standard_normal(n) for _ in range(4)]
    rows += [rng.standard_normal(n) for _ in range(4)]
    return zscore_rows(np.vstack(rows)), labels


def main():
    F, labels = synthetic()
    calibration = np.arange(160)

    gates = positive_correlation_gates(F, labels, indices=calibration)
    assert not gates.fallback
    assert gates.gates.shape == (F.shape[0],)
    assert np.isclose(np.sqrt(np.mean(gates.gates ** 2)), 1.0)
    assert gates.gates[:4].mean() > gates.gates[4:].mean()

    one_class_labels = np.zeros(F.shape[1], dtype=int)
    one_class = positive_correlation_gates(
        F, one_class_labels, indices=np.arange(8)
    )
    assert one_class.fallback and one_class.fallback_reason == "one_class"
    assert np.array_equal(one_class.gates, np.ones(F.shape[0]))

    # Reversing a continuous target makes every formerly positive correlation
    # negative; the declared all-zero fallback is an ungated graph.
    monotone = np.vstack([
        np.linspace(-1.0, 1.0, F.shape[1]) + 0.01 * index
        for index in range(F.shape[0])
    ])
    decreasing = -np.linspace(-1.0, 1.0, F.shape[1])
    no_positive = positive_correlation_gates(
        monotone, decreasing, require_two_classes=False
    )
    assert no_positive.fallback
    assert no_positive.fallback_reason == "no_positive_correlation"
    assert np.array_equal(no_positive.gates, np.ones(F.shape[0]))

    anchored = target_anchored_laplacian_fit(
        F, labels, calibration, lambda_=0.1, k=7
    )
    assert anchored.fit.w.shape == (F.shape[0],)
    assert anchored.fit.baseline.rho_hat.shape == (F.shape[0],)
    assert np.isfinite(anchored.fit.w).all()
    assert anchored.gate_result.gates[4:].min() == 0.0
    # Graph gating does not delete coordinates from rho, U2, or final weights.
    assert len(anchored.fit.w) == F.shape[0]
    assert len(anchored.fit.baseline.rho_hat) == F.shape[0]

    ordinary = target_anchored_laplacian_fit(
        F, one_class_labels, np.arange(8), lambda_=0.0, k=7
    )
    assert np.array_equal(ordinary.fit.w, ordinary.fit.baseline.w)

    ordinary_scores = anchored.fit.baseline.w @ F
    pseudo = pseudo_anchor_laplacian_fit(
        F, ordinary_scores, lambda_=0.1, k=7
    )
    direct_pseudo = positive_correlation_gates(
        F, ordinary_scores, require_two_classes=False
    )
    assert np.array_equal(pseudo.gate_result.gates, direct_pseudo.gates)

    ridge_zero = projected_ridge_fit(F, lambda_=0.0)
    assert np.array_equal(ridge_zero.w, ridge_zero.baseline.w)
    ridge = projected_ridge_fit(F, lambda_=0.1)
    assert np.isfinite(ridge.w).all()
    assert ridge.diagnostics["zero_equation_weight_error"] < 1e-10

    coordinates, basis = ordinary_u2_coordinates(F)
    assert coordinates.shape == (F.shape[1], 2)
    assert basis.shape == (F.shape[0], 2)
    assert np.allclose(coordinates.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(coordinates.std(axis=0), 1.0, atol=1e-12)

    logistic, diagnostic = fixed_logistic_scores(coordinates, labels, calibration)
    assert logistic.shape == labels.shape
    assert np.all((logistic >= 0.0) & (logistic <= 1.0))
    assert not diagnostic["fallback"]

    constant, diagnostic = fixed_logistic_scores(
        coordinates, one_class_labels, np.arange(8)
    )
    assert diagnostic["fallback"]
    assert np.array_equal(constant, np.zeros(F.shape[1]))

    # A paired target swap cannot alter any label-free artifact.
    alt_labels = 1 - labels
    baseline_a = projected_ridge_fit(F, lambda_=0.1)
    baseline_b = projected_ridge_fit(F.copy(), lambda_=0.1)
    assert np.array_equal(baseline_a.w, baseline_b.w)
    assert np.array_equal(baseline_a.w @ F, baseline_b.w @ F)
    assert not np.array_equal(
        positive_correlation_gates(F, labels, indices=calibration).gates,
        positive_correlation_gates(F, alt_labels, indices=calibration).gates,
    )

    print("TARGET-ANCHORED LIU TEST PASS")
    print({
        "signal_gate_mean": float(gates.gates[:4].mean()),
        "noise_gate_mean": float(gates.gates[4:].mean()),
        "positive_gate_count": gates.diagnostics["positive_gate_count"],
        "logistic_iterations": diagnostic.get("n_iter", 0),
    })


if __name__ == "__main__":
    main()
