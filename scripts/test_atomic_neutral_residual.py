#!/usr/bin/env python3
"""Invariant tests for the label-free atomic neutral-residual operator."""

from __future__ import annotations

import numpy as np

from spectral_utils.atomic_neutral_residual import (
    atomic_contribution_space,
    atomic_neutral_score,
    fit_atomic_neutral_calibration,
)


def synthetic(seed, names, n=320):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(len(names), n))
    F = (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)
    weights = rng.uniform(0.2, 1.0, size=len(names))
    weights /= weights.sum()
    return F, weights


def run():
    names = tuple(f"f{index}" for index in range(7))
    inputs = [synthetic(100 + index, names) for index in range(5)]
    spaces = [
        atomic_contribution_space(F, names, weights)
        for F, weights in inputs
    ]
    calibration = fit_atomic_neutral_calibration(
        spaces, null_draws=80, null_seed=991
    )
    assert calibration.diagnostics["uses_labels"] is False
    assert np.isclose(np.linalg.norm(calibration.direction), 1.0)
    assert np.any(calibration.neutral_mask)

    F, weights = inputs[0]
    scored = atomic_neutral_score(spaces[0], weights, calibration)
    reconstructed = scored.effective_weights @ F + scored.intercept
    assert np.allclose(scored.score, reconstructed, atol=1e-9, rtol=1e-8)
    assert abs(scored.diagnostics["baseline_correction_covariance"]) < 1e-10
    assert np.isclose(
        scored.diagnostics["correction_scale"],
        1.0 / np.sqrt(len(names)),
    )

    permutation = np.asarray([3, 0, 6, 2, 5, 1, 4])
    permuted_names = tuple(names[index] for index in permutation)
    permuted_spaces = [
        atomic_contribution_space(F[permutation], permuted_names, w[permutation])
        for F, w in inputs
    ]
    permuted_calibration = fit_atomic_neutral_calibration(
        permuted_spaces, null_draws=80, null_seed=991
    )
    assert permuted_calibration.feature_names == calibration.feature_names
    assert np.allclose(
        permuted_calibration.residual_covariance,
        calibration.residual_covariance,
    )
    permuted_score = atomic_neutral_score(
        permuted_spaces[0], weights[permutation], permuted_calibration
    )
    assert np.allclose(permuted_score.score, scored.score, atol=1e-9, rtol=1e-8)
    print("atomic neutral residual invariant tests: PASS")


if __name__ == "__main__":
    run()
