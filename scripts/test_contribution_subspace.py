#!/usr/bin/env python3
"""Known-answer tests for HARP-inspired IU contribution subspaces."""

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.contribution_subspace import (  # noqa: E402
    cardinality_balanced_contribution_score,
    cardinality_balanced_iu_fit,
    fit_anchored_contribution_head,
    fit_contribution_transform,
    fit_neutral_residual_mode_calibration,
    iu_family_contributions,
    leverage_balanced_contribution_score,
    leverage_balanced_iu_fit,
    neutral_residual_mode_score,
)


def zscore_rows(values):
    values = np.asarray(values, dtype=float)
    return (
        values - values.mean(axis=1, keepdims=True)
    ) / values.std(axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(311)
    n = 600
    target = rng.standard_normal(n)
    nuisance = rng.standard_normal(n)
    names = (
        "epr",
        "spectral_entropy",
        "sw_var_peak",
        "epr_spilled",
        "cusum_max_spilled",
        "mean_top1_logprob",
        "logprob_margin",
    )
    F = zscore_rows(np.vstack([
        0.20 * target + 0.80 * nuisance + 0.50 * rng.standard_normal(n),
        0.75 * target + 0.30 * rng.standard_normal(n),
        0.65 * target + 0.45 * rng.standard_normal(n),
        -0.15 * target + 0.90 * nuisance + 0.40 * rng.standard_normal(n),
        -0.10 * target + 0.75 * nuisance + 0.50 * rng.standard_normal(n),
        0.55 * target + 0.55 * rng.standard_normal(n),
        0.45 * target + 0.65 * rng.standard_normal(n),
    ]))
    labels = (target + 0.30 * rng.standard_normal(n) > 0).astype(int)
    weights = np.asarray([1.0, 0.05, 0.05, 0.8, 0.8, 0.05, 0.05])
    weights /= np.linalg.norm(weights)
    train = np.arange(400)
    test = np.arange(400, n)

    space = iu_family_contributions(F, names, weights)
    assert space.contributions.shape == (n, len(space.families))
    assert np.allclose(
        space.contributions.sum(axis=1),
        space.baseline_score,
        rtol=0.0,
        atol=space.diagnostics["reconstruction_error"] + 1e-14,
    )

    transform = fit_contribution_transform(space, train)
    baseline, residuals = transform.apply(
        space.baseline_score, space.contributions
    )
    assert np.allclose(baseline[train].mean(), 0.0, atol=1e-12)
    assert np.allclose(baseline[train].std(), 1.0, atol=1e-12)
    assert np.allclose(residuals[train].mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(residuals[train].std(axis=0), 1.0, atol=1e-12)
    assert np.max(np.abs(baseline[train] @ residuals[train])) < 1e-9

    # The zero correction is the standardized IU-PCR score exactly.
    assert np.array_equal(baseline, baseline + residuals @ np.zeros(residuals.shape[1]))

    head = fit_anchored_contribution_head(
        space,
        labels,
        train[:60],
        transform_indices=train,
        prior_strength=0.1,
    )
    corrected = head.score(space.baseline_score, space.contributions)
    assert corrected.shape == (n,)
    assert np.isfinite(corrected).all()
    assert head.delta.shape == (len(space.families),)
    assert head.diagnostics["n_training"] == 60
    assert head.diagnostics["n_transform"] == len(train)
    baseline_auc = roc_auc_score(labels[test], baseline[test])
    corrected_auc = roc_auc_score(labels[test], corrected[test])
    assert corrected_auc > baseline_auc + 0.05, (baseline_auc, corrected_auc)

    # The label-free correction stays orthogonal to IU on its fit rows and has
    # the parameter-free 1/G trust scale.
    balanced = leverage_balanced_contribution_score(space, weights, train)
    assert balanced.score.shape == (n,)
    assert np.isfinite(balanced.score).all()
    assert abs(balanced.diagnostics["baseline_correction_covariance"]) < 1e-12
    assert np.isclose(
        balanced.diagnostics["correction_scale"],
        1.0 / len(space.families),
    )
    assert np.allclose(
        balanced.effective_weights @ F + balanced.intercept,
        balanced.score,
        atol=1e-12,
    )
    assert balanced.diagnostics["weight_reconstruction_error"] < 1e-12

    wrapped = leverage_balanced_iu_fit(F, names, train)
    assert np.allclose(
        wrapped.effective_weights @ F + wrapped.intercept,
        wrapped.score,
        atol=1e-12,
    )
    assert wrapped.contribution_space.families == space.families

    cardinality_balanced = cardinality_balanced_contribution_score(
        space, weights, train
    )
    assert cardinality_balanced.score.shape == (n,)
    assert np.isfinite(cardinality_balanced.score).all()
    assert abs(
        cardinality_balanced.diagnostics[
            "baseline_correction_covariance"
        ]
    ) < 1e-12
    assert np.isclose(
        cardinality_balanced.diagnostics["correction_scale"],
        1.0 / len(space.families),
    )
    assert np.allclose(
        cardinality_balanced.effective_weights @ F
        + cardinality_balanced.intercept,
        cardinality_balanced.score,
        atol=1e-12,
    )
    cardinality_wrapped = cardinality_balanced_iu_fit(F, names, train)
    assert np.allclose(
        cardinality_wrapped.effective_weights @ F
        + cardinality_wrapped.intercept,
        cardinality_wrapped.score,
        atol=1e-12,
    )

    # The neutral-mode calibration reads contribution spaces only.  Source
    # cell order cannot change its pairwise covariance or oriented direction.
    calibration_names = (
        "epr",
        "spectral_entropy",
        "epr_spilled",
        "epr_energy",
        "mean_top1_logprob",
        "trace_length",
    )
    calibration_spaces = []
    for seed in (811, 812, 813):
        local = np.random.default_rng(seed)
        common = local.standard_normal(n)
        calibration_F = zscore_rows(np.vstack([
            common + 0.30 * local.standard_normal(n),
            0.8 * common + 0.55 * local.standard_normal(n),
            0.6 * common + 0.75 * local.standard_normal(n),
            0.7 * common + 0.65 * local.standard_normal(n),
            0.9 * common + 0.40 * local.standard_normal(n),
            0.5 * common + 0.85 * local.standard_normal(n),
        ]))
        calibration_spaces.append(iu_family_contributions(
            calibration_F,
            calibration_names,
            np.full(len(calibration_names), 1.0 / len(calibration_names)),
        ))
    neutral_calibration = fit_neutral_residual_mode_calibration(
        calibration_spaces
    )
    reversed_calibration = fit_neutral_residual_mode_calibration(
        reversed(calibration_spaces)
    )
    assert neutral_calibration.diagnostics["uses_labels"] is False
    assert np.isclose(np.linalg.norm(neutral_calibration.direction), 1.0)
    assert np.sum(neutral_calibration.direction) >= 0.0
    assert np.allclose(
        neutral_calibration.residual_covariance,
        reversed_calibration.residual_covariance,
        atol=1e-12,
    )
    assert np.allclose(
        neutral_calibration.direction,
        reversed_calibration.direction,
        atol=1e-10,
    )
    neutral = neutral_residual_mode_score(
        space, weights, neutral_calibration, train
    )
    assert neutral.diagnostics["uses_labels"] is False
    assert np.isclose(
        neutral.diagnostics["correction_scale"],
        1.0 / len(space.families),
    )
    assert abs(neutral.diagnostics["baseline_correction_covariance"]) < 1e-12
    assert np.allclose(
        neutral.effective_weights @ F + neutral.intercept,
        neutral.score,
        atol=1e-12,
    )

    # Common positive scaling of every IU weight changes neither ranking nor
    # the standardized contribution correction.
    scaled_space = iu_family_contributions(F, names, 7.0 * weights)
    scaled = leverage_balanced_contribution_score(
        scaled_space, 7.0 * weights, train
    )
    assert np.allclose(balanced.score, scaled.score, atol=1e-12)

    # Equal family leverage is the exact IU identity, rather than a numerically
    # tiny learned perturbation.
    equal_weights = np.zeros_like(weights)
    for member in space.members.values():
        equal_weights[member] = 1.0 / len(member)
    equal_space = iu_family_contributions(F, names, equal_weights)
    equal = leverage_balanced_contribution_score(
        equal_space, equal_weights, train
    )
    assert equal.diagnostics["zero_correction"]
    assert np.array_equal(equal.score, equal.baseline_score)

    # One feature from every present family also makes the cardinality rule an
    # exact identity.
    equal_cardinality_members = np.asarray([
        member[0] for member in space.members.values()
    ])
    equal_cardinality_space = iu_family_contributions(
        F[equal_cardinality_members],
        np.asarray(names)[equal_cardinality_members],
        weights[equal_cardinality_members],
    )
    equal_cardinality = cardinality_balanced_contribution_score(
        equal_cardinality_space,
        weights[equal_cardinality_members],
        train,
    )
    assert equal_cardinality.diagnostics["zero_correction"]
    assert np.array_equal(
        equal_cardinality.score, equal_cardinality.baseline_score
    )

    # Fitting cannot silently accept a one-class target.
    one_class_failed = False
    try:
        fit_anchored_contribution_head(
            space, np.zeros(n, dtype=int), train, prior_strength=0.1
        )
    except ValueError:
        one_class_failed = True
    assert one_class_failed

    print("CONTRIBUTION SUBSPACE TEST PASS")
    print({
        "families": space.families,
        "reconstruction_error": space.diagnostics["reconstruction_error"],
        "baseline_auc": baseline_auc,
        "corrected_auc": corrected_auc,
        "delta_norm": head.diagnostics["delta_norm"],
    })


if __name__ == "__main__":
    main()
