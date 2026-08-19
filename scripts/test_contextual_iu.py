#!/usr/bin/env python3
"""Mechanical and known-answer tests for DSP-contextual IU-PCR."""

from __future__ import annotations

import inspect

import numpy as np

from spectral_utils.contextual_iu import (
    ContextualIUModel,
    family_partition,
    flatten_dsp_context,
)
from spectral_utils.local_online_comprehensive import causal_operator_matrices
from spectral_utils.upcr import upcr_fit, upcr_fit_covariance


IU = {
    "loss": "l2",
    "exclusion": False,
    "difficulty_gate": False,
    "simple_avg_fallback": False,
    "recompute_after_exclusion": False,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "n_components": 2,
    "auto_components": False,
}


def _fixture(seed: int = 7, questions: int = 72, landmarks: int = 4):
    rng = np.random.default_rng(seed)
    rows, positions, groups, dsp = [], [], [], []
    for question in range(questions):
        latent = rng.normal(size=2)
        level = rng.normal(scale=0.35, size=(landmarks, 6))
        level[:, :3] += latent[0]
        level[:, 3:] += latent[1]
        states = causal_operator_matrices(level)
        rows.append(level)
        positions.extend(range(1, landmarks + 1))
        groups.extend([f"q-{question}"] * landmarks)
        dsp.append(flatten_dsp_context(states))
    return (
        np.vstack(rows),
        np.asarray(positions),
        np.asarray(groups),
        np.vstack(dsp),
    )


def test_covariance_api_identity() -> None:
    rng = np.random.default_rng(2)
    F = rng.normal(size=(9, 180))
    F = (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)
    direct = upcr_fit(F, **IU)
    moment = upcr_fit_covariance(F @ F.T / F.shape[1], **IU)
    np.testing.assert_array_equal(direct.w, moment.w)
    np.testing.assert_array_equal(direct.rho_hat, moment.rho_hat)
    assert direct.meta == moment.meta


def test_fallback_and_question_duplication() -> None:
    X, positions, groups, dsp = _fixture(questions=20)
    model = ContextualIUModel.fit(
        X, positions, groups, family_partition(6), dsp_context=dsp, mode="dsp"
    )
    result = model.score(X[:12], positions[:12], dsp_context=dsp[:12])
    assert np.all(result.fallback)
    np.testing.assert_array_equal(result.score, result.baseline_score)

    duplicated = np.repeat(np.arange(len(X)), 2)
    duplicate_model = ContextualIUModel.fit(
        X[duplicated],
        positions[duplicated],
        groups[duplicated],
        family_partition(6),
        dsp_context=dsp[duplicated],
        mode="dsp",
    )
    np.testing.assert_allclose(
        model.global_covariance, duplicate_model.global_covariance, atol=1e-14
    )
    np.testing.assert_allclose(
        model.global_result.w, duplicate_model.global_result.w, atol=1e-12
    )


def test_contextual_scores_and_family_mass() -> None:
    X, positions, groups, dsp = _fixture()
    model = ContextualIUModel.fit(
        X, positions, groups, family_partition(6), dsp_context=dsp, mode="dsp"
    )
    result = model.score(X[:25], positions[:25], dsp_context=dsp[:25])
    assert result.score.shape == (25,)
    assert result.weights.shape == (25, 6)
    assert result.family_mass.shape == (25, 6)
    np.testing.assert_allclose(result.family_mass.sum(axis=1), 1.0, atol=1e-10)
    assert np.isfinite(result.score).all()
    assert np.all(result.n_eff >= 0.0)
    assert np.all((result.alpha >= 0.0) & (result.alpha <= 1.0))


def test_core_mode_and_permutation_control() -> None:
    X, positions, groups, dsp = _fixture(seed=11)
    model = ContextualIUModel.fit(
        X, positions, groups, family_partition(6), mode="core"
    )
    result = model.score(X[:20], positions[:20])
    rng = np.random.default_rng(19)
    permutation = rng.permutation(20)
    changed = model.score(
        X[:20],
        positions[:20],
        context_override=model._query_context(
            (X[:20] - model.feature_mean) / model.feature_scale,
            positions[:20],
            None,
        )[permutation],
    )
    assert not np.allclose(result.weights, changed.weights)
    np.testing.assert_array_equal(result.baseline_score, changed.baseline_score)


def test_causal_suffix_identity() -> None:
    rng = np.random.default_rng(23)
    level = rng.normal(size=(90, 6))
    changed = level.copy()
    changed[45:] = rng.normal(loc=20.0, size=(45, 6))
    left = flatten_dsp_context(causal_operator_matrices(level[:45]))
    right = flatten_dsp_context(causal_operator_matrices(changed[:45]))
    np.testing.assert_array_equal(left, right)
    full = flatten_dsp_context(causal_operator_matrices(level))
    np.testing.assert_array_equal(left, full[:45])


def test_label_firewall_and_feature_partition() -> None:
    parameters = inspect.signature(ContextualIUModel.fit).parameters
    assert "labels" not in parameters
    assert "targets" not in parameters
    assert family_partition(3, 2) == ((0, 3), (1, 4), (2, 5))


def main() -> None:
    tests = (
        test_covariance_api_identity,
        test_fallback_and_question_duplication,
        test_contextual_scores_and_family_mass,
        test_core_mode_and_permutation_control,
        test_causal_suffix_identity,
        test_label_firewall_and_feature_partition,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all ({len(tests)} tests)")


if __name__ == "__main__":
    main()
