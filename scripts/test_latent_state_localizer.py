#!/usr/bin/env python3
"""Known-answer tests for the label-free latent-state localizer."""

from __future__ import annotations

import inspect
import sys
from unittest.mock import patch
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spectral_utils.latent_state_localizer as latent_state  # noqa: E402

from spectral_utils.latent_state_localizer import (  # noqa: E402
    apply_latent_state_fit,
    fit_upcr_initialized_hmm,
    forward_backward,
    posterior_entry_curve,
)


def planted_sequences(seed=0, n=36, length=120):
    rng = np.random.default_rng(seed)
    sequences, changes = [], []
    for index in range(n):
        change = 35 + (index % 45)
        values = np.concatenate([
            rng.normal(-1.1, 0.28, change),
            rng.normal(1.2, 0.28, length - change),
        ])
        sequences.append(values)
        changes.append(change)
    return sequences, np.asarray(changes, dtype=int)


def test_forward_backward_identities():
    values = np.array([-1.0, -0.7, 0.8, 1.2])
    means = np.array([-1.0, 1.0])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    start = np.array([0.95, 0.05])
    gamma, xi, likelihood = forward_backward(
        values, means, 0.25, transition, start
    )
    assert gamma.shape == (4, 2)
    assert xi.shape == (3, 2, 2)
    assert np.isfinite(likelihood)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0)
    assert np.allclose(xi.sum(axis=2), gamma[:-1], atol=1e-9)
    assert np.allclose(xi.sum(axis=1), gamma[1:], atol=1e-9)


def test_forward_backward_is_stable_when_only_reachable_emission_is_tiny():
    """Structural zeros must not overflow the backward recursion."""

    values = np.full(64, 20.0)
    gamma, xi, log_likelihood = forward_backward(
        values,
        means=np.array([-20.0, 20.0]),
        variance=1e-3,
        transition=np.array([[0.99, 0.01], [0.0, 1.0]]),
        start=np.array([1.0, 0.0]),
    )

    assert np.isfinite(log_likelihood)
    assert np.isfinite(gamma).all()
    assert np.isfinite(xi).all()
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0)
    np.testing.assert_allclose(xi.sum(axis=(1, 2)), 1.0)
    np.testing.assert_allclose(xi.sum(axis=2), gamma[:-1], atol=1e-9)
    np.testing.assert_allclose(xi.sum(axis=1), gamma[1:], atol=1e-9)
    assert gamma[0, 0] == 1.0


def test_absorbing_hmm_recovers_planted_onsets():
    sequences, changes = planted_sequences()
    fit = fit_upcr_initialized_hmm(sequences, kind="absorbing")
    assert not fit.fallback, fit.diagnostics
    assert fit.selected is not None
    assert fit.selected.means[1] > fit.selected.means[0]
    assert np.array_equal(fit.selected.start, np.array([1.0, 0.0]))
    assert fit.selected.transition[1, 0] == 0.0
    assert fit.selected.transition[1, 1] == 1.0
    curves, locators, diag = apply_latent_state_fit(fit, sequences)
    assert all(curve[0] == 0.0 for curve in curves)
    assert not diag["used_fallback"]
    assert all(len(curve) == len(sequence) for curve, sequence in zip(curves, sequences))
    assert float(np.median(np.abs(locators - changes))) <= 2.0


def test_reversible_hmm_recovers_transient_entries():
    rng = np.random.default_rng(4)
    sequences, changes = [], []
    for index in range(42):
        length = 130
        start = 30 + (index % 55)
        values = rng.normal(-0.9, 0.30, length)
        values[start:start + 9] = rng.normal(1.4, 0.25, 9)
        sequences.append(values)
        changes.append(start)
    changes = np.asarray(changes, dtype=int)
    fit = fit_upcr_initialized_hmm(sequences, kind="reversible")
    assert not fit.fallback, fit.diagnostics
    curves, locators, _ = apply_latent_state_fit(fit, sequences)
    assert fit.selected.transition[1, 0] > 0.0
    assert float(np.median(np.abs(locators - changes))) <= 2.0
    # Entry, not occupancy: the peak belongs at the leading edge of the burst.
    one = posterior_entry_curve(fit.selected, sequences[0])
    assert abs(int(np.argmax(one)) - int(changes[0])) <= 2


def test_determinism_and_affine_scale_invariance():
    rng = np.random.default_rng(8)
    sequences = []
    for index in range(30):
        values = rng.normal(-0.8, 0.3, 110)
        onset = 25 + index % 45
        values[onset:onset + 10] = rng.normal(1.3, 0.25, 10)
        sequences.append(values)
    fit_a = fit_upcr_initialized_hmm(sequences, kind="reversible")
    fit_b = fit_upcr_initialized_hmm(sequences, kind="reversible")
    assert fit_a.selected is not None and fit_b.selected is not None
    curves_a, loc_a, _ = apply_latent_state_fit(fit_a, sequences)
    curves_b, loc_b, _ = apply_latent_state_fit(fit_b, sequences)
    assert fit_a.selected.seed == fit_b.selected.seed
    assert np.array_equal(loc_a, loc_b)
    assert all(np.array_equal(a, b) for a, b in zip(curves_a, curves_b))

    shifted = [7.5 + 3.2 * values for values in sequences]
    fit_c = fit_upcr_initialized_hmm(shifted, kind="reversible")
    curves_c, loc_c, _ = apply_latent_state_fit(fit_c, shifted)
    assert np.array_equal(loc_a, loc_c)
    assert all(np.allclose(a, c, atol=2e-7) for a, c in zip(curves_a, curves_c))


def test_guarded_fallback_is_exact_iu_argmax():
    sequences = [np.ones(40), np.ones(55), np.ones(70)]
    fit = fit_upcr_initialized_hmm(sequences, kind="reversible")
    assert fit.fallback
    assert fit.fallback_reason == "constant_iu_risk"
    curves, locators, diag = apply_latent_state_fit(fit, sequences)
    assert diag["used_fallback"]
    assert diag["output_curve_kind"] == "iu_risk_fallback"
    assert np.isnan(diag["mean_entry_mass"])
    assert all(np.array_equal(a, b) for a, b in zip(curves, sequences))
    assert np.array_equal(locators, np.zeros(len(sequences), dtype=int))


def test_seed_locator_agreement_guard_is_registered_and_passes_known_signal():
    sequences, _ = planted_sequences(seed=19)
    fit = fit_upcr_initialized_hmm(sequences, kind="absorbing")
    assert fit.diagnostics["minimum_seed_locator_agreement"] == 0.80
    assert fit.diagnostics["n_valid_candidates"] >= 2
    assert fit.diagnostics["mean_pair_exact_argmax_agreement"] >= 0.80
    assert fit.diagnostics["seed_locator_agreement_guard_passed"]
    assert not fit.fallback


def test_one_numerical_seed_failure_does_not_abort_other_starts():
    sequences, _ = planted_sequences(seed=21)
    original = latent_state._fit_one

    def fail_seed_11(*args, seed, **kwargs):
        if seed == 11:
            raise FloatingPointError("synthetic seed-local failure")
        return original(*args, seed=seed, **kwargs)

    with patch.object(latent_state, "_fit_one", side_effect=fail_seed_11):
        fit = fit_upcr_initialized_hmm(sequences, kind="absorbing")
    assert len(fit.candidates) == 3
    assert fit.candidates[0].guard_failures == ("numerical_failure",)
    assert fit.selected is not None
    assert fit.selected.seed in {23, 37}


def test_all_numerical_seed_failures_use_exact_iu_fallback():
    sequences, _ = planted_sequences(seed=22)
    with patch.object(
        latent_state, "_fit_one", side_effect=FloatingPointError("synthetic")
    ):
        fit = fit_upcr_initialized_hmm(sequences, kind="absorbing")
    curves, locators, diagnostics = apply_latent_state_fit(fit, sequences)
    assert fit.fallback
    assert fit.fallback_reason == "all_hmm_starts_failed_parameter_guards"
    assert diagnostics["used_fallback"]
    assert all(np.array_equal(left, right) for left, right in zip(curves, sequences))
    assert np.array_equal(
        locators, np.asarray([np.argmax(values) for values in sequences], dtype=int)
    )


def test_structural_fit_errors_are_not_hidden_as_numerical_fallbacks():
    sequences, _ = planted_sequences(seed=23)
    with patch.object(latent_state, "_fit_one", side_effect=ValueError("structural")):
        try:
            fit_upcr_initialized_hmm(sequences, kind="absorbing")
        except ValueError as error:
            assert str(error) == "structural"
        else:
            raise AssertionError("structural ValueError should propagate")


def test_fit_api_cannot_receive_labels_or_step_spans():
    parameters = inspect.signature(fit_upcr_initialized_hmm).parameters
    forbidden = {"labels", "label", "step_spans", "step_token_spans"}
    assert forbidden.isdisjoint(parameters)


def main():
    tests = [
        test_forward_backward_identities,
        test_forward_backward_is_stable_when_only_reachable_emission_is_tiny,
        test_absorbing_hmm_recovers_planted_onsets,
        test_reversible_hmm_recovers_transient_entries,
        test_determinism_and_affine_scale_invariance,
        test_guarded_fallback_is_exact_iu_argmax,
        test_seed_locator_agreement_guard_is_registered_and_passes_known_signal,
        test_one_numerical_seed_failure_does_not_abort_other_starts,
        test_all_numerical_seed_failures_use_exact_iu_fallback,
        test_structural_fit_errors_are_not_hidden_as_numerical_fallbacks,
        test_fit_api_cannot_receive_labels_or_step_spans,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"latent_state_localizer: {len(tests)} tests passed")


if __name__ == "__main__":
    main()
