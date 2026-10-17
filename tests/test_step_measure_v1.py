"""Tests for the step-measurement stage.

The one that matters is `test_whitening_recovers_a_buried_impulse`: it plants an impulse
in AR(1) noise, checks that the raw Top-K readout picks the wrong step and the whitened
one picks the right step, and so demonstrates the mechanism the experiment is testing
rather than asserting it. If that test ever stops passing on synthetic data, no result on
real data should be believed either.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_utils.step_measure_v1 import (
    accumulate_autocorrelation, answer_standardize, best_window_steps, topk_mean_steps,
    whiten, yule_walker,
)


def _ar1(n, rho, rng, scale=1.0):
    e = rng.normal(scale=scale, size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


# ------------------------------------------------------------------- basic contracts
def test_answer_standardize_handles_a_constant_series():
    assert np.all(answer_standardize(np.full(7, 3.0)) == 0.0)
    with pytest.raises(ValueError):
        answer_standardize(np.array([]))


def test_topk_mean_matches_the_naive_definition():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    spans = np.array([[0, 17], [17, 40], [40, 60]])
    for k in (1, 3, 10, 999):
        got = topk_mean_steps(x, spans, k)
        want = [np.sort(x[a:b])[-min(k, b - a):].mean() for a, b in spans]
        assert np.allclose(got, want)


def test_best_window_matches_the_naive_definition_and_never_crosses_a_boundary():
    rng = np.random.default_rng(2)
    x = rng.normal(size=50)
    spans = np.array([[0, 12], [12, 31], [31, 50]])
    for w in (1, 3, 8, 999):
        got = best_window_steps(x, spans, w)
        want = []
        for a, b in spans:
            ww = min(w, b - a)
            want.append(max(x[s:s + ww].mean() for s in range(a, b - ww + 1)))
        assert np.allclose(got, want)
    # A spike sitting just outside a step must not raise that step's score.
    x = np.zeros(20)
    x[10] = 50.0
    spans = np.array([[0, 10], [10, 20]])
    assert best_window_steps(x, spans, 4)[0] == pytest.approx(0.0)


def test_width_one_window_equals_top_one_mean():
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    spans = np.array([[0, 25], [25, 52], [52, 80]])
    assert np.allclose(best_window_steps(x, spans, 1), topk_mean_steps(x, spans, 1))


def test_empty_step_scores_zero_rather_than_raising():
    x = np.arange(10, dtype=float)
    spans = np.array([[0, 4], [4, 4], [4, 10]])
    assert topk_mean_steps(x, spans, 3)[1] == 0.0
    assert best_window_steps(x, spans, 3)[1] == 0.0


# ----------------------------------------------------------------------- the whitener
def test_yule_walker_recovers_a_known_ar1():
    rho = 0.6
    r = rho ** np.arange(9)
    model = yule_walker(r)
    assert model["coefficients"][0] == pytest.approx(rho, abs=1e-9)
    assert np.allclose(model["coefficients"][1:], 0.0, atol=1e-9)
    assert model["residual_std"] == pytest.approx(np.sqrt(1 - rho ** 2), abs=1e-9)


def test_yule_walker_rejects_an_unnormalized_sequence():
    with pytest.raises(ValueError):
        yule_walker(np.array([2.0, 0.5, 0.2]))


def test_accumulate_autocorrelation_finds_the_planted_rho():
    rng = np.random.default_rng(4)
    r = accumulate_autocorrelation((_ar1(4000, 0.5, rng) for _ in range(40)), order=4)
    assert r[0] == 1.0
    assert r[1] == pytest.approx(0.5, abs=0.03)
    assert r[2] == pytest.approx(0.25, abs=0.04)


def test_whitening_removes_the_autocorrelation_it_was_fitted_on():
    rng = np.random.default_rng(5)
    train = [_ar1(3000, 0.7, rng) for _ in range(30)]
    model = yule_walker(accumulate_autocorrelation(iter(train), order=6))
    x = _ar1(20000, 0.7, rng)
    before = np.corrcoef(x[:-1], x[1:])[0, 1]
    e = whiten(x, model)
    after = np.corrcoef(e[:-1], e[1:])[0, 1]
    assert before == pytest.approx(0.7, abs=0.03)
    assert abs(after) < 0.05


def test_whiten_preserves_length_so_step_zero_stays_scoreable():
    rng = np.random.default_rng(6)
    model = yule_walker(0.5 ** np.arange(9))
    for n in (3, 9, 50, 500):
        assert len(whiten(rng.normal(size=n), model)) == n


def test_a_series_shorter_than_the_order_uses_the_lower_order_predictors():
    """Not a passthrough. Superseded contract, 2026-09-19.

    The first version returned the standardized series unchanged whenever it was shorter
    than the AR order. That is wrong for the same reason the x[0] padding was wrong: at
    position t there IS history, just less of it, and the order-t predictor uses exactly
    what exists. Only t = 0, which has no history at all, passes through.
    """
    model = yule_walker(0.5 ** np.arange(9))
    x = np.array([1.0, 4.0, 2.0])
    z = answer_standardize(x)
    out = whiten(x, model)
    assert out[0] == pytest.approx(z[0])
    for t in (1, 2):
        a_t, sd_t = model["ladder"][t - 1]
        assert out[t] == pytest.approx((z[t] - a_t @ z[t - 1::-1][:t]) / sd_t)


# ------------------------------------------------------- the mechanism, on synthetic data
def test_whitening_recovers_a_buried_impulse():
    """An impulse in AR(1) noise: raw Top-K is fooled, whitened Top-K is not.

    This is the experiment's hypothesis in miniature. Correlated noise produces smooth
    excursions that a window mean rewards, so the raw readout tends to crown a noise
    swell; whitening flattens the swells and leaves the impulse standing.
    """
    rng = np.random.default_rng(7)
    model = yule_walker(accumulate_autocorrelation(
        (_ar1(3000, 0.8, rng) for _ in range(30)), order=6))
    spans = np.array([[i * 60, (i + 1) * 60] for i in range(8)])
    truth = 3
    raw_hits = white_hits = 0
    trials = 300
    for _ in range(trials):
        x = _ar1(480, 0.8, rng)
        x[truth * 60 + 30] += 6.0            # one impulse, in one step
        raw_hits += int(np.argmax(topk_mean_steps(x, spans, 10)) == truth)
        white_hits += int(np.argmax(topk_mean_steps(whiten(x, model), spans, 10)) == truth)
    assert white_hits > raw_hits + 0.15 * trials


def test_contiguous_window_beats_unordered_topk_on_a_planted_burst():
    """And the converse control: when the signal IS a contiguous burst, the window wins."""
    rng = np.random.default_rng(8)
    spans = np.array([[i * 60, (i + 1) * 60] for i in range(8)])
    truth = 5
    topk_hits = window_hits = 0
    trials = 300
    for _ in range(trials):
        x = rng.normal(size=480)
        x[truth * 60 + 20: truth * 60 + 28] += 1.4      # an 8-token burst
        topk_hits += int(np.argmax(topk_mean_steps(x, spans, 24)) == truth)
        window_hits += int(np.argmax(best_window_steps(x, spans, 8)) == truth)
    assert window_hits > topk_hits


# ------------------------------------------- the start-of-answer artefact (audit, 2026-09-19)
def test_whiten_has_no_start_of_answer_offset():
    """The bug the 2026-09-19 audit found: padding with x[0] biased the first tokens.

    Constant padding made e[0] a shrunken copy of x[0] and injected a large offset into
    e[1] of every answer, which pushed 22-24% of whitened predictions onto step 0 against
    a 12.25% base rate. The predictor ladder must leave the first `order` positions with
    the same first and second moments as the steady state.
    """
    rng = np.random.default_rng(31)
    model = yule_walker(accumulate_autocorrelation(
        (_ar1(3000, 0.5, rng) for _ in range(30)), order=8))
    head = np.zeros((600, 8))
    tail = []
    for row in range(600):
        e = whiten(_ar1(400, 0.5, rng), model)
        head[row] = e[:8]
        tail.append(e[12:])
    tail = np.concatenate(tail)
    # every one of the first eight positions must look like the steady state
    assert np.abs(head.mean(axis=0)).max() < 0.15, head.mean(axis=0)
    assert np.abs(head.std(axis=0) - tail.std()).max() < 0.15, head.std(axis=0)


def test_whiten_first_sample_is_the_standardized_value_itself():
    # With no history the best predictor is the unconditional mean, so e[0] = x[0]. This
    # keeps step 0 reachable without giving it a manufactured score.
    rng = np.random.default_rng(32)
    model = yule_walker(0.5 ** np.arange(9))
    x = rng.normal(size=40)
    assert whiten(x, model)[0] == pytest.approx(answer_standardize(x)[0])


def test_whiten_steady_state_is_unchanged_by_the_ladder():
    # The fix must only touch the first `order` positions.
    rng = np.random.default_rng(33)
    model = yule_walker(accumulate_autocorrelation(
        (_ar1(2000, 0.6, rng) for _ in range(20)), order=8))
    x = _ar1(500, 0.6, rng)
    a = model["coefficients"]
    from scipy.signal import lfilter as _lf
    reference = _lf(np.concatenate([[1.0], -a]), [1.0], answer_standardize(x))[8:] \
        / model["residual_std"]
    assert np.allclose(whiten(x, model)[8:], reference, atol=1e-12)


def test_yule_walker_ladder_is_consistent_with_its_own_orders():
    r = 0.6 ** np.arange(9)
    model = yule_walker(r)
    assert len(model["ladder"]) == 8
    # For a true AR(1), every order-t predictor should reduce to the same single lag.
    for t, (a_t, sd_t) in enumerate(model["ladder"], start=1):
        assert a_t[0] == pytest.approx(0.6, abs=1e-9)
        assert np.allclose(a_t[1:], 0.0, atol=1e-9)
        assert sd_t == pytest.approx(np.sqrt(1 - 0.36), abs=1e-9)
