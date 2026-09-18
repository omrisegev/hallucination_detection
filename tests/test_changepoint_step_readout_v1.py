"""Tests for the change-point readouts.

Two of these matter more than the rest.

`test_bocpd_matches_brute_force_partitions` re-runs the verification that
`docs/reviews/bocpd_boundary_audit_2026-09-07.md` applied to the original filter:
enumerate EVERY partition of EVERY prefix of a short series, compute each segment's
marginal evidence independently under the Gaussian block covariance
``R I + prior_variance 11^T``, and check that the recursion's reset posterior agrees.
The copy in this branch must not drift from the verified one.

`test_cusum_closed_form_matches_the_recursion` checks the vectorized prefix-minimum form
against the literal ``S_t = max(0, S_{t-1} + z_t - k)`` loop, because the closed form is
the only reason this is affordable at 7M tokens and a silent mismatch there would be
invisible in every downstream number.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from spectral_utils.changepoint_step_readout_v1 import (
    argmax_readout, bocpd_filter, bocpd_readout, first_crossing_readout, page_cusum,
    page_cusum_readout, token_index_to_step, zscore_within,
)


# ------------------------------------------------------------------ CUSUM closed form
def _cusum_loop(z, k):
    out, s = [], 0.0
    for value in z:
        s = max(0.0, s + value - k)
        out.append(s)
    return np.asarray(out)


@pytest.mark.parametrize("k", [0.0, 0.25, 0.5, 1.0])
def test_cusum_closed_form_matches_the_recursion(k):
    rng = np.random.default_rng(7)
    for _ in range(20):
        x = rng.normal(size=rng.integers(3, 40))
        run = page_cusum(x, k=k)
        assert np.allclose(run["statistic"], _cusum_loop(zscore_within(x), k), atol=1e-12)


def test_cusum_onset_is_the_start_of_the_excursion():
    # Flat, then a sustained lift from index 6. The excursion that raises the alarm must
    # be traced back to 6, not to the index where the statistic happened to cross.
    x = np.r_[np.zeros(6), np.full(8, 4.0)]
    run = page_cusum(x, k=0.5)
    alarm = int(np.flatnonzero(run["statistic"] >= 2.0)[0])
    assert alarm > 6
    assert run["onset"][alarm] == 6
    assert page_cusum_readout(x, k=0.5, h=2.0, estimator="onset") == 6
    assert page_cusum_readout(x, k=0.5, h=2.0, estimator="alarm") == alarm


def test_cusum_can_return_index_zero():
    # Both estimators must be able to name the first index; 12.25% of this population's
    # first errors are at step 0, so a rule that cannot reach it forfeits that mass.
    x = np.r_[np.full(8, 5.0), np.zeros(8)]
    assert page_cusum_readout(x, k=0.5, h=0.5, estimator="alarm") == 0
    assert page_cusum_readout(x, k=0.5, h=4.0, estimator="onset") == 0


def test_cusum_fallbacks_are_different_rules():
    # A sustained late shift plus one early noise spike: accumulated evidence and raw size
    # give different answers, which is the whole reason the fallback has to be declared.
    rng = np.random.default_rng(19)
    x = np.r_[rng.normal(size=60), rng.normal(loc=1.0, size=60)]
    x[7] = 6.0
    by_statistic = page_cusum_readout(x, k=0.5, h=1e6, fallback="statistic")
    by_series = page_cusum_readout(x, k=0.5, h=1e6, fallback="series")
    assert by_series == argmax_readout(x) == 7
    assert by_statistic == int(np.argmax(page_cusum(x, k=0.5)["statistic"]))
    assert by_statistic != by_series


def test_on_a_short_standardized_series_the_cusum_fallback_usually_is_the_argmax():
    # Documented, not incidental: with within-answer z-scoring and k > 0, a lone spike
    # dominates the accumulated statistic on a short series, so the statistic fallback
    # coincides with the raw argmax. This is why a CUSUM run over ~8 step scores is
    # expected to behave like argmax, and it is asserted so the expectation is explicit.
    rng = np.random.default_rng(23)
    agree = 0
    for _ in range(400):
        x = rng.normal(size=int(rng.integers(4, 12)))
        agree += page_cusum_readout(x, k=0.5, h=1e6, fallback="statistic") == argmax_readout(x)
    assert agree / 400 > 0.75


# --------------------------------------------------------------------- first crossing
def test_first_crossing_degenerates_to_argmax_at_q_one():
    rng = np.random.default_rng(3)
    for _ in range(50):
        x = rng.normal(size=rng.integers(3, 30))
        assert first_crossing_readout(x, q=1.0) == argmax_readout(x)


def test_first_crossing_is_earlier_than_argmax_when_the_bar_is_low():
    x = np.array([1.0, 0.2, 0.3, 5.0, 0.1])
    assert argmax_readout(x) == 3
    assert first_crossing_readout(x, q=0.5) == 0


def test_constant_series_is_handled_not_nan():
    x = np.full(9, 2.5)
    assert np.all(zscore_within(x) == 0.0)
    assert first_crossing_readout(x, q=0.9) == 0
    assert page_cusum_readout(x) == 0
    assert bocpd_readout(x) == 0


# ------------------------------------------------------------------------------ BOCPD
def _segment_log_evidence(values, observation_variance, prior_mean, prior_variance):
    """Marginal evidence of one segment, computed independently of the recursion."""
    n = len(values)
    cov = observation_variance * np.eye(n) + prior_variance * np.ones((n, n))
    delta = np.asarray(values) - prior_mean
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    return -0.5 * (n * np.log(2 * np.pi) + logdet + delta @ np.linalg.solve(cov, delta))


def _partitions(n):
    """Every ordered partition of 0..n-1 into contiguous blocks."""
    for cuts in itertools.chain.from_iterable(
            itertools.combinations(range(1, n), r) for r in range(n)):
        bounds = (0,) + cuts + (n,)
        yield [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def test_bocpd_matches_brute_force_partitions():
    hazard, obs, prior_mean, prior_var = 0.2, 0.7, 0.0, 1.3
    x = np.array([0.4, -1.1, 2.3, 0.2, -0.7])
    out = bocpd_filter(x, hazard=hazard, observation_variance=obs, prior_mean=prior_mean,
                       prior_variance=prior_var, standardize=False)
    for t in range(len(x)):
        prefix = x[:t + 1]
        total, reset = [], []
        for blocks in _partitions(len(prefix)):
            # P(partition): each internal boundary costs the hazard, each continuation
            # costs (1 - hazard). The first block's start is not a boundary event.
            boundaries = len(blocks) - 1
            continuations = len(prefix) - 1 - boundaries
            log_prior = boundaries * np.log(hazard) + continuations * np.log(1 - hazard)
            log_joint = log_prior + sum(
                _segment_log_evidence(prefix[a:b], obs, prior_mean, prior_var)
                for a, b in blocks)
            total.append(log_joint)
            if blocks[-1][0] == t:          # the last datum opened a fresh segment
                reset.append(log_joint)
        total = np.asarray(total)
        evidence = np.log(np.exp(total - total.max()).sum()) + total.max()
        assert np.isclose(out["log_predictive"][:t + 1].sum(), evidence, atol=1e-10)
        if t:
            reset = np.asarray(reset)
            mass = np.exp(np.log(np.exp(reset - reset.max()).sum()) + reset.max() - evidence)
            assert np.isclose(out["reset_probability"][t], mass, atol=1e-10)


def test_bocpd_truncation_is_close_to_exact():
    rng = np.random.default_rng(11)
    x = np.r_[rng.normal(size=60), rng.normal(loc=3.0, size=60)]
    exact = bocpd_filter(x, r_max=None)
    capped = bocpd_filter(x, r_max=24)
    assert np.allclose(exact["rise"], capped["rise"], atol=5e-3)
    assert argmax_readout(exact["rise"]) == argmax_readout(capped["rise"])


def test_bocpd_rise_finds_a_planted_shift():
    rng = np.random.default_rng(5)
    x = np.r_[rng.normal(scale=0.3, size=40), rng.normal(loc=4.0, scale=0.3, size=40)]
    assert abs(bocpd_readout(x, curve="rise") - 40) <= 1


def test_bocpd_reset_at_t0_is_the_hazard_by_construction():
    # Documented convention, asserted so it cannot change silently: the rise curve is
    # the one that keeps index 0 reachable, the reset curve is flat there.
    rng = np.random.default_rng(13)
    x = rng.normal(size=25)
    for hazard in (1 / 8, 1 / 32, 1 / 128):
        assert bocpd_filter(x, hazard=hazard)["reset_probability"][0] == pytest.approx(hazard)
    planted = np.r_[np.full(6, 6.0), np.zeros(30)]
    assert bocpd_readout(planted, curve="rise") == 0


# ------------------------------------------------------------------------ granularity
def test_token_index_to_step_maps_into_the_containing_span():
    spans = np.array([[0, 4], [4, 4], [4, 9], [9, 12]])   # note the empty step
    assert [token_index_to_step(i, spans) for i in (0, 3, 4, 8, 9, 11)] == [0, 0, 2, 2, 3, 3]
    assert token_index_to_step(99, spans) == 3


def test_token_index_to_step_rejects_bad_geometry():
    with pytest.raises(ValueError):
        token_index_to_step(0, np.zeros((0, 2), dtype=int))
    with pytest.raises(ValueError):
        token_index_to_step(0, np.arange(6))
