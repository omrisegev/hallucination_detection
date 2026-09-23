"""Tests for the extended within-step readouts.

The one that matters most is `test_matches_frozen_core_profiles`: the new reducers must
read the SAME answer-standardised numbers as the seven frozen readouts of
`cvf_v2.core.profiles`, otherwise a comparison between old and new readouts would be a
comparison between two standardisations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
from cvf_v2.core import profiles as frozen_profiles, encode  # noqa: E402
from spectral_utils.changepoint_step_readout_v1 import page_cusum  # noqa: E402
from spectral_utils.step_readouts_v1 import (  # noqa: E402
    EXT_READOUTS, answer_max_by_offsets, argmax_earliest, consensus_readout_choice,
    ext_profiles, page_first_crossing, page_threshold, robust_standardize_tokens,
    soft_mass, step_boxcar_max, step_frac_above, step_jump, step_mean, step_page_wmax,
    step_slope, step_std, step_topk, tied_argmax,
)


def _fixture(seed=3, T=300, C=11, steps=12):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, C)) * rng.uniform(.5, 3, C) + rng.normal(size=C)
    cuts = np.sort(rng.choice(np.arange(1, T), steps - 1, replace=False))
    bounds = np.r_[0, cuts, T]
    spans = np.column_stack([bounds[:-1], bounds[1:]])
    return x, spans


def test_matches_frozen_core_profiles():
    x, spans = _fixture()
    z = robust_standardize_tokens(x)
    frozen = frozen_profiles(x, spans)
    np.testing.assert_allclose(step_topk(z, spans, 5), frozen[:, :, 0], rtol=0, atol=0)
    np.testing.assert_allclose(step_topk(z, spans, 10), frozen[:, :, 1], rtol=0, atol=0)
    np.testing.assert_allclose(step_mean(z, spans), frozen[:, :, 3], rtol=0, atol=0)
    ext = ext_profiles(z, spans)
    assert ext.shape == (len(spans), 11, len(EXT_READOUTS)) and np.isfinite(ext).all()


def test_shapes_and_nan_rejection():
    x, spans = _fixture(T=40, C=2, steps=3)
    assert step_std(x, spans).shape == (3, 2)
    with pytest.raises(ValueError):
        step_mean(np.r_[x[:-1], [[np.nan, 0.]]], spans)
    with pytest.raises(ValueError):
        step_mean(x, [(0, 10), (10, 41)])
    with pytest.raises(ValueError):
        robust_standardize_tokens(np.zeros((0, 2)))


def test_frac_above_boundary_and_slope_on_ramp():
    x = np.array([[0.], [1.], [2.], [3.]])
    spans = np.array([[0, 4]])
    assert step_frac_above(x, spans, threshold=1.0)[0, 0] == pytest.approx(.75)  # tie counts as above
    assert step_frac_above(x, spans, threshold=-10.)[0, 0] == 1.
    assert step_slope(x, spans)[0, 0] == pytest.approx(1.)
    assert step_slope(x[:1], np.array([[0, 1]]))[0, 0] == 0.
    assert step_std(x[:1], np.array([[0, 1]]))[0, 0] == 0.


def test_jump_keeps_step_zero_reachable():
    x = np.r_[np.full((5, 1), 2.), np.full((5, 1), 2.), np.full((5, 1), 5.)]
    spans = np.array([[0, 5], [5, 10], [10, 15]])
    j = step_jump(x, spans)[:, 0]
    np.testing.assert_allclose(j, [2., 0., 3.])
    assert argmax_earliest(step_jump(np.full((6, 1), 4.), np.array([[0, 3], [3, 6]])))[0] == 0


def test_boxcar_matches_bruteforce():
    x, spans = _fixture(seed=5, T=120, C=3, steps=7)
    got = step_boxcar_max(x, spans, width=8)
    for s, (a, b) in enumerate(spans):
        w = min(8, b - a)
        for j in range(3):
            assert got[s, j] == pytest.approx(max(x[i:i + w, j].mean() for i in range(a, b - w + 1)))


def test_page_wmax_matches_recursion_and_is_monotone_in_k():
    x, spans = _fixture(seed=9, T=90, C=2, steps=5)
    z = robust_standardize_tokens(x)
    for k in [.25, .5, 1.]:
        got = step_page_wmax(z, spans, k=k)
        for j in range(2):
            s, loop = 0., []
            zz = (z[:, j] - z[:, j].mean()) / z[:, j].std()   # mean-centred increments, drift -k
            for v in zz:
                s = max(0., s + v - k); loop.append(s)
            loop = np.asarray(loop)
            np.testing.assert_allclose(got[:, j], [loop[a:b].max() for a, b in spans], atol=1e-12)
    assert (step_page_wmax(z, spans, k=.25) >= step_page_wmax(z, spans, k=1.)).all()


def test_first_crossing_on_planted_excursion_and_fallback():
    # Flat noise-free series, then a sustained lift from token 20 (step 2 of 4).
    x = np.r_[np.zeros(20), np.full(20, 3.)][:, None]
    spans = np.array([[0, 10], [10, 20], [20, 30], [30, 40]])
    wmax = step_page_wmax(x, spans, k=.5)
    profile, crossed = page_first_crossing(wmax, h=np.array([5.]))
    assert crossed[0] and argmax_earliest(profile)[0] == 2
    assert np.isneginf(profile[3, 0]) and np.isfinite(profile[:3, 0]).all()
    too_high, crossed = page_first_crossing(wmax, h=np.array([1e9]))
    assert not crossed[0] and np.array_equal(too_high, wmax)  # finite fallback, argmax of the statistic
    # Crossing on step 0 is reachable.
    early, crossed = page_first_crossing(np.array([[9.], [1.]]), h=np.array([5.]))
    assert crossed[0] and argmax_earliest(early)[0] == 0 and np.isneginf(early[1, 0])
    # Threshold is a quantile of per-answer maxima, labels never enter.
    maxima = answer_max_by_offsets(np.arange(10.)[:, None], np.array([0, 4, 10]))
    np.testing.assert_allclose(maxima[:, 0], [3., 9.])
    assert page_threshold(maxima, .5)[0] == pytest.approx(6.)
    with pytest.raises(ValueError):
        page_threshold(maxima, 1.5)


def test_soft_mass_matches_core_soft_encoding():
    p = np.array([[0., -np.inf], [2., 1.], [1., -np.inf]])
    # core.encode gives 2*cumsum(mass)-1 without the last row; recover mass and compare.
    cum = (encode(p, 'soft', 'pb') + 1) / 2
    mass = soft_mass(p)
    np.testing.assert_allclose(np.cumsum(mass, axis=0)[:-1], cum)
    assert mass[0, 1] == 0 and mass[2, 1] == 0 and mass[1, 1] == pytest.approx(1.)
    np.testing.assert_allclose(soft_mass(np.zeros((1, 3))), np.ones((1, 3)))


def test_tied_argmax_flags_discrete_channels():
    p = np.array([[1., 0.], [1., 2.], [0., 1.]])
    assert tied_argmax(p).tolist() == [True, False]


def test_consensus_recovers_a_planted_readout():
    rng = np.random.default_rng(11)
    answers, cells = [], []
    for i in range(120):
        S = rng.integers(3, 9); truth = rng.integers(0, S)
        p = rng.normal(size=(S, 5, 3)) * .3
        p[truth, :, 1] += 3.                     # readout 1 carries the shared signal
        p[:, :, 2] = rng.normal(size=(S, 5))     # readout 2 is noise
        p[:, 4, 1] = rng.normal(size=S)          # channel 4's readout 1 is broken; its readout 0 works
        p[truth, 4, 0] += 3.
        answers.append(p); cells.append('a' if i % 2 else 'b')
    choice, agreement = consensus_readout_choice(answers, base=0, sweeps=2, cells=cells)
    assert choice.tolist() == [1, 1, 1, 1, 0]
    assert agreement.shape == (2, 5, 3)
    masked, _ = consensus_readout_choice(answers, base=0, sweeps=2, candidate_mask=[True, False, True])
    assert 1 not in masked.tolist()
    with pytest.raises(ValueError):
        consensus_readout_choice([answers[0][:1]])
