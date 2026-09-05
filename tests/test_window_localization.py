import numpy as np
import pytest

from spectral_utils.window_localization import (
    WINDOW_FEATURE_NAMES, WindowMatrix, build_window_matrix, feasible_widths, make_window_plan,
    matrix_diagnostics, tokens_to_official_steps, windows_to_tokens,
)


def test_dense_windows_do_not_inflate_fit_sample_count():
    plan = make_window_plan(101, 32, 7)
    assert len(plan.starts) > len(plan.fit_indices)
    assert plan.starts[plan.fit_indices].tolist() == [0, 32, 64]
    assert plan.ends[-1] == 101
    assert np.allclose(windows_to_tokens(plan, np.ones(len(plan.starts))), 1)


def test_short_features_cannot_silently_become_zero_measurements():
    with pytest.raises(ValueError, match="STFT"):
        make_window_plan(256, 16)
    with pytest.raises(ValueError, match="TRACE_TOO_SHORT"):
        make_window_plan(20, 32)


def test_gap_stride_is_rejected():
    with pytest.raises(ValueError, match="stride"):
        make_window_plan(256, 32, 64)


def test_full_feature_schema_is_retained_and_constant_length_is_disclosed():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(256, 3))
    names = ("entropy_series", "spilled_series", "energy_series")
    result = build_window_matrix(raw, names, make_window_plan(256, 32))
    assert result.values.shape == (8, 30)
    length = WINDOW_FEATURE_NAMES.index("trace_length")
    assert np.all(result.values[:, length] == 32)
    assert not result.active[length]
    assert result.inactive_reasons[length] == "CONSTANT_ON_FIT_WINDOWS"
    lp = WINDOW_FEATURE_NAMES.index("mean_top1_logprob")
    assert not result.active[lp]
    assert result.inactive_reasons[lp] == "UNAVAILABLE_OR_NONFINITE_ON_FIT_WINDOWS"


def test_recomputed_window_does_not_inherit_outside_answer_or_prefix_statistics():
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(128, 3))
    names = ("entropy_series", "spilled_series", "energy_series")
    plan = make_window_plan(128, 32)
    first = build_window_matrix(raw, names, plan)
    changed = raw.copy()
    changed[:32] *= 1000
    second = build_window_matrix(changed, names, plan)
    # An earlier-window outlier must not alter the other windows' CUSUM,
    # spectrum, energy or raw feature values. Normalization is not pooled here.
    np.testing.assert_allclose(first.values[1:], second.values[1:], equal_nan=True)


def test_known_window_overlap_maps_to_original_official_spans():
    plan = make_window_plan(80, 32)
    token = windows_to_tokens(plan, [1, 3, 5])
    assert np.all(token[:32] == 1)
    assert np.all(token[32:48] == 3)
    assert np.all(token[48:64] == 4)
    assert np.all(token[64:] == 5)
    np.testing.assert_allclose(tokens_to_official_steps(token, [16, 48], [48, 80]), [2, 4.5])
    with pytest.raises(ValueError, match="finite"):
        windows_to_tokens(plan, [1, np.nan, 5])


def test_geometry_filter_does_not_claim_full_rank_or_accuracy():
    rows = feasible_widths(270, widths=(16, 32, 48))
    assert not rows[0]["feasible_geometry"]
    assert rows[1]["feasible_geometry"]
    assert rows[1]["fit_windows"] == 8
    assert rows[1]["sample_covariance_rank_cap"] == 7
    assert not rows[2]["feasible_geometry"]


def test_constant_trace_is_flagged_without_inventing_localization_signal():
    raw = np.ones((256, 3))
    names = ("entropy_series", "spilled_series", "energy_series")
    matrix = build_window_matrix(raw, names, make_window_plan(256, 32))
    assert matrix_diagnostics(matrix)["status"] == "INSUFFICIENT_VARIATION"


def test_two_window_rank_respects_centering_even_with_large_feature_offsets():
    x = np.asarray([[1e9 + .1, 1e8 + .2, 1e7 + .3],
                    [1e9 + .7, 1e8 + .9, 1e7 + .5]])
    matrix = WindowMatrix(make_window_plan(64, 32), x, ("a", "b", "c"),
                          np.ones(3, dtype=bool), (None, None, None))
    result = matrix_diagnostics(matrix)
    assert result["rank"] == result["rank_cap"] == 1
    assert result["participation_rank"] == pytest.approx(1)
