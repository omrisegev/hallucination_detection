"""Unit tests for probability normalization and preprocessing ablation v1."""
from __future__ import annotations

import numpy as np

from spectral_utils import probability_normalization_ablation as model


def _logprobs(seed=7, tokens=37, width=50):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(tokens):
        full = rng.dirichlet(np.geomspace(3.0, 0.2, width + 17))
        rows.append(np.log(np.sort(full)[::-1][:width]))
    return np.asarray(rows)


def test_raw_head_scale_invariance():
    logprobs = _logprobs()
    for k in (15, 50):
        p, q = model.conditional_head(logprobs, k)
        for alpha in (0.0, 0.75, 1.0, 2.0):
            raw = model.escort_varentropy_raw_head(p, alpha)
            conditional = model.escort_varentropy_distribution(q, alpha)
            np.testing.assert_allclose(raw, conditional, atol=2e-9, rtol=2e-9)


def test_tail_bucket_is_probability_distribution():
    distribution, tail = model.tail_bucket_distribution(_logprobs(), 15)
    assert distribution.shape == (37, 16)
    assert tail.shape == (37,)
    assert np.all(tail >= 0)
    np.testing.assert_allclose(distribution.sum(axis=1), 1.0, atol=1e-12, rtol=0)


def test_feature_bank_preprocessing_invariances():
    bank = model.feature_bank(_logprobs())
    assert set(bank["local"]) == set(model.LOCAL_METHODS)
    raw = bank["local"]["equal4_raw"]
    centered = bank["local"]["equal4_center_only"]
    scaled = bank["local"]["equal4_scale_only"]
    answer_z = bank["local"]["equal4_answer_z"]
    np.testing.assert_array_equal(np.argsort(raw), np.argsort(centered))
    np.testing.assert_array_equal(np.argsort(scaled), np.argsort(answer_z))
    assert max(bank["invariance_error"].values()) < 2e-8


def test_grouped_standardizer_equalizes_group_not_answer_count():
    stats = {
        0: {"m_mean": np.array([0.0]), "m_second": np.array([1.0])},
        1: {"m_mean": np.array([2.0]), "m_second": np.array([5.0])},
        2: {"m_mean": np.array([10.0]), "m_second": np.array([101.0])},
    }
    groups = {0: "a", 1: "a", 2: "b"}
    fitted = model.fit_grouped_standardizer(stats, groups, [0, 1, 2], "m")
    # Group a mean is 1, group b mean is 10, and groups receive equal weight.
    np.testing.assert_allclose(fitted.mean, [5.5], atol=1e-12, rtol=0)
    assert fitted.training_groups == ("a", "b")
    assert fitted.training_answers == 3


def test_step_readout_uses_top10_or_all_if_shorter():
    token = np.arange(15, dtype=float)
    spans = np.array([[0, 5], [5, 15]])
    observed = model.step_readout(token, spans)
    np.testing.assert_allclose(observed, [2.0, 9.5], atol=0, rtol=0)


def run():
    tests = [
        test_raw_head_scale_invariance,
        test_tail_bucket_is_probability_distribution,
        test_feature_bank_preprocessing_invariances,
        test_grouped_standardizer_equalizes_group_not_answer_count,
        test_step_readout_uses_top10_or_all_if_shorter,
    ]
    for test in tests:
        test()
    return {"status": "PASS", "tests": len(tests)}


if __name__ == "__main__":
    print(run())
