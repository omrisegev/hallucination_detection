"""Numerical and label-firewall tests for Renyi position-varying fusion."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spectral_utils import renyi_position_fusion as model
from spectral_utils.answer_position_fusion import position_overlap
from spectral_utils.renyi_alpha_sweep import anchor_stream, sweep_matrix


def _logprobs(rng, tokens=83, width=50):
    logits = rng.normal(size=(tokens, width))
    logits.sort(axis=1)
    logits = logits[:, ::-1]
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def run():
    rng = np.random.default_rng(20260914)
    logprobs = _logprobs(rng)
    features = model.feature_bank(logprobs)
    checks = []

    assert features["z"].shape == (83, 4)
    np.testing.assert_allclose(features["z"].mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(features["z"].std(0), 1.0, atol=1e-12)
    checks.append("frozen four-view bank and answer standardization")

    full, names = sweep_matrix(logprobs)
    anchor = anchor_stream(logprobs)
    for method, name in zip(model.SINGLE_METHODS, model.FEATURE_NAMES):
        expected = full[:, names.index(name)].copy()
        if name.startswith("ve") and np.corrcoef(expected, anchor)[0, 1] < 0:
            expected *= -1.0
        np.testing.assert_array_equal(features["singles"][method], expected)
    np.testing.assert_allclose(features["singles"]["view__ve1"], anchor, atol=1e-12, rtol=1e-10)
    checks.append("Stage-3b single-view orientation and VE1 identity")

    stats = model.regional_statistics(features["z"], "training-answer")
    shifted = model.regional_statistics(features["z"] + 7.0, "training-answer")
    for key in ("real", "shuffle"):
        covariance = stats[key + "_second"] - np.einsum(
            "ji,jk->jik", stats[key + "_mean"], stats[key + "_mean"]
        )
        shifted_covariance = shifted[key + "_second"] - np.einsum(
            "ji,jk->jik", shifted[key + "_mean"], shifted[key + "_mean"]
        )
        np.testing.assert_allclose(covariance, shifted_covariance, atol=1e-11)
    checks.append("whole-answer regional covariance and translation invariance")

    external, fit_info = model.fit_external_model(stats, group_count=41)
    assert set(fit_info) == {
        "external_iu_static",
        "external_iu_position_mean",
        "external_iu_position",
        "external_iu_position_shuffled",
    }
    for name in fit_info:
        assert external[name]["coefficients"].shape == (model.BINS, 4)
    checks.append("stationary and position-dependent external IU fits")

    spans = np.array([[0, 7], [7, 25], [25, 57], [57, 83]])
    scores, health, maps = model.score_answer(features, spans, external, "held-answer")
    assert set(scores) == set(model.METHODS)
    assert all(value.shape == (4,) for value in scores.values())
    assert all(value.shape == (model.BINS, 4) for value in maps.values())
    assert all(entry["status"] == "OK" for entry in health.values()), health
    checks.append("complete method roster, Top10 step scores and compact weight maps")

    # No borrowing at alpha zero is an exact replay of the local IU baseline.
    zero_methods = (
        "local_iu",
        "local_shrink_pooled",
        "local_shrink_position",
        "local_shrink_position_scale_only",
        "local_shrink_position_shuffled",
    )
    zero_scores, zero_health, zero_maps = model.score_answer(
        features, spans, external, "held-answer", methods=zero_methods, alpha=0.0
    )
    for name in zero_methods[1:]:
        np.testing.assert_array_equal(zero_scores[name], zero_scores["local_iu"])
        np.testing.assert_array_equal(zero_maps[name], zero_maps["local_iu"])
    assert all(entry["status"] == "OK" for entry in zero_health.values())
    checks.append("alpha-zero exact local-IU replay")

    scale_info = health["local_shrink_position_scale_only"]
    assert scale_info["direction_change_mean"] == 0.0
    checks.append("scale-only control removes position-dependent direction changes")

    first = position_overlap(83, "held-answer", True)
    second = position_overlap(83, "held-answer", True)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(1), 1.0, atol=1e-15)
    checks.append("deterministic position shuffle preserves every token row")

    public = (model.feature_bank, model.regional_statistics, model.fit_external_model, model.prepare_local, model.score_answer)
    banned = {"label", "labels", "target", "targets", "error_position"}
    for function in public:
        assert banned.isdisjoint(inspect.signature(function).parameters)
    checks.append("public fit and score APIs accept no benchmark labels")

    invalid = [np.zeros((2, 50)), np.full((8, 50), np.nan)]
    for value in invalid:
        try:
            model.feature_bank(value)
        except (ValueError, FloatingPointError):
            pass
        else:
            raise AssertionError("invalid bank silently accepted")
    checks.append("short and nonfinite inputs fail without fallback")

    return dict(status="PASS", checks=checks, count=len(checks))


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
