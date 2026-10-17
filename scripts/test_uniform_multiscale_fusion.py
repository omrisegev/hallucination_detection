"""Mechanism tests for the benchmark-uniform multiscale fusion experiment."""
from __future__ import annotations

import inspect
import numpy as np

from spectral_utils import uniform_multiscale_fusion as model


def _logprobs(seed=31, tokens=67, width=50):
    rng = np.random.default_rng(seed)
    logits = np.sort(rng.normal(size=(tokens, width)), axis=1)[:, ::-1]
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def _records():
    return [
        {"cell": "pb_a", "group_id": "g0"},
        {"cell": "pb_a", "group_id": "g1"},
        {"cell": "prm_a", "group_id": "g2"},
        {"cell": "prm_a", "group_id": "g3"},
    ]


def test_bank_and_q15_identity():
    bank = model.feature_bank(_logprobs())
    assert bank["bank"].shape == (67, 8)
    assert bank["q15_frozen_identity_max"] < 2e-9
    assert tuple(model.SUPPORTS) == ("q15", "q50", "ms8")


def test_step_bank_uses_per_view_top10():
    bank = model.feature_bank(_logprobs())["bank"]
    spans = np.array([[0, 7], [7, 29], [29, 67]])
    steps = model.step_bank(bank, spans)
    assert steps.shape == (3, 8)
    for j in range(8):
        np.testing.assert_allclose(steps[:, j], [np.sort(bank[a:b, j])[-min(10, b-a):].mean() for a, b in spans])


def test_global_scale_is_one_shared_hierarchy():
    stats = {
        i: {"mean": np.arange(8, dtype=float) + i, "second": (np.arange(8, dtype=float) + i) ** 2 + 1}
        for i in range(4)
    }
    fitted = model.fit_global_scale(stats, _records(), range(4))
    assert fitted.panels == ("pb", "prm")
    assert fitted.cells == ("pb_a", "prm_a")
    assert fitted.training_groups == ("g0", "g1", "g2", "g3")
    assert np.all(fitted.scale > 0)


def test_simplex_constraints_threshold_and_balancing():
    records = _records()
    steps = {
        0: np.array([[0]*8, [1]*8, [2]*8], float),
        1: np.array([[0]*8, [1]*8, [3]*8], float),
        2: np.array([[0]*8, [2]*8, [4]*8], float),
        3: np.array([[0]*8, [3]*8, [5]*8], float),
    }
    # Break symmetry so SLSQP has a stable direction.
    for i in steps:
        steps[i][:, 1:] += np.arange(7)[None] * 0.01
    labels = {i: np.array([0, 0, 1]) for i in range(4)}
    weights, info = model.fit_simplex(steps, labels, records, range(4), model.SUPPORTS["q15"], np.ones(8))
    assert info["status"] == "FIT"
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
    assert np.all((weights == 0) | (weights >= model.WEIGHT_EPSILON))
    np.testing.assert_allclose([info["negative_weight"], info["positive_weight"]], [0.5, 0.5], atol=1e-12)


def test_natural_reconstruction_and_centered_ordering():
    rng = np.random.default_rng(9)
    steps = rng.normal(size=(11, 8))
    scale = np.geomspace(0.2, 4.0, 8)
    for columns in model.SUPPORTS.values():
        natural = model.natural_weights(columns, scale)
        observed = model.score(steps, columns, scale, natural)
        expected = steps[:, columns].sum(axis=1) / scale[columns].sum()
        np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)
        equal = np.full(len(columns), 1 / len(columns))
        full = model.score(steps, columns, scale, equal)
        centered = model.score(steps, columns, scale, equal, centered=True)
        np.testing.assert_array_equal(np.argsort(full, kind="stable"), np.argsort(centered, kind="stable"))


def test_label_firewall_surface():
    for function in (model.feature_bank, model.fit_global_scale, model.step_bank, model.score):
        assert "labels" not in inspect.signature(function).parameters
    assert "labels" in inspect.signature(model.fit_simplex).parameters


def run():
    tests = [
        test_bank_and_q15_identity,
        test_step_bank_uses_per_view_top10,
        test_global_scale_is_one_shared_hierarchy,
        test_simplex_constraints_threshold_and_balancing,
        test_natural_reconstruction_and_centered_ordering,
        test_label_firewall_surface,
    ]
    for test in tests:
        test()
    return {"status": "PASS", "tests": len(tests)}


if __name__ == "__main__":
    print(run())
