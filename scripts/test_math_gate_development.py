"""Deterministic unit checks for the cross-panel math gate bank."""
from __future__ import annotations

import numpy as np

from spectral_utils import gate_feature_readout as gate
from spectral_utils import math_gate_selection as selection


def _logprobs(tokens: int = 12, support: int = 50) -> np.ndarray:
    base = np.linspace(0.0, -7.0, support)
    rows = []
    for index in range(tokens):
        logits = base * (0.7 + 0.04 * index)
        logits -= np.log(np.exp(logits).sum())
        rows.append(logits)
    return np.asarray(rows)


def run() -> dict:
    values = np.arange(1.0, 13.0)
    readouts = gate.apply_temporal_readouts(values)
    assert tuple(readouts) == gate.TEMPORAL_READOUT_NAMES
    assert readouts["token_mean"] == 6.5
    assert readouts["token_top10"] == 7.5
    assert readouts["rolling8_max"] == 8.5
    assert readouts["region_q1_mean"] == 2.0
    assert readouts["region_q2_mean"] == 5.0
    assert readouts["region_q3_mean"] == 8.0
    assert readouts["region_q4_mean"] == 11.0
    assert abs(readouts["position_slope"] - 11.0) < 1e-12

    short = gate.apply_temporal_readouts(np.asarray([2.0, 5.0]))
    assert np.isfinite(list(short.values())).all()
    assert short["rolling8_max"] == 3.5

    logprobs = _logprobs()
    p15 = np.exp(logprobs[:, :15])
    q15 = p15 / p15.sum(axis=1, keepdims=True)
    entropy = -(q15 * np.log(q15 + 1e-12)).sum(axis=1)
    detectors = gate.answer_temporal_detectors(logprobs, entropy)
    assert tuple(detectors) == gate.TEMPORAL_METHODS
    assert len(detectors) == 121
    assert np.isfinite(list(detectors.values())).all()
    assert abs(
        detectors["entropy_native__token_mean"]
        - detectors["q15_H1__token_mean"]
    ) < 3e-12
    assert all("mean_step_top10" not in name for name in detectors)

    cells = np.asarray(["a"] * 4 + ["b"] * 4)
    families = np.asarray(["gsm8k"] * 4 + ["math500"] * 4)
    raw = np.asarray([3.0, 1.0, 4.0, 2.0, 40.0, 10.0, 30.0, 20.0])
    percentile = selection.percentile_by_cell(raw, cells)
    np.testing.assert_allclose(
        percentile,
        np.asarray([0.625, 0.125, 0.875, 0.375, 0.875, 0.125, 0.625, 0.375]),
    )
    y = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    scored = selection.select_q(y, percentile, cells, families)
    assert scored["selected_q"] == 0.5
    assert scored["selected"]["family_macro"]["macro_f1"] == 1.0

    simplex = selection.fit_simplex(
        np.column_stack((percentile, 1.0 - percentile)), y, cells, families
    )
    assert np.all(simplex["weights"] >= 0)
    assert abs(simplex["weights"].sum() - 1.0) < 1e-12
    assert np.all(
        (simplex["weights"] == 0.0)
        | (simplex["weights"] >= selection.SIMPLEX_EPSILON)
    )
    return {
        "schema": "math-gate-development-unit-v1",
        "status": "PASS",
        "signals": len(gate.SIGNAL_NAMES),
        "readouts": len(gate.TEMPORAL_READOUT_NAMES),
        "candidates": len(detectors),
    }


if __name__ == "__main__":
    print(run())
