"""Mechanism checks for cumulative q15 plus tail15 gate integration."""
from __future__ import annotations

import numpy as np

from scripts.run_integrated_q15_tail15_gate_replay_v1 import gated_prediction


def run():
    detector = np.asarray([0.1, 0.3, 0.5, np.nan])
    peak = np.asarray([2, 1, 0, 3])
    valid = np.asarray([True, True, True, True])
    prediction, usable = gated_prediction(detector, 0.3, peak, valid)
    assert prediction.tolist() == [-1, 1, 0, -1]
    assert usable.tolist() == [True, True, True, False]
    prediction, usable = gated_prediction(detector, 0.3, peak, np.asarray([True, False, True, True]))
    assert prediction.tolist() == [-1, -1, 0, -1]
    assert usable.tolist() == [True, False, True, False]
    return {"status": "PASS", "checks": 4}


if __name__ == "__main__":
    print(run())
