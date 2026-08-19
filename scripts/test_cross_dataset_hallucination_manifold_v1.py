#!/usr/bin/env python3
"""Focused mechanical tests for the cross-dataset manifold diagnostic."""

from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.cross_dataset_hallucination_manifold_v1 import (
    covariance,
    family,
    gaussian_log_density,
    unit,
)


def main():
    assert family("trace_math500_qwenmath15b_k10") == "math500"
    assert family("lapeigvals_gsm8k_llama8b") == "gsm8k"
    assert np.allclose(unit(np.array([3.0, 4.0])), [0.6, 0.8])
    rows = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    assert np.allclose(covariance(rows), [[1.0, 2.0], [2.0, 4.0]])
    model = {"mean": np.zeros(2), "precision": np.eye(2), "logdet": 0.0}
    scores = gaussian_log_density(np.array([[0.0, 0.0], [1.0, 0.0]]), model)
    assert scores[0] > scores[1]
    print("cross-dataset manifold focused tests: PASS")


if __name__ == "__main__":
    main()
