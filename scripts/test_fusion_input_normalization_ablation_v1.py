#!/usr/bin/env python3
"""Unit checks for the input-normalization ablation."""
from __future__ import annotations

import numpy as np

from scripts.run_fusion_input_normalization_ablation_v1 import transform_bank


def run() -> dict:
    rng = np.random.default_rng(2026091403)
    raw = rng.normal(size=(31, 4)) * np.array([7.0, 2.0, 0.4, 0.1]) + np.array([3.0, -2.0, 0.7, 0.05])
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    z = (raw - mean) / scale
    arrays = {"z": z, "mean": mean, "scale": scale}

    actual_z = transform_bank(arrays, "answer_z")
    actual_scale = transform_bank(arrays, "scale_only")
    actual_raw = transform_bank(arrays, "raw")
    np.testing.assert_allclose(actual_z, z, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(actual_scale, raw / scale, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(actual_raw, raw, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(actual_z.mean(axis=0), 0.0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(actual_z.std(axis=0), 1.0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(actual_scale.std(axis=0), 1.0, atol=1e-14, rtol=0.0)
    np.testing.assert_array_equal(np.argsort(actual_z.mean(axis=1)), np.argsort(actual_scale.mean(axis=1)))
    return {"schema": "fusion-input-normalization-unit-v1", "status": "PASS", "checks": 7}


if __name__ == "__main__":
    print(run())
