#!/usr/bin/env python3
"""Focused mechanical tests for the target-free Feature Contract V2."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("build_feature_contract_v2.py")
SPEC = importlib.util.spec_from_file_location("build_feature_contract_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    names = (
        "epr",
        "trace_length",
        "low_band_power",
        "high_band_power",
        "hl_ratio",
        "mean_logprob_entropy",
        "logprob_margin",
    )
    low = np.asarray([0.2, 0.4, 0.8], dtype=np.float64)
    high = np.asarray([0.3, 0.1, 0.6], dtype=np.float64)
    X = np.column_stack(
        (
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            low,
            high,
            high / (low + 1e-12),
            [1.1, 2.2, 3.3],
            [0.4, 0.5, 0.6],
        )
    )
    Y, output_names, metrics = MODULE.transform_contract(X, names)
    assert output_names == (
        "entropy_common",
        "entropy_support_delta",
        "low_band_power",
        "high_band_power",
        "logprob_margin",
    )
    common, delta = Y[:, 0], Y[:, 1]
    assert np.array_equal(common - delta / 2, X[:, 0])
    assert np.allclose(common + delta / 2, X[:, 5], atol=1e-15, rtol=0)
    assert metrics["entropy_roundtrip_max_abs"] <= 5e-16
    assert metrics["hl_ratio_roundtrip_max_abs"] == 0.0
    assert "trace_length" not in output_names and "hl_ratio" not in output_names

    row_ids = ("r0", "r1", "r2", "r3")
    group_ids = ("g0", "g1", "g2", "g3")
    first = MODULE.stable_hex(MODULE.SCHEMA, *row_ids)
    second = MODULE.stable_hex(MODULE.SCHEMA, *row_ids)
    assert first == second
    assert len(set(group_ids)) == len(group_ids)
    print("feature_contract_v2 mechanical tests: PASS")


if __name__ == "__main__":
    main()
