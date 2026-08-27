#!/usr/bin/env python3
"""Focused mechanical tests for the graph-geometry post-report audit."""

from __future__ import annotations

from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.graph_geometry_selection_postreport_audit import (  # noqa: E402
    choose_max_mean,
    choose_one_se,
    corrected_actuator_semantics,
    matched_did,
)


def test_policy_selectors_are_separate():
    groups = ("a", "b", "c")
    priority = {"g0": 0, "g1": 1}
    conservative = ("g0", 0.03, 0.5)
    aggressive = ("g1", 3.0, 1.0)
    values = {
        conservative: {"a": 0.010, "b": 0.010, "c": 0.010},
        aggressive: {"a": 0.025, "b": 0.025, "c": -0.004},
    }
    one_se, one_diag = choose_one_se(values, groups, priority)
    max_mean, max_diag = choose_max_mean(values, groups, priority)
    assert one_se == conservative
    assert max_mean == aggressive
    assert one_diag["selected"]["mean"] != max_diag["selected"]["mean"]


def test_matched_did_subtracts_fixed_generalization_gap():
    value = matched_did(
        searched_inner=0.0060,
        fixed_inner=0.0045,
        searched_outer=0.0044,
        fixed_outer=0.0040,
    )
    assert abs(value - 0.0011) < 1e-15
    raw_searched_gap = 0.0060 - 0.0044
    assert abs(value - raw_searched_gap) > 1e-6


def test_actuator_legacy_column_is_reinterpreted_not_recomputed():
    rows = [
        {
            "geometry_id": "g0", "selector": "one_se",
            "trust_class": "canonical", "actuator": "full",
            "lambda_is_a_cross_parameter": "True",
        },
        {
            "geometry_id": "g0", "selector": "one_se",
            "trust_class": "canonical", "actuator": "cross",
            "lambda_is_a_cross_parameter": "False",
        },
    ]
    corrected = corrected_actuator_semantics(rows)
    assert corrected[0]["lambda_is_full_parameter"] is True
    assert corrected[1]["lambda_is_full_parameter"] is False
    assert all(row["cross_lambda_parameter"] is None for row in corrected)


def test_actuator_mismatch_fails_closed():
    bad = [{
        "geometry_id": "g0", "selector": "one_se",
        "trust_class": "canonical", "actuator": "cross",
        "lambda_is_a_cross_parameter": "True",
    }]
    try:
        corrected_actuator_semantics(bad)
    except RuntimeError:
        return
    raise AssertionError("misaligned legacy actuator semantics did not fail closed")


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} graph-geometry post-report audit tests")


if __name__ == "__main__":
    main()
