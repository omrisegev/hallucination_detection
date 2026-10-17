#!/usr/bin/env python3
"""Unit checks for the preregistered Phase-2 reducer ladder."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from scripts.reasoning_localization import integrate_phase2_reducer_candidate as INTEGRATOR


REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "scripts/reasoning_localization/run_phase2_reducer.py"
SPEC = importlib.util.spec_from_file_location("phase2_reducer", PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ReducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.arange(1.0, 11.0)
        self.starts = [0]
        self.ends = [10]

    def score(self, variant: str) -> float:
        return float(RUNNER.reduce_steps(variant, self.values, self.starts, self.ends)[0])

    def test_fixed_k_ladder(self) -> None:
        self.assertEqual(10.0, self.score("P2R_A_MAX_K1"))
        self.assertEqual(9.5, self.score("P2R_A_TOPK2"))
        self.assertEqual(9.0, self.score("P2R_A_TOPK3"))
        self.assertEqual(8.0, self.score(RUNNER.REFERENCE))
        self.assertEqual(6.5, self.score("P2R_A_TOPK8"))
        self.assertEqual(5.5, self.score("P2R_A_TOPK10"))

    def test_fraction_quantile_mean_and_median(self) -> None:
        self.assertEqual(9.0, self.score("P2R_A_TOPQ25"))
        self.assertEqual(8.0, self.score("P2R_A_TOPQ50"))
        self.assertEqual(10.0, self.score("P2R_A_TOPQ10_EXPLORATORY"))
        self.assertEqual(10.0, self.score("P2R_A_TOPQ05_EXPLORATORY"))
        self.assertEqual(7.75, self.score("P2R_A_QUANTILE75"))
        self.assertEqual(9.1, self.score("P2R_A_QUANTILE90"))
        self.assertEqual(5.5, self.score("P2R_A_MEAN_ALL"))
        self.assertEqual(5.5, self.score("P2R_A_MEDIAN"))

    def test_fixed_k_clamps_to_short_step(self) -> None:
        observed = RUNNER.reduce_steps(RUNNER.REFERENCE, [2.0, 4.0], [0], [2])
        np.testing.assert_allclose(observed, [3.0])

    def test_exploratory_top_fraction_uses_ceil_and_minimum_one(self) -> None:
        values = np.arange(1.0, 22.0)
        top_ten = RUNNER.reduce_steps(
            "P2R_A_TOPQ10_EXPLORATORY", values, [0], [21]
        )
        top_five = RUNNER.reduce_steps(
            "P2R_A_TOPQ05_EXPLORATORY", values, [0], [21]
        )
        # ceil(0.10 * 21) = 3, while ceil(0.05 * 21) = 2.
        np.testing.assert_allclose(top_ten, [20.0])
        np.testing.assert_allclose(top_five, [20.5])

    def test_length_stratum_boundaries(self) -> None:
        cuts = {"q1": 4.0, "q2": 8.0}
        self.assertEqual("short", RUNNER._stratum(4, cuts))
        self.assertEqual("medium", RUNNER._stratum(8, cuts))
        self.assertEqual("long", RUNNER._stratum(9, cuts))

    def test_reference_is_first_in_frozen_order(self) -> None:
        self.assertEqual(RUNNER.REFERENCE, RUNNER.STAGE_A_VARIANTS[0])

    def test_candidate_statistical_status_keeps_harm_and_hard_failure_distinct(self) -> None:
        supported_harm = {"delta": -0.03, "ci_low": -0.04, "ci_high": -0.02}
        self.assertEqual(
            "SUPPORTED_HARM",
            INTEGRATOR.statistical_status(supported_harm, hard_failure=False),
        )
        self.assertEqual(
            "HARD_FAILURE",
            INTEGRATOR.statistical_status(supported_harm, hard_failure=True),
        )

    def test_positive_interval_crossing_zero_is_promising_not_rejected(self) -> None:
        row = {"delta": 0.004, "ci_low": -0.006, "ci_high": 0.014}
        self.assertEqual(
            "PROMISING_UNCONFIRMED",
            INTEGRATOR.statistical_status(row, hard_failure=False),
        )


if __name__ == "__main__":
    unittest.main()
