#!/usr/bin/env python3
"""Focused unit checks for the frozen Phase-1 baseline runner."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "scripts/reasoning_localization/run_phase1_baseline.py"
SPEC = importlib.util.spec_from_file_location("reasoning_phase1_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Phase1RunnerTests(unittest.TestCase):
    def test_topk_mean_handles_short_steps_without_padding(self) -> None:
        score = RUNNER.topk_step_mean(
            [1.0, 4.0, 2.0, 8.0, 3.0], [0, 3], [3, 5], k=5
        )
        np.testing.assert_allclose(score, [7.0 / 3.0, 5.5])

    def test_topk_mean_uses_only_largest_k_values(self) -> None:
        score = RUNNER.topk_step_mean(range(10), [0], [10], k=3)
        np.testing.assert_allclose(score, [8.0])

    def test_common_detector_is_monotone_geometric_rank_fusion(self) -> None:
        cell = SimpleNamespace(
            cell_id="synthetic",
            segment_starts=np.array([0, 1, 2]),
            segment_offsets=np.array([0, 2, 3]),
            method_ids=("other", RUNNER.GLOBAL_METHOD),
            response_scores=np.array([[0.0, 0.0], [2.0, 1.0]]),
        )
        observed = RUNNER.combine_with_common_detector(cell, [1.0, 3.0, 2.0])
        local = RUNNER.empirical_midrank(np.array([1.0, 3.0, 2.0]))
        response = np.repeat(RUNNER.empirical_midrank(np.array([2.0, 1.0])), [2, 1])
        np.testing.assert_allclose(observed, np.sqrt(local * response))

    def test_registered_order_is_exact_and_current_r2_is_not_historical(self) -> None:
        self.assertEqual(
            RUNNER.VARIANTS,
            (
                "R0_ENTROPY_MAX",
                "R1_ENTROPY_TOP5",
                "R2_FAMILY6_TOP5_CURRENT",
                "R3_IU29",
                "R4_MIND_GAP",
            ),
        )


if __name__ == "__main__":
    unittest.main()
