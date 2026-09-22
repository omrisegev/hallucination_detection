#!/usr/bin/env python3
"""Unit tests for the frozen Phase-2 atomic score transforms."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.reconstruction_benchmark.localization_contract import empirical_midrank
from scripts.reasoning_localization.run_phase2_atomic_c1 import (
    fuse_step_channels,
    response_reset_swvar,
    suffix_invariance_audit,
    trailing_population_variance,
)
from scripts.reasoning_localization.run_phase2_atomic_c2 import (
    adaptive_trailing_population_variance,
    adaptive_window,
    adaptive_suffix_invariance_audit,
)
from scripts.reasoning_localization.run_phase2_atomic_remaining import (
    causal_cusum,
    edis_onset,
    ewma16,
    persistence,
    positive_area,
    prefix_replay_error,
    response_map,
)
from scripts.reasoning_localization import run_phase2_confirmation as confirmation


class AtomicC1TransformTests(unittest.TestCase):
    def test_trailing_variance_matches_direct_population_variance(self) -> None:
        values = np.asarray([1.0, 4.0, -2.0, 3.0, 9.0, 0.5], dtype=np.float64)
        actual = trailing_population_variance(values, window=3)
        expected = np.asarray([
            np.var(values[max(0, index - 2):index + 1], ddof=0)
            for index in range(len(values))
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)

    def test_response_boundaries_reset_history(self) -> None:
        values = np.asarray([0.0, 2.0, 4.0, 100.0, 102.0], dtype=np.float64)
        actual = response_reset_swvar(values, [0, 3, 5])
        self.assertEqual(0.0, actual[3])
        np.testing.assert_allclose(
            actual,
            np.concatenate((
                trailing_population_variance(values[:3]),
                trailing_population_variance(values[3:]),
            )),
            atol=0.0,
            rtol=0.0,
        )

    def test_suffix_invariance_is_exact_for_prefix_replay(self) -> None:
        values = np.linspace(-3.0, 7.0, 57) ** 2 / 11.0
        self.assertLessEqual(suffix_invariance_audit(values), 1e-12)

    def test_fusion_endpoints_and_equal_weight(self) -> None:
        entropy = np.asarray([3.0, 1.0, 1.0, 8.0])
        variance = np.asarray([0.0, 9.0, 2.0, 2.0])
        entropy_rank = empirical_midrank(entropy)
        variance_rank = empirical_midrank(variance)
        np.testing.assert_allclose(
            fuse_step_channels(entropy, variance, sw_weight=0.0), entropy_rank
        )
        np.testing.assert_allclose(
            fuse_step_channels(entropy, variance, sw_weight=1.0), variance_rank
        )
        np.testing.assert_allclose(
            fuse_step_channels(entropy, variance),
            0.5 * entropy_rank + 0.5 * variance_rank,
        )


class AtomicC2TransformTests(unittest.TestCase):
    def test_adaptive_window_matches_frozen_floor_and_clip_rule(self) -> None:
        self.assertEqual(3, adaptive_window(1))
        self.assertEqual(3, adaptive_window(39))
        self.assertEqual(4, adaptive_window(40))
        self.assertEqual(31, adaptive_window(319))
        self.assertEqual(32, adaptive_window(320))
        self.assertEqual(32, adaptive_window(1000))

    def test_adaptive_variance_matches_direct_prefix_calculation(self) -> None:
        values = np.linspace(-4.0, 5.0, 83) ** 2
        actual = adaptive_trailing_population_variance(values)
        expected = []
        for index in range(len(values)):
            prefix_length = index + 1
            width = min(prefix_length, adaptive_window(prefix_length))
            expected.append(np.var(values[prefix_length - width:prefix_length], ddof=0))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    def test_adaptive_transform_is_suffix_invariant(self) -> None:
        values = np.sin(np.linspace(0.0, 15.0, 777))
        self.assertLessEqual(adaptive_suffix_invariance_audit(values), 1e-12)


class AtomicRemainingTransformTests(unittest.TestCase):
    def test_two_sided_cusum_uses_reset_recursion(self) -> None:
        values = np.asarray([1.0, 1.0, -4.0, -1.0, 2.0])
        np.testing.assert_allclose(causal_cusum(values), [1.0, 2.0, 4.0, 5.0, 3.0])

    def test_response_map_resets_every_transform(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 10.0, 11.0])
        actual = response_map(values, [0, 3, 5], causal_cusum)
        expected = np.concatenate([causal_cusum(values[:3]), causal_cusum(values[3:])])
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

    def test_registered_dsp_operators_are_prefix_invariant(self) -> None:
        values = np.sin(np.linspace(0.0, 12.0, 101)) + np.linspace(-1.0, 1.0, 101)
        for transform in (ewma16, positive_area, persistence):
            self.assertLessEqual(prefix_replay_error(values, transform), 1e-12)

    def test_edis_onset_is_causal_and_nonnegative(self) -> None:
        values = np.asarray([0.0, 0.2, 2.0, -1.0, 0.5, 2.1])
        output = edis_onset(values)
        self.assertTrue(np.all(output >= 0.0))
        self.assertGreater(output[2], 0.0)
        self.assertLessEqual(prefix_replay_error(values, edis_onset), 1e-12)


class Phase2ConfirmationContractTests(unittest.TestCase):
    def test_confirmation_is_bounded_to_two_candidates_and_four_llama_cells(self) -> None:
        self.assertEqual(("P2C_C7_EDIS_LLAMA4", "P2C_C8_INNOV_LLAMA4"), confirmation.VARIANTS)
        self.assertEqual(4, len(confirmation.LLAMA_CELLS))
        self.assertTrue(all(cell.endswith("_llama31_8b") for cell in confirmation.LLAMA_CELLS))
        self.assertEqual(4, confirmation.PRIMARY_FAMILY_SIZE)

    def test_primary_contrast_uses_simultaneous_family_interval(self) -> None:
        left = {
            "by_cell": [{"cell_id": f"c{i}", "official_macro_f1": 0.4 + i / 100} for i in range(4)],
            "samples": {"official_macro_f1": np.linspace(0.3, 0.5, 20000)},
        }
        right = {
            "by_cell": [{"cell_id": f"c{i}", "official_macro_f1": 0.39 + i / 100} for i in range(4)],
            "samples": {"official_macro_f1": np.linspace(0.29, 0.49, 20000)},
        }
        row = confirmation.contrast(left, right, "candidate", "reference", "official_macro_f1", primary=True)
        self.assertEqual(4, row["multiplicity_family_size"])
        self.assertAlmostEqual(0.01, row["delta"])
        self.assertEqual(4, row["wins"])
        self.assertIn("Bonferroni", row["inference"])


if __name__ == "__main__":
    unittest.main()
