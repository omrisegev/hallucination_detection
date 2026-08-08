#!/usr/bin/env python3
"""Unit and protocol checks for repeated-measurement reliability fusion."""

import inspect
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.repeated_measurement_reliability_benchmark import fit_label_free
from spectral_utils.dufs_liu_feature_contract import dufs_liu_mixed_v2_matrix
from spectral_utils.repeated_measurement_reliability import (
    FixedMixedV2Transformer,
    bootstrap_trace_row,
    circular_moving_block_indices,
    covariance_components,
    generalized_reliability,
)


class RepeatedMeasurementReliabilityTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_length_preserving(self):
        first = circular_moving_block_indices(37, 8, np.random.default_rng(19))
        second = circular_moving_block_indices(37, 8, np.random.default_rng(19))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (37,))
        self.assertGreaterEqual(int(first.min()), 0)
        self.assertLess(int(first.max()), 37)

    def test_channels_share_exactly_the_same_bootstrap_indices(self):
        row = {
            "token_entropies": np.arange(20),
            "token_spilled_energies": np.arange(20) + 100,
            "token_logsumexp": np.arange(20) + 200,
            "top_k_logprobs": {
                "ids": np.column_stack([np.arange(20), np.arange(20) + 300]),
                "logprobs": np.column_stack([np.arange(20), np.arange(20) + 400]),
            },
        }
        indices = circular_moving_block_indices(20, 6, np.random.default_rng(5))
        boot = bootstrap_trace_row(row, indices)
        self.assertTrue(np.array_equal(boot["token_entropies"], indices))
        self.assertTrue(np.array_equal(boot["token_spilled_energies"] - 100, indices))
        self.assertTrue(np.array_equal(boot["token_logsumexp"] - 200, indices))
        self.assertTrue(np.array_equal(boot["top_k_logprobs"]["ids"][:, 0], indices))

    def test_fixed_transform_matches_mixed_v2_on_training_population(self):
        rng = np.random.default_rng(11)
        names = ("epr", "pe_mean", "stft_spectral_entropy", "logprob_margin")
        raw = rng.normal(size=(201, len(names)))
        fixed = FixedMixedV2Transformer.fit(raw, names)
        expected, expected_names, _ = dufs_liu_mixed_v2_matrix(raw, names)
        self.assertEqual(expected_names, names)
        self.assertLess(float(np.max(np.abs(fixed.training_output - expected))), 1e-10)

    def test_generalized_solver_prefers_high_signal_low_noise_direction(self):
        signal = np.diag([4.0, 1.0, 0.1])
        within = np.diag([0.2, 1.0, 2.0])
        result = generalized_reliability(signal, within)
        self.assertGreater(result["eigenvalues"][0], result["eigenvalues"][1])
        leading = int(np.argmax(np.abs(result["vectors"][:, 0])))
        self.assertEqual(leading, 0)

    def test_covariance_decomposition_recovers_nonzero_within_noise(self):
        rng = np.random.default_rng(7)
        original = rng.normal(size=(300, 4))
        replicates = original[:, None, :] + rng.normal(size=(300, 20, 4)) * 0.2
        total, within, signal = covariance_components(original, replicates)
        self.assertTrue(np.all(np.diag(within) > 0.02))
        self.assertTrue(np.all(np.diag(within) < 0.08))
        self.assertTrue(np.allclose(signal, total - within))

    def test_label_free_fit_function_has_no_outcome_field_reference(self):
        source = inspect.getsource(fit_label_free)
        self.assertNotIn("final_answer_correct", source)
        self.assertNotIn("roc_auc", source)


if __name__ == "__main__":
    unittest.main()
