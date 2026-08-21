#!/usr/bin/env python3
import unittest

import numpy as np

from scripts.direct_dufs_length_residualization_v1 import (
    apply_residualizer,
    decide,
    fit_residualizer,
    length_basis,
)


class LengthResidualizationTests(unittest.TestCase):
    def test_basis_is_finite_centered_and_bounded(self):
        basis = length_basis([0, 1, 2, 3, 10, 100, np.nan])
        self.assertTrue(np.isfinite(basis).all())
        np.testing.assert_allclose(basis.mean(axis=0), 0.0, atol=1e-12)
        self.assertLessEqual(np.max(np.abs(basis[:, 0])), 5.0)

    def test_training_only_coefficients_remove_shared_length_curve(self):
        lengths_a = np.arange(1, 101, dtype=float)
        lengths_b = np.arange(5, 205, 2, dtype=float)
        beta = np.asarray([0.6, -0.2, 0.08])
        train = []
        for cell, lengths in (("a", lengths_a), ("b", lengths_b)):
            basis = length_basis(lengths)
            matrix = np.column_stack([basis @ beta + 0.01 * np.sin(lengths), np.cos(lengths)])
            train.append({"cell": cell, "matrix": matrix, "names": ("signal", "control"), "length": lengths})
        coefficients, diagnostics = fit_residualizer(train, ("signal", "control"))
        np.testing.assert_allclose(coefficients["signal"], beta, atol=0.02)
        held_length = np.arange(2, 152, 3, dtype=float)
        held = np.column_stack([length_basis(held_length) @ beta, np.sin(held_length)])
        residual = apply_residualizer(held, ("signal", "control"), held_length, coefficients)
        self.assertEqual(residual.shape, held.shape)
        np.testing.assert_allclose(residual.mean(axis=0), 0.0, atol=1e-10)
        self.assertEqual(diagnostics["training_cell_count"], 2)

    def test_decision_requires_every_lane(self):
        summary = []
        for lane in ("global24", "processbench", "ragtruth"):
            summary.extend([
                {"lane": lane, "condition": "drop_length_refit_gates", "median_length_smoothness_effect": 0.5},
                {
                    "lane": lane,
                    "condition": "train_residualized_refit_gates",
                    "median_length_smoothness_effect": 0.2,
                    "fraction_target_smoother_than_length": 0.75,
                },
            ])
        decision, _ = decide(summary)
        self.assertEqual(decision, "RESIDUALIZATION_REVEALS_TARGET_SPECIFIC_GEOMETRY")
        summary[-1]["fraction_target_smoother_than_length"] = 0.0
        decision, _ = decide(summary)
        self.assertEqual(decision, "RESIDUALIZATION_REMOVES_LENGTH_BUT_NOT_TARGET_SPECIFIC")


if __name__ == "__main__":
    unittest.main()
