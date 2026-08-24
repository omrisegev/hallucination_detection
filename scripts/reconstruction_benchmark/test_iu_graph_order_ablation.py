#!/usr/bin/env python3
"""Focused tests for the IU graph-order mechanism ablation."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from spectral_utils.reconstruction_benchmark.contracts import (
    CONTRACT_VERSION,
    PreparedCell,
)
from spectral_utils.reconstruction_benchmark.iu_graph_order_ablation import (
    arm_id,
    expected_arm_ids,
    fit_cell,
    validate_config,
)
from spectral_utils.specrage_views import FEATURE_TO_VIEW


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/reconstruction_benchmark_v1/iu_graph_order_ablation_v1.json"


def _cell(*, reverse_rows: bool = False) -> PreparedCell:
    rng = np.random.default_rng(4321)
    n = 72
    names = tuple(FEATURE_TO_VIEW)
    latent = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, len(names)))
    matrix = latent @ loadings + 0.35 * rng.normal(size=(n, len(names)))
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0)
    rows = np.asarray([f"row-{index:04d}" for index in range(n)])
    if reverse_rows:
        order = np.arange(n - 1, -1, -1)
        matrix = matrix[order]
        rows = rows[order]
    return PreparedCell(
        population_id="synthetic",
        cell_id="synthetic_cell",
        domain="QA",
        matrix=matrix,
        feature_names=names,
        row_ids=tuple(rows.tolist()),
        feature_contract=CONTRACT_VERSION,
        preprocessing_steps=(CONTRACT_VERSION,),
        preprocessed=True,
    )


class IUGraphOrderAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_config_and_roster_are_exact(self) -> None:
        lambdas = validate_config(self.config)
        arms = expected_arm_ids(lambdas)
        self.assertEqual(len(lambdas), 6)
        self.assertEqual(len(arms), 26)
        self.assertEqual(arms[:2], ("iu_pcr", "equal_family_mean"))
        self.assertIn(arm_id("residual_ridge_correction", 1.0), arms)

    def test_fit_is_finite_deterministic_and_target_free(self) -> None:
        first = fit_cell(_cell(), self.config)
        second = fit_cell(_cell(), self.config)
        self.assertEqual(tuple(first.scores), expected_arm_ids(validate_config(self.config)))
        self.assertEqual(first.diagnostics["runtime_labels_used"], False)
        for arm in first.scores:
            self.assertTrue(np.array_equal(first.scores[arm], second.scores[arm]), arm)
            self.assertTrue(np.isfinite(first.scores[arm]).all(), arm)
            self.assertAlmostEqual(float(np.std(first.scores[arm])), 1.0, places=10)

    def test_closed_form_never_increases_its_registered_objective(self) -> None:
        result = fit_cell(_cell(), self.config)
        for value in validate_config(self.config):
            diagnostics = result.diagnostics["arms"][
                arm_id("residual_ridge_correction", value)
            ]
            self.assertLessEqual(
                float(diagnostics["objective_after"]),
                float(diagnostics["objective_before"]) + 1e-10,
            )

    def test_row_permutation_is_equivariant(self) -> None:
        forward = fit_cell(_cell(), self.config)
        reverse = fit_cell(_cell(reverse_rows=True), self.config)
        reverse_index = {row: index for index, row in enumerate(reverse.row_ids)}
        order = np.asarray([reverse_index[row] for row in forward.row_ids], dtype=int)
        for arm in forward.scores:
            self.assertTrue(
                np.allclose(forward.scores[arm], reverse.scores[arm][order], atol=1e-9, rtol=1e-9),
                arm,
            )

    def test_invalid_or_label_claiming_config_is_rejected(self) -> None:
        bad = dict(self.config)
        bad["runtime_labels_used"] = True
        with self.assertRaisesRegex(ValueError, "targets"):
            validate_config(bad)
        bad = dict(self.config)
        bad["lambda_grid"] = [0.1, 0.1]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_config(bad)


if __name__ == "__main__":
    unittest.main()
