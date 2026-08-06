#!/usr/bin/env python3
"""Known-answer tests for the U2-prior reconciliation helpers."""

import os
import sys
import types
import unittest

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.feature_contract import consensus_anchor                 # noqa: E402
from spectral_utils.semi_supervised_fusion import (                         # noqa: E402
    orient_weight,
    spectral_score_basis,
)
from spectral_utils.u2_prior_reconciliation import (                        # noqa: E402
    CURRENT_UPCR_KWARGS,
    basis_alignment,
    covariance_normalized_u2_basis,
    fit_equivalence_diagnostics,
    fit_prior_head,
    optimistic_endpoint_controls,
)
from spectral_utils.upcr import upcr_fit                                     # noqa: E402
from scripts.u2_prior_reconciliation import (                                # noqa: E402
    CORE_SYNTHETIC_METHODS,
    literal_cell_stop_rule,
    pack_and_validate_row_scores,
    real_cell_method_means,
    safe_metrics,
    validate_cartesian_rows,
)


def zscore_rows(F):
    F = np.asarray(F, dtype=float)
    return (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)


class U2PriorReconciliationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(731)
        n = 320
        g = rng.normal(size=n)
        u = rng.normal(size=n)
        rows = [g + 0.5 * rng.normal(size=n) for _ in range(5)]
        rows += [u + 0.25 * rng.normal(size=n) for _ in range(5)]
        self.F = zscore_rows(np.vstack(rows))
        self.matrix = self.F.T
        self.labels = (g + 0.25 * rng.normal(size=n) > 0).astype(int)

    def test_current_iu_anchored_basis_is_u2_reparameterization(self):
        reference, _, _ = covariance_normalized_u2_basis(self.F)
        fit = upcr_fit(self.F, **CURRENT_UPCR_KWARGS)
        weight = orient_weight(
            fit.w,
            self.matrix,
            consensus_anchor(self.matrix),
        )
        source = spectral_score_basis(self.matrix, weight, rank=2)
        alignment = basis_alignment(self.F, reference, source)
        self.assertTrue(alignment.geometrically_equivalent)
        self.assertLess(alignment.max_principal_angle_rad, 1e-10)

    def test_orthogonal_basis_change_preserves_fitted_head(self):
        reference, _, _ = covariance_normalized_u2_basis(self.F)
        theta = 0.61
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        source = reference @ rotation
        alignment = basis_alignment(self.F, reference, source)
        self.assertTrue(alignment.geometrically_equivalent)
        calibration = np.arange(48)
        source_head, source_scores = fit_prior_head(
            self.matrix, self.labels, calibration, source, np.array([1.0, 0.0])
        )
        reference_head, reference_scores = fit_prior_head(
            self.matrix,
            self.labels,
            calibration,
            reference,
            alignment.source_prior_in_reference,
        )
        diagnostics = fit_equivalence_diagnostics(
            source_head, source_scores, reference_head, reference_scores
        )
        self.assertTrue(diagnostics["fit_equivalent"])

    def test_nonmatching_subspace_is_rejected(self):
        reference, _, _ = covariance_normalized_u2_basis(self.F)
        covariance = self.F @ self.F.T / self.F.shape[1]
        values, vectors = np.linalg.eigh(covariance)
        third = vectors[:, np.argsort(values)[::-1][2]]
        third /= np.std(self.matrix @ third)
        source = np.column_stack([reference[:, 0], third])
        alignment = basis_alignment(self.F, reference, source)
        self.assertFalse(alignment.geometrically_equivalent)
        self.assertGreater(alignment.max_principal_angle_rad, 0.1)

    def test_endpoint_switch_is_not_called_an_interpolation_ceiling(self):
        rng = np.random.default_rng(19)
        score_a = rng.normal(size=800)
        score_b = rng.normal(size=800)
        labels = (score_a + score_b > 0).astype(int)
        controls = optimistic_endpoint_controls(
            labels, score_a, score_b, np.arange(len(labels))
        )
        endpoint = controls["metrics"]["optimistic_endpoint_switch"]["auroc"]
        interpolation = controls["metrics"]["optimistic_interpolation"]["auroc"]
        self.assertGreater(interpolation, endpoint + 0.1)
        self.assertGreater(
            controls["metrics"]["optimistic_interpolation"]["alpha_u2"], 0.0
        )
        self.assertLess(
            controls["metrics"]["optimistic_interpolation"]["alpha_u2"], 1.0
        )

    def test_real_cartesian_validator_rejects_compensating_duplicate(self):
        rows = []
        for cell in ("cell_a", "cell_b"):
            for repetition in range(2):
                for method in ("upcr", "candidate"):
                    rows.append({
                        "unit": cell,
                        "repetition": str(repetition),
                        "method": method,
                        "group": "QA" if cell == "cell_a" else "math",
                        "domain": "QA" if cell == "cell_a" else "math",
                        "auc": "0.5",
                    })
        validate_cartesian_rows(
            rows, ("cell_a", "cell_b"), range(2), ("upcr", "candidate")
        )
        malformed = list(rows)
        malformed[-1] = dict(malformed[-2])
        with self.assertRaisesRegex(RuntimeError, "Cartesian product"):
            validate_cartesian_rows(
                malformed,
                ("cell_a", "cell_b"),
                range(2),
                ("upcr", "candidate"),
            )

    def test_real_aggregation_averages_repetitions_inside_cells(self):
        rows = [
            {"unit": "a", "method": "upcr", "auc": "0.4"},
            {"unit": "a", "method": "upcr", "auc": "0.6"},
            {"unit": "a", "method": "candidate", "auc": "0.7"},
            {"unit": "a", "method": "candidate", "auc": "0.9"},
            {"unit": "b", "method": "upcr", "auc": "0.8"},
            {"unit": "b", "method": "upcr", "auc": "1.0"},
            {"unit": "b", "method": "candidate", "auc": "0.3"},
            {"unit": "b", "method": "candidate", "auc": "0.5"},
        ]
        means = real_cell_method_means(rows, repetitions_per_cell=2)
        self.assertAlmostEqual(means[("a", "upcr")], 0.5)
        self.assertAlmostEqual(means[("a", "candidate")], 0.8)
        self.assertAlmostEqual(means[("b", "upcr")], 0.9)
        self.assertAlmostEqual(means[("b", "candidate")], 0.4)
        cell_macro_delta = np.mean([
            means[(cell, "candidate")] - means[(cell, "upcr")]
            for cell in ("a", "b")
        ])
        self.assertAlmostEqual(cell_macro_delta, -0.1)

    def test_packed_scores_align_and_reconstruct_calibration_and_metrics(self):
        row_arrays = {key: [] for key in (
            "replicate", "task", "draw", "budget", "sample_index",
            "is_calibration", "label",
        )}
        for method in CORE_SYNTHETIC_METHODS:
            row_arrays[f"score__{method}"] = []
        labels = np.array([0, 1, 0, 1], dtype=np.int8)
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        for draw, calibration in enumerate((np.array([0]), np.array([1]))):
            row_arrays["replicate"].append(np.zeros(4, dtype=np.int16))
            row_arrays["task"].append(np.zeros(4, dtype=np.int8))
            row_arrays["draw"].append(np.full(4, draw, dtype=np.int8))
            row_arrays["budget"].append(np.ones(4, dtype=np.int16))
            row_arrays["sample_index"].append(np.arange(4, dtype=np.int16))
            row_arrays["is_calibration"].append(np.isin(np.arange(4), calibration))
            row_arrays["label"].append(labels)
            for method in CORE_SYNTHETIC_METHODS:
                row_arrays[f"score__{method}"].append(scores)
        packed = pack_and_validate_row_scores(
            row_arrays, expected_rows=8, expected_groups=2, samples_per_group=4
        )
        group = (
            (packed["replicate"] == 0)
            & (packed["task"] == 0)
            & (packed["draw"] == 0)
            & (packed["budget"] == 1)
        )
        evaluation = group & ~packed["is_calibration"]
        auc, ap = safe_metrics(
            packed["label"][evaluation],
            packed["score__iu"][evaluation],
        )
        expected_auc, expected_ap = safe_metrics(labels[1:], scores[1:])
        self.assertAlmostEqual(auc, expected_auc)
        self.assertAlmostEqual(ap, expected_ap)
        self.assertEqual(int(np.sum(group & packed["is_calibration"])), 1)

    def test_stop_rule_is_literal_at_cell_level(self):
        summaries = [
            {"strictly_improves_cell_macro": False},
            {"strictly_improves_cell_macro": False},
        ]
        replay = [
            {"optimistic_endpoint_switch_gain_pp": 0.1},
            {"optimistic_endpoint_switch_gain_pp": 1.01},
        ]
        no_improvement, all_cells_below, stop = literal_cell_stop_rule(
            summaries, replay
        )
        self.assertTrue(no_improvement)
        self.assertFalse(all_cells_below)
        self.assertFalse(stop)


if __name__ == "__main__":
    unittest.main(verbosity=2)
