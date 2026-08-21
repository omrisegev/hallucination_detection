#!/usr/bin/env python3
"""Focused adversarial tests for supervised conditional manifold discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.supervised_conditional_manifold_discovery_v1 import (  # noqa: E402
    _save_null_checkpoint,
    candidate_definitions,
    exact_eligible,
    summarize_outer_families,
)
from spectral_utils.supervised_manifold_discovery import (  # noqa: E402
    conditional_residual_smoothness,
    cross_fitted_length_residual,
    deterministic_subsample,
    feature_relevance,
    fit_metric_ensemble,
    graph_is_healthy,
    median_pairwise_cosine,
    median_pairwise_jaccard,
    metric_matrix,
    select_label_free_graph,
    support_indices,
    target_blind_tie_keys,
)
from scripts.validate_supervised_conditional_manifold_v1 import load_manifest_cells  # noqa: E402


def synthetic_cells(seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    cells = []
    for family_index, family in enumerate(("a", "b", "c", "d")):
        for replicate in range(2):
            n = 180
            length = rng.integers(5, 80, size=n).astype(float)
            target = rng.integers(0, 2, size=n)
            signal = (2 * target - 1) + rng.normal(scale=.35, size=n)
            matrix = np.column_stack((
                signal,
                np.log1p(length) + rng.normal(scale=.1, size=n),
                rng.normal(size=n),
                rng.normal(size=n),
                rng.normal(size=n),
                rng.normal(size=n),
            ))
            matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
            cells.append({
                "cell": f"{family}_{replicate}",
                "family": family,
                "X": matrix,
                "y": target,
                "length": length,
            })
    return cells


class ResidualAndWeightTests(unittest.TestCase):
    def test_cross_fitted_residual_reduces_length_only_target_relation(self):
        rng = np.random.default_rng(2)
        length = rng.integers(1, 150, size=600).astype(float)
        target = (length > np.median(length)).astype(int)
        residual = cross_fitted_length_residual(target, length, seed=7)
        raw = abs(np.corrcoef(target, np.log1p(length))[0, 1])
        held = abs(np.corrcoef(residual, np.log1p(length))[0, 1])
        self.assertLess(held, raw * .35)

    def test_supervised_metric_recovers_planted_feature(self):
        cells = synthetic_cells()
        mean, members = fit_metric_ensemble(cells)
        self.assertEqual(int(np.argmax(mean)), 0)
        self.assertGreater(mean[0], 2 * np.median(mean[1:]))
        self.assertEqual(set(members), {17, 29, 43})

    def test_held_labels_cannot_change_donor_weights(self):
        cells = synthetic_cells()
        donors = [cell for cell in cells if cell["family"] != "d"]
        original, _ = fit_metric_ensemble(donors)
        for cell in cells:
            if cell["family"] == "d":
                cell["y"] = 1 - cell["y"]
        repeated, _ = fit_metric_ensemble(donors)
        np.testing.assert_array_equal(original, repeated)

    def test_feature_relevance_rejects_pure_length_coordinate(self):
        rng = np.random.default_rng(8)
        n = 1000
        length = rng.integers(1, 200, size=n).astype(float)
        target = (rng.random(n) < 1 / (1 + np.exp(-(np.log1p(length) - 3.5)))).astype(int)
        independent = rng.normal(size=n)
        matrix = np.column_stack((np.log1p(length), independent))
        relevance = feature_relevance(matrix, target, length, seed=91)
        self.assertLess(relevance[0], .12)


class GraphAndSupportTests(unittest.TestCase):
    def test_graph_selection_is_label_free_and_healthy(self):
        rng = np.random.default_rng(4)
        matrix = rng.normal(size=(160, 4))
        keys = target_blind_tie_keys(len(matrix), namespace="unit", seed=101)
        graph, diagnostics = select_label_free_graph(matrix, tie_keys=keys)
        self.assertIsNotNone(graph)
        self.assertTrue(diagnostics["eligible"])
        self.assertTrue(graph_is_healthy(diagnostics))

    def test_support_ties_break_by_feature_name(self):
        names = ("z", "a", "m", "b")
        weights = np.ones(4)
        indexes = support_indices(weights, names, 2)
        self.assertEqual({names[index] for index in indexes}, {"a", "b"})

    def test_metric_scaling_preserves_equal_weight_geometry(self):
        rng = np.random.default_rng(1)
        matrix = rng.normal(size=(30, 5))
        weights = np.ones(5) / 5
        scaled = metric_matrix(matrix, weights, np.arange(5))
        np.testing.assert_allclose(scaled, matrix)

    def test_target_blind_subsample_does_not_accept_target(self):
        left = deterministic_subsample(1000, namespace="cell", max_rows=100)
        right = deterministic_subsample(1000, namespace="cell", max_rows=100)
        np.testing.assert_array_equal(left, right)
        self.assertEqual(len(left), 100)

    def test_duplicate_rows_do_not_break_metric_graph(self):
        rng = np.random.default_rng(3)
        matrix = np.vstack((np.zeros((20, 3)), rng.normal(size=(60, 3))))
        keys = target_blind_tie_keys(len(matrix), namespace="dupes", seed=307)
        graph, diagnostics = select_label_free_graph(matrix, tie_keys=keys)
        self.assertIsNotNone(graph)
        self.assertTrue(diagnostics["eligible"])
        self.assertGreater(np.min(graph.data), 0)


class StabilityAndCheckpointTests(unittest.TestCase):
    def test_stability_metrics_have_expected_extremes(self):
        self.assertAlmostEqual(median_pairwise_cosine([np.ones(3), np.ones(3)]), 1.0)
        self.assertAlmostEqual(
            median_pairwise_jaccard([np.array([0, 1]), np.array([0, 1])]), 1.0
        )
        self.assertAlmostEqual(
            median_pairwise_jaccard([np.array([0]), np.array([1])]), 0.0
        )

    def test_null_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "null.npz"
            values = {
                "exact_metric": np.arange(4, dtype=float),
                "exact_advantage": np.arange(4, dtype=float) + 1,
                "crt_metric": np.arange(4, dtype=float) + 2,
                "crt_advantage": np.arange(4, dtype=float) + 3,
            }
            _save_null_checkpoint(path, values, 4)
            with np.load(path, allow_pickle=False) as loaded:
                self.assertEqual(int(loaded["reruns"][0]), 4)
                np.testing.assert_array_equal(loaded["crt_advantage"], values["crt_advantage"])

    def test_exact_control_eligibility_is_fail_closed(self):
        self.assertFalse(exact_eligible({
            "movable_fraction": .19,
            "movable_rows": 100,
            "mixed_strata": 10,
        }))

    def test_candidate_family_is_fixed(self):
        names = tuple(f"f{i}" for i in range(16))
        definitions = candidate_definitions(np.arange(1, 17) / 136, names)
        self.assertEqual(
            [value["candidate"] for value in definitions],
            ["equal_all", "supervised_s5", "supervised_s10", "supervised_s15", "supervised_sall"],
        )

    def test_family_summary_does_not_count_tie_seeds_as_cells(self):
        rows = []
        for seed, effect in ((101, .10), (211, .20), (307, .30)):
            rows.append({
                "held_family": "a",
                "cell": "a_0",
                "candidate": "supervised_s5",
                "candidate_role": "supervised_metric",
                "graph_role": "metric_graph",
                "tie_seed": seed,
                "graph_eligible": True,
                "exact_eligible": True,
                "crt_eligible": True,
                "exact_effect": effect,
                "crt_effect": effect + .01,
                "min_conditional_effect": effect,
                "liu_delta_auroc": .001,
            })
        summary = summarize_outer_families(rows)[0]
        self.assertEqual(summary["cell_count"], 1)
        self.assertEqual(summary["tie_seed_count"], 3)
        self.assertAlmostEqual(summary["mean_exact_effect"], .20)


class ConditionalStatisticTests(unittest.TestCase):
    def test_planted_local_target_is_smoother_than_random(self):
        rng = np.random.default_rng(14)
        target = np.repeat((0, 1), 100)
        matrix = np.column_stack((target * .9 + rng.normal(scale=.55, size=200), rng.normal(size=200)))
        length = rng.integers(5, 50, size=200)
        keys = target_blind_tie_keys(200, namespace="smooth", seed=101)
        graph, _ = select_label_free_graph(matrix, tie_keys=keys)
        self.assertIsNotNone(graph)
        observed = conditional_residual_smoothness(graph, target, length, seed=10)
        permuted = conditional_residual_smoothness(
            graph, rng.permutation(target), length, seed=10
        )
        self.assertGreater(observed, permuted)


class ExternalValidationContractTests(unittest.TestCase):
    def _candidate(self):
        return {
            "feature_names": ["a", "b"],
            "weights": [.6, .4],
            "support_indices": [0, 1],
        }

    def _write_fixture(self, directory: Path, *, independent=True, dataset_new=True):
        rng = np.random.default_rng(55)
        np.savez_compressed(
            directory / "cell.npz",
            X=rng.normal(size=(40, 2)),
            feature_names=np.asarray(["a", "b"]),
            trace_length=rng.integers(3, 20, size=40),
            hallucination_target=np.tile((0, 1), 20),
        )
        manifest = {
            "version": "supervised-conditional-manifold-validation-manifest-v1",
            "validation_name": "unit",
            "standardization_contract": "registered_per_cell_unlabeled",
            "discovery_dataset_families": ["old"],
            "cells": [{
                "cell": "held",
                "lane": "global_final_answer_error",
                "dataset_family": "new" if dataset_new else "old",
                "model_family": "new_model",
                "dataset_new": dataset_new,
                "model_new": True,
                "npz": "cell.npz",
                "matrix_key": "X",
                "feature_names_key": "feature_names",
                "length_key": "trace_length",
                "target_key": "hallucination_target",
                "independent_rows": independent,
            }],
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_valid_external_manifest_loads_frozen_feature_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_fixture(Path(directory))
            _, cells = load_manifest_cells(path, self._candidate())
            self.assertEqual(cells[0]["X"].shape, (40, 2))

    def test_external_manifest_rejects_unhandled_dependent_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_fixture(Path(directory), independent=False)
            with self.assertRaisesRegex(RuntimeError, "group-aware null"):
                load_manifest_cells(path, self._candidate())

    def test_external_manifest_rejects_false_new_dataset_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_fixture(Path(directory), dataset_new=False)
            manifest = json.loads(path.read_text())
            manifest["cells"][0]["dataset_new"] = True
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "dataset_new conflicts"):
                load_manifest_cells(path, self._candidate())


if __name__ == "__main__":
    unittest.main()
