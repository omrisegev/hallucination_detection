#!/usr/bin/env python3
"""Focused mechanical tests for the reconstruction core-method layer.

These tests use synthetic, target-free matrices only.  They do not load the
24-cell bundle, labels, or evaluation code, and they do not run the expensive
DUFS/SpecRaGE/DEEM optimizers.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from spectral_utils.reconstruction_benchmark import (
    CONTRACT_VERSION,
    FitStatus,
    PRIMARY_METHOD_IDS,
    PRIMARY_METHOD_SPECS,
    PreparedCell,
    SCORE_SEMANTICS_CONVERSION,
    run_method,
)


SIX_FAMILY_NAMES = (
    "epr",
    "trace_length",
    "spectral_entropy",
    "epr_spilled",
    "epr_energy",
    "mean_top1_logprob",
)


def standardized_matrix(names=SIX_FAMILY_NAMES, *, n=96, seed=7):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    columns = []
    for index, _ in enumerate(names):
        loading = 0.45 + 0.08 * (index % 5)
        columns.append(loading * latent + rng.normal(scale=0.75, size=n))
    matrix = np.column_stack(columns)
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)


def prepared_cell(names=SIX_FAMILY_NAMES, *, n=96, seed=7):
    return PreparedCell(
        population_id="synthetic_population",
        cell_id="synthetic_cell",
        domain="synthetic_domain",
        matrix=standardized_matrix(names, n=n, seed=seed),
        feature_names=tuple(names),
        row_ids=tuple(f"row-{index:04d}" for index in range(n)),
    )


class PreparedCellContractTest(unittest.TestCase):
    def test_core_import_graph_excludes_label_selected_subset_module(self):
        probe = (
            "import sys; "
            "import spectral_utils.reconstruction_benchmark; "
            "assert 'spectral_utils.subset_sweep' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_primary_roster_has_exactly_thirteen_unique_arms(self):
        self.assertEqual(len(PRIMARY_METHOD_IDS), 13)
        self.assertEqual(len(set(PRIMARY_METHOD_IDS)), 13)
        self.assertEqual(tuple(PRIMARY_METHOD_SPECS), PRIMARY_METHOD_IDS)
        self.assertNotIn("upcr_estimated_sign", PRIMARY_METHOD_IDS)
        self.assertNotIn("residual_graph_deem", PRIMARY_METHOD_IDS)

    def test_specs_have_stable_versions_and_hashes(self):
        for method_id, spec in PRIMARY_METHOD_SPECS.items():
            self.assertEqual(spec.method_id, method_id)
            self.assertTrue(spec.method_version_id)
            self.assertEqual(len(spec.config_sha256), 64)
            self.assertEqual(spec.config["feature_contract"], CONTRACT_VERSION)
        self.assertEqual(
            PRIMARY_METHOD_SPECS["family_nrm_a"].development_status,
            "new_unrun_ablation",
        )
        self.assertEqual(
            PRIMARY_METHOD_SPECS["pgrd_a"].development_status,
            "new_unrun_ablation",
        )

    def test_prepared_matrix_is_copied_frozen_and_hashed(self):
        original = standardized_matrix()
        cell = PreparedCell(
            "population", "cell", "domain", original,
            SIX_FAMILY_NAMES,
            tuple(f"row-{index}" for index in range(len(original))),
        )
        original[0, 0] = 100.0
        self.assertNotEqual(cell.matrix[0, 0], 100.0)
        self.assertFalse(cell.matrix.flags.writeable)
        self.assertEqual(len(cell.matrix_sha256), 64)
        rebuilt = PreparedCell(
            "population", "cell", "domain", cell.matrix,
            SIX_FAMILY_NAMES,
            cell.row_ids,
            declared_matrix_sha256=cell.matrix_sha256,
        )
        self.assertEqual(rebuilt.matrix_sha256, cell.matrix_sha256)

    def test_runner_recomputes_hash_instead_of_trusting_declared_value(self):
        cell = prepared_cell()
        frozen_hash = cell.matrix_sha256
        cell.matrix.setflags(write=True)
        cell.matrix[0, 0] += 0.25
        result = run_method("equal_feature_mean", cell)
        self.assertEqual(result.status, FitStatus.INPUT_INVALID)
        self.assertNotEqual(result.prepared_matrix_sha256, frozen_hash)
        self.assertIn("changed after", result.diagnostics["error"])

    def test_contract_rejects_wrong_or_double_preprocessing(self):
        matrix = standardized_matrix()
        common = dict(
            population_id="population",
            cell_id="cell",
            domain="domain",
            matrix=matrix,
            feature_names=SIX_FAMILY_NAMES,
            row_ids=tuple(f"row-{index}" for index in range(len(matrix))),
        )
        with self.assertRaisesRegex(ValueError, "expected feature contract"):
            PreparedCell(**common, feature_contract="fixed-stable-v1")
        with self.assertRaisesRegex(ValueError, "exactly once"):
            PreparedCell(
                **common,
                preprocessing_steps=(CONTRACT_VERSION, CONTRACT_VERSION),
            )
        with self.assertRaisesRegex(ValueError, "already-prepared"):
            PreparedCell(**common, preprocessed=False)

    def test_contract_rejects_raw_unordered_duplicate_or_target_payloads(self):
        matrix = standardized_matrix()
        rows = tuple(f"row-{index}" for index in range(len(matrix)))
        with self.assertRaisesRegex(ValueError, "not centered"):
            PreparedCell(
                "population", "cell", "domain", matrix + 2.0,
                SIX_FAMILY_NAMES, rows,
            )
        with self.assertRaisesRegex(ValueError, "label-free 30-feature order"):
            PreparedCell(
                "population", "cell", "domain", matrix[:, ::-1],
                tuple(reversed(SIX_FAMILY_NAMES)), rows,
            )
        duplicate_rows = list(rows)
        duplicate_rows[-1] = duplicate_rows[0]
        with self.assertRaisesRegex(ValueError, "row_ids must be unique"):
            PreparedCell(
                "population", "cell", "domain", matrix,
                SIX_FAMILY_NAMES, tuple(duplicate_rows),
            )
        payload = {
            "population_id": "population",
            "cell_id": "cell",
            "domain": "domain",
            "matrix": matrix,
            "feature_names": SIX_FAMILY_NAMES,
            "row_ids": rows,
            "correctness": np.zeros(len(rows), dtype=int),
        }
        with self.assertRaisesRegex(ValueError, "forbidden/unknown"):
            PreparedCell.from_mapping(payload)


class CheapCoreMethodTest(unittest.TestCase):
    def test_means_use_confidence_matrix_and_return_risk(self):
        cell = prepared_cell()
        feature = run_method("equal_feature_mean", cell)
        family = run_method("equal_family_mean", cell)
        np.testing.assert_allclose(feature.score, -cell.matrix.mean(axis=1))
        # One selected feature belongs to each of the six present families here.
        np.testing.assert_allclose(family.score, -cell.matrix.mean(axis=1))
        for result in (feature, family):
            self.assertEqual(result.status, FitStatus.OK)
            self.assertEqual(result.score_semantics, "higher_is_incorrect")
            self.assertEqual(result.positive_class, "incorrect")
            self.assertEqual(
                dict(result.score_semantics_conversion),
                dict(SCORE_SEMANTICS_CONVERSION),
            )

    def test_existing_cpu_methods_and_new_within_cell_ablations_fit(self):
        cell = prepared_cell()
        methods = (
            "continuous_lsml",
            "upcr",
            "iu_pcr",
            "su_pcr",
            "family_nrm_a",
            "pgrd_a",
        )
        for method_id in methods:
            with self.subTest(method_id=method_id):
                result = run_method(method_id, cell)
                self.assertIn(result.status, (FitStatus.OK, FitStatus.OK_FALLBACK))
                self.assertEqual(result.score.shape, (len(cell.row_ids),))
                self.assertTrue(np.isfinite(result.score).all())
                self.assertEqual(result.prepared_matrix_sha256, cell.matrix_sha256)
        for method_id in ("family_nrm_a", "pgrd_a"):
            result = run_method(method_id, cell)
            self.assertEqual(result.diagnostics["regime"], "A_within_cell_fully_unsupervised")
            self.assertEqual(result.diagnostics["donor_cells_used"], 0)
            self.assertFalse(result.diagnostics["targets_used"])

    def test_family_methods_have_explicit_low_family_fallback(self):
        # epr is entropy_level; the other two columns are entropy_dynamics.
        names = ("epr", "spectral_entropy", "low_band_power")
        cell = prepared_cell(names, seed=19)
        for method_id in ("family_nrm_a", "pgrd_a"):
            result = run_method(method_id, cell)
            self.assertEqual(result.status, FitStatus.OK_FALLBACK)
            self.assertIn("fewer than three", result.fallback_reason)
            self.assertTrue(np.isfinite(result.score).all())
            self.assertIn("baseline_standardized", result.artifacts)
            self.assertEqual(result.selected_features, cell.feature_names)

    def test_parameter_free_dufs_fallback_is_not_silent(self):
        cell = prepared_cell()
        gates = np.full(len(cell.feature_names), -1.0)
        with patch(
            "spectral_utils.selectors.a2_groupfs.dufs_pf_gates",
            return_value=gates,
        ):
            result = run_method("dufs_pf_lsml", cell)
        self.assertEqual(result.status, FitStatus.OK_FALLBACK)
        self.assertIn("fewer than three", result.fallback_reason)
        self.assertEqual(result.selected_features, cell.feature_names)

    def test_stability_selected_dufs_uses_a2_dufs_arm(self):
        cell = prepared_cell()
        fake = {
            "cols": np.asarray([0, 2, 4]),
            "fallback": False,
            "diag": {"lambda_dufs": 0.4, "stability": 0.8},
        }
        with patch(
            "spectral_utils.selectors.a2_groupfs.dufs_stability_selection",
            return_value=fake,
        ):
            result = run_method("dufs_stability_lsml", cell)
        self.assertEqual(result.status, FitStatus.OK)
        self.assertEqual(
            result.selected_features,
            tuple(cell.feature_names[index] for index in (0, 2, 4)),
        )
        self.assertEqual(result.diagnostics["selector"]["variant"], "a2.dufs")


class ExpensiveAdapterGateTest(unittest.TestCase):
    def test_dufs_liu_uses_frozen_gate_and_graph_configuration(self):
        cell = prepared_cell(n=48)
        seen = {}

        def fake_gates(F, *, seeds, epochs):
            seen["seeds"] = seeds
            seen["epochs"] = epochs
            p = F.shape[0]
            probabilities = np.full((len(seeds), p), 0.75)
            return np.ones(p), {
                "per_seed_probabilities": probabilities,
                "raw_probabilities": probabilities.mean(axis=0),
            }

        with patch(
            "spectral_utils.laplacian_upcr.dufs_soft_gates",
            side_effect=fake_gates,
        ):
            result = run_method("dufs_liu", cell)
        self.assertEqual(result.status, FitStatus.OK)
        self.assertEqual(seen, {"seeds": (11, 23, 37), "epochs": 80})
        self.assertEqual(result.artifacts["graph"].shape, (48, 48))
        self.assertEqual(
            result.artifacts["gate_probabilities_per_seed"].shape,
            (3, len(cell.feature_names)),
        )

    def test_ca_atomic_wiring_keeps_provenance_prior_explicit(self):
        cell = prepared_cell(n=48)
        base = np.ones((48, 48), dtype=float) - np.eye(48)
        graph = csr_matrix(base)
        seen = {}

        def fake_fit(views, *, config, seeds, view_prior, tie_keys):
            seen["seeds"] = tuple(seeds)
            seen["lambda_config"] = config.n_neighbors
            seen["prior"] = dict(view_prior)
            seen["tie_keys"] = np.asarray(tie_keys)
            view_names = tuple(views)
            alpha = np.full((48, len(view_names)), 1.0 / len(view_names))
            seed_results = tuple(
                SimpleNamespace(seed=seed, graph=graph, alpha=alpha)
                for seed in seeds
            )
            return SimpleNamespace(
                graph=graph,
                embedding_graph=graph,
                base_graphs=tuple(graph for _ in view_names),
                alpha=alpha,
                view_names=view_names,
                view_prior=np.asarray([view_prior[name] for name in view_names]),
                seed_results=seed_results,
                diagnostics={},
            )

        with patch(
            "spectral_utils.specrage_laplacian.fit_specrage_graph",
            side_effect=fake_fit,
        ):
            result = run_method("ca_specrage_atomic", cell)
        self.assertEqual(result.status, FitStatus.OK)
        self.assertEqual(seen["seeds"], (11, 23))
        self.assertEqual(seen["lambda_config"], 15)
        self.assertAlmostEqual(sum(seen["prior"].values()), 1.0)
        self.assertEqual(len(np.unique(seen["tie_keys"])), len(cell.row_ids))
        self.assertFalse(result.diagnostics["historical_loco_micro_prior_reused"])
        self.assertIsInstance(result.artifacts["base_graphs"], dict)
        self.assertEqual(result.artifacts["alpha_per_seed"].shape[0], 2)

    def test_deem_receives_only_common_whole_matrix_risk_view_and_all_five_seeds(self):
        cell = prepared_cell(n=48)
        seen = []

        def fake_fit(X_risk, feature_names, *, seed, config):
            seen.append((seed, np.asarray(X_risk).copy(), tuple(feature_names)))
            score = np.linspace(0.1, 0.9, len(X_risk)) + 0.001 * seed
            return SimpleNamespace(
                seed=seed,
                score=score,
                health={
                    "healthy": True,
                    "finite": True,
                    "posterior_sd": float(np.std(score)),
                    "contribution_reconstruction_max_abs": 0.0,
                    "epochs_completed": 100,
                    "runtime_seconds": 10.0 + seed,
                },
            )

        with patch(
            "spectral_utils.residual_graph_deem.fit_continuous_deem",
            side_effect=fake_fit,
        ):
            result = run_method("deem_b3", cell)
        self.assertEqual(result.status, FitStatus.OK)
        self.assertEqual([row[0] for row in seen], [0, 1, 2, 3, 4])
        for _, matrix, names in seen:
            np.testing.assert_array_equal(matrix, -cell.matrix)
            self.assertEqual(names, cell.feature_names)
        expected = np.linspace(0.1, 0.9, len(cell.row_ids)) + 0.002
        np.testing.assert_allclose(result.score, expected)
        self.assertIn("whole-matrix risk view", result.diagnostics["solver_coordinate_semantics"])
        self.assertTrue(all(
            "runtime_seconds" not in row
            for row in result.diagnostics["seed_health"]
        ))

    def test_deem_rejects_one_unhealthy_seed_without_survivor_average(self):
        cell = prepared_cell(n=48)

        def fake_fit(X_risk, feature_names, *, seed, config):
            score = np.linspace(0.1, 0.9, len(X_risk))
            return SimpleNamespace(
                seed=seed,
                score=score,
                health={
                    "healthy": seed != 3,
                    "finite": True,
                    "posterior_sd": float(np.std(score)),
                    "contribution_reconstruction_max_abs": 0.0,
                    "epochs_completed": 100,
                },
            )

        with patch(
            "spectral_utils.residual_graph_deem.fit_continuous_deem",
            side_effect=fake_fit,
        ):
            result = run_method("deem_b3", cell)
        self.assertEqual(result.status, FitStatus.FIT_FAILED)
        self.assertIsNone(result.score)
        self.assertIn("all-seed health gate", result.diagnostics["error"])

    def test_sparse_graph_artifacts_serialize_without_opening_targets(self):
        from spectral_utils.reconstruction_benchmark.serialization import write_score_result

        cell = prepared_cell(n=48)
        result = run_method("pgrd_a", cell)
        self.assertEqual(result.status, FitStatus.OK)
        with tempfile.TemporaryDirectory() as temporary:
            record = write_score_result(result, cell.row_ids, temporary)
            self.assertEqual(record["score_n"], len(cell.row_ids))
            self.assertIsNotNone(record["score_sha256"])
            self.assertIsNotNone(record["artifacts_sha256"])


if __name__ == "__main__":
    unittest.main()
