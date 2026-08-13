#!/usr/bin/env python3
"""Tests for the frozen automatic group-free IU Phase A4 utilities."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_utils.paired_repeatability import (
    FEATURE_ROSTER,
    algebraic_sign,
    conditional_derangement,
    fit_cca_common,
    fit_corrca,
    fisher_macro,
    sanitize_processbench_row,
    scalar_confound_metrics,
    select_nested,
    validate_sanitized_row,
)
from scripts.automatic_group_free_phase_a4 import _outer_null_pairing, _training_nulls


def telemetry_row(**extra):
    row = {
        "id": "item-1",
        "problem": "What is two plus two?",
        "steps": ["Two plus two is four."],
        "align_diag": {"problems": [], "n_tokens": 17, "secret_nested": "ignored"},
        "token_entropies": np.linspace(0.1, 0.9, 12),
        "token_spilled_energies": np.linspace(0.3, 1.1, 12),
        "token_logsumexp": np.linspace(3.0, 4.0, 12),
        "top_k_logprobs": {
            "ids": np.tile(np.arange(3), (12, 1)),
            "logprobs": np.tile(np.asarray([-0.1, -1.0, -2.0]), (12, 1)),
        },
    }
    row.update(extra)
    return row


class FirewallTests(unittest.TestCase):
    def test_target_keys_do_not_cross_sanitizer(self):
        clean = sanitize_processbench_row("1", telemetry_row())
        poisoned = sanitize_processbench_row(
            "1", telemetry_row(label=3, final_answer_correct=False, step_token_spans=[(0, 2)]),
        )
        self.assertEqual(set(clean), set(poisoned))
        self.assertNotIn("label", poisoned)
        self.assertNotIn("align_diag", poisoned)
        np.testing.assert_array_equal(clean["token_entropies"], poisoned["token_entropies"])

    def test_fit_boundary_rejects_raw_or_target_keys(self):
        with self.assertRaises(ValueError):
            validate_sanitized_row({"label": 1})
        with self.assertRaises(ValueError):
            validate_sanitized_row({"align_diag": {}})

    def test_alignment_failure_is_dropped_before_sanitization(self):
        self.assertIsNone(sanitize_processbench_row(
            "1", telemetry_row(align_diag={"problems": ["bad"]}),
        ))


class EstimatorTests(unittest.TestCase):
    def test_roster_is_exact_29_without_min_spilled(self):
        self.assertEqual(len(FEATURE_ROSTER), 29)
        self.assertEqual(len(set(FEATURE_ROSTER)), 29)
        self.assertNotIn("min_spilled", FEATURE_ROSTER)

    def test_corrca_recovers_repeatable_loading(self):
        rng = np.random.default_rng(17)
        n, p = 900, 8
        loading = rng.normal(size=p)
        loading /= np.linalg.norm(loading)
        latent = rng.normal(size=n)
        x4 = latent[:, None] * loading + rng.normal(0.0, 0.6, (n, p))
        x8 = latent[:, None] * loading + rng.normal(0.0, 0.6, (n, p))
        fitted, value = fit_corrca(x4, x8, 0.01)
        self.assertGreater(abs(float(fitted @ loading)), 0.9)
        self.assertGreater(value, 0.0)

    def test_cca_common_is_normalized_and_repeatable(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(500, 6))
        x8 = x + rng.normal(0.0, 0.2, x.shape)
        loading = fit_cca_common(x, x8, 0.1)
        self.assertAlmostEqual(float(np.linalg.norm(loading)), 1.0)
        left, right = x @ loading, x8 @ loading
        self.assertGreater(float(np.corrcoef(left, right)[0, 1]), 0.9)

    def test_algebraic_sign_is_deterministic(self):
        left = algebraic_sign(np.asarray([-2.0, 2.0, 0.5]))
        np.testing.assert_array_equal(left, np.asarray([2.0, -2.0, -0.5]))


class NullAndAggregationTests(unittest.TestCase):
    def test_conditional_derangement_preserves_strata_without_fixed_points(self):
        subsets = np.repeat(np.arange(4), 40)
        lengths = np.tile(np.repeat(np.arange(10), 4), 4).astype(float)
        permutation, rows = conditional_derangement(
            subsets, lengths, np.random.default_rng(29),
        )
        self.assertTrue(np.all(permutation != np.arange(len(permutation))))
        np.testing.assert_array_equal(subsets, subsets[permutation])
        np.testing.assert_array_equal(lengths, lengths[permutation])
        self.assertEqual(sum(row["n"] for row in rows), len(subsets))

    def test_fisher_macro_does_not_equal_raw_concatenation_shortcut(self):
        self.assertAlmostEqual(fisher_macro([0.2, 0.6]), np.tanh(
            0.5 * (np.arctanh(0.2) + np.arctanh(0.6))
        ))

    def test_nested_pair_null_breaks_full_validation_pairing(self):
        rng = np.random.default_rng(31)
        n, p = 160, len(FEATURE_ROSTER)
        subset = np.repeat(np.arange(4), n // 4)
        latent = rng.normal(size=(n, p))
        raw = np.stack([
            latent + rng.normal(0.0, 0.2, (n, p)),
            latent + rng.normal(0.0, 0.2, (n, p)),
            latent + rng.normal(0.0, 0.2, (n, p)),
        ], axis=1)
        covariates = rng.normal(size=(n, 3, 6))
        # One nontrivial conditional stratum per subset in the small fixture.
        covariates[:, :, 1] = 3.0
        groups = np.asarray([f"group-{index}" for index in range(n)])
        result = select_nested(
            raw, covariates, subset, groups, FEATURE_ROSTER,
            transform_kind="raw_z", seed=37, pair_null_seed=41, baselines=False,
        )
        permutations = result["pair_null_permutations"]
        self.assertEqual(len(permutations), 4)
        for row in permutations:
            self.assertTrue(np.all(row["train"] != np.arange(len(row["train"]))))
            self.assertTrue(np.all(row["valid"] != np.arange(len(row["valid"]))))

    def test_scalar_confound_uses_full_fit_residual_coordinates(self):
        rng = np.random.default_rng(43)
        n_train, n_held, p, d = 120, 48, len(FEATURE_ROSTER), 17
        train_design = rng.normal(size=(2 * n_train, d))
        held_design = rng.normal(size=(n_held, 3, d))
        coefficient = rng.normal(size=d)
        full_train = rng.normal(size=(n_train, 2, p))
        held = rng.normal(size=(n_held, 3, p))
        # Feature zero is deliberately fully text-predictable in the same
        # full-fit residual coordinate system.
        full_train[:, 0, 0] = train_design[:n_train] @ coefficient
        full_train[:, 1, 0] = train_design[n_train:] @ coefficient
        for view in range(3):
            held[:, view, 0] = held_design[:, view] @ coefficient
        prepared = type("Fixture", (), {
            "train_residuals": rng.normal(size=(n_train, 2, p)),
            "train_full_residuals": full_train,
            "held_residuals": held,
            "train_design": train_design,
            "held_design": held_design,
        })()
        loading = np.zeros(p)
        loading[0] = 1.0
        subsets = np.tile(np.arange(4), n_held // 4)
        rows = scalar_confound_metrics(prepared, loading, subsets)
        self.assertEqual(len(rows), 12)
        self.assertGreater(min(row["r2"] for row in rows), 0.99)
        self.assertGreater(min(row["abs_correlation"] for row in rows), 0.99)

    def test_outer_null_breaks_held_pairs_with_training_edges(self):
        rng = np.random.default_rng(47)
        p = len(FEATURE_ROSTER)
        train_subset = np.repeat(np.arange(4), 80)
        held_subset = np.repeat(np.arange(4), 20)
        raw_train = rng.normal(size=(320, 3, p))
        raw_held = rng.normal(size=(80, 3, p))
        cov_train = rng.normal(size=(320, 3, 6))
        cov_held = rng.normal(size=(80, 3, 6))
        # One stratum per subset; this makes the source of edges auditable.
        cov_train[:, :, 1] = 2.0
        cov_held[:, :, 1] = 2.0
        train_groups = np.asarray([f"train-{index}" for index in range(320)])
        held_groups = np.asarray([f"held-{index}" for index in range(80)])
        _, _, held_raw_null, _, diagnostics = _outer_null_pairing(
            raw_train, cov_train, train_subset, train_groups,
            raw_held, cov_held, held_subset, held_groups, seed=53,
        )
        permutation = np.asarray(diagnostics["held_permutation"])
        self.assertTrue(np.all(permutation != np.arange(len(held_subset))))
        self.assertTrue(np.all(held_groups[permutation] != held_groups))
        np.testing.assert_array_equal(held_raw_null[:, 1], raw_held[permutation, 1])
        self.assertEqual(set(diagnostics["length_edges"]), set(range(4)))

    def test_training_null_artifact_retains_zero_fixed_point_evidence(self):
        # Exercise the compact diagnostic serializer without paying for 200
        # full draws by temporarily shrinking the registered loop in-module.
        import scripts.automatic_group_free_phase_a4 as runner
        rng = np.random.default_rng(59)
        n, p = 160, len(FEATURE_ROSTER)
        subsets = np.repeat(np.arange(4), 40)
        folds = np.tile(np.repeat(np.arange(5), 8), 4)
        groups = np.asarray([f"group-{index}" for index in range(n)])
        raw = rng.normal(size=(n, 3, p))
        covariates = rng.normal(size=(n, 3, 6))
        covariates[:, :, 1] = 2.0
        arrays = {
            "raw": raw, "covariates": covariates, "subset": subsets,
            "group_id": groups, "outer_fold": folds,
        }
        original = runner.TRAIN_NULL_DRAWS
        runner.TRAIN_NULL_DRAWS = 1
        try:
            values, diagnostics = _training_nulls(arrays)
        finally:
            runner.TRAIN_NULL_DRAWS = original
        self.assertEqual(values.shape, (1,))
        self.assertEqual(len(diagnostics), 1)
        for fold in diagnostics[0]["folds"]:
            self.assertEqual(fold["outer_train_fixed_points"], 0)
            self.assertEqual(fold["outer_held_fixed_points"], 0)
            self.assertTrue(fold["outer_train_permutation_sha256"])
            self.assertTrue(fold["outer_held_permutation_sha256"])


if __name__ == "__main__":
    unittest.main()
