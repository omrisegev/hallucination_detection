#!/usr/bin/env python
"""CPU-only regression tests for the Fair Comparison v1 evaluator contract."""

from __future__ import annotations

import os
import random
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spectral_utils.fair_comparisons import evaluator as E  # noqa: E402
from spectral_utils.fair_comparisons import folds as F  # noqa: E402


class HashAndFoldTests(unittest.TestCase):
    def _rows(self):
        rows = []
        for family in ("math", "qa"):
            for label in (0, 1):
                for index in range(13):
                    group_id = f"{family}:{label}:{index}"
                    for method in ("unified28", "incumbent", "copy"):
                        rows.append(
                            {
                                "group_id": group_id,
                                "family": family,
                                "stratify_label": label,
                                "label": -1 if label == 0 else index,
                                "method_id": method,
                            }
                        )
        return rows

    def test_canonical_hash_is_order_stable_but_population_hash_is_order_sensitive(self):
        self.assertEqual(F.canonical_sha256({"b": 2, "a": 1}), F.canonical_sha256({"a": 1, "b": 2}))
        self.assertNotEqual(F.ordered_id_sha256(["a", "b"]), F.ordered_id_sha256(["b", "a"]))
        with self.assertRaisesRegex(ValueError, "duplicate ordered row ids"):
            F.ordered_id_sha256(["a", "a"])

    def test_folds_are_input_order_independent_and_copies_are_isolated(self):
        rows = self._rows()
        first = F.assign_group_folds(rows)
        shuffled = list(rows)
        random.Random(20260818).shuffle(shuffled)
        second = F.assign_group_folds(shuffled)
        self.assertEqual(first, second)
        attached = F.attach_folds(rows, first)
        F.assert_group_fold_isolation(attached)
        for family in ("math", "qa"):
            for label in (0, 1):
                counts = [
                    sum(
                        group.startswith(f"{family}:{label}:") and fold == target
                        for group, fold in first.items()
                    )
                    for target in range(5)
                ]
                self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(F.fold_assignment_sha256(first), F.fold_assignment_sha256(second))
        one_stratum = sorted(
            [group for group in first if group.startswith("math:0:")],
            key=lambda group: (F.sha256_text(group), group),
        )
        self.assertEqual([first[group] for group in one_stratum], [index % 5 for index in range(13)])

    def test_group_label_or_family_conflict_fails(self):
        rows = [
            {"group_id": "q", "family": "math", "stratify_label": 0, "label": -1},
            {"group_id": "q", "family": "math", "stratify_label": 1, "label": 7},
        ]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            F.assign_group_folds(rows)

    def test_raw_lane_label_is_separate_from_binary_stratification_label(self):
        rows = [
            {"group_id": "clean", "family": "pb", "stratify_label": 0, "label": -1},
            {"group_id": "error", "family": "pb", "stratify_label": 1, "label": 23},
        ]
        assignments = F.assign_group_folds(rows)
        self.assertEqual(set(assignments), {"clean", "error"})
        # The helper accepts an explicitly registered opaque hashable stratum, too.
        alternate = [
            {"group_id": "a", "family": "f", "stratify_label": ("clean", 0)},
            {"group_id": "b", "family": "f", "stratify_label": ("error", 1)},
        ]
        self.assertEqual(set(F.assign_group_folds(alternate)), {"a", "b"})

    def test_manual_fold_split_fails(self):
        with self.assertRaisesRegex(ValueError, "appears in folds"):
            F.assert_group_fold_isolation(
                [
                    {"group_id": "q", "fold": 0},
                    {"group_id": "q", "fold": 1},
                ]
            )


class DetectionMetricTests(unittest.TestCase):
    def test_positive_class_is_error_and_orientation_is_not_flipped(self):
        labels = [0, 0, 1, 1]
        risk = [0.0, 0.1, 0.9, 1.0]
        metrics = E.detection_metrics(labels, risk)
        self.assertEqual(metrics["positive_class"], "error/risk")
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["error_auprc"], 1.0)
        self.assertEqual(E.auroc(labels, [-value for value in risk]), 0.0)

    def test_auroc_and_ap_handle_ties_without_row_order_dependence(self):
        self.assertEqual(E.auroc([1, 0], [7.0, 7.0]), 0.5)
        labels = [1, 0, 1, 0]
        scores = [1.0, 1.0, 0.0, 0.0]
        self.assertAlmostEqual(E.average_precision(labels, scores), 0.5)
        order = [3, 1, 2, 0]
        self.assertAlmostEqual(
            E.average_precision([labels[i] for i in order], [scores[i] for i in order]),
            0.5,
        )

    def test_nonfinite_scores_fail_instead_of_changing_population(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            E.detection_metrics([0, 1], [0.0, float("nan")])
        with self.assertRaisesRegex(ValueError, "binary"):
            E.detection_metrics([0, -1], [0.0, 1.0])

    def test_fixed_fpr_uses_correct_rows_only_and_never_splits_ties(self):
        labels = [0] * 20 + [1, 1]
        scores = list(range(20)) + [10_000, 20_000]
        ledger = E.calibrate_correct_only_threshold(labels, scores, target_fpr=0.05)
        self.assertEqual(ledger["n_correct_calibration"], 20)
        self.assertEqual(ledger["n_false_positive_calibration"], 1)
        self.assertAlmostEqual(ledger["observed_calibration_fpr"], 0.05)
        tied = E.calibrate_correct_only_threshold([0] * 20, [1.0] * 20, target_fpr=0.05)
        self.assertEqual(tied["n_false_positive_calibration"], 0)
        self.assertGreater(tied["threshold"], 1.0)

    def test_operating_point_reports_error_tpr_precision_and_observed_fpr(self):
        result = E.operating_point([0, 0, 1, 1], [0.0, 0.8, 0.7, 0.9], 0.75)
        self.assertAlmostEqual(result["error_tpr"], 0.5)
        self.assertAlmostEqual(result["error_precision"], 0.5)
        self.assertAlmostEqual(result["observed_fpr"], 0.5)

    def test_foldwise_guard_rejects_refit_pooling_and_heterogeneous_cells(self):
        with self.assertRaisesRegex(ValueError, "continuously refit"):
            E.pooled_detection_metrics([0, 1], [0.0, 1.0], scores_refit_within_fold=True)
        with self.assertRaisesRegex(ValueError, "heterogeneous"):
            E.pooled_detection_metrics([0, 1], [0.0, 1.0], cell_ids=["a", "b"])
        rows = [
            {"label": 0, "score": 0.0, "fold": 0, "cell_id": "a"},
            {"label": 1, "score": 1.0, "fold": 0, "cell_id": "a"},
            {"label": 0, "score": 0.2, "fold": 1, "cell_id": "a"},
            {"label": 1, "score": 0.8, "fold": 1, "cell_id": "a"},
        ]
        result = E.foldwise_detection_metrics(rows)
        self.assertEqual(result["equal_fold_mean"]["auroc"], 1.0)
        self.assertEqual(result["aggregation"], "evaluate_within_fold_then_equal_fold_mean")
        rows[-1] = {**rows[-1], "cell_id": "b"}
        with self.assertRaisesRegex(ValueError, "heterogeneous"):
            E.foldwise_detection_metrics(rows)


class PrefixTests(unittest.TestCase):
    def test_budget_eligibility_is_strictly_greater(self):
        self.assertTrue(E.eligible_at_budget(17, 16))
        self.assertFalse(E.eligible_at_budget(16, 16))
        self.assertFalse(E.eligible_at_budget(15, 16))
        rows = [{"final_length": 15}, {"final_length": 16}, {"final_length": 17}]
        self.assertEqual(E.unfinished_rows(rows, 16), [rows[-1]])

    def test_prefix_calibration_uses_complete_trace_maxima(self):
        paths = []
        for index in range(20):
            # The maximum is deliberately at a different budget than the first score.
            path = [0.0] * 6
            path[index % 6] = float(index)
            paths.append(path)
        ledgers = E.calibrate_prefix_ever_warning_thresholds([0] * 20, paths)
        five = ledgers["fpr_05"]
        self.assertEqual(five["calibration_population"], "correct_trace_six_budget_maximum")
        self.assertEqual(five["n_false_positive_calibration"], 1)
        self.assertAlmostEqual(five["observed_calibration_fpr"], 0.05)
        with self.assertRaisesRegex(ValueError, "complete horizon"):
            E.calibrate_prefix_ever_warning_thresholds([0], [[0.0] * 5])

    def test_prefix_warning_metrics_use_the_common_grid(self):
        rows = [
            [0.0, 0.0, 0.8, 0.8, 0.8, 0.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.9, 0.9, 0.9, 0.9],
        ]
        result = E.prefix_warning_metrics([1, 1, 0], rows, threshold=0.5)
        self.assertAlmostEqual(result["wrong_trace_warning_coverage"], 0.5)
        self.assertAlmostEqual(result["correct_trace_ever_warning_fpr"], 1.0)
        self.assertEqual(result["first_warning_budgets"], [64, None, 32])

    def test_recovered_signal_is_above_chance_not_raw_auroc(self):
        self.assertAlmostEqual(E.recovered_above_chance_signal(0.595, 0.6), 0.95)
        self.assertEqual(
            E.earliest_budget_reaching_signal([16, 32], [0.58, 0.595], 0.6),
            32,
        )


class LocalizationTests(unittest.TestCase):
    SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")

    def _crossfit_rows(self):
        rows = []
        for fold in range(5):
            for subset in self.SUBSETS:
                rows.extend(
                    [
                        {
                            "group_id": f"{subset}:clean:{fold}",
                            "subset": subset,
                            "first_error": -1,
                            "step_scores": [0.1, 0.2],
                            "fold": fold,
                        },
                        {
                            "group_id": f"{subset}:error:{fold}",
                            "subset": subset,
                            "first_error": 1,
                            "step_scores": [0.1, 0.9],
                            "fold": fold,
                        },
                    ]
                )
        return rows

    def test_processbench_clean_is_minus_one_and_unparsed_counts_wrong(self):
        metrics = E.processbench_f1([None, 0, -1], [0, -1, -1])
        self.assertEqual(E.NO_ERROR, -1)
        self.assertEqual(metrics["n_unparsed"], 1)
        self.assertEqual(metrics["error_acc"], 0.0)
        self.assertAlmostEqual(metrics["correct_acc"], 0.5)

    def test_threshold_fit_and_five_fold_discrete_crossfit(self):
        rows = self._crossfit_rows()
        fit = E.fit_localization_threshold(rows, expected_subsets=self.SUBSETS)
        self.assertAlmostEqual(fit["equal_subset_macro_f1"], 1.0)
        self.assertEqual(fit["tie_break"], ["macro_f1", "clean_accuracy", "higher_threshold"])
        result = E.crossfit_localization_threshold(rows, expected_subsets=self.SUBSETS)
        self.assertEqual(len(result["calibration_ledgers"]), 5)
        self.assertEqual(result["official_oof_metrics"]["equal_subset_macro_f1"], 1.0)
        self.assertEqual(result["aggregation"], "concatenated_discrete_out_of_fold_predictions")
        self.assertTrue(all(len(item["calibration_hash"]) == 64 for item in result["calibration_ledgers"]))

    def test_within_one_is_erroneous_traces_only(self):
        rows = [
            {"subset": "s", "first_error": -1},
            {"subset": "s", "first_error": 3},
        ]
        metrics = E.localization_metrics(rows, [-1, 4])
        self.assertEqual(metrics["within_one_error_accuracy"], 1.0)


class BootstrapTests(unittest.TestCase):
    def test_default_2000_draw_paired_family_bootstrap_is_reproducible_and_refits(self):
        groups = {
            f"q{index}": {
                "unified28": {"b16": index + 0.5, "b32": index + 1.0},
                "incumbent": {"b16": index, "b32": index + 0.25},
                "copy": {"unified28": index + 0.5},
            }
            for index in range(8)
        }
        strata = {f"q{index}": "math" if index < 4 else "qa" for index in range(8)}

        refit_calls = {"n": 0}

        def recompute(payloads):
            refit_calls["n"] += 1
            return sum(item["incumbent"]["b16"] for item in payloads) / len(payloads)

        def statistic(payloads, fitted):
            # Presence checks ensure each resampled source payload carries methods,
            # budgets, and the repeated scorer copy together.
            self.assertTrue(all(set(item) == {"unified28", "incumbent", "copy"} for item in payloads))
            delta16 = sum(
                item["unified28"]["b16"] - item["incumbent"]["b16"]
                for item in payloads
            ) / len(payloads)
            delta32 = sum(
                item["unified28"]["b32"] - item["incumbent"]["b32"]
                for item in payloads
            ) / len(payloads)
            # ``fitted`` is intentionally used so omitting per-replicate recomputation
            # would alter the returned distribution.
            return {"delta_b16": delta16 + 0.0 * fitted, "delta_b32": delta32}

        first = E.paired_grouped_bootstrap(
            groups, statistic, recompute=recompute, strata=strata
        )
        self.assertEqual(refit_calls["n"], 2001)
        refit_calls["n"] = 0
        second = E.paired_grouped_bootstrap(
            groups, statistic, recompute=recompute, strata=strata
        )
        self.assertEqual(first, second)
        self.assertEqual(refit_calls["n"], 2001)
        self.assertEqual(first["n_boot"], 2000)
        self.assertEqual(first["seed"], 20260818)
        self.assertEqual(first["n_groups_by_stratum"], {"math": 4, "qa": 4})
        self.assertTrue(first["recomputed_each_replicate"])
        self.assertAlmostEqual(first["statistics"]["delta_b16"]["point"], 0.5)
        self.assertAlmostEqual(first["statistics"]["delta_b32"]["point"], 0.75)
        self.assertEqual(first["statistics"]["delta_b16"]["n_valid"], 2000)


class IdentityTests(unittest.TestCase):
    def test_evaluator_identity_pins_contract(self):
        summary = E.summary_dict()
        self.assertEqual(summary["derived_from"], "paper_exact_evaluator_v1.0.0")
        self.assertEqual(summary["positive_class"], "error/risk")
        self.assertEqual(summary["processbench_clean_label"], -1)
        self.assertEqual(summary["bootstrap_replicates"], 2000)
        self.assertEqual(summary["bootstrap_seed"], 20260818)

    def test_realized_token_accounting_includes_closure_and_rejects_missing_closure(self):
        records = [
            {
                "n_reasoning_tokens": 80,
                "n_closure_tokens": 20,
                "stopped_early": True,
                "closure_generated": True,
            },
            {
                "n_reasoning_tokens": 50,
                "n_closure_tokens": 0,
                "stopped_early": True,
                "closure_generated": False,
            },
        ]
        accounting = E.token_accounting(records)
        self.assertEqual(accounting["total_tokens"], 150)
        self.assertEqual(accounting["closure_tokens"], 20)
        self.assertEqual(accounting["n_stopped_without_closure"], 1)
        self.assertFalse(accounting["realized_savings_valid"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
