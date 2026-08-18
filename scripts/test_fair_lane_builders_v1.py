#!/usr/bin/env python3
"""Focused tests for Global and Localization fair-comparison lane builders."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spectral_utils.fair_comparisons.evaluator import (  # noqa: E402
    fit_localization_threshold,
    localization_metrics,
)
from spectral_utils.fair_comparisons.global_lane import (  # noqa: E402
    REGISTERED_QWEN8_GSM8K_DUFS_ANCHOR_SHA256,
    crossfit_operating_points,
    evaluate_global_panel,
    load_classic_global_fit_ids,
    verify_registered_dufs_provenance,
)
from spectral_utils.fair_comparisons.localization import crossfit_score_method  # noqa: E402
from spectral_utils.historical_multitask_baselines import (  # noqa: E402
    REGISTERED_DUFS_EPOCHS,
    REGISTERED_DUFS_K,
    REGISTERED_DUFS_LAMBDA,
    REGISTERED_DUFS_SEEDS,
    fit_registered_dufs_global,
)
from scripts.build_fair_paper_exact_comparisons_v1 import (  # noqa: E402
    GLOBAL_ALL_METHOD_IDS,
    GLOBAL_CONTEXT_METHOD_IDS,
    GLOBAL_DIRECT_METHOD_IDS,
    PB_GLOBAL_POPULATION,
    TWENTYFOUR_SOURCE_TOKEN,
    _evaluator_run_contract,
    _global_join_expectations,
    _partition_global_report_rows,
    _portable_twentyfour_output,
    _portable_unified_tree_manifest,
    _report_identity,
)
from spectral_utils.fair_comparisons.registry import canonical_sha256  # noqa: E402
from spectral_utils.fair_comparisons.reporting import write_reports  # noqa: E402


HASH = "a" * 64
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


class PublicationBuilderPresentationTests(unittest.TestCase):
    def test_testing_report_is_visibly_non_publication_and_binds_actual_bootstrap(self):
        identity = _report_identity(
            testing_only=True,
            n_boot=7,
            testing_deviations=("bootstrap_replicates=7 (required=2000)",),
        )
        self.assertTrue(identity["title"].startswith("TEST-ONLY"))
        self.assertIn("TEST-ONLY NON-PUBLICATION OUTPUT", identity["summary"])
        self.assertTrue(identity["testing_only"])
        self.assertFalse(identity["publication_build_mode_eligible"])
        self.assertEqual(
            identity["publication_acceptance_status_at_build"],
            "ineligible-testing-only",
        )
        self.assertEqual(identity["bootstrap_replicates"], 7)
        self.assertEqual(identity["bootstrap_seed"], 20260818)
        self.assertIn("not publication intervals", identity["confidence_interval_status"])

        evaluator = _evaluator_run_contract(7)
        self.assertEqual(evaluator["bootstrap_replicates"], 7)
        self.assertEqual(evaluator["publication_default_bootstrap_replicates"], 2000)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_reports(
                output,
                title=identity["title"],
                summary=identity["summary"],
                direct_tables=(),
                native_context_tables=(),
                partial_blocked_tables=(),
                provenance={
                    "testing_only": identity["testing_only"],
                    "publication_build_mode_eligible": identity[
                        "publication_build_mode_eligible"
                    ],
                    "publication_acceptance_status_at_build": identity[
                        "publication_acceptance_status_at_build"
                    ],
                    "bootstrap_replicates": identity["bootstrap_replicates"],
                    "bootstrap_seed": identity["bootstrap_seed"],
                    "confidence_interval_status": identity[
                        "confidence_interval_status"
                    ],
                },
            )
            markdown = (output / "REPORT.md").read_text(encoding="utf-8")
            advisor = (output / "REPORT.html").read_text(encoding="utf-8")
        for rendered in (markdown, advisor):
            self.assertIn("TEST-ONLY", rendered)
            self.assertIn("NON-PUBLICATION OUTPUT", rendered)
            self.assertIn("bootstrap_replicates", rendered)

    def test_publication_report_retains_unwatermarked_title_and_ci_contract(self):
        identity = _report_identity(
            testing_only=False,
            n_boot=2000,
            testing_deviations=(),
        )
        self.assertEqual(identity["title"], "Fair Paper-Exact Comparison Package v1")
        self.assertNotIn("TEST-ONLY", identity["summary"])
        self.assertFalse(identity["testing_only"])
        self.assertTrue(identity["publication_build_mode_eligible"])
        self.assertTrue(identity["publication_acceptance_requires_independent_rebuild"])
        self.assertEqual(
            identity["publication_acceptance_status_at_build"],
            "pending-independent-byte-identical-rebuild",
        )
        self.assertIn("95% percentile intervals", identity["confidence_interval_status"])

    def test_twentyfour_runtime_roots_serialize_byte_identically(self):
        def source_payload(root: Path) -> dict[str, object]:
            audit = {
                "schema": "24cell_partial_identity_audit_v1",
                "audits": [
                    {
                        "source": {
                            "raw_path": str(root / "cell" / "raw.pkl"),
                            "manifest_path": str(root / "cell" / "manifest.json"),
                        }
                    }
                ],
            }
            return {
                "sources": audit["audits"],
                "identity_audit": {
                    **audit,
                    "audit_sha256": canonical_sha256(audit),
                },
            }

        first_root = Path("/private/tmp/twentyfour-stage-a")
        second_root = Path("/private/tmp/twentyfour-stage-b")
        first = _portable_twentyfour_output(
            source_payload(first_root), source_root=first_root
        )
        second = _portable_twentyfour_output(
            source_payload(second_root), source_root=second_root
        )
        self.assertEqual(first, second)
        serialized = json.dumps(first, sort_keys=True)
        self.assertIn(TWENTYFOUR_SOURCE_TOKEN, serialized)
        self.assertNotIn(str(first_root), serialized)
        self.assertNotIn(str(second_root), serialized)
        portable_audit = dict(first["identity_audit"])
        observed_hash = portable_audit.pop("audit_sha256")
        self.assertEqual(observed_hash, canonical_sha256(portable_audit))

    def test_unified_worktree_manifest_is_directory_name_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "worktree-one"
            second = root / "renamed-worktree"
            first.mkdir()
            second.mkdir()
            (first / "frozen.txt").write_text("identical\n", encoding="utf-8")
            (second / "frozen.txt").write_text("identical\n", encoding="utf-8")
            first_manifest = _portable_unified_tree_manifest(first)
            second_manifest = _portable_unified_tree_manifest(second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["root_label"], "${UNIFIED_WORKTREE}")


class FastLocalizationThresholdTests(unittest.TestCase):
    def test_single_locator_sweep_matches_literal_bruteforce(self):
        rng = np.random.default_rng(20260818)
        rows = []
        for subset in SUBSETS:
            for index in range(35):
                label = -1 if index % 4 == 0 else index % 6
                rows.append(
                    {
                        "subset": subset,
                        "first_error": label,
                        "step_scores": [float(rng.integers(0, 9))],
                        "step_indices": [int(rng.integers(0, 6))],
                    }
                )
        fitted = fit_localization_threshold(rows, expected_subsets=SUBSETS)
        candidates = sorted(
            {row["step_scores"][0] for row in rows}
            | {float(np.nextafter(max(row["step_scores"][0] for row in rows), np.inf))}
        )
        literal = []
        for threshold in candidates:
            predictions = [
                row["step_indices"][0] if row["step_scores"][0] >= threshold else -1
                for row in rows
            ]
            metric = localization_metrics(
                rows, predictions, expected_subsets=SUBSETS
            )
            literal.append(
                (
                    metric["equal_subset_macro_f1"],
                    metric["equal_subset_clean_accuracy"],
                    threshold,
                )
            )
        expected = max(literal)
        self.assertEqual(fitted["threshold"], expected[2])
        self.assertAlmostEqual(fitted["equal_subset_macro_f1"], expected[0])
        self.assertAlmostEqual(fitted["equal_subset_clean_accuracy"], expected[1])
        self.assertEqual(fitted["threshold_sweep"], "single_frozen_locator_blockwise_exact")

    def test_crossfit_score_method_changes_threshold_only(self):
        rows = []
        for subset in SUBSETS:
            for index in range(20):
                row_id = f"pb::{subset}::{index}"
                rows.append(
                    {
                        "row_id": row_id,
                        "group_id": row_id,
                        "cell_id": f"cell::{subset}",
                        "family": subset,
                        "subset": subset,
                        "first_error": -1 if index % 4 == 0 else index % 3,
                        "stratify_label": int(index % 4 != 0),
                        "fold": index % 5,
                        "continuous_score": float(index % 7),
                        "locator": index % 3,
                        "source_artifact_hash": HASH,
                    }
                )
        result = crossfit_score_method(
            rows, method_id="frozen", population_id="population::localization"
        )
        self.assertEqual(len(result["records"]), len(rows))
        for source, scored in zip(rows, result["records"]):
            self.assertEqual(scored["continuous_score"], source["continuous_score"])
            self.assertEqual(scored["locator"], source["locator"])
            self.assertEqual(scored["fold"], source["fold"])
            self.assertRegex(scored["calibration_hash"], r"^[0-9a-f]{64}$")


class GlobalLaneTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        ordered = []
        for family in ("a", "b"):
            for index in range(20):
                row_id = f"row::{family}::{index}"
                ordered.append(row_id)
                label = index % 2
                for method, offset in (("u28", 0.3), ("incumbent", 0.0)):
                    rows.append(
                        {
                            "method_id": method,
                            "row_id": row_id,
                            "group_id": row_id,
                            "cell_id": f"cell::{family}",
                            "family": family,
                            "budget": "final",
                            "label": label,
                            "continuous_score": float(label + offset + (index % 3) / 10),
                            "fold": index % 5,
                        }
                    )
        return rows, ordered

    def test_crossfit_operating_points_never_fit_on_held_fold(self):
        rows, _ = self.rows()
        method_rows = [row for row in rows if row["method_id"] == "u28"]
        result = crossfit_operating_points(method_rows)
        self.assertEqual(set(result), {"fpr_05", "fpr_10"})
        for point in result.values():
            self.assertEqual(len(point["calibration_ledgers"]), 5)
            for ledger in point["calibration_ledgers"]:
                self.assertNotIn(ledger["held_out_fold"], ledger["train_folds"])
            self.assertLessEqual(point["observed_fpr"], 0.25)

    def test_panel_uses_error_auprc_and_equal_family_macro(self):
        rows, ordered = self.rows()
        panel = evaluate_global_panel(
            rows, ordered_ids=ordered, method_ids=("u28", "incumbent")
        )
        self.assertEqual(panel["positive_class"], "final_answer_wrong")
        self.assertEqual(len(panel["methods"]), 2)
        for method in panel["methods"]:
            self.assertTrue(math.isfinite(method["equal_family_error_auprc"]))
            self.assertNotIn("auprc_correct", method)


class GlobalReportRosterTests(unittest.TestCase):
    def test_direct_and_context_rosters_are_exact_and_disjoint(self):
        self.assertEqual(
            GLOBAL_DIRECT_METHOD_IDS,
            (
                "unified28",
                "classic_mixed_v2_no_length",
                "mixed_v2_dufs_liu_l0p1_no_length",
                "max_entropy_global",
            ),
        )
        self.assertEqual(
            GLOBAL_CONTEXT_METHOD_IDS,
            (
                "unified28_dufs_l0p1",
                "unified28_dufs_l0p3",
                "unified28_dufs_l1",
                "unified28_dufs_l3",
                "unified28_task_reweighted_a0p5_historical",
                "ordinary36_historical_control",
            ),
        )
        self.assertFalse(set(GLOBAL_DIRECT_METHOD_IDS) & set(GLOBAL_CONTEXT_METHOD_IDS))
        self.assertEqual(
            GLOBAL_ALL_METHOD_IDS,
            GLOBAL_DIRECT_METHOD_IDS + GLOBAL_CONTEXT_METHOD_IDS,
        )

    def test_report_partition_cannot_promote_frozen_search_controls(self):
        source = [
            {"method_id": method_id, "headline_eligible": True}
            for method_id in reversed(GLOBAL_ALL_METHOD_IDS)
        ]
        direct, context = _partition_global_report_rows(source)
        self.assertEqual(
            [row["method_id"] for row in direct], list(GLOBAL_DIRECT_METHOD_IDS)
        )
        self.assertEqual(
            [row["method_id"] for row in context], list(GLOBAL_CONTEXT_METHOD_IDS)
        )
        self.assertTrue(all(row["headline_eligible"] for row in direct))
        self.assertTrue(all(not row["headline_eligible"] for row in context))
        self.assertTrue(
            all(row["comparison_scope"] == "frozen-search-context-only" for row in context)
        )
        with self.assertRaisesRegex(ValueError, "Global report roster mismatch"):
            _partition_global_report_rows(source[:-1])

    def test_join_audit_marks_only_direct_roster_as_headline(self):
        expectations = _global_join_expectations(
            {
                "method_ids": GLOBAL_ALL_METHOD_IDS,
                "direct_method_ids": GLOBAL_DIRECT_METHOD_IDS,
                "context_method_ids": GLOBAL_CONTEXT_METHOD_IDS,
            }
        )
        self.assertEqual(
            [row["method_id"] for row in expectations if row["headline"]],
            list(GLOBAL_DIRECT_METHOD_IDS),
        )
        self.assertEqual(
            [row["method_id"] for row in expectations if not row["headline"]],
            list(GLOBAL_CONTEXT_METHOD_IDS),
        )
        self.assertTrue(
            all(row["population_id"] == PB_GLOBAL_POPULATION for row in expectations)
        )


class RegisteredGlobalDUFSReplayTests(unittest.TestCase):
    def test_classic_fit_identity_adapter_reads_no_labels_or_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classic.jsonl"
            rows = []
            for family in SUBSETS:
                for index in range(32):
                    unit = f"{index:04d}"
                    for repeat in range(3):
                        rows.append(
                            {
                                "candidate": "classic_mixed_v2_no_length",
                                "family": family,
                                "model": "qwen3_8b",
                                "unit": unit,
                                "source_group": f"{family}::{unit}",
                                "repeat": repeat,
                                "fold": repeat,
                                # These values are intentionally nonsensical.  The
                                # identity-only adapter must never inspect them.
                                "wrong": "forbidden-label",
                                "global_score": "forbidden-score",
                            }
                        )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            audit = load_classic_global_fit_ids(path)
        self.assertEqual(audit["observed_rows"], 384)
        self.assertFalse(audit["labels_or_scores_read"])
        self.assertEqual(
            {family: len(values) for family, values in audit["fit_ids_by_family"].items()},
            {family: 32 for family in SUBSETS},
        )

    def test_frozen_dufs_constants_and_score_fit_ignore_labels(self):
        rng = np.random.default_rng(20260818)
        rows = []
        for index in range(24):
            entropy = rng.normal(1.5 + index / 80.0, 0.25, size=48)
            spilled = rng.normal(0.8 - index / 100.0, 0.18, size=48)
            rows.append(
                {
                    "token_entropies": entropy,
                    "token_spilled_energies": spilled,
                    "token_logsumexp": rng.normal(5.0, 0.3, size=48),
                    "top_k_logprobs": {
                        "ids": np.zeros((48, 8), dtype=np.int32),
                        "logprobs": np.sort(
                            rng.normal(-2.0, 0.6, size=(48, 8)), axis=1
                        )[:, ::-1],
                    },
                    "label": index % 2,
                }
            )
        permuted = [
            {**row, "label": 1 - int(row["label"]), "final_answer_correct": None}
            for row in rows
        ]
        stripped = [
            {key: value for key, value in row.items() if key != "label"}
            for row in rows
        ]
        first = fit_registered_dufs_global(rows)
        second = fit_registered_dufs_global(permuted)
        third = fit_registered_dufs_global(stripped)
        self.assertEqual(first.names, second.names)
        np.testing.assert_array_equal(first.weights, second.weights)
        np.testing.assert_array_equal(first.training_scores, second.training_scores)
        np.testing.assert_array_equal(first.weights, third.weights)
        np.testing.assert_array_equal(first.training_scores, third.training_scores)
        self.assertNotIn("trace_length", first.names)
        self.assertEqual(tuple(first.diagnostics["dufs_seeds"]), REGISTERED_DUFS_SEEDS)
        self.assertEqual(first.diagnostics["dufs_epochs"], REGISTERED_DUFS_EPOCHS)
        self.assertEqual(first.diagnostics["graph_k"], REGISTERED_DUFS_K)
        self.assertEqual(first.diagnostics["lambda"], REGISTERED_DUFS_LAMBDA)
        self.assertFalse(first.diagnostics["labels_seen_during_fit"])

    def test_provenance_gate_binds_anchor_and_both_source_ledgers(self):
        hashes = {family: (str(index + 1) * 64)[:64] for index, family in enumerate(SUBSETS)}
        llama_hashes = {
            family: (chr(ord("a") + index) * 64)
            for index, family in enumerate(SUBSETS)
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            definition = root / "definition.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "model": "qwen3_8b",
                                "subset": "gsm8k",
                                "score_hashes": {
                                    "global_mixed_v2_dufs": (
                                        REGISTERED_QWEN8_GSM8K_DUFS_ANCHOR_SHA256
                                    )
                                },
                            }
                        ],
                        "labels_or_step_spans_read": False,
                        "global_detector": "mixed-v2 DUFS-LIU, lambda=0.1, k=7",
                    }
                ),
                encoding="utf-8",
            )
            definition.write_text(
                json.dumps(
                    {
                        "classic_contract": (
                            "registered mixed-v2 30-feature contract with final length excluded"
                        ),
                        "classic_labels_seen_during_fit": False,
                        "inventory": [
                            {"model": "qwen3_8b", "family": family, "sha256": value}
                            for family, value in hashes.items()
                        ],
                        "validation_inventory": [
                            {
                                "model": "llama31_8b",
                                "family": family,
                                "sha256": value,
                            }
                            for family, value in llama_hashes.items()
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit = verify_registered_dufs_provenance(
                anchor_manifest_path=manifest,
                classic_run_definition_path=definition,
                qwen_source_hashes=hashes,
                llama_source_hashes=llama_hashes,
            )
            self.assertTrue(audit["passed"])
            with self.assertRaisesRegex(ValueError, "Qwen replay telemetry hashes"):
                verify_registered_dufs_provenance(
                    anchor_manifest_path=manifest,
                    classic_run_definition_path=definition,
                    qwen_source_hashes={**hashes, "gsm8k": "f" * 64},
                    llama_source_hashes=llama_hashes,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
