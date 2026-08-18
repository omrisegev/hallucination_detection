#!/usr/bin/env python3
"""CPU-only regression tests for the Fair Comparison v1 S2 stopping lane."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.fair_comparisons import stopping as S  # noqa: E402
from spectral_utils.fair_comparisons.registry import (  # noqa: E402
    ordered_id_sha256,
    validate_comparison_record,
)
from spectral_utils.paper_exact.manifest import sha256_order  # noqa: E402


def _raw_record(question_id: str, arm: str, answer: str, gold: str, *, correct=False):
    reasoning = 0 if arm == "nocot" else (8 if arm == "leash" else 12)
    closure = 2
    stopped = arm == "leash"
    return {
        "trace_key": f"{arm}:central:{question_id}",
        "question_id": question_id,
        "arm": arm,
        "setting_label": "central",
        "prompt_text": "fixture",
        "prompt_token_ids": [1],
        "gen_token_ids": list(range(reasoning)),
        "full_text": "fixture rationale",
        "answer_text": answer,
        "answer_token_ids": [2, 3],
        "n_reasoning_tokens": reasoning,
        "n_closure_tokens": closure,
        "n_total_tokens": reasoning + closure,
        "stop_reason": "policy" if stopped else ("n/a" if arm == "nocot" else "max_tokens"),
        "stopped_early": stopped,
        "closure_generated": True,
        "gold_answer": gold,
        # These deliberately imitate the broken numeric AQuA evaluation.
        "correct": bool(correct),
        "pred_answer": "17",
        "parse_status": "fallback_number",
        "wall_s": {"cot": 3.0, "leash": 2.0, "nocot": 1.0}[arm],
    }


def _write_run(root: Path, *, dataset="aqua", model="fixture/model", answers=None) -> Path:
    run = root / f"s2_leash_{model.replace('/', '-')}_{dataset}"
    (run / "shards").mkdir(parents=True)
    if dataset == "aqua":
        question_ids = ["aqua:0", "aqua:1", "aqua:2"]
        golds = ["A", "B", "C"]
        answers = answers or {
            "cot": ["A) alpha", "B) beta", r"\boxed{C}"],
            "leash": ["The correct option is A.", "The answer is C.", "choice C)"],
            "nocot": ["17", "B", "numeric only: 3"],
        }
        source = "deepmind/aqua_rat"
    else:
        question_ids = ["gsm8k:0", "gsm8k:1", "gsm8k:2"]
        golds = ["1", "2", "3"]
        answers = answers or {
            "cot": [r"\boxed{1}", r"\boxed{2}", r"\boxed{3}"],
            "leash": ["1", "2", "0"],
            "nocot": ["1", "0", "3"],
        }
        source = "openai/gsm8k"

    # Match the acquisition's arm-major shard order; the loader must return a canonical
    # question-major order regardless of how shards were cut.
    records = [
        _raw_record(question_id, arm, answers[arm][index], golds[index])
        for arm in ("leash", "cot", "nocot")
        for index, question_id in enumerate(question_ids)
    ]
    shard = run / "shards" / "shard_00000.pkl"
    with shard.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    payload = shard.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    index = {
        "shard": 0,
        "path": "shards/shard_00000.pkl",
        "n_traces": len(records),
        "bytes": len(payload),
        "sha256": digest,
        "keys": [record["trace_key"] for record in records],
        "question_ids": sorted(question_ids),
    }
    (run / "INDEX.jsonl").write_text(json.dumps(index) + "\n", encoding="utf-8")
    manifest = {
        "schema": "paper_exact_acquisition_v1",
        "run_id": run.name,
        "fidelity": "paper-specified-partial",
        "dataset_source": source,
        "dataset_revision": "test",
        "dataset_example_ids": question_ids,
        "dataset_order_sha256": sha256_order(question_ids),
        "model_id": model,
        "model_revision": "fixture-revision",
        "evaluator_revision": "paper_exact_evaluator_v1.0.0",
        "expected_traces": len(records),
        "extra": {"arms": ["leash", "cot", "nocot"], "sweep": False},
    }
    (run / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    status = {
        "n_expected": len(records),
        "n_finished": len(records),
        "n_failed": 0,
        "n_shards": 1,
        "bytes_total": len(payload),
        "failures": [],
        "complete": True,
    }
    (run / "STATUS.json").write_text(json.dumps(status), encoding="utf-8")
    return run


class AquaOptionParserTests(unittest.TestCase):
    def test_frozen_high_confidence_forms(self):
        fixtures = {
            r"\boxed{\text{D}}": ("D", "boxed_option"),
            "Therefore the final answer is B.": ("B", "explicit_option"),
            "This corresponds to option C) 12.5.": ("C", "explicit_option"),
            "**A) 36**": ("A", "leading_option"),
            ": \\( E) 110": ("E", "leading_option"),
        }
        for text, expected in fixtures.items():
            with self.subTest(text=text):
                parsed = S.parse_aqua_option(text)
                self.assertEqual((parsed["answer"], parsed["status"]), expected)
                self.assertEqual(parsed["parser_revision"], S.AQUA_PARSER_REVISION)

    def test_numeric_and_incidental_letters_are_unparsed_and_wrong(self):
        for text in ("78.20", "5(sqrt(3)+1)", "A car travels 5 miles", ""):
            with self.subTest(text=text):
                self.assertIsNone(S.parse_aqua_option(text)["answer"])
        graded = S.grade_aqua_option("78.20", "E")
        self.assertFalse(graded["correct"])
        self.assertEqual(graded["parse_status"], "none")

    def test_last_explicit_final_answer_wins_without_gold_access(self):
        parsed = S.parse_aqua_option("Option A looks plausible, but the final answer is C.")
        self.assertEqual(parsed["answer"], "C")
        with self.assertRaisesRegex(S.StoppingIntegrityError, "A--E"):
            S.grade_aqua_option("A)", "17")


class ArtifactIntegrityTests(unittest.TestCase):
    def test_complete_run_is_hash_checked_rescored_and_canonicalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_run(Path(temporary))
            loaded = S.load_s2_run(run)
        self.assertEqual(len(loaded["records"]), 9)
        self.assertTrue(loaded["audit"]["status_complete"])
        self.assertTrue(loaded["audit"]["identical_arm_question_ids"])
        defect = loaded["audit"]["upstream_aqua_parser_defect"]
        self.assertTrue(defect["detected"])
        self.assertFalse(defect["raw_artifacts_mutated"])
        self.assertFalse(defect["stored_summary_usable_for_accuracy"])
        self.assertGreater(loaded["audit"]["rescored_pass_at_1"], 0.0)
        self.assertEqual(
            loaded["audit"]["registered_question_ids"],
            ["aqua:0", "aqua:1", "aqua:2"],
        )
        self.assertEqual(
            loaded["audit"]["paired_group_order_sha256"],
            ordered_id_sha256(loaded["audit"]["registered_group_ids"]),
        )
        first = loaded["records"][0]
        self.assertEqual(validate_comparison_record(first), first)
        self.assertEqual(first["question_id"], "aqua:0")
        self.assertEqual(first["arm"], "cot")
        self.assertEqual(
            first["row_id"], "test::aqua::aqua:0::fixture/model::cot"
        )
        self.assertEqual(first["group_id"], "test::aqua::aqua:0")
        numeric = next(
            record for record in loaded["records"]
            if record["question_id"] == "aqua:0" and record["arm"] == "nocot"
        )
        self.assertTrue(numeric["parser_failure"])
        self.assertFalse(numeric["correct"])

    def test_gsm8k_uses_the_frozen_math_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_run(Path(temporary), dataset="gsm8k")
            loaded = S.load_s2_run(run)
        self.assertEqual(loaded["audit"]["rescoring_parser_revision"], S.GSM8K_PARSER_REVISION)
        self.assertFalse(loaded["audit"]["upstream_aqua_parser_defect"]["detected"])
        cot = [record for record in loaded["records"] if record["arm"] == "cot"]
        self.assertTrue(all(record["correct"] for record in cot))

    def test_incomplete_status_fails_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_run(Path(temporary))
            status_path = run / "STATUS.json"
            status = json.loads(status_path.read_text())
            status["complete"] = False
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(S.StoppingIntegrityError, "not complete"):
                S.load_s2_run(run)

    def test_index_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_run(Path(temporary))
            index_path = run / "INDEX.jsonl"
            index = json.loads(index_path.read_text())
            index["sha256"] = "0" * 64
            index_path.write_text(json.dumps(index) + "\n")
            with self.assertRaisesRegex(S.StoppingIntegrityError, "SHA-256 mismatch"):
                S.load_s2_run(run)

    def test_duplicate_and_missing_arm_rows_fail_the_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _write_run(root, model="fixture/one")
            second = _write_run(root, model="fixture/two")
            one = S.load_s2_run(first, verify_hashes=True)["records"]
            two = S.load_s2_run(second, verify_hashes=True)["records"]
        damaged = copy.deepcopy(two)
        damaged[-1]["question_id"] = "aqua:999"
        with self.assertRaisesRegex(S.StoppingIntegrityError, "identical IDs"):
            S.score_stopping_records(one + damaged, n_boot=20)


class MetricsAndBootstrapTests(unittest.TestCase):
    def _two_model_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = S.load_s2_run(_write_run(root, model="fixture/one"))["records"]
            two = S.load_s2_run(_write_run(root, model="fixture/two"))["records"]
        return one + two

    def test_metrics_count_real_closure_tokens_and_frontier(self):
        rows = self._two_model_rows()
        result = S.score_stopping_records(rows, n_boot=50, seed=20260818)
        self.assertTrue(result["pairing_audit"]["identical_question_ids_across_model_copies"])
        metrics = {
            (row["model"], row["arm"]): row for row in result["cell_metrics"]
        }
        leash = metrics[("fixture/one", "leash")]
        self.assertEqual(leash["reasoning_tokens"], 24)
        self.assertEqual(leash["closure_tokens"], 6)
        self.assertEqual(leash["total_tokens"], 30)
        self.assertEqual(leash["n_stopped_early"], 3)
        self.assertEqual(leash["n_forced_closure"], 3)
        self.assertTrue(leash["realized_savings_valid"])
        frontier = {
            (row["model"], row["arm"]): row
            for row in result["accuracy_compute_frontier"]
        }
        self.assertIn("pareto_efficient_within_cell", frontier[("fixture/one", "cot")])

    def test_paired_bootstrap_is_reproducible_and_carries_model_copies(self):
        rows = self._two_model_rows()
        first = S.paired_question_bootstrap(rows, n_boot=100, seed=20260818)
        second = S.paired_question_bootstrap(rows, n_boot=100, seed=20260818)
        self.assertEqual(first, second)
        lookup = {
            (row["model"], row["arm"], row["metric"]): row for row in first
        }
        # The two fixture models have identical question-level data. Shared bootstrap
        # draws therefore produce byte-identical interval endpoints.
        for arm in S.S2_ARMS:
            for metric in ("pass_at_1", "mean_tokens_per_question", "parser_failure_rate"):
                one = lookup[("fixture/one", arm, metric)]
                two = lookup[("fixture/two", arm, metric)]
                self.assertEqual((one["point"], one["lo"], one["hi"]),
                                 (two["point"], two["lo"], two["hi"]))
        contrast = lookup[("fixture/one", "leash", "accuracy_delta_vs_cot")]
        self.assertEqual(contrast["reference_arm"], "cot")
        self.assertEqual(contrast["contrast_direction"], "arm_minus_cot")
        self.assertEqual(contrast["n_groups"], 3)
        self.assertEqual(contrast["n_valid"], 100)
        # Explicit paired deltas use the same draws and arm-minus-CoT arithmetic.
        self.assertAlmostEqual(
            lookup[("fixture/one", "leash", "pass_at_1_delta_vs_cot")]["point"],
            -1.0 / 3.0,
        )
        self.assertAlmostEqual(
            lookup[("fixture/one", "leash", "mean_token_delta_vs_cot")]["point"],
            -4.0,
        )
        self.assertAlmostEqual(
            lookup[("fixture/one", "leash", "total_token_delta_vs_cot")]["point"],
            -12.0,
        )
        self.assertAlmostEqual(
            lookup[("fixture/one", "leash", "mean_wall_s_delta_vs_cot")]["point"],
            -1.0,
        )
        self.assertAlmostEqual(
            lookup[("fixture/one", "nocot", "pass_at_1_delta_vs_cot")]["point"],
            -2.0 / 3.0,
        )
        self.assertAlmostEqual(
            lookup[("fixture/one", "nocot", "total_token_delta_vs_cot")]["point"],
            -36.0,
        )
        self.assertAlmostEqual(
            lookup[("fixture/one", "nocot", "mean_wall_s_delta_vs_cot")]["point"],
            -2.0,
        )
        for arm in ("leash", "nocot"):
            for metric in (
                "pass_at_1_delta_vs_cot",
                "mean_token_delta_vs_cot",
                "total_token_delta_vs_cot",
                "mean_wall_s_delta_vs_cot",
                "early_stop_rate_delta_vs_cot",
                "forced_closure_rate_delta_vs_cot",
                "parser_failure_rate_delta_vs_cot",
            ):
                row = lookup[("fixture/one", arm, metric)]
                self.assertLessEqual(row["lo"], row["point"])
                self.assertGreaterEqual(row["hi"], row["point"])

    def test_missing_real_closure_invalidates_savings(self):
        rows = self._two_model_rows()
        bad = copy.deepcopy(rows)
        target = next(row for row in bad if row["arm"] == "leash")
        target["closure_generated"] = False
        target["forced_closure"] = False
        with self.assertRaisesRegex(S.StoppingIntegrityError, "lacks a real closure"):
            S.score_stopping_records(bad, n_boot=20)


@unittest.skipUnless(
    os.environ.get("RUN_REAL_S2_STOPPING_TEST") == "1",
    "set RUN_REAL_S2_STOPPING_TEST=1 for the 406 MB local-cache integrity test",
)
class RealCacheIntegrationTests(unittest.TestCase):
    def test_six_complete_downloaded_cells(self):
        cache = REPO_ROOT / "local_cache" / "fair_paper_exact_comparisons_v1"
        package = S.build_s2_stopping_lane(cache, n_boot=20)
        self.assertEqual(package["schema"], "fair_s2_stopping_lane_package_v1")
        self.assertEqual(package["suite_audit"]["n_cells"], 6)
        self.assertEqual(len(package["cell_metrics"]), 18)
        aqua_audits = [row for row in package["run_audits"] if row["dataset"] == "aqua"]
        self.assertEqual(len(aqua_audits), 3)
        self.assertTrue(all(row["upstream_aqua_parser_defect"]["detected"] for row in aqua_audits))


if __name__ == "__main__":
    unittest.main(verbosity=2)
