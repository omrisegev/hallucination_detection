#!/usr/bin/env python3
"""Mechanical checks for the frozen-24 published-comparator registry.

The registry deliberately keeps paper-table values out of the direct v2
leaderboard.  These tests guard that boundary and, when the pinned v2 release
is present, confirm that each recorded point-estimate leader is reproduced
from its atomic metrics table.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    ROOT
    / "configs"
    / "reconstruction_benchmark_v1"
    / "frozen24_published_comparator_registry_v1.json"
)
FROZEN_CELLS_PATH = (
    ROOT / "configs" / "reconstruction_benchmark_v1" / "frozen24_cells.json"
)
METHODS_PATH = ROOT / "configs" / "reconstruction_benchmark_v1" / "methods.json"

EXPECTED_PUBLISHED = {
    "ars_gsm8k_r1distill8b": ("ARS (CCS)", 0.7472, "ars_2601_17467", "Table 1"),
    "epr_triviaqa_mistral24b": ("EPR", 0.746, "epr_2509_04492", "Table 1 (K=15)"),
    "internalstates_gsm8k_qwen25_7b": ("Internal-States+RC", 0.7915, "internal_states_2510_11529", "Table 1"),
    "lapeigvals_gsm8k_llama3b": ("AttentionScore", 0.717, "attention_spectral_2502_17598", "Table 1"),
    "lapeigvals_gsm8k_llama8b": ("AttentionScore", 0.720, "attention_spectral_2502_17598", "Table 1"),
    "lapeigvals_gsm8k_mistral24b": ("AttentionScore", 0.576, "attention_spectral_2502_17598", "Table 1"),
    "lapeigvals_gsm8k_nemo": ("AttentionScore", 0.630, "attention_spectral_2502_17598", "Table 1"),
    "lapeigvals_gsm8k_phi35": ("AttentionScore", 0.666, "attention_spectral_2502_17598", "Table 1"),
    "losnet_hotpotqa_mistral7b": ("LOS-Net", 0.7292, "losnet_2503_14043", "Table 1"),
    "math500_r1distill8b": ("ARS (CCS)", 0.8638, "ars_2601_17467", "Table 1"),
    "math500_r1distill8b_mn4096": ("ARS (CCS)", 0.8638, "ars_2601_17467", "Table 1"),
    "noise_gsm8k_mistral7b": ("Noise Injection", 0.7850, "noise_injection_2502_03799", "Table 3"),
    "noise_gsm8k_phi3mini": ("Noise Injection", 0.7251, "noise_injection_2502_03799", "Table 3"),
    "sciq_llama8b": ("HCPD", 0.8604, "hcpd_2606_12900", "Table 2"),
    "se_nq_open_llama8b": ("HCPD", 0.9038, "hcpd_2606_12900", "Table 2"),
    "se_squad_v2_llama8b": ("Lexical Similarity", 0.5988, "als_2605_26366", "Table 2"),
    "seiclr_triviaqa_opt30b": ("Semantic Entropy", 0.830, "semantic_uncertainty_2302_09664", "Table 2"),
    "semenergy_triviaqa_qwen3_8b": ("Semantic Energy", 0.748, "semantic_energy_2508_14496", "Table 1"),
    "spilled_triviaqa_llama8b": ("HCPD", 0.8625, "hcpd_2606_12900", "Table 2"),
    "trace_gsm8k_llama8b_k10": ("AttentionScore", 0.720, "attention_spectral_2502_17598", "Table 1"),
    "truthfulqa_llama8b": ("TSV", 0.842, "tsv_2503_01917", "Table 1"),
}

EXPECTED_V2_LEADERS = {
    "ars_gsm8k_r1distill8b": ("equal_family_mean", 0.781472, 0.735666, 0.824666),
    "epr_triviaqa_mistral24b": ("upcr", 0.745750, 0.701458, 0.789180),
    "internalstates_gsm8k_qwen25_7b": ("equal_family_mean", 0.727872, 0.679618, 0.774822),
    "lapeigvals_gsm8k_llama3b": ("dufs_pf_lsml", 0.708735, 0.681072, 0.735991),
    "lapeigvals_gsm8k_llama8b": ("family_nrm_a", 0.819801, 0.774949, 0.861660),
    "lapeigvals_gsm8k_mistral24b": ("deem_b3", 0.852908, 0.811338, 0.891088),
    "lapeigvals_gsm8k_nemo": ("equal_family_mean", 0.811270, 0.777967, 0.842872),
    "lapeigvals_gsm8k_phi35": ("deem_b3", 0.816713, 0.784804, 0.847559),
    "losnet_hotpotqa_mistral7b": ("equal_family_mean", 0.582009, 0.528093, 0.635016),
    "math500_dsmath7b": ("deem_b3", 0.812938, 0.751984, 0.868105),
    "math500_qwenmath7b": ("iu_pcr", 0.930781, 0.890094, 0.965190),
    "math500_r1distill8b": ("continuous_lsml", 0.845794, 0.795077, 0.893447),
    "math500_r1distill8b_mn4096": ("ca_specrage_atomic", 0.837654, 0.789910, 0.882150),
    "noise_gsm8k_mistral7b": ("equal_family_mean", 0.793635, 0.769077, 0.817619),
    "noise_gsm8k_phi3mini": ("upcr", 0.683409, 0.650844, 0.714865),
    "sciq_llama8b": ("pgrd_a", 0.746269, 0.673489, 0.814063),
    "se_nq_open_llama8b": ("dufs_pf_lsml", 0.754783, 0.729952, 0.778606),
    "se_squad_v2_llama8b": ("pgrd_a", 0.826250, 0.785336, 0.862560),
    "seiclr_triviaqa_opt30b": ("ca_specrage_atomic", 0.830063, 0.799716, 0.858488),
    "semenergy_triviaqa_qwen3_8b": ("upcr", 0.813097, 0.776296, 0.847669),
    "spilled_triviaqa_llama8b": ("equal_feature_mean", 0.951333, 0.883927, 0.996032),
    "trace_gsm8k_llama8b_k10": ("ca_specrage_atomic", 0.815029, 0.791082, 0.837429),
    "trace_math500_qwenmath15b_k10": ("equal_family_mean", 0.701489, 0.656131, 0.743460),
    "truthfulqa_llama8b": ("pgrd_a", 0.676009, 0.634304, 0.715207),
}


class Frozen24PublishedComparatorRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads(REGISTRY_PATH.read_text())
        cls.frozen_cells = json.loads(FROZEN_CELLS_PATH.read_text())
        cls.methods = json.loads(METHODS_PATH.read_text())
        cls.rows = cls.registry["cells"]
        cls.by_cell = {row["cell_id"]: row for row in cls.rows}
        cls.primary_method_ids = {
            method["method_id"] for method in cls.methods["methods"]
        }

    def test_exactly_one_row_for_each_frozen_cell(self):
        expected = {row["cell_id"] for row in self.frozen_cells["cells"]}
        observed = [row["cell_id"] for row in self.rows]
        self.assertEqual(len(observed), 24)
        self.assertEqual(len(set(observed)), 24)
        self.assertEqual(set(observed), expected)

    def test_status_counts_and_no_paper_delta(self):
        counts = {}
        for row in self.rows:
            counts[row["comparison_status"]] = (
                counts.get(row["comparison_status"], 0) + 1
            )
            self.assertFalse(row["delta_eligible"])
            self.assertNotIn("delta", row)
            expected_replay_status = (
                "NOT_APPLICABLE_NO_COMPARATOR"
                if row["comparison_status"] == "NO_PUBLISHED_COMPARATOR"
                else "BLOCKED_PENDING_COMMON_ROW_RERUN"
            )
            self.assertEqual(row["common_replay_status"], expected_replay_status)
            self.assertTrue(row["mismatch_reasons"])
        self.assertEqual(
            counts,
            {
                "PUBLISHED_CONTEXT_ONLY": 17,
                "RELATED_PUBLISHED_CONTEXT_ONLY": 4,
                "NO_PUBLISHED_COMPARATOR": 3,
            },
        )
        report_contract = self.registry["report_contract"]
        self.assertIn("13 v2 methods", report_contract["direct_panel"])
        self.assertIn("separate", report_contract["published_context_panel"])
        self.assertIn("null delta", report_contract["export_rule"])
        self.assertIn("visible legend", report_contract["legend_rule"].lower())

    def test_every_v2_leader_is_a_primary_method(self):
        self.assertEqual(len(self.primary_method_ids), 13)
        self.assertEqual(set(self.by_cell), set(EXPECTED_V2_LEADERS))
        for row in self.rows:
            leader = row["v2_point_estimate_leader"]
            self.assertIn(leader["method_id"], self.primary_method_ids)
            self.assertLessEqual(0.0, leader["auroc"])
            self.assertLessEqual(leader["auroc"], 1.0)
            lo, hi = leader["ci95"]
            self.assertLessEqual(lo, leader["auroc"])
            self.assertLessEqual(leader["auroc"], hi)
            expected = EXPECTED_V2_LEADERS[row["cell_id"]]
            self.assertEqual(leader["method_id"], expected[0])
            self.assertAlmostEqual(leader["auroc"], expected[1], places=6)
            self.assertAlmostEqual(lo, expected[2], places=6)
            self.assertAlmostEqual(hi, expected[3], places=6)

    def test_published_values_resolve_to_verified_primary_sources(self):
        sources = self.registry["source_catalog"]
        required_axes = {
            "dataset_revision",
            "model",
            "row_ids",
            "generation",
            "labels_grader",
            "prediction_unit",
            "metric",
            "evaluation_protocol",
        }
        for row in self.rows:
            comparator = row["published_comparator"]
            self.assertEqual(set(row["match_axes"]), required_axes)
            if row["comparison_status"] == "NO_PUBLISHED_COMPARATOR":
                self.assertIsNone(comparator["method"])
                self.assertIsNone(comparator["auroc"])
                self.assertIsNone(comparator["source_id"])
                self.assertEqual(
                    set(row["match_axes"].values()), {"NOT_APPLICABLE"}
                )
                continue
            self.assertIsInstance(comparator["auroc"], (int, float))
            self.assertIn(comparator["source_id"], sources)
            self.assertIn(
                comparator["mapping_status"],
                {
                    "VERIFIED_PRIMARY_SOURCE",
                    "VERIFIED_PRIMARY_SOURCE_RELATED_CONTEXT",
                    "VERIFIED_PRIMARY_SOURCE_TABLE_CORRECTED",
                    "CORRECTED_PRIMARY_SOURCE",
                },
            )
            self.assertNotEqual(row["match_axes"]["row_ids"], "EXACT")
            source = sources[comparator["source_id"]]
            self.assertTrue(source["url"].startswith("https://arxiv.org/abs/"))
            for local_source in source["local_sources"]:
                self.assertTrue((ROOT / local_source).is_file(), local_source)

        self.assertEqual(
            {
                row["cell_id"]
                for row in self.rows
                if row["published_comparator"]["auroc"] is not None
            },
            set(EXPECTED_PUBLISHED),
        )
        for cell_id, expected in EXPECTED_PUBLISHED.items():
            comparator = self.by_cell[cell_id]["published_comparator"]
            observed = (
                comparator["method"],
                comparator["auroc"],
                comparator["source_id"],
                comparator["table"],
            )
            self.assertEqual(observed, expected, cell_id)

    def test_semantic_entropy_and_tsv_use_correct_primary_papers(self):
        semantic_entropy = self.by_cell["seiclr_triviaqa_opt30b"]
        se_comparator = semantic_entropy["published_comparator"]
        self.assertEqual(
            se_comparator["source_id"], "semantic_uncertainty_2302_09664"
        )
        self.assertEqual(se_comparator["table"], "Table 2")
        self.assertAlmostEqual(se_comparator["auroc"], 0.830)
        self.assertIn(
            "2302.09664",
            self.registry["source_catalog"][se_comparator["source_id"]]["url"],
        )
        se_evidence = self.registry["source_catalog"][se_comparator["source_id"]][
            "primary_evidence"
        ]
        self.assertIn("OPT-30B TriviaQA", se_evidence["locator"])
        self.assertIn("0.83", se_evidence["locator"])

        tsv = self.by_cell["truthfulqa_llama8b"]
        tsv_comparator = tsv["published_comparator"]
        self.assertEqual(tsv_comparator["source_id"], "tsv_2503_01917")
        self.assertEqual(tsv_comparator["table"], "Table 1")
        self.assertAlmostEqual(tsv_comparator["auroc"], 0.842)
        self.assertIn("32 labeled examples", tsv_comparator["supervision"])
        self.assertIn(
            "2503.01917",
            self.registry["source_catalog"][tsv_comparator["source_id"]]["url"],
        )
        tsv_evidence = self.registry["source_catalog"][tsv_comparator["source_id"]][
            "primary_evidence"
        ]
        self.assertIn("TruthfulQA", tsv_evidence["locator"])
        self.assertIn("32 labeled examples", tsv_evidence["locator"])

    def test_supervision_and_label_axes_are_not_overclaimed(self):
        for cell_id in (
            "sciq_llama8b",
            "se_nq_open_llama8b",
            "spilled_triviaqa_llama8b",
        ):
            comparator = self.by_cell[cell_id]["published_comparator"]
            self.assertIn("target-dataset proxy-label supervised", comparator["supervision"])
            self.assertIn("no manual hallucination labels", comparator["supervision"])
            self.assertIn("GRPO", comparator["supervision"])
            self.assertEqual(
                comparator["passes"], "K=5 scoring-agent samples per target response"
            )

        for cell_id in ("noise_gsm8k_mistral7b", "noise_gsm8k_phi3mini"):
            supervision = self.by_cell[cell_id]["published_comparator"]["supervision"]
            self.assertIn("label-informed validation selection", supervision)

        self.assertEqual(
            self.by_cell["ars_gsm8k_r1distill8b"]["match_axes"][
                "dataset_revision"
            ],
            "PARTIAL",
        )

        self.assertEqual(
            self.by_cell["seiclr_triviaqa_opt30b"]["match_axes"]["labels_grader"],
            "EXACT",
        )
        for cell_id in (
            "ars_gsm8k_r1distill8b",
            "math500_r1distill8b",
            "math500_r1distill8b_mn4096",
        ):
            self.assertEqual(
                self.by_cell[cell_id]["match_axes"]["labels_grader"],
                "DIFFERENT",
            )

        trace_context = self.by_cell["trace_gsm8k_llama8b_k10"]
        self.assertEqual(
            trace_context["comparison_status"],
            "RELATED_PUBLISHED_CONTEXT_ONLY",
        )
        self.assertEqual(trace_context["match_axes"]["prediction_unit"], "DIFFERENT")

    def test_noise_injection_values_point_to_table_three(self):
        expected = {
            "noise_gsm8k_mistral7b": 0.7850,
            "noise_gsm8k_phi3mini": 0.7251,
        }
        for cell_id, auroc in expected.items():
            comparator = self.by_cell[cell_id]["published_comparator"]
            self.assertEqual(comparator["source_id"], "noise_injection_2502_03799")
            self.assertEqual(comparator["table"], "Table 3")
            self.assertAlmostEqual(comparator["auroc"], auroc)

    def test_v2_point_estimate_leaders_match_pinned_atomic_metrics_if_present(self):
        source = ROOT / self.registry["v2_metrics_source"]["path"]
        if not source.is_file():
            self.skipTest("Pinned reporting-v2 metrics are not materialized in this checkout")

        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(
            actual_sha256,
            self.registry["v2_metrics_source"]["sha256"],
        )
        by_cell = {}
        with source.open(newline="") as handle:
            for metric in csv.DictReader(handle):
                cell_id = metric["cell_id"]
                if cell_id not in self.by_cell:
                    continue
                if metric["metric_id"] != "auroc":
                    continue
                if metric["method_id"] not in self.primary_method_ids:
                    continue
                by_cell.setdefault(cell_id, []).append(metric)

        self.assertEqual(set(by_cell), set(self.by_cell))
        for cell_id, metrics in by_cell.items():
            self.assertEqual(len(metrics), 13, cell_id)
            observed = max(metrics, key=lambda metric: float(metric["value"]))
            expected = self.by_cell[cell_id]["v2_point_estimate_leader"]
            self.assertEqual(observed["method_id"], expected["method_id"])
            self.assertAlmostEqual(float(observed["value"]), expected["auroc"], places=6)
            self.assertAlmostEqual(float(observed["ci_low"]), expected["ci95"][0], places=6)
            self.assertAlmostEqual(float(observed["ci_high"]), expected["ci95"][1], places=6)


if __name__ == "__main__":
    unittest.main()
