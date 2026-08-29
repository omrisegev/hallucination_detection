#!/usr/bin/env python3
"""Artifact-contract tests for completed Reasoning Localization Phase 0 states."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s0_historical_replay"
S1_DIR = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s1_reducer_bridge"
S1_REGISTRY = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "P0_S1_EXECUTION_REGISTRY.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Phase0HistoricalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((STATE_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        cls.verification = json.loads((STATE_DIR / "P0_S0_VERIFICATION.json").read_text(encoding="utf-8"))
        cls.population = json.loads((STATE_DIR / "P0_S0_POPULATION.json").read_text(encoding="utf-8"))

    def test_replay_is_checksum_equivalent_without_new_inference(self) -> None:
        self.assertEqual("COMPLETE", self.manifest["status"])
        self.assertFalse(self.manifest["new_inference"])
        self.assertFalse(self.manifest["source_mutation"])
        self.assertEqual(0, self.manifest["gpu_hours"])
        self.assertEqual("CHECKSUM_EQUIVALENT", self.verification["status"])
        self.assertTrue(all(self.verification["checks"][key] for key in (
            "per_question_byte_exact",
            "cell_metrics_semantic_exact",
            "aggregate_semantic_exact",
            "intervals_semantic_exact",
        )))

    def test_manifest_binds_every_output(self) -> None:
        for row in self.manifest["outputs"]:
            path = STATE_DIR / row["file"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], sha256_file(path))

    def test_population_identity_and_grouping_are_frozen(self) -> None:
        self.assertEqual(8, self.population["n_cells"])
        self.assertEqual(1270, self.population["n_scorer_rows"])
        self.assertEqual(635, self.population["n_source_question_groups"])
        self.assertTrue(self.population["scorer_copies_grouped"])
        self.assertEqual(self.population["source_question_group_sha256"], self.verification["population_sha256"])

    def test_registered_metric_is_the_exact_historical_anchor(self) -> None:
        with (STATE_DIR / "P0_S0_METRICS.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(1, len(rows))
        self.assertEqual("R2_HISTORICAL_FAMILY6_BRIDGE", rows[0]["variant_id"])
        self.assertEqual("0.3662328341717007", rows[0]["value"])
        self.assertEqual("RETROSPECTIVE", rows[0]["evidence_status"])


class Phase0ReducerBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((S1_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        cls.verification = json.loads((S1_DIR / "P0_S1_VERIFICATION.json").read_text(encoding="utf-8"))
        cls.population = json.loads((S1_DIR / "P0_S1_POPULATION.json").read_text(encoding="utf-8"))
        cls.registry = json.loads(S1_REGISTRY.read_text(encoding="utf-8"))

    def test_s1_reconstructs_s0_and_changes_only_the_reducer(self) -> None:
        self.assertEqual("COMPLETE", self.manifest["status"])
        self.assertTrue(self.verification["s0_reconstruction_exact"])
        self.assertEqual(
            {"factor": "step_reducer", "from": "step_top5mean", "to": "step_max_token_argmax"},
            self.verification["single_changed_factor"],
        )
        self.assertFalse(self.manifest["new_inference"])
        self.assertFalse(self.manifest["source_mutation"])
        self.assertEqual(0, self.manifest["gpu_hours"])
        self.assertEqual(20000, self.manifest["bootstrap_draws"])
        self.assertEqual(2026082901, self.manifest["bootstrap_seed"])

    def test_s1_manifest_binds_every_emitted_output(self) -> None:
        for row in self.manifest["outputs"]:
            path = S1_DIR / row["file"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], sha256_file(path))
        self.assertEqual(self.manifest["runner_sha256"], self.registry["runner_sha256"])
        self.assertEqual(self.manifest["execution_registry_sha256"], sha256_file(S1_REGISTRY))

    def test_s1_population_is_identical_to_s0(self) -> None:
        self.assertEqual(8, self.population["n_cells"])
        self.assertEqual(1270, self.population["n_scorer_rows"])
        self.assertEqual(635, self.population["n_source_question_groups"])
        self.assertEqual(
            "d12d651cad9bec326686c2c83070644d22ca058ed57e942f683452050e757a05",
            self.population["source_question_group_sha256"],
        )

    def test_s1_macro_result_and_paired_interval_are_frozen(self) -> None:
        aggregate = {
            row["metric_id"]: row
            for row in self.verification["aggregate"]
        }
        self.assertEqual(0.33007771561392063, aggregate["macro_f1"]["value"])
        contrast = next(
            row for row in self.verification["contrasts"]
            if row["metric_id"] == "macro_f1"
        )
        self.assertEqual(-0.03615511855778009, contrast["delta"])
        self.assertLess(contrast["ci_high"], 0.0)
        self.assertEqual((0, 0, 4), (contrast["wins"], contrast["ties"], contrast["losses"]))

    def test_s1_flip_audit_covers_every_scorer_row(self) -> None:
        self.assertEqual(1270, sum(self.verification["prediction_flip_counts"].values()))
        self.assertEqual(978, self.verification["prediction_flip_counts"]["NO_FLIP"])


if __name__ == "__main__":
    unittest.main()
