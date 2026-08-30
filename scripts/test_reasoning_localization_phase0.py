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
S2A_REGISTRY = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "P0_S2A_EXECUTION_REGISTRY.json"
S2A_RUNNER = REPO / "scripts" / "reasoning_localization" / "run_phase0_detector_bridge.py"
S2A_DIR = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s2a_detector_bridge"
S2I_REGISTRY = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "P0_S2I_EXECUTION_REGISTRY.json"
S2I_RUNNER = REPO / "scripts" / "reasoning_localization" / "run_phase0_interaction_control.py"
S2I_DIR = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s2i_interaction_control"
S2B_REGISTRY = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "P0_S2B_EXECUTION_REGISTRY.json"
S2B_RUNNER = REPO / "scripts" / "reasoning_localization" / "run_phase0_pure_local_detector_bridge.py"
S2B_DIR = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s2b_pure_local_detector_bridge"
S3A_REGISTRY = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "P0_S3A_EXECUTION_REGISTRY.json"
S3A_RUNNER = REPO / "scripts" / "reasoning_localization" / "run_phase0_raw_entropy_representation_bridge.py"
S3A_DIR = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s3a_raw_entropy_representation_bridge"
S3B_REGISTRY = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "P0_S3B_EXECUTION_REGISTRY.json"
S3B_RUNNER = REPO / "scripts" / "reasoning_localization" / "run_phase0_iu29_representation_bridge.py"
S3B_DIR = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s3b_iu29_representation_bridge"
S4_REGISTRY = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "P0_S4_EXECUTION_REGISTRY.json"
S4_RUNNER = REPO / "scripts" / "reasoning_localization" / "run_phase0_split_bridge.py"
S4_DIR = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s4_fivefold_split_bridge"
S5_REGISTRY = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "P0_S5_EXECUTION_REGISTRY.json"
S5_RUNNER = REPO / "scripts" / "reasoning_localization" / "run_phase0_population_bridges.py"
S5_DIR = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s5_population_bridges"


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


class Phase0DetectorBridgeFreezeTests(unittest.TestCase):
    def test_s2a_registry_freezes_one_detector_only(self) -> None:
        registry = json.loads(S2A_REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual("FROZEN_BEFORE_RUN", registry["status"])
        self.assertEqual(
            {
                "factor": "answer_detector",
                "from": "RegisteredGlobal mixed-v2 ordinary IU",
                "to": "calibration-only answer_dufs_liu_mixed",
            },
            registry["single_changed_factor"],
        )
        self.assertEqual("step_max_token_argmax", registry["fixed_factors"]["step_reducer"])
        self.assertEqual("family6", registry["fixed_factors"]["representation"])
        self.assertEqual(7, registry["modern_detector"]["k"])
        self.assertEqual(0.1, registry["modern_detector"]["lambda"])
        self.assertEqual([11, 23, 37], registry["modern_detector"]["dufs_seeds"])
        self.assertEqual(80, registry["modern_detector"]["dufs_epochs"])
        self.assertFalse(registry["modern_detector"]["labels_seen_during_fit"])
        self.assertEqual(registry["runner_sha256"], sha256_file(S2A_RUNNER))


class Phase0DetectorBridgeResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((S2A_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        cls.verification = json.loads((S2A_DIR / "P0_S2A_VERIFICATION.json").read_text(encoding="utf-8"))
        cls.population = json.loads((S2A_DIR / "P0_S2A_POPULATION.json").read_text(encoding="utf-8"))

    def test_s2a_reconstructs_s1_and_changes_only_the_detector(self) -> None:
        self.assertEqual("COMPLETE", self.manifest["status"])
        self.assertTrue(self.verification["s1_reconstruction_exact"])
        self.assertEqual(
            {
                "factor": "answer_detector",
                "from": "RegisteredGlobal mixed-v2 ordinary IU",
                "to": "calibration-only answer_dufs_liu_mixed",
            },
            self.verification["single_changed_factor"],
        )
        self.assertFalse(self.verification["new_model_inference"])
        self.assertFalse(self.verification["source_mutation"])
        self.assertFalse(self.verification["s2b_opened"])
        self.assertTrue(self.verification["label_free_method_fit"])

    def test_s2a_manifest_binds_every_emitted_output(self) -> None:
        for row in self.manifest["outputs"]:
            path = S2A_DIR / row["file"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], sha256_file(path))
        self.assertEqual(self.manifest["runner_sha256"], sha256_file(S2A_RUNNER))
        self.assertEqual(self.manifest["execution_registry_sha256"], sha256_file(S2A_REGISTRY))

    def test_s2a_population_is_identical_to_s1(self) -> None:
        self.assertEqual(8, self.population["n_cells"])
        self.assertEqual(1270, self.population["n_scorer_rows"])
        self.assertEqual(635, self.population["n_source_question_groups"])
        self.assertEqual(
            "d12d651cad9bec326686c2c83070644d22ca058ed57e942f683452050e757a05",
            self.population["source_question_group_sha256"],
        )

    def test_s2a_macro_is_unsupported_and_operating_tradeoff_is_explicit(self) -> None:
        aggregate = {row["metric_id"]: row for row in self.verification["aggregate"]}
        self.assertEqual(0.32859546976358334, aggregate["macro_f1"]["value"])
        contrasts = {row["metric_id"]: row for row in self.verification["contrasts"]}
        macro = contrasts["macro_f1"]
        self.assertEqual(-0.001482245850337259, macro["delta"])
        self.assertLess(macro["ci_low"], 0.0)
        self.assertGreater(macro["ci_high"], 0.0)
        self.assertLess(contrasts["clean_abstention"]["ci_high"], 0.0)
        self.assertGreater(contrasts["within_one"]["ci_low"], 0.0)

    def test_s2a_flip_audit_covers_every_scorer_row(self) -> None:
        self.assertEqual(1270, sum(self.verification["prediction_flip_counts"].values()))
        self.assertEqual(1224, self.verification["prediction_flip_counts"]["NO_FLIP"])


class Phase0InteractionControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(S2I_REGISTRY.read_text(encoding="utf-8"))
        cls.manifest = json.loads((S2I_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        cls.verification = json.loads((S2I_DIR / "P0_S2I_VERIFICATION.json").read_text(encoding="utf-8"))
        cls.population = json.loads((S2I_DIR / "P0_S2I_POPULATION.json").read_text(encoding="utf-8"))

    def test_s2i_is_frozen_as_one_reducer_only_and_keeps_s2b_closed(self) -> None:
        self.assertEqual("FROZEN_BEFORE_RUN", self.registry["status"])
        self.assertEqual(
            {
                "factor": "step_reducer",
                "from": "step_max_token_argmax",
                "to": "step_top5mean",
            },
            self.verification["single_changed_factor"],
        )
        self.assertTrue(self.verification["s2a_reconstruction_exact"])
        self.assertFalse(self.verification["s2b_opened"])
        self.assertFalse(self.verification["new_model_inference"])
        self.assertFalse(self.verification["source_mutation"])
        self.assertTrue(self.verification["label_free_method_fit"])

    def test_s2i_manifest_binds_every_output_and_frozen_runner(self) -> None:
        for row in self.manifest["outputs"]:
            path = S2I_DIR / row["file"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], sha256_file(path))
        self.assertEqual(self.registry["runner_sha256"], sha256_file(S2I_RUNNER))
        self.assertEqual(self.manifest["runner_sha256"], sha256_file(S2I_RUNNER))
        self.assertEqual(self.manifest["execution_registry_sha256"], sha256_file(S2I_REGISTRY))

    def test_s2i_population_and_existing_edges_are_exact(self) -> None:
        self.assertEqual(8, self.population["n_cells"])
        self.assertEqual(1270, self.population["n_scorer_rows"])
        self.assertEqual(635, self.population["n_source_question_groups"])
        self.assertEqual(
            "d12d651cad9bec326686c2c83070644d22ca058ed57e942f683452050e757a05",
            self.population["source_question_group_sha256"],
        )
        self.assertLessEqual(self.verification["existing_edge_max_abs_delta"], 1e-12)

    def test_s2i_separates_adjacent_anchor_and_interaction_estimands(self) -> None:
        aggregate = {row["metric_id"]: row for row in self.verification["aggregate"]}
        self.assertEqual(0.3632846791052713, aggregate["macro_f1"]["value"])
        effects = {
            row["effect_id"]: row for row in self.verification["factorial_effects"]
            if row["metric_id"] == "macro_f1"
        }
        adjacent = effects["P0_S2I_VS_S2A_ADJACENT_POOLING"]
        self.assertEqual(0.03468920934168793, adjacent["delta"])
        self.assertGreater(adjacent["ci_low"], 0.0)
        self.assertEqual((4, 0, 0), (adjacent["wins"], adjacent["ties"], adjacent["losses"]))
        anchor = effects["P0_S2I_VS_S0_DETECTOR_TOP5"]
        self.assertLess(anchor["ci_low"], 0.0)
        self.assertGreater(anchor["ci_high"], 0.0)
        interaction = effects["P0_REDUCER_X_DETECTOR_INTERACTION"]
        self.assertLess(interaction["ci_low"], 0.0)
        self.assertGreater(interaction["ci_high"], 0.0)

    def test_s2i_flip_audit_covers_every_scorer_row(self) -> None:
        self.assertEqual(1270, sum(self.verification["prediction_flip_counts"].values()))
        self.assertEqual(974, self.verification["prediction_flip_counts"]["NO_FLIP"])


class Phase0PureLocalDetectorBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(S2B_REGISTRY.read_text(encoding="utf-8"))
        cls.manifest = json.loads((S2B_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        cls.verification = json.loads((S2B_DIR / "P0_S2B_VERIFICATION.json").read_text(encoding="utf-8"))
        cls.population = json.loads((S2B_DIR / "P0_S2B_POPULATION.json").read_text(encoding="utf-8"))

    def test_s2b_is_frozen_as_one_detector_only(self) -> None:
        self.assertEqual("FROZEN_BEFORE_RUN", self.registry["status"])
        self.assertEqual(
            {
                "factor": "answer_detector",
                "from": "calibration-only answer_dufs_liu_mixed",
                "to": "maximum of the fitted family6 level local-risk curve",
            },
            self.verification["single_changed_factor"],
        )
        self.assertTrue(self.verification["s2a_reconstruction_exact"])
        self.assertTrue(self.verification["locator_unchanged"])
        self.assertFalse(self.verification["new_model_inference"])
        self.assertFalse(self.verification["source_mutation"])
        self.assertTrue(self.verification["label_free_method_fit"])
        self.assertFalse(self.verification["later_bridge_opened"])

    def test_s2b_manifest_binds_every_output_and_frozen_runner(self) -> None:
        for row in self.manifest["outputs"]:
            path = S2B_DIR / row["file"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], sha256_file(path))
        self.assertEqual(self.registry["runner_sha256"], sha256_file(S2B_RUNNER))
        self.assertEqual(self.manifest["runner_sha256"], sha256_file(S2B_RUNNER))
        self.assertEqual(self.manifest["execution_registry_sha256"], sha256_file(S2B_REGISTRY))

    def test_s2b_population_and_supported_loss_are_frozen(self) -> None:
        self.assertEqual(8, self.population["n_cells"])
        self.assertEqual(1270, self.population["n_scorer_rows"])
        self.assertEqual(635, self.population["n_source_question_groups"])
        aggregate = {row["metric_id"]: row for row in self.verification["aggregate"]}
        self.assertEqual(0.3065027012935364, aggregate["macro_f1"]["value"])
        contrasts = {row["metric_id"]: row for row in self.verification["contrasts"]}
        macro = contrasts["macro_f1"]
        self.assertEqual(-0.0220927684700469, macro["delta"])
        self.assertLess(macro["ci_high"], 0.0)
        self.assertEqual((0, 0, 4), (macro["wins"], macro["ties"], macro["losses"]))
        self.assertLess(contrasts["exact_error"]["ci_high"], 0.0)
        self.assertLess(contrasts["clean_abstention"]["ci_low"], 0.0)
        self.assertGreater(contrasts["clean_abstention"]["ci_high"], 0.0)

    def test_s2b_flip_audit_covers_every_scorer_row(self) -> None:
        self.assertEqual(1270, sum(self.verification["prediction_flip_counts"].values()))
        self.assertEqual(1012, self.verification["prediction_flip_counts"]["NO_FLIP"])


class Phase0RawEntropyRepresentationBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(S3A_REGISTRY.read_text(encoding="utf-8"))
        cls.manifest = json.loads((S3A_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        cls.verification = json.loads((S3A_DIR / "P0_S3A_VERIFICATION.json").read_text(encoding="utf-8"))
        cls.population = json.loads((S3A_DIR / "P0_S3A_POPULATION.json").read_text(encoding="utf-8"))

    def test_s3a_is_frozen_as_one_representation_only(self) -> None:
        self.assertEqual("FROZEN_BEFORE_RUN", self.registry["status"])
        self.assertEqual(
            {
                "factor": "local_risk_representation",
                "from": "fitted family6 level risk curve",
                "to": "raw token entropy",
            },
            self.verification["single_changed_factor"],
        )
        self.assertTrue(self.verification["s2b_reconstruction_exact"])
        self.assertTrue(all(row["all_curves_finite"] for row in self.verification["representation_audits"]))
        self.assertTrue(all(not row["labels_seen_during_representation_construction"] for row in self.verification["representation_audits"]))
        self.assertFalse(self.verification["new_model_inference"])
        self.assertFalse(self.verification["source_mutation"])
        self.assertTrue(self.verification["label_free_method_fit"])
        self.assertFalse(self.verification["iu29_bridge_opened"])

    def test_s3a_manifest_binds_every_output_and_frozen_runner(self) -> None:
        for row in self.manifest["outputs"]:
            path = S3A_DIR / row["file"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], sha256_file(path))
        self.assertEqual(self.registry["runner_sha256"], sha256_file(S3A_RUNNER))
        self.assertEqual(self.manifest["runner_sha256"], sha256_file(S3A_RUNNER))
        self.assertEqual(self.manifest["execution_registry_sha256"], sha256_file(S3A_REGISTRY))

    def test_s3a_population_and_unsupported_point_gain_are_frozen(self) -> None:
        self.assertEqual(8, self.population["n_cells"])
        self.assertEqual(1270, self.population["n_scorer_rows"])
        self.assertEqual(635, self.population["n_source_question_groups"])
        aggregate = {row["metric_id"]: row for row in self.verification["aggregate"]}
        self.assertEqual(0.3110940034934562, aggregate["macro_f1"]["value"])
        contrasts = {row["metric_id"]: row for row in self.verification["contrasts"]}
        macro = contrasts["macro_f1"]
        self.assertEqual(0.004591302199919791, macro["delta"])
        self.assertLess(macro["ci_low"], 0.0)
        self.assertGreater(macro["ci_high"], 0.0)
        self.assertEqual((2, 0, 2), (macro["wins"], macro["ties"], macro["losses"]))
        self.assertLess(contrasts["within_one"]["ci_low"], 0.0)
        self.assertGreater(contrasts["within_one"]["ci_high"], 0.0)
        self.assertEqual(559, self.verification["locator_change_count"])

    def test_s3a_flip_audit_covers_every_scorer_row(self) -> None:
        self.assertEqual(1270, sum(self.verification["prediction_flip_counts"].values()))
        self.assertEqual(712, self.verification["prediction_flip_counts"]["NO_FLIP"])


class Phase0IU29RepresentationBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(S3B_REGISTRY.read_text(encoding="utf-8"))
        cls.manifest = json.loads((S3B_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        cls.verification = json.loads((S3B_DIR / "P0_S3B_VERIFICATION.json").read_text(encoding="utf-8"))
        cls.population = json.loads((S3B_DIR / "P0_S3B_POPULATION.json").read_text(encoding="utf-8"))

    def test_s3b_is_frozen_as_one_representation_only(self) -> None:
        self.assertEqual("FROZEN_BEFORE_RUN", self.registry["status"])
        self.assertEqual(
            {
                "factor": "local_risk_representation",
                "from": "raw token entropy",
                "to": "calibration-only LOCAL_IU29 token risk",
            },
            self.verification["single_changed_factor"],
        )
        self.assertTrue(self.verification["s3a_reconstruction_exact"])
        self.assertTrue(self.verification["calibration_rows_only"])
        self.assertTrue(self.verification["label_free_method_fit"])
        self.assertFalse(self.verification["new_model_inference"])
        self.assertFalse(self.verification["source_mutation"])
        self.assertFalse(self.verification["split_bridge_opened"])
        audits = self.verification["representation_audits"]
        self.assertEqual(8, len(audits))
        self.assertTrue(all(row["n_input_streams"] == 29 for row in audits))
        self.assertTrue(all(row["n_kept_streams"] == 29 for row in audits))
        self.assertTrue(all(row["all_curves_finite"] for row in audits))
        self.assertTrue(all(row["deterministic_within_1e_12"] for row in audits))
        self.assertTrue(all(row["score_reconstruction_within_1e_12"] for row in audits))
        self.assertTrue(all(not row["labels_seen_during_representation_fit"] for row in audits))

    def test_s3b_manifest_binds_every_output_and_frozen_runner(self) -> None:
        for row in self.manifest["outputs"]:
            path = S3B_DIR / row["file"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], sha256_file(path))
        self.assertEqual(self.registry["runner_sha256"], sha256_file(S3B_RUNNER))
        self.assertEqual(self.manifest["runner_sha256"], sha256_file(S3B_RUNNER))
        self.assertEqual(self.manifest["execution_registry_sha256"], sha256_file(S3B_REGISTRY))

    def test_s3b_population_and_inconclusive_point_loss_are_frozen(self) -> None:
        self.assertEqual(8, self.population["n_cells"])
        self.assertEqual(1270, self.population["n_scorer_rows"])
        self.assertEqual(635, self.population["n_source_question_groups"])
        self.assertEqual(
            "d12d651cad9bec326686c2c83070644d22ca058ed57e942f683452050e757a05",
            self.population["source_question_group_sha256"],
        )
        aggregate = {row["metric_id"]: row for row in self.verification["aggregate"]}
        self.assertEqual(0.2996587594711835, aggregate["macro_f1"]["value"])
        contrasts = {row["metric_id"]: row for row in self.verification["contrasts"]}
        macro = contrasts["macro_f1"]
        self.assertEqual(-0.011435244022272775, macro["delta"])
        self.assertLess(macro["ci_low"], 0.0)
        self.assertGreater(macro["ci_high"], 0.0)
        self.assertEqual((2, 0, 2), (macro["wins"], macro["ties"], macro["losses"]))
        self.assertEqual(-0.20003988831272437, macro["worst_unit_delta"])
        self.assertEqual(501, self.verification["locator_change_count"])

    def test_s3b_flip_audit_covers_every_scorer_row(self) -> None:
        self.assertEqual(1270, sum(self.verification["prediction_flip_counts"].values()))
        self.assertEqual(792, self.verification["prediction_flip_counts"]["NO_FLIP"])


class Phase0SplitBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(S4_REGISTRY.read_text(encoding="utf-8"))
        cls.manifest = json.loads((S4_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        cls.verification = json.loads((S4_DIR / "P0_S4_VERIFICATION.json").read_text(encoding="utf-8"))

    def test_s4_is_frozen_and_manifest_bound(self) -> None:
        self.assertEqual("FROZEN_BEFORE_RUN", self.registry["status"])
        self.assertEqual(self.registry["runner_sha256"], sha256_file(S4_RUNNER))
        self.assertEqual(self.manifest["execution_registry_sha256"], sha256_file(S4_REGISTRY))
        for row in self.manifest["outputs"]:
            path = S4_DIR / row["file"]
            self.assertEqual(row["sha256"], sha256_file(path))

    def test_s4_macro_is_inconclusive_but_clean_loss_is_supported(self) -> None:
        contrasts = {r["metric_id"]: r for r in self.verification["contrasts"]}
        self.assertEqual(-0.005639186754005907, contrasts["macro_f1"]["delta"])
        self.assertLess(contrasts["macro_f1"]["ci_low"], 0.0)
        self.assertGreater(contrasts["macro_f1"]["ci_high"], 0.0)
        self.assertLess(contrasts["clean_abstention"]["ci_high"], 0.0)
        self.assertTrue(self.verification["score_and_locator_unchanged"])


class Phase0PopulationBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(S5_REGISTRY.read_text(encoding="utf-8"))
        cls.manifest = json.loads((S5_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        cls.verification = json.loads((S5_DIR / "P0_S5_VERIFICATION.json").read_text(encoding="utf-8"))

    def test_s5_is_frozen_dual_build_and_manifest_bound(self) -> None:
        self.assertEqual("FROZEN_BEFORE_RUN", self.registry["status"])
        self.assertEqual(self.registry["runner_sha256"], sha256_file(S5_RUNNER))
        self.assertEqual(self.manifest["execution_registry_sha256"], sha256_file(S5_REGISTRY))
        self.assertTrue(self.verification["dual_build"]["dual_build_decisions_identical"])
        self.assertTrue(self.verification["dual_build"]["dual_build_metrics_identical"])
        for row in self.manifest["outputs"]:
            path = S5_DIR / row["file"]
            self.assertEqual(row["sha256"], sha256_file(path))

    def test_s5_population_states_and_nonpaired_boundary(self) -> None:
        a = {r["metric_id"]: r for r in self.verification["s5a"]["aggregate"]}
        b = {r["metric_id"]: r for r in self.verification["s5b"]["aggregate"]}
        self.assertEqual(0.2931182814184147, a["macro_f1"]["value"])
        self.assertEqual(0.2943961703375378, b["macro_f1"]["value"])
        macro = next(r for r in self.verification["composition_contrasts"] if r["metric_id"] == "macro_f1")
        self.assertLess(macro["ci_low"], 0.0)
        self.assertGreater(macro["ci_high"], 0.0)
        self.assertFalse(self.verification["s4_adjacent_contrast_available"])
        self.assertFalse(self.verification["prediction_flips_available"])


if __name__ == "__main__":
    unittest.main()
