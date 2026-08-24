#!/usr/bin/env python3
"""Deterministic fail-closed tests for the external final-answer boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

from spectral_utils.dufs_liu_feature_contract import CONTRACT_VERSION
from spectral_utils.reconstruction_benchmark.external_final_answer import (
    CANONICAL_FEATURE_NAMES,
    LABEL_ADAPTERS,
    RAW_ADAPTERS,
    SCORE_FREEZE_SCHEMA_VERSION,
    ExternalCellSpec,
    ExternalContractError,
    ExternalRegistry,
    RawFeatureCell,
    ReadinessStatus,
    SourceFile,
    apply_mixed_v2_once,
    load_labels_after_score_freeze,
    load_external_registry,
    load_prepared_external_cell,
    load_raw_feature_cell,
    prepare_external_cell,
    resolve_sources,
    verify_source_file,
)
from spectral_utils.reconstruction_benchmark.io import (
    canonical_tree_manifest,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.external_ab import (
    AB_CERTIFICATE_SCHEMA_VERSION,
    AB_VERIFICATION_SOURCES,
    assert_external_ab_certificate,
    current_feature_contract_bindings,
    validate_scientific_score_freeze,
    verify_current_source_snapshot,
)
from spectral_utils.reconstruction_benchmark.methods import PRIMARY_METHOD_IDS
from spectral_utils.reconstruction_benchmark.methods import run_method
from spectral_utils.reconstruction_benchmark.external_evaluation import (
    aurc_x1000,
    grouped_paired_bootstrap,
    population_grouped_paired_bootstrap,
)


REPO = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExternalFinalAnswerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.bin"
        self.source.write_bytes(b"contains a secret label but no fitting adapter may copy it")
        self.spec = ExternalCellSpec.from_mapping({
            "cell_id": "synthetic_cell",
            "population_id": "synthetic_population",
            "dataset_id": "synthetic_dataset",
            "model_id": "synthetic_model",
            "slice_id": "overall",
            "domain": "qa_response_detection",
            "comparison_group_id": "synthetic_group",
            "expected_rows": 12,
            "adapter_id": "synthetic_v1",
            "fit_policy": "run_if_compatible",
            "panel_role": "test",
            "expected_incorrect": 6,
            "expected_correct": 6,
            "source": {
                "kind": "explicit",
                "files": [{"path": "source.bin", "sha256": _digest(self.source)}],
            },
        })
        self.registry_path = self.root / "registry.json"
        self.population_path = self.root / "populations.json"
        self.registry_path.write_text("{}", encoding="utf-8")
        self.population_path.write_text("{}", encoding="utf-8")
        self.registry = ExternalRegistry(
            path=self.registry_path,
            sha256="a" * 64,
            population_registry_path=self.population_path,
            population_registry_sha256="b" * 64,
            raw={"feature_contract_id": CONTRACT_VERSION},
            cells=(self.spec,),
        )
        rng = np.random.default_rng(20260824)
        self.raw_matrix = rng.normal(size=(12, len(CANONICAL_FEATURE_NAMES)))
        self.row_ids = tuple(f"row:{index:03d}" for index in range(12))
        self.group_ids = tuple(f"group:{index // 2:03d}" for index in range(12))

        def raw_adapter(spec, sources):
            return RawFeatureCell(
                spec,
                self.raw_matrix,
                self.row_ids,
                self.group_ids,
                sources.feature_files,
            )

        def label_adapter(spec, sources):
            labels = [index % 2 for index in range(12)]
            return list(self.row_ids), list(self.group_ids), labels, {"label_rule": "synthetic alternating"}

        self.raw_patch = mock.patch.dict(RAW_ADAPTERS, {"synthetic_v1": raw_adapter})
        self.label_patch = mock.patch.dict(LABEL_ADAPTERS, {"synthetic_v1": label_adapter})
        self.raw_patch.start()
        self.label_patch.start()

    def tearDown(self) -> None:
        self.label_patch.stop()
        self.raw_patch.stop()
        self.temp.cleanup()

    def _freeze(self, *, complete: bool = True) -> dict:
        value = {
            "schema_version": SCORE_FREEZE_SCHEMA_VERSION,
            "all_expected_scores_present": complete,
            "labels_opened_by_fit": False,
            "runtime_labels_used": False,
            "external_registry_sha256": self.registry.sha256,
        }
        value["payload_sha256"] = sha256_bytes(canonical_json_bytes(value))
        return value

    def test_prepare_is_exactly_once_target_free_and_byte_deterministic(self) -> None:
        left = self.root / "A.npz"
        right = self.root / "B.npz"
        first = prepare_external_cell(
            registry=self.registry, spec=self.spec, repo=self.root, output_path=left,
        )
        second = prepare_external_cell(
            registry=self.registry, spec=self.spec, repo=self.root, output_path=right,
        )
        self.assertEqual(first["mixed_v2_applied_count"], 1)
        self.assertEqual(first["n_features"], 30)
        self.assertEqual(len(first["present_feature_roster_sha256"]), 64)
        self.assertEqual(first["prepared_matrix_sha256"], second["prepared_matrix_sha256"])
        self.assertEqual(sha256_file(left), sha256_file(right))
        bundle = load_npz_no_pickle(left)
        self.assertEqual(
            set(bundle),
            {"X_confidence", "feature_names", "family_ids", "row_ids", "group_ids", "row_index"},
        )
        self.assertFalse(any(
            token in name.lower()
            for name in bundle
            for token in ("label", "target", "correct", "gold")
        ))
        record = {**first, "artifact_path": left.name}
        prepared, groups = load_prepared_external_cell(artifact_path=left, record=record)
        self.assertEqual(prepared.feature_names, CANONICAL_FEATURE_NAMES)
        self.assertEqual(groups, self.group_ids)
        result = run_method("equal_feature_mean", prepared)
        self.assertEqual(result.status.value, "OK")
        self.assertEqual(result.score.shape, (12,))

    def test_missing_feature_rejects_whole_cell(self) -> None:
        broken = self.raw_matrix.copy()
        broken[3, 29] = np.nan
        with self.assertRaises(ExternalContractError) as raised:
            RawFeatureCell(
                self.spec, broken, self.row_ids, self.group_ids,
                (SourceFile(self.source, "source.bin", _digest(self.source)),),
            )
        self.assertEqual(raised.exception.status, ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT)

    def test_uniformly_absent_view_is_bound_as_a_29_feature_cell(self) -> None:
        repgrid_path = self.root / "uniform_absent.pkl"
        data = {
            problem: {"candidates": [{"label": bool(problem % 2)}]}
            for problem in range(12)
        }
        with repgrid_path.open("wb") as handle:
            pickle.dump(data, handle)
        spec = ExternalCellSpec.from_mapping({
            "cell_id": "uniform_absent", "population_id": "synthetic_population",
            "dataset_id": "repgrid", "model_id": "model", "slice_id": "k1",
            "domain": "qa_response_detection", "comparison_group_id": "repgrid",
            "expected_rows": 12, "expected_group_count": 12,
            "adapter_id": "repgrid_embedded_label_v1", "fit_policy": "run_if_compatible",
            "panel_role": "test",
            "source": {"kind": "explicit", "files": [{"path": "uniform_absent.pkl", "sha256": _digest(repgrid_path)}]},
        })
        sources = resolve_sources(self.registry, spec, repo=self.root)
        absent = CANONICAL_FEATURE_NAMES[-1]
        values = {
            name: float(index + 1)
            for index, name in enumerate(CANONICAL_FEATURE_NAMES)
            if name != absent
        }
        with mock.patch(
            "spectral_utils.reconstruction_benchmark.external_final_answer._telemetry_features",
            return_value=values,
        ):
            raw = load_raw_feature_cell(spec, sources)
        self.assertEqual(raw.raw_matrix.shape, (12, 29))
        self.assertEqual(raw.feature_names, CANONICAL_FEATURE_NAMES[:-1])
        transformed, _ = apply_mixed_v2_once(raw)
        self.assertEqual(transformed.shape, (12, 29))

    def test_row_count_never_shrinks_silently(self) -> None:
        with self.assertRaises(ExternalContractError) as raised:
            RawFeatureCell(
                self.spec, self.raw_matrix[:-1], self.row_ids[:-1], self.group_ids[:-1],
                (SourceFile(self.source, "source.bin", _digest(self.source)),),
            )
        self.assertEqual(raised.exception.status, ReadinessStatus.ROW_CONTRACT_MISMATCH)

    def test_second_preprocessing_pass_is_rejected(self) -> None:
        with self.assertRaises(ExternalContractError) as raised:
            RawFeatureCell(
                self.spec, self.raw_matrix, self.row_ids, self.group_ids,
                (SourceFile(self.source, "source.bin", _digest(self.source)),),
                preprocessing_steps=(CONTRACT_VERSION,),
            )
        self.assertEqual(raised.exception.status, ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT)

    def test_source_hash_mismatch_and_lfs_pointer_are_explicit(self) -> None:
        with self.assertRaises(ExternalContractError) as bad_hash:
            verify_source_file(SourceFile(self.source, "source.bin", "0" * 64))
        self.assertEqual(bad_hash.exception.status, ReadinessStatus.SOURCE_HASH_MISMATCH)

        pointer = self.root / "large.pkl"
        oid = "1" * 64
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{oid}\nsize 12345\n",
            encoding="utf-8",
        )
        with self.assertRaises(ExternalContractError) as blocked:
            verify_source_file(SourceFile(pointer, "large.pkl", oid, 12345))
        self.assertEqual(blocked.exception.status, ReadinessStatus.BLOCKED_ASSET)

    def test_prepared_artifact_tamper_is_refused(self) -> None:
        path = self.root / "tamper.npz"
        record = prepare_external_cell(
            registry=self.registry, spec=self.spec, repo=self.root, output_path=path,
        )
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        with self.assertRaises(RuntimeError):
            load_prepared_external_cell(artifact_path=path, record=record)

    def test_grouped_paired_bootstrap_is_deterministic_and_group_preserving(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
        groups = ("a", "a", "b", "b", "c", "c", "d", "d")
        scores = {
            "iu_pcr": np.asarray([0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.9, 0.6]),
            "candidate": np.asarray([0.2, 0.1, 0.7, 0.8, 0.4, 0.3, 0.6, 0.9]),
        }
        left = grouped_paired_bootstrap(
            labels=labels, scores_by_method=scores, group_ids=groups,
            draws=250, seed=17,
        )
        right = grouped_paired_bootstrap(
            labels=labels, scores_by_method=scores, group_ids=groups,
            draws=250, seed=17,
        )
        self.assertEqual(left, right)
        self.assertEqual(left["bootstrap_unit"], "source_group")
        self.assertTrue(left["paired"])
        self.assertEqual(left["n_groups"], 4)
        self.assertEqual(left["contrasts"]["candidate"]["auroc"]["reference_method"], "iu_pcr")

    def test_aurc_is_invariant_to_row_order_inside_score_ties(self) -> None:
        labels = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
        scores = np.asarray([0.2, 0.2, 0.2, 0.8, 0.8, 0.8], dtype=float)
        permutation = np.asarray([2, 0, 1, 5, 3, 4])
        self.assertEqual(
            aurc_x1000(labels, scores),
            aurc_x1000(labels[permutation], scores[permutation]),
        )
        constant = np.zeros(len(labels), dtype=float)
        self.assertEqual(
            aurc_x1000(labels, constant),
            aurc_x1000(labels[::-1], constant[::-1]),
        )

    def test_population_bootstrap_links_rosters_and_equal_weights_cells(self) -> None:
        groups = ("q0", "q1", "q2", "q3")
        cells = {
            "model_a": {
                "labels": [0, 1, 0, 1],
                "group_ids": groups,
                "scores_by_method": {
                    "iu_pcr": [0.1, 0.9, 0.2, 0.8],
                    "candidate": [0.2, 0.8, 0.1, 0.9],
                },
            },
            "model_b": {
                "labels": [0, 1, 1, 0],
                "group_ids": groups,
                "scores_by_method": {
                    "iu_pcr": [0.2, 0.8, 0.7, 0.3],
                    "candidate": [0.1, 0.9, 0.8, 0.2],
                },
            },
        }
        kwargs = dict(
            cells=cells,
            link_keys={"model_a": "same_questions", "model_b": "same_questions"},
            draws=200,
            seed=91,
        )
        left = population_grouped_paired_bootstrap(**kwargs)
        right = population_grouped_paired_bootstrap(**kwargs)
        self.assertEqual(left, right)
        self.assertTrue(left["linked_resampling"])
        self.assertEqual(left["point_estimate_unit"], "cell")
        self.assertEqual(left["n_resampling_groups"], 4)
        broken = json.loads(json.dumps(cells))
        broken["model_b"]["group_ids"][-1] = "different"
        with self.assertRaisesRegex(ValueError, "linked group roster mismatch"):
            population_grouped_paired_bootstrap(
                cells=broken,
                link_keys={"model_a": "same_questions", "model_b": "same_questions"},
                draws=10,
            )

    def test_hle_style_bootstrap_is_stratified_at_the_source_group(self) -> None:
        interval = population_grouped_paired_bootstrap(
            cells={
                "hle": {
                    "labels": [0, 0, 0, 1],
                    "group_ids": ["a", "b", "c", "d"],
                    "scores_by_method": {
                        "iu_pcr": [0.1, 0.2, 0.3, 0.9],
                        "candidate": [0.2, 0.1, 0.4, 0.8],
                    },
                }
            },
            draws=100,
            seed=4070942594,
            weighting="single_cell",
            stratify_by_label=True,
        )
        self.assertTrue(interval["stratified_by_group_label"])
        self.assertEqual(interval["valid_draws"], 100)
        self.assertEqual(interval["link_blocks"][0]["groups_by_label"], {"0": 3, "1": 1})

    def test_source_snapshot_rejects_stale_current_file(self) -> None:
        file_path = self.root / "source.py"
        file_path.write_text("first", encoding="utf-8")
        snapshot = {
            "files": [{"path": "source.py", "sha256": sha256_file(file_path)}],
        }
        snapshot["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(snapshot))
        verify_current_source_snapshot(
            snapshot, repo=self.root, required_paths=("source.py",), name="synthetic",
        )
        file_path.write_text("second", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed or is missing"):
            verify_current_source_snapshot(
                snapshot, repo=self.root, required_paths=("source.py",), name="synthetic",
            )

    def test_debug_freeze_is_refused_by_publication_validator(self) -> None:
        path = self.root / "debug_freeze.json"
        value = {
            "schema_version": SCORE_FREEZE_SCHEMA_VERSION,
            "scientific_full": False,
            "all_expected_scores_present": True,
        }
        value["payload_sha256"] = sha256_bytes(canonical_json_bytes(value))
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "debug or partial"):
            validate_scientific_score_freeze(
                path,
                registry=self.registry,
                repo=REPO,
                input_root=self.root,
                fit_root=self.root,
            )

    def test_ab_certificate_is_required_and_tree_tamper_is_refused(self) -> None:
        release_id = "synthetic_release"
        for build_id in ("A", "B"):
            build = self.root / release_id / f"build_{build_id}" / "external_final_answer"
            inputs, fit = build / "inputs", build / "fit"
            inputs.mkdir(parents=True)
            fit.mkdir(parents=True)
            (inputs / "MANIFEST.json").write_text(f"input-{build_id}", encoding="utf-8")
            (fit / "SCORE_FREEZE_MANIFEST.json").write_text(f"score-{build_id}", encoding="utf-8")
        verification_snapshot = {
            "files": [
                {"path": relative, "sha256": sha256_file(REPO / relative)}
                for relative in AB_VERIFICATION_SOURCES
            ]
        }
        verification_snapshot["snapshot_sha256"] = sha256_bytes(
            canonical_json_bytes(verification_snapshot)
        )
        builds = {}
        for build_id in ("A", "B"):
            build = self.root / release_id / f"build_{build_id}" / "external_final_answer"
            input_path = build / "inputs" / "MANIFEST.json"
            freeze_path = build / "fit" / "SCORE_FREEZE_MANIFEST.json"
            builds[build_id] = {
                "input_manifest_sha256": sha256_file(input_path),
                "score_freeze_sha256": sha256_file(freeze_path),
                "input_tree": canonical_tree_manifest(build / "inputs"),
                "fit_tree": canonical_tree_manifest(build / "fit"),
            }
        comparison_records = [
            {"cell_id": "synthetic_cell", "method_id": method_id}
            for method_id in PRIMARY_METHOD_IDS
        ]
        certificate = {
            "schema_version": AB_CERTIFICATE_SCHEMA_VERSION,
            "release_id": release_id,
            "status": "PASS",
            "scientific_full": True,
            "external_registry_sha256": self.registry.sha256,
            "population_registry_sha256": self.registry.population_registry_sha256,
            "feature_contract_bindings": current_feature_contract_bindings(REPO),
            "verification_source_snapshot": verification_snapshot,
            "method_registry_sha256": sha256_file(REPO / "configs/reconstruction_benchmark_v1/methods.json"),
            "method_ids": list(PRIMARY_METHOD_IDS),
            "cell_ids": ["synthetic_cell"],
            "n_method_comparisons": len(PRIMARY_METHOD_IDS),
            "comparison_records": comparison_records,
            "comparison_records_sha256": sha256_bytes(canonical_json_bytes(comparison_records)),
            "builds": builds,
        }
        certificate["certificate_sha256"] = sha256_bytes(canonical_json_bytes(certificate))
        certificate_path = self.root / "AB_VERIFICATION.json"
        certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
        assert_external_ab_certificate(
            certificate_path,
            release_id=release_id,
            release_root=self.root,
            selected_build="A",
            registry=self.registry,
            repo=REPO,
        )
        tampered = self.root / release_id / "build_B" / "external_final_answer" / "fit" / "SCORE_FREEZE_MANIFEST.json"
        tampered.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed after A/B certification"):
            assert_external_ab_certificate(
                certificate_path,
                release_id=release_id,
                release_root=self.root,
                selected_build="A",
                registry=self.registry,
                repo=REPO,
            )

    def test_production_registry_binds_semgrad_to_the_declared_qwen_model(self) -> None:
        registry = load_external_registry(
            repo=REPO,
            registry_path="configs/reconstruction_benchmark_v1/external_final_answer.json",
            population_registry_path="configs/reconstruction_benchmark_v1/populations.json",
        )
        self.assertEqual(registry.by_cell["semgrad_sciq_bem"].model_id, "qwen3_4b_instruct_2507")
        self.assertEqual(registry.by_cell["semgrad_truthfulqa_bem"].model_id, "qwen3_4b_instruct_2507")

    def test_labels_are_blocked_until_valid_score_freeze(self) -> None:
        with self.assertRaises(ExternalContractError) as blocked:
            load_labels_after_score_freeze(
                registry=self.registry,
                spec=self.spec,
                repo=self.root,
                score_freeze=self._freeze(complete=False),
                expected_row_ids=self.row_ids,
                expected_group_ids=self.group_ids,
            )
        self.assertEqual(blocked.exception.status, ReadinessStatus.LABEL_PROVENANCE_BLOCKED)

        labels = load_labels_after_score_freeze(
            registry=self.registry,
            spec=self.spec,
            repo=self.root,
            score_freeze=self._freeze(),
            expected_row_ids=self.row_ids,
            expected_group_ids=self.group_ids,
        )
        self.assertEqual(int(labels.incorrect.sum()), 6)

    def test_label_row_order_mismatch_is_fatal(self) -> None:
        reversed_rows = tuple(reversed(self.row_ids))
        with self.assertRaises(ExternalContractError) as mismatch:
            load_labels_after_score_freeze(
                registry=self.registry,
                spec=self.spec,
                repo=self.root,
                score_freeze=self._freeze(),
                expected_row_ids=reversed_rows,
                expected_group_ids=self.group_ids,
            )
        self.assertEqual(mismatch.exception.status, ReadinessStatus.ROW_CONTRACT_MISMATCH)

    def test_forbidden_population_never_gets_an_adapter(self) -> None:
        quarantined = ExternalCellSpec.from_mapping({
            "cell_id": "coqa_bad", "population_id": "coqa", "dataset_id": "coqa",
            "model_id": "llama_base", "slice_id": "historical", "domain": "quarantine",
            "comparison_group_id": "none", "expected_rows": 1, "adapter_id": "none",
            "fit_policy": "forbidden", "configured_status": "QUARANTINED",
            "status_reason": "invalid prompt", "panel_role": "quarantine",
        })
        with self.assertRaises(ExternalContractError) as raised:
            load_raw_feature_cell(
                quarantined,
                resolve_sources(self.registry, self.spec, repo=self.root),
            )
        self.assertEqual(raised.exception.status, ReadinessStatus.QUARANTINED)

    def test_repgrid_adapter_keeps_all_rows_and_never_reads_labels_for_features(self) -> None:
        repgrid_path = self.root / "repgrid.pkl"
        data = {
            problem: {
                "candidates": [
                    {"label": bool(index % 2), "secret_target": index}
                    for index in range(2)
                ]
            }
            for problem in range(6)
        }
        with repgrid_path.open("wb") as handle:
            pickle.dump(data, handle)
        repgrid_spec = ExternalCellSpec.from_mapping({
            "cell_id": "repgrid", "population_id": "synthetic_population",
            "dataset_id": "repgrid", "model_id": "model", "slice_id": "k2",
            "domain": "negative_stress", "comparison_group_id": "repgrid",
            "expected_rows": 12, "expected_group_count": 6,
            "adapter_id": "repgrid_embedded_label_v1", "fit_policy": "run_if_compatible",
            "panel_role": "test",
            "source": {"kind": "explicit", "files": [{"path": "repgrid.pkl", "sha256": _digest(repgrid_path)}]},
        })
        sources = resolve_sources(self.registry, repgrid_spec, repo=self.root)
        values = {name: float(index + 1) for index, name in enumerate(CANONICAL_FEATURE_NAMES)}
        with mock.patch(
            "spectral_utils.reconstruction_benchmark.external_final_answer._telemetry_features",
            return_value=values,
        ) as extractor:
            raw = load_raw_feature_cell(repgrid_spec, sources)
        self.assertEqual(raw.raw_matrix.shape, (12, 30))
        self.assertEqual(len(set(raw.group_ids)), 6)
        self.assertEqual(extractor.call_count, 12)
        for call in extractor.call_args_list:
            passed = call.args[0]
            self.assertIn("label", passed)  # source may contain it
            # The extractor is a telemetry whitelist; its return has no target.
        self.assertEqual(tuple(raw.feature_names), CANONICAL_FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
