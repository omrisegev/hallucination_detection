#!/usr/bin/env python3
"""Deterministic fail-closed tests for the external final-answer boundary."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from spectral_utils.dufs_liu_feature_contract import CONTRACT_VERSION
from spectral_utils.reconstruction_benchmark.external_final_answer import (
    CANONICAL_FEATURE_NAMES,
    ID_CONTRACT_VERSION,
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
    apply_external_id_contract,
    assert_score_freeze,
    canonicalize_external_identity_order,
    external_id_contract_binding,
    fit_safe_external_cell_record,
    identity_key_id,
    load_identity_key,
    load_labels_after_score_freeze,
    load_external_registry,
    load_prepared_external_cell,
    load_raw_feature_cell,
    prepare_external_cell,
    resolve_sources,
    sealed_group_roster_commitment,
    verify_source_file,
)
from spectral_utils.reconstruction_benchmark.io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_tree_manifest,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.external_ab import (
    AB_CERTIFICATE_SCHEMA_VERSION,
    AB_VERIFICATION_SOURCES,
    FIT_SAFE_INPUT_MANIFEST_SCHEMA_VERSION,
    assert_external_ab_certificate,
    _assert_input_population_count_binding,
    current_feature_contract_bindings,
    validate_fit_safe_input_manifest,
    validate_scientific_score_freeze,
    verify_current_source_snapshot,
)
from spectral_utils.reconstruction_benchmark.fit_firewall import (
    build_fit_audit_policy,
)
from spectral_utils.reconstruction_benchmark.external_fit_contract import (
    build_fit_row_identity_contract,
)
from spectral_utils.reconstruction_benchmark.external_fit_safe import (
    FIT_SAFE_INPUT_MANIFEST_FIELDS,
)
from scripts.reconstruction_benchmark.run_external_final_answer_methods import (
    FIT_CAPSULE_CODE_ALLOWLIST,
    FIT_CAPSULE_CONFIG_ALLOWLIST,
    _copy_fit_capsule,
    _launch_worker,
    _worker_policy,
)
from scripts.reconstruction_benchmark.evaluate_external_final_answer import (
    _AtomicEvaluationStage,
    _remove_verified_empty_directory_tree,
    _restore_validated_score_freeze,
)
from scripts.reconstruction_benchmark import (
    evaluate_external_final_answer as external_evaluation_cli,
)
from spectral_utils.reconstruction_benchmark.methods import PRIMARY_METHOD_IDS
from spectral_utils.reconstruction_benchmark.methods import run_method
from spectral_utils.reconstruction_benchmark.serialization import write_score_result
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
            raw={
                "feature_contract_id": CONTRACT_VERSION,
                "population_aggregates": {
                    "synthetic_population": {
                        "enabled": True,
                        "link_cells_by": "none",
                    },
                },
                "identity_contract": {
                    "version": ID_CONTRACT_VERSION,
                    "digest_algorithm": "hmac-sha256-canonical-json-v1",
                    "identity_key_contract_version": "reconstruction-external-identity-key-v1",
                    "identity_key_bytes": 32,
                    "opaque_row_id_prefix": "xridv2_",
                    "opaque_group_id_prefix": "xgidv2_",
                    "row_namespace_scope": "cell",
                    "canonical_row_order": "lexicographic_opaque_row_id",
                    "group_namespace_by_population": {
                        "synthetic_population": "cell",
                    },
                },
            },
            cells=(self.spec,),
        )
        rng = np.random.default_rng(20260824)
        self.raw_matrix = rng.normal(size=(12, len(CANONICAL_FEATURE_NAMES)))
        self.raw_row_ids = tuple(
            f"deception_error_family_correct_row:{index:03d}" for index in range(12)
        )
        self.raw_group_ids = tuple(
            f"step_contradiction_label_group:{index // 2:03d}" for index in range(12)
        )
        self.identity_key = b"\x17" * 32
        self.identity = apply_external_id_contract(
            self.registry,
            self.spec,
            self.raw_row_ids,
            self.raw_group_ids,
            identity_key=self.identity_key,
        )
        self.fit_identity = build_fit_row_identity_contract(
            self.identity.contract_binding,
            identity_key=self.identity_key,
        )
        _, self.identity = canonicalize_external_identity_order(
            self.raw_matrix, self.identity
        )
        self.row_ids = self.identity.row_ids
        self.group_ids = self.identity.group_ids
        self.group_commitment = sealed_group_roster_commitment(self.identity)

        def raw_adapter(spec, sources):
            return RawFeatureCell(
                spec,
                self.raw_matrix,
                self.raw_row_ids,
                self.raw_group_ids,
                sources.feature_files,
            )

        def label_adapter(spec, sources):
            labels = [index % 2 for index in range(12)]
            return list(self.raw_row_ids), list(self.raw_group_ids), labels, {"label_rule": "synthetic alternating"}

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
            "identity_contract": self.fit_identity,
            "id_contract_version": ID_CONTRACT_VERSION,
        }
        value["payload_sha256"] = sha256_bytes(canonical_json_bytes(value))
        return value

    def _write_fit_safe_manifest(
        self,
        *,
        input_root: Path,
        record: dict,
        release_id: str = "synthetic_fit",
        build_id: str = "A",
    ) -> dict:
        safe_record = fit_safe_external_cell_record(record)
        manifest = {
            "schema_version": FIT_SAFE_INPUT_MANIFEST_SCHEMA_VERSION,
            "prepared_cell_schema_version": record["schema_version"],
            "identity_contract": safe_record["identity_contract"],
            "id_contract_version": ID_CONTRACT_VERSION,
            "release_id": release_id,
            "build_id": build_id,
            "scientific_full_build": False,
            "applicability_complete": False,
            "complete_eligible_roster": False,
            "external_registry_sha256": self.registry.sha256,
            "population_registry_sha256": self.registry.population_registry_sha256,
            "preparation_manifest_sha256": "1" * 64,
            "preparation_manifest_payload_sha256": "2" * 64,
            "preparation_attestation_sha256": "3" * 64,
            "feature_contract_id": CONTRACT_VERSION,
            "mixed_v2_applied_exactly_once": True,
            "target_data_opened": False,
            "historical_scores_opened": False,
            "n_registered_cells": 1,
            "n_runnable_cells": 1,
            "n_prepared_cells": 1,
            "cells": [safe_record],
        }
        manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        atomic_write_json(input_root / "MANIFEST.json", manifest)
        return manifest

    def test_prepare_is_exactly_once_target_free_and_byte_deterministic(self) -> None:
        left = self.root / "A.npz"
        right = self.root / "B.npz"
        first = prepare_external_cell(
            registry=self.registry, spec=self.spec, repo=self.root, output_path=left,
            identity_key=self.identity_key,
        )
        second = prepare_external_cell(
            registry=self.registry, spec=self.spec, repo=self.root, output_path=right,
            identity_key=self.identity_key,
        )
        self.assertEqual(first["mixed_v2_applied_count"], 1)
        self.assertEqual(first["n_features"], 30)
        self.assertEqual(len(first["present_feature_roster_sha256"]), 64)
        self.assertEqual(first["prepared_matrix_sha256"], second["prepared_matrix_sha256"])
        self.assertEqual(sha256_file(left), sha256_file(right))
        bundle = load_npz_no_pickle(left)
        self.assertEqual(
            set(bundle),
            {
                "X_confidence", "feature_names", "family_ids", "row_ids",
                "row_index", "id_contract_version", "id_contract_sha256",
                "row_namespace_sha256", "identity_key_id",
            },
        )
        self.assertFalse(any(
            token in name.lower()
            for name in bundle
            for token in ("label", "target", "correct", "gold")
        ))
        leaked_source_fragments = (
            b"deception_error_family_correct_row",
            b"step_contradiction_label_group",
        )
        serialized_record = canonical_json_bytes(first)
        for fragment in leaked_source_fragments:
            self.assertNotIn(fragment, left.read_bytes())
            self.assertNotIn(fragment, serialized_record)
        record = {**first, "artifact_path": left.name}
        safe_record = fit_safe_external_cell_record(record)
        prepared = load_prepared_external_cell(
            artifact_path=left,
            record=safe_record,
            identity_contract=safe_record["identity_contract"],
        )
        self.assertEqual(prepared.feature_names, CANONICAL_FEATURE_NAMES)
        result = run_method("equal_feature_mean", prepared)
        self.assertEqual(result.status.value, "OK")
        self.assertEqual(result.score.shape, (12,))
        fit_root = self.root / "fit_record"
        fit_record = write_score_result(
            result,
            prepared.row_ids,
            fit_root,
            identity_contract={
                "identity_contract": safe_record["identity_contract"],
                "id_contract_version": first["id_contract_version"],
                "id_contract_sha256": safe_record["id_contract_sha256"],
                "identity_key_id": first["identity_key_id"],
                "row_namespace_sha256": first["row_namespace_sha256"],
                "row_roster_sha256": first["row_roster_sha256"],
            },
        )
        self.assertEqual(fit_record["id_contract_version"], ID_CONTRACT_VERSION)
        for fragment in leaked_source_fragments:
            self.assertNotIn(fragment, (fit_root / "RECORD.json").read_bytes())
            self.assertNotIn(fragment, (fit_root / "score.npz").read_bytes())

    def test_release_identity_key_is_sealed_and_changes_every_opaque_id(self) -> None:
        key_path = self.root / "private_control" / "external-id-v2.key"
        first = load_identity_key(key_path, create=True)
        second = load_identity_key(key_path, create=False)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
        self.assertTrue(identity_key_id(first).startswith("xkidv1_"))

        alternate = bytes(value ^ 0xFF for value in first)
        left = apply_external_id_contract(
            self.registry, self.spec, self.raw_row_ids, self.raw_group_ids,
            identity_key=first,
        )
        right = apply_external_id_contract(
            self.registry, self.spec, self.raw_row_ids, self.raw_group_ids,
            identity_key=alternate,
        )
        self.assertTrue(set(left.row_ids).isdisjoint(right.row_ids))
        self.assertTrue(set(left.group_ids).isdisjoint(right.group_ids))

        key_path.chmod(0o644)
        with self.assertRaises(PermissionError):
            load_identity_key(key_path, create=False)

    def test_fit_safe_artifact_omits_groups_and_rejects_family_string_tamper(self) -> None:
        input_root = self.root / "safe_inputs"
        input_root.mkdir()
        path = input_root / "synthetic.npz"
        record = dict(prepare_external_cell(
            registry=self.registry,
            spec=self.spec,
            repo=self.root,
            output_path=path,
            identity_key=self.identity_key,
        ))
        record["artifact_path"] = path.name
        manifest = self._write_fit_safe_manifest(
            input_root=input_root, record=record,
        )
        safe_record = manifest["cells"][0]
        self.assertEqual(set(manifest), set(FIT_SAFE_INPUT_MANIFEST_FIELDS))
        self.assertEqual(
            set(manifest["identity_contract"]),
            {
                "schema_version", "version", "digest_algorithm",
                "identity_key_contract_version", "identity_key_bytes",
                "opaque_row_id_prefix", "row_namespace_scope",
                "canonical_row_order", "key_id",
                "private_group_linkage_commitment", "contract_sha256",
            },
        )
        self.assertTrue(
            manifest["identity_contract"]["private_group_linkage_commitment"]
            .startswith("xglcv1_")
        )
        for forbidden in (
            "opaque_group_id_prefix", "group_namespace_by_population",
            "group_ids", "group_count", "group_namespace_sha256",
        ):
            self.assertNotIn(forbidden, manifest["identity_contract"])
        for forbidden in (
            "group_count", "group_ids", "group_namespace_sha256",
            "sealed_group_roster_commitment_sha256", "source_files",
            "source_root", "expected_correct", "expected_incorrect",
            "excluded_raw_id_fingerprints", "adapter_id",
        ):
            self.assertNotIn(forbidden, safe_record)
        arrays = load_npz_no_pickle(path)
        self.assertNotIn("group_ids", arrays)
        self.assertNotIn("group_namespace_sha256", arrays)
        self.assertNotIn("sealed_group_roster_commitment_sha256", arrays)

        changed_linkage = dict(self.identity.contract_binding)
        changed_linkage["group_namespace_by_population"] = {
            "synthetic_population": "population:synthetic_population"
        }
        changed_fit_identity = build_fit_row_identity_contract(
            changed_linkage, identity_key=self.identity_key,
        )
        self.assertEqual(
            changed_fit_identity["key_id"],
            manifest["identity_contract"]["key_id"],
        )
        self.assertNotEqual(
            changed_fit_identity["private_group_linkage_commitment"],
            manifest["identity_contract"]["private_group_linkage_commitment"],
        )

        injected = dict(manifest)
        injected["group_count"] = 6
        injected["payload_sha256"] = sha256_bytes(canonical_json_bytes({
            key: value for key, value in injected.items()
            if key != "payload_sha256"
        }))
        atomic_write_json(input_root / "MANIFEST.json", injected)
        with self.assertRaisesRegex(RuntimeError, "controller-only/unknown"):
            validate_fit_safe_input_manifest(
                input_root / "MANIFEST.json", repo=REPO, input_root=input_root,
                require_scientific=False,
            )
        atomic_write_json(input_root / "MANIFEST.json", manifest)

        arrays["family_ids"] = arrays["family_ids"].copy()
        arrays["family_ids"][0] = "deception_error_family_label"
        atomic_write_npz(path, arrays)
        tampered = {**safe_record, "artifact_sha256": sha256_file(path)}
        with self.assertRaisesRegex(RuntimeError, "family roster"):
            load_prepared_external_cell(
                artifact_path=path,
                record=tampered,
                identity_contract=manifest["identity_contract"],
            )

    def test_old_unkeyed_fit_manifest_schema_is_scientifically_refused(self) -> None:
        path = self.root / "old_v1_manifest.json"
        value = {
            "schema_version": "reconstruction-external-target-free-build-v1",
            "scientific_full_build": True,
        }
        value["payload_sha256"] = sha256_bytes(canonical_json_bytes(value))
        atomic_write_json(path, value)
        with self.assertRaisesRegex(RuntimeError, "fit-safe external manifest schema"):
            validate_fit_safe_input_manifest(
                path, repo=REPO, input_root=self.root,
            )

    def test_real_prepared_provenance_uses_strict_n_rows_count_binding(self) -> None:
        manifest_path = (
            REPO / "results/reconstruction_benchmark_v1/private_control"
            / "2026-08-24_external_final_answer_v2_opaque"
            / "external_final_answer/build_A/preparation_provenance/MANIFEST.json"
        )
        if not manifest_path.is_file():
            self.skipTest("real external preparation manifest is not materialized")
        registry = load_external_registry(
            repo=REPO,
            registry_path=(
                REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
            ),
            population_registry_path=(
                REPO / "configs/reconstruction_benchmark_v1/populations.json"
            ),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = next(
            item for item in manifest["cells"]
            if item["cell_id"] == "processbench_gsm8k_qwen3_4b"
        )
        spec = next(
            item for item in registry.cells
            if item.cell_id == "processbench_gsm8k_qwen3_4b"
        )
        self.assertEqual(row["status"], "ELIGIBLE")
        self.assertNotIn("expected_rows", row)
        self.assertEqual(row["n_rows"], spec.expected_rows)
        _assert_input_population_count_binding(row, spec)

        wrong_actual_count = {**row, "n_rows": spec.expected_rows - 1}
        with self.assertRaisesRegex(RuntimeError, "population/count binding"):
            _assert_input_population_count_binding(wrong_actual_count, spec)

        wrong_population = {**row, "population_id": "wrong_population"}
        with self.assertRaisesRegex(RuntimeError, "population/count binding"):
            _assert_input_population_count_binding(wrong_population, spec)

        ambiguous_prepared = {**row, "expected_rows": spec.expected_rows}
        with self.assertRaisesRegex(RuntimeError, "population/count binding"):
            _assert_input_population_count_binding(ambiguous_prepared, spec)

        terminal = {
            "cell_id": spec.cell_id,
            "population_id": spec.population_id,
            "status": "INCOMPATIBLE_FEATURE_CONTRACT",
            "expected_rows": spec.expected_rows,
        }
        _assert_input_population_count_binding(terminal, spec)
        with self.assertRaisesRegex(RuntimeError, "population/count binding"):
            _assert_input_population_count_binding(
                {**terminal, "expected_rows": spec.expected_rows - 1}, spec,
            )
        with self.assertRaisesRegex(RuntimeError, "population/count binding"):
            _assert_input_population_count_binding(
                {**terminal, "n_rows": spec.expected_rows}, spec,
            )

    def test_restricted_external_worker_exact_closure_runs_all_thirteen(self) -> None:
        release_id = "synthetic_worker"
        build_id = "A"
        # CA-SpecRaGE's frozen k=15 split needs 16 validation rows plus two
        # disjoint 16-row training batches.  Keep this smoke above that exact
        # label-free feasibility boundary without changing any method config.
        worker_rows = 64
        worker_spec = ExternalCellSpec.from_mapping({
            "cell_id": self.spec.cell_id,
            "population_id": self.spec.population_id,
            "dataset_id": self.spec.dataset_id,
            "model_id": self.spec.model_id,
            "slice_id": self.spec.slice_id,
            "domain": self.spec.domain,
            "comparison_group_id": self.spec.comparison_group_id,
            "expected_rows": worker_rows,
            "adapter_id": self.spec.adapter_id,
            "fit_policy": self.spec.fit_policy,
            "panel_role": self.spec.panel_role,
            "expected_incorrect": worker_rows // 2,
            "expected_correct": worker_rows // 2,
            "source": {
                "kind": "explicit",
                "files": [{"path": "source.bin", "sha256": _digest(self.source)}],
            },
        })
        worker_registry = ExternalRegistry(
            path=self.registry.path,
            sha256=self.registry.sha256,
            population_registry_path=self.registry.population_registry_path,
            population_registry_sha256=self.registry.population_registry_sha256,
            raw=self.registry.raw,
            cells=(worker_spec,),
        )
        worker_rng = np.random.default_rng(2026082401)
        worker_matrix = worker_rng.normal(
            size=(worker_rows, len(CANONICAL_FEATURE_NAMES))
        )
        worker_row_ids = tuple(
            f"deception_error_family_correct_row:{index:03d}"
            for index in range(worker_rows)
        )
        worker_group_ids = tuple(
            f"step_contradiction_label_group:{index // 2:03d}"
            for index in range(worker_rows)
        )

        def worker_raw_adapter(spec, sources):
            return RawFeatureCell(
                spec, worker_matrix, worker_row_ids, worker_group_ids,
                sources.feature_files,
            )

        input_root = self.root / "release" / "inputs"
        cell_root = input_root / "cells"
        cell_root.mkdir(parents=True)
        path = cell_root / "synthetic_cell.npz"
        with mock.patch.dict(RAW_ADAPTERS, {"synthetic_v1": worker_raw_adapter}):
            record = dict(prepare_external_cell(
                registry=worker_registry,
                spec=worker_spec,
                repo=self.root,
                output_path=path,
                identity_key=self.identity_key,
            ))
        record["artifact_path"] = path.relative_to(input_root).as_posix()
        self._write_fit_safe_manifest(
            input_root=input_root,
            record=record,
            release_id=release_id,
            build_id=build_id,
        )
        capsule_root = self.root / "release" / "fit_capsule"
        code_root = _copy_fit_capsule(capsule_root)
        closure = json.loads(
            (code_root / "FIT_CODE_CLOSURE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [row["path"] for row in closure["source_files"]],
            list(FIT_CAPSULE_CODE_ALLOWLIST),
        )
        self.assertEqual(
            [row["path"] for row in closure["config_files"]],
            list(FIT_CAPSULE_CONFIG_ALLOWLIST),
        )
        for forbidden_relative in (
            "spectral_utils/reconstruction_benchmark/external_final_answer.py",
            "spectral_utils/reconstruction_benchmark/external_ab.py",
            "spectral_utils/reconstruction_benchmark/external_evaluation.py",
            "spectral_utils/reconstruction_benchmark/evaluation.py",
            "spectral_utils/residual_graph_deem_labels.py",
            "spectral_utils/selectors/a6_pseudolabel_gates.py",
        ):
            self.assertFalse((code_root / forbidden_relative).exists())
        fit_root = self.root / "release" / "fit"
        key_path = self.root / "private_control" / "external-id-v2.key"
        key_path.parent.mkdir(parents=True)
        key_path.write_bytes(self.identity_key)
        key_path.chmod(0o600)
        controller_manifest = self.root / "private_control" / "MANIFEST.json"
        controller_manifest.write_text(
            '{"expected_incorrect":6,"source_root":"secret"}', encoding="utf-8"
        )
        policy = _worker_policy(
            code_root=code_root,
            input_root=input_root,
            fit_root=fit_root,
            forbidden_paths=[
                ("full_external_registry", self.registry_path),
                ("controller_identity_key", key_path),
                ("controller_provenance", controller_manifest),
                ("raw_telemetry_source", self.source),
                (
                    "preparation_adapter_module",
                    REPO / "spectral_utils/reconstruction_benchmark/external_final_answer.py",
                ),
                (
                    "postfreeze_evaluation_module",
                    REPO / "spectral_utils/reconstruction_benchmark/external_evaluation.py",
                ),
                (
                    "controller_ab_module",
                    REPO / "spectral_utils/reconstruction_benchmark/external_ab.py",
                ),
                (
                    "error_taxonomy_module",
                    REPO / "spectral_utils/residual_graph_deem_labels.py",
                ),
            ],
        )
        _launch_worker(
            code_root=code_root,
            input_root=input_root,
            fit_root=fit_root,
            release_id=release_id,
            build_id=build_id,
            cells=[worker_spec.cell_id],
            methods=PRIMARY_METHOD_IDS,
            policy=policy,
        )
        worker = json.loads(
            (fit_root / "WORKER_RESULT_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(worker["firewall_violations"], [])
        self.assertEqual(
            worker["denial_probes"],
            [
                {"probe_id": value, "read_denied": True}
                for value in (
                    "full_external_registry", "controller_identity_key",
                    "controller_provenance", "raw_telemetry_source",
                    "preparation_adapter_module", "postfreeze_evaluation_module",
                    "controller_ab_module", "error_taxonomy_module",
                )
            ],
        )
        self.assertTrue(worker["all_candidate_scores_present"])
        self.assertEqual(worker["method_ids"], list(PRIMARY_METHOD_IDS))
        self.assertEqual(len(worker["records"]), len(PRIMARY_METHOD_IDS))
        self.assertTrue(all(
            row["status"] in {"OK", "OK_FALLBACK"}
            for row in worker["records"]
        ))
        self.assertFalse((fit_root / "SCORE_FREEZE_MANIFEST.json").exists())
        score = load_npz_no_pickle(fit_root / worker["records"][0]["score_path"])
        self.assertEqual(
            set(score),
            {
                "row_ids", "score", "id_contract_version",
                "id_contract_sha256", "identity_key_id",
                "row_namespace_sha256", "row_roster_sha256",
            },
        )
        rendered = b"\n".join(
            file.read_bytes()
            for file in fit_root.rglob("*")
            if file.is_file()
        )
        for fragment in (
            b"deception_error_family_correct_row",
            b"step_contradiction_label_group",
            self.identity_key,
        ):
            self.assertNotIn(fragment, rendered)

    def test_firewall_records_caught_escape_attempts_and_allows_fit_io(self) -> None:
        safe_read = self.root / "capsule_read"
        safe_write = self.root / "capsule_write"
        forbidden = self.root / "controller_only" / "raw_labels.bin"
        safe_read.mkdir()
        safe_write.mkdir()
        forbidden.parent.mkdir()
        (safe_read / "allowed.txt").write_text("allowed", encoding="utf-8")
        forbidden.write_bytes(b"target-bearing sentinel")
        runtime_roots = (
            Path(sys.prefix), Path(sys.base_prefix), Path("/usr"),
            Path("/System"), Path("/Library"), Path("/dev"),
        )
        policy = build_fit_audit_policy(
            allowed_read_roots=(safe_read, *runtime_roots),
            allowed_read_files=(safe_read / "allowed.txt",),
            allowed_write_roots=(safe_write,),
            allowed_native_roots=runtime_roots[:-1],
            forbidden_probes=(({"probe_id": "raw_source", "path": str(forbidden)}),),
        )
        encoded = base64.b64encode(canonical_json_bytes(policy)).decode("ascii")
        firewall_path = (
            REPO / "spectral_utils/reconstruction_benchmark/fit_firewall.py"
        )
        child = r'''
import base64, ctypes, importlib.util, json, os, pathlib, socket, subprocess, sys
firewall_path, encoded, allowed, output, forbidden = sys.argv[1:]
spec = importlib.util.spec_from_file_location("_fit_firewall_probe", firewall_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
policy = json.loads(base64.b64decode(encoded).decode("utf-8"))
module.install_fit_audit_hook(policy)
probes = module.run_forbidden_read_probes(policy)
assert pathlib.Path(allowed).read_text(encoding="utf-8") == "allowed"
(pathlib.Path(output) / "success.txt").write_text("success", encoding="utf-8")
results = {}
def denied(name, operation):
    try:
        operation()
    except PermissionError:
        results[name] = True
    else:
        results[name] = False
denied("read", lambda: open(forbidden, "rb").read(1))
denied("write", lambda: open(str(pathlib.Path(forbidden).with_name("outside.bin")), "wb"))
denied("hardlink", lambda: os.link(forbidden, str(pathlib.Path(output) / "hardlink")))
denied("symlink", lambda: os.symlink(forbidden, str(pathlib.Path(output) / "symlink")))
denied("rename", lambda: os.rename(forbidden, str(pathlib.Path(output) / "renamed")))
denied("listdir", lambda: os.listdir(str(pathlib.Path(forbidden).parent)))
denied("subprocess", lambda: subprocess.run(["/usr/bin/true"], check=False))
denied("socket", lambda: socket.socket())
denied("ctypes", lambda: ctypes.CDLL(forbidden))
if hasattr(os, "fork"):
    denied("fork", lambda: os.fork())
print(json.dumps({"probes": probes, "results": results, "violations": module.fit_firewall_violations()}, sort_keys=True))
'''
        completed = subprocess.run(
            [
                sys.executable, "-I", "-B", "-c", child,
                str(firewall_path), encoded, str(safe_read / "allowed.txt"),
                str(safe_write), str(forbidden),
            ],
            check=False,
            capture_output=True,
            text=True,
            close_fds=True,
            stdin=subprocess.DEVNULL,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(result["probes"], [{"probe_id": "raw_source", "read_denied": True}])
        self.assertTrue(all(result["results"].values()), result)
        self.assertGreaterEqual(len(result["violations"]), len(result["results"]))
        self.assertEqual((safe_write / "success.txt").read_text(encoding="utf-8"), "success")
        self.assertTrue(forbidden.exists())

    def test_prepared_identity_namespace_and_version_tamper_are_rejected(self) -> None:
        path = self.root / "identity.npz"
        record = prepare_external_cell(
            registry=self.registry, spec=self.spec, repo=self.root, output_path=path,
            identity_key=self.identity_key,
        )
        safe_record = fit_safe_external_cell_record(record)
        wrong_version = dict(safe_record)
        wrong_version["id_contract_version"] = "reconstruction-external-opaque-id-v999"
        with self.assertRaisesRegex(RuntimeError, "identity binding drifted"):
            load_prepared_external_cell(
                artifact_path=path,
                record=wrong_version,
                identity_contract=safe_record["identity_contract"],
            )

        arrays = load_npz_no_pickle(path)
        arrays["row_namespace_sha256"] = np.asarray(["0" * 64], dtype="<U64")
        atomic_write_npz(path, arrays)
        tampered_artifact = dict(safe_record)
        tampered_artifact["artifact_sha256"] = sha256_file(path)
        with self.assertRaisesRegex(RuntimeError, "artifact identity binding drifted"):
            load_prepared_external_cell(
                artifact_path=path,
                record=tampered_artifact,
                identity_contract=safe_record["identity_contract"],
            )

        order_path = self.root / "identity_order.npz"
        order_record = prepare_external_cell(
            registry=self.registry, spec=self.spec, repo=self.root, output_path=order_path,
            identity_key=self.identity_key,
        )
        order_safe_record = fit_safe_external_cell_record(order_record)
        order_arrays = load_npz_no_pickle(order_path)
        order_arrays["row_ids"] = order_arrays["row_ids"][::-1]
        order_arrays["X_confidence"] = order_arrays["X_confidence"][::-1]
        atomic_write_npz(order_path, order_arrays)
        order_safe_record = {
            **order_safe_record,
            "artifact_sha256": sha256_file(order_path),
        }
        with self.assertRaisesRegex(RuntimeError, "canonical opaque order"):
            load_prepared_external_cell(
                artifact_path=order_path,
                record=order_safe_record,
                identity_contract=order_safe_record["identity_contract"],
            )

    def test_raw_adapter_permutation_canonicalizes_to_identical_prepared_bytes(self) -> None:
        left = self.root / "canonical_a.npz"
        right = self.root / "canonical_b.npz"
        first = prepare_external_cell(
            registry=self.registry, spec=self.spec, repo=self.root, output_path=left,
            identity_key=self.identity_key,
        )
        reverse = np.arange(len(self.raw_row_ids) - 1, -1, -1)

        def reversed_adapter(spec, sources):
            return RawFeatureCell(
                spec,
                self.raw_matrix[reverse],
                tuple(self.raw_row_ids[index] for index in reverse.tolist()),
                tuple(self.raw_group_ids[index] for index in reverse.tolist()),
                sources.feature_files,
            )

        with mock.patch.dict(RAW_ADAPTERS, {"synthetic_v1": reversed_adapter}):
            second = prepare_external_cell(
                registry=self.registry, spec=self.spec, repo=self.root, output_path=right,
                identity_key=self.identity_key,
            )
        self.assertEqual(sha256_file(left), sha256_file(right))
        self.assertEqual(first["prepared_matrix_sha256"], second["prepared_matrix_sha256"])
        self.assertEqual(first["row_roster_sha256"], second["row_roster_sha256"])
        arrays = load_npz_no_pickle(left)
        self.assertEqual(
            tuple(map(str, arrays["row_ids"].tolist())),
            tuple(sorted(map(str, arrays["row_ids"].tolist()))),
        )

    def test_missing_feature_rejects_whole_cell(self) -> None:
        broken = self.raw_matrix.copy()
        broken[3, 29] = np.nan
        with self.assertRaises(ExternalContractError) as raised:
            RawFeatureCell(
                self.spec, broken, self.raw_row_ids, self.raw_group_ids,
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
                self.spec, self.raw_matrix[:-1], self.raw_row_ids[:-1], self.raw_group_ids[:-1],
                (SourceFile(self.source, "source.bin", _digest(self.source)),),
            )
        self.assertEqual(raised.exception.status, ReadinessStatus.ROW_CONTRACT_MISMATCH)

    def test_second_preprocessing_pass_is_rejected(self) -> None:
        with self.assertRaises(ExternalContractError) as raised:
            RawFeatureCell(
                self.spec, self.raw_matrix, self.raw_row_ids, self.raw_group_ids,
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
            identity_key=self.identity_key,
        )
        safe_record = fit_safe_external_cell_record(record)
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        with self.assertRaises(RuntimeError):
            load_prepared_external_cell(
                artifact_path=path,
                record=safe_record,
                identity_contract=safe_record["identity_contract"],
            )

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

    def test_real_v3_verified_freeze_remains_canonical_for_label_gate(self) -> None:
        science_repo = REPO.parent / "reconstruction-science-run-v1"
        release_id = "2026-08-24_external_final_answer_v3_opaque"
        lane = (
            science_repo / "results/reconstruction_benchmark_v1/releases"
            / release_id / "build_A/external_final_answer"
        )
        freeze_path = lane / "fit/SCORE_FREEZE_MANIFEST.json"
        key_path = (
            science_repo / "results/reconstruction_benchmark_v1/private_control"
            / release_id / "external_final_answer/external-id-v2.key"
        )
        if not freeze_path.is_file() or not key_path.is_file():
            self.skipTest("real v3 external scientific freeze is not materialized")
        registry = load_external_registry(
            repo=science_repo,
            registry_path=(
                science_repo
                / "configs/reconstruction_benchmark_v1/external_final_answer.json"
            ),
            population_registry_path=(
                science_repo / "configs/reconstruction_benchmark_v1/populations.json"
            ),
        )
        input_root = lane / "inputs"
        fit_manifest = validate_fit_safe_input_manifest(
            input_root / "MANIFEST.json",
            repo=science_repo,
            input_root=input_root,
        )
        raw_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        validated = validate_scientific_score_freeze(
            freeze_path,
            registry=registry,
            repo=science_repo,
            input_root=input_root,
            fit_root=lane / "fit",
            input_manifest=fit_manifest,
        )
        self.assertEqual(
            set(validated) - set(raw_freeze), {"_validated_fit_manifest"}
        )
        restored = _restore_validated_score_freeze(
            validated, fit_manifest=fit_manifest,
        )
        self.assertEqual(restored, raw_freeze)
        identity_key = load_identity_key(key_path, create=False)
        assert_score_freeze(
            restored, registry=registry, identity_key=identity_key,
        )

        missing_augmentation = dict(raw_freeze)
        with self.assertRaisesRegex(RuntimeError, "derived-field roster"):
            _restore_validated_score_freeze(
                missing_augmentation, fit_manifest=fit_manifest,
            )

        extra_augmentation = {
            **validated,
            "_unexpected_validator_state": True,
        }
        with self.assertRaisesRegex(RuntimeError, "derived-field roster"):
            _restore_validated_score_freeze(
                extra_augmentation, fit_manifest=fit_manifest,
            )

        wrong_manifest = dict(fit_manifest)
        wrong_manifest["release_id"] = "wrong_release"
        wrong_attachment = {
            **validated,
            "_validated_fit_manifest": wrong_manifest,
        }
        with self.assertRaisesRegex(RuntimeError, "another fit manifest"):
            _restore_validated_score_freeze(
                wrong_attachment, fit_manifest=fit_manifest,
            )

        signed_field_tamper = {**restored, "build_id": "B"}
        with self.assertRaisesRegex(ExternalContractError, "payload hash failed"):
            assert_score_freeze(
                signed_field_tamper, registry=registry,
                identity_key=identity_key,
            )

    def test_atomic_evaluation_stage_cleans_failure_and_supports_retry(self) -> None:
        final_root = self.root / "external_final_answer/evaluation"
        (final_root / "labels").mkdir(parents=True)
        _remove_verified_empty_directory_tree(final_root)
        self.assertFalse(final_root.exists())

        failed = _AtomicEvaluationStage(final_root)
        (failed.path / "labels").mkdir()
        (failed.path / "labels/partial.npz").write_bytes(b"partial label output")
        failed.cleanup()
        self.assertFalse(failed.path.exists())
        self.assertFalse(final_root.exists())

        retry = _AtomicEvaluationStage(final_root)
        (retry.path / "labels").mkdir()
        (retry.path / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
        retry.commit()
        retry.cleanup()
        self.assertTrue(retry.committed)
        self.assertEqual(
            (final_root / "MANIFEST.json").read_text(encoding="utf-8"), "{}\n"
        )

        protected = self.root / "protected_evaluation"
        protected.mkdir()
        sentinel = protected / "MANIFEST.json"
        sentinel.write_text("do not overwrite\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "material output"):
            _remove_verified_empty_directory_tree(protected)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not overwrite\n")

        broken_link = self.root / "broken_evaluation_link"
        broken_link.symlink_to(self.root / "missing_evaluation_target")
        with self.assertRaisesRegex(FileExistsError, "not a directory"):
            _remove_verified_empty_directory_tree(broken_link)
        self.assertTrue(broken_link.is_symlink())

    def test_evaluator_preflight_failure_opens_no_labels_or_output(self) -> None:
        fit_manifest = {"payload_sha256": "fit-safe-manifest"}
        valid_signed_freeze = self._freeze()
        failure_cases = {
            "wrong_verifier_attachment": (
                {
                    **valid_signed_freeze,
                    "_validated_fit_manifest": {"payload_sha256": "wrong"},
                },
                "another fit manifest",
            ),
            "signed_payload_tamper": (
                {
                    **valid_signed_freeze,
                    "build_id": "tampered-after-signing",
                    "_validated_fit_manifest": fit_manifest,
                },
                "payload hash failed",
            ),
        }
        for case_id, (validated_freeze, message) in failure_cases.items():
            with self.subTest(case_id=case_id):
                release_root = self.root / case_id / "releases"
                final_root = (
                    release_root / "synthetic_release/build_A"
                    / "external_final_answer/evaluation"
                )
                external_evaluation_cli._ACTIVE_EVALUATION_STAGE = None
                argv = [
                    "evaluate_external_final_answer.py",
                    "--release-id", "synthetic_release",
                    "--build", "A",
                    "--release-root", str(release_root),
                    "--identity-key", str(self.root / "sealed.key"),
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(
                        external_evaluation_cli, "load_external_registry",
                        return_value=self.registry,
                    ),
                    mock.patch.object(
                        external_evaluation_cli, "load_identity_key",
                        return_value=self.identity_key,
                    ),
                    mock.patch.object(
                        external_evaluation_cli, "assert_external_ab_certificate",
                        return_value={},
                    ),
                    mock.patch.object(
                        external_evaluation_cli, "validate_fit_safe_input_manifest",
                        return_value=fit_manifest,
                    ),
                    mock.patch.object(
                        external_evaluation_cli, "validate_scientific_input_manifest",
                        return_value={"cells": []},
                    ),
                    mock.patch.object(
                        external_evaluation_cli, "assert_fit_safe_matches_preparation",
                    ),
                    mock.patch.object(
                        external_evaluation_cli, "validate_scientific_score_freeze",
                        return_value=validated_freeze,
                    ),
                    mock.patch.object(
                        external_evaluation_cli, "load_labels_after_score_freeze",
                    ) as label_loader,
                    mock.patch.object(
                        external_evaluation_cli, "_AtomicEvaluationStage",
                    ) as stage_constructor,
                ):
                    with self.assertRaisesRegex(
                        (RuntimeError, ExternalContractError), message,
                    ):
                        external_evaluation_cli.main()
                label_loader.assert_not_called()
                stage_constructor.assert_not_called()
                self.assertFalse(final_root.exists())
                self.assertEqual(
                    list(final_root.parent.glob(".evaluation.staging-*")), []
                )

    def test_ab_certificate_is_required_and_tree_tamper_is_refused(self) -> None:
        release_id = "synthetic_release"
        release_root = self.root / "releases"
        controller_root = (
            self.root / "private_control" / release_id / "external_final_answer"
        )
        for build_id in ("A", "B"):
            build = release_root / release_id / f"build_{build_id}" / "external_final_answer"
            inputs, fit = build / "inputs", build / "fit"
            inputs.mkdir(parents=True)
            fit.mkdir(parents=True)
            (build / "fit_capsule").mkdir(parents=True)
            (inputs / "MANIFEST.json").write_text(f"input-{build_id}", encoding="utf-8")
            (fit / "SCORE_FREEZE_MANIFEST.json").write_text(f"score-{build_id}", encoding="utf-8")
            provenance = controller_root / f"build_{build_id}" / "preparation_provenance"
            provenance.mkdir(parents=True)
            (provenance / "MANIFEST.json").write_text(
                f"provenance-{build_id}", encoding="utf-8"
            )
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
            build = release_root / release_id / f"build_{build_id}" / "external_final_answer"
            input_path = build / "inputs" / "MANIFEST.json"
            freeze_path = build / "fit" / "SCORE_FREEZE_MANIFEST.json"
            provenance = controller_root / f"build_{build_id}" / "preparation_provenance"
            preparation_path = provenance / "MANIFEST.json"
            builds[build_id] = {
                "input_manifest_sha256": sha256_file(input_path),
                "score_freeze_sha256": sha256_file(freeze_path),
                "preparation_manifest_sha256": sha256_file(preparation_path),
                "input_tree": canonical_tree_manifest(build / "inputs"),
                "fit_tree": canonical_tree_manifest(build / "fit"),
                "capsule_tree": canonical_tree_manifest(build / "fit_capsule"),
                "preparation_provenance_tree": canonical_tree_manifest(provenance),
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
            "identity_contract": self.fit_identity,
            "id_contract_version": ID_CONTRACT_VERSION,
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
            release_root=release_root,
            selected_build="A",
            registry=self.registry,
            repo=REPO,
        )
        tampered = release_root / release_id / "build_B" / "external_final_answer" / "fit" / "SCORE_FREEZE_MANIFEST.json"
        tampered.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed after A/B certification"):
            assert_external_ab_certificate(
                certificate_path,
                release_id=release_id,
                release_root=release_root,
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

    def test_production_registry_contains_only_opaque_prmbench_exclusions(self) -> None:
        registry_path = REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
        raw_bytes = registry_path.read_bytes()
        for leaked in (
            b"confidence_confidence_prm_train_p1_303",
            b"deception_deception_prm_test_p1_87",
            b"step_contradiction_step_contradiction_prm_test_p2_991",
        ):
            self.assertNotIn(leaked, raw_bytes)
        registry = load_external_registry(
            repo=REPO,
            registry_path=registry_path,
            population_registry_path=REPO / "configs/reconstruction_benchmark_v1/populations.json",
        )
        exclusions = registry.by_cell[
            "prmbench_response_qwen3_8b"
        ].excluded_raw_id_fingerprints
        self.assertEqual(len(exclusions), 3)
        self.assertTrue(all(value.startswith("xrfpv1_") for value in exclusions))

    def test_group_namespace_preserves_only_registered_cross_cell_linkage(self) -> None:
        def linked_spec(cell_id: str, model_id: str, slice_id: str) -> ExternalCellSpec:
            return ExternalCellSpec.from_mapping({
                "cell_id": cell_id,
                "population_id": "linked_population",
                "dataset_id": "processbench",
                "model_id": model_id,
                "slice_id": slice_id,
                "domain": "qa_response_detection",
                "comparison_group_id": "linked",
                "expected_rows": 2,
                "adapter_id": "synthetic_v1",
                "fit_policy": "run_if_compatible",
                "panel_role": "test",
                "source": {"kind": "explicit", "files": []},
            })

        left = linked_spec("gsm8k_model_a", "model_a", "gsm8k")
        right = linked_spec("gsm8k_model_b", "model_b", "gsm8k")
        other_slice = linked_spec("math_model_a", "model_a", "math")
        raw = dict(self.registry.raw)
        raw["population_aggregates"] = {
            "linked_population": {"enabled": True, "link_cells_by": "slice_id"},
        }
        raw["identity_contract"] = {
            **dict(raw["identity_contract"]),
            "group_namespace_by_population": {
                "linked_population": "population_slice",
            },
        }
        registry = ExternalRegistry(
            path=self.registry.path,
            sha256=self.registry.sha256,
            population_registry_path=self.registry.population_registry_path,
            population_registry_sha256=self.registry.population_registry_sha256,
            raw=raw,
            cells=(left, right, other_slice),
        )
        raw_rows = ("question-7-response", "question-8-response")
        raw_groups = ("question-7", "question-8")
        left_ids = apply_external_id_contract(
            registry, left, raw_rows, raw_groups, identity_key=self.identity_key
        )
        right_ids = apply_external_id_contract(
            registry, right, raw_rows, raw_groups, identity_key=self.identity_key
        )
        other_ids = apply_external_id_contract(
            registry, other_slice, raw_rows, raw_groups,
            identity_key=self.identity_key,
        )
        self.assertEqual(left_ids.group_ids, right_ids.group_ids)
        self.assertNotEqual(left_ids.row_ids, right_ids.row_ids)
        self.assertNotEqual(left_ids.group_ids, other_ids.group_ids)

    def test_labels_are_blocked_until_valid_score_freeze(self) -> None:
        with self.assertRaises(ExternalContractError) as blocked:
            load_labels_after_score_freeze(
                registry=self.registry,
                spec=self.spec,
                repo=self.root,
                score_freeze=self._freeze(complete=False),
                expected_row_ids=self.row_ids,
                expected_group_roster_commitment_sha256=self.group_commitment,
                identity_key=self.identity_key,
            )
        self.assertEqual(blocked.exception.status, ReadinessStatus.LABEL_PROVENANCE_BLOCKED)

        labels = load_labels_after_score_freeze(
            registry=self.registry,
            spec=self.spec,
            repo=self.root,
            score_freeze=self._freeze(),
            expected_row_ids=self.row_ids,
            expected_group_roster_commitment_sha256=self.group_commitment,
            identity_key=self.identity_key,
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
                expected_group_roster_commitment_sha256=self.group_commitment,
                identity_key=self.identity_key,
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
