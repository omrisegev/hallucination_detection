#!/usr/bin/env python3
"""Synthetic-only contract tests for the EDIS reconstruction lane."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.dufs_liu_feature_contract import CONTRACT_VERSION, dufs_liu_mixed_v2_matrix
from spectral_utils.reconstruction_benchmark import edis_preparation as preparation
from spectral_utils.reconstruction_benchmark.edis_ab import (
    _validate_evaluation_table_rosters,
    _validate_prediction_label_bindings,
    assert_ab_certificate,
    canonical_evaluation_table_sha256,
    verify_current_source_snapshot,
)
from spectral_utils.reconstruction_benchmark.edis_bootstrap import (
    grouped_paired_bootstrap_auroc_auprc,
    population_grouped_paired_bootstrap_auroc_auprc,
)
from spectral_utils.reconstruction_benchmark.edis_evaluation import (
    _common,
    _metric_rows,
    _prediction_row,
    _validate_partial_status_roster,
    evaluate,
    load_postfreeze_registry,
)
from spectral_utils.reconstruction_benchmark.edis_fit import load_fit_registry, load_prepared_cell
from spectral_utils.reconstruction_benchmark.edis_identity import (
    SharedEdisIdentityController,
    load_edis_identity_controller,
)
from spectral_utils.reconstruction_benchmark.edis_preparation import (
    EdisCellSpec,
    NOMINAL_FEATURE_NAMES,
    audit_nominal_matrix,
    assert_expected_preparation_status_roster,
    extract_target_free_cell,
    load_preparation_registry,
    prepare_build,
)
from spectral_utils.reconstruction_benchmark.io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.external_evaluation import (
    grouped_paired_bootstrap,
    population_grouped_paired_bootstrap,
)
from spectral_utils.reconstruction_reporting.schemas import validate_records
from spectral_utils.reconstruction_reporting.io import (
    read_parquet,
    read_tidy_csv,
    write_parquet,
    write_tidy_csv,
)
from scripts.reconstruction_benchmark.run_edis_methods import (
    EDIS_RUNTIME_READ_FILES,
    FIT_CAPSULE_MODULES,
    FIT_CAPSULE_RECONSTRUCTION_MODULES,
    _copy_fit_capsule,
    _launch_worker,
    _load_and_validate_controller_identity,
    _resolve_requested_identity_key,
    _validate_preparation_preflight,
    _worker_policy,
)


TARGET_FREE = REPO / "configs/reconstruction_benchmark_v1/edis_target_free.json"
POSTFREEZE = REPO / "configs/reconstruction_benchmark_v1/edis_postfreeze.json"


def _payload_sha(value) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _fake_features(candidate):
    base = float(candidate.get("telemetry_id", 0.0))
    return {
        name: base + 0.01 * (index + 1) + 0.0001 * base * (index + 1)
        for index, name in enumerate(NOMINAL_FEATURE_NAMES)
    }


def _raw_fixture(*, reverse: bool = False):
    keys = list(range(3))
    if reverse:
        keys.reverse()
    return {
        question: {
            "question": f"synthetic question {question}",
            "candidates": [
                {
                    "telemetry_id": 10 * question + candidate,
                    "token_entropies": [0.1 + 0.01 * index for index in range(8)],
                    "token_spilled_energies": [
                        0.2 + 0.01 * index for index in range(8)
                    ],
                    "token_logsumexp": None,
                    "top_k_logprobs": None,
                    "label": bool((question + candidate) % 2),
                    "label_lexical": False,
                }
                for candidate in range(2)
            ]
        }
        for question in keys
    }


class GuardedCandidate(dict):
    def __init__(self):
        super().__init__({
            "token_entropies": np.linspace(0.1, 1.1, 32),
            "token_spilled_energies": np.linspace(0.2, 1.2, 32),
            "token_logsumexp": np.linspace(-2.0, -1.0, 32),
            "top_k_logprobs": None,
            "label": True,
            "correct": True,
            "answer_key": "secret",
        })
        self.requested = []

    def get(self, key, default=None):
        self.requested.append(key)
        if key in {"label", "correct", "answer_key"}:
            raise AssertionError("target-like member was accessed")
        return super().get(key, default)


class EdisRegistryTests(unittest.TestCase):
    def test_real_registry_is_exact_and_postfreeze_is_separate(self):
        registry = load_preparation_registry(TARGET_FREE)
        post = load_postfreeze_registry(POSTFREEZE, registry)
        self.assertEqual(len(registry.cells), 12)
        self.assertEqual({cell.dataset_id for cell in registry.cells}, {"aime24", "amc23", "gsm8k", "math500"})
        self.assertEqual(len(post["cells"]), 12)
        target_free_text = TARGET_FREE.read_text(encoding="utf-8")
        self.assertNotIn("expected_correct", target_free_text)
        self.assertNotIn("expected_incorrect", target_free_text)
        self.assertFalse(post["evidence_boundary"]["headline_eligible"])
        self.assertEqual(post["evidence_boundary"]["status"], "DESCRIPTIVE_GATE_FAILED")
        expected = registry.expected_status_by_cell
        self.assertIsNotNone(expected)
        self.assertEqual(sum(status == "READY" for status in expected.values()), 4)
        self.assertEqual(
            sum(status == "BLOCKED_TRACE_BELOW_FROZEN_MIN" for status in expected.values()),
            8,
        )

    def test_nominal_feature_audit_allows_only_whole_view_absence(self):
        rows = [
            {name: float(row + column) for column, name in enumerate(NOMINAL_FEATURE_NAMES)}
            for row in range(6)
        ]
        matrix, names, absent = audit_nominal_matrix(rows, expected_rows=6, cell_id="synthetic")
        self.assertEqual(matrix.shape, (6, 30))
        self.assertEqual(names, NOMINAL_FEATURE_NAMES)
        self.assertEqual(absent, ())

        whole_absent = [dict(row) for row in rows]
        missing_name = NOMINAL_FEATURE_NAMES[-1]
        for row in whole_absent:
            row.pop(missing_name)
        matrix, names, absent = audit_nominal_matrix(whole_absent, expected_rows=6, cell_id="synthetic")
        self.assertEqual(matrix.shape, (6, 29))
        self.assertNotIn(missing_name, names)
        self.assertEqual(absent, (missing_name,))

        partial = [dict(row) for row in whole_absent]
        partial[0][missing_name] = 1.0
        with self.assertRaisesRegex(RuntimeError, "partially available"):
            audit_nominal_matrix(partial, expected_rows=6, cell_id="synthetic")

    def test_telemetry_extractor_never_requests_targets(self):
        candidate = GuardedCandidate()
        preparation._telemetry_features(candidate)
        self.assertEqual(
            set(candidate.requested),
            {"token_entropies", "token_spilled_energies", "token_logsumexp", "top_k_logprobs"},
        )


class EdisIdentityTests(unittest.TestCase):
    def test_shared_keyed_hmac_is_deterministic_and_key_separated(self):
        first = SharedEdisIdentityController(bytes(range(32)))
        same = SharedEdisIdentityController(bytes(range(32)))
        other = SharedEdisIdentityController(bytes(reversed(range(32))))
        namespace = {"lane_id": "edis", "scope": "cell", "cell_id": "c"}
        left = first.row_id(namespace=namespace, raw_identity="guessable:0:0")
        self.assertEqual(left, same.row_id(namespace=namespace, raw_identity="guessable:0:0"))
        self.assertNotEqual(left, other.row_id(namespace=namespace, raw_identity="guessable:0:0"))
        self.assertRegex(left, r"^xridv2_[0-9a-f]{64}$")
        self.assertNotIn("guessable", left)
        binding = dict(first.public_binding)
        self.assertRegex(binding["key_id"], r"^xkidv1_[0-9a-f]{64}$")
        self.assertNotIn("identity_key", binding)
        self.assertNotIn("path", json.dumps(binding))
        self.assertNotIn("group", json.dumps(binding).lower())
        self.assertEqual(
            first.private_identity_commitment_sha256,
            _payload_sha(first.private_identity_binding),
        )

    def test_controller_key_is_private_and_outside_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            controller = load_edis_identity_controller(
                private_control_root=root / "private",
                release_id="r1",
                create=True,
                release_root=root / "releases",
            )
            path = root / "private/r1/external-id-v2.key"
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(controller.identity_key), 32)
            again = load_edis_identity_controller(
                private_control_root=root / "private",
                release_id="r1",
                create=False,
                release_root=root / "releases",
            )
            self.assertEqual(controller.public_binding, again.public_binding)
            with self.assertRaisesRegex(ValueError, "outside every release"):
                load_edis_identity_controller(
                    private_control_root=root / "releases/controller",
                    release_id="r1",
                    create=True,
                    release_root=root / "releases",
                )


class EdisPreparationTests(unittest.TestCase):
    def _spec(self, source: Path) -> EdisCellSpec:
        return EdisCellSpec(
            lane_id="edis_aime_reconstruction_v1",
            dataset_id="gsm8k",
            population_id="synthetic",
            population_kind="pilot3",
            model_id="synthetic",
            temperature=0.2,
            cell_id="synthetic_t0p2",
            expected_rows=6,
            expected_questions=3,
            candidates_per_question=2,
            source_path=str(source),
            source_sha256="0" * 64,
            source_size_bytes=1,
            manifest_path="manifest.json",
            manifest_sha256="0" * 64,
            manifest_size_bytes=1,
        )

    def test_source_iteration_order_cannot_change_prepared_rows(self):
        controller = SharedEdisIdentityController(bytes(range(32)))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left_path, right_path = root / "left.pkl", root / "right.pkl"
            for path, value in ((left_path, _raw_fixture(reverse=False)), (right_path, _raw_fixture(reverse=True))):
                with path.open("wb") as handle:
                    pickle.dump(value, handle)
            with mock.patch.object(preparation, "_telemetry_features", side_effect=_fake_features):
                left = extract_target_free_cell(spec=self._spec(left_path), source_path=left_path, identity=controller)
                right = extract_target_free_cell(spec=self._spec(right_path), source_path=right_path, identity=controller)
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:], right[1:])
            self.assertTrue(all(row.startswith("xridv2_") for row in left[3]))
            self.assertEqual(
                set(left[4]),
                {
                    "group_membership_commitment_sha256",
                    "question_roster_commitment_sha256",
                    "row_roster_sha256",
                },
            )
            self.assertNotIn("xgidv2_", json.dumps(left[4]))

    def _synthetic_registry(self, root: Path) -> Path:
        datasets = []
        for dataset_id in ("aime24", "amc23", "gsm8k", "math500"):
            directory = root / "raw" / dataset_id
            directory.mkdir(parents=True, exist_ok=True)
            manifest = directory / "manifest.json"
            manifest.write_text('{"synthetic":true}\n', encoding="utf-8")
            cells = []
            for temperature, token in ((0.2, "0p2"), (0.6, "0p6"), (1.0, "1p0")):
                source = directory / f"raw_{token}.pkl"
                with source.open("wb") as handle:
                    pickle.dump(_raw_fixture(reverse=temperature == 0.6), handle)
                cells.append({
                    "cell_id": f"edis_{dataset_id}_t{token}",
                    "temperature": temperature,
                    "expected_rows": 6,
                    "source": {
                        "path": source.relative_to(root).as_posix(),
                        "sha256": sha256_file(source),
                        "size_bytes": source.stat().st_size,
                    },
                })
            datasets.append({
                "dataset_id": dataset_id,
                "population_id": f"synthetic_{dataset_id}",
                "population_kind": "synthetic",
                "model_id": "synthetic_model",
                "questions": 3,
                "samples_per_question_temperature": 2,
                "manifest": {
                    "path": manifest.relative_to(root).as_posix(),
                    "sha256": sha256_file(manifest),
                    "size_bytes": manifest.stat().st_size,
                },
                "cells": cells,
            })
        value = {
            "schema_version": "reconstruction-edis-target-free-registry-v1",
            "lane_id": "edis_aime_reconstruction_v1",
            "track": "multi_sample_inference",
            "feature_contract_id": CONTRACT_VERSION,
            "nominal_feature_count": 30,
            "feature_rule": "synthetic",
            "trace_status_contract": json.loads(
                TARGET_FREE.read_text(encoding="utf-8")
            )["trace_status_contract"],
            "fit_contract": {
                "method_roster": "all_13_primary_methods",
                "labels_available_to_fit": False,
                "class_counts_available_to_fit": False,
                "raw_source_paths_available_to_fit": False,
                "historical_scores_available_to_fit": False,
                "donors_available_to_fit": False,
                "score_semantics": "higher_is_incorrect",
                "mixed_v2_application_count": 1,
                "score_freeze_required_before_labels": True,
            },
            "identity_contract": {"required": "keyed_hmac_with_release_sealed_key"},
            "datasets": datasets,
        }
        path = root / "registry.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_ab_preparation_is_deterministic_and_fit_artifacts_hide_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = load_preparation_registry(self._synthetic_registry(root))
            identity = SharedEdisIdentityController(bytes(range(32)))
            snapshot = {"files": []}
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            with mock.patch.object(preparation, "_telemetry_features", side_effect=_fake_features):
                left = prepare_build(
                    release_id="r1", build_id="A", registry=registry, identity=identity,
                    source_root=root, release_root=root / "releases",
                    private_control_root=root / "private", preparation_source_snapshot=snapshot,
                )
                right = prepare_build(
                    release_id="r1", build_id="B", registry=registry, identity=identity,
                    source_root=root, release_root=root / "releases",
                    private_control_root=root / "private", preparation_source_snapshot=snapshot,
                )
            public_left = {key: value for key, value in left.items() if key not in {"build_id", "payload_sha256"}}
            public_right = {key: value for key, value in right.items() if key not in {"build_id", "payload_sha256"}}
            self.assertEqual(public_left, public_right)
            rendered = json.dumps(left).lower()
            for forbidden in (
                "source_path", "expected_correct", "expected_incorrect",
                "group", "gate_status",
            ):
                self.assertNotIn(forbidden, rendered)
            for cell in left["cells"]:
                artifact = root / "releases/r1/build_A/edis/inputs" / cell["artifact_path"]
                arrays = load_npz_no_pickle(artifact)
                self.assertNotIn("group_ids", arrays)
                self.assertNotIn("label", arrays)
                self.assertEqual(set(arrays), {
                    "X_confidence", "feature_names", "family_ids", "row_ids", "row_index",
                    "identity_contract_version", "identity_key_id",
                })

    def test_controller_key_must_match_both_public_and_private_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = load_preparation_registry(self._synthetic_registry(root))
            expected_identity = SharedEdisIdentityController(bytes(range(32)))
            snapshot = {"files": []}
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            with mock.patch.object(
                preparation, "_telemetry_features", side_effect=_fake_features
            ):
                fit_registry = prepare_build(
                    release_id="rkey", build_id="A", registry=registry,
                    identity=expected_identity, source_root=root,
                    release_root=root / "releases",
                    private_control_root=root / "private",
                    preparation_source_snapshot=snapshot,
                )
            private_path = root / "private/rkey/edis/build_A/PREPARATION_PROVENANCE.json"
            private = json.loads(private_path.read_text(encoding="utf-8"))
            key_path = root / "private/rkey/external-id-v2.key"
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(bytes(range(32)))
            key_path.chmod(0o600)
            validated = _load_and_validate_controller_identity(
                private_control_root=root / "private", release_id="rkey",
                release_root=root / "releases", repo=REPO,
                fit_registry=fit_registry,
                private_provenance=private,
            )
            self.assertEqual(validated.public_binding, expected_identity.public_binding)
            key_path.write_bytes(bytes(reversed(range(32))))
            key_path.chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _load_and_validate_controller_identity(
                    private_control_root=root / "private", release_id="rkey",
                    release_root=root / "releases", repo=REPO,
                    fit_registry=fit_registry,
                    private_provenance=private,
                )
            key_path.unlink()
            with self.assertRaises(FileNotFoundError):
                _load_and_validate_controller_identity(
                    private_control_root=root / "private", release_id="rkey",
                    release_root=root / "releases", repo=REPO,
                    fit_registry=fit_registry,
                    private_provenance=private,
                )

    def test_custom_identity_key_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = _resolve_requested_identity_key(
                requested=None,
                private_control_root=root / "private",
                release_id="rpath",
            )
            self.assertEqual(canonical, (root / "private/rpath/external-id-v2.key").resolve())
            with self.assertRaisesRegex(ValueError, "sealed controller key"):
                _resolve_requested_identity_key(
                    requested=root / "releases/rpath/build_A/edis/inputs/key.bin",
                    private_control_root=root / "private",
                    release_id="rpath",
                )

    def test_fit_preflight_rejects_stale_preparation_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = []
            for relative in preparation.PREPARATION_SOURCE_PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"synthetic {relative}\n", encoding="utf-8")
                files.append({"path": relative, "sha256": sha256_file(path)})
            self.assertIn(
                "spectral_utils/fusion_utils.py", preparation.PREPARATION_SOURCE_PATHS
            )
            self.assertIn(
                "spectral_utils/streaming_utils.py", preparation.PREPARATION_SOURCE_PATHS
            )
            snapshot = {"files": files}
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            private = {"preparation_source_snapshot": snapshot}
            _validate_preparation_preflight(
                private_provenance=private, repo=root, build_id="A"
            )
            (root / "spectral_utils/fusion_utils.py").write_text(
                "changed\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "changed or is missing"):
                _validate_preparation_preflight(
                    private_provenance=private, repo=root, build_id="A"
                )

    def test_cross_temperature_linkage_requires_identical_question_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry_path = self._synthetic_registry(root)
            registry_raw = json.loads(registry_path.read_text(encoding="utf-8"))
            cell = registry_raw["datasets"][0]["cells"][1]
            source = root / cell["source"]["path"]
            with source.open("rb") as handle:
                raw = pickle.load(handle)
            raw[1]["question"] = "changed question at the same ordinal"
            with source.open("wb") as handle:
                pickle.dump(raw, handle)
            cell["source"]["sha256"] = sha256_file(source)
            cell["source"]["size_bytes"] = source.stat().st_size
            registry_path.write_text(json.dumps(registry_raw), encoding="utf-8")
            registry = load_preparation_registry(registry_path)
            snapshot = {"files": []}
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            with mock.patch.object(
                preparation, "_telemetry_features", side_effect=_fake_features
            ), self.assertRaisesRegex(
                RuntimeError, "question content/order differs across temperatures"
            ):
                prepare_build(
                    release_id="rlink", build_id="A", registry=registry,
                    identity=SharedEdisIdentityController(bytes(range(32))),
                    source_root=root, release_root=root / "releases",
                    private_control_root=root / "private",
                    preparation_source_snapshot=snapshot,
                )

    def test_restricted_fit_worker_denies_controller_and_raw_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = load_preparation_registry(self._synthetic_registry(root))
            identity = SharedEdisIdentityController(bytes(range(32)))
            snapshot = {"files": []}
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            with mock.patch.object(
                preparation, "_telemetry_features", side_effect=_fake_features
            ):
                prepare_build(
                    release_id="rfit", build_id="A", registry=registry,
                    identity=identity, source_root=root,
                    release_root=root / "releases",
                    private_control_root=root / "private",
                    preparation_source_snapshot=snapshot,
                )
            input_root = root / "releases/rfit/build_A/edis/inputs"
            capsule = root / "releases/rfit/build_A/edis/fit_capsule"
            code_root = _copy_fit_capsule(capsule)
            expected_python = {
                "spectral_utils/__init__.py",
                "spectral_utils/selectors/__init__.py",
                "spectral_utils/selectors/a2_groupfs.py",
                "spectral_utils/reconstruction_benchmark/__init__.py",
                "scripts/reconstruction_benchmark/edis_fit_worker.py",
                *{f"spectral_utils/{name}" for name in FIT_CAPSULE_MODULES},
                *{
                    f"spectral_utils/reconstruction_benchmark/{name}"
                    for name in FIT_CAPSULE_RECONSTRUCTION_MODULES
                },
            }
            actual_python = {
                path.relative_to(code_root).as_posix()
                for path in code_root.rglob("*.py")
            }
            self.assertEqual(actual_python, expected_python)
            self.assertFalse((code_root / "spectral_utils/reconstruction_benchmark/edis_evaluation.py").exists())
            self.assertFalse((code_root / "spectral_utils/reconstruction_benchmark/edis_preparation.py").exists())
            fit_root = root / "releases/rfit/build_A/edis/fit"
            controller_secret = root / "private/rfit/external-id-v2.key"
            controller_secret.parent.mkdir(parents=True, exist_ok=True)
            controller_secret.write_bytes(bytes(range(32)))
            raw_source = root / registry.cells[0].source_path
            policy = _worker_policy(
                code_root=code_root,
                input_root=input_root,
                fit_root=fit_root,
                forbidden_paths=(
                    ("controller_identity_key", controller_secret),
                    ("raw_telemetry_source", raw_source),
                ),
            )
            self.assertEqual(
                [str(path.resolve()) for path in EDIS_RUNTIME_READ_FILES],
                ["/proc/self/maps"],
            )
            self.assertIn("/proc/self/maps", policy["allowed_read_files"])
            self.assertNotIn("/proc", policy["allowed_read_roots"])
            self.assertNotIn("/proc/self", policy["allowed_read_roots"])
            selected_cell = registry.cells[0].cell_id
            _launch_worker(
                code_root=code_root,
                input_root=input_root,
                fit_root=fit_root,
                release_id="rfit",
                build_id="A",
                policy=policy,
                cells=(selected_cell,),
                # dufs_pf_lsml exercises the real restricted PyTorch import
                # that performs a best-effort open('/proc/self/maps').
                methods=("equal_feature_mean", "dufs_pf_lsml"),
            )
            worker = json.loads(
                (fit_root / "WORKER_RESULT_MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertEqual(worker["firewall_violations"], [])
            self.assertEqual(worker["denial_probes"], [
                {"probe_id": "controller_identity_key", "read_denied": True},
                {"probe_id": "raw_telemetry_source", "read_denied": True},
            ])
            self.assertEqual(
                worker["method_ids"],
                ["equal_feature_mean", "dufs_pf_lsml"],
            )
            self.assertEqual(worker["cell_ids"], [selected_cell])
            rendered = json.dumps(worker).lower()
            self.assertNotIn("group_id", rendered)
            self.assertNotIn("label", rendered)
            self.assertEqual(len(worker["records"]), 2)
            for record in worker["records"]:
                score = load_npz_no_pickle(fit_root / record["score_path"])
                self.assertEqual(set(score), {"row_ids", "score"})

    def test_source_tamper_becomes_visible_blocked_asset_without_substitution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = load_preparation_registry(self._synthetic_registry(root))
            source = root / registry.cells[0].source_path
            source.write_bytes(source.read_bytes() + b"tamper")
            snapshot = {"files": []}
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            with mock.patch.object(
                preparation, "_telemetry_features", side_effect=_fake_features
            ):
                manifest = prepare_build(
                    release_id="r2", build_id="A", registry=registry,
                    identity=SharedEdisIdentityController(bytes(range(32))),
                    source_root=root, release_root=root / "releases",
                    private_control_root=root / "private", preparation_source_snapshot=snapshot,
                )
            self.assertTrue(manifest["partial_descriptive_build"])
            self.assertFalse(manifest["scientific_full_build"])
            self.assertFalse(manifest["aggregate_metrics_allowed"])
            self.assertEqual(manifest["ready_cell_count"], 11)
            status = preparation.load_preparation_status(
                root / "releases/r2/build_A/edis/PREPARATION_STATUS.json"
            )
            blocked = [row for row in status["cells"] if row["status"] == "BLOCKED_ASSET"]
            self.assertEqual([row["cell_id"] for row in blocked], [registry.cells[0].cell_id])
            self.assertNotIn(registry.cells[0].source_path, json.dumps(status))
            self.assertEqual(status["dataset_aggregate_status"], "BLOCKED_INCOMPLETE_CELL_ROSTER")

    def test_materialized_roster_asset_drift_publishes_status_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry_path = self._synthetic_registry(root)
            raw_registry = json.loads(registry_path.read_text(encoding="utf-8"))
            flat_cells = [
                cell
                for dataset in raw_registry["datasets"]
                for cell in dataset["cells"]
            ]
            ready_indexes = {3, 6, 9, 10}
            blocked_rows = []
            ready_ids = []
            for index, cell in enumerate(flat_cells):
                if index in ready_indexes:
                    ready_ids.append(cell["cell_id"])
                    continue
                source = root / cell["source"]["path"]
                with source.open("rb") as handle:
                    payload = pickle.load(handle)
                candidate = payload[0]["candidates"][0]
                candidate["token_entropies"] = candidate["token_entropies"][:7]
                candidate["token_spilled_energies"] = candidate[
                    "token_spilled_energies"
                ][:7]
                with source.open("wb") as handle:
                    pickle.dump(payload, handle)
                cell["source"]["sha256"] = sha256_file(source)
                cell["source"]["size_bytes"] = source.stat().st_size
                blocked_rows.append({
                    "cell_id": cell["cell_id"],
                    "status": "BLOCKED_TRACE_BELOW_FROZEN_MIN",
                })
            raw_registry["target_free_status_roster_contract"] = {
                "contract_id": "edis-materialized-status-roster-v1-2026-08-24",
                "registered_stage": (
                    "after_target_free_telemetry_audit_before_any_scores_or_labels"
                ),
                "labels_used": False,
                "ready_cell_ids": ready_ids,
                "blocked_cells": blocked_rows,
                "blocked_labels_may_be_opened": False,
                "dataset_or_task_aggregates_allowed": False,
                "headline_or_publication_eligible": False,
            }
            registry_path.write_text(json.dumps(raw_registry), encoding="utf-8")
            registry = load_preparation_registry(registry_path)
            # Tamper one preregistered READY asset after pinning it.
            ready_spec = registry.by_cell[ready_ids[0]]
            ready_source = root / ready_spec.source_path
            ready_source.write_bytes(ready_source.read_bytes() + b"tamper")
            snapshot = {"files": []}
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            with mock.patch.object(
                preparation, "_telemetry_features", side_effect=_fake_features
            ):
                manifest = prepare_build(
                    release_id="rstatusonly", build_id="A", registry=registry,
                    identity=SharedEdisIdentityController(bytes(range(32))),
                    source_root=root, release_root=root / "releases",
                    private_control_root=root / "private",
                    preparation_source_snapshot=snapshot,
                )
            lane = root / "releases/rstatusonly/build_A/edis"
            status = preparation.load_preparation_status(
                lane / "PREPARATION_STATUS.json"
            )
            self.assertTrue(manifest["status_only_build"])
            self.assertFalse(manifest["fit_registry_available"])
            self.assertFalse(status["status_roster_contract_match"])
            self.assertEqual(
                [row["cell_id"] for row in status["cells"] if row["status"] == "BLOCKED_ASSET"],
                [ready_spec.cell_id],
            )
            self.assertFalse((lane / "inputs/FIT_REGISTRY.json").exists())
            self.assertTrue((lane / "FIT_UNAVAILABLE.json").is_file())

    def test_short_trace_blocks_whole_cell_and_has_ab_stable_public_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry_path = self._synthetic_registry(root)
            raw_registry = json.loads(registry_path.read_text(encoding="utf-8"))
            source_row = raw_registry["datasets"][0]["cells"][0]["source"]
            source = root / source_row["path"]
            with source.open("rb") as handle:
                raw = pickle.load(handle)
            candidate = raw[0]["candidates"][0]
            candidate["token_entropies"] = candidate["token_entropies"][:7]
            candidate["token_spilled_energies"] = candidate[
                "token_spilled_energies"
            ][:7]
            with source.open("wb") as handle:
                pickle.dump(raw, handle)
            source_row["sha256"] = sha256_file(source)
            source_row["size_bytes"] = source.stat().st_size
            registry_path.write_text(json.dumps(raw_registry), encoding="utf-8")
            registry = load_preparation_registry(registry_path)
            snapshot = {"files": []}
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            identity = SharedEdisIdentityController(bytes(range(32)))
            with mock.patch.object(
                preparation, "_telemetry_features", side_effect=_fake_features
            ):
                for build in ("A", "B"):
                    prepare_build(
                        release_id="rshort", build_id=build, registry=registry,
                        identity=identity, source_root=root,
                        release_root=root / "releases",
                        private_control_root=root / "private",
                        preparation_source_snapshot=snapshot,
                    )
            statuses = [
                preparation.load_preparation_status(
                    root / f"releases/rshort/build_{build}/edis/PREPARATION_STATUS.json"
                )
                for build in ("A", "B")
            ]
            self.assertEqual(
                statuses[0]["status_commitment_sha256"],
                statuses[1]["status_commitment_sha256"],
            )
            self.assertNotEqual(statuses[0]["payload_sha256"], statuses[1]["payload_sha256"])
            blocked = [
                row for row in statuses[0]["cells"]
                if row["status"] == "BLOCKED_TRACE_BELOW_FROZEN_MIN"
            ]
            self.assertEqual(len(blocked), 1)
            self.assertEqual(blocked[0]["blocking_row_count"], 1)
            self.assertEqual(blocked[0]["frozen_minimum_trace_tokens"], 8)
            self.assertEqual(len(blocked[0]["opaque_blocking_row_ids"]), 1)
            self.assertRegex(blocked[0]["opaque_blocking_row_ids"][0], r"^xridv2_[0-9a-f]{64}$")
            self.assertEqual(set(blocked[0]["nominal_feature_finite_counts"]), set(NOMINAL_FEATURE_NAMES))
            rendered = json.dumps(statuses[0])
            for forbidden in ("source_question", "candidate_index", "question_fingerprint"):
                self.assertNotIn(forbidden, rendered)
            fit_registry = load_fit_registry(
                root / "releases/rshort/build_A/edis/inputs/FIT_REGISTRY.json"
            )
            self.assertEqual(fit_registry["ready_cell_count"], 11)
            self.assertTrue(fit_registry["partial_descriptive_build"])
            self.assertFalse(fit_registry["aggregate_metrics_allowed"])

    def test_malformed_feature_channels_block_cell_before_reduction(self):
        mutations = {
            "logsumexp_length": lambda candidate: candidate.update({
                "token_logsumexp": [0.1] * 7,
            }),
            "topk_time": lambda candidate: candidate.update({
                "top_k_logprobs": {
                    "ids": np.zeros((7, 2), dtype=np.int32),
                    "logprobs": np.zeros((7, 2), dtype=np.float32),
                },
            }),
            "topk_shape": lambda candidate: candidate.update({
                "top_k_logprobs": {
                    "ids": np.zeros((8, 3), dtype=np.int32),
                    "logprobs": np.zeros((8, 2), dtype=np.float32),
                },
            }),
            "topk_nonfinite": lambda candidate: candidate.update({
                "top_k_logprobs": {
                    "ids": np.zeros((8, 2), dtype=np.int32),
                    "logprobs": np.asarray([[0.0, -1.0]] * 7 + [[np.nan, -1.0]]),
                },
            }),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "raw.pkl"
                raw = _raw_fixture()
                mutate(raw[0]["candidates"][0])
                with path.open("wb") as handle:
                    pickle.dump(raw, handle)
                with mock.patch.object(
                    preparation, "_telemetry_features", side_effect=_fake_features
                ), self.assertRaises(preparation.EdisCellBlocked) as caught:
                    extract_target_free_cell(
                        spec=self._spec(path), source_path=path,
                        identity=SharedEdisIdentityController(bytes(range(32))),
                    )
                self.assertEqual(caught.exception.status, "BLOCKED_MALFORMED_TELEMETRY")

    def test_failed_stage_is_clean_and_exact_empty_legacy_tree_is_recoverable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = load_preparation_registry(self._synthetic_registry(root))
            snapshot = {"files": []}
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            stale = root / "releases/rretry/build_A/edis/inputs/cells"
            stale.mkdir(parents=True)
            with mock.patch.object(
                preparation, "_telemetry_features", side_effect=RuntimeError("synthetic failure")
            ), self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                prepare_build(
                    release_id="rretry", build_id="A", registry=registry,
                    identity=SharedEdisIdentityController(bytes(range(32))),
                    source_root=root, release_root=root / "releases",
                    private_control_root=root / "private",
                    preparation_source_snapshot=snapshot,
                )
            self.assertFalse((root / "releases/rretry/build_A/edis").exists())
            self.assertTrue(
                (root / "private/rretry/edis/recovery/public_build_A_zero_file_residue").is_dir()
            )
            self.assertEqual(
                list((root / "releases/rretry/build_A").glob(".edis_build_A_preparing_*")),
                [],
            )
            with mock.patch.object(
                preparation, "_telemetry_features", side_effect=_fake_features
            ):
                result = prepare_build(
                    release_id="rretry", build_id="A", registry=registry,
                    identity=SharedEdisIdentityController(bytes(range(32))),
                    source_root=root, release_root=root / "releases",
                    private_control_root=root / "private",
                    preparation_source_snapshot=snapshot,
                )
            self.assertEqual(result["ready_cell_count"], 12)
            self.assertTrue(
                (root / "releases/rretry/build_A/edis/inputs/FIT_REGISTRY.json").is_file()
            )


class EdisFitAndReportingTests(unittest.TestCase):
    @staticmethod
    def _evaluation_kwargs(root: Path) -> dict:
        return {
            "release_id": "r-evaluation",
            "build_id": "A",
            "release_root": root / "releases",
            "private_control_root": root / "private",
            "source_root": root,
            "preparation_registry_path": root / "registry.json",
            "postfreeze_registry_path": root / "postfreeze.json",
            "identity": SharedEdisIdentityController(bytes(range(32))),
            "repo": root,
        }

    def test_partial_prediction_rows_round_trip_csv_and_parquet_as_booleans(self):
        base = _common(
            release_id="r", run_id="r::edis::A::postfreeze", spec=None,
            dataset_id="amc23", population_id="edis_amc23_full_v1",
            cell_id="edis_amc23_t0p2", slice_id="temperature_0p2",
            cohort_id="cohort::synthetic", method_id="iu_pcr",
            method_version_id="iu-pcr-v1", comparison_group_id="comparison::synthetic",
        )
        labels = np.asarray([0, 1, 1, 0], dtype=np.int8)
        rows = [
            _prediction_row(
                base=base,
                row_id=f"xridv2_{index}",
                group_id=f"xgidv2_{index // 2}",
                score=0.1 * index,
                label=label,
                fallback_used=False,
                score_hash="a" * 64,
            )
            for index, label in enumerate(labels)
        ]
        validated = validate_records("predictions", rows)
        self.assertEqual([row["label"] for row in validated], [False, True, True, False])
        self.assertTrue(all(type(row["label"]) is bool for row in validated))
        with self.assertRaisesRegex(ValueError, "must be binary"):
            _prediction_row(
                base=base, row_id="xridv2_bad", group_id="xgidv2_bad",
                score=0.0, label=np.int8(2), fallback_used=False,
                score_hash="a" * 64,
            )
        for invalid in (0.9, 1.9, "1", np.float64(1.0)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                TypeError, "bool/integer"
            ):
                _prediction_row(
                    base=base, row_id="xridv2_bad", group_id="xgidv2_bad",
                    score=0.0, label=invalid, fallback_used=False,
                    score_hash="a" * 64,
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "predictions.csv"
            parquet_path = root / "predictions.parquet"
            write_tidy_csv(csv_path, "predictions", rows)
            write_parquet(parquet_path, "predictions", rows)
            csv_rows = read_tidy_csv(csv_path, "predictions")
            parquet_rows = read_parquet(parquet_path, "predictions")
            self.assertEqual(csv_rows, parquet_rows)
            self.assertEqual(
                [row["label"] for row in parquet_rows],
                [False, True, True, False],
            )

    def test_evaluation_failure_discards_only_unpublished_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_labels = root / "sealed-source-labels.bin"
            label_bytes = b"synthetic source labels remain recoverable"
            source_labels.write_bytes(label_bytes)

            def fail_after_partial_writes(**kwargs):
                output = kwargs["output"]
                labels = output / "labels"
                labels.mkdir()
                (labels / "ready-cell.npz").write_bytes(source_labels.read_bytes())
                (output / "predictions.csv").write_text(
                    "synthetic unpublished predictions\n", encoding="utf-8"
                )
                raise RuntimeError("synthetic Parquet failure")

            with mock.patch(
                "spectral_utils.reconstruction_benchmark.edis_evaluation._evaluate_in_stage",
                side_effect=fail_after_partial_writes,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic Parquet failure"):
                    evaluate(**self._evaluation_kwargs(root))

            final = root / "releases/r-evaluation/build_A/edis/evaluation"
            self.assertFalse(final.exists())
            self.assertEqual(source_labels.read_bytes(), label_bytes)
            self.assertEqual(
                list(final.parent.glob(".evaluation.staging-*")),
                [],
            )

    def test_evaluation_no_clobber_preserves_material_partial_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final = root / "releases/r-evaluation/build_A/edis/evaluation"
            labels = final / "labels"
            labels.mkdir(parents=True)
            expected: dict[Path, bytes] = {}
            for index in range(4):
                path = labels / f"ready-{index}.npz"
                expected[path] = f"label-bytes-{index}".encode("ascii")
                path.write_bytes(expected[path])
            predictions = final / "predictions.csv"
            expected[predictions] = b"partial predictions\n"
            predictions.write_bytes(expected[predictions])

            with mock.patch(
                "spectral_utils.reconstruction_benchmark.edis_evaluation._evaluate_in_stage"
            ) as core:
                with self.assertRaisesRegex(FileExistsError, "material output"):
                    evaluate(**self._evaluation_kwargs(root))
                core.assert_not_called()

            self.assertEqual(
                {path: path.read_bytes() for path in expected},
                expected,
            )
            self.assertEqual(
                list(final.parent.glob(".evaluation.staging-*")),
                [],
            )

    def test_evaluation_commits_complete_stage_over_empty_retry_scaffold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final = root / "releases/r-evaluation/build_A/edis/evaluation"
            (final / "labels").mkdir(parents=True)
            label_bytes = b"complete staged labels"

            def succeed(**kwargs):
                output = kwargs["output"]
                labels = output / "labels"
                labels.mkdir()
                (labels / "ready-cell.npz").write_bytes(label_bytes)
                atomic_write_json(output / "MANIFEST.json", {"status": "complete"})
                return {"status": "complete"}

            with mock.patch(
                "spectral_utils.reconstruction_benchmark.edis_evaluation._evaluate_in_stage",
                side_effect=succeed,
            ):
                result = evaluate(**self._evaluation_kwargs(root))

            self.assertEqual(result, {"status": "complete"})
            self.assertEqual((final / "labels/ready-cell.npz").read_bytes(), label_bytes)
            self.assertTrue((final / "MANIFEST.json").is_file())
            self.assertEqual(
                list(final.parent.glob(".evaluation.staging-*")),
                [],
            )

    def test_evaluation_commit_race_preserves_published_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final = root / "releases/r-evaluation/build_A/edis/evaluation"
            winner_bytes = b"complete competing evaluation"

            def finish_stage(**kwargs):
                (kwargs["output"] / "MANIFEST.json").write_text(
                    '{"status":"complete"}\n', encoding="utf-8"
                )
                return {"status": "complete"}

            def competing_publish(_source, target):
                target = Path(target)
                target.mkdir()
                (target / "winner.bin").write_bytes(winner_bytes)
                raise OSError("synthetic nonempty target race")

            with mock.patch(
                "spectral_utils.reconstruction_benchmark.edis_evaluation._evaluate_in_stage",
                side_effect=finish_stage,
            ), mock.patch(
                "spectral_utils.reconstruction_benchmark.edis_evaluation.os.replace",
                side_effect=competing_publish,
            ):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    evaluate(**self._evaluation_kwargs(root))

            self.assertEqual((final / "winner.bin").read_bytes(), winner_bytes)
            self.assertEqual(
                list(final.parent.glob(".evaluation.staging-*")),
                [],
            )

    def test_partial_status_roster_accepts_exact_4_8_and_rejects_drift(self):
        preparation_registry = load_preparation_registry(TARGET_FREE)
        expected = preparation_registry.expected_status_by_cell
        rows = [
            {
                "cell_id": cell.cell_id,
                "status": expected[cell.cell_id],
                "status_detail": "synthetic target-free status",
            }
            for cell in preparation_registry.cells
        ]
        ready = [row["cell_id"] for row in rows if row["status"] == "READY"]
        certificate = {
            "scientific_full": False,
            "descriptive_partial": True,
            "headline_eligible": False,
            "aggregate_metrics_allowed": False,
            "cell_ids": ready,
            "registered_cell_statuses": rows,
            "registered_cell_count": 12,
            "ready_cell_count": 4,
            "blocked_cell_count": 8,
        }
        self.assertEqual(
            _validate_partial_status_roster(
                certificate=certificate, preparation=preparation_registry
            ),
            expected,
        )
        assert_expected_preparation_status_roster(
            registry=preparation_registry, status={"cells": rows}
        )
        eleven_ready = [dict(row) for row in rows]
        eleven_ready[0]["status"] = "READY"
        with self.assertRaisesRegex(RuntimeError, "preregistered 4-ready/8-blocked"):
            assert_expected_preparation_status_roster(
                registry=preparation_registry, status={"cells": eleven_ready}
            )
        drifted = {**certificate, "registered_cell_statuses": [dict(row) for row in rows]}
        drifted["registered_cell_statuses"][0]["status"] = "READY"
        with self.assertRaisesRegex(RuntimeError, "roster drifted"):
            _validate_partial_status_roster(
                certificate=drifted, preparation=preparation_registry
            )

    def test_partial_table_verifier_rejects_any_cartesian_omission(self):
        preparation_registry = load_preparation_registry(TARGET_FREE)
        expected = preparation_registry.expected_status_by_cell
        status_rows = [
            {
                "cell_id": cell.cell_id,
                "status": expected[cell.cell_id],
                "expected_rows": 1,
            }
            for cell in preparation_registry.cells
        ]
        ready = [row["cell_id"] for row in status_rows if row["status"] == "READY"]
        methods = list(__import__(
            "spectral_utils.reconstruction_benchmark.methods",
            fromlist=["PRIMARY_METHOD_IDS"],
        ).PRIMARY_METHOD_IDS)
        certificate = {
            "registered_cell_statuses": status_rows,
            "cell_ids": ready,
        }
        coverage = []
        predictions = []
        metrics = []
        contrasts = []
        for row in status_rows:
            cell = row["cell_id"]
            is_ready = row["status"] == "READY"
            for method in methods:
                coverage.append({
                    "cell_id": cell, "method_id": method,
                    "status": "CONTEXT_ONLY" if is_ready else "INPUT_INVALID",
                    "expected_n": 1,
                    "eligible_n": 1 if is_ready else 0,
                    "scored_n": 1 if is_ready else 0,
                    "excluded_n": 0,
                    "failed_n": 0 if is_ready else 1,
                    "coverage_fraction": 1.0 if is_ready else 0.0,
                })
                if is_ready:
                    predictions.append({
                        "cell_id": cell, "method_id": method,
                        "row_id": f"xridv2_{_payload_sha([cell, method])}",
                    })
                    for metric in ("auroc", "auprc"):
                        metrics.append({
                            "cell_id": cell, "method_id": method,
                            "metric_id": metric, "aggregation_level": "cell",
                        })
                        if method != "iu_pcr":
                            contrasts.append({
                                "cell_id": cell, "method_id": method,
                                "metric_id": metric, "aggregation_level": "cell",
                            })
        tables = {
            "coverage": coverage, "predictions": predictions,
            "metrics": metrics, "contrasts": contrasts,
        }
        _validate_evaluation_table_rosters(
            table_rows=tables, score_certificate=certificate,
            descriptive_partial=True,
        )
        for table in ("coverage", "metrics", "contrasts", "predictions"):
            omitted = {key: list(value) for key, value in tables.items()}
            omitted[table] = omitted[table][1:]
            with self.subTest(table=table), self.assertRaises(RuntimeError):
                _validate_evaluation_table_rosters(
                    table_rows=omitted, score_certificate=certificate,
                    descriptive_partial=True,
                )
        truncated = {key: list(value) for key, value in tables.items()}
        first_prediction = truncated["predictions"].pop(0)
        pair = (first_prediction["cell_id"], first_prediction["method_id"])
        truncated["coverage"] = [dict(row) for row in truncated["coverage"]]
        for row in truncated["coverage"]:
            if (row["cell_id"], row["method_id"]) == pair:
                row.update({
                    "expected_n": 0, "eligible_n": 0, "scored_n": 0,
                    "coverage_fraction": 0.0,
                })
        with self.assertRaisesRegex(RuntimeError, "expected_rows"):
            _validate_evaluation_table_rosters(
                table_rows=truncated, score_certificate=certificate,
                descriptive_partial=True,
            )

    def test_prediction_label_binding_is_independent_of_group_sort_order(self):
        methods = list(__import__(
            "spectral_utils.reconstruction_benchmark.methods",
            fromlist=["PRIMARY_METHOD_IDS"],
        ).PRIMARY_METHOD_IDS)
        label_bindings = {
            "cell": {
                "xridv2_a": ("xgidv2_z", 0),
                "xridv2_b": ("xgidv2_a", 1),
            }
        }
        predictions = []
        for method in methods:
            # Persisted prediction order follows group_id, opposite row_id here.
            predictions.extend([
                {
                    "cell_id": "cell", "method_id": method,
                    "row_id": "xridv2_b", "group_id": "xgidv2_a", "label": 1,
                },
                {
                    "cell_id": "cell", "method_id": method,
                    "row_id": "xridv2_a", "group_id": "xgidv2_z", "label": 0,
                },
            ])
        _validate_prediction_label_bindings(
            predictions=predictions, label_bindings=label_bindings,
            ready_cell_ids=["cell"],
        )
        tampered = [dict(row) for row in predictions]
        tampered[0]["group_id"] = "xgidv2_wrong"
        with self.assertRaisesRegex(RuntimeError, "group/label binding"):
            _validate_prediction_label_bindings(
                predictions=tampered, label_bindings=label_bindings,
                ready_cell_ids=["cell"],
            )

    def test_partial_ab_certificate_cannot_open_current_full_only_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            certificate = {
                "schema_version": "reconstruction-edis-ab-certificate-v1",
                "release_id": "rpartial",
                "status": "PASS",
                "scientific_full": False,
                "descriptive_partial": True,
                "headline_eligible": False,
                "aggregate_metrics_allowed": False,
                "certificate_scope": "DESCRIPTIVE_PARTIAL_READY_CELLS_ONLY",
            }
            certificate["certificate_sha256"] = _payload_sha(certificate)
            path = root / "certificate.json"
            path.write_text(json.dumps(certificate), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "full-roster only"):
                assert_ab_certificate(
                    path=path, release_id="rpartial",
                    release_root=root / "releases", selected_build="A",
                    preparation_registry_path=root / "registry.json",
                    private_control_root=root / "private", repo=root,
                )

    def test_current_source_snapshot_rejects_stale_preparation_or_fit_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            snapshot = {
                "git_clean": True,
                "files": [{"path": "source.py", "sha256": sha256_file(source)}],
            }
            snapshot["snapshot_sha256"] = _payload_sha(snapshot)
            verify_current_source_snapshot(
                snapshot, repo=root, name="synthetic source", require_clean=True,
                expected_paths=("source.py",),
            )
            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed or is missing"):
                verify_current_source_snapshot(
                    snapshot, repo=root, name="synthetic source", require_clean=True,
                    expected_paths=("source.py",),
                )

    def test_evaluation_ab_hash_removes_only_build_run_identifier(self):
        def rows(build):
            base = {
                method: _common(
                    release_id="r", run_id=f"r::edis::{build}::postfreeze",
                    spec=None, dataset_id="d", population_id="p", cell_id="c",
                    slice_id="s", cohort_id="cohort::x", method_id=method,
                    method_version_id=f"{method}-v", comparison_group_id="g",
                )
                for method in __import__(
                    "spectral_utils.reconstruction_benchmark.methods",
                    fromlist=["PRIMARY_METHOD_IDS"],
                ).PRIMARY_METHOD_IDS
            }
            metric_value = {
                "value": 0.7, "ci_low": 0.6, "ci_high": 0.8,
                "valid_draws": 20_000,
            }
            interval = {
                "bootstrap_unit": "source_question",
                "draws_requested": 20_000,
                "metrics": {
                    method: {"auroc": metric_value, "auprc": metric_value}
                    for method in base
                },
            }
            return _metric_rows(
                interval=interval, base_by_method=base, aggregation_id="a",
                aggregation_level="cell", component_ids=["c"], n_rows=12,
                n_groups=3, n_positive=6, n_negative=6,
            )

        left, right = rows("A"), rows("B")
        left_hash = canonical_evaluation_table_sha256(
            table="metrics", rows=left, release_id="r", build_id="A"
        )
        right_hash = canonical_evaluation_table_sha256(
            table="metrics", rows=right, release_id="r", build_id="B"
        )
        self.assertEqual(left_hash, right_hash)
        right[0] = {**right[0], "value": 0.71}
        changed_hash = canonical_evaluation_table_sha256(
            table="metrics", rows=right, release_id="r", build_id="B"
        )
        self.assertNotEqual(left_hash, changed_hash)

    def test_vectorized_grouped_bootstrap_matches_direct_reference(self):
        labels = np.asarray([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        groups = ("q0", "q0", "q1", "q1", "q2", "q2", "q3", "q3")
        scores = {
            "iu_pcr": np.asarray([0.1, 0.8, 0.3, 0.7, 0.7, 0.2, 0.2, 0.9]),
            "candidate": np.asarray([0.2, 0.9, 0.4, 0.6, 0.6, 0.1, 0.1, 0.8]),
        }
        direct = grouped_paired_bootstrap(
            labels=labels, scores_by_method=scores, group_ids=groups,
            draws=500, seed=91,
        )
        fast = grouped_paired_bootstrap_auroc_auprc(
            labels=labels, scores_by_method=scores, group_ids=groups,
            draws=500, seed=91,
        )
        self.assertEqual(fast["valid_draws"], direct["valid_draws"])
        for method_id in scores:
            for metric_id in ("auroc", "auprc"):
                for field in ("value", "ci_low", "ci_high"):
                    self.assertAlmostEqual(
                        fast["metrics"][method_id][metric_id][field],
                        direct["metrics"][method_id][metric_id][field],
                        places=12,
                    )
                if method_id != "iu_pcr":
                    for field in ("delta", "ci_low", "ci_high", "probability_delta_le_zero"):
                        self.assertAlmostEqual(
                            fast["contrasts"][method_id][metric_id][field],
                            direct["contrasts"][method_id][metric_id][field],
                            places=12,
                        )

    def test_vectorized_linked_population_matches_direct_reference(self):
        groups = ("q0", "q0", "q1", "q1", "q2", "q2", "q3", "q3")
        cells = {
            "t0p2": {
                "labels": [0, 1, 0, 1, 1, 0, 0, 1],
                "group_ids": groups,
                "scores_by_method": {
                    "iu_pcr": [0.1, 0.8, 0.3, 0.7, 0.6, 0.2, 0.4, 0.9],
                    "candidate": [0.2, 0.9, 0.4, 0.6, 0.7, 0.1, 0.3, 0.8],
                },
            },
            "t0p6": {
                "labels": [0, 1, 1, 0, 1, 0, 0, 1],
                "group_ids": groups,
                "scores_by_method": {
                    "iu_pcr": [0.2, 0.7, 0.8, 0.3, 0.9, 0.2, 0.1, 0.8],
                    "candidate": [0.1, 0.8, 0.9, 0.2, 0.8, 0.3, 0.2, 0.7],
                },
            },
        }
        links = {cell_id: "same_questions" for cell_id in cells}
        direct = population_grouped_paired_bootstrap(
            cells=cells, link_keys=links, draws=400, seed=17,
        )
        fast = population_grouped_paired_bootstrap_auroc_auprc(
            cells=cells, link_keys=links, draws=400, seed=17,
        )
        self.assertEqual(fast["valid_draws"], direct["valid_draws"])
        self.assertEqual(fast["link_blocks"], direct["link_blocks"])
        for method_id in ("iu_pcr", "candidate"):
            for metric_id in ("auroc", "auprc"):
                for field in ("value", "ci_low", "ci_high"):
                    self.assertAlmostEqual(
                        fast["metrics"][method_id][metric_id][field],
                        direct["metrics"][method_id][metric_id][field],
                        places=12,
                    )

    def test_prepared_loader_rejects_group_member_and_raw_matrix(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(12, len(NOMINAL_FEATURE_NAMES)))
        matrix, names, _ = dufs_liu_mixed_v2_matrix(raw, NOMINAL_FEATURE_NAMES)
        identity = SharedEdisIdentityController(bytes(range(32)))
        rows = tuple(sorted(
            identity.row_id(namespace={"lane_id": "x", "scope": "cell", "cell_id": "c"}, raw_identity=str(index))
            for index in range(12)
        ))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "cell.npz"
            arrays = {
                "X_confidence": np.asarray(matrix, dtype="<f8"),
                "feature_names": np.asarray(names, dtype="<U64"),
                "family_ids": np.asarray([preparation.FEATURE_TO_VIEW[name] for name in names], dtype="<U32"),
                "row_ids": np.asarray(rows, dtype="<U80"),
                "row_index": np.arange(len(rows), dtype="<i8"),
                "identity_contract_version": np.asarray([identity.public_binding["contract_version"]], dtype="<U64"),
                "identity_key_id": np.asarray([identity.public_binding["key_id"]], dtype="<U80"),
            }
            artifact_sha = atomic_write_npz(artifact, arrays)
            from spectral_utils.reconstruction_benchmark.contracts import prepared_matrix_sha256
            record = {
                "cell_id": "c", "population_id": "p", "artifact_sha256": artifact_sha,
                "feature_names": list(names), "row_roster_sha256": _payload_sha(list(rows)),
                "prepared_matrix_sha256": prepared_matrix_sha256(matrix, names, rows),
            }
            loaded = load_prepared_cell(
                artifact_path=artifact, record=record,
                identity_binding=identity.public_binding,
            )
            self.assertEqual(loaded.row_ids, rows)
            arrays["group_ids"] = np.asarray(["xgidv2_" + "0" * 64] * len(rows), dtype="<U80")
            artifact_sha = atomic_write_npz(root / "bad.npz", arrays)
            bad = dict(record, artifact_sha256=artifact_sha)
            with self.assertRaisesRegex(RuntimeError, "unexpected members"):
                load_prepared_cell(
                    artifact_path=root / "bad.npz", record=bad,
                    identity_binding=identity.public_binding,
                )

            raw_arrays = dict(arrays)
            raw_arrays.pop("group_ids")
            raw_arrays["X_confidence"] = raw
            raw_sha = atomic_write_npz(root / "raw.npz", raw_arrays)
            raw_record = dict(record, artifact_sha256=raw_sha)
            raw_record["prepared_matrix_sha256"] = prepared_matrix_sha256(raw, names, rows)
            with self.assertRaisesRegex(ValueError, "not centered"):
                load_prepared_cell(
                    artifact_path=root / "raw.npz", record=raw_record,
                    identity_binding=identity.public_binding,
                )

    def test_fit_module_has_no_raw_or_label_loader_import(self):
        source = (REPO / "spectral_utils/reconstruction_benchmark/edis_fit.py").read_text(encoding="utf-8")
        self.assertNotIn("import pickle", source)
        self.assertNotIn("edis_evaluation", source)
        self.assertNotIn("edis_preparation", source)
        self.assertNotIn("external_final_answer", source)

    def test_reporting_rows_validate_as_context_only(self):
        base = {
            method: _common(
                release_id="r", run_id="run", spec=None, dataset_id="d",
                population_id="p", cell_id="c", slice_id="s", cohort_id="cohort::x",
                method_id=method, method_version_id=f"{method}-v",
                comparison_group_id="g",
            )
            for method in __import__(
                "spectral_utils.reconstruction_benchmark.methods", fromlist=["PRIMARY_METHOD_IDS"]
            ).PRIMARY_METHOD_IDS
        }
        metric_value = {"value": 0.7, "ci_low": 0.6, "ci_high": 0.8, "valid_draws": 20_000}
        interval = {
            "bootstrap_unit": "source_group", "draws_requested": 20_000,
            "metrics": {method: {"auroc": metric_value, "auprc": metric_value} for method in base},
        }
        rows = _metric_rows(
            interval=interval, base_by_method=base, aggregation_id="a",
            aggregation_level="cell", component_ids=["c"], n_rows=12,
            n_groups=3, n_positive=6, n_negative=6,
        )
        validated = validate_records("metrics", rows)
        self.assertEqual(len(validated), 26)
        self.assertTrue(all(row["status"] == "CONTEXT_ONLY" for row in validated))
        self.assertTrue(all(row["access_contract_id"] == "gray_box_multi_pass" for row in validated))

    def test_evaluator_cannot_reach_label_loader_without_certificate(self):
        registry = load_preparation_registry(TARGET_FREE)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch(
                "spectral_utils.reconstruction_benchmark.edis_evaluation.reconstruct_labels_and_groups",
                side_effect=AssertionError("labels opened before certificate"),
            ) as label_loader, mock.patch(
                "spectral_utils.reconstruction_benchmark.edis_evaluation.load_postfreeze_registry",
                side_effect=AssertionError("target-derived counts opened before certificate"),
            ) as target_registry:
                with self.assertRaises(FileNotFoundError):
                    evaluate(
                        release_id="missing", build_id="A",
                        release_root=root / "releases",
                        private_control_root=root / "private",
                        source_root=REPO,
                        preparation_registry_path=TARGET_FREE,
                        postfreeze_registry_path=POSTFREEZE,
                        identity=SharedEdisIdentityController(bytes(range(32))),
                        repo=REPO,
                    )
                label_loader.assert_not_called()
                target_registry.assert_not_called()

    def test_tampered_certificate_payload_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "certificate.json"
            value = {
                "schema_version": "reconstruction-edis-ab-certificate-v1",
                "status": "PASS", "release_id": "r", "scientific_full": True,
            }
            value["certificate_sha256"] = _payload_sha(value)
            atomic_write_json(path, value)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["status"] = "FAIL"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "certificate_sha256 failed"):
                assert_ab_certificate(
                    path=path, release_id="r", release_root=Path(temp) / "releases",
                    selected_build="A", preparation_registry_path=TARGET_FREE,
                    private_control_root=Path(temp) / "private", repo=REPO,
                )


if __name__ == "__main__":
    unittest.main()
