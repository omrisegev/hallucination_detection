#!/usr/bin/env python3
"""Focused tests for the strict reconstruction 24-cell evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from spectral_utils.reconstruction_benchmark.contracts import (
    CONTRACT_VERSION,
    FitStatus,
    ScoreResult,
    prepared_matrix_sha256,
)
from spectral_utils.reconstruction_benchmark.evaluation import (
    BOOTSTRAP_DRAW_COUNT,
    EvaluationContractError,
    GROUP_EVIDENCE_SCHEMA_VERSION,
    GROUP_SIDECAR_SCHEMA_VERSION,
    VerifiedCell,
    VerifiedRelease,
    _ordered_text_hash,
    _weighted_binary_metric_draws,
    evaluate_verified_release,
    open_correctness_labels,
    prediction_snapshot_arrays,
    row_group_binding_sha256,
    verify_release_before_labels,
)
from spectral_utils.reconstruction_benchmark.io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.methods import (
    PRIMARY_METHOD_IDS,
    PRIMARY_METHOD_SPECS,
)
from spectral_utils.reconstruction_benchmark.preparation import SCHEMA_VERSION
from spectral_utils.reconstruction_benchmark.serialization import write_score_result


REPO = Path(__file__).resolve().parents[2]
CELL_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "frozen24_cells.json"
METHOD_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "methods.json"
FEATURE_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "feature_contract.json"
FEATURE_NAMES = (
    "epr",
    "trace_length",
    "spectral_entropy",
    "epr_spilled",
    "epr_energy",
    "mean_top1_logprob",
)


def _with_payload_hash(value: dict, field: str = "payload_sha256") -> dict:
    output = dict(value)
    output[field] = sha256_bytes(canonical_json_bytes(output))
    return output


def _standard_matrix(n: int = 6) -> np.ndarray:
    base = np.arange(n * len(FEATURE_NAMES), dtype=np.float64).reshape(n, -1)
    base = np.sin(base * 0.37) + np.cos(base * 0.11)
    return (base - base.mean(axis=0)) / base.std(axis=0)


def _verified_release_for_metrics() -> tuple[VerifiedRelease, dict[str, np.ndarray]]:
    registry = json.loads(CELL_REGISTRY.read_text(encoding="utf-8"))
    rows = tuple(f"row-{index}" for index in range(8))
    groups = ("q0", "q0", "q1", "q1", "q2", "q2", "q3", "q3")
    base = np.asarray([-1.2, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4])
    cells = {}
    labels = {}
    for cell_index, metadata in enumerate(registry["cells"]):
        score_map = {
            method_id: np.asarray(base + 0.01 * index, dtype=np.float64)
            for index, method_id in enumerate(PRIMARY_METHOD_IDS)
        }
        for value in score_map.values():
            value.setflags(write=False)
        cells[metadata["cell_id"]] = VerifiedCell(
            metadata=metadata,
            row_ids=rows,
            group_ids=groups,
            prepared_matrix_sha256="1" * 64,
            score_by_method=score_map,
            score_sha256_by_method={method_id: "2" * 64 for method_id in PRIMARY_METHOD_IDS},
            fit_status_by_method={method_id: "OK" for method_id in PRIMARY_METHOD_IDS},
            fallback_reason_by_method={method_id: None for method_id in PRIMARY_METHOD_IDS},
            group_artifact_sha256="3" * 64,
            group_binding_sha256="4" * 64,
            group_evidence_sha256="5" * 64,
            group_source_sha256="6" * 64,
        )
        labels[metadata["cell_id"]] = np.asarray(
            [1, 0, 1, 0, 1, 0, 1, 0] if cell_index % 2 == 0
            else [0, 1, 0, 1, 0, 1, 0, 1],
            dtype=np.int8,
        )
    return VerifiedRelease(
        release_root=Path("/synthetic"),
        population_id=registry["population_id"],
        method_ids=tuple(PRIMARY_METHOD_IDS),
        cells=cells,
        label_bundle=Path("/synthetic/labels.npz"),
        provenance={"labels_opened": False},
    ), labels


def _build_full_fixture(root: Path) -> dict[str, Path]:
    registry = json.loads(CELL_REGISTRY.read_text(encoding="utf-8"))
    feature_config = json.loads(FEATURE_REGISTRY.read_text(encoding="utf-8"))
    cell_ids = tuple(row["cell_id"] for row in registry["cells"])
    release = root / "release"
    group_root = root / "groups"
    group_root.mkdir(parents=True)
    labels_path = root / "labels.npz"
    y_correct = np.asarray([1, 0, 1, 0, 1, 0], dtype="<i1")
    atomic_write_npz(labels_path, {f"{cell_id}__labels": y_correct for cell_id in cell_ids})
    label_sha = sha256_file(labels_path)
    matrix = _standard_matrix()
    family_ids = np.asarray(
        ["entropy_level", "structural", "entropy_dynamics", "sampled_token_energy", "partition_energy", "topk_distribution"],
        dtype="<U32",
    )
    rows_by_cell = {
        cell_id: tuple(f"{cell_id}:matrix_row:{index:08d}" for index in range(len(matrix)))
        for cell_id in cell_ids
    }
    input_manifests = {}
    input_records = {}
    matrix_hashes = {}
    for build_id in ("A", "B"):
        input_root = release / f"build_{build_id}" / "inputs"
        records = []
        for metadata in registry["cells"]:
            cell_id = metadata["cell_id"]
            row_ids = rows_by_cell[cell_id]
            artifact = input_root / "cells" / f"{cell_id}.npz"
            artifact_sha = atomic_write_npz(artifact, {
                "X_confidence": np.asarray(matrix, dtype="<f8"),
                "feature_names": np.asarray(FEATURE_NAMES, dtype="<U64"),
                "family_ids": family_ids,
                "row_ids": np.asarray(row_ids, dtype="<U128"),
                "row_index": np.arange(len(row_ids), dtype="<i8"),
            })
            matrix_hash = prepared_matrix_sha256(matrix, FEATURE_NAMES, row_ids)
            matrix_hashes[cell_id] = matrix_hash
            records.append({
                "cell_id": cell_id,
                "domain": metadata["domain"],
                "n_rows": len(row_ids),
                "n_features": len(FEATURE_NAMES),
                "feature_names": list(FEATURE_NAMES),
                "present_families": list(family_ids),
                "cohort_id": f"cohort-{cell_id}",
                "feature_matrix_sha256": "0" * 64,
                "artifact_path": f"cells/{cell_id}.npz",
                "artifact_sha256": artifact_sha,
                "transform_details": {},
            })
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "scientific_run": True,
            "feature_contract_id": CONTRACT_VERSION,
            "source_bundle_sha256": label_sha,
            "feature_contract_config_sha256": sha256_file(FEATURE_REGISTRY),
            "transform_source_sha256": feature_config["transform_source_sha256"],
            "orientation_source_sha256": feature_config["orientation_source_sha256"],
            "roster_source_sha256": feature_config["roster_source_sha256"],
            "label_arrays_accessed": False,
            "n_cells": 24,
            "n_rows": 24 * len(matrix),
            "cells": records,
        }
        manifest["manifest_payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        atomic_write_json(input_root / "MANIFEST.json", manifest)
        input_manifests[build_id] = manifest
        input_records[build_id] = {row["cell_id"]: row for row in records}

    freeze_by_build = {}
    source_snapshot = _with_payload_hash(
        {
            "schema_version": "reconstruction-source-snapshot-v1",
            "git_head": "synthetic",
            "git_status_sha256": "7" * 64,
            "git_status_clean": True,
            "files": [],
        },
        field="snapshot_sha256",
    )
    for build_id in ("A", "B"):
        fit_root = release / f"build_{build_id}" / "fit"
        prefit = {
            "schema_version": "reconstruction-fit-source-snapshot-v1",
            "release_id": "synthetic-release",
            "build_id": build_id,
            "input_manifest_payload_sha256": input_manifests[build_id]["manifest_payload_sha256"],
            "source_snapshot": source_snapshot,
            "source_snapshot_sha256": source_snapshot["snapshot_sha256"],
            "method_ids": list(PRIMARY_METHOD_IDS),
            "cell_ids": list(cell_ids),
        }
        prefit = _with_payload_hash(prefit)
        prefit_path = fit_root / "FIT_SOURCE_SNAPSHOT.json"
        atomic_write_json(prefit_path, prefit)
        summaries = []
        for cell_index, cell_id in enumerate(cell_ids):
            row_ids = rows_by_cell[cell_id]
            for method_index, method_id in enumerate(PRIMARY_METHOD_IDS):
                spec = PRIMARY_METHOD_SPECS[method_id]
                score = np.linspace(-1.0, 1.0, len(row_ids)) + 0.01 * method_index + 0.001 * cell_index
                result = ScoreResult(
                    method_id=method_id,
                    method_version_id=spec.method_version_id,
                    config_sha256=spec.config_sha256,
                    status=FitStatus.OK,
                    score=score,
                    population_id=registry["population_id"],
                    cell_id=cell_id,
                    feature_contract=CONTRACT_VERSION,
                    prepared_matrix_sha256=matrix_hashes[cell_id],
                    diagnostics={"synthetic": True},
                )
                record = write_score_result(
                    result, row_ids, fit_root / "cells" / cell_id / method_id
                )
                summaries.append({
                    "cell_id": cell_id,
                    "method_id": method_id,
                    "method_version_id": record["method_version_id"],
                    "config_sha256": record["config_sha256"],
                    "status": record["status"],
                    "prepared_matrix_sha256": record["prepared_matrix_sha256"],
                    "score_sha256": record["score_sha256"],
                    "artifacts_sha256": record["artifacts_sha256"],
                    "artifact_index_sha256": record["artifact_index_sha256"],
                    "record_sha256": record["record_sha256"],
                })
        freeze = {
            "schema_version": "reconstruction-score-freeze-v1",
            "build_id": build_id,
            "scientific_run": True,
            "feature_contract_id": CONTRACT_VERSION,
            "score_semantics": "higher_is_incorrect",
            "positive_class": "incorrect",
            "labels_opened_by_fit": False,
            "runtime_labels_used": False,
            "preprocessing_selected_after_outcomes_were_opened": True,
            "evidence_status": "D0_reused_development",
            "all_headline_scores_present": True,
            "input_manifest_payload_sha256": input_manifests[build_id]["manifest_payload_sha256"],
            "input_manifest_file_sha256": sha256_file(
                release / f"build_{build_id}" / "inputs" / "MANIFEST.json"
            ),
            "cell_registry_sha256": sha256_file(CELL_REGISTRY),
            "method_registry_sha256": sha256_file(METHOD_REGISTRY),
            "feature_config_sha256": sha256_file(FEATURE_REGISTRY),
            "source_snapshot": source_snapshot,
            "source_snapshot_sha256": source_snapshot["snapshot_sha256"],
            "prefit_snapshot_payload_sha256": prefit["payload_sha256"],
            "prefit_snapshot_file_sha256": sha256_file(prefit_path),
            "n_cells": 24,
            "n_methods": 13,
            "n_records": 312,
            "expected_records": 312,
            "cell_ids": list(cell_ids),
            "method_ids": list(PRIMARY_METHOD_IDS),
            "method_specs": {
                method_id: {
                    "method_version_id": PRIMARY_METHOD_SPECS[method_id].method_version_id,
                    "config": dict(PRIMARY_METHOD_SPECS[method_id].config),
                    "config_sha256": PRIMARY_METHOD_SPECS[method_id].config_sha256,
                }
                for method_id in PRIMARY_METHOD_IDS
            },
            "records": summaries,
        }
        freeze = _with_payload_hash(freeze)
        freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
        atomic_write_json(freeze_path, freeze)
        freeze_by_build[build_id] = freeze

    attestation = {
        "schema_version": "reconstruction-score-ab-verification-v1",
        "pass": True,
        "n_cells": 24,
        "n_methods": 13,
        "n_pairs": 312,
        "cell_ids": list(cell_ids),
        "method_ids": list(PRIMARY_METHOD_IDS),
        "freeze_A_sha256": sha256_file(release / "build_A" / "fit" / "SCORE_FREEZE_MANIFEST.json"),
        "freeze_B_sha256": sha256_file(release / "build_B" / "fit" / "SCORE_FREEZE_MANIFEST.json"),
        "input_manifest_A_sha256": sha256_file(release / "build_A" / "inputs" / "MANIFEST.json"),
        "input_manifest_B_sha256": sha256_file(release / "build_B" / "inputs" / "MANIFEST.json"),
        "pairs": [
            {
                "cell_id": row["cell_id"],
                "method_id": row["method_id"],
                "score_sha256": row["score_sha256"],
                "artifacts_sha256": row["artifacts_sha256"],
                "record_sha256": row["record_sha256"],
                "byte_identical": True,
            }
            for row in freeze_by_build["A"]["records"]
        ],
    }
    atomic_write_json(release / "SCORE_AB_VERIFICATION.json", _with_payload_hash(attestation))

    group_rows = []
    for metadata in registry["cells"]:
        cell_id = metadata["cell_id"]
        row_ids = rows_by_cell[cell_id]
        group_ids = ("source-0", "source-0", "source-1", "source-1", "source-2", "source-2")
        artifact = group_root / "sidecars" / f"{cell_id}.npz"
        artifact_sha = atomic_write_npz(artifact, {
            "row_ids": np.asarray(row_ids, dtype="<U128"),
            "group_ids": np.asarray(group_ids, dtype="<U32"),
        })
        source = group_root / "sources" / f"{cell_id}.npz"
        source_sha = atomic_write_npz(source, {
            "source_group_ids": np.asarray(group_ids, dtype="<U32"),
        })
        binding = row_group_binding_sha256(row_ids, group_ids)
        evidence = _with_payload_hash({
            "schema_version": GROUP_EVIDENCE_SCHEMA_VERSION,
            "cell_id": cell_id,
            "verification_status": "VERIFIED",
            "labels_used": False,
            "source_artifact_sha256": source_sha,
            "group_artifact_sha256": artifact_sha,
            "row_group_binding_sha256": binding,
            "verifier_id": "synthetic-test-verifier",
            "verification_method": "exact synthetic row/source join",
            "checks": {
                "source_hash_verified": True,
                "row_count_verified": True,
                "row_order_verified": True,
                "group_semantics_verified": True,
            },
        })
        evidence_path = group_root / "evidence" / f"{cell_id}.json"
        evidence_sha = atomic_write_json(evidence_path, evidence)
        group_rows.append({
            "cell_id": cell_id,
            "verification_status": "VERIFIED",
            "labels_used": False,
            "group_unit": "source_question_id",
            "cohort_id": input_records["A"][cell_id]["cohort_id"],
            "prepared_matrix_sha256": matrix_hashes[cell_id],
            "registry_source_group_status": metadata["source_group_status"],
            "artifact_path": artifact.relative_to(group_root).as_posix(),
            "artifact_sha256": artifact_sha,
            "row_group_binding_sha256": binding,
            "row_ids_sha256": _ordered_text_hash(row_ids, field="row_ids"),
            "group_ids_sha256": _ordered_text_hash(group_ids, field="group_ids"),
            "n_rows": len(row_ids),
            "n_groups": len(set(group_ids)),
            "source_artifact_path": source.relative_to(group_root).as_posix(),
            "source_artifact_sha256": source_sha,
            "identity_evidence_path": evidence_path.relative_to(group_root).as_posix(),
            "identity_evidence_sha256": evidence_sha,
        })
    group_manifest = _with_payload_hash({
        "schema_version": GROUP_SIDECAR_SCHEMA_VERSION,
        "population_id": registry["population_id"],
        "label_bundle_sha256": label_sha,
        "cells": group_rows,
    })
    group_manifest_path = group_root / "GROUP_SIDECARS.json"
    atomic_write_json(group_manifest_path, group_manifest)
    return {
        "release": release,
        "labels": labels_path,
        "groups": group_manifest_path,
    }


class WeightedMetricTest(unittest.TestCase):
    def test_exact_parity_with_sklearn_including_score_ties(self):
        y = np.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int8)
        score = np.asarray([0.1, 0.1, 0.4, 0.7, 0.7, 0.2, 0.9, 0.4])
        group_columns = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        counts = np.asarray([[1, 1, 1, 1], [2, 0, 1, 1], [0, 2, 1, 1]], dtype=np.int32)
        auc, ap = _weighted_binary_metric_draws(y, score, group_columns, counts)
        for index, group_weights in enumerate(counts):
            weights = group_weights[group_columns]
            self.assertAlmostEqual(auc[index], roc_auc_score(y, score, sample_weight=weights), places=14)
            self.assertAlmostEqual(ap[index], average_precision_score(y, score, sample_weight=weights), places=14)


class GroupedEvaluationTest(unittest.TestCase):
    def test_shared_draws_equal_cell_aggregates_and_noncanonical_headline_gate(self):
        verified, labels = _verified_release_for_metrics()
        first, first_arrays = evaluate_verified_release(
            verified, labels, bootstrap_draws=127, bootstrap_chunk_size=19
        )
        second, second_arrays = evaluate_verified_release(
            verified, labels, bootstrap_draws=127, bootstrap_chunk_size=31
        )
        self.assertEqual(BOOTSTRAP_DRAW_COUNT, 20_000)
        self.assertEqual(first["headline_status"], "HEADLINE_BLOCKED_INCOMPLETE_OR_NONCANONICAL")
        self.assertEqual(first["headline_macro24_auroc"], [])
        self.assertEqual(first["payload_sha256"], second["payload_sha256"])
        self.assertEqual(set(first_arrays), set(second_arrays))
        for key in first_arrays:
            np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
        cell = next(iter(verified.cells))
        baseline = first_arrays[f"cell__{cell}__iu_pcr__auroc"]
        for method_id in PRIMARY_METHOD_IDS:
            np.testing.assert_array_equal(
                baseline, first_arrays[f"cell__{cell}__{method_id}__auroc"]
            )
        macro = [
            row for row in first["aggregate_metrics"]
            if row["scope_type"] == "macro24"
            and row["method_id"] == "iu_pcr"
            and row["metric"] == "auroc"
        ]
        self.assertEqual(len(macro), 1)
        self.assertEqual(macro[0]["n_cells"], 24)
        contrasts = [
            row for row in first["paired_contrasts_vs_iu_pcr"]
            if row["scope_type"] == "macro24" and row["metric"] == "auroc"
        ]
        self.assertEqual(len(contrasts), 12)
        self.assertTrue(all(abs(row["delta"]) < 1e-14 for row in contrasts))


class FullBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.paths = _build_full_fixture(Path(cls.temporary.name))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _verify(self) -> VerifiedRelease:
        return verify_release_before_labels(
            release_root=self.paths["release"],
            cell_registry_path=CELL_REGISTRY,
            method_registry_path=METHOD_REGISTRY,
            feature_registry_path=FEATURE_REGISTRY,
            label_bundle=self.paths["labels"],
            group_manifest_path=self.paths["groups"],
        )

    def test_complete_boundary_verifies_before_labels_and_converts_correctness(self):
        verified = self._verify()
        self.assertEqual(len(verified.cells), 24)
        self.assertEqual(len(verified.method_ids), 13)
        self.assertFalse(verified.provenance["labels_opened"])
        y_correct = open_correctness_labels(verified)
        evaluation, _ = evaluate_verified_release(
            verified, y_correct, bootstrap_draws=41, bootstrap_chunk_size=7
        )
        first = evaluation["label_provenance"][0]
        self.assertEqual(first["conversion"], "y_error=1-y_correct")
        self.assertEqual(first["n_correct"], 3)
        self.assertEqual(first["n_error"], 3)
        snapshot = prediction_snapshot_arrays(verified, y_correct)
        first_cell = next(iter(verified.cells))
        self.assertEqual(
            set(name for name in snapshot if name.startswith(first_cell + "__")),
            {
                f"{first_cell}__row_ids",
                f"{first_cell}__group_ids",
                f"{first_cell}__y_error",
                *{
                    f"{first_cell}__{method_id}__score"
                    for method_id in PRIMARY_METHOD_IDS
                },
            },
        )
        np.testing.assert_array_equal(
            snapshot[f"{first_cell}__y_error"],
            1 - y_correct[first_cell],
        )

    def test_score_byte_drift_is_rejected(self):
        score_path = (
            self.paths["release"] / "build_B" / "fit" / "cells"
            / json.loads(CELL_REGISTRY.read_text())["cells"][0]["cell_id"]
            / PRIMARY_METHOD_IDS[0] / "score.npz"
        )
        original = score_path.read_bytes()
        try:
            score_path.write_bytes(original + b"drift")
            with self.assertRaisesRegex(EvaluationContractError, "score hash drift"):
                self._verify()
        finally:
            score_path.write_bytes(original)

    def test_unverified_or_iid_group_sidecar_is_rejected(self):
        manifest_path = self.paths["groups"]
        original = manifest_path.read_bytes()
        try:
            manifest = json.loads(original)
            manifest["cells"][0]["verification_status"] = "PENDING"
            manifest.pop("payload_sha256")
            atomic_write_json(manifest_path, _with_payload_hash(manifest))
            with self.assertRaisesRegex(EvaluationContractError, "not verified"):
                self._verify()
        finally:
            manifest_path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
