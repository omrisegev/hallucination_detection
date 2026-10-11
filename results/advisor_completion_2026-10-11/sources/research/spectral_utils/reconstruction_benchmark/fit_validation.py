"""Fail-closed validation for target-free inputs and frozen score builds.

This module never opens labels.  It binds a scientific score freeze to the
exact 24-cell registry, mixed-v2 source snapshot, 13 executable method specs,
and every on-disk score/artifact record.  It also supplies the independent A/B
equality gate required before the evaluator may open targets.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    CONTRACT_VERSION,
    OUTPUT_SCORE_SEMANTICS,
    POSITIVE_CLASS,
    SCORE_SEMANTICS_CONVERSION,
    MethodSpec,
    PreparedCell,
    prepared_matrix_sha256,
)
from .io import canonical_json_bytes, load_npz_no_pickle, sha256_bytes, sha256_file
from .preparation import SCHEMA_VERSION, _cohort_id, _matrix_hash
from ..specrage_views import FEATURE_TO_VIEW, view_members


SUCCESS = {"OK", "OK_FALLBACK"}


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return sha256_bytes(canonical_json_bytes(payload))


def require_payload_hash(value: Mapping[str, Any], field: str) -> None:
    expected = value.get(field)
    observed = payload_sha256(value, field)
    if expected != observed:
        raise RuntimeError(
            f"invalid {field}: expected self-declared {expected!r}, observed {observed}"
        )


def validate_static_sources(repo: Path, feature_config: Mapping[str, Any]) -> None:
    if feature_config.get("schema_version") != "reconstruction-feature-contract-v1":
        raise RuntimeError("unexpected feature-contract schema")
    if feature_config.get("contract_id") != CONTRACT_VERSION:
        raise RuntimeError("feature-contract ID disagrees with executable contract")
    for path_key, hash_key in (
        ("input_artifact", "input_sha256"),
        ("input_manifest", "input_manifest_sha256"),
        ("transform_source", "transform_source_sha256"),
        ("orientation_source", "orientation_source_sha256"),
        ("roster_source", "roster_source_sha256"),
    ):
        path = (repo / str(feature_config[path_key])).resolve()
        if not path.is_relative_to(repo.resolve()):
            raise RuntimeError(f"configured source escapes repository: {path_key}")
        observed = sha256_file(path)
        if observed != feature_config[hash_key]:
            raise RuntimeError(
                f"configured source drifted for {path_key}: "
                f"expected {feature_config[hash_key]}, got {observed}"
            )


def validate_prepared_manifest(
    *,
    input_root: Path,
    build_id: str,
    repo: Path,
    feature_config: Mapping[str, Any],
    cell_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return one complete target-free 24-cell input manifest."""

    validate_static_sources(repo, feature_config)
    if cell_registry.get("schema_version") != "reconstruction-frozen24-cell-registry-v1":
        raise RuntimeError("unexpected frozen-24 cell-registry schema")
    if cell_registry.get("population_id") != "frozen24_response_v1":
        raise RuntimeError("unexpected frozen-24 population ID")
    expected_cells = list(cell_registry.get("cells", []))
    expected_ids = [str(item["cell_id"]) for item in expected_cells]
    if len(expected_ids) != 24 or len(set(expected_ids)) != 24:
        raise RuntimeError("frozen-24 registry must contain exactly 24 unique cells")

    path = input_root / "MANIFEST.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unexpected prepared-input manifest schema")
    require_payload_hash(manifest, "manifest_payload_sha256")
    required_header = {
        "build_id": build_id,
        "scientific_run": True,
        "feature_contract_id": CONTRACT_VERSION,
        "source_bundle_sha256": feature_config["input_sha256"],
        "feature_contract_config_sha256": sha256_file(
            repo / "configs" / "reconstruction_benchmark_v1" / "feature_contract.json"
        ),
        "transform_source_sha256": feature_config["transform_source_sha256"],
        "orientation_source_sha256": feature_config["orientation_source_sha256"],
        "roster_source_sha256": feature_config["roster_source_sha256"],
        "label_arrays_accessed": False,
        "matrix_semantics": "higher_is_confidence",
        "n_cells": 24,
    }
    for key, expected in required_header.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"prepared manifest field {key} is {manifest.get(key)!r}, "
                f"expected {expected!r}"
            )

    records = list(manifest.get("cells", []))
    record_ids = [str(item.get("cell_id")) for item in records]
    if record_ids != expected_ids or len(set(record_ids)) != 24:
        raise RuntimeError("prepared manifest does not match exact frozen-24 order")
    expected_by_id = {str(item["cell_id"]): item for item in expected_cells}
    total_rows = 0
    for record in records:
        cell_id = str(record["cell_id"])
        expected = expected_by_id[cell_id]
        if record.get("domain") != expected.get("domain"):
            raise RuntimeError(f"{cell_id}: domain disagrees with frozen registry")
        relative = Path(str(record.get("artifact_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"{cell_id}: unsafe prepared artifact path")
        if relative.as_posix() != f"cells/{cell_id}.npz":
            raise RuntimeError(f"{cell_id}: noncanonical prepared artifact path")
        artifact = input_root / relative
        if sha256_file(artifact) != record.get("artifact_sha256"):
            raise RuntimeError(f"{cell_id}: prepared artifact hash mismatch")
        arrays = load_npz_no_pickle(artifact)
        required_arrays = {
            "X_confidence", "feature_names", "family_ids", "row_ids", "row_index"
        }
        if set(arrays) != required_arrays:
            raise RuntimeError(f"{cell_id}: prepared artifact member mismatch")
        matrix = np.asarray(arrays["X_confidence"], dtype=np.float64)
        names = tuple(str(value) for value in arrays["feature_names"].tolist())
        rows = tuple(str(value) for value in arrays["row_ids"].tolist())
        families = tuple(str(value) for value in arrays["family_ids"].tolist())
        indices = np.asarray(arrays["row_index"], dtype=np.int64)
        if matrix.shape != (int(record["n_rows"]), int(record["n_features"])):
            raise RuntimeError(f"{cell_id}: prepared shape disagrees with manifest")
        if names != tuple(record["feature_names"]):
            raise RuntimeError(f"{cell_id}: feature names disagree with manifest")
        if families != tuple(FEATURE_TO_VIEW[name] for name in names):
            raise RuntimeError(f"{cell_id}: family IDs disagree with frozen map")
        if tuple(view_members(names)) != tuple(record["present_families"]):
            raise RuntimeError(f"{cell_id}: present-family roster drifted")
        expected_rows = tuple(
            f"{cell_id}:matrix_row:{index:08d}" for index in range(matrix.shape[0])
        )
        if rows != expected_rows or not np.array_equal(indices, np.arange(len(rows))):
            raise RuntimeError(f"{cell_id}: consolidated row identity/order drifted")
        matrix_hash = _matrix_hash(matrix, names)
        if matrix_hash != record.get("feature_matrix_sha256"):
            raise RuntimeError(f"{cell_id}: feature-matrix hash mismatch")
        cohort = _cohort_id(
            cell_id,
            len(rows),
            feature_matrix_sha256=matrix_hash,
            source_bundle_sha256=feature_config["input_sha256"],
        )
        if cohort != record.get("cohort_id"):
            raise RuntimeError(f"{cell_id}: cohort ID does not bind matrix/source")
        # This constructor independently checks canonical order, mixed-v2
        # standardization, and its stronger X+names+rows hash.
        PreparedCell(
            population_id=cell_registry["population_id"],
            cell_id=cell_id,
            domain=str(record["domain"]),
            matrix=matrix,
            feature_names=names,
            row_ids=rows,
            feature_contract=CONTRACT_VERSION,
            preprocessing_steps=(CONTRACT_VERSION,),
            preprocessed=True,
        )
        total_rows += len(rows)
    if int(manifest.get("n_rows", -1)) != total_rows:
        raise RuntimeError("prepared manifest total row count is wrong")
    return manifest


def validate_score_record(
    *,
    method_dir: Path,
    spec: MethodSpec,
    cell: PreparedCell,
) -> dict[str, Any]:
    """Validate every byte referenced by one score record."""

    record_path = method_dir / "RECORD.json"
    record = json.loads(record_path.read_text())
    expected = {
        "schema_version": "reconstruction-score-record-v1",
        "method_id": spec.method_id,
        "method_version_id": spec.method_version_id,
        "config_sha256": spec.config_sha256,
        "population_id": cell.population_id,
        "cell_id": cell.cell_id,
        "feature_contract": cell.feature_contract,
        "prepared_matrix_sha256": cell.matrix_sha256,
        "score_semantics": OUTPUT_SCORE_SEMANTICS,
        "positive_class": POSITIVE_CLASS,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(
                f"{cell.cell_id}/{spec.method_id}: record {key} mismatch"
            )
    if record.get("status") not in SUCCESS:
        raise RuntimeError(
            f"{cell.cell_id}/{spec.method_id}: non-success status {record.get('status')}"
        )
    if record.get("score_semantics_conversion") != dict(SCORE_SEMANTICS_CONVERSION):
        raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: score conversion drifted")
    score_path = method_dir / str(record.get("score_path"))
    if record.get("score_path") != "score.npz":
        raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: noncanonical score path")
    if sha256_file(score_path) != record.get("score_sha256"):
        raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: score hash mismatch")
    score_arrays = load_npz_no_pickle(score_path)
    if set(score_arrays) != {"row_ids", "score"}:
        raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: score members drifted")
    rows = tuple(str(value) for value in score_arrays["row_ids"].tolist())
    score = np.asarray(score_arrays["score"], dtype=float)
    if rows != cell.row_ids or score.shape != (len(rows),) or not np.isfinite(score).all():
        raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: score rows/vector invalid")
    if int(record.get("score_n", -1)) != len(rows):
        raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: score_n mismatch")

    index_path = method_dir / "ARTIFACT_INDEX.json"
    if sha256_file(index_path) != record.get("artifact_index_sha256"):
        raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: artifact index mismatch")
    artifact_path = record.get("artifacts_path")
    artifact_sha = record.get("artifacts_sha256")
    if (artifact_path is None) != (artifact_sha is None):
        raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: artifact fields disagree")
    if artifact_path is not None:
        if artifact_path != "artifacts.npz":
            raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: noncanonical artifact path")
        if sha256_file(method_dir / artifact_path) != artifact_sha:
            raise RuntimeError(f"{cell.cell_id}/{spec.method_id}: artifact hash mismatch")
    record["record_sha256"] = sha256_file(record_path)
    return record


def _records_by_pair(freeze: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for record in freeze.get("records", []):
        key = (str(record.get("cell_id")), str(record.get("method_id")))
        if key in output:
            raise RuntimeError(f"duplicate frozen score pair: {key!r}")
        output[key] = dict(record)
    return output


def verify_frozen_build(
    *,
    build_id: str,
    fit_root: Path,
    input_root: Path,
    input_manifest: Mapping[str, Any],
    method_specs: Mapping[str, MethodSpec],
    population_id: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema_version") != "reconstruction-score-freeze-v1":
        raise RuntimeError("unexpected score-freeze schema")
    require_payload_hash(freeze, "payload_sha256")
    required_header = {
        "build_id": build_id,
        "scientific_run": True,
        "feature_contract_id": CONTRACT_VERSION,
        "score_semantics": OUTPUT_SCORE_SEMANTICS,
        "positive_class": POSITIVE_CLASS,
        "labels_opened_by_fit": False,
        "runtime_labels_used": False,
        "preprocessing_selected_after_outcomes_were_opened": True,
        "evidence_status": "D0_reused_development",
        "n_cells": 24,
        "n_methods": len(method_specs),
        "expected_records": 24 * len(method_specs),
        "all_headline_scores_present": True,
    }
    for key, expected in required_header.items():
        if freeze.get(key) != expected:
            raise RuntimeError(f"score-freeze {key} mismatch")
    if freeze.get("input_manifest_payload_sha256") != input_manifest.get(
        "manifest_payload_sha256"
    ):
        raise RuntimeError("score-freeze is not bound to prepared manifest")
    if freeze.get("input_manifest_file_sha256") != sha256_file(
        input_root / "MANIFEST.json"
    ):
        raise RuntimeError("score-freeze prepared-manifest file hash mismatch")

    prefit_path = fit_root / "FIT_SOURCE_SNAPSHOT.json"
    prefit = json.loads(prefit_path.read_text())
    require_payload_hash(prefit, "payload_sha256")
    if prefit.get("schema_version") != "reconstruction-fit-source-snapshot-v1":
        raise RuntimeError("unexpected pre-fit source-snapshot schema")
    if prefit.get("build_id") != build_id:
        raise RuntimeError("pre-fit source-snapshot build mismatch")
    if freeze.get("prefit_snapshot_payload_sha256") != prefit.get("payload_sha256"):
        raise RuntimeError("score-freeze is not bound to the pre-fit source snapshot")
    if freeze.get("prefit_snapshot_file_sha256") != sha256_file(prefit_path):
        raise RuntimeError("pre-fit source-snapshot file hash mismatch")
    if prefit.get("input_manifest_payload_sha256") != input_manifest.get(
        "manifest_payload_sha256"
    ):
        raise RuntimeError("pre-fit source snapshot is not bound to prepared input")
    if prefit.get("source_snapshot_sha256") != freeze.get("source_snapshot_sha256"):
        raise RuntimeError("pre-fit and score-freeze source snapshots differ")
    if prefit.get("source_snapshot") != freeze.get("source_snapshot"):
        raise RuntimeError("pre-fit and score-freeze source payloads differ")

    expected_cells = tuple(str(item["cell_id"]) for item in input_manifest["cells"])
    expected_methods = tuple(method_specs)
    if tuple(freeze.get("cell_ids", ())) != expected_cells:
        raise RuntimeError("score-freeze cell roster/order mismatch")
    if tuple(freeze.get("method_ids", ())) != expected_methods:
        raise RuntimeError("score-freeze method roster/order mismatch")
    if tuple(prefit.get("cell_ids", ())) != expected_cells:
        raise RuntimeError("pre-fit source-snapshot cell roster/order mismatch")
    if tuple(prefit.get("method_ids", ())) != expected_methods:
        raise RuntimeError("pre-fit source-snapshot method roster/order mismatch")
    expected_pairs = {
        (cell_id, method_id)
        for cell_id in expected_cells
        for method_id in expected_methods
    }
    frozen_pairs = _records_by_pair(freeze)
    if set(frozen_pairs) != expected_pairs or len(frozen_pairs) != 24 * len(method_specs):
        raise RuntimeError("score-freeze does not contain exact 24 x method Cartesian set")

    input_records = {str(item["cell_id"]): item for item in input_manifest["cells"]}
    for cell_id in expected_cells:
        prepared_path = input_root / input_records[cell_id]["artifact_path"]
        arrays = load_npz_no_pickle(prepared_path)
        cell = PreparedCell(
            population_id=population_id,
            cell_id=cell_id,
            domain=str(input_records[cell_id]["domain"]),
            matrix=np.asarray(arrays["X_confidence"], dtype=np.float64),
            feature_names=tuple(str(v) for v in arrays["feature_names"].tolist()),
            row_ids=tuple(str(v) for v in arrays["row_ids"].tolist()),
            feature_contract=CONTRACT_VERSION,
            preprocessing_steps=(CONTRACT_VERSION,),
            preprocessed=True,
        )
        for method_id, spec in method_specs.items():
            method_dir = fit_root / "cells" / cell_id / method_id
            observed = validate_score_record(method_dir=method_dir, spec=spec, cell=cell)
            frozen = frozen_pairs[(cell_id, method_id)]
            for field in (
                "method_version_id", "config_sha256", "status",
                "prepared_matrix_sha256", "score_sha256", "artifacts_sha256",
                "artifact_index_sha256", "record_sha256",
            ):
                if frozen.get(field) != observed.get(field):
                    raise RuntimeError(
                        f"{cell_id}/{method_id}: frozen {field} disagrees with disk"
                    )
    return freeze, frozen_pairs


def compare_frozen_builds(
    *,
    release_root: Path,
    input_manifests: Mapping[str, Mapping[str, Any]],
    method_specs: Mapping[str, MethodSpec],
    population_id: str,
) -> dict[str, Any]:
    """Verify independent A/B score outputs and return a signed PASS payload."""

    freezes: dict[str, dict[str, Any]] = {}
    pairs: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for build_id in ("A", "B"):
        freeze, by_pair = verify_frozen_build(
            build_id=build_id,
            fit_root=release_root / f"build_{build_id}" / "fit",
            input_root=release_root / f"build_{build_id}" / "inputs",
            input_manifest=input_manifests[build_id],
            method_specs=method_specs,
            population_id=population_id,
        )
        freezes[build_id] = freeze
        pairs[build_id] = by_pair
    if set(pairs["A"]) != set(pairs["B"]):
        raise RuntimeError("A/B frozen score pair sets differ")
    comparisons = []
    for cell_id, method_id in sorted(pairs["A"]):
        left = pairs["A"][(cell_id, method_id)]
        right = pairs["B"][(cell_id, method_id)]
        equal = all(
            left.get(field) == right.get(field)
            for field in (
                "method_version_id", "config_sha256", "status",
                "prepared_matrix_sha256", "score_sha256", "artifacts_sha256",
                "artifact_index_sha256", "record_sha256",
            )
        )
        if not equal:
            raise RuntimeError(f"A/B score or artifact mismatch: {cell_id}/{method_id}")
        comparisons.append(
            {
                "cell_id": cell_id,
                "method_id": method_id,
                "score_sha256": left["score_sha256"],
                "artifacts_sha256": left["artifacts_sha256"],
                "record_sha256": left["record_sha256"],
                "byte_identical": True,
            }
        )
    if freezes["A"].get("source_snapshot_sha256") != freezes["B"].get(
        "source_snapshot_sha256"
    ):
        raise RuntimeError("A/B source snapshots differ")
    result = {
        "schema_version": "reconstruction-score-ab-verification-v1",
        "pass": True,
        "n_cells": 24,
        "n_methods": len(method_specs),
        "n_pairs": len(comparisons),
        "cell_ids": [str(item["cell_id"]) for item in input_manifests["A"]["cells"]],
        "method_ids": list(method_specs),
        "freeze_A_sha256": sha256_file(
            release_root / "build_A" / "fit" / "SCORE_FREEZE_MANIFEST.json"
        ),
        "freeze_B_sha256": sha256_file(
            release_root / "build_B" / "fit" / "SCORE_FREEZE_MANIFEST.json"
        ),
        "input_manifest_A_sha256": sha256_file(
            release_root / "build_A" / "inputs" / "MANIFEST.json"
        ),
        "input_manifest_B_sha256": sha256_file(
            release_root / "build_B" / "inputs" / "MANIFEST.json"
        ),
        "source_snapshot_sha256": freezes["A"].get("source_snapshot_sha256"),
        "pairs": comparisons,
    }
    result["payload_sha256"] = payload_sha256(result, "payload_sha256")
    return result


__all__ = [
    "compare_frozen_builds",
    "payload_sha256",
    "require_payload_hash",
    "validate_prepared_manifest",
    "validate_score_record",
    "validate_static_sources",
    "verify_frozen_build",
]
