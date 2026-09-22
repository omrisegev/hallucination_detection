"""Strict post-freeze evaluation for the reconstructed 24-cell benchmark.

This module is the target-access boundary.  :func:`verify_release_before_labels`
must finish successfully before :func:`open_correctness_labels` is called.  In
particular, a score-freeze declaration is not trusted on its own: both builds,
every score record, every declared artifact, and the independent A/B
verification record are re-hashed here.

The legacy 24-cell bundle does not contain source-question identifiers.  A
separate, audited group sidecar is therefore mandatory.  Treating matrix rows
as independent bootstrap units is deliberately unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .contracts import (
    CONTRACT_VERSION,
    OUTPUT_SCORE_SEMANTICS,
    POSITIVE_CLASS,
    SCORE_SEMANTICS_CONVERSION,
    canonical_sha256,
    prepared_matrix_sha256,
)
from .methods import PRIMARY_METHOD_IDS, PRIMARY_METHOD_SPECS
from .io import (
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from .fit_validation import validate_static_sources


EVALUATION_SCHEMA_VERSION = "reconstruction-24cell-evaluation-v1"
EVALUATION_MANIFEST_SCHEMA_VERSION = "reconstruction-evaluation-manifest-v1"
GROUP_SIDECAR_SCHEMA_VERSION = "reconstruction-group-sidecars-v1"
GROUP_EVIDENCE_SCHEMA_VERSION = "reconstruction-group-identity-evidence-v1"
AB_VERIFICATION_SCHEMA_VERSION = "reconstruction-score-ab-verification-v1"
BOOTSTRAP_DRAW_COUNT = 20_000
BOOTSTRAP_BASE_SEED = 20_260_824
BOOTSTRAP_CHUNK_SIZE = 128
MIN_VALID_BOOTSTRAP_FRACTION = 0.95
REFERENCE_METHOD_ID = "iu_pcr"
SUCCESS_FIT_STATUSES = frozenset({"OK", "OK_FALLBACK"})
ALLOWED_GROUP_UNITS = frozenset({
    "source_question_id",
    "source_prompt_id",
    "source_item_id",
    "problem_id",
})
REQUIRED_IDENTITY_CHECKS = frozenset({
    "source_hash_verified",
    "row_count_verified",
    "row_order_verified",
    "group_semantics_verified",
})
_SAFE_BOOTSTRAP_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


class EvaluationContractError(RuntimeError):
    """A fail-closed precondition or evaluation contract was violated."""


@dataclass(frozen=True)
class VerifiedCell:
    """All target-free material needed to evaluate one cell."""

    metadata: Mapping[str, Any]
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    prepared_matrix_sha256: str
    score_by_method: Mapping[str, np.ndarray]
    score_sha256_by_method: Mapping[str, str]
    fit_status_by_method: Mapping[str, str]
    fallback_reason_by_method: Mapping[str, str | None]
    group_artifact_sha256: str
    group_binding_sha256: str
    group_evidence_sha256: str
    group_source_sha256: str


@dataclass(frozen=True)
class VerifiedRelease:
    """A fully verified, still target-free 24x13 release."""

    release_root: Path
    population_id: str
    method_ids: tuple[str, ...]
    cells: Mapping[str, VerifiedCell]
    label_bundle: Path
    provenance: Mapping[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    """Read JSON while rejecting duplicate object keys."""

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise EvaluationContractError(f"duplicate JSON key {key!r} in {path}")
            output[key] = value
        return output

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"cannot read canonical JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationContractError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationContractError(message)


def _verify_payload_hash(payload: Mapping[str, Any], field: str, *, context: str) -> str:
    declared = payload.get(field)
    _require(isinstance(declared, str) and len(declared) == 64, f"{context}: missing {field}")
    body = dict(payload)
    body.pop(field, None)
    observed = sha256_bytes(canonical_json_bytes(body))
    _require(observed == declared, f"{context}: {field} mismatch")
    return observed


def _resolve_relative(base: Path, relative: Any, *, context: str) -> Path:
    _require(isinstance(relative, str) and relative.strip(), f"{context}: invalid path")
    candidate = Path(relative)
    _require(not candidate.is_absolute(), f"{context}: artifact path must be relative")
    resolved_base = base.resolve()
    resolved = (base / candidate).resolve()
    _require(
        resolved == resolved_base or resolved_base in resolved.parents,
        f"{context}: artifact path escapes its root",
    )
    _require(resolved.is_file(), f"{context}: artifact is missing: {resolved}")
    return resolved


def _resolve_provenance_path(base: Path, value: Any, *, context: str) -> Path:
    _require(isinstance(value, str) and value.strip(), f"{context}: invalid provenance path")
    candidate = Path(value)
    resolved = candidate if candidate.is_absolute() else base / candidate
    resolved = resolved.resolve()
    _require(resolved.is_file(), f"{context}: provenance artifact is missing: {resolved}")
    return resolved


def _ordered_text_hash(values: Sequence[str], *, field: str) -> str:
    return sha256_bytes(canonical_json_bytes({field: [str(value) for value in values]}))


def row_group_binding_sha256(row_ids: Sequence[str], group_ids: Sequence[str]) -> str:
    """Hash the exact row-to-source-group mapping."""

    return sha256_bytes(canonical_json_bytes({
        "row_ids": [str(value) for value in row_ids],
        "group_ids": [str(value) for value in group_ids],
    }))


def _validate_registries(cell_registry_path: Path, method_registry_path: Path) -> tuple[dict, dict]:
    cells = _load_json(cell_registry_path)
    methods = _load_json(method_registry_path)
    _require(
        cells.get("schema_version") == "reconstruction-frozen24-cell-registry-v1",
        "unexpected frozen-cell registry schema",
    )
    _require(cells.get("population_id") == "frozen24_response_v1", "unexpected population")
    _require(cells.get("positive_class") == POSITIVE_CLASS, "cell-registry positive class drift")
    cell_rows = cells.get("cells")
    _require(isinstance(cell_rows, list) and len(cell_rows) == 24, "registry must contain 24 cells")
    cell_ids = [str(row.get("cell_id", "")) for row in cell_rows if isinstance(row, dict)]
    _require(len(cell_ids) == 24 and len(set(cell_ids)) == 24, "cell IDs must be 24 unique values")
    for row in cell_rows:
        _require(isinstance(row, dict), "invalid cell-registry row")
        for field in ("cell_id", "domain", "dataset_id", "dataset_family", "model_id", "model_family"):
            _require(isinstance(row.get(field), str) and row[field], f"cell registry missing {field}")
        _require(row["domain"] in {"QA", "math"}, f"invalid domain for {row['cell_id']}")

    _require(
        methods.get("schema_version") == "reconstruction-method-registry-v1",
        "unexpected method registry schema",
    )
    _require(methods.get("score_semantics") == OUTPUT_SCORE_SEMANTICS, "method score semantics drift")
    method_rows = methods.get("methods")
    _require(isinstance(method_rows, list) and len(method_rows) == 13, "registry must contain 13 methods")
    method_ids = tuple(str(row.get("method_id", "")) for row in method_rows if isinstance(row, dict))
    _require(method_ids == tuple(PRIMARY_METHOD_IDS), "method registry and executable roster disagree")
    _require(methods.get("primary_roster_size") == 13, "primary roster size drift")
    for row in method_rows:
        method_id = row["method_id"]
        _require(
            row.get("method_version_id") == PRIMARY_METHOD_SPECS[method_id].method_version_id,
            f"method version drift: {method_id}",
        )
    return cells, methods


def _verify_input_manifest(
    release_root: Path,
    build_id: str,
    expected_cell_ids: tuple[str, ...],
    label_bundle_sha256: str,
    feature_config: Mapping[str, Any],
    feature_config_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict], dict[str, tuple[str, ...]], dict[str, str]]:
    input_root = release_root / f"build_{build_id}" / "inputs"
    manifest_path = input_root / "MANIFEST.json"
    manifest = _load_json(manifest_path)
    _verify_payload_hash(manifest, "manifest_payload_sha256", context=f"input build {build_id}")
    _require(manifest.get("schema_version") == "reconstruction-target-free-input-v1", "input schema drift")
    _require(manifest.get("build_id") == build_id, f"input build ID mismatch: {build_id}")
    _require(manifest.get("scientific_run") is True, f"build {build_id} is not scientific")
    _require(manifest.get("label_arrays_accessed") is False, f"build {build_id} opened labels")
    _require(manifest.get("feature_contract_id") == CONTRACT_VERSION, "feature contract drift")
    _require(manifest.get("source_bundle_sha256") == label_bundle_sha256, "label/source bundle hash drift")
    _require(
        manifest.get("feature_contract_config_sha256") == feature_config_sha256,
        f"input build {build_id} feature-contract config drift",
    )
    for field in (
        "transform_source_sha256",
        "orientation_source_sha256",
        "roster_source_sha256",
    ):
        _require(
            manifest.get(field) == feature_config.get(field),
            f"input build {build_id} {field} drift",
        )
    _require(manifest.get("n_cells") == 24, f"input build {build_id} is not 24-cell")
    rows = manifest.get("cells")
    _require(isinstance(rows, list) and len(rows) == 24, f"input build {build_id} has wrong cell count")
    records = {str(row.get("cell_id", "")): row for row in rows if isinstance(row, dict)}
    _require(set(records) == set(expected_cell_ids), f"input build {build_id} roster mismatch")
    row_ids: dict[str, tuple[str, ...]] = {}
    matrix_hashes: dict[str, str] = {}
    for cell_id in expected_cell_ids:
        record = records[cell_id]
        artifact = _resolve_relative(input_root, record.get("artifact_path"), context=f"input {build_id}/{cell_id}")
        _require(sha256_file(artifact) == record.get("artifact_sha256"), f"prepared artifact hash drift: {build_id}/{cell_id}")
        arrays = load_npz_no_pickle(artifact)
        _require(
            set(arrays) == {"X_confidence", "feature_names", "family_ids", "row_ids", "row_index"},
            f"prepared artifact member drift: {build_id}/{cell_id}",
        )
        rows_here = tuple(str(value) for value in arrays["row_ids"].tolist())
        names = tuple(str(value) for value in arrays["feature_names"].tolist())
        matrix = np.asarray(arrays["X_confidence"], dtype=np.float64)
        _require(len(rows_here) == record.get("n_rows"), f"prepared row count drift: {build_id}/{cell_id}")
        _require(len(set(rows_here)) == len(rows_here), f"duplicate prepared row IDs: {build_id}/{cell_id}")
        _require(
            np.array_equal(np.asarray(arrays["row_index"]), np.arange(len(rows_here))),
            f"prepared row index drift: {build_id}/{cell_id}",
        )
        observed = prepared_matrix_sha256(matrix, names, rows_here)
        row_ids[cell_id] = rows_here
        matrix_hashes[cell_id] = observed
    return manifest, records, row_ids, matrix_hashes


def _verify_freeze_manifest(
    release_root: Path,
    build_id: str,
    expected_cell_ids: tuple[str, ...],
    expected_method_ids: tuple[str, ...],
    input_manifest: Mapping[str, Any],
    input_manifest_path: Path,
    prepared_rows: Mapping[str, tuple[str, ...]],
    prepared_hashes: Mapping[str, str],
    cell_registry_sha256: str,
    method_registry_sha256: str,
    feature_registry_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict], dict[str, dict[str, np.ndarray]]]:
    fit_root = release_root / f"build_{build_id}" / "fit"
    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = _load_json(freeze_path)
    _verify_payload_hash(freeze, "payload_sha256", context=f"score freeze {build_id}")
    _require(freeze.get("schema_version") == "reconstruction-score-freeze-v1", "score-freeze schema drift")
    _require(freeze.get("build_id") == build_id, f"score-freeze build ID mismatch: {build_id}")
    _require(freeze.get("scientific_run") is True, f"score freeze {build_id} is not scientific")
    _require(freeze.get("labels_opened_by_fit") is False, f"fit build {build_id} opened labels")
    _require(freeze.get("runtime_labels_used") is False, f"fit build {build_id} used runtime labels")
    _require(
        freeze.get("preprocessing_selected_after_outcomes_were_opened") is True,
        f"fit build {build_id} omits the retrospective-preprocessing disclosure",
    )
    _require(freeze.get("evidence_status") == "D0_reused_development", f"fit build {build_id} evidence status drift")
    _require(freeze.get("all_headline_scores_present") is True, f"fit build {build_id} is incomplete")
    _require(freeze.get("feature_contract_id") == CONTRACT_VERSION, "fit feature-contract drift")
    _require(freeze.get("score_semantics") == OUTPUT_SCORE_SEMANTICS, "fit score semantics drift")
    _require(freeze.get("positive_class") == POSITIVE_CLASS, "fit positive class drift")
    _require(freeze.get("n_cells") == 24 and freeze.get("n_methods") == 13, "freeze is not 24x13")
    _require(freeze.get("n_records") == 312 and freeze.get("expected_records") == 312, "freeze lacks 312 records")
    _require(tuple(freeze.get("method_ids", ())) == expected_method_ids, "freeze method roster drift")
    _require(
        freeze.get("input_manifest_payload_sha256") == input_manifest.get("manifest_payload_sha256"),
        f"freeze {build_id} is not bound to its input manifest",
    )
    _require(
        freeze.get("input_manifest_file_sha256") == sha256_file(input_manifest_path),
        f"freeze {build_id} input-manifest file hash drift",
    )
    prefit_path = fit_root / "FIT_SOURCE_SNAPSHOT.json"
    prefit = _load_json(prefit_path)
    _verify_payload_hash(
        prefit, "payload_sha256", context=f"pre-fit source snapshot {build_id}"
    )
    _require(
        prefit.get("schema_version") == "reconstruction-fit-source-snapshot-v1",
        f"pre-fit source-snapshot schema drift: {build_id}",
    )
    _require(prefit.get("build_id") == build_id, f"pre-fit build ID drift: {build_id}")
    _require(
        freeze.get("prefit_snapshot_payload_sha256") == prefit.get("payload_sha256"),
        f"freeze {build_id} is not bound to the pre-fit snapshot payload",
    )
    _require(
        freeze.get("prefit_snapshot_file_sha256") == sha256_file(prefit_path),
        f"freeze {build_id} pre-fit snapshot file hash drift",
    )
    _require(
        prefit.get("input_manifest_payload_sha256")
        == input_manifest.get("manifest_payload_sha256"),
        f"pre-fit snapshot {build_id} input binding drift",
    )
    _require(
        tuple(prefit.get("cell_ids", ())) == expected_cell_ids,
        f"pre-fit snapshot {build_id} cell roster drift",
    )
    _require(
        tuple(prefit.get("method_ids", ())) == expected_method_ids,
        f"pre-fit snapshot {build_id} method roster drift",
    )
    _require(freeze.get("cell_registry_sha256") == cell_registry_sha256, f"freeze {build_id} cell-registry drift")
    _require(freeze.get("method_registry_sha256") == method_registry_sha256, f"freeze {build_id} method-registry drift")
    _require(freeze.get("feature_config_sha256") == feature_registry_sha256, f"freeze {build_id} feature-registry drift")
    source_snapshot = freeze.get("source_snapshot")
    _require(isinstance(source_snapshot, dict), f"freeze {build_id} lacks a source snapshot")
    _verify_payload_hash(source_snapshot, "snapshot_sha256", context=f"source snapshot {build_id}")
    _require(
        freeze.get("source_snapshot_sha256") == source_snapshot.get("snapshot_sha256"),
        f"freeze {build_id} source-snapshot binding drift",
    )
    _require(
        prefit.get("source_snapshot_sha256") == freeze.get("source_snapshot_sha256")
        and prefit.get("source_snapshot") == source_snapshot,
        f"pre-fit/freeze source snapshots differ: {build_id}",
    )
    for method_id in expected_method_ids:
        expected_spec = PRIMARY_METHOD_SPECS[method_id]
        observed_spec = freeze.get("method_specs", {}).get(method_id, {})
        _require(observed_spec.get("method_version_id") == expected_spec.method_version_id, f"freeze method version drift: {method_id}")
        _require(observed_spec.get("config_sha256") == expected_spec.config_sha256, f"freeze config drift: {method_id}")
        _require(
            isinstance(observed_spec.get("config"), dict)
            and canonical_sha256(observed_spec["config"]) == expected_spec.config_sha256,
            f"freeze config payload drift: {method_id}",
        )

    rows = freeze.get("records")
    _require(isinstance(rows, list) and len(rows) == 312, f"freeze {build_id} record count drift")
    records: dict[tuple[str, str], dict] = {}
    scores: dict[str, dict[str, np.ndarray]] = {cell_id: {} for cell_id in expected_cell_ids}
    for summary in rows:
        _require(isinstance(summary, dict), f"freeze {build_id} has a non-object record")
        key = (str(summary.get("cell_id", "")), str(summary.get("method_id", "")))
        _require(key not in records, f"duplicate freeze pair: {build_id}/{key}")
        _require(key[0] in prepared_rows and key[1] in expected_method_ids, f"unknown freeze pair: {build_id}/{key}")
        _require(summary.get("status") in SUCCESS_FIT_STATUSES, f"unsuccessful freeze pair: {build_id}/{key}")
        method_dir = fit_root / "cells" / key[0] / key[1]
        record_path = method_dir / "RECORD.json"
        _require(record_path.is_file(), f"missing score record: {build_id}/{key}")
        record_sha = sha256_file(record_path)
        _require(record_sha == summary.get("record_sha256"), f"record hash drift: {build_id}/{key}")
        record = _load_json(record_path)
        _require(record.get("schema_version") == "reconstruction-score-record-v1", f"record schema drift: {build_id}/{key}")
        for field in (
            "cell_id", "method_id", "method_version_id", "config_sha256", "status",
            "prepared_matrix_sha256", "score_sha256", "artifacts_sha256",
            "artifact_index_sha256",
        ):
            _require(record.get(field) == summary.get(field), f"freeze/record {field} mismatch: {build_id}/{key}")
        _require(record.get("population_id") == "frozen24_response_v1", f"record population drift: {build_id}/{key}")
        expected_spec = PRIMARY_METHOD_SPECS[key[1]]
        _require(record.get("method_version_id") == expected_spec.method_version_id, f"record method version drift: {build_id}/{key}")
        _require(record.get("config_sha256") == expected_spec.config_sha256, f"record method config drift: {build_id}/{key}")
        _require(record.get("feature_contract") == CONTRACT_VERSION, f"record feature contract drift: {build_id}/{key}")
        _require(record.get("score_semantics") == OUTPUT_SCORE_SEMANTICS, f"record score semantics drift: {build_id}/{key}")
        _require(record.get("positive_class") == POSITIVE_CLASS, f"record positive class drift: {build_id}/{key}")
        _require(record.get("score_semantics_conversion") == dict(SCORE_SEMANTICS_CONVERSION), f"record score conversion drift: {build_id}/{key}")
        _require(record.get("prepared_matrix_sha256") == prepared_hashes[key[0]], f"record matrix hash drift: {build_id}/{key}")
        _require(record.get("score_n") == len(prepared_rows[key[0]]), f"score length declaration drift: {build_id}/{key}")

        _require(record.get("score_path") == "score.npz", f"noncanonical score path: {build_id}/{key}")
        score_path = _resolve_relative(method_dir, record.get("score_path"), context=f"score {build_id}/{key}")
        _require(sha256_file(score_path) == record.get("score_sha256"), f"score hash drift: {build_id}/{key}")
        score_bundle = load_npz_no_pickle(score_path)
        _require(set(score_bundle) == {"row_ids", "score"}, f"score members drift: {build_id}/{key}")
        score_rows = tuple(str(value) for value in score_bundle["row_ids"].tolist())
        values = np.asarray(score_bundle["score"], dtype=np.float64)
        _require(score_rows == prepared_rows[key[0]], f"score row order drift: {build_id}/{key}")
        _require(values.shape == (len(score_rows),) and np.isfinite(values).all(), f"invalid scores: {build_id}/{key}")

        artifact_index = method_dir / "ARTIFACT_INDEX.json"
        _require(artifact_index.is_file(), f"missing artifact index: {build_id}/{key}")
        _require(sha256_file(artifact_index) == record.get("artifact_index_sha256"), f"artifact-index hash drift: {build_id}/{key}")
        if record.get("artifacts_path") is None:
            _require(record.get("artifacts_sha256") is None, f"partial artifact declaration: {build_id}/{key}")
        else:
            _require(record.get("artifacts_path") == "artifacts.npz", f"noncanonical artifact path: {build_id}/{key}")
            artifact_path = _resolve_relative(method_dir, record["artifacts_path"], context=f"artifacts {build_id}/{key}")
            _require(sha256_file(artifact_path) == record.get("artifacts_sha256"), f"artifact hash drift: {build_id}/{key}")
        records[key] = {
            "summary": summary,
            "record": record,
            "record_sha256": record_sha,
            "score_file_sha256": sha256_file(score_path),
            "artifact_index_sha256": sha256_file(artifact_index),
        }
        scores[key[0]][key[1]] = values

    expected_pairs = {(cell_id, method_id) for cell_id in expected_cell_ids for method_id in expected_method_ids}
    _require(set(records) == expected_pairs, f"freeze {build_id} does not cover exact 24x13 Cartesian product")
    return freeze, records, scores


def _verify_ab_attestation(
    release_root: Path,
    freeze_paths: Mapping[str, Path],
    input_paths: Mapping[str, Path],
    expected_cells: tuple[str, ...],
    expected_methods: tuple[str, ...],
    observed_records: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict, str]:
    path = release_root / "SCORE_AB_VERIFICATION.json"
    record = _load_json(path)
    _verify_payload_hash(record, "payload_sha256", context="independent A/B verification")
    _require(record.get("schema_version") == AB_VERIFICATION_SCHEMA_VERSION, "A/B verification schema drift")
    _require(record.get("pass") is True, "independent A/B verification did not pass")
    _require(record.get("n_cells") == 24, "A/B verification does not cover 24 cells")
    _require(record.get("n_methods") == 13, "A/B verification does not cover 13 methods")
    _require(record.get("n_pairs") == 312, "A/B verification does not cover 312 pairs")
    _require(tuple(record.get("cell_ids", ())) == expected_cells, "A/B verification cell roster drift")
    _require(tuple(record.get("method_ids", ())) == expected_methods, "A/B verification method roster drift")
    pair_rows = record.get("pairs")
    _require(isinstance(pair_rows, list) and len(pair_rows) == 312, "A/B verification pair ledger drift")
    attested_pairs: dict[tuple[str, str], dict] = {}
    for row in pair_rows:
        _require(isinstance(row, dict), "A/B verification has a non-object pair")
        key = (str(row.get("cell_id", "")), str(row.get("method_id", "")))
        _require(key not in attested_pairs, f"duplicate A/B verification pair: {key}")
        _require(key in observed_records, f"unknown A/B verification pair: {key}")
        _require(row.get("byte_identical") is True, f"A/B pair is not byte-identical: {key}")
        observed = observed_records[key]
        _require(row.get("score_sha256") == observed["score_file_sha256"], f"A/B score certificate drift: {key}")
        _require(row.get("record_sha256") == observed["record_sha256"], f"A/B record certificate drift: {key}")
        _require(row.get("artifacts_sha256") == observed["record"].get("artifacts_sha256"), f"A/B artifact certificate drift: {key}")
        attested_pairs[key] = row
    _require(set(attested_pairs) == set(observed_records), "A/B verification omits score pairs")
    for build_id in ("A", "B"):
        _require(
            record.get(f"freeze_{build_id}_sha256") == sha256_file(freeze_paths[build_id]),
            f"A/B verification freeze hash drift: {build_id}",
        )
        _require(
            record.get(f"input_manifest_{build_id}_sha256") == sha256_file(input_paths[build_id]),
            f"A/B verification input hash drift: {build_id}",
        )
    return record, sha256_file(path)


def _compare_builds(
    input_a: Mapping[str, Any],
    input_b: Mapping[str, Any],
    input_records_a: Mapping[str, dict],
    input_records_b: Mapping[str, dict],
    row_ids_a: Mapping[str, tuple[str, ...]],
    row_ids_b: Mapping[str, tuple[str, ...]],
    matrix_hash_a: Mapping[str, str],
    matrix_hash_b: Mapping[str, str],
    records_a: Mapping[tuple[str, str], dict],
    records_b: Mapping[tuple[str, str], dict],
    scores_a: Mapping[str, Mapping[str, np.ndarray]],
    scores_b: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    _require(input_a.get("source_bundle_sha256") == input_b.get("source_bundle_sha256"), "A/B source snapshots differ")
    _require(input_records_a == input_records_b, "A/B prepared input records differ")
    _require(row_ids_a == row_ids_b and matrix_hash_a == matrix_hash_b, "A/B prepared matrices differ")
    _require(set(records_a) == set(records_b), "A/B score pair sets differ")
    for key in sorted(records_a):
        a, b = records_a[key], records_b[key]
        for field in ("record_sha256", "score_file_sha256", "artifact_index_sha256"):
            _require(a[field] == b[field], f"A/B {field} mismatch: {key}")
        _require(a["record"].get("artifacts_sha256") == b["record"].get("artifacts_sha256"), f"A/B artifact hash mismatch: {key}")
        _require(np.array_equal(scores_a[key[0]][key[1]], scores_b[key[0]][key[1]]), f"A/B score values differ: {key}")


def _verify_group_sidecars(
    manifest_path: Path,
    cell_registry: Mapping[str, Any],
    input_records: Mapping[str, dict],
    prepared_rows: Mapping[str, tuple[str, ...]],
    prepared_hashes: Mapping[str, str],
    label_bundle_sha256: str,
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, str]], dict, str]:
    manifest = _load_json(manifest_path)
    _verify_payload_hash(manifest, "payload_sha256", context="group-sidecar manifest")
    _require(manifest.get("schema_version") == GROUP_SIDECAR_SCHEMA_VERSION, "group-sidecar schema drift")
    _require(manifest.get("population_id") == cell_registry.get("population_id"), "group-sidecar population drift")
    _require(manifest.get("label_bundle_sha256") == label_bundle_sha256, "group sidecars bind another source bundle")
    rows = manifest.get("cells")
    _require(isinstance(rows, list) and len(rows) == 24, "group manifest must cover 24 cells")
    by_cell = {str(row.get("cell_id", "")): row for row in rows if isinstance(row, dict)}
    _require(len(by_cell) == 24 and set(by_cell) == set(prepared_rows), "group-sidecar roster mismatch")
    registry_by_cell = {row["cell_id"]: row for row in cell_registry["cells"]}
    groups: dict[str, tuple[str, ...]] = {}
    provenance: dict[str, dict[str, str]] = {}
    hash_cache: dict[Path, str] = {}

    def cached_hash(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        return hash_cache[path]

    for cell_id in prepared_rows:
        row = by_cell[cell_id]
        context = f"group sidecar {cell_id}"
        _require(row.get("verification_status") == "VERIFIED", f"{context}: not verified")
        _require(row.get("labels_used") is False, f"{context}: group identity used labels")
        _require(row.get("group_unit") in ALLOWED_GROUP_UNITS, f"{context}: invalid group unit")
        _require(row.get("cohort_id") == input_records[cell_id].get("cohort_id"), f"{context}: cohort mismatch")
        _require(row.get("prepared_matrix_sha256") == prepared_hashes[cell_id], f"{context}: matrix mismatch")
        _require(row.get("registry_source_group_status") == registry_by_cell[cell_id].get("source_group_status"), f"{context}: registry status mismatch")

        artifact = _resolve_relative(manifest_path.parent, row.get("artifact_path"), context=context)
        artifact_sha = cached_hash(artifact)
        _require(artifact_sha == row.get("artifact_sha256"), f"{context}: artifact hash drift")
        arrays = load_npz_no_pickle(artifact)
        _require(set(arrays) == {"row_ids", "group_ids"}, f"{context}: unexpected artifact members")
        sidecar_rows = tuple(str(value) for value in arrays["row_ids"].tolist())
        sidecar_groups = tuple(str(value) for value in arrays["group_ids"].tolist())
        _require(sidecar_rows == prepared_rows[cell_id], f"{context}: row order mismatch")
        _require(len(sidecar_groups) == len(sidecar_rows), f"{context}: group length mismatch")
        _require(all(value.strip() for value in sidecar_groups), f"{context}: blank group ID")
        _require(sidecar_groups != sidecar_rows, f"{context}: row IDs were reused as IID bootstrap groups")
        binding = row_group_binding_sha256(sidecar_rows, sidecar_groups)
        _require(binding == row.get("row_group_binding_sha256"), f"{context}: row/group binding drift")
        _require(row.get("row_ids_sha256") == _ordered_text_hash(sidecar_rows, field="row_ids"), f"{context}: row hash drift")
        _require(row.get("group_ids_sha256") == _ordered_text_hash(sidecar_groups, field="group_ids"), f"{context}: group hash drift")
        _require(row.get("n_rows") == len(sidecar_rows), f"{context}: row count drift")
        _require(row.get("n_groups") == len(set(sidecar_groups)), f"{context}: group count drift")
        _require(len(set(sidecar_groups)) >= 2, f"{context}: fewer than two source groups")

        source_path = _resolve_provenance_path(manifest_path.parent, row.get("source_artifact_path"), context=context)
        source_sha = cached_hash(source_path)
        _require(source_sha == row.get("source_artifact_sha256"), f"{context}: source hash drift")
        evidence_path = _resolve_provenance_path(manifest_path.parent, row.get("identity_evidence_path"), context=context)
        evidence_sha = cached_hash(evidence_path)
        _require(evidence_sha == row.get("identity_evidence_sha256"), f"{context}: evidence hash drift")
        evidence = _load_json(evidence_path)
        _verify_payload_hash(evidence, "payload_sha256", context=f"identity evidence {cell_id}")
        _require(evidence.get("schema_version") == GROUP_EVIDENCE_SCHEMA_VERSION, f"{context}: evidence schema drift")
        _require(evidence.get("cell_id") == cell_id, f"{context}: evidence cell mismatch")
        _require(evidence.get("verification_status") == "VERIFIED", f"{context}: evidence not verified")
        _require(evidence.get("labels_used") is False, f"{context}: evidence used labels")
        _require(evidence.get("source_artifact_sha256") == source_sha, f"{context}: evidence source mismatch")
        _require(evidence.get("group_artifact_sha256") == artifact_sha, f"{context}: evidence sidecar mismatch")
        _require(evidence.get("row_group_binding_sha256") == binding, f"{context}: evidence binding mismatch")
        _require(isinstance(evidence.get("verifier_id"), str) and evidence["verifier_id"].strip(), f"{context}: missing verifier")
        _require(isinstance(evidence.get("verification_method"), str) and evidence["verification_method"].strip(), f"{context}: missing verification method")
        checks = evidence.get("checks")
        _require(isinstance(checks, dict), f"{context}: missing evidence checks")
        _require(all(checks.get(name) is True for name in REQUIRED_IDENTITY_CHECKS), f"{context}: incomplete identity audit")
        groups[cell_id] = sidecar_groups
        provenance[cell_id] = {
            "group_artifact_sha256": artifact_sha,
            "group_binding_sha256": binding,
            "group_evidence_sha256": evidence_sha,
            "group_source_sha256": source_sha,
        }
    return groups, provenance, manifest, sha256_file(manifest_path)


def verify_release_before_labels(
    *,
    release_root: str | Path,
    cell_registry_path: str | Path,
    method_registry_path: str | Path,
    feature_registry_path: str | Path,
    label_bundle: str | Path,
    group_manifest_path: str | Path,
) -> VerifiedRelease:
    """Verify the complete target-free boundary; never indexes ``__labels``."""

    release_root = Path(release_root).resolve()
    cell_registry_path = Path(cell_registry_path).resolve()
    method_registry_path = Path(method_registry_path).resolve()
    feature_registry_path = Path(feature_registry_path).resolve()
    label_bundle = Path(label_bundle).resolve()
    group_manifest_path = Path(group_manifest_path).resolve()
    _require(release_root.is_dir(), f"release root is missing: {release_root}")
    _require(label_bundle.is_file(), f"label/source bundle is missing: {label_bundle}")
    cell_registry, _method_registry = _validate_registries(cell_registry_path, method_registry_path)
    feature_config = _load_json(feature_registry_path)
    feature_repo = feature_registry_path.parents[2]
    validate_static_sources(feature_repo, feature_config)
    cell_ids = tuple(row["cell_id"] for row in cell_registry["cells"])
    method_ids = tuple(PRIMARY_METHOD_IDS)
    label_bundle_sha = sha256_file(label_bundle)  # Hashing bytes does not materialize target arrays.
    cell_registry_sha = sha256_file(cell_registry_path)
    method_registry_sha = sha256_file(method_registry_path)
    feature_registry_sha = sha256_file(feature_registry_path)

    inputs: dict[str, dict] = {}
    input_records: dict[str, dict[str, dict]] = {}
    row_ids: dict[str, dict[str, tuple[str, ...]]] = {}
    matrix_hashes: dict[str, dict[str, str]] = {}
    freezes: dict[str, dict] = {}
    records: dict[str, dict[tuple[str, str], dict]] = {}
    scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for build_id in ("A", "B"):
        inputs[build_id], input_records[build_id], row_ids[build_id], matrix_hashes[build_id] = _verify_input_manifest(
            release_root,
            build_id,
            cell_ids,
            label_bundle_sha,
            feature_config,
            feature_registry_sha,
        )
        freezes[build_id], records[build_id], scores[build_id] = _verify_freeze_manifest(
            release_root,
            build_id,
            cell_ids,
            method_ids,
            inputs[build_id],
            release_root / f"build_{build_id}" / "inputs" / "MANIFEST.json",
            row_ids[build_id],
            matrix_hashes[build_id],
            cell_registry_sha,
            method_registry_sha,
            feature_registry_sha,
        )
    _compare_builds(
        inputs["A"], inputs["B"], input_records["A"], input_records["B"],
        row_ids["A"], row_ids["B"], matrix_hashes["A"], matrix_hashes["B"],
        records["A"], records["B"], scores["A"], scores["B"],
    )
    _require(
        freezes["A"]["source_snapshot_sha256"] == freezes["B"]["source_snapshot_sha256"],
        "A/B source snapshots differ",
    )
    freeze_paths = {
        build_id: release_root / f"build_{build_id}" / "fit" / "SCORE_FREEZE_MANIFEST.json"
        for build_id in ("A", "B")
    }
    input_paths = {
        build_id: release_root / f"build_{build_id}" / "inputs" / "MANIFEST.json"
        for build_id in ("A", "B")
    }
    _ab_record, ab_sha = _verify_ab_attestation(
        release_root, freeze_paths, input_paths, cell_ids, method_ids, records["A"]
    )
    groups, group_provenance, _group_manifest, group_manifest_sha = _verify_group_sidecars(
        group_manifest_path,
        cell_registry,
        input_records["A"],
        row_ids["A"],
        matrix_hashes["A"],
        label_bundle_sha,
    )

    metadata = {row["cell_id"]: MappingProxyType(dict(row)) for row in cell_registry["cells"]}
    verified_cells: dict[str, VerifiedCell] = {}
    for cell_id in cell_ids:
        score_map = {
            method_id: np.asarray(scores["A"][cell_id][method_id], dtype=np.float64)
            for method_id in method_ids
        }
        for values in score_map.values():
            values.setflags(write=False)
        proof = group_provenance[cell_id]
        verified_cells[cell_id] = VerifiedCell(
            metadata=metadata[cell_id],
            row_ids=row_ids["A"][cell_id],
            group_ids=groups[cell_id],
            prepared_matrix_sha256=matrix_hashes["A"][cell_id],
            score_by_method=MappingProxyType(score_map),
            score_sha256_by_method=MappingProxyType({
                method_id: records["A"][(cell_id, method_id)]["score_file_sha256"]
                for method_id in method_ids
            }),
            fit_status_by_method=MappingProxyType({
                method_id: str(records["A"][(cell_id, method_id)]["record"]["status"])
                for method_id in method_ids
            }),
            fallback_reason_by_method=MappingProxyType({
                method_id: records["A"][(cell_id, method_id)]["record"].get("fallback_reason")
                for method_id in method_ids
            }),
            group_artifact_sha256=proof["group_artifact_sha256"],
            group_binding_sha256=proof["group_binding_sha256"],
            group_evidence_sha256=proof["group_evidence_sha256"],
            group_source_sha256=proof["group_source_sha256"],
        )

    provenance = {
        "cell_registry_sha256": cell_registry_sha,
        "method_registry_sha256": method_registry_sha,
        "label_bundle_sha256": label_bundle_sha,
        "group_manifest_sha256": group_manifest_sha,
        "score_ab_verification_sha256": ab_sha,
        "freeze_A_sha256": sha256_file(freeze_paths["A"]),
        "freeze_B_sha256": sha256_file(freeze_paths["B"]),
        "input_manifest_A_sha256": sha256_file(input_paths["A"]),
        "input_manifest_B_sha256": sha256_file(input_paths["B"]),
        "evaluation_module_sha256": sha256_file(Path(__file__)),
        "numpy_version": np.__version__,
        "labels_opened": False,
        "verified_cell_method_pairs": 312,
    }
    return VerifiedRelease(
        release_root=release_root,
        population_id=cell_registry["population_id"],
        method_ids=method_ids,
        cells=MappingProxyType(verified_cells),
        label_bundle=label_bundle,
        provenance=MappingProxyType(provenance),
    )


def open_correctness_labels(verified: VerifiedRelease) -> dict[str, np.ndarray]:
    """Open only correctness arrays after the target-free verification gate."""

    _require(len(verified.cells) == 24, "label gate requires 24 verified cells")
    _require(tuple(verified.method_ids) == tuple(PRIMARY_METHOD_IDS), "label gate requires the canonical 13 methods")
    _require(
        verified.provenance.get("verified_cell_method_pairs") == 312
        and isinstance(verified.provenance.get("score_ab_verification_sha256"), str),
        "label gate lacks the completed 24x13 A/B verification proof",
    )
    labels: dict[str, np.ndarray] = {}
    try:
        handle = verified.label_bundle.open("rb")
    except Exception as exc:
        raise EvaluationContractError(f"cannot open label bundle: {exc}") from exc
    with handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        _require(
            digest.hexdigest() == verified.provenance.get("label_bundle_sha256"),
            "label bundle changed after the target-free verification gate",
        )
        handle.seek(0)
        try:
            bundle = np.load(handle, allow_pickle=False)
        except Exception as exc:
            raise EvaluationContractError(f"cannot parse label bundle: {exc}") from exc
        with bundle:
            for cell_id, cell in verified.cells.items():
                member = f"{cell_id}__labels"
                _require(member in bundle.files, f"missing correctness labels: {cell_id}")
                # This line is the first semantic target-array access in the pipeline.
                y_correct = np.asarray(bundle[member])
                _require(y_correct.shape == (len(cell.row_ids),), f"label length mismatch: {cell_id}")
                _require(np.isin(y_correct, (0, 1)).all(), f"non-binary correctness labels: {cell_id}")
                y_correct = np.asarray(y_correct, dtype=np.int8)
                y_correct.setflags(write=False)
                labels[cell_id] = y_correct
    return labels


def prediction_snapshot_arrays(
    verified: VerifiedRelease,
    y_correct_by_cell: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Materialize the exact row-level inputs to the published metrics.

    The snapshot is produced only after :func:`open_correctness_labels` crosses
    the target boundary.  A later reporting step can therefore build
    ``predictions.parquet`` without reopening the label bundle or independently
    re-evaluating a method.  Every score array is the already verified Build-A
    score that was byte-identical to Build B.
    """

    _require(set(y_correct_by_cell) == set(verified.cells), "prediction label roster mismatch")
    arrays: dict[str, np.ndarray] = {}
    for cell_id, cell in verified.cells.items():
        y_correct = np.asarray(y_correct_by_cell[cell_id], dtype=np.int8)
        _require(y_correct.shape == (len(cell.row_ids),), f"prediction label length drift: {cell_id}")
        _require(np.isin(y_correct, (0, 1)).all(), f"prediction labels are non-binary: {cell_id}")
        arrays[f"{cell_id}__row_ids"] = np.asarray(cell.row_ids)
        arrays[f"{cell_id}__group_ids"] = np.asarray(cell.group_ids)
        arrays[f"{cell_id}__y_error"] = np.asarray(1 - y_correct, dtype="<i1")
        for method_id in verified.method_ids:
            score = np.asarray(cell.score_by_method[method_id], dtype="<f8")
            _require(score.shape == y_correct.shape, f"prediction score length drift: {cell_id}/{method_id}")
            _require(np.isfinite(score).all(), f"prediction score is non-finite: {cell_id}/{method_id}")
            arrays[f"{cell_id}__{method_id}__score"] = score
    return arrays


def _utf8_sorted(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values}, key=lambda value: value.encode("utf-8")))


def _bootstrap_seed(cell_id: str) -> int:
    payload = f"{EVALUATION_SCHEMA_VERSION}|{BOOTSTRAP_BASE_SEED}|{cell_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def grouped_bootstrap_multiplicity_chunks(
    group_ids: Sequence[str],
    *,
    cell_id: str = "standalone",
    draws: int = BOOTSTRAP_DRAW_COUNT,
    chunk_size: int = BOOTSTRAP_CHUNK_SIZE,
) -> tuple[tuple[str, ...], np.ndarray, int, Iterator[tuple[int, np.ndarray, bytes]]]:
    """Return a deterministic grouped-bootstrap chunk iterator.

    Each draw samples exactly ``G`` of the ``G`` verified source groups with
    replacement.  The returned row-to-group columns and every multiplicity
    chunk are shared by all methods in the cell.
    """

    _require(draws > 0 and chunk_size > 0, "bootstrap draws/chunk size must be positive")
    ordered = _utf8_sorted(group_ids)
    _require(len(ordered) >= 2, "grouped bootstrap requires at least two groups")
    columns_by_group = {group: index for index, group in enumerate(ordered)}
    group_columns = np.asarray([columns_by_group[str(value)] for value in group_ids], dtype=np.int64)
    seed = _bootstrap_seed(str(cell_id))

    def iterator() -> Iterator[tuple[int, np.ndarray, bytes]]:
        rng = np.random.Generator(np.random.PCG64(seed))
        group_count = len(ordered)
        offset = 0
        while offset < draws:
            count = min(chunk_size, draws - offset)
            sampled = rng.integers(0, group_count, size=(count, group_count), dtype=np.int64)
            multiplicities = np.zeros((count, group_count), dtype=np.int32)
            np.add.at(
                multiplicities,
                (np.repeat(np.arange(count), group_count), sampled.reshape(-1)),
                1,
            )
            _require(np.all(multiplicities.sum(axis=1) == group_count), "bootstrap draw size drift")
            yield offset, multiplicities, np.asarray(sampled, dtype="<i8").tobytes(order="C")
            offset += count

    return ordered, group_columns, seed, iterator()


def _weighted_binary_metric_draws(
    y_error: np.ndarray,
    score: np.ndarray,
    group_columns: np.ndarray,
    multiplicities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized exact weighted AUROC and average precision for shared draws."""

    y = np.asarray(y_error, dtype=np.int8)
    values = np.asarray(score, dtype=np.float64)
    columns = np.asarray(group_columns, dtype=np.int64)
    counts = np.asarray(multiplicities)
    _require(y.ndim == 1 and y.shape == values.shape == columns.shape, "metric input shape mismatch")
    _require(np.isin(y, (0, 1)).all() and np.isfinite(values).all(), "invalid metric input")
    _require(counts.ndim == 2 and counts.shape[1] > int(columns.max()), "invalid bootstrap multiplicities")
    _require(np.issubdtype(counts.dtype, np.integer) and np.all(counts >= 0), "invalid bootstrap weights")

    order = np.argsort(values, kind="mergesort")
    sorted_score = values[order]
    sorted_y = y[order]
    weights = counts[:, columns[order]].astype(np.float64, copy=False)
    starts = np.r_[0, 1 + np.flatnonzero(sorted_score[1:] != sorted_score[:-1])]
    positive_blocks = np.add.reduceat(weights * sorted_y, starts, axis=1)
    negative_blocks = np.add.reduceat(weights * (1 - sorted_y), starts, axis=1)
    positive_total = positive_blocks.sum(axis=1)
    negative_total = negative_blocks.sum(axis=1)

    negative_before = np.cumsum(negative_blocks, axis=1) - negative_blocks
    concordant = np.sum(
        positive_blocks * (negative_before + 0.5 * negative_blocks), axis=1
    )
    auc_denominator = positive_total * negative_total
    auroc = np.divide(
        concordant,
        auc_denominator,
        out=np.full(len(counts), np.nan, dtype=np.float64),
        where=auc_denominator > 0,
    )

    positive_desc = positive_blocks[:, ::-1]
    negative_desc = negative_blocks[:, ::-1]
    cumulative_positive = np.cumsum(positive_desc, axis=1)
    cumulative_total = cumulative_positive + np.cumsum(negative_desc, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    ap_numerator = np.sum(positive_desc * precision, axis=1)
    auprc = np.divide(
        ap_numerator,
        positive_total,
        out=np.full(len(counts), np.nan, dtype=np.float64),
        where=positive_total > 0,
    )
    return auroc, auprc


def _linear_quantile(values: np.ndarray, probability: float) -> float | None:
    finite = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if len(finite) == 0:
        return None
    position = (len(finite) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(finite[lower])
    fraction = position - lower
    return float(finite[lower] * (1.0 - fraction) + finite[upper] * fraction)


def _interval(draws: np.ndarray) -> dict[str, Any]:
    values = np.asarray(draws, dtype=np.float64)
    valid = int(np.isfinite(values).sum())
    return {
        "bootstrap_draws_requested": int(len(values)),
        "bootstrap_draws_valid": valid,
        "ci_lower": _linear_quantile(values, 0.025),
        "bootstrap_median": _linear_quantile(values, 0.5),
        "ci_upper": _linear_quantile(values, 0.975),
        "ci_level": 0.95,
        "ci_quantile_rule": "linear_type7",
    }


def _array_hash(values: np.ndarray, *, dtype: str) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(values, dtype=dtype)).tobytes(order="C"))


def _bootstrap_cell(
    cell_id: str,
    cell: VerifiedCell,
    y_error: np.ndarray,
    method_ids: tuple[str, ...],
    *,
    draws: int,
    chunk_size: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    ordered_groups = _utf8_sorted(cell.group_ids)
    columns_by_group = {group: index for index, group in enumerate(ordered_groups)}
    group_columns = np.asarray([columns_by_group[value] for value in cell.group_ids], dtype=np.int64)
    seed = _bootstrap_seed(cell_id)
    rng = np.random.Generator(np.random.PCG64(seed))
    output = {
        method_id: {
            "auroc": np.empty(draws, dtype=np.float64),
            "auprc": np.empty(draws, dtype=np.float64),
        }
        for method_id in method_ids
    }
    digest = hashlib.sha256()
    group_count = len(ordered_groups)
    offset = 0
    while offset < draws:
        count = min(chunk_size, draws - offset)
        sampled = rng.integers(0, group_count, size=(count, group_count), dtype=np.int64)
        digest.update(np.asarray(sampled, dtype="<i8").tobytes(order="C"))
        multiplicities = np.zeros((count, group_count), dtype=np.int32)
        np.add.at(
            multiplicities,
            (np.repeat(np.arange(count), group_count), sampled.reshape(-1)),
            1,
        )
        _require(np.all(multiplicities.sum(axis=1) == group_count), f"bootstrap size drift: {cell_id}")
        for method_id in method_ids:
            auc, ap = _weighted_binary_metric_draws(
                y_error,
                cell.score_by_method[method_id],
                group_columns,
                multiplicities,
            )
            output[method_id]["auroc"][offset:offset + count] = auc
            output[method_id]["auprc"][offset:offset + count] = ap
        offset += count
    return output, {
        "cell_id": cell_id,
        "draws": draws,
        "seed": seed,
        "rng": "numpy.PCG64",
        "resampling_unit": "verified_source_group",
        "n_groups": group_count,
        "group_order_sha256": _ordered_text_hash(ordered_groups, field="ordered_group_ids"),
        "sampled_group_positions_sha256": digest.hexdigest(),
        "shared_across_all_methods": True,
    }


def _scope_definitions(verified: VerifiedRelease) -> list[dict[str, Any]]:
    cell_ids = tuple(verified.cells)
    scopes = [{
        "scope_type": "macro24",
        "scope_value": "all_24_cells",
        "cell_ids": list(cell_ids),
        "headline_eligible": True,
    }]
    for cell_id in cell_ids:
        scopes.append({
            "scope_type": "cell",
            "scope_value": cell_id,
            "cell_ids": [cell_id],
            "headline_eligible": False,
        })
    for field, scope_type in (
        ("domain", "domain"),
        ("dataset_family", "dataset_family"),
        ("model_family", "model_family"),
    ):
        values = sorted({str(cell.metadata[field]) for cell in verified.cells.values()}, key=lambda value: value.encode("utf-8"))
        for value in values:
            scopes.append({
                "scope_type": scope_type,
                "scope_value": value,
                "cell_ids": [cell_id for cell_id, cell in verified.cells.items() if cell.metadata[field] == value],
                "headline_eligible": False,
            })
    return scopes


def evaluate_verified_release(
    verified: VerifiedRelease,
    y_correct_by_cell: Mapping[str, np.ndarray],
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAW_COUNT,
    bootstrap_chunk_size: int = BOOTSTRAP_CHUNK_SIZE,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate verified scores with paired, shared grouped-bootstrap draws."""

    _require(bootstrap_draws > 0, "bootstrap draw count must be positive")
    _require(len(verified.cells) == 24, "strict evaluator requires all 24 cells")
    _require(tuple(verified.method_ids) == tuple(PRIMARY_METHOD_IDS), "strict evaluator requires the canonical 13 methods")
    _require(set(y_correct_by_cell) == set(verified.cells), "label roster mismatch")
    scientific_draw_count = bootstrap_draws == BOOTSTRAP_DRAW_COUNT
    minimum_valid_draws = int(math.ceil(MIN_VALID_BOOTSTRAP_FRACTION * bootstrap_draws))
    method_versions = {
        method_id: PRIMARY_METHOD_SPECS[method_id].method_version_id
        for method_id in verified.method_ids
    }
    cell_metrics: list[dict[str, Any]] = []
    bootstrap: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    bootstrap_manifest: list[dict[str, Any]] = []
    label_provenance: list[dict[str, Any]] = []
    bootstrap_arrays: dict[str, np.ndarray] = {}
    fit_outcomes: list[dict[str, Any]] = []

    for cell_id, cell in verified.cells.items():
        y_correct = np.asarray(y_correct_by_cell[cell_id], dtype=np.int8)
        _require(y_correct.shape == (len(cell.row_ids),), f"label length drift: {cell_id}")
        _require(np.isin(y_correct, (0, 1)).all(), f"non-binary labels: {cell_id}")
        y_error = np.asarray(1 - y_correct, dtype=np.int8)
        for method_id in verified.method_ids:
            fit_status = cell.fit_status_by_method[method_id]
            fit_outcomes.append({
                "cell_id": cell_id,
                "method_id": method_id,
                "method_version_id": method_versions[method_id],
                "fit_status": fit_status,
                "fallback_used": fit_status == "OK_FALLBACK",
                "fallback_reason": cell.fallback_reason_by_method[method_id],
                "score_file_sha256": cell.score_sha256_by_method[method_id],
                "prepared_matrix_sha256": cell.prepared_matrix_sha256,
            })
        label_provenance.append({
            "cell_id": cell_id,
            "n_rows": len(y_error),
            "n_correct": int(y_correct.sum()),
            "n_error": int(y_error.sum()),
            "error_prevalence": float(y_error.mean()),
            "y_correct_sha256": _array_hash(y_correct, dtype="<i1"),
            "y_error_sha256": _array_hash(y_error, dtype="<i1"),
            "conversion": "y_error=1-y_correct",
        })
        two_classes = set(np.unique(y_error)) == {0, 1}
        if two_classes:
            bootstrap[cell_id], draw_record = _bootstrap_cell(
                cell_id,
                cell,
                y_error,
                verified.method_ids,
                draws=bootstrap_draws,
                chunk_size=bootstrap_chunk_size,
            )
            bootstrap_manifest.append(draw_record)
        else:
            bootstrap[cell_id] = {
                method_id: {
                    metric: np.full(bootstrap_draws, np.nan, dtype=np.float64)
                    for metric in ("auroc", "auprc")
                }
                for method_id in verified.method_ids
            }
            bootstrap_manifest.append({
                "cell_id": cell_id,
                "draws": bootstrap_draws,
                "status": "METRIC_UNDEFINED_SINGLE_CLASS",
                "shared_across_all_methods": True,
            })

        one_draw = np.ones((1, len(_utf8_sorted(cell.group_ids))), dtype=np.int32)
        group_lookup = {group: index for index, group in enumerate(_utf8_sorted(cell.group_ids))}
        group_columns = np.asarray([group_lookup[group] for group in cell.group_ids], dtype=np.int64)
        for method_id in verified.method_ids:
            if two_classes:
                point_auc, point_ap = _weighted_binary_metric_draws(
                    y_error, cell.score_by_method[method_id], group_columns, one_draw
                )
                point = {"auroc": float(point_auc[0]), "auprc": float(point_ap[0])}
            else:
                point = {"auroc": None, "auprc": None}
            for metric in ("auroc", "auprc"):
                draws_here = bootstrap[cell_id][method_id][metric]
                interval = _interval(draws_here)
                status = "OK" if point[metric] is not None else "METRIC_UNDEFINED_SINGLE_CLASS"
                if status == "OK" and interval["bootstrap_draws_valid"] < minimum_valid_draws:
                    status = "BOOTSTRAP_INSUFFICIENT_VALID_DRAWS"
                cell_metrics.append({
                    "status": status,
                    "population_id": verified.population_id,
                    "cell_id": cell_id,
                    "domain": cell.metadata["domain"],
                    "dataset_id": cell.metadata["dataset_id"],
                    "dataset_family": cell.metadata["dataset_family"],
                    "model_id": cell.metadata["model_id"],
                    "model_family": cell.metadata["model_family"],
                    "method_id": method_id,
                    "method_version_id": method_versions[method_id],
                    "metric": metric,
                    "estimate": point[metric],
                    "n_rows": len(y_error),
                    "n_groups": len(set(cell.group_ids)),
                    "positive_class": POSITIVE_CLASS,
                    "score_semantics": OUTPUT_SCORE_SEMANTICS,
                    **interval,
                })
                key = f"cell__{cell_id}__{method_id}__{metric}"
                _require(_SAFE_BOOTSTRAP_KEY.fullmatch(key) is not None, f"unsafe bootstrap key: {key}")
                bootstrap_arrays[key] = draws_here

    point_lookup = {
        (row["cell_id"], row["method_id"], row["metric"]): row["estimate"]
        for row in cell_metrics
    }
    scopes = _scope_definitions(verified)
    aggregates: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for scope in scopes:
        components = tuple(scope["cell_ids"])
        for method_id in verified.method_ids:
            for metric in ("auroc", "auprc"):
                point_values = [point_lookup[(cell_id, method_id, metric)] for cell_id in components]
                complete = len(components) > 0 and all(value is not None for value in point_values)
                matrix = np.vstack([bootstrap[cell_id][method_id][metric] for cell_id in components])
                valid_draw = np.all(np.isfinite(matrix), axis=0)
                aggregate_draws = np.full(bootstrap_draws, np.nan, dtype=np.float64)
                aggregate_draws[valid_draw] = np.mean(matrix[:, valid_draw], axis=0)
                interval = _interval(aggregate_draws)
                status = "OK" if complete else "INCOMPLETE_COMPONENT_CELLS"
                if status == "OK" and interval["bootstrap_draws_valid"] < minimum_valid_draws:
                    status = "BOOTSTRAP_INSUFFICIENT_VALID_DRAWS"
                if scope["headline_eligible"] and (len(components) != 24 or not complete):
                    status = "HEADLINE_BLOCKED_INCOMPLETE_24"
                aggregates.append({
                    "status": status,
                    "population_id": verified.population_id,
                    "scope_type": scope["scope_type"],
                    "scope_value": scope["scope_value"],
                    "cell_ids": list(components),
                    "n_cells": len(components),
                    "aggregation": "equal_cell_mean",
                    "headline_eligible": bool(scope["headline_eligible"] and status == "OK" and len(components) == 24),
                    "method_id": method_id,
                    "method_version_id": method_versions[method_id],
                    "metric": metric,
                    "estimate": float(np.mean(point_values)) if complete else None,
                    **interval,
                })

        for candidate in verified.method_ids:
            if candidate == REFERENCE_METHOD_ID:
                continue
            for metric in ("auroc", "auprc"):
                candidate_points = [point_lookup[(cell_id, candidate, metric)] for cell_id in components]
                reference_points = [point_lookup[(cell_id, REFERENCE_METHOD_ID, metric)] for cell_id in components]
                complete = all(value is not None for value in candidate_points + reference_points)
                candidate_matrix = np.vstack([bootstrap[cell_id][candidate][metric] for cell_id in components])
                reference_matrix = np.vstack([bootstrap[cell_id][REFERENCE_METHOD_ID][metric] for cell_id in components])
                valid_draw = np.all(np.isfinite(candidate_matrix) & np.isfinite(reference_matrix), axis=0)
                delta_draws = np.full(bootstrap_draws, np.nan, dtype=np.float64)
                delta_draws[valid_draw] = np.mean(
                    candidate_matrix[:, valid_draw] - reference_matrix[:, valid_draw], axis=0
                )
                if complete:
                    deltas = np.asarray(candidate_points, dtype=float) - np.asarray(reference_points, dtype=float)
                    tolerance = 1e-12
                    wins = int(np.sum(deltas > tolerance))
                    losses = int(np.sum(deltas < -tolerance))
                    ties = int(len(deltas) - wins - losses)
                    estimate = float(np.mean(deltas))
                else:
                    wins = losses = ties = 0
                    estimate = None
                interval = _interval(delta_draws)
                status = "OK" if complete else "INCOMPLETE_COMPONENT_CELLS"
                if status == "OK" and interval["bootstrap_draws_valid"] < minimum_valid_draws:
                    status = "BOOTSTRAP_INSUFFICIENT_VALID_DRAWS"
                if scope["headline_eligible"] and (len(components) != 24 or not complete):
                    status = "HEADLINE_BLOCKED_INCOMPLETE_24"
                finite_delta = delta_draws[np.isfinite(delta_draws)]
                contrasts.append({
                    "status": status,
                    "population_id": verified.population_id,
                    "scope_type": scope["scope_type"],
                    "scope_value": scope["scope_value"],
                    "cell_ids": list(components),
                    "n_cells": len(components),
                    "aggregation": "equal_cell_mean_of_paired_deltas",
                    "reference_method_id": REFERENCE_METHOD_ID,
                    "candidate_method_id": candidate,
                    "metric": metric,
                    "delta": estimate,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "tie_tolerance": 1e-12,
                    "bootstrap_probability_delta_positive": (
                        float(np.mean(finite_delta > 0.0)) if len(finite_delta) else None
                    ),
                    **interval,
                })

    macro_rows = [
        row for row in aggregates
        if row["scope_type"] == "macro24" and row["metric"] == "auroc"
    ]
    headline_ok = (
        scientific_draw_count
        and len(macro_rows) == 13
        and all(row["status"] == "OK" and row["headline_eligible"] for row in macro_rows)
    )
    evaluation = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "OK" if headline_ok else "HEADLINE_BLOCKED",
        "headline_status": "OK" if headline_ok else "HEADLINE_BLOCKED_INCOMPLETE_OR_NONCANONICAL",
        "population_id": verified.population_id,
        "positive_class": POSITIVE_CLASS,
        "label_conversion": "y_error=1-y_correct",
        "score_semantics": OUTPUT_SCORE_SEMANTICS,
        "metric_definitions": {
            "auroc": "weighted Mann-Whitney AUROC with half credit for score ties",
            "auprc": "weighted non-interpolated average precision (sklearn average_precision convention)",
        },
        "n_cells": len(verified.cells),
        "n_methods": len(verified.method_ids),
        "method_ids": list(verified.method_ids),
        "reference_method_id": REFERENCE_METHOD_ID,
        "bootstrap": {
            "draws": bootstrap_draws,
            "canonical_draw_count": BOOTSTRAP_DRAW_COUNT,
            "minimum_valid_fraction": MIN_VALID_BOOTSTRAP_FRACTION,
            "minimum_valid_draws": minimum_valid_draws,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "rng": "numpy.PCG64",
            "resampling_unit": "verified_source_group_within_cell",
            "shared_draws": "one draw stream per cell, reused by every method and paired contrast",
            "aggregate_rule": "same draw index, then equal-cell mean; invalid component makes aggregate draw invalid",
            "inference_boundary": (
                "source-group sampling uncertainty conditional on the frozen cells; "
                "cells, datasets, and model families are not resampled"
            ),
            "cell_draw_manifests": bootstrap_manifest,
        },
        "provenance": {**dict(verified.provenance), "labels_opened": True},
        "fit_outcomes": fit_outcomes,
        "label_provenance": label_provenance,
        "cell_metrics": cell_metrics,
        "aggregate_metrics": aggregates,
        "paired_contrasts_vs_iu_pcr": contrasts,
        "headline_macro24_auroc": macro_rows if headline_ok else [],
    }
    evaluation["payload_sha256"] = sha256_bytes(canonical_json_bytes(evaluation))
    return evaluation, bootstrap_arrays


__all__ = [
    "AB_VERIFICATION_SCHEMA_VERSION",
    "ALLOWED_GROUP_UNITS",
    "BOOTSTRAP_BASE_SEED",
    "BOOTSTRAP_CHUNK_SIZE",
    "BOOTSTRAP_DRAW_COUNT",
    "EVALUATION_MANIFEST_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationContractError",
    "GROUP_EVIDENCE_SCHEMA_VERSION",
    "GROUP_SIDECAR_SCHEMA_VERSION",
    "MIN_VALID_BOOTSTRAP_FRACTION",
    "REFERENCE_METHOD_ID",
    "VerifiedCell",
    "VerifiedRelease",
    "evaluate_verified_release",
    "grouped_bootstrap_multiplicity_chunks",
    "open_correctness_labels",
    "prediction_snapshot_arrays",
    "row_group_binding_sha256",
    "verify_release_before_labels",
]
