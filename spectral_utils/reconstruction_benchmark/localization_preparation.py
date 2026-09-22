"""Target-free preparation for ProcessBench and PRMBench localization.

The sole response-risk source is a previously signed external-final-answer
A/B build.  This module verifies that certificate, joins the exact opaque row
IDs, extracts only token telemetry and step boundaries, applies the frozen
29-stream mixed-v2 transform once, and emits the narrow fit mount.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .external_ab import assert_external_ab_certificate
from .external_final_answer import (
    ExternalCellSpec,
    ExternalRegistry,
    apply_external_id_contract,
    external_id_contract_binding,
    identity_key_id,
    load_external_registry,
    load_identity_key,
    load_raw_feature_cell,
    resolve_sources,
    verify_sources,
)
from .external_fit_contract import (
    ID_CONTRACT_VERSION,
    build_fit_row_identity_contract,
    fit_row_roster_sha256,
    row_namespace_sha256,
)
from .io import (
    atomic_write_json,
    atomic_write_npz,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from .localization_contract import (
    FIT_MANIFEST_SCHEMA_VERSION,
    FIT_SAFE_CELL_FIELDS,
    PREPARED_SCHEMA_VERSION,
    TOKEN_CONTRACT_ID,
    assert_no_target_named_members,
    load_localization_registry,
    payload_sha256,
)
from .methods import PRIMARY_METHOD_IDS
from ..fixed_application_pipelines import (
    SHARED_TOKEN_VIEWS,
    fit_shared_mixed_transformer,
    raw_token_feature_matrix,
)


PREPARATION_PROVENANCE_SCHEMA_VERSION = "reconstruction-localization-preparation-v1"
RAW_CONTAINER_NOTICE = (
    "The trusted preparation adapter unpickles a container that co-locates task "
    "targets, but selects no target value. The restricted fit mount contains only "
    "opaque IDs, scores, token coordinates, and step boundaries."
)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _external_score_freeze_root(
    release_root: Path,
    external_release_id: str,
    build_id: str,
) -> tuple[Path, dict[str, Any]]:
    fit_root = (
        release_root / external_release_id / f"build_{build_id}"
        / "external_final_answer" / "fit"
    )
    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    return fit_root, freeze


def _load_external_response_scores(
    *,
    fit_root: Path,
    freeze: Mapping[str, Any],
    cell_id: str,
    expected_row_ids: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    records = {
        (str(row["cell_id"]), str(row["method_id"])): row
        for row in freeze.get("records", ())
    }
    if len(records) != len(freeze.get("records", ())):
        raise RuntimeError("external score freeze contains duplicate records")
    scores: list[np.ndarray] = []
    bindings: list[dict[str, Any]] = []
    expected_rows = tuple(map(str, expected_row_ids))
    for method_id in PRIMARY_METHOD_IDS:
        record = records.get((cell_id, method_id))
        if record is None or record.get("status") not in {"OK", "OK_FALLBACK"}:
            raise RuntimeError(f"{cell_id}/{method_id}: signed external score is absent")
        record_path = fit_root / str(record["record_path"])
        score_path = fit_root / str(record["score_path"])
        if sha256_file(record_path) != record.get("record_sha256"):
            raise RuntimeError(f"{cell_id}/{method_id}: external score record hash failed")
        if sha256_file(score_path) != record.get("score_sha256"):
            raise RuntimeError(f"{cell_id}/{method_id}: external score artifact hash failed")
        full_record = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            full_record.get("score_semantics") != "higher_is_incorrect"
            or full_record.get("positive_class") != "incorrect"
            or full_record.get("method_id") != method_id
        ):
            raise RuntimeError(f"{cell_id}/{method_id}: external score semantics drifted")
        arrays = load_npz_no_pickle(score_path)
        if "row_ids" not in arrays or "score" not in arrays:
            raise RuntimeError(f"{cell_id}/{method_id}: external score arrays are incomplete")
        rows = tuple(map(str, arrays["row_ids"].tolist()))
        values = np.asarray(arrays["score"], dtype=np.float64)
        if rows != expected_rows or values.shape != (len(rows),) or not np.isfinite(values).all():
            raise RuntimeError(f"{cell_id}/{method_id}: external score cohort/order drifted")
        scores.append(values)
        bindings.append({
            "cell_id": cell_id,
            "method_id": method_id,
            "method_version_id": record["method_version_id"],
            "config_sha256": record["config_sha256"],
            "record_sha256": record["record_sha256"],
            "score_sha256": record["score_sha256"],
            "row_roster_sha256": record["row_roster_sha256"],
        })
    return np.vstack(scores), bindings


def _source_rows(
    spec: ExternalCellSpec,
    source_path: Path,
    admitted_raw_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    cache = _load_pickle(source_path)
    if not isinstance(cache, Mapping):
        raise RuntimeError(f"{spec.cell_id}: token telemetry cache is not a mapping")
    by_id: dict[str, Mapping[str, Any]] = {}
    for key in sorted(cache, key=lambda value: (str(type(value)), str(value))):
        row = cache[key]
        if not isinstance(row, Mapping):
            raise RuntimeError(f"{spec.cell_id}: token telemetry row is not a mapping")
        raw_id = str(row.get("id" if spec.dataset_id == "processbench" else "idx", ""))
        if not raw_id or raw_id in by_id:
            raise RuntimeError(f"{spec.cell_id}: raw token row identity is empty/duplicated")
        by_id[raw_id] = row
    admitted = tuple(map(str, admitted_raw_ids))
    if any(raw_id not in by_id for raw_id in admitted):
        raise RuntimeError(f"{spec.cell_id}: response and token cohorts do not join")
    return [by_id[raw_id] for raw_id in admitted]


def _transformer_binding(transformer: Any) -> dict[str, Any]:
    arrays = {
        "raw_median": np.asarray(transformer.raw_median, dtype="<f8"),
        "oriented_mean": np.asarray(transformer.oriented_mean, dtype="<f8"),
        "oriented_std": np.asarray(transformer.oriented_std, dtype="<f8"),
        "mode_centres": np.asarray(transformer.mode_centres, dtype="<f8"),
        "output_mean": np.asarray(transformer.output_mean, dtype="<f8"),
        "output_std": np.asarray(transformer.output_std, dtype="<f8"),
    }
    array_hashes = {
        name: sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))
        for name, value in arrays.items()
    }
    sorted_hashes = [
        sha256_bytes(np.ascontiguousarray(value, dtype="<f8").tobytes(order="C"))
        for value in transformer.sorted_oriented
    ]
    value = {
        "schema_version": "localization-token-transform-binding-v1",
        "token_contract_id": TOKEN_CONTRACT_ID,
        "names": list(transformer.names),
        "array_sha256": array_hashes,
        "sorted_oriented_sha256": sorted_hashes,
        "n_fit_tokens": int(len(transformer.training_output)),
        "preprocessing_count": 1,
        "matrix_semantics": "higher_is_confidence",
    }
    value["binding_sha256"] = payload_sha256(value)
    return value


def prepare_localization_cell(
    *,
    registry: ExternalRegistry,
    spec: ExternalCellSpec,
    source_root: str | Path,
    fit_root: str | Path,
    external_fit_root: str | Path,
    external_score_freeze: Mapping[str, Any],
    external_certificate: Mapping[str, Any],
    identity_key: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = resolve_sources(registry, spec, repo=source_root)
    verified_sources = verify_sources(sources, include_labels=False)
    if len(sources.feature_files) != 1:
        raise RuntimeError(f"{spec.cell_id}: localization requires one telemetry cache")
    raw = load_raw_feature_cell(spec, sources)
    identity = apply_external_id_contract(
        registry, spec, raw.row_ids, raw.group_ids, identity_key=identity_key
    )
    order = np.asarray(
        sorted(range(len(identity.row_ids)), key=lambda index: identity.row_ids[index]),
        dtype=np.int64,
    )
    row_ids = tuple(identity.row_ids[index] for index in order.tolist())
    if row_ids != tuple(sorted(row_ids)):
        raise AssertionError("localization opaque row ordering failed")
    response_scores, response_bindings = _load_external_response_scores(
        fit_root=Path(external_fit_root),
        freeze=external_score_freeze,
        cell_id=spec.cell_id,
        expected_row_ids=row_ids,
    )

    rows_raw_order = _source_rows(spec, sources.feature_files[0].path, raw.row_ids)
    rows = [rows_raw_order[index] for index in order.tolist()]
    raw_token_records: list[tuple[str, np.ndarray]] = []
    local_spans: list[list[tuple[int, int]]] = []
    for opaque_id, row in zip(row_ids, rows):
        matrix = raw_token_feature_matrix(row)
        spans = row.get("step_token_spans")
        if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)) or not spans:
            raise RuntimeError(f"{spec.cell_id}: admitted row lacks step spans")
        clean_spans: list[tuple[int, int]] = []
        for span in spans:
            if (
                span is None or not isinstance(span, Sequence) or len(span) != 2
                or int(span[0]) < 0 or int(span[1]) <= int(span[0])
                or int(span[1]) > len(matrix)
            ):
                raise RuntimeError(f"{spec.cell_id}: admitted row has an invalid step span")
            clean_spans.append((int(span[0]), int(span[1])))
        raw_token_records.append((opaque_id, matrix))
        local_spans.append(clean_spans)

    transformer = fit_shared_mixed_transformer(raw_token_records)
    transform_binding = _transformer_binding(transformer)
    token_parts: list[np.ndarray] = []
    token_offsets = [0]
    segment_offsets = [0]
    segment_starts: list[int] = []
    segment_ends: list[int] = []
    for (_, raw_matrix), spans in zip(raw_token_records, local_spans):
        confidence = np.asarray(transformer.transform(raw_matrix), dtype=np.float64)
        if confidence.shape != raw_matrix.shape or not np.isfinite(confidence).all():
            raise RuntimeError(f"{spec.cell_id}: token mixed-v2 output is invalid")
        base = token_offsets[-1]
        token_parts.append(confidence)
        token_offsets.append(base + len(confidence))
        for lo, hi in spans:
            segment_starts.append(base + lo)
            segment_ends.append(base + hi)
        segment_offsets.append(len(segment_starts))
    token_confidence = np.vstack(token_parts)

    fit_identity = build_fit_row_identity_contract(
        external_id_contract_binding(registry, identity_key=identity_key),
        identity_key=identity_key,
    )
    if fit_identity != external_certificate.get("identity_contract"):
        raise RuntimeError("localization and signed external identity contracts disagree")
    fit_namespace = row_namespace_sha256(contract=fit_identity, cell_id=spec.cell_id)
    if fit_namespace != identity.row_namespace_sha256:
        raise RuntimeError("localization and external row namespaces disagree")
    row_roster = fit_row_roster_sha256(
        row_ids, contract=fit_identity, row_namespace_sha256_value=fit_namespace
    )
    response_binding_hash = payload_sha256(response_bindings)
    certificate_sha = str(external_certificate["certificate_sha256"])
    target = Path(fit_root) / "cells" / f"{spec.cell_id}.npz"
    arrays = {
        "token_confidence": token_confidence.astype("<f8", copy=False),
        "token_offsets": np.asarray(token_offsets, dtype="<i8"),
        "segment_offsets": np.asarray(segment_offsets, dtype="<i8"),
        "segment_starts": np.asarray(segment_starts, dtype="<i8"),
        "segment_ends": np.asarray(segment_ends, dtype="<i8"),
        "row_ids": np.asarray(row_ids, dtype="<U80"),
        "response_scores": response_scores.astype("<f8", copy=False),
        "method_ids": np.asarray(PRIMARY_METHOD_IDS, dtype="<U48"),
        "id_contract_version": np.asarray([ID_CONTRACT_VERSION], dtype="<U64"),
        "id_contract_sha256": np.asarray([fit_identity["contract_sha256"]], dtype="<U64"),
        "identity_key_id": np.asarray([fit_identity["key_id"]], dtype="<U80"),
        "row_namespace_sha256": np.asarray([fit_namespace], dtype="<U64"),
        "external_certificate_sha256": np.asarray([certificate_sha], dtype="<U64"),
        "external_score_bindings_sha256": np.asarray([response_binding_hash], dtype="<U64"),
        "token_transform_sha256": np.asarray([transform_binding["binding_sha256"]], dtype="<U64"),
    }
    assert_no_target_named_members(tuple(arrays))
    artifact_sha = atomic_write_npz(target, arrays)
    fit_record = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "cell_id": spec.cell_id,
        "population_id": spec.population_id,
        "dataset_id": spec.dataset_id,
        "model_id": spec.model_id,
        "slice_id": spec.slice_id,
        "status": "ELIGIBLE",
        "n_rows": len(row_ids),
        "n_tokens": len(token_confidence),
        "n_segments": len(segment_starts),
        "n_token_streams": len(SHARED_TOKEN_VIEWS),
        "method_ids": list(PRIMARY_METHOD_IDS),
        "token_contract_id": TOKEN_CONTRACT_ID,
        "token_mixed_v2_applied_count": 1,
        "token_matrix_semantics": "higher_is_confidence",
        "identity_contract": fit_identity,
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": fit_identity["contract_sha256"],
        "identity_key_id": fit_identity["key_id"],
        "row_namespace_sha256": fit_namespace,
        "row_roster_sha256": row_roster,
        "external_certificate_sha256": certificate_sha,
        "external_score_bindings_sha256": response_binding_hash,
        "token_transform_sha256": transform_binding["binding_sha256"],
        "artifact_path": target.relative_to(Path(fit_root)).as_posix(),
        "artifact_sha256": artifact_sha,
    }
    if set(fit_record) != FIT_SAFE_CELL_FIELDS:
        raise AssertionError("localization fit-safe record schema drifted")
    provenance = {
        **fit_record,
        "adapter_id": spec.adapter_id,
        "comparison_group_id": spec.comparison_group_id,
        "source_files": verified_sources,
        "raw_container_notice": RAW_CONTAINER_NOTICE,
        "raw_container_target_fields_co_located": True,
        "target_values_selected": False,
        "step_boundaries_selected": True,
        "response_score_bindings": response_bindings,
        "token_transform_binding": transform_binding,
    }
    return fit_record, provenance


def prepare_localization_build(
    *,
    repo: str | Path,
    localization_registry_path: str | Path,
    external_registry_path: str | Path,
    population_registry_path: str | Path,
    release_root: str | Path,
    localization_release_id: str,
    external_release_id: str,
    build_id: str,
    source_root: str | Path,
    identity_key_path: str | Path,
    scientific_full: bool = True,
) -> dict[str, Any]:
    if build_id not in {"A", "B"}:
        raise ValueError("localization build must be A or B")
    repo_path = Path(repo).resolve()
    release_root_path = Path(release_root).resolve()
    config = load_localization_registry(localization_registry_path)
    registry = load_external_registry(
        repo=repo_path,
        registry_path=external_registry_path,
        population_registry_path=population_registry_path,
    )
    certificate_path = (
        release_root_path / external_release_id / "external_final_answer"
        / "AB_VERIFICATION.json"
    )
    certificate = assert_external_ab_certificate(
        certificate_path,
        release_id=external_release_id,
        release_root=release_root_path,
        selected_build=build_id,
        registry=registry,
        repo=repo_path,
    )
    identity_key = load_identity_key(identity_key_path)
    if identity_key_id(identity_key) != certificate["identity_contract"]["key_id"]:
        raise RuntimeError("localization identity key differs from external certificate")
    external_fit_root, freeze = _external_score_freeze_root(
        release_root_path, external_release_id, build_id
    )
    required_cells = [
        *config["processbench"]["source_cells"],
        config["prmbench"]["source_cell"],
    ]
    if any(cell_id not in certificate["cell_ids"] for cell_id in required_cells):
        raise RuntimeError("signed external release lacks a required localization cell")

    build_root = (
        release_root_path / localization_release_id / f"build_{build_id}"
        / "localization"
    )
    input_root = build_root / "inputs"
    if input_root.exists() and any(input_root.iterdir()):
        raise FileExistsError(f"localization input root is not empty: {input_root}")
    (input_root / "cells").mkdir(parents=True, exist_ok=False)
    controller_root = (
        release_root_path.parent / "private_control" / localization_release_id
        / "localization" / f"build_{build_id}" / "preparation_provenance"
    )
    if controller_root.exists() and any(controller_root.iterdir()):
        raise FileExistsError(f"localization provenance root is not empty: {controller_root}")
    controller_root.mkdir(parents=True, exist_ok=False)

    fit_cells: list[dict[str, Any]] = []
    provenance_cells: list[dict[str, Any]] = []
    for cell_id in required_cells:
        fit_record, provenance = prepare_localization_cell(
            registry=registry,
            spec=registry.by_cell[cell_id],
            source_root=source_root,
            fit_root=input_root,
            external_fit_root=external_fit_root,
            external_score_freeze=freeze,
            external_certificate=certificate,
            identity_key=identity_key,
        )
        fit_cells.append(fit_record)
        provenance_cells.append(provenance)
    fit_identity = certificate["identity_contract"]
    fit_manifest = {
        "schema_version": FIT_MANIFEST_SCHEMA_VERSION,
        "release_id": localization_release_id,
        "external_release_id": external_release_id,
        "build_id": build_id,
        "scientific_full_build": bool(scientific_full),
        "target_values_selected": False,
        "historical_localization_scores_opened": False,
        "external_certificate_sha256": certificate["certificate_sha256"],
        "external_registry_sha256": registry.sha256,
        "method_registry_sha256": certificate["method_registry_sha256"],
        "identity_contract": fit_identity,
        "id_contract_version": ID_CONTRACT_VERSION,
        "token_contract_id": TOKEN_CONTRACT_ID,
        "token_mixed_v2_applied_exactly_once": True,
        "n_cells": len(fit_cells),
        "cells": fit_cells,
    }
    fit_manifest["payload_sha256"] = payload_sha256(fit_manifest)
    atomic_write_json(input_root / "MANIFEST.json", fit_manifest)
    provenance = {
        "schema_version": PREPARATION_PROVENANCE_SCHEMA_VERSION,
        "release_id": localization_release_id,
        "external_release_id": external_release_id,
        "build_id": build_id,
        "localization_registry_sha256": sha256_file(localization_registry_path),
        "external_registry_sha256": registry.sha256,
        "population_registry_sha256": registry.population_registry_sha256,
        "external_certificate_path": str(certificate_path),
        "external_certificate_sha256": certificate["certificate_sha256"],
        "external_score_freeze_sha256": sha256_file(
            external_fit_root / "SCORE_FREEZE_MANIFEST.json"
        ),
        "identity_key_id": fit_identity["key_id"],
        "target_values_selected": False,
        "raw_container_notice": RAW_CONTAINER_NOTICE,
        "fit_manifest_sha256": sha256_file(input_root / "MANIFEST.json"),
        "fit_manifest_payload_sha256": fit_manifest["payload_sha256"],
        "cells": provenance_cells,
    }
    # Comparator containers co-locate targets and scores.  Declassify them to
    # a score-only tree outside the fit mount; the fit worker never receives
    # this directory or its raw source paths.
    from .localization_comparators import project_localization_comparators

    comparator_manifest = project_localization_comparators(
        localization_registry_path=localization_registry_path,
        registry=registry,
        source_root=source_root,
        output_root=build_root / "comparator_projections",
        identity_key=identity_key,
        build_id=build_id,
    )
    provenance["comparator_projection_manifest_sha256"] = sha256_file(
        build_root / "comparator_projections" / "MANIFEST.json"
    )
    provenance["comparator_projection_payload_sha256"] = comparator_manifest[
        "payload_sha256"
    ]
    provenance["payload_sha256"] = payload_sha256(provenance)
    atomic_write_json(controller_root / "MANIFEST.json", provenance)
    return fit_manifest


__all__ = [
    "PREPARATION_PROVENANCE_SCHEMA_VERSION", "prepare_localization_build",
    "prepare_localization_cell",
]
