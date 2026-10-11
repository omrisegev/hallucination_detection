"""Fit-only execution boundary for the EDIS/AIME reconstruction lane.

This module intentionally has no pickle import, raw-source resolver, identity
key loader, label adapter, class-count registry, group IDs, or evaluator.  It
accepts only the public fit-safe registry and prepared NPZ matrices.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..dufs_liu_feature_contract import CONTRACT_VERSION
from ..specrage_views import FEATURE_TO_VIEW
from .contracts import PreparedCell, prepared_matrix_sha256
from .io import (
    atomic_write_json,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from .methods import PRIMARY_METHOD_IDS, PRIMARY_METHOD_SPECS, run_method
from .serialization import write_score_result


FIT_REGISTRY_SCHEMA = "reconstruction-edis-fit-safe-registry-v1"
SCORE_FREEZE_SCHEMA = "reconstruction-edis-score-freeze-v2-firewalled"
PREFIT_SCHEMA = "reconstruction-edis-prefit-v2-controller-worker"
WORKER_RESULT_SCHEMA = "reconstruction-edis-fit-worker-result-v1"
_OPAQUE_ROW = re.compile(r"^xridv2_[0-9a-f]{64}$")
_SUCCESS = {"OK", "OK_FALLBACK"}
_PREPARED_MEMBERS = {
    "X_confidence", "feature_names", "family_ids", "row_ids", "row_index",
    "identity_contract_version", "identity_key_id",
}


def _payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _verify_payload(value: Mapping[str, Any], *, name: str) -> None:
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != _payload_sha256(payload):
        raise RuntimeError(f"{name} payload hash failed")


def load_fit_registry(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    _verify_payload(raw, name="EDIS fit registry")
    if raw.get("schema_version") != FIT_REGISTRY_SCHEMA:
        raise RuntimeError("unexpected EDIS fit-registry schema")
    scientific_full = raw.get("scientific_full_build") is True
    descriptive_partial = raw.get("partial_descriptive_build") is True
    if scientific_full == descriptive_partial:
        raise RuntimeError("EDIS fit registry must be exactly one of full or partial")
    required = {
        "feature_contract_id": CONTRACT_VERSION,
        "nominal_feature_count": len(FEATURE_TO_VIEW),
        "method_roster": "all_13_primary_methods",
        "score_semantics": "higher_is_incorrect",
        "mixed_v2_applied_exactly_once": True,
        "labels_opened": False,
        "historical_scores_opened": False,
        "raw_sources_serialized": False,
        "headline_eligible": False,
        "aggregate_metrics_allowed": scientific_full,
        "fit_registry_available": True,
        "status_only_build": False,
        "status_roster_contract_match": True,
        "trace_status_contract_id": "edis-frozen-min-trace-status-v1-2026-08-24",
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise RuntimeError(f"EDIS fit-registry {key} attestation failed")
    cells = raw.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RuntimeError("EDIS fit registry contains no runnable cells")
    registered = int(raw.get("registered_cell_count", -1))
    ready = int(raw.get("ready_cell_count", -1))
    blocked = int(raw.get("blocked_cell_count", -1))
    if (
        registered != 12
        or ready != len(cells)
        or ready + blocked != registered
        or (scientific_full and (ready != 12 or blocked != 0))
        or (descriptive_partial and not (0 < ready < 12 and blocked > 0))
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(raw.get("preparation_status_commitment_sha256", ""))
        )
    ):
        raise RuntimeError("EDIS fit registry cell-status accounting drifted")
    ids = [str(item.get("cell_id", "")) for item in cells]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("EDIS fit registry has empty or duplicate cell IDs")
    forbidden = {
        "source", "source_path", "raw_path", "manifest", "questions",
        "samples_per_question_temperature", "group_ids", "group_count",
        "group_membership", "expected_correct", "expected_incorrect",
        "gate_status", "gate_reasons",
    }
    for cell in cells:
        lowered = {str(key).lower() for key in cell}
        if lowered & forbidden:
            raise RuntimeError("EDIS fit registry exposes source, target, or group structure")
        if int(cell.get("mixed_v2_applied_count", -1)) != 1:
            raise RuntimeError(f"{cell.get('cell_id')}: mixed-v2 count is not exactly one")
        if cell.get("preprocessing_steps") != [CONTRACT_VERSION]:
            raise RuntimeError(f"{cell.get('cell_id')}: preprocessing contract drifted")
    binding = raw.get("identity_contract", {})
    if not isinstance(binding, Mapping):
        raise RuntimeError("EDIS fit registry lacks its public identity binding")
    binding_payload = dict(binding)
    binding_sha = binding_payload.pop("contract_sha256", None)
    if binding_sha != _payload_sha256(binding_payload):
        raise RuntimeError("EDIS fit registry identity-binding hash failed")
    if (
        not str(binding.get("contract_version", "")).strip()
        or not re.fullmatch(r"xkidv1_[0-9a-f]{64}", str(binding.get("key_id", "")))
        or binding.get("opaque_row_id_prefix") != "xridv2_"
        or binding.get("canonical_row_order") != "lexicographic_opaque_row_id"
        or binding.get("row_namespace_scope") != "dataset_temperature_cell"
    ):
        raise RuntimeError("EDIS fit registry has an invalid public identity binding")
    if set(binding) != {
        "contract_version", "digest_algorithm", "identity_key_contract_version",
        "identity_key_bytes", "opaque_row_id_prefix", "canonical_row_order",
        "row_namespace_scope", "key_id", "contract_sha256",
    }:
        raise RuntimeError("EDIS fit registry identity binding is not row-only")
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(raw.get("private_identity_contract_commitment_sha256", "")),
    ):
        raise RuntimeError("EDIS fit registry lacks its private identity commitment")
    for cell in cells:
        if cell.get("identity_contract") != binding:
            raise RuntimeError(f"{cell.get('cell_id')}: cell identity binding drifted")
    return raw


def load_prepared_cell(
    *, artifact_path: str | Path, record: Mapping[str, Any], identity_binding: Mapping[str, Any]
) -> PreparedCell:
    path = Path(artifact_path)
    if sha256_file(path) != str(record.get("artifact_sha256")):
        raise RuntimeError(f"prepared artifact hash mismatch: {path}")
    arrays = load_npz_no_pickle(path)
    if set(arrays) != _PREPARED_MEMBERS:
        raise RuntimeError(f"prepared EDIS artifact exposes unexpected members: {sorted(set(arrays) ^ _PREPARED_MEMBERS)}")
    names = tuple(map(str, arrays["feature_names"].tolist()))
    expected_names = tuple(name for name in FEATURE_TO_VIEW if name in set(names))
    if names != expected_names or tuple(record.get("feature_names", ())) != names:
        raise RuntimeError("prepared EDIS feature roster drifted")
    if tuple(map(str, arrays["family_ids"].tolist())) != tuple(FEATURE_TO_VIEW[name] for name in names):
        raise RuntimeError("prepared EDIS feature-family roster drifted")
    rows = tuple(map(str, arrays["row_ids"].tolist()))
    if (
        not rows
        or len(set(rows)) != len(rows)
        or rows != tuple(sorted(rows))
        or any(_OPAQUE_ROW.fullmatch(row) is None for row in rows)
    ):
        raise RuntimeError("prepared EDIS rows are not canonical keyed opaque IDs")
    if _payload_sha256(list(rows)) != record.get("row_roster_sha256"):
        raise RuntimeError("prepared EDIS row roster commitment failed")
    if not np.array_equal(np.asarray(arrays["row_index"], dtype=np.int64), np.arange(len(rows), dtype=np.int64)):
        raise RuntimeError("prepared EDIS row index is not canonical")
    contract_scalar = np.asarray(arrays["identity_contract_version"])
    key_scalar = np.asarray(arrays["identity_key_id"])
    if contract_scalar.shape != (1,) or key_scalar.shape != (1,):
        raise RuntimeError("prepared EDIS identity bindings are not scalar")
    if str(contract_scalar.tolist()[0]) != str(identity_binding.get("contract_version")):
        raise RuntimeError("prepared EDIS identity-contract version drifted")
    if str(key_scalar.tolist()[0]) != str(identity_binding.get("key_id")):
        raise RuntimeError("prepared EDIS key ID drifted")
    matrix = np.asarray(arrays["X_confidence"], dtype=np.float64)
    observed = prepared_matrix_sha256(matrix, names, rows)
    if observed != record.get("prepared_matrix_sha256"):
        raise RuntimeError("prepared EDIS matrix commitment failed")
    return PreparedCell(
        population_id=str(record["population_id"]),
        cell_id=str(record["cell_id"]),
        domain="multi_sample_trace_detection",
        matrix=matrix,
        feature_names=names,
        row_ids=rows,
        feature_contract=CONTRACT_VERSION,
        preprocessing_steps=(CONTRACT_VERSION,),
        preprocessed=True,
        declared_matrix_sha256=observed,
    )


def validate_method_registry(repo: Path) -> str:
    path = repo / "configs/reconstruction_benchmark_v1/methods.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = tuple(str(item.get("method_id")) for item in raw.get("methods", ()))
    if ids != PRIMARY_METHOD_IDS or int(raw.get("primary_roster_size", -1)) != 13:
        raise RuntimeError("EDIS executable roster is not the canonical 13 methods")
    by_id = {str(item["method_id"]): item for item in raw["methods"]}
    for method_id in PRIMARY_METHOD_IDS:
        spec = PRIMARY_METHOD_SPECS[method_id]
        if by_id[method_id].get("method_version_id") != spec.method_version_id:
            raise RuntimeError(f"{method_id}: method version drifted")
        if by_id[method_id].get("config_sha256") != spec.config_sha256:
            raise RuntimeError(f"{method_id}: method config drifted")
    return sha256_file(path)


def validate_fit_safe_feature_contract(repo: Path) -> str:
    path = repo / "configs/reconstruction_benchmark_v1/fit_safe_feature_contract.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "reconstruction-feature-contract-v1",
        "contract_id": CONTRACT_VERSION,
        "nominal_feature_count": len(FEATURE_TO_VIEW),
        "preprocessing_count": 1,
        "matrix_semantics": "higher_is_confidence",
        "fit_boundary": (
            "prepared mixed-v2 matrix only; no raw paths or target metadata"
        ),
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise RuntimeError(f"fit-safe feature contract {key} drifted")
    bindings = (
        ("transform_source", "transform_source_sha256"),
        ("orientation_source", "orientation_source_sha256"),
        ("roster_source", "roster_source_sha256"),
    )
    for path_key, hash_key in bindings:
        relative = str(raw.get(path_key, ""))
        source = (repo / relative).resolve()
        try:
            source.relative_to(repo.resolve())
        except ValueError as error:
            raise RuntimeError("fit-safe feature source escapes capsule") from error
        if sha256_file(source) != raw.get(hash_key):
            raise RuntimeError(f"fit-safe feature source hash failed: {relative}")
    return sha256_file(path)


def run_fit_worker(
    *,
    release_id: str,
    build_id: str,
    input_root: str | Path,
    fit_root: str | Path,
    repo: str | Path,
    audit_policy_sha256: str,
    denial_probes: Sequence[Mapping[str, Any]],
    requested_cells: Sequence[str] | None = None,
    requested_methods: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    """Fit prepared matrices inside an already-installed irreversible firewall."""

    if build_id not in {"A", "B"}:
        raise ValueError("build_id must be A or B")
    inputs = Path(input_root).resolve()
    output = Path(fit_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"EDIS fit directory is not empty: {fit_root}")
    output.mkdir(parents=True, exist_ok=False)
    fit_registry_path = inputs / "FIT_REGISTRY.json"
    fit_registry = load_fit_registry(fit_registry_path)
    if fit_registry.get("release_id") != release_id or fit_registry.get("build_id") != build_id:
        raise RuntimeError("EDIS fit registry release/build binding failed")
    repo_root = Path(repo).resolve()
    method_registry_sha = validate_method_registry(repo_root)
    feature_contract_sha = validate_fit_safe_feature_contract(repo_root)
    method_ids = tuple(requested_methods or PRIMARY_METHOD_IDS)
    if len(set(method_ids)) != len(method_ids):
        raise ValueError("duplicate EDIS worker method request")
    if any(method_id not in PRIMARY_METHOD_IDS for method_id in method_ids):
        raise KeyError("EDIS worker requested a method outside the primary roster")
    cells = list(fit_registry["cells"])
    requested = None if requested_cells is None else set(requested_cells)
    if requested is not None:
        available = {str(item["cell_id"]) for item in cells}
        if requested - available:
            raise KeyError(
                f"EDIS worker requested unavailable cells: {sorted(requested - available)}"
            )
        cells = [item for item in cells if item["cell_id"] in requested]
    if not cells or not method_ids:
        raise RuntimeError("EDIS worker has no cells or methods to fit")

    records: list[dict[str, Any]] = []
    for cell_record in cells:
        cell = load_prepared_cell(
            artifact_path=inputs / cell_record["artifact_path"],
            record=cell_record,
            identity_binding=fit_registry["identity_contract"],
        )
        cell_root = output / "cells" / cell.cell_id
        method_records: list[dict[str, Any]] = []
        for method_id in method_ids:
            result = run_method(method_id, cell)
            record = write_score_result(result, cell.row_ids, cell_root / method_id)
            worker_record = {
                "cell_id": cell.cell_id,
                "population_id": cell.population_id,
                "method_id": method_id,
                "method_version_id": record["method_version_id"],
                "config_sha256": record["config_sha256"],
                "status": record["status"],
                "prepared_matrix_sha256": record["prepared_matrix_sha256"],
                "row_roster_sha256": cell_record["row_roster_sha256"],
                "score_sha256": record["score_sha256"],
                "record_sha256": record["record_sha256"],
                "artifacts_sha256": record["artifacts_sha256"],
                "artifact_index_sha256": record["artifact_index_sha256"],
                "record_path": (cell_root / method_id / "RECORD.json").relative_to(output).as_posix(),
                "score_path": (
                    (cell_root / method_id / "score.npz").relative_to(output).as_posix()
                    if record["score_path"] else None
                ),
                "artifacts_path": (
                    (cell_root / method_id / "artifacts.npz").relative_to(output).as_posix()
                    if record["artifacts_path"] else None
                ),
                "artifact_index_path": (cell_root / method_id / "ARTIFACT_INDEX.json").relative_to(output).as_posix(),
            }
            records.append(worker_record)
            method_records.append(worker_record)
        atomic_write_json(cell_root / "CELL_FIT_MANIFEST.json", {
            "schema_version": "reconstruction-edis-cell-fit-v1-worker-candidate",
            "cell_id": cell.cell_id,
            "population_id": cell.population_id,
            "prepared_matrix_sha256": cell.matrix_sha256,
            "row_roster_sha256": cell_record["row_roster_sha256"],
            "identity_contract": fit_registry["identity_contract"],
            "target_data_opened": False,
            "raw_sources_opened": False,
            "group_structure_opened": False,
            "method_records": method_records,
        })
    expected = len(cells) * len(method_ids)
    complete = (
        len(records) == expected
        and len({(row["cell_id"], row["method_id"]) for row in records}) == expected
        and all(row["status"] in _SUCCESS and row["score_sha256"] for row in records)
    )
    result_manifest = {
        "schema_version": WORKER_RESULT_SCHEMA,
        "release_id": release_id,
        "build_id": build_id,
        "all_candidate_scores_present": bool(complete),
        "scientific_full_build": fit_registry["scientific_full_build"],
        "partial_descriptive_build": fit_registry["partial_descriptive_build"],
        "headline_eligible": False,
        "aggregate_metrics_allowed": fit_registry["aggregate_metrics_allowed"],
        "preparation_status_commitment_sha256": fit_registry[
            "preparation_status_commitment_sha256"
        ],
        "fit_registry_sha256": sha256_file(fit_registry_path),
        "fit_registry_payload_sha256": fit_registry["payload_sha256"],
        "method_registry_sha256": method_registry_sha,
        "fit_safe_feature_contract_sha256": feature_contract_sha,
        "audit_policy_sha256": audit_policy_sha256,
        "denial_probes": [dict(item) for item in denial_probes],
        "firewall_violations": [],
        "method_ids": list(method_ids),
        "cell_ids": [item["cell_id"] for item in cells],
        "n_records": len(records),
        "expected_records": expected,
        "target_data_opened": False,
        "raw_sources_opened": False,
        "group_structure_opened": False,
        "historical_scores_opened": False,
        "records": records,
    }
    result_manifest["payload_sha256"] = _payload_sha256(result_manifest)
    atomic_write_json(output / "WORKER_RESULT_MANIFEST.json", result_manifest)
    if not complete:
        raise RuntimeError("EDIS fit worker candidates are incomplete")
    return result_manifest


__all__ = [
    "FIT_REGISTRY_SCHEMA",
    "PREFIT_SCHEMA",
    "SCORE_FREEZE_SCHEMA",
    "WORKER_RESULT_SCHEMA",
    "load_fit_registry",
    "load_prepared_cell",
    "run_fit_worker",
    "validate_fit_safe_feature_contract",
    "validate_method_registry",
]
