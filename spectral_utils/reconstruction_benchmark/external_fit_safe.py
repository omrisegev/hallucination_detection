"""Fit-capsule-only validation of redacted external inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .external_fit_contract import (
    CANONICAL_FEATURE_NAMES,
    ID_CONTRACT_VERSION,
    PREPARED_SCHEMA_VERSION,
    load_prepared_external_cell,
    validate_fit_row_identity_contract,
    validate_fit_safe_cell_record,
)
from .io import canonical_json_bytes, sha256_bytes, sha256_file
from ..dufs_liu_feature_contract import CONTRACT_VERSION


FIT_SAFE_INPUT_MANIFEST_SCHEMA_VERSION = "reconstruction-external-fit-safe-build-v1"
FEATURE_CONFIG_PATH = "configs/reconstruction_benchmark_v1/fit_safe_feature_contract.json"
TRANSFORM_SOURCE_PATH = "spectral_utils/dufs_liu_feature_contract.py"
ORIENTATION_SOURCE_PATH = "spectral_utils/feature_contract.py"
FEATURE_ROSTER_SOURCE_PATH = "spectral_utils/specrage_views.py"

# Exact top-level registry visible to fitting.  In particular there is no
# source path, target/class count, exclusion policy, group count, group ID,
# linkage rule, label adapter, or post-freeze metadata field.
FIT_SAFE_INPUT_MANIFEST_FIELDS = frozenset({
    "schema_version", "prepared_cell_schema_version", "identity_contract",
    "id_contract_version", "release_id", "build_id",
    "scientific_full_build", "applicability_complete",
    "complete_eligible_roster", "external_registry_sha256",
    "population_registry_sha256", "preparation_manifest_sha256",
    "preparation_manifest_payload_sha256",
    "preparation_attestation_sha256", "feature_contract_id",
    "mixed_v2_applied_exactly_once", "target_data_opened",
    "historical_scores_opened", "n_registered_cells", "n_runnable_cells",
    "n_prepared_cells", "cells", "payload_sha256",
})


def _payload_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _safe_child(root: Path, relative: str, *, description: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{description} escapes its root: {relative!r}") from error
    return candidate


def current_fit_feature_contract_bindings(repo: str | Path) -> dict[str, Any]:
    root = Path(repo).resolve()
    config_path = root / FEATURE_CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "reconstruction-feature-contract-v1":
        raise RuntimeError("unexpected reconstruction feature-contract schema")
    if config.get("contract_id") != CONTRACT_VERSION:
        raise RuntimeError("feature-contract config names another executable contract")
    if (
        config.get("preprocessing_count") != 1
        or config.get("nominal_feature_count") != len(CANONICAL_FEATURE_NAMES)
    ):
        raise RuntimeError("feature-contract preprocessing count or roster size drifted")
    expected_sources = {
        "transform_source": TRANSFORM_SOURCE_PATH,
        "orientation_source": ORIENTATION_SOURCE_PATH,
    }
    for key, relative in expected_sources.items():
        if config.get(key) != relative:
            raise RuntimeError(f"feature-contract {key} changed")
        if config.get(f"{key}_sha256") != sha256_file(root / relative):
            raise RuntimeError(f"feature-contract {key} hash is stale")
    declared = str(config.get("roster_source", ""))
    if not declared:
        raise RuntimeError("feature-contract config lacks its roster source")
    roster_path = _safe_child(root, declared, description="declared contract roster")
    if config.get("roster_source_sha256") != sha256_file(roster_path):
        raise RuntimeError("feature-contract roster-source hash is stale")
    return {
        "feature_contract_id": CONTRACT_VERSION,
        "feature_contract_config_sha256": sha256_file(config_path),
        "transform_source_sha256": sha256_file(root / TRANSFORM_SOURCE_PATH),
        "orientation_source_sha256": sha256_file(root / ORIENTATION_SOURCE_PATH),
        "feature_roster_source_sha256": sha256_file(root / FEATURE_ROSTER_SOURCE_PATH),
        "declared_roster_source": declared,
        "declared_roster_source_sha256": sha256_file(roster_path),
        "nominal_feature_roster_sha256": _payload_hash(list(CANONICAL_FEATURE_NAMES)),
        "nominal_feature_count": len(CANONICAL_FEATURE_NAMES),
    }


def validate_fit_safe_input_manifest(
    path: str | Path,
    *,
    repo: str | Path,
    input_root: str | Path | None = None,
    require_scientific: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = dict(manifest)
    recorded = payload.pop("payload_sha256", None)
    if recorded != _payload_hash(payload):
        raise RuntimeError("fit-safe external manifest payload hash failed")
    if manifest.get("schema_version") != FIT_SAFE_INPUT_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unexpected fit-safe external manifest schema")
    if set(manifest) != FIT_SAFE_INPUT_MANIFEST_FIELDS:
        raise RuntimeError(
            "fit-safe external manifest contains controller-only/unknown fields"
        )
    if require_scientific and manifest.get("scientific_full_build") is not True:
        raise RuntimeError("partial fit-safe external input cannot be scientific")
    if require_scientific and (
        manifest.get("applicability_complete") is not True
        or manifest.get("complete_eligible_roster") is not True
    ):
        raise RuntimeError("fit-safe external applicability roster is incomplete")
    if manifest.get("target_data_opened") is not False:
        raise RuntimeError("fit-safe manifest does not prove target isolation")
    if manifest.get("historical_scores_opened") is not False:
        raise RuntimeError("fit-safe manifest opened historical scores")
    if manifest.get("feature_contract_id") != CONTRACT_VERSION:
        raise RuntimeError("fit-safe manifest binds another feature contract")
    if manifest.get("mixed_v2_applied_exactly_once") is not True:
        raise RuntimeError("fit-safe manifest does not attest one mixed-v2 pass")
    for key in (
        "external_registry_sha256", "population_registry_sha256",
        "preparation_manifest_sha256", "preparation_manifest_payload_sha256",
        "preparation_attestation_sha256",
    ):
        value = str(manifest.get(key, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(f"fit-safe external manifest has malformed {key}")
    identity = validate_fit_row_identity_contract(manifest.get("identity_contract", {}))
    if manifest.get("id_contract_version") != ID_CONTRACT_VERSION:
        raise RuntimeError("fit-safe external identity version is stale")
    rows = manifest.get("cells", ())
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("fit-safe external manifest has no cells")
    identifiers = [str(row.get("cell_id", "")) for row in rows]
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise RuntimeError("fit-safe external manifest has empty or duplicate cells")
    registered = int(manifest.get("n_registered_cells", -1))
    if (
        (require_scientific and registered != len(rows))
        or (not require_scientific and registered < len(rows))
    ):
        raise RuntimeError("fit-safe external registered-cell count drifted")
    artifact_root = Path(input_root) if input_root is not None else manifest_path.parent
    eligible: list[str] = []
    for row in rows:
        validate_fit_safe_cell_record(row)
        if row.get("status") != "ELIGIBLE":
            continue
        eligible.append(str(row["cell_id"]))
        exact = {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "feature_contract_id": CONTRACT_VERSION,
            "preprocessing_steps": [CONTRACT_VERSION],
            "mixed_v2_applied_count": 1,
            "identity_contract": identity,
            "id_contract_version": ID_CONTRACT_VERSION,
            "id_contract_sha256": identity["contract_sha256"],
            "identity_key_id": identity["key_id"],
        }
        for key, expected in exact.items():
            if row.get(key) != expected:
                raise RuntimeError(f"{row.get('cell_id')}: fit-safe field {key} drifted")
        names = tuple(map(str, row.get("feature_names", ())))
        if names != tuple(name for name in CANONICAL_FEATURE_NAMES if name in set(names)):
            raise RuntimeError(f"{row.get('cell_id')}: feature roster is invalid")
        relative = str(row.get("artifact_path", ""))
        if not relative:
            raise RuntimeError(f"{row.get('cell_id')}: prepared artifact path is absent")
        load_prepared_external_cell(
            artifact_path=_safe_child(
                artifact_root, relative, description="fit-safe prepared artifact"
            ),
            record=row,
            identity_contract=identity,
        )
    if int(manifest.get("n_prepared_cells", -1)) != len(eligible):
        raise RuntimeError("fit-safe external prepared-cell count drifted")
    manifest["_eligible_cell_ids"] = eligible
    manifest["_validated_feature_contract_bindings"] = (
        current_fit_feature_contract_bindings(repo)
    )
    return manifest


__all__ = [
    "FEATURE_CONFIG_PATH",
    "FIT_SAFE_INPUT_MANIFEST_SCHEMA_VERSION",
    "FIT_SAFE_INPUT_MANIFEST_FIELDS",
    "current_fit_feature_contract_bindings",
    "validate_fit_safe_input_manifest",
]
