"""Fit-only external row-identity and prepared-artifact contract.

This module is deliberately independent of the external preparation and label
adapters.  It is the only external-identity module copied into the restricted
fit capsule.  Group identities, linkage rules, source paths, exclusions,
labels, and evaluation code are outside its import and filesystem closure.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import PreparedCell, prepared_matrix_sha256
from .io import canonical_json_bytes, load_npz_no_pickle, sha256_bytes, sha256_file
from ..dufs_liu_feature_contract import CONTRACT_VERSION
from ..specrage_views import FEATURE_TO_VIEW


PREPARED_SCHEMA_VERSION = "reconstruction-external-target-free-input-v2"
ID_CONTRACT_VERSION = "reconstruction-external-keyed-hmac-id-v1"
IDENTITY_KEY_CONTRACT_VERSION = "reconstruction-external-identity-key-v1"
IDENTITY_KEY_BYTES = 32
ID_DIGEST_ALGORITHM = "hmac-sha256-canonical-json-v1"
IDENTITY_KEY_ID_PREFIX = "xkidv1_"
OPAQUE_ROW_ID_PREFIX = "xridv2_"
GROUP_LINKAGE_COMMITMENT_PREFIX = "xglcv1_"
FIT_ROW_IDENTITY_SCHEMA_VERSION = "reconstruction-external-fit-row-identity-v1"
CANONICAL_FEATURE_NAMES = tuple(FEATURE_TO_VIEW)

_IDENTITY_KEY_ID_RE = re.compile(r"^xkidv1_[0-9a-f]{64}$")
_OPAQUE_ROW_ID_RE = re.compile(r"^xridv2_[0-9a-f]{64}$")
_GROUP_LINKAGE_COMMITMENT_RE = re.compile(r"^xglcv1_[0-9a-f]{64}$")


def _payload_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _identity_key_id(identity_key: bytes) -> str:
    key = bytes(identity_key)
    if len(key) != IDENTITY_KEY_BYTES:
        raise ValueError("external identity key has the wrong length")
    digest = hashlib.sha256(
        IDENTITY_KEY_CONTRACT_VERSION.encode("utf-8")
        + b"\0external_identity_key_id\0"
        + key
    ).hexdigest()
    return IDENTITY_KEY_ID_PREFIX + digest


def build_fit_row_identity_contract(
    full_binding: Mapping[str, Any],
    *,
    identity_key: bytes,
) -> dict[str, Any]:
    """Derive the fit-visible row contract and a sealed linkage commitment."""

    key_id = _identity_key_id(identity_key)
    if str(full_binding.get("key_id")) != key_id:
        raise RuntimeError("full identity contract binds another release key")
    expected_public = {
        "version": ID_CONTRACT_VERSION,
        "digest_algorithm": ID_DIGEST_ALGORITHM,
        "identity_key_contract_version": IDENTITY_KEY_CONTRACT_VERSION,
        "identity_key_bytes": IDENTITY_KEY_BYTES,
        "opaque_row_id_prefix": OPAQUE_ROW_ID_PREFIX,
        "row_namespace_scope": "cell",
        "canonical_row_order": "lexicographic_opaque_row_id",
        "key_id": key_id,
    }
    for field, expected in expected_public.items():
        if full_binding.get(field) != expected:
            raise RuntimeError(f"full identity contract row field drifted: {field}")
    private_linkage = {
        "schema_version": "reconstruction-external-private-group-linkage-v1",
        "id_contract_version": ID_CONTRACT_VERSION,
        "digest_algorithm": full_binding.get("digest_algorithm"),
        "opaque_group_id_prefix": full_binding.get("opaque_group_id_prefix"),
        "group_namespace_by_population": full_binding.get(
            "group_namespace_by_population"
        ),
    }
    commitment = GROUP_LINKAGE_COMMITMENT_PREFIX + hmac.new(
        bytes(identity_key),
        canonical_json_bytes(private_linkage),
        hashlib.sha256,
    ).hexdigest()
    value = {
        "schema_version": FIT_ROW_IDENTITY_SCHEMA_VERSION,
        **expected_public,
        "private_group_linkage_commitment": commitment,
    }
    value["contract_sha256"] = _payload_hash(value)
    return value


def validate_fit_row_identity_contract(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact row-only contract accepted by a fit worker."""

    value = dict(binding)
    expected_keys = {
        "schema_version", "version", "digest_algorithm",
        "identity_key_contract_version", "identity_key_bytes",
        "opaque_row_id_prefix", "row_namespace_scope",
        "canonical_row_order", "key_id",
        "private_group_linkage_commitment", "contract_sha256",
    }
    if set(value) != expected_keys:
        raise RuntimeError("fit row identity contract contains private/unknown fields")
    exact = {
        "schema_version": FIT_ROW_IDENTITY_SCHEMA_VERSION,
        "version": ID_CONTRACT_VERSION,
        "digest_algorithm": ID_DIGEST_ALGORITHM,
        "identity_key_contract_version": IDENTITY_KEY_CONTRACT_VERSION,
        "identity_key_bytes": IDENTITY_KEY_BYTES,
        "opaque_row_id_prefix": OPAQUE_ROW_ID_PREFIX,
        "row_namespace_scope": "cell",
        "canonical_row_order": "lexicographic_opaque_row_id",
    }
    for field, expected in exact.items():
        if value.get(field) != expected:
            raise RuntimeError(f"fit row identity contract drifted: {field}")
    if _IDENTITY_KEY_ID_RE.fullmatch(str(value.get("key_id"))) is None:
        raise RuntimeError("fit row identity key commitment is malformed")
    if _GROUP_LINKAGE_COMMITMENT_RE.fullmatch(
        str(value.get("private_group_linkage_commitment"))
    ) is None:
        raise RuntimeError("private group-linkage commitment is malformed")
    payload = dict(value)
    recorded = payload.pop("contract_sha256")
    if recorded != _payload_hash(payload):
        raise RuntimeError("fit row identity contract hash failed")
    return value


def row_namespace_sha256(*, contract: Mapping[str, Any], cell_id: str) -> str:
    value = validate_fit_row_identity_contract(contract)
    namespace = {
        "contract_version": ID_CONTRACT_VERSION,
        "identity_kind": "row",
        "scope": "cell",
        "cell_id": str(cell_id),
    }
    return _payload_hash(namespace)


def fit_row_roster_sha256(
    row_ids: Sequence[str],
    *,
    contract: Mapping[str, Any],
    row_namespace_sha256_value: str,
) -> str:
    binding = validate_fit_row_identity_contract(contract)
    rows = tuple(map(str, row_ids))
    if not rows or len(set(rows)) != len(rows) or any(
        _OPAQUE_ROW_ID_RE.fullmatch(value) is None for value in rows
    ):
        raise RuntimeError("fit opaque row roster is empty, duplicated, or malformed")
    return _payload_hash({
        "schema_version": "reconstruction-external-fit-row-roster-v3",
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": binding["contract_sha256"],
        "key_id": binding["key_id"],
        "row_namespace_sha256": str(row_namespace_sha256_value),
        "row_ids": list(rows),
    })


FIT_SAFE_PUBLIC_CELL_FIELDS = (
    "cell_id", "population_id", "dataset_id", "model_id", "slice_id", "domain",
    "comparison_group_id", "panel_role", "status", "prepared",
)
FIT_SAFE_ELIGIBLE_FIELDS = (
    "schema_version", "n_rows", "n_features", "feature_names",
    "present_feature_roster_sha256", "nominal_feature_count",
    "nominal_feature_roster_sha256", "absent_feature_names",
    "feature_contract_id", "preprocessing_steps", "mixed_v2_applied_count",
    "matrix_semantics", "prepared_matrix_sha256", "identity_contract",
    "id_contract_version", "id_contract_sha256", "identity_key_id",
    "row_namespace_sha256", "row_roster_sha256", "artifact_path",
    "artifact_sha256",
)


def validate_fit_safe_cell_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any fit record outside the exact public/eligible schema."""

    value = dict(record)
    status = str(value.get("status", ""))
    allowed = set(FIT_SAFE_PUBLIC_CELL_FIELDS)
    if status == "ELIGIBLE":
        allowed.update(FIT_SAFE_ELIGIBLE_FIELDS)
    if set(value) - allowed:
        raise RuntimeError("controller-only metadata crossed the fit cell boundary")
    required_public = {
        "cell_id", "population_id", "dataset_id", "model_id", "slice_id",
        "domain", "comparison_group_id", "panel_role", "status",
    }
    if not required_public.issubset(value):
        raise RuntimeError("fit-safe cell record lacks public task identity")
    if status == "ELIGIBLE":
        missing = set(FIT_SAFE_ELIGIBLE_FIELDS) - set(value)
        if missing:
            raise RuntimeError(f"eligible fit-safe record lacks fields: {sorted(missing)}")
        validate_fit_row_identity_contract(value["identity_contract"])
    elif value.get("prepared") is not False:
        raise RuntimeError("noneligible fit-safe record is not closed")
    return value


def load_prepared_external_cell(
    *,
    artifact_path: str | Path,
    record: Mapping[str, Any],
    identity_contract: Mapping[str, Any],
) -> PreparedCell:
    """Load one target-free NPZ under the row-only fit identity contract."""

    path = Path(artifact_path)
    if sha256_file(path) != str(record["artifact_sha256"]):
        raise RuntimeError(f"prepared artifact hash mismatch: {path}")
    arrays = load_npz_no_pickle(path)
    allowed = {
        "X_confidence", "feature_names", "family_ids", "row_ids",
        "row_index", "id_contract_version", "id_contract_sha256",
        "row_namespace_sha256", "identity_key_id",
    }
    if set(arrays) != allowed:
        raise RuntimeError(f"unexpected prepared arrays: {sorted(set(arrays) ^ allowed)}")
    names = tuple(map(str, arrays["feature_names"].tolist()))
    expected_names = tuple(name for name in CANONICAL_FEATURE_NAMES if name in set(names))
    if names != expected_names or names != tuple(record.get("feature_names", ())):
        raise RuntimeError("prepared external present-feature roster/order drifted")
    if _payload_hash(list(names)) != record.get("present_feature_roster_sha256"):
        raise RuntimeError("prepared external present-feature roster hash drifted")
    family_ids = tuple(map(str, arrays["family_ids"].tolist()))
    if family_ids != tuple(FEATURE_TO_VIEW[name] for name in names):
        raise RuntimeError("prepared external family roster/order drifted")
    rows = tuple(map(str, arrays["row_ids"].tolist()))
    if not rows or len(set(rows)) != len(rows) or any(
        _OPAQUE_ROW_ID_RE.fullmatch(value) is None for value in rows
    ):
        raise RuntimeError("prepared external row IDs are not keyed opaque v2 IDs")
    if rows != tuple(sorted(rows)):
        raise RuntimeError("prepared external row IDs are not in canonical opaque order")
    if not np.array_equal(
        np.asarray(arrays["row_index"], dtype=np.int64),
        np.arange(len(rows), dtype=np.int64),
    ):
        raise RuntimeError("prepared external row_index is not canonical")
    expected_contract = validate_fit_row_identity_contract(identity_contract)
    row_namespace = row_namespace_sha256(
        contract=expected_contract, cell_id=str(record.get("cell_id", ""))
    )
    identity_exact = {
        "identity_contract": expected_contract,
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": expected_contract["contract_sha256"],
        "identity_key_id": expected_contract["key_id"],
        "row_namespace_sha256": row_namespace,
    }
    for key, expected in identity_exact.items():
        if record.get(key) != expected:
            raise RuntimeError(f"prepared external identity binding drifted: {key}")
    for key in (
        "id_contract_version", "id_contract_sha256",
        "row_namespace_sha256", "identity_key_id",
    ):
        scalar = np.asarray(arrays[key])
        if scalar.shape != (1,) or str(scalar.tolist()[0]) != identity_exact[key]:
            raise RuntimeError(f"prepared artifact identity binding drifted: {key}")
    matrix = np.asarray(arrays["X_confidence"], dtype=np.float64)
    observed = prepared_matrix_sha256(matrix, names, rows)
    if observed != record.get("prepared_matrix_sha256"):
        raise RuntimeError("prepared matrix/row hash mismatch")
    if fit_row_roster_sha256(
        rows,
        contract=expected_contract,
        row_namespace_sha256_value=row_namespace,
    ) != record.get("row_roster_sha256"):
        raise RuntimeError("prepared opaque row-roster signature mismatch")
    return PreparedCell(
        population_id=str(record["population_id"]),
        cell_id=str(record["cell_id"]),
        domain=str(record["domain"]),
        matrix=matrix,
        feature_names=names,
        row_ids=rows,
        feature_contract=CONTRACT_VERSION,
        preprocessing_steps=(CONTRACT_VERSION,),
        preprocessed=True,
        declared_matrix_sha256=observed,
    )


__all__ = [
    "CANONICAL_FEATURE_NAMES",
    "FIT_ROW_IDENTITY_SCHEMA_VERSION",
    "FIT_SAFE_ELIGIBLE_FIELDS",
    "FIT_SAFE_PUBLIC_CELL_FIELDS",
    "GROUP_LINKAGE_COMMITMENT_PREFIX",
    "ID_CONTRACT_VERSION",
    "ID_DIGEST_ALGORITHM",
    "IDENTITY_KEY_BYTES",
    "IDENTITY_KEY_CONTRACT_VERSION",
    "OPAQUE_ROW_ID_PREFIX",
    "PREPARED_SCHEMA_VERSION",
    "build_fit_row_identity_contract",
    "fit_row_roster_sha256",
    "load_prepared_external_cell",
    "row_namespace_sha256",
    "validate_fit_row_identity_contract",
    "validate_fit_safe_cell_record",
]
