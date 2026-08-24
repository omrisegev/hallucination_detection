"""Strict, target-free contracts for the reconstruction localization lane.

The response component is never fitted here.  It is imported only from a
passing external-final-answer A/B certificate.  The task-specific fit capsule
receives an already oriented 29-stream token matrix, opaque row identities,
step boundaries, and the thirteen signed response-risk vectors.  It receives
no source paths, group linkage, labels, error-family names, or comparator data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
REGISTRY_SCHEMA_VERSION = "reconstruction-localization-registry-v1"
PREPARED_SCHEMA_VERSION = "reconstruction-localization-target-free-input-v1"
FIT_MANIFEST_SCHEMA_VERSION = "reconstruction-localization-fit-safe-build-v1"
SCORE_SCHEMA_VERSION = "reconstruction-localization-score-bundle-v1"
SCORE_FREEZE_SCHEMA_VERSION = "reconstruction-localization-score-freeze-v1"
TOKEN_CONTRACT_ID = "localization-token-iu29-mixed-v2-v1"
COMBINED_ADAPTER_ID = "response-token-midrank-geomean-v1"
RESPONSE_ONLY_ADAPTER_ID = "response-midrank-only-null-v1"
TOKEN_ONLY_ADAPTER_ID = "token-iu29-midrank-only-null-v1"
FIT_TOKEN_CAP = 60_000
TOKEN_STREAM_COUNT = 29
ID_CONTRACT_VERSION = "reconstruction-external-keyed-hmac-id-v1"
OPAQUE_ROW_ID_PREFIX = "xridv2_"
FIT_ROW_IDENTITY_SCHEMA_VERSION = "reconstruction-external-fit-row-identity-v1"
ID_DIGEST_ALGORITHM = "hmac-sha256-canonical-json-v1"
IDENTITY_KEY_CONTRACT_VERSION = "reconstruction-external-identity-key-v1"
IDENTITY_KEY_BYTES = 32
NO_ERROR = -1

_OPAQUE_ROW_RE = re.compile(rf"^{re.escape(OPAQUE_ROW_ID_PREFIX)}[0-9a-f]{{64}}$")
_TARGET_FRAGMENTS = (
    "label", "target", "correct", "incorrect", "classification",
    "error", "answer", "outcome", "group", "source_idx", "first_error",
)


def payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def validate_fit_row_identity_contract(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Fit-capsule-local validation of the external row-only identity contract."""

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
    if re.fullmatch(r"xkidv1_[0-9a-f]{64}", str(value.get("key_id"))) is None:
        raise RuntimeError("fit identity key commitment is malformed")
    if re.fullmatch(
        r"xglcv1_[0-9a-f]{64}", str(value.get("private_group_linkage_commitment"))
    ) is None:
        raise RuntimeError("fit group-linkage commitment is malformed")
    payload = dict(value)
    recorded = payload.pop("contract_sha256")
    if recorded != payload_sha256(payload):
        raise RuntimeError("fit row identity contract hash failed")
    return value


def load_localization_registry(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise RuntimeError("unexpected localization registry schema")
    pb = value.get("processbench", {})
    models = tuple(map(str, pb.get("models", ())))
    subsets = tuple(map(str, pb.get("subsets", ())))
    cells = tuple(map(str, pb.get("source_cells", ())))
    if len(models) != 3 or len(subsets) != 4 or len(cells) != 12:
        raise RuntimeError("localization registry is not the exact 3x4 ProcessBench roster")
    expected = {
        f"processbench_{subset}_{model}" for model in models for subset in subsets
    }
    if set(cells) != expected:
        raise RuntimeError("ProcessBench source cell roster drifted")
    populations = pb.get("population_id_by_model", {})
    row_counts = pb.get("expected_rows_by_subset", {})
    balances = pb.get("expected_first_error_balance_by_subset", {})
    threshold = pb.get("threshold_fit", {})
    bootstrap = pb.get("bootstrap", {})
    if (
        set(populations) != set(models)
        or any(not str(populations[model]) for model in models)
        or set(row_counts) != set(subsets)
        or set(balances) != set(subsets)
        or any(
            int(balances[subset].get("error", -1))
            + int(balances[subset].get("clean", -1))
            != int(row_counts[subset])
            for subset in subsets
        )
        or int(threshold.get("folds", -1)) != 5
        or threshold.get("stage") != "post-score-freeze only"
        or int(bootstrap.get("draws", -1)) != 20_000
        or bootstrap.get("unit") != "source question"
        or bootstrap.get("paired") is not True
    ):
        raise RuntimeError("ProcessBench population/count/crossfit/bootstrap contract drifted")
    prm = value.get("prmbench", {})
    families = tuple(map(str, prm.get("error_families", ())))
    if len(families) != 9 or len(set(families)) != 9 or "multi_solutions" not in families:
        raise RuntimeError("PRMBench registry must expose all nine error families")
    expected_by_family = prm.get("expected_by_family", {})
    prm_bootstrap = prm.get("bootstrap", {})
    if (
        prm.get("source_cell") != "prmbench_response_qwen3_8b"
        or prm.get("source_slice_id") != "overall"
        or set(expected_by_family) != set(families)
        or sum(int(expected_by_family[family].get("responses", -1)) for family in families)
        != int(prm.get("expected_error_responses", -2))
        or sum(int(expected_by_family[family].get("steps", -1)) for family in families)
        != int(prm.get("expected_steps", -2))
        or sum(int(expected_by_family[family].get("positive_steps", -1)) for family in families)
        != int(prm.get("expected_positive_steps", -2))
        or int(expected_by_family["multi_solutions"].get("positive_steps", -1)) != 0
        or int(prm_bootstrap.get("draws", -1)) != 20_000
        or prm_bootstrap.get("unit") != "source_idx"
        or prm_bootstrap.get("paired") is not True
    ):
        raise RuntimeError("PRMBench population/count/single-class/bootstrap contract drifted")
    adapter = value.get("adapter_contract", {})
    if (
        adapter.get("adapter_id") != COMBINED_ADAPTER_ID
        or adapter.get("historical_075_025_blend") != "FORBIDDEN"
        or int(adapter.get("primary_system_count", -1)) != 13
    ):
        raise RuntimeError("localization adapter contract drifted")
    token = value.get("token_contract", {})
    if token.get("contract_id") != TOKEN_CONTRACT_ID or int(token.get("n_streams", -1)) != 29:
        raise RuntimeError("localization token contract drifted")
    comparators = value.get("comparators", ())
    comparator_ids = [str(row.get("system_id", "")) for row in comparators]
    if (
        len(comparators) != 4
        or len(set(comparator_ids)) != 4
        or sum(row.get("dataset_id") == "processbench" for row in comparators) != 3
        or sum(row.get("dataset_id") == "prmbench" for row in comparators) != 1
        or any(
            not row.get("access_level") or not row.get("fidelity")
            or "only" not in str(row.get("projection", ""))
            for row in comparators
        )
    ):
        raise RuntimeError("localization comparator access/fidelity/projection roster drifted")
    return value


def primary_system_roster(method_ids: Sequence[str]) -> tuple[dict[str, str], ...]:
    methods = tuple(map(str, method_ids))
    if len(methods) != 13 or len(set(methods)) != 13:
        raise ValueError("the localization adapter requires the exact 13 response methods")
    rows: list[dict[str, str]] = []
    for method_id in methods:
        rows.append({
            "method_id": method_id,
            "adapter_id": COMBINED_ADAPTER_ID,
            "system_id": f"{method_id}__loc_geomean_v1",
            "role": "primary_localization_adapter",
        })
    for method_id in methods:
        rows.append({
            "method_id": method_id,
            "adapter_id": RESPONSE_ONLY_ADAPTER_ID,
            "system_id": f"{method_id}__response_only_null_v1",
            "role": "adapter_ablation",
        })
    rows.append({
        "method_id": "token_iu29",
        "adapter_id": TOKEN_ONLY_ADAPTER_ID,
        "system_id": "token_iu29__step_only_null_v1",
        "role": "adapter_ablation",
    })
    if len(rows) != 27 or len({row["system_id"] for row in rows}) != 27:
        raise AssertionError("localization system roster is not 27 unique systems")
    return tuple(rows)


def empirical_midrank(values: Sequence[float]) -> np.ndarray:
    """Return deterministic empirical midranks ``(rank - 0.5) / n``.

    Equal values receive the average of their occupied one-based ranks.  The
    output is strictly between zero and one, so the registered geometric mean
    needs no arbitrary epsilon or clipping constant.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("midrank input must be a nonempty finite vector")
    order = np.argsort(array, kind="mergesort")
    ranked = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        average_one_based = 0.5 * ((start + 1) + end)
        ranked[order[start:end]] = (average_one_based - 0.5) / len(array)
        start = end
    if not ((ranked > 0.0) & (ranked < 1.0)).all():
        raise AssertionError("empirical midranks left the open unit interval")
    return ranked


@dataclass(frozen=True)
class PreparedLocalizationCell:
    cell_id: str
    population_id: str
    dataset_id: str
    model_id: str
    slice_id: str
    row_ids: tuple[str, ...]
    token_confidence: np.ndarray
    token_offsets: np.ndarray
    segment_offsets: np.ndarray
    segment_starts: np.ndarray
    segment_ends: np.ndarray
    response_scores: np.ndarray
    method_ids: tuple[str, ...]
    identity_contract: Mapping[str, Any]
    external_certificate_sha256: str
    external_score_bindings_sha256: str
    token_transform_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        rows = tuple(map(str, self.row_ids))
        if not rows or rows != tuple(sorted(rows)) or len(set(rows)) != len(rows):
            raise RuntimeError("localization prepared rows are not canonical and unique")
        if any(_OPAQUE_ROW_RE.fullmatch(value) is None for value in rows):
            raise RuntimeError("localization prepared rows are not keyed opaque IDs")
        methods = tuple(map(str, self.method_ids))
        if len(methods) != 13 or len(set(methods)) != 13:
            raise RuntimeError("localization prepared response roster is not 13 methods")
        token = np.asarray(self.token_confidence, dtype=np.float64)
        if token.ndim != 2 or token.shape[1] != TOKEN_STREAM_COUNT:
            raise RuntimeError("localization token matrix is not tokens x 29")
        if not len(token) or not np.isfinite(token).all():
            raise RuntimeError("localization token matrix is empty or non-finite")
        token_offsets = np.asarray(self.token_offsets, dtype=np.int64)
        segment_offsets = np.asarray(self.segment_offsets, dtype=np.int64)
        starts = np.asarray(self.segment_starts, dtype=np.int64)
        ends = np.asarray(self.segment_ends, dtype=np.int64)
        if (
            token_offsets.shape != (len(rows) + 1,)
            or segment_offsets.shape != (len(rows) + 1,)
            or token_offsets[0] != 0
            or token_offsets[-1] != len(token)
            or segment_offsets[0] != 0
            or segment_offsets[-1] != len(starts)
            or starts.shape != ends.shape
            or np.any(np.diff(token_offsets) <= 0)
            or np.any(np.diff(segment_offsets) <= 0)
        ):
            raise RuntimeError("localization token/segment offsets are malformed")
        for row_index in range(len(rows)):
            token_lo, token_hi = token_offsets[row_index:row_index + 2]
            seg_lo, seg_hi = segment_offsets[row_index:row_index + 2]
            if np.any(starts[seg_lo:seg_hi] < token_lo) or np.any(ends[seg_lo:seg_hi] > token_hi):
                raise RuntimeError("a localization segment escapes its response token range")
        if np.any(ends <= starts):
            raise RuntimeError("localization segments must be nonempty half-open spans")
        response = np.asarray(self.response_scores, dtype=np.float64)
        if response.shape != (len(methods), len(rows)) or not np.isfinite(response).all():
            raise RuntimeError("localization response score matrix is malformed")
        validate_fit_row_identity_contract(self.identity_contract)
        for value in (
            self.external_certificate_sha256,
            self.external_score_bindings_sha256,
            self.token_transform_sha256,
            self.artifact_sha256,
        ):
            if len(str(value)) != 64:
                raise RuntimeError("localization prepared hash binding is malformed")
        for array in (token, token_offsets, segment_offsets, starts, ends, response):
            array.setflags(write=False)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "method_ids", methods)
        object.__setattr__(self, "token_confidence", token)
        object.__setattr__(self, "token_offsets", token_offsets)
        object.__setattr__(self, "segment_offsets", segment_offsets)
        object.__setattr__(self, "segment_starts", starts)
        object.__setattr__(self, "segment_ends", ends)
        object.__setattr__(self, "response_scores", response)


FIT_SAFE_CELL_FIELDS = frozenset({
    "schema_version", "cell_id", "population_id", "dataset_id", "model_id",
    "slice_id", "status", "n_rows", "n_tokens", "n_segments",
    "n_token_streams", "method_ids", "token_contract_id",
    "token_mixed_v2_applied_count", "token_matrix_semantics",
    "identity_contract", "id_contract_version", "id_contract_sha256",
    "identity_key_id", "row_namespace_sha256", "row_roster_sha256",
    "external_certificate_sha256", "external_score_bindings_sha256",
    "token_transform_sha256", "artifact_path", "artifact_sha256",
})


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(arrays[name])
    if value.shape != (1,):
        raise RuntimeError(f"prepared localization scalar {name} is malformed")
    return str(value.tolist()[0])


def load_prepared_localization_cell(
    artifact_path: str | Path,
    record: Mapping[str, Any],
) -> PreparedLocalizationCell:
    if set(record) != FIT_SAFE_CELL_FIELDS:
        raise RuntimeError("fit-safe localization record contains controller-only fields")
    if record.get("schema_version") != PREPARED_SCHEMA_VERSION or record.get("status") != "ELIGIBLE":
        raise RuntimeError("localization fit cell is not eligible")
    path = Path(artifact_path)
    if sha256_file(path) != record.get("artifact_sha256"):
        raise RuntimeError("localization prepared artifact hash mismatch")
    arrays = load_npz_no_pickle(path)
    allowed = {
        "token_confidence", "token_offsets", "segment_offsets", "segment_starts",
        "segment_ends", "row_ids", "response_scores", "method_ids",
        "id_contract_version", "id_contract_sha256", "identity_key_id",
        "row_namespace_sha256", "external_certificate_sha256",
        "external_score_bindings_sha256", "token_transform_sha256",
    }
    if set(arrays) != allowed:
        raise RuntimeError("localization prepared artifact contains unknown arrays")
    exact_scalars = {
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": str(record["id_contract_sha256"]),
        "identity_key_id": str(record["identity_key_id"]),
        "row_namespace_sha256": str(record["row_namespace_sha256"]),
        "external_certificate_sha256": str(record["external_certificate_sha256"]),
        "external_score_bindings_sha256": str(record["external_score_bindings_sha256"]),
        "token_transform_sha256": str(record["token_transform_sha256"]),
    }
    for name, expected in exact_scalars.items():
        if _scalar(arrays, name) != expected:
            raise RuntimeError(f"localization prepared scalar drifted: {name}")
    value = PreparedLocalizationCell(
        cell_id=str(record["cell_id"]),
        population_id=str(record["population_id"]),
        dataset_id=str(record["dataset_id"]),
        model_id=str(record["model_id"]),
        slice_id=str(record["slice_id"]),
        row_ids=tuple(map(str, arrays["row_ids"].tolist())),
        token_confidence=np.asarray(arrays["token_confidence"], dtype=np.float64),
        token_offsets=np.asarray(arrays["token_offsets"], dtype=np.int64),
        segment_offsets=np.asarray(arrays["segment_offsets"], dtype=np.int64),
        segment_starts=np.asarray(arrays["segment_starts"], dtype=np.int64),
        segment_ends=np.asarray(arrays["segment_ends"], dtype=np.int64),
        response_scores=np.asarray(arrays["response_scores"], dtype=np.float64),
        method_ids=tuple(map(str, arrays["method_ids"].tolist())),
        identity_contract=record["identity_contract"],
        external_certificate_sha256=str(record["external_certificate_sha256"]),
        external_score_bindings_sha256=str(record["external_score_bindings_sha256"]),
        token_transform_sha256=str(record["token_transform_sha256"]),
        artifact_sha256=str(record["artifact_sha256"]),
    )
    if len(value.row_ids) != int(record["n_rows"]):
        raise RuntimeError("localization prepared row count drifted")
    if len(value.token_confidence) != int(record["n_tokens"]):
        raise RuntimeError("localization prepared token count drifted")
    if len(value.segment_starts) != int(record["n_segments"]):
        raise RuntimeError("localization prepared segment count drifted")
    return value


def validate_fit_manifest(
    path: str | Path,
    *,
    input_root: str | Path | None = None,
    require_scientific: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != payload_sha256(payload):
        raise RuntimeError("localization fit manifest payload hash failed")
    expected_fields = {
        "schema_version", "release_id", "external_release_id", "build_id",
        "scientific_full_build", "target_values_selected", "historical_localization_scores_opened",
        "external_certificate_sha256", "external_registry_sha256",
        "method_registry_sha256", "identity_contract", "id_contract_version",
        "token_contract_id", "token_mixed_v2_applied_exactly_once",
        "n_cells", "cells", "payload_sha256",
    }
    if set(value) != expected_fields:
        raise RuntimeError("localization fit manifest contains unknown/controller fields")
    if value.get("schema_version") != FIT_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unexpected localization fit manifest schema")
    if require_scientific and value.get("scientific_full_build") is not True:
        raise RuntimeError("partial localization preparation cannot be scientific")
    if value.get("target_values_selected") is not False:
        raise RuntimeError("target values crossed the localization fit boundary")
    if value.get("historical_localization_scores_opened") is not False:
        raise RuntimeError("historical localization scores crossed the fit boundary")
    if value.get("token_contract_id") != TOKEN_CONTRACT_ID:
        raise RuntimeError("localization fit manifest binds another token contract")
    if value.get("token_mixed_v2_applied_exactly_once") is not True:
        raise RuntimeError("localization token preprocessing count is not one")
    identity = validate_fit_row_identity_contract(value.get("identity_contract", {}))
    if value.get("id_contract_version") != ID_CONTRACT_VERSION:
        raise RuntimeError("localization fit identity version drifted")
    cells = value.get("cells", ())
    if not isinstance(cells, list) or len(cells) != int(value.get("n_cells", -1)):
        raise RuntimeError("localization fit cell roster is malformed")
    root = Path(input_root) if input_root is not None else manifest_path.parent
    seen: set[str] = set()
    for record in cells:
        cell_id = str(record.get("cell_id", ""))
        if not cell_id or cell_id in seen:
            raise RuntimeError("localization fit manifest has duplicate/empty cells")
        seen.add(cell_id)
        if record.get("identity_contract") != identity:
            raise RuntimeError("localization fit cells disagree on identity contract")
        load_prepared_localization_cell(root / str(record["artifact_path"]), record)
    return value


def assert_no_target_named_members(names: Sequence[str]) -> None:
    forbidden = [
        str(name) for name in names
        if any(fragment in str(name).lower() for fragment in _TARGET_FRAGMENTS)
    ]
    if forbidden:
        raise RuntimeError(f"target/group-like members entered fit artifact: {forbidden}")


__all__ = [
    "COMBINED_ADAPTER_ID", "FIT_MANIFEST_SCHEMA_VERSION", "FIT_SAFE_CELL_FIELDS",
    "FIT_TOKEN_CAP", "NO_ERROR", "PREPARED_SCHEMA_VERSION",
    "PreparedLocalizationCell", "REGISTRY_SCHEMA_VERSION",
    "RESPONSE_ONLY_ADAPTER_ID", "SCORE_FREEZE_SCHEMA_VERSION",
    "SCORE_SCHEMA_VERSION", "TOKEN_CONTRACT_ID", "TOKEN_ONLY_ADAPTER_ID",
    "TOKEN_STREAM_COUNT",
    "assert_no_target_named_members", "empirical_midrank",
    "load_localization_registry", "load_prepared_localization_cell",
    "payload_sha256", "primary_system_roster", "validate_fit_manifest",
    "validate_fit_row_identity_contract",
]
