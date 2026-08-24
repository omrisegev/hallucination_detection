"""Audited source-group sidecars for the reconstructed 24-cell benchmark.

The consolidated ``cells.npz`` archive preserves row order but not source
question IDs.  This module reconstructs only the equivalence relation needed
by the grouped bootstrap, without indexing or deriving from correctness labels:

* a frozen ``k=1`` source manifest proves that every admitted row is its own
  source item, so stable singleton pseudonyms are sufficient and row order is
  immaterial to the grouping relation;
* repeated-generation cells are rebuilt from their hash-frozen raw pickle,
  admitted with the existing target-free A0 adapter, restored to the exact
  historical feature-cache order, and aligned positionally only after at least
  eight complete feature columns reproduce the consolidated matrix.

There is deliberately no repeated-cell fallback to IID row groups.  A failed
source hash, admission count, cache-order reconstruction, or feature proof is
reported as ``GROUP_IDS_UNAVAILABLE`` and blocks the 24-cell headline.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from ..a5_target_free_data import CORE_FEATURES, FROZEN_A0_SOURCE_SPECS
from ..dufs_liu_feature_contract import dufs_liu_mixed_v2_from_bundle
from ..fair_comparisons.twentyfour import admit_source_rows, verify_source_artifact
from .contracts import prepared_matrix_sha256
from .fit_validation import payload_sha256, validate_prepared_manifest
from .io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)


GROUP_SIDECAR_SCHEMA_VERSION = "reconstruction-group-sidecars-v1"
GROUP_EVIDENCE_SCHEMA_VERSION = "reconstruction-group-identity-evidence-v1"
SOURCE_REGISTRY_SCHEMA_VERSION = "residual_graph_deem_24cell_v1_registry"
SOURCE_REGISTRY_CONTENT_FIELD = "registry_content_sha256"
SOURCE_REGISTRY_CONTENT_SHA256 = (
    "84fa79e396672d9d7f930d385202c0270facb0a32da6b7e78b42926133d5b776"
)
BUILDER_ID = "reconstruction-group-sidecar-builder-v1.0.0"
MIN_EXACT_IDENTITY_FEATURES = 8
IDENTITY_ATOL = 1e-10

REPEATED_SOURCE_STATUS = "raw_source_ids_required_for_row_level_interval"
SINGLETON_SOURCE_STATUS = "one_row_per_answer_assumed_pending_identity_audit"

_ALLOWED_SOURCE_BUNDLE_MEMBERS = frozenset({"V", "pool", "hand_signs"})


class GroupSidecarError(RuntimeError):
    """A source-group identity or provenance gate failed."""


def _with_payload_hash(value: Mapping[str, Any], field: str = "payload_sha256") -> dict:
    output = dict(value)
    output[field] = payload_sha256(output, field)
    return output


def ordered_text_sha256(values: Sequence[str], *, field: str) -> str:
    return sha256_bytes(canonical_json_bytes({field: [str(value) for value in values]}))


def row_group_binding_sha256(
    row_ids: Sequence[str], group_ids: Sequence[str]
) -> str:
    return sha256_bytes(canonical_json_bytes({
        "row_ids": [str(value) for value in row_ids],
        "group_ids": [str(value) for value in group_ids],
    }))


def load_target_free_bundle_cell(
    source_bundle: Any,
    cell_id: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Materialize exactly three whitelisted arrays from the legacy bundle."""

    required = {
        member: f"{cell_id}__{member}"
        for member in _ALLOWED_SOURCE_BUNDLE_MEMBERS
    }
    missing = [key for key in required.values() if key not in source_bundle.files]
    if missing:
        raise GroupSidecarError(
            f"{cell_id}: source bundle lacks identity members {missing!r}"
        )
    # Keep these accesses visibly literal.  Downstream identity code receives only
    # these returned arrays, and cannot construct a label key from ``member``.
    matrix = np.asarray(source_bundle[f"{cell_id}__V"], dtype=float)
    pool = tuple(str(value) for value in source_bundle[f"{cell_id}__pool"].tolist())
    hand_signs = np.asarray(source_bundle[f"{cell_id}__hand_signs"], dtype=float)
    return matrix, pool, hand_signs


def load_source_registry(path: str | Path) -> dict[str, Any]:
    """Load the frozen 24-cell raw-source registry and verify its content hash."""

    registry_path = Path(path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise GroupSidecarError("source registry must be a JSON object")
    if registry.get("schema") != SOURCE_REGISTRY_SCHEMA_VERSION:
        raise GroupSidecarError("unexpected source-registry schema")
    declared = registry.get(SOURCE_REGISTRY_CONTENT_FIELD)
    body = dict(registry)
    body.pop(SOURCE_REGISTRY_CONTENT_FIELD, None)
    observed = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if declared != observed:
        raise GroupSidecarError(
            f"source-registry content hash mismatch: {declared!r} != {observed}"
        )
    if observed != SOURCE_REGISTRY_CONTENT_SHA256:
        raise GroupSidecarError(
            "source registry is internally valid but is not the frozen v1 content"
        )
    cells = registry.get("cells")
    if not isinstance(cells, list) or len(cells) != 24:
        raise GroupSidecarError("source registry must contain exactly 24 cells")
    ids = [str(row.get("cell_id", "")) for row in cells if isinstance(row, dict)]
    if len(ids) != 24 or len(set(ids)) != 24:
        raise GroupSidecarError("source registry cell IDs are not 24 unique values")
    for row in cells:
        source = row.get("source")
        if not isinstance(source, dict) or source.get("environment_id") != row.get("cell_id"):
            raise GroupSidecarError("source registry cell/source identity mismatch")
    return registry


def _manifest_k_values(manifest: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
    try:
        header_k = int(manifest["k"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GroupSidecarError("source manifest has no integer k") from exc
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise GroupSidecarError("source manifest has no cell records")
    try:
        cell_k = tuple(int(row["k"]) for row in cells)
    except (KeyError, TypeError, ValueError) as exc:
        raise GroupSidecarError("source manifest cell record has no integer k") from exc
    return header_k, cell_k


def singleton_group_ids(
    *,
    cell_id: str,
    row_ids: Sequence[str],
    manifest_path: str | Path,
    source_spec: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Prove and construct singleton source groups from a frozen ``k=1`` manifest.

    The generated IDs are pseudonyms for equivalence classes, not claims that the
    original question identifiers were recovered.  With one candidate per source
    item, the partition itself is uniquely determined: every row is a singleton.
    """

    path = Path(manifest_path)
    observed_manifest_sha = sha256_file(path)
    if observed_manifest_sha != source_spec.get("manifest_sha256"):
        raise GroupSidecarError(f"{cell_id}: frozen source-manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise GroupSidecarError(f"{cell_id}: source manifest is not an object")
    if manifest.get("dataset") != source_spec.get("dataset"):
        raise GroupSidecarError(f"{cell_id}: source-manifest dataset mismatch")
    if manifest.get("split") != source_spec.get("split"):
        raise GroupSidecarError(f"{cell_id}: source-manifest split mismatch")
    header_k, cell_k = _manifest_k_values(manifest)
    if header_k != 1 or any(value != 1 for value in cell_k):
        raise GroupSidecarError(
            f"{cell_id}: singleton proof requires k=1 in header and every cell"
        )
    expected = int(source_spec.get("expected_admitted_count", -1))
    if len(row_ids) != expected:
        raise GroupSidecarError(
            f"{cell_id}: prepared/admitted count mismatch {len(row_ids)} != {expected}"
        )
    groups = tuple(
        f"{cell_id}::manifest-k1-item::{index:08d}"
        for index in range(len(row_ids))
    )
    if groups == tuple(str(value) for value in row_ids):
        raise GroupSidecarError(f"{cell_id}: singleton group IDs reused prepared row IDs")
    if len(set(groups)) != len(groups):
        raise GroupSidecarError(f"{cell_id}: singleton pseudonyms are not unique")
    detail = {
        "proof_type": "frozen_manifest_k1_singleton_partition",
        "source_manifest_path": str(path.resolve()),
        "source_manifest_sha256": observed_manifest_sha,
        "manifest_header_k": header_k,
        "manifest_cell_k": list(cell_k),
        "expected_admitted_count": expected,
        "prepared_row_count": len(row_ids),
        "raw_source_read": False,
        "source_identity_recovered": False,
        "equivalence_relation": "one unique source item per admitted row",
        "row_mapping_requirement": "order_invariant_for_singleton_partition",
    }
    return groups, detail


def historical_featcache_order(
    source: Mapping[Any, Any], identities: Sequence[Any]
) -> tuple[Any, ...]:
    """Restore ``build_repgrid_featcache.py``'s exact row iteration order.

    The fair-comparison adapter intentionally sorts source keys by their string
    representation to make canonical IDs portable.  The historical feature cache
    instead used Python's native ``sorted(data.keys())``.  The consolidated matrix
    follows the latter, so identity rows must be restored to that order before a
    positional feature proof is meaningful.
    """

    lookup: dict[tuple[str, int], Any] = {}
    for identity in identities:
        key = (str(identity.item_group_id), int(identity.candidate_ordinal))
        if key in lookup:
            raise GroupSidecarError(f"duplicate admitted source identity: {key!r}")
        lookup[key] = identity
    ordered: list[Any] = []
    try:
        source_keys = sorted(source.keys())
    except TypeError as exc:
        raise GroupSidecarError(
            "raw source keys cannot reproduce historical native-sort order"
        ) from exc
    for problem_key in source_keys:
        raw_row = source[problem_key]
        if not isinstance(raw_row, Mapping):
            raise GroupSidecarError(f"raw source row is not a mapping: {problem_key!r}")
        candidates = raw_row.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise GroupSidecarError(f"raw candidates are not a sequence: {problem_key!r}")
        for ordinal in range(len(candidates)):
            identity = lookup.get((str(problem_key), ordinal))
            if identity is not None:
                ordered.append(identity)
    if len(ordered) != len(identities) or {
        (str(row.item_group_id), int(row.candidate_ordinal)) for row in ordered
    } != set(lookup):
        raise GroupSidecarError("historical source-order reconstruction is incomplete")
    return tuple(ordered)


def _zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise GroupSidecarError("identity feature must be one finite vector")
    scale = float(array.std(ddof=0))
    return array - array.mean() if scale <= 1e-8 else (array - array.mean()) / scale


def audit_feature_group_collisions(
    feature_matrix: np.ndarray,
    group_ids: Sequence[str],
    *,
    rounded_decimals: Sequence[int] = (12, 10),
) -> dict[str, Any]:
    """Reject feature-indistinguishable rows assigned to different source groups."""

    matrix = np.asarray(feature_matrix, dtype=np.float64)
    groups = tuple(str(value) for value in group_ids)
    if matrix.ndim != 2 or matrix.shape[0] != len(groups):
        raise GroupSidecarError("collision-audit matrix/group shape mismatch")
    regimes: dict[str, dict[str, int]] = {}
    for decimals in (None, *tuple(int(value) for value in rounded_decimals)):
        values = matrix if decimals is None else np.round(matrix, decimals=decimals)
        classes: dict[bytes, list[int]] = {}
        for index, row in enumerate(values):
            signature = np.ascontiguousarray(row, dtype="<f8").tobytes(order="C")
            classes.setdefault(signature, []).append(index)
        duplicated = [indices for indices in classes.values() if len(indices) > 1]
        cross_group = [
            indices
            for indices in duplicated
            if len({groups[index] for index in indices}) > 1
        ]
        key = "exact_float64" if decimals is None else f"rounded_{decimals}_decimals"
        regimes[key] = {
            "duplicated_classes": len(duplicated),
            "rows_in_duplicated_classes": sum(map(len, duplicated)),
            "cross_group_classes": len(cross_group),
            "rows_in_cross_group_classes": sum(map(len, cross_group)),
        }
    failures = {
        key: value for key, value in regimes.items() if value["cross_group_classes"]
    }
    if failures:
        raise GroupSidecarError(
            "feature fingerprint collision spans different source groups: "
            f"{failures!r}"
        )
    return {
        "proof_type": "feature_collision_classes_do_not_cross_source_groups",
        "n_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "regimes": regimes,
        "within_group_duplicates_allowed": True,
        "cross_group_collisions": 0,
    }


def prove_positional_feature_alignment(
    *,
    ordered_identities: Sequence[Any],
    bundle_matrix: np.ndarray,
    pool: Sequence[str],
    hand_signs: np.ndarray,
    min_exact_features: int = MIN_EXACT_IDENTITY_FEATURES,
    atol: float = IDENTITY_ATOL,
) -> dict[str, Any]:
    """Prove direct raw-to-consolidated row order with full-column fingerprints."""

    rows = tuple(ordered_identities)
    matrix = np.asarray(bundle_matrix, dtype=float)
    names = tuple(str(value) for value in pool)
    signs = np.asarray(hand_signs, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(rows), len(names)):
        raise GroupSidecarError(
            f"bundle/admitted shape mismatch: {matrix.shape} != {(len(rows), len(names))}"
        )
    if signs.shape != (len(names),) or not np.isin(signs, (-1.0, 1.0)).all():
        raise GroupSidecarError("invalid frozen hand-sign vector")
    if not np.isfinite(matrix).all():
        raise GroupSidecarError("consolidated identity matrix contains non-finite values")
    core = np.vstack([np.asarray(row.core_features, dtype=float) for row in rows])
    if core.shape != (len(rows), len(CORE_FEATURES)) or not np.isfinite(core).all():
        raise GroupSidecarError("reconstructed A0 core feature matrix is invalid")
    common = tuple(name for name in CORE_FEATURES if name in names)
    if len(common) < int(min_exact_features):
        raise GroupSidecarError(
            f"only {len(common)} common identity features; need {min_exact_features}"
        )
    reconstructed_by_feature = {
        name: _zscore(
            core[:, CORE_FEATURES.index(name)] * signs[names.index(name)]
        )
        for name in common
    }
    # Ambiguity must be tested in exactly the reconstructed coordinates visible
    # in the frozen bundle.  A difference in an absent raw feature cannot identify
    # a bundle row.
    collision_audit = audit_feature_group_collisions(
        np.column_stack([reconstructed_by_feature[name] for name in common]),
        tuple(str(row.group_id) for row in rows),
    )
    max_abs_by_feature: dict[str, float] = {}
    exact: list[str] = []
    for name in common:
        reconstructed = reconstructed_by_feature[name]
        observed = matrix[:, names.index(name)]
        error = float(np.max(np.abs(reconstructed - observed)))
        max_abs_by_feature[name] = error
        if np.allclose(reconstructed, observed, rtol=0.0, atol=float(atol)):
            exact.append(name)
    if len(exact) < int(min_exact_features):
        raise GroupSidecarError(
            "direct positional feature proof failed: "
            f"{len(exact)} exact columns < {min_exact_features}; "
            f"best max error={min(max_abs_by_feature.values()):.3e}"
        )
    # Do not permit the gate to pass on a convenient subset while silently
    # ignoring drift in another shared identity coordinate.  The minimum count
    # protects sparse rosters; all available common coordinates must agree.
    if len(exact) != len(common):
        failed = {name: max_abs_by_feature[name] for name in common if name not in exact}
        raise GroupSidecarError(
            f"direct positional feature proof has drifting common columns: {failed!r}"
        )
    feature_payload = {
        name: hashlib.sha256(
            np.ascontiguousarray(matrix[:, names.index(name)], dtype="<f8").tobytes()
        ).hexdigest()
        for name in exact
    }
    return {
        "proof_type": "direct_historical_order_full_column_feature_fingerprint",
        "source_order_contract": (
            "scripts/build_repgrid_featcache.py::build_cell/"
            "for idx in sorted(data.keys()), candidate order, complete-case admission"
        ),
        "n_rows": len(rows),
        "common_features": list(common),
        "exact_features": list(exact),
        "n_exact_features": len(exact),
        "minimum_exact_features": int(min_exact_features),
        "absolute_tolerance": float(atol),
        "max_abs_error": max(max_abs_by_feature.values()),
        "max_abs_by_feature": max_abs_by_feature,
        "observed_column_sha256": feature_payload,
        "collision_audit": collision_audit,
    }


def prove_prepared_row_alignment(
    *,
    source_matrix: np.ndarray,
    source_names: Sequence[str],
    source_hand_signs: np.ndarray,
    prepared_matrix: np.ndarray,
    prepared_names: Sequence[str],
) -> dict[str, Any]:
    """Prove that prepared rows are the direct mixed-v2 transform of bundle rows."""

    reconstructed, reconstructed_names, details = dufs_liu_mixed_v2_from_bundle(
        source_matrix,
        source_names,
        source_hand_signs,
    )
    expected = np.asarray(reconstructed, dtype=np.float64)
    observed = np.asarray(prepared_matrix, dtype=np.float64)
    expected_names = tuple(str(value) for value in reconstructed_names)
    observed_names = tuple(str(value) for value in prepared_names)
    if expected_names != observed_names:
        raise GroupSidecarError("prepared/source feature-name order mismatch")
    if expected.shape != observed.shape or not np.isfinite(observed).all():
        raise GroupSidecarError(
            f"prepared/source transformed shape mismatch: {observed.shape} != "
            f"{expected.shape}"
        )
    if not np.array_equal(expected, observed):
        max_error = float(np.max(np.abs(expected - observed)))
        raise GroupSidecarError(
            "prepared rows do not exactly reproduce the frozen mixed-v2 transform; "
            f"max error={max_error:.3e}"
        )
    return {
        "proof_type": "exact_bundle_to_prepared_mixed_v2_rebuild",
        "n_rows": int(observed.shape[0]),
        "n_features": int(observed.shape[1]),
        "feature_names": list(observed_names),
        "exact_array_equality": True,
        "maximum_absolute_error": 0.0,
        "transform_details": details,
        "prepared_value_sha256": hashlib.sha256(
            np.ascontiguousarray(observed, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
    }


def _frozen_a0_spec(cell_id: str, source_spec: Mapping[str, Any]):
    matches = [spec for spec in FROZEN_A0_SOURCE_SPECS if spec.environment_id == cell_id]
    if len(matches) != 1:
        raise GroupSidecarError(f"{cell_id}: repeated source lacks one frozen A0 spec")
    spec = matches[0]
    expected = {
        "environment_id": spec.environment_id,
        "dataset": spec.dataset,
        "split": spec.split,
        "dataset_family": spec.dataset_family,
        "expected_admitted_count": spec.expected_admitted_count,
        "admission_mode": spec.admission_mode,
        "raw_relative_path": spec.raw_relative_path,
        "source_sha256": spec.source_sha256,
        "source_size": spec.source_size,
        "manifest_sha256": spec.manifest_sha256,
    }
    for key, value in expected.items():
        if source_spec.get(key) != value:
            raise GroupSidecarError(f"{cell_id}: residual/A0 source spec drift at {key}")
    return spec


def repeated_group_ids(
    *,
    repo_root: Path,
    raw_root: Path,
    cell_id: str,
    row_ids: Sequence[str],
    source_spec: Mapping[str, Any],
    bundle_matrix: np.ndarray,
    bundle_pool: Sequence[str],
    bundle_hand_signs: np.ndarray,
    prepared_matrix: np.ndarray,
    prepared_names: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, Any], Path, str]:
    """Reconstruct one repeated cell and prove its row-to-group mapping."""

    frozen_spec = _frozen_a0_spec(cell_id, source_spec)
    source_audit = verify_source_artifact(
        repo_root,
        cell_id,
        source_root=raw_root,
        verify_sha256=True,
    )
    manifest_path = Path(str(source_audit["manifest_path"])).resolve()
    manifest_sha = str(source_audit["manifest_sha256"])
    if manifest_sha != source_spec.get("manifest_sha256"):
        raise GroupSidecarError(f"{cell_id}: source verifier/registry manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    header_k, cell_k = _manifest_k_values(manifest)
    if header_k <= 1 or any(value <= 1 for value in cell_k):
        raise GroupSidecarError(f"{cell_id}: repeated-source manifest does not declare k>1")
    raw_path = Path(str(source_audit["raw_path"])).resolve()
    observed_size = int(source_audit["source_size"])
    if observed_size != int(source_spec.get("source_size", -1)):
        raise GroupSidecarError(f"{cell_id}: source verifier/registry size drift")
    observed_sha = str(source_audit["source_sha256"])
    if observed_sha != source_spec.get("source_sha256"):
        raise GroupSidecarError(f"{cell_id}: source verifier/registry SHA-256 drift")
    with raw_path.open("rb") as handle:
        source = pickle.load(handle)
    identities = admit_source_rows(source, frozen_spec)
    ordered = historical_featcache_order(source, identities)
    if len(ordered) != len(row_ids):
        raise GroupSidecarError(
            f"{cell_id}: admitted/prepared count mismatch {len(ordered)} != {len(row_ids)}"
        )

    matrix = np.asarray(bundle_matrix, dtype=float)
    pool = tuple(str(value) for value in bundle_pool)
    hand_signs = np.asarray(bundle_hand_signs, dtype=float)
    feature_proof = prove_positional_feature_alignment(
        ordered_identities=ordered,
        bundle_matrix=matrix,
        pool=pool,
        hand_signs=hand_signs,
    )
    prepared_proof = prove_prepared_row_alignment(
        source_matrix=matrix,
        source_names=pool,
        source_hand_signs=hand_signs,
        prepared_matrix=prepared_matrix,
        prepared_names=prepared_names,
    )
    groups = tuple(str(identity.group_id) for identity in ordered)
    counts = Counter(groups)
    if len(groups) != len(row_ids) or len(set(groups)) < 2:
        raise GroupSidecarError(f"{cell_id}: invalid repeated group vector")
    if max(counts.values()) <= 1:
        raise GroupSidecarError(
            f"{cell_id}: repeated cell unexpectedly collapsed to IID singleton groups"
        )
    if max(counts.values()) > max((header_k, *cell_k)):
        raise GroupSidecarError(f"{cell_id}: source group exceeds manifest candidate count")
    detail = {
        "proof_type": "hash_frozen_raw_rebuild_plus_positional_feature_fingerprint",
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": manifest_sha,
        "raw_source_path": str(raw_path.resolve()),
        "raw_source_sha256": observed_sha,
        "raw_source_size": observed_size,
        "manifest_header_k": header_k,
        "manifest_cell_k": list(cell_k),
        "admission_mode": frozen_spec.admission_mode,
        "expected_admitted_count": frozen_spec.expected_admitted_count,
        "admitted_count": len(ordered),
        "n_groups": len(counts),
        "minimum_group_size": min(counts.values()),
        "maximum_group_size": max(counts.values()),
        "labels_indexed": False,
        "target_like_fields_indexed": False,
        "raw_pickle_materialized_after_hash_verification": True,
        "raw_top_level_fields_indexed_by_adapter": [
            "question",
            "candidates",
        ],
        "raw_candidate_fields_indexed_by_adapter": [
            "token_entropies",
            "token_spilled_energies",
            "token_logsumexp",
            "top_k_logprobs",
            "token_offsets (cropped_all_rows only)",
            "full_text (cropped_all_rows only; generated text, not a target)",
        ],
        "opaque_source_candidate_retained_but_not_indexed_for_targets": True,
        "source_bundle_members_accessed": ["V", "pool", "hand_signs"],
        "feature_alignment": feature_proof,
        "prepared_alignment": prepared_proof,
    }
    return groups, detail, raw_path.resolve(), observed_sha


def _source_registry_by_cell(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["cell_id"]): dict(row) for row in registry["cells"]}


def _write_verified_cell(
    *,
    out_root: Path,
    cell_id: str,
    row_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    group_unit: str,
    input_record: Mapping[str, Any],
    registry_record: Mapping[str, Any],
    source_artifact_path: Path,
    source_artifact_sha256: str,
    proof: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    width_rows = max(1, max(map(len, row_ids), default=1))
    width_groups = max(1, max(map(len, group_ids), default=1))
    artifact = out_root / "sidecars" / f"{cell_id}.npz"
    artifact_sha = atomic_write_npz(artifact, {
        "row_ids": np.asarray(row_ids, dtype=f"<U{width_rows}"),
        "group_ids": np.asarray(group_ids, dtype=f"<U{width_groups}"),
    })
    binding = row_group_binding_sha256(row_ids, group_ids)
    evidence = _with_payload_hash({
        "schema_version": GROUP_EVIDENCE_SCHEMA_VERSION,
        "cell_id": cell_id,
        "verification_status": "VERIFIED",
        "labels_used": False,
        "source_artifact_sha256": source_artifact_sha256,
        "group_artifact_sha256": artifact_sha,
        "row_group_binding_sha256": binding,
        "verifier_id": BUILDER_ID,
        "verification_method": str(proof["proof_type"]),
        "group_unit": group_unit,
        "checks": {
            "source_hash_verified": True,
            "row_count_verified": True,
            "row_order_verified": True,
            "group_semantics_verified": True,
        },
        "data_firewall": {
            "group_builder_role": (
                "post-fit provenance sidecar construction; not method fitting"
            ),
            "raw_cache_objects_may_contain_labels": True,
            "legacy_bundle_contains_label_members": True,
            "labels_or_correctness_indexed": False,
            "labels_or_correctness_derived": False,
        },
        "proof": dict(proof),
        "provenance": dict(provenance),
    })
    evidence_path = out_root / "evidence" / f"{cell_id}.json"
    evidence_sha = atomic_write_json(evidence_path, evidence)
    return {
        "cell_id": cell_id,
        "verification_status": "VERIFIED",
        "labels_used": False,
        "group_unit": group_unit,
        "cohort_id": input_record["cohort_id"],
        "prepared_matrix_sha256": provenance["prepared_matrix_sha256"],
        "registry_source_group_status": registry_record["source_group_status"],
        "artifact_path": artifact.relative_to(out_root).as_posix(),
        "artifact_sha256": artifact_sha,
        "row_group_binding_sha256": binding,
        "row_ids_sha256": ordered_text_sha256(row_ids, field="row_ids"),
        "group_ids_sha256": ordered_text_sha256(group_ids, field="group_ids"),
        "n_rows": len(row_ids),
        "n_groups": len(set(group_ids)),
        "source_artifact_path": str(source_artifact_path),
        "source_artifact_sha256": source_artifact_sha256,
        "identity_evidence_path": evidence_path.relative_to(out_root).as_posix(),
        "identity_evidence_sha256": evidence_sha,
    }


def build_group_sidecars(
    *,
    repo_root: str | Path,
    release_root: str | Path,
    out_root: str | Path,
    raw_root: str | Path,
    label_bundle: str | Path,
    feature_config_path: str | Path,
    cell_registry_path: str | Path,
    source_registry_path: str | Path,
    build_id: str = "A",
) -> dict[str, Any]:
    """Build all 24 sidecars, preserving explicit failures in the manifest."""

    repo = Path(repo_root).resolve()
    release = Path(release_root).resolve()
    output = Path(out_root).resolve()
    raw_source_root = Path(raw_root).resolve()
    label_path = Path(label_bundle).resolve()
    feature_path = Path(feature_config_path).resolve()
    cell_path = Path(cell_registry_path).resolve()
    source_path = Path(source_registry_path).resolve()
    if build_id not in {"A", "B"}:
        raise GroupSidecarError("build_id must be A or B")
    if output.exists():
        raise FileExistsError(f"group-sidecar output already exists: {output}")

    feature_config = json.loads(feature_path.read_text(encoding="utf-8"))
    cell_registry = json.loads(cell_path.read_text(encoding="utf-8"))
    source_registry = load_source_registry(source_path)
    input_root = release / f"build_{build_id}" / "inputs"
    input_manifest = validate_prepared_manifest(
        input_root=input_root,
        build_id=build_id,
        repo=repo,
        feature_config=feature_config,
        cell_registry=cell_registry,
    )
    label_sha = sha256_file(label_path)
    if label_sha != input_manifest.get("source_bundle_sha256"):
        raise GroupSidecarError("label/source bundle hash does not match prepared build")
    if label_sha != feature_config.get("input_sha256"):
        raise GroupSidecarError("label/source bundle hash does not match frozen feature config")

    input_by_cell = {str(row["cell_id"]): row for row in input_manifest["cells"]}
    frozen_by_cell = {str(row["cell_id"]): row for row in cell_registry["cells"]}
    source_by_cell = _source_registry_by_cell(source_registry)
    expected_ids = [str(row["cell_id"]) for row in cell_registry["cells"]]
    if set(source_by_cell) != set(expected_ids):
        raise GroupSidecarError("source registry and frozen-24 roster disagree")

    code_paths = (
        repo / "spectral_utils" / "reconstruction_benchmark" / "group_sidecars.py",
        repo / "spectral_utils" / "a5_target_free_data.py",
        repo / "spectral_utils" / "answer_span.py",
        repo / "spectral_utils" / "dufs_liu_feature_contract.py",
        repo / "spectral_utils" / "fair_comparisons" / "twentyfour.py",
        repo / "spectral_utils" / "feature_contract.py",
        repo / "spectral_utils" / "feature_utils.py",
        repo / "scripts" / "build_repgrid_featcache.py",
    )
    code_hashes = {path.relative_to(repo).as_posix(): sha256_file(path) for path in code_paths}
    common_provenance = {
        "builder_id": BUILDER_ID,
        "build_id": build_id,
        "input_manifest_sha256": sha256_file(input_root / "MANIFEST.json"),
        "label_bundle_sha256": label_sha,
        "cell_registry_sha256": sha256_file(cell_path),
        "feature_config_sha256": sha256_file(feature_path),
        "source_registry_file_sha256": sha256_file(source_path),
        "source_registry_content_sha256": source_registry[SOURCE_REGISTRY_CONTENT_FIELD],
        "code_sha256": code_hashes,
        "labels_used": False,
        "group_builder_role": "post-fit provenance sidecar construction; not fitting",
        "raw_cache_objects_may_contain_labels": True,
        "legacy_bundle_contains_label_members": True,
        "labels_or_correctness_indexed": False,
        "labels_or_correctness_derived": False,
    }

    rows: list[dict[str, Any]] = []
    with np.load(label_path, allow_pickle=True) as source_bundle:
        # Do not leave a directory that resembles a partial audit when a global
        # provenance/bundle gate fails.  Cell-level failures below are
        # intentionally recorded in a complete 24-row manifest.
        output.mkdir(parents=True)
        for cell_id in expected_ids:
            input_record = input_by_cell[cell_id]
            registry_record = frozen_by_cell[cell_id]
            source_record = source_by_cell[cell_id]
            source_spec = source_record["source"]
            try:
                artifact_path = input_root / str(input_record["artifact_path"])
                arrays = load_npz_no_pickle(artifact_path)
                row_ids = tuple(str(value) for value in arrays["row_ids"].tolist())
                names = tuple(str(value) for value in arrays["feature_names"].tolist())
                matrix = np.asarray(arrays["X_confidence"], dtype=float)
                strong_matrix_hash = prepared_matrix_sha256(matrix, names, row_ids)
                provenance = {
                    **common_provenance,
                    "prepared_artifact_sha256": sha256_file(artifact_path),
                    "prepared_matrix_sha256": strong_matrix_hash,
                }
                status = str(registry_record.get("source_group_status"))
                if status == SINGLETON_SOURCE_STATUS:
                    manifest_path = repo / "dataset_cache" / "repgrid" / cell_id / "manifest.json"
                    group_ids, proof = singleton_group_ids(
                        cell_id=cell_id,
                        row_ids=row_ids,
                        manifest_path=manifest_path,
                        source_spec=source_spec,
                    )
                    source_artifact = manifest_path.resolve()
                    source_sha = sha256_file(source_artifact)
                    group_unit = "source_item_id"
                elif status == REPEATED_SOURCE_STATUS:
                    bundle_matrix, bundle_pool, bundle_hand_signs = (
                        load_target_free_bundle_cell(source_bundle, cell_id)
                    )
                    group_ids, proof, source_artifact, source_sha = repeated_group_ids(
                        repo_root=repo,
                        raw_root=raw_source_root,
                        cell_id=cell_id,
                        row_ids=row_ids,
                        source_spec=source_spec,
                        bundle_matrix=bundle_matrix,
                        bundle_pool=bundle_pool,
                        bundle_hand_signs=bundle_hand_signs,
                        prepared_matrix=matrix,
                        prepared_names=names,
                    )
                    group_unit = "problem_id"
                else:
                    raise GroupSidecarError(
                        f"{cell_id}: unknown frozen source_group_status {status!r}"
                    )
                rows.append(_write_verified_cell(
                    out_root=output,
                    cell_id=cell_id,
                    row_ids=row_ids,
                    group_ids=group_ids,
                    group_unit=group_unit,
                    input_record=input_record,
                    registry_record=registry_record,
                    source_artifact_path=source_artifact,
                    source_artifact_sha256=source_sha,
                    proof=proof,
                    provenance=provenance,
                ))
            except Exception as exc:  # preserve every cell-level failure in one audit
                rows.append({
                    "cell_id": cell_id,
                    "verification_status": "GROUP_IDS_UNAVAILABLE",
                    "labels_used": False,
                    "registry_source_group_status": registry_record.get("source_group_status"),
                    "failure_code": type(exc).__name__,
                    "failure_detail": str(exc),
                })

    n_verified = sum(row["verification_status"] == "VERIFIED" for row in rows)
    manifest = _with_payload_hash({
        "schema_version": GROUP_SIDECAR_SCHEMA_VERSION,
        "population_id": cell_registry["population_id"],
        "label_bundle_sha256": label_sha,
        "builder_id": BUILDER_ID,
        "labels_used": False,
        "all_verified": n_verified == 24,
        "n_verified": n_verified,
        "n_failed": 24 - n_verified,
        "cell_registry_sha256": sha256_file(cell_path),
        "source_registry_file_sha256": sha256_file(source_path),
        "source_registry_content_sha256": source_registry[SOURCE_REGISTRY_CONTENT_FIELD],
        "input_manifest_sha256": sha256_file(input_root / "MANIFEST.json"),
        "cells": rows,
    })
    atomic_write_json(output / "GROUP_SIDECARS.json", manifest)
    return manifest


__all__ = [
    "BUILDER_ID",
    "GROUP_EVIDENCE_SCHEMA_VERSION",
    "GROUP_SIDECAR_SCHEMA_VERSION",
    "GroupSidecarError",
    "audit_feature_group_collisions",
    "build_group_sidecars",
    "historical_featcache_order",
    "load_target_free_bundle_cell",
    "load_source_registry",
    "ordered_text_sha256",
    "prove_prepared_row_alignment",
    "prove_positional_feature_alignment",
    "repeated_group_ids",
    "row_group_binding_sha256",
    "singleton_group_ids",
]
