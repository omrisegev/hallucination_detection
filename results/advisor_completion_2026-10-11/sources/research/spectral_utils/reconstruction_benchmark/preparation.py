"""Prepare target-free mixed-v2 inputs for reconstruction benchmark v1.

This module is the only fitting-side component allowed to open the legacy
``cells.npz`` bundle.  It reads only measurement, name, and legacy-sign arrays;
it never indexes a ``__labels`` member.  The resulting per-cell archives contain
no target, anchor, historical polarity, or label-like field.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import atomic_write_json, atomic_write_npz, canonical_json_bytes, sha256_file
from ..dufs_liu_feature_contract import (
    CONTRACT_VERSION,
    dufs_liu_mixed_v2_from_bundle,
)
from ..specrage_views import FEATURE_TO_VIEW, VIEW_SCHEMA_VERSION, view_members


SCHEMA_VERSION = "reconstruction-target-free-input-v1"
FORBIDDEN_FIELD_FRAGMENTS = (
    "label",
    "target",
    "correct",
    "answer_key",
    "auc",
    "auprc",
    "rho_polarity",
    "anchor",
)


@dataclass(frozen=True)
class PreparedCellRecord:
    cell_id: str
    domain: str
    n_rows: int
    n_features: int
    feature_names: tuple[str, ...]
    present_families: tuple[str, ...]
    cohort_id: str
    feature_matrix_sha256: str
    artifact_path: str
    artifact_sha256: str
    transform_details: dict


def _assert_target_free_names(names: Iterable[str]) -> None:
    violations = []
    for name in names:
        lowered = str(name).lower()
        if any(token in lowered for token in FORBIDDEN_FIELD_FRAGMENTS):
            violations.append(str(name))
    if violations:
        raise RuntimeError("target-like fields are forbidden in fitting artifacts: " + ", ".join(violations))


def _matrix_hash(matrix: np.ndarray, feature_names: tuple[str, ...]) -> str:
    value = np.ascontiguousarray(matrix, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(value.tobytes(order="C"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(feature_names))
    return digest.hexdigest()


def _cohort_id(
    cell_id: str,
    n_rows: int,
    *,
    feature_matrix_sha256: str,
    source_bundle_sha256: str,
) -> str:
    # The frozen matrix has stable within-cell row order but not source IDs for
    # every cell.  The ID says exactly what it is; it must not be presented as a
    # raw-generation identity proof.
    payload = {
        "cell_id": cell_id,
        "row_identity": "consolidated_matrix_index",
        "row_indices": [0, int(n_rows)],
        "feature_matrix_sha256": str(feature_matrix_sha256),
        "source_bundle_sha256": str(source_bundle_sha256),
    }
    return "cohort_" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:20]


def prepare_build(
    *,
    source_bundle: str | Path,
    out_dir: str | Path,
    roster: Iterable[str],
    domains: dict[str, str],
    expected_source_sha256: str,
    feature_contract_config_sha256: str,
    transform_source_sha256: str,
    orientation_source_sha256: str,
    roster_source_sha256: str,
    build_id: str,
) -> dict:
    """Independently rebuild one target-free input tree.

    No output from another build is accepted.  Existing files are refused so a
    rebuild cannot silently mix definitions.
    """

    source = Path(source_bundle)
    output = Path(out_dir)
    roster = tuple(str(cell) for cell in roster)
    if build_id not in {"A", "B"}:
        raise ValueError("build_id must be A or B")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"target-free build directory is not empty: {output}")
    source_hash = sha256_file(source)
    if source_hash != expected_source_sha256:
        raise RuntimeError(
            f"source bundle hash mismatch: expected {expected_source_sha256}, got {source_hash}"
        )

    output.mkdir(parents=True, exist_ok=True)
    cell_dir = output / "cells"
    cell_dir.mkdir()
    records: list[dict] = []

    # allow_pickle is needed only for the legacy string arrays.  Label arrays are
    # never indexed, materialized, copied, or passed to the transformer.
    with np.load(source, allow_pickle=True) as bundle:
        source_cells = sorted({name.split("__", 1)[0] for name in bundle.files})
        if sorted(roster) != source_cells:
            raise RuntimeError(
                f"source roster mismatch: expected {len(roster)} cells, found {len(source_cells)}"
            )
        for cell_id in roster:
            required = [f"{cell_id}__V", f"{cell_id}__pool", f"{cell_id}__hand_signs"]
            missing = [name for name in required if name not in bundle.files]
            if missing:
                raise KeyError(f"{cell_id}: missing source arrays {missing}")

            stored = np.asarray(bundle[f"{cell_id}__V"], dtype=np.float64)
            names = tuple(str(item) for item in bundle[f"{cell_id}__pool"].tolist())
            legacy_signs = np.asarray(bundle[f"{cell_id}__hand_signs"], dtype=np.int8)
            matrix, transformed_names, details = dufs_liu_mixed_v2_from_bundle(
                stored, names, legacy_signs
            )
            matrix = np.asarray(matrix, dtype=np.float64)
            transformed_names = tuple(transformed_names)
            if transformed_names != names:
                raise RuntimeError(f"{cell_id}: mixed-v2 changed the column roster")
            if matrix.shape != stored.shape or not np.isfinite(matrix).all():
                raise RuntimeError(f"{cell_id}: invalid transformed matrix")
            means = matrix.mean(axis=0)
            scales = matrix.std(axis=0, ddof=0)
            if np.max(np.abs(means)) > 1e-9:
                raise RuntimeError(f"{cell_id}: transformed columns are not centered")
            valid_scale = np.logical_or(
                np.abs(scales - 1.0) <= 1e-8,
                scales <= 1e-12,
            )
            if not np.all(valid_scale):
                bad = np.flatnonzero(~valid_scale).tolist()
                raise RuntimeError(
                    f"{cell_id}: transformed columns are neither unit-scaled nor "
                    f"constant-zero; bad columns {bad!r}"
                )

            family_ids = tuple(FEATURE_TO_VIEW[name] for name in names)
            present_families = tuple(view_members(names))
            row_ids = np.asarray(
                [f"{cell_id}:matrix_row:{index:08d}" for index in range(matrix.shape[0])],
                dtype=f"<U{len(cell_id) + 32}",
            )
            arrays = {
                "X_confidence": matrix.astype("<f8", copy=False),
                "feature_names": np.asarray(names, dtype="<U64"),
                "family_ids": np.asarray(family_ids, dtype="<U32"),
                "row_ids": row_ids,
                "row_index": np.arange(matrix.shape[0], dtype="<i8"),
            }
            _assert_target_free_names(arrays)
            artifact = cell_dir / f"{cell_id}.npz"
            artifact_hash = atomic_write_npz(artifact, arrays)
            matrix_hash = _matrix_hash(matrix, names)
            record = PreparedCellRecord(
                cell_id=cell_id,
                domain=str(domains[cell_id]),
                n_rows=int(matrix.shape[0]),
                n_features=int(matrix.shape[1]),
                feature_names=names,
                present_families=present_families,
                cohort_id=_cohort_id(
                    cell_id,
                    matrix.shape[0],
                    feature_matrix_sha256=matrix_hash,
                    source_bundle_sha256=source_hash,
                ),
                feature_matrix_sha256=matrix_hash,
                artifact_path=artifact.relative_to(output).as_posix(),
                artifact_sha256=artifact_hash,
                transform_details=details,
            )
            records.append(
                {
                    **record.__dict__,
                    "feature_names": list(record.feature_names),
                    "present_families": list(record.present_families),
                }
            )

    total_rows = sum(int(item["n_rows"]) for item in records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "scientific_run": True,
        "feature_contract_id": CONTRACT_VERSION,
        "view_schema_version": VIEW_SCHEMA_VERSION,
        "source_bundle": str(source),
        "source_bundle_sha256": source_hash,
        "feature_contract_config_sha256": str(feature_contract_config_sha256),
        "transform_source_sha256": str(transform_source_sha256),
        "orientation_source_sha256": str(orientation_source_sha256),
        "roster_source_sha256": str(roster_source_sha256),
        "label_arrays_accessed": False,
        "score_semantics": "not_applicable_pre_fit",
        "matrix_semantics": "higher_is_confidence",
        "row_identity_level": "consolidated_matrix_index; not raw-generation identity",
        "n_cells": len(records),
        "n_rows": total_rows,
        "cells": records,
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    # A file cannot contain its own cryptographic hash without a special
    # envelope.  ``manifest_payload_sha256`` is the self-verifying content
    # digest; callers may hash the completed file separately when they bind it
    # into a later score-freeze record.
    atomic_write_json(output / "MANIFEST.json", manifest)
    return manifest


def compare_prepared_builds(left_dir: str | Path, right_dir: str | Path) -> dict:
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    left = json.loads((left_dir / "MANIFEST.json").read_text())
    right = json.loads((right_dir / "MANIFEST.json").read_text())
    left_cells = {item["cell_id"]: item for item in left["cells"]}
    right_cells = {item["cell_id"]: item for item in right["cells"]}
    if set(left_cells) != set(right_cells):
        raise RuntimeError("A/B prepared rosters differ")
    comparisons = []
    for cell_id in sorted(left_cells):
        a = left_cells[cell_id]
        b = right_cells[cell_id]
        byte_equal = sha256_file(left_dir / a["artifact_path"]) == sha256_file(
            right_dir / b["artifact_path"]
        )
        semantic_equal = (
            a["feature_matrix_sha256"] == b["feature_matrix_sha256"]
            and a["cohort_id"] == b["cohort_id"]
            and a["feature_names"] == b["feature_names"]
        )
        comparisons.append(
            {"cell_id": cell_id, "byte_equal": byte_equal, "semantic_equal": semantic_equal}
        )
    result = {
        "schema_version": "reconstruction-prepared-ab-comparison-v1",
        "left_build": left["build_id"],
        "right_build": right["build_id"],
        "n_cells": len(comparisons),
        "all_byte_equal": all(item["byte_equal"] for item in comparisons),
        "all_semantic_equal": all(item["semantic_equal"] for item in comparisons),
        "cells": comparisons,
    }
    if not result["all_byte_equal"] or not result["all_semantic_equal"]:
        raise RuntimeError("independent prepared builds are not identical")
    return result


__all__ = [
    "FORBIDDEN_FIELD_FRAGMENTS",
    "PreparedCellRecord",
    "compare_prepared_builds",
    "prepare_build",
]
