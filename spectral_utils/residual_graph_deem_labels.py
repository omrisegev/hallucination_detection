"""Evaluation-only label sidecars for Residual-Graph DEEM v1.

This module must never be imported by the Stage-A fit runner.  It is the single
Python module in the experiment allowed to translate source correctness into
``y_H`` and to join targets to frozen scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .residual_graph_deem import (
    ResidualGraphDeemError,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from .residual_graph_deem_data import TargetFreeCellBundle


SIDECAR_SCHEMA = "residual_graph_deem_label_sidecar_v1"


@dataclass(frozen=True)
class LabelSidecar:
    cell_id: str
    row_ids: tuple[str, ...]
    y_h: np.ndarray
    sidecar_sha256: str = ""


def require_complete_score_freeze(path: str | Path, expected_cells: Sequence[str]) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("status") != "complete" or value.get("debug"):
        raise ResidualGraphDeemError("label sidecar requires a complete non-debug score freeze")
    if sorted(value.get("cells", [])) != sorted(str(cell) for cell in expected_cells):
        raise ResidualGraphDeemError("score freeze cell roster mismatch")
    if value.get("missing_seeds") or value.get("incomplete_fits"):
        raise ResidualGraphDeemError("score freeze is incomplete")
    return value


def build_label_sidecar(
    bundle: TargetFreeCellBundle,
    identities: Sequence[Any],
) -> LabelSidecar:
    from .fair_comparisons.twentyfour import _binary_correct_label

    identity_by_row = {identity.row_id: identity for identity in identities}
    if set(identity_by_row) != set(bundle.row_ids):
        raise ResidualGraphDeemError("label identity roster differs from frozen fit bundle")
    y_h = np.asarray(
        [
            1 - _binary_correct_label(identity_by_row[row_id].source_candidate)
            for row_id in bundle.row_ids
        ],
        dtype=np.int8,
    )
    return LabelSidecar(cell_id=bundle.cell_id, row_ids=bundle.row_ids, y_h=y_h)


def write_label_sidecar(path: str | Path, sidecar: LabelSidecar) -> dict[str, Any]:
    digest = atomic_save_npz(
        path,
        schema=np.asarray(SIDECAR_SCHEMA),
        cell_id=np.asarray(sidecar.cell_id),
        row_id=np.asarray(sidecar.row_ids, dtype=str),
        y_H=np.asarray(sidecar.y_h, dtype=np.int8),
    )
    manifest = {
        "schema": SIDECAR_SCHEMA,
        "cell_id": sidecar.cell_id,
        "n_rows": len(sidecar.row_ids),
        "sidecar_sha256": digest,
        "unordered_row_id_sha256": canonical_sha256(sorted(sidecar.row_ids)),
    }
    atomic_write_json(Path(path).with_suffix(".manifest.json"), manifest)
    return manifest


def load_label_sidecar(path: str | Path) -> LabelSidecar:
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"].item()) != SIDECAR_SCHEMA:
            raise ResidualGraphDeemError("label-sidecar schema mismatch")
        sidecar = LabelSidecar(
            cell_id=str(data["cell_id"].item()),
            row_ids=tuple(str(value) for value in data["row_id"].tolist()),
            y_h=np.asarray(data["y_H"], dtype=np.int8),
            sidecar_sha256=digest,
        )
    if len(sidecar.row_ids) != len(set(sidecar.row_ids)) or sidecar.y_h.shape != (len(sidecar.row_ids),):
        raise ResidualGraphDeemError("label sidecar IDs/targets are not unique and aligned")
    if not set(np.unique(sidecar.y_h)).issubset({0, 1}):
        raise ResidualGraphDeemError("label sidecar target is not binary")
    return sidecar


def join_labels_by_id(bundle: TargetFreeCellBundle, sidecar: LabelSidecar) -> np.ndarray:
    if bundle.cell_id != sidecar.cell_id:
        raise ResidualGraphDeemError("bundle/sidecar cell mismatch")
    if len(sidecar.row_ids) != len(bundle.row_ids) or set(sidecar.row_ids) != set(bundle.row_ids):
        raise ResidualGraphDeemError("bundle/sidecar join is not bijective")
    labels = dict(zip(sidecar.row_ids, sidecar.y_h.tolist()))
    return np.asarray([labels[row_id] for row_id in bundle.row_ids], dtype=np.int8)


__all__ = [
    "LabelSidecar", "SIDECAR_SCHEMA", "build_label_sidecar", "join_labels_by_id",
    "load_label_sidecar", "require_complete_score_freeze", "write_label_sidecar",
]
