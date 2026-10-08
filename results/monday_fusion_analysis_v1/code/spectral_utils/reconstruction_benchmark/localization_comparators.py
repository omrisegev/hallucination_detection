"""Trusted score-only declassification of localization comparator caches.

The historical pickle containers co-locate task targets with model outputs.
They are never mounted in the fit capsule.  This projector uses an exact
per-kind field allowlist and emits only keyed opaque row IDs, native decisions,
and continuous score vectors.  The manifest states the trust boundary rather
than pretending the raw container lacked targets.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .external_final_answer import (
    ExternalRegistry,
    apply_external_id_contract,
    load_raw_feature_cell,
    resolve_sources,
)
from .io import atomic_write_json, atomic_write_npz, sha256_file
from .localization_contract import load_localization_registry, payload_sha256


PROJECTION_SCHEMA_VERSION = "reconstruction-localization-score-only-projection-v1"
PROJECTION_MANIFEST_SCHEMA_VERSION = "reconstruction-localization-comparator-projections-v1"


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _source_path(source_root: Path, template: str, *, slice_id: str) -> Path:
    relative = template.format(slice_id=slice_id)
    path = (source_root / relative).resolve()
    try:
        path.relative_to(source_root.resolve())
    except ValueError as error:
        raise RuntimeError("localization comparator path escaped source root") from error
    return path


def _row_index(cache: Mapping[Any, Any], *, dataset_id: str) -> dict[str, Mapping[str, Any]]:
    key_name = "id" if dataset_id == "processbench" else "idx"
    output: dict[str, Mapping[str, Any]] = {}
    for row in cache.values():
        if not isinstance(row, Mapping):
            raise RuntimeError("localization comparator row is not a mapping")
        row_id = str(row.get(key_name, ""))
        if not row_id or row_id in output:
            raise RuntimeError("localization comparator raw IDs are empty/duplicated")
        output[row_id] = row
    return output


def _project_rows(
    *,
    kind: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    decisions: list[int] = []
    offsets = [0]
    values: list[float] = []
    coverage: list[int] = []
    for row in rows:
        if kind == "processbench_prm":
            rewards = np.asarray(row.get("rewards", ()), dtype=np.float64)
            if rewards.ndim != 1 or not len(rewards) or not np.isfinite(rewards).all():
                raise RuntimeError("ProcessBench PRM projection has invalid rewards")
            values.extend((-rewards).tolist())
            decisions.append(int(row.get("prediction", -2)))
            coverage.append(1)
        elif kind == "processbench_fixed_prediction":
            prediction = row.get("prediction")
            decisions.append(-2 if prediction is None else int(prediction))
            coverage.append(int(prediction is not None))
        elif kind == "processbench_uprm":
            candidate = row.get("scores")
            if not isinstance(candidate, Mapping) or not candidate:
                decisions.append(-2)
                coverage.append(0)
            else:
                ordered = [float(candidate[key]) for key in sorted(candidate, key=lambda x: int(x))]
                if not np.isfinite(ordered).all():
                    raise RuntimeError("uPRM candidate projection is non-finite")
                values.extend(ordered)
                prediction = row.get("prediction")
                decisions.append(-2 if prediction is None else int(prediction))
                coverage.append(int(prediction is not None))
        elif kind == "prmbench_prm":
            rewards = np.asarray(row.get("rewards", ()), dtype=np.float64)
            if rewards.ndim != 1 or not len(rewards) or not np.isfinite(rewards).all():
                raise RuntimeError("PRMBench PRM projection has invalid rewards")
            values.extend((-rewards).tolist())
            decisions.append(-2)
            coverage.append(1)
        else:
            raise KeyError(f"unknown localization comparator projection kind: {kind}")
        offsets.append(len(values))
    return (
        np.asarray(decisions, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        np.asarray(coverage, dtype=np.int8),
    )


def _project_one(
    *,
    comparator: Mapping[str, Any],
    source_spec: Any,
    registry: ExternalRegistry,
    source_root: Path,
    output_root: Path,
    identity_key: bytes,
) -> dict[str, Any]:
    dataset_id = str(comparator["dataset_id"])
    slice_id = str(source_spec.slice_id)
    path = _source_path(
        source_root, str(comparator["path_template"]), slice_id=slice_id
    )
    cache = _load_pickle(path)
    if not isinstance(cache, Mapping):
        raise RuntimeError("localization comparator cache is not a mapping")
    by_id = _row_index(cache, dataset_id=dataset_id)
    sources = resolve_sources(registry, source_spec, repo=source_root)
    raw = load_raw_feature_cell(source_spec, sources)
    if any(row_id not in by_id for row_id in raw.row_ids):
        raise RuntimeError("comparator projection does not cover the exact response cohort")
    identity = apply_external_id_contract(
        registry, source_spec, raw.row_ids, raw.group_ids, identity_key=identity_key
    )
    order = sorted(range(len(identity.row_ids)), key=lambda index: identity.row_ids[index])
    row_ids = tuple(identity.row_ids[index] for index in order)
    rows = [by_id[raw.row_ids[index]] for index in order]
    decisions, offsets, scores, coverage = _project_rows(
        kind=str(comparator["kind"]), rows=rows
    )
    target = output_root / str(comparator["system_id"]) / f"{source_spec.cell_id}.npz"
    artifact_sha = atomic_write_npz(target, {
        "row_ids": np.asarray(row_ids, dtype="<U80"),
        "native_prediction": decisions.astype("<i8", copy=False),
        "score_offsets": offsets.astype("<i8", copy=False),
        "score": scores.astype("<f8", copy=False),
        "coverage": coverage.astype("<i1", copy=False),
    })
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "system_id": comparator["system_id"],
        "dataset_id": dataset_id,
        "kind": comparator["kind"],
        "cell_id": source_spec.cell_id,
        "model_id": source_spec.model_id,
        "slice_id": slice_id,
        "n_rows": len(row_ids),
        "n_scores": len(scores),
        "n_covered": int(coverage.sum()),
        "access_level": comparator["access_level"],
        "fidelity": comparator["fidelity"],
        "projection_contract": comparator["projection"],
        "raw_container_target_fields_co_located": True,
        "target_fields_selected": False,
        "raw_source_path": path.relative_to(source_root).as_posix(),
        "raw_source_sha256": sha256_file(path),
        "artifact_path": target.relative_to(output_root).as_posix(),
        "artifact_sha256": artifact_sha,
    }


def project_localization_comparators(
    *,
    localization_registry_path: str | Path,
    registry: ExternalRegistry,
    source_root: str | Path,
    output_root: str | Path,
    identity_key: bytes,
    build_id: str,
) -> dict[str, Any]:
    config = load_localization_registry(localization_registry_path)
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"localization comparator projection root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=False)
    source_root_path = Path(source_root).resolve()
    records: list[dict[str, Any]] = []
    for comparator in config["comparators"]:
        if comparator["dataset_id"] == "processbench":
            # Materialize the same score-only comparator against every exact
            # scorer-cell namespace. This preserves row identity while access
            # groups remain separate in evaluation/reporting.
            cell_ids = config["processbench"]["source_cells"]
        else:
            cell_ids = [config["prmbench"]["source_cell"]]
        for cell_id in cell_ids:
            records.append(_project_one(
                comparator=comparator,
                source_spec=registry.by_cell[cell_id],
                registry=registry,
                source_root=source_root_path,
                output_root=root,
                identity_key=identity_key,
            ))
    records.sort(key=lambda row: (row["system_id"], row["cell_id"]))
    manifest = {
        "schema_version": PROJECTION_MANIFEST_SCHEMA_VERSION,
        "build_id": build_id,
        "localization_registry_sha256": sha256_file(localization_registry_path),
        "external_registry_sha256": registry.sha256,
        "raw_containers_target_fields_co_located": True,
        "target_fields_selected": False,
        "fit_capsule_mount": False,
        "records": records,
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    atomic_write_json(root / "MANIFEST.json", manifest)
    return manifest


__all__ = [
    "PROJECTION_MANIFEST_SCHEMA_VERSION", "PROJECTION_SCHEMA_VERSION",
    "project_localization_comparators",
]
