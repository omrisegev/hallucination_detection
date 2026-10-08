"""Physical target firewall and raw-source builder for Residual-Graph DEEM v1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .a5_target_free_data import (
    FrozenSourceSpec,
    _a0_admitted,
    _cropped_telemetry_only,
    _energy_features_from_logsumexp,
    _logprob_features,
    _validated_telemetry,
)
from .fair_comparisons.twentyfour import admit_source_rows
from .feature_contract import confidence_sign_vector
from .feature_utils import extract_all_features
from .residual_graph_deem import (
    ResidualGraphDeemError,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
    validate_inventory,
)


REGISTRY_SCHEMA = "residual_graph_deem_24cell_v1_registry"
BUNDLE_SCHEMA = "residual_graph_deem_target_free_bundle_v1"
TARGET_LIKE = re.compile(
    r"(?:^|_)(?:label|labels|correct|correctness|is_correct|gold|answer|answers|"
    r"reference|target|first_error|error_label|y_h)(?:$|_)", re.IGNORECASE
)


@dataclass(frozen=True)
class TargetFreeCellBundle:
    cell_id: str
    X_raw: np.ndarray
    feature_names: tuple[str, ...]
    confidence_signs: np.ndarray
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    raw_trace_length: np.ndarray
    dataset_family: str
    task_type: str
    source_sha256: str
    manifest_sha256: str
    admission_sha256: str
    inventory_sha256: str
    bundle_sha256: str = ""


def load_registry(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != REGISTRY_SCHEMA:
        raise ResidualGraphDeemError("registry schema mismatch")
    expected = value.get("registry_content_sha256")
    unhashed = dict(value)
    unhashed.pop("registry_content_sha256", None)
    if canonical_sha256(unhashed) != expected:
        raise ResidualGraphDeemError("registry content hash mismatch")
    cells = value.get("cells", [])
    if len(cells) != 24 or sum(int(cell["n_rows"]) for cell in cells) != 48_607:
        raise ResidualGraphDeemError("registry population mismatch")
    return value


def registry_cell(registry: Mapping[str, Any], cell_id: str) -> Mapping[str, Any]:
    matches = [cell for cell in registry["cells"] if cell["cell_id"] == str(cell_id)]
    if len(matches) != 1:
        raise ResidualGraphDeemError(f"unregistered or duplicate cell: {cell_id}")
    return matches[0]


def _source_spec(cell: Mapping[str, Any]) -> FrozenSourceSpec:
    source = cell["source"]
    return FrozenSourceSpec(
        environment_id=source["environment_id"],
        dataset=source["dataset"],
        split=source["split"],
        dataset_family=source["dataset_family"],
        expected_admitted_count=int(source["expected_admitted_count"]),
        admission_mode=source["admission_mode"],
        raw_relative_path=source["raw_relative_path"],
        source_sha256=source["source_sha256"],
        source_size=int(source["source_size"]),
        manifest_sha256=source["manifest_sha256"],
    )


def _source_paths(
    repo_root: Path,
    cell: Mapping[str, Any],
    source_root: str | Path | None,
) -> tuple[Path, Path]:
    source = cell["source"]
    relative = Path(source["raw_relative_path"])
    if source_root is None:
        raw = (repo_root / relative).resolve()
        parent = raw.parent
    else:
        parent = (Path(source_root).resolve() / cell["cell_id"]).resolve()
        raw = parent / relative.name
    canonical_manifest = (
        repo_root / "dataset_cache" / "repgrid" / cell["cell_id"] / "manifest.json"
    ).resolve()
    external_manifest = parent / "manifest.json"
    manifest = canonical_manifest
    if not manifest.is_file() or sha256_file(manifest) != source["manifest_sha256"]:
        manifest = external_manifest
    return raw, manifest


def verify_source(
    repo_root: str | Path,
    cell: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    repo = Path(repo_root).resolve()
    raw, manifest = _source_paths(repo, cell, source_root)
    source = cell["source"]
    if not raw.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"missing raw source/manifest: {cell['cell_id']}")
    if raw.stat().st_size != int(source["source_size"]):
        detail = " (Git-LFS pointer)" if raw.stat().st_size < 1024 else ""
        raise ResidualGraphDeemError(f"source size mismatch: {cell['cell_id']}{detail}")
    if sha256_file(manifest) != source["manifest_sha256"]:
        raise ResidualGraphDeemError(f"manifest hash mismatch: {cell['cell_id']}")
    if sha256_file(raw) != source["source_sha256"]:
        raise ResidualGraphDeemError(f"source byte hash mismatch: {cell['cell_id']}")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_value.get("dataset") != source["dataset"] or manifest_value.get("split") != source["split"]:
        raise ResidualGraphDeemError(f"source dataset/split mismatch: {cell['cell_id']}")
    cells = manifest_value.get("cells")
    if not isinstance(cells, list) or len(cells) != 1 or cells[0].get("pkl") != raw.name:
        raise ResidualGraphDeemError(f"manifest does not bind raw source: {cell['cell_id']}")
    return raw, {
        "source_sha256": source["source_sha256"],
        "source_size": int(source["source_size"]),
        "manifest_sha256": source["manifest_sha256"],
        "admission_sha256": source["admission_contract_sha256"],
    }


def _inventory_features(telemetry: Mapping[str, Any], *, allow_short: bool) -> dict[str, float]:
    entropy, spilled, logsum, topk = _validated_telemetry(telemetry)
    output = extract_all_features(
        entropy, spilled_energies=spilled, allow_short=allow_short
    ) or {}
    output.update(_energy_features_from_logsumexp(logsum))
    output.update(_logprob_features(topk))
    return {str(name): float(value) for name, value in output.items()}


def build_target_free_cell(
    repo_root: str | Path,
    registry: Mapping[str, Any],
    cell_id: str,
    *,
    source_root: str | Path | None = None,
) -> tuple[TargetFreeCellBundle, tuple[Any, ...]]:
    cell = registry_cell(registry, cell_id)
    raw_path, audit = verify_source(repo_root, cell, source_root=source_root)
    with raw_path.open("rb") as handle:
        raw_source = pickle.load(handle)
    spec = _source_spec(cell)
    identities = admit_source_rows(raw_source, spec)
    names = tuple(str(name) for name in cell["feature_names"])
    rows = []
    for identity in identities:
        values = _inventory_features(
            identity.telemetry, allow_short=spec.admission_mode == "cropped_all_rows"
        )
        missing = [name for name in names if name not in values or not np.isfinite(values[name])]
        if missing:
            raise ResidualGraphDeemError(
                f"registered inventory undefined for {identity.row_id}: {missing}"
            )
        rows.append([values[name] for name in names])
    X_raw = validate_inventory(np.asarray(rows, dtype=np.float64), names)
    signs = confidence_sign_vector(names).astype(np.int8)
    if names != tuple(cell["feature_names"]) or not np.array_equal(signs, cell["confidence_signs"]):
        raise ResidualGraphDeemError("inventory order/sign mismatch")
    if len(X_raw) != int(cell["n_rows"]):
        raise ResidualGraphDeemError("admitted row count mismatch")
    inventory_hash = canonical_sha256(
        {"feature_names": names, "confidence_signs": signs.tolist()}
    )
    if inventory_hash != cell["inventory_sha256"]:
        raise ResidualGraphDeemError("inventory content hash mismatch")
    bundle = TargetFreeCellBundle(
        cell_id=str(cell_id),
        X_raw=X_raw,
        feature_names=names,
        confidence_signs=signs,
        row_ids=tuple(identity.row_id for identity in identities),
        group_ids=tuple(identity.group_id for identity in identities),
        raw_trace_length=np.asarray(
            [len(identity.telemetry["token_entropies"]) for identity in identities],
            dtype=np.int64,
        ),
        dataset_family=str(cell["dataset_family"]),
        task_type=str(cell["task_type"]),
        source_sha256=audit["source_sha256"],
        manifest_sha256=audit["manifest_sha256"],
        admission_sha256=audit["admission_sha256"],
        inventory_sha256=inventory_hash,
    )
    return bundle, identities


def bundle_arrays(bundle: TargetFreeCellBundle) -> dict[str, np.ndarray]:
    arrays = {
        "schema": np.asarray(BUNDLE_SCHEMA),
        "cell_id": np.asarray(bundle.cell_id),
        "X_raw": np.asarray(bundle.X_raw, dtype=np.float64),
        "feature_names": np.asarray(bundle.feature_names, dtype=str),
        "confidence_signs": np.asarray(bundle.confidence_signs, dtype=np.int8),
        "row_id": np.asarray(bundle.row_ids, dtype=str),
        "group_id": np.asarray(bundle.group_ids, dtype=str),
        "raw_trace_length": np.asarray(bundle.raw_trace_length, dtype=np.int64),
        "dataset_family": np.asarray(bundle.dataset_family),
        "task_type": np.asarray(bundle.task_type),
        "source_sha256": np.asarray(bundle.source_sha256),
        "source_manifest_sha256": np.asarray(bundle.manifest_sha256),
        "admission_sha256": np.asarray(bundle.admission_sha256),
        "inventory_sha256": np.asarray(bundle.inventory_sha256),
    }
    assert_no_target_fields(arrays)
    return arrays


def assert_no_target_fields(payload: Mapping[str, Any]) -> None:
    forbidden = sorted(key for key in payload if TARGET_LIKE.search(str(key)))
    if forbidden:
        raise ResidualGraphDeemError("target-like fields in fit payload: " + ", ".join(forbidden))


def write_target_free_bundle(path: str | Path, bundle: TargetFreeCellBundle) -> dict[str, Any]:
    digest = atomic_save_npz(path, **bundle_arrays(bundle))
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "cell_id": bundle.cell_id,
        "n_rows": len(bundle.row_ids),
        "n_features": len(bundle.feature_names),
        "bundle_sha256": digest,
        "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
        "inventory_sha256": bundle.inventory_sha256,
        "source_sha256": bundle.source_sha256,
        "manifest_sha256": bundle.manifest_sha256,
        "admission_sha256": bundle.admission_sha256,
        "labels_accessed": False,
        "allow_pickle": False,
    }
    atomic_write_json(Path(path).with_suffix(".manifest.json"), manifest)
    return manifest


def load_target_free_bundle(path: str | Path) -> TargetFreeCellBundle:
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        assert_no_target_fields({name: None for name in data.files})
        if str(data["schema"].item()) != BUNDLE_SCHEMA:
            raise ResidualGraphDeemError("target-free bundle schema mismatch")
        bundle = TargetFreeCellBundle(
            cell_id=str(data["cell_id"].item()),
            X_raw=np.asarray(data["X_raw"], dtype=float),
            feature_names=tuple(str(value) for value in data["feature_names"].tolist()),
            confidence_signs=np.asarray(data["confidence_signs"], dtype=np.int8),
            row_ids=tuple(str(value) for value in data["row_id"].tolist()),
            group_ids=tuple(str(value) for value in data["group_id"].tolist()),
            raw_trace_length=np.asarray(data["raw_trace_length"], dtype=np.int64),
            dataset_family=str(data["dataset_family"].item()),
            task_type=str(data["task_type"].item()),
            source_sha256=str(data["source_sha256"].item()),
            manifest_sha256=str(data["source_manifest_sha256"].item()),
            admission_sha256=str(data["admission_sha256"].item()),
            inventory_sha256=str(data["inventory_sha256"].item()),
            bundle_sha256=digest,
        )
    validate_inventory(bundle.X_raw, bundle.feature_names)
    if len(bundle.row_ids) != len(set(bundle.row_ids)) or len(bundle.row_ids) != len(bundle.X_raw):
        raise ResidualGraphDeemError("bundle row IDs are not unique/aligned")
    return bundle


__all__ = [
    "BUNDLE_SCHEMA", "REGISTRY_SCHEMA", "TargetFreeCellBundle",
    "assert_no_target_fields", "build_target_free_cell", "bundle_arrays",
    "load_registry", "load_target_free_bundle", "registry_cell", "verify_source",
    "write_target_free_bundle",
]
