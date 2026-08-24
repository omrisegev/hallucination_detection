"""Strict external final-answer reconstruction boundary.

The external populations were collected by several historical pipelines.  This
module gives them one fail-closed interface without pretending that every saved
cache satisfies the same feature contract.  Fitting code can receive only a
target-free :class:`PreparedCell`; labels are opened by a separate function and
only after a complete score-freeze manifest has been verified.

The contract is intentionally stricter than older application scripts.  All 30
nominal views are audited: a present view must be finite on every registered
row, while a uniformly unavailable view may remain absent.  Partial
availability, convenience column selection, row selection, median imputation,
raw/pre-warper substitution, and a second orientation pass are forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import CONTRACT_VERSION, PreparedCell, prepared_matrix_sha256
from .io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from ..dufs_liu_feature_contract import dufs_liu_mixed_v2_matrix
from ..fair_comparisons.stopping import grade_aqua_option
from ..repgrid_scoring import _candidate_features, logprob_features_extended
from ..specrage_views import FEATURE_TO_VIEW


SCHEMA_VERSION = "reconstruction-external-final-answer-registry-v1"
PREPARED_SCHEMA_VERSION = "reconstruction-external-target-free-input-v1"
AUDIT_SCHEMA_VERSION = "reconstruction-external-applicability-v1"
LABEL_SCHEMA_VERSION = "reconstruction-external-label-vector-v1"
SCORE_FREEZE_SCHEMA_VERSION = "reconstruction-external-score-freeze-v1"
CANONICAL_FEATURE_NAMES = tuple(FEATURE_TO_VIEW)

LFS_MARKER = b"version https://git-lfs.github.com/spec/v1\n"
TARGET_LIKE_FRAGMENTS = (
    "label",
    "target",
    "correct",
    "answer_key",
    "gold",
    "auc",
    "auprc",
)


class ReadinessStatus(str, Enum):
    READY_FOR_TARGET_FREE_PREPARATION = "READY_FOR_TARGET_FREE_PREPARATION"
    ELIGIBLE = "ELIGIBLE"
    BLOCKED_ASSET = "BLOCKED_ASSET"
    SOURCE_HASH_MISMATCH = "SOURCE_HASH_MISMATCH"
    SOURCE_INVENTORY_MISMATCH = "SOURCE_INVENTORY_MISMATCH"
    ROW_CONTRACT_MISMATCH = "ROW_CONTRACT_MISMATCH"
    LABEL_PROVENANCE_BLOCKED = "LABEL_PROVENANCE_BLOCKED"
    INCOMPATIBLE_FEATURE_CONTRACT = "INCOMPATIBLE_FEATURE_CONTRACT"
    PROTOCOL_GATE_FAILED = "PROTOCOL_GATE_FAILED"
    QUARANTINED = "QUARANTINED"


class ExternalContractError(RuntimeError):
    def __init__(self, status: ReadinessStatus, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    sha256: str
    size_bytes: int | None = None
    role: str = "telemetry"


@dataclass(frozen=True)
class ExternalCellSpec:
    cell_id: str
    population_id: str
    dataset_id: str
    model_id: str
    slice_id: str
    domain: str
    comparison_group_id: str
    expected_rows: int
    adapter_id: str
    fit_policy: str
    panel_role: str
    source: Mapping[str, Any]
    expected_incorrect: int | None = None
    expected_correct: int | None = None
    expected_group_count: int | None = None
    excluded_row_ids: tuple[str, ...] = ()
    configured_status: str | None = None
    status_reason: str | None = None
    known_contract_risk: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExternalCellSpec":
        required = {
            "cell_id", "population_id", "dataset_id", "model_id", "slice_id",
            "domain", "comparison_group_id", "expected_rows", "adapter_id",
            "fit_policy", "panel_role",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"external cell is missing fields: {missing}")
        fit_policy = str(value["fit_policy"])
        source = value.get("source", {})
        if fit_policy == "run_if_compatible" and not source:
            raise ValueError(f"{value['cell_id']}: runnable cell lacks a source")
        return cls(
            cell_id=str(value["cell_id"]),
            population_id=str(value["population_id"]),
            dataset_id=str(value["dataset_id"]),
            model_id=str(value["model_id"]),
            slice_id=str(value["slice_id"]),
            domain=str(value["domain"]),
            comparison_group_id=str(value["comparison_group_id"]),
            expected_rows=int(value["expected_rows"]),
            adapter_id=str(value["adapter_id"]),
            fit_policy=fit_policy,
            panel_role=str(value["panel_role"]),
            source=dict(source),
            expected_incorrect=_optional_int(value.get("expected_incorrect")),
            expected_correct=_optional_int(value.get("expected_correct")),
            expected_group_count=_optional_int(value.get("expected_group_count")),
            excluded_row_ids=tuple(map(str, value.get("excluded_row_ids", ()))),
            configured_status=(
                None if value.get("configured_status") is None
                else str(value["configured_status"])
            ),
            status_reason=(
                None if value.get("status_reason") is None
                else str(value["status_reason"])
            ),
            known_contract_risk=(
                None if value.get("known_contract_risk") is None
                else str(value["known_contract_risk"])
            ),
        )


@dataclass(frozen=True)
class ExternalRegistry:
    path: Path
    sha256: str
    population_registry_path: Path
    population_registry_sha256: str
    raw: Mapping[str, Any]
    cells: tuple[ExternalCellSpec, ...]

    @property
    def by_cell(self) -> dict[str, ExternalCellSpec]:
        return {cell.cell_id: cell for cell in self.cells}


@dataclass(frozen=True)
class ResolvedSources:
    feature_files: tuple[SourceFile, ...]
    label_files: tuple[SourceFile, ...]
    provenance_files: tuple[SourceFile, ...]
    inventory_bindings: tuple[Mapping[str, Any], ...]

    @property
    def all_files(self) -> tuple[SourceFile, ...]:
        unique: dict[str, SourceFile] = {}
        for item in (*self.feature_files, *self.label_files, *self.provenance_files):
            unique[item.relative_path] = item
        return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True)
class RawFeatureCell:
    spec: ExternalCellSpec
    raw_matrix: np.ndarray
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    source_files: tuple[SourceFile, ...]
    feature_names: tuple[str, ...] = CANONICAL_FEATURE_NAMES
    preprocessing_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.raw_matrix, dtype=float)
        names = tuple(map(str, self.feature_names))
        expected_names = tuple(name for name in CANONICAL_FEATURE_NAMES if name in set(names))
        if names != expected_names or len(set(names)) != len(names):
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{self.spec.cell_id}: present features are not an ordered subset of the nominal frozen 30",
            )
        if len(names) < 3:
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{self.spec.cell_id}: fewer than three fully present features",
            )
        if matrix.shape != (self.spec.expected_rows, len(names)):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: raw matrix shape {matrix.shape} does not equal "
                f"({self.spec.expected_rows}, {len(names)})",
            )
        if self.preprocessing_steps:
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{self.spec.cell_id}: source claims prior preprocessing; mixed-v2 must run once",
            )
        if not np.isfinite(matrix).all():
            locations = np.argwhere(~np.isfinite(matrix))[:10].tolist()
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{self.spec.cell_id}: present-feature matrix has non-finite values at {locations}",
            )
        rows = tuple(map(str, self.row_ids))
        groups = tuple(map(str, self.group_ids))
        if len(rows) != len(groups) or len(rows) != len(matrix):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: row/group/matrix lengths disagree",
            )
        if any(not value for value in rows) or len(set(rows)) != len(rows):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: row IDs are empty or duplicated",
            )
        if any(not value for value in groups):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: group IDs must be nonempty",
            )
        if (
            self.spec.expected_group_count is not None
            and len(set(groups)) != self.spec.expected_group_count
        ):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: expected {self.spec.expected_group_count} groups, "
                f"found {len(set(groups))}",
            )
        matrix = np.array(matrix, dtype=np.float64, order="C", copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "raw_matrix", matrix)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "feature_names", names)


@dataclass(frozen=True)
class LabelVector:
    cell_id: str
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    incorrect: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        labels = np.asarray(self.incorrect, dtype=np.int8)
        if labels.shape != (len(self.row_ids),):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.cell_id}: label and row counts disagree",
            )
        if not np.isin(labels, (0, 1)).all():
            raise ExternalContractError(
                ReadinessStatus.LABEL_PROVENANCE_BLOCKED,
                f"{self.cell_id}: labels are not binary incorrect indicators",
            )
        labels = np.array(labels, dtype=np.int8, copy=True)
        labels.setflags(write=False)
        object.__setattr__(self, "incorrect", labels)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def load_external_registry(
    *,
    repo: str | Path,
    registry_path: str | Path,
    population_registry_path: str | Path,
) -> ExternalRegistry:
    root = Path(repo).resolve()
    registry_file = _resolve_under(root, registry_path)
    population_file = _resolve_under(root, population_registry_path)
    raw = _read_json(registry_file)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected external final-answer registry schema")
    if raw.get("feature_contract_id") != CONTRACT_VERSION:
        raise ValueError("external registry does not bind the frozen mixed-v2 contract")
    cells = tuple(ExternalCellSpec.from_mapping(item) for item in raw.get("cells", ()))
    if not cells or len({item.cell_id for item in cells}) != len(cells):
        raise ValueError("external cell registry is empty or contains duplicate IDs")
    populations = _read_json(population_file)
    population_rows = {
        str(item["population_id"]): item
        for item in populations.get("populations", ())
    }
    known = set(population_rows)
    missing = sorted({item.population_id for item in cells} - known)
    if missing:
        raise ValueError("external cells reference unregistered populations: " + ", ".join(missing))
    aliases = {str(key): str(value) for key, value in raw.get("model_id_aliases", {}).items()}
    for item in cells:
        if item.fit_policy not in {"run_if_compatible", "forbidden"}:
            raise ValueError(f"{item.cell_id}: unknown fit_policy {item.fit_policy}")
        population = population_rows[item.population_id]
        declared_models = []
        if population.get("model") is not None:
            declared_models.append(str(population["model"]))
        declared_models.extend(map(str, population.get("scorer_models", ())))
        declared_models.extend(map(str, population.get("models", ())))
        if item.fit_policy == "run_if_compatible" and declared_models:
            unaliased = sorted(set(declared_models) - set(aliases))
            if unaliased:
                raise ValueError(
                    f"{item.population_id}: population model metadata lacks canonical aliases: "
                    + ", ".join(unaliased)
                )
            expected_model_ids = {aliases[value] for value in declared_models}
            if item.model_id not in expected_model_ids:
                raise ValueError(
                    f"{item.cell_id}: model_id {item.model_id!r} conflicts with population "
                    f"metadata {sorted(expected_model_ids)!r}"
                )
        if item.fit_policy == "forbidden":
            if item.configured_status not in {
                ReadinessStatus.PROTOCOL_GATE_FAILED.value,
                ReadinessStatus.QUARANTINED.value,
            }:
                raise ValueError(f"{item.cell_id}: forbidden cell lacks a terminal status")
        if item.fit_policy == "run_if_compatible" and item.source.get("kind") == "explicit":
            manifest_registry = raw.get("explicit_source_manifests", {})
            for source_file in item.source.get("files", ()):
                if source_file.get("role", "telemetry") != "telemetry":
                    continue
                manifest = str(Path(str(source_file["path"])).parent / "manifest.json")
                if manifest not in manifest_registry:
                    raise ValueError(
                        f"{item.cell_id}: explicit telemetry lacks a hash-frozen source manifest"
                    )
    expectations = raw.get("population_expectations", {})
    aggregates = raw.get("population_aggregates", {})
    if set(expectations) != set(aggregates):
        raise ValueError("population expectation and aggregate registries must have identical IDs")
    for population_id, expected in expectations.items():
        members = [item for item in cells if item.population_id == population_id]
        observed = {
            "rows": sum(item.expected_rows for item in members),
            "cells": len(members),
        }
        for key, value in observed.items():
            if int(expected.get(key, -1)) != value:
                raise ValueError(
                    f"{population_id}: external {key} total {value} conflicts with "
                    f"registered expectation {expected.get(key)!r}"
                )
        aggregate = aggregates[population_id]
        if aggregate.get("enabled") is True:
            if aggregate.get("weighting") not in {"single_cell", "equal_cell"}:
                raise ValueError(f"{population_id}: unsupported aggregate weighting")
            if aggregate.get("link_cells_by") not in {"none", "slice_id", "all"}:
                raise ValueError(f"{population_id}: unsupported cross-cell linkage rule")
    return ExternalRegistry(
        path=registry_file,
        sha256=sha256_file(registry_file),
        population_registry_path=population_file,
        population_registry_sha256=sha256_file(population_file),
        raw=raw,
        cells=cells,
    )


def _resolve_under(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source path escapes the repository root: {path}") from error
    return resolved


def _inventory(root: Path, entry: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    path = _resolve_under(root, str(entry["path"]))
    if not path.is_file():
        raise ExternalContractError(
            ReadinessStatus.BLOCKED_ASSET, f"source inventory is missing: {path}"
        )
    observed = sha256_file(path)
    if observed != str(entry["sha256"]):
        raise ExternalContractError(
            ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
            f"source inventory hash mismatch for {path}: expected {entry['sha256']}, got {observed}",
        )
    return path, _read_json(path)


def _source_file(
    root: Path,
    relative: str,
    digest: str,
    *,
    size_bytes: int | None = None,
    role: str = "telemetry",
) -> SourceFile:
    return SourceFile(
        path=_resolve_under(root, relative),
        relative_path=str(relative),
        sha256=str(digest),
        size_bytes=_optional_int(size_bytes),
        role=role,
    )


def resolve_sources(
    registry: ExternalRegistry,
    spec: ExternalCellSpec,
    *,
    repo: str | Path,
) -> ResolvedSources:
    root = Path(repo).resolve()
    source = spec.source
    kind = str(source.get("kind", ""))
    feature: list[SourceFile] = []
    labels: list[SourceFile] = []
    provenance: list[SourceFile] = []
    bindings: list[Mapping[str, Any]] = []
    if kind == "explicit":
        for item in source.get("files", ()):
            role = str(item.get("role", "telemetry"))
            resolved = _source_file(
                root,
                str(item["path"]),
                str(item["sha256"]),
                size_bytes=item.get("size_bytes"),
                role=role,
            )
            if role == "telemetry":
                feature.append(resolved)
            elif role == "labels":
                labels.append(resolved)
            elif role in {"manifest", "inventory", "provenance"}:
                provenance.append(resolved)
            else:
                raise ExternalContractError(
                    ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                    f"{spec.cell_id}: unsupported explicit source role {role!r}",
                )
        registered_manifests = registry.raw.get("explicit_source_manifests", {})
        for telemetry in feature:
            manifest_relative = str(Path(telemetry.relative_path).parent / "manifest.json")
            digest = registered_manifests.get(manifest_relative)
            if digest is not None:
                provenance.append(_source_file(
                    root, manifest_relative, str(digest), role="manifest"
                ))
    elif kind == "data_readiness":
        inv_entry = registry.raw["source_inventories"]["data_readiness"]
        inv_path, inventory = _inventory(root, inv_entry)
        candidates = [
            item for item in inventory.get("datasets", ())
            if str(item.get("dataset_id")) == str(source["dataset_id"])
        ]
        if len(candidates) != 1:
            raise ExternalContractError(
                ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                f"{spec.cell_id}: data-readiness selector did not resolve exactly once",
            )
        row = candidates[0]
        hashes = {str(key): str(value) for key, value in row.get("file_hashes", {}).items()}
        suffix = str(source["path_suffix"])
        matches = [path for path in row.get("source_paths", ()) if str(path).endswith(suffix)]
        if len(matches) != 1 or matches[0] not in hashes:
            raise ExternalContractError(
                ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                f"{spec.cell_id}: feature source suffix {suffix!r} is ambiguous or unhashed",
            )
        feature.append(_source_file(root, matches[0], hashes[matches[0]]))
        label_suffix = source.get("label_path_suffix")
        if label_suffix is not None:
            label_matches = [
                path for path in row.get("source_paths", ())
                if str(path).endswith(str(label_suffix))
            ]
            if len(label_matches) != 1 or label_matches[0] not in hashes:
                raise ExternalContractError(
                    ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                    f"{spec.cell_id}: label source suffix is ambiguous or unhashed",
                )
            labels.append(_source_file(
                root, label_matches[0], hashes[label_matches[0]], role="labels"
            ))
        for relative in row.get("source_paths", ()):
            if (
                str(relative).endswith(("manifest.json", "_manifest.json"))
                and relative in hashes
            ):
                provenance.append(_source_file(
                    root, relative, hashes[relative], role="manifest"
                ))
        provenance.append(SourceFile(
            inv_path,
            inv_path.relative_to(root).as_posix(),
            str(inv_entry["sha256"]),
            inv_path.stat().st_size,
            "inventory",
        ))
        bindings.append({
            "inventory": inv_path.relative_to(root).as_posix(),
            "inventory_sha256": str(inv_entry["sha256"]),
            "selector": {"dataset_id": source["dataset_id"]},
        })
    elif kind == "aqua_external":
        inv_entry = registry.raw["source_inventories"]["aqua_external"]
        inv_path, inventory = _inventory(root, inv_entry)
        candidates = [
            item for item in inventory.get("cells", ())
            if str(item.get("cell")) == str(source["cell"])
        ]
        if len(candidates) != 1:
            raise ExternalContractError(
                ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                f"{spec.cell_id}: AQuA inventory selector did not resolve exactly once",
            )
        row = candidates[0]
        hashes = {str(key): str(value) for key, value in row.get("source_sha256", {}).items()}
        for relative in row.get("source_paths", ()):
            if relative not in hashes:
                raise ExternalContractError(
                    ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                    f"{spec.cell_id}: AQuA source lacks an expected hash: {relative}",
                )
            item = _source_file(root, relative, hashes[relative])
            if str(relative).endswith(".pkl"):
                feature.append(item)
            else:
                provenance.append(SourceFile(
                    item.path, item.relative_path, item.sha256, item.size_bytes, "manifest"
                ))
        provenance.append(SourceFile(
            inv_path,
            inv_path.relative_to(root).as_posix(),
            str(inv_entry["sha256"]),
            inv_path.stat().st_size,
            "inventory",
        ))
        bindings.append({
            "inventory": inv_path.relative_to(root).as_posix(),
            "inventory_sha256": str(inv_entry["sha256"]),
            "selector": {"cell": source["cell"]},
        })
    else:
        raise ExternalContractError(
            ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
            f"{spec.cell_id}: unsupported source resolver {kind!r}",
        )
    if not feature:
        raise ExternalContractError(
            ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
            f"{spec.cell_id}: no feature source resolved",
        )
    return ResolvedSources(tuple(feature), tuple(labels), tuple(provenance), tuple(bindings))


def _parse_lfs_pointer(path: Path) -> tuple[str, int] | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        prefix = handle.read(256)
    if not prefix.startswith(LFS_MARKER):
        return None
    fields: dict[str, str] = {}
    for line in prefix.decode("utf-8", errors="strict").splitlines()[1:]:
        if " " in line:
            key, value = line.split(" ", 1)
            fields[key] = value
    oid = fields.get("oid", "")
    if not oid.startswith("sha256:") or "size" not in fields:
        raise ExternalContractError(
            ReadinessStatus.BLOCKED_ASSET, f"malformed Git-LFS pointer: {path}"
        )
    return oid.removeprefix("sha256:"), int(fields["size"])


def verify_source_file(item: SourceFile) -> Mapping[str, Any]:
    if not item.path.is_file():
        raise ExternalContractError(
            ReadinessStatus.BLOCKED_ASSET, f"missing source asset: {item.relative_path}"
        )
    pointer = _parse_lfs_pointer(item.path)
    if pointer is not None:
        oid, size = pointer
        if oid != item.sha256 or (item.size_bytes is not None and size != item.size_bytes):
            raise ExternalContractError(
                ReadinessStatus.SOURCE_HASH_MISMATCH,
                f"Git-LFS pointer metadata disagrees with the registry: {item.relative_path}",
            )
        raise ExternalContractError(
            ReadinessStatus.BLOCKED_ASSET,
            f"Git-LFS object is not materialized: {item.relative_path} ({oid}, {size} bytes)",
        )
    observed_size = item.path.stat().st_size
    if item.size_bytes is not None and observed_size != item.size_bytes:
        raise ExternalContractError(
            ReadinessStatus.SOURCE_HASH_MISMATCH,
            f"source size mismatch for {item.relative_path}: expected {item.size_bytes}, got {observed_size}",
        )
    observed_hash = sha256_file(item.path)
    if observed_hash != item.sha256:
        raise ExternalContractError(
            ReadinessStatus.SOURCE_HASH_MISMATCH,
            f"source hash mismatch for {item.relative_path}: expected {item.sha256}, got {observed_hash}",
        )
    return {
        "path": item.relative_path,
        "sha256": observed_hash,
        "size_bytes": observed_size,
        "role": item.role,
        "materialized": True,
    }


def verify_sources(sources: ResolvedSources, *, include_labels: bool = False) -> list[Mapping[str, Any]]:
    selected = [*sources.feature_files, *sources.provenance_files]
    if include_labels:
        selected.extend(sources.label_files)
    return [verify_source_file(item) for item in selected]


def _stable_key(value: Any) -> tuple[int, int | str, str]:
    text = str(value)
    try:
        return (0, int(text), text)
    except ValueError:
        return (1, text, text)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _telemetry_features(row: Mapping[str, Any], *, allow_short: bool) -> Mapping[str, float]:
    # Only telemetry members are copied.  Embedded answer/label fields never
    # enter this function's output or a fitting artifact.
    telemetry = {
        "token_entropies": row.get("token_entropies"),
        "token_spilled_energies": row.get("token_spilled_energies"),
        "token_logsumexp": row.get("token_logsumexp"),
        "top_k_logprobs": row.get("top_k_logprobs"),
    }
    output = dict(_candidate_features(telemetry, allow_short=allow_short))
    if telemetry["top_k_logprobs"] is not None:
        output.update(logprob_features_extended(telemetry["top_k_logprobs"]))
    return output


def _matrix_from_feature_rows(
    spec: ExternalCellSpec,
    rows: Sequence[Mapping[str, float]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if len(rows) != spec.expected_rows:
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: expected {spec.expected_rows} rows, found {len(rows)}",
        )
    nominal = np.asarray([
        [row.get(name, np.nan) for name in CANONICAL_FEATURE_NAMES]
        for row in rows
    ], dtype=np.float64)
    if nominal.shape != (spec.expected_rows, len(CANONICAL_FEATURE_NAMES)):
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: cannot audit the nominal raw 30-feature roster",
        )
    finite_counts = np.sum(np.isfinite(nominal), axis=0)
    partial = {
        name: int(count)
        for name, count in zip(CANONICAL_FEATURE_NAMES, finite_counts)
        if 0 < int(count) < spec.expected_rows
    }
    if partial:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: partially available views are forbidden; "
            f"finite_rows_by_feature={partial}, expected_rows={spec.expected_rows}",
        )
    keep = finite_counts == spec.expected_rows
    names = tuple(
        name for name, present in zip(CANONICAL_FEATURE_NAMES, keep) if bool(present)
    )
    if len(names) < 3:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: only {len(names)} nominal views are fully present",
        )
    matrix = nominal[:, keep]
    if not np.isfinite(matrix).all():
        raise AssertionError("present-feature mask admitted a non-finite value")
    return matrix, names


def _processbench_raw(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    if len(sources.feature_files) != 1:
        raise ExternalContractError(ReadinessStatus.SOURCE_INVENTORY_MISMATCH, "ProcessBench needs one cache")
    cache = _load_pickle(sources.feature_files[0].path)
    if not isinstance(cache, Mapping):
        raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "ProcessBench cache is not a mapping")
    feature_rows: list[Mapping[str, float]] = []
    row_ids: list[str] = []
    for key in sorted(cache, key=_stable_key):
        row = cache[key]
        if not isinstance(row, Mapping):
            raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "ProcessBench row is not a mapping")
        alignment = row.get("align_diag", {})
        problems = alignment.get("problems")
        if problems or alignment.get("ok") is False:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{spec.cell_id}: registered population contains alignment problems at {key}: {problems}",
            )
        official_id = row.get("id")
        if not isinstance(official_id, str) or not official_id:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"ProcessBench source row {key} lacks its official string ID",
            )
        row_id = official_id
        feature_rows.append(_telemetry_features(row, allow_short=True))
        row_ids.append(row_id)
    matrix, names = _matrix_from_feature_rows(spec, feature_rows)
    return RawFeatureCell(
        spec, matrix, tuple(row_ids), tuple(row_ids), sources.feature_files,
        feature_names=names,
    )


def _prmbench_raw(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    if len(sources.feature_files) != 1:
        raise ExternalContractError(ReadinessStatus.SOURCE_INVENTORY_MISMATCH, "PRMBench needs one cache")
    cache = _load_pickle(sources.feature_files[0].path)
    if not isinstance(cache, Mapping):
        raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "PRMBench cache is not a mapping")
    excluded = set(spec.excluded_row_ids)
    seen_excluded: set[str] = set()
    feature_rows: list[Mapping[str, float]] = []
    row_ids: list[str] = []
    groups: list[str] = []
    for key in sorted(cache, key=_stable_key):
        row = cache[key]
        row_id = str(row.get("idx", ""))
        if not row_id:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"PRMBench source row {key} lacks idx",
            )
        if row_id in excluded:
            seen_excluded.add(row_id)
            continue
        alignment = row.get("align_diag", {})
        problems = alignment.get("problems")
        if problems or alignment.get("ok") is False:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{spec.cell_id}: unexpected non-preregistered alignment exclusion {row_id}",
            )
        feature_rows.append(_telemetry_features(row, allow_short=True))
        row_ids.append(row_id)
        source_group = str(row.get("source_idx", ""))
        if not source_group:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"PRMBench row {row_id} lacks source_idx",
            )
        groups.append(source_group)
    if seen_excluded != excluded:
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: preregistered exclusions missing from source: {sorted(excluded - seen_excluded)}",
        )
    matrix, names = _matrix_from_feature_rows(spec, feature_rows)
    return RawFeatureCell(
        spec, matrix, tuple(row_ids), tuple(groups), sources.feature_files,
        feature_names=names,
    )


def _repgrid_raw(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    if len(sources.feature_files) != 1:
        raise ExternalContractError(ReadinessStatus.SOURCE_INVENTORY_MISMATCH, "repgrid cell needs one cache")
    data = _load_pickle(sources.feature_files[0].path)
    if not isinstance(data, Mapping):
        raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "repgrid cache is not a mapping")
    feature_rows: list[Mapping[str, float]] = []
    row_ids: list[str] = []
    groups: list[str] = []
    for key in sorted(data, key=_stable_key):
        entry = data[key]
        candidates = entry.get("candidates") if isinstance(entry, Mapping) else None
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{spec.cell_id}: source group {key} lacks a candidate sequence",
            )
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "candidate is not a mapping")
            feature_rows.append(_telemetry_features(candidate, allow_short=False))
            group = str(key)
            row_ids.append(f"{group}:candidate:{index:04d}")
            groups.append(group)
    matrix, names = _matrix_from_feature_rows(spec, feature_rows)
    return RawFeatureCell(
        spec, matrix, tuple(row_ids), tuple(groups), sources.feature_files,
        feature_names=names,
    )


def _aqua_raw(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    records: list[Mapping[str, Any]] = []
    for item in sorted(sources.feature_files, key=lambda value: value.relative_path):
        shard = _load_pickle(item.path)
        if not isinstance(shard, Sequence) or isinstance(shard, (str, bytes)):
            raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "AQuA shard is not a row sequence")
        records.extend(shard)
    selected = [
        row for row in records
        if row.get("arm") == "cot" and row.get("setting_label") == "central"
    ]
    selected.sort(key=lambda row: _stable_key(str(row.get("question_id", "")).split(":")[-1]))
    if len(selected) != spec.expected_rows:
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: expected {spec.expected_rows} cot|central rows, found {len(selected)}",
        )
    feature_rows: list[Mapping[str, float]] = []
    row_ids: list[str] = []
    for row in selected:
        # The frozen detector was developed on canonical post-warper streams.
        # Historical AQuA shards often contain only raw/pre-warper streams and
        # a NaN sampled-entropy channel.  Reconstructing a surrogate from raw
        # top-k values would define a different method and is forbidden.
        channels = row.get("channels", {})
        entropy = channels.get("sampled_entropy")
        spilled = channels.get("spilled_energy")
        logsumexp = channels.get("logsumexp")
        topk = row.get("top_k_logprobs")
        if any(value is None for value in (entropy, spilled, logsumexp, topk)):
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{spec.cell_id}: canonical post-warper entropy/spilled/logsumexp/top-k streams are absent; raw substitution is forbidden",
            )
        candidate = {
            "token_entropies": entropy,
            "token_spilled_energies": spilled,
            "token_logsumexp": logsumexp,
            "top_k_logprobs": topk,
        }
        feature_rows.append(_telemetry_features(candidate, allow_short=False))
        row_ids.append(str(row.get("question_id", "")))
    matrix, names = _matrix_from_feature_rows(spec, feature_rows)
    return RawFeatureCell(
        spec, matrix, tuple(row_ids), tuple(row_ids), sources.feature_files,
        feature_names=names,
    )


RAW_ADAPTERS = {
    "processbench_teacher_forced_v1": _processbench_raw,
    "prmbench_teacher_forced_v1": _prmbench_raw,
    "repgrid_embedded_label_v1": _repgrid_raw,
    "hle_separate_judge_v1": _repgrid_raw,
    "aqua_paper_exact_cot_v1": _aqua_raw,
}


def load_raw_feature_cell(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    if spec.fit_policy != "run_if_compatible":
        status = ReadinessStatus(str(spec.configured_status))
        raise ExternalContractError(status, spec.status_reason or f"{spec.cell_id} is not runnable")
    adapter = RAW_ADAPTERS.get(spec.adapter_id)
    if adapter is None:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: no target-free adapter for {spec.adapter_id}",
        )
    try:
        return adapter(spec, sources)
    except ExternalContractError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: telemetry cannot reconstruct the frozen raw 30-feature contract: "
            f"{type(error).__name__}: {error}",
        ) from error


def apply_mixed_v2_once(raw: RawFeatureCell) -> tuple[np.ndarray, Mapping[str, Any]]:
    if raw.preprocessing_steps:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{raw.spec.cell_id}: preprocessing already present; double transform forbidden",
        )
    transformed, names, details = dufs_liu_mixed_v2_matrix(
        raw.raw_matrix, raw.feature_names
    )
    if tuple(names) != raw.feature_names or transformed.shape != raw.raw_matrix.shape:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{raw.spec.cell_id}: mixed-v2 changed the frozen present-feature roster",
        )
    if not np.isfinite(transformed).all():
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{raw.spec.cell_id}: mixed-v2 produced non-finite values",
        )
    return np.asarray(transformed, dtype=np.float64), details


def _row_signature(row_ids: Sequence[str], group_ids: Sequence[str]) -> str:
    return _payload_hash({"row_ids": list(row_ids), "group_ids": list(group_ids)})


def prepare_external_cell(
    *,
    registry: ExternalRegistry,
    spec: ExternalCellSpec,
    repo: str | Path,
    output_path: str | Path,
) -> Mapping[str, Any]:
    sources = resolve_sources(registry, spec, repo=repo)
    verified = verify_sources(sources, include_labels=False)
    raw = load_raw_feature_cell(spec, sources)
    matrix, transform_details = apply_mixed_v2_once(raw)
    matrix_hash = prepared_matrix_sha256(matrix, raw.feature_names, raw.row_ids)
    # PreparedCell validation is the executable gate shared by all 13 methods.
    PreparedCell(
        population_id=spec.population_id,
        cell_id=spec.cell_id,
        domain=spec.domain,
        matrix=matrix,
        feature_names=raw.feature_names,
        row_ids=raw.row_ids,
        feature_contract=CONTRACT_VERSION,
        preprocessing_steps=(CONTRACT_VERSION,),
        preprocessed=True,
        declared_matrix_sha256=matrix_hash,
    )
    target = Path(output_path)
    arrays = {
        "X_confidence": matrix.astype("<f8", copy=False),
        "feature_names": np.asarray(raw.feature_names, dtype="<U64"),
        "family_ids": np.asarray([FEATURE_TO_VIEW[name] for name in raw.feature_names], dtype="<U32"),
        "row_ids": np.asarray(raw.row_ids, dtype="<U256"),
        "group_ids": np.asarray(raw.group_ids, dtype="<U256"),
        "row_index": np.arange(len(raw.row_ids), dtype="<i8"),
    }
    forbidden = [
        name for name in arrays
        if any(fragment in name.lower() for fragment in TARGET_LIKE_FRAGMENTS)
    ]
    if forbidden:
        raise RuntimeError(f"target-like members in prepared artifact: {forbidden}")
    artifact_sha = atomic_write_npz(target, arrays)
    return {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": ReadinessStatus.ELIGIBLE.value,
        "cell_id": spec.cell_id,
        "population_id": spec.population_id,
        "dataset_id": spec.dataset_id,
        "model_id": spec.model_id,
        "slice_id": spec.slice_id,
        "domain": spec.domain,
        "comparison_group_id": spec.comparison_group_id,
        "panel_role": spec.panel_role,
        "adapter_id": spec.adapter_id,
        "n_rows": len(raw.row_ids),
        "n_features": len(raw.feature_names),
        "feature_names": list(raw.feature_names),
        "present_feature_roster_sha256": _payload_hash(list(raw.feature_names)),
        "nominal_feature_count": len(CANONICAL_FEATURE_NAMES),
        "nominal_feature_roster_sha256": _payload_hash(list(CANONICAL_FEATURE_NAMES)),
        "absent_feature_names": [
            name for name in CANONICAL_FEATURE_NAMES if name not in set(raw.feature_names)
        ],
        "feature_contract_id": CONTRACT_VERSION,
        "preprocessing_steps": [CONTRACT_VERSION],
        "mixed_v2_applied_count": 1,
        "matrix_semantics": "higher_is_confidence",
        "prepared_matrix_sha256": matrix_hash,
        "row_signature_sha256": _row_signature(raw.row_ids, raw.group_ids),
        "group_count": len(set(raw.group_ids)),
        "artifact_path": target.name,
        "artifact_sha256": artifact_sha,
        "source_files": verified,
        "source_inventory_bindings": list(sources.inventory_bindings),
        "transform_details": transform_details,
        "labels_opened": False,
        "historical_scores_opened": False,
    }


def load_prepared_external_cell(
    *,
    artifact_path: str | Path,
    record: Mapping[str, Any],
) -> tuple[PreparedCell, tuple[str, ...]]:
    path = Path(artifact_path)
    if sha256_file(path) != str(record["artifact_sha256"]):
        raise RuntimeError(f"prepared artifact hash mismatch: {path}")
    arrays = load_npz_no_pickle(path)
    allowed = {"X_confidence", "feature_names", "family_ids", "row_ids", "group_ids", "row_index"}
    if set(arrays) != allowed:
        raise RuntimeError(f"unexpected prepared arrays: {sorted(set(arrays) ^ allowed)}")
    if any(
        any(fragment in name.lower() for fragment in TARGET_LIKE_FRAGMENTS)
        for name in arrays
    ):
        raise RuntimeError("target-like array crossed the fitting boundary")
    names = tuple(map(str, arrays["feature_names"].tolist()))
    expected_names = tuple(name for name in CANONICAL_FEATURE_NAMES if name in set(names))
    if names != expected_names or names != tuple(record.get("feature_names", ())):
        raise RuntimeError("prepared external present-feature roster/order drifted")
    if _payload_hash(list(names)) != record.get("present_feature_roster_sha256"):
        raise RuntimeError("prepared external present-feature roster hash drifted")
    rows = tuple(map(str, arrays["row_ids"].tolist()))
    groups = tuple(map(str, arrays["group_ids"].tolist()))
    matrix = np.asarray(arrays["X_confidence"], dtype=np.float64)
    observed = prepared_matrix_sha256(matrix, names, rows)
    if observed != record.get("prepared_matrix_sha256"):
        raise RuntimeError("prepared matrix/row hash mismatch")
    if _row_signature(rows, groups) != record.get("row_signature_sha256"):
        raise RuntimeError("prepared row/group signature mismatch")
    cell = PreparedCell(
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
    return cell, groups


def audit_external_registry(
    *,
    registry: ExternalRegistry,
    repo: str | Path,
    deep: bool = False,
) -> Mapping[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for spec in registry.cells:
        base = {
            "cell_id": spec.cell_id,
            "population_id": spec.population_id,
            "dataset_id": spec.dataset_id,
            "model_id": spec.model_id,
            "slice_id": spec.slice_id,
            "comparison_group_id": spec.comparison_group_id,
            "panel_role": spec.panel_role,
            "fit_policy": spec.fit_policy,
            "expected_rows": spec.expected_rows,
        }
        if spec.fit_policy == "forbidden":
            rows.append({
                **base,
                "status": str(spec.configured_status),
                "reason": spec.status_reason,
                "fit_allowed": False,
                "sources_verified": False,
                "label_assets_hash_verified_without_parsing": False,
                "feature_contract_verified": False,
            })
            continue
        try:
            sources = resolve_sources(registry, spec, repo=repo)
            # Hash label ledgers for readiness without parsing their contents.
            # The label vector and even its class counts remain unopened until
            # the score-freeze gate in ``load_labels_after_score_freeze``.
            files = verify_sources(sources, include_labels=True)
            result: dict[str, Any] = {
                **base,
                "status": ReadinessStatus.READY_FOR_TARGET_FREE_PREPARATION.value,
                "reason": None,
                "fit_allowed": True,
                "sources_verified": True,
                "label_assets_hash_verified_without_parsing": True,
                "feature_contract_verified": False,
                "verified_files": files,
                "inventory_bindings": list(sources.inventory_bindings),
            }
            if deep:
                raw = load_raw_feature_cell(spec, sources)
                matrix, _ = apply_mixed_v2_once(raw)
                result.update({
                    "status": ReadinessStatus.ELIGIBLE.value,
                    "feature_contract_verified": True,
                    "n_rows": len(raw.row_ids),
                    "n_features": matrix.shape[1],
                    "row_signature_sha256": _row_signature(raw.row_ids, raw.group_ids),
                })
            rows.append(result)
        except ExternalContractError as error:
            rows.append({
                **base,
                "status": error.status.value,
                "reason": str(error),
                "fit_allowed": False,
                "sources_verified": error.status not in {
                    ReadinessStatus.BLOCKED_ASSET,
                    ReadinessStatus.SOURCE_HASH_MISMATCH,
                    ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                },
                "label_assets_hash_verified_without_parsing": False,
                "feature_contract_verified": False,
            })
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "deep_feature_audit": bool(deep),
        "registry_path": registry.path.as_posix(),
        "registry_sha256": registry.sha256,
        "population_registry_path": registry.population_registry_path.as_posix(),
        "population_registry_sha256": registry.population_registry_sha256,
        "source_root": str(Path(repo).resolve()),
        "feature_contract_id": CONTRACT_VERSION,
        "row_dropping_or_imputation_forbidden": True,
        "partial_feature_availability_forbidden": True,
        "uniformly_unavailable_nominal_views_may_remain_absent": True,
        "historical_scores_opened": False,
        "labels_opened": False,
        "cells": rows,
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({str(row["status"]) for row in rows})
        },
    }
    manifest["payload_sha256"] = _payload_hash(manifest)
    return manifest


def _label_rows_processbench(spec: ExternalCellSpec, sources: ResolvedSources):
    cache = _load_pickle(sources.feature_files[0].path)
    rows, groups, labels = [], [], []
    for key in sorted(cache, key=_stable_key):
        row = cache[key]
        alignment = row.get("align_diag", {})
        problems = alignment.get("problems")
        if problems or alignment.get("ok") is False:
            raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "ProcessBench label roster contains misaligned row")
        official_id = row.get("id")
        if not isinstance(official_id, str) or not official_id:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"ProcessBench label source row {key} lacks its official string ID",
            )
        row_id = official_id
        if "final_answer_correct" not in row:
            raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"{row_id}: final_answer_correct is absent")
        rows.append(row_id)
        groups.append(row_id)
        labels.append(1 - int(bool(row["final_answer_correct"])))
    return rows, groups, labels, {"label_rule": "1 - bool(final_answer_correct)", "source": "ProcessBench fixed response"}


def _label_rows_prmbench(spec: ExternalCellSpec, sources: ResolvedSources):
    cache = _load_pickle(sources.feature_files[0].path)
    excluded = set(spec.excluded_row_ids)
    rows, groups, labels = [], [], []
    for key in sorted(cache, key=_stable_key):
        row = cache[key]
        row_id = str(row.get("idx", ""))
        if not row_id:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"PRMBench label source row {key} lacks idx",
            )
        if row_id in excluded:
            continue
        classification = str(row.get("classification", ""))
        if not classification:
            raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"{row_id}: classification is absent")
        rows.append(row_id)
        source_group = str(row.get("source_idx", ""))
        if not source_group:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"PRMBench label row {row_id} lacks source_idx",
            )
        groups.append(source_group)
        labels.append(int(classification != "correct"))
    return rows, groups, labels, {"label_rule": "classification != correct", "excluded_row_ids": sorted(excluded)}


def _label_rows_repgrid(spec: ExternalCellSpec, sources: ResolvedSources):
    data = _load_pickle(sources.feature_files[0].path)
    rows, groups, labels = [], [], []
    for key in sorted(data, key=_stable_key):
        candidates = data[key]["candidates"]
        for index, candidate in enumerate(candidates):
            if "label" not in candidate:
                raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"{key}/{index}: embedded label absent")
            group = str(key)
            rows.append(f"{group}:candidate:{index:04d}")
            groups.append(group)
            labels.append(1 - int(bool(candidate["label"])))
    return rows, groups, labels, {"label_rule": "1 - bool(candidate.label)", "source": "frozen embedded grader output"}


def _label_rows_hle(spec: ExternalCellSpec, sources: ResolvedSources):
    if len(sources.label_files) != 1:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "HLE needs one separate judge ledger")
    judged: dict[str, int] = {}
    with sources.label_files[0].path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            value = str(record.get("correct", "")).strip().lower()
            if value not in {"yes", "no"}:
                raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"invalid HLE judge value {value!r}")
            key = str(record["row_key"])
            if key in judged:
                raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"duplicate HLE judge key {key}")
            judged[key] = int(value == "no")
    raw = _load_pickle(sources.feature_files[0].path)
    rows, groups, labels = [], [], []
    for key in sorted(raw, key=_stable_key):
        candidates = raw[key]["candidates"]
        if len(candidates) != 1:
            raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "HLE must have one response per question")
        group = str(key)
        if group not in judged:
            raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"missing HLE judge key {group}")
        rows.append(f"{group}:candidate:0000")
        groups.append(group)
        labels.append(judged[group])
    if set(judged) != set(groups):
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "HLE judge/source rosters differ")
    return rows, groups, labels, {
        "label_rule": "Codex interim judge correct=yes -> incorrect=0; correct=no -> incorrect=1",
        "paper_faithful_grader": False,
        "judge_file": sources.label_files[0].relative_path,
    }


def _label_rows_aqua(spec: ExternalCellSpec, sources: ResolvedSources):
    records: list[Mapping[str, Any]] = []
    for item in sorted(sources.feature_files, key=lambda value: value.relative_path):
        records.extend(_load_pickle(item.path))
    selected = [row for row in records if row.get("arm") == "cot" and row.get("setting_label") == "central"]
    selected.sort(key=lambda row: _stable_key(str(row.get("question_id", "")).split(":")[-1]))
    rows, labels = [], []
    for row in selected:
        row_id = str(row.get("question_id", ""))
        graded = grade_aqua_option(row.get("answer_text"), str(row.get("gold_answer", "")))
        rows.append(row_id)
        labels.append(1 - int(bool(graded["correct"])))
    return rows, rows, labels, {"label_rule": "fair_aqua_option_parser_v1.0.0", "derived_after_score_freeze": True}


LABEL_ADAPTERS = {
    "processbench_teacher_forced_v1": _label_rows_processbench,
    "prmbench_teacher_forced_v1": _label_rows_prmbench,
    "repgrid_embedded_label_v1": _label_rows_repgrid,
    "hle_separate_judge_v1": _label_rows_hle,
    "aqua_paper_exact_cot_v1": _label_rows_aqua,
}


def assert_score_freeze(
    freeze: Mapping[str, Any],
    *,
    registry: ExternalRegistry,
) -> None:
    if freeze.get("schema_version") != SCORE_FREEZE_SCHEMA_VERSION:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "external score freeze schema is missing")
    if freeze.get("all_expected_scores_present") is not True:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "score freeze is incomplete")
    if freeze.get("labels_opened_by_fit") is not False or freeze.get("runtime_labels_used") is not False:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "fit freeze does not prove label isolation")
    if freeze.get("external_registry_sha256") != registry.sha256:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "score freeze binds another external registry")
    payload = dict(freeze)
    recorded = payload.pop("payload_sha256", None)
    if recorded != _payload_hash(payload):
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "score freeze payload hash failed")


def load_labels_after_score_freeze(
    *,
    registry: ExternalRegistry,
    spec: ExternalCellSpec,
    repo: str | Path,
    score_freeze: Mapping[str, Any],
    expected_row_ids: Sequence[str],
    expected_group_ids: Sequence[str],
) -> LabelVector:
    assert_score_freeze(score_freeze, registry=registry)
    sources = resolve_sources(registry, spec, repo=repo)
    verified = verify_sources(sources, include_labels=True)
    adapter = LABEL_ADAPTERS.get(spec.adapter_id)
    if adapter is None:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"no label adapter for {spec.adapter_id}")
    try:
        rows, groups, labels, provenance = adapter(spec, sources)
    except ExternalContractError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ExternalContractError(
            ReadinessStatus.LABEL_PROVENANCE_BLOCKED,
            f"{spec.cell_id}: label adapter failed closed: {type(error).__name__}: {error}",
        ) from error
    rows = tuple(map(str, rows))
    groups = tuple(map(str, groups))
    if rows != tuple(map(str, expected_row_ids)) or groups != tuple(map(str, expected_group_ids)):
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: post-freeze label roster/order differs from the frozen score cohort",
        )
    values = np.asarray(labels, dtype=np.int8)
    incorrect = int(values.sum())
    correct = int(len(values) - incorrect)
    if spec.expected_incorrect is not None and incorrect != spec.expected_incorrect:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"{spec.cell_id}: expected {spec.expected_incorrect} incorrect, got {incorrect}")
    if spec.expected_correct is not None and correct != spec.expected_correct:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"{spec.cell_id}: expected {spec.expected_correct} correct, got {correct}")
    label_provenance = {
        **dict(provenance),
        "verified_source_files": verified,
        "row_label_sha256": _payload_hash({"row_ids": list(rows), "incorrect": values.tolist()}),
        "positive_class": "incorrect",
        "n_incorrect": incorrect,
        "n_correct": correct,
        "score_freeze_payload_sha256": score_freeze["payload_sha256"],
    }
    return LabelVector(spec.cell_id, rows, groups, values, label_provenance)


def write_label_vector(path: str | Path, value: LabelVector) -> Mapping[str, Any]:
    target = Path(path)
    artifact_sha = atomic_write_npz(target, {
        "row_ids": np.asarray(value.row_ids, dtype="<U256"),
        "group_ids": np.asarray(value.group_ids, dtype="<U256"),
        "incorrect": np.asarray(value.incorrect, dtype="i1"),
    })
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "cell_id": value.cell_id,
        "n_rows": len(value.row_ids),
        "artifact_path": target.name,
        "artifact_sha256": artifact_sha,
        "provenance": dict(value.provenance),
    }


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CANONICAL_FEATURE_NAMES",
    "ExternalCellSpec",
    "ExternalContractError",
    "ExternalRegistry",
    "LABEL_SCHEMA_VERSION",
    "LabelVector",
    "PREPARED_SCHEMA_VERSION",
    "RAW_ADAPTERS",
    "ReadinessStatus",
    "ResolvedSources",
    "SCORE_FREEZE_SCHEMA_VERSION",
    "SourceFile",
    "apply_mixed_v2_once",
    "assert_score_freeze",
    "audit_external_registry",
    "load_external_registry",
    "load_labels_after_score_freeze",
    "load_prepared_external_cell",
    "load_raw_feature_cell",
    "prepare_external_cell",
    "resolve_sources",
    "verify_source_file",
    "verify_sources",
    "write_label_vector",
]
