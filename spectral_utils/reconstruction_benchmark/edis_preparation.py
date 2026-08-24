"""Target-free preparation for the frozen EDIS/AIME reconstruction lane.

This module is deliberately preparation-only.  It reads the pinned telemetry
pickles, extracts the nominal 30 label-free views, applies mixed-v2 exactly
once, and emits a fit-safe registry.  The fit-safe registry and NPZ files do
not contain raw source paths, class counts, labels, source-question IDs, group
IDs, group membership, or group multiplicities.

Opaque identities are supplied by the shared keyed-HMAC controller.  Keeping
that controller behind a tiny protocol lets this module reuse the repository's
sealed-key implementation without importing any post-freeze label adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..dufs_liu_feature_contract import CONTRACT_VERSION, dufs_liu_mixed_v2_matrix
from ..repgrid_scoring import _candidate_features, logprob_features_extended
from ..specrage_views import FEATURE_TO_VIEW
from .contracts import PreparedCell, prepared_matrix_sha256
from .io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


REGISTRY_SCHEMA = "reconstruction-edis-target-free-registry-v1"
FIT_REGISTRY_SCHEMA = "reconstruction-edis-fit-safe-registry-v1"
PRIVATE_PROVENANCE_SCHEMA = "reconstruction-edis-private-preparation-v1"
PREPARATION_SOURCE_PATHS = (
    "configs/reconstruction_benchmark_v1/edis_target_free.json",
    "configs/reconstruction_benchmark_v1/feature_contract.json",
    "scripts/reconstruction_benchmark/prepare_edis.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/feature_utils.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/repgrid_scoring.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/streaming_utils.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/edis_identity.py",
    "spectral_utils/reconstruction_benchmark/edis_preparation.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/io.py",
)
NOMINAL_FEATURE_NAMES = tuple(FEATURE_TO_VIEW)
_OPAQUE_ROW = re.compile(r"^xridv2_[0-9a-f]{64}$")
_OPAQUE_GROUP = re.compile(r"^xgidv2_[0-9a-f]{64}$")
_TARGET_FRAGMENTS = ("label", "target", "correct", "class", "gold", "answer_key")
_FIT_FORBIDDEN_KEYS = {
    "source", "source_path", "raw_path", "manifest", "questions",
    "samples_per_question_temperature", "group_ids", "group_count",
    "group_membership", "expected_correct", "expected_incorrect",
    "gate_status", "gate_reasons",
}
_SAFE_ATTESTATION_KEYS = {
    "labels_opened", "historical_scores_opened", "raw_sources_serialized",
}


class KeyedIdentityController(Protocol):
    """Narrow interface implemented by the shared sealed-key identity code."""

    @property
    def public_binding(self) -> Mapping[str, Any]: ...

    def row_id(self, *, namespace: Mapping[str, str], raw_identity: str) -> str: ...

    def group_id(self, *, namespace: Mapping[str, str], raw_identity: str) -> str: ...


@dataclass(frozen=True)
class EdisCellSpec:
    lane_id: str
    dataset_id: str
    population_id: str
    population_kind: str
    model_id: str
    temperature: float
    cell_id: str
    expected_rows: int
    expected_questions: int
    candidates_per_question: int
    source_path: str
    source_sha256: str
    source_size_bytes: int
    manifest_path: str
    manifest_sha256: str
    manifest_size_bytes: int


@dataclass(frozen=True)
class EdisPreparationRegistry:
    path: Path
    sha256: str
    lane_id: str
    raw: Mapping[str, Any]
    cells: tuple[EdisCellSpec, ...]

    @property
    def by_cell(self) -> dict[str, EdisCellSpec]:
        return {cell.cell_id: cell for cell in self.cells}


def _payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _safe_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes source root: {relative!r}") from error
    return path


def load_preparation_registry(path: str | Path) -> EdisPreparationRegistry:
    target = Path(path).resolve()
    raw = json.loads(target.read_text(encoding="utf-8"))
    if raw.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("unexpected EDIS target-free registry schema")
    if raw.get("feature_contract_id") != CONTRACT_VERSION:
        raise ValueError("EDIS registry does not bind the frozen mixed-v2 contract")
    if int(raw.get("nominal_feature_count", -1)) != len(NOMINAL_FEATURE_NAMES):
        raise ValueError("EDIS nominal feature count drifted")
    contract = raw.get("fit_contract", {})
    required_contract = {
        "method_roster": "all_13_primary_methods",
        "labels_available_to_fit": False,
        "class_counts_available_to_fit": False,
        "raw_source_paths_available_to_fit": False,
        "historical_scores_available_to_fit": False,
        "donors_available_to_fit": False,
        "score_semantics": "higher_is_incorrect",
        "mixed_v2_application_count": 1,
        "score_freeze_required_before_labels": True,
    }
    if contract != required_contract:
        raise ValueError("EDIS fit contract drifted")
    if raw.get("identity_contract", {}).get("required") != "keyed_hmac_with_release_sealed_key":
        raise ValueError("EDIS registry requires the shared keyed-HMAC identity contract")

    cells: list[EdisCellSpec] = []
    for dataset in raw.get("datasets", ()):
        required_dataset = {
            "dataset_id", "population_id", "population_kind", "model_id",
            "questions", "samples_per_question_temperature", "manifest", "cells",
        }
        if not required_dataset.issubset(dataset):
            raise ValueError("EDIS dataset registry row is incomplete")
        manifest = dataset["manifest"]
        for item in dataset["cells"]:
            cells.append(EdisCellSpec(
                lane_id=str(raw["lane_id"]),
                dataset_id=str(dataset["dataset_id"]),
                population_id=str(dataset["population_id"]),
                population_kind=str(dataset["population_kind"]),
                model_id=str(dataset["model_id"]),
                temperature=float(item["temperature"]),
                cell_id=str(item["cell_id"]),
                expected_rows=int(item["expected_rows"]),
                expected_questions=int(dataset["questions"]),
                candidates_per_question=int(dataset["samples_per_question_temperature"]),
                source_path=str(item["source"]["path"]),
                source_sha256=str(item["source"]["sha256"]),
                source_size_bytes=int(item["source"]["size_bytes"]),
                manifest_path=str(manifest["path"]),
                manifest_sha256=str(manifest["sha256"]),
                manifest_size_bytes=int(manifest["size_bytes"]),
            ))
    if len(cells) != 12 or len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("EDIS registry must contain exactly 12 unique dataset-temperature cells")
    datasets = {cell.dataset_id for cell in cells}
    if datasets != {"aime24", "amc23", "gsm8k", "math500"}:
        raise ValueError("EDIS dataset roster drifted")
    for dataset_id in datasets:
        temperatures = sorted(cell.temperature for cell in cells if cell.dataset_id == dataset_id)
        if temperatures != [0.2, 0.6, 1.0]:
            raise ValueError(f"{dataset_id}: temperature roster drifted")
    return EdisPreparationRegistry(
        path=target,
        sha256=sha256_file(target),
        lane_id=str(raw["lane_id"]),
        raw=raw,
        cells=tuple(cells),
    )


def verify_pinned_file(
    *, root: str | Path, relative: str, expected_sha256: str, expected_size: int
) -> Mapping[str, Any]:
    source_root = Path(root).resolve()
    path = _safe_under(source_root, relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = path.stat().st_size
    if observed_size != int(expected_size):
        raise RuntimeError(f"source size mismatch for {relative}")
    observed_sha = sha256_file(path)
    if observed_sha != str(expected_sha256):
        raise RuntimeError(f"source hash mismatch for {relative}")
    return {"path": relative, "sha256": observed_sha, "size_bytes": observed_size}


def _stable_problem_key(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("boolean source-question keys are forbidden")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"source-question key is not integer-like: {type(value).__name__}") from error
    if str(value) != str(integer):
        raise ValueError(f"noncanonical source-question key: {value!r}")
    return integer


def _telemetry_features(candidate: Mapping[str, Any]) -> Mapping[str, float]:
    """Extract only telemetry fields; target-like members are never requested."""

    telemetry = {
        "token_entropies": candidate.get("token_entropies"),
        "token_spilled_energies": candidate.get("token_spilled_energies"),
        "token_logsumexp": candidate.get("token_logsumexp"),
        "top_k_logprobs": candidate.get("top_k_logprobs"),
    }
    features = dict(_candidate_features(telemetry, allow_short=False))
    if telemetry["top_k_logprobs"] is not None:
        features.update(logprob_features_extended(telemetry["top_k_logprobs"]))
    return features


def audit_nominal_matrix(
    rows: Sequence[Mapping[str, float]], *, expected_rows: int, cell_id: str
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if len(rows) != int(expected_rows):
        raise RuntimeError(f"{cell_id}: expected {expected_rows} rows, found {len(rows)}")
    nominal = np.asarray([
        [row.get(name, np.nan) for name in NOMINAL_FEATURE_NAMES]
        for row in rows
    ], dtype=np.float64)
    if nominal.shape != (expected_rows, len(NOMINAL_FEATURE_NAMES)):
        raise RuntimeError(f"{cell_id}: nominal 30-view matrix shape failed")
    finite = np.sum(np.isfinite(nominal), axis=0)
    partial = {
        name: int(count)
        for name, count in zip(NOMINAL_FEATURE_NAMES, finite)
        if 0 < int(count) < expected_rows
    }
    if partial:
        raise RuntimeError(
            f"{cell_id}: partially available features are forbidden: {partial}"
        )
    keep = finite == expected_rows
    names = tuple(name for name, present in zip(NOMINAL_FEATURE_NAMES, keep) if bool(present))
    absent = tuple(name for name, present in zip(NOMINAL_FEATURE_NAMES, keep) if not bool(present))
    if len(names) < 3:
        raise RuntimeError(f"{cell_id}: fewer than three fully present views")
    matrix = nominal[:, keep]
    if not np.isfinite(matrix).all():
        raise AssertionError("present-view audit admitted a non-finite value")
    return matrix, names, absent


def _question_fingerprint(dataset_id: str, question: int, text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            f"{dataset_id}: source question {question} lacks saved question text"
        )
    return _payload_sha256({
        "dataset_id": dataset_id,
        "source_question": int(question),
        "question_text": text,
    })


def _raw_identity(
    dataset_id: str,
    temperature: float,
    question: int,
    question_fingerprint: str,
    candidate: int,
) -> str:
    return canonical_json_bytes({
        "dataset_id": dataset_id,
        "temperature": float(temperature),
        "source_question": int(question),
        "question_fingerprint": question_fingerprint,
        "candidate_index": int(candidate),
    }).decode("ascii")


def _raw_group_identity(dataset_id: str, question_fingerprint: str) -> str:
    return canonical_json_bytes({
        "dataset_id": dataset_id,
        "question_fingerprint": question_fingerprint,
    }).decode("ascii")


def extract_target_free_cell(
    *,
    spec: EdisCellSpec,
    source_path: str | Path,
    identity: KeyedIdentityController,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...], Mapping[str, Any]]:
    """Return raw features plus keyed IDs; labels are never read or returned."""

    with Path(source_path).open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, Mapping):
        raise TypeError(f"{spec.cell_id}: raw pickle is not a mapping")
    keyed_rows: list[str] = []
    keyed_groups: list[str] = []
    question_fingerprints: list[str] = []
    feature_rows: list[Mapping[str, float]] = []
    problem_keys = sorted((_stable_problem_key(key), key) for key in data)
    canonical_questions = [integer for integer, _ in problem_keys]
    if canonical_questions != list(range(spec.expected_questions)):
        raise RuntimeError(
            f"{spec.cell_id}: expected canonical question keys 0..{spec.expected_questions - 1}"
        )
    row_namespace = {
        "lane_id": spec.lane_id,
        "scope": "dataset_temperature_cell",
        "cell_id": spec.cell_id,
    }
    group_namespace = {
        "lane_id": spec.lane_id,
        "scope": "dataset_across_temperatures",
        "dataset_id": spec.dataset_id,
    }
    for problem, original_key in problem_keys:
        entry = data[original_key]
        if not isinstance(entry, Mapping):
            raise TypeError(f"{spec.cell_id}: source-question entry is not a mapping")
        candidates = entry.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise TypeError(f"{spec.cell_id}: source-question entry lacks candidates")
        if len(candidates) != spec.candidates_per_question:
            raise RuntimeError(
                f"{spec.cell_id}: question {problem} has {len(candidates)} candidates; "
                f"expected {spec.candidates_per_question}"
            )
        question_fingerprint = _question_fingerprint(
            spec.dataset_id, problem, entry.get("question")
        )
        question_fingerprints.append(question_fingerprint)
        group_id = identity.group_id(
            namespace=group_namespace,
            raw_identity=_raw_group_identity(spec.dataset_id, question_fingerprint),
        )
        if _OPAQUE_GROUP.fullmatch(group_id) is None:
            raise RuntimeError("shared identity controller returned a malformed group ID")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise TypeError(f"{spec.cell_id}: candidate is not a mapping")
            row_id = identity.row_id(
                namespace=row_namespace,
                raw_identity=_raw_identity(
                    spec.dataset_id,
                    spec.temperature,
                    problem,
                    question_fingerprint,
                    candidate_index,
                ),
            )
            if _OPAQUE_ROW.fullmatch(row_id) is None:
                raise RuntimeError("shared identity controller returned a malformed row ID")
            keyed_rows.append(row_id)
            keyed_groups.append(group_id)
            feature_rows.append(_telemetry_features(candidate))
    if len(set(keyed_rows)) != len(keyed_rows):
        raise RuntimeError(f"{spec.cell_id}: keyed row IDs are not unique")
    raw_matrix, names, absent = audit_nominal_matrix(
        feature_rows, expected_rows=spec.expected_rows, cell_id=spec.cell_id
    )
    order = np.asarray(sorted(range(len(keyed_rows)), key=lambda index: keyed_rows[index]), dtype=np.int64)
    ordered_rows = tuple(keyed_rows[index] for index in order.tolist())
    ordered_groups = tuple(keyed_groups[index] for index in order.tolist())
    ordered_matrix = np.asarray(raw_matrix[order], dtype=np.float64)
    if ordered_rows != tuple(sorted(ordered_rows)):
        raise AssertionError("keyed row canonicalization failed")
    private_group_commitment = _payload_sha256([
        {"row_id": row_id, "group_id": group_id}
        for row_id, group_id in zip(ordered_rows, ordered_groups)
    ])
    private = {
        "group_membership_commitment_sha256": private_group_commitment,
        "question_roster_commitment_sha256": _payload_sha256(
            question_fingerprints
        ),
        "row_roster_sha256": _payload_sha256(list(ordered_rows)),
    }
    return ordered_matrix, names, absent, ordered_rows, private


def _assert_fit_safe(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            lowered = name.lower()
            if (
                lowered not in _SAFE_ATTESTATION_KEYS
                and (
                    lowered in _FIT_FORBIDDEN_KEYS
                    or any(fragment in lowered for fragment in _TARGET_FRAGMENTS)
                )
            ):
                raise RuntimeError(f"fit-visible registry contains forbidden key {path}.{name}")
            _assert_fit_safe(child, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_fit_safe(child, path=f"{path}[{index}]")


def prepare_build(
    *,
    release_id: str,
    build_id: str,
    registry: EdisPreparationRegistry,
    identity: KeyedIdentityController,
    source_root: str | Path,
    release_root: str | Path,
    private_control_root: str | Path,
    preparation_source_snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Prepare one independent A/B build without opening row labels."""

    if build_id not in {"A", "B"}:
        raise ValueError("build_id must be A or B")
    public_root = Path(release_root) / release_id / f"build_{build_id}" / "edis" / "inputs"
    private_root = Path(private_control_root) / release_id / "edis" / f"build_{build_id}"
    if public_root.exists() and any(public_root.iterdir()):
        raise FileExistsError(f"EDIS input directory is not empty: {public_root}")
    if private_root.exists() and any(private_root.iterdir()):
        raise FileExistsError(f"EDIS private preparation directory is not empty: {private_root}")
    # Validate the complete pinned source roster before creating any release
    # directory.  A missing/tampered late cell therefore cannot leave behind a
    # plausible-looking partial input tree that blocks a clean retry.
    root = Path(source_root).resolve()
    verified: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    manifest_cache: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for spec in registry.cells:
        source = verify_pinned_file(
            root=root,
            relative=spec.source_path,
            expected_sha256=spec.source_sha256,
            expected_size=spec.source_size_bytes,
        )
        manifest_key = (
            spec.manifest_path,
            spec.manifest_sha256,
            spec.manifest_size_bytes,
        )
        manifest = manifest_cache.get(manifest_key)
        if manifest is None:
            manifest = verify_pinned_file(
                root=root,
                relative=spec.manifest_path,
                expected_sha256=spec.manifest_sha256,
                expected_size=spec.manifest_size_bytes,
            )
            manifest_cache[manifest_key] = manifest
        verified[spec.cell_id] = (source, manifest)
    (public_root / "cells").mkdir(parents=True, exist_ok=False)
    private_root.mkdir(parents=True, exist_ok=False)

    public_cells: list[dict[str, Any]] = []
    private_cells: list[dict[str, Any]] = []
    for spec in registry.cells:
        source, manifest = verified[spec.cell_id]
        raw, names, absent, row_ids, private = extract_target_free_cell(
            spec=spec,
            source_path=_safe_under(root, spec.source_path),
            identity=identity,
        )
        transformed, transformed_names, details = dufs_liu_mixed_v2_matrix(raw, names)
        if tuple(transformed_names) != names or transformed.shape != raw.shape:
            raise RuntimeError(f"{spec.cell_id}: mixed-v2 changed the present-view roster")
        matrix = np.asarray(transformed, dtype=np.float64)
        matrix_hash = prepared_matrix_sha256(matrix, names, row_ids)
        PreparedCell(
            population_id=spec.population_id,
            cell_id=spec.cell_id,
            domain="multi_sample_trace_detection",
            matrix=matrix,
            feature_names=names,
            row_ids=row_ids,
            preprocessing_steps=(CONTRACT_VERSION,),
            preprocessed=True,
            declared_matrix_sha256=matrix_hash,
        )
        binding = dict(identity.public_binding)
        artifact = public_root / "cells" / f"{spec.cell_id}.npz"
        artifact_sha = atomic_write_npz(artifact, {
            "X_confidence": matrix.astype("<f8", copy=False),
            "feature_names": np.asarray(names, dtype="<U64"),
            "family_ids": np.asarray([FEATURE_TO_VIEW[name] for name in names], dtype="<U32"),
            "row_ids": np.asarray(row_ids, dtype="<U80"),
            "row_index": np.arange(len(row_ids), dtype="<i8"),
            "identity_contract_version": np.asarray([str(binding["contract_version"])], dtype="<U64"),
            "identity_key_id": np.asarray([str(binding["key_id"])], dtype="<U80"),
        })
        public_cells.append({
            "cell_id": spec.cell_id,
            "population_id": spec.population_id,
            "dataset_id": spec.dataset_id,
            "model_id": spec.model_id,
            "temperature": spec.temperature,
            "n_rows": spec.expected_rows,
            "artifact_path": artifact.relative_to(public_root).as_posix(),
            "artifact_sha256": artifact_sha,
            "prepared_matrix_sha256": matrix_hash,
            "feature_names": list(names),
            "absent_feature_names": list(absent),
            "present_feature_roster_sha256": _payload_sha256(list(names)),
            "nominal_feature_roster_sha256": _payload_sha256(list(NOMINAL_FEATURE_NAMES)),
            "row_roster_sha256": private["row_roster_sha256"],
            "identity_contract": binding,
            "mixed_v2_applied_count": 1,
            "preprocessing_steps": [CONTRACT_VERSION],
        })
        private_cells.append({
            "cell_id": spec.cell_id,
            "source": source,
            "source_manifest": manifest,
            "group_membership_commitment_sha256": private["group_membership_commitment_sha256"],
            "question_roster_commitment_sha256": private["question_roster_commitment_sha256"],
            "row_roster_sha256": private["row_roster_sha256"],
            "transform_details": details,
        })

    for dataset_id in sorted({spec.dataset_id for spec in registry.cells}):
        commitments = {
            row["question_roster_commitment_sha256"]
            for row, spec in zip(private_cells, registry.cells)
            if spec.dataset_id == dataset_id
        }
        if len(commitments) != 1:
            raise RuntimeError(
                f"{dataset_id}: saved question content/order differs across temperatures"
            )

    private_identity_commitment = getattr(
        identity, "private_identity_commitment_sha256", None
    )
    private_identity_binding = getattr(identity, "private_identity_binding", None)
    if not isinstance(private_identity_commitment, str) or len(private_identity_commitment) != 64:
        raise RuntimeError("identity controller lacks its private-contract commitment")
    if not isinstance(private_identity_binding, Mapping):
        raise RuntimeError("identity controller lacks its private identity binding")
    public = {
        "schema_version": FIT_REGISTRY_SCHEMA,
        "release_id": release_id,
        "build_id": build_id,
        "lane_id": registry.lane_id,
        "scientific_full_build": True,
        "feature_contract_id": CONTRACT_VERSION,
        "nominal_feature_count": len(NOMINAL_FEATURE_NAMES),
        "method_roster": "all_13_primary_methods",
        "score_semantics": "higher_is_incorrect",
        "identity_contract": dict(identity.public_binding),
        "private_identity_contract_commitment_sha256": private_identity_commitment,
        "preparation_registry_sha256": registry.sha256,
        "preparation_source_snapshot_sha256": str(preparation_source_snapshot["snapshot_sha256"]),
        "mixed_v2_applied_exactly_once": True,
        "labels_opened": False,
        "historical_scores_opened": False,
        "raw_sources_serialized": False,
        "cells": public_cells,
    }
    _assert_fit_safe(public)
    public["payload_sha256"] = _payload_sha256(public)
    atomic_write_json(public_root / "FIT_REGISTRY.json", public)

    private_provenance = {
        "schema_version": PRIVATE_PROVENANCE_SCHEMA,
        "release_id": release_id,
        "build_id": build_id,
        "lane_id": registry.lane_id,
        "preparation_registry_path": str(registry.path),
        "preparation_registry_sha256": registry.sha256,
        "identity_contract": dict(identity.public_binding),
        "private_identity_contract": dict(private_identity_binding),
        "private_identity_contract_commitment_sha256": private_identity_commitment,
        "preparation_source_snapshot": dict(preparation_source_snapshot),
        "labels_opened": False,
        "historical_scores_opened": False,
        "cells": private_cells,
    }
    private_provenance["payload_sha256"] = _payload_sha256(private_provenance)
    atomic_write_json(private_root / "PREPARATION_PROVENANCE.json", private_provenance)
    return public


__all__ = [
    "EdisCellSpec",
    "EdisPreparationRegistry",
    "FIT_REGISTRY_SCHEMA",
    "KeyedIdentityController",
    "NOMINAL_FEATURE_NAMES",
    "PREPARATION_SOURCE_PATHS",
    "PRIVATE_PROVENANCE_SCHEMA",
    "REGISTRY_SCHEMA",
    "audit_nominal_matrix",
    "extract_target_free_cell",
    "load_preparation_registry",
    "prepare_build",
    "verify_pinned_file",
]
