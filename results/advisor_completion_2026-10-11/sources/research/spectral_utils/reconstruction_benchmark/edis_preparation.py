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
import shutil
import tempfile
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
PREPARATION_STATUS_SCHEMA = "reconstruction-edis-preparation-status-v1"
TRACE_STATUS_CONTRACT_ID = "edis-frozen-min-trace-status-v1-2026-08-24"
STATUS_ROSTER_CONTRACT_ID = "edis-materialized-status-roster-v1-2026-08-24"
MINIMUM_ENTROPY_TOKENS = 8
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


class PartialFeatureAvailabilityError(RuntimeError):
    def __init__(self, *, cell_id: str, partial: Mapping[str, int]):
        self.cell_id = cell_id
        self.partial = dict(partial)
        super().__init__(
            f"{cell_id}: partially available features are forbidden: {self.partial}"
        )


class EdisCellBlocked(RuntimeError):
    """Structured target-free cell failure; it never contains label values."""

    def __init__(
        self,
        *,
        cell_id: str,
        status: str,
        status_detail: str,
        public_audit: Mapping[str, Any],
        private_audit: Mapping[str, Any],
    ):
        self.cell_id = cell_id
        self.status = status
        self.status_detail = status_detail
        self.public_audit = dict(public_audit)
        self.private_audit = dict(private_audit)
        super().__init__(f"{cell_id}: {status}: {status_detail}")


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

    @property
    def expected_status_by_cell(self) -> dict[str, str] | None:
        contract = self.raw.get("target_free_status_roster_contract")
        if contract is None:
            return None
        ready = {str(cell_id): "READY" for cell_id in contract["ready_cell_ids"]}
        blocked = {
            str(row["cell_id"]): str(row["status"])
            for row in contract["blocked_cells"]
        }
        return {**ready, **blocked}


def _payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def preparation_status_commitment_sha256(value: Mapping[str, Any]) -> str:
    """Return the A/B-stable commitment to the complete public status roster."""

    canonical = {
        key: child
        for key, child in value.items()
        if key not in {
            "build_id",
            "payload_sha256",
            "status_commitment_sha256",
        }
    }
    return _payload_sha256(canonical)


def _safe_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes source root: {relative!r}") from error
    return path


def load_preparation_status(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = dict(raw)
    recorded = payload.pop("payload_sha256", None)
    if recorded != _payload_sha256(payload):
        raise RuntimeError("EDIS preparation-status payload hash failed")
    if raw.get("schema_version") != PREPARATION_STATUS_SCHEMA:
        raise RuntimeError("unexpected EDIS preparation-status schema")
    if raw.get("status_commitment_sha256") != preparation_status_commitment_sha256(raw):
        raise RuntimeError("EDIS preparation-status canonical commitment failed")
    rows = raw.get("cells")
    if not isinstance(rows, list) or len(rows) != 12:
        raise RuntimeError("EDIS preparation status must expose all 12 cells")
    ids = [str(row.get("cell_id", "")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != 12:
        raise RuntimeError("EDIS preparation-status cell roster is invalid")
    ready = sum(row.get("status") == "READY" for row in rows)
    blocked = len(rows) - ready
    roster_match = raw.get("status_roster_contract_match") is True
    expected_fit_available = roster_match and ready > 0
    expected_full = expected_fit_available and blocked == 0
    expected_partial = expected_fit_available and blocked > 0
    expected_status_only = not expected_fit_available
    if (
        int(raw.get("registered_cell_count", -1)) != 12
        or int(raw.get("ready_cell_count", -1)) != ready
        or int(raw.get("blocked_cell_count", -1)) != blocked
        or raw.get("scientific_full_build") is not expected_full
        or raw.get("partial_descriptive_build") is not expected_partial
        or raw.get("status_only_build") is not expected_status_only
        or raw.get("headline_eligible") is not False
        or raw.get("fit_registry_available") is not expected_fit_available
        or raw.get("row_exclusion_performed") is not False
        or raw.get("imputation_performed") is not False
        or raw.get("feature_substitution_performed") is not False
        or raw.get("trace_status_contract_id") != TRACE_STATUS_CONTRACT_ID
    ):
        raise RuntimeError("EDIS preparation-status accounting drifted")
    allowed = {
        "READY",
        "BLOCKED_TRACE_BELOW_FROZEN_MIN",
        "BLOCKED_MALFORMED_TELEMETRY",
        "BLOCKED_PARTIAL_FEATURE_AVAILABILITY",
        "BLOCKED_ASSET",
    }
    if any(row.get("status") not in allowed for row in rows):
        raise RuntimeError("EDIS preparation status contains an unknown cell status")
    if blocked and (
        raw.get("dataset_aggregate_status") != "BLOCKED_INCOMPLETE_CELL_ROSTER"
        or raw.get("task_aggregate_status") != "BLOCKED_INCOMPLETE_CELL_ROSTER"
    ):
        raise RuntimeError("partial EDIS preparation did not block aggregates")
    return raw


def assert_expected_preparation_status_roster(
    *, registry: EdisPreparationRegistry, status: Mapping[str, Any]
) -> None:
    """Fail closed on any drift from a preregistered target-free cell roster."""

    expected = registry.expected_status_by_cell
    if expected is None:
        return
    rows = status.get("cells")
    if not isinstance(rows, list):
        raise RuntimeError("EDIS preparation status roster is absent")
    observed = [
        (str(row.get("cell_id", "")), str(row.get("status", "")))
        for row in rows
    ]
    required = [
        (cell.cell_id, expected[cell.cell_id])
        for cell in registry.cells
    ]
    if observed != required:
        raise RuntimeError(
            "EDIS preparation status differs from the preregistered 4-ready/8-blocked roster"
        )


def load_preparation_registry(path: str | Path) -> EdisPreparationRegistry:
    target = Path(path).resolve()
    raw = json.loads(target.read_text(encoding="utf-8"))
    if raw.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("unexpected EDIS target-free registry schema")
    if raw.get("feature_contract_id") != CONTRACT_VERSION:
        raise ValueError("EDIS registry does not bind the frozen mixed-v2 contract")
    if int(raw.get("nominal_feature_count", -1)) != len(NOMINAL_FEATURE_NAMES):
        raise ValueError("EDIS nominal feature count drifted")
    trace_contract = raw.get("trace_status_contract")
    expected_trace_contract = {
        "contract_id": TRACE_STATUS_CONTRACT_ID,
        "registered_stage": (
            "target_free_preparation_remediation_before_any_scores_or_labels"
        ),
        "minimum_entropy_tokens": MINIMUM_ENTROPY_TOKENS,
        "short_trace_status": "BLOCKED_TRACE_BELOW_FROZEN_MIN",
        "short_trace_action": "block_entire_dataset_temperature_cell",
        "malformed_trace_status": "BLOCKED_MALFORMED_TELEMETRY",
        "partial_feature_status": "BLOCKED_PARTIAL_FEATURE_AVAILABILITY",
        "asset_status": "BLOCKED_ASSET",
        "row_exclusion_allowed": False,
        "imputation_allowed": False,
        "feature_substitution_allowed": False,
        "allow_short_override_allowed": False,
        "partial_release_policy": (
            "ready_cells_may_run_descriptive_only; every blocked cell remains "
            "an explicit status row; incomplete dataset and task aggregates are "
            "status-only"
        ),
        "labels_used": False,
    }
    if trace_contract != expected_trace_contract:
        raise ValueError("EDIS frozen trace-status contract drifted")
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
    status_roster = raw.get("target_free_status_roster_contract")
    if status_roster is not None:
        required_status_roster = {
            "contract_id": STATUS_ROSTER_CONTRACT_ID,
            "registered_stage": (
                "after_target_free_telemetry_audit_before_any_scores_or_labels"
            ),
            "labels_used": False,
            "blocked_labels_may_be_opened": False,
            "dataset_or_task_aggregates_allowed": False,
            "headline_or_publication_eligible": False,
        }
        for key, expected in required_status_roster.items():
            if status_roster.get(key) != expected:
                raise ValueError(f"EDIS target-free status-roster {key} drifted")
        ready_ids = [str(value) for value in status_roster.get("ready_cell_ids", ())]
        blocked_rows = status_roster.get("blocked_cells")
        if not isinstance(blocked_rows, list):
            raise ValueError("EDIS target-free blocked-cell roster is absent")
        blocked_ids = [str(row.get("cell_id", "")) for row in blocked_rows]
        if (
            len(ready_ids) != 4
            or len(blocked_ids) != 8
            or any(
                row.get("status") != "BLOCKED_TRACE_BELOW_FROZEN_MIN"
                for row in blocked_rows
            )
            or set(ready_ids) & set(blocked_ids)
            or set(ready_ids) | set(blocked_ids)
            != {cell.cell_id for cell in cells}
            or ready_ids
            != [cell.cell_id for cell in cells if cell.cell_id in set(ready_ids)]
            or blocked_ids
            != [cell.cell_id for cell in cells if cell.cell_id in set(blocked_ids)]
        ):
            raise ValueError("EDIS target-free 4-ready/8-blocked roster drifted")
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
        raise PartialFeatureAvailabilityError(cell_id=cell_id, partial=partial)
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


def _frozen_trace_length(candidate: Mapping[str, Any]) -> int:
    """Validate every feature-bearing trajectory without consulting any target."""

    entropies = candidate.get("token_entropies")
    try:
        entropy_array = np.asarray(entropies, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("token_entropies is not a numeric trajectory") from error
    if entropy_array.ndim != 1 or entropy_array.size == 0:
        raise ValueError("token_entropies must be a nonempty one-dimensional trajectory")
    if not np.isfinite(entropy_array).all():
        raise ValueError("token_entropies contains non-finite values")
    spilled = candidate.get("token_spilled_energies")
    if spilled is not None:
        try:
            spilled_array = np.asarray(spilled, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("token_spilled_energies is not numeric") from error
        if spilled_array.ndim != 1 or spilled_array.shape != entropy_array.shape:
            raise ValueError(
                "token_spilled_energies is not aligned to token_entropies"
            )
        if not np.isfinite(spilled_array).all():
            raise ValueError("token_spilled_energies contains non-finite values")
    logsumexp = candidate.get("token_logsumexp")
    if logsumexp is not None:
        try:
            logsumexp_array = np.asarray(logsumexp, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("token_logsumexp is not numeric") from error
        if logsumexp_array.ndim != 1 or logsumexp_array.shape != entropy_array.shape:
            raise ValueError("token_logsumexp is not aligned to token_entropies")
        if not np.isfinite(logsumexp_array).all():
            raise ValueError("token_logsumexp contains non-finite values")
    top_k = candidate.get("top_k_logprobs")
    if top_k is not None:
        if not isinstance(top_k, Mapping) or not {"ids", "logprobs"}.issubset(top_k):
            raise ValueError("top_k_logprobs must contain ids and logprobs")
        try:
            logprobs = np.asarray(top_k["logprobs"], dtype=np.float64)
            ids = np.asarray(top_k["ids"])
        except (TypeError, ValueError) as error:
            raise ValueError("top_k_logprobs arrays are not numeric") from error
        if (
            logprobs.ndim != 2
            or logprobs.shape[0] != entropy_array.size
            or logprobs.shape[1] < 2
        ):
            raise ValueError("top_k_logprobs has an invalid T-by-K shape")
        if ids.ndim != 2 or ids.shape != logprobs.shape:
            raise ValueError("top_k ids/logprobs shapes are not aligned")
        if not np.isfinite(logprobs).all():
            raise ValueError("top_k logprobs contain non-finite values")
        if not np.issubdtype(ids.dtype, np.integer):
            try:
                numeric_ids = np.asarray(ids, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError("top_k ids are not integer-like") from error
            if (
                not np.isfinite(numeric_ids).all()
                or not np.equal(numeric_ids, np.floor(numeric_ids)).all()
            ):
                raise ValueError("top_k ids are not integer-like")
    return int(entropy_array.size)


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
    short_rows: list[dict[str, Any]] = []
    malformed_rows: list[dict[str, Any]] = []
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
            try:
                trace_length = _frozen_trace_length(candidate)
            except ValueError as error:
                malformed_rows.append({
                    "row_id": row_id,
                    "source_question": problem,
                    "candidate_index": candidate_index,
                    "question_fingerprint": question_fingerprint,
                    "reason": str(error),
                })
                continue
            features = _telemetry_features(candidate)
            feature_rows.append(features)
            if trace_length < MINIMUM_ENTROPY_TOKENS:
                short_rows.append({
                    "row_id": row_id,
                    "source_question": problem,
                    "candidate_index": candidate_index,
                    "question_fingerprint": question_fingerprint,
                    "trace_length": trace_length,
                })
                continue
    if len(set(keyed_rows)) != len(keyed_rows):
        raise RuntimeError(f"{spec.cell_id}: keyed row IDs are not unique")
    order = np.asarray(sorted(range(len(keyed_rows)), key=lambda index: keyed_rows[index]), dtype=np.int64)
    ordered_rows = tuple(keyed_rows[index] for index in order.tolist())
    ordered_groups = tuple(keyed_groups[index] for index in order.tolist())
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
    if malformed_rows:
        raise EdisCellBlocked(
            cell_id=spec.cell_id,
            status="BLOCKED_MALFORMED_TELEMETRY",
            status_detail=(
                f"{len(malformed_rows)} row(s) have malformed target-free telemetry"
            ),
            public_audit={
                "source_row_count": len(keyed_rows),
                "blocking_row_count": len(malformed_rows),
                "row_roster_sha256": private["row_roster_sha256"],
            },
            private_audit={**private, "blocking_rows": malformed_rows},
        )
    if short_rows:
        lengths = [int(row["trace_length"]) for row in short_rows]
        finite_counts = {
            name: int(sum(np.isfinite(row.get(name, np.nan)) for row in feature_rows))
            for name in NOMINAL_FEATURE_NAMES
        }
        raise EdisCellBlocked(
            cell_id=spec.cell_id,
            status="BLOCKED_TRACE_BELOW_FROZEN_MIN",
            status_detail=(
                f"{len(short_rows)} valid trace(s) contain fewer than "
                f"{MINIMUM_ENTROPY_TOKENS} tokens; no rows or features were dropped"
            ),
            public_audit={
                "source_row_count": len(keyed_rows),
                "blocking_row_count": len(short_rows),
                "minimum_observed_trace_tokens": min(lengths),
                "maximum_blocking_trace_tokens": max(lengths),
                "frozen_minimum_trace_tokens": MINIMUM_ENTROPY_TOKENS,
                "nominal_feature_finite_counts": finite_counts,
                "opaque_blocking_row_ids": sorted(
                    str(row["row_id"]) for row in short_rows
                ),
                "row_roster_sha256": private["row_roster_sha256"],
            },
            private_audit={**private, "blocking_rows": short_rows},
        )
    try:
        raw_matrix, names, absent = audit_nominal_matrix(
            feature_rows, expected_rows=spec.expected_rows, cell_id=spec.cell_id
        )
    except PartialFeatureAvailabilityError as error:
        raise EdisCellBlocked(
            cell_id=spec.cell_id,
            status="BLOCKED_PARTIAL_FEATURE_AVAILABILITY",
            status_detail=(
                "one or more nominal features are present on only a strict subset "
                "of rows; no rows or features were dropped"
            ),
            public_audit={
                "source_row_count": len(keyed_rows),
                "partial_feature_finite_counts": error.partial,
                "row_roster_sha256": private["row_roster_sha256"],
            },
            private_audit={**private, "partial_feature_finite_counts": error.partial},
        ) from error
    ordered_matrix = np.asarray(raw_matrix[order], dtype=np.float64)
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


def _is_zero_file_preparation_residue(path: Path) -> bool:
    """Recognize only the exact empty skeleton left by the old failed writer."""

    if not path.is_dir():
        return False
    descendants = list(path.rglob("*"))
    if any(item.is_file() or item.is_symlink() for item in descendants):
        return False
    relative_dirs = {
        item.relative_to(path).as_posix()
        for item in descendants
        if item.is_dir()
    }
    return relative_dirs in ({"inputs", "inputs/cells"}, {"inputs"}, set())


def _quarantine_zero_file_residue(
    *, path: Path, quarantine_root: Path, label: str
) -> None:
    if not path.exists():
        return
    if not _is_zero_file_preparation_residue(path):
        raise FileExistsError(
            f"EDIS {label} destination already contains material artifacts: {path}"
        )
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / f"{label}_zero_file_residue"
    if target.exists():
        raise FileExistsError(f"EDIS recovery quarantine already exists: {target}")
    path.replace(target)


def _expected_pinned_binding(
    *, relative: str, sha256: str, size_bytes: int
) -> dict[str, Any]:
    return {
        "path": str(relative),
        "sha256": str(sha256),
        "size_bytes": int(size_bytes),
    }


def _prepare_build_into(
    *,
    release_id: str,
    build_id: str,
    registry: EdisPreparationRegistry,
    identity: KeyedIdentityController,
    source_root: str | Path,
    public_lane_root: str | Path,
    private_root: str | Path,
    preparation_source_snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Populate already-private staging roots without opening row labels."""

    if build_id not in {"A", "B"}:
        raise ValueError("build_id must be A or B")
    lane_root = Path(public_lane_root)
    public_root = lane_root / "inputs"
    private_root = Path(private_root)
    if (lane_root.exists() and any(lane_root.iterdir())) or (
        private_root.exists() and any(private_root.iterdir())
    ):
        raise FileExistsError("EDIS staging destinations must be empty")
    (public_root / "cells").mkdir(parents=True, exist_ok=False)
    private_root.mkdir(parents=True, exist_ok=True)

    # Audit every pinned asset independently.  An unavailable or tampered file
    # becomes an explicit cell status; it never disappears behind the first
    # exception and never causes another population to be substituted.
    root = Path(source_root).resolve()
    verified: dict[
        str,
        tuple[Mapping[str, Any], Mapping[str, Any]] | None,
    ] = {}
    asset_failures: dict[str, list[dict[str, str]]] = {}
    manifest_cache: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for spec in registry.cells:
        failures: list[dict[str, str]] = []
        source: Mapping[str, Any] | None = None
        manifest: Mapping[str, Any] | None = None
        try:
            source = verify_pinned_file(
                root=root,
                relative=spec.source_path,
                expected_sha256=spec.source_sha256,
                expected_size=spec.source_size_bytes,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            failures.append({
                "asset_role": "telemetry_source",
                "failure_type": type(error).__name__,
            })
        manifest_key = (
            spec.manifest_path,
            spec.manifest_sha256,
            spec.manifest_size_bytes,
        )
        try:
            manifest = manifest_cache.get(manifest_key)
            if manifest is None:
                manifest = verify_pinned_file(
                    root=root,
                    relative=spec.manifest_path,
                    expected_sha256=spec.manifest_sha256,
                    expected_size=spec.manifest_size_bytes,
                )
                manifest_cache[manifest_key] = manifest
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            failures.append({
                "asset_role": "source_manifest",
                "failure_type": type(error).__name__,
            })
        if failures:
            asset_failures[spec.cell_id] = failures
            verified[spec.cell_id] = None
        else:
            if source is None or manifest is None:
                raise AssertionError("verified EDIS asset binding was lost")
            verified[spec.cell_id] = (source, manifest)

    public_cells: list[dict[str, Any]] = []
    private_cells: list[dict[str, Any]] = []
    cell_statuses: list[dict[str, Any]] = []
    for spec in registry.cells:
        verified_assets = verified[spec.cell_id]
        if verified_assets is None:
            failures = asset_failures[spec.cell_id]
            cell_statuses.append({
                "cell_id": spec.cell_id,
                "dataset_id": spec.dataset_id,
                "population_id": spec.population_id,
                "model_id": spec.model_id,
                "temperature": spec.temperature,
                "status": "BLOCKED_ASSET",
                "status_detail": (
                    f"{len(failures)} pinned asset verification failure(s); "
                    "no source population was substituted"
                ),
                "expected_rows": spec.expected_rows,
                "source_row_count": 0,
                "blocking_row_count": spec.expected_rows,
                "asset_failure_types": failures,
            })
            private_cells.append({
                "cell_id": spec.cell_id,
                "status": "BLOCKED_ASSET",
                "status_detail": (
                    f"{len(failures)} pinned asset verification failure(s)"
                ),
                "source": _expected_pinned_binding(
                    relative=spec.source_path,
                    sha256=spec.source_sha256,
                    size_bytes=spec.source_size_bytes,
                ),
                "source_manifest": _expected_pinned_binding(
                    relative=spec.manifest_path,
                    sha256=spec.manifest_sha256,
                    size_bytes=spec.manifest_size_bytes,
                ),
                "asset_verification_failures": failures,
            })
            continue
        source, manifest = verified_assets
        try:
            raw, names, absent, row_ids, private = extract_target_free_cell(
                spec=spec,
                source_path=_safe_under(root, spec.source_path),
                identity=identity,
            )
        except EdisCellBlocked as blocked:
            cell_statuses.append({
                "cell_id": spec.cell_id,
                "dataset_id": spec.dataset_id,
                "population_id": spec.population_id,
                "model_id": spec.model_id,
                "temperature": spec.temperature,
                "status": blocked.status,
                "status_detail": blocked.status_detail,
                "expected_rows": spec.expected_rows,
                **blocked.public_audit,
            })
            private_cells.append({
                "cell_id": spec.cell_id,
                "status": blocked.status,
                "status_detail": blocked.status_detail,
                "source": source,
                "source_manifest": manifest,
                **blocked.private_audit,
            })
            continue
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
            "status": "READY",
            "source": source,
            "source_manifest": manifest,
            "group_membership_commitment_sha256": private["group_membership_commitment_sha256"],
            "question_roster_commitment_sha256": private["question_roster_commitment_sha256"],
            "row_roster_sha256": private["row_roster_sha256"],
            "transform_details": details,
        })
        cell_statuses.append({
            "cell_id": spec.cell_id,
            "dataset_id": spec.dataset_id,
            "population_id": spec.population_id,
            "model_id": spec.model_id,
            "temperature": spec.temperature,
            "status": "READY",
            "status_detail": "all retained views are whole-cell finite",
            "expected_rows": spec.expected_rows,
            "source_row_count": spec.expected_rows,
            "blocking_row_count": 0,
            "present_feature_count": len(names),
            "absent_feature_names": list(absent),
            "row_roster_sha256": private["row_roster_sha256"],
        })

    private_by_cell = {row["cell_id"]: row for row in private_cells}
    for dataset_id in sorted({spec.dataset_id for spec in registry.cells}):
        commitments = {
            private_by_cell[spec.cell_id]["question_roster_commitment_sha256"]
            for spec in registry.cells
            if spec.dataset_id == dataset_id
            and "question_roster_commitment_sha256" in private_by_cell[spec.cell_id]
        }
        if len(commitments) > 1:
            raise RuntimeError(
                f"{dataset_id}: saved question content/order differs across temperatures"
            )

    ready_count = sum(row["status"] == "READY" for row in cell_statuses)
    blocked_count = len(cell_statuses) - ready_count
    if len(cell_statuses) != len(registry.cells):
        raise RuntimeError("EDIS preparation did not status every registered cell")
    try:
        assert_expected_preparation_status_roster(
            registry=registry, status={"cells": cell_statuses}
        )
        status_roster_contract_match = True
        status_roster_contract_detail = "matches preregistered target-free roster"
    except RuntimeError:
        status_roster_contract_match = False
        status_roster_contract_detail = (
            "does not match preregistered target-free roster; status-only, no fit allowed"
        )
    fit_registry_available = status_roster_contract_match and ready_count > 0
    scientific_full_build = fit_registry_available and blocked_count == 0
    partial_descriptive_build = fit_registry_available and blocked_count > 0
    status_only_build = not fit_registry_available
    preparation_status = {
        "schema_version": PREPARATION_STATUS_SCHEMA,
        "release_id": release_id,
        "build_id": build_id,
        "lane_id": registry.lane_id,
        "status": (
            "READY_FULL_ROSTER"
            if scientific_full_build
            else (
                "PARTIAL_DESCRIPTIVE_BLOCKED_CELLS_VISIBLE"
                if partial_descriptive_build
                else "STATUS_ONLY_ROSTER_MISMATCH_OR_NO_RUNNABLE_CELLS"
            )
        ),
        "scientific_full_build": scientific_full_build,
        "partial_descriptive_build": partial_descriptive_build,
        "status_only_build": status_only_build,
        "status_roster_contract_match": status_roster_contract_match,
        "status_roster_contract_detail": status_roster_contract_detail,
        "headline_eligible": False,
        "fit_registry_available": fit_registry_available,
        "dataset_aggregate_status": (
            "AVAILABLE_COMPLETE_ROSTER"
            if scientific_full_build
            else "BLOCKED_INCOMPLETE_CELL_ROSTER"
        ),
        "task_aggregate_status": (
            "AVAILABLE_COMPLETE_ROSTER"
            if scientific_full_build
            else "BLOCKED_INCOMPLETE_CELL_ROSTER"
        ),
        "registered_cell_count": len(registry.cells),
        "ready_cell_count": ready_count,
        "blocked_cell_count": blocked_count,
        "trace_status_contract_id": TRACE_STATUS_CONTRACT_ID,
        "labels_opened": False,
        "row_exclusion_performed": False,
        "imputation_performed": False,
        "feature_substitution_performed": False,
        "cells": cell_statuses,
    }
    preparation_status["status_commitment_sha256"] = (
        preparation_status_commitment_sha256(preparation_status)
    )
    preparation_status["payload_sha256"] = _payload_sha256(preparation_status)
    preparation_status_path = lane_root / "PREPARATION_STATUS.json"
    atomic_write_json(preparation_status_path, preparation_status)
    preparation_status_commitment = preparation_status["status_commitment_sha256"]

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
        "scientific_full_build": scientific_full_build,
        "partial_descriptive_build": partial_descriptive_build,
        "status_only_build": status_only_build,
        "status_roster_contract_match": status_roster_contract_match,
        "headline_eligible": False,
        "aggregate_metrics_allowed": scientific_full_build,
        "fit_registry_available": fit_registry_available,
        "registered_cell_count": len(registry.cells),
        "ready_cell_count": ready_count,
        "blocked_cell_count": blocked_count,
        "preparation_status_commitment_sha256": preparation_status_commitment,
        "trace_status_contract_id": TRACE_STATUS_CONTRACT_ID,
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
    if fit_registry_available:
        atomic_write_json(public_root / "FIT_REGISTRY.json", public)
    else:
        atomic_write_json(lane_root / "FIT_UNAVAILABLE.json", {
            "release_id": release_id,
            "build_id": build_id,
            "status": "STATUS_ONLY_ROSTER_MISMATCH_OR_NO_RUNNABLE_CELLS",
            "preparation_status_commitment_sha256": preparation_status_commitment,
            "labels_opened": False,
        })

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
        "scientific_full_build": scientific_full_build,
        "partial_descriptive_build": partial_descriptive_build,
        "status_only_build": status_only_build,
        "status_roster_contract_match": status_roster_contract_match,
        "headline_eligible": False,
        "aggregate_metrics_allowed": scientific_full_build,
        "fit_registry_available": fit_registry_available,
        "preparation_status_commitment_sha256": preparation_status_commitment,
        "trace_status_contract_id": TRACE_STATUS_CONTRACT_ID,
        "labels_opened": False,
        "historical_scores_opened": False,
        "cells": private_cells,
    }
    private_provenance["payload_sha256"] = _payload_sha256(private_provenance)
    atomic_write_json(private_root / "PREPARATION_PROVENANCE.json", private_provenance)
    return public


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
    """Stage an EDIS preparation completely, then publish it by directory rename."""

    if build_id not in {"A", "B"}:
        raise ValueError("build_id must be A or B")
    release = Path(release_root) / release_id
    final_lane = release / f"build_{build_id}" / "edis"
    private_edits = Path(private_control_root) / release_id / "edis"
    final_private = private_edits / f"build_{build_id}"
    quarantine = private_edits / "recovery"

    # The old writer created inputs/cells before feature extraction.  Recover
    # only that exact zero-file skeleton, preserving it outside the release.
    _quarantine_zero_file_residue(
        path=final_lane,
        quarantine_root=quarantine,
        label=f"public_build_{build_id}",
    )
    _quarantine_zero_file_residue(
        path=final_private,
        quarantine_root=quarantine,
        label=f"private_build_{build_id}",
    )
    final_lane.parent.mkdir(parents=True, exist_ok=True)
    final_private.parent.mkdir(parents=True, exist_ok=True)
    stage_lane = Path(tempfile.mkdtemp(
        prefix=f".edis_build_{build_id}_preparing_",
        dir=final_lane.parent,
    ))
    stage_private = Path(tempfile.mkdtemp(
        prefix=f".build_{build_id}_preparing_",
        dir=final_private.parent,
    ))
    try:
        result = _prepare_build_into(
            release_id=release_id,
            build_id=build_id,
            registry=registry,
            identity=identity,
            source_root=source_root,
            public_lane_root=stage_lane,
            private_root=stage_private,
            preparation_source_snapshot=preparation_source_snapshot,
        )
        if final_lane.exists() or final_private.exists():
            raise FileExistsError("EDIS final preparation destination appeared during staging")
        # Both destinations were proven absent immediately above.  Publish the
        # controller tree first so a public tree never points to missing private
        # provenance; a failure before the second rename remains recognizable.
        stage_private.replace(final_private)
        try:
            stage_lane.replace(final_lane)
        except BaseException:
            final_private.replace(stage_private)
            raise
        return result
    finally:
        if stage_lane.exists():
            shutil.rmtree(stage_lane)
        if stage_private.exists():
            shutil.rmtree(stage_private)


__all__ = [
    "EdisCellSpec",
    "EdisPreparationRegistry",
    "FIT_REGISTRY_SCHEMA",
    "EdisCellBlocked",
    "KeyedIdentityController",
    "NOMINAL_FEATURE_NAMES",
    "PREPARATION_SOURCE_PATHS",
    "PREPARATION_STATUS_SCHEMA",
    "PRIVATE_PROVENANCE_SCHEMA",
    "REGISTRY_SCHEMA",
    "STATUS_ROSTER_CONTRACT_ID",
    "TRACE_STATUS_CONTRACT_ID",
    "audit_nominal_matrix",
    "assert_expected_preparation_status_roster",
    "extract_target_free_cell",
    "load_preparation_registry",
    "load_preparation_status",
    "preparation_status_commitment_sha256",
    "prepare_build",
    "verify_pinned_file",
]
