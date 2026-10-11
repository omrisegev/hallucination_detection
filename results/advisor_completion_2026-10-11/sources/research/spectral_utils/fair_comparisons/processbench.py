"""Strict ProcessBench adapters for Fair Paper-Exact Comparisons v1.

This module is deliberately a join layer, not an evaluator.  It turns trusted
ProcessBench telemetry, frozen Unified-28 JSONL records, and external competitor
predictions into ``comparison_record_v1`` dictionaries.  Every join is by the
official row identifier.  Mapping keys, list positions, and source iteration order
are never used as substitutes for a missing ID.

The trusted population is built once from the official 3,400-row telemetry bundle.
All method adapters then join against that population and emit a machine-readable
audit before any metric code is allowed to run.  A strict adapter fails on duplicate
IDs, label disagreement, extra rows, missing fields, or less than 100% coverage.

An external method's unparsed output is not dropped.  It is represented by
``discrete_prediction=None`` and ``prediction_status='unparsed'``; the frozen
ProcessBench evaluator consequently counts it as wrong.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .folds import ordered_id_sha256


PROCESSBENCH_DATASET_REVISION = "processbench@e8024636bcab"
PROCESSBENCH_POPULATION_ID = f"{PROCESSBENCH_DATASET_REVISION}::official-3400"
PROCESSBENCH_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
PROCESSBENCH_EXPECTED_COUNTS = {
    "gsm8k": 400,
    "math": 1_000,
    "olympiadbench": 1_000,
    "omnimath": 1_000,
}
PROCESSBENCH_NO_ERROR = -1
PROCESSBENCH_PREFIX_BUDGETS = (16, 32, 64, 128, 256, 512)
COMPARISON_RECORD_SCHEMA = "comparison_record_v1"
PROCESSBENCH_ADAPTER_REVISION = "fair_processbench_adapter_v1.0.0"
UPRM_EQ6_QWEN25_14B_METHOD_ID = "uprm_eq6_qwen2_5_14b_control"
UPRM_EQ6_QWEN25_14B_DISPLAY_NAME = "uPRM Eq.6 Qwen2.5-14B control"
UPRM_EQ6_QWEN25_14B_FIDELITY = "paper-specified-partial"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a local source artifact without mutating it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_processbench_id(subset: str, official_id: str) -> str:
    """Build the only accepted ProcessBench row key.

    ``official_id`` must be present in the artifact itself.  In particular, callers
    must not pass a pickle mapping key or a list position when the row's ``id`` field
    is absent.  The known ``<subset>-...`` prefix is checked when present.
    """

    if subset not in PROCESSBENCH_SUBSETS:
        raise ValueError(f"unknown ProcessBench subset {subset!r}")
    if not isinstance(official_id, str) or not official_id.strip():
        raise ValueError("official ProcessBench id must be a non-empty string")
    if "::" in official_id:
        raise ValueError(f"official ProcessBench id contains reserved delimiter: {official_id!r}")
    official_id = official_id.strip()
    prefix = official_id.split("-", 1)[0].lower()
    if prefix in PROCESSBENCH_SUBSETS and prefix != subset:
        raise ValueError(
            f"ProcessBench id/subset disagreement: id={official_id!r}, subset={subset!r}"
        )
    return f"{PROCESSBENCH_DATASET_REVISION}::{subset}::{official_id}"


@dataclass(frozen=True)
class ProcessBenchPopulationRow:
    """One trusted source question and its two distinct ProcessBench targets."""

    row_id: str
    group_id: str
    subset: str
    official_id: str
    model: str
    localization_label: int
    wrong_label: int
    final_answer_correct: bool
    n_steps: int
    trace_length: int

    @property
    def cell_id(self) -> str:
        return f"{PROCESSBENCH_DATASET_REVISION}::{self.model}::{self.subset}"


@dataclass(frozen=True)
class ProcessBenchPopulation:
    """Canonical ordered ProcessBench population used by every direct table."""

    population_id: str
    rows: Mapping[str, ProcessBenchPopulationRow]
    ordered_ids: tuple[str, ...]
    ordered_id_sha256: str
    source_hashes: Mapping[str, str] = field(default_factory=dict)

    def row(self, row_id: str) -> ProcessBenchPopulationRow:
        try:
            return self.rows[row_id]
        except KeyError as exc:
            raise KeyError(f"row id is not in {self.population_id}: {row_id}") from exc

    def ids_for_subset(self, subset: str) -> tuple[str, ...]:
        return tuple(row_id for row_id in self.ordered_ids if self.rows[row_id].subset == subset)


@dataclass
class JoinAudit:
    """Serializable fail-closed join report."""

    source: str
    lane: str
    population_id: str
    population_ordered_id_sha256: str | None
    expected_rows: int
    observed_source_rows: int = 0
    emitted_records: int = 0
    coverage_by_method: dict[str, float] = field(default_factory=dict)
    expected_by_method: dict[str, int] = field(default_factory=dict)
    observed_by_method: dict[str, int] = field(default_factory=dict)
    missing_ids_by_method: dict[str, list[str]] = field(default_factory=dict)
    duplicate_record_keys: list[str] = field(default_factory=list)
    duplicate_source_ids: list[str] = field(default_factory=list)
    extra_ids: list[str] = field(default_factory=list)
    label_conflicts: list[dict[str, Any]] = field(default_factory=list)
    missing_fields: list[dict[str, Any]] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    n_unparsed: int = 0

    @property
    def ok(self) -> bool:
        complete = bool(self.coverage_by_method) and all(
            value == 1.0 for value in self.coverage_by_method.values()
        )
        return complete and not any(
            (
                self.duplicate_record_keys,
                self.duplicate_source_ids,
                self.extra_ids,
                self.label_conflicts,
                self.missing_fields,
                self.schema_errors,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "processbench_join_audit_v1",
            "adapter_revision": PROCESSBENCH_ADAPTER_REVISION,
            "source": self.source,
            "lane": self.lane,
            "population_id": self.population_id,
            "population_ordered_id_sha256": self.population_ordered_id_sha256,
            "expected_rows": self.expected_rows,
            "observed_source_rows": self.observed_source_rows,
            "emitted_records": self.emitted_records,
            "coverage_by_method": dict(sorted(self.coverage_by_method.items())),
            "expected_by_method": dict(sorted(self.expected_by_method.items())),
            "observed_by_method": dict(sorted(self.observed_by_method.items())),
            "missing_ids_by_method": {
                key: value for key, value in sorted(self.missing_ids_by_method.items())
            },
            "duplicate_record_keys": self.duplicate_record_keys,
            "duplicate_source_ids": self.duplicate_source_ids,
            "extra_ids": self.extra_ids,
            "label_conflicts": self.label_conflicts,
            "missing_fields": self.missing_fields,
            "schema_errors": self.schema_errors,
            "n_unparsed": self.n_unparsed,
            "ok": self.ok,
        }


class ProcessBenchJoinError(ValueError):
    """Raised when a ProcessBench adapter fails its headline join gate."""

    def __init__(self, audit: JoinAudit):
        self.audit = audit
        summary = (
            f"ProcessBench join failed for {audit.source}: "
            f"coverage={audit.coverage_by_method}, duplicates="
            f"{len(audit.duplicate_record_keys) + len(audit.duplicate_source_ids)}, "
            f"label_conflicts={len(audit.label_conflicts)}, "
            f"missing_fields={len(audit.missing_fields)}, extras={len(audit.extra_ids)}"
        )
        super().__init__(summary)


@dataclass(frozen=True)
class PopulationBuildResult:
    population: ProcessBenchPopulation
    audit: JoinAudit


@dataclass(frozen=True)
class AdapterResult:
    records: tuple[dict[str, Any], ...]
    audit: JoinAudit


def _iter_rows(source: Mapping[Any, Any] | Sequence[Any]) -> Iterator[Mapping[str, Any]]:
    """Yield row payloads while intentionally discarding container positions/keys."""

    values: Iterable[Any]
    if isinstance(source, Mapping):
        values = source.values()
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        values = source
    else:
        raise TypeError("row source must be a mapping or a non-string sequence")
    for row in values:
        if not isinstance(row, Mapping):
            raise TypeError(f"row payload must be a mapping, got {type(row).__name__}")
        yield row


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _binary(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise TypeError(f"{name} must be binary, got {value!r}")


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _artifact_hash(value: str | None, *, path: str | Path | None = None) -> str:
    if value is None and path is not None:
        value = sha256_file(path)
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("source artifact SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _calibration_hash(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("calibration hash must be null or a lowercase SHA-256")
    return value


def _source_hash_for_subset(source_hash: str | Mapping[str, str], subset: str) -> str:
    value = source_hash.get(subset) if isinstance(source_hash, Mapping) else source_hash
    return _artifact_hash(value)


def _fold_for(row_id: str, folds: Mapping[str, int] | None) -> int | None:
    if folds is None:
        return None
    if row_id not in folds:
        raise KeyError(f"fold assignment missing for {row_id}")
    fold = _strict_int(folds[row_id], name="fold")
    if not 0 <= fold < 5:
        raise ValueError(f"fold must be in [0,4], got {fold}")
    return fold


def _record(
    *,
    lane: str,
    population: ProcessBenchPopulation,
    population_row: ProcessBenchPopulationRow,
    method_id: str,
    continuous_score: float | None,
    discrete_prediction: int | None,
    label: int,
    budget: str | int,
    source_artifact_hash: str,
    calibration_hash: str | None,
    fold: int | None,
    prediction_status: str,
) -> dict[str, Any]:
    record = {
        "schema": COMPARISON_RECORD_SCHEMA,
        "lane": lane,
        "population_id": population.population_id,
        "row_id": population_row.row_id,
        "group_id": population_row.group_id,
        "cell_id": population_row.cell_id,
        "family": population_row.subset,
        # Fold stratification is binary trace status even in the localization lane;
        # the raw first-error index remains in ``label`` and is never conflated with it.
        "stratify_label": int(label != PROCESSBENCH_NO_ERROR) if lane == "localization" else label,
        "method_id": method_id,
        "continuous_score": continuous_score,
        "discrete_prediction": discrete_prediction,
        "label": label,
        "budget": budget,
        "fold": fold,
        "calibration_hash": calibration_hash,
        "source_artifact_hash": source_artifact_hash,
        "prediction_status": prediction_status,
    }
    validate_comparison_record(record)
    return record


def validate_comparison_record(record: Mapping[str, Any]) -> None:
    """Validate the ProcessBench specialization of ``comparison_record_v1``."""

    required = {
        "schema",
        "lane",
        "population_id",
        "row_id",
        "group_id",
        "cell_id",
        "family",
        "stratify_label",
        "method_id",
        "continuous_score",
        "discrete_prediction",
        "label",
        "budget",
        "fold",
        "calibration_hash",
        "source_artifact_hash",
        "prediction_status",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"comparison record missing fields: {missing}")
    if record["schema"] != COMPARISON_RECORD_SCHEMA:
        raise ValueError(f"unexpected comparison schema {record['schema']!r}")
    lane = record["lane"]
    if lane not in {"global", "localization", "prefix"}:
        raise ValueError(f"unsupported ProcessBench lane {lane!r}")
    if record["population_id"] != PROCESSBENCH_POPULATION_ID:
        raise ValueError(f"unexpected ProcessBench population {record['population_id']!r}")
    expected_prefix = f"{PROCESSBENCH_DATASET_REVISION}::"
    for name in ("row_id", "group_id", "cell_id"):
        if not isinstance(record[name], str) or not record[name].startswith(expected_prefix):
            raise ValueError(f"{name} is not a canonical ProcessBench identifier")
    if record["row_id"] != record["group_id"]:
        raise ValueError("ProcessBench source question must be its own group_id")
    if record["family"] not in PROCESSBENCH_SUBSETS:
        raise ValueError(f"invalid ProcessBench family {record['family']!r}")
    _binary(record["stratify_label"], name="stratify_label")
    if not isinstance(record["method_id"], str) or not record["method_id"].strip():
        raise ValueError("method_id must be a non-empty string")
    score = record["continuous_score"]
    if score is not None:
        _finite(score, name="continuous_score")
    prediction = record["discrete_prediction"]
    if prediction is not None:
        _strict_int(prediction, name="discrete_prediction")
    status = record["prediction_status"]
    if status not in {"parsed", "unparsed", "not_applicable"}:
        raise ValueError(f"invalid prediction_status {status!r}")
    if status == "parsed" and prediction is None:
        raise ValueError("parsed prediction must not be null")
    if status in {"unparsed", "not_applicable"} and prediction is not None:
        raise ValueError(f"{status} prediction must be null")
    if lane in {"global", "prefix"}:
        _binary(record["label"], name="label")
        if record["stratify_label"] != record["label"]:
            raise ValueError(f"{lane} stratify_label must equal the binary risk label")
        if score is None:
            raise ValueError(f"{lane} record requires a continuous risk score")
    else:
        label = _strict_int(record["label"], name="label")
        if label < PROCESSBENCH_NO_ERROR:
            raise ValueError(f"invalid ProcessBench localization label {label}")
        if record["stratify_label"] != int(label != PROCESSBENCH_NO_ERROR):
            raise ValueError("localization stratify_label must be 0 for clean and 1 for error")
        if score is None and status == "not_applicable":
            raise ValueError("localization record requires a score or a prediction")
    budget = record["budget"]
    if lane == "prefix":
        if budget not in PROCESSBENCH_PREFIX_BUDGETS:
            raise ValueError(f"invalid causal-prefix budget {budget!r}")
    elif budget != "final":
        raise ValueError(f"{lane} record budget must be 'final'")
    if record["fold"] is not None:
        fold = _strict_int(record["fold"], name="fold")
        if not 0 <= fold < 5:
            raise ValueError(f"fold must be in [0,4], got {fold}")
    _calibration_hash(record["calibration_hash"])
    _artifact_hash(record["source_artifact_hash"])


def build_processbench_population(
    rows_by_subset: Mapping[str, Mapping[Any, Any] | Sequence[Any]],
    *,
    model: str = "llama31_8b",
    source_hashes: Mapping[str, str] | None = None,
    expected_counts: Mapping[str, int] = PROCESSBENCH_EXPECTED_COUNTS,
    strict: bool = True,
) -> PopulationBuildResult:
    """Build the trusted ID/label population from ProcessBench telemetry.

    The order is the official dataset order: fixed subset order followed by the
    registered ``<subset>-0 .. <subset>-(N-1)`` identifiers.  Cache insertion order is
    deliberately ignored because parallel acquisition produced shuffled dictionaries.
    Every later method is rearranged by ID to match this order; source positions are
    never used to invent an identity.
    """

    audit = JoinAudit(
        source="processbench_telemetry",
        lane="population",
        population_id=PROCESSBENCH_POPULATION_ID,
        population_ordered_id_sha256=None,
        expected_rows=sum(expected_counts.values()),
    )
    population_rows: dict[str, ProcessBenchPopulationRow] = {}
    ordered: list[str] = []
    subset_seen: dict[str, int] = {}
    hashes: dict[str, str] = {}
    registered_missing_ids: list[str] = []

    for subset in PROCESSBENCH_SUBSETS:
        source = rows_by_subset.get(subset)
        if source is None:
            audit.schema_errors.append(f"missing ProcessBench telemetry subset {subset}")
            subset_seen[subset] = 0
            continue
        if source_hashes is not None:
            try:
                hashes[subset] = _artifact_hash(source_hashes.get(subset))
            except Exception as exc:  # noqa: BLE001 - report all join defects together
                audit.schema_errors.append(f"{subset} source hash: {exc}")
        count = 0
        try:
            iterator = _iter_rows(source)
            for source_index, row in enumerate(iterator):
                audit.observed_source_rows += 1
                count += 1
                required = (
                    "id",
                    "label",
                    "final_answer_correct",
                    "steps",
                    "step_token_spans",
                    "gen_token_ids",
                )
                missing = [name for name in required if name not in row]
                if missing:
                    audit.missing_fields.append(
                        {"subset": subset, "source_index": source_index, "fields": missing}
                    )
                    continue
                try:
                    official_id = row["id"]
                    row_id = canonical_processbench_id(subset, official_id)
                    label = _strict_int(row["label"], name="label")
                    steps = row["steps"]
                    spans = row["step_token_spans"]
                    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
                        raise TypeError("steps must be a sequence")
                    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
                        raise TypeError("step_token_spans must be a sequence")
                    n_steps = len(steps)
                    if len(spans) != n_steps:
                        raise ValueError(
                            f"step span count {len(spans)} disagrees with {n_steps} steps"
                        )
                    if label < PROCESSBENCH_NO_ERROR or label >= n_steps:
                        raise ValueError(f"localization label {label} outside [-1,{n_steps - 1}]")
                    final_correct = bool(_binary(row["final_answer_correct"], name="final_answer_correct"))
                    token_ids = row["gen_token_ids"]
                    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
                        raise TypeError("gen_token_ids must be a sequence")
                    if not token_ids:
                        raise ValueError("empty gen_token_ids cannot support causal budgets")
                    align_diag = row.get("align_diag")
                    if isinstance(align_diag, Mapping) and align_diag.get("ok") is False:
                        raise ValueError("telemetry row reports failed step alignment")
                except Exception as exc:  # noqa: BLE001
                    audit.schema_errors.append(f"{subset} source row {source_index}: {exc}")
                    continue
                if row_id in population_rows:
                    audit.duplicate_source_ids.append(row_id)
                    continue
                pop_row = ProcessBenchPopulationRow(
                    row_id=row_id,
                    group_id=row_id,
                    subset=subset,
                    official_id=official_id,
                    model=model,
                    localization_label=label,
                    wrong_label=int(not final_correct),
                    final_answer_correct=final_correct,
                    n_steps=n_steps,
                    trace_length=len(token_ids),
                )
                population_rows[row_id] = pop_row
        except Exception as exc:  # noqa: BLE001
            audit.schema_errors.append(f"{subset} row container: {exc}")
        subset_seen[subset] = count

    for subset, expected in expected_counts.items():
        observed = subset_seen.get(subset, 0)
        if observed != expected:
            audit.schema_errors.append(
                f"{subset} row count {observed} does not equal registered count {expected}"
            )
        expected_ids = [
            canonical_processbench_id(subset, f"{subset}-{index}")
            for index in range(expected)
        ]
        expected_set = set(expected_ids)
        observed_set = {
            row_id for row_id, row in population_rows.items() if row.subset == subset
        }
        missing_official = sorted(expected_set.difference(observed_set))
        unexpected_official = sorted(observed_set.difference(expected_set))
        registered_missing_ids.extend(missing_official)
        audit.extra_ids.extend(unexpected_official)
        if missing_official:
            audit.schema_errors.append(
                f"{subset} missing registered official IDs: {missing_official[:10]}"
            )
        if unexpected_official:
            audit.schema_errors.append(
                f"{subset} has unexpected official IDs: {unexpected_official[:10]}"
            )
        ordered.extend(row_id for row_id in expected_ids if row_id in population_rows)
    order_hash = ordered_id_sha256(ordered)
    population = ProcessBenchPopulation(
        population_id=PROCESSBENCH_POPULATION_ID,
        rows=population_rows,
        ordered_ids=tuple(ordered),
        ordered_id_sha256=order_hash,
        source_hashes=hashes,
    )
    audit.population_ordered_id_sha256 = order_hash
    audit.expected_by_method = {"population": audit.expected_rows}
    audit.observed_by_method = {"population": len(population_rows)}
    audit.missing_ids_by_method = {"population": registered_missing_ids}
    audit.coverage_by_method = {
        "population": len(population_rows) / audit.expected_rows if audit.expected_rows else 0.0
    }
    audit.emitted_records = len(population_rows)
    if strict and not audit.ok:
        raise ProcessBenchJoinError(audit)
    return PopulationBuildResult(population=population, audit=audit)


def _jsonl_rows(source: str | Path | Iterable[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], str | None]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        rows: list[Mapping[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError(f"JSONL line {line_number} is not an object")
                rows.append(value)
        return rows, sha256_file(path)
    rows = list(source)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("JSONL-equivalent iterable contains a non-mapping row")
    return rows, None


def _new_audit(source: str, lane: str, population: ProcessBenchPopulation) -> JoinAudit:
    return JoinAudit(
        source=source,
        lane=lane,
        population_id=population.population_id,
        population_ordered_id_sha256=population.ordered_id_sha256,
        expected_rows=len(population.ordered_ids),
    )


def _finalize(
    *,
    raw_records: list[dict[str, Any]],
    audit: JoinAudit,
    population: ProcessBenchPopulation,
    methods: Sequence[str],
    budgets: Sequence[str | int],
    strict: bool,
) -> AdapterResult:
    by_key: dict[tuple[str, str | int, str], dict[str, Any]] = {}
    for record in raw_records:
        key = (record["method_id"], record["budget"], record["row_id"])
        if key in by_key:
            audit.duplicate_record_keys.append(
                json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            )
            continue
        by_key[key] = record

    expected_ids = set(population.ordered_ids)
    ordered_records: list[dict[str, Any]] = []
    for method in methods:
        method_observed = 0
        missing: list[str] = []
        for budget in budgets:
            eligible_ids = [
                row_id
                for row_id in population.ordered_ids
                if budget == "final" or population.rows[row_id].trace_length > int(budget)
            ]
            for row_id in eligible_ids:
                record = by_key.get((method, budget, row_id))
                if record is None:
                    missing.append(row_id if budget == "final" else f"{row_id}@{budget}")
                else:
                    ordered_records.append(record)
                    method_observed += 1
        expected = sum(
            1
            for budget in budgets
            for row_id in population.ordered_ids
            if budget == "final" or population.rows[row_id].trace_length > int(budget)
        )
        audit.expected_by_method[method] = expected
        audit.observed_by_method[method] = method_observed
        audit.missing_ids_by_method[method] = missing
        audit.coverage_by_method[method] = method_observed / expected if expected else 0.0

    known_method_budget = {(method, budget) for method in methods for budget in budgets}
    for (method, budget, row_id) in by_key:
        if row_id not in expected_ids:
            audit.extra_ids.append(row_id)
        elif (method, budget) not in known_method_budget:
            audit.schema_errors.append(f"unexpected method/budget record: {method}/{budget}")
    audit.emitted_records = len(ordered_records)
    audit.n_unparsed = sum(
        record["prediction_status"] == "unparsed" for record in ordered_records
    )
    if strict and not audit.ok:
        raise ProcessBenchJoinError(audit)
    return AdapterResult(records=tuple(ordered_records), audit=audit)


def adapt_unified_validation_records(
    source: str | Path | Iterable[Mapping[str, Any]],
    population: ProcessBenchPopulation,
    *,
    lane: str,
    candidate_methods: Mapping[str, str] | None = None,
    budgets: Sequence[int] = PROCESSBENCH_PREFIX_BUDGETS,
    source_artifact_hash: str | None = None,
    calibration_hash: str | None = None,
    folds: Mapping[str, int] | None = None,
    strict: bool = True,
) -> AdapterResult:
    """Adapt frozen Unified validation JSONL records by explicit ``family``/``unit`` IDs."""

    if lane not in {"global", "localization", "prefix"}:
        raise ValueError(f"unsupported Unified ProcessBench lane {lane!r}")
    candidate_methods = candidate_methods or {"base7_full28": "unified28"}
    if not candidate_methods:
        raise ValueError("candidate_methods must not be empty")
    rows, path_hash = _jsonl_rows(source)
    artifact_hash = _artifact_hash(source_artifact_hash or path_hash)
    cal_hash = _calibration_hash(calibration_hash)
    audit = _new_audit("unified_validation_records", lane, population)
    audit.observed_source_rows = len(rows)
    raw: list[dict[str, Any]] = []
    budget_values: Sequence[str | int] = ("final",) if lane != "prefix" else tuple(budgets)

    for source_index, row in enumerate(rows):
        candidate = row.get("candidate")
        if candidate not in candidate_methods:
            continue
        required = (
            "candidate",
            "family",
            "unit",
            "source_group",
            "wrong",
            "target_step",
        )
        if lane == "global":
            required += ("global_score",)
        elif lane == "localization":
            required += ("localization_score", "prediction")
        else:
            required += tuple(f"risk_at_{budget}" for budget in budget_values)
        missing = [name for name in required if name not in row]
        if missing:
            audit.missing_fields.append(
                {"source_index": source_index, "candidate": candidate, "fields": missing}
            )
            continue
        try:
            subset = row["family"]
            row_id = canonical_processbench_id(subset, row["unit"])
            if row["source_group"] != f"{subset}::{row['unit']}":
                raise ValueError("source_group is not exactly family::unit")
        except Exception as exc:  # noqa: BLE001
            audit.schema_errors.append(f"source row {source_index}: {exc}")
            continue
        pop = population.rows.get(row_id)
        if pop is None:
            audit.extra_ids.append(row_id)
            continue
        try:
            wrong = _binary(row["wrong"], name="wrong")
            target = _strict_int(row["target_step"], name="target_step")
        except Exception as exc:  # noqa: BLE001
            audit.schema_errors.append(f"{row_id}/{candidate}: {exc}")
            continue
        if wrong != pop.wrong_label or target != pop.localization_label:
            audit.label_conflicts.append(
                {
                    "row_id": row_id,
                    "method": candidate_methods[candidate],
                    "artifact_wrong": wrong,
                    "population_wrong": pop.wrong_label,
                    "artifact_localization": target,
                    "population_localization": pop.localization_label,
                }
            )
            continue
        method = candidate_methods[candidate]
        try:
            if lane == "global":
                raw.append(
                    _record(
                        lane=lane,
                        population=population,
                        population_row=pop,
                        method_id=method,
                        continuous_score=_finite(row["global_score"], name="global_score"),
                        discrete_prediction=None,
                        label=wrong,
                        budget="final",
                        source_artifact_hash=artifact_hash,
                        calibration_hash=cal_hash,
                        fold=_fold_for(row_id, folds),
                        prediction_status="not_applicable",
                    )
                )
            elif lane == "localization":
                raw.append(
                    _record(
                        lane=lane,
                        population=population,
                        population_row=pop,
                        method_id=method,
                        continuous_score=_finite(
                            row["localization_score"], name="localization_score"
                        ),
                        discrete_prediction=_strict_int(row["prediction"], name="prediction"),
                        label=target,
                        budget="final",
                        source_artifact_hash=artifact_hash,
                        calibration_hash=cal_hash,
                        fold=_fold_for(row_id, folds),
                        prediction_status="parsed",
                    )
                )
            else:
                for budget in budget_values:
                    if pop.trace_length <= int(budget):
                        continue
                    raw.append(
                        _record(
                            lane=lane,
                            population=population,
                            population_row=pop,
                            method_id=method,
                            continuous_score=_finite(
                                row[f"risk_at_{budget}"], name=f"risk_at_{budget}"
                            ),
                            discrete_prediction=None,
                            label=wrong,
                            budget=int(budget),
                            source_artifact_hash=artifact_hash,
                            calibration_hash=cal_hash,
                            fold=_fold_for(row_id, folds),
                            prediction_status="not_applicable",
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            audit.schema_errors.append(f"{row_id}/{candidate}: {exc}")

    return _finalize(
        raw_records=raw,
        audit=audit,
        population=population,
        methods=tuple(candidate_methods.values()),
        budgets=budget_values,
        strict=strict,
    )


def adapt_unified_global_records(
    source: str | Path | Iterable[Mapping[str, Any]],
    population: ProcessBenchPopulation,
    *,
    method_columns: Mapping[str, str] | None = None,
    source_artifact_hash: str | None = None,
    folds: Mapping[str, int] | None = None,
    strict: bool = True,
) -> AdapterResult:
    """Adapt the paired Unified-28/global-incumbent 3,400-row JSONL bundle."""

    method_columns = method_columns or {
        "base7_full28": "unified28",
        "classic_mixed_v2_no_length": "classic_mixed_v2_no_length",
    }
    rows, path_hash = _jsonl_rows(source)
    artifact_hash = _artifact_hash(source_artifact_hash or path_hash)
    audit = _new_audit("unified_global_records", "global", population)
    audit.observed_source_rows = len(rows)
    raw: list[dict[str, Any]] = []

    for source_index, row in enumerate(rows):
        required = ("family", "unit", "source_group", "wrong", *method_columns.keys())
        missing = [name for name in required if name not in row]
        if missing:
            audit.missing_fields.append({"source_index": source_index, "fields": missing})
            continue
        try:
            subset = row["family"]
            row_id = canonical_processbench_id(subset, row["unit"])
            if row["source_group"] != f"{subset}::{row['unit']}":
                raise ValueError("source_group is not exactly family::unit")
        except Exception as exc:  # noqa: BLE001
            audit.schema_errors.append(f"source row {source_index}: {exc}")
            continue
        pop = population.rows.get(row_id)
        if pop is None:
            audit.extra_ids.append(row_id)
            continue
        try:
            wrong = _binary(row["wrong"], name="wrong")
        except Exception as exc:  # noqa: BLE001
            audit.schema_errors.append(f"{row_id}: {exc}")
            continue
        if wrong != pop.wrong_label:
            audit.label_conflicts.append(
                {
                    "row_id": row_id,
                    "artifact_wrong": wrong,
                    "population_wrong": pop.wrong_label,
                }
            )
            continue
        for column, method in method_columns.items():
            try:
                raw.append(
                    _record(
                        lane="global",
                        population=population,
                        population_row=pop,
                        method_id=method,
                        continuous_score=_finite(row[column], name=column),
                        discrete_prediction=None,
                        label=wrong,
                        budget="final",
                        source_artifact_hash=artifact_hash,
                        calibration_hash=None,
                        fold=_fold_for(row_id, folds),
                        prediction_status="not_applicable",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                audit.schema_errors.append(f"{row_id}/{method}: {exc}")

    return _finalize(
        raw_records=raw,
        audit=audit,
        population=population,
        methods=tuple(method_columns.values()),
        budgets=("final",),
        strict=strict,
    )


def adapt_external_localization_records(
    rows_by_subset: Mapping[str, Mapping[Any, Any] | Sequence[Any]],
    population: ProcessBenchPopulation,
    *,
    method_id: str,
    source_artifact_hashes: str | Mapping[str, str],
    id_field: str = "id",
    label_field: str = "label",
    prediction_field: str = "prediction",
    folds: Mapping[str, int] | None = None,
    strict: bool = True,
) -> AdapterResult:
    """Adapt PRM, critic, or reconstructed Eq.6 fixed predictions.

    ``prediction=None`` is retained as an unparsed row.  No decision threshold is fitted
    here; these are fixed outputs and therefore carry ``calibration_hash=None``.
    """

    audit = _new_audit(method_id, "localization", population)
    raw: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()

    for subset in PROCESSBENCH_SUBSETS:
        source = rows_by_subset.get(subset)
        if source is None:
            audit.schema_errors.append(f"missing external prediction subset {subset}")
            continue
        try:
            iterator = _iter_rows(source)
            for source_index, row in enumerate(iterator):
                audit.observed_source_rows += 1
                required = (id_field, label_field, prediction_field)
                missing = [name for name in required if name not in row]
                if missing:
                    audit.missing_fields.append(
                        {"subset": subset, "source_index": source_index, "fields": missing}
                    )
                    continue
                try:
                    row_id = canonical_processbench_id(subset, row[id_field])
                except Exception as exc:  # noqa: BLE001
                    audit.schema_errors.append(f"{subset} source row {source_index}: {exc}")
                    continue
                if row_id in seen_source_ids:
                    audit.duplicate_source_ids.append(row_id)
                    continue
                seen_source_ids.add(row_id)
                pop = population.rows.get(row_id)
                if pop is None:
                    audit.extra_ids.append(row_id)
                    continue
                try:
                    label = _strict_int(row[label_field], name=label_field)
                except Exception as exc:  # noqa: BLE001
                    audit.schema_errors.append(f"{row_id}: {exc}")
                    continue
                if label != pop.localization_label:
                    audit.label_conflicts.append(
                        {
                            "row_id": row_id,
                            "artifact_localization": label,
                            "population_localization": pop.localization_label,
                        }
                    )
                    continue
                prediction_value = row[prediction_field]
                try:
                    prediction = (
                        None
                        if prediction_value is None
                        else _strict_int(prediction_value, name=prediction_field)
                    )
                    raw.append(
                        _record(
                            lane="localization",
                            population=population,
                            population_row=pop,
                            method_id=method_id,
                            continuous_score=None,
                            discrete_prediction=prediction,
                            label=label,
                            budget="final",
                            source_artifact_hash=_source_hash_for_subset(
                                source_artifact_hashes, subset
                            ),
                            calibration_hash=None,
                            fold=_fold_for(row_id, folds),
                            prediction_status="parsed" if prediction is not None else "unparsed",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    audit.schema_errors.append(f"{row_id}: {exc}")
        except Exception as exc:  # noqa: BLE001
            audit.schema_errors.append(f"{subset} row container: {exc}")

    return _finalize(
        raw_records=raw,
        audit=audit,
        population=population,
        methods=(method_id,),
        budgets=("final",),
        strict=strict,
    )


def adapt_eq6_shard_records(
    rows: Iterable[Mapping[str, Any]],
    population: ProcessBenchPopulation,
    *,
    method_id: str = UPRM_EQ6_QWEN25_14B_METHOD_ID,
    source_artifact_hash: str,
    folds: Mapping[str, int] | None = None,
    strict: bool = True,
) -> AdapterResult:
    """Adapt L1 sharded Eq.6 records using their explicit ``subset:id`` question key."""

    by_subset: dict[str, list[dict[str, Any]]] = {subset: [] for subset in PROCESSBENCH_SUBSETS}
    malformed: list[str] = []
    for source_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            malformed.append(f"source row {source_index} is not a mapping")
            continue
        subset = row.get("subset")
        question_id = row.get("question_id")
        if subset not in PROCESSBENCH_SUBSETS or not isinstance(question_id, str):
            malformed.append(f"source row {source_index} lacks subset/question_id")
            continue
        expected_prefix = f"{subset}:"
        if not question_id.startswith(expected_prefix) or len(question_id) == len(expected_prefix):
            malformed.append(
                f"source row {source_index} question_id is not exactly {subset}:<official_id>"
            )
            continue
        converted = dict(row)
        converted["id"] = question_id[len(expected_prefix) :]
        by_subset[subset].append(converted)
    result = adapt_external_localization_records(
        by_subset,
        population,
        method_id=method_id,
        source_artifact_hashes=source_artifact_hash,
        folds=folds,
        strict=False,
    )
    result.audit.source = "l1_eq6_shards"
    result.audit.schema_errors.extend(malformed)
    if strict and not result.audit.ok:
        raise ProcessBenchJoinError(result.audit)
    return result


def load_pickle_rows(path: str | Path) -> Mapping[Any, Any] | Sequence[Any]:
    """Read a local prediction pickle without interpreting its container keys."""

    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"pickle row container must be mapping/list: {path}")


def load_pickle_bundle(
    paths_by_subset: Mapping[str, str | Path],
) -> tuple[dict[str, Mapping[Any, Any] | Sequence[Any]], dict[str, str]]:
    """Read and hash a four-subset local pickle bundle for a strict adapter."""

    rows: dict[str, Mapping[Any, Any] | Sequence[Any]] = {}
    hashes: dict[str, str] = {}
    for subset in PROCESSBENCH_SUBSETS:
        if subset not in paths_by_subset:
            raise KeyError(f"pickle bundle missing subset path {subset}")
        path = Path(paths_by_subset[subset])
        rows[subset] = load_pickle_rows(path)
        hashes[subset] = sha256_file(path)
    return rows, hashes


__all__ = [
    "AdapterResult",
    "COMPARISON_RECORD_SCHEMA",
    "JoinAudit",
    "PROCESSBENCH_ADAPTER_REVISION",
    "PROCESSBENCH_DATASET_REVISION",
    "PROCESSBENCH_EXPECTED_COUNTS",
    "PROCESSBENCH_NO_ERROR",
    "PROCESSBENCH_POPULATION_ID",
    "PROCESSBENCH_PREFIX_BUDGETS",
    "PROCESSBENCH_SUBSETS",
    "PopulationBuildResult",
    "ProcessBenchJoinError",
    "ProcessBenchPopulation",
    "ProcessBenchPopulationRow",
    "UPRM_EQ6_QWEN25_14B_DISPLAY_NAME",
    "UPRM_EQ6_QWEN25_14B_FIDELITY",
    "UPRM_EQ6_QWEN25_14B_METHOD_ID",
    "adapt_eq6_shard_records",
    "adapt_external_localization_records",
    "adapt_unified_global_records",
    "adapt_unified_validation_records",
    "build_processbench_population",
    "canonical_processbench_id",
    "load_pickle_bundle",
    "load_pickle_rows",
    "sha256_file",
    "validate_comparison_record",
]
