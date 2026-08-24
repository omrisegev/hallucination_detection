"""Strict tidy-record contracts for reconstruction benchmark releases.

This module contains no experiment logic and never reads labels or score files.
It defines the stable interchange rows consumed by the evaluator, DuckDB layer,
and static report.  The contracts are deliberately explicit: a missing result
is a status row, not an absent row, and a leaderboard group is rankable only
when every method shares the same estimand and data/access contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence


SCHEMA_REVISION = "reconstruction_reporting_v1.2.0"

STATUS_VALUES = (
    "OK",
    "OK_FALLBACK",
    "NOT_APPLICABLE",
    "NOT_RUN",
    "ADAPTER_MISSING",
    "BLOCKED_ASSET",
    "INPUT_INVALID",
    "FIT_FAILED",
    "SCORE_INCOMPLETE",
    "METRIC_UNDEFINED_SINGLE_CLASS",
    "EXCLUDED_BY_PROTOCOL",
    "QUARANTINED",
    "UNVERIFIED",
    "CONTEXT_ONLY",
)
RANKABLE_STATUSES = frozenset(("OK", "OK_FALLBACK"))
NUMERIC_CONTEXT_STATUSES = frozenset(("CONTEXT_ONLY", "UNVERIFIED"))

BETTER_DIRECTIONS = ("higher", "lower")
AGGREGATION_LEVELS = ("cell", "dataset", "task", "release", "context")
LABEL_STAGES = ("label_free", "post_freeze_labels", "not_applicable")
EVIDENCE_GRADES = ("D0", "D1", "D2", "context", "ungraded")
ACCESS_TIERS = (
    "gray_box_single_pass",
    "gray_box_multi_pass",
    "white_box",
    "external_judge",
    "supervised_probe",
    "mixed",
    "context_only",
)

_ID_RE = re.compile(r"^[^\x00-\x1f\x7f]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SchemaError(ValueError):
    """A reporting record or cross-record relationship is invalid."""


class MissingOptionalDependency(RuntimeError):
    """A requested release format needs an optional reporting dependency."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and reject NaN/Infinity."""

    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"value is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derive_cohort_id(prediction_rows: Iterable[Mapping[str, Any]]) -> str:
    """Bind one evaluated cohort to its complete row/group identity.

    Only eligible, successfully scored rows enter a rankable metric.  Sorting the
    ``(row_id, group_id)`` pairs makes the identifier independent of file discovery
    order while retaining the grouping used by a grouped bootstrap.  A producer-
    supplied nickname or row count is intentionally insufficient.
    """

    identities: list[dict[str, str]] = []
    for index, row in enumerate(prediction_rows):
        if not isinstance(row, Mapping):
            raise SchemaError(f"cohort prediction row {index} must be a mapping")
        if not bool(row.get("eligible", False)):
            continue
        if row.get("status") not in RANKABLE_STATUSES:
            continue
        if row.get("continuous_score") is None:
            continue
        identities.append(
            {
                "row_id": _id(row.get("row_id"), field="cohort.row_id"),
                "group_id": _id(row.get("group_id"), field="cohort.group_id"),
            }
        )
    identities.sort(key=lambda item: (item["row_id"], item["group_id"]))
    pairs = [(item["row_id"], item["group_id"]) for item in identities]
    if len(pairs) != len(set(pairs)):
        raise SchemaError("cohort repeats a (row_id, group_id) identity")
    payload = {
        "schema": "reconstruction_cohort_identity_v1",
        "identity_fields": ["row_id", "group_id"],
        "rows": identities,
    }
    return f"cohort::{canonical_sha256(payload)}"


def derive_aggregate_cohort_id(
    unit_field: str,
    components: Iterable[Mapping[str, Any]],
) -> str:
    """Bind an equal-unit aggregate to every component cohort it contains."""

    if unit_field not in ("cell_id", "dataset_id"):
        raise SchemaError("aggregate cohort unit_field must be cell_id or dataset_id")
    identities = [
        {
            "unit_id": _id(row.get(unit_field), field=f"aggregate_cohort.{unit_field}"),
            "cohort_id": _id(row.get("cohort_id"), field="aggregate_cohort.cohort_id"),
        }
        for row in components
    ]
    identities.sort(key=lambda item: (item["unit_id"], item["cohort_id"]))
    unit_ids = [item["unit_id"] for item in identities]
    if len(unit_ids) != len(set(unit_ids)):
        raise SchemaError("aggregate cohort repeats a component unit")
    payload = {
        "schema": "reconstruction_aggregate_cohort_identity_v1",
        "unit_field": unit_field,
        "components": identities,
    }
    return f"cohort::{canonical_sha256(payload)}"


def _require_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{where} must be a mapping")
    return value


def _require_fields(row: Mapping[str, Any], fields: Sequence[str], *, where: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise SchemaError(f"{where} missing required fields: {missing}")


def _id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SchemaError(f"{field} must be a non-empty, trimmed string")
    if _ID_RE.fullmatch(value) is None:
        raise SchemaError(f"{field} contains a control character")
    return value


def _optional_id(value: Any, *, field: str) -> Optional[str]:
    if value is None:
        return None
    return _id(value, field=field)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field} must be non-empty text")
    return value.strip()


def _boolean(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(f"{field} must be boolean")
    return value


def _integer(value: Any, *, field: str, minimum: Optional[int] = None) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise SchemaError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaError(f"{field} must be >= {minimum}")
    return value


def _finite(value: Any, *, field: str, allow_none: bool = False) -> Optional[float]:
    if value is None and allow_none:
        return None
    if type(value) is bool or not isinstance(value, (int, float)):
        raise SchemaError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{field} must be finite")
    return result


def _choice(value: Any, choices: Sequence[str], *, field: str) -> str:
    value = _id(value, field=field)
    if value not in choices:
        raise SchemaError(f"{field} must be one of {tuple(choices)!r}; got {value!r}")
    return value


def _sha256(value: Any, *, field: str) -> str:
    value = _id(value, field=field)
    if _SHA256_RE.fullmatch(value) is None:
        raise SchemaError(f"{field} must be a lowercase SHA-256 digest")
    return value


COMMON_RESULT_FIELDS = (
    "release_id",
    "run_id",
    "lane_id",
    "task_id",
    "dataset_id",
    "population_id",
    "cell_id",
    "slice_id",
    "cohort_id",
    "method_id",
    "method_version_id",
    "adapter_id",
    "system_id",
    "comparison_group_id",
    "feature_contract_id",
    "access_contract_id",
    "evaluator_id",
    "evidence_grade",
    "status",
    "status_detail",
)

PREDICTION_FIELDS = COMMON_RESULT_FIELDS + (
    "row_id",
    "group_id",
    "continuous_score",
    "discrete_prediction",
    "label",
    "eligible",
    "fallback_used",
    "score_hash",
)

METRIC_FIELDS = COMMON_RESULT_FIELDS + (
    "aggregation_id",
    "aggregation_level",
    "metric_id",
    "metric_label",
    "metric_unit",
    "positive_class",
    "better_direction",
    "value",
    "ci_low",
    "ci_high",
    "n_rows",
    "n_groups",
    "n_positive",
    "n_negative",
    "bootstrap_unit",
    "bootstrap_draws",
    "is_primary",
    "fidelity",
    "component_ids",
)

CONTRAST_FIELDS = COMMON_RESULT_FIELDS + (
    "aggregation_id",
    "aggregation_level",
    "metric_id",
    "metric_unit",
    "positive_class",
    "better_direction",
    "left_system_id",
    "right_system_id",
    "delta",
    "ci_low",
    "ci_high",
    "wins",
    "ties",
    "losses",
    "n_pairs",
    "bootstrap_unit",
    "bootstrap_draws",
    "paired",
    "fidelity",
)

COVERAGE_FIELDS = COMMON_RESULT_FIELDS + (
    "expected_n",
    "eligible_n",
    "scored_n",
    "fallback_n",
    "excluded_n",
    "failed_n",
    "coverage_fraction",
)

GRAPH_DIAGNOSTIC_FIELDS = COMMON_RESULT_FIELDS + (
    "graph_id",
    "graph_variant",
    "graph_hash",
    "matrix_hash",
    "diagnostic_id",
    "diagnostic_label",
    "diagnostic_unit",
    "value",
    "null_value",
    "effect",
    "p_value",
    "permutation_count",
    "label_stage",
    "n_nodes",
    "n_edges",
    "notes",
)

GRAPH_EXAMPLE_FIELDS = COMMON_RESULT_FIELDS + (
    "example_id",
    "selection_rule_id",
    "selection_label_free",
    "row_kind",
    "source_row_id",
    "node_index",
    "embedding_x",
    "embedding_y",
    "y_error",
    "nuisance_name",
    "nuisance_available",
    "nuisance_value",
    "edge_source_index",
    "edge_target_index",
    "edge_weight",
    "graph_hash",
    "matrix_hash",
    "operator_hash",
    "label_stage",
    "notes",
)

TABLE_FIELDS = {
    "predictions": PREDICTION_FIELDS,
    "metrics": METRIC_FIELDS,
    "contrasts": CONTRAST_FIELDS,
    "coverage": COVERAGE_FIELDS,
    "graph_diagnostics": GRAPH_DIAGNOSTIC_FIELDS,
    "graph_examples": GRAPH_EXAMPLE_FIELDS,
}

INTEGER_FIELDS = frozenset(
    (
        "n_rows",
        "n_groups",
        "n_positive",
        "n_negative",
        "bootstrap_draws",
        "wins",
        "ties",
        "losses",
        "n_pairs",
        "expected_n",
        "eligible_n",
        "scored_n",
        "fallback_n",
        "excluded_n",
        "failed_n",
        "permutation_count",
        "n_nodes",
        "n_edges",
        "node_index",
        "edge_source_index",
        "edge_target_index",
    )
)
FLOAT_FIELDS = frozenset(
    (
        "continuous_score",
        "value",
        "ci_low",
        "ci_high",
        "delta",
        "coverage_fraction",
        "null_value",
        "effect",
        "p_value",
        "embedding_x",
        "embedding_y",
        "nuisance_value",
        "edge_weight",
    )
)
BOOLEAN_FIELDS = frozenset(
    (
        "discrete_prediction", "label", "eligible", "fallback_used", "is_primary", "paired",
        "selection_label_free", "y_error", "nuisance_available",
    )
)
JSON_FIELDS = frozenset(("component_ids",))


def _validate_common(row: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    _require_fields(row, COMMON_RESULT_FIELDS, where=where)
    normalized = dict(row)
    for field in COMMON_RESULT_FIELDS:
        if field in ("status_detail",):
            if not isinstance(row[field], str):
                raise SchemaError(f"{where}.{field} must be text (empty is allowed)")
        elif field == "evidence_grade":
            _choice(row[field], EVIDENCE_GRADES, field=f"{where}.{field}")
        elif field == "status":
            _choice(row[field], STATUS_VALUES, field=f"{where}.{field}")
        else:
            _id(row[field], field=f"{where}.{field}")
    return normalized


def validate_prediction_record(row: Mapping[str, Any]) -> dict[str, Any]:
    row = _require_mapping(row, where="prediction")
    normalized = _validate_common(row, where="prediction")
    _require_fields(row, PREDICTION_FIELDS, where="prediction")
    for field in ("row_id", "group_id", "score_hash"):
        _id(row[field], field=f"prediction.{field}")
    _boolean(row["eligible"], field="prediction.eligible")
    _boolean(row["fallback_used"], field="prediction.fallback_used")
    status = row["status"]
    score = _finite(row["continuous_score"], field="prediction.continuous_score", allow_none=True)
    if status in RANKABLE_STATUSES and row["eligible"] and score is None:
        raise SchemaError("rankable eligible prediction must have continuous_score")
    for field in ("discrete_prediction", "label"):
        value = row[field]
        if value is not None and type(value) not in (bool, int):
            raise SchemaError(f"prediction.{field} must be bool/int or None")
    if row["score_hash"] != "not_applicable":
        _sha256(row["score_hash"], field="prediction.score_hash")
    return normalized


def _validate_interval(
    row: Mapping[str, Any],
    *,
    point_field: str,
    where: str,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    point = _finite(row[point_field], field=f"{where}.{point_field}", allow_none=True)
    low = _finite(row["ci_low"], field=f"{where}.ci_low", allow_none=True)
    high = _finite(row["ci_high"], field=f"{where}.ci_high", allow_none=True)
    if (low is None) != (high is None):
        raise SchemaError(f"{where} CI endpoints must both be present or both absent")
    if low is not None:
        if point is None:
            raise SchemaError(f"{where} cannot have a CI without a point estimate")
        if not low <= point <= high:
            raise SchemaError(f"{where} must satisfy ci_low <= point <= ci_high")
    return point, low, high


def validate_metric_record(row: Mapping[str, Any]) -> dict[str, Any]:
    row = _require_mapping(row, where="metric")
    normalized = _validate_common(row, where="metric")
    _require_fields(row, METRIC_FIELDS, where="metric")
    for field in (
        "aggregation_id",
        "metric_id",
        "metric_label",
        "metric_unit",
        "positive_class",
        "bootstrap_unit",
        "fidelity",
    ):
        _id(row[field], field=f"metric.{field}")
    _choice(row["aggregation_level"], AGGREGATION_LEVELS, field="metric.aggregation_level")
    _choice(row["better_direction"], BETTER_DIRECTIONS, field="metric.better_direction")
    _boolean(row["is_primary"], field="metric.is_primary")
    for field in ("n_rows", "n_groups", "n_positive", "n_negative", "bootstrap_draws"):
        _integer(row[field], field=f"metric.{field}", minimum=0)
    components = row["component_ids"]
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise SchemaError("metric.component_ids must be a sequence")
    component_ids = [_id(value, field="metric.component_ids[]") for value in components]
    if len(component_ids) != len(set(component_ids)):
        raise SchemaError("metric.component_ids must not repeat IDs")
    point, _, _ = _validate_interval(row, point_field="value", where="metric")
    if row["status"] in RANKABLE_STATUSES and point is None:
        raise SchemaError("rankable metric must have a numeric value")
    if row["status"] not in RANKABLE_STATUSES | NUMERIC_CONTEXT_STATUSES and point is not None:
        raise SchemaError(
            f"status {row['status']!r} cannot carry a metric value; use CONTEXT_ONLY/UNVERIFIED"
        )
    if row["n_positive"] + row["n_negative"] > row["n_rows"]:
        raise SchemaError("metric class counts cannot exceed n_rows")
    if row["status"] == "METRIC_UNDEFINED_SINGLE_CLASS" and row["n_positive"] > 0 and row["n_negative"] > 0:
        raise SchemaError("single-class status requires at least one zero class count")
    normalized["component_ids"] = component_ids
    return normalized


def validate_contrast_record(row: Mapping[str, Any]) -> dict[str, Any]:
    row = _require_mapping(row, where="contrast")
    normalized = _validate_common(row, where="contrast")
    _require_fields(row, CONTRAST_FIELDS, where="contrast")
    for field in (
        "aggregation_id",
        "metric_id",
        "metric_unit",
        "positive_class",
        "left_system_id",
        "right_system_id",
        "bootstrap_unit",
        "fidelity",
    ):
        _id(row[field], field=f"contrast.{field}")
    if row["left_system_id"] == row["right_system_id"]:
        raise SchemaError("contrast sides must be different systems")
    _choice(row["aggregation_level"], AGGREGATION_LEVELS, field="contrast.aggregation_level")
    _choice(row["better_direction"], BETTER_DIRECTIONS, field="contrast.better_direction")
    _boolean(row["paired"], field="contrast.paired")
    for field in ("wins", "ties", "losses", "n_pairs", "bootstrap_draws"):
        _integer(row[field], field=f"contrast.{field}", minimum=0)
    if row["wins"] + row["ties"] + row["losses"] not in (0, row["n_pairs"]):
        raise SchemaError("contrast W/T/L must be zero or sum exactly to n_pairs")
    point, _, _ = _validate_interval(row, point_field="delta", where="contrast")
    if row["status"] in RANKABLE_STATUSES and point is None:
        raise SchemaError("rankable contrast must have a numeric delta")
    if row["status"] not in RANKABLE_STATUSES | NUMERIC_CONTEXT_STATUSES and point is not None:
        raise SchemaError(f"status {row['status']!r} cannot carry a contrast value")
    return normalized


def validate_coverage_record(row: Mapping[str, Any]) -> dict[str, Any]:
    row = _require_mapping(row, where="coverage")
    normalized = _validate_common(row, where="coverage")
    _require_fields(row, COVERAGE_FIELDS, where="coverage")
    for field in ("expected_n", "eligible_n", "scored_n", "fallback_n", "excluded_n", "failed_n"):
        _integer(row[field], field=f"coverage.{field}", minimum=0)
    if row["eligible_n"] > row["expected_n"]:
        raise SchemaError("coverage.eligible_n cannot exceed expected_n")
    if row["scored_n"] > row["eligible_n"]:
        raise SchemaError("coverage.scored_n cannot exceed eligible_n")
    if row["fallback_n"] > row["scored_n"]:
        raise SchemaError("coverage.fallback_n cannot exceed scored_n")
    if row["excluded_n"] + row["failed_n"] + row["scored_n"] > row["expected_n"]:
        raise SchemaError("coverage accounted rows cannot exceed expected_n")
    fraction = _finite(row["coverage_fraction"], field="coverage.coverage_fraction")
    expected_fraction = (row["scored_n"] / row["eligible_n"]) if row["eligible_n"] else 0.0
    if abs(fraction - expected_fraction) > 1e-12:
        raise SchemaError("coverage_fraction must equal scored_n / eligible_n (or zero if none)")
    return normalized


def validate_graph_diagnostic_record(row: Mapping[str, Any]) -> dict[str, Any]:
    row = _require_mapping(row, where="graph_diagnostic")
    normalized = _validate_common(row, where="graph_diagnostic")
    _require_fields(row, GRAPH_DIAGNOSTIC_FIELDS, where="graph_diagnostic")
    for field in (
        "graph_id",
        "graph_variant",
        "diagnostic_id",
        "diagnostic_label",
        "diagnostic_unit",
    ):
        _id(row[field], field=f"graph_diagnostic.{field}")
    for field in ("graph_hash", "matrix_hash"):
        value = row[field]
        if value != "not_applicable":
            _sha256(value, field=f"graph_diagnostic.{field}")
    _choice(row["label_stage"], LABEL_STAGES, field="graph_diagnostic.label_stage")
    for field in ("n_nodes", "n_edges", "permutation_count"):
        _integer(row[field], field=f"graph_diagnostic.{field}", minimum=0)
    for field in ("value", "null_value", "effect", "p_value"):
        _finite(row[field], field=f"graph_diagnostic.{field}", allow_none=True)
    if row["p_value"] is not None and not 0.0 <= float(row["p_value"]) <= 1.0:
        raise SchemaError("graph_diagnostic.p_value must lie in [0, 1]")
    if not isinstance(row["notes"], str):
        raise SchemaError("graph_diagnostic.notes must be text")
    if row["status"] in RANKABLE_STATUSES and row["value"] is None:
        raise SchemaError("OK graph diagnostic must have a value")
    return normalized


def validate_graph_example_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one node or edge from a label-free-selected graph example."""

    row = _require_mapping(row, where="graph_example")
    normalized = _validate_common(row, where="graph_example")
    _require_fields(row, GRAPH_EXAMPLE_FIELDS, where="graph_example")
    for field in ("example_id", "selection_rule_id", "row_kind", "source_row_id", "nuisance_name"):
        _id(row[field], field=f"graph_example.{field}")
    _choice(row["row_kind"], ("node", "edge"), field="graph_example.row_kind")
    _boolean(row["selection_label_free"], field="graph_example.selection_label_free")
    if row["selection_label_free"] is not True:
        raise SchemaError("graph example selection must be label-free")
    _boolean(row["nuisance_available"], field="graph_example.nuisance_available")
    for field in ("graph_hash", "matrix_hash", "operator_hash"):
        _sha256(row[field], field=f"graph_example.{field}")
    _choice(row["label_stage"], LABEL_STAGES, field="graph_example.label_stage")
    if row["label_stage"] != "post_freeze_labels":
        raise SchemaError("graph example rows containing correctness colors must be post_freeze_labels")
    for field in ("node_index", "edge_source_index", "edge_target_index"):
        _integer(row[field], field=f"graph_example.{field}", minimum=-1)
    if not isinstance(row["notes"], str):
        raise SchemaError("graph_example.notes must be text")
    if row["row_kind"] == "node":
        if row["node_index"] < 0 or row["edge_source_index"] != -1 or row["edge_target_index"] != -1:
            raise SchemaError("graph example node indices are inconsistent")
        _finite(row["embedding_x"], field="graph_example.embedding_x")
        _finite(row["embedding_y"], field="graph_example.embedding_y")
        _boolean(row["y_error"], field="graph_example.y_error")
        if row["edge_weight"] is not None:
            raise SchemaError("graph example node cannot carry edge_weight")
        if row["nuisance_available"]:
            _finite(row["nuisance_value"], field="graph_example.nuisance_value")
        elif row["nuisance_value"] is not None:
            raise SchemaError("unavailable graph-example nuisance cannot carry a value")
    else:
        if row["node_index"] != -1 or row["edge_source_index"] < 0 or row["edge_target_index"] < 0:
            raise SchemaError("graph example edge indices are inconsistent")
        if any(row[field] is not None for field in ("embedding_x", "embedding_y", "y_error", "nuisance_value")):
            raise SchemaError("graph example edge cannot carry node values")
        weight = _finite(row["edge_weight"], field="graph_example.edge_weight")
        if weight < 0:
            raise SchemaError("graph example edge weight must be nonnegative")
    return normalized


VALIDATORS: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "predictions": validate_prediction_record,
    "metrics": validate_metric_record,
    "contrasts": validate_contrast_record,
    "coverage": validate_coverage_record,
    "graph_diagnostics": validate_graph_diagnostic_record,
    "graph_examples": validate_graph_example_record,
}


def validate_records(table: str, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if table not in VALIDATORS:
        raise SchemaError(f"unknown reporting table {table!r}")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            normalized.append(VALIDATORS[table](row))
        except SchemaError as exc:
            raise SchemaError(f"{table} row {index}: {exc}") from exc
    return normalized


COMPARISON_SIGNATURE_FIELDS = (
    "lane_id",
    "task_id",
    "dataset_id",
    "population_id",
    "cell_id",
    "slice_id",
    "cohort_id",
    "aggregation_id",
    "aggregation_level",
    "metric_id",
    "metric_unit",
    "positive_class",
    "better_direction",
    "feature_contract_id",
    "access_contract_id",
    "evaluator_id",
    "evidence_grade",
    "fidelity",
)


def comparison_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    _require_fields(row, COMPARISON_SIGNATURE_FIELDS, where="comparison signature")
    return {field: row[field] for field in COMPARISON_SIGNATURE_FIELDS}


def derive_comparison_group_id(row: Mapping[str, Any]) -> str:
    """Return a readable, content-addressed group identifier.

    The prefix helps humans identify accidental cross-task joins while the hash
    binds every field that defines a rankable estimand.
    """

    signature = comparison_signature(row)
    prefix = f"{signature['task_id']}::{signature['dataset_id']}::{signature['metric_id']}"
    return f"{prefix}::{canonical_sha256(signature)[:16]}"


def validate_comparison_groups(rows: Iterable[Mapping[str, Any]]) -> None:
    """Reject mixed estimands, forged group IDs, or duplicate systems."""

    groups: MutableMapping[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_id(row.get("comparison_group_id"), field="comparison_group_id")].append(row)
    for group_id, members in groups.items():
        signatures = {canonical_sha256(comparison_signature(row)) for row in members}
        if len(signatures) != 1:
            varying = [
                field
                for field in COMPARISON_SIGNATURE_FIELDS
                if len({canonical_json_bytes(row[field]) for row in members}) > 1
            ]
            raise SchemaError(
                f"comparison group {group_id!r} mixes incompatible fields: {varying}"
            )
        expected_group_id = derive_comparison_group_id(members[0])
        if group_id != expected_group_id:
            raise SchemaError(
                f"comparison group {group_id!r} is not content-addressed; "
                f"expected {expected_group_id!r}"
            )
        systems = [row["system_id"] for row in members]
        duplicates = sorted({value for value in systems if systems.count(value) > 1})
        if duplicates:
            raise SchemaError(
                f"comparison group {group_id!r} repeats system rows: {duplicates}"
            )


def rank_metric_group(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank one exact comparison group and mark a conservative CI-overlap set.

    The `uncertainty_tie` flag is descriptive: it means the row's interval is
    not disjoint from the point leader's interval.  Paired contrasts remain the
    inferential source of truth and are presented separately.
    """

    if not rows:
        return []
    validate_comparison_groups(rows)
    rankable = [row for row in rows if row["status"] in RANKABLE_STATUSES]
    if not rankable:
        return []
    direction = rankable[0]["better_direction"]
    reverse = direction == "higher"
    ordered = sorted(
        rankable,
        key=lambda row: (
            -float(row["value"]) if reverse else float(row["value"]),
            str(row["system_id"]),
        ),
    )
    best_value = float(ordered[0]["value"])
    leaders = [row for row in ordered if float(row["value"]) == best_value]
    output: list[dict[str, Any]] = []
    last_value: Optional[float] = None
    rank = 0
    for index, row in enumerate(ordered, start=1):
        value = float(row["value"])
        if last_value is None or value != last_value:
            rank = index
            last_value = value
        is_leader = value == best_value
        overlap = is_leader or any(
            None not in (row["ci_low"], row["ci_high"], leader["ci_low"], leader["ci_high"])
            and not (
                float(row["ci_high"]) < float(leader["ci_low"])
                or float(row["ci_low"]) > float(leader["ci_high"])
            )
            for leader in leaders
        )
        item = dict(row)
        item["point_rank"] = rank
        item["point_leader"] = is_leader
        item["uncertainty_tie"] = overlap
        item["uncertainty_tie_rule"] = "95% marginal CI overlaps the point leader; inspect paired contrasts"
        output.append(item)
    return output


def validate_expected_coverage(
    expected: Iterable[Mapping[str, Any]],
    coverage_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Require one explicit coverage/status row per registered system population."""

    key_fields = ("release_id", "population_id", "cell_id", "slice_id", "system_id")
    expected_keys = {
        tuple(_id(row.get(field), field=f"expected.{field}") for field in key_fields)
        for row in expected
    }
    observed_list = [
        tuple(_id(row.get(field), field=f"coverage.{field}") for field in key_fields)
        for row in coverage_rows
    ]
    observed_keys = set(observed_list)
    if len(observed_list) != len(observed_keys):
        raise SchemaError("coverage contains duplicate expected-combination rows")
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing or extra:
        raise SchemaError(
            f"coverage registry mismatch: missing={missing[:5]!r}, extra={extra[:5]!r}"
        )


def validate_equal_unit_aggregates(
    metric_rows: Iterable[Mapping[str, Any]],
    aggregations: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = 1e-12,
) -> None:
    """Verify registered equal-cell/dataset means from their component rows.

    Confidence intervals are intentionally not reconstructed here; those must
    come from the registered grouped bootstrap.  This gate verifies only that
    the displayed point aggregate is the declared equal-unit mean.
    """

    rows = list(metric_rows)
    by_aggregation = {row["aggregation_id"]: row for row in aggregations}
    for aggregate in rows:
        if aggregate["aggregation_level"] == "cell" or aggregate["status"] not in RANKABLE_STATUSES:
            continue
        definition = by_aggregation.get(aggregate["aggregation_id"])
        if definition is None:
            raise SchemaError(
                f"aggregate metric references unknown aggregation {aggregate['aggregation_id']!r}"
            )
        if definition.get("rule") != "equal_unit_mean":
            continue
        component_ids = list(definition.get("component_ids", []))
        if not component_ids:
            raise SchemaError("equal_unit_mean aggregation must register component_ids")
        unit_field = definition.get("unit_field")
        if unit_field not in ("cell_id", "dataset_id"):
            raise SchemaError("equal_unit_mean unit_field must be cell_id or dataset_id")
        if list(aggregate.get("component_ids", [])) != component_ids:
            raise SchemaError(
                f"aggregate {aggregate['aggregation_id']!r} metric component_ids "
                "must exactly match its registry definition"
            )
        compatibility_fields = (
            "release_id",
            "run_id",
            "lane_id",
            "task_id",
            "system_id",
            "method_id",
            "method_version_id",
            "adapter_id",
            "metric_id",
            "metric_unit",
            "positive_class",
            "better_direction",
            "feature_contract_id",
            "access_contract_id",
            "evaluator_id",
            "evidence_grade",
            "fidelity",
        )
        by_unit: MutableMapping[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get(unit_field) not in component_ids:
                continue
            if row.get("status") not in RANKABLE_STATUSES:
                continue
            if row.get("aggregation_level") == aggregate["aggregation_level"]:
                continue
            if all(row.get(field) == aggregate.get(field) for field in compatibility_fields):
                by_unit[str(row[unit_field])].append(row)
        counts = {unit_id: len(by_unit.get(unit_id, [])) for unit_id in component_ids}
        if any(count != 1 for count in counts.values()):
            raise SchemaError(
                f"aggregate {aggregate['aggregation_id']!r} requires exactly one "
                f"compatible row per component: {counts!r}"
            )
        components = [by_unit[unit_id][0] for unit_id in component_ids]
        expected_cohort = derive_aggregate_cohort_id(unit_field, components)
        if aggregate["cohort_id"] != expected_cohort:
            raise SchemaError(
                f"aggregate {aggregate['aggregation_id']!r} cohort_id does not bind "
                "its exact component cohorts"
            )
        expected = sum(float(row["value"]) for row in components) / len(components)
        if abs(float(aggregate["value"]) - expected) > tolerance:
            raise SchemaError(
                f"aggregate {aggregate['aggregation_id']!r} value {aggregate['value']} "
                f"does not equal component mean {expected}"
            )


def record_sort_key(table: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Stable scientific sort independent of input discovery order."""

    common = (
        row.get("release_id", ""),
        row.get("run_id", ""),
        row.get("lane_id", ""),
        row.get("task_id", ""),
        row.get("dataset_id", ""),
        row.get("population_id", ""),
        row.get("cell_id", ""),
        row.get("slice_id", ""),
        row.get("cohort_id", ""),
        row.get("comparison_group_id", ""),
        row.get("feature_contract_id", ""),
        row.get("access_contract_id", ""),
        row.get("evaluator_id", ""),
        row.get("evidence_grade", ""),
        row.get("system_id", ""),
    )
    if table == "predictions":
        specific = (row.get("group_id", ""), row.get("row_id", ""))
    elif table == "metrics":
        specific = (
            row.get("aggregation_level", ""),
            row.get("aggregation_id", ""),
            row.get("metric_id", ""),
            row.get("fidelity", ""),
        )
    elif table == "contrasts":
        specific = (
            row.get("aggregation_level", ""),
            row.get("aggregation_id", ""),
            row.get("metric_id", ""),
            row.get("fidelity", ""),
            row.get("left_system_id", ""),
            row.get("right_system_id", ""),
        )
    elif table == "coverage":
        specific = ()
    elif table == "graph_diagnostics":
        specific = (
            row.get("graph_id", ""),
            row.get("graph_variant", ""),
            row.get("graph_hash", ""),
            row.get("matrix_hash", ""),
            row.get("diagnostic_id", ""),
            row.get("label_stage", ""),
        )
    elif table == "graph_examples":
        specific = (
            row.get("example_id", ""),
            row.get("row_kind", ""),
            row.get("node_index", -1),
            row.get("edge_source_index", -1),
            row.get("edge_target_index", -1),
            row.get("source_row_id", ""),
        )
    else:
        raise SchemaError(f"unknown table {table!r}")
    # The canonical row bytes are a final total-order tie-breaker.  This prevents
    # duplicate scientific keys with different payloads from inheriting input
    # discovery order and changing table or plot hashes.
    return common + specific + (canonical_json_bytes(dict(row)),)


def table_sha256(table: str, rows: Iterable[Mapping[str, Any]]) -> str:
    normalized = validate_records(table, rows)
    ordered = sorted(normalized, key=lambda row: record_sort_key(table, row))
    return canonical_sha256({"schema": SCHEMA_REVISION, "table": table, "rows": ordered})


PLOT_KINDS = (
    "forest",
    "heatmap",
    "faceted_heatmap",
    "contrast_forest",
    "line",
    "scatter",
    "diagnostic_heatmap",
    "diagnostic_summary",
    "diagnostic_scatter",
    "graph_embedding_pair",
)
PLOT_SOURCE_TABLES = tuple(TABLE_FIELDS)


def make_plot_spec(
    *,
    plot_id: str,
    title: str,
    kind: str,
    source_table: str,
    rows: Iterable[Mapping[str, Any]],
    filters: Mapping[str, Any],
    encodings: Mapping[str, Any],
    legend: Sequence[str],
    caption: str,
    better_direction: str,
    ci_definition: str,
    selection_rule: str,
) -> dict[str, Any]:
    """Create a content-addressed plot contract from its exact source rows."""

    source_rows = validate_records(source_table, rows)
    selected_rows = []
    for row in source_rows:
        matches = True
        for field, expected in filters.items():
            if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
                matches = row.get(field) in expected
            else:
                matches = row.get(field) == expected
            if not matches:
                break
        if matches:
            selected_rows.append(row)
    comparison_groups = {
        row["comparison_group_id"] for row in selected_rows
        if row.get("comparison_group_id")
    }
    multi_group_kinds = {"faceted_heatmap", "diagnostic_summary", "diagnostic_scatter"}
    if len(comparison_groups) > 1 and kind not in multi_group_kinds:
        raise SchemaError(
            f"plot {plot_id!r} mixes comparison groups: {sorted(comparison_groups)!r}"
        )
    if kind == "faceted_heatmap":
        if not selected_rows:
            raise SchemaError(f"faceted heatmap {plot_id!r} has no source rows")
        if any(row.get("aggregation_level") != "cell" for row in selected_rows):
            raise SchemaError("faceted heatmaps may contain only cell-level metric rows")
        compatibility_fields = (
            "release_id",
            "run_id",
            "lane_id",
            "task_id",
            "metric_id",
            "metric_unit",
            "positive_class",
            "better_direction",
            "feature_contract_id",
            "access_contract_id",
            "evaluator_id",
            "evidence_grade",
            "fidelity",
            "adapter_id",
        )
        signatures = {
            tuple(row.get(field) for field in compatibility_fields)
            for row in selected_rows
        }
        if len(signatures) != 1:
            raise SchemaError(
                f"faceted heatmap {plot_id!r} mixes scientific contracts"
            )
        group_cells: MutableMapping[str, set[str]] = defaultdict(set)
        tile_keys: list[tuple[str, str, str]] = []
        for row in selected_rows:
            group_cells[str(row["comparison_group_id"])].add(str(row["cell_id"]))
            tile_keys.append(
                (str(row["comparison_group_id"]), str(row["cell_id"]), str(row["system_id"]))
            )
        if any(len(cell_ids) != 1 for cell_ids in group_cells.values()):
            raise SchemaError(
                f"faceted heatmap {plot_id!r} has a comparison group spanning multiple cells"
            )
        if len(tile_keys) != len(set(tile_keys)):
            raise SchemaError(f"faceted heatmap {plot_id!r} repeats a cell/system tile")
    if kind == "graph_embedding_pair":
        if not selected_rows:
            raise SchemaError(f"graph embedding {plot_id!r} has no source rows")
        example_ids = {row.get("example_id") for row in selected_rows}
        if len(example_ids) != 1 or any(row.get("selection_label_free") is not True for row in selected_rows):
            raise SchemaError("graph embedding plot must contain one label-free-selected example")
    if kind in ("diagnostic_summary", "diagnostic_scatter"):
        if not selected_rows:
            raise SchemaError(f"{kind} {plot_id!r} has no source rows")
        scientific_fields = (
            "release_id", "run_id", "lane_id", "task_id", "feature_contract_id",
            "access_contract_id", "evaluator_id", "evidence_grade", "adapter_id",
        )
        if len({tuple(row.get(field) for field in scientific_fields) for row in selected_rows}) != 1:
            raise SchemaError(f"{kind} mixes scientific contracts")
    spec = {
        "plot_id": plot_id,
        "title": title,
        "kind": kind,
        "source_table": source_table,
        "filters": dict(filters),
        "encodings": dict(encodings),
        "legend": list(legend),
        "caption": caption,
        "better_direction": better_direction,
        "ci_definition": ci_definition,
        "selection_rule": selection_rule,
        "data_sha256": table_sha256(source_table, selected_rows),
        "n_source_rows": len(selected_rows),
    }
    return validate_plot_spec(spec)


def validate_plot_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    spec = _require_mapping(spec, where="plot spec")
    required = (
        "plot_id",
        "title",
        "kind",
        "source_table",
        "filters",
        "encodings",
        "legend",
        "caption",
        "better_direction",
        "ci_definition",
        "selection_rule",
        "data_sha256",
        "n_source_rows",
    )
    _require_fields(spec, required, where="plot spec")
    for field in ("plot_id", "title", "caption", "ci_definition", "selection_rule"):
        _text(spec[field], field=f"plot_spec.{field}")
    _choice(spec["kind"], PLOT_KINDS, field="plot_spec.kind")
    _choice(spec["source_table"], PLOT_SOURCE_TABLES, field="plot_spec.source_table")
    _choice(spec["better_direction"], ("higher", "lower", "context_dependent"), field="plot_spec.better_direction")
    _sha256(spec["data_sha256"], field="plot_spec.data_sha256")
    _integer(spec["n_source_rows"], field="plot_spec.n_source_rows", minimum=0)
    if not isinstance(spec["filters"], Mapping):
        raise SchemaError("plot_spec.filters must be a mapping")
    unknown_filters = sorted(
        set(spec["filters"]) - set(TABLE_FIELDS[spec["source_table"]])
    )
    if unknown_filters:
        raise SchemaError(f"plot_spec.filters contains unknown fields: {unknown_filters}")
    if not isinstance(spec["encodings"], Mapping) or not spec["encodings"]:
        raise SchemaError("plot_spec.encodings must be a non-empty mapping")
    legend = spec["legend"]
    if isinstance(legend, (str, bytes)) or not isinstance(legend, Sequence) or not legend:
        raise SchemaError("plot_spec.legend must contain visible legend statements")
    for index, item in enumerate(legend):
        _text(item, field=f"plot_spec.legend[{index}]")
    return dict(spec)


def validate_plot_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _require_mapping(manifest, where="plot manifest")
    _require_fields(manifest, ("schema", "release_id", "plots"), where="plot manifest")
    if manifest["schema"] != "reconstruction_plot_manifest_v1":
        raise SchemaError("unknown plot manifest schema")
    _id(manifest["release_id"], field="plot_manifest.release_id")
    plots = manifest["plots"]
    if isinstance(plots, (str, bytes)) or not isinstance(plots, Sequence):
        raise SchemaError("plot_manifest.plots must be a sequence")
    normalized = sorted(
        (validate_plot_spec(item) for item in plots),
        key=lambda item: item["plot_id"],
    )
    ids = [item["plot_id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise SchemaError("plot manifest contains duplicate plot_id values")
    output = dict(manifest)
    output["plots"] = normalized
    expected_hash = canonical_sha256({"release_id": manifest["release_id"], "plots": normalized})
    if "manifest_sha256" in manifest and manifest["manifest_sha256"] != expected_hash:
        raise SchemaError("plot manifest_sha256 does not match its content")
    output["manifest_sha256"] = expected_hash
    return output
