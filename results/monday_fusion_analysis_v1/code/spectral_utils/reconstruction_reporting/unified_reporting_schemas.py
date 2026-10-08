"""Typed, label-source-closed schemas for the certified multi-lane report.

This module is intentionally independent of the prediction-backed reporting
schema.  It represents already-certified evaluation summaries and the integer
first-error decision relation without pretending that every lane has boolean
predictions or a shared estimand.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from io import BytesIO, StringIO
import json
import math
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "reconstruction-unified-reporting-v1"


class UnifiedReportingError(ValueError):
    """Raised when a certified reporting record violates its typed contract."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    nullable: bool = False


def _s(name: str, nullable: bool = False) -> FieldSpec:
    return FieldSpec(name, "string", nullable)


def _i(name: str, nullable: bool = False) -> FieldSpec:
    return FieldSpec(name, "int64", nullable)


def _f(name: str, nullable: bool = False) -> FieldSpec:
    return FieldSpec(name, "float64", nullable)


def _b(name: str, nullable: bool = False) -> FieldSpec:
    return FieldSpec(name, "bool", nullable)


SOURCE_FIELDS = (
    _s("release_id"), _s("source_binding_id"), _s("source_id"),
    _s("source_release_id"), _b("certified"), _s("source_status"),
    _s("source_root_id", True), _s("certificate_schema", True),
    _s("certificate_file_sha256", True), _s("certificate_payload_sha256", True),
    _s("manifest_schema", True), _s("manifest_file_sha256", True),
    _s("manifest_payload_sha256", True), _s("logical_binding_sha256"),
)

SOURCE_ARTIFACT_FIELDS = (
    _s("release_id"), _s("source_binding_id"), _s("source_id"),
    _s("artifact_role"), _s("logical_name"), _s("relative_path"),
    _s("format"), _s("schema_version", True), _s("file_sha256"),
    _s("payload_sha256", True), _b("authenticated"),
)

SCOPE_FIELDS = (
    _s("release_id"), _s("lane_id"), _s("task_id"),
    _s("source_dataset_id"), _s("dataset_id"), _s("population_id"),
    _s("cell_id"), _s("slice_id"), _s("prediction_unit"),
    _s("estimand_id"), _s("access_level"), _s("supervision"),
    _s("fidelity"), _s("report_partition"),
)

PROVENANCE_FIELDS = (
    _s("source_binding_id"), _s("source_table"),
    _s("source_row_locator"), _s("source_row_sha256"),
)

STATUS_FIELDS = SCOPE_FIELDS + (
    _s("status_id"), _s("status_scope"), _s("system_id"),
    _s("method_id"), _s("metric_id"), _s("aggregation_level"),
    _s("status_class"), _s("source_status"), _s("status_detail", True),
    _i("expected_n", True), _i("observed_n", True), _b("rankable"),
) + PROVENANCE_FIELDS

METRIC_FIELDS = SCOPE_FIELDS + (
    _s("system_id"), _s("method_id"), _s("metric_id"),
    _s("metric_unit"), _s("positive_class"), _s("better_direction"),
    _s("aggregation_id"), _s("aggregation_level"),
    _s("aggregation_rule"), _s("aggregation_unit"), _s("cohort_id"),
    _s("comparison_group_id"), _s("source_comparison_group_id", True),
    _f("value", True), _f("ci_low", True), _f("ci_high", True),
    _i("n_rows", True), _i("n_groups", True), _i("n_positive", True),
    _i("n_negative", True), _s("bootstrap_unit", True),
    _i("bootstrap_draws", True), _s("status_class"),
    _s("source_status"), _s("status_detail", True), _b("rankable"),
) + PROVENANCE_FIELDS

CONTRAST_FIELDS = SCOPE_FIELDS + (
    _s("left_system_id"), _s("right_system_id"), _s("left_method_id"),
    _s("right_method_id"), _s("metric_id"), _s("metric_unit"),
    _s("positive_class"), _s("better_direction"), _s("aggregation_id"),
    _s("aggregation_level"), _s("aggregation_rule"),
    _s("aggregation_unit"), _s("cohort_id"), _s("comparison_group_id"),
    _s("source_comparison_group_id", True), _f("delta", True),
    _f("ci_low", True), _f("ci_high", True), _i("n_pairs", True),
    _s("bootstrap_unit", True), _i("bootstrap_draws", True),
    _b("paired"), _s("status_class"), _s("source_status"),
    _s("status_detail", True), _b("rankable"),
) + PROVENANCE_FIELDS

COVERAGE_FIELDS = SCOPE_FIELDS + (
    _s("system_id"), _s("method_id"), _i("expected_n", True),
    _i("eligible_n", True), _i("scored_n", True), _i("fallback_n", True),
    _i("excluded_n", True), _i("failed_n", True),
    _f("coverage_fraction", True), _s("cohort_id"),
    _s("status_class"), _s("source_status"), _s("status_detail", True),
) + PROVENANCE_FIELDS

LOCALIZATION_DECISION_FIELDS = SCOPE_FIELDS + (
    _s("model_id"), _s("system_id"), _s("row_id"), _s("cohort_id"),
    _s("group_id"), _i("fold"), _i("predicted_first_error"),
    _i("true_first_error"), _s("comparison_group_id"),
    _s("source_comparison_group_id", True), _s("source_status"),
) + PROVENANCE_FIELDS

WINNER_SET_FIELDS = SCOPE_FIELDS + (
    _s("comparison_group_id"), _s("source_comparison_group_id"),
    _s("aggregation_id"), _s("aggregation_level"), _s("metric_id"),
    _s("better_direction"), _s("winner_reference_method_id"),
    _s("method_id"), _f("method_value"), _s("membership_status"),
    _b("in_winner_reference_set"), _s("interpretation_code"),
    _b("equivalence_claim"), _b("simultaneous_coverage"),
    _b("winner_selection_adjusted"), _s("multiplicity_adjustment"),
) + PROVENANCE_FIELDS

WINNER_CONTRAST_FIELDS = SCOPE_FIELDS + (
    _s("comparison_group_id"), _s("source_comparison_group_id"),
    _s("aggregation_id"), _s("aggregation_level"), _s("metric_id"),
    _s("better_direction"), _s("winner_reference_method_id"),
    _s("candidate_method_id"), _f("winner_value"), _f("candidate_value"),
    _f("delta_candidate_minus_winner"), _f("ci_low"), _f("ci_high"),
    _s("membership_status"), _b("in_winner_reference_set"),
    _s("bootstrap_unit"), _i("bootstrap_draws"),
    _s("multiplicity_adjustment"), _b("equivalence_claim"),
) + PROVENANCE_FIELDS


TABLE_SCHEMAS: Mapping[str, tuple[FieldSpec, ...]] = {
    "source_bindings": SOURCE_FIELDS,
    "source_artifacts": SOURCE_ARTIFACT_FIELDS,
    "status": STATUS_FIELDS,
    "metrics": METRIC_FIELDS,
    "context_metrics": METRIC_FIELDS,
    "contrasts": CONTRAST_FIELDS,
    "context_contrasts": CONTRAST_FIELDS,
    "coverage": COVERAGE_FIELDS,
    "context_coverage": COVERAGE_FIELDS,
    "localization_decisions": LOCALIZATION_DECISION_FIELDS,
    "context_localization_decisions": LOCALIZATION_DECISION_FIELDS,
    "winner_reference_sets": WINNER_SET_FIELDS,
    "context_winner_reference_sets": WINNER_SET_FIELDS,
    "winner_reference_contrasts": WINNER_CONTRAST_FIELDS,
    "context_winner_reference_contrasts": WINNER_CONTRAST_FIELDS,
}

STATUS_CLASSES = frozenset(
    {
        "OK", "CONTEXT", "PARTIAL", "UNDEFINED", "BLOCKED",
        "NOT_APPLICABLE", "NOT_CERTIFIED", "UNVERIFIED",
    }
)
REPORT_PARTITIONS = frozenset({"primary", "context"})
WINNER_MEMBERSHIP_STATUSES = frozenset(
    {
        "POINT_WINNER",
        "NOT_SEPARATED_FROM_POINT_WINNER_95CI",
        "SEPARATED_FROM_POINT_WINNER_95CI",
    }
)
COMPARISON_SIGNATURE_FIELDS = (
    "lane_id", "task_id", "dataset_id", "population_id", "cell_id",
    "slice_id", "prediction_unit", "estimand_id", "metric_id",
    "metric_unit", "positive_class", "better_direction", "cohort_id",
    "access_level", "supervision", "fidelity", "report_partition",
    "aggregation_id", "aggregation_level", "aggregation_rule",
    "aggregation_unit",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derive_comparison_group_id(row: Mapping[str, Any]) -> str:
    missing = [field for field in COMPARISON_SIGNATURE_FIELDS if field not in row]
    if missing:
        raise UnifiedReportingError(f"comparison signature missing fields: {missing}")
    signature = {field: row[field] for field in COMPARISON_SIGNATURE_FIELDS}
    return f"cmpv1_{canonical_sha256(signature)[:24]}"


def _normalize_value(spec: FieldSpec, value: Any, *, where: str) -> Any:
    if value is None:
        if spec.nullable:
            return None
        raise UnifiedReportingError(f"{where}.{spec.name} is required")
    if spec.kind == "string":
        if not isinstance(value, str) or not value:
            raise UnifiedReportingError(f"{where}.{spec.name} must be a non-empty string")
        return value
    if spec.kind == "bool":
        if type(value) is not bool:
            raise UnifiedReportingError(f"{where}.{spec.name} must be boolean")
        return value
    if spec.kind == "int64":
        if type(value) is bool or not isinstance(value, int):
            raise UnifiedReportingError(f"{where}.{spec.name} must be an integer")
        return value
    if spec.kind == "float64":
        if type(value) is bool or not isinstance(value, (int, float)):
            raise UnifiedReportingError(f"{where}.{spec.name} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise UnifiedReportingError(f"{where}.{spec.name} must be finite")
        return result
    raise UnifiedReportingError(f"unknown field kind: {spec.kind}")


def validate_row(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if table not in TABLE_SCHEMAS:
        raise UnifiedReportingError(f"unknown unified reporting table: {table}")
    specs = TABLE_SCHEMAS[table]
    expected = {spec.name for spec in specs}
    missing = sorted(expected - set(row))
    extra = sorted(set(row) - expected)
    if missing or extra:
        raise UnifiedReportingError(
            f"{table} field drift: missing={missing}, extra={extra}"
        )
    normalized = {
        spec.name: _normalize_value(spec, row[spec.name], where=table)
        for spec in specs
    }
    if "report_partition" in normalized and normalized["report_partition"] not in REPORT_PARTITIONS:
        raise UnifiedReportingError("report_partition must be primary or context")
    if "status_class" in normalized and normalized["status_class"] not in STATUS_CLASSES:
        raise UnifiedReportingError(f"unknown status_class: {normalized['status_class']}")
    if "rankable" in normalized:
        allowed = (
            normalized.get("status_class") == "OK"
            and normalized.get("report_partition") == "primary"
        )
        if normalized["rankable"] and not allowed:
            raise UnifiedReportingError("rankable row is not primary/OK")
    if table in {"metrics", "context_metrics", "contrasts", "context_contrasts"}:
        expected_group = derive_comparison_group_id(normalized)
        if normalized["comparison_group_id"] != expected_group:
            raise UnifiedReportingError("comparison_group_id is not content-addressed")
        numeric_field = "value" if "metrics" in table else "delta"
        if normalized["rankable"] and normalized[numeric_field] is None:
            raise UnifiedReportingError("rankable result requires a numeric value")
        if normalized["status_class"] in {"BLOCKED", "UNDEFINED", "UNVERIFIED"}:
            if normalized[numeric_field] is not None:
                raise UnifiedReportingError("blocked/undefined/unverified result must be null")
    if table in {"localization_decisions", "context_localization_decisions"}:
        for field in ("predicted_first_error", "true_first_error"):
            if normalized[field] < -1:
                raise UnifiedReportingError(f"{field} must be -1 or a non-negative index")
        if normalized["fold"] < 0:
            raise UnifiedReportingError("localization fold must be non-negative")
    if table in {"winner_reference_sets", "context_winner_reference_sets"}:
        if normalized["equivalence_claim"] or normalized["simultaneous_coverage"]:
            raise UnifiedReportingError("winner-reference membership is not equivalence/simultaneous inference")
        if normalized["winner_selection_adjusted"]:
            raise UnifiedReportingError("winner-reference inference is not selection adjusted")
        if normalized["multiplicity_adjustment"] != "NONE":
            raise UnifiedReportingError("winner-reference multiplicity must remain NONE")
        if normalized["interpretation_code"] != "DIRECT_PAIRED_NONSEPARATION_95":
            raise UnifiedReportingError("winner-reference interpretation drift")
        if normalized["membership_status"] not in WINNER_MEMBERSHIP_STATUSES:
            raise UnifiedReportingError("winner-reference membership is not a direct-winner status")
        expected_membership = normalized["membership_status"] != "SEPARATED_FROM_POINT_WINNER_95CI"
        if normalized["in_winner_reference_set"] is not expected_membership:
            raise UnifiedReportingError("winner-reference membership flag/status mismatch")
        is_reference = normalized["method_id"] == normalized["winner_reference_method_id"]
        if (normalized["membership_status"] == "POINT_WINNER") is not is_reference:
            raise UnifiedReportingError("POINT_WINNER must identify exactly the reference method")
    if table in {"winner_reference_contrasts", "context_winner_reference_contrasts"}:
        if normalized["equivalence_claim"]:
            raise UnifiedReportingError("winner-reference contrast cannot claim equivalence")
        if normalized["multiplicity_adjustment"] != "NONE":
            raise UnifiedReportingError("winner-reference multiplicity must remain NONE")
        if normalized["membership_status"] not in WINNER_MEMBERSHIP_STATUSES:
            raise UnifiedReportingError("winner-reference contrast is not a direct-winner status")
        expected_membership = normalized["membership_status"] != "SEPARATED_FROM_POINT_WINNER_95CI"
        if normalized["in_winner_reference_set"] is not expected_membership:
            raise UnifiedReportingError("winner-reference contrast flag/status mismatch")
        is_reference = normalized["candidate_method_id"] == normalized["winner_reference_method_id"]
        if (normalized["membership_status"] == "POINT_WINNER") is not is_reference:
            raise UnifiedReportingError("POINT_WINNER contrast must be the reference self-contrast")
    primary_tables = {
        "metrics", "contrasts", "coverage", "localization_decisions",
        "winner_reference_sets", "winner_reference_contrasts",
    }
    context_tables = {
        "context_metrics", "context_contrasts", "context_coverage",
        "context_localization_decisions", "context_winner_reference_sets",
        "context_winner_reference_contrasts",
    }
    if table in primary_tables and normalized["report_partition"] != "primary":
        raise UnifiedReportingError(f"{table} may contain only primary rows")
    if table in context_tables and normalized["report_partition"] != "context":
        raise UnifiedReportingError(f"{table} may contain only context rows")
    return normalized


def validate_rows(table: str, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        try:
            output.append(validate_row(table, row))
        except UnifiedReportingError as exc:
            raise UnifiedReportingError(f"{table} row {index}: {exc}") from exc
    return sorted(output, key=canonical_json_bytes)


def table_logical_sha256(table: str, rows: Iterable[Mapping[str, Any]]) -> str:
    return canonical_sha256({"schema": SCHEMA_VERSION, "table": table, "rows": validate_rows(table, rows)})


def _encode_csv(spec: FieldSpec, value: Any) -> str:
    if value is None:
        return ""
    if spec.kind == "bool":
        return "true" if value else "false"
    if spec.kind == "float64":
        return format(float(value), ".17g")
    return str(value)


def csv_bytes(table: str, rows: Iterable[Mapping[str, Any]]) -> tuple[bytes, list[dict[str, Any]]]:
    normalized = validate_rows(table, rows)
    specs = TABLE_SCHEMAS[table]
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=[spec.name for spec in specs], extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in normalized:
        writer.writerow({spec.name: _encode_csv(spec, row[spec.name]) for spec in specs})
    return stream.getvalue().encode("utf-8"), normalized


def _decode_csv(spec: FieldSpec, value: str) -> Any:
    if value == "":
        if spec.nullable:
            return None
        raise UnifiedReportingError(f"non-nullable CSV field {spec.name} is empty")
    if spec.kind == "string":
        return value
    if spec.kind == "bool":
        if value not in {"true", "false"}:
            raise UnifiedReportingError(f"invalid boolean for {spec.name}: {value!r}")
        return value == "true"
    if spec.kind == "int64":
        try:
            return int(value)
        except ValueError as exc:
            raise UnifiedReportingError(f"invalid integer for {spec.name}: {value!r}") from exc
    if spec.kind == "float64":
        try:
            result = float(value)
        except ValueError as exc:
            raise UnifiedReportingError(f"invalid float for {spec.name}: {value!r}") from exc
        if not math.isfinite(result):
            raise UnifiedReportingError(f"non-finite float for {spec.name}")
        return result
    raise UnifiedReportingError(f"unknown field kind: {spec.kind}")


def read_csv_bytes(table: str, payload: bytes) -> list[dict[str, Any]]:
    specs = TABLE_SCHEMAS[table]
    reader = csv.DictReader(StringIO(payload.decode("utf-8"), newline=""))
    expected = [spec.name for spec in specs]
    if reader.fieldnames != expected:
        raise UnifiedReportingError(
            f"{table} CSV header drift: expected={expected}, observed={reader.fieldnames}"
        )
    return validate_rows(
        table,
        ({spec.name: _decode_csv(spec, row[spec.name]) for spec in specs} for row in reader),
    )


def _arrow_type(kind: str) -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise UnifiedReportingError("PyArrow is required for unified reporting Parquet") from exc
    return {
        "string": pa.string(), "bool": pa.bool_(), "int64": pa.int64(),
        "float64": pa.float64(),
    }[kind]


def parquet_bytes(table: str, rows: Iterable[Mapping[str, Any]]) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise UnifiedReportingError("PyArrow is required for unified reporting Parquet") from exc
    normalized = validate_rows(table, rows)
    specs = TABLE_SCHEMAS[table]
    schema = pa.schema(
        [pa.field(spec.name, _arrow_type(spec.kind), nullable=spec.nullable) for spec in specs],
        metadata={b"reconstruction_schema": SCHEMA_VERSION.encode(), b"table": table.encode()},
    )
    arrow_table = pa.Table.from_pylist(normalized, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(
        arrow_table, sink, compression="zstd", use_dictionary=False,
        write_statistics=True, data_page_version="1.0",
    )
    return sink.getvalue().to_pybytes(), normalized


def read_parquet_bytes(table: str, payload: bytes) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise UnifiedReportingError("PyArrow is required for unified reporting Parquet") from exc
    rows = pq.read_table(BytesIO(payload)).to_pylist()
    return validate_rows(table, rows)


def assert_csv_parquet_parity(table: str, csv_payload: bytes, parquet_payload: bytes) -> str:
    csv_rows = read_csv_bytes(table, csv_payload)
    parquet_rows = read_parquet_bytes(table, parquet_payload)
    if csv_rows != parquet_rows:
        raise UnifiedReportingError(f"{table} CSV/Parquet logical mismatch")
    return table_logical_sha256(table, csv_rows)


def schema_bundle() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tables": {
            table: [
                {"name": spec.name, "kind": spec.kind, "nullable": spec.nullable}
                for spec in specs
            ]
            for table, specs in sorted(TABLE_SCHEMAS.items())
        },
        "comparison_signature_fields": list(COMPARISON_SIGNATURE_FIELDS),
        "status_classes": sorted(STATUS_CLASSES),
    }


__all__ = [
    "COMPARISON_SIGNATURE_FIELDS", "REPORT_PARTITIONS", "SCHEMA_VERSION",
    "STATUS_CLASSES", "TABLE_SCHEMAS", "WINNER_MEMBERSHIP_STATUSES", "UnifiedReportingError",
    "assert_csv_parquet_parity", "canonical_json_bytes", "canonical_sha256",
    "csv_bytes", "derive_comparison_group_id", "parquet_bytes",
    "read_csv_bytes", "read_parquet_bytes", "schema_bundle",
    "table_logical_sha256", "validate_row", "validate_rows",
]
