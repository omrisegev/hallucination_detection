"""Normalize certified reconstruction lanes into one comparison-safe bridge."""

from __future__ import annotations

from collections import defaultdict
import csv
from io import StringIO
import json
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .unified_reporting_schemas import (
    UnifiedReportingError,
    canonical_json_bytes,
    canonical_sha256,
    derive_comparison_group_id,
    validate_rows,
)
from .unified_reporting_sources import AuthenticatedSource


BRIDGE_SCHEMA = "reconstruction-unified-reporting-bridge-v1"
NO_VALUE = "__all__"


def _csv(payload: bytes) -> list[dict[str, str]]:
    try:
        return list(csv.DictReader(StringIO(payload.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise UnifiedReportingError(f"invalid certified CSV: {exc}") from exc


def _payload_json(payload: bytes, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnifiedReportingError(f"invalid certified JSON in {where}") from exc
    if not isinstance(value, dict):
        raise UnifiedReportingError(f"{where} must be a JSON object")
    if "payload_sha256" in value:
        body = dict(value)
        observed = body.pop("payload_sha256")
        if observed != canonical_sha256(body):
            raise UnifiedReportingError(f"{where} payload hash mismatch")
    return value


def _opt_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value)
    return result if result else None


def _text(value: Any, default: str = NO_VALUE) -> str:
    result = _opt_text(value)
    return default if result is None else result


def _int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if type(value) is bool:
        raise UnifiedReportingError("boolean cannot be coerced to integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise UnifiedReportingError(f"invalid integer: {value!r}") from exc
    if str(result) != str(value) and not isinstance(value, int):
        raise UnifiedReportingError(f"lossy integer conversion: {value!r}")
    return result


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if type(value) is bool:
        raise UnifiedReportingError("boolean cannot be coerced to float")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise UnifiedReportingError(f"invalid float: {value!r}") from exc


def _bool(value: Any) -> bool:
    if type(value) is bool:
        return value
    if value in ("true", "True", "1", 1):
        return True
    if value in ("false", "False", "0", 0):
        return False
    raise UnifiedReportingError(f"invalid boolean: {value!r}")


def _ns(lane_id: str, kind: str, value: Any) -> str:
    return f"{lane_id}::{kind}::{_text(value)}"


def _status_class(source_status: Any, *, context: bool = False) -> str:
    status = _text(source_status).upper()
    if status in {"OK", "OK_FALLBACK", "PASS", "READY", "OK_AGGREGATED"}:
        return "CONTEXT" if context else "OK"
    if status == "CONTEXT_ONLY":
        return "CONTEXT"
    if "PARTIAL" in status or "INCOMPLETE" in status:
        return "PARTIAL"
    if "UNDEFINED" in status or "SINGLE_CLASS" in status:
        return "UNDEFINED"
    if status in {"NOT_APPLICABLE", "EXCLUDED_BY_PROTOCOL"}:
        return "NOT_APPLICABLE"
    if status == "NOT_CERTIFIED":
        return "NOT_CERTIFIED"
    if status == "UNVERIFIED":
        return "UNVERIFIED"
    if any(
        token in status
        for token in (
            "BLOCKED", "FAILED", "INVALID", "INCOMPATIBLE", "NOT_AGGREGATED",
            "DISABLED", "MISSING", "GATE_FAILED", "ADAPTER_MISSING", "NOT_RUN",
        )
    ):
        return "BLOCKED"
    raise UnifiedReportingError(f"unmapped producer status: {source_status!r}")


def _direction(metric_id: str, explicit: Any = None) -> str:
    if explicit not in (None, ""):
        value = str(explicit).lower()
        if value in {"higher", "lower"}:
            return value
    metric = metric_id.lower()
    return "lower" if "aurc" in metric or metric.startswith("mean_") or metric.endswith("failure_rate") else "higher"


def _metric_unit(metric_id: str, explicit: Any = None) -> str:
    if explicit not in (None, ""):
        return str(explicit)
    return "x1000" if "x1000" in metric_id.lower() else "unit_interval"


def _provenance(source: AuthenticatedSource, table: str, locator: str, raw: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source_binding_id": source.source_binding_id,
        "source_table": table,
        "source_row_locator": locator,
        "source_row_sha256": canonical_sha256(raw),
    }


def _lane(contract: Mapping[str, Any], source_id: str) -> Mapping[str, Any]:
    try:
        return contract["lanes"][source_id]
    except KeyError as exc:
        raise UnifiedReportingError(f"contract has no lane for {source_id}") from exc


def _partition(
    contract: Mapping[str, Any], source_id: str, *, panel_role: str | None = None,
    access_level: str | None = None,
) -> str:
    lane = _lane(contract, source_id)
    if panel_role is not None:
        try:
            return str(contract["access_partitions"]["external_final_answer"][panel_role])
        except KeyError as exc:
            raise UnifiedReportingError(f"unmapped external panel role: {panel_role}") from exc
    if access_level is not None and source_id == "localization_v1":
        try:
            return str(contract["access_partitions"]["localization"][access_level])
        except KeyError as exc:
            raise UnifiedReportingError(f"unmapped localization access: {access_level}") from exc
    return str(lane["report_partition"])


def _scope(
    *, release_id: str, lane_id: str, task_id: str, source_dataset_id: str,
    population_id: Any, cell_id: Any, slice_id: Any, prediction_unit: str,
    estimand_id: str, access_level: str, supervision: str, fidelity: str,
    report_partition: str,
) -> dict[str, Any]:
    return {
        "release_id": release_id,
        "lane_id": lane_id,
        "task_id": task_id,
        "source_dataset_id": source_dataset_id,
        "dataset_id": _ns(lane_id, "dataset", source_dataset_id),
        "population_id": _ns(lane_id, "population", population_id),
        "cell_id": _ns(lane_id, "cell", cell_id),
        "slice_id": _ns(lane_id, "slice", slice_id),
        "prediction_unit": prediction_unit,
        "estimand_id": estimand_id,
        "access_level": access_level,
        "supervision": supervision,
        "fidelity": fidelity,
        "report_partition": report_partition,
    }


def _aggregation(level: str, aggregation_id: Any = None) -> tuple[str, str, str]:
    level = _text(level, "cell")
    aggregation_id = _text(aggregation_id, f"native_{level}")
    units = {
        "cell": "source_row", "dataset": "cell", "population": "cell",
        "task": "dataset", "release": "task", "context": "native",
    }
    return aggregation_id, ("native_metric" if level == "cell" else "producer_declared"), units.get(level, "native")


def _metric_row(
    *, scope: Mapping[str, Any], system_id: str, method_id: str,
    metric_id: str, metric_unit: str, positive_class: str, better_direction: str,
    aggregation_id: str, aggregation_level: str, aggregation_rule: str,
    aggregation_unit: str, cohort_id: str, source_comparison_group_id: str | None,
    value: float | None, ci_low: float | None, ci_high: float | None,
    n_rows: int | None, n_groups: int | None, n_positive: int | None,
    n_negative: int | None, bootstrap_unit: str | None,
    bootstrap_draws: int | None, source_status: str, status_detail: str | None,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    status_class = _status_class(source_status, context=scope["report_partition"] == "context")
    row = {
        **scope,
        "system_id": _ns(scope["lane_id"], "system", system_id),
        "method_id": method_id,
        "metric_id": metric_id,
        "metric_unit": metric_unit,
        "positive_class": positive_class,
        "better_direction": better_direction,
        "aggregation_id": aggregation_id,
        "aggregation_level": aggregation_level,
        "aggregation_rule": aggregation_rule,
        "aggregation_unit": aggregation_unit,
        "cohort_id": cohort_id,
        "comparison_group_id": "pending",
        "source_comparison_group_id": source_comparison_group_id,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_rows": n_rows,
        "n_groups": n_groups,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "bootstrap_unit": bootstrap_unit,
        "bootstrap_draws": bootstrap_draws,
        "status_class": status_class,
        "source_status": source_status,
        "status_detail": status_detail,
        "rankable": status_class == "OK" and scope["report_partition"] == "primary" and value is not None,
        **provenance,
    }
    row["comparison_group_id"] = derive_comparison_group_id(row)
    return row


def _contrast_row(
    *, scope: Mapping[str, Any], left_system: str, right_system: str,
    left_method: str, right_method: str, metric_id: str, metric_unit: str,
    positive_class: str, better_direction: str, aggregation_id: str,
    aggregation_level: str, aggregation_rule: str, aggregation_unit: str,
    cohort_id: str, source_comparison_group_id: str | None,
    delta: float | None, ci_low: float | None, ci_high: float | None,
    n_pairs: int | None, bootstrap_unit: str | None, bootstrap_draws: int | None,
    paired: bool, source_status: str, status_detail: str | None,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    status_class = _status_class(source_status, context=scope["report_partition"] == "context")
    row = {
        **scope,
        "left_system_id": _ns(scope["lane_id"], "system", left_system),
        "right_system_id": _ns(scope["lane_id"], "system", right_system),
        "left_method_id": left_method,
        "right_method_id": right_method,
        "metric_id": metric_id,
        "metric_unit": metric_unit,
        "positive_class": positive_class,
        "better_direction": better_direction,
        "aggregation_id": aggregation_id,
        "aggregation_level": aggregation_level,
        "aggregation_rule": aggregation_rule,
        "aggregation_unit": aggregation_unit,
        "cohort_id": cohort_id,
        "comparison_group_id": "pending",
        "source_comparison_group_id": source_comparison_group_id,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_pairs": n_pairs,
        "bootstrap_unit": bootstrap_unit,
        "bootstrap_draws": bootstrap_draws,
        "paired": paired,
        "status_class": status_class,
        "source_status": source_status,
        "status_detail": status_detail,
        "rankable": status_class == "OK" and scope["report_partition"] == "primary" and delta is not None,
        **provenance,
    }
    row["comparison_group_id"] = derive_comparison_group_id(row)
    return row


def _generic_metric(
    release_id: str, source: AuthenticatedSource, lane: Mapping[str, Any],
    raw: Mapping[str, Any], locator: str,
) -> dict[str, Any]:
    partition = str(lane["report_partition"])
    access = _text(raw.get("access_contract_id"), "producer_registered_access")
    scope = _scope(
        release_id=release_id, lane_id=str(lane["lane_id"]),
        task_id=_text(raw.get("task_id"), str(lane["task_id"])),
        source_dataset_id=_text(raw.get("dataset_id")),
        population_id=raw.get("population_id"), cell_id=raw.get("cell_id"),
        slice_id=raw.get("slice_id"), prediction_unit=str(lane["default_prediction_unit"]),
        estimand_id=str(lane["default_estimand_id"]), access_level=access,
        supervision="unsupervised", fidelity=_text(raw.get("fidelity"), "producer_certified"),
        report_partition=partition,
    )
    level = _text(raw.get("aggregation_level"), "cell")
    aid, rule, unit = _aggregation(level, raw.get("aggregation_id"))
    return _metric_row(
        scope=scope, system_id=_text(raw.get("system_id")), method_id=_text(raw.get("method_id")),
        metric_id=_text(raw.get("metric_id")), metric_unit=_metric_unit(_text(raw.get("metric_id")), raw.get("metric_unit")),
        positive_class=_text(raw.get("positive_class"), "incorrect"),
        better_direction=_direction(_text(raw.get("metric_id")), raw.get("better_direction")),
        aggregation_id=aid, aggregation_level=level, aggregation_rule=rule,
        aggregation_unit=unit, cohort_id=_text(raw.get("cohort_id"), "producer_cohort_not_reported"),
        source_comparison_group_id=_opt_text(raw.get("comparison_group_id")),
        value=_float(raw.get("value")), ci_low=_float(raw.get("ci_low")),
        ci_high=_float(raw.get("ci_high")), n_rows=_int(raw.get("n_rows")),
        n_groups=_int(raw.get("n_groups")), n_positive=_int(raw.get("n_positive")),
        n_negative=_int(raw.get("n_negative")), bootstrap_unit=_opt_text(raw.get("bootstrap_unit")),
        bootstrap_draws=_int(raw.get("bootstrap_draws")), source_status=_text(raw.get("status")),
        status_detail=_opt_text(raw.get("status_detail")),
        provenance=_provenance(source, "metrics", locator, raw),
    )


def _generic_contrast(
    release_id: str, source: AuthenticatedSource, lane: Mapping[str, Any],
    raw: Mapping[str, Any], locator: str, method_by_system: Mapping[str, str],
) -> dict[str, Any]:
    access = _text(raw.get("access_contract_id"), "producer_registered_access")
    scope = _scope(
        release_id=release_id, lane_id=str(lane["lane_id"]),
        task_id=_text(raw.get("task_id"), str(lane["task_id"])),
        source_dataset_id=_text(raw.get("dataset_id")), population_id=raw.get("population_id"),
        cell_id=raw.get("cell_id"), slice_id=raw.get("slice_id"),
        prediction_unit=str(lane["default_prediction_unit"]),
        estimand_id=str(lane["default_estimand_id"]), access_level=access,
        supervision="unsupervised", fidelity=_text(raw.get("fidelity"), "producer_certified"),
        report_partition=str(lane["report_partition"]),
    )
    level = _text(raw.get("aggregation_level"), "cell")
    aid, rule, unit = _aggregation(level, raw.get("aggregation_id"))
    left, right = _text(raw.get("left_system_id")), _text(raw.get("right_system_id"))
    return _contrast_row(
        scope=scope, left_system=left, right_system=right,
        left_method=method_by_system.get(left, left), right_method=method_by_system.get(right, right),
        metric_id=_text(raw.get("metric_id")), metric_unit=_metric_unit(_text(raw.get("metric_id")), raw.get("metric_unit")),
        positive_class=_text(raw.get("positive_class"), "incorrect"),
        better_direction=_direction(_text(raw.get("metric_id")), raw.get("better_direction")),
        aggregation_id=aid, aggregation_level=level, aggregation_rule=rule,
        aggregation_unit=unit, cohort_id=_text(raw.get("cohort_id"), "producer_cohort_not_reported"),
        source_comparison_group_id=_opt_text(raw.get("comparison_group_id")),
        delta=_float(raw.get("delta")), ci_low=_float(raw.get("ci_low")),
        ci_high=_float(raw.get("ci_high")), n_pairs=_int(raw.get("n_pairs")),
        bootstrap_unit=_opt_text(raw.get("bootstrap_unit")), bootstrap_draws=_int(raw.get("bootstrap_draws")),
        paired=_bool(raw.get("paired", True)), source_status=_text(raw.get("status")),
        status_detail=_opt_text(raw.get("status_detail")),
        provenance=_provenance(source, "contrasts", locator, raw),
    )


def _generic_coverage(
    release_id: str, source: AuthenticatedSource, lane: Mapping[str, Any],
    raw: Mapping[str, Any], locator: str,
) -> dict[str, Any]:
    scope = _scope(
        release_id=release_id, lane_id=str(lane["lane_id"]),
        task_id=_text(raw.get("task_id"), str(lane["task_id"])),
        source_dataset_id=_text(raw.get("dataset_id")), population_id=raw.get("population_id"),
        cell_id=raw.get("cell_id"), slice_id=raw.get("slice_id"),
        prediction_unit=str(lane["default_prediction_unit"]), estimand_id=str(lane["default_estimand_id"]),
        access_level=_text(raw.get("access_contract_id"), "producer_registered_access"),
        supervision="unsupervised", fidelity="producer_certified_coverage",
        report_partition=str(lane["report_partition"]),
    )
    return {
        **scope,
        "system_id": _ns(scope["lane_id"], "system", raw.get("system_id")),
        "method_id": _text(raw.get("method_id")),
        "expected_n": _int(raw.get("expected_n")), "eligible_n": _int(raw.get("eligible_n")),
        "scored_n": _int(raw.get("scored_n")), "fallback_n": _int(raw.get("fallback_n")),
        "excluded_n": _int(raw.get("excluded_n")), "failed_n": _int(raw.get("failed_n")),
        "coverage_fraction": _float(raw.get("coverage_fraction")),
        "cohort_id": _text(raw.get("cohort_id"), "producer_cohort_not_reported"),
        "status_class": _status_class(raw.get("status"), context=scope["report_partition"] == "context"),
        "source_status": _text(raw.get("status")), "status_detail": _opt_text(raw.get("status_detail")),
        **_provenance(source, "coverage", locator, raw),
    }


def _normalize_generic(
    release_id: str, source: AuthenticatedSource, lane: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_source = _csv(source.files["metrics"])
    metrics = [_generic_metric(release_id, source, lane, raw, f"row:{index + 2}") for index, raw in enumerate(metric_source)]
    method_by_system = {
        _text(raw.get("system_id")): _text(raw.get("method_id")) for raw in metric_source
    }
    contrasts = [
        _generic_contrast(release_id, source, lane, raw, f"row:{index + 2}", method_by_system)
        for index, raw in enumerate(_csv(source.files["contrasts"]))
    ]
    coverage = [
        _generic_coverage(release_id, source, lane, raw, f"row:{index + 2}")
        for index, raw in enumerate(_csv(source.files["coverage"]))
    ]
    return metrics, contrasts, coverage


def _normalize_external(
    release_id: str, source: AuthenticatedSource, contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lane = _lane(contract, source.source_id)
    metrics = []
    for index, raw in enumerate(_csv(source.files["metrics"])):
        partition = _partition(contract, source.source_id, panel_role=_text(raw.get("panel_role")))
        level = "population" if raw.get("record_level") == "population" else "cell"
        aid, rule, unit = _aggregation(level, raw.get("aggregate_weighting") or "single_cell")
        scope = _scope(
            release_id=release_id, lane_id=str(lane["lane_id"]), task_id=str(lane["task_id"]),
            source_dataset_id=_text(raw.get("dataset_id")), population_id=raw.get("population_id"),
            cell_id=raw.get("cell_id"), slice_id=raw.get("slice_id"),
            prediction_unit=str(lane["default_prediction_unit"]), estimand_id=str(lane["default_estimand_id"]),
            access_level="gray_box_single_pass", supervision="unsupervised",
            fidelity=f"external_v3::{_text(raw.get('panel_role'))}", report_partition=partition,
        )
        metric_id = _text(raw.get("metric_id"))
        metrics.append(_metric_row(
            scope=scope, system_id=_text(raw.get("method_id")), method_id=_text(raw.get("method_id")),
            metric_id=metric_id, metric_unit=_metric_unit(metric_id), positive_class="incorrect",
            better_direction=_direction(metric_id), aggregation_id=aid, aggregation_level=level,
            aggregation_rule=rule, aggregation_unit=unit,
            cohort_id=_text(raw.get("cohort_id"), "producer_cohort_not_reported"),
            source_comparison_group_id=_opt_text(raw.get("comparison_group_id")), value=_float(raw.get("value")),
            ci_low=_float(raw.get("ci_low")), ci_high=_float(raw.get("ci_high")), n_rows=_int(raw.get("n")),
            n_groups=_int(raw.get("n_groups")), n_positive=_int(raw.get("n_incorrect")),
            n_negative=_int(raw.get("n_correct")), bootstrap_unit=_opt_text(raw.get("bootstrap_unit")),
            bootstrap_draws=_int(raw.get("bootstrap_draws")), source_status=_text(raw.get("status")),
            status_detail=None, provenance=_provenance(source, "metrics", f"row:{index + 2}", raw),
        ))
    contrasts = []
    for index, raw in enumerate(_csv(source.files["contrasts"])):
        partition = _partition(contract, source.source_id, panel_role=_text(raw.get("panel_role")))
        level = "population" if raw.get("record_level") == "population" else "cell"
        aid, rule, unit = _aggregation(level, raw.get("aggregate_weighting") or "single_cell")
        scope = _scope(
            release_id=release_id, lane_id=str(lane["lane_id"]), task_id=str(lane["task_id"]),
            source_dataset_id=_text(raw.get("dataset_id")), population_id=raw.get("population_id"),
            cell_id=raw.get("cell_id"), slice_id=raw.get("slice_id"),
            prediction_unit=str(lane["default_prediction_unit"]), estimand_id=str(lane["default_estimand_id"]),
            access_level="gray_box_single_pass", supervision="unsupervised",
            fidelity=f"external_v3::{_text(raw.get('panel_role'))}", report_partition=partition,
        )
        metric_id = _text(raw.get("metric_id"))
        contrasts.append(_contrast_row(
            scope=scope, left_system=_text(raw.get("method_id")), right_system=_text(raw.get("reference_method_id")),
            left_method=_text(raw.get("method_id")), right_method=_text(raw.get("reference_method_id")),
            metric_id=metric_id, metric_unit=_metric_unit(metric_id), positive_class="incorrect",
            better_direction=_direction(metric_id, "higher" if _bool(raw.get("higher_is_better")) else "lower"),
            aggregation_id=aid, aggregation_level=level, aggregation_rule=rule, aggregation_unit=unit,
            cohort_id=_text(raw.get("cohort_id"), "producer_cohort_not_reported"),
            source_comparison_group_id=_opt_text(raw.get("comparison_group_id")), delta=_float(raw.get("delta")),
            ci_low=_float(raw.get("ci_low")), ci_high=_float(raw.get("ci_high")), n_pairs=_int(raw.get("n")),
            bootstrap_unit=_opt_text(raw.get("bootstrap_unit")), bootstrap_draws=_int(raw.get("bootstrap_draws")),
            paired=True, source_status=_text(raw.get("status")), status_detail=None,
            provenance=_provenance(source, "contrasts", f"row:{index + 2}", raw),
        ))
    return metrics, contrasts, []


def _localization_scope(
    release_id: str, lane: Mapping[str, Any], contract: Mapping[str, Any], raw: Mapping[str, Any]
) -> dict[str, Any]:
    task_id = _text(raw.get("task_id"))
    access = _text(raw.get("access_level"))
    # Cross-access producer contrasts are retained for auditability but cannot
    # enter a primary comparison set.  Their composite access label is created
    # by this bridge, so it is intentionally outside the producer access map.
    partition = (
        "context"
        if access.startswith("incompatible::")
        else _partition(contract, "localization_v1", access_level=access)
    )
    prediction_unit = "trace_first_error_index" if task_id == "first_error_localization" else "step"
    estimand = "first_error_index_decision" if task_id == "first_error_localization" else "annotated_error_step_ranking"
    supervision = "supervised_or_judge" if access != "saved_output_probability_telemetry_one_pass" else "unsupervised"
    return _scope(
        release_id=release_id, lane_id=str(lane["lane_id"]), task_id=task_id,
        source_dataset_id=_text(raw.get("dataset_id")), population_id=raw.get("population_id"),
        cell_id=raw.get("cell_id"), slice_id=raw.get("slice_id"), prediction_unit=prediction_unit,
        estimand_id=estimand, access_level=access, supervision=supervision,
        fidelity=_text(raw.get("fidelity")), report_partition=partition,
    )


def _normalize_localization(
    release_id: str, source: AuthenticatedSource, contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lane = _lane(contract, source.source_id)
    metrics = []
    for index, raw in enumerate(_csv(source.files["metrics"])):
        scope = _localization_scope(release_id, lane, contract, raw)
        metric_id = _text(raw.get("metric_id"))
        metrics.append(_metric_row(
            scope=scope, system_id=_text(raw.get("system_id")), method_id=_text(raw.get("system_id")),
            metric_id=metric_id, metric_unit="unit_interval", positive_class="annotated_error",
            better_direction="higher", aggregation_id="native_cell", aggregation_level="cell",
            aggregation_rule="native_metric", aggregation_unit="source_row",
            cohort_id=_text(raw.get("cohort_id"), "producer_cohort_not_reported"),
            source_comparison_group_id=_opt_text(raw.get("comparison_group_id")), value=_float(raw.get("value")),
            ci_low=_float(raw.get("ci_low")), ci_high=_float(raw.get("ci_high")),
            n_rows=_int(raw.get("n_examples")), n_groups=_int(raw.get("n_examples")),
            n_positive=_int(raw.get("n_positive")), n_negative=_int(raw.get("n_negative")),
            bootstrap_unit=_opt_text(raw.get("bootstrap_unit")), bootstrap_draws=_int(raw.get("bootstrap_draws")),
            source_status=_text(raw.get("status")), status_detail=None,
            provenance=_provenance(source, "metrics", f"row:{index + 2}", raw),
        ))
    contrasts = []
    for index, raw in enumerate(_csv(source.files["contrasts"])):
        candidate_access = _text(raw.get("candidate_access_level"))
        reference_access = _text(raw.get("reference_access_level"))
        candidate_fidelity = _text(raw.get("candidate_fidelity"))
        reference_fidelity = _text(raw.get("reference_fidelity"))
        composite = dict(raw)
        if candidate_access == reference_access and candidate_fidelity == reference_fidelity:
            composite["access_level"] = candidate_access
            composite["fidelity"] = candidate_fidelity
        else:
            composite["access_level"] = f"incompatible::{candidate_access}::{reference_access}"
            composite["fidelity"] = f"incompatible::{candidate_fidelity}::{reference_fidelity}"
        scope = _localization_scope(release_id, lane, contract, composite)
        if candidate_access != reference_access or candidate_fidelity != reference_fidelity:
            scope["report_partition"] = "context"
        metric_id = _text(raw.get("metric_id"))
        contrasts.append(_contrast_row(
            scope=scope, left_system=_text(raw.get("candidate_system_id")),
            right_system=_text(raw.get("reference_system_id")), left_method=_text(raw.get("candidate_system_id")),
            right_method=_text(raw.get("reference_system_id")), metric_id=metric_id,
            metric_unit="unit_interval", positive_class="annotated_error", better_direction="higher",
            aggregation_id="native_cell", aggregation_level="cell", aggregation_rule="native_metric",
            aggregation_unit="source_row", cohort_id=_text(raw.get("cohort_id"), "producer_cohort_not_reported"),
            source_comparison_group_id=_opt_text(raw.get("comparison_group_id")), delta=_float(raw.get("delta")),
            ci_low=_float(raw.get("ci_low")), ci_high=_float(raw.get("ci_high")), n_pairs=_int(raw.get("n_valid")),
            bootstrap_unit=_opt_text(raw.get("bootstrap_unit")), bootstrap_draws=_int(raw.get("bootstrap_draws")),
            paired=True, source_status=_text(raw.get("status")), status_detail=None,
            provenance=_provenance(source, "contrasts", f"row:{index + 2}", raw),
        ))
    coverage = []
    for index, raw in enumerate(_csv(source.files["coverage"])):
        scope = _localization_scope(release_id, lane, contract, raw)
        expected, scored = _int(raw.get("n_expected")), _int(raw.get("n_scored"))
        coverage.append({
            **scope, "system_id": _ns(scope["lane_id"], "system", raw.get("system_id")),
            "method_id": _text(raw.get("system_id")), "expected_n": expected,
            "eligible_n": None, "scored_n": scored, "fallback_n": _int(raw.get("n_fallback")),
            "excluded_n": _int(raw.get("n_excluded")), "failed_n": _int(raw.get("n_failed")),
            "coverage_fraction": (None if expected in (None, 0) or scored is None else scored / expected),
            "cohort_id": _text(raw.get("cohort_id"), "producer_cohort_not_reported"),
            "status_class": _status_class(raw.get("status"), context=scope["report_partition"] == "context"),
            "source_status": _text(raw.get("status")), "status_detail": None,
            **_provenance(source, "coverage", f"row:{index + 2}", raw),
        })
    decisions = []
    for index, raw in enumerate(_csv(source.files["localization_decisions"])):
        scope = _localization_scope(release_id, lane, contract, raw)
        decision = {
            **scope, "model_id": _text(raw.get("model_id")),
            "system_id": _ns(scope["lane_id"], "system", raw.get("system_id")),
            "row_id": _text(raw.get("row_id")), "cohort_id": _text(raw.get("cohort_id")),
            "group_id": _text(raw.get("group_id")), "fold": _int(raw.get("fold")),
            # The certified producer uses an empty value only for explicitly
            # unscorable comparator rows.  Preserve those rows and encode the
            # documented abstention sentinel in the typed decision relation.
            "predicted_first_error": (
                -1 if raw.get("prediction_step") in (None, "") else _int(raw.get("prediction_step"))
            ),
            "true_first_error": _int(raw.get("true_first_error")),
            "comparison_group_id": "pending",
            "source_comparison_group_id": _opt_text(raw.get("comparison_group_id")),
            "source_status": _text(raw.get("status")),
            **_provenance(source, "localization_decisions", f"row:{index + 2}", raw),
        }
        signature = {
            field: decision.get(field, NO_VALUE)
            for field in (
                "lane_id", "task_id", "dataset_id", "population_id", "cell_id", "slice_id",
                "prediction_unit", "estimand_id", "cohort_id", "access_level", "supervision",
                "fidelity", "report_partition",
            )
        }
        decision["comparison_group_id"] = f"decisionv1_{canonical_sha256(signature)[:24]}"
        decisions.append(decision)
    return metrics, contrasts, coverage, decisions


def _normalize_prefix(
    release_id: str, source: AuthenticatedSource, contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lane = _lane(contract, source.source_id)
    metrics_payload = _payload_json(source.files["metrics"], where="prefix metrics")
    contrasts_payload = _payload_json(source.files["contrasts"], where="prefix contrasts")
    metrics = []
    population = _text(metrics_payload.get("population_id"), "processbench_llama31_prefix_v1")
    for section, rows in (("per_subset_rows", metrics_payload.get("per_subset_rows", [])), ("rows", metrics_payload.get("rows", []))):
        if not isinstance(rows, list):
            raise UnifiedReportingError(f"prefix {section} must be a list")
        for index, raw in enumerate(rows):
            aggregate = section == "rows"
            family = "all_registered_subsets" if aggregate else _text(raw.get("family"))
            budget = _text(raw.get("budget"))
            scope = _scope(
                release_id=release_id, lane_id=str(lane["lane_id"]), task_id=str(lane["task_id"]),
                source_dataset_id=f"processbench::{family}", population_id=population,
                cell_id=family, slice_id=f"budget_{budget}", prediction_unit=str(lane["default_prediction_unit"]),
                estimand_id=str(lane["default_estimand_id"]),
                access_level="saved_output_probability_prefix", supervision="unsupervised",
                fidelity="causal_early_fixed_trace", report_partition="primary",
            )
            metric_id = _text(raw.get("metric"))
            status = _text(raw.get("status"))
            metrics.append(_metric_row(
                scope=scope, system_id=_text(raw.get("method_id")), method_id=_text(raw.get("method_id")),
                metric_id=metric_id, metric_unit="unit_interval", positive_class="incorrect_final_answer",
                better_direction="higher", aggregation_id=("equal_subset_macro" if aggregate else "single_subset"),
                aggregation_level=("population" if aggregate else "cell"),
                aggregation_rule=("equal_unit_mean" if aggregate else "native_metric"),
                aggregation_unit=("subset" if aggregate else "source_trace"),
                cohort_id=f"prefix_budget_{budget}", source_comparison_group_id=None,
                value=_float(raw.get("point")), ci_low=_float(raw.get("ci_low")), ci_high=_float(raw.get("ci_high")),
                n_rows=_int(raw.get("n_traces")), n_groups=(None if aggregate else _int(raw.get("n_traces"))),
                n_positive=_int(raw.get("n_positive")), n_negative=_int(raw.get("n_negative")),
                bootstrap_unit=("source_question_within_subset" if aggregate else None),
                bootstrap_draws=_int(raw.get("bootstrap_draws")), source_status=status,
                status_detail=(None if status == "OK" else _text(raw.get("missing_subset_policy"), "producer_status")),
                provenance=_provenance(source, "metrics", f"/{section}/{index}", raw),
            ))
    contrasts = []
    rows = contrasts_payload.get("rows", [])
    if not isinstance(rows, list):
        raise UnifiedReportingError("prefix contrast rows must be a list")
    for index, raw in enumerate(rows):
        budget = _text(raw.get("budget"))
        scope = _scope(
            release_id=release_id, lane_id=str(lane["lane_id"]), task_id=str(lane["task_id"]),
            source_dataset_id="processbench::all_registered_subsets", population_id=population,
            cell_id="all_registered_subsets", slice_id=f"budget_{budget}",
            prediction_unit=str(lane["default_prediction_unit"]), estimand_id=str(lane["default_estimand_id"]),
            access_level="saved_output_probability_prefix", supervision="unsupervised",
            fidelity="causal_early_fixed_trace", report_partition="primary",
        )
        contrasts.append(_contrast_row(
            scope=scope, left_system=_text(raw.get("left_method_id")), right_system=_text(raw.get("right_method_id")),
            left_method=_text(raw.get("left_method_id")), right_method=_text(raw.get("right_method_id")),
            metric_id=_text(raw.get("metric")), metric_unit="unit_interval", positive_class="incorrect_final_answer",
            better_direction="higher", aggregation_id="equal_subset_macro", aggregation_level="population",
            aggregation_rule="equal_unit_mean", aggregation_unit="subset", cohort_id=f"prefix_budget_{budget}",
            source_comparison_group_id=None, delta=_float(raw.get("point_delta")), ci_low=_float(raw.get("ci_low")),
            ci_high=_float(raw.get("ci_high")), n_pairs=None,
            bootstrap_unit=_opt_text(raw.get("resampling_unit")), bootstrap_draws=_int(raw.get("bootstrap_draws")),
            paired=_bool(raw.get("paired")), source_status=_text(raw.get("status")),
            status_detail=(None if raw.get("status") == "OK" else _text(raw.get("missing_subset_policy"), "producer_status")),
            provenance=_provenance(source, "contrasts", f"/rows/{index}", raw),
        ))
    return metrics, contrasts


def _status_from_result(row: Mapping[str, Any], *, status_scope: str) -> dict[str, Any]:
    source = {
        "binding": row["source_binding_id"], "table": row["source_table"],
        "row": row["source_row_locator"], "row_sha256": row["source_row_sha256"],
        "scope": status_scope,
    }
    payload = {
        field: row[field]
        for field in (
            "release_id", "lane_id", "task_id", "source_dataset_id", "dataset_id",
            "population_id", "cell_id", "slice_id", "prediction_unit", "estimand_id",
            "access_level", "supervision", "fidelity", "report_partition",
        )
    }
    metric_id = _text(row.get("metric_id"))
    aggregation_level = _text(row.get("aggregation_level"), "cell")
    output = {
        **payload,
        "status_id": f"statusv1_{canonical_sha256(source)[:24]}",
        "status_scope": status_scope,
        "system_id": _text(row.get("system_id", row.get("left_system_id"))),
        "method_id": _text(row.get("method_id", row.get("left_method_id"))),
        "metric_id": metric_id,
        "aggregation_level": aggregation_level,
        "status_class": row["status_class"],
        "source_status": row["source_status"],
        "status_detail": row.get("status_detail"),
        "expected_n": row.get("expected_n"),
        "observed_n": row.get("scored_n", row.get("n_rows")),
        "rankable": bool(row.get("rankable", False)),
        "source_binding_id": row["source_binding_id"],
        "source_table": row["source_table"],
        "source_row_locator": row["source_row_locator"],
        "source_row_sha256": row["source_row_sha256"],
    }
    return output


def _external_population_statuses(
    release_id: str, source: AuthenticatedSource, contract: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if source.manifest is None:
        return []
    lane = _lane(contract, source.source_id)
    partition_by_population: MutableMapping[str, set[str]] = defaultdict(set)
    for row in metrics:
        partition_by_population[row["population_id"]].add(row["report_partition"])
    output = []
    for index, raw in enumerate(source.manifest.get("population_checks", [])):
        population = _text(raw.get("population_id"))
        canonical_population = _ns(str(lane["lane_id"]), "population", population)
        partitions = partition_by_population.get(canonical_population, {"context"})
        partition = "primary" if partitions == {"primary"} else "context"
        source_status = _text(raw.get("status"))
        scope = _scope(
            release_id=release_id, lane_id=str(lane["lane_id"]), task_id=str(lane["task_id"]),
            source_dataset_id=f"population::{population}", population_id=population,
            cell_id=NO_VALUE, slice_id=NO_VALUE, prediction_unit=str(lane["default_prediction_unit"]),
            estimand_id=str(lane["default_estimand_id"]), access_level="gray_box_single_pass",
            supervision="unsupervised", fidelity="external_v3_population_contract",
            report_partition=partition,
        )
        provenance = _provenance(source, "manifest.population_checks", f"/population_checks/{index}", raw)
        detail = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        identity = {"binding": source.source_binding_id, "population": population, "status": source_status}
        output.append({
            **scope, "status_id": f"statusv1_{canonical_sha256(identity)[:24]}",
            "status_scope": "aggregate", "system_id": NO_VALUE, "method_id": NO_VALUE,
            "metric_id": NO_VALUE, "aggregation_level": "population",
            "status_class": _status_class(source_status, context=partition == "context"),
            "source_status": source_status, "status_detail": detail,
            "expected_n": _int(raw.get("atomic_expected", {}).get("rows") if isinstance(raw.get("atomic_expected"), Mapping) else None),
            "observed_n": _int(raw.get("observed", {}).get("rows") if isinstance(raw.get("observed"), Mapping) else None),
            "rankable": False, **provenance,
        })
    return output


def _placeholder_status(
    release_id: str, source: AuthenticatedSource, contract: Mapping[str, Any]
) -> dict[str, Any]:
    lane = _lane(contract, source.source_id)
    scope = _scope(
        release_id=release_id, lane_id=str(lane["lane_id"]), task_id=str(lane["task_id"]),
        source_dataset_id="NOT_CERTIFIED", population_id=NO_VALUE, cell_id=NO_VALUE,
        slice_id=NO_VALUE, prediction_unit=str(lane["default_prediction_unit"]),
        estimand_id=str(lane["default_estimand_id"]), access_level="NOT_CERTIFIED",
        supervision="NOT_CERTIFIED", fidelity="NOT_CERTIFIED",
        report_partition=str(lane["report_partition"]),
    )
    identity = {"source_binding_id": source.source_binding_id, "status": "NOT_CERTIFIED"}
    return {
        **scope, "status_id": f"statusv1_{canonical_sha256(identity)[:24]}",
        "status_scope": "lane", "system_id": NO_VALUE, "method_id": NO_VALUE,
        "metric_id": NO_VALUE, "aggregation_level": "lane",
        "status_class": "NOT_CERTIFIED", "source_status": "NOT_CERTIFIED",
        "status_detail": "No reviewed final evaluation A/B certificate is bound in the source lock.",
        "expected_n": None, "observed_n": None, "rankable": False,
        "source_binding_id": source.source_binding_id, "source_table": "source_lock",
        "source_row_locator": f"source:{source.source_id}",
        "source_row_sha256": source.logical_binding_sha256,
    }


def _rollup_statuses(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    severity = {
        "OK": 0, "CONTEXT": 1, "NOT_APPLICABLE": 2, "PARTIAL": 3,
        "UNDEFINED": 4, "BLOCKED": 5, "UNVERIFIED": 6,
        "NOT_CERTIFIED": 7,
    }
    output: list[dict[str, Any]] = []
    specs = (
        ("dataset", (
            "lane_id", "task_id", "dataset_id", "prediction_unit", "estimand_id",
            "access_level", "supervision", "fidelity", "report_partition",
        )),
        ("cell", (
            "lane_id", "task_id", "dataset_id", "population_id", "cell_id",
            "prediction_unit", "estimand_id", "access_level", "supervision",
            "fidelity", "report_partition",
        )),
        ("system", (
            "lane_id", "task_id", "dataset_id", "population_id", "cell_id",
            "system_id", "prediction_unit", "estimand_id", "access_level",
            "supervision", "fidelity", "report_partition",
        )),
    )
    for status_scope, fields in specs:
        groups: MutableMapping[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["status_scope"] in {"metric", "aggregate", "system"}:
                groups[tuple(row[field] for field in fields)].append(row)
        for key, children in groups.items():
            worst = max(children, key=lambda row: severity[row["status_class"]])
            first = children[0]
            binding_ids = sorted({row["source_binding_id"] for row in children})
            if len(binding_ids) != 1:
                raise UnifiedReportingError("status rollup crossed a certified source binding")
            child_ids = sorted(row["status_id"] for row in children)
            identity = {"scope": status_scope, "key": key, "children": child_ids}
            rollup_scope = {field: first[field] for field in (
                    "release_id", "lane_id", "task_id", "source_dataset_id", "dataset_id",
                    "population_id", "cell_id", "slice_id", "prediction_unit", "estimand_id",
                    "access_level", "supervision", "fidelity", "report_partition",
                )}
            if status_scope == "dataset":
                rollup_scope.update(population_id=NO_VALUE, cell_id=NO_VALUE, slice_id=NO_VALUE)
            else:
                rollup_scope["slice_id"] = NO_VALUE
            output.append({
                **rollup_scope,
                "status_id": f"statusv1_{canonical_sha256(identity)[:24]}",
                "status_scope": status_scope,
                "system_id": (first["system_id"] if status_scope == "system" else NO_VALUE),
                "method_id": (first["method_id"] if status_scope == "system" else NO_VALUE),
                "metric_id": NO_VALUE, "aggregation_level": status_scope,
                "status_class": worst["status_class"], "source_status": "DERIVED_ROLLUP",
                "status_detail": json.dumps(
                    {"child_count": len(children), "status_counts": {
                        name: sum(row["status_class"] == name for row in children)
                        for name in sorted({row["status_class"] for row in children})
                    }}, sort_keys=True, separators=(",", ":"),
                ),
                "expected_n": None, "observed_n": None,
                "rankable": all(row["rankable"] for row in children),
                "source_binding_id": binding_ids[0], "source_table": "derived_status",
                "source_row_locator": f"derived:{status_scope}:{canonical_sha256(key)[:16]}",
                "source_row_sha256": canonical_sha256(child_ids),
            })
    return output


def _source_binding_rows(release_id: str, sources: Sequence[AuthenticatedSource]) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        certificate = source.lock_record.get("certificate", {})
        manifest = source.lock_record.get("manifest", {})
        rows.append({
            "release_id": release_id, "source_binding_id": source.source_binding_id,
            "source_id": source.source_id, "source_release_id": source.source_release_id,
            "certified": source.certified, "source_status": source.source_status,
            "source_root_id": source.source_root_id,
            "certificate_schema": certificate.get("schema_version"),
            "certificate_file_sha256": certificate.get("file_sha256"),
            "certificate_payload_sha256": certificate.get("self_hash"),
            "manifest_schema": manifest.get("schema_version"),
            "manifest_file_sha256": manifest.get("file_sha256"),
            "manifest_payload_sha256": manifest.get("self_hash"),
            "logical_binding_sha256": source.logical_binding_sha256,
        })
    return rows


def _source_artifact_rows(
    release_id: str, sources: Sequence[AuthenticatedSource]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        if not source.certified:
            continue
        bindings: list[tuple[str, str, Mapping[str, Any]]] = [
            ("certificate", "certificate", source.lock_record["certificate"]),
        ]
        if "manifest" in source.lock_record:
            bindings.append(("manifest", "manifest", source.lock_record["manifest"]))
        bindings.extend(
            ("evaluation_artifact", name, binding)
            for name, binding in sorted(source.lock_record.get("files", {}).items())
        )
        for role, name, binding in bindings:
            rows.append({
                "release_id": release_id,
                "source_binding_id": source.source_binding_id,
                "source_id": source.source_id,
                "artifact_role": role,
                "logical_name": name,
                "relative_path": str(binding["path"]),
                "format": _text(binding.get("format"), "json" if role != "evaluation_artifact" else "binary"),
                "schema_version": _opt_text(binding.get("schema_version")),
                "file_sha256": str(binding["file_sha256"]),
                "payload_sha256": _opt_text(binding.get("self_hash")),
                "authenticated": True,
            })
    return rows


def _winner_rows(
    release_id: str, source: AuthenticatedSource, contract: Mapping[str, Any],
    base_metrics: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lane = _lane(contract, source.source_id)
    base_source = str(lane["base_source"])
    base_binding = next(
        (row["source_binding_id"] for row in base_metrics if row["source_table"] == "metrics" and row["lane_id"] == lane["lane_id"]),
        None,
    )
    if base_binding is None:
        raise UnifiedReportingError(f"winner source {source.source_id} has no normalized base metrics")
    # Require the method's own certified point estimate so winner tables cannot
    # silently bind a stale release.  External-v3 preserves its source group in
    # the base table.  Frozen24 winner production intentionally created a new
    # scope-oriented group, so bind it to the exact producer aggregation id.
    source_group_index: MutableMapping[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    aggregation_index: MutableMapping[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in base_metrics:
        if row["source_binding_id"] == base_binding:
            source_group_index[
                (str(row.get("source_comparison_group_id")), row["metric_id"], row["method_id"])
            ].append(row)
            aggregation_index[
                (row["aggregation_id"], row["metric_id"], row["method_id"])
            ].append(row)

    def base(raw: Mapping[str, Any], method_field: str) -> Mapping[str, Any]:
        metric_id = _text(raw.get("metric_id"))
        method_id = _text(raw.get(method_field))
        if base_source == "frozen24":
            scope_type = _text(raw.get("scope_type"))
            scope_value = _text(raw.get("scope_value"))
            aggregation_id = (
                f"frozen24::cell::{scope_value}"
                if scope_type == "cell"
                else f"frozen24::equal-cell::{scope_type}-{scope_value}"
            )
            key = (aggregation_id, metric_id, method_id)
            matches = aggregation_index.get(key, [])
        else:
            key = (_text(raw.get("source_comparison_group_id")), metric_id, method_id)
            matches = source_group_index.get(key, [])
        if len(matches) != 1:
            raise UnifiedReportingError(f"winner row does not resolve to one base metric: {key}")
        point = matches[0]
        if point["source_binding_id"] != base_binding:
            raise UnifiedReportingError("winner row resolved outside its certified base binding")
        if raw.get("record_level") == "cell" and point["source_dataset_id"] != _text(raw.get("dataset_id")):
            raise UnifiedReportingError("winner cell dataset differs from base metric")
        return point

    sets = []
    for index, raw in enumerate(_csv(source.files["winner_reference_sets"])):
        point = base(raw, "method_id")
        if abs(float(raw["method_value"]) - float(point["value"])) > 1e-12:
            raise UnifiedReportingError("winner-reference method value differs from base metric")
        sets.append({
            **{field: point[field] for field in (
                "release_id", "lane_id", "task_id", "source_dataset_id", "dataset_id",
                "population_id", "cell_id", "slice_id", "prediction_unit", "estimand_id",
                "access_level", "supervision", "fidelity", "report_partition",
            )},
            "comparison_group_id": point["comparison_group_id"],
            "source_comparison_group_id": _text(raw.get("source_comparison_group_id")),
            "aggregation_id": _text(raw.get("aggregation")),
            "aggregation_level": _text(raw.get("record_level")),
            "metric_id": _text(raw.get("metric_id")),
            "better_direction": "higher" if _bool(raw.get("higher_is_better")) else "lower",
            "winner_reference_method_id": _text(raw.get("winner_reference_method_id")),
            "method_id": _text(raw.get("method_id")), "method_value": _float(raw.get("method_value")),
            "membership_status": _text(raw.get("membership_status")),
            "in_winner_reference_set": _bool(raw.get("in_winner_reference_set")),
            "interpretation_code": "DIRECT_PAIRED_NONSEPARATION_95",
            "equivalence_claim": False, "simultaneous_coverage": False,
            "winner_selection_adjusted": False, "multiplicity_adjustment": "NONE",
            **_provenance(source, "winner_reference_sets", f"row:{index + 2}", raw),
        })
    contrasts = []
    for index, raw in enumerate(_csv(source.files["winner_reference_contrasts"])):
        candidate = base(raw, "candidate_method_id")
        winner = base(raw, "winner_reference_method_id")
        if candidate["comparison_group_id"] != winner["comparison_group_id"]:
            raise UnifiedReportingError("winner-reference contrast crossed a comparison group")
        if abs(float(raw["candidate_value"]) - float(candidate["value"])) > 1e-12:
            raise UnifiedReportingError("winner-reference candidate value differs from base metric")
        if abs(float(raw["winner_value"]) - float(winner["value"])) > 1e-12:
            raise UnifiedReportingError("winner-reference winner value differs from base metric")
        contrasts.append({
            **{field: candidate[field] for field in (
                "release_id", "lane_id", "task_id", "source_dataset_id", "dataset_id",
                "population_id", "cell_id", "slice_id", "prediction_unit", "estimand_id",
                "access_level", "supervision", "fidelity", "report_partition",
            )},
            "comparison_group_id": candidate["comparison_group_id"],
            "source_comparison_group_id": _text(raw.get("source_comparison_group_id")),
            "aggregation_id": _text(raw.get("aggregation")),
            "aggregation_level": _text(raw.get("record_level")),
            "metric_id": _text(raw.get("metric_id")),
            "better_direction": "higher" if _bool(raw.get("higher_is_better")) else "lower",
            "winner_reference_method_id": _text(raw.get("winner_reference_method_id")),
            "candidate_method_id": _text(raw.get("candidate_method_id")),
            "winner_value": _float(raw.get("winner_value")),
            "candidate_value": _float(raw.get("candidate_value")),
            "delta_candidate_minus_winner": _float(raw.get("delta_candidate_minus_winner")),
            "ci_low": _float(raw.get("delta_ci_low")), "ci_high": _float(raw.get("delta_ci_high")),
            "membership_status": _text(raw.get("membership_status")),
            "in_winner_reference_set": _bool(raw.get("in_winner_reference_set")),
            "bootstrap_unit": _text(raw.get("bootstrap_unit")),
            "bootstrap_draws": _int(raw.get("bootstrap_draws")),
            "multiplicity_adjustment": _text(raw.get("multiplicity_adjustment")),
            "equivalence_claim": False,
            **_provenance(source, "winner_reference_contrasts", f"row:{index + 2}", raw),
        })
    return sets, contrasts


def _ensure_unique(rows: Iterable[Mapping[str, Any]], fields: Sequence[str], *, where: str) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in seen:
            raise UnifiedReportingError(f"duplicate {where} key: {key}")
        seen.add(key)


def build_unified_rows(
    *, release_id: str, contract: Mapping[str, Any], sources: Sequence[AuthenticatedSource]
) -> dict[str, list[dict[str, Any]]]:
    if not release_id:
        raise UnifiedReportingError("release_id is required")
    metrics: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    winner_sources: list[AuthenticatedSource] = []
    for source in sources:
        lane = _lane(contract, source.source_id)
        adapter = lane["adapter"]
        if adapter == "not_certified":
            statuses.append(_placeholder_status(release_id, source, contract))
        elif adapter in {"frozen24_v1", "edis_v2"}:
            lane_metrics, lane_contrasts, lane_coverage = _normalize_generic(release_id, source, lane)
            metrics.extend(lane_metrics); contrasts.extend(lane_contrasts); coverage.extend(lane_coverage)
        elif adapter == "external_v3":
            lane_metrics, lane_contrasts, lane_coverage = _normalize_external(release_id, source, contract)
            metrics.extend(lane_metrics); contrasts.extend(lane_contrasts); coverage.extend(lane_coverage)
            statuses.extend(_external_population_statuses(release_id, source, contract, lane_metrics))
        elif adapter == "localization_v1":
            lane_metrics, lane_contrasts, lane_coverage, lane_decisions = _normalize_localization(release_id, source, contract)
            metrics.extend(lane_metrics); contrasts.extend(lane_contrasts); coverage.extend(lane_coverage); decisions.extend(lane_decisions)
        elif adapter == "prefix_v1":
            lane_metrics, lane_contrasts = _normalize_prefix(release_id, source, contract)
            metrics.extend(lane_metrics); contrasts.extend(lane_contrasts)
        elif adapter == "winner_reference_v1":
            winner_sources.append(source)
        else:
            raise UnifiedReportingError(f"unsupported unified reporting adapter: {adapter}")
    for row in metrics:
        statuses.append(_status_from_result(row, status_scope=("aggregate" if row["aggregation_level"] != "cell" else "metric")))
    for row in coverage:
        statuses.append(_status_from_result(row, status_scope="system"))
    statuses.extend(_rollup_statuses(statuses))
    winner_sets: list[dict[str, Any]] = []
    winner_contrasts: list[dict[str, Any]] = []
    for source in winner_sources:
        sets, direct = _winner_rows(release_id, source, contract, metrics)
        winner_sets.extend(sets); winner_contrasts.extend(direct)
    primary_metrics = [row for row in metrics if row["report_partition"] == "primary"]
    context_metrics = [row for row in metrics if row["report_partition"] == "context"]
    primary_contrasts = [row for row in contrasts if row["report_partition"] == "primary"]
    context_contrasts = [row for row in contrasts if row["report_partition"] == "context"]
    primary_coverage = [row for row in coverage if row["report_partition"] == "primary"]
    context_coverage = [row for row in coverage if row["report_partition"] == "context"]
    primary_decisions = [row for row in decisions if row["report_partition"] == "primary"]
    context_decisions = [row for row in decisions if row["report_partition"] == "context"]
    primary_winner_sets = [row for row in winner_sets if row["report_partition"] == "primary"]
    context_winner_sets = [row for row in winner_sets if row["report_partition"] == "context"]
    primary_winner_contrasts = [row for row in winner_contrasts if row["report_partition"] == "primary"]
    context_winner_contrasts = [row for row in winner_contrasts if row["report_partition"] == "context"]
    tables = {
        "source_bindings": _source_binding_rows(release_id, sources),
        "source_artifacts": _source_artifact_rows(release_id, sources),
        "status": statuses,
        "metrics": primary_metrics,
        "context_metrics": context_metrics,
        "contrasts": primary_contrasts,
        "context_contrasts": context_contrasts,
        "coverage": primary_coverage,
        "context_coverage": context_coverage,
        "localization_decisions": primary_decisions,
        "context_localization_decisions": context_decisions,
        "winner_reference_sets": primary_winner_sets,
        "context_winner_reference_sets": context_winner_sets,
        "winner_reference_contrasts": primary_winner_contrasts,
        "context_winner_reference_contrasts": context_winner_contrasts,
    }
    for table, rows in list(tables.items()):
        tables[table] = validate_rows(table, rows)
    _ensure_unique(tables["source_bindings"], ("source_binding_id",), where="source binding")
    _ensure_unique(
        tables["source_artifacts"],
        ("source_binding_id", "artifact_role", "logical_name"),
        where="source artifact",
    )
    _ensure_unique(tables["status"], ("status_id",), where="status")
    for table in ("localization_decisions", "context_localization_decisions"):
        _ensure_unique(
            tables[table], ("source_binding_id", "system_id", "row_id"),
            where=table,
        )
    for table in ("metrics", "context_metrics"):
        _ensure_unique(tables[table], ("comparison_group_id", "system_id"), where=table)
    for table in ("contrasts", "context_contrasts"):
        _ensure_unique(tables[table], ("comparison_group_id", "left_system_id", "right_system_id"), where=table)
    return tables


def advisor_inputs(
    *, release_id: str, contract: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    status_counts: MutableMapping[str, MutableMapping[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in tables["status"]:
        status_counts[row["lane_id"]][row["status_class"]] += 1
    value = {
        "schema_version": "reconstruction-unified-advisor-inputs-v1",
        "release_id": release_id,
        "claim_boundaries": contract.get("claim_boundaries", {}),
        "lane_status_counts": {
            lane: dict(sorted(counts.items())) for lane, counts in sorted(status_counts.items())
        },
        "scientific_cross_task_macro_computed": False,
        "winner_reference_equivalence_claim": False,
    }
    value["payload_sha256"] = canonical_sha256(value)
    return value


__all__ = ["BRIDGE_SCHEMA", "advisor_inputs", "build_unified_rows"]
