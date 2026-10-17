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

LEASH_AGGREGATE_COLUMNS = (
    "arm", "dataset", "fidelity", "metric", "scope", "value",
)
LEASH_BOOTSTRAP_COLUMNS = (
    "arm", "dataset", "grouping", "hi", "lo", "metric", "model",
    "n_boot", "n_groups", "point", "reference_arm", "scope", "seed",
)
LEASH_CELL_COLUMNS = (
    "actual_stopping_claim_eligible", "arm", "cell_id", "closure_tokens",
    "dataset", "dataset_revision", "early_stop_rate", "fidelity",
    "forced_closure_rate", "mean_closure_tokens", "mean_reasoning_tokens",
    "mean_tokens_per_question", "mean_total_tokens", "mean_wall_s", "method_id",
    "model", "n_forced_closure", "n_parser_failures", "n_questions",
    "n_stopped_early", "n_stopped_without_closure", "parser_failure_rate",
    "parser_revision", "pass_at_1", "realized_savings_valid", "reasoning_tokens",
    "schema", "total_tokens", "total_wall_s",
)
LEASH_CONTRAST_COLUMNS = (
    "arm", "cell_id", "contrast_direction", "dataset",
    "early_stop_rate_delta_vs_cot", "forced_closure_rate_delta_vs_cot",
    "matched_accuracy_claim", "mean_closure_tokens_delta_vs_cot",
    "mean_reasoning_tokens_delta_vs_cot", "mean_total_tokens_delta_vs_cot",
    "mean_wall_s_delta_vs_cot", "model", "parser_failure_rate_delta_vs_cot",
    "pass_at_1_delta_vs_cot", "reference_arm", "token_reduction_vs_cot",
)
LEASH_COVERAGE_COLUMNS = (
    "actual_policy_execution_observed", "actual_stopping_claim_eligible",
    "coverage_status", "dataset", "fidelity", "model", "n_expected",
    "n_failed", "n_finished", "n_leash_policy_stops", "n_leash_rows_replayed",
    "n_policy_replay_mismatches", "reason", "run_id", "usable_for_evaluation",
)
LEASH_FRONTIER_COLUMNS = (
    "accuracy_delta_vs_cot", "arm", "cell_id", "dataset", "dataset_revision",
    "dominated_by", "mean_tokens_per_question", "mean_wall_s", "model",
    "pareto_efficient_within_cell", "pass_at_1", "schema",
    "token_reduction_vs_cot",
)
LEASH_BASE_METRICS = (
    "early_stop_rate", "forced_closure_rate", "mean_closure_tokens",
    "mean_reasoning_tokens", "mean_total_tokens", "mean_wall_s",
    "parser_failure_rate", "pass_at_1",
)
LEASH_DELTA_METRICS = tuple(f"{name}_delta_vs_cot" for name in LEASH_BASE_METRICS)

RAG_METRIC_COLUMNS = (
    "panel_id", "dataset", "unit", "access", "estimand", "split", "subgroup",
    "method_id", "metric", "value", "ci_low", "ci_high", "n", "n_groups",
    "positive_rate", "bootstrap_draws", "status",
)
RAG_CONTRAST_COLUMNS = (
    "panel_id", "split", "subgroup", "left_method", "right_method", "metric",
    "delta", "ci_low", "ci_high", "n", "n_groups", "bootstrap_draws", "status",
)
RAG_STATUS_COLUMNS = (
    "panel_id", "status", "metric_rows", "prediction_rows",
    "cross_panel_macro_contribution",
)
RAG_REFCHECKER_SUBGROUPS = (
    "accurate_context", "noisy_context", "zero_context",
)
RAG_PANEL_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "ragtruth_evidence_contrast_answer": {
        "dataset": "RAGTruth", "unit": "answer",
        "access": "teacher_forced_full_noctx_loo_where_available",
        "estimand": "response_hallucination_ranking", "methods": ("fixed_rag_iu_pcr",),
        "metrics": ("auroc", "auprc"), "splits": ("dev", "test"),
        "bootstrap_unit": "source_id", "supervision": "unsupervised",
        "fidelity": "registered_ragtruth_evidence_contrast",
    },
    "ragtruth_evidence_contrast_sentence": {
        "dataset": "RAGTruth", "unit": "sentence",
        "access": "teacher_forced_full_noctx_loo_where_available",
        "estimand": "sentence_hallucination_ranking", "methods": ("fixed_rag_iu_pcr",),
        "metrics": ("auroc", "auprc"), "splits": ("dev", "test"),
        "bootstrap_unit": "source_id", "supervision": "unsupervised",
        "fidelity": "registered_ragtruth_evidence_contrast",
    },
    "ragtruth_evidence_contrast_token": {
        "dataset": "RAGTruth", "unit": "scorer_token",
        "access": "teacher_forced_full_noctx_loo_where_available",
        "estimand": "token_overlap_hallucination_ranking", "methods": ("fixed_rag_iu_pcr",),
        "metrics": ("auroc", "auprc"), "splits": ("dev", "test"),
        "bootstrap_unit": "source_id", "supervision": "unsupervised",
        "fidelity": "registered_ragtruth_evidence_contrast",
    },
    "gasp_protocol_sentence": {
        "dataset": "RAGTruth balanced GASP cohort", "unit": "sentence",
        "access": "teacher_forced_full_noctx_loo_exact_full_vocab_jsd",
        "estimand": "sentence_hallucination_ranking_on_local_protocol_sample",
        "methods": ("gasp_threshold", "fixed_rag_iu_pcr_matched"),
        "metrics": ("auroc", "auprc"), "splits": ("local_400_response_sample",),
        "bootstrap_unit": "source_id", "supervision": "unsupervised",
        "fidelity": "protocol_reproduction_own_ids_and_splitter",
    },
    "lettucedetect_example": {
        "dataset": "RAGTruth", "unit": "example",
        "access": "supervised_ragtruth_token_classifier",
        "estimand": "any_predicted_span_vs_any_gold_span",
        "methods": ("lettucedetect_large_modernbert",),
        "metrics": ("f1", "precision", "recall"), "splits": ("test",),
        "bootstrap_unit": "source_id", "supervision": "supervised",
        "fidelity": "exact_local_reproduction",
    },
    "refchecker_threeway": {
        "dataset": "KnowHalBench fixed claims", "unit": "fixed_claim",
        "access": "supervised_nli_checker", "estimand": "three_way_claim_checking",
        "methods": ("refchecker_nli",), "metrics": ("accuracy", "macro_f1"),
        "splits": ("official_fixed_claims",), "bootstrap_unit": "example_id",
        "supervision": "supervised", "fidelity": "fixed_claim_reference_execution",
    },
    "refchecker_binary_claim": {
        "dataset": "KnowHalBench fixed claims", "unit": "fixed_claim",
        "access": "teacher_forced_full_noctx",
        "estimand": "unsupported_claim_ranking_binary_collapse",
        "methods": ("fixed_rag_iu_pcr_transfer",), "metrics": ("auroc", "auprc"),
        "splits": ("official_fixed_claims",), "bootstrap_unit": "example_id",
        "supervision": "unsupervised", "fidelity": "registered_binary_transfer",
    },
}


def _csv(payload: bytes) -> list[dict[str, str]]:
    try:
        return list(csv.DictReader(StringIO(payload.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise UnifiedReportingError(f"invalid certified CSV: {exc}") from exc


def _csv_exact(
    payload: bytes, columns: Sequence[str], *, where: str,
) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(StringIO(payload.decode("utf-8"), newline=""))
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise UnifiedReportingError(
                f"{where} header drift: expected={list(columns)}, observed={reader.fieldnames}"
            )
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise UnifiedReportingError(f"invalid certified CSV in {where}: {exc}") from exc
    if any(None in row for row in rows):
        raise UnifiedReportingError(f"{where} contains fields outside its typed header")
    return rows


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


def _same_number(left: Any, right: Any, *, where: str) -> None:
    left_value, right_value = _float(left), _float(right)
    if left_value is None or right_value is None or abs(left_value - right_value) > 1e-12:
        raise UnifiedReportingError(f"{where} point estimate drift")


def _leash_scope(
    release_id: str, lane: Mapping[str, Any], *, dataset: str, model: str,
    scope_name: str,
) -> tuple[dict[str, Any], str, str, str, str]:
    if scope_name == "cell":
        if not dataset or not model:
            raise UnifiedReportingError("LEASH cell scope requires dataset and model")
        population, cell, level = dataset, f"{dataset}::{model}", "cell"
        aggregation_id, aggregation_unit = "leash_cell", "source_question"
        source_dataset = dataset
    elif scope_name == "equal_model_within_dataset":
        if not dataset or model:
            raise UnifiedReportingError("LEASH equal-model scope identity drift")
        population, cell, level = dataset, NO_VALUE, "dataset"
        aggregation_id, aggregation_unit = "leash_equal_model_within_dataset", "model"
        source_dataset = dataset
    elif scope_name == "equal_dataset_after_equal_model":
        if dataset or model:
            raise UnifiedReportingError("LEASH equal-dataset scope identity drift")
        population, cell, level = NO_VALUE, NO_VALUE, "task"
        aggregation_id, aggregation_unit = "leash_equal_dataset_after_equal_model", "dataset"
        source_dataset = "all_registered_datasets"
    else:
        raise UnifiedReportingError(f"unknown LEASH aggregation scope: {scope_name!r}")
    scope = _scope(
        release_id=release_id, lane_id=str(lane["lane_id"]), task_id=str(lane["task_id"]),
        source_dataset_id=source_dataset, population_id=population, cell_id=cell,
        slice_id=NO_VALUE, prediction_unit=str(lane["default_prediction_unit"]),
        estimand_id=str(lane["default_estimand_id"]),
        access_level="actual_callback_execution_with_cot_nocot_references",
        supervision="unsupervised", fidelity="paper-specified-partial",
        report_partition="context",
    )
    return scope, aggregation_id, level, "producer_declared", aggregation_unit


def _leash_metric_metadata(metric_id: str) -> tuple[str, str, str]:
    if metric_id in {"mean_closure_tokens", "mean_reasoning_tokens", "mean_total_tokens"}:
        return "tokens", NO_VALUE, "lower"
    if metric_id == "mean_wall_s":
        return "seconds", NO_VALUE, "lower"
    if metric_id == "pass_at_1":
        return "unit_interval", "correct_final_answer", "higher"
    if metric_id == "parser_failure_rate":
        return "unit_interval", "parser_failure", "lower"
    if metric_id in {"early_stop_rate", "forced_closure_rate"}:
        return "unit_interval", "policy_stop", "higher"
    if metric_id == "token_reduction":
        return "unit_interval", NO_VALUE, "higher"
    raise UnifiedReportingError(f"unregistered LEASH metric: {metric_id}")


def _normalize_leash(
    release_id: str, source: AuthenticatedSource, contract: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]],
]:
    lane = _lane(contract, source.source_id)
    if lane["report_partition"] != "context":
        raise UnifiedReportingError("LEASH may only enter the context partition")
    allowed_arms = tuple(lane["allowed_arms"])
    reference_arm = str(lane["reference_arm"])
    draws = int(lane["bootstrap_draws"])
    aggregate_rows = _csv_exact(
        source.files["aggregate_metrics"], LEASH_AGGREGATE_COLUMNS,
        where="LEASH aggregate_metrics",
    )
    bootstrap_rows = _csv_exact(
        source.files["bootstrap_intervals"], LEASH_BOOTSTRAP_COLUMNS,
        where="LEASH bootstrap_intervals",
    )
    cell_rows = _csv_exact(
        source.files["cell_metrics"], LEASH_CELL_COLUMNS, where="LEASH cell_metrics",
    )
    contrast_rows = _csv_exact(
        source.files["contrasts"], LEASH_CONTRAST_COLUMNS, where="LEASH contrasts",
    )
    coverage_rows = _csv_exact(
        source.files["coverage"], LEASH_COVERAGE_COLUMNS, where="LEASH coverage",
    )
    frontier_rows = _csv_exact(
        source.files["frontier"], LEASH_FRONTIER_COLUMNS, where="LEASH frontier",
    )

    cell_points: dict[tuple[str, str, str, str], str] = {}
    cell_methods: dict[tuple[str, str, str], str] = {}
    cell_n: dict[tuple[str, str, str], int] = {}
    for raw in cell_rows:
        arm, dataset, model = raw["arm"], raw["dataset"], raw["model"]
        if (
            arm not in allowed_arms
            or raw["fidelity"] != "paper-specified-partial"
            or raw["dataset_revision"] != "test"
            or raw["schema"] != "s2_stopping_cell_metric_v1"
            or raw["cell_id"] != f"s2::{dataset}::{model}"
            or raw["method_id"] != f"{arm}|central"
            or not _bool(raw["realized_savings_valid"])
            or _bool(raw["actual_stopping_claim_eligible"]) is not (arm == "leash")
        ):
            raise UnifiedReportingError("LEASH cell metric contract drift")
        identity = (arm, dataset, model)
        if identity in cell_methods:
            raise UnifiedReportingError(f"duplicate LEASH cell arm: {identity}")
        cell_methods[identity] = raw["method_id"]
        cell_n[identity] = int(_int(raw["n_questions"]) or 0)
        for metric in LEASH_BASE_METRICS:
            cell_points[(*identity, metric)] = raw[metric]
    expected_cells = {(dataset, model) for _, dataset, model in cell_methods}
    if len(cell_rows) != 18 or any(
        {arm for arm, observed_dataset, observed_model in cell_methods if (observed_dataset, observed_model) == cell}
        != set(allowed_arms)
        for cell in expected_cells
    ):
        raise UnifiedReportingError("LEASH cell/arm coverage drift")

    aggregate_points: dict[tuple[str, str, str, str], str] = {}
    for raw in aggregate_rows:
        key = (raw["scope"], raw["arm"], raw["dataset"], raw["metric"])
        if (
            raw["arm"] not in allowed_arms
            or raw["metric"] not in LEASH_BASE_METRICS
            or raw["fidelity"] != "paper-specified-partial"
            or raw["scope"] not in {
                "equal_model_within_dataset", "equal_dataset_after_equal_model",
            }
            or key in aggregate_points
        ):
            raise UnifiedReportingError("LEASH aggregate metric contract drift")
        aggregate_points[key] = raw["value"]
    if len(aggregate_rows) != 72:
        raise UnifiedReportingError("LEASH aggregate metric row-count drift")

    cell_contrasts: dict[tuple[str, str, str, str], str] = {}
    for raw in contrast_rows:
        if (
            raw["arm"] not in {"leash", "nocot"}
            or raw["cell_id"] != f"s2::{raw['dataset']}::{raw['model']}"
            or raw["reference_arm"] != reference_arm
            or raw["contrast_direction"] != "arm_minus_cot"
            or _bool(raw["matched_accuracy_claim"])
        ):
            raise UnifiedReportingError("LEASH unregistered comparison escaped")
        for metric in (*LEASH_DELTA_METRICS, "token_reduction_vs_cot"):
            cell_contrasts[(raw["arm"], raw["dataset"], raw["model"], metric)] = raw[metric]
    if len(contrast_rows) != 12 or len(cell_contrasts) != 108:
        raise UnifiedReportingError("LEASH contrast coverage drift")

    metrics: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for index, raw in enumerate(bootstrap_rows):
        arm, dataset, model = raw["arm"], raw["dataset"], raw["model"]
        scope_name, metric_name = raw["scope"], raw["metric"]
        if (
            arm not in allowed_arms
            or _int(raw["n_boot"]) != draws
            or _int(raw["seed"]) != 2026082406
            or raw["grouping"]
            != "source_question_stratified_within_dataset_shared_across_arms_and_models"
            or (_int(raw["n_groups"]) or 0) <= 0
        ):
            raise UnifiedReportingError("LEASH bootstrap contract drift")
        scope, aggregation_id, level, rule, unit = _leash_scope(
            release_id, lane, dataset=dataset, model=model, scope_name=scope_name,
        )
        method_id = f"{arm}|central"
        cohort = f"leash::{scope_name}::{dataset or 'all'}::{model or 'all'}"
        provenance = _provenance(source, "bootstrap_intervals", f"row:{index + 2}", raw)
        if raw["reference_arm"] == "":
            if metric_name not in LEASH_BASE_METRICS:
                raise UnifiedReportingError("LEASH point interval uses an unregistered metric")
            if scope_name == "cell":
                _same_number(
                    raw["point"], cell_points.get((arm, dataset, model, metric_name)),
                    where="LEASH cell/bootstrap",
                )
                n_rows = cell_n[(arm, dataset, model)]
            else:
                _same_number(
                    raw["point"], aggregate_points.get((scope_name, arm, dataset, metric_name)),
                    where="LEASH aggregate/bootstrap",
                )
                n_rows = None
            metric_unit, positive_class, direction = _leash_metric_metadata(metric_name)
            metrics.append(_metric_row(
                scope=scope, system_id=arm, method_id=method_id, metric_id=metric_name,
                metric_unit=metric_unit, positive_class=positive_class,
                better_direction=direction, aggregation_id=aggregation_id,
                aggregation_level=level, aggregation_rule=rule, aggregation_unit=unit,
                cohort_id=cohort, source_comparison_group_id=f"leash::{scope_name}",
                value=_float(raw["point"]), ci_low=_float(raw["lo"]), ci_high=_float(raw["hi"]),
                n_rows=n_rows, n_groups=_int(raw["n_groups"]), n_positive=None,
                n_negative=None, bootstrap_unit="source_question_within_dataset",
                bootstrap_draws=draws, source_status="READY", status_detail=None,
                provenance=provenance,
            ))
            continue
        if (
            raw["reference_arm"] != reference_arm
            or arm not in {"leash", "nocot"}
            or metric_name not in {*LEASH_DELTA_METRICS, "token_reduction_vs_cot"}
        ):
            raise UnifiedReportingError("LEASH interval contains an unregistered comparison")
        if scope_name == "cell":
            _same_number(
                raw["point"], cell_contrasts.get((arm, dataset, model, metric_name)),
                where="LEASH contrast/bootstrap",
            )
        base_metric = (
            "token_reduction" if metric_name == "token_reduction_vs_cot"
            else metric_name.removesuffix("_delta_vs_cot")
        )
        metric_unit, positive_class, direction = _leash_metric_metadata(base_metric)
        contrasts.append(_contrast_row(
            scope=scope, left_system=arm, right_system=reference_arm,
            left_method=method_id, right_method=f"{reference_arm}|central",
            metric_id=base_metric, metric_unit=metric_unit, positive_class=positive_class,
            better_direction=direction, aggregation_id=aggregation_id,
            aggregation_level=level, aggregation_rule=rule, aggregation_unit=unit,
            cohort_id=cohort, source_comparison_group_id=f"leash::{scope_name}",
            delta=_float(raw["point"]), ci_low=_float(raw["lo"]), ci_high=_float(raw["hi"]),
            n_pairs=_int(raw["n_groups"]), bootstrap_unit="source_question_within_dataset",
            bootstrap_draws=draws, paired=True, source_status="READY",
            status_detail="registered arm-minus-CoT reference contrast",
            provenance=provenance,
        ))
    if len(bootstrap_rows) != 378 or len(metrics) != 216 or len(contrasts) != 162:
        raise UnifiedReportingError("LEASH bootstrap reporting roster drift")

    frontier_metrics: list[dict[str, Any]] = []
    frontier_specs = (
        ("frontier_pass_at_1", "pass_at_1", "pass_at_1"),
        ("frontier_mean_total_tokens", "mean_tokens_per_question", "mean_total_tokens"),
        ("frontier_mean_wall_s", "mean_wall_s", "mean_wall_s"),
        ("frontier_token_reduction", "token_reduction_vs_cot", "token_reduction"),
        ("frontier_accuracy_delta", "accuracy_delta_vs_cot", "pass_at_1"),
    )
    for index, raw in enumerate(frontier_rows):
        arm, dataset, model = raw["arm"], raw["dataset"], raw["model"]
        if (
            arm not in allowed_arms
            or raw["schema"] != "s2_accuracy_compute_frontier_point_v1"
            or raw["dataset_revision"] != "test"
            or raw["cell_id"] != f"s2::{dataset}::{model}"
        ):
            raise UnifiedReportingError("LEASH frontier contract drift")
        _same_number(raw["pass_at_1"], cell_points[(arm, dataset, model, "pass_at_1")], where="LEASH frontier/pass@1")
        _same_number(raw["mean_tokens_per_question"], cell_points[(arm, dataset, model, "mean_total_tokens")], where="LEASH frontier/tokens")
        _same_number(raw["mean_wall_s"], cell_points[(arm, dataset, model, "mean_wall_s")], where="LEASH frontier/wall")
        if arm == reference_arm:
            _same_number(raw["accuracy_delta_vs_cot"], 0.0, where="LEASH frontier/reference accuracy")
            _same_number(raw["token_reduction_vs_cot"], 0.0, where="LEASH frontier/reference tokens")
        else:
            _same_number(raw["accuracy_delta_vs_cot"], cell_contrasts[(arm, dataset, model, "pass_at_1_delta_vs_cot")], where="LEASH frontier/accuracy")
            _same_number(raw["token_reduction_vs_cot"], cell_contrasts[(arm, dataset, model, "token_reduction_vs_cot")], where="LEASH frontier/reduction")
        scope, _, level, _, unit = _leash_scope(
            release_id, lane, dataset=dataset, model=model, scope_name="cell",
        )
        detail = json.dumps({
            "pareto_efficient_within_cell": _bool(raw["pareto_efficient_within_cell"]),
            "dominated_by": raw["dominated_by"],
            "frontier_interpretation": "accuracy versus realized total-token compute",
            "inferential_status": "descriptive_only_no_interval",
            "matched_accuracy_claim": False,
        }, sort_keys=True, separators=(",", ":"))
        for output_metric, field, semantic_metric in frontier_specs:
            metric_unit, positive_class, direction = _leash_metric_metadata(semantic_metric)
            frontier_metrics.append(_metric_row(
                scope=scope, system_id=arm, method_id=f"{arm}|central",
                metric_id=output_metric, metric_unit=metric_unit,
                positive_class=positive_class, better_direction=direction,
                aggregation_id="leash_accuracy_compute_frontier_cell",
                aggregation_level=level, aggregation_rule="producer_frontier_point",
                aggregation_unit=unit, cohort_id=f"leash_frontier::{dataset}::{model}",
                source_comparison_group_id="leash::accuracy_compute_frontier",
                value=_float(raw[field]), ci_low=None, ci_high=None,
                n_rows=cell_n[(arm, dataset, model)], n_groups=cell_n[(arm, dataset, model)],
                n_positive=None, n_negative=None, bootstrap_unit=None, bootstrap_draws=None,
                source_status="READY", status_detail=detail,
                provenance=_provenance(source, "frontier", f"row:{index + 2}:{field}", raw),
            ))
    if len(frontier_rows) != 18:
        raise UnifiedReportingError("LEASH frontier row-count drift")
    metrics.extend(frontier_metrics)

    coverage: list[dict[str, Any]] = []
    for index, raw in enumerate(coverage_rows):
        dataset, model = raw["dataset"], raw["model"]
        source_status = raw["coverage_status"]
        ready = source_status == "READY"
        if (
            source_status not in {"READY", "PROTOCOL_GATE_FAILED"}
            or raw["fidelity"] != "paper-specified-partial"
            or _bool(raw["usable_for_evaluation"]) is not ready
            or _bool(raw["actual_policy_execution_observed"]) is not ready
            or _bool(raw["actual_stopping_claim_eligible"]) is not ready
        ):
            raise UnifiedReportingError("LEASH coverage/status contract drift")
        scope, _, _, _, _ = _leash_scope(
            release_id, lane, dataset=dataset, model=model, scope_name="cell",
        )
        expected, finished = _int(raw["n_expected"]), _int(raw["n_finished"])
        detail = json.dumps({
            "run_id": raw["run_id"], "reason": raw["reason"] or None,
            "actual_callback_execution": _bool(raw["actual_policy_execution_observed"]),
            "actual_stopping_claim_eligible": _bool(raw["actual_stopping_claim_eligible"]),
            "policy_replay_mismatches": _int(raw["n_policy_replay_mismatches"]),
        }, sort_keys=True, separators=(",", ":"))
        coverage.append({
            **scope, "system_id": _ns(str(lane["lane_id"]), "system", "leash_stopping_bundle"),
            "method_id": "leash|central", "expected_n": expected, "eligible_n": finished,
            "scored_n": finished, "fallback_n": 0, "excluded_n": 0,
            "failed_n": _int(raw["n_failed"]),
            "coverage_fraction": (
                None if expected in (None, 0) or finished is None else finished / expected
            ),
            "cohort_id": f"leash_coverage::{dataset}::{model}",
            "status_class": _status_class(source_status, context=True),
            "source_status": source_status, "status_detail": detail,
            **_provenance(source, "coverage", f"row:{index + 2}", raw),
        })
    if len(coverage) != 8:
        raise UnifiedReportingError("LEASH coverage row-count drift")

    manifest = source.manifest or {}
    manifest_scope, _, _, _, _ = _leash_scope(
        release_id, lane, dataset="", model="",
        scope_name="equal_dataset_after_equal_model",
    )
    ready_count = sum(row["coverage_status"] == "READY" for row in coverage_rows)
    manifest_identity = {
        "source_binding_id": source.source_binding_id,
        "claim_status": manifest.get("claim_status"),
    }
    explicit_status = [{
        **manifest_scope,
        "status_id": f"statusv1_{canonical_sha256(manifest_identity)[:24]}",
        "status_scope": "task", "system_id": NO_VALUE, "method_id": NO_VALUE,
        "metric_id": NO_VALUE, "aggregation_level": "task",
        "status_class": "CONTEXT", "source_status": _text(manifest.get("claim_status")),
        "status_detail": json.dumps({
            "claim_scope": manifest.get("claim_scope"),
            "fidelity": manifest.get("fidelity"),
            "paper_exact_claim": False, "matched_accuracy_claim": False,
            "cross_task_or_access_macro": False,
        }, sort_keys=True, separators=(",", ":")),
        "expected_n": len(coverage_rows), "observed_n": ready_count, "rankable": False,
        **_provenance(source, "manifest", "/claim_status", manifest_identity),
    }]
    return metrics, contrasts, coverage, explicit_status


def _rag_scope(
    release_id: str, lane: Mapping[str, Any], panel_id: str, *, split: str,
    subgroup: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    try:
        panel = RAG_PANEL_CONTRACTS[panel_id]
    except KeyError as exc:
        raise UnifiedReportingError(f"unregistered RAG evidence panel: {panel_id}") from exc
    scope = _scope(
        release_id=release_id, lane_id=str(lane["lane_id"]), task_id=str(lane["task_id"]),
        source_dataset_id=str(panel["dataset"]), population_id=panel_id,
        cell_id=f"{panel_id}::{split}", slice_id=subgroup,
        prediction_unit=str(panel["unit"]), estimand_id=str(panel["estimand"]),
        access_level=str(panel["access"]), supervision=str(panel["supervision"]),
        fidelity=str(panel["fidelity"]), report_partition="context",
    )
    return scope, panel


def _rag_positive_class(panel_id: str) -> str:
    if panel_id == "refchecker_threeway":
        return "three_way_claim_label"
    if panel_id == "refchecker_binary_claim":
        return "unsupported_claim"
    if panel_id == "lettucedetect_example":
        return "any_gold_hallucination_span"
    return "hallucinated"


def _normalize_rag(
    release_id: str, source: AuthenticatedSource, contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lane = _lane(contract, source.source_id)
    if lane["report_partition"] != "context":
        raise UnifiedReportingError("RAG evidence may only enter the context partition")
    panel_ids = tuple(lane["panel_ids"])
    if panel_ids != tuple(RAG_PANEL_CONTRACTS):
        raise UnifiedReportingError("RAG panel roster drift")
    draws = int(lane["bootstrap_draws"])
    raw_metrics = _csv_exact(source.files["metrics"], RAG_METRIC_COLUMNS, where="RAG metrics")
    raw_contrasts = _csv_exact(
        source.files["contrasts"], RAG_CONTRAST_COLUMNS, where="RAG contrasts",
    )
    raw_status = _csv_exact(
        source.files["panel_status"], RAG_STATUS_COLUMNS, where="RAG panel_status",
    )
    metric_groups: MutableMapping[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    group_counts: dict[tuple[str, str, str], tuple[int, int]] = {}
    metrics: list[dict[str, Any]] = []
    seen_metric_keys: set[tuple[str, str, str, str, str]] = set()
    for index, raw in enumerate(raw_metrics):
        panel_id, split, subgroup = raw["panel_id"], raw["split"], raw["subgroup"]
        scope, panel = _rag_scope(
            release_id, lane, panel_id, split=split, subgroup=subgroup,
        )
        key = (panel_id, split, subgroup, raw["method_id"], raw["metric"])
        if (
            raw["dataset"] != panel["dataset"]
            or raw["unit"] != panel["unit"]
            or raw["access"] != panel["access"]
            or raw["estimand"] != panel["estimand"]
            or split not in panel["splits"]
            or raw["method_id"] not in panel["methods"]
            or raw["metric"] not in panel["metrics"]
            or key in seen_metric_keys
            or _int(raw["bootstrap_draws"]) != draws
            or (_int(raw["n"]) or 0) <= 0
            or (_int(raw["n_groups"]) or 0) <= 0
        ):
            raise UnifiedReportingError("RAG metric panel contract drift")
        seen_metric_keys.add(key)
        if panel_id.startswith("refchecker_"):
            if subgroup not in RAG_REFCHECKER_SUBGROUPS:
                raise UnifiedReportingError("RefChecker setting pooling/roster drift")
        elif not subgroup or subgroup.lower() in {"pooled", "macro", "cross_panel"}:
            raise UnifiedReportingError("RAG task subgroup is invalid")
        source_status = raw["status"]
        if source_status not in {"OK", "METRIC_UNDEFINED_SINGLE_CLASS"}:
            raise UnifiedReportingError("RAG metric status drift")
        value, ci_low, ci_high = _float(raw["value"]), _float(raw["ci_low"]), _float(raw["ci_high"])
        if (source_status == "OK") is not (value is not None and ci_low is not None and ci_high is not None):
            raise UnifiedReportingError("RAG metric value/status mismatch")
        group = (panel_id, split, subgroup)
        counts = (_int(raw["n"]) or 0, _int(raw["n_groups"]) or 0)
        if group in group_counts and group_counts[group] != counts:
            raise UnifiedReportingError("RAG within-panel subgroup count drift")
        group_counts[group] = counts
        metric_groups[group].add((raw["method_id"], raw["metric"]))
        positive_rate = _float(raw["positive_rate"])
        n_positive = None
        if positive_rate is not None:
            candidate = round(positive_rate * counts[0])
            if abs(candidate - positive_rate * counts[0]) <= 1e-8:
                n_positive = int(candidate)
        detail = json.dumps(
            {"positive_rate": positive_rate, "panel_id": panel_id, "split": split,
             "subgroup": subgroup, "bootstrap_seed": 2026082407,
             "interval_interpretation": "nominal_grouped_95_ci"},
            sort_keys=True, separators=(",", ":"),
        )
        metrics.append(_metric_row(
            scope=scope, system_id=raw["method_id"], method_id=raw["method_id"],
            metric_id=raw["metric"], metric_unit="unit_interval",
            positive_class=_rag_positive_class(panel_id), better_direction="higher",
            aggregation_id=f"rag_panel::{panel_id}::{split}::{subgroup}",
            aggregation_level="cell", aggregation_rule="grouped_bootstrap_metric",
            aggregation_unit=str(panel["unit"]),
            cohort_id=f"rag::{panel_id}::{split}::{subgroup}",
            source_comparison_group_id=f"rag::{panel_id}::{split}::{subgroup}",
            value=value, ci_low=ci_low, ci_high=ci_high, n_rows=counts[0],
            n_groups=counts[1], n_positive=n_positive,
            n_negative=(None if n_positive is None else counts[0] - n_positive),
            bootstrap_unit=str(panel["bootstrap_unit"]), bootstrap_draws=draws,
            source_status=source_status, status_detail=detail,
            provenance=_provenance(source, "metrics", f"row:{index + 2}", raw),
        ))
    observed_splits: MutableMapping[str, set[str]] = defaultdict(set)
    observed_subgroups: MutableMapping[tuple[str, str], set[str]] = defaultdict(set)
    for panel_id, split, subgroup in metric_groups:
        observed_splits[panel_id].add(split)
        observed_subgroups[(panel_id, split)].add(subgroup)
        expected_pairs = {
            (method, metric)
            for method in RAG_PANEL_CONTRACTS[panel_id]["methods"]
            for metric in RAG_PANEL_CONTRACTS[panel_id]["metrics"]
        }
        if metric_groups[(panel_id, split, subgroup)] != expected_pairs:
            raise UnifiedReportingError("RAG within-panel method/metric matrix is incomplete")
    if set(observed_splits) != set(panel_ids) or any(
        observed_splits[panel_id] != set(RAG_PANEL_CONTRACTS[panel_id]["splits"])
        for panel_id in panel_ids
    ):
        raise UnifiedReportingError("RAG panel/split coverage drift")
    for panel_id in panel_ids:
        for split in RAG_PANEL_CONTRACTS[panel_id]["splits"]:
            subgroups = observed_subgroups[(panel_id, split)]
            if panel_id.startswith("refchecker_"):
                if subgroups != set(RAG_REFCHECKER_SUBGROUPS):
                    raise UnifiedReportingError("RefChecker required setting coverage drift")
            elif "all" not in subgroups:
                raise UnifiedReportingError("RAG non-RefChecker panel omits its all subgroup")
    for split in ("dev", "test"):
        ragtruth_sets = {
            tuple(sorted(observed_subgroups[(panel_id, split)]))
            for panel_id in panel_ids[:3]
        }
        if len(ragtruth_sets) != 1:
            raise UnifiedReportingError("RAGTruth unit panels disagree on task subgroups")

    contrasts: list[dict[str, Any]] = []
    seen_contrasts: set[tuple[str, str]] = set()
    gasp_groups = {
        subgroup for panel_id, split, subgroup in metric_groups
        if panel_id == "gasp_protocol_sentence" and split == "local_400_response_sample"
    }
    for index, raw in enumerate(raw_contrasts):
        if (
            raw["panel_id"] != "gasp_protocol_sentence"
            or raw["split"] != "local_400_response_sample"
            or raw["subgroup"] not in gasp_groups
            or raw["left_method"] != "gasp_threshold"
            or raw["right_method"] != "fixed_rag_iu_pcr_matched"
            or raw["metric"] not in {"auroc", "auprc"}
            or (raw["subgroup"], raw["metric"]) in seen_contrasts
            or _int(raw["bootstrap_draws"]) != draws
            or raw["status"] not in {"OK", "METRIC_UNDEFINED_SINGLE_CLASS"}
        ):
            raise UnifiedReportingError("RAG unregistered/cross-panel contrast escaped")
        seen_contrasts.add((raw["subgroup"], raw["metric"]))
        counts = group_counts[(raw["panel_id"], raw["split"], raw["subgroup"])]
        if counts != (_int(raw["n"]), _int(raw["n_groups"])):
            raise UnifiedReportingError("RAG contrast/metric population mismatch")
        delta, ci_low, ci_high = _float(raw["delta"]), _float(raw["ci_low"]), _float(raw["ci_high"])
        if (raw["status"] == "OK") is not (
            delta is not None and ci_low is not None and ci_high is not None
        ):
            raise UnifiedReportingError("RAG contrast value/status mismatch")
        scope, panel = _rag_scope(
            release_id, lane, raw["panel_id"], split=raw["split"], subgroup=raw["subgroup"],
        )
        contrasts.append(_contrast_row(
            scope=scope, left_system=raw["left_method"], right_system=raw["right_method"],
            left_method=raw["left_method"], right_method=raw["right_method"],
            metric_id=raw["metric"], metric_unit="unit_interval",
            positive_class=_rag_positive_class(raw["panel_id"]), better_direction="higher",
            aggregation_id=f"rag_panel::{raw['panel_id']}::{raw['split']}::{raw['subgroup']}",
            aggregation_level="cell", aggregation_rule="paired_grouped_bootstrap_delta",
            aggregation_unit=str(panel["unit"]),
            cohort_id=f"rag::{raw['panel_id']}::{raw['split']}::{raw['subgroup']}",
            source_comparison_group_id=f"rag::{raw['panel_id']}::{raw['split']}::{raw['subgroup']}",
            delta=delta, ci_low=ci_low, ci_high=ci_high, n_pairs=counts[0],
            bootstrap_unit=str(panel["bootstrap_unit"]), bootstrap_draws=draws,
            paired=True, source_status=raw["status"],
            status_detail=(
                "registered within-GASP paired reference contrast; "
                "nominal grouped 95% CI; seed=2026082407"
            ),
            provenance=_provenance(source, "contrasts", f"row:{index + 2}", raw),
        ))
    if seen_contrasts != {(subgroup, metric) for subgroup in gasp_groups for metric in ("auroc", "auprc")}:
        raise UnifiedReportingError("RAG GASP paired contrast coverage drift")

    statuses: list[dict[str, Any]] = []
    if tuple(raw["panel_id"] for raw in raw_status) != panel_ids:
        raise UnifiedReportingError("RAG panel-status order/roster drift")
    manifest_status = (source.manifest or {}).get("panel_status")
    for index, raw in enumerate(raw_status):
        panel_id = raw["panel_id"]
        metric_count = sum(row["panel_id"] == panel_id for row in raw_metrics)
        if (
            raw["status"] != "PASS"
            or raw["cross_panel_macro_contribution"] != "FORBIDDEN"
            or _int(raw["metric_rows"]) != metric_count
            or (_int(raw["prediction_rows"]) or 0) <= 0
        ):
            raise UnifiedReportingError("RAG panel-status content drift")
        manifest_row = manifest_status[index] if isinstance(manifest_status, list) else None
        if not isinstance(manifest_row, Mapping) or {
            "panel_id": raw["panel_id"], "status": raw["status"],
            "metric_rows": _int(raw["metric_rows"]),
            "prediction_rows": _int(raw["prediction_rows"]),
            "cross_panel_macro_contribution": raw["cross_panel_macro_contribution"],
        } != dict(manifest_row):
            raise UnifiedReportingError("RAG panel-status table/manifest mismatch")
        panel = RAG_PANEL_CONTRACTS[panel_id]
        scope, _ = _rag_scope(
            release_id, lane, panel_id, split=NO_VALUE, subgroup=NO_VALUE,
        )
        identity = {"binding": source.source_binding_id, "panel": panel_id, "status": "PASS"}
        statuses.append({
            **scope, "status_id": f"statusv1_{canonical_sha256(identity)[:24]}",
            "status_scope": "cell", "system_id": NO_VALUE, "method_id": NO_VALUE,
            "metric_id": NO_VALUE, "aggregation_level": "cell",
            "status_class": "CONTEXT", "source_status": "PASS",
            "status_detail": json.dumps({
                "metric_rows": _int(raw["metric_rows"]),
                "prediction_rows": _int(raw["prediction_rows"]),
                "cross_panel_macro_contribution": "FORBIDDEN",
                "unit": panel["unit"], "access": panel["access"],
                "estimand": panel["estimand"],
            }, sort_keys=True, separators=(",", ":")),
            "expected_n": _int(raw["prediction_rows"]),
            "observed_n": _int(raw["prediction_rows"]), "rankable": False,
            **_provenance(source, "panel_status", f"row:{index + 2}", raw),
        })
    return metrics, contrasts, statuses


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
        # Certification state is controlled only by the source lock.  A lane may
        # declare its future typed adapter while remaining a source-closed
        # placeholder; no artifact bytes exist on this branch.
        if not source.certified:
            statuses.append(_placeholder_status(release_id, source, contract))
        elif adapter == "not_certified":
            raise UnifiedReportingError(
                f"certified source {source.source_id} has no typed adapter"
            )
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
        elif adapter == "leash_v1":
            lane_metrics, lane_contrasts, lane_coverage, lane_statuses = _normalize_leash(
                release_id, source, contract,
            )
            metrics.extend(lane_metrics); contrasts.extend(lane_contrasts)
            coverage.extend(lane_coverage); statuses.extend(lane_statuses)
        elif adapter == "rag_evidence_v1":
            lane_metrics, lane_contrasts, lane_statuses = _normalize_rag(
                release_id, source, contract,
            )
            metrics.extend(lane_metrics); contrasts.extend(lane_contrasts)
            statuses.extend(lane_statuses)
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
