"""Deterministic reporting for the Reasoning Localization 0.3662 program.

The report is deliberately a renderer over frozen registries and long-form
artifacts.  Scientific values are never encoded in the HTML template.  This
module also owns the fail-closed validation rules that keep historical context,
ProcessBench, PRMBench, and early detection in separate comparison regimes.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_ID = "REASONING_LOCALIZATION_03662_ANCHOR_V1"
REGISTRY_FILES = (
    "METHOD_REGISTRY.json",
    "VARIANT_REGISTRY.json",
    "EXPERIMENT_REGISTRY.json",
    "METRICS_LONG.csv",
    "CONTRASTS_LONG.csv",
    "GATES_LONG.csv",
    "CLAIMS.json",
    "EXAMPLES.json",
    "PLOT_MANIFEST.json",
)
EXPECTED_NEW_VARIANTS = (
    "R0_ENTROPY_MAX",
    "R1_ENTROPY_TOP5",
    "R2_FAMILY6_TOP5_CURRENT",
    "R2_HISTORICAL_FAMILY6_BRIDGE",
    "R3_IU29",
    "R4_MIND_GAP",
    "C1_ENT_SW16",
    "C2_ENT_SWADAPT",
    "C3_ENT_CCUSUM",
    "C4_ENT_SAMPLED",
    "C5_ENT_ENERGY",
    "C6_DSP12",
    "C7_EDIS_ONSET",
    "C8_SELF_INNOV",
    "P3T_T0_FROZEN_PARENT",
    "P3T_T1_DSP_FIRST",
    "P3T_T2_CAUSAL_TEMPORAL",
    "P3T_T3_TWO_AXIS_LOWRANK",
)
TASK_LABELS = {
    "processbench_first_error": "ProcessBench",
    "prmbench_step_error": "PRMBench",
    "early_detection": "Early",
}
PHASE_ORDER = {"CONTEXT": 0, "REPORTING": 1, "P0": 2, "P1": 3, "P2": 4, "P3": 5, "P4": 6, "P5": 7}
NUMERIC_METRIC_FIELDS = ("value", "ci_low", "ci_high")
NUMERIC_CONTRAST_FIELDS = ("delta", "ci_low", "ci_high", "p_adjusted", "worst_unit_delta")
CLAIM_REF_IDS = {"TABLE_VARIANTS", "TABLE_METRICS", "TABLE_CONTRASTS", "TABLE_GATES"}


class ReportingValidationError(ValueError):
    """Raised when a report artifact violates the frozen reporting contract."""


@dataclass(frozen=True)
class ReportBuild:
    html_bytes: bytes
    manifest: dict[str, Any]
    resolved_plots: list[dict[str, Any]]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReportingValidationError(f"{path.name} must contain a JSON object")
    return value


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ReportingValidationError(f"{path.name} has no header")
        return list(reader.fieldnames), list(reader)


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ReportingValidationError(f"artifact escapes repository root: {path}") from exc


def _as_decimal(value: str, label: str) -> Decimal | None:
    if value == "":
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ReportingValidationError(f"{label} is not a finite decimal: {value!r}") from exc


def _selector_parts(selector: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for component in selector.split(";"):
        if not component:
            continue
        if "=" not in component:
            raise ReportingValidationError(f"invalid source selector component: {component!r}")
        key, value = component.split("=", 1)
        if not key or key in result:
            raise ReportingValidationError(f"invalid or duplicate selector key: {key!r}")
        result[key] = value
    return result


def _matches(row: Mapping[str, str], selection: Mapping[str, Any]) -> bool:
    for key, expected in selection.items():
        if expected in (None, "", [], {}):
            continue
        if isinstance(expected, list):
            if row.get(key, "") not in {str(value) for value in expected}:
                return False
        elif str(row.get(key, "")) != str(expected):
            return False
    return True


def _task_for_variant(variant: Mapping[str, Any], experiments: Sequence[Mapping[str, Any]]) -> str:
    phase = variant["phase"]
    tasks: list[str] = list(variant.get("task_ids", []))
    if not tasks:
        for experiment in experiments:
            if experiment["phase"] == phase:
                tasks.extend(experiment.get("task_ids", []))
    if phase == "CONTEXT" or variant["execution_status"] == "CONTEXT_ONLY":
        tasks = ["processbench_first_error"]
    labels = [TASK_LABELS.get(task, task) for task in dict.fromkeys(tasks)]
    return ", ".join(labels) if labels else "Audit / context"


def load_bundle(report_dir: Path) -> dict[str, Any]:
    method_registry = _load_json(report_dir / "METHOD_REGISTRY.json")
    variant_registry = _load_json(report_dir / "VARIANT_REGISTRY.json")
    experiment_registry = _load_json(report_dir / "EXPERIMENT_REGISTRY.json")
    claims = _load_json(report_dir / "CLAIMS.json")
    examples = _load_json(report_dir / "EXAMPLES.json")
    plot_manifest = _load_json(report_dir / "PLOT_MANIFEST.json")
    metric_fields, metrics = _load_csv(report_dir / "METRICS_LONG.csv")
    contrast_fields, contrasts = _load_csv(report_dir / "CONTRASTS_LONG.csv")
    gate_fields, gates = _load_csv(report_dir / "GATES_LONG.csv")
    return {
        "method_registry": method_registry,
        "variant_registry": variant_registry,
        "experiment_registry": experiment_registry,
        "claims": claims,
        "examples": examples,
        "plot_manifest": plot_manifest,
        "metric_fields": metric_fields,
        "metrics": metrics,
        "contrast_fields": contrast_fields,
        "contrasts": contrasts,
        "gate_fields": gate_fields,
        "gates": gates,
    }


def validate_bundle(bundle: Mapping[str, Any], report_dir: Path, repo_root: Path) -> list[str]:
    """Validate the complete reporting bundle and return source-artifact paths."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    method_registry = bundle["method_registry"]
    variant_registry = bundle["variant_registry"]
    experiment_registry = bundle["experiment_registry"]
    claims_registry = bundle["claims"]
    examples = bundle["examples"]
    plot_manifest = bundle["plot_manifest"]
    metrics = bundle["metrics"]
    contrasts = bundle["contrasts"]
    gates = bundle["gates"]

    for registry_name, registry in (
        ("METHOD_REGISTRY.json", method_registry),
        ("VARIANT_REGISTRY.json", variant_registry),
        ("EXPERIMENT_REGISTRY.json", experiment_registry),
        ("PLOT_MANIFEST.json", plot_manifest),
    ):
        require(registry.get("report_id") == REPORT_ID, f"{registry_name}: report_id mismatch")

    methods = method_registry.get("methods", [])
    method_ids = [row.get("method_id") for row in methods]
    require(len(method_ids) == len(set(method_ids)), "duplicate method_id")
    required_method_fields = {
        "method_id", "display_name", "problem", "plain_summary", "input_operation_output",
        "novelty", "assumptions", "limitations", "references",
    }
    for row in methods:
        require(required_method_fields <= set(row), f"method {row.get('method_id')} misses required fields")

    variants = variant_registry.get("variants", [])
    variant_ids = [row.get("variant_id") for row in variants]
    require(len(variant_ids) == len(set(variant_ids)), "duplicate variant_id")
    require(set(EXPECTED_NEW_VARIANTS) <= set(variant_ids), "R0-R4/C1-C8 roster is incomplete")
    allowed_execution = set(variant_registry.get("allowed_execution_statuses", []))
    allowed_decisions = set(variant_registry.get("allowed_decision_statuses", []))
    allowed_evidence = set(variant_registry.get("allowed_evidence_statuses", []))
    allowed_statistics = set(variant_registry.get("allowed_statistical_statuses", []))
    required_variant_fields = {
        "variant_id", "display_name", "method_id", "phase", "role", "parent_variant_ids",
        "signals", "transforms", "detector", "step_reducer", "fusion", "novelty",
        "access_tier", "supervision", "causal_validity", "prior_evidence", "limitations",
        "failure_hypothesis", "execution_status", "decision_status", "evidence_status", "statistical_status",
        "rankable", "display_order",
    }
    variant_map = {row.get("variant_id"): row for row in variants}
    for row in variants:
        vid = row.get("variant_id")
        require(required_variant_fields <= set(row), f"variant {vid} misses required fields")
        require(row.get("method_id") in method_ids, f"variant {vid} has unknown method_id")
        require(row.get("execution_status") in allowed_execution, f"variant {vid} has invalid execution status")
        require(row.get("decision_status") in allowed_decisions, f"variant {vid} has invalid decision status")
        require(row.get("evidence_status") in allowed_evidence, f"variant {vid} has invalid evidence status")
        require(row.get("statistical_status") in allowed_statistics, f"variant {vid} has invalid statistical status")
        require(isinstance(row.get("rankable"), bool), f"variant {vid} rankable must be boolean")
        for parent in row.get("parent_variant_ids", []):
            require(parent in variant_map, f"variant {vid} has unknown parent {parent}")
        if row.get("execution_status") == "CONTEXT_ONLY":
            require(not row.get("rankable"), f"context variant {vid} cannot be rankable")
            require(row.get("evidence_status") == "CONTEXT_ONLY", f"context variant {vid} evidence must be context-only")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(vid: str) -> None:
        if vid in visiting:
            errors.append(f"variant parent cycle at {vid}")
            return
        if vid in visited or vid not in variant_map:
            return
        visiting.add(vid)
        for parent in variant_map[vid].get("parent_variant_ids", []):
            visit(parent)
        visiting.remove(vid)
        visited.add(vid)

    for vid in variant_map:
        visit(vid)

    experiments = experiment_registry.get("experiments", [])
    experiment_ids = [row.get("experiment_id") for row in experiments]
    require(len(experiment_ids) == len(set(experiment_ids)), "duplicate experiment_id")
    require(experiment_ids and experiment_ids[0] == "REPORTING", "REPORTING must be the first experiment")
    require(experiments[0].get("execution_status") == "COMPLETE", "Reporting Phase must be complete")

    source_artifacts: set[str] = set()
    source_cache: dict[str, tuple[str, list[dict[str, str]]]] = {}

    def source_rows(row: Mapping[str, str], row_label: str) -> list[dict[str, str]]:
        artifact = row.get("source_artifact", "")
        source_hash = row.get("source_sha256", "")
        selector = row.get("source_row_selector", "")
        if not artifact:
            require(row.get("status") not in {"COMPLETE", "CONTEXT_ONLY"}, f"{row_label}: completed/context row has no source artifact")
            return []
        source_artifacts.add(artifact)
        path = (repo_root / artifact).resolve()
        try:
            _repo_relative(path, repo_root)
        except ReportingValidationError as exc:
            errors.append(str(exc))
            return []
        require(path.is_file(), f"{row_label}: source artifact does not exist: {artifact}")
        if not path.is_file():
            return []
        actual_hash = sha256_file(path)
        require(source_hash == actual_hash, f"{row_label}: source SHA mismatch for {artifact}")
        if artifact not in source_cache:
            _, rows = _load_csv(path)
            source_cache[artifact] = (actual_hash, rows)
        selected = [candidate for candidate in source_cache[artifact][1] if _matches(candidate, _selector_parts(selector))]
        require(len(selected) == 1, f"{row_label}: selector must resolve exactly one source row (found {len(selected)})")
        return selected

    metric_comparison_regimes: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for index, row in enumerate(metrics, start=2):
        label = f"METRICS_LONG.csv:{index}"
        vid = row.get("variant_id", "")
        require(vid in variant_map, f"{label}: unknown variant {vid}")
        require(row.get("experiment_id") in experiment_ids, f"{label}: unknown experiment")
        require(row.get("status") in allowed_execution, f"{label}: invalid status")
        require(row.get("evidence_status") in allowed_evidence, f"{label}: invalid evidence")
        for field in NUMERIC_METRIC_FIELDS:
            try:
                _as_decimal(row.get(field, ""), f"{label}:{field}")
            except ReportingValidationError as exc:
                errors.append(str(exc))
        numeric = row.get("value", "") != ""
        require(numeric == (row.get("status") in {"COMPLETE", "CONTEXT_ONLY"}), f"{label}: numeric value/status mismatch")
        if row.get("status") == "CONTEXT_ONLY":
            require(not variant_map.get(vid, {}).get("rankable", True), f"{label}: context metric cannot be rankable")
        group = row.get("comparison_group_id", "")
        if group:
            metric_comparison_regimes[group].add((row.get("task_id", ""), row.get("population_id", ""), row.get("metric_id", "")))
        selected = source_rows(row, label)
        if selected:
            field = row.get("source_value_field", "")
            require(field in selected[0], f"{label}: missing source value field {field!r}")
            if field in selected[0]:
                require(row.get("value") == selected[0][field], f"{label}: copied metric does not equal source")

    for group, regimes in metric_comparison_regimes.items():
        require(len(regimes) == 1, f"comparison group {group} mixes task/population/metric regimes: {sorted(regimes)}")

    contrast_comparison_regimes: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for index, row in enumerate(contrasts, start=2):
        label = f"CONTRASTS_LONG.csv:{index}"
        left = row.get("left_variant_id", "")
        right = row.get("right_variant_id", "")
        require(left in variant_map and right in variant_map, f"{label}: unknown contrast variant")
        require(row.get("experiment_id") in experiment_ids, f"{label}: unknown experiment")
        require(row.get("status") in allowed_execution, f"{label}: invalid status")
        require(row.get("evidence_status") in allowed_evidence, f"{label}: invalid evidence")
        for field in NUMERIC_CONTRAST_FIELDS:
            try:
                _as_decimal(row.get(field, ""), f"{label}:{field}")
            except ReportingValidationError as exc:
                errors.append(str(exc))
        numeric = row.get("delta", "") != ""
        require(numeric == (row.get("status") in {"COMPLETE", "CONTEXT_ONLY"}), f"{label}: numeric delta/status mismatch")
        group = row.get("comparison_group_id", "")
        if group:
            contrast_comparison_regimes[group].add((row.get("task_id", ""), row.get("population_id", ""), row.get("metric_id", "")))
        selected = source_rows(row, label)
        if selected:
            source = selected[0]
            checks = {
                "delta": "delta", "ci_low": "ci_low", "ci_high": "ci_high",
                "ties": "ties", "worst_unit_delta": "worst_unit_delta",
            }
            for target, source_field in checks.items():
                if source_field in source:
                    require(row.get(target, "") == source.get(source_field, ""), f"{label}: {target} does not equal source {source_field}")
            for target, current_field, legacy_field in (
                ("wins", "wins", "family_wins"),
                ("losses", "losses", "family_losses"),
            ):
                source_field = current_field if current_field in source else legacy_field
                require(row.get(target, "") == source.get(source_field, ""), f"{label}: {target} does not equal source {source_field}")

    for group, regimes in contrast_comparison_regimes.items():
        require(len(regimes) == 1, f"contrast group {group} mixes task/population/metric regimes: {sorted(regimes)}")

    for index, row in enumerate(gates, start=2):
        label = f"GATES_LONG.csv:{index}"
        require(row.get("variant_id") in variant_map, f"{label}: unknown variant")
        require(row.get("experiment_id") in experiment_ids, f"{label}: unknown experiment")
        require(row.get("passed") in {"true", "false", "", "NA"}, f"{label}: invalid passed value")
        selected = source_rows(row, label)
        if selected:
            source = selected[0]
            source_value_field = row.get("source_value_field", "")
            require(source_value_field in source, f"{label}: missing source value field {source_value_field!r}")
            if source_value_field in source:
                require(row.get("observed", "") == source[source_value_field], f"{label}: copied gate observation does not equal source")
            for field in ("threshold", "direction", "passed", "status", "evidence_status"):
                require(row.get(field, "") == source.get(field, ""), f"{label}: {field} does not equal source")

    # Planned roster entries cannot silently acquire a scientific score.
    metric_variants = {row["variant_id"] for row in metrics if row.get("value", "") != ""}
    for variant in variants:
        if variant["execution_status"] == "PLANNED":
            require(variant["variant_id"] not in metric_variants, f"planned variant {variant['variant_id']} has a numeric metric")

    plot_ids = [plot.get("plot_id") for plot in plot_manifest.get("plots", [])]
    require(len(plot_ids) == len(set(plot_ids)), "duplicate plot_id")
    table_names = set(REGISTRY_FILES)
    for plot in plot_manifest.get("plots", []):
        require(plot.get("source_table") in table_names, f"plot {plot.get('plot_id')} has unknown source table")
        for field in ("plot_id", "phase", "kind", "title", "caption", "source_table", "selection", "legend", "comparison_group", "bootstrap_definition", "selection_rule"):
            require(field in plot, f"plot {plot.get('plot_id')} misses {field}")

    allowed_verdicts = set(claims_registry.get("allowed_verdicts", []))
    for claim in claims_registry.get("claims", []):
        require(claim.get("verdict") in allowed_verdicts, f"claim {claim.get('claim_id')} has invalid verdict")
        summary = claim.get("statistical_summary")
        if summary is not None:
            required_summary = {"metric", "point_delta", "ci_low", "ci_high", "benefit_bound", "harm_bound", "bound_basis", "multiplicity"}
            require(required_summary <= set(summary), f"claim {claim.get('claim_id')} has incomplete statistical summary")
            if required_summary <= set(summary):
                require(summary["ci_low"] <= summary["point_delta"] <= summary["ci_high"], f"claim {claim.get('claim_id')} point delta lies outside interval")
                require(summary["harm_bound"] <= summary["benefit_bound"], f"claim {claim.get('claim_id')} has inverted practical bounds")
        for reference in claim.get("evidence_refs", []):
            if reference in CLAIM_REF_IDS:
                continue
            if reference.startswith("PLOT_"):
                require(reference in plot_ids, f"claim {claim.get('claim_id')} refers to unknown plot {reference}")
            elif reference.startswith("CONTRAST:"):
                parts = reference.split(":")
                require(len(parts) == 3, f"claim {claim.get('claim_id')} has malformed contrast reference")
                if len(parts) == 3:
                    require(any(row["left_variant_id"] == parts[1] and row["right_variant_id"] == parts[2] for row in contrasts), f"claim {claim.get('claim_id')} refers to missing contrast")
            elif reference.startswith("MANIFEST:"):
                require(reference == "MANIFEST:REPORT_MANIFEST.json", f"claim {claim.get('claim_id')} has unknown manifest reference")
            else:
                errors.append(f"claim {claim.get('claim_id')} has unknown evidence reference {reference}")

    required_categories = set(examples.get("required_categories", []))
    example_categories = [row.get("category") for row in examples.get("examples", [])]
    require(len(example_categories) == len(set(example_categories)), "duplicate deterministic example category")
    require(set(example_categories) <= required_categories, "unknown deterministic example category")
    if examples.get("status") == "COMPLETE":
        require(set(example_categories) == required_categories, "complete examples artifact must represent all categories")
    else:
        require(not examples.get("examples"), "non-complete examples artifact must not contain hand-picked cases")

    if errors:
        raise ReportingValidationError("reporting validation failed:\n- " + "\n- ".join(errors))
    return sorted(source_artifacts)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: str, digits: int = 4) -> str:
    if value in ("", None):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return _esc(value)


def _status_badge(value: str, kind: str = "status") -> str:
    css = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return f'<span class="badge {kind}-{css}">{_esc(value)}</span>'


def _semantic_table(table_id: str, caption: str, headers: Sequence[tuple[str, str]], rows: Sequence[Sequence[str]], classes: str = "") -> str:
    head = "".join(f'<th scope="col" data-key="{_esc(key)}">{_esc(label)}</th>' for key, label in headers)
    if rows:
        body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    else:
        body = f'<tr><td colspan="{len(headers)}" class="empty">No registered rows. Missing is not zero.</td></tr>'
    return (
        f'<div class="table-wrap"><table id="{_esc(table_id)}" class="{_esc(classes)}">'
        f'<caption>{_esc(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _scale(value: float, low: float, high: float, left: float, right: float) -> float:
    if high <= low:
        return (left + right) / 2
    return left + (value - low) / (high - low) * (right - left)


def _forest_svg(plot: Mapping[str, Any], rows: Sequence[Mapping[str, str]], variant_map: Mapping[str, Mapping[str, Any]]) -> str:
    value_field = "delta" if plot["kind"] == "contrast_forest" else plot.get("x_field", "value")
    usable = [row for row in rows if row.get(value_field, "") != ""]
    if not usable:
        return _pending_plot(plot)
    values: list[float] = []
    for row in usable:
        values.append(float(row[value_field]))
        for field in ("ci_low", "ci_high"):
            if row.get(field, ""):
                values.append(float(row[field]))
    low, high = min(values), max(values)
    if value_field == "delta":
        low, high = min(low, 0.0), max(high, 0.0)
    pad = max((high - low) * 0.12, 0.005)
    low, high = low - pad, high + pad
    width = 920
    left, right = 270, 875
    row_height = 52
    height = 70 + row_height * len(usable)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{plot["plot_id"]}-title {plot["plot_id"]}-desc">',
             f'<title id="{plot["plot_id"]}-title">{_esc(plot["title"])}</title>',
             f'<desc id="{plot["plot_id"]}-desc">{_esc(plot["caption"])}</desc>']
    if low <= 0 <= high:
        zx = _scale(0, low, high, left, right)
        parts.append(f'<line x1="{zx:.2f}" x2="{zx:.2f}" y1="24" y2="{height-28}" class="zero"/>')
    for tick in range(6):
        val = low + tick * (high - low) / 5
        x = _scale(val, low, high, left, right)
        parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{height-28}" y2="{height-22}" class="axis"/>')
        parts.append(f'<text x="{x:.2f}" y="{height-6}" text-anchor="middle" class="tick">{val:.3f}</text>')
    for index, row in enumerate(usable):
        y = 36 + index * row_height
        variant_id = row.get("left_variant_id") or row.get("variant_id", "")
        label = variant_map.get(variant_id, {}).get("display_name", variant_id)
        parts.append(f'<text x="8" y="{y+5}" class="label">{_esc(label)}</text>')
        if row.get("ci_low", "") and row.get("ci_high", ""):
            x1 = _scale(float(row["ci_low"]), low, high, left, right)
            x2 = _scale(float(row["ci_high"]), low, high, left, right)
            parts.append(f'<line x1="{x1:.2f}" x2="{x2:.2f}" y1="{y}" y2="{y}" class="interval"/>')
        x = _scale(float(row[value_field]), low, high, left, right)
        context = row.get("status") == "CONTEXT_ONLY"
        parts.append(f'<circle cx="{x:.2f}" cy="{y}" r="6" class="point {"context-point" if context else ""}"/>')
        parts.append(f'<text x="{right}" y="{y-9}" text-anchor="end" class="value">{float(row[value_field]):.4f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_start(plot: Mapping[str, Any], width: int, height: int) -> list[str]:
    return [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{plot["plot_id"]}-title {plot["plot_id"]}-desc">',
        f'<title id="{plot["plot_id"]}-title">{_esc(plot["title"])}</title>',
        f'<desc id="{plot["plot_id"]}-desc">{_esc(plot["caption"])}</desc>',
    ]


def _waterfall_svg(plot: Mapping[str, Any], rows: Sequence[Mapping[str, str]], variant_map: Mapping[str, Mapping[str, Any]]) -> str:
    usable = [row for row in rows if row.get("value", "") != ""]
    if not usable:
        return _pending_plot(plot)
    usable.sort(key=lambda row: (float(row.get("display_order") or 0), row.get("variant_id", "")))
    values = [float(row["value"]) for row in usable]
    low, high = min(0.0, min(values)), max(values)
    pad = max((high - low) * 0.1, 0.005)
    low, high = low - pad, high + pad
    width, height = 920, 420
    left, right, top, bottom = 72, 890, 35, 335
    bar_width = max(18, min(72, (right - left) / max(len(usable), 1) * 0.62))
    parts = _svg_start(plot, width, height)
    zero_y = _scale(0, low, high, bottom, top)
    parts.append(f'<line x1="{left}" x2="{right}" y1="{zero_y:.2f}" y2="{zero_y:.2f}" class="axis"/>')
    previous: float | None = None
    for index, row in enumerate(usable):
        value = float(row["value"])
        x = left + (index + 0.5) * (right - left) / len(usable)
        y = _scale(value, low, high, bottom, top)
        y0 = _scale(0, low, high, bottom, top)
        color_class = "waterfall-up" if previous is None or value >= previous else "waterfall-down"
        parts.append(f'<rect x="{x-bar_width/2:.2f}" y="{min(y,y0):.2f}" width="{bar_width:.2f}" height="{max(abs(y-y0),1):.2f}" class="{color_class}"/>')
        if previous is not None:
            py = _scale(previous, low, high, bottom, top)
            px = left + (index - 0.5) * (right - left) / len(usable)
            parts.append(f'<line x1="{px+bar_width/2:.2f}" x2="{x-bar_width/2:.2f}" y1="{py:.2f}" y2="{y:.2f}" class="waterfall-link"/>')
        label = variant_map.get(row.get("variant_id", ""), {}).get("display_name", row.get("variant_id", ""))
        label_lines: list[str] = []
        for word in label.split():
            if label_lines and len(label_lines[-1]) + len(word) + 1 <= 22:
                label_lines[-1] += f" {word}"
            else:
                label_lines.append(word)
        tspans = "".join(
            f'<tspan x="{x:.2f}" dy="{0 if line_index == 0 else 15}">{_esc(line)}</tspan>'
            for line_index, line in enumerate(label_lines)
        )
        parts.append(f'<text x="{x:.2f}" y="{bottom+24}" text-anchor="middle" class="tick">{tspans}</text>')
        parts.append(f'<text x="{x:.2f}" y="{max(top+12,y-8):.2f}" text-anchor="middle" class="value">{value:.4f}</text>')
        previous = value
    parts.append("</svg>")
    return "".join(parts)


def _heatmap_svg(plot: Mapping[str, Any], rows: Sequence[Mapping[str, str]], variant_map: Mapping[str, Mapping[str, Any]]) -> str:
    usable = [row for row in rows if row.get("value", "") != ""]
    if not usable:
        return _pending_plot(plot)
    x_field, y_field = plot["x_field"], plot["y_field"]
    x_values = sorted({row.get(x_field, "") for row in usable})
    y_values = sorted({row.get(y_field, "") for row in usable}, key=lambda value: variant_map.get(value, {}).get("display_order", 9999))
    numeric = [float(row["value"]) for row in usable]
    low, high = min(numeric), max(numeric)
    cell_w, cell_h = 92, 34
    left, top = 210, 58
    width = max(760, left + len(x_values) * cell_w + 30)
    height = top + len(y_values) * cell_h + 35
    lookup = {(row.get(x_field, ""), row.get(y_field, "")): row for row in usable}
    parts = _svg_start(plot, width, height)
    for x_index, x_value in enumerate(x_values):
        x = left + x_index * cell_w
        parts.append(f'<text x="{x+cell_w/2:.2f}" y="38" text-anchor="middle" class="tick">{_esc(x_value)}</text>')
    for y_index, y_value in enumerate(y_values):
        y = top + y_index * cell_h
        label = variant_map.get(y_value, {}).get("display_name", y_value)
        parts.append(f'<text x="{left-9}" y="{y+22}" text-anchor="end" class="label">{_esc(label)}</text>')
        for x_index, x_value in enumerate(x_values):
            x = left + x_index * cell_w
            row = lookup.get((x_value, y_value))
            if row is None:
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" class="heat-missing"/>')
                continue
            value = float(row["value"])
            ratio = 0.5 if high <= low else (value - low) / (high - low)
            hue = 215 - 55 * ratio
            light = 92 - 38 * ratio
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" rx="3" style="fill:hsl({hue:.1f} 68% {light:.1f}%)"/>')
            parts.append(f'<text x="{x+(cell_w-2)/2:.2f}" y="{y+21}" text-anchor="middle" class="heat-value">{value:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _gate_matrix_svg(plot: Mapping[str, Any], rows: Sequence[Mapping[str, str]], variant_map: Mapping[str, Mapping[str, Any]]) -> str:
    if not rows:
        return _pending_plot(plot)
    x_field, y_field = plot["x_field"], plot["y_field"]
    x_values = sorted({row.get(x_field, "") for row in rows})
    y_values = sorted({row.get(y_field, "") for row in rows}, key=lambda value: variant_map.get(value, {}).get("display_order", 9999))
    if not x_values or not y_values:
        return _pending_plot(plot)
    cell_w, cell_h = 108, 38
    left, top = 220, 72
    width = max(760, left + len(x_values) * cell_w + 25)
    height = top + len(y_values) * cell_h + 35
    lookup = {(row.get(x_field, ""), row.get(y_field, "")): row.get("passed", "") for row in rows}
    parts = _svg_start(plot, width, height)
    for x_index, value in enumerate(x_values):
        x = left + x_index * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.2f}" y="48" text-anchor="middle" class="tick">{_esc(value[:15])}</text>')
    for y_index, value in enumerate(y_values):
        y = top + y_index * cell_h
        label = variant_map.get(value, {}).get("display_name", value)
        parts.append(f'<text x="{left-9}" y="{y+24}" text-anchor="end" class="label">{_esc(label)}</text>')
        for x_index, x_value in enumerate(x_values):
            status = lookup.get((x_value, value), "")
            css = "gate-pass" if status == "true" else "gate-fail" if status == "false" else "gate-pending"
            symbol = "PASS" if status == "true" else "FAIL" if status == "false" else "—"
            x = left + x_index * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" rx="4" class="{css}"/><text x="{x+(cell_w-3)/2:.2f}" y="{y+24}" text-anchor="middle" class="gate-text">{symbol}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _measure_for_field(row: Mapping[str, str], field: str) -> float | None:
    direct = row.get(field, "")
    if direct != "":
        try:
            return float(direct)
        except ValueError:
            return None
    metric_id = row.get("metric_id", "")
    task_id = row.get("task_id", "")
    expected_metric = field
    expected_task = ""
    if field.startswith("processbench_delta_"):
        expected_task = "processbench_first_error"
        expected_metric = field.removeprefix("processbench_delta_")
    elif field.startswith("prmbench_delta_"):
        expected_task = "prmbench_step_error"
        expected_metric = field.removeprefix("prmbench_delta_")
    if expected_task and task_id != expected_task:
        return None
    metric_matches = metric_id in {expected_metric, field, f"paired_delta_{expected_metric}"}
    if not metric_matches:
        return None
    for value_field in ("value", "delta", "observed"):
        if row.get(value_field, "") != "":
            return float(row[value_field])
    return None


def _scatter_points(plot: Mapping[str, Any], rows: Sequence[Mapping[str, str]]) -> list[tuple[str, float, float]]:
    x_field, y_field = plot["x_field"], plot["y_field"]
    direct: list[tuple[str, float, float]] = []
    for row in rows:
        x_value, y_value = _measure_for_field(row, x_field), _measure_for_field(row, y_field)
        if x_value is not None and y_value is not None:
            direct.append((row.get("variant_id") or row.get("left_variant_id") or row.get(plot.get("series_field", ""), ""), x_value, y_value))
    if direct:
        return direct
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = row.get("variant_id") or row.get("left_variant_id") or row.get(plot.get("series_field", ""), "")
        for field in (x_field, y_field):
            value = _measure_for_field(row, field)
            if value is not None:
                grouped[key][field] = value
    return [(key, values[x_field], values[y_field]) for key, values in sorted(grouped.items()) if x_field in values and y_field in values]


def _scatter_svg(plot: Mapping[str, Any], rows: Sequence[Mapping[str, str]], variant_map: Mapping[str, Mapping[str, Any]]) -> str:
    points = _scatter_points(plot, rows)
    if not points:
        return _pending_plot(plot)
    xs, ys = [point[1] for point in points], [point[2] for point in points]
    x_low, x_high = min(xs + [0.0]), max(xs + [0.0])
    y_low, y_high = min(ys + [0.0]), max(ys + [0.0])
    x_pad, y_pad = max((x_high-x_low)*0.12, 0.002), max((y_high-y_low)*0.12, 0.002)
    x_low, x_high, y_low, y_high = x_low-x_pad, x_high+x_pad, y_low-y_pad, y_high+y_pad
    width, height = 920, 470
    left, right, top, bottom = 90, 880, 35, 390
    parts = _svg_start(plot, width, height)
    zx, zy = _scale(0, x_low, x_high, left, right), _scale(0, y_low, y_high, bottom, top)
    parts.append(f'<line x1="{zx:.2f}" x2="{zx:.2f}" y1="{top}" y2="{bottom}" class="zero"/><line x1="{left}" x2="{right}" y1="{zy:.2f}" y2="{zy:.2f}" class="zero"/>')
    for key, x_value, y_value in points:
        x, y = _scale(x_value, x_low, x_high, left, right), _scale(y_value, y_low, y_high, bottom, top)
        label = variant_map.get(key, {}).get("display_name", key)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" class="scatter-point"><title>{_esc(label)}: {x_value:.4f}, {y_value:.4f}</title></circle><text x="{x+9:.2f}" y="{y-8:.2f}" class="tick">{_esc(key)}</text>')
    parts.append(f'<text x="{(left+right)/2:.2f}" y="{height-16}" text-anchor="middle" class="label">{_esc(plot["x_field"])}</text><text x="18" y="{(top+bottom)/2:.2f}" class="label">{_esc(plot["y_field"])}</text></svg>')
    return "".join(parts)


def _line_svg(plot: Mapping[str, Any], rows: Sequence[Mapping[str, str]]) -> str:
    usable = [row for row in rows if row.get(plot["x_field"], "") != "" and row.get("value", "") != ""]
    if not usable:
        return _pending_plot(plot)
    series_field = plot.get("series_field", "metric_id")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in usable:
        key = f'{row.get("variant_id", "")} · {row.get(series_field, "")}'.strip(" ·")
        grouped[key].append((float(row[plot["x_field"]]), float(row["value"])))
    xs = [value[0] for values in grouped.values() for value in values]
    ys = [value[1] for values in grouped.values() for value in values]
    x_low, x_high, y_low, y_high = min(xs), max(xs), min(ys), max(ys)
    x_pad, y_pad = max((x_high-x_low)*.08, 1), max((y_high-y_low)*.1, .002)
    width, height = 920, 470
    left, right, top, bottom = 84, 880, 32, 390
    palette = ["#2463eb", "#008b86", "#ad6800", "#7a3bc2", "#b52a38", "#15805d"]
    parts = _svg_start(plot, width, height)
    for index, (name, values) in enumerate(sorted(grouped.items())):
        values.sort()
        points = " ".join(f'{_scale(x,x_low-x_pad,x_high+x_pad,left,right):.2f},{_scale(y,y_low-y_pad,y_high+y_pad,bottom,top):.2f}' for x, y in values)
        color = palette[index % len(palette)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"><title>{_esc(name)}</title></polyline>')
        for x, y in values:
            px, py = _scale(x,x_low-x_pad,x_high+x_pad,left,right), _scale(y,y_low-y_pad,y_high+y_pad,bottom,top)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{right-5}" y="{top+15+index*16}" text-anchor="end" style="fill:{color};font-size:11px">{_esc(name)}</text>')
    parts.append(f'<text x="{(left+right)/2:.2f}" y="{height-18}" text-anchor="middle" class="label">{_esc(plot["x_field"])}</text></svg>')
    return "".join(parts)


def _lineage_svg(plot: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]) -> str:
    phases = ["CONTEXT", "P0", "P1", "P2", "P3", "P4", "P5"]
    grouped: dict[str, list[Mapping[str, Any]]] = {phase: [] for phase in phases}
    for variant in variants:
        phase = variant["phase"] if variant["phase"] in grouped else "CONTEXT"
        grouped[phase].append(variant)
    for values in grouped.values():
        values.sort(key=lambda row: (row["display_order"], row["variant_id"]))
    width = 1420
    column_width = width / len(phases)
    max_rows = max(len(values) for values in grouped.values())
    row_height = 74
    height = 100 + max_rows * row_height
    positions: dict[str, tuple[float, float]] = {}
    for phase_index, phase in enumerate(phases):
        x = phase_index * column_width + column_width / 2
        for row_index, variant in enumerate(grouped[phase]):
            positions[variant["variant_id"]] = (x, 76 + row_index * row_height)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{plot["plot_id"]}-title {plot["plot_id"]}-desc">',
             f'<title id="{plot["plot_id"]}-title">{_esc(plot["title"])}</title>',
             f'<desc id="{plot["plot_id"]}-desc">{_esc(plot["caption"])}</desc>']
    for phase_index, phase in enumerate(phases):
        x = phase_index * column_width
        parts.append(f'<rect x="{x:.2f}" y="0" width="{column_width:.2f}" height="{height}" class="phase-bg {"context-bg" if phase == "CONTEXT" else ""}"/>')
        parts.append(f'<text x="{x+column_width/2:.2f}" y="26" text-anchor="middle" class="phase-label">{phase}</text>')
    for variant in variants:
        target = positions.get(variant["variant_id"])
        if not target:
            continue
        for parent in variant.get("parent_variant_ids", []):
            source = positions.get(parent)
            if source:
                parts.append(f'<path d="M {source[0]+72:.2f} {source[1]:.2f} C {source[0]+105:.2f} {source[1]:.2f}, {target[0]-105:.2f} {target[1]:.2f}, {target[0]-72:.2f} {target[1]:.2f}" class="edge"/>')
    for variant in variants:
        if variant["variant_id"] not in positions:
            continue
        x, y = positions[variant["variant_id"]]
        decision = re.sub(r"[^a-z]+", "-", variant["decision_status"].lower()).strip("-")
        evidence = re.sub(r"[^a-z]+", "-", variant["evidence_status"].lower()).strip("-")
        parts.append(f'<g class="node decision-{decision} evidence-{evidence}"><rect x="{x-72:.2f}" y="{y-22:.2f}" width="144" height="44" rx="8"/><title>{_esc(variant["display_name"])}; {variant["decision_status"]}; {variant["evidence_status"]}</title><text x="{x:.2f}" y="{y-2:.2f}" text-anchor="middle">{_esc(variant["variant_id"])}</text><text x="{x:.2f}" y="{y+13:.2f}" text-anchor="middle" class="node-status">{_esc(variant["execution_status"])}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def _pipeline_svg(plot: Mapping[str, Any]) -> str:
    """Render a small design-contract flow without inventing result values."""

    selection = plot.get("selection", {})
    nodes = selection.get("nodes", [])
    transfer = selection.get("transfer_node", "")
    transfer_from = int(selection.get("transfer_from", max(0, len(nodes) - 2)))
    if not isinstance(nodes, list) or len(nodes) < 2 or not all(isinstance(x, str) and x for x in nodes):
        raise ReportingValidationError(f"{plot['plot_id']}: pipeline nodes are malformed")
    if transfer and not (0 <= transfer_from < len(nodes)):
        raise ReportingValidationError(f"{plot['plot_id']}: pipeline transfer index is invalid")
    width, height = 1420, 360
    margin, gap = 36, 22
    box_width = (width - 2 * margin - gap * (len(nodes) - 1)) / len(nodes)
    box_height, y = 92, 76
    parts = _svg_start(plot, width, height)
    for index, label in enumerate(nodes):
        x = margin + index * (box_width + gap)
        if index:
            previous_right = x - gap
            parts.append(
                f'<line x1="{previous_right:.2f}" y1="{y+box_height/2:.2f}" '
                f'x2="{x-6:.2f}" y2="{y+box_height/2:.2f}" class="pipeline-arrow"/>'
                f'<path d="M {x-13:.2f} {y+box_height/2-6:.2f} L {x-6:.2f} {y+box_height/2:.2f} '
                f'L {x-13:.2f} {y+box_height/2+6:.2f}" class="pipeline-head"/>'
            )
        words = label.split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            if len(" ".join(current + [word])) > 24 and current:
                lines.append(" ".join(current)); current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        parts.append(f'<rect x="{x:.2f}" y="{y}" width="{box_width:.2f}" height="{box_height}" rx="12" class="pipeline-node"/>')
        for line_index, line in enumerate(lines[:3]):
            line_y = y + box_height / 2 - (len(lines[:3]) - 1) * 9 + line_index * 18
            parts.append(f'<text x="{x+box_width/2:.2f}" y="{line_y:.2f}" text-anchor="middle" class="pipeline-text">{_esc(line)}</text>')
    if transfer:
        source_x = margin + transfer_from * (box_width + gap) + box_width / 2
        branch_y = 242
        transfer_width = min(350.0, box_width * 1.7)
        transfer_x = min(width - margin - transfer_width, max(margin, source_x - transfer_width / 2))
        parts.append(
            f'<path d="M {source_x:.2f} {y+box_height:.2f} C {source_x:.2f} 205, '
            f'{transfer_x+transfer_width/2:.2f} 205, {transfer_x+transfer_width/2:.2f} {branch_y:.2f}" '
            f'class="pipeline-transfer"/>'
            f'<rect x="{transfer_x:.2f}" y="{branch_y:.2f}" width="{transfer_width:.2f}" height="70" rx="12" class="pipeline-transfer-node"/>'
            f'<text x="{transfer_x+transfer_width/2:.2f}" y="{branch_y+32:.2f}" text-anchor="middle" class="pipeline-text">{_esc(transfer)}</text>'
            f'<text x="{transfer_x+transfer_width/2:.2f}" y="{branch_y+52:.2f}" text-anchor="middle" class="pipeline-note">separate evaluator; no task average</text>'
        )
    parts.append('</svg>')
    return "".join(parts)


def _pending_plot(plot: Mapping[str, Any]) -> str:
    return (
        '<div class="pending-plot" role="status">'
        f'<span class="pending-icon" aria-hidden="true">◇</span><strong>PLANNED — no eligible registered rows</strong>'
        f'<p>This panel will render from <code>{_esc(plot["source_table"])}</code> only after its selection contract is satisfied. No zero is imputed.</p></div>'
    )


def _plot_rows(plot: Mapping[str, Any], bundle: Mapping[str, Any]) -> Sequence[Mapping[str, str]]:
    source = plot["source_table"]
    if source == "METRICS_LONG.csv":
        rows = bundle["metrics"]
    elif source == "CONTRASTS_LONG.csv":
        rows = bundle["contrasts"]
    elif source == "GATES_LONG.csv":
        rows = bundle["gates"]
    else:
        return []
    selection = plot.get("selection", {})
    return [row for row in rows if _matches(row, selection)]


def _render_plot(plot: Mapping[str, Any], bundle: Mapping[str, Any], variant_map: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    if plot["kind"] == "lineage":
        return _lineage_svg(plot, bundle["variant_registry"]["variants"]), "RENDERED"
    if plot["kind"] == "pipeline":
        return _pipeline_svg(plot), "RENDERED"
    rows = _plot_rows(plot, bundle)
    if plot["kind"] in {"forest", "contrast_forest"}:
        rendered = _forest_svg(plot, rows, variant_map)
        return rendered, "RENDERED" if rows and any(row.get("value") or row.get("delta") for row in rows) else "PENDING"
    if plot["kind"] == "waterfall":
        rendered = _waterfall_svg(plot, rows, variant_map)
        return rendered, "RENDERED" if any(row.get("value") for row in rows) else "PENDING"
    if plot["kind"] == "heatmap":
        rendered = _heatmap_svg(plot, rows, variant_map)
        return rendered, "RENDERED" if any(row.get("value") for row in rows) else "PENDING"
    if plot["kind"] == "gate_matrix":
        rendered = _gate_matrix_svg(plot, rows, variant_map)
        return rendered, "RENDERED" if rows else "PENDING"
    if plot["kind"] == "scatter":
        rendered = _scatter_svg(plot, rows, variant_map)
        return rendered, "RENDERED" if _scatter_points(plot, rows) else "PENDING"
    if plot["kind"] == "line":
        rendered = _line_svg(plot, rows)
        return rendered, "RENDERED" if any(row.get(plot["x_field"], "") and row.get("value", "") for row in rows) else "PENDING"
    return _pending_plot(plot), "PENDING"


def _method_cards(bundle: Mapping[str, Any]) -> str:
    methods = {row["method_id"]: row for row in bundle["method_registry"]["methods"]}
    variants = sorted(bundle["variant_registry"]["variants"], key=lambda row: (PHASE_ORDER.get(row["phase"], 99), row["display_order"], row["variant_id"]))

    def card(variant: Mapping[str, Any], context: bool) -> str:
        method = methods[variant["method_id"]]
        parents = ", ".join(variant["parent_variant_ids"]) or "root"
        refs = "".join(f"<li><code>{_esc(ref)}</code></li>" for ref in method["references"])
        signals = ", ".join(variant["signals"])
        transforms = ", ".join(variant["transforms"])
        return f'''<article class="method-card {'context-card' if context else ''}" id="method-{_esc(variant['variant_id'].lower())}">
          <header><span class="eyebrow">{_esc(variant['phase'])} · {_esc(variant['role'])}</span><h3>{_esc(variant['display_name'])}</h3><code>{_esc(variant['variant_id'])}</code></header>
          <p class="method-summary">{_esc(method['plain_summary'])}</p>
          <dl>
            <dt>Problem</dt><dd>{_esc(method['problem'])}</dd>
            <dt>Flow</dt><dd>{_esc(method['input_operation_output'])}</dd>
            <dt>Signals / transforms</dt><dd>{_esc(signals)} / {_esc(transforms)}</dd>
            <dt>Detector / reducer</dt><dd>{_esc(variant['detector'])} / {_esc(variant['step_reducer'])}</dd>
            <dt>Parent → change</dt><dd>{_esc(parents)} → {_esc(variant['novelty'])}</dd>
            <dt>Fusion</dt><dd>{_esc(variant['fusion'])}</dd>
            <dt>Access / supervision</dt><dd>{_esc(variant['access_tier'])}; {_esc(variant['supervision'])}</dd>
            <dt>Causal validity</dt><dd>{_esc(variant['causal_validity'])}</dd>
            <dt>Prior evidence</dt><dd>{_esc(variant['prior_evidence'])}</dd>
            <dt>Limits / failure hypothesis</dt><dd>{_esc(variant['limitations'])} {_esc(variant['failure_hypothesis'])}</dd>
          </dl>
          <details><summary>Repository evidence</summary><ul>{refs}</ul></details>
          <footer>{_status_badge(variant['execution_status'], 'execution')} {_status_badge(variant['decision_status'], 'decision')} {_status_badge(variant['evidence_status'], 'evidence')} {_status_badge(variant['statistical_status'], 'statistical')}</footer>
        </article>'''

    new_cards = "".join(card(row, False) for row in variants if row["execution_status"] != "CONTEXT_ONLY")
    context_cards = "".join(card(row, True) for row in variants if row["execution_status"] == "CONTEXT_ONLY")
    return f'''<div class="section-intro"><p>Each card states the exact input, transform, reducer, parent delta, access, and failure hypothesis. A card is a method definition—not evidence that it works.</p></div>
      <h3 class="subheading">New program roster</h3><div class="card-grid">{new_cards}</div>
      <h3 class="subheading">Historical context — not rankable</h3><p class="context-note">These gray cards preserve provenance and negative evidence. They never enter the common-protocol leaderboard.</p><div class="card-grid">{context_cards}</div>'''


def _best_task_metric(variant_id: str, task_id: str, metrics: Sequence[Mapping[str, str]]) -> str:
    candidates = [row for row in metrics if row["variant_id"] == variant_id and row["task_id"] == task_id and row.get("value", "") != "" and row.get("cell_id") == "aggregate"]
    if not candidates:
        return "—"
    candidates.sort(key=lambda row: (row.get("display_order", ""), row["metric_id"]))
    return "<br>".join(f'{_esc(row["metric_id"])} {_fmt(row["value"])}' for row in candidates)


def _variant_table(bundle: Mapping[str, Any]) -> str:
    experiments = bundle["experiment_registry"]["experiments"]
    variants = sorted(bundle["variant_registry"]["variants"], key=lambda row: (PHASE_ORDER.get(row["phase"], 99), row["display_order"], row["variant_id"]))
    rows: list[list[str]] = []
    for variant in variants:
        task = _task_for_variant(variant, experiments)
        parents = ", ".join(variant["parent_variant_ids"]) or "—"
        row_classes = f'<span class="sr-only">{_esc(variant["phase"])} {_esc(task)} {_esc(variant["execution_status"])} {_esc(variant["evidence_status"])} {_esc(variant["statistical_status"])}</span>'
        rows.append([
            row_classes + f'<a href="#method-{_esc(variant["variant_id"].lower())}"><code>{_esc(variant["variant_id"])}</code></a><br><span class="muted">{_esc(variant["display_name"])}</span>',
            _esc(variant["phase"]), _esc(task), _esc(variant["method_id"]), _esc(parents),
            _status_badge(variant["execution_status"], "execution"),
            _status_badge(variant["decision_status"], "decision"),
            _status_badge(variant["evidence_status"], "evidence"),
            _status_badge(variant["statistical_status"], "statistical"),
            _best_task_metric(variant["variant_id"], "processbench_first_error", bundle["metrics"]),
            _best_task_metric(variant["variant_id"], "prmbench_step_error", bundle["metrics"]),
            _best_task_metric(variant["variant_id"], "early_detection", bundle["metrics"]),
        ])
    headers = (
        ("variant", "Variant"), ("phase", "Phase"), ("task", "Task"), ("family", "Family"),
        ("parent", "Parent"), ("execution", "Execution"), ("decision", "Decision"),
        ("evidence", "Evidence"), ("statistical", "Statistical status"), ("pb", "ProcessBench"), ("prm", "PRMBench"), ("early", "Early"),
    )
    return _semantic_table("table-variants", "Master variant roster. Task columns are separate; no overall score exists.", headers, rows, "sortable master-table")


def _raw_table(kind: str, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    useful = {
        "metrics": ["phase_id", "variant_id", "task_id", "population_id", "cell_id", "metric_id", "value", "ci_low", "ci_high", "comparison_group_id", "status", "evidence_status"],
        "contrasts": ["phase_id", "left_variant_id", "right_variant_id", "task_id", "population_id", "metric_id", "delta", "ci_low", "ci_high", "wins", "ties", "losses", "worst_unit_delta", "status"],
        "gates": ["phase_id", "variant_id", "gate_id", "metric_id", "observed", "threshold", "direction", "passed", "status"],
    }[kind]
    selected_fields = [field for field in useful if field in fields]
    rendered_rows = [[_esc(row.get(field, "")) if row.get(field, "") != "" else "—" for field in selected_fields] for row in rows]
    return _semantic_table(f"table-{kind}", f"Registered {kind} rows. Empty fields are displayed as em dashes, never zeros.", [(field, field.replace("_", " ")) for field in selected_fields], rendered_rows, "raw-table")


def _experiment_sections(bundle: Mapping[str, Any], resolved_plot_html: Mapping[str, str], resolved_plots: Sequence[Mapping[str, Any]]) -> str:
    plots_by_phase: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for plot in resolved_plots:
        plots_by_phase[plot["phase"]].append(plot)
    chunks: list[str] = []
    for experiment in bundle["experiment_registry"]["experiments"]:
        if experiment["phase"] == "REPORTING":
            continue
        phase = experiment["phase"]
        gates = "".join(f"<li>{_esc(gate)}</li>" for gate in experiment["promotion_gates"])
        plot_cards: list[str] = []
        phase_plots = [
            plot for plot in plots_by_phase.get(phase, [])
            if plot.get("selection", {}).get("experiment_id", "") in {"", experiment["experiment_id"]}
        ]
        for plot in phase_plots:
            legends = "".join(f"<li>{_esc(item)}</li>" for item in plot["legend"])
            plot_cards.append(f'''<figure id="{_esc(plot['plot_id'])}" class="plot-card {plot['render_status'].lower()}">
              <header><span class="eyebrow">{_esc(plot['phase'])} · {_esc(plot['kind'])} · {_esc(plot['render_status'])}</span><h4>{_esc(plot['title'])}</h4></header>
              <div class="plot-frame">{resolved_plot_html[plot['plot_id']]}</div>
              <figcaption><strong>Figure contract.</strong> {_esc(plot['caption'])}
                <dl class="caption-meta"><dt>Comparison group</dt><dd>{_esc(plot['comparison_group'])}</dd><dt>Bootstrap</dt><dd>{_esc(plot['bootstrap_definition'])}</dd><dt>Selection rule</dt><dd>{_esc(plot['selection_rule'])}</dd><dt>Source</dt><dd><code>{_esc(plot['source_table'])}</code> · SHA <code>{_esc(plot['source_sha256'])}</code></dd></dl>
                <details><summary>Legend</summary><ul>{legends}</ul></details>
              </figcaption>
            </figure>''')
        chunks.append(f'''<section class="experiment-block" id="experiment-{_esc(experiment['experiment_id'].lower())}">
          <header class="experiment-header"><span class="phase-pill">{_esc(phase)}</span><div><h3>{_esc(experiment['display_name'])}</h3><p>{_esc(experiment['question'])}</p></div>{_status_badge(experiment['execution_status'], 'execution')}</header>
          <div class="experiment-contract"><div><strong>Prerequisite</strong><p>{_esc(experiment['prerequisite'])}</p></div><div><strong>Bootstrap</strong><p>{_esc(experiment['bootstrap'])}</p></div><div><strong>Promotion gates</strong><ul>{gates}</ul></div></div>
          <div class="plots-grid">{''.join(plot_cards) if plot_cards else '<p class="empty">No plot contracts registered.</p>'}</div>
        </section>''')
    return "".join(chunks)


def _claim_ledger(bundle: Mapping[str, Any]) -> str:
    rows = []
    for claim in bundle["claims"]["claims"]:
        refs = []
        for ref in claim["evidence_refs"]:
            if ref.startswith("PLOT_"):
                target = ref
            elif ref == "TABLE_VARIANTS":
                target = "table-variants"
            elif ref == "TABLE_METRICS":
                target = "table-metrics"
            elif ref == "TABLE_CONTRASTS" or ref.startswith("CONTRAST:"):
                target = "table-contrasts"
            elif ref == "TABLE_GATES":
                target = "table-gates"
            else:
                target = "part-provenance"
            refs.append(f'<a href="#{_esc(target)}"><code>{_esc(ref)}</code></a>')
        summary = claim.get("statistical_summary")
        if summary is None:
            estimate = "—"
        else:
            estimate = (
                f'<strong>{_esc(summary["metric"])}</strong><br>'
                f'Δ {_fmt(summary["point_delta"])}; CI [{_fmt(summary["ci_low"])}, {_fmt(summary["ci_high"])}]<br>'
                f'benefit &gt; {_fmt(summary["benefit_bound"])}; harm &lt; {_fmt(summary["harm_bound"])}<br>'
                f'<span class="muted">{_esc(summary["bound_basis"])} {_esc(summary["multiplicity"])}</span>'
            )
        rows.append([
            f'<code>{_esc(claim["claim_id"])}</code><br>{_esc(claim["text"])}',
            _status_badge(claim["verdict"], "claim"), estimate, _esc(claim["task_scope"]), "<br>".join(refs),
            _esc(claim["worst_case_behavior"]), _esc(claim["claim_boundary"]),
            "Yes" if claim["fresh_confirmation_required"] else "No",
        ])
    headers = (("claim", "Claim"), ("verdict", "Statistical verdict"), ("estimate", "Delta / CI / bounds"), ("scope", "Task / population"), ("evidence", "Evidence"), ("worst", "Worst case"), ("boundary", "Boundary"), ("fresh", "Fresh confirmation"))
    policy = bundle["experiment_registry"]["statistical_status_contract"]
    policy_panel = (
        '<div class="section-intro"><p><strong>Statistical status is separate from execution, decision, and evidence.</strong> '
        'An interval crossing zero means the directional improvement claim is unsupported; it is not equality, generic failure, or rejection. '
        f'{_esc(policy["continuation_rule"])} {_esc(policy["display_rule"])}</p></div>'
    )
    return policy_panel + _semantic_table("table-claims", "Claim–evidence ledger with raw deltas, intervals, practical bounds, and multiplicity status. Failed gates cannot be overridden by prose.", headers, rows, "claim-table")


def _examples_section(bundle: Mapping[str, Any]) -> str:
    examples = bundle["examples"]
    by_category = {row["category"]: row for row in examples["examples"]}
    cards = []
    for category in examples["required_categories"]:
        example = by_category.get(category)
        if example is None:
            content = '<strong>NO_ELIGIBLE_CASE</strong><p>Winner is not frozen or this category has no eligible registered trace. No manual substitute is allowed.</p>'
        else:
            content = f'<strong>{_esc(example.get("source_question_id", ""))}</strong><p>{_esc(example.get("summary", ""))}</p>'
        cards.append(f'<article class="case-card"><span class="eyebrow">{_esc(category)}</span>{content}</article>')
    return f'''<p>Seed <code>{examples['selection_seed']}</code>. {_esc(examples['selection_rule'])}</p><div class="case-grid">{''.join(cards)}</div>'''


def _manifest_input_rows(manifest: Mapping[str, Any]) -> str:
    rows = [[f'<code>{_esc(row["path"])}</code>', _esc(row["role"]), f'<code>{_esc(row["sha256"])}</code>', str(row["bytes"])] for row in manifest["inputs"]]
    return _semantic_table("table-provenance", "Files bound into this report build.", (("path", "Artifact"), ("role", "Role"), ("sha", "SHA256"), ("bytes", "Bytes")), rows, "provenance-table")


def _css() -> str:
    return r'''
:root{--ink:#142133;--muted:#617085;--line:#d7dee8;--paper:#f5f7fb;--card:#fff;--navy:#172f55;--blue:#2463eb;--teal:#008b86;--green:#15805d;--amber:#ad6800;--red:#b52a38;--gray:#788494;--context:#eef0f3;--shadow:0 8px 28px rgba(23,47,85,.08)}
*{box-sizing:border-box}html{scroll-behavior:smooth;max-width:100%;overflow-x:hidden}body{margin:0;max-width:100%;overflow-x:hidden;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--blue)}code{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}header.hero{background:linear-gradient(135deg,#0d1d37,#193e68 63%,#126b70);color:#fff;padding:64px max(24px,calc((100vw - 1260px)/2)) 48px}.hero h1{font-size:clamp(34px,5vw,64px);line-height:1.02;margin:.2em 0}.hero .kicker{text-transform:uppercase;letter-spacing:.18em;font-size:12px;color:#9fdce0}.hero p{max-width:850px;font-size:18px;color:#d9e6f2}.hero-status{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px}.shell{max-width:1340px;margin:auto;padding:0 24px 80px}.shell,section,.appendix,.plots-grid,.plot-card,.experiment-block,.method-card{min-width:0}.toc{position:sticky;top:0;z-index:10;margin:0 -24px 28px;padding:10px 24px;background:rgba(245,247,251,.94);backdrop-filter:blur(12px);border-bottom:1px solid var(--line);display:flex;gap:8px;overflow:auto}.toc a{white-space:nowrap;text-decoration:none;color:var(--navy);font-weight:700;padding:8px 12px;border-radius:99px}.toc a:hover,.toc a:focus{background:#e4ebf8}section.part{scroll-margin-top:72px;margin:38px 0 64px}.part>header{display:grid;grid-template-columns:auto 1fr;gap:18px;align-items:start;margin-bottom:24px}.part-number{font-size:12px;letter-spacing:.12em;font-weight:800;color:var(--blue);border:1px solid #a9c2f7;border-radius:99px;padding:6px 10px}.part h2{font-size:clamp(27px,3vw,42px);line-height:1.1;margin:0}.part>header p{grid-column:2;margin:0;color:var(--muted);max-width:850px}.subheading{margin:34px 0 14px}.section-intro,.context-note{color:var(--muted);max-width:900px}.card-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card-grid>*,.case-grid>*,.experiment-contract>*,.provenance-summary>*{min-width:0}.method-card{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:14px;padding:20px;box-shadow:var(--shadow);scroll-margin-top:80px}.method-card.context-card{background:var(--context);border-top-color:var(--gray);box-shadow:none}.method-card h3{margin:4px 0 2px;font-size:20px}.method-card header code{color:var(--muted)}.method-summary{font-size:16px}.method-card dl,.caption-meta{display:grid;grid-template-columns:minmax(110px,.38fr) 1fr;gap:7px 12px}.method-card dt,.caption-meta dt{font-weight:800;color:var(--navy)}.method-card dd,.caption-meta dd{margin:0}.method-card footer{border-top:1px solid var(--line);margin-top:16px;padding-top:12px;display:flex;flex-wrap:wrap;gap:5px}.eyebrow{font-size:11px;font-weight:850;letter-spacing:.1em;text-transform:uppercase;color:var(--teal)}.badge{display:inline-block;border:1px solid currentColor;border-radius:99px;padding:3px 7px;font-size:10px;font-weight:850;letter-spacing:.04em;white-space:nowrap}.execution-complete,.decision-promoted,.claim-supported{color:var(--green);background:#e8f7f1}.execution-planned,.decision-pending{color:var(--blue);background:#edf3ff}.execution-context-only,.evidence-context-only,.claim-descriptive{color:var(--gray);background:#eef0f3}.execution-hard-fail,.decision-rejected,.claim-not-supported,.claim-blocked{color:var(--red);background:#fff0f1}.execution-blocked,.execution-not-run-by-gate,.decision-no-promotion,.claim-noninferior-only{color:var(--amber);background:#fff7e8}.decision-processbench-specialist,.decision-prmbench-specialist,.claim-pb-specialist,.claim-prm-specialist{color:#7a3bc2;background:#f5edff}.evidence-retrospective{color:var(--amber);background:#fff7e8}.evidence-development{color:var(--blue);background:#edf3ff}.evidence-transfer,.evidence-fresh-confirmation{color:var(--green);background:#e8f7f1}.controls{display:flex;flex-wrap:wrap;gap:10px;background:#fff;border:1px solid var(--line);padding:14px;border-radius:12px;margin-bottom:12px}.controls label{font-size:12px;font-weight:800;color:var(--navy)}.controls select,.controls input{display:block;margin-top:4px;border:1px solid #aeb9c8;border-radius:7px;padding:8px;background:#fff;min-width:150px}.controls button,.download-btn{border:0;border-radius:8px;padding:9px 12px;background:var(--navy);color:#fff;font-weight:800;cursor:pointer}.table-wrap{overflow:auto;max-width:100%;background:#fff;border:1px solid var(--line);border-radius:12px;margin:12px 0 24px}table{border-collapse:collapse;width:100%;min-width:940px}caption{text-align:left;font-weight:800;padding:13px 15px;background:#edf2f8;color:var(--navy)}th,td{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top}th{background:#f7f9fc;color:var(--navy);font-size:12px;position:sticky;top:0}th[data-key]{cursor:pointer}tbody tr:hover{background:#f8fbff}.muted,.empty{color:var(--muted)}.empty{text-align:center;padding:30px}.plot-card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0;box-shadow:var(--shadow);scroll-margin-top:80px}.plot-card h4{font-size:20px;margin:3px 0 12px}.plot-frame{overflow:auto;max-width:100%;border:1px solid #e3e8ef;border-radius:9px;background:#fbfcfe;padding:8px}.plot-frame svg{min-width:760px;width:100%;height:auto}.pending-plot{min-height:190px;display:grid;place-items:center;text-align:center;color:var(--muted);padding:28px}.pending-plot strong{color:var(--blue)}.pending-icon{font-size:34px;color:#a8b7cd}.plot-card figcaption{margin-top:12px;color:var(--muted)}.caption-meta{margin-top:12px;font-size:13px}.plots-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:12px}.experiment-block{margin:30px 0 48px;padding-top:20px;border-top:2px solid var(--line);scroll-margin-top:75px}.experiment-header{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:14px;align-items:start}.experiment-header h3{margin:0;font-size:25px}.experiment-header p{margin:.3em 0;color:var(--muted)}.phase-pill{display:grid;place-items:center;background:var(--navy);color:#fff;border-radius:10px;min-width:56px;height:42px;font-weight:850}.experiment-contract{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr) minmax(0,1.8fr);gap:12px;margin:16px 0}.experiment-contract>div{background:#eaf0f7;border-radius:9px;padding:12px}.experiment-contract p,.experiment-contract ul{margin:5px 0}.case-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.case-card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;min-height:130px}.provenance-summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.stat{background:var(--navy);color:#fff;padding:17px;border-radius:12px}.stat span{display:block;color:#b9d2ea;font-size:12px}.stat strong{font-size:20px}.axis{stroke:#677489}.zero{stroke:#8a96a8;stroke-dasharray:5 4}.interval{stroke:#2463eb;stroke-width:3}.point{fill:#2463eb}.context-point{fill:#788494}.tick,.value,.label{fill:#394a61;font-size:12px}.phase-bg{fill:#f8faff;stroke:#dfe5ed}.context-bg{fill:#eef0f3}.phase-label{fill:#172f55;font-weight:800;font-size:14px}.edge{fill:none;stroke:#98a8bc;stroke-width:1.5}.node rect{stroke:#2463eb;stroke-width:2;fill:#edf3ff}.node text{font-size:10px;font-weight:800;fill:#172f55}.node .node-status{font-size:8px;font-weight:600}.node.evidence-context-only rect{stroke:#788494;fill:#eef0f3}.node.evidence-retrospective rect{stroke:#ad6800;stroke-dasharray:5 3}.node.evidence-transfer rect{stroke:#15805d}.node.decision-promoted rect{fill:#e8f7f1}.node.decision-rejected rect{fill:#fff0f1}.node.decision-processbench-specialist rect,.node.decision-prmbench-specialist rect{fill:#f5edff}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.appendix details{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0;min-width:0;max-width:100%;overflow:hidden}.appendix summary{font-weight:800;cursor:pointer}.no-print{display:block}
.execution-running{color:var(--teal);background:#e6f7f6}.case-card{overflow-wrap:anywhere}.waterfall-up{fill:#2463eb}.waterfall-down{fill:#b52a38}.waterfall-link{stroke:#98a8bc;stroke-width:1.5;stroke-dasharray:4 3}.heat-missing{fill:#e5e8ed}.heat-value{fill:#12233b;font-size:11px;font-weight:700}.gate-pass{fill:#b9ead8}.gate-fail{fill:#ffc9cf}.gate-pending{fill:#e4e8ee}.gate-text{fill:#24364e;font-size:10px;font-weight:850}.scatter-point{fill:#2463eb;stroke:#fff;stroke-width:2}
.pipeline-node{fill:#edf3ff;stroke:#2463eb;stroke-width:2}.pipeline-transfer-node{fill:#e8f7f1;stroke:#15805d;stroke-width:2}.pipeline-arrow{stroke:#65758a;stroke-width:2.5}.pipeline-head{fill:none;stroke:#65758a;stroke-width:2.5}.pipeline-transfer{fill:none;stroke:#15805d;stroke-width:2.5;stroke-dasharray:7 5}.pipeline-text{fill:#172f55;font-size:12px;font-weight:800}.pipeline-note{fill:#617085;font-size:10px}
@media(max-width:760px){header.hero{padding:42px 20px}.shell{padding:0 14px 60px}.toc{margin:0 -14px;padding:8px 14px}.card-grid,.case-grid,.provenance-summary,.experiment-contract{grid-template-columns:1fr}.part>header{grid-template-columns:1fr}.part>header p{grid-column:1}.method-card{padding:16px}.method-card dl,.caption-meta{grid-template-columns:1fr}.method-card dd,.caption-meta dd{margin-bottom:8px}.experiment-header{grid-template-columns:auto 1fr}.experiment-header>.badge{grid-column:2}.controls label{width:100%}.controls select,.controls input{width:100%;min-width:0}.plot-frame svg{min-width:680px}}
@media print{body{background:#fff;font-size:10pt}.hero{background:#fff!important;color:#111!important;padding:0!important}.hero p,.hero .kicker{color:#333!important}.shell{max-width:none;padding:0}.toc,.controls,.download-btn,.no-print{display:none!important}.method-card,.plot-card,.case-card,.table-wrap{box-shadow:none;break-inside:avoid}.card-grid{grid-template-columns:1fr 1fr}section.part{page-break-before:always}.table-wrap{overflow:visible}table{min-width:0;font-size:8pt}.plot-frame svg{min-width:0}.pending-plot{min-height:90px}}
'''


def _javascript() -> str:
    return r'''
(()=>{const table=document.getElementById('table-variants');const rows=[...table.tBodies[0].rows];const filters=['phase','task','execution','evidence'];const normalize=s=>(s||'').toLowerCase();function apply(){const q=normalize(document.getElementById('filter-query').value);const values=Object.fromEntries(filters.map(k=>[k,normalize(document.getElementById('filter-'+k).value)]));rows.forEach(row=>{const text=normalize(row.innerText);const cells=[...row.cells];const map={phase:cells[1]?.innerText,task:cells[2]?.innerText,execution:cells[5]?.innerText,evidence:cells[7]?.innerText};const visible=(!q||text.includes(q))&&filters.every(k=>!values[k]||normalize(map[k]).includes(values[k]));row.hidden=!visible;});}filters.forEach(k=>document.getElementById('filter-'+k).addEventListener('change',apply));document.getElementById('filter-query').addEventListener('input',apply);document.getElementById('reset-filters').addEventListener('click',()=>{document.getElementById('filter-query').value='';filters.forEach(k=>document.getElementById('filter-'+k).value='');apply();});document.querySelectorAll('table.sortable th').forEach((th,index)=>{let direction=1;th.tabIndex=0;th.setAttribute('role','button');const sort=()=>{rows.sort((a,b)=>a.cells[index].innerText.localeCompare(b.cells[index].innerText,undefined,{numeric:true})*direction);direction*=-1;rows.forEach(row=>table.tBodies[0].appendChild(row));apply();};th.addEventListener('click',sort);th.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();sort();}});});function csvCell(v){const s=String(v??'');return /[",\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;}document.querySelectorAll('[data-download]').forEach(button=>button.addEventListener('click',()=>{const key=button.dataset.download;const payload=JSON.parse(document.getElementById('report-data').textContent);const rows=payload[key];if(!Array.isArray(rows)||!rows.length){alert('No registered rows to download.');return;}const fields=[...new Set(rows.flatMap(Object.keys))];const csv=[fields.join(','),...rows.map(row=>fields.map(field=>csvCell(row[field])).join(','))].join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=key.toUpperCase()+'.csv';a.click();URL.revokeObjectURL(url);}));})();
'''


def render_report(bundle: Mapping[str, Any], manifest_shell: Mapping[str, Any], resolved_plots: Sequence[Mapping[str, Any]]) -> bytes:
    variants = bundle["variant_registry"]["variants"]
    variant_map = {row["variant_id"]: row for row in variants}
    resolved_plot_html: dict[str, str] = {}
    for plot in resolved_plots:
        rendered, status = _render_plot(plot, bundle, variant_map)
        if status != plot["render_status"]:
            raise ReportingValidationError(f"plot render status drift for {plot['plot_id']}")
        resolved_plot_html[plot["plot_id"]] = rendered
    lineage = resolved_plot_html["PLOT_LINEAGE"]
    historical = resolved_plot_html["PLOT_HISTORICAL_ANCHOR"]
    historical_spec = next(plot for plot in resolved_plots if plot["plot_id"] == "PLOT_HISTORICAL_ANCHOR")
    robustness = next(plot for plot in resolved_plots if plot["plot_id"] == "PLOT_ROBUSTNESS_WORST_UNIT")
    data_payload = {
        "methods": bundle["method_registry"]["methods"],
        "variants": variants,
        "experiments": bundle["experiment_registry"]["experiments"],
        "metrics": bundle["metrics"],
        "contrasts": bundle["contrasts"],
        "gates": bundle["gates"],
        "claims": bundle["claims"]["claims"],
        "examples": bundle["examples"],
        "plots": resolved_plots,
    }
    payload_text = json.dumps(data_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    filter_phases = sorted({row["phase"] for row in variants}, key=lambda phase: PHASE_ORDER.get(phase, 99))
    filter_tasks = ["ProcessBench", "PRMBench", "Early", "Audit / context"]
    execution = bundle["variant_registry"]["allowed_execution_statuses"]
    evidence = bundle["variant_registry"]["allowed_evidence_statuses"]
    experiments_by_id = {row["experiment_id"]: row for row in bundle["experiment_registry"]["experiments"]}
    p0_status = experiments_by_id["P0_BRIDGE"]["execution_status"]
    p0_badge = f'<span class="badge execution-{_esc(p0_status.lower())}">P0 {_esc(p0_status)}</span>'
    later_statuses = {experiments_by_id[experiment_id]["execution_status"] for experiment_id in ("P1_BASELINES", "P2_ATOMIC", "P3_FUSION", "P4_PRMBENCH_TRANSFER", "P5_EARLY_TRANSFER")}
    later_label = "P1–P5 PLANNED" if later_statuses == {"PLANNED"} else "P1–P5 MIXED STATUS"
    later_css = "execution-planned" if later_statuses == {"PLANNED"} else "execution-running"
    later_badge = f'<span class="badge {later_css}">{later_label}</span>'
    options = lambda values: "".join(f'<option value="{_esc(value)}">{_esc(value)}</option>' for value in values)
    html_text = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reasoning Localization 0.3662 — Research Program</title><style>{_css()}</style></head>
<body><header class="hero" id="top"><span class="kicker">Research program · living evidence report</span><h1>Reasoning Localization<br>0.3662</h1><p>A compact, evidence-gated program for first-error localization on ProcessBench, transfer to PRMBench step localization, and lower-priority causal early detection.</p><div class="hero-status">{_status_badge('REPORTING COMPLETE','execution')} {p0_badge} {later_badge} {_status_badge('NO COMMON-PROTOCOL WINNER','decision')}</div></header>
<main class="shell"><nav class="toc" aria-label="Report sections"><a href="#part-methods">I Methods</a><a href="#part-variants">II Variants</a><a href="#part-experiments">III Experiments</a><a href="#part-claims">IV Claims</a><a href="#part-cases">V Trace cases</a><a href="#part-provenance">VI Provenance</a></nav>
<section class="part" id="part-methods"><header><span class="part-number">PART I</span><h2>Methods and novelty</h2><p>Definitions and hypotheses first; outcomes later. Historical methods are visibly isolated.</p></header>{_method_cards(bundle)}</section>
<section class="part" id="part-variants"><header><span class="part-number">PART II</span><h2>All variants at a glance</h2><p>Execution, decision, and evidence are independent fields. ProcessBench, PRMBench, and Early stay in separate columns.</p></header>
<div class="controls no-print" aria-label="Variant filters"><label>Search<input id="filter-query" type="search" placeholder="variant or method"></label><label>Phase<select id="filter-phase"><option value="">All</option>{options(filter_phases)}</select></label><label>Task<select id="filter-task"><option value="">All</option>{options(filter_tasks)}</select></label><label>Execution<select id="filter-execution"><option value="">All</option>{options(execution)}</select></label><label>Evidence<select id="filter-evidence"><option value="">All</option>{options(evidence)}</select></label><button id="reset-filters" type="button">Reset</button><button class="download-btn" type="button" data-download="variants">Download roster CSV</button></div>
{_variant_table(bundle)}
<figure id="PLOT_LINEAGE" class="plot-card rendered"><header><span class="eyebrow">ALL · lineage · RENDERED</span><h4>Variant development lineage</h4></header><div class="plot-frame">{lineage}</div><figcaption>Edges are registered parent relationships. Fill is decision status; border style is evidence grade. Context is gray.</figcaption></figure>
</section>
<section class="part" id="part-experiments"><header><span class="part-number">PART III</span><h2>Experiments and evidence</h2><p>The live report exposes the whole preregistered ladder. A panel renders only from exact registered rows; unexecuted states stay visibly PLANNED. Missing is not zero.</p></header>
<figure id="PLOT_HISTORICAL_ANCHOR" class="plot-card rendered"><header><span class="eyebrow">CONTEXT · historical anchor · RENDERED</span><h4>{_esc(historical_spec['title'])}</h4></header><div class="plot-frame">{historical}</div><figcaption>{_esc(historical_spec['caption'])}<dl class="caption-meta"><dt>Comparison group</dt><dd>{_esc(historical_spec['comparison_group'])}</dd><dt>Selection rule</dt><dd>{_esc(historical_spec['selection_rule'])}</dd><dt>Source</dt><dd><code>{_esc(historical_spec['source_table'])}</code> · SHA <code>{_esc(historical_spec['source_sha256'])}</code></dd></dl></figcaption></figure>
{_experiment_sections(bundle, resolved_plot_html, resolved_plots)}
<section class="experiment-block" id="experiment-robustness"><header class="experiment-header"><span class="phase-pill">R</span><div><h3>Cross-phase robustness audits</h3><p>Worst-unit behavior, W/T/L, missing-score, and population audits remain mandatory companions to headline results.</p></div>{_status_badge('PLANNED','execution')}</header><figure id="PLOT_ROBUSTNESS_WORST_UNIT" class="plot-card pending"><header><span class="eyebrow">ROBUSTNESS · {_esc(robustness['kind'])} · {_esc(robustness['render_status'])}</span><h4>{_esc(robustness['title'])}</h4></header><div class="plot-frame">{resolved_plot_html['PLOT_ROBUSTNESS_WORST_UNIT']}</div><figcaption>{_esc(robustness['caption'])}<dl class="caption-meta"><dt>Comparison group</dt><dd>{_esc(robustness['comparison_group'])}</dd><dt>Bootstrap</dt><dd>{_esc(robustness['bootstrap_definition'])}</dd><dt>Selection rule</dt><dd>{_esc(robustness['selection_rule'])}</dd><dt>Source</dt><dd><code>{_esc(robustness['source_table'])}</code> · SHA <code>{_esc(robustness['source_sha256'])}</code></dd></dl></figcaption></figure></section>
</section>
<section class="part" id="part-claims"><header><span class="part-number">PART IV</span><h2>Claim–evidence ledger</h2><p>Claims are machine-linked to registered plots, tables, contrasts, and manifests.</p></header>{_claim_ledger(bundle)}</section>
<section class="part" id="part-cases"><header><span class="part-number">PART V</span><h2>Deterministic trace cases</h2><p>Four winner cases will be selected by the frozen hash rule, never by visual appeal.</p></header>{_examples_section(bundle)}</section>
<section class="part" id="part-provenance"><header><span class="part-number">PART VI</span><h2>Limitations and provenance</h2><p>This living release includes the P0-S0 checksum audit. It does not validate or promote a new localizer.</p></header>
<div class="provenance-summary"><div class="stat"><span>Source commit</span><strong><code>{_esc(manifest_shell['source_commit'][:12])}</code></strong></div><div class="stat"><span>Embedded bundle SHA</span><strong><code>{_esc(manifest_shell['embedded_data_sha256'][:12])}</code></strong></div><div class="stat"><span>Build status</span><strong>Deterministic</strong></div></div>
<h3>Evaluator and leakage contract</h3><ul><li>ProcessBench estimates one first-error-or-clean decision per response; PRMBench estimates dense step ranking on error responses. They are not the same evaluator contract.</li><li>No score, threshold, scaling, orientation, or weight may be retuned on the transfer task after freeze.</li><li>Calibration donors, held-fold thresholds, source-question groups, response groups, and evaluation rows must be disjoint according to the experiment registry.</li><li>Historical Stage-4 rows are retrospective context only. Fresh common-population results must carry exact population and comparison-group hashes.</li><li>Missing, blocked, single-class, and gate-stopped results remain statuses, never numeric zeros.</li></ul>
{_manifest_input_rows(manifest_shell)}
<div class="appendix"><h3>Appendices and embedded downloads</h3><p>All downloadable CSVs are reconstructed in-browser from the exact JSON embedded below; there are no external data dependencies.</p><div class="no-print"><button class="download-btn" data-download="metrics">Metrics CSV</button> <button class="download-btn" data-download="contrasts">Contrasts CSV</button> <button class="download-btn" data-download="gates">Gates CSV</button></div><details open><summary>Metrics</summary>{_raw_table('metrics', bundle['metric_fields'], bundle['metrics'])}</details><details><summary>Contrasts</summary>{_raw_table('contrasts', bundle['contrast_fields'], bundle['contrasts'])}</details><details><summary>Gates</summary>{_raw_table('gates', bundle['gate_fields'], bundle['gates'])}</details></div>
</section></main><script id="report-data" type="application/json">{payload_text}</script><script>{_javascript()}</script></body></html>'''
    return html_text.encode("utf-8")


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _resolved_plots(bundle: Mapping[str, Any], report_dir: Path) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    variant_map = {row["variant_id"]: row for row in bundle["variant_registry"]["variants"]}
    for spec in bundle["plot_manifest"]["plots"]:
        row = dict(spec)
        source_path = report_dir / spec["source_table"]
        row["source_sha256"] = sha256_file(source_path)
        _, render_status = _render_plot(spec, bundle, variant_map)
        row["render_status"] = render_status
        resolved.append(row)
    return resolved


def prepare_build(report_dir: Path, repo_root: Path, generator_path: Path | None = None) -> ReportBuild:
    report_dir = report_dir.resolve()
    repo_root = repo_root.resolve()
    bundle = load_bundle(report_dir)
    source_artifacts = validate_bundle(bundle, report_dir, repo_root)
    generator = (generator_path or Path(__file__)).resolve()
    input_paths = [report_dir / name for name in REGISTRY_FILES] + [repo_root / path for path in source_artifacts] + [generator]
    unique_paths = sorted({path.resolve() for path in input_paths}, key=lambda path: _repo_relative(path, repo_root))
    inputs = []
    for path in unique_paths:
        relative = _repo_relative(path, repo_root)
        if relative in source_artifacts:
            role = "registered_result_artifact" if "/phase_" in relative else "historical_source_artifact"
        elif path == generator:
            role = "report_generator"
        else:
            role = "report_input"
        inputs.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size, "role": role})
    resolved_plots = _resolved_plots(bundle, report_dir)
    embedded = {
        "methods": bundle["method_registry"]["methods"], "variants": bundle["variant_registry"]["variants"],
        "experiments": bundle["experiment_registry"]["experiments"], "metrics": bundle["metrics"],
        "contrasts": bundle["contrasts"], "gates": bundle["gates"], "claims": bundle["claims"]["claims"],
        "examples": bundle["examples"], "plots": resolved_plots,
    }
    manifest_shell = {
        "schema": "reasoning_localization_report_manifest_v1",
        "report_id": REPORT_ID,
        "source_commit": _git_head(repo_root),
        "inputs": inputs,
        "embedded_data_sha256": sha256_bytes(canonical_json_bytes(embedded)),
        "plots": [
            {
                "plot_id": row["plot_id"],
                "source_table": row["source_table"],
                "source_sha256": row["source_sha256"],
                "render_status": row["render_status"],
                "selection": row["selection"],
                "selection_rule": row["selection_rule"],
                "comparison_group": row["comparison_group"],
                "bootstrap_definition": row["bootstrap_definition"],
            }
            for row in resolved_plots
        ],
    }
    html_bytes = render_report(bundle, manifest_shell, resolved_plots)
    manifest = dict(manifest_shell)
    report_relative = _repo_relative(report_dir / "REPORT.html", repo_root)
    manifest["output"] = {"path": report_relative, "sha256": sha256_bytes(html_bytes), "bytes": len(html_bytes)}
    manifest["report_manifest_sha256"] = sha256_bytes(canonical_json_bytes({key: value for key, value in manifest.items() if key != "report_manifest_sha256"}))
    return ReportBuild(html_bytes=html_bytes, manifest=manifest, resolved_plots=resolved_plots)


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_build(report_dir: Path, build: ReportBuild) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "REPORT.html").write_bytes(build.html_bytes)
    (report_dir / "REPORT_MANIFEST.json").write_bytes(manifest_bytes(build.manifest))


def check_build(report_dir: Path, build: ReportBuild) -> None:
    expected = {"REPORT.html": build.html_bytes, "REPORT_MANIFEST.json": manifest_bytes(build.manifest)}
    for name, content in expected.items():
        path = report_dir / name
        if not path.is_file():
            raise ReportingValidationError(f"missing generated output: {path}")
        if path.read_bytes() != content:
            raise ReportingValidationError(f"generated output is stale or nondeterministic: {path}")


def create_immutable_snapshot(report_dir: Path, label: str, build: ReportBuild, repo_root: Path) -> Path:
    if not re.fullmatch(r"(?:reporting|phase_[0-5]|amendment_[a-z0-9_]+)", label):
        raise ReportingValidationError("snapshot label must be reporting, phase_0..phase_5, or amendment_<slug>")
    snapshot_dir = report_dir / "snapshots" / label
    snapshot_report = snapshot_dir / "REPORT.html"
    snapshot_manifest_path = snapshot_dir / "REPORT_MANIFEST.json"
    snapshot_manifest = json.loads(json.dumps(build.manifest))
    snapshot_manifest["snapshot"] = {"label": label, "immutable": True}
    snapshot_manifest["output"]["path"] = _repo_relative(snapshot_report, repo_root)
    snapshot_manifest["report_manifest_sha256"] = sha256_bytes(canonical_json_bytes({key: value for key, value in snapshot_manifest.items() if key != "report_manifest_sha256"}))
    expected_manifest = manifest_bytes(snapshot_manifest)
    if snapshot_dir.exists():
        if not snapshot_report.is_file() or not snapshot_manifest_path.is_file():
            raise ReportingValidationError(f"existing snapshot is incomplete and cannot be overwritten: {snapshot_dir}")
        if snapshot_report.read_bytes() != build.html_bytes or snapshot_manifest_path.read_bytes() != expected_manifest:
            raise ReportingValidationError(f"immutable snapshot differs from requested build: {snapshot_dir}")
        return snapshot_dir
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    snapshot_report.write_bytes(build.html_bytes)
    snapshot_manifest_path.write_bytes(expected_manifest)
    return snapshot_dir
