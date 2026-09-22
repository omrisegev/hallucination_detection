"""Fail-closed published-comparator context for the frozen 24-cell report.

Published paper values in this project were measured on independently generated
populations.  They are useful context, but they are not rows in the common-v2
benchmark.  This module validates that boundary and projects the versioned
source registry into a small report-only artifact.  The projection is never a
metrics or contrasts table and every paper-to-v2 delta is explicitly null.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .schemas import RANKABLE_STATUSES, SchemaError, canonical_sha256


PUBLISHED_COMPARATOR_REGISTRY_SCHEMA = (
    "reconstruction-frozen24-published-comparator-registry-v1"
)
PUBLISHED_CONTEXT_SCHEMA = "reconstruction-frozen24-published-context-v1"
PUBLISHED_CONTEXT_STATUSES = (
    "PUBLISHED_CONTEXT_ONLY",
    "RELATED_PUBLISHED_CONTEXT_ONLY",
    "NO_PUBLISHED_COMPARATOR",
)
EXPECTED_STATUS_COUNTS = {
    "PUBLISHED_CONTEXT_ONLY": 17,
    "RELATED_PUBLISHED_CONTEXT_ONLY": 4,
    "NO_PUBLISHED_COMPARATOR": 3,
}
REQUIRED_MATCH_AXES = (
    "dataset_revision",
    "model",
    "row_ids",
    "generation",
    "labels_grader",
    "prediction_unit",
    "metric",
    "evaluation_protocol",
)
MATCH_AXIS_STATUSES = {
    "EXACT",
    "PARTIAL",
    "DIFFERENT",
    "UNKNOWN",
    "NOT_APPLICABLE",
}
DISPLAY_MODE_BY_STATUS = {
    "PUBLISHED_CONTEXT_ONLY": "SEPARATE_CONTEXT_CARD",
    "RELATED_PUBLISHED_CONTEXT_ONLY": "SEPARATE_RELATED_CONTEXT_CARD",
    "NO_PUBLISHED_COMPARATOR": "EXPLICIT_NO_ELIGIBLE_COMPARATOR",
}
ANCHOR_ROLE_BY_STATUS = {
    "PUBLISHED_CONTEXT_ONLY": "primary",
    "RELATED_PUBLISHED_CONTEXT_ONLY": "related",
    "NO_PUBLISHED_COMPARATOR": "none",
}
FIDELITY_BY_STATUS = {
    "PUBLISHED_CONTEXT_ONLY": "published context; different generated population",
    "RELATED_PUBLISHED_CONTEXT_ONLY": "related published context; different population or protocol",
    "NO_PUBLISHED_COMPARATOR": "no eligible published comparator",
}
MAPPING_STATUSES_BY_STATUS = {
    "PUBLISHED_CONTEXT_ONLY": {
        "VERIFIED_PRIMARY_SOURCE",
        "VERIFIED_PRIMARY_SOURCE_TABLE_CORRECTED",
        "CORRECTED_PRIMARY_SOURCE",
    },
    "RELATED_PUBLISHED_CONTEXT_ONLY": {
        "VERIFIED_PRIMARY_SOURCE_RELATED_CONTEXT"
    },
    "NO_PUBLISHED_COMPARATOR": {"NOT_AVAILABLE"},
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEADER_TOLERANCE = 5.1e-7
_PROJECTION_FIELDS = {
    "schema_version",
    "release_id",
    "source_registry_schema_version",
    "source_registry_file_sha256",
    "source_registry_content_sha256",
    "metrics_cell_auroc_logical_sha256",
    "status_counts",
    "ranking_policy",
    "delta_policy",
    "rows",
}
_PROJECTED_ROW_FIELDS = {
    "cell_id",
    "comparison_status",
    "anchor_role",
    "allowed_display",
    "fidelity",
    "published_method",
    "published_auroc",
    "mapping_status",
    "supervision",
    "access",
    "passes",
    "source_id",
    "source_title",
    "source_url",
    "source_table",
    "match_axes",
    "mismatch_reasons",
    "common_replay_status",
    "delta_eligible",
    "paper_to_v2_delta",
}
_RANKING_POLICY = (
    "report-only context; never a metric, contrast, rank, forest, or heatmap row"
)
_DELTA_POLICY = "paper-to-v2 delta forbidden; every projected delta is null"
_ATOMIC_METRIC_BINDING_FIELDS = (
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
    "aggregation_id",
    "aggregation_level",
    "metric_id",
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise SchemaError(f"duplicate JSON key {key!r} in {path}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def load_published_comparator_registry(path: str | Path) -> dict[str, Any]:
    """Load the source registry with duplicate-key rejection."""

    return _strict_json_object(
        Path(path).resolve(), label="published-comparator registry"
    )


def load_published_context_projection(path: str | Path) -> dict[str, Any]:
    """Load a projected report artifact with duplicate-key rejection."""

    return _strict_json_object(
        Path(path).resolve(), label="published-context projection"
    )


def _finite_number(value: Any, *, context: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{context} must be a finite number",
    )
    return float(value)


def _expected_cell_ids(values: Sequence[str]) -> tuple[str, ...]:
    cell_ids = tuple(str(value) for value in values)
    _require(len(cell_ids) == 24, "published context requires exactly 24 frozen cells")
    _require(len(set(cell_ids)) == 24, "frozen cell roster contains duplicates")
    _require(
        not any(value.startswith("aggregate::") for value in cell_ids),
        "published context cell roster may not contain aggregate rows",
    )
    return tuple(sorted(cell_ids, key=lambda value: value.encode("utf-8")))


def _metric_leaders(
    metrics_rows: Iterable[Mapping[str, Any]],
    *,
    expected_cell_ids: Sequence[str],
    expected_method_ids: Sequence[str],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    cells = set(expected_cell_ids)
    methods = set(str(value) for value in expected_method_ids)
    _require(len(methods) == 13, "frozen published context requires the 13 primary methods")
    by_cell: dict[str, dict[str, Mapping[str, Any]]] = {cell_id: {} for cell_id in cells}
    for row in metrics_rows:
        if (
            row.get("aggregation_level") != "cell"
            or row.get("metric_id") != "auroc"
            or row.get("cell_id") not in cells
        ):
            continue
        method_id = str(row.get("method_id", ""))
        if method_id not in methods:
            continue
        _require(
            row.get("status") in RANKABLE_STATUSES and row.get("value") is not None,
            f"published context cannot bind a non-rankable AUROC row: {row.get('cell_id')} / {method_id}",
        )
        _require(
            method_id not in by_cell[str(row["cell_id"])],
            f"duplicate cell AUROC method row: {row['cell_id']} / {method_id}",
        )
        by_cell[str(row["cell_id"])][method_id] = row
    for cell_id, rows in by_cell.items():
        _require(
            set(rows) == methods,
            f"published context metric roster mismatch for {cell_id}: expected 13 primary methods",
        )
    return by_cell


def frozen24_cell_auroc_logical_sha256(
    metrics_rows: Iterable[Mapping[str, Any]],
    *,
    expected_cell_ids: Sequence[str],
    expected_method_ids: Sequence[str],
) -> str:
    """Hash only the comparable 24 x 13 atomic AUROC scientific facts.

    The complete metrics CSV also contains reporting-only aggregates and their
    dataset metadata.  Those may change when the report taxonomy is corrected
    without changing any atomic benchmark result.  This projection therefore
    binds the paper-context registry to the exact per-cell method estimates,
    intervals, status, and sample sizes that its recorded v2 leaders cite.
    """

    by_cell = _metric_leaders(
        metrics_rows,
        expected_cell_ids=expected_cell_ids,
        expected_method_ids=expected_method_ids,
    )
    projection = [
        {field: row[field] for field in _ATOMIC_METRIC_BINDING_FIELDS}
        for cell_id in sorted(by_cell, key=lambda value: value.encode("utf-8"))
        for method_id, row in sorted(
            by_cell[cell_id].items(), key=lambda item: item[0].encode("utf-8")
        )
    ]
    return canonical_sha256(projection)


def _validate_recorded_leaders(
    cells: Sequence[Mapping[str, Any]],
    metric_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    expected_method_display_names: Mapping[str, str] | None = None,
) -> None:
    for cell in cells:
        cell_id = str(cell["cell_id"])
        leader = cell.get("v2_point_estimate_leader")
        _require(isinstance(leader, Mapping), f"{cell_id}: missing v2 point leader")
        method_id = str(leader.get("method_id", ""))
        _require(method_id in metric_rows[cell_id], f"{cell_id}: unknown v2 leader method")
        if expected_method_display_names is not None:
            _require(
                leader.get("display_name")
                == expected_method_display_names.get(method_id),
                f"{cell_id}: recorded v2 leader display name drift",
            )
        source = metric_rows[cell_id][method_id]
        source_value = _finite_number(source.get("value"), context=f"{cell_id} source AUROC")
        recorded_value = _finite_number(leader.get("auroc"), context=f"{cell_id} recorded AUROC")
        _require(
            abs(source_value - recorded_value) <= _LEADER_TOLERANCE,
            f"{cell_id}: recorded v2 leader AUROC does not match report metrics",
        )
        maximum = max(float(row["value"]) for row in metric_rows[cell_id].values())
        _require(
            abs(source_value - maximum) <= _LEADER_TOLERANCE,
            f"{cell_id}: recorded v2 leader is not a highest point estimate",
        )
        ci = leader.get("ci95")
        _require(
            isinstance(ci, list) and len(ci) == 2,
            f"{cell_id}: recorded v2 leader CI must have two values",
        )
        for recorded, field in zip(ci, ("ci_low", "ci_high")):
            source_ci = _finite_number(source.get(field), context=f"{cell_id} source {field}")
            recorded_ci = _finite_number(recorded, context=f"{cell_id} recorded {field}")
            _require(
                abs(source_ci - recorded_ci) <= _LEADER_TOLERANCE,
                f"{cell_id}: recorded v2 leader {field} does not match report metrics",
            )


def _validate_source_cell(
    cell: Mapping[str, Any],
    *,
    source_catalog: Mapping[str, Any],
) -> None:
    cell_id = str(cell.get("cell_id", ""))
    status = cell.get("comparison_status")
    _require(status in PUBLISHED_CONTEXT_STATUSES, f"{cell_id}: unknown context status")
    _require(
        cell.get("delta_eligible") is False,
        f"{cell_id}: every published comparator must set delta_eligible to false",
    )
    for key in cell:
        if key != "delta_eligible" and (key == "delta" or key.endswith("_delta")):
            raise SchemaError(f"{cell_id}: paper-to-v2 delta fields are forbidden")
    _require(
        cell.get("allowed_display") == DISPLAY_MODE_BY_STATUS[status],
        f"{cell_id}: context status and allowed_display disagree",
    )
    comparator = cell.get("published_comparator")
    _require(isinstance(comparator, Mapping), f"{cell_id}: missing published comparator object")
    _require(
        comparator.get("anchor_role") == ANCHOR_ROLE_BY_STATUS[status],
        f"{cell_id}: context status and anchor role disagree",
    )
    match_axes = cell.get("match_axes")
    _require(isinstance(match_axes, Mapping), f"{cell_id}: match_axes must be an object")
    _require(
        set(match_axes) == set(REQUIRED_MATCH_AXES),
        f"{cell_id}: published context must declare every match axis",
    )
    _require(
        set(match_axes.values()) <= MATCH_AXIS_STATUSES,
        f"{cell_id}: unknown match-axis status",
    )
    _require(
        comparator.get("mapping_status") in MAPPING_STATUSES_BY_STATUS[status],
        f"{cell_id}: comparison status and mapping status disagree",
    )
    reasons = cell.get("mismatch_reasons")
    _require(
        isinstance(reasons, list)
        and reasons
        and all(isinstance(reason, str) and reason.strip() for reason in reasons),
        f"{cell_id}: mismatch reasons must be nonempty strings",
    )
    if status == "NO_PUBLISHED_COMPARATOR":
        _require(
            all(comparator.get(field) is None for field in ("method", "auroc", "source_id", "table", "supervision", "access", "passes")),
            f"{cell_id}: no-comparator row contains paper metadata",
        )
        _require(
            set(match_axes.values()) == {"NOT_APPLICABLE"},
            f"{cell_id}: no-comparator axes must be NOT_APPLICABLE",
        )
        _require(
            cell.get("common_replay_status") == "NOT_APPLICABLE_NO_COMPARATOR",
            f"{cell_id}: no-comparator replay status must be not applicable",
        )
        return
    _require(
        any(value != "EXACT" for value in match_axes.values()),
        f"{cell_id}: context-only paper row claims every axis is exact",
    )
    _require(
        match_axes.get("row_ids") != "EXACT",
        f"{cell_id}: a context-only paper row may not claim exact row IDs",
    )
    _require(
        cell.get("common_replay_status") == "BLOCKED_PENDING_COMMON_ROW_RERUN",
        f"{cell_id}: context-only replay status must remain blocked",
    )
    for field in ("method", "source_id", "table", "supervision", "access", "passes"):
        _require(
            isinstance(comparator.get(field), str) and comparator[field].strip(),
            f"{cell_id}: published comparator {field} must be visible and nonempty",
        )
    auroc = _finite_number(comparator.get("auroc"), context=f"{cell_id} paper AUROC")
    _require(0.0 <= auroc <= 1.0, f"{cell_id}: paper AUROC is outside [0,1]")
    source_id = str(comparator["source_id"])
    _require(source_id in source_catalog, f"{cell_id}: unresolved paper source_id")
    source = source_catalog[source_id]
    _require(isinstance(source, Mapping), f"{cell_id}: paper source entry must be an object")
    _require(
        isinstance(source.get("title"), str) and source["title"].strip(),
        f"{cell_id}: paper source title is missing",
    )
    _require(
        isinstance(source.get("url"), str) and source["url"].startswith("https://"),
        f"{cell_id}: paper source URL must be an https URL",
    )


def validate_and_project_published_comparators(
    value: Mapping[str, Any],
    *,
    release_id: str,
    expected_cell_ids: Sequence[str],
    expected_method_ids: Sequence[str],
    metrics_rows: Iterable[Mapping[str, Any]],
    source_registry_file_sha256: str,
    expected_method_display_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the source registry and return its report-only projection."""

    _require(isinstance(value, Mapping), "published-comparator registry must be an object")
    _require(
        value.get("schema_version") == PUBLISHED_COMPARATOR_REGISTRY_SCHEMA,
        "published-comparator registry schema drift",
    )
    _require(value.get("release_id") == release_id, "published-comparator release_id mismatch")
    _require(
        isinstance(source_registry_file_sha256, str)
        and _SHA256.fullmatch(source_registry_file_sha256) is not None,
        "published-comparator registry file hash is invalid",
    )
    expected_cells = _expected_cell_ids(expected_cell_ids)
    cells = value.get("cells")
    _require(isinstance(cells, list), "published-comparator cells must be a list")
    observed_ids = [str(cell.get("cell_id", "")) for cell in cells if isinstance(cell, Mapping)]
    _require(
        len(observed_ids) == len(cells) == 24 and len(set(observed_ids)) == 24,
        "published-comparator registry must contain 24 unique cell rows",
    )
    _require(
        tuple(sorted(observed_ids, key=lambda item: item.encode("utf-8"))) == expected_cells,
        "published-comparator cell coverage does not match the report registry",
    )
    source_catalog = value.get("source_catalog")
    _require(isinstance(source_catalog, Mapping), "published source catalog must be an object")
    for cell in cells:
        _require(isinstance(cell, Mapping), "published-comparator cell row must be an object")
        _validate_source_cell(cell, source_catalog=source_catalog)
    counts = Counter(str(cell["comparison_status"]) for cell in cells)
    _require(dict(counts) == EXPECTED_STATUS_COUNTS, "published-comparator 17/4/3 status roster drift")
    metrics_source = value.get("v2_metrics_source")
    _require(isinstance(metrics_source, Mapping), "published-comparator metrics source is missing")
    metrics_sha256 = metrics_source.get("cell_auroc_logical_sha256")
    _require(
        isinstance(metrics_sha256, str) and _SHA256.fullmatch(metrics_sha256) is not None,
        "published-comparator atomic cell-AUROC logical SHA-256 is invalid",
    )
    metrics_rows = list(metrics_rows)
    metric_rows = _metric_leaders(
        metrics_rows,
        expected_cell_ids=expected_cells,
        expected_method_ids=expected_method_ids,
    )
    _validate_recorded_leaders(
        cells,
        metric_rows,
        expected_method_display_names=expected_method_display_names,
    )
    observed_metrics_sha256 = frozen24_cell_auroc_logical_sha256(
        (
            row
            for cell_rows in metric_rows.values()
            for row in cell_rows.values()
        ),
        expected_cell_ids=expected_cells,
        expected_method_ids=expected_method_ids,
    )
    _require(
        observed_metrics_sha256 == metrics_sha256,
        "published-comparator/report atomic cell-AUROC logical hash mismatch",
    )

    projected_rows: list[dict[str, Any]] = []
    for cell in sorted(cells, key=lambda row: str(row["cell_id"]).encode("utf-8")):
        status = str(cell["comparison_status"])
        comparator = cell["published_comparator"]
        source = (
            source_catalog[str(comparator["source_id"])]
            if comparator.get("source_id") is not None
            else None
        )
        fidelity = FIDELITY_BY_STATUS[status]
        projected_rows.append(
            {
                "cell_id": str(cell["cell_id"]),
                "comparison_status": status,
                "anchor_role": str(comparator["anchor_role"]),
                "allowed_display": str(cell["allowed_display"]),
                "fidelity": fidelity,
                "published_method": comparator.get("method"),
                "published_auroc": comparator.get("auroc"),
                "mapping_status": str(comparator.get("mapping_status", "")),
                "supervision": comparator.get("supervision"),
                "access": comparator.get("access"),
                "passes": comparator.get("passes"),
                "source_id": comparator.get("source_id"),
                "source_title": source.get("title") if source is not None else None,
                "source_url": source.get("url") if source is not None else None,
                "source_table": comparator.get("table"),
                "match_axes": {
                    axis: str(cell["match_axes"][axis]) for axis in REQUIRED_MATCH_AXES
                },
                "mismatch_reasons": [str(reason) for reason in cell["mismatch_reasons"]],
                "common_replay_status": str(cell["common_replay_status"]),
                "delta_eligible": False,
                "paper_to_v2_delta": None,
            }
        )
    output = {
        "schema_version": PUBLISHED_CONTEXT_SCHEMA,
        "release_id": release_id,
        "source_registry_schema_version": PUBLISHED_COMPARATOR_REGISTRY_SCHEMA,
        "source_registry_file_sha256": source_registry_file_sha256,
        "source_registry_content_sha256": canonical_sha256(value),
        "metrics_cell_auroc_logical_sha256": str(metrics_sha256),
        "status_counts": {status: EXPECTED_STATUS_COUNTS[status] for status in PUBLISHED_CONTEXT_STATUSES},
        "ranking_policy": _RANKING_POLICY,
        "delta_policy": _DELTA_POLICY,
        "rows": projected_rows,
    }
    return validate_published_context_projection(
        output,
        release_id=release_id,
        expected_cell_ids=expected_cells,
    )


def validate_published_context_projection(
    value: Mapping[str, Any],
    *,
    release_id: str,
    expected_cell_ids: Sequence[str],
    metrics_cell_auroc_logical_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the signed report-only projection and optional metrics hash."""

    _require(isinstance(value, Mapping), "published context projection must be an object")
    _require(
        set(value) == _PROJECTION_FIELDS,
        "published context projection top-level fields drift",
    )
    _require(value.get("schema_version") == PUBLISHED_CONTEXT_SCHEMA, "published context schema drift")
    _require(value.get("release_id") == release_id, "published context release_id mismatch")
    _require(
        value.get("source_registry_schema_version")
        == PUBLISHED_COMPARATOR_REGISTRY_SCHEMA,
        "published context source-registry schema drift",
    )
    for field in ("source_registry_file_sha256", "source_registry_content_sha256"):
        _require(
            isinstance(value.get(field), str)
            and _SHA256.fullmatch(str(value[field])) is not None,
            f"published context {field} is invalid",
        )
    _require(
        value.get("ranking_policy") == _RANKING_POLICY,
        "published context ranking policy drift",
    )
    _require(
        value.get("delta_policy") == _DELTA_POLICY,
        "published context delta policy drift",
    )
    expected_cells = _expected_cell_ids(expected_cell_ids)
    rows = value.get("rows")
    _require(isinstance(rows, list), "published context rows must be a list")
    observed = [str(row.get("cell_id", "")) for row in rows if isinstance(row, Mapping)]
    _require(
        len(observed) == len(rows) == 24 and len(set(observed)) == 24,
        "published context projection must contain 24 unique rows",
    )
    _require(
        tuple(sorted(observed, key=lambda item: item.encode("utf-8"))) == expected_cells,
        "published context projection cell coverage mismatch",
    )
    _require(
        observed == list(expected_cells),
        "published context rows are not in canonical cell order",
    )
    counts = Counter(str(row.get("comparison_status")) for row in rows)
    _require(dict(counts) == EXPECTED_STATUS_COUNTS, "published context projection 17/4/3 status drift")
    _require(value.get("status_counts") == EXPECTED_STATUS_COUNTS, "published context declared status counts drift")
    expected_metrics_hash = value.get("metrics_cell_auroc_logical_sha256")
    _require(
        isinstance(expected_metrics_hash, str)
        and _SHA256.fullmatch(expected_metrics_hash) is not None,
        "published context atomic cell-AUROC logical SHA-256 is invalid",
    )
    if metrics_cell_auroc_logical_sha256 is not None:
        _require(
            isinstance(metrics_cell_auroc_logical_sha256, str)
            and _SHA256.fullmatch(metrics_cell_auroc_logical_sha256) is not None,
            "report atomic cell-AUROC logical SHA-256 is invalid",
        )
        _require(
            metrics_cell_auroc_logical_sha256 == expected_metrics_hash,
            "published context/report atomic cell-AUROC logical hash mismatch",
        )
    for row in rows:
        _require(isinstance(row, Mapping), "published context row must be an object")
        _require(
            set(row) == _PROJECTED_ROW_FIELDS,
            "published context projected row fields drift",
        )
        cell_id = str(row.get("cell_id", ""))
        status = row.get("comparison_status")
        _require(status in PUBLISHED_CONTEXT_STATUSES, f"{cell_id}: unknown projected context status")
        _require(
            row.get("allowed_display") == DISPLAY_MODE_BY_STATUS[status],
            f"{cell_id}: projected context status/display mismatch",
        )
        _require(
            row.get("anchor_role") == ANCHOR_ROLE_BY_STATUS[status],
            f"{cell_id}: projected context status/anchor mismatch",
        )
        _require(
            row.get("fidelity") == FIDELITY_BY_STATUS[status],
            f"{cell_id}: projected fidelity drift",
        )
        _require(
            row.get("mapping_status") in MAPPING_STATUSES_BY_STATUS[status],
            f"{cell_id}: projected mapping status drift",
        )
        _require(row.get("delta_eligible") is False, f"{cell_id}: projected delta_eligible must be false")
        _require(row.get("paper_to_v2_delta") is None, f"{cell_id}: projected paper delta must be null")
        match_axes = row.get("match_axes")
        _require(
            isinstance(match_axes, Mapping)
            and set(match_axes) == set(REQUIRED_MATCH_AXES),
            f"{cell_id}: projected match axes drift",
        )
        _require(
            set(match_axes.values()) <= MATCH_AXIS_STATUSES,
            f"{cell_id}: projected match-axis status drift",
        )
        reasons = row.get("mismatch_reasons")
        _require(
            isinstance(reasons, list)
            and reasons
            and all(isinstance(reason, str) and reason.strip() for reason in reasons),
            f"{cell_id}: projected mismatch reasons are invalid",
        )
        if status == "NO_PUBLISHED_COMPARATOR":
            _require(
                all(row.get(field) is None for field in ("published_method", "published_auroc", "source_id", "source_title", "source_url", "source_table", "supervision", "access", "passes")),
                f"{cell_id}: projected no-comparator row contains paper metadata",
            )
            _require(
                set(match_axes.values()) == {"NOT_APPLICABLE"},
                f"{cell_id}: projected no-comparator axes drift",
            )
            _require(
                row.get("common_replay_status")
                == "NOT_APPLICABLE_NO_COMPARATOR",
                f"{cell_id}: projected no-comparator replay status drift",
            )
        else:
            for field in (
                "published_method",
                "source_id",
                "source_title",
                "source_table",
                "supervision",
                "access",
                "passes",
            ):
                _require(
                    isinstance(row.get(field), str) and row[field].strip(),
                    f"{cell_id}: projected {field} is invalid",
                )
            _require(
                isinstance(row.get("source_url"), str) and row["source_url"].startswith("https://"),
                f"{cell_id}: projected source URL is invalid",
            )
            auroc = _finite_number(
                row.get("published_auroc"),
                context=f"{cell_id} projected paper AUROC",
            )
            _require(
                0.0 <= auroc <= 1.0,
                f"{cell_id}: projected paper AUROC is outside [0,1]",
            )
            _require(
                match_axes.get("row_ids") != "EXACT",
                f"{cell_id}: projected context-only row IDs may not be exact",
            )
            _require(
                row.get("common_replay_status")
                == "BLOCKED_PENDING_COMMON_ROW_RERUN",
                f"{cell_id}: projected context replay status drift",
            )
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


__all__ = [
    "EXPECTED_STATUS_COUNTS",
    "PUBLISHED_COMPARATOR_REGISTRY_SCHEMA",
    "PUBLISHED_CONTEXT_SCHEMA",
    "PUBLISHED_CONTEXT_STATUSES",
    "REQUIRED_MATCH_AXES",
    "frozen24_cell_auroc_logical_sha256",
    "load_published_comparator_registry",
    "load_published_context_projection",
    "validate_and_project_published_comparators",
    "validate_published_context_projection",
]
