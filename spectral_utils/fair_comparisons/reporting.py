"""Deterministic report rendering for the fair-comparison package.

The advisor HTML has three ordered views.  Only direct, identical-population
comparisons are visible when the file opens; native-paper/context material and
partial/blocked coverage require an explicit tab click.  The Markdown report
contains the same material in that same order for archival readability.

Rendering is deliberately data-only: no metrics are computed here and no
external JavaScript, fonts, or network resources are loaded.  Callers pass
table specifications containing already audited rows.
"""

from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from .registry import ACCESS_FIELDS, FIDELITY_LABELS, LANES, RegistryError


DIRECT_TIER = "direct"
CONTEXT_TIER = "native-context"
PARTIAL_TIER = "partial-blocked"
PRESENTATION_TIERS = (DIRECT_TIER, CONTEXT_TIER, PARTIAL_TIER)

_DIRECT_FORBIDDEN_FIDELITY = frozenset(("published-context-only", "blocked-assets"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INTERVAL_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\]\s*$"
)


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def _markdown_cell(value: Any) -> str:
    return _format_value(value).replace("|", "\\|").replace("\n", "<br>")


def _normalize_columns(columns: Any, rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    if columns is None:
        keys = sorted({key for row in rows for key in row})
        return [(key, key.replace("_", " ").title()) for key in keys]
    if isinstance(columns, Mapping):
        result = [(str(key), str(label)) for key, label in columns.items()]
    elif isinstance(columns, Sequence) and not isinstance(columns, (str, bytes)):
        result = []
        for item in columns:
            if isinstance(item, str):
                result.append((item, item.replace("_", " ").title()))
            elif isinstance(item, Mapping):
                if set(item) != {"key", "label"}:
                    raise RegistryError("column mappings must contain exactly key and label")
                result.append((str(item["key"]), str(item["label"])))
            elif isinstance(item, Sequence) and len(item) == 2:
                result.append((str(item[0]), str(item[1])))
            else:
                raise RegistryError(f"invalid column specification: {item!r}")
    else:
        raise RegistryError("columns must be a mapping or sequence")
    if not result:
        raise RegistryError("a rendered table must declare at least one column")
    keys = [key for key, _ in result]
    if len(keys) != len(set(keys)):
        raise RegistryError("rendered table contains duplicate column keys")
    return result


def _row_access(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    if isinstance(row.get("access"), Mapping):
        access = dict(row["access"])
    else:
        access = {field: row.get(field) for field in ACCESS_FIELDS if field in row}
    if not all(field in access for field in ACCESS_FIELDS):
        return None
    if not isinstance(access["input_type"], str) or not access["input_type"].strip():
        return None
    if not isinstance(access["supervision"], str) or not access["supervision"].strip():
        return None
    return access


def _is_paired_interval(value: Any) -> bool:
    if isinstance(value, Mapping):
        triple = (value.get("point"), value.get("ci_low"), value.get("ci_high"))
    elif isinstance(value, str):
        match = _INTERVAL_RE.fullmatch(value)
        if match is None:
            return False
        triple = tuple(float(item) for item in match.groups())
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        triple = tuple(value)
    else:
        return False
    try:
        point, low, high = (float(item) for item in triple)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) for item in (point, low, high)) and low <= high


def _validate_direct_claim_contract(
    table: Mapping[str, Any],
    *,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    required_methods: Sequence[str],
) -> dict[str, Any]:
    """Require every rendered direct claim to bind population, evaluator and CI."""

    contract = table.get("direct_claim_contract")
    if not isinstance(contract, Mapping):
        raise RegistryError(
            f"direct table {title!r} must declare direct_claim_contract"
        )
    group_by = contract.get("group_by", [])
    if isinstance(group_by, (str, bytes)) or not isinstance(group_by, Sequence):
        raise RegistryError(f"direct table {title!r} contract group_by must be a sequence")
    group_by = [str(field) for field in group_by]
    if any(not field for field in group_by) or len(group_by) != len(set(group_by)):
        raise RegistryError(f"direct table {title!r} has invalid contract group_by")
    hash_fields = contract.get("eligible_population_hash_fields")
    if (
        isinstance(hash_fields, (str, bytes))
        or not isinstance(hash_fields, Sequence)
        or not hash_fields
    ):
        raise RegistryError(
            f"direct table {title!r} must declare eligible_population_hash_fields"
        )
    hash_fields = [str(field) for field in hash_fields]
    if any(not field for field in hash_fields) or len(hash_fields) != len(set(hash_fields)):
        raise RegistryError(f"direct table {title!r} has invalid eligible hash fields")
    evaluator_field = contract.get("evaluator_hash_field")
    if not isinstance(evaluator_field, str) or not evaluator_field:
        raise RegistryError(f"direct table {title!r} must declare evaluator_hash_field")
    paired = contract.get("paired_intervals")
    if isinstance(paired, (str, bytes)) or not isinstance(paired, Sequence) or not paired:
        raise RegistryError(
            f"direct table {title!r} must declare at least one required paired interval"
        )
    paired_specs: list[dict[str, str]] = []
    for index, item in enumerate(paired):
        if not isinstance(item, Mapping):
            raise RegistryError(
                f"direct table {title!r} paired interval {index} is not a mapping"
            )
        required = ("left_method_id", "right_method_id", "field")
        if any(not isinstance(item.get(field), str) or not item[field] for field in required):
            raise RegistryError(
                f"direct table {title!r} paired interval {index} lacks {required}"
            )
        paired_specs.append({field: str(item[field]) for field in required})

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row_index, row in enumerate(rows):
        missing_group = [field for field in group_by if field not in row]
        if missing_group:
            raise RegistryError(
                f"direct table {title!r} row {row_index} lacks group fields {missing_group}"
            )
        groups.setdefault(tuple(row[field] for field in group_by), []).append(row)
    if not groups:
        raise RegistryError(f"direct table {title!r} cannot make a claim with zero rows")

    unified_ids = {"unified28", "unified-28"}
    for group_key, group_rows in groups.items():
        by_method: dict[str, list[Mapping[str, Any]]] = {}
        for row in group_rows:
            by_method.setdefault(str(row.get("method_id")), []).append(row)
        missing = [method for method in required_methods if method not in by_method]
        if missing:
            raise RegistryError(
                f"direct table {title!r} comparison group {group_key!r} is missing "
                f"required methods {missing}"
            )
        duplicated = [method for method, values in by_method.items() if len(values) != 1]
        if duplicated:
            raise RegistryError(
                f"direct table {title!r} comparison group {group_key!r} repeats methods "
                f"{duplicated}"
            )
        for field in hash_fields:
            values = [row.get(field) for row in group_rows]
            if any(not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in values):
                raise RegistryError(
                    f"direct table {title!r} comparison group {group_key!r} has an "
                    f"invalid/missing eligible population hash in {field!r}"
                )
            if len(set(values)) != 1:
                raise RegistryError(
                    f"direct table {title!r} comparison group {group_key!r} does not "
                    f"share eligible population hash {field!r}"
                )
        evaluators = [row.get(evaluator_field) for row in group_rows]
        if any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in evaluators
        ):
            raise RegistryError(
                f"direct table {title!r} comparison group {group_key!r} has an "
                "invalid/missing evaluator hash"
            )
        if len(set(evaluators)) != 1:
            raise RegistryError(
                f"direct table {title!r} comparison group {group_key!r} does not share "
                "one evaluator hash"
            )
        for spec in paired_specs:
            left = spec["left_method_id"]
            right = spec["right_method_id"]
            if left not in by_method or right not in by_method:
                raise RegistryError(
                    f"direct table {title!r} comparison group {group_key!r} lacks paired "
                    f"contrast methods {left!r}/{right!r}"
                )
            interval = by_method[left][0].get(spec["field"])
            if not _is_paired_interval(interval):
                raise RegistryError(
                    f"direct table {title!r} comparison group {group_key!r} lacks a valid "
                    f"paired interval {spec['field']!r} for {left!r} vs {right!r}"
                )
        present_unified = unified_ids.intersection(by_method)
        if present_unified and not any(
            spec["left_method_id"] in present_unified
            and spec["right_method_id"] in required_methods
            for spec in paired_specs
        ):
            raise RegistryError(
                f"direct table {title!r} comparison group {group_key!r} lacks the "
                "required Unified-vs-incumbent paired interval"
            )
    return {
        "group_by": group_by,
        "eligible_population_hash_fields": hash_fields,
        "evaluator_hash_field": evaluator_field,
        "paired_intervals": paired_specs,
    }


def _validate_table(table: Mapping[str, Any], *, tier: str) -> dict[str, Any]:
    if not isinstance(table, Mapping):
        raise RegistryError("report table must be a mapping")
    title = table.get("title")
    if not isinstance(title, str) or not title.strip():
        raise RegistryError("report table title must be a non-empty string")
    lane = table.get("lane")
    if lane is not None and lane not in LANES:
        raise RegistryError(f"table lane {lane!r} not in {LANES}")
    rows_value = table.get("rows", [])
    if isinstance(rows_value, (str, bytes)) or not isinstance(rows_value, Sequence):
        raise RegistryError(f"table {title!r} rows must be a sequence")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows_value):
        if not isinstance(row, Mapping):
            raise RegistryError(f"table {title!r} row {index} is not a mapping")
        normalized = dict(row)
        if lane is not None and "lane" in normalized and normalized["lane"] != lane:
            raise RegistryError(
                f"table {title!r} mixes lane {lane!r} with row lane {normalized['lane']!r}"
            )
        if table.get("require_fidelity", True):
            fidelity = normalized.get("fidelity")
            if fidelity not in FIDELITY_LABELS:
                raise RegistryError(
                    f"table {title!r} row {index} has invalid/missing fidelity {fidelity!r}"
                )
            access = _row_access(normalized)
            if access is None:
                raise RegistryError(
                    f"table {title!r} row {index} lacks the four orthogonal access fields"
                )
            if tier == DIRECT_TIER and fidelity in _DIRECT_FORBIDDEN_FIDELITY:
                raise RegistryError(
                    f"{fidelity} row cannot enter a direct-comparison table: {title!r}"
                )
            if tier == DIRECT_TIER and normalized.get("headline_eligible") is False:
                raise RegistryError(
                    f"ineligible row cannot enter a direct-comparison table: {title!r}"
                )
            if tier == DIRECT_TIER:
                for field in ("model_passes_per_question", "traces_per_question"):
                    value = access[field]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) < 0
                    ):
                        raise RegistryError(
                            f"direct table {title!r} row {index} has invalid access.{field}"
                        )
        rows.append(normalized)

    required_methods = list(table.get("required_method_ids", []))
    if required_methods:
        observed_methods = {row.get("method_id") for row in rows}
        missing = [method for method in required_methods if method not in observed_methods]
        if missing:
            raise RegistryError(
                f"table {title!r} is missing required methods {missing}; "
                "do not render an eligible direct table without Unified-28 and its incumbent"
            )

    direct_claim_contract = None
    if tier == DIRECT_TIER:
        direct_claim_contract = _validate_direct_claim_contract(
            table,
            title=title,
            rows=rows,
            required_methods=required_methods,
        )

    return {
        "table_id": str(table.get("table_id", title)),
        "title": title,
        "lane": lane,
        "description": str(table.get("description", "")),
        "note": str(table.get("note", "")),
        "columns": _normalize_columns(table.get("columns"), rows),
        "rows": rows,
        "require_fidelity": bool(table.get("require_fidelity", True)),
        "required_method_ids": required_methods,
        "direct_claim_contract": direct_claim_contract,
    }


def validate_report_tables(
    direct_tables: Sequence[Mapping[str, Any]],
    native_context_tables: Sequence[Mapping[str, Any]],
    partial_blocked_tables: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Validate tier separation and normalize table/column specifications."""

    return {
        DIRECT_TIER: [_validate_table(table, tier=DIRECT_TIER) for table in direct_tables],
        CONTEXT_TIER: [
            _validate_table(table, tier=CONTEXT_TIER) for table in native_context_tables
        ],
        PARTIAL_TIER: [
            _validate_table(table, tier=PARTIAL_TIER) for table in partial_blocked_tables
        ],
    }


def presentation_tier(row: Mapping[str, Any]) -> str:
    """Classify a row when a caller is assembling report tables.

    An explicit ``presentation_tier`` wins.  Otherwise blocked assets and any
    incomplete/partial acquisition state go to the coverage view; published
    numbers and ``native_only`` rows go to context; complete common-protocol
    rows remain direct.  ``paper-specified-partial`` does *not* automatically
    mean acquisition-partial: a fully completed LEASH cell may legitimately be
    a direct row with that fidelity label.
    """

    explicit = row.get("presentation_tier")
    if explicit is not None:
        if explicit not in PRESENTATION_TIERS:
            raise RegistryError(f"unknown presentation_tier {explicit!r}")
        return str(explicit)
    fidelity = row.get("fidelity")
    status = str(row.get("status", "")).lower()
    if fidelity == "blocked-assets" or any(
        marker in status for marker in ("blocked", "incomplete", "partial", "failed")
    ):
        return PARTIAL_TIER
    if fidelity == "published-context-only" or bool(row.get("native_only")):
        return CONTEXT_TIER
    return DIRECT_TIER


def partition_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {tier: [] for tier in PRESENTATION_TIERS}
    for row in rows:
        result[presentation_tier(row)].append(dict(row))
    return result


def render_markdown_table(table: Mapping[str, Any], *, tier: str = DIRECT_TIER) -> str:
    normalized = _validate_table(table, tier=tier)
    lines = [f"### {normalized['title']}", ""]
    if normalized["lane"]:
        lines.extend((f"Lane: `{normalized['lane']}`", ""))
    if normalized["description"]:
        lines.extend((normalized["description"], ""))
    columns = normalized["columns"]
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    if normalized["rows"]:
        for row in normalized["rows"]:
            lines.append("| " + " | ".join(_markdown_cell(row.get(key)) for key, _ in columns) + " |")
    else:
        lines.append("| " + " | ".join("—" for _ in columns) + " |")
    if normalized["note"]:
        lines.extend(("", f"Note: {normalized['note']}"))
    return "\n".join(lines)


def _markdown_section(
    title: str,
    intro: str,
    tables: Sequence[Mapping[str, Any]],
    *,
    tier: str,
) -> list[str]:
    lines = [f"## {title}", "", intro, ""]
    if not tables:
        lines.extend(("No rows are registered in this section.", ""))
        return lines
    for table in tables:
        lines.extend((render_markdown_table(table, tier=tier), ""))
    return lines


def render_markdown_report(
    *,
    title: str,
    direct_tables: Sequence[Mapping[str, Any]],
    native_context_tables: Sequence[Mapping[str, Any]] = (),
    partial_blocked_tables: Sequence[Mapping[str, Any]] = (),
    summary: str = "",
    provenance: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render the archival Markdown report in the frozen presentation order."""

    normalized = validate_report_tables(
        direct_tables, native_context_tables, partial_blocked_tables
    )
    lines = [f"# {title}", ""]
    if summary:
        lines.extend((summary.strip(), ""))
    lines.extend(
        _markdown_section(
            "Direct comparisons",
            "Identical-row comparisons using the registered population, evaluator, and paired uncertainty contract.",
            normalized[DIRECT_TIER],
            tier=DIRECT_TIER,
        )
    )
    lines.extend(
        _markdown_section(
            "Native-paper and context",
            "Metric-incompatible native-paper results and published references are context, not substitutes for direct replays.",
            normalized[CONTEXT_TIER],
            tier=CONTEXT_TIER,
        )
    )
    lines.extend(
        _markdown_section(
            "Partial and blocked coverage",
            "Incomplete acquisitions, failed cells, and unavailable official assets never enter headline aggregates.",
            normalized[PARTIAL_TIER],
            tier=PARTIAL_TIER,
        )
    )
    if provenance:
        lines.extend(("## Reproducibility", ""))
        for key in sorted(provenance):
            lines.append(f"- `{key}`: `{_markdown_cell(provenance[key])}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _html_table(table: Mapping[str, Any], *, tier: str) -> str:
    normalized = _validate_table(table, tier=tier)
    lane_badge = (
        f'<span class="lane">{html.escape(normalized["lane"])}</span>'
        if normalized["lane"]
        else ""
    )
    description = (
        f'<p class="description">{html.escape(normalized["description"])}</p>'
        if normalized["description"]
        else ""
    )
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in normalized["columns"])
    body_rows = []
    for row in normalized["rows"]:
        cells = "".join(
            f"<td>{html.escape(_format_value(row.get(key)))}</td>"
            for key, _ in normalized["columns"]
        )
        body_rows.append(f"<tr>{cells}</tr>")
    if not body_rows:
        body_rows.append(
            f'<tr><td colspan="{len(normalized["columns"])}" class="empty">No rows</td></tr>'
        )
    note = (
        f'<p class="note"><strong>Note:</strong> {html.escape(normalized["note"])}</p>'
        if normalized["note"]
        else ""
    )
    return (
        '<article class="table-card">'
        f'<header><h3>{html.escape(normalized["title"])}</h3>{lane_badge}</header>'
        f"{description}"
        '<div class="table-scroll"><table><thead><tr>'
        f"{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>{note}</article>"
    )


def _html_panel(
    panel_id: str,
    heading: str,
    intro: str,
    tables: Sequence[Mapping[str, Any]],
    *,
    tier: str,
    hidden: bool,
) -> str:
    hidden_attr = " hidden" if hidden else ""
    table_html = "".join(_html_table(table, tier=tier) for table in tables)
    if not table_html:
        table_html = '<p class="empty-panel">No rows are registered in this section.</p>'
    return (
        f'<section id="panel-{panel_id}" class="panel" role="tabpanel" '
        f'aria-labelledby="tab-{panel_id}"{hidden_attr}>'
        f"<h2>{html.escape(heading)}</h2><p class=\"panel-intro\">{html.escape(intro)}</p>"
        f"{table_html}</section>"
    )


def render_advisor_html(
    *,
    title: str,
    direct_tables: Sequence[Mapping[str, Any]],
    native_context_tables: Sequence[Mapping[str, Any]] = (),
    partial_blocked_tables: Sequence[Mapping[str, Any]] = (),
    summary: str = "",
    provenance: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render a standalone advisor report with direct comparisons as default."""

    normalized = validate_report_tables(
        direct_tables, native_context_tables, partial_blocked_tables
    )
    provenance_html = ""
    if provenance:
        items = "".join(
            f"<dt>{html.escape(str(key))}</dt><dd><code>{html.escape(_format_value(provenance[key]))}</code></dd>"
            for key in sorted(provenance)
        )
        provenance_html = f'<footer><h2>Reproducibility</h2><dl>{items}</dl></footer>'
    summary_html = f'<p class="summary">{html.escape(summary.strip())}</p>' if summary else ""
    direct_panel = _html_panel(
        "direct",
        "Direct comparisons",
        "Identical-row comparisons using the registered population, evaluator, and paired uncertainty contract.",
        normalized[DIRECT_TIER],
        tier=DIRECT_TIER,
        hidden=False,
    )
    context_panel = _html_panel(
        "context",
        "Native-paper and context",
        "Metric-incompatible native-paper results and published references are context, not substitutes for direct replays.",
        normalized[CONTEXT_TIER],
        tier=CONTEXT_TIER,
        hidden=True,
    )
    partial_panel = _html_panel(
        "partial",
        "Partial and blocked coverage",
        "Incomplete acquisitions, failed cells, and unavailable official assets never enter headline aggregates.",
        normalized[PARTIAL_TIER],
        tier=PARTIAL_TIER,
        hidden=True,
    )
    # The secondary panels occur after direct in source order and carry the
    # native HTML hidden attribute, so direct is the only visible default even
    # when JavaScript is disabled.
    return f"""<!doctype html>
<html lang="en" data-default-view="direct">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#19212b;--muted:#66717f;--line:#d8dee8;--paper:#fff;--wash:#f4f7fb;--accent:#173b67;--accent2:#e8f0fa;--warn:#7c4313}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:0 auto;padding:32px}} h1{{font-size:30px;margin:0 0 8px}} h2{{font-size:22px;margin:10px 0 4px}} h3{{font-size:17px;margin:0}}
.summary,.panel-intro,.description,.note{{color:var(--muted)}} .tabs{{display:flex;gap:8px;margin:24px 0 16px;flex-wrap:wrap}}
.tabs button{{border:1px solid var(--line);background:var(--paper);padding:9px 14px;border-radius:8px;cursor:pointer;color:var(--ink)}}
.tabs button[aria-selected="true"]{{background:var(--accent);border-color:var(--accent);color:white}}
.panel[hidden]{{display:none}} .table-card{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:18px;margin:16px 0}}
.table-card header{{display:flex;align-items:center;gap:10px}} .lane{{background:var(--accent2);color:var(--accent);border-radius:999px;padding:2px 8px;font-size:12px}}
.table-scroll{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top;white-space:nowrap}} th{{background:#f8fafc;position:sticky;top:0}} td.empty{{text-align:center;color:var(--muted)}}
footer{{margin-top:28px;padding-top:16px;border-top:1px solid var(--line)}} dl{{display:grid;grid-template-columns:max-content 1fr;gap:6px 14px}} dt{{font-weight:600}} dd{{margin:0;overflow-wrap:anywhere}} code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}}
@media(max-width:700px){{main{{padding:18px}} dl{{grid-template-columns:1fr}}}}
@media print{{body{{background:white}} .tabs{{display:none}} .panel[hidden]{{display:none!important}} main{{max-width:none;padding:0}} .table-card{{break-inside:avoid}}}}
</style>
</head>
<body><main>
<header><h1>{html.escape(title)}</h1>{summary_html}</header>
<nav class="tabs" role="tablist" aria-label="Report views">
<button id="tab-direct" role="tab" aria-controls="panel-direct" aria-selected="true" data-panel="direct">Direct comparisons</button>
<button id="tab-context" role="tab" aria-controls="panel-context" aria-selected="false" data-panel="context">Native / context</button>
<button id="tab-partial" role="tab" aria-controls="panel-partial" aria-selected="false" data-panel="partial">Partial / blocked</button>
</nav>
{direct_panel}
{context_panel}
{partial_panel}
{provenance_html}
</main>
<script>
document.querySelectorAll('[data-panel]').forEach(function(button){{
  button.addEventListener('click',function(){{
    document.querySelectorAll('[data-panel]').forEach(function(item){{item.setAttribute('aria-selected',String(item===button));}});
    document.querySelectorAll('.panel').forEach(function(panel){{panel.hidden=panel.id!=='panel-'+button.dataset.panel;}});
  }});
}});
</script>
</body></html>
"""


def write_reports(
    output_dir: os.PathLike[str] | str,
    *,
    title: str,
    direct_tables: Sequence[Mapping[str, Any]],
    native_context_tables: Sequence[Mapping[str, Any]] = (),
    partial_blocked_tables: Sequence[Mapping[str, Any]] = (),
    summary: str = "",
    provenance: Optional[Mapping[str, Any]] = None,
    markdown_name: str = "REPORT.md",
    html_name: str = "REPORT.html",
) -> dict[str, str]:
    """Write the two deterministic report views and return their paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown_report(
        title=title,
        direct_tables=direct_tables,
        native_context_tables=native_context_tables,
        partial_blocked_tables=partial_blocked_tables,
        summary=summary,
        provenance=provenance,
    )
    advisor_html = render_advisor_html(
        title=title,
        direct_tables=direct_tables,
        native_context_tables=native_context_tables,
        partial_blocked_tables=partial_blocked_tables,
        summary=summary,
        provenance=provenance,
    )
    markdown_path = output / markdown_name
    html_path = output / html_name
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    html_path.write_text(advisor_html, encoding="utf-8", newline="\n")
    return {"markdown": str(markdown_path), "html": str(html_path)}


__all__ = [
    "CONTEXT_TIER",
    "DIRECT_TIER",
    "PARTIAL_TIER",
    "PRESENTATION_TIERS",
    "partition_rows",
    "presentation_tier",
    "render_advisor_html",
    "render_markdown_report",
    "render_markdown_table",
    "validate_report_tables",
    "write_reports",
]
