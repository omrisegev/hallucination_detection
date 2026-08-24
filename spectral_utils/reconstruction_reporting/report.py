"""Self-contained, file://-safe reconstruction benchmark explorer."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .io import _atomic_write_bytes, validate_plot_data_sources
from .registry import registry_indexes, validate_registry, validate_result_references
from .schemas import (
    RANKABLE_STATUSES,
    SchemaError,
    canonical_sha256,
    make_plot_spec,
    rank_metric_group,
    record_sort_key,
    validate_comparison_groups,
    validate_plot_manifest,
    validate_records,
)


REPORT_SCHEMA = "reconstruction_static_report_v1"
GRAPH_DISPLAY_EDGE_LIMIT = 2_000
_EMBEDDED_DIAGNOSTIC_FIELDS = (
    "task_id",
    "dataset_id",
    "cell_id",
    "slice_id",
    "comparison_group_id",
    "method_id",
    "system_id",
    "graph_variant",
    "diagnostic_label",
    "value",
    "null_value",
    "effect",
    "p_value",
    "label_stage",
    "status",
)


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_for_script(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _embedded_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project verified diagnostics to the fields used by the browser UI.

    Validation, plot hashing, generated report identity, downloadable CSV and
    the Parquet/DuckDB release all continue to use the complete signed rows.
    This projection only avoids repeating unused provenance strings inside the
    self-contained HTML file.
    """

    return [
        {field: row[field] for field in _EMBEDDED_DIAGNOSTIC_FIELDS}
        for row in rows
    ]


def _marker_svg(marker: str, color: str, *, size: int = 22) -> str:
    """Render the same eight registered marker shapes used by the plots."""

    geometry = {
        "circle": '<circle cx="11" cy="11" r="6"/>',
        "square": '<rect x="5" y="5" width="12" height="12"/>',
        "triangle": '<polygon points="11,4 18,17 4,17"/>',
        "diamond": '<polygon points="11,3 19,11 11,19 3,11"/>',
        "cross": '<path d="M5 5 L17 17 M17 5 L5 17"/>',
        "plus": '<path d="M11 3 L11 19 M3 11 L19 11"/>',
        "star": '<polygon points="11,2 13.6,7.4 19.5,8.1 15.2,12.2 16.4,18 11,15.1 5.6,18 6.8,12.2 2.5,8.1 8.4,7.4"/>',
        "hexagon": '<polygon points="6,3.5 16,3.5 21,11 16,18.5 6,18.5 1,11"/>',
    }
    if marker not in geometry:
        raise SchemaError(f"unsupported method marker {marker!r}")
    return (
        f'<svg class="method-marker" width="{size}" height="{size}" '
        f'viewBox="0 0 22 22" role="img" aria-label="{_e(marker)} marker" '
        f'style="--method-color:{_e(color)}">{geometry[marker]}</svg>'
    )


def _method_legend(registry: Mapping[str, Any]) -> str:
    return "".join(
        f'<span class="method-legend-item">{_marker_svg(method["style"]["marker"], method["style"]["color"], size=18)}'
        f'<a href="#method-{_e(method["method_id"])}">{_e(method["display_name"])}</a></span>'
        for method in registry["methods"]
    )


_STATUS_EXPLANATIONS = {
    "OK": "Completed with the registered method.",
    "OK_FALLBACK": "Completed with the explicitly recorded fallback.",
    "NOT_APPLICABLE": "The registered system does not define this estimand.",
    "NOT_RUN": "The registered run has not been executed.",
    "ADAPTER_MISSING": "No frozen adapter exists for this task.",
    "BLOCKED_ASSET": "A required source artifact is unavailable or unverified.",
    "INPUT_INVALID": "The input failed its frozen contract.",
    "FIT_FAILED": "The method fit failed; no numeric result is substituted.",
    "SCORE_INCOMPLETE": "Some required example scores are missing.",
    "METRIC_UNDEFINED_SINGLE_CLASS": "The requested metric is undefined for a one-class slice.",
    "EXCLUDED_BY_PROTOCOL": "The preregistered protocol excludes this row.",
    "QUARANTINED": "A known data or protocol defect prevents comparison.",
    "UNVERIFIED": "A numeric value is visible but has not passed the required audit.",
    "CONTEXT_ONLY": "A non-comparable published or historical value shown for context.",
}


def _status_table() -> str:
    rows = "".join(
        f'<tr data-status="{_e(status)}"><td class="status {_e(status)}">{_e(status)}</td>'
        f'<td>{_e(description)}</td><td>{"yes" if status in RANKABLE_STATUSES else "no"}</td></tr>'
        for status, description in _STATUS_EXPLANATIONS.items()
    )
    return (
        '<div class="table-wrap status-table"><table><thead><tr><th>Status</th>'
        f'<th>Meaning</th><th>Rankable</th></tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _method_cards(registry: Mapping[str, Any]) -> str:
    cards = []
    for method in registry["methods"]:
        terms = "".join(
            f"<dt><code>{_e(symbol)}</code></dt><dd>{_e(description)}</dd>"
            for symbol, description in method["formula_terms"].items()
        )
        assumptions = "".join(f"<li>{_e(item)}</li>" for item in method["assumptions"])
        fallbacks = "".join(f"<li>{_e(item)}</li>" for item in method["fallbacks"])
        limitations = "".join(f"<li>{_e(item)}</li>" for item in method["limitations"])
        references = "".join(
            f"<li>{_e(reference['citation'])}</li>" for reference in method["references"]
        ) or "<li>No external paper; project-defined control or adaptation.</li>"
        style = method["style"]
        cards.append(
            f"""
<article class="guide-card method-card" id="method-{_e(method['method_id'])}">
  <header>
    {_marker_svg(style['marker'], style['color'])}
    <div><h3>{_e(method['display_name'])}</h3>
    <p class="subtitle">{_e(method['acronym_expansion'])}</p></div>
    <span class="badge">{_e(method['role'])}</span>
    <span class="badge subtle">{_e(method['research_stage'])}</span>
  </header>
  <p class="plain-summary">{_e(method['plain_summary'])}</p>
  <p><strong>Input → operation → output:</strong> {_e(method['input_operation_output'])}</p>
  <div class="formula"><code>{_e(method['formula'])}</code></div>
  <dl class="formula-terms">{terms}</dl>
  <div class="guide-grid">
    <div><h4>Origin and development</h4>
      <p>{_e(method['origin']['title'])} ({_e(method['origin']['year'])}). {_e(method['origin']['relationship'])}</p>
      <p>{_e(method['development_history'])}</p>
    </div>
    <div><h4>Information used</h4>
      <p>{_e(method['inputs'])}</p>
      <p>Access: <code>{_e(method['access_tier'])}</code>; supervision: {_e(method['supervision'])}; donors: <code>{_e(method['donor_regime'])}</code>; passes: {_e(method['model_passes'])}.</p>
    </div>
    <div><h4>Assumptions</h4><ul>{assumptions}</ul></div>
    <div><h4>Fallbacks and limitations</h4><ul>{fallbacks}{limitations}</ul></div>
  </div>
  <details><summary>References</summary><ul>{references}</ul></details>
</article>""".strip()
        )
    return "\n".join(cards)


def _dataset_cards(registry: Mapping[str, Any]) -> str:
    return "\n".join(
        f"""
<article class="guide-card dataset-card" id="dataset-{_e(dataset['dataset_id'])}">
  <header><div><h3>{_e(dataset['display_name'])}</h3>
    <p class="subtitle">{_e(dataset['dataset_family'])} · revision {_e(dataset['revision'])}</p></div></header>
  <p>{_e(dataset['description'])}</p>
  <dl class="compact-dl">
    <dt>Prediction unit</dt><dd>{_e(dataset['prediction_unit'])}</dd>
    <dt>Label</dt><dd>{_e(dataset['label_definition'])}</dd>
    <dt>Positive class</dt><dd>{_e(dataset['positive_class'])}</dd>
    <dt>Why included</dt><dd>{_e(dataset['inclusion_reason'])}</dd>
  </dl>
  <p class="caveat"><strong>Limits:</strong> {_e('; '.join(dataset['limitations']) or 'None registered.')}</p>
</article>""".strip()
        for dataset in registry["datasets"]
    )


def _plot_csv_link(plot_manifest: Mapping[str, Any], *, kind: str, example_id: str | None = None) -> str:
    for plot in plot_manifest["plots"]:
        if plot["kind"] != kind:
            continue
        if example_id is not None and plot.get("filters", {}).get("example_id") != example_id:
            continue
        return f'plot_data/{_e(plot["plot_id"])}.csv'
    return ""


def _scaled_coordinates(nodes: Sequence[Mapping[str, Any]]) -> dict[int, tuple[float, float]]:
    xs = [float(row["embedding_x"]) for row in nodes]
    ys = [float(row["embedding_y"]) for row in nodes]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    return {
        int(row["node_index"]): (
            24.0 + 312.0 * (float(row["embedding_x"]) - x_min) / x_span,
            206.0 - 182.0 * (float(row["embedding_y"]) - y_min) / y_span,
        )
        for row in nodes
    }


def _nuisance_color(value: float, low: float, high: float) -> str:
    q = 0.5 if high == low else max(0.0, min(1.0, (value - low) / (high - low)))
    red = round(49 + 194 * q)
    green = round(103 + 39 * q)
    blue = round(213 - 168 * q)
    return f"rgb({red},{green},{blue})"


def _example_svg(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    color_mode: str,
) -> str:
    coordinates = _scaled_coordinates(nodes)
    edge_markup = []
    for row in edges:
        source = coordinates[int(row["edge_source_index"])]
        target = coordinates[int(row["edge_target_index"])]
        edge_markup.append(
            f'<line x1="{source[0]:.6f}" y1="{source[1]:.6f}" x2="{target[0]:.6f}" y2="{target[1]:.6f}" '
            f'stroke="currentColor" stroke-opacity="0.20" stroke-width="1"><title>edge weight {_e(format(float(row["edge_weight"]), ".5g"))}</title></line>'
        )
    nuisance_values = [float(row["nuisance_value"]) for row in nodes if row["nuisance_value"] is not None]
    low = min(nuisance_values) if nuisance_values else 0.0
    high = max(nuisance_values) if nuisance_values else 1.0
    node_markup = []
    for row in nodes:
        x, y = coordinates[int(row["node_index"])]
        if color_mode == "error":
            fill = "#c43b3b" if row["y_error"] else "#2f6fed"
            meaning = "incorrect" if row["y_error"] else "correct"
        else:
            fill = _nuisance_color(float(row["nuisance_value"]), low, high)
            meaning = f"{row['nuisance_name']}={format(float(row['nuisance_value']), '.5g')}"
        node_markup.append(
            f'<circle cx="{x:.6f}" cy="{y:.6f}" r="4.2" fill="{fill}" stroke="white" stroke-width="0.7">'
            f'<title>{_e(row["source_row_id"])} · {_e(meaning)}</title></circle>'
        )
    return (
        '<svg class="example-graph-svg" viewBox="0 0 360 230" role="img" '
        f'aria-label="Graph spectral embedding colored by {_e(color_mode)}">'
        '<rect x="0" y="0" width="360" height="230" rx="8" fill="transparent" stroke="currentColor" stroke-opacity=".14"/>'
        + "".join(edge_markup) + "".join(node_markup) + "</svg>"
    )


def _display_edges(
    edges: Sequence[Mapping[str, Any]],
    *,
    graph_hash: str,
) -> list[Mapping[str, Any]]:
    """Choose a fixed label-free edge sample for an intelligible SVG.

    The signed graph table and downloadable plot CSV retain every edge.  A
    dense graph can contain hundreds of thousands of edges, which is both
    visually opaque and needlessly makes the self-contained report hundreds
    of megabytes.  Hash sampling is independent of labels, node colors, and
    performance, and the same sampled edge set is used in both color panels.
    """

    if len(edges) <= GRAPH_DISPLAY_EDGE_LIMIT:
        return list(edges)
    ranked = sorted(
        edges,
        key=lambda row: (
            canonical_sha256(
                {
                    "rule": "label-free-hash-edge-sample-v1",
                    "graph_hash": graph_hash,
                    "source": int(row["edge_source_index"]),
                    "target": int(row["edge_target_index"]),
                    "weight": float(row["edge_weight"]),
                }
            ),
            int(row["edge_source_index"]),
            int(row["edge_target_index"]),
        ),
    )
    return ranked[:GRAPH_DISPLAY_EDGE_LIMIT]


def _graph_example_cards(
    examples: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    plot_manifest: Mapping[str, Any],
) -> str:
    if not examples:
        return '<p class="empty" id="graph-examples-unavailable">No verified label-free-selected example graph was supplied.</p>'
    methods = {row["method_id"]: row for row in registry["methods"]}
    cards = []
    for example_id in sorted({str(row["example_id"]) for row in examples}, key=lambda value: value.encode("utf-8")):
        group = [row for row in examples if row["example_id"] == example_id]
        nodes = sorted((row for row in group if row["row_kind"] == "node"), key=lambda row: int(row["node_index"]))
        edges = sorted((row for row in group if row["row_kind"] == "edge"), key=lambda row: (int(row["edge_source_index"]), int(row["edge_target_index"])))
        if not nodes:
            raise SchemaError(f"graph example {example_id!r} has no nodes")
        expected_indices = list(range(len(nodes)))
        if [int(row["node_index"]) for row in nodes] != expected_indices:
            raise SchemaError(f"graph example {example_id!r} node indices are not contiguous")
        if any(int(row["edge_source_index"]) not in expected_indices or int(row["edge_target_index"]) not in expected_indices for row in edges):
            raise SchemaError(f"graph example {example_id!r} edge references an unknown node")
        invariants = ("method_id", "cell_id", "cohort_id", "selection_rule_id", "graph_hash", "matrix_hash", "operator_hash", "nuisance_name", "nuisance_available")
        if any(len({str(row[field]) for row in group}) != 1 for field in invariants):
            raise SchemaError(f"graph example {example_id!r} mixes identities")
        method = methods[str(nodes[0]["method_id"])]
        nuisance_available = bool(nodes[0]["nuisance_available"])
        displayed_edges = _display_edges(edges, graph_hash=str(nodes[0]["graph_hash"]))
        left = _example_svg(nodes, displayed_edges, color_mode="error")
        if nuisance_available:
            right = _example_svg(nodes, displayed_edges, color_mode="nuisance")
            nuisance_legend = '<span><i class="legend-gradient"></i>Blue → orange = low → high frozen trace-length coordinate</span>'
        else:
            right = '<div class="unavailable-panel" role="img" aria-label="Trace-length nuisance unavailable">Trace-length nuisance coordinate unavailable for this selected cell. No substitute feature was used.</div>'
            nuisance_legend = '<span>Right panel unavailable: the signed source did not contain the nuisance coordinate.</span>'
        csv_link = _plot_csv_link(plot_manifest, kind="graph_embedding_pair", example_id=example_id)
        cards.append(f'''
<article class="figure-card graph-example" data-example-id="{_e(example_id)}">
 <h3 class="plot-title">{_e(method['display_name'])} · {_e(nodes[0]['cell_id'])}</h3>
 <p class="plot-subtitle">One embedding and one edge set, shown with two different color keys.</p>
 <div class="embedding-pair"><div><h4>Correctness (opened after freeze)</h4>{left}</div><div><h4>Trace-length nuisance</h4>{right}</div></div>
 <div class="legend"><span><i class="legend-swatch correct"></i>Blue = correct</span><span><i class="legend-swatch error"></i>Red = incorrect</span>{nuisance_legend}</div>
 <figcaption>Selection rule: <code>{_e(nodes[0]['selection_rule_id'])}</code> (label-free). Cohort: <code>{_e(nodes[0]['cohort_id'])}</code>. Stage: {_e(nodes[0]['label_stage'])}. Nodes={len(nodes)}; displayed edges={len(displayed_edges)} of {len(edges)}, selected by <code>label-free-hash-edge-sample-v1</code>. The linked CSV retains the complete graph. <a href="{csv_link}" download>Source CSV</a>.</figcaption>
</article>'''.strip())
    return "".join(cards)


def _is_alignment_cell_diagnostic(row: Mapping[str, Any]) -> bool:
    if str(row.get("cell_id", "")).startswith("aggregate::"):
        return False
    try:
        source = json.loads(str(row.get("notes", "")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        source.get("source_panel_id") == "alignment_vs_improvement"
        and source.get("source_metric_id") == "published_cell_auroc_delta_vs_iu_pcr"
    )


def _alignment_scatter(
    diagnostics: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    plot_manifest: Mapping[str, Any],
) -> str:
    points = []
    for row in diagnostics:
        if not _is_alignment_cell_diagnostic(row) or row.get("value") is None:
            continue
        try:
            source_x = float(json.loads(str(row["notes"]))["x_value"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SchemaError(f"alignment diagnostic {row.get('diagnostic_id')!r} lacks verified source x_value") from exc
        points.append((source_x, float(row["value"]), row))
    if not points:
        return '<p class="empty" id="alignment-scatter-unavailable">No verified cell-level alignment-versus-improvement rows were supplied.</p>'
    points.sort(key=lambda item: (str(item[2]["method_id"]), str(item[2]["cell_id"])))
    xs, ys = [item[0] for item in points], [item[1] for item in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = (x_max - x_min) * 0.08 or 0.01
    y_pad = (y_max - y_min) * 0.08 or 0.01
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    sx = lambda value: 75.0 + 630.0 * (value - x_min) / (x_max - x_min)
    sy = lambda value: 335.0 - 285.0 * (value - y_min) / (y_max - y_min)
    method_map = {row["method_id"]: row for row in registry["methods"]}
    point_markup = []
    for x, y, row in points:
        color = method_map[row["method_id"]]["style"]["color"]
        point_markup.append(
            f'<circle cx="{sx(x):.6f}" cy="{sy(y):.6f}" r="5" fill="{_e(color)}" stroke="white" stroke-width="1">'
            f'<title>{_e(row["cell_id"])} · {_e(method_map[row["method_id"]]["display_name"])} · alignment={_e(format(x, ".5g"))} · AUROC Δ={_e(format(y, ".5g"))}</title></circle>'
        )
    x_zero = f'<line x1="{sx(0):.6f}" x2="{sx(0):.6f}" y1="45" y2="335" stroke="currentColor" stroke-opacity=".35" stroke-dasharray="4 4"/>' if x_min <= 0 <= x_max else ""
    y_zero = f'<line x1="75" x2="705" y1="{sy(0):.6f}" y2="{sy(0):.6f}" stroke="currentColor" stroke-opacity=".35" stroke-dasharray="4 4"/>' if y_min <= 0 <= y_max else ""
    methods_used = sorted({row[2]["method_id"] for row in points}, key=lambda value: value.encode("utf-8"))
    legend = "".join(f'<span><i class="legend-swatch" style="background:{_e(method_map[mid]["style"]["color"])}"></i>{_e(method_map[mid]["display_name"])}</span>' for mid in methods_used)
    csv_link = _plot_csv_link(plot_manifest, kind="diagnostic_scatter")
    return f'''
<figure class="figure-card" id="alignment-scatter-card">
 <h3 class="plot-title">Target alignment versus paired AUROC change</h3>
 <p class="plot-subtitle">Each point is one frozen cell; no cell is selected by performance.</p>
 <div class="plot-shell"><svg id="alignment-scatter-svg" viewBox="0 0 760 385" role="img" aria-label="Target alignment effect versus paired AUROC delta scatter">
  <line x1="75" x2="705" y1="335" y2="335" stroke="currentColor"/><line x1="75" x2="75" y1="45" y2="335" stroke="currentColor"/>{x_zero}{y_zero}{''.join(point_markup)}
  <text x="390" y="375" text-anchor="middle" fill="currentColor">Target alignment effect (node-permutation median − observed roughness)</text>
  <text x="18" y="190" text-anchor="middle" fill="currentColor" transform="rotate(-90 18 190)">Paired AUROC Δ versus IU-PCR</text>
 </svg></div><div class="legend">{legend}<span>Dashed zero = no alignment/no performance change</span></div>
 <figcaption>Post-freeze descriptive check. Every point retains its exact cohort and method in the signed source rows. <a href="{csv_link}" download>Source CSV</a>.</figcaption>
</figure>'''.strip()


def default_plot_manifest(
    release_id: str,
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    metrics = list(rows_by_table.get("metrics", []))
    contrasts = list(rows_by_table.get("contrasts", []))
    diagnostics = list(rows_by_table.get("graph_diagnostics", []))
    examples = list(rows_by_table.get("graph_examples", []))
    plots: list[dict[str, Any]] = []

    # A plot is an inferential view, not merely a convenient collection of
    # numbers.  Build one contract per exact comparison group so neither the
    # static plot-data CSV nor the browser can silently mix estimands.
    metric_groups = sorted({row["comparison_group_id"] for row in metrics})
    for group_id in metric_groups:
        suffix = canonical_sha256({"group": group_id})[:16]
        plots.append(make_plot_spec(
            plot_id=f"metric_forest_{suffix}",
            title="Which methods lead within each exact comparison group?",
            kind="forest",
            source_table="metrics",
            rows=metrics,
            filters={"comparison_group_id": group_id},
            encodings={"x": "value", "y": "system_id", "interval": ["ci_low", "ci_high"], "color": "method_id"},
            legend=(
                "Color and marker identify the method card shown above.",
                "Horizontal lines are registered 95% intervals; missing intervals are shown as points only.",
                "Hollow marks are context/unverified rows and do not enter ranking.",
            ),
            caption="Only methods with the same comparison_group_id are rankable. Higher/lower direction is read from each metric row. The uncertainty set is descriptive; paired contrasts are the inferential comparison.",
            better_direction="context_dependent",
            ci_definition="Registered grouped bootstrap interval stored in metrics_long; see bootstrap_unit and bootstrap_draws.",
            selection_rule="All registered metric rows; interactive filters select task, dataset, cell, slice, metric, and group without changing scores.",
        ))
    # A 24-cell heatmap necessarily spans 24 exact row cohorts.  It is therefore
    # a faceted descriptive view, never a single rankable comparison group.  The
    # special plot contract below permits that view only when every non-cell
    # scientific field is identical and each comparison group belongs to one
    # cell.  Ranking continues to happen inside each group only.
    facet_fields = (
        # population_id is intentionally cell-specific in the strict bridge.
        # The heatmap may span those populations because every tile retains
        # its own exact cohort/comparison group; all other estimand fields must
        # remain identical.  Including population_id here would silently turn
        # the requested 24 x method matrix into 24 one-cell plots.
        "release_id", "run_id", "lane_id", "task_id",
        "metric_id", "metric_unit", "positive_class", "better_direction",
        "feature_contract_id", "access_contract_id", "evaluator_id",
        "evidence_grade", "fidelity", "adapter_id",
    )
    cell_rows = [row for row in metrics if row["aggregation_level"] == "cell"]
    facet_keys = sorted({tuple(row[field] for field in facet_fields) for row in cell_rows})
    for key in facet_keys:
        filters = {field: value for field, value in zip(facet_fields, key)}
        filters["aggregation_level"] = "cell"
        suffix = canonical_sha256({"faceted_heatmap": filters})[:16]
        plots.append(make_plot_spec(
            plot_id=f"cell_method_faceted_heatmap_{suffix}",
            title="How do methods behave across the registered cells?",
            kind="faceted_heatmap",
            source_table="metrics",
            rows=metrics,
            filters=filters,
            encodings={
                "x": "cell_id",
                "y": "system_id",
                "fill": "value",
                "facet_group": "comparison_group_id",
            },
            legend=(
                "Each tile is one exact cell × system metric row.",
                "Cells remain separate comparison groups; color shows the common absolute metric scale.",
                "Gray entries are explicit non-OK statuses, never zeros.",
            ),
            caption=(
                "This descriptive matrix aligns one metric and one feature/access/evaluator contract "
                "across cells. It does not pool rows or rank systems across incompatible cohorts."
            ),
            better_direction=key[7],
            ci_definition="Intervals remain in the per-cell table; the heatmap shows point estimates only.",
            selection_rule="All registered cells under one common scientific contract; no cell is selected by performance.",
        ))

    contrast_groups = sorted({row["comparison_group_id"] for row in contrasts})
    for group_id in contrast_groups:
        suffix = canonical_sha256({"group": group_id})[:16]
        plots.append(make_plot_spec(
            plot_id=f"paired_contrasts_{suffix}",
            title="How large is the paired change from the matched control?",
            kind="contrast_forest",
            source_table="contrasts",
            rows=contrasts,
            filters={"comparison_group_id": group_id},
            encodings={"x": "delta", "y": "left_system_id", "interval": ["ci_low", "ci_high"], "reference": 0},
            legend=(
                "Points are left system minus right system under the registered metric direction.",
                "Horizontal lines are paired grouped-bootstrap 95% intervals.",
                "The vertical zero line means no paired change.",
            ),
            caption="Contrasts are shown only on registered matched cohorts. W/T/L counts use the registered independent unit and must sum to n_pairs when available.",
            better_direction="higher",
            ci_definition="Paired grouped bootstrap; exact unit and draws are printed per row.",
            selection_rule="All registered contrast rows; no post-hoc comparator substitution.",
        ))

    if diagnostics:
        plots.append(make_plot_spec(
            plot_id="graph_assumption_summary",
            title="Graph assumption checks by named diagnostic",
            kind="diagnostic_summary",
            source_table="graph_diagnostics",
            rows=diagnostics,
            filters={},
            encodings={"x": "cell_id", "y": "method_id", "fill": "effect", "selector": "diagnostic_label"},
            legend=(
                "Label-free graph-health rows are marked separately from post-freeze target checks.",
                "Effects compare the real graph with the registered null/control when one exists.",
                "A healthy or stable graph shows that the mechanism ran; it does not by itself show correctness alignment.",
            ),
            caption="Choose one named diagnostic; rows become a cell-by-method matrix instead of a sparse matrix of unique source IDs.",
            better_direction="context_dependent",
            ci_definition="Permutation count and p_value are stored per diagnostic when a null test is defined.",
            selection_rule="All registered cells are summarized. Any example cell must have a preregistered label-free selection rule in its plot specification.",
        ))
        alignment_labels = sorted({
            str(row["diagnostic_label"])
            for row in diagnostics
            if _is_alignment_cell_diagnostic(row)
        })
        if alignment_labels:
            if len(alignment_labels) != 1:
                raise SchemaError("alignment diagnostics do not share one readable label")
            alignment_label = alignment_labels[0]
            plots.append(make_plot_spec(
                plot_id="graph_alignment_vs_auroc_delta",
                title="Does stronger target alignment predict a gain over IU-PCR?",
                kind="diagnostic_scatter",
                source_table="graph_diagnostics",
                rows=diagnostics,
                filters={"diagnostic_label": alignment_label},
                encodings={"x": "notes.source_x_value", "y": "value", "color": "method_id", "label": "cell_id"},
                legend=(
                    "Horizontal axis = signed null-minus-real target-alignment effect copied from the verified diagnostic source.",
                    "Vertical axis = paired cell AUROC delta versus IU-PCR copied from the frozen evaluator.",
                    "Color and marker identify method; aggregate correlation rows are excluded from plotted points.",
                ),
                caption="This is a descriptive across-cell relationship after score freeze, not a causal or independent validation claim.",
                better_direction="context_dependent",
                ci_definition="No new interval is computed by reporting; each point is one frozen cell comparison.",
                selection_rule="All non-aggregate cell rows with the producer-defined alignment-vs-improvement diagnostic; no cell is selected by performance.",
            ))
    for example_id in sorted({row["example_id"] for row in examples}, key=lambda value: value.encode("utf-8")):
        plots.append(make_plot_spec(
            plot_id=f"graph_example_{canonical_sha256({'example_id': example_id})[:16]}",
            title=f"Label-free-selected graph example: {example_id}",
            kind="graph_embedding_pair",
            source_table="graph_examples",
            rows=examples,
            filters={"example_id": example_id},
            encodings={"x": "embedding_x", "y": "embedding_y", "edge": ["edge_source_index", "edge_target_index"], "color_pair": ["y_error", "nuisance_value"]},
            legend=(
                "Both panels use the identical frozen two-dimensional spectral embedding and identical edges.",
                "Left color = final-answer correctness after score freeze; right color = frozen trace-length nuisance coordinate.",
                "If that nuisance coordinate is unavailable, the right panel states this explicitly instead of substituting another feature.",
            ),
            caption="The example was selected only by the registered label-free graph-health rule; performance labels did not choose the cell.",
            better_direction="context_dependent",
            ci_definition="Illustrative embedding; no confidence interval and no performance claim.",
            selection_rule=next(row["selection_rule_id"] for row in examples if row["example_id"] == example_id),
        ))
    value = {
        "schema": "reconstruction_plot_manifest_v1",
        "release_id": release_id,
        "plots": plots,
    }
    return validate_plot_manifest(value)


def _validate_inputs(
    registry: Mapping[str, Any],
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    registry = validate_registry(registry)
    normalized = {
        table: sorted(
            validate_records(table, rows_by_table.get(table, [])),
            key=lambda row: record_sort_key(table, row),
        )
        for table in ("predictions", "metrics", "contrasts", "coverage", "graph_diagnostics", "graph_examples")
    }
    validate_result_references(registry, normalized)
    validate_comparison_groups(normalized["metrics"])
    return registry, normalized


_CSS = r"""
:root{color-scheme:light dark;--bg:#f6f7fb;--panel:#fff;--ink:#18202b;--muted:#5f6b7a;--line:#d8dee8;--accent:#315ea8;--ok:#16794a;--warn:#a35b00;--bad:#a52828;--shadow:0 8px 30px rgba(24,32,43,.08)}
@media(prefers-color-scheme:dark){:root{--bg:#11151b;--panel:#191f28;--ink:#edf2f7;--muted:#a9b4c1;--line:#35404f;--accent:#8ab4f8;--ok:#66d59b;--warn:#ffc36a;--bad:#ff8d8d;--shadow:0 8px 30px rgba(0,0,0,.28)}}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:28px}.hero{padding:28px 30px;background:linear-gradient(135deg,#17345f,#315ea8);color:#fff;border-radius:18px;box-shadow:var(--shadow)}.hero h1{margin:0 0 6px;font-size:clamp(28px,4vw,46px)}.hero p{max-width:1000px;margin:8px 0}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:800;opacity:.8}.toc{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 0}.toc a{color:#fff;border:1px solid rgba(255,255,255,.35);padding:6px 10px;border-radius:999px;text-decoration:none}.section{margin-top:30px}.section>h2{font-size:28px;margin:0 0 5px}.section-intro{color:var(--muted);max-width:1050px;margin:0 0 18px}.guide-grid-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.guide-card,.panel,.figure-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}.guide-card header{display:flex;align-items:flex-start;gap:10px}.guide-card h3{margin:0;font-size:20px}.guide-card h4{margin:10px 0 4px}.subtitle{margin:2px 0;color:var(--muted)}.plain-summary{font-size:16px}.method-marker{flex:0 0 auto;margin-top:3px;overflow:visible}.method-marker>*{fill:var(--method-color);stroke:var(--method-color);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}.method-marker path{fill:none}.method-legend{display:flex;flex-wrap:wrap;gap:8px 16px;padding:12px 14px;margin:0 0 16px;background:var(--panel);border:1px solid var(--line);border-radius:12px}.method-legend-item{display:inline-flex;align-items:center;gap:5px}.method-legend-item a{color:var(--ink);text-decoration:none}.badge{margin-left:auto;background:color-mix(in srgb,var(--accent),transparent 83%);color:var(--accent);border-radius:999px;padding:3px 8px;font-size:11px;font-weight:800}.badge.subtle{margin-left:0;color:var(--muted);background:color-mix(in srgb,var(--muted),transparent 88%)}.formula{overflow:auto;background:color-mix(in srgb,var(--accent),transparent 92%);border:1px solid color-mix(in srgb,var(--accent),transparent 75%);padding:12px;border-radius:9px}.formula-terms,.compact-dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 12px}.formula-terms dt,.compact-dl dt{font-weight:800}.formula-terms dd,.compact-dl dd{margin:0}.guide-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 18px}.guide-grid ul{margin-top:4px;padding-left:20px}.caveat{color:var(--muted)}.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;padding:14px;background:var(--panel);border:1px solid var(--line);border-radius:14px;position:sticky;top:6px;z-index:10;box-shadow:var(--shadow)}label{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:12px;font-weight:800}select,button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit}button{cursor:pointer;font-weight:800;color:var(--accent)}.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:12px 0}.summary-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}.summary-card b{display:block;font-size:22px}.summary-card span{color:var(--muted)}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:14px}.plot-title{margin:0 0 4px;font-size:18px}.plot-subtitle{color:var(--muted);margin:0 0 10px}.plot-shell{width:100%;overflow:auto;min-height:170px}.plot-shell svg{display:block;min-width:520px}.legend{display:flex;flex-wrap:wrap;gap:7px 14px;padding:9px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:8px 0;font-size:12px}.legend span{display:inline-flex;align-items:center;gap:5px}.legend-dot{width:10px;height:10px;border-radius:50%;background:var(--accent)}figcaption{color:var(--muted);font-size:12px}.table-wrap{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}.status-table{max-height:none;margin:12px 0 18px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{position:sticky;top:0;background:var(--panel);z-index:2}tr[data-status]:not([data-status="OK"]):not([data-status="OK_FALLBACK"]){color:var(--muted)}.status{font-weight:800}.status.OK,.status.OK_FALLBACK{color:var(--ok)}.status.CONTEXT_ONLY,.status.UNVERIFIED{color:var(--warn)}.empty{color:var(--muted);padding:28px;text-align:center}.manifest-meta{font-size:12px;color:var(--muted)}.warning{border-left:4px solid var(--warn);padding:10px 14px;background:color-mix(in srgb,var(--warn),transparent 92%);border-radius:8px}.row-link{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;color:var(--muted)}details summary{cursor:pointer;font-weight:800}@media(max-width:720px){main{padding:14px}.guide-grid{grid-template-columns:1fr}.controls{position:static}.badge{display:none}}
.embedding-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.embedding-pair h4{margin:4px 0 6px}.example-graph-svg{display:block;width:100%;min-height:230px}.unavailable-panel{min-height:230px;border:1px dashed var(--line);border-radius:8px;display:grid;place-items:center;padding:24px;text-align:center;color:var(--muted)}.legend-swatch{display:inline-block;width:11px;height:11px;border-radius:3px;background:var(--accent)}.legend-swatch.correct{background:#2f6fed}.legend-swatch.error{background:#c43b3b}.legend-gradient{display:inline-block;width:52px;height:10px;border-radius:3px;background:linear-gradient(90deg,rgb(49,103,213),rgb(243,142,45))}.diagnostic-controls{max-width:520px;margin:10px 0}@media(max-width:760px){.embedding-pair{grid-template-columns:1fr}}
"""


_JS = r"""
const parseData=id=>JSON.parse(document.getElementById(id).textContent);
const DATA={metrics:parseData('metrics-data'),contrasts:parseData('contrasts-data'),coverage:parseData('coverage-data'),diagnostics:parseData('diagnostics-data')};
const REG=parseData('registry-data'), PLOTS=parseData('plot-manifest-data');
const METHODS=Object.fromEntries(REG.methods.map(x=>[x.method_id,x]));
const SYSTEMS=Object.fromEntries(REG.systems.map(x=>[x.system_id,x]));
const CONTEXT=new Set(['CONTEXT_ONLY','UNVERIFIED']);
const OK=new Set(['OK','OK_FALLBACK']);
const controls=['task_id','dataset_id','cell_id','slice_id','metric_id','comparison_group_id','method_id'];
const el=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x===null||x===undefined||Number.isNaN(Number(x))?'—':Number(x).toFixed(4);
const ROW_IDENTITY=['release_id','run_id','lane_id','task_id','dataset_id','population_id','cell_id','slice_id','cohort_id','comparison_group_id','feature_contract_id','access_contract_id','evaluator_id','evidence_grade','method_id','method_version_id','adapter_id','system_id','aggregation_id','aggregation_level','metric_id','fidelity'];
const rowKey=r=>ROW_IDENTITY.map(field=>String(r[field]??'')).join('␟');
function unique(rows,field){return [...new Set(rows.map(r=>r[field]).filter(x=>x!==null&&x!==undefined&&x!==''))].sort((a,b)=>String(a).localeCompare(String(b)))}
function selected(field){return el('filter-'+field).value}
function baseFilter(row,skip=''){for(const f of controls){if(f===skip)continue;const v=selected(f);if(v&&Object.prototype.hasOwnProperty.call(row,f)&&row[f]!==v)return false}return true}
function optionLabel(field,value){if(field==='method_id')return METHODS[value]?.display_name||value;return value}
function refreshOptions(){
  for(const field of controls){const select=el('filter-'+field), old=select.value;const vals=unique(DATA.metrics.filter(r=>baseFilter(r,field)),field);select.innerHTML='<option value="">All</option>'+vals.map(v=>`<option value="${esc(v)}">${esc(optionLabel(field,v))}</option>`).join('');if(vals.includes(old))select.value=old}
}
function filtered(table){return DATA[table].filter(r=>baseFilter(r))}
function metricGroups(rows){const groups={};for(const r of rows)(groups[r.comparison_group_id]??=[]).push(r);return groups}
function rankGroup(rows){
 const good=rows.filter(r=>OK.has(r.status)&&r.value!==null);if(!good.length)return [];
 const high=good[0].better_direction==='higher';good.sort((a,b)=>(high?b.value-a.value:a.value-b.value)||a.system_id.localeCompare(b.system_id));
 const best=Number(good[0].value), leaders=good.filter(r=>Number(r.value)===best);let previous=null,rank=0;
 return good.map((r,i)=>{const value=Number(r.value);if(previous===null||value!==previous)rank=i+1;previous=value;const pointLeader=value===best;const uncertaintyTie=pointLeader||leaders.some(lead=>r.ci_low!==null&&r.ci_high!==null&&lead.ci_low!==null&&lead.ci_high!==null&&!(r.ci_high<lead.ci_low||r.ci_low>lead.ci_high));return {...r,point_rank:rank,point_leader:pointLeader,uncertainty_tie:uncertaintyTie}})
}
function renderSummary(rows){
 const groups=metricGroups(rows), ranked=Object.values(groups).flatMap(rankGroup);const leaders=ranked.filter(r=>r.point_leader);const ties=ranked.filter(r=>r.uncertainty_tie);
 const coverage=filtered('coverage');const scored=coverage.reduce((s,r)=>s+(r.scored_n||0),0), expected=coverage.reduce((s,r)=>s+(r.expected_n||0),0);
 el('summary').innerHTML=[['Displayed rows',rows.length],['Exact groups',Object.keys(groups).length],['Point leaders',leaders.length?leaders.map(r=>SYSTEMS[r.system_id]?.display_name||r.system_id).join('; '):'—'],['Uncertainty set',ties.length?ties.map(r=>SYSTEMS[r.system_id]?.display_name||r.system_id).join('; '):'—'],['Coverage',expected?`${scored}/${expected}`:'—']].map(([k,v])=>`<div class="summary-card"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join('')
}
function marker(svg,x,y,method,status,size=6){
 const style=METHODS[method]?.style||{color:'#315ea8',marker:'circle'}, ns='http://www.w3.org/2000/svg';let node;
 if(style.marker==='square'){node=document.createElementNS(ns,'rect');node.setAttribute('x',x-size);node.setAttribute('y',y-size);node.setAttribute('width',size*2);node.setAttribute('height',size*2)}
 else if(style.marker==='triangle'){node=document.createElementNS(ns,'polygon');node.setAttribute('points',`${x},${y-size*1.2} ${x-size*1.1},${y+size} ${x+size*1.1},${y+size}`)}
 else if(style.marker==='diamond'){node=document.createElementNS(ns,'polygon');node.setAttribute('points',`${x},${y-size*1.3} ${x-size*1.1},${y} ${x},${y+size*1.3} ${x+size*1.1},${y}`)}
 else if(style.marker==='cross'||style.marker==='plus'){node=document.createElementNS(ns,'path');const diagonal=style.marker==='cross';node.setAttribute('d',diagonal?`M ${x-size} ${y-size} L ${x+size} ${y+size} M ${x+size} ${y-size} L ${x-size} ${y+size}`:`M ${x} ${y-size*1.25} L ${x} ${y+size*1.25} M ${x-size*1.25} ${y} L ${x+size*1.25} ${y}`);node.setAttribute('fill','none')}
 else if(style.marker==='star'){node=document.createElementNS(ns,'polygon');const points=[];for(let i=0;i<10;i++){const angle=-Math.PI/2+i*Math.PI/5,r=i%2===0?size*1.35:size*.58;points.push(`${x+Math.cos(angle)*r},${y+Math.sin(angle)*r}`)}node.setAttribute('points',points.join(' '))}
 else if(style.marker==='hexagon'){node=document.createElementNS(ns,'polygon');const points=[];for(let i=0;i<6;i++){const angle=Math.PI/6+i*Math.PI/3;points.push(`${x+Math.cos(angle)*size*1.2},${y+Math.sin(angle)*size*1.2}`)}node.setAttribute('points',points.join(' '))}
 else{node=document.createElementNS(ns,'circle');node.setAttribute('cx',x);node.setAttribute('cy',y);node.setAttribute('r',size)}
 if(!['cross','plus'].includes(style.marker))node.setAttribute('fill',CONTEXT.has(status)?'none':style.color);node.setAttribute('stroke',style.color);node.setAttribute('stroke-width','2');node.setAttribute('role','img');node.setAttribute('aria-label',`${METHODS[method]?.display_name||method}: ${style.marker} marker`);svg.appendChild(node);return node
}
function clearPlot(id,msg=''){const shell=el(id);shell.innerHTML=msg?`<div class="empty">${esc(msg)}</div>`:'';return shell}
function requireExactGroup(rows,target){const groups=unique(rows,'comparison_group_id');if(groups.length!==1){clearPlot(target,groups.length?'Select one exact comparison group to draw this plot.':'No rows match these filters.');return false}return true}
function forest(rows,target='forest-plot',valueField='value',lowField='ci_low',highField='ci_high',labelField='system_id'){
 if(!requireExactGroup(rows,target))return;const shell=clearPlot(target);const good=rows.filter(r=>r[valueField]!==null);if(!good.length){clearPlot(target,'No numeric rows match these filters.');return}
 const values=good.flatMap(r=>[r[lowField],r[valueField],r[highField]].filter(x=>x!==null).map(Number));if(valueField==='delta')values.push(0);let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){lo-=.01;hi+=.01}const pad=(hi-lo)*.08;lo-=pad;hi+=pad;
 const w=760,rowH=28,left=220,right=55,h=55+good.length*rowH,ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('width','100%');svg.setAttribute('height',h);shell.appendChild(svg);const sx=v=>left+(Number(v)-lo)/(hi-lo)*(w-left-right);
 for(let t=0;t<=4;t++){const v=lo+(hi-lo)*t/4,x=sx(v);const line=document.createElementNS(ns,'line');line.setAttribute('x1',x);line.setAttribute('x2',x);line.setAttribute('y1',18);line.setAttribute('y2',h-25);line.setAttribute('stroke','var(--line)');svg.appendChild(line);const tx=document.createElementNS(ns,'text');tx.setAttribute('x',x);tx.setAttribute('y',h-7);tx.setAttribute('text-anchor','middle');tx.setAttribute('fill','currentColor');tx.setAttribute('font-size','10');tx.textContent=v.toFixed(3);svg.appendChild(tx)}
 if(valueField==='delta'){const zero=document.createElementNS(ns,'line');zero.setAttribute('x1',sx(0));zero.setAttribute('x2',sx(0));zero.setAttribute('y1',14);zero.setAttribute('y2',h-25);zero.setAttribute('stroke','currentColor');zero.setAttribute('stroke-width','2');zero.setAttribute('stroke-dasharray','4 3');zero.setAttribute('aria-label','zero: no paired change');svg.appendChild(zero)}
 good.forEach((r,i)=>{const y=28+i*rowH;const label=document.createElementNS(ns,'text');label.setAttribute('x',left-8);label.setAttribute('y',y+4);label.setAttribute('text-anchor','end');label.setAttribute('fill','currentColor');label.setAttribute('font-size','11');label.textContent=SYSTEMS[r[labelField]]?.display_name||r[labelField];svg.appendChild(label);if(r[lowField]!==null&&r[highField]!==null){const ci=document.createElementNS(ns,'line');ci.setAttribute('x1',sx(r[lowField]));ci.setAttribute('x2',sx(r[highField]));ci.setAttribute('y1',y);ci.setAttribute('y2',y);ci.setAttribute('stroke',METHODS[r.method_id]?.style?.color||'#315ea8');ci.setAttribute('stroke-width','2');svg.appendChild(ci)}const m=marker(svg,sx(r[valueField]),y,r.method_id,r.status);m.dataset.rowKey=rowKey(r);const title=document.createElementNS(ns,'title');title.textContent=`${label.textContent}: ${fmt(r[valueField])} [${fmt(r[lowField])}, ${fmt(r[highField])}] · ${r.status}`;m.appendChild(title)})
}
function compatibleFacetedCells(rows,target){
 if(!rows.length){clearPlot(target,'No per-cell rows match these filters.');return false}
 const fixed=['release_id','run_id','lane_id','task_id','metric_id','metric_unit','positive_class','better_direction','feature_contract_id','access_contract_id','evaluator_id','evidence_grade','fidelity','adapter_id'];
 const mixed=fixed.filter(field=>unique(rows,field).length!==1);if(mixed.length){clearPlot(target,`Select one compatible metric/access view. Mixed fields: ${mixed.join(', ')}.`);return false}
 const groupCells={};for(const r of rows)(groupCells[r.comparison_group_id]??=new Set()).add(r.cell_id);if(Object.values(groupCells).some(values=>values.size!==1)){clearPlot(target,'A comparison group spans more than one cell; the faceted heatmap is blocked.');return false}
 const tiles=rows.map(r=>`${r.comparison_group_id}::${r.cell_id}::${r.system_id}`);if(new Set(tiles).size!==tiles.length){clearPlot(target,'Duplicate cell/system tiles were found; the faceted heatmap is blocked.');return false}return true
}
function heatmap(rows,target='heatmap-plot'){
 const shell=clearPlot(target), good=rows.filter(r=>r.aggregation_level==='cell');if(!compatibleFacetedCells(good,target))return;const cells=unique(good,'cell_id'),systems=unique(good,'system_id');if(!cells.length||!systems.length)return;
 const vals=good.filter(r=>r.value!==null&&OK.has(r.status)).map(r=>Number(r.value)),lo=Math.min(...vals),hi=Math.max(...vals);const cw=36,rh=24,left=205,top=110,w=left+cells.length*cw+20,h=top+systems.length*rh+25,ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('width','100%');svg.setAttribute('height',h);shell.appendChild(svg);const by=new Map(good.map(r=>[`${r.system_id}::${r.cell_id}`,r]));
 cells.forEach((c,i)=>{const t=document.createElementNS(ns,'text');t.setAttribute('x',left+i*cw+cw*.6);t.setAttribute('y',top-6);t.setAttribute('transform',`rotate(-55 ${left+i*cw+cw*.6} ${top-6})`);t.setAttribute('font-size','9');t.setAttribute('fill','currentColor');t.textContent=c;svg.appendChild(t)});
 systems.forEach((s,j)=>{const t=document.createElementNS(ns,'text');t.setAttribute('x',left-7);t.setAttribute('y',top+j*rh+16);t.setAttribute('text-anchor','end');t.setAttribute('font-size','10');t.setAttribute('fill','currentColor');t.textContent=SYSTEMS[s]?.display_name||s;svg.appendChild(t);cells.forEach((c,i)=>{const r=by.get(`${s}::${c}`),rect=document.createElementNS(ns,'rect');rect.setAttribute('x',left+i*cw);rect.setAttribute('y',top+j*rh);rect.setAttribute('width',cw-1);rect.setAttribute('height',rh-1);if(!r||r.value===null||!OK.has(r.status)){rect.setAttribute('fill','var(--line)');rect.setAttribute('opacity','.45')}else{const q=hi===lo?.5:(Number(r.value)-lo)/(hi-lo);rect.setAttribute('fill',`hsl(${220-190*q} 62% ${72-28*q}%)`)}const title=document.createElementNS(ns,'title');title.textContent=r?`${c} · ${s}: ${fmt(r.value)} · ${r.status}`:`${c} · ${s}: no registered row`;rect.appendChild(title);svg.appendChild(rect)})})
}
function renderTable(rows){const rankedMap=new Map(Object.values(metricGroups(rows)).flatMap(rankGroup).map(r=>[rowKey(r),r]));const body=rows.slice().sort((a,b)=>a.comparison_group_id.localeCompare(b.comparison_group_id)||(rankedMap.get(rowKey(a))?.point_rank??999)-(rankedMap.get(rowKey(b))?.point_rank??999)||rowKey(a).localeCompare(rowKey(b))).map(r=>{const rr=rankedMap.get(rowKey(r));return `<tr id="row-${esc(rowKey(r))}" data-status="${esc(r.status)}"><td>${esc(SYSTEMS[r.system_id]?.display_name||r.system_id)}</td><td>${fmt(r.value)}</td><td>[${fmt(r.ci_low)}, ${fmt(r.ci_high)}]</td><td>${rr?.point_rank??'—'}</td><td>${rr?.uncertainty_tie?'yes':'no'}</td><td class="status ${esc(r.status)}">${esc(r.status)}</td><td>${esc(r.n_rows)}</td><td>${esc(r.n_groups)}</td><td>${esc(r.evidence_grade)}</td><td>${esc(r.fidelity)}</td><td>${esc(r.access_contract_id)}</td><td>${esc(r.comparison_group_id)}</td><td class="row-link">${esc(rowKey(r))}</td></tr>`}).join('');el('results-body').innerHTML=body||'<tr><td colspan="13" class="empty">No rows match these filters.</td></tr>'}
function renderDiagnostics(rows){const body=rows.map(r=>`<tr data-status="${esc(r.status)}"><td>${esc(SYSTEMS[r.system_id]?.display_name||r.system_id)}</td><td>${esc(r.cell_id)}</td><td>${esc(r.diagnostic_label)}</td><td>${fmt(r.value)}</td><td>${fmt(r.null_value)}</td><td>${fmt(r.effect)}</td><td>${fmt(r.p_value)}</td><td>${esc(r.label_stage)}</td><td>${esc(r.status)}</td></tr>`).join('');el('diagnostics-body').innerHTML=body||'<tr><td colspan="9" class="empty">No graph diagnostics match these filters.</td></tr>'}
function diagnosticKey(r){const series=(r.graph_variant.match(/;series=([^;]*)/)||[])[1]||'observed',x=(r.graph_variant.match(/;x=([^;]*)/)||[])[1]||'0';return `${r.diagnostic_label} · ${series} · x=${x}`}
function selectedDiagnostics(rows){const select=el('diagnostic-selector'),keys=[...new Set(rows.map(diagnosticKey))].sort();const previous=select.value;select.innerHTML=keys.map(k=>`<option value="${esc(k)}">${esc(k)}</option>`).join('');const preferred=keys.find(k=>k.startsWith('target_vs_nuisance_roughness: error_label_roughness'))||keys[0]||'';select.value=keys.includes(previous)?previous:preferred;return rows.filter(r=>diagnosticKey(r)===select.value)}
function diagnosticHeatmap(rows,target='diagnostic-plot'){
 const shell=clearPlot(target),good=rows.filter(r=>r.effect!==null||r.value!==null);if(!good.length){clearPlot(target,'No numeric rows exist for this named diagnostic.');return}const columns=unique(good,'cell_id'),rowIds=unique(good,'system_id'),values=good.map(r=>Number(r.effect??r.value)),bound=Math.max(...values.map(Math.abs),1e-12),cw=44,rh=28,left=220,top=115,w=left+columns.length*cw+25,h=top+rowIds.length*rh+25,ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('width','100%');svg.setAttribute('height',h);shell.appendChild(svg);const by=new Map(good.map(r=>[`${r.system_id}::${r.cell_id}`,r]));columns.forEach((c,i)=>{const t=document.createElementNS(ns,'text');t.setAttribute('x',left+i*cw+cw*.55);t.setAttribute('y',top-7);t.setAttribute('transform',`rotate(-55 ${left+i*cw+cw*.55} ${top-7})`);t.setAttribute('font-size','9');t.setAttribute('fill','currentColor');t.textContent=c;svg.appendChild(t)});rowIds.forEach((s,j)=>{const t=document.createElementNS(ns,'text');t.setAttribute('x',left-7);t.setAttribute('y',top+j*rh+18);t.setAttribute('text-anchor','end');t.setAttribute('font-size','10');t.setAttribute('fill','currentColor');t.textContent=SYSTEMS[s]?.display_name||s;svg.appendChild(t);columns.forEach((c,i)=>{const r=by.get(`${s}::${c}`),rect=document.createElementNS(ns,'rect');rect.setAttribute('x',left+i*cw);rect.setAttribute('y',top+j*rh);rect.setAttribute('width',cw-1);rect.setAttribute('height',rh-1);if(!r||!OK.has(r.status)){rect.setAttribute('fill','var(--line)');rect.setAttribute('opacity','.45')}else{const v=Number(r.effect??r.value),q=(v/bound+1)/2;rect.setAttribute('fill',`hsl(${235-220*q} 60% ${72-25*Math.abs(v/bound)}%)`)}const title=document.createElementNS(ns,'title');title.textContent=r?`${r.diagnostic_label}: effect ${fmt(r.effect)}, value ${fmt(r.value)}, null ${fmt(r.null_value)} · ${r.label_stage} · ${r.status}`:`${c} · ${s}: no registered row`;rect.appendChild(title);svg.appendChild(rect)})})
}
function downloadFiltered(){const rows=filtered('metrics'),fields=Object.keys(DATA.metrics[0]||{});const quote=v=>`"${String(v??'').replaceAll('"','""')}"`;const csv=[fields.map(quote).join(','),...rows.map(r=>fields.map(f=>quote(Array.isArray(r[f])?JSON.stringify(r[f]):r[f])).join(','))].join('\n')+'\n';const blob=new Blob([csv],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='filtered_metrics_long.csv';a.click();URL.revokeObjectURL(a.href)}
function render(){refreshOptions();const rows=filtered('metrics');renderSummary(rows);forest(rows);heatmap(rows);renderTable(rows);const diagnostics=selectedDiagnostics(filtered('diagnostics'));renderDiagnostics(diagnostics);diagnosticHeatmap(diagnostics);const cs=filtered('contrasts');forest(cs,'contrast-plot','delta','ci_low','ci_high','left_system_id')}
for(const f of controls)el('filter-'+f).addEventListener('change',render);el('diagnostic-selector').addEventListener('change',render);el('download').addEventListener('click',downloadFiltered);render();
"""


def render_report(
    *,
    registry: Mapping[str, Any],
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
    plot_manifest: Optional[Mapping[str, Any]] = None,
    title: str = "Reconstruction benchmark explorer",
) -> str:
    registry, rows = _validate_inputs(registry, rows_by_table)
    plot_manifest = validate_plot_manifest(
        plot_manifest or default_plot_manifest(registry["release_id"], rows)
    )
    if plot_manifest["release_id"] != registry["release_id"]:
        raise SchemaError("plot manifest release_id does not match registry")
    supported_plot_kinds = {
        "forest",
        "heatmap",
        "faceted_heatmap",
        "contrast_forest",
        "diagnostic_heatmap",
        "diagnostic_summary",
        "diagnostic_scatter",
        "graph_embedding_pair",
    }
    unsupported = sorted(
        {
            plot["kind"]
            for plot in plot_manifest["plots"]
            if plot["kind"] not in supported_plot_kinds
        }
    )
    if unsupported:
        raise SchemaError(
            f"static report has no renderer for plot kinds: {unsupported!r}"
        )
    # Every declared plot must match the exact registered subset and hash
    # before any HTML is emitted.  Interactive result/diagnostic tables are
    # embedded.  Full graph-example rows stay in their signed CSV/Parquet;
    # static SVGs embed a fixed label-free edge sample for usability.
    plot_manifest, _ = validate_plot_data_sources(plot_manifest, rows)
    graph_example_html = _graph_example_cards(rows["graph_examples"], registry, plot_manifest)
    alignment_scatter_html = _alignment_scatter(rows["graph_diagnostics"], registry, plot_manifest)
    generated_id = canonical_sha256(
        {
            "schema": REPORT_SCHEMA,
            "registry_sha256": registry["registry_sha256"],
            "plot_manifest_sha256": plot_manifest["manifest_sha256"],
            "table_hashes": {
                table: canonical_sha256(value) for table, value in sorted(rows.items())
            },
        }
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{_CSS}</style></head>
<body><main>
<header class="hero"><div class="eyebrow">{_e(registry['release_id'])} · deterministic report {_e(generated_id[:12])}</div>
<h1>{_e(title)}</h1>
<p>This report starts by explaining every method and dataset. Results are then searchable as task → dataset → cell → slice → method. A point leader is not automatically a statistically distinct winner.</p>
<p>All numeric result plots are rendered from embedded copies of the same tidy rows used by CSV, Parquet, and DuckDB. Graph examples are rendered from a fixed label-free edge sample while the complete signed graph remains in the linked CSV. No score is typed into this page.</p>
<nav class="toc"><a href="#method-guide">Method guide</a><a href="#status-guide">Status guide</a><a href="#dataset-guide">Dataset guide</a><a href="#results">Results</a><a href="#graph-checks">Graph assumption checks</a><a href="#provenance">Provenance</a></nav></header>

<section class="section" id="method-guide"><h2>Method guide</h2>
<p class="section-intro">Read this before the leaderboard. Each card states what the name means, the mathematical operation, the source or project origin, the information it uses, and the assumption that can fail. The 30 input coordinates are oriented only once by the frozen feature contract. If a spectral solver leaves only the sign of its whole fused score unresolved, one common label-free rule aligns that whole score with the equal-family confidence average; it never reorients an individual feature. The aligned confidence score is then negated once so higher means more likely incorrect.</p>
<div class="method-legend" aria-label="Complete method legend">{_method_legend(registry)}</div>
<div class="guide-grid-cards">{_method_cards(registry)}</div></section>

<section class="section" id="status-guide"><h2>Status guide</h2>
<p class="section-intro">Every expected result remains visible. A failed, blocked, quarantined, context-only, or undefined result is a named status, never an empty cell silently interpreted as zero.</p>
{_status_table()}</section>

<section class="section" id="dataset-guide"><h2>Dataset guide</h2>
<p class="section-intro">A dataset name is not enough: the prediction unit, label, positive class, revision, and inclusion reason define the estimand.</p>
<div class="guide-grid-cards">{_dataset_cards(registry)}</div></section>

<section class="section" id="results"><h2>Results explorer</h2>
<p class="section-intro">Ranking is allowed only inside one exact <code>comparison_group_id</code>. Context-only and unverified rows remain visible but are not ranked.</p>
<div class="method-legend" aria-label="Complete result-plot method legend">{_method_legend(registry)}</div>
<div class="controls">
  <label>Task<select id="filter-task_id"></select></label><label>Dataset<select id="filter-dataset_id"></select></label>
  <label>Cell<select id="filter-cell_id"></select></label><label>Slice<select id="filter-slice_id"></select></label>
  <label>Metric<select id="filter-metric_id"></select></label><label>Exact comparison group<select id="filter-comparison_group_id"></select></label>
  <label>Method<select id="filter-method_id"></select></label><button id="download" type="button">Download filtered CSV</button>
</div>
<div class="summary-grid" id="summary"></div>
<div class="plots">
 <figure class="figure-card"><h3 class="plot-title">Metric forest</h3><p class="plot-subtitle">Point estimate and registered interval for the current filters.</p><div class="plot-shell" id="forest-plot"></div>
 <div class="legend"><span><i class="legend-dot"></i>Color/marker = method card</span><span>Line = registered 95% interval</span><span>Hollow = context/unverified, excluded from rank</span></div>
 <figcaption>Better direction, positive class, bootstrap unit, cohort, and access contract are stored per row. Compare only one exact group.</figcaption></figure>
 <figure class="figure-card"><h3 class="plot-title">Cell × method heatmap</h3><p class="plot-subtitle">Per-cell values; gray means an explicit missing/non-OK status.</p><div class="plot-shell" id="heatmap-plot"></div>
 <div class="legend"><span><i class="legend-dot"></i>Color = point estimate within the filtered metric</span><span>Gray = missing/non-OK, never zero</span></div>
 <figcaption>Color is meaningful only after selecting one metric-compatible view. Hover a tile for its cell, method, value, and status.</figcaption></figure>
 <figure class="figure-card"><h3 class="plot-title">Paired contrasts</h3><p class="plot-subtitle">Candidate minus its registered matched control.</p><div class="plot-shell" id="contrast-plot"></div>
 <div class="legend"><span><i class="legend-dot"></i>Point = paired delta</span><span>Line = paired grouped-bootstrap 95% interval</span><span>Zero = no change</span></div>
 <figcaption>The right-system comparator, pairing unit, W/T/L counts, and draw count are in <code>contrasts_long</code>.</figcaption></figure>
</div>
<h3>Exact result rows</h3><div class="table-wrap"><table><thead><tr><th>Method/system</th><th>Value</th><th>95% interval</th><th>Point rank</th><th>Uncertainty set</th><th>Status</th><th>N rows</th><th>N groups</th><th>Evidence</th><th>Fidelity</th><th>Access contract</th><th>Comparison group</th><th>Row key</th></tr></thead><tbody id="results-body"></tbody></table></div>
</section>

<section class="section" id="graph-checks"><h2>Graph assumption checks</h2>
<p class="section-intro">These checks can support or refute a mechanism. Connectivity or stability shows that a graph operated; it does not prove that the graph follows correctness. <code>label_stage</code> separates label-free health checks from analyses opened only after scores and graph hashes were frozen.</p>
<div class="warning">Node embeddings are illustrative only. Any example cell must be chosen by a preregistered label-free graph-health rule, never because it gives a favorable AUROC.</div>
<label class="diagnostic-controls">Named diagnostic<select id="diagnostic-selector"></select></label>
<figure class="figure-card"><h3 class="plot-title">Graph diagnostic cell × method matrix</h3><p class="plot-subtitle">One named and human-readable diagnostic at a time; no unique raw diagnostic IDs are used as axes.</p><div class="plot-shell" id="diagnostic-plot"></div>
<div class="legend"><span>Blue → red = negative → positive effect on a symmetric scale</span><span>Gray = missing or non-OK</span><span>Each hover states label stage and null value</span></div>
<figcaption>Each tile retains its exact cell cohort and comparison group. Connectivity checks describe the frozen operator. The grouped sensitivity panel only reweights that fitted graph; it does not refit edges and therefore does not establish graph-learning stability. Target alignment is a separate, post-freeze check.</figcaption></figure>
<div class="table-wrap"><table><thead><tr><th>Method</th><th>Cell</th><th>Diagnostic</th><th>Value</th><th>Null</th><th>Effect</th><th>p</th><th>Label stage</th><th>Status</th></tr></thead><tbody id="diagnostics-body"></tbody></table></div>
<h3>Alignment and utility</h3>{alignment_scatter_html}
<h3>Label-free-selected example graphs</h3><p class="section-intro">For each graph method, both panels below use the exact same nodes, edges, and coordinates. Only the color changes.</p>
<div class="guide-grid-cards" id="graph-example-cards">{graph_example_html}</div>
</section>

<section class="section" id="provenance"><h2>Plot contracts and provenance</h2>
<p class="section-intro">Every plot has a visible legend and caption, an exact source-table hash, a declared selection rule, and a materialized CSV in <code>07_reports/plot_data/</code>.</p>
<div class="guide-grid-cards">{''.join(f'''<article class="guide-card"><h3>{_e(plot['title'])}</h3><p>{_e(plot['caption'])}</p><ul>{''.join(f'<li>{_e(item)}</li>' for item in plot['legend'])}</ul><p class="manifest-meta">plot_id={_e(plot['plot_id'])}<br>source={_e(plot['source_table'])} · rows={_e(plot['n_source_rows'])}<br>data_sha256={_e(plot['data_sha256'])}<br>CI: {_e(plot['ci_definition'])}<br>Selection: {_e(plot['selection_rule'])}</p></article>''' for plot in plot_manifest['plots'])}</div>
</section>
</main>
<script type="application/json" id="registry-data">{_json_for_script(registry)}</script>
<script type="application/json" id="metrics-data">{_json_for_script(rows['metrics'])}</script>
<script type="application/json" id="contrasts-data">{_json_for_script(rows['contrasts'])}</script>
<script type="application/json" id="coverage-data">{_json_for_script(rows['coverage'])}</script>
<script type="application/json" id="diagnostics-data">{_json_for_script(_embedded_diagnostics(rows['graph_diagnostics']))}</script>
<script type="application/json" id="plot-manifest-data">{_json_for_script(plot_manifest)}</script>
<script>{_JS}</script></body></html>
"""


def write_report(
    path: os.PathLike[str] | str,
    *,
    registry: Mapping[str, Any],
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
    plot_manifest: Optional[Mapping[str, Any]] = None,
    title: str = "Reconstruction benchmark explorer",
) -> dict[str, Any]:
    rendered = render_report(
        registry=registry,
        rows_by_table=rows_by_table,
        plot_manifest=plot_manifest,
        title=title,
    )
    target = Path(path)
    payload = rendered.encode("utf-8")
    _atomic_write_bytes(target, payload)
    return {
        "schema": REPORT_SCHEMA,
        "path": target.name,
        "size_bytes": len(payload),
        "sha256": canonical_sha256({"html_utf8": rendered}),
        "self_contained": True,
    }
