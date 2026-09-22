"""Self-contained, file://-safe reconstruction benchmark explorer."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlparse

from .io import _atomic_write_bytes, validate_plot_data_sources
from .published_context import (
    frozen24_cell_auroc_logical_sha256,
    validate_published_context_projection,
)
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


REPORT_SCHEMA = "reconstruction_static_report_v3"
GRAPH_DISPLAY_EDGE_LIMIT = 2_000
_MECHANISM_PLOT_IDS = {
    "continuous_lsml": "continuous_lsml_correlation_clusters",
    "family_nrm_a": "family_nrm_residual_structure",
    "su_pcr": "su_pcr_support_and_eigenspectrum",
    "pgrd_a": "pgrd_energy_decomposition",
}
_ASSUMPTION_SUMMARY_PLOT_IDS = {
    "random_family_graph_control": "random_family_graph_control_summary",
    "ca_alpha_controls": "ca_registered_control_summary",
    "ca_view_weights": "ca_view_weight_summary",
    "dufs_gate_weights": "dufs_gate_weight_summary",
    "fixed_graph_group_bootstrap_stability": "fixed_graph_weight_sensitivity_summary",
    "family_nrm_family_contributions": "family_nrm_family_contribution_summary",
    "su_pcr_sparse_support_stability": "su_pcr_support_stability_summary",
}
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
    "diagnostic_unit",
    "value",
    "null_value",
    "effect",
    "p_value",
    "permutation_count",
    "label_stage",
    "status",
)
_GENERIC_DIAGNOSTIC_PANELS = {
    "graph_health",
    "target_vs_nuisance_roughness",
    "roughness_null_summary",
    "length_only_graph_control",
    "graph_operator_similarity",
    "dufs_gate_stability",
    "dufs_seed_graph_stability",
    "ca_alpha_stability",
    "ca_seed_graph_stability",
    "pgrd_seed_graph_stability",
}


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
    This projection also keeps the generic selector to a bounded set of
    one-row-per-cell assumption panels. Dense residual coordinates, seed
    sweeps, random-family enumerations, and aggregate association rows already
    have dedicated signed figure contracts/panels and remain downloadable, but
    are not duplicated into thousands of selector options.
    """

    projected = []
    for row in rows:
        variant = str(row["graph_variant"])
        panel = next(
            (
                item.split("=", 1)[1]
                for item in variant.split(";")
                if item.startswith("panel=")
            ),
            None,
        )
        if panel is not None and panel not in _GENERIC_DIAGNOSTIC_PANELS:
            continue
        if str(row["cell_id"]).startswith("aggregate::"):
            continue
        projected.append(
            {field: row[field] for field in _EMBEDDED_DIAGNOSTIC_FIELDS}
        )
    return projected


def _diagnostic_selector_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    variant = _variant_fields(row)
    return (
        str(row["diagnostic_label"]),
        variant.get("series", "observed"),
        variant.get("x", "0"),
    )


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


_PUBLISHED_SOURCE_HOSTS = {"arxiv.org", "www.arxiv.org"}


def _published_source_url(value: Any) -> str:
    parsed = urlparse(str(value))
    if parsed.scheme != "https" or parsed.hostname not in _PUBLISHED_SOURCE_HOSTS:
        raise SchemaError("published-context source URL is not on the report allowlist")
    return str(value)


def _published_context_cards(published_context: Mapping[str, Any] | None) -> str:
    if published_context is None:
        return ""
    axis_labels = {
        "dataset_revision": "Dataset revision",
        "model": "Model",
        "row_ids": "Exact rows",
        "generation": "Generation",
        "labels_grader": "Labels / grader",
        "prediction_unit": "Prediction unit",
        "metric": "Metric",
        "evaluation_protocol": "Evaluation protocol",
    }
    cards: list[str] = []
    for row in published_context["rows"]:
        status = str(row["comparison_status"])
        cell_id = str(row["cell_id"])
        axes = "".join(
            f'<tr><th>{_e(axis_labels[axis])}</th><td><span class="axis-status axis-{_e(value)}">{_e(value.replace("_", " ").lower())}</span></td></tr>'
            for axis, value in row["match_axes"].items()
        )
        reasons = "".join(
            f"<li>{_e(reason)}</li>" for reason in row["mismatch_reasons"]
        )
        if status == "NO_PUBLISHED_COMPARATOR":
            headline = "No eligible published comparator for this cell"
            paper = (
                '<p class="context-none">No paper value is registered for this exact cell. '
                "Nothing is ranked, plotted, or subtracted.</p>"
            )
            source = "<span>Not applicable</span>"
            access = supervision = passes = "Not applicable"
        else:
            published_auroc = f'{float(row["published_auroc"]):.4f}'
            headline = (
                f'{_e(row["published_method"])} · published AUROC '
                f'{_e(published_auroc)}'
            )
            paper = (
                '<p class="context-caveat">This hollow-gray paper value is context only. '
                "It uses a different population or protocol and never enters a leaderboard, "
                "forest, heatmap, or delta.</p>"
            )
            source_url = _published_source_url(row["source_url"])
            source = (
                f'<a href="{_e(source_url)}" rel="noopener noreferrer">'
                f'{_e(row["source_title"])}</a> · {_e(row["source_table"])}'
            )
            access = _e(row["access"])
            supervision = _e(row["supervision"])
            passes = _e(row["passes"])
        related = (
            '<span class="context-badge related">Related protocol</span>'
            if status == "RELATED_PUBLISHED_CONTEXT_ONLY"
            else ""
        )
        cards.append(
            f'''<article class="published-context-card" data-cell-id="{_e(cell_id)}" data-context-status="{_e(status)}" hidden>
 <header><i class="published-context-marker" aria-hidden="true"></i><div><p class="context-kicker">Published comparator context · {_e(cell_id)}</p><h3>{headline}</h3></div>{related}<span class="context-badge">{_e(status.replace("_", " ").lower())}</span></header>
 {paper}
 <dl class="context-meta"><dt>Fidelity</dt><dd>{_e(row["fidelity"])}</dd><dt>Access</dt><dd>{access}</dd><dt>Supervision</dt><dd>{supervision}</dd><dt>Passes</dt><dd>{passes}</dd><dt>Source</dt><dd>{source}</dd><dt>Common-row replay</dt><dd>{_e(row["common_replay_status"].replace("_", " ").lower())}</dd><dt>Paper → v2 delta</dt><dd><strong>Not computed</strong> (forbidden by contract)</dd></dl>
 <div class="context-grid"><div><h4>Protocol match axes</h4><div class="table-wrap context-axis-table"><table><tbody>{axes}</tbody></table></div></div><div><h4>Why this is context only</h4><ul>{reasons}</ul></div></div>
</article>'''.strip()
        )
    return "\n".join(cards)


def _plot_csv_link(
    plot_manifest: Mapping[str, Any],
    *,
    kind: str | None = None,
    plot_id: str | None = None,
    example_id: str | None = None,
) -> str:
    for plot in plot_manifest["plots"]:
        if kind is not None and plot["kind"] != kind:
            continue
        if plot_id is not None and plot["plot_id"] != plot_id:
            continue
        if example_id is not None and plot.get("filters", {}).get("example_id") != example_id:
            continue
        return f'plot_data/{_e(plot["plot_id"])}.csv'
    return ""


def _plot_spec(
    plot_manifest: Mapping[str, Any],
    *,
    kind: str | None = None,
    plot_id: str | None = None,
    example_id: str | None = None,
) -> Mapping[str, Any] | None:
    for plot in plot_manifest["plots"]:
        if kind is not None and plot["kind"] != kind:
            continue
        if plot_id is not None and plot["plot_id"] != plot_id:
            continue
        if example_id is not None and plot.get("filters", {}).get("example_id") != example_id:
            continue
        return plot
    return None


def _plot_csv_attestation(
    plot_manifest: Mapping[str, Any],
    *,
    kind: str | None = None,
    plot_id: str | None = None,
    example_id: str | None = None,
    label: str = "Exact signed source CSV",
) -> str:
    plot = _plot_spec(
        plot_manifest,
        kind=kind,
        plot_id=plot_id,
        example_id=example_id,
    )
    if plot is None:
        raise SchemaError("visible figure lacks an exact plot-data contract")
    link = f'plot_data/{_e(plot["plot_id"])}.csv'
    return (
        f'<span class="figure-source"><a href="{link}" download>{_e(label)}</a> '
        f'<code>data_sha256={_e(plot.get("data_sha256", "UNMATERIALIZED_TEST_CONTRACT"))}</code></span>'
    )


def _variant_fields(row: Mapping[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in str(row.get("graph_variant", "")).split(";"):
        key, separator, value = item.partition("=")
        if separator:
            fields[key] = value
    return fields


def _source_notes(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("notes", "")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _panel_rows(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    panel_id: str,
    method_id: str | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in diagnostics
        if _variant_fields(row).get("panel") == panel_id
        and (method_id is None or row.get("method_id") == method_id)
        and row.get("status") in RANKABLE_STATUSES
        and row.get("value") is not None
    ]


def _source_csv_link(plot_manifest: Mapping[str, Any], plot_id: str) -> str:
    link = _plot_csv_link(plot_manifest, plot_id=plot_id)
    if not link:
        raise SchemaError(
            f"visible diagnostic panel {plot_id!r} lacks an exact plot-data contract"
        )
    return link


def _interpolate_rgb(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    fraction: float,
) -> str:
    q = max(0.0, min(1.0, fraction))
    values = tuple(round(left + (right - left) * q) for left, right in zip(start, end))
    return f"rgb({values[0]},{values[1]},{values[2]})"


def _diverging_color(value: float, bound: float) -> str:
    if bound <= 0:
        return "rgb(244,244,244)"
    q = max(-1.0, min(1.0, value / bound))
    if q < 0:
        return _interpolate_rgb((45, 105, 190), (244, 244, 244), q + 1.0)
    return _interpolate_rgb((244, 244, 244), (196, 59, 59), q)


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
        '<svg class="example-graph-svg" viewBox="0 0 360 246" role="img" '
        f'aria-label="Graph spectral embedding colored by {_e(color_mode)}">'
        '<rect x="0" y="0" width="360" height="230" rx="8" fill="transparent" stroke="currentColor" stroke-opacity=".14"/>'
        + "".join(edge_markup)
        + "".join(node_markup)
        + '<text x="180" y="242" text-anchor="middle" fill="currentColor" opacity=".72" font-size="10">spectral coordinate 1 (arbitrary)</text>'
        + '<text x="9" y="115" text-anchor="middle" fill="currentColor" opacity=".72" font-size="10" transform="rotate(-90 9 115)">spectral coordinate 2 (arbitrary)</text>'
        + "</svg>"
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
            nuisance_values = [float(row["nuisance_value"]) for row in nodes]
            nuisance_low = min(nuisance_values)
            nuisance_high = max(nuisance_values)
            nuisance_legend = (
                '<span><i class="legend-gradient"></i>'
                f'{_e(nodes[0]["nuisance_name"])}: blue={_e(format(nuisance_low, ".5g"))} '
                f'→ orange={_e(format(nuisance_high, ".5g"))}</span>'
            )
        else:
            right = '<div class="unavailable-panel" role="img" aria-label="Trace-length nuisance unavailable">Trace-length nuisance coordinate unavailable for this selected cell. No substitute feature was used.</div>'
            nuisance_legend = '<span>Right panel unavailable: the signed source did not contain the nuisance coordinate.</span>'
        source_attestation = _plot_csv_attestation(
            plot_manifest,
            kind="graph_embedding_pair",
            example_id=example_id,
            label="Complete signed graph CSV",
        )
        cards.append(f'''
<article class="figure-card graph-example" data-example-id="{_e(example_id)}">
 <h3 class="plot-title">{_e(method['display_name'])} · {_e(nodes[0]['cell_id'])}</h3>
 <p class="plot-subtitle">One embedding and one edge set, shown with two different color keys. Both spectral-coordinate axes are arbitrary embedding coordinates, not measured quantities.</p>
 <div class="embedding-pair"><div><h4>Correctness (opened after freeze)</h4>{left}</div><div><h4>Trace-length nuisance</h4>{right}</div></div>
 <div class="legend"><span><i class="legend-swatch correct"></i>Blue = correct</span><span><i class="legend-swatch error"></i>Red = incorrect</span>{nuisance_legend}</div>
 <figcaption>Selection rule: <code>{_e(nodes[0]['selection_rule_id'])}</code> (label-free). Cohort: <code>{_e(nodes[0]['cohort_id'])}</code>. Stage: {_e(nodes[0]['label_stage'])}. Coordinates are an illustrative two-dimensional spectral projection and have no absolute scale. Nodes={len(nodes)}; displayed edges={len(displayed_edges)} of {len(edges)}, selected by <code>label-free-hash-edge-sample-v1</code>. The linked CSV retains the complete graph. {source_attestation}</figcaption>
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
    axis_ticks: list[str] = []
    for tick_index in range(5):
        x_value = x_min + (x_max - x_min) * tick_index / 4
        x_pos = sx(x_value)
        y_value = y_min + (y_max - y_min) * tick_index / 4
        y_pos = sy(y_value)
        axis_ticks.append(
            f'<line x1="{x_pos:.6f}" x2="{x_pos:.6f}" y1="335" y2="340" stroke="currentColor"/>'
            f'<text x="{x_pos:.6f}" y="353" text-anchor="middle" fill="currentColor" font-size="9">{x_value:.3g}</text>'
            f'<line x1="70" x2="75" y1="{y_pos:.6f}" y2="{y_pos:.6f}" stroke="currentColor"/>'
            f'<text x="66" y="{y_pos + 3:.6f}" text-anchor="end" fill="currentColor" font-size="9">{y_value:.3g}</text>'
        )
    methods_used = sorted({row[2]["method_id"] for row in points}, key=lambda value: value.encode("utf-8"))
    legend = "".join(f'<span><i class="legend-swatch" style="background:{_e(method_map[mid]["style"]["color"])}"></i>{_e(method_map[mid]["display_name"])}</span>' for mid in methods_used)
    association_rows = _panel_rows(
        diagnostics,
        panel_id="alignment_vs_improvement_summary",
    )
    associations: dict[str, dict[str, float]] = {}
    for row in association_rows:
        method_id = str(row["method_id"])
        label = str(row["diagnostic_label"]).lower()
        if "spearman" in label:
            associations.setdefault(method_id, {})["spearman"] = float(row["value"])
        elif "pearson" in label:
            associations.setdefault(method_id, {})["pearson"] = float(row["value"])
    association_markup = "".join(
        '<span class="association-stat">'
        f'{_e(method_map[method_id]["display_name"])}: '
        f'Spearman ρ={_e(format(values.get("spearman", float("nan")), ".3f"))}; '
        f'Pearson r={_e(format(values.get("pearson", float("nan")), ".3f"))}'
        "</span>"
        for method_id, values in sorted(associations.items())
        if method_id in method_map
        and "spearman" in values
        and "pearson" in values
    )
    source_attestation = _plot_csv_attestation(
        plot_manifest,
        kind="diagnostic_scatter",
    )
    return f'''
<figure class="figure-card" id="alignment-scatter-card">
 <h3 class="plot-title">Target alignment versus paired AUROC change</h3>
 <p class="plot-subtitle">Each point is one frozen cell; no cell is selected by performance.</p>
 <div class="plot-shell"><svg id="alignment-scatter-svg" viewBox="0 0 760 385" role="img" aria-label="Target alignment effect versus paired AUROC delta scatter">
  <line x1="75" x2="705" y1="335" y2="335" stroke="currentColor"/><line x1="75" x2="75" y1="45" y2="335" stroke="currentColor"/>{x_zero}{y_zero}{''.join(axis_ticks)}{''.join(point_markup)}
  <text x="390" y="375" text-anchor="middle" fill="currentColor">Target alignment effect (node-permutation median − observed roughness)</text>
  <text x="18" y="190" text-anchor="middle" fill="currentColor" transform="rotate(-90 18 190)">Paired AUROC Δ versus IU-PCR</text>
 </svg></div><div class="legend">{legend}<span>Dashed zero = no alignment/no performance change</span></div>
 <div class="association-summary" aria-label="Across-cell descriptive correlations">{association_markup}</div>
 <figcaption>Post-freeze descriptive check. The displayed Spearman and Pearson coefficients are signed aggregate diagnostic rows; they have no reporting-layer confidence interval or p-value and are not an independent validation test. Every point retains its exact cohort and method in the signed source rows. {source_attestation}</figcaption>
</figure>'''.strip()


def _continuous_lsml_panel(
    diagnostics: Sequence[Mapping[str, Any]],
    plot_manifest: Mapping[str, Any],
) -> str:
    correlation_rows = _panel_rows(
        diagnostics,
        panel_id="continuous_lsml_correlation_clusters",
        method_id="continuous_lsml",
    )
    if not correlation_rows:
        return ""
    cluster_rows = _panel_rows(
        diagnostics,
        panel_id="continuous_lsml_cluster_boundaries",
        method_id="continuous_lsml",
    )
    cluster_counts: dict[str, set[int]] = {}
    for row in cluster_rows:
        cluster_counts.setdefault(str(row["cell_id"]), set()).add(int(round(float(row["value"]))))
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in correlation_rows:
        variant = _variant_fields(row)
        notes = _source_notes(row)
        source_note = str(notes.get("source_note") or "")
        note_fields = dict(
            item.split("=", 1)
            for item in source_note.split(";")
            if "=" in item
        )
        row_feature = variant.get("series")
        column_feature = note_fields.get("column_feature")
        same_cluster = note_fields.get("same_cluster")
        if not row_feature or not column_feature or row_feature == column_feature:
            continue
        if same_cluster not in {"true", "false"}:
            continue
        bucket = "within" if same_cluster == "true" else "cross"
        grouped.setdefault(str(row["cell_id"]), {"within": [], "cross": []})[
            bucket
        ].append(abs(float(row["value"])))
    summaries = [
        (
            cell_id,
            sum(values["within"]) / len(values["within"]),
            sum(values["cross"]) / len(values["cross"]),
            len(cluster_counts.get(cell_id, set())),
        )
        for cell_id, values in grouped.items()
        if values["within"] and values["cross"]
    ]
    if not summaries:
        return ""
    summaries.sort(key=lambda item: item[0].encode("utf-8"))
    width = 920
    left, right, top, row_height = 285, 32, 28, 23
    height = top + len(summaries) * row_height + 48
    plot_width = width - left - right
    x = lambda value: left + plot_width * max(0.0, min(1.0, value))
    markup = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x_tick = x(tick)
        markup.append(
            f'<line x1="{x_tick:.3f}" x2="{x_tick:.3f}" y1="{top - 8}" y2="{height - 34}" stroke="currentColor" stroke-opacity=".10"/>'
            f'<text x="{x_tick:.3f}" y="{height - 15}" text-anchor="middle" fill="currentColor" opacity=".72" font-size="10">{tick:.2f}</text>'
        )
    for index, (cell_id, within, cross, n_clusters) in enumerate(summaries):
        y = top + index * row_height + 10
        markup.append(
            f'<text x="{left - 10}" y="{y + 3}" text-anchor="end" fill="currentColor" font-size="10">{_e(cell_id)} (K={n_clusters})</text>'
            f'<line x1="{x(cross):.3f}" x2="{x(within):.3f}" y1="{y}" y2="{y}" stroke="currentColor" stroke-opacity=".28"/>'
            f'<circle cx="{x(cross):.3f}" cy="{y}" r="4" fill="#d8872d"><title>{_e(cell_id)} · cross-cluster mean |r|={cross:.5g}</title></circle>'
            f'<circle cx="{x(within):.3f}" cy="{y}" r="4" fill="#315ea8"><title>{_e(cell_id)} · within-cluster mean |r|={within:.5g}</title></circle>'
        )
    source_attestation = _plot_csv_attestation(
        plot_manifest,
        plot_id=_MECHANISM_PLOT_IDS["continuous_lsml"],
    )
    return f'''
<figure class="figure-card mechanism-card" id="continuous-lsml-cluster-panel">
 <h3 class="plot-title">Continuous L-SML: do the correlation clusters separate?</h3>
 <p class="plot-subtitle">All cells are shown. Each line compares mean absolute feature correlation inside and across the frozen L-SML clusters.</p>
 <div class="plot-shell"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Within-cluster and cross-cluster mean absolute feature correlation for every cell">{''.join(markup)}<text x="{left + plot_width / 2:.3f}" y="{height - 2}" text-anchor="middle" fill="currentColor" font-size="11">Mean absolute Pearson correlation |r|</text></svg></div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8"></i>Within-cluster pairs</span><span><i class="legend-swatch" style="background:#d8872d"></i>Cross-cluster pairs</span><span>K = number of frozen clusters</span></div>
 <figcaption>Label-free mechanism check. Diagonal self-correlations are excluded. A larger within-minus-cross gap shows that the clustering found correlation blocks; it does not show that those blocks track correctness or improve AUROC. No cell was selected by performance. {source_attestation}</figcaption>
</figure>'''.strip()


def _family_nrm_panel(
    diagnostics: Sequence[Mapping[str, Any]],
    plot_manifest: Mapping[str, Any],
) -> str:
    covariance_rows = _panel_rows(
        diagnostics,
        panel_id="family_nrm_residual_covariance",
        method_id="family_nrm_a",
    )
    eigen_rows = _panel_rows(
        diagnostics,
        panel_id="family_nrm_residual_eigenspectrum",
        method_id="family_nrm_a",
    )
    if not covariance_rows or not eigen_rows:
        return ""
    covariance_values: dict[tuple[str, str], list[float]] = {}
    families: set[str] = set()
    for row in covariance_rows:
        row_family = _variant_fields(row).get("series")
        source_note = str(_source_notes(row).get("source_note") or "")
        column_family = next(
            (
                item.split("=", 1)[1]
                for item in source_note.split(";")
                if item.startswith("column_family=")
            ),
            None,
        )
        if not row_family or not column_family:
            continue
        families.update((row_family, column_family))
        covariance_values.setdefault((row_family, column_family), []).append(
            float(row["value"])
        )
    ordered_families = sorted(families, key=lambda value: value.encode("utf-8"))
    covariance_medians = {
        pair: median(values) for pair, values in covariance_values.items()
    }
    bound = max((abs(value) for value in covariance_medians.values()), default=1.0) or 1.0
    cell_size = 54
    left, top = 148, 118
    covariance_width = left + cell_size * len(ordered_families) + 28
    covariance_height = top + cell_size * len(ordered_families) + 30
    covariance_markup = []
    for index, family in enumerate(ordered_families):
        x_pos = left + index * cell_size + cell_size / 2
        y_pos = top + index * cell_size + cell_size / 2
        covariance_markup.append(
            f'<text x="{x_pos:.3f}" y="{top - 8}" text-anchor="end" fill="currentColor" font-size="9" transform="rotate(-50 {x_pos:.3f} {top - 8})">{_e(family)}</text>'
            f'<text x="{left - 8}" y="{y_pos + 3:.3f}" text-anchor="end" fill="currentColor" font-size="9">{_e(family)}</text>'
        )
    for row_index, row_family in enumerate(ordered_families):
        for column_index, column_family in enumerate(ordered_families):
            values = []
            if (row_family, column_family) in covariance_medians:
                values.append(covariance_medians[(row_family, column_family)])
            if (column_family, row_family) in covariance_medians:
                values.append(covariance_medians[(column_family, row_family)])
            if not values:
                continue
            value = sum(values) / len(values)
            x_pos = left + column_index * cell_size
            y_pos = top + row_index * cell_size
            text_color = "#ffffff" if abs(value) / bound > 0.55 else "#17202a"
            covariance_markup.append(
                f'<rect x="{x_pos}" y="{y_pos}" width="{cell_size - 1}" height="{cell_size - 1}" fill="{_diverging_color(value, bound)}"><title>{_e(row_family)} × {_e(column_family)} · median covariance={value:.5g}</title></rect>'
                f'<text x="{x_pos + cell_size / 2:.3f}" y="{y_pos + cell_size / 2 + 3:.3f}" text-anchor="middle" fill="{text_color}" font-size="8">{value:.2f}</text>'
            )
    spectra: dict[str, list[float]] = {}
    for row in eigen_rows:
        spectra.setdefault(str(row["cell_id"]), []).append(float(row["value"]))
    shares: dict[str, list[float]] = {}
    for cell_id, values in spectra.items():
        ranked = sorted((max(value, 0.0) for value in values), reverse=True)
        total = sum(ranked)
        if total > 0:
            shares[cell_id] = [value / total for value in ranked]
    max_rank = max((len(values) for values in shares.values()), default=0)
    if max_rank == 0:
        return ""
    spectrum_width, spectrum_height = 520, covariance_height
    spectrum_left, spectrum_right, spectrum_top, spectrum_bottom = 58, 25, 34, 55
    spectrum_plot_width = spectrum_width - spectrum_left - spectrum_right
    spectrum_plot_height = spectrum_height - spectrum_top - spectrum_bottom
    y_max = max(max(values) for values in shares.values()) * 1.05 or 1.0
    sx = lambda rank: spectrum_left + spectrum_plot_width * (rank - 1) / max(1, max_rank - 1)
    sy = lambda value: spectrum_top + spectrum_plot_height * (1.0 - value / y_max)
    spectrum_markup = []
    for tick_index in range(5):
        value = y_max * tick_index / 4
        y_pos = sy(value)
        spectrum_markup.append(
            f'<line x1="{spectrum_left}" x2="{spectrum_width - spectrum_right}" y1="{y_pos:.3f}" y2="{y_pos:.3f}" stroke="currentColor" stroke-opacity=".10"/>'
            f'<text x="{spectrum_left - 8}" y="{y_pos + 3:.3f}" text-anchor="end" fill="currentColor" opacity=".72" font-size="9">{value:.2f}</text>'
        )
    for cell_id, values in sorted(shares.items()):
        points = " ".join(
            f"{sx(rank):.3f},{sy(value):.3f}"
            for rank, value in enumerate(values, start=1)
        )
        spectrum_markup.append(
            f'<polyline points="{points}" fill="none" stroke="#315ea8" stroke-opacity=".16" stroke-width="1"><title>{_e(cell_id)} residual covariance eigenvalue shares</title></polyline>'
        )
    median_shares = [
        median(values[rank] for values in shares.values() if len(values) > rank)
        for rank in range(max_rank)
    ]
    median_points = " ".join(
        f"{sx(rank):.3f},{sy(value):.3f}"
        for rank, value in enumerate(median_shares, start=1)
    )
    spectrum_markup.append(
        f'<polyline points="{median_points}" fill="none" stroke="#c43b3b" stroke-width="3"/>'
    )
    for rank in range(1, max_rank + 1):
        spectrum_markup.append(
            f'<text x="{sx(rank):.3f}" y="{spectrum_height - 31}" text-anchor="middle" fill="currentColor" font-size="9">{rank}</text>'
        )
    source_attestation = _plot_csv_attestation(
        plot_manifest,
        plot_id=_MECHANISM_PLOT_IDS["family_nrm_a"],
    )
    return f'''
<figure class="figure-card mechanism-card" id="family-nrm-residual-panel">
 <h3 class="plot-title">Family-NRM-A: residual covariance and eigenspectrum</h3>
 <p class="plot-subtitle">The covariance matrix is the across-cell median for each family pair; the spectrum shows every cell and its median.</p>
 <div class="mechanism-grid"><div><h4>Median residual covariance</h4><div class="plot-shell"><svg viewBox="0 0 {covariance_width} {covariance_height}" role="img" aria-label="Median Family-NRM residual covariance across cells">{''.join(covariance_markup)}</svg></div></div><div><h4>Residual covariance eigenvalue share</h4><div class="plot-shell"><svg viewBox="0 0 {spectrum_width} {spectrum_height}" role="img" aria-label="Family-NRM residual covariance eigenspectrum in every cell">{''.join(spectrum_markup)}<text x="{spectrum_left + spectrum_plot_width / 2:.3f}" y="{spectrum_height - 7}" text-anchor="middle" fill="currentColor" font-size="10">Eigenvalue rank</text><text x="13" y="{spectrum_top + spectrum_plot_height / 2:.3f}" text-anchor="middle" fill="currentColor" font-size="10" transform="rotate(-90 13 {spectrum_top + spectrum_plot_height / 2:.3f})">Share of covariance trace</text></svg></div></div></div>
 <div class="legend"><span><i class="legend-gradient diverging"></i>Covariance: blue negative, white zero, red positive; numbers are medians</span><span><i class="legend-swatch" style="background:#315ea8;opacity:.35"></i>One cell</span><span><i class="legend-swatch" style="background:#c43b3b"></i>Across-cell median spectrum</span></div>
 <figcaption>Label-free mechanism check over all available cells. Tiny negative eigenvalues are clipped to zero only when converting the signed covariance eigenvalues to variance shares for this display. Concentrated residual covariance shows a shared residual axis; it does not identify its correctness orientation or imply an AUROC gain. {source_attestation}</figcaption>
</figure>'''.strip()


def _cell_series_svg(
    cell_values: Mapping[str, Mapping[str, float]],
    *,
    series: Sequence[tuple[str, str, str]],
    axis_label: str,
    sort_series: str,
) -> str:
    cells = sorted(
        cell_values,
        key=lambda cell_id: (
            float(cell_values[cell_id].get(sort_series, 0.0)),
            cell_id.encode("utf-8"),
        ),
    )
    values = [
        float(value)
        for cell_id in cells
        for value in cell_values[cell_id].values()
    ] + [0.0]
    low, high = min(values), max(values)
    if low == high:
        low -= 0.01
        high += 0.01
    pad = (high - low) * 0.08
    low, high = low - pad, high + pad
    width = 800
    left, right, top, row_height = 280, 32, 26, 23
    height = top + len(cells) * row_height + 49
    plot_width = width - left - right
    sx = lambda value: left + plot_width * (value - low) / (high - low)
    markup = []
    for tick_index in range(5):
        value = low + (high - low) * tick_index / 4
        x_pos = sx(value)
        markup.append(
            f'<line x1="{x_pos:.3f}" x2="{x_pos:.3f}" y1="{top - 8}" y2="{height - 34}" stroke="currentColor" stroke-opacity=".10"/>'
            f'<text x="{x_pos:.3f}" y="{height - 15}" text-anchor="middle" fill="currentColor" opacity=".72" font-size="9">{value:.3g}</text>'
        )
    if low <= 0 <= high:
        markup.append(
            f'<line x1="{sx(0):.3f}" x2="{sx(0):.3f}" y1="{top - 8}" y2="{height - 34}" stroke="currentColor" stroke-opacity=".55" stroke-dasharray="4 3"/>'
        )
    for row_index, cell_id in enumerate(cells):
        y_pos = top + row_index * row_height + 10
        markup.append(
            f'<text x="{left - 9}" y="{y_pos + 3}" text-anchor="end" fill="currentColor" font-size="10">{_e(cell_id)}</text>'
        )
        if len(series) == 1 and series[0][0] in cell_values[cell_id]:
            value = float(cell_values[cell_id][series[0][0]])
            markup.append(
                f'<line x1="{sx(0):.3f}" x2="{sx(value):.3f}" y1="{y_pos}" y2="{y_pos}" stroke="{series[0][1]}" stroke-opacity=".45" stroke-width="3"/>'
            )
        for series_index, (series_id, color, marker) in enumerate(series):
            if series_id not in cell_values[cell_id]:
                continue
            value = float(cell_values[cell_id][series_id])
            y_offset = (series_index - (len(series) - 1) / 2) * 5
            if marker == "square":
                shape = f'<rect x="{sx(value) - 4:.3f}" y="{y_pos + y_offset - 4:.3f}" width="8" height="8" fill="{color}"/>'
            elif marker == "diamond":
                shape = f'<polygon points="{sx(value):.3f},{y_pos + y_offset - 5:.3f} {sx(value) + 5:.3f},{y_pos + y_offset:.3f} {sx(value):.3f},{y_pos + y_offset + 5:.3f} {sx(value) - 5:.3f},{y_pos + y_offset:.3f}" fill="{color}"/>'
            else:
                shape = f'<circle cx="{sx(value):.3f}" cy="{y_pos + y_offset:.3f}" r="4" fill="{color}"/>'
            markup.append(
                f'<g>{shape}<title>{_e(cell_id)} · {_e(series_id)}={value:.6g}</title></g>'
            )
    markup.append(
        f'<text x="{left + plot_width / 2:.3f}" y="{height - 2}" text-anchor="middle" fill="currentColor" font-size="10">{_e(axis_label)}</text>'
    )
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(axis_label)} by cell">{"".join(markup)}</svg>'


def _cell_collapsed_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    category,
    value_field: str = "value",
) -> dict[str, list[float]]:
    """Collapse repeated seeds/draws within cell, then retain all cell medians."""

    by_cell: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        category_id = str(category(row))
        by_cell.setdefault((category_id, str(row["cell_id"])), []).append(
            float(value)
        )
    collapsed: dict[str, list[float]] = {}
    for (category_id, _cell_id), values in by_cell.items():
        collapsed.setdefault(category_id, []).append(float(median(values)))
    return collapsed


def _range_summary_svg(
    category_values: Mapping[str, Sequence[float]],
    *,
    axis_label: str,
    signed: bool = False,
) -> str:
    """Render min--median--max across cell-level summaries with numeric ticks."""

    summaries = [
        (category, min(values), float(median(values)), max(values), len(values))
        for category, values in category_values.items()
        if values
    ]
    summaries.sort(key=lambda item: (item[2], item[0].encode("utf-8")))
    if not summaries:
        return '<p class="empty">No numeric signed rows were available.</p>'
    values = [value for _, low, center, high, _ in summaries for value in (low, center, high)]
    if signed:
        values.append(0.0)
    low, high = min(values), max(values)
    if low == high:
        pad = abs(low) * 0.05 or 0.01
        low, high = low - pad, high + pad
    pad = (high - low) * 0.07
    low, high = low - pad, high + pad
    width = 920
    left, right, top, row_height = 365, 42, 28, 23
    height = top + len(summaries) * row_height + 50
    plot_width = width - left - right
    sx = lambda value: left + plot_width * (float(value) - low) / (high - low)
    markup: list[str] = []
    for tick_index in range(5):
        value = low + (high - low) * tick_index / 4
        x_pos = sx(value)
        markup.append(
            f'<line x1="{x_pos:.3f}" x2="{x_pos:.3f}" y1="{top - 8}" y2="{height - 34}" stroke="currentColor" stroke-opacity=".10"/>'
            f'<text x="{x_pos:.3f}" y="{height - 15}" text-anchor="middle" fill="currentColor" opacity=".72" font-size="9">{value:.3g}</text>'
        )
    if low <= 0 <= high:
        markup.append(
            f'<line x1="{sx(0):.3f}" x2="{sx(0):.3f}" y1="{top - 8}" y2="{height - 34}" stroke="currentColor" stroke-opacity=".55" stroke-dasharray="4 3"/>'
        )
    for row_index, (category, minimum, center, maximum, n_cells) in enumerate(summaries):
        y_pos = top + row_index * row_height + 10
        markup.append(
            f'<text x="{left - 10}" y="{y_pos + 3}" text-anchor="end" fill="currentColor" font-size="9">{_e(category)}</text>'
            f'<line x1="{sx(minimum):.3f}" x2="{sx(maximum):.3f}" y1="{y_pos}" y2="{y_pos}" stroke="#8a929d" stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="{sx(center):.3f}" cy="{y_pos}" r="4.5" fill="#315ea8" stroke="white" stroke-width=".8"><title>{_e(category)} · cell median={center:.6g}; cell range=[{minimum:.6g}, {maximum:.6g}]; n_cells={n_cells}</title></circle>'
        )
    markup.append(
        f'<text x="{left + plot_width / 2:.3f}" y="{height - 2}" text-anchor="middle" fill="currentColor" font-size="10">{_e(axis_label)}</text>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_e(axis_label)}: cell median and full cell range">'
        + "".join(markup)
        + "</svg>"
    )


def _diagnostic_summary_panels(
    diagnostics: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    plot_manifest: Mapping[str, Any],
) -> str:
    """Render bounded, audit-linked summaries for the promised mechanism checks."""

    method_names = {
        str(method["method_id"]): str(method["display_name"])
        for method in registry["methods"]
    }
    figures: list[str] = []

    random_rows = _panel_rows(diagnostics, panel_id="random_family_graph_control")
    if random_rows:
        short = {
            "Random feature-family graph control — error label roughness": "error roughness",
            "Random feature-family graph control — trace length coordinate roughness": "length roughness",
            "Random feature-family graph control — edge support jaccard vs fitted": "edge Jaccard vs fitted",
            "Random feature-family graph control — operator cosine vs fitted": "operator cosine vs fitted",
        }
        values = _cell_collapsed_values(
            random_rows,
            category=lambda row: (
                f'{method_names.get(str(row["method_id"]), row["method_id"])} · '
                f'{short.get(str(row["diagnostic_label"]), row["diagnostic_label"])}'
            ),
        )
        attestation = _plot_csv_attestation(
            plot_manifest,
            plot_id=_ASSUMPTION_SUMMARY_PLOT_IDS["random_family_graph_control"],
        )
        figures.append(f'''
<figure class="figure-card mechanism-card" id="random-family-control-panel">
 <h3 class="plot-title">Random-family graph controls</h3>
 <p class="plot-subtitle">Each cell is first summarized over its preregistered target-blind family draws; the dot and line then show the across-cell median and full cell range.</p>
 <div class="plot-shell">{_range_summary_svg(values, axis_label="Registered roughness/similarity value (separate named rows; descriptive)")}</div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8;border-radius:50%"></i>Dot = median of cell medians</span><span><i class="legend-swatch" style="background:#8a929d;width:28px;height:3px"></i>Line = full range across cells</span><span>Family draws are target-blind; lower roughness is smoother, higher similarity is closer to the fitted graph</span></div>
 <figcaption>This is the visible random-family control promised by the graph-assumption plan. Roughness and operator-similarity rows share a bounded display but retain separate names and exact values; no combined score is computed. {attestation}</figcaption>
</figure>'''.strip())

    ca_control_rows = _panel_rows(diagnostics, panel_id="ca_alpha_controls")
    if ca_control_rows:
        values = _cell_collapsed_values(
            ca_control_rows,
            category=lambda row: (
                f'{_variant_fields(row).get("series", "control")} · '
                + (
                    "error roughness"
                    if "error label" in str(row["diagnostic_label"]).lower()
                    else "length roughness"
                )
            ),
            value_field="effect",
        )
        attestation = _plot_csv_attestation(
            plot_manifest,
            plot_id=_ASSUMPTION_SUMMARY_PLOT_IDS["ca_alpha_controls"],
        )
        figures.append(f'''
<figure class="figure-card mechanism-card" id="ca-control-panel">
 <h3 class="plot-title">CA-SpecRaGE: equal, prior, global-mean and permuted controls</h3>
 <p class="plot-subtitle">Control roughness minus learned-graph roughness, summarized across every registered cell.</p>
 <div class="plot-shell">{_range_summary_svg(values, axis_label="Control roughness − learned-graph roughness", signed=True)}</div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8;border-radius:50%"></i>Dot = median cell effect</span><span><i class="legend-swatch" style="background:#8a929d;width:28px;height:3px"></i>Line = full cell range</span><span>Positive = learned graph is smoother than that named control; dashed zero = equal roughness</span></div>
 <figcaption>Error-label roughness is a post-freeze explanation; trace-length roughness is a nuisance check. Equal/prior/permuted controls are not ranked as methods. {attestation}</figcaption>
</figure>'''.strip())

    ca_weight_rows = _panel_rows(diagnostics, panel_id="ca_view_weights")
    if ca_weight_rows:
        paired: dict[tuple[str, str], dict[str, float]] = {}
        for row in ca_weight_rows:
            key = (str(row["cell_id"]), _variant_fields(row).get("series", "unknown"))
            label = str(row["diagnostic_label"]).lower()
            name = "learned" if "learned alpha" in label else "prior"
            paired.setdefault(key, {})[name] = float(row["value"])
        differences: dict[str, list[float]] = {}
        for (_cell_id, feature), values_for_pair in paired.items():
            if set(values_for_pair) == {"learned", "prior"}:
                differences.setdefault(feature, []).append(
                    values_for_pair["learned"] - values_for_pair["prior"]
                )
        attestation = _plot_csv_attestation(
            plot_manifest,
            plot_id=_ASSUMPTION_SUMMARY_PLOT_IDS["ca_view_weights"],
        )
        figures.append(f'''
<figure class="figure-card mechanism-card" id="ca-view-weight-panel">
 <h3 class="plot-title">CA-SpecRaGE: learned view weights versus the frozen prior</h3>
 <p class="plot-subtitle">For each of the 30 monotone features, the display shows learned alpha minus its frozen prior across cells.</p>
 <div class="plot-shell">{_range_summary_svg(differences, axis_label="Mean learned alpha − frozen view prior", signed=True)}</div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8;border-radius:50%"></i>Dot = across-cell median shift</span><span><i class="legend-swatch" style="background:#8a929d;width:28px;height:3px"></i>Line = full cell range</span><span>Positive = more weight than prior; dashed zero = unchanged</span></div>
 <figcaption>This panel shows what CA reweighted; it does not say that a larger alpha is better or causally responsible for AUROC. {attestation}</figcaption>
</figure>'''.strip())

    dufs_weight_rows = _panel_rows(diagnostics, panel_id="dufs_gate_weights")
    dufs_seed_rows = _panel_rows(diagnostics, panel_id="dufs_gate_weights_per_seed")
    if dufs_weight_rows and dufs_seed_rows:
        weights = _cell_collapsed_values(
            dufs_weight_rows,
            category=lambda row: _variant_fields(row).get("series", "unknown"),
        )
        survival = _cell_collapsed_values(
            dufs_seed_rows,
            category=lambda row: _variant_fields(row).get("series", "unknown"),
        )
        attestation = _plot_csv_attestation(
            plot_manifest,
            plot_id=_ASSUMPTION_SUMMARY_PLOT_IDS["dufs_gate_weights"],
        )
        figures.append(f'''
<figure class="figure-card mechanism-card" id="dufs-gate-weight-panel">
 <h3 class="plot-title">DUFS-LIU: feature gate weight and seed survival</h3>
 <p class="plot-subtitle">All 30 oriented features are shown. Repeated seed rows are collapsed within cell before the across-cell range is drawn.</p>
 <div class="mechanism-grid"><div><h4>RMS-normalized gate weight</h4><div class="plot-shell">{_range_summary_svg(weights, axis_label="RMS-normalized DUFS gate weight")}</div></div><div><h4>Gate survival across seeds</h4><div class="plot-shell">{_range_summary_svg(survival, axis_label="Gate survival probability across registered seeds")}</div></div></div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8;border-radius:50%"></i>Dot = across-cell median</span><span><i class="legend-swatch" style="background:#8a929d;width:28px;height:3px"></i>Line = full cell range</span><span>Higher weight/survival means DUFS retained that feature more strongly/often; it does not imply label relevance</span></div>
 <figcaption>The gate is fitted without hallucination labels. This is the visible feature-selection diagnostic required by the DUFS mechanism check. {attestation}</figcaption>
</figure>'''.strip())

    fixed_rows = _panel_rows(
        diagnostics, panel_id="fixed_graph_group_bootstrap_stability"
    )
    if fixed_rows:
        kept_phrases = (
            "edge support jaccard",
            "weighted graph frobenius cosine",
            "normalized laplacian frobenius cosine",
            "normalized laplacian relative difference",
        )
        selected_rows = [
            row
            for row in fixed_rows
            if any(
                phrase in str(row["diagnostic_label"]).lower()
                for phrase in kept_phrases
            )
        ]
        values = _cell_collapsed_values(
            selected_rows,
            category=lambda row: (
                f'{method_names.get(str(row["method_id"]), row["method_id"])} · '
                f'{str(row["diagnostic_label"]).split(" — ")[-1]}'
            ),
        )
        attestation = _plot_csv_attestation(
            plot_manifest,
            plot_id=_ASSUMPTION_SUMMARY_PLOT_IDS[
                "fixed_graph_group_bootstrap_stability"
            ],
        )
        figures.append(f'''
<figure class="figure-card mechanism-card" id="fixed-graph-bootstrap-panel">
 <h3 class="plot-title">Fixed-graph source-group bootstrap sensitivity</h3>
 <p class="plot-subtitle">Each cell is summarized over registered group-resampling draws; the fitted graph is reweighted, not refit.</p>
 <div class="plot-shell">{_range_summary_svg(values, axis_label="Similarity/difference after source-group reweighting")}</div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8;border-radius:50%"></i>Dot = median of cell medians</span><span><i class="legend-swatch" style="background:#8a929d;width:28px;height:3px"></i>Line = full cell range</span><span>Higher cosine/Jaccard = more stable; lower relative difference = more stable</span></div>
 <figcaption>This is weight sensitivity on one frozen graph, not stability of refitting graph edges. Retained-node fraction and effective row mass remain in the linked source CSV. {attestation}</figcaption>
</figure>'''.strip())

    family_rows = _panel_rows(
        diagnostics,
        panel_id="family_nrm_family_contributions",
        method_id="family_nrm_a",
    )
    if family_rows:
        shares = _cell_collapsed_values(
            [
                row
                for row in family_rows
                if "absolute direction share" in str(row["diagnostic_label"]).lower()
            ],
            category=lambda row: _variant_fields(row).get("series", "unknown"),
        )
        coefficients = _cell_collapsed_values(
            [
                row
                for row in family_rows
                if "direction coefficient" in str(row["diagnostic_label"]).lower()
            ],
            category=lambda row: _variant_fields(row).get("series", "unknown"),
        )
        attestation = _plot_csv_attestation(
            plot_manifest,
            plot_id=_ASSUMPTION_SUMMARY_PLOT_IDS[
                "family_nrm_family_contributions"
            ],
        )
        figures.append(f'''
<figure class="figure-card mechanism-card" id="family-nrm-contribution-panel">
 <h3 class="plot-title">Family-NRM-A: contribution of each measurement family</h3>
 <p class="plot-subtitle">The six registered families are shown separately, so the residual direction is not presented as an unexplained single number.</p>
 <div class="mechanism-grid"><div><h4>Absolute share of the correction direction</h4><div class="plot-shell">{_range_summary_svg(shares, axis_label="Absolute direction share")}</div></div><div><h4>Signed direction coefficient</h4><div class="plot-shell">{_range_summary_svg(coefficients, axis_label="Signed residual-direction coefficient", signed=True)}</div></div></div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8;border-radius:50%"></i>Dot = across-cell median</span><span><i class="legend-swatch" style="background:#8a929d;width:28px;height:3px"></i>Line = full cell range</span><span>Share measures magnitude; coefficient preserves sign</span></div>
 <figcaption>These are label-free fitted contributions. A consistent family coefficient does not establish its correctness orientation; performance is evaluated separately. {attestation}</figcaption>
</figure>'''.strip())

    su_stability_rows = _panel_rows(
        diagnostics,
        panel_id="su_pcr_sparse_support_stability",
        method_id="su_pcr",
    )
    if su_stability_rows:
        values = _cell_collapsed_values(
            su_stability_rows,
            category=lambda row: str(row["diagnostic_label"]).split(" — ")[-1],
        )
        attestation = _plot_csv_attestation(
            plot_manifest,
            plot_id=_ASSUMPTION_SUMMARY_PLOT_IDS[
                "su_pcr_sparse_support_stability"
            ],
        )
        figures.append(f'''
<figure class="figure-card mechanism-card" id="su-pcr-support-stability-panel">
 <h3 class="plot-title">SU-PCR: sparse-support stability under source-group bootstrap</h3>
 <p class="plot-subtitle">Bootstrap draws are collapsed within cell; all cells remain in the across-cell range.</p>
 <div class="plot-shell">{_range_summary_svg(values, axis_label="Registered SU-PCR bootstrap diagnostic")}</div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8;border-radius:50%"></i>Dot = median of cell medians</span><span><i class="legend-swatch" style="background:#8a929d;width:28px;height:3px"></i>Line = full cell range</span><span>Higher support Jaccard = more stable; convergence is 0/1; residual/support fraction retain their named units</span></div>
 <figcaption>The plotted rows diagnose stability of the fitted sparse structure, not AUROC. No cross-metric average is computed. {attestation}</figcaption>
</figure>'''.strip())

    if not figures:
        return '<p class="empty" id="diagnostic-summary-panels-unavailable">No signed bounded diagnostic summaries were supplied.</p>'
    return '<div class="mechanism-panels">' + "".join(figures) + "</div>"


def _su_pcr_panel(
    diagnostics: Sequence[Mapping[str, Any]],
    plot_manifest: Mapping[str, Any],
) -> str:
    support_rows = [
        row
        for row in _panel_rows(
            diagnostics,
            panel_id="su_pcr_decomposition",
            method_id="su_pcr",
        )
        if "sparse support fraction off diagonal" in str(row["diagnostic_label"]).lower()
    ]
    eigen_rows = _panel_rows(
        diagnostics,
        panel_id="su_pcr_low_rank_eigenspectrum",
        method_id="su_pcr",
    )
    if not support_rows or not eigen_rows:
        return ""
    support = {
        str(row["cell_id"]): {"support_fraction": float(row["value"])}
        for row in support_rows
    }
    eigenvalues: dict[str, list[float]] = {}
    for row in eigen_rows:
        eigenvalues.setdefault(str(row["cell_id"]), []).append(float(row["value"]))
    minimum_eigenvalue = {
        cell_id: {"minimum_eigenvalue": min(values)}
        for cell_id, values in eigenvalues.items()
        if values
    }
    support_svg = _cell_series_svg(
        support,
        series=(("support_fraction", "#315ea8", "circle"),),
        axis_label="Off-diagonal sparse-support fraction (0 = empty support)",
        sort_series="support_fraction",
    )
    eigen_svg = _cell_series_svg(
        minimum_eigenvalue,
        series=(("minimum_eigenvalue", "#c43b3b", "diamond"),),
        axis_label="Minimum eigenvalue of the fitted low-rank component",
        sort_series="minimum_eigenvalue",
    )
    source_attestation = _plot_csv_attestation(
        plot_manifest,
        plot_id=_MECHANISM_PLOT_IDS["su_pcr"],
    )
    return f'''
<figure class="figure-card mechanism-card" id="su-pcr-structure-panel">
 <h3 class="plot-title">SU-PCR: sparse support and low-rank eigenvalues</h3>
 <p class="plot-subtitle">Every cell is shown; the two views use separate axes because support fraction and covariance eigenvalues have different units.</p>
 <div class="mechanism-grid"><div><h4>Recovered sparse dependence</h4><div class="plot-shell">{support_svg}</div></div><div><h4>Low-rank decomposition warning</h4><div class="plot-shell">{eigen_svg}</div></div></div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8"></i>Off-diagonal nonzero support fraction</span><span><i class="legend-swatch" style="background:#c43b3b;transform:rotate(45deg)"></i>Minimum fitted low-rank eigenvalue</span><span>Dashed zero = empty support / no negative eigenvalue</span></div>
 <figcaption>Label-free decomposition check. Zero support means the fitted sparse component added no off-diagonal dependencies in that cell. A negative low-rank eigenvalue flags a fitted decomposition property; this panel does not establish that it caused any performance change. {source_attestation}</figcaption>
</figure>'''.strip()


def _pgrd_panel(
    diagnostics: Sequence[Mapping[str, Any]],
    plot_manifest: Mapping[str, Any],
) -> str:
    rows = _panel_rows(
        diagnostics,
        panel_id="pgrd_cross_gradient",
        method_id="pgrd_a",
    )
    labels = {
        "cross_term": "cross term at registered direction",
        "quadratic_term": "quadratic term at registered direction",
        "unit_step_change": "predicted energy change at unit step",
    }
    cell_values: dict[str, dict[str, float]] = {}
    for row in rows:
        label = str(row["diagnostic_label"]).lower()
        for series_id, phrase in labels.items():
            if phrase in label:
                cell_values.setdefault(str(row["cell_id"]), {})[series_id] = float(
                    row["value"]
                )
    cell_values = {
        cell_id: values
        for cell_id, values in cell_values.items()
        if set(values) == set(labels)
    }
    if not cell_values:
        return ""
    svg = _cell_series_svg(
        cell_values,
        series=(
            ("cross_term", "#315ea8", "circle"),
            ("quadratic_term", "#d8872d", "square"),
            ("unit_step_change", "#8b3fb1", "diamond"),
        ),
        axis_label="Registered graph-energy term (negative = energy descent)",
        sort_series="unit_step_change",
    )
    descending = sum(
        values["unit_step_change"] < 0 for values in cell_values.values()
    )
    source_attestation = _plot_csv_attestation(
        plot_manifest,
        plot_id=_MECHANISM_PLOT_IDS["pgrd_a"],
    )
    return f'''
<figure class="figure-card mechanism-card" id="pgrd-energy-panel">
 <h3 class="plot-title">PGRD-A: registered graph-energy decomposition</h3>
 <p class="plot-subtitle">For each cell, the predicted unit-step change equals the signed cross term plus the non-negative quadratic term.</p>
 <div class="plot-shell">{svg}</div>
 <div class="legend"><span><i class="legend-swatch" style="background:#315ea8"></i>Cross term / directional derivative at zero</span><span><i class="legend-swatch" style="background:#d8872d"></i>Quadratic term</span><span><i class="legend-swatch" style="background:#8b3fb1;transform:rotate(45deg)"></i>Total predicted unit-step energy change</span><span>{descending}/{len(cell_values)} cells have negative predicted unit-step change</span></div>
 <figcaption>Label-free mechanism check over every cell. A negative total means the registered direction lowers this graph energy at a unit step. It does not mean that the direction follows correctness or improves AUROC; those are separate post-freeze results. {source_attestation}</figcaption>
</figure>'''.strip()


def _mechanism_panels(
    diagnostics: Sequence[Mapping[str, Any]],
    plot_manifest: Mapping[str, Any],
) -> str:
    panels = [
        _continuous_lsml_panel(diagnostics, plot_manifest),
        _family_nrm_panel(diagnostics, plot_manifest),
        _su_pcr_panel(diagnostics, plot_manifest),
        _pgrd_panel(diagnostics, plot_manifest),
    ]
    available = [panel for panel in panels if panel]
    if not available:
        return '<p class="empty" id="mechanism-panels-unavailable">No signed method-specific mechanism panels were supplied.</p>'
    return '<div class="mechanism-panels">' + "".join(available) + "</div>"


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
            ci_definition="No reporting-layer interval is added. A node-permutation median without a producer-supplied p_value is descriptive only; grouped source resampling reweights a fixed fitted graph and is not a refit-stability interval.",
            selection_rule="All registered cells are summarized. Any example cell must have a preregistered label-free selection rule in its plot specification.",
        ))
        selector_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
        for row in _embedded_diagnostics(diagnostics):
            selector_groups.setdefault(_diagnostic_selector_key(row), []).append(row)
        for (label, series_id, x_id), selected_rows in sorted(
            selector_groups.items(),
            key=lambda item: tuple(
                value.encode("utf-8") for value in item[0]
            ),
        ):
            graph_variants = sorted(
                {str(row["graph_variant"]) for row in selected_rows},
                key=lambda value: value.encode("utf-8"),
            )
            cell_ids = sorted(
                {str(row["cell_id"]) for row in selected_rows},
                key=lambda value: value.encode("utf-8"),
            )
            system_ids = sorted(
                {str(row["system_id"]) for row in selected_rows},
                key=lambda value: value.encode("utf-8"),
            )
            filters = {
                "diagnostic_label": label,
                "graph_variant": graph_variants,
                "cell_id": cell_ids,
                "system_id": system_ids,
            }
            suffix = canonical_sha256(
                {
                    "diagnostic_selector": {
                        "label": label,
                        "series": series_id,
                        "x": x_id,
                        "filters": filters,
                    }
                }
            )[:16]
            plots.append(make_plot_spec(
                plot_id=f"diagnostic_selector_{suffix}",
                title=f"{label} · {series_id} · x={x_id}",
                kind="diagnostic_summary",
                source_table="graph_diagnostics",
                rows=diagnostics,
                filters=filters,
                encodings={
                    "x": "cell_id",
                    "y": "system_id",
                    "fill": "effect_or_value",
                },
                legend=(
                    "This CSV is the exact signed row set for the currently selectable diagnostic key.",
                    "Color semantics, unit, null/control and label stage are printed beside the matrix.",
                    "Gray means a named missing or non-OK status, never zero.",
                ),
                caption="One bounded diagnostic selector key across its exact registered cells and systems.",
                better_direction="context_dependent",
                ci_definition="No reporting-layer interval is added; producer-supplied null and p-value fields remain in the exact CSV.",
                selection_rule="Exact readable diagnostic label + registered series + registered x coordinate; no performance-based selection.",
            ))
        mechanism_specs = (
            (
                _MECHANISM_PLOT_IDS["continuous_lsml"],
                "continuous_lsml",
                [
                    "Continuous L-SML cluster boundaries — cluster id",
                    "Continuous L-SML feature-correlation clusters — feature correlation",
                ],
                "Do Continuous L-SML clusters separate feature-correlation blocks?",
                (
                    "Blue and orange identify within-cluster and cross-cluster feature pairs.",
                    "The plot uses absolute Pearson correlation and excludes diagonal self-correlations.",
                    "Every registered cell is shown; no correctness label or performance value selects a cell.",
                ),
                "Signed correlation and cluster-assignment rows for every Continuous L-SML cell; the report derives only the within/cross means shown.",
            ),
            (
                _MECHANISM_PLOT_IDS["family_nrm_a"],
                "family_nrm_a",
                [
                    "Family-NRM residual covariance — residual covariance",
                    "Family-NRM residual eigenspectrum — residual eigenvalue",
                ],
                "What residual structure does Family-NRM-A recover?",
                (
                    "Covariance color is centered at zero: blue negative, white zero, red positive.",
                    "Thin eigenspectrum curves are cells; the thick curve is the across-cell median.",
                    "The rows are label-free and do not determine the sign of a correctness correction.",
                ),
                "All signed Family-NRM-A residual-covariance and eigenspectrum rows; no cell is selected.",
            ),
            (
                _MECHANISM_PLOT_IDS["su_pcr"],
                "su_pcr",
                [
                    "SU-PCR low-rank plus sparse decomposition — sparse support fraction off diagonal",
                    "SU-PCR low-rank eigenspectrum — low rank eigenvalue",
                ],
                "Where does SU-PCR recover sparse support or negative low-rank eigenvalues?",
                (
                    "Support fraction and eigenvalue use separate axes and units.",
                    "Zero support means no fitted off-diagonal sparse coefficients in that cell.",
                    "Negative eigenvalues describe the fitted decomposition and are not a causal performance diagnosis.",
                ),
                "All signed SU-PCR off-diagonal support-fraction and low-rank eigenspectrum rows; no cell is selected.",
            ),
            (
                _MECHANISM_PLOT_IDS["pgrd_a"],
                "pgrd_a",
                [
                    "PGRD residual-gradient decomposition — cross term at registered direction",
                    "PGRD residual-gradient decomposition — quadratic term at registered direction",
                    "PGRD residual-gradient decomposition — predicted energy change at unit step",
                ],
                "Does the registered PGRD-A direction lower its graph energy?",
                (
                    "Cross, quadratic, and total unit-step terms are shown separately for every cell.",
                    "Negative total change means graph-energy descent, not improved correctness detection.",
                    "All terms are label-free; AUROC is deliberately absent from this panel.",
                ),
                "All signed PGRD-A energy-decomposition rows for the three displayed terms; no cell is selected.",
            ),
        )
        available_by_method: dict[str, set[str]] = {}
        for row in diagnostics:
            available_by_method.setdefault(str(row["method_id"]), set()).add(
                str(row["diagnostic_label"])
            )
        for plot_id, method_id, labels, plot_title, legend, selection_rule in mechanism_specs:
            if not set(labels).issubset(available_by_method.get(method_id, set())):
                continue
            plots.append(make_plot_spec(
                plot_id=plot_id,
                title=plot_title,
                kind="diagnostic_summary",
                source_table="graph_diagnostics",
                rows=diagnostics,
                filters={"method_id": method_id, "diagnostic_label": labels},
                encodings={
                    "cell": "cell_id",
                    "series": "graph_variant.series",
                    "x": "graph_variant.x",
                    "value": "value",
                },
                legend=legend,
                caption="Method-specific label-free mechanism panel derived directly from the signed diagnostic rows.",
                better_direction="context_dependent",
                ci_definition="No reporting-layer confidence interval; the panel displays every registered cell and its signed diagnostic values.",
                selection_rule=selection_rule,
            ))
        assumption_specs = (
            (
                _ASSUMPTION_SUMMARY_PLOT_IDS["random_family_graph_control"],
                {"random_family_graph_control"},
                "How do target-blind random-family graphs compare with fitted graphs?",
                "Within each cell, registered target-blind family draws are summarized before the across-cell median and full range are displayed.",
            ),
            (
                _ASSUMPTION_SUMMARY_PLOT_IDS["ca_alpha_controls"],
                {"ca_alpha_controls"},
                "Does CA-SpecRaGE differ from its registered weight controls?",
                "All learned-versus-equal/prior/global-mean/permuted roughness contrasts are retained; no control is selected by its outcome.",
            ),
            (
                _ASSUMPTION_SUMMARY_PLOT_IDS["ca_view_weights"],
                {"ca_view_weights"},
                "Which feature views does CA-SpecRaGE reweight?",
                "All learned-alpha and frozen-prior rows for all features and cells are retained; the report derives only their paired difference.",
            ),
            (
                _ASSUMPTION_SUMMARY_PLOT_IDS["dufs_gate_weights"],
                {"dufs_gate_weights", "dufs_gate_weights_per_seed"},
                "Which features does DUFS retain, and is that choice stable across seeds?",
                "All signed RMS gate-weight and per-seed survival rows are retained for every feature and cell.",
            ),
            (
                _ASSUMPTION_SUMMARY_PLOT_IDS[
                    "fixed_graph_group_bootstrap_stability"
                ],
                {"fixed_graph_group_bootstrap_stability"},
                "How sensitive is a fixed fitted graph to source-group reweighting?",
                "All registered source-group bootstrap rows are retained. The graph is reweighted, not refit.",
            ),
            (
                _ASSUMPTION_SUMMARY_PLOT_IDS[
                    "family_nrm_family_contributions"
                ],
                {"family_nrm_family_contributions"},
                "Which measurement families contribute to Family-NRM-A?",
                "All six-family contribution rows are retained for every cell; the report displays signed coefficients and absolute direction shares.",
            ),
            (
                _ASSUMPTION_SUMMARY_PLOT_IDS[
                    "su_pcr_sparse_support_stability"
                ],
                {"su_pcr_sparse_support_stability"},
                "Is the SU-PCR sparse support stable under source-group bootstrap?",
                "All registered support-refit bootstrap rows are retained and collapsed within cell only for the visible summary.",
            ),
        )
        for plot_id, panel_ids, plot_title, selection_rule in assumption_specs:
            labels = sorted(
                {
                    str(row["diagnostic_label"])
                    for row in diagnostics
                    if _variant_fields(row).get("panel") in panel_ids
                },
                key=lambda value: value.encode("utf-8"),
            )
            if not labels:
                continue
            plots.append(make_plot_spec(
                plot_id=plot_id,
                title=plot_title,
                kind="diagnostic_summary",
                source_table="graph_diagnostics",
                rows=diagnostics,
                filters={"diagnostic_label": labels},
                encodings={
                    "cell": "cell_id",
                    "series": "graph_variant.series",
                    "draw": "notes.draw_index",
                    "value": "value",
                    "effect": "effect",
                },
                legend=(
                    "Dots are across-cell medians after any registered repeated draws are collapsed within cell.",
                    "Horizontal ranges retain the minimum and maximum cell summaries; no cell is selected by performance.",
                    "The exact signed source CSV contains every input row and its provenance hash.",
                ),
                caption="Bounded mechanism/control summary derived from all matching signed diagnostic rows.",
                better_direction="context_dependent",
                ci_definition="Descriptive median and full cell range only; no reporting-layer confidence interval is added.",
                selection_rule=selection_rule,
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
            association_labels = sorted({
                str(row["diagnostic_label"])
                for row in diagnostics
                if _variant_fields(row).get("panel")
                == "alignment_vs_improvement_summary"
            })
            plots.append(make_plot_spec(
                plot_id="graph_alignment_vs_auroc_delta",
                title="Does stronger target alignment predict a gain over IU-PCR?",
                kind="diagnostic_scatter",
                source_table="graph_diagnostics",
                rows=diagnostics,
                filters={
                    "diagnostic_label": [alignment_label, *association_labels]
                },
                encodings={"x": "notes.source_x_value", "y": "value", "color": "method_id", "label": "cell_id"},
                legend=(
                    "Horizontal axis = signed null-minus-real target-alignment effect copied from the verified diagnostic source.",
                    "Vertical axis = paired cell AUROC delta versus IU-PCR copied from the frozen evaluator.",
                    "Color identifies method; signed aggregate Spearman and Pearson rows are printed below the scatter rather than plotted as points.",
                ),
                caption="This is a descriptive across-cell relationship after score freeze. The aggregate coefficients have no reporting-layer p-value or interval and are not a causal or independent validation claim.",
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
.legend-gradient.diverging{background:linear-gradient(90deg,rgb(45,105,190),rgb(244,244,244),rgb(196,59,59))}.association-summary{display:flex;flex-wrap:wrap;gap:8px 16px;margin:8px 0;padding:10px 12px;border:1px solid var(--line);border-radius:9px;background:color-mix(in srgb,var(--accent),transparent 95%);font-size:12px}.association-stat{font-variant-numeric:tabular-nums}.diagnostic-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px;margin:9px 0}.diagnostic-meta>div{padding:9px 10px;border:1px solid var(--line);border-radius:9px;background:color-mix(in srgb,var(--accent),transparent 96%);font-size:12px}.diagnostic-meta b{display:block;margin-bottom:2px}.diagnostic-scale{display:flex;align-items:center;gap:8px;margin-top:5px}.diagnostic-colorbar{display:inline-block;width:110px;height:11px;border:1px solid var(--line);border-radius:4px}.diagnostic-scale-values{font-variant-numeric:tabular-nums;color:var(--muted)}.mechanism-panels{display:grid;gap:16px}.mechanism-card{margin-top:12px}.mechanism-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.mechanism-grid h4{margin:2px 0 7px}.mechanism-card .plot-shell svg{min-width:460px}@media(max-width:980px){.mechanism-grid{grid-template-columns:1fr}}
.published-context-shell{margin:14px 0}.published-context-legend{display:flex;align-items:center;gap:9px;padding:10px 12px;border:1px dashed #8a929d;border-radius:10px;color:var(--muted);background:color-mix(in srgb,#8a929d,transparent 94%)}.published-context-marker{display:inline-block;width:13px;height:13px;border:2px solid #8a929d;border-radius:50%;background:transparent;flex:0 0 auto}.published-context-card{margin-top:10px;border:2px solid #8a929d;border-radius:13px;padding:16px;background:color-mix(in srgb,#8a929d,transparent 96%);box-shadow:none}.published-context-card[hidden]{display:none}.published-context-card header{display:flex;align-items:flex-start;gap:10px}.published-context-card h3{margin:0;font-size:19px}.context-kicker{margin:0;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.07em;font-weight:800}.context-badge{margin-left:auto;border:1px solid #8a929d;border-radius:999px;padding:3px 8px;font-size:10px;color:var(--muted);font-weight:800}.context-badge.related{margin-left:0;color:var(--warn);border-color:var(--warn)}.context-caveat,.context-none{margin:10px 0;color:var(--muted)}.context-meta{display:grid;grid-template-columns:max-content 1fr;gap:4px 12px;padding:10px 12px;border:1px solid var(--line);border-radius:9px;background:var(--panel)}.context-meta dt{font-weight:800}.context-meta dd{margin:0;white-space:normal}.context-grid{display:grid;grid-template-columns:minmax(280px,.8fr) minmax(300px,1.2fr);gap:14px;margin-top:10px}.context-grid h4{margin:0 0 6px}.context-grid ul{margin:0;padding-left:20px}.context-axis-table{max-height:none}.context-axis-table th{position:static}.axis-status{font-size:10px;font-weight:800}.axis-EXACT{color:var(--ok)}.axis-DIFFERENT,.axis-UNKNOWN{color:var(--warn)}.preset-control{max-width:520px;margin:0 0 8px}.metric-heatmap-scale{padding:7px 0;align-items:center}.metric-heatmap-scale strong{font-size:11px}.figure-source{display:block;margin-top:5px;overflow-wrap:anywhere}.figure-source code{font-size:10px}.figure-source a[aria-disabled="true"]{color:var(--muted);pointer-events:none}@media(max-width:800px){.context-grid{grid-template-columns:1fr}.published-context-card header{flex-wrap:wrap}.context-badge{margin-left:0}}
"""


_JS = r"""
const parseData=id=>JSON.parse(document.getElementById(id).textContent);
const DATA={metrics:parseData('metrics-data'),contrasts:parseData('contrasts-data'),coverage:parseData('coverage-data'),diagnostics:parseData('diagnostics-data')};
const REG=parseData('registry-data'), PLOTS=parseData('plot-manifest-data'), PUBLISHED_CONTEXT=parseData('published-context-data');
const METHODS=Object.fromEntries(REG.methods.map(x=>[x.method_id,x]));
const SYSTEMS=Object.fromEntries(REG.systems.map(x=>[x.system_id,x]));
const CELLS=Object.fromEntries(REG.cells.map(x=>[x.cell_id,x]));
const CONTEXT=new Set(['CONTEXT_ONLY','UNVERIFIED']);
const OK=new Set(['OK','OK_FALLBACK']);
const controls=['task_id','dataset_id','generation_model_id','scorer_model_id','cell_id','slice_id','metric_id','comparison_group_id','method_id'];
const el=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x===null||x===undefined||Number.isNaN(Number(x))?'—':Number(x).toFixed(4);
const ROW_IDENTITY=['release_id','run_id','lane_id','task_id','dataset_id','population_id','cell_id','slice_id','cohort_id','comparison_group_id','feature_contract_id','access_contract_id','evaluator_id','evidence_grade','method_id','method_version_id','adapter_id','system_id','aggregation_id','aggregation_level','metric_id','fidelity'];
const rowKey=r=>ROW_IDENTITY.map(field=>String(r[field]??'')).join('␟');
function fieldValue(row,field){if(field==='generation_model_id'||field==='scorer_model_id')return CELLS[row.cell_id]?.[field]??'';return row[field]}
function unique(rows,field){return [...new Set(rows.map(r=>fieldValue(r,field)).filter(x=>x!==null&&x!==undefined&&x!==''))].sort((a,b)=>String(a).localeCompare(String(b)))}
function selected(field){return el('filter-'+field).value}
function baseFilter(row,skip=''){for(const f of controls){if(f===skip)continue;const v=selected(f);if(v&&fieldValue(row,f)!==v)return false}return true}
function optionLabel(field,value){if(field==='method_id')return METHODS[value]?.display_name||value;return value}
function refreshOptions(){
  for(const field of controls){const select=el('filter-'+field), old=select.value;const vals=unique(DATA.metrics.filter(r=>baseFilter(r,field)),field);select.innerHTML='<option value="">All</option>'+vals.map(v=>`<option value="${esc(v)}">${esc(optionLabel(field,v))}</option>`).join('');if(vals.includes(old))select.value=old}
}
function filtered(table){return DATA[table].filter(r=>baseFilter(r))}
function headlineViews(){
 const candidates=REG.aggregations.filter(a=>a.rule==='equal_unit_mean'&&a.unit_field==='cell_id'&&Array.isArray(a.component_ids)&&a.component_ids.length===24&&String(a.display_name||'').startsWith('Equal-cell macro24:'));
 if(candidates.length!==1)return null;const macroAggregation=candidates[0],frozenCells=new Set(macroAggregation.component_ids);
 const macros=DATA.metrics.filter(r=>r.metric_id==='auroc'&&r.aggregation_id===macroAggregation.aggregation_id);
 if(!macros.length)return null;const groupIds=unique(macros,'comparison_group_id'),macroSystems=unique(macros,'system_id');if(groupIds.length!==1||macroSystems.length!==REG.systems.length)return null;
 const anchor=macros[0],cells=DATA.metrics.filter(r=>r.aggregation_level==='cell'&&r.metric_id==='auroc'&&frozenCells.has(r.cell_id)&&r.task_id===anchor.task_id&&r.feature_contract_id===anchor.feature_contract_id&&r.access_contract_id===anchor.access_contract_id&&r.evaluator_id===anchor.evaluator_id);
 if(unique(cells,'cell_id').length!==24||cells.length!==24*macroSystems.length)return null;const expectedSystems=new Set(macroSystems);for(const cellId of frozenCells){const systems=new Set(cells.filter(r=>r.cell_id===cellId).map(r=>r.system_id));if(systems.size!==expectedSystems.size||[...expectedSystems].some(system=>!systems.has(system)))return null}
 const iuSystem=macros.find(r=>r.method_id==='iu_pcr')?.system_id;if(!iuSystem)return null;
 const contrasts=DATA.contrasts.filter(r=>r.metric_id==='auroc'&&r.aggregation_id===macroAggregation.aggregation_id&&r.comparison_group_id===anchor.comparison_group_id&&r.right_system_id===iuSystem);
 if(contrasts.length!==macroSystems.length-1)return null;return {forest:macros,heatmap:cells,contrasts};
}
function plotFilterMatches(row,filters){return Object.entries(filters||{}).every(([field,expected])=>Array.isArray(expected)?expected.includes(row[field]):row[field]===expected)}
function exactPlotContract(kind,rows){return PLOTS.plots.find(plot=>plot.kind===kind&&plot.n_source_rows===rows.length&&rows.every(row=>plotFilterMatches(row,plot.filters)))}
function setPlotSource(prefix,plot){const link=el(prefix+'-source-link'),hash=el(prefix+'-source-hash');if(!plot){link.removeAttribute('href');link.setAttribute('aria-disabled','true');link.textContent='No exact source contract for this custom view';hash.textContent='';return}link.href=`plot_data/${encodeURIComponent(plot.plot_id)}.csv`;link.removeAttribute('aria-disabled');link.textContent='Download exact source CSV';hash.textContent=`data_sha256=${plot.data_sha256}`}
function updateVisiblePlotSources(forestRows,heatRows,contrastRows,diagnosticRows){setPlotSource('forest',exactPlotContract('forest',forestRows));setPlotSource('heatmap',exactPlotContract('faceted_heatmap',heatRows));setPlotSource('contrast',exactPlotContract('contrast_forest',contrastRows));setPlotSource('diagnostic',exactPlotContract('diagnostic_summary',diagnosticRows))}
function renderPublishedContext(){
 const chosen=selected('cell_id'),cards=[...document.querySelectorAll('.published-context-card')];let shown=0;
 for(const card of cards){const visible=Boolean(chosen)&&card.dataset.cellId===chosen;card.hidden=!visible;if(visible)shown+=1}
 const note=el('published-context-note');
 if(!PUBLISHED_CONTEXT){note.textContent='No signed published-comparator context artifact was supplied for this generic report.';return}
 note.textContent=shown?'':chosen?'This aggregate or non-frozen cell has no published-context card. Select one of the 24 frozen cells.':'Select one frozen cell to show its separate published-comparator context card.';
}
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
function clearHeatmapScale(){el('heatmap-scale-values').textContent='No compatible numeric view';el('heatmap-direction').textContent='—';el('heatmap-colorbar').style.background='var(--line)'}
function heatmap(rows,target='heatmap-plot'){
 const shell=clearPlot(target), good=rows.filter(r=>r.aggregation_level==='cell');if(!compatibleFacetedCells(good,target)){clearHeatmapScale();return}const cells=unique(good,'cell_id'),systems=unique(good,'system_id');if(!cells.length||!systems.length){clearHeatmapScale();return}
 const vals=good.filter(r=>r.value!==null&&OK.has(r.status)).map(r=>Number(r.value));if(!vals.length){clearPlot(target,'No numeric OK rows exist for this heatmap.');clearHeatmapScale();return}const lo=Math.min(...vals),hi=Math.max(...vals),direction=unique(good,'better_direction')[0];el('heatmap-colorbar').style.background='linear-gradient(90deg,hsl(220 62% 72%),hsl(30 62% 44%))';el('heatmap-scale-values').textContent=`${fmt(lo)} → ${fmt(hi)}`;el('heatmap-direction').textContent=direction==='higher'?'Higher is better':direction==='lower'?'Lower is better':'Direction is context-dependent';const cw=36,rh=24,left=205,top=110,w=left+cells.length*cw+20,h=top+systems.length*rh+25,ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('width','100%');svg.setAttribute('height',h);shell.appendChild(svg);const by=new Map(good.map(r=>[`${r.system_id}::${r.cell_id}`,r]));
 cells.forEach((c,i)=>{const t=document.createElementNS(ns,'text');t.setAttribute('x',left+i*cw+cw*.6);t.setAttribute('y',top-6);t.setAttribute('transform',`rotate(-55 ${left+i*cw+cw*.6} ${top-6})`);t.setAttribute('font-size','9');t.setAttribute('fill','currentColor');t.textContent=c;svg.appendChild(t)});
 systems.forEach((s,j)=>{const t=document.createElementNS(ns,'text');t.setAttribute('x',left-7);t.setAttribute('y',top+j*rh+16);t.setAttribute('text-anchor','end');t.setAttribute('font-size','10');t.setAttribute('fill','currentColor');t.textContent=SYSTEMS[s]?.display_name||s;svg.appendChild(t);cells.forEach((c,i)=>{const r=by.get(`${s}::${c}`),rect=document.createElementNS(ns,'rect');rect.setAttribute('x',left+i*cw);rect.setAttribute('y',top+j*rh);rect.setAttribute('width',cw-1);rect.setAttribute('height',rh-1);if(!r||r.value===null||!OK.has(r.status)){rect.setAttribute('fill','var(--line)');rect.setAttribute('opacity','.45')}else{const q=hi===lo?.5:(Number(r.value)-lo)/(hi-lo);rect.setAttribute('fill',`hsl(${220-190*q} 62% ${72-28*q}%)`)}const title=document.createElementNS(ns,'title');title.textContent=r?`${c} · ${s}: ${fmt(r.value)} · ${r.status}`:`${c} · ${s}: no registered row`;rect.appendChild(title);svg.appendChild(rect)})})
}
function renderTable(rows){const rankedMap=new Map(Object.values(metricGroups(rows)).flatMap(rankGroup).map(r=>[rowKey(r),r]));const body=rows.slice().sort((a,b)=>a.comparison_group_id.localeCompare(b.comparison_group_id)||(rankedMap.get(rowKey(a))?.point_rank??999)-(rankedMap.get(rowKey(b))?.point_rank??999)||rowKey(a).localeCompare(rowKey(b))).map(r=>{const rr=rankedMap.get(rowKey(r));return `<tr id="row-${esc(rowKey(r))}" data-status="${esc(r.status)}"><td>${esc(SYSTEMS[r.system_id]?.display_name||r.system_id)}</td><td>${fmt(r.value)}</td><td>[${fmt(r.ci_low)}, ${fmt(r.ci_high)}]</td><td>${rr?.point_rank??'—'}</td><td>${rr?.uncertainty_tie?'yes':'no'}</td><td class="status ${esc(r.status)}">${esc(r.status)}</td><td>${esc(r.n_rows)}</td><td>${esc(r.n_groups)}</td><td>${esc(r.evidence_grade)}</td><td>${esc(r.fidelity)}</td><td>${esc(r.access_contract_id)}</td><td>${esc(r.comparison_group_id)}</td><td class="row-link">${esc(rowKey(r))}</td></tr>`}).join('');el('results-body').innerHTML=body||'<tr><td colspan="13" class="empty">No rows match these filters.</td></tr>'}
function renderDiagnostics(rows){const body=rows.map(r=>`<tr data-status="${esc(r.status)}"><td>${esc(SYSTEMS[r.system_id]?.display_name||r.system_id)}</td><td>${esc(r.cell_id)}</td><td>${esc(r.diagnostic_label)}</td><td>${esc(String(r.diagnostic_unit||'').replaceAll('_',' '))}</td><td>${fmt(r.value)}</td><td>${fmt(r.null_value)}</td><td>${fmt(r.effect)}</td><td>${fmt(r.p_value)}</td><td>${esc(r.label_stage)}</td><td>${esc(r.status)}</td></tr>`).join('');el('diagnostics-body').innerHTML=body||'<tr><td colspan="10" class="empty">No graph diagnostics match these filters.</td></tr>'}
function diagnosticKey(r){const series=(r.graph_variant.match(/;series=([^;]*)/)||[])[1]||'observed',x=(r.graph_variant.match(/;x=([^;]*)/)||[])[1]||'0';return `${r.diagnostic_label} · ${series} · x=${x}`}
function selectedDiagnostics(rows){const select=el('diagnostic-selector'),keys=[...new Set(rows.map(diagnosticKey))].sort();const previous=select.value;select.innerHTML=keys.map(k=>`<option value="${esc(k)}">${esc(k)}</option>`).join('');const preferred=keys.find(k=>k.startsWith('Correctness and trace-length smoothness — error label roughness'))||keys.find(k=>k.startsWith('Graph health — normalized spectral gap'))||keys[0]||'';select.value=keys.includes(previous)?previous:preferred;return rows.filter(r=>diagnosticKey(r)===select.value)}
function variantValue(r,key){return (r.graph_variant.match(new RegExp(`(?:^|;)${key}=([^;]*)`))||[])[1]||''}
function diagnosticSemantics(rows){
 const row=rows[0]||{},label=String(row.diagnostic_label||''),lower=label.toLowerCase(),units=unique(rows,'diagnostic_unit').map(value=>String(value).replaceAll('_',' '));
 const hasReference=rows.some(r=>r.null_value!==null&&r.effect!==null&&Math.abs(Number(r.effect)-Number(r.value))>1e-12),ratioToNull=lower.includes('ratio to null median');
 const panel=variantValue(row,'panel'),nullId=variantValue(row,'null'),permutations=Math.max(0,...rows.map(r=>Number(r.permutation_count||0))),hasP=rows.some(r=>r.p_value!==null);
 let field=hasReference?'effect':'value',scale='sequential',center=null,direction='Descriptive only; there is no universal better direction.';
 if(hasReference){scale='diverging';center=0;direction='Positive color means null/control minus observed is positive; interpret that difference using the named metric.'}
 else if(ratioToNull){scale='diverging';center=1;direction='Below 1 means the signal is smoother on the fitted graph than the descriptive node-permutation median; above 1 means rougher.'}
 else if(/correlation|coefficient|eigenvalue|directional derivative|cross term|energy change/.test(lower)){scale='diverging';center=0;direction='Sign and magnitude are descriptive; zero is the neutral reference.'}
 if(panel==='fixed_graph_group_bootstrap_stability')direction='This measures sensitivity after source-group resampling reweights the already fitted graph. Edges are not refit, so even a high similarity is not graph-learning/refit stability.';
 else if(lower.includes('normalized spectral gap'))direction='Larger gap indicates stronger connectivity/mixing of the frozen graph; it does not imply correctness alignment.';
 else if(lower.includes('n components')||lower.includes('isolated nodes'))direction='Lower is healthier connectivity; one component and zero isolated nodes are the graph-health targets.';
 else if(lower.includes('degree cv'))direction='Lower means more even weighted degree; this is graph health, not predictive quality.';
 else if(lower.includes('edge jaccard')||lower.includes('frobenius cosine')||lower.includes('seed cosine')||lower.includes('seed spearman'))direction='Higher means more similar/stable under the named comparison; it does not imply better correctness detection.';
 else if(lower.includes('relative difference'))direction='Lower means more similar operators under the named comparison.';
 else if(lower.includes('roughness')&&hasReference)direction='Positive null − observed means the signal is smoother on the fitted graph than on the reference; this is not a correctness-performance claim.';
 else if(lower.includes('published cell auroc delta'))direction='Positive means higher AUROC than IU-PCR in that frozen cell; this is a post-freeze performance value.';
 else if(panel==='pgrd_cross_gradient'&&(lower.includes('cross term')||lower.includes('directional derivative')||lower.includes('energy change')))direction='Negative means descent of the registered graph energy; it does not imply better correctness detection.';
 let nullText='No null or control is defined for this diagnostic.';
 if(ratioToNull) nullText='The displayed ratio already uses a descriptive fixed-signal node-permutation median. Raw null draws and an inferential p-value are not supplied in this reporting row.';
 else if(hasReference&&permutations>1&&!hasP) nullText=`null_value is a descriptive reference from ${permutations} producer permutations/resamples; no inferential p-value was supplied.`;
 else if(hasReference&&hasP) nullText=`null_value is the registered reference (${permutations||'recorded'} permutations); the producer-supplied p-value is shown in the table.`;
 else if(hasReference) nullText=`null_value is the registered ${nullId||'control'} reference; the difference is descriptive and has no producer-supplied p-value.`;
 else if(panel==='fixed_graph_group_bootstrap_stability') nullText='No null test: each row is a source-group resample that changes node/row weights on a fixed fitted graph; it does not refit edges.';
 const stages=unique(rows,'label_stage').map(value=>value==='label_free'?'label-free (no correctness labels opened)':value==='post_freeze_labels'?'post-freeze explanation (labels opened only after scores/graphs were frozen)':String(value));
 return {label,unit:units.join(' / ')||'unit not registered',field,scale,center,direction,nullText,stage:stages.join(' / ')}
}
function diagnosticScale(rows,sem){const values=rows.map(r=>Number(r[sem.field])).filter(Number.isFinite);if(!values.length)return {lo:0,hi:1,center:null,kind:'sequential'};if(sem.scale==='diverging'){const center=Number(sem.center),bound=Math.max(...values.map(v=>Math.abs(v-center)),1e-12);return {lo:center-bound,hi:center+bound,center,kind:'diverging'}}let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){const pad=Math.abs(lo)*.05||.01;lo-=pad;hi+=pad}return {lo,hi,center:null,kind:'sequential'}}
function diagnosticColor(value,scale){if(scale.kind==='diverging'){const q=Math.max(-1,Math.min(1,(value-scale.center)/(scale.hi-scale.center)));return q<0?`hsl(${215-5*(q+1)} 58% ${43+47*(q+1)}%)`:`hsl(${10-2*q} 58% ${90-47*q}%)`}const q=Math.max(0,Math.min(1,(value-scale.lo)/(scale.hi-scale.lo)));return `hsl(${215-185*q} 62% ${86-42*q}%)`}
function updateDiagnosticMeta(sem,scale){el('diagnostic-unit').textContent=`${sem.unit}; color uses ${sem.field==='effect'?'null/control − observed effect':'observed value'}`;el('diagnostic-direction').textContent=sem.direction;el('diagnostic-null').textContent=sem.nullText;el('diagnostic-stage').textContent=sem.stage;const bar=el('diagnostic-colorbar');bar.style.background=scale.kind==='diverging'?'linear-gradient(90deg,hsl(215 58% 43%),hsl(0 0% 90%),hsl(8 58% 43%))':'linear-gradient(90deg,hsl(215 62% 86%),hsl(30 62% 44%))';el('diagnostic-scale-values').textContent=scale.center===null?`${fmt(scale.lo)} → ${fmt(scale.hi)}`:`${fmt(scale.lo)} · neutral ${fmt(scale.center)} · ${fmt(scale.hi)}`}
function diagnosticHeatmap(rows,target='diagnostic-plot'){
 const shell=clearPlot(target),good=rows.filter(r=>r.effect!==null||r.value!==null);if(!good.length){clearPlot(target,'No numeric rows exist for this named diagnostic.');return}const sem=diagnosticSemantics(good),scale=diagnosticScale(good,sem);updateDiagnosticMeta(sem,scale);const columns=unique(good,'cell_id'),rowIds=unique(good,'system_id'),cw=44,rh=28,left=220,top=115,w=left+columns.length*cw+25,h=top+rowIds.length*rh+25,ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('width','100%');svg.setAttribute('height',h);shell.appendChild(svg);const by=new Map(good.map(r=>[`${r.system_id}::${r.cell_id}`,r]));columns.forEach((c,i)=>{const t=document.createElementNS(ns,'text');t.setAttribute('x',left+i*cw+cw*.55);t.setAttribute('y',top-7);t.setAttribute('transform',`rotate(-55 ${left+i*cw+cw*.55} ${top-7})`);t.setAttribute('font-size','9');t.setAttribute('fill','currentColor');t.textContent=c;svg.appendChild(t)});rowIds.forEach((s,j)=>{const t=document.createElementNS(ns,'text');t.setAttribute('x',left-7);t.setAttribute('y',top+j*rh+18);t.setAttribute('text-anchor','end');t.setAttribute('font-size','10');t.setAttribute('fill','currentColor');t.textContent=SYSTEMS[s]?.display_name||s;svg.appendChild(t);columns.forEach((c,i)=>{const r=by.get(`${s}::${c}`),rect=document.createElementNS(ns,'rect');rect.setAttribute('x',left+i*cw);rect.setAttribute('y',top+j*rh);rect.setAttribute('width',cw-1);rect.setAttribute('height',rh-1);if(!r||!OK.has(r.status)){rect.setAttribute('fill','var(--line)');rect.setAttribute('opacity','.45')}else{const v=Number(r[sem.field]);rect.setAttribute('fill',diagnosticColor(v,scale))}const title=document.createElementNS(ns,'title');title.textContent=r?`${r.diagnostic_label}: ${sem.field} ${fmt(r[sem.field])}, observed ${fmt(r.value)}, null ${fmt(r.null_value)} · ${r.diagnostic_unit} · ${r.label_stage} · ${r.status}`:`${c} · ${s}: no registered row`;rect.appendChild(title);svg.appendChild(rect)})})
}
function downloadFiltered(){const rows=filtered('metrics'),fields=Object.keys(DATA.metrics[0]||{});const quote=v=>`"${String(v??'').replaceAll('"','""')}"`;const csv=[fields.map(quote).join(','),...rows.map(r=>fields.map(f=>quote(Array.isArray(r[f])?JSON.stringify(r[f]):r[f])).join(','))].join('\n')+'\n';const blob=new Blob([csv],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='filtered_metrics_long.csv';a.click();URL.revokeObjectURL(a.href)}
function render(){
 refreshOptions();renderPublishedContext();const headline=el('view-preset').value==='headline_24cell_auroc'?headlineViews():null;
 const rows=headline?.forest||filtered('metrics'),heatRows=headline?.heatmap||rows,contrastRows=headline?.contrasts||filtered('contrasts');
 el('preset-note').textContent=headline?'Headline preset: Macro-24 AUROC forest, all 24 cell AUROC tiles, and registered Macro-24 paired contrasts versus IU-PCR. Filters below apply after switching to Custom explorer.':el('view-preset').value==='headline_24cell_auroc'?'The headline preset is unavailable because this generic report does not contain one complete frozen 24-cell AUROC contract; showing the custom explorer instead.':'Custom explorer: every plot follows the exact filters below.';
 renderSummary(rows);forest(rows);heatmap(heatRows);renderTable(rows);const diagnostics=selectedDiagnostics(filtered('diagnostics'));renderDiagnostics(diagnostics);diagnosticHeatmap(diagnostics);forest(contrastRows,'contrast-plot','delta','ci_low','ci_high','left_system_id');updateVisiblePlotSources(rows,heatRows,contrastRows,diagnostics);
}
for(const f of controls)el('filter-'+f).addEventListener('change',()=>{el('view-preset').value='custom';render()});el('view-preset').addEventListener('change',render);el('diagnostic-selector').addEventListener('change',render);el('download').addEventListener('click',downloadFiltered);render();
"""


def render_report(
    *,
    registry: Mapping[str, Any],
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
    plot_manifest: Optional[Mapping[str, Any]] = None,
    title: str = "Reconstruction benchmark explorer",
    published_context: Mapping[str, Any] | None = None,
) -> str:
    registry, rows = _validate_inputs(registry, rows_by_table)
    cell_ids = sorted(
        {
            str(row["cell_id"])
            for row in rows["metrics"]
            if row.get("aggregation_level") == "cell"
        },
        key=lambda value: value.encode("utf-8"),
    )
    if published_context is not None:
        published_context = validate_published_context_projection(
            published_context,
            release_id=str(registry["release_id"]),
            expected_cell_ids=cell_ids,
            metrics_cell_auroc_logical_sha256=(
                frozen24_cell_auroc_logical_sha256(
                    rows["metrics"],
                    expected_cell_ids=cell_ids,
                    expected_method_ids=[
                        str(method["method_id"])
                        for method in registry["methods"]
                    ],
                )
            ),
        )
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
    mechanism_panel_html = _mechanism_panels(
        rows["graph_diagnostics"], plot_manifest
    )
    diagnostic_summary_panel_html = _diagnostic_summary_panels(
        rows["graph_diagnostics"], registry, plot_manifest
    )
    published_context_html = _published_context_cards(published_context)
    generated_id = canonical_sha256(
        {
            "schema": REPORT_SCHEMA,
            "registry_sha256": registry["registry_sha256"],
            "plot_manifest_sha256": plot_manifest["manifest_sha256"],
            "table_hashes": {
                table: canonical_sha256(value) for table, value in sorted(rows.items())
            },
            "published_context_sha256": (
                canonical_sha256(published_context)
                if published_context is not None
                else None
            ),
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
<label class="preset-control">Named view<select id="view-preset"><option value="headline_24cell_auroc">Headline: frozen 24-cell AUROC</option><option value="custom">Custom explorer</option></select></label>
<p class="section-intro" id="preset-note"></p>
<div class="controls">
  <label>Task<select id="filter-task_id"></select></label><label>Dataset<select id="filter-dataset_id"></select></label>
  <label>Generation model<select id="filter-generation_model_id"></select></label><label>Scorer model<select id="filter-scorer_model_id"></select></label>
  <label>Cell<select id="filter-cell_id"></select></label><label>Slice<select id="filter-slice_id"></select></label>
  <label>Metric<select id="filter-metric_id"></select></label><label>Exact comparison group<select id="filter-comparison_group_id"></select></label>
  <label>Method<select id="filter-method_id"></select></label><button id="download" type="button">Download filtered CSV</button>
</div>
<aside class="published-context-shell" id="published-context" aria-label="Published comparator context">
 <h3>Published comparator context for the selected cell</h3>
 <div class="published-context-legend"><i class="published-context-marker" aria-hidden="true"></i><span><strong>Hollow gray = published context only.</strong> It is displayed separately and is never ranked, plotted, or used to compute a paper-to-v2 delta.</span></div>
 <p class="section-intro" id="published-context-note"></p>
 <div id="published-context-cards">{published_context_html}</div>
</aside>
<div class="summary-grid" id="summary"></div>
<div class="plots">
 <figure class="figure-card"><h3 class="plot-title">Metric forest</h3><p class="plot-subtitle">Point estimate and registered interval for the current filters.</p><div class="plot-shell" id="forest-plot"></div>
 <div class="legend"><span><i class="legend-dot"></i>Color/marker = method card (one of the 13 benchmark methods)</span><span>Line = registered 95% interval</span><span>Published paper context is never added to this forest</span></div>
 <figcaption>Better direction, positive class, bootstrap unit, cohort, and access contract are stored per row. Compare only one exact group. <span class="figure-source"><a id="forest-source-link" download></a> <code id="forest-source-hash"></code></span></figcaption></figure>
 <figure class="figure-card"><h3 class="plot-title">Cell × method heatmap</h3><p class="plot-subtitle">Per-cell values; gray means an explicit missing/non-OK status.</p><div class="plot-shell" id="heatmap-plot"></div>
 <div class="diagnostic-scale metric-heatmap-scale"><i class="diagnostic-colorbar" id="heatmap-colorbar"></i><span class="diagnostic-scale-values" id="heatmap-scale-values"></span><strong id="heatmap-direction"></strong></div>
 <div class="legend"><span><i class="legend-dot"></i>Blue at the visible minimum → orange at the visible maximum</span><span>Gray = missing/non-OK, never zero</span></div>
 <figcaption>Visible numeric limits and direction define the color scale; hover adds the exact cell value. <span class="figure-source"><a id="heatmap-source-link" download></a> <code id="heatmap-source-hash"></code></span></figcaption></figure>
 <figure class="figure-card"><h3 class="plot-title">Paired contrasts</h3><p class="plot-subtitle">Candidate minus its registered matched control.</p><div class="plot-shell" id="contrast-plot"></div>
 <div class="legend"><span><i class="legend-dot"></i>Point = paired delta</span><span>Line = paired grouped-bootstrap 95% interval</span><span>Zero = no change</span></div>
 <figcaption>The right-system comparator, pairing unit, W/T/L counts, and draw count are in <code>contrasts_long</code>. <span class="figure-source"><a id="contrast-source-link" download></a> <code id="contrast-source-hash"></code></span></figcaption></figure>
</div>
<h3>Exact result rows</h3><div class="table-wrap"><table><thead><tr><th>Method/system</th><th>Value</th><th>95% interval</th><th>Point rank</th><th>Uncertainty set</th><th>Status</th><th>N rows</th><th>N groups</th><th>Evidence</th><th>Fidelity</th><th>Access contract</th><th>Comparison group</th><th>Row key</th></tr></thead><tbody id="results-body"></tbody></table></div>
</section>

<section class="section" id="graph-checks"><h2>Graph assumption checks</h2>
<p class="section-intro">These checks can support or refute a mechanism. Connectivity or stability shows that a graph operated; it does not prove that the graph follows correctness. <code>label_stage</code> separates label-free health checks from analyses opened only after scores and graph hashes were frozen. A node-permutation median is a descriptive reference unless the producer supplied an inferential p-value. Source-group bootstrap rows reweight one fixed fitted graph; they are weight-sensitivity checks, not graph-refit stability.</p>
<div class="warning">Node embeddings are illustrative only. Any example cell must be chosen by a preregistered label-free graph-health rule, never because it gives a favorable AUROC.</div>
<label class="diagnostic-controls">Named diagnostic<select id="diagnostic-selector"></select></label>
<figure class="figure-card"><h3 class="plot-title">Graph diagnostic cell × method matrix</h3><p class="plot-subtitle">One named and human-readable diagnostic at a time; no unique raw diagnostic IDs are used as axes.</p><div class="plot-shell" id="diagnostic-plot"></div>
<div class="diagnostic-meta" aria-live="polite"><div><b>Unit and color value</b><span id="diagnostic-unit">Choose a diagnostic.</span><div class="diagnostic-scale"><i class="diagnostic-colorbar" id="diagnostic-colorbar"></i><span class="diagnostic-scale-values" id="diagnostic-scale-values"></span></div></div><div><b>Direction</b><span id="diagnostic-direction">Choose a diagnostic.</span></div><div><b>Null/control semantics</b><span id="diagnostic-null">Choose a diagnostic.</span></div><div><b>Label stage</b><span id="diagnostic-stage">Choose a diagnostic.</span></div></div>
<div class="legend"><span>Color scale, neutral point, unit and direction update with the selected metric</span><span>Gray = missing or non-OK, never zero</span><span>Hover gives the exact observed, null and displayed color value</span></div>
<figcaption>Each tile retains its exact cell cohort and comparison group. Connectivity checks describe the frozen operator. The grouped sensitivity panel only reweights that fitted graph; it does not refit edges and therefore does not establish graph-learning stability. Target alignment is a separate, post-freeze check. <span class="figure-source"><a id="diagnostic-source-link" download></a> <code id="diagnostic-source-hash"></code></span></figcaption></figure>
<div class="table-wrap"><table><thead><tr><th>Method</th><th>Cell</th><th>Diagnostic</th><th>Unit</th><th>Value</th><th>Null</th><th>Effect</th><th>p</th><th>Label stage</th><th>Status</th></tr></thead><tbody id="diagnostics-body"></tbody></table></div>
<h3>Method-specific mechanism panels</h3><p class="section-intro">These panels make the main mathematical diagnostics visible without searching the selector. They use every available cell and no performance-selected example.</p>
{mechanism_panel_html}
<h3>Bounded control and stability summaries</h3><p class="section-intro">These panels keep the promised random-family, equal/prior/permuted, feature-weight, family-contribution and bootstrap checks visible without exposing thousands of raw draw keys in the selector. Each panel links the complete signed rows.</p>
{diagnostic_summary_panel_html}
<h3>Alignment and utility</h3>{alignment_scatter_html}
<h3>Label-free-selected example graphs</h3><p class="section-intro">For each graph method, both panels below use the exact same nodes, edges, and arbitrary two-dimensional spectral coordinates. Only the color changes; the visible nuisance scale gives its exact range.</p>
<div class="guide-grid-cards" id="graph-example-cards">{graph_example_html}</div>
</section>

<section class="section" id="provenance"><h2>Plot contracts and provenance</h2>
<p class="section-intro">Every plot has a visible legend and caption, an exact source-table hash, a declared selection rule, and a materialized CSV in <code>07_reports/plot_data/</code>.</p>
<div class="guide-grid-cards">{''.join(f'''<article class="guide-card plot-contract-card"><h3>{_e(plot['title'])}</h3><p>{_e(plot['caption'])}</p><ul>{''.join(f'<li>{_e(item)}</li>' for item in plot['legend'])}</ul><p class="manifest-meta">plot_id={_e(plot['plot_id'])}<br>source={_e(plot['source_table'])} · rows={_e(plot['n_source_rows'])}<br>data_sha256={_e(plot['data_sha256'])}<br>CI: {_e(plot['ci_definition'])}<br>Selection: {_e(plot['selection_rule'])}</p><p><a class="plot-data-link" href="plot_data/{_e(plot['plot_id'])}.csv" download>Download the exact CSV for this figure contract</a></p></article>''' for plot in plot_manifest['plots'])}</div>
</section>
</main>
<script type="application/json" id="registry-data">{_json_for_script(registry)}</script>
<script type="application/json" id="metrics-data">{_json_for_script(rows['metrics'])}</script>
<script type="application/json" id="contrasts-data">{_json_for_script(rows['contrasts'])}</script>
<script type="application/json" id="coverage-data">{_json_for_script(rows['coverage'])}</script>
<script type="application/json" id="diagnostics-data">{_json_for_script(_embedded_diagnostics(rows['graph_diagnostics']))}</script>
<script type="application/json" id="plot-manifest-data">{_json_for_script(plot_manifest)}</script>
<script type="application/json" id="published-context-data">{_json_for_script(published_context)}</script>
<script>{_JS}</script></body></html>
"""


def write_report(
    path: os.PathLike[str] | str,
    *,
    registry: Mapping[str, Any],
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
    plot_manifest: Optional[Mapping[str, Any]] = None,
    title: str = "Reconstruction benchmark explorer",
    published_context: Mapping[str, Any] | None = None,
    atomic: bool = True,
) -> dict[str, Any]:
    rendered = render_report(
        registry=registry,
        rows_by_table=rows_by_table,
        plot_manifest=plot_manifest,
        title=title,
        published_context=published_context,
    )
    target = Path(path)
    payload = rendered.encode("utf-8")
    _atomic_write_bytes(target, payload, atomic=atomic)
    return {
        "schema": REPORT_SCHEMA,
        "path": target.name,
        "size_bytes": len(payload),
        "sha256": canonical_sha256({"html_utf8": rendered}),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "self_contained": True,
    }
