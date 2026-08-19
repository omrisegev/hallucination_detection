#!/usr/bin/env python3
"""Render the frozen white-box layer-fusion benchmark as self-contained HTML.

The benchmark runner owns fitting and metric computation.  This module only
reads its CSV/JSON artifacts, renders audit plots, and records the hashes of
everything it consumed.  Promotion is intentionally fail-closed: the page is
VALIDATED only when ``validation_status.json`` explicitly passes both the
corrected Gate B and the architecture-fidelity pilot, and all three freeze
documents are present.

Usage::

    python scripts/whitebox_layer_fusion_report.py \
        --results-dir results/whitebox_layer_fusion_v2
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

VERSION = "whitebox-layer-fusion-report-v2"
DEFAULT_RESULTS = Path("results/whitebox_layer_fusion_v2")
SOURCE_FILES = (
    "per_cell_metrics.csv",
    "headline_summary.csv",
    "cohort_summary.csv",
    "comparator_fidelity.csv",
    "paired_comparisons.csv",
    "data_coverage.csv",
    "layer_diagnostics.csv",
    "dependence_diagnostics.csv",
    "weights_diagnostics.csv",
    "validation_status.json",
    "RUN_DEFINITION.json",
    "SOURCE_FREEZE_MANIFEST.json",
    "SCORE_FREEZE_MANIFEST.json",
)
FREEZE_FILES = (
    "RUN_DEFINITION.json",
    "SOURCE_FREEZE_MANIFEST.json",
    "SCORE_FREEZE_MANIFEST.json",
)
FIGURES = (
    "macro_forest.svg",
    "per_cell_heatmap.svg",
    "paired_deltas.svg",
    "layer_curves.svg",
    "layer_correlation_heatmap.svg",
    "dependence_diagnostics.svg",
    "weights_diagnostics.svg",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _first(row: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _number(row: Mapping[str, Any], *names: str) -> float:
    value = _first(row, *names, default="")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _integer(row: Mapping[str, Any], *names: str) -> int | None:
    value = _number(row, *names)
    return int(round(value)) if math.isfinite(value) else None


def _display_method(row: Mapping[str, Any]) -> str:
    parts = [str(_first(row, "method", "method_key", "method_name", default="method"))]
    for aliases in (
        ("feature_contract", "contract"),
        ("layer_subset", "subset"),
        ("structured", "structure"),
    ):
        value = str(_first(row, *aliases, default="")).strip()
        if value and value.lower() not in {"none", "false", "flat", "nan"}:
            parts.append(value)
    return " · ".join(dict.fromkeys(parts))


def _pretty(value: Any) -> str:
    return str(value).replace("_", " ").replace("__", " · ").strip().title()


def _fmt(value: Any, column: str = "") -> str:
    if value is None or str(value).strip() == "":
        return "—"
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return html.escape(text)
    if not math.isfinite(number):
        return "—"
    lower = column.lower()
    if any(token in lower for token in ("n_samples", "n_groups", "wins", "ties", "losses", "layer")):
        return f"{int(round(number)):,}"
    if "sha" in lower:
        return html.escape(text)
    return f"{number:.4f}"


def _normal_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _gate_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "passed", "true", "ok", "validated", "complete"}:
            return True
        if normalized in {"fail", "failed", "false", "blocked", "incomplete", "missing"}:
            return False
    if isinstance(value, Mapping):
        for key in ("pass", "passed", "ok", "value", "status", "result"):
            if key in value:
                parsed = _gate_value(value[key])
                if parsed is not None:
                    return parsed
    return None


def _walk_mapping(value: Any) -> Iterable[tuple[str, Any]]:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        yield str(key), child
        yield from _walk_mapping(child)


def _find_gate(payload: Mapping[str, Any], aliases: Sequence[str]) -> bool | None:
    normalized = {_normal_key(alias) for alias in aliases}
    for key, value in _walk_mapping(payload):
        if _normal_key(key) in normalized:
            parsed = _gate_value(value)
            if parsed is not None:
                return parsed
    return None


def derive_validation_status(
    payload: Mapping[str, Any], results_dir: Path
) -> dict[str, Any]:
    """Return the fail-closed promotion state and its audit rows."""
    gate_b = _find_gate(
        payload,
        (
            "corrected_layer_gate_b_all_pass",
            "corrected_layer_gate_b_pass",
            "corrected_layer_gate_b_status",
            "gate_b_all_pass",
            "gate_b_pass",
            "corrected_gate_b",
            "entropy_gate_b",
        ),
    )
    architecture = _find_gate(
        payload,
        (
            "architecture_pilot_pass",
            "architecture_pilot",
            "architecture_fidelity_pass",
            "architecture_fidelity",
            "architecture_recapture_pass",
            "fidelity_pilot_pass",
        ),
    )
    freeze_presence = {name: (results_dir / name).is_file() for name in FREEZE_FILES}
    score_freeze = _read_json(results_dir / "SCORE_FREEZE_MANIFEST.json")
    labels_seen = _find_gate(
        score_freeze,
        ("labels_seen_during_fit", "fit_labels_seen", "labels_seen"),
    )
    frozen_before_labels = _find_gate(
        score_freeze,
        (
            "scores_frozen_before_labels",
            "score_files_verified_before_labels",
            "score_freeze_before_labels",
            "frozen_before_labels",
        ),
    )
    rows = [
        {
            "gate": "Corrected live Gate B (all 14 cells)",
            "result": "PASS" if gate_b is True else "FAIL" if gate_b is False else "MISSING",
            "required": "yes",
        },
        {
            "gate": "Two-cell architecture-fidelity pilot",
            "result": "PASS" if architecture is True else "FAIL" if architecture is False else "MISSING",
            "required": "yes",
        },
        {
            "gate": "Leakage boundary: labels_seen_during_fit is false",
            "result": "PASS" if labels_seen is False else "FAIL" if labels_seen is True else "MISSING",
            "required": "yes",
        },
        {
            "gate": "Score hashes frozen before labels opened",
            "result": "PASS" if frozen_before_labels is True else "FAIL" if frozen_before_labels is False else "MISSING",
            "required": "yes",
        },
    ]
    for name, present in freeze_presence.items():
        rows.append(
            {
                "gate": f"Provenance artifact: {name}",
                "result": "PRESENT" if present else "MISSING",
                "required": "yes",
            }
        )
    validated = (
        gate_b is True
        and architecture is True
        and labels_seen is False
        and frozen_before_labels is True
        and all(freeze_presence.values())
    )
    blockers = [row["gate"] for row in rows if row["result"] not in {"PASS", "PRESENT"}]
    return {
        "validated": validated,
        "status": "VALIDATED" if validated else "PRELIMINARY / VALIDATION BLOCKED",
        "gate_b": gate_b,
        "architecture_pilot": architecture,
        "labels_seen_during_fit": labels_seen,
        "scores_frozen_before_labels": frozen_before_labels,
        "rows": rows,
        "blockers": blockers,
    }


def _svg_text(value: Any, limit: int = 52) -> str:
    text = str(value)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return html.escape(text)


def _svg_document(width: int, height: int, title: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">
<title id="title">{html.escape(title)}</title><style>
text{{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;fill:#18202a}}
.title{{font-size:18px;font-weight:700}}.label{{font-size:11px}}.small{{font-size:9px;fill:#637080}}
.axis{{stroke:#aeb6c2;stroke-width:1}}.grid{{stroke:#e5e8eb;stroke-width:1}}.tick{{font-size:10px;fill:#637080}}
</style><rect width="100%" height="100%" fill="#fff"/>{body}</svg>'''


def _write_svg(path: Path, width: int, height: int, title: str, body: str) -> None:
    path.write_text(_svg_document(width, height, title, body), encoding="utf-8")


def _scale(value: float, low: float, high: float, left: float, right: float) -> float:
    if not math.isfinite(value) or high <= low:
        return (left + right) / 2
    return left + (value - low) / (high - low) * (right - left)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _save_empty_figure(path: Path, title: str, message: str) -> None:
    body = (
        f'<text class="title" x="460" y="120" text-anchor="middle">{_svg_text(title)}</text>'
        f'<text class="label" x="460" y="155" text-anchor="middle">{_svg_text(message, 110)}</text>'
    )
    _write_svg(path, 920, 270, title, body)


def _plot_forest(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plotted = []
    for row in rows:
        estimate = _number(row, "macro_auroc", "auroc", "estimate")
        if not math.isfinite(estimate):
            continue
        low = _number(row, "macro_auroc_ci_low", "auroc_ci_low", "ci_low", "low")
        high = _number(row, "macro_auroc_ci_high", "auroc_ci_high", "ci_high", "high")
        plotted.append((_display_method(row), estimate, low, high))
    if not plotted:
        _save_empty_figure(path, "Macro AUROC", "headline_summary.csv has no plottable rows")
        return
    plotted = plotted[:28]
    finite = [value for _, estimate, low, high in plotted for value in (estimate, low, high) if math.isfinite(value)]
    domain_low = min(0.5, min(finite) - 0.02)
    domain_high = min(1.0, max(0.55, max(finite) + 0.02))
    width, left, right, top, row_h = 1120, 385, 1080, 58, 28
    height = top + row_h * len(plotted) + 50
    body = [f'<text class="title" x="{left}" y="27">Registered layer-fusion methods</text>']
    for tick in [domain_low + i * (domain_high - domain_low) / 5 for i in range(6)]:
        x = _scale(tick, domain_low, domain_high, left, right)
        body.append(f'<line class="grid" x1="{x:.1f}" y1="42" x2="{x:.1f}" y2="{height-34}"/>')
        body.append(f'<text class="tick" x="{x:.1f}" y="{height-16}" text-anchor="middle">{tick:.2f}</text>')
    if domain_low <= 0.5 <= domain_high:
        x = _scale(0.5, domain_low, domain_high, left, right)
        body.append(f'<line x1="{x:.1f}" y1="42" x2="{x:.1f}" y2="{height-34}" stroke="#8d98a5" stroke-dasharray="4 4"/>')
    for index, (label, estimate, low, high) in enumerate(plotted):
        y = top + index * row_h
        body.append(f'<text class="label" x="{left-12}" y="{y+4}" text-anchor="end">{_svg_text(label)}</text>')
        x = _scale(estimate, domain_low, domain_high, left, right)
        if math.isfinite(low) and math.isfinite(high) and low <= estimate <= high:
            x0, x1 = _scale(low, domain_low, domain_high, left, right), _scale(high, domain_low, domain_high, left, right)
            body.append(f'<line x1="{x0:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" stroke="#176f5b" stroke-width="2"/>')
            body.append(f'<line x1="{x0:.1f}" y1="{y-4}" x2="{x0:.1f}" y2="{y+4}" stroke="#176f5b"/>')
            body.append(f'<line x1="{x1:.1f}" y1="{y-4}" x2="{x1:.1f}" y2="{y+4}" stroke="#176f5b"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y}" r="4" fill="#176f5b"/>')
        body.append(f'<text class="small" x="{x+7:.1f}" y="{y-6}">{estimate:.4f}</text>')
    _write_svg(path, width, height, "Macro AUROC forest plot", "".join(body))


def _heat_color(value: float, low: float, high: float) -> str:
    fraction = 0.5 if high <= low else min(1.0, max(0.0, (value - low) / (high - low)))
    start, end = (236, 241, 238), (23, 111, 91)
    rgb = tuple(round(a + fraction * (b - a)) for a, b in zip(start, end))
    return "#" + "".join(f"{part:02x}" for part in rgb)


def _plot_heatmap(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    cells, methods = [], []
    for row in rows:
        value = _number(row, "auroc", "roc_auc")
        cell = str(_first(row, "cell", "cell_id", "dataset", default="")).strip()
        method = _display_method(row)
        if cell and math.isfinite(value):
            values[(method, cell)].append(value)
            if cell not in cells:
                cells.append(cell)
            if method not in methods:
                methods.append(method)
    if not values:
        _save_empty_figure(path, "Per-cell AUROC", "per_cell_metrics.csv has no plottable rows")
        return
    methods = methods[:32]
    left, top, cell_w, row_h = 385, 72, 105, 27
    width = left + cell_w * len(cells) + 20
    height = top + row_h * len(methods) + 35
    all_values = [v for entries in values.values() for v in entries]
    low, high = min(0.5, min(all_values)), max(0.8, max(all_values))
    body = ['<text class="title" x="18" y="27">Per-cell candidate-level AUROC</text>']
    for column, cell in enumerate(cells):
        x = left + column * cell_w + cell_w / 2
        body.append(f'<text class="small" x="{x:.1f}" y="58" text-anchor="middle">{_svg_text(_pretty(cell), 16)}</text>')
    for i, method in enumerate(methods):
        y = top + i * row_h
        body.append(f'<text class="label" x="{left-10}" y="{y+17}" text-anchor="end">{_svg_text(method)}</text>')
        for j, cell in enumerate(cells):
            entries = values.get((method, cell), [])
            x = left + j * cell_w
            if not entries:
                body.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{row_h-2}" fill="#f1f3f4"/>')
                continue
            value = _mean(entries)
            fill = _heat_color(value, low, high)
            text_color = "#fff" if value > (low + high) / 2 else "#18202a"
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{row_h-2}" rx="2" fill="{fill}"/>')
            body.append(f'<text x="{x+(cell_w-2)/2:.1f}" y="{y+17}" text-anchor="middle" font-size="10" fill="{text_color}" style="fill:{text_color}">{value:.3f}</text>')
    _write_svg(path, width, height, "Per-cell AUROC heatmap", "".join(body))


def _plot_deltas(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plotted = []
    for row in rows:
        delta = _number(row, "delta", "mean_delta", "estimate")
        if not math.isfinite(delta):
            continue
        name = str(_first(row, "contrast", "comparison", default="")).strip()
        if not name:
            name = f'{_first(row, "lhs", "method")} − {_first(row, "rhs", "reference")}'
        metric = str(_first(row, "metric", default="auroc")).upper()
        plotted.append((f"{_pretty(name)} [{metric}]", delta,
                        _number(row, "ci_low", "delta_ci_low", "low"),
                        _number(row, "ci_high", "delta_ci_high", "high")))
    if not plotted:
        _save_empty_figure(path, "Registered paired deltas", "paired_comparisons.csv has no plottable rows")
        return
    plotted = plotted[:24]
    finite = [v for _, d, lo, hi in plotted for v in (d, lo, hi) if math.isfinite(v)]
    bound = max(0.01, max(abs(min(finite)), abs(max(finite))) * 1.15)
    width, left, right, top, row_h = 1120, 430, 1080, 56, 29
    height = top + len(plotted) * row_h + 48
    zero = _scale(0, -bound, bound, left, right)
    body = [f'<text class="title" x="{left}" y="27">Pre-registered paired contrasts</text>',
            f'<line class="axis" x1="{zero:.1f}" y1="42" x2="{zero:.1f}" y2="{height-32}"/>']
    for index, (label, delta, low, high) in enumerate(plotted):
        y = top + index * row_h
        color = "#176f5b" if delta >= 0 else "#b3453f"
        body.append(f'<text class="label" x="{left-12}" y="{y+4}" text-anchor="end">{_svg_text(label)}</text>')
        if math.isfinite(low) and math.isfinite(high):
            x0, x1 = _scale(low, -bound, bound, left, right), _scale(high, -bound, bound, left, right)
            body.append(f'<line x1="{x0:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        x = _scale(delta, -bound, bound, left, right)
        body.append(f'<circle cx="{x:.1f}" cy="{y}" r="4" fill="{color}"/>')
        body.append(f'<text class="small" x="{x+7:.1f}" y="{y-6}">{delta:+.4f}</text>')
    for value in (-bound, 0, bound):
        x = _scale(value, -bound, bound, left, right)
        body.append(f'<text class="tick" x="{x:.1f}" y="{height-14}" text-anchor="middle">{value:+.3f}</text>')
    _write_svg(path, width, height, "Paired method deltas", "".join(body))


def _plot_layer_curves(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        layer = _number(row, "layer", "layer_index")
        value = _number(row, "auroc", "roc_auc")
        if math.isfinite(layer) and math.isfinite(value):
            label = " · ".join(
                str(_first(row, name, default="")) for name in ("metric", "module")
                if str(_first(row, name, default="")).strip()
            ) or "single view"
            series[label].append((layer, value))
    if not series:
        _save_empty_figure(path, "Layer diagnostics", "layer_diagnostics.csv has no plottable rows")
        return
    colors = ("#176f5b", "#286f9b", "#a36527", "#7757a0", "#b3453f", "#4a7b36",
              "#16697a", "#9a5579", "#645f28", "#4f67a8", "#a04a24", "#647080")
    width, height, left, right, top, bottom = 1080, 560, 70, 790, 48, 505
    all_points = [point for points in series.values() for point in points]
    x_low, x_high = min(p[0] for p in all_points), max(p[0] for p in all_points)
    y_low, y_high = min(0.48, min(p[1] for p in all_points) - 0.02), max(0.55, max(p[1] for p in all_points) + 0.02)
    body = ['<text class="title" x="70" y="27">Signal by layer, metric, and hook position</text>']
    for value in [y_low + i * (y_high - y_low) / 5 for i in range(6)]:
        y = _scale(value, y_low, y_high, bottom, top)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        body.append(f'<text class="tick" x="{left-8}" y="{y+3:.1f}" text-anchor="end">{value:.2f}</text>')
    for index, (label, points) in enumerate(list(series.items())[:18]):
        points.sort()
        color = colors[index % len(colors)]
        coordinates = " ".join(
            f'{_scale(x, x_low, x_high, left, right):.1f},{_scale(y, y_low, y_high, bottom, top):.1f}'
            for x, y in points
        )
        body.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="1.7" opacity=".88"/>')
        legend_y = 58 + index * 24
        legend_x = 825 if index < 18 else 950
        body.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+20}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text class="small" x="{legend_x+26}" y="{legend_y+3}">{_svg_text(label, 28)}</text>')
    for value in range(int(x_low), int(x_high) + 1, max(1, int((x_high - x_low) // 8) or 1)):
        x = _scale(value, x_low, x_high, left, right)
        body.append(f'<text class="tick" x="{x:.1f}" y="{bottom+20}" text-anchor="middle">{value}</text>')
    body.append(f'<text class="label" x="{(left+right)/2:.1f}" y="{bottom+43}" text-anchor="middle">Transformer layer</text>')
    _write_svg(path, width, height, "Layer signal curves", "".join(body))


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
        if part != ""
    )


def _correlation_color(value: float) -> str:
    """Blue-white-red diverging color for correlations in [-1, 1]."""
    value = max(-1.0, min(1.0, value))
    blue, middle, red = (43, 111, 154), (247, 247, 243), (178, 66, 58)
    if value < 0:
        fraction = value + 1.0
        rgb = tuple(round(a + fraction * (b - a)) for a, b in zip(blue, middle))
    else:
        fraction = value
        rgb = tuple(round(a + fraction * (b - a)) for a, b in zip(middle, red))
    return "#" + "".join(f"{part:02x}" for part in rgb)


def _plot_correlation_heatmap(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    """Render an explicit layer/view correlation matrix from generic pair rows."""
    pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
    labels: set[str] = set()
    for row in rows:
        first = str(
            _first(
                row,
                "feature_a",
                "layer_a",
                "layer_i",
                "row_feature",
                "source_feature",
                default="",
            )
        ).strip()
        second = str(
            _first(
                row,
                "feature_b",
                "layer_b",
                "layer_j",
                "column_feature",
                "target_feature",
                default="",
            )
        ).strip()
        value = _number(
            row,
            "value",
            "correlation",
            "spearman",
            "pearson",
            "rho",
        )
        diagnostic = str(_first(row, "diagnostic", "kind", default="")).lower()
        if first and second and math.isfinite(value) and (
            not diagnostic or any(token in diagnostic for token in ("corr", "spearman", "pearson", "rho"))
        ):
            labels.update((first, second))
            pairs[(first, second)].append(value)
    if not pairs:
        _save_empty_figure(
            path,
            "Layer-correlation heatmap",
            "No feature_a/feature_b correlation rows were supplied",
        )
        return

    ordered = sorted(labels, key=_natural_key)[:32]
    left, top, cell_size = 150, 116, max(16, min(25, 650 // max(1, len(ordered))))
    matrix_size = cell_size * len(ordered)
    width, height = left + matrix_size + 92, top + matrix_size + 72
    body = [
        '<text class="title" x="18" y="27">Layer-correlation heatmap</text>',
        '<text class="small" x="18" y="47">Mean supplied correlation for each pair; diagonal is identity when omitted</text>',
    ]
    label_step = max(1, math.ceil(len(ordered) / 12))
    for i, first in enumerate(ordered):
        for j, second in enumerate(ordered):
            candidates = pairs.get((first, second), []) + pairs.get((second, first), [])
            if first == second and not candidates:
                candidates = [1.0]
            x, y = left + j * cell_size, top + i * cell_size
            if candidates:
                value = _mean(candidates)
                fill = _correlation_color(value)
                label = f"{first} × {second}: correlation {value:.4f}"
            else:
                fill, label = "#e9ecef", f"{first} × {second}: unavailable"
            body.append(
                f'<rect x="{x}" y="{y}" width="{cell_size-1}" height="{cell_size-1}" '
                f'fill="{fill}"><title>{html.escape(label)}</title></rect>'
            )
        if i % label_step == 0 or i == len(ordered) - 1:
            axis_y = top + i * cell_size + cell_size * 0.68
            axis_x = left + i * cell_size + cell_size * 0.5
            body.append(
                f'<text class="small" x="{left-7}" y="{axis_y:.1f}" text-anchor="end">'
                f'{_svg_text(first, 19)}</text>'
            )
            body.append(
                f'<text class="small" transform="translate({axis_x:.1f} {top-7}) rotate(-55)" '
                f'text-anchor="start">{_svg_text(first, 19)}</text>'
            )
    legend_x = left + matrix_size + 25
    legend_h = max(120, min(260, matrix_size))
    steps = 40
    for index in range(steps):
        value = 1.0 - index * 2.0 / (steps - 1)
        body.append(
            f'<rect x="{legend_x}" y="{top + index * legend_h / steps:.1f}" width="15" '
            f'height="{legend_h / steps + .5:.1f}" fill="{_correlation_color(value)}"/>'
        )
    body.extend(
        (
            f'<text class="tick" x="{legend_x+21}" y="{top+6}">+1</text>',
            f'<text class="tick" x="{legend_x+21}" y="{top+legend_h/2+4:.1f}">0</text>',
            f'<text class="tick" x="{legend_x+21}" y="{top+legend_h+4:.1f}">−1</text>',
        )
    )
    _write_svg(path, width, height, "Layer-correlation heatmap", "".join(body))


def _plot_dependence(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    distance_points: list[tuple[float, float]] = []
    rank_points: list[tuple[str, float]] = []
    for index, row in enumerate(rows):
        diagnostic = _normal_key(str(_first(row, "diagnostic", "kind", default="")))
        distance = _number(row, "layer_distance", "layer_gap", "distance")
        correlation = _number(
            row,
            "mean_abs_spearman",
            "mean_abs_correlation",
            "abs_spearman",
            "correlation",
            "spearman",
            "rho",
            "value",
        )
        if (
            math.isfinite(distance)
            and math.isfinite(correlation)
            and "effectiverank" not in diagnostic
        ):
            distance_points.append((distance, correlation))

        rank = _number(row, "effective_rank")
        if not math.isfinite(rank) and "effectiverank" in diagnostic:
            rank = _number(row, "value")
        if math.isfinite(rank):
            cell = str(_first(row, "cell", "cell_id", default="cell"))
            contract = str(_first(row, "contract", "feature_contract", default=""))
            label = " · ".join(item for item in (cell, contract) if item) or f"row {index + 1}"
            rank_points.append((label, rank))

    if not distance_points and not rank_points:
        _save_empty_figure(
            path,
            "Layer dependence",
            "No layer-distance correlation or effective-rank values were supplied",
        )
        return

    width, height, top, bottom = 1120, 480, 66, 405
    left0, right0, left1, right1 = 68, 700, 770, 1085
    body = [
        '<text class="title" x="18" y="27">Dependence versus distance and effective rank</text>',
        f'<text class="label" x="{(left0+right0)/2:.1f}" y="51" text-anchor="middle">Correlation decay across layer distance</text>',
        f'<text class="label" x="{(left1+right1)/2:.1f}" y="51" text-anchor="middle">Effective rank by cell / contract</text>',
    ]

    if distance_points:
        distance_points.sort()
        x_low, x_high = min(p[0] for p in distance_points), max(p[0] for p in distance_points)
        y_low, y_high = min(p[1] for p in distance_points), max(p[1] for p in distance_points)
        x_pad = (x_high - x_low) * 0.05 or 1.0
        y_pad = (y_high - y_low) * 0.08 or 0.05
        x_low, x_high = x_low - x_pad, x_high + x_pad
        y_low, y_high = y_low - y_pad, y_high + y_pad
        for tick_index in range(5):
            x_value = x_low + tick_index * (x_high - x_low) / 4
            x = _scale(x_value, x_low, x_high, left0, right0)
            body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>')
            body.append(f'<text class="tick" x="{x:.1f}" y="{bottom+18}" text-anchor="middle">{x_value:.1f}</text>')
        for tick_index in range(5):
            y_value = y_low + tick_index * (y_high - y_low) / 4
            y = _scale(y_value, y_low, y_high, bottom, top)
            body.append(f'<line class="grid" x1="{left0}" y1="{y:.1f}" x2="{right0}" y2="{y:.1f}"/>')
            body.append(f'<text class="tick" x="{left0-7}" y="{y+3:.1f}" text-anchor="end">{y_value:.2f}</text>')
        for x_value, y_value in distance_points:
            x = _scale(x_value, x_low, x_high, left0, right0)
            y = _scale(y_value, y_low, y_high, bottom, top)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="#286f9b" opacity=".72"/>')
        unique_x = sorted({point[0] for point in distance_points})
        means = [(x, _mean([point[1] for point in distance_points if point[0] == x])) for x in unique_x]
        coordinates = " ".join(
            f'{_scale(x, x_low, x_high, left0, right0):.1f},{_scale(y, y_low, y_high, bottom, top):.1f}'
            for x, y in means
        )
        body.append(f'<polyline points="{coordinates}" fill="none" stroke="#15445e" stroke-width="2"/>')
        body.append(f'<text class="label" x="{(left0+right0)/2:.1f}" y="{height-13}" text-anchor="middle">Layer distance</text>')
        body.append(f'<text class="label" transform="translate(14 {(top+bottom)/2:.1f}) rotate(-90)" text-anchor="middle">Correlation</text>')
    else:
        body.append(f'<text class="small" x="{(left0+right0)/2:.1f}" y="{(top+bottom)/2:.1f}" text-anchor="middle">No distance rows</text>')

    if rank_points:
        aggregated: dict[str, list[float]] = defaultdict(list)
        for label, rank in rank_points:
            aggregated[label].append(rank)
        ranked = sorted(
            ((label, _mean(values)) for label, values in aggregated.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:14]
        rank_high = max(value for _, value in ranked) * 1.08 or 1.0
        row_h = (bottom - top) / max(1, len(ranked))
        for index, (label, value) in enumerate(ranked):
            y = top + index * row_h
            bar_width = _scale(value, 0, rank_high, 0, right1 - left1)
            body.append(f'<rect x="{left1}" y="{y:.1f}" width="{bar_width:.1f}" height="{max(4,row_h-4):.1f}" fill="#176f5b" opacity=".82"/>')
            body.append(f'<text class="small" x="{left1-7}" y="{y+row_h-7:.1f}" text-anchor="end">{_svg_text(label, 18)}</text>')
            body.append(f'<text class="small" x="{left1+bar_width+5:.1f}" y="{y+row_h-7:.1f}">{value:.2f}</text>')
        body.append(f'<text class="label" x="{(left1+right1)/2:.1f}" y="{height-13}" text-anchor="middle">Effective rank</text>')
    else:
        body.append(f'<text class="small" x="{(left1+right1)/2:.1f}" y="{(top+bottom)/2:.1f}" text-anchor="middle">No effective-rank rows</text>')
    _write_svg(path, width, height, "Layer dependence diagnostics", "".join(body))


def _plot_weights(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    weight_values: list[tuple[str, float]] = []
    gate_values: list[tuple[str, float]] = []
    graph_rows: list[Mapping[str, Any]] = []
    for row in rows:
        kind = _normal_key(str(_first(row, "kind", "diagnostic", "type", default="")))
        identity = " · ".join(
            item
            for item in (
                str(_first(row, "method", "solver", default="")),
                str(_first(row, "feature", "view", "group", default="")),
                str(_first(row, "cell", "cell_id", default="")),
            )
            if item
        ) or "diagnostic"

        weight = _number(row, "weight", "signed_weight", "coefficient", "abs_weight")
        gate = _number(row, "gate_probability", "survival_probability", "probability", "gate")
        generic = _number(row, "value")
        if not math.isfinite(weight) and math.isfinite(generic) and any(
            token in kind for token in ("weight", "coefficient", "fusion")
        ):
            weight = generic
        if not math.isfinite(gate) and math.isfinite(generic) and any(
            token in kind for token in ("gate", "dufs", "survival")
        ):
            gate = generic
        if math.isfinite(weight):
            weight_values.append((identity, weight))
        if math.isfinite(gate):
            gate_values.append((identity, gate))

        if (
            any(token in kind for token in ("graph", "converg", "optim"))
            or any(
                str(_first(row, name, default="")).strip()
                for name in (
                    "graph_components",
                    "mean_degree",
                    "spectral_gap",
                    "converged",
                    "epoch",
                    "n_epochs",
                )
            )
        ):
            graph_rows.append(row)

    if not weight_values and not gate_values and not graph_rows:
        _save_empty_figure(
            path,
            "Fusion diagnostics",
            "No fusion-weight, DUFS-gate, graph, or convergence rows were supplied",
        )
        return

    width, height = 1120, 690
    left0, right0, left1, right1, top, panel_bottom = 178, 535, 735, 1085, 78, 410
    body = [
        '<text class="title" x="18" y="27">Fusion weights, DUFS gates, graph health, and convergence</text>',
        f'<text class="label" x="{(left0+right0)/2:.1f}" y="54" text-anchor="middle">Largest absolute fusion weights</text>',
        f'<text class="label" x="{(left1+right1)/2:.1f}" y="54" text-anchor="middle">Largest DUFS survival probabilities</text>',
    ]

    def bars(
        values: Sequence[tuple[str, float]],
        left: float,
        right: float,
        *,
        signed: bool,
        color: str,
    ) -> None:
        ranked = sorted(values, key=lambda item: abs(item[1]), reverse=True)[:14][::-1]
        if not ranked:
            body.append(f'<text class="small" x="{(left+right)/2:.1f}" y="{(top+panel_bottom)/2:.1f}" text-anchor="middle">Not supplied</text>')
            return
        row_h = (panel_bottom - top) / len(ranked)
        if signed:
            domain = max(abs(value) for _, value in ranked) or 1.0
            zero = _scale(0, -domain, domain, left, right)
            body.append(f'<line class="axis" x1="{zero:.1f}" y1="{top}" x2="{zero:.1f}" y2="{panel_bottom}"/>')
        else:
            low, high = 0.0, max(1.0, max(value for _, value in ranked))
            zero = left
            body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{panel_bottom}"/>')
        for index, (label, value) in enumerate(ranked):
            y = top + index * row_h
            x = _scale(value, -domain, domain, left, right) if signed else _scale(value, low, high, left, right)
            body.append(
                f'<rect x="{min(x, zero):.1f}" y="{y:.1f}" width="{max(1,abs(x-zero)):.1f}" '
                f'height="{max(3,row_h-4):.1f}" fill="{color}" opacity=".84"/>'
            )
            body.append(f'<text class="small" x="{left-7}" y="{y+row_h-7:.1f}" text-anchor="end">{_svg_text(label, 24)}</text>')
            body.append(f'<text class="small" x="{x+5 if x>=zero else x-5:.1f}" y="{y+row_h-7:.1f}" text-anchor="{"start" if x>=zero else "end"}">{value:.3f}</text>')

    bars(weight_values, left0, right0, signed=True, color="#7757a0")
    bars(gate_values, left1, right1, signed=False, color="#176f5b")

    body.append('<text class="label" x="18" y="458">Graph and optimizer health (generic runner fields)</text>')
    if graph_rows:
        graph_columns = (
            ("Components", ("graph_components", "n_components")),
            ("Mean degree", ("mean_degree", "graph_mean_degree")),
            ("Spectral gap", ("spectral_gap", "algebraic_connectivity")),
            ("Converged", ("converged", "optimizer_converged")),
            ("Seed", ("seed",)),
            ("Epoch", ("epoch", "n_epochs")),
        )
        for index, row in enumerate(graph_rows[:7]):
            y = 485 + index * 27
            label = " · ".join(
                item for item in (
                    str(_first(row, "cell", "cell_id", default="")),
                    str(_first(row, "method", "solver", default="")),
                    str(_first(row, "contract", "feature_contract", default="")),
                ) if item
            ) or f"row {index+1}"
            fields = []
            for field_label, aliases in graph_columns:
                value = _first(row, *aliases, default="")
                if str(value).strip():
                    fields.append(f"{field_label}: {value}")
            body.append(f'<text class="small" x="28" y="{y}">{_svg_text(label, 33)}</text>')
            body.append(f'<text class="small" x="290" y="{y}">{_svg_text("  ·  ".join(fields) or "diagnostic row supplied", 106)}</text>')
    else:
        body.append('<text class="small" x="28" y="493">No graph or convergence fields supplied</text>')
    _write_svg(path, width, height, "Fusion and graph diagnostics", "".join(body))


def _data_uri(path: Path) -> str:
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


ColumnKey = str | Sequence[str]


def _column_value(row: Mapping[str, Any], key: ColumnKey) -> Any:
    if isinstance(key, str):
        return _first(row, key, default="")
    return _first(row, *key, default="")


def _column_name(key: ColumnKey) -> str:
    return key if isinstance(key, str) else key[0]


def _table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[ColumnKey, str]],
    *,
    empty: str,
    caption: str,
    table_class: str = "",
) -> str:
    if not rows:
        return f'<p class="empty">{html.escape(empty)}</p>'
    head = "".join(f'<th scope="col">{html.escape(label)}</th>' for _, label in columns)
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="{"num" if isinstance(value := _column_value(row, key), (int, float)) or str(value).replace(".", "", 1).replace("-", "", 1).isdigit() else ""}">'
            f'{_fmt(value, _column_name(key))}</td>'
            for key, _ in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap"><table class="{html.escape(table_class)}">'
        f'<caption>{html.escape(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _dynamic_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    empty: str,
    caption: str,
    max_columns: int = 12,
) -> str:
    if not rows:
        return f'<p class="empty">{html.escape(empty)}</p>'
    keys = list(dict.fromkeys(key for row in rows for key in row))[:max_columns]
    return _table(
        rows,
        [(key, _pretty(key)) for key in keys],
        empty=empty,
        caption=caption,
    )


def _figure(path: Path, title: str, caption: str) -> str:
    return (
        f'<figure><img src="{_data_uri(path)}" alt="{html.escape(title)}">'
        f'<figcaption><strong>{html.escape(title)}.</strong> {html.escape(caption)}</figcaption></figure>'
    )


def _is_label_using_diagnostic(row: Mapping[str, Any]) -> bool:
    label_use = _normal_key(
        str(_first(row, "label_use", "eligibility", "analysis_role", default=""))
    )
    if label_use in {
        "supervisedceiling",
        "evaluationonly",
        "labelusing",
        "diagnosticonly",
        "oracle",
    }:
        return True
    method = _normal_key(
        str(_first(row, "method", "method_key", "method_name", default=""))
    )
    return any(
        token in method
        for token in (
            "logisticregression",
            "supervisedlr",
            "supervisedceiling",
            "bestsinglelayer",
            "layeroracle",
        )
    )


def _is_curated_visual_method(row: Mapping[str, Any]) -> bool:
    """Keep plots readable without outcome-based selection; tables retain all rows."""

    method = str(_first(row, "method", default=""))
    contract = str(_first(row, "feature_contract", default=""))
    structure = str(_first(row, "structured", default="flat"))
    if contract == "resid-core-L" and structure in {"flat", "dependency", "hierarchical-bands"}:
        return method in {
            "final_layer_nll", "equal_mean", "upcr", "iu_pcr", "dufs_liu_pcr",
            "su_pcr", "lsml_continuous", "clustered_upcr",
        }
    if contract in {"resid-core-8", "lens-96", "resid-core-L-length-residualized"}:
        return method == "dufs_liu_pcr"
    if contract == "trilens-entropy-3L" and structure == "flat":
        return method in {"trilens_equal_mean", "upcr", "iu_pcr", "dufs_liu_pcr"}
    if contract == "dola-kl-proxy-L" and structure == "flat":
        return method in {"dola_kl_equal_mean", "dufs_liu_pcr"}
    return method in {
        "haloscope_direct_proxy", "spilled_energy_eq8_mean_proxy",
        "spilled_energy_eq8_min_proxy", "generation_entropy_mean",
        "realized_token_nll_mean",
    }


def _weight_row_groups(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    weights, gates, graph = [], [], []
    for row in rows:
        kind = _normal_key(str(_first(row, "kind", "diagnostic", "type", default="")))
        if (
            any(math.isfinite(_number(row, name)) for name in ("weight", "signed_weight", "coefficient", "abs_weight"))
            or any(token in kind for token in ("weight", "coefficient", "fusion"))
        ):
            weights.append(row)
        if (
            any(math.isfinite(_number(row, name)) for name in ("gate_probability", "survival_probability", "probability", "gate"))
            or any(token in kind for token in ("gate", "dufs", "survival"))
        ):
            gates.append(row)
        if (
            any(token in kind for token in ("graph", "converg", "optim"))
            or any(
                str(_first(row, name, default="")).strip()
                for name in (
                    "graph_components",
                    "mean_degree",
                    "spectral_gap",
                    "converged",
                    "epoch",
                    "n_epochs",
                )
            )
        ):
            graph.append(row)
    return weights, gates, graph


CSS = r"""
:root{--bg:#f4f3ef;--panel:#fff;--ink:#18202a;--muted:#647080;--line:#d9dde2;
 --accent:#176f5b;--accent-soft:#dceee8;--warn:#9b6100;--warn-bg:#fff1d2;
 --bad:#a53d39;--bad-bg:#fbe8e6;--blue:#286f9b;--code:#edf0f2}
:root[data-theme="dark"]{--bg:#11161c;--panel:#19212a;--ink:#edf2f6;--muted:#a2adba;
 --line:#34404c;--accent:#55c7a9;--accent-soft:#183c34;--warn:#ffc35c;--warn-bg:#392d17;
 --bad:#ff9088;--bad-bg:#3d2525;--blue:#75b9e2;--code:#252f39}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11161c;--panel:#19212a;
 --ink:#edf2f6;--muted:#a2adba;--line:#34404c;--accent:#55c7a9;--accent-soft:#183c34;
 --warn:#ffc35c;--warn-bg:#392d17;--bad:#ff9088;--bad-bg:#3d2525;--blue:#75b9e2;--code:#252f39}}
*{box-sizing:border-box}html,body{max-width:100%}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{width:100%;max-width:1180px;margin:auto;padding:30px 22px 80px;overflow-wrap:anywhere}.hero{padding:22px 0 12px}
.eyebrow{margin:0;color:var(--muted);font-size:11px;letter-spacing:.15em;text-transform:uppercase}
h1{font-size:clamp(27px,5vw,47px);line-height:1.08;letter-spacing:-.035em;margin:7px 0 10px;max-width:880px}
h2{font-size:23px;margin:46px 0 8px;border-top:1px solid var(--line);padding-top:24px}
h3{font-size:17px;margin:25px 0 5px}p{max-width:82ch}.lede{font-size:17px;color:var(--muted)}
.validation-banner{display:flex;gap:12px;align-items:center;justify-content:space-between;border:2px solid var(--accent);background:var(--accent-soft);padding:11px 14px;margin:0 0 20px;border-radius:10px;font-weight:700}
body[data-validation-status="blocked"] .validation-banner{border-color:var(--bad);background:var(--bad-bg);color:var(--bad)}
.status{display:inline-block;border-radius:999px;padding:5px 12px;font-size:12px;font-weight:750;letter-spacing:.05em}
.status.valid{background:var(--accent-soft);color:var(--accent)}.status.blocked{background:var(--warn-bg);color:var(--warn)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr));gap:13px;margin:20px 0}
.tile,.card,figure{min-width:0;max-width:100%;background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:17px 18px}
.tile .label{color:var(--muted);font-size:12px}.tile .value{font-size:28px;font-weight:720;line-height:1.15;margin:5px 0}
.tile .foot{color:var(--muted);font-size:12px}.callout{border-left:4px solid var(--accent);background:var(--panel);padding:12px 16px;border-radius:0 10px 10px 0;margin:16px 0;max-width:90ch}
.callout.warn{border-color:var(--warn);background:var(--warn-bg)}.callout.bad{border-color:var(--bad);background:var(--bad-bg)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:15px;align-items:start}.grid>*{min-width:0}
figure{margin:14px 0;overflow:hidden}figure img{width:100%;max-width:100%;height:auto;display:block;border-radius:5px;background:#fff}
figcaption{color:var(--muted);font-size:12.5px;margin-top:10px}.table-wrap{max-width:100%;overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;border:1px solid var(--line);border-radius:10px;background:var(--panel);margin:12px 0 18px}
table{border-collapse:collapse;width:100%;font-size:13px}caption{text-align:left;padding:9px 10px;font-weight:700;color:var(--ink);background:var(--panel)}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere}
th{font-size:11px;text-transform:uppercase;letter-spacing:.055em;color:var(--muted);background:color-mix(in srgb,var(--panel) 90%,var(--line))}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}tbody tr:last-child td{border-bottom:0}
code{background:var(--code);padding:1px 5px;border-radius:4px}.empty{color:var(--muted);font-style:italic}.muted{color:var(--muted)}
.ok{color:var(--accent);font-weight:700}.fail{color:var(--bad);font-weight:700}.hash{font:11px ui-monospace,SFMono-Regular,monospace;word-break:break-all;white-space:normal}
.theme{position:fixed;z-index:5;right:12px;top:10px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:5px 10px;cursor:pointer}
@media(max-width:680px){main{padding:22px 13px 60px}.grid,.tiles{grid-template-columns:minmax(0,1fr)}.validation-banner{align-items:flex-start;flex-direction:column}.tile .value{font-size:23px}th,td{padding:6px 8px}figure{padding:10px}.theme{position:absolute}}
@media print{.theme{display:none}body{background:#fff}main{max-width:none}.tile,.card,figure{break-inside:avoid}}
"""


THEME_SCRIPT = r"""<script>
(function(){const b=document.createElement('button');b.className='theme';b.type='button';
b.textContent='◐ theme';b.setAttribute('aria-label','Toggle light and dark theme');
b.onclick=function(){const r=document.documentElement;let v=r.getAttribute('data-theme');
if(!v){v=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light'}
r.setAttribute('data-theme',v==='dark'?'light':'dark')};document.body.appendChild(b)})();
</script>"""


def _headline_cards(
    headline: Sequence[Mapping[str, Any]],
    per_cell: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> str:
    cells = {
        str(_first(row, "cell", "cell_id", "dataset", default=""))
        for row in per_cell
        if str(_first(row, "cell", "cell_id", "dataset", default=""))
    }
    sample_by_cell = {}
    for row in per_cell:
        cell = str(_first(row, "cell", "cell_id", "dataset", default=""))
        n = _integer(row, "n_samples", "n")
        if cell and n is not None:
            sample_by_cell[cell] = max(n, sample_by_cell.get(cell, 0))
    first = headline[0] if headline else {}
    macro = _number(first, "macro_auroc", "auroc", "estimate")
    primary = next(
        (
            row
            for row in paired
            if str(_first(row, "primary", default="")).lower() in {"1", "true", "yes"}
        ),
        paired[0] if paired else {},
    )
    delta = _number(primary, "delta", "mean_delta", "estimate")
    cards = (
        ("Evidence status", validation["status"], "promotion is fail-closed"),
        ("Evaluated cells", str(len(cells)) if cells else "—", "equal-cell macro; never pooled headline"),
        ("Candidate rows", f"{sum(sample_by_cell.values()):,}" if sample_by_cell else "—", "unique within-cell counts"),
        (
            "First registered macro AUROC",
            f"{macro:.4f}" if math.isfinite(macro) else "—",
            _display_method(first) if first else "headline artifact unavailable",
        ),
        (
            "First registered paired delta",
            f"{delta:+.4f}" if math.isfinite(delta) else "—",
            _pretty(_first(primary, "contrast", default="paired artifact unavailable")),
        ),
    )
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div><div class="foot">{html.escape(foot)}</div></div>'
        for label, value, foot in cards
    ) + "</div>"


def _claim_assessment(
    validation: Mapping[str, Any], paired: Sequence[Mapping[str, Any]]
) -> str:
    primary = [
        row
        for row in paired
        if str(_first(row, "primary", default="")).strip().lower() in {"1", "true", "yes"}
        or "dufs" in str(_first(row, "contrast", default="")).lower()
        and any(token in str(_first(row, "contrast", default="")).lower() for token in ("final", "iu"))
    ]
    primary = primary[:2]
    intervals_positive = bool(primary) and all(
        _number(row, "ci_low", "delta_ci_low", "low") > 0 for row in primary
    )
    if validation["validated"] and len(primary) >= 2 and intervals_positive:
        return (
            '<div class="callout" role="status"><strong>Registered claim supported.</strong> '
            "Both primary macro-AUROC intervals exclude zero on the positive side, and the "
            "capture-validation requirements pass.</div>"
        )
    reason = "the capture-validation requirements are incomplete"
    if validation["validated"]:
        reason = "the two registered primary intervals do not both exclude zero"
    return (
        '<div class="callout bad" role="alert"><strong>No robust-improvement claim is promoted.</strong> '
        + html.escape(reason)
        + ". Results below are descriptive and retain their pre-registered names.</div>"
    )


def render_html(
    *,
    results_dir: Path,
    inputs: Mapping[str, Sequence[Mapping[str, Any]] | Mapping[str, Any]],
    validation: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> str:
    per_cell = list(inputs["per_cell"])
    diagnostic_ceiling_rows = [row for row in per_cell if _is_label_using_diagnostic(row)]
    appendix_rows = [
        row for row in per_cell
        if "appendix" in str(_first(row, "status", default="")).lower()
    ]
    eligible_per_cell = [
        row for row in per_cell
        if not _is_label_using_diagnostic(row) and row not in appendix_rows
    ]
    headline = [row for row in inputs["headline"] if not _is_label_using_diagnostic(row)]
    paired = list(inputs["paired"])
    coverage = list(inputs["coverage"])
    layer = list(inputs["layer"])
    dependence = list(inputs["dependence"])
    weights = list(inputs["weights"])
    cohorts = list(inputs.get("cohorts", []))
    comparator_fidelity = list(inputs.get("comparator_fidelity", []))
    figures = results_dir / "figures"
    status_class = "valid" if validation["validated"] else "blocked"
    gate_rows = validation["rows"]
    coverage_roster_columns = [
        (("cell", "cell_id"), "Cell"),
        ("dataset", "Dataset"),
        ("model_family", "Model family"),
        ("n_layers", "Layers"),
        ("n_source_rows", "Source rows"),
        (("n_samples", "n"), "Joined candidates"),
        ("n_excluded_rows", "Excluded"),
        (("n_groups", "problem_groups"), "Problem groups"),
        ("prevalence", "Hallucination rate"),
        ("architecture_status", "Architecture"),
        ("geometry_status", "Geometry"),
        ("protocol_scope", "Scope"),
        ("status", "Status"),
        ("exclusion_reason", "Exclusion reason"),
    ]
    coverage_gate_columns = [
        (("cell", "cell_id"), "Cell"),
        ("raw_backfill_gate_b_status", "Raw backfill Gate B"),
        ("raw_backfill_gate_b_median", "Raw median error"),
        ("raw_backfill_gate_b_first", "Raw first-token error"),
        ("raw_backfill_gate_b_fraction", "Raw fraction ≤0.05"),
        (("corrected_layer_gate_b_status", "gate_b_status"), "Corrected live Gate B"),
        ("corrected_layer_gate_b_median", "Live median error"),
        ("corrected_layer_gate_b_first", "Live first-token error"),
        ("corrected_layer_gate_b_fraction", "Live fraction ≤0.05"),
    ]
    metric_columns = [
        ("cell", "Cell"),
        ("method", "Method"),
        ("feature_contract", "Contract"),
        ("layer_subset", "Layers"),
        ("structured", "Structure"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("prevalence", "Prevalence"),
        ("n_samples", "N"),
        ("n_groups", "Groups"),
        ("label_use", "Label use"),
    ]
    paired_columns = [
        ("contrast", "Contrast"),
        ("metric", "Metric"),
        ("delta", "Delta"),
        ("ci_low", "CI low"),
        ("ci_high", "CI high"),
        ("wins", "Wins"),
        ("ties", "Ties"),
        ("losses", "Losses"),
        ("worst_cell_delta", "Worst cell"),
        ("p_raw", "p raw"),
        ("p_holm", "p Holm"),
    ]
    cohort_columns = [
        ("cohort", "Cohort"), ("n_cells", "Cells"), ("method", "Method"),
        ("feature_contract", "Contract"), ("structured", "Structure"),
        ("macro_auroc", "Macro AUROC"), ("macro_auprc", "Macro AUPRC"),
    ]
    fidelity_columns = [
        ("method", "Comparator"), ("implementation", "What was run"),
        ("label_use", "Label use"), ("fidelity", "Fidelity"),
        ("limitation", "Boundary"),
    ]
    provenance_rows = [
        {"artifact": name, "sha256": digest, "bytes": (results_dir / name).stat().st_size}
        for name, digest in source_hashes.items()
    ]
    blockers = ", ".join(validation["blockers"]) if validation["blockers"] else "none"
    confound_rows = [
        row
        for row in per_cell
        if any(
            token in " ".join(str(value).lower() for value in row.values())
            for token in ("token_length", "token length", "length_residual", "length residual")
        )
    ]
    leakage = _first(inputs["score_freeze"], "labels_seen_during_fit", default="not recorded")
    run_protocol = _first(inputs["run_definition"], "protocol_signature", "run_fingerprint", default="not recorded")
    fusion_weight_rows, dufs_gate_rows, graph_rows = _weight_row_groups(weights)
    weight_columns = [
        (("cell", "cell_id"), "Cell"),
        (("method", "solver"), "Method"),
        (("contract", "feature_contract"), "Contract"),
        ("kind", "Kind"),
        (("feature", "view", "group"), "Feature / view"),
        (("weight", "signed_weight", "coefficient", "abs_weight", "value"), "Weight"),
        ("seed", "Seed"),
    ]
    gate_columns = [
        (("cell", "cell_id"), "Cell"),
        (("method", "solver"), "Method"),
        (("contract", "feature_contract"), "Contract"),
        ("kind", "Kind"),
        (("feature", "view", "group"), "Feature / view"),
        (("gate_probability", "survival_probability", "probability", "gate", "value"), "Gate probability"),
        ("seed", "Seed"),
    ]
    graph_columns = [
        (("cell", "cell_id"), "Cell"),
        (("method", "solver"), "Method"),
        (("contract", "feature_contract"), "Contract"),
        ("kind", "Kind"),
        ("graph_components", "Graph components"),
        ("mean_degree", "Mean degree"),
        ("spectral_gap", "Spectral gap"),
        ("converged", "Converged"),
        ("seed", "Seed"),
        (("epoch", "n_epochs"), "Epoch"),
    ]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>White-box Layer-Fusion Benchmark</title><style>{CSS}</style></head>
<body data-validation-status="{'validated' if validation['validated'] else 'blocked'}"><main>
<header class="hero">
<div class="validation-banner" role="{'status' if validation['validated'] else 'alert'}" aria-live="polite">
<span>{html.escape(validation['status'])}</span><span>{'All promotion gates passed.' if validation['validated'] else 'Numerical comparisons are descriptive only; no improvement claim may be promoted.'}</span>
</div>
<p class="eyebrow">{VERSION} · nine model families · 14 cells (13 primary + one rejected appendix)</p>
<span class="status {status_class}">{html.escape(validation['status'])}</span>
<h1>Can label-free fusion turn internal layer trajectories into a better hallucination detector?</h1>
<p class="lede">A frozen cross-architecture comparison of U-PCR, IU-PCR, DUFS-LIU-PCR, structured depth fusion, and fidelity-scoped TriLens, HaloScope, DoLa, Spilled Energy, and INSIDE arms. Higher scores mean greater hallucination risk. The headline is an equal-cell macro over 13 protocol-eligible cells; every cell remains visible.</p>
{_headline_cards(headline, per_cell, paired, validation)}
{_claim_assessment(validation, paired)}
</header>

<section aria-labelledby="validation"><h2 id="validation">1. Validation and claim boundary</h2>
<p>The original <code>whitebox/per-layer-views</code> branch was recovered and its hook/capture implementation is now source-frozen. Numerical results still cannot be promoted until corrected live Gate B covers all 14 cells and the independent two-cell architecture pilot passes. A claimed status string is ignored: the validation evidence, no-label fitting declaration, pre-label score freeze, and every freeze record must all pass explicitly.</p>
{_table(gate_rows, (("gate", "Evidence gate"), ("result", "Result"), ("required", "Required")), empty="No validation evidence was supplied.", caption="Fail-closed promotion gates")}
<div class="callout {'warn' if validation['blockers'] else ''}"><strong>Current blockers:</strong> {html.escape(blockers)}.</div>
</section>

<section aria-labelledby="coverage"><h2 id="coverage">2. Data roster and integrity coverage</h2>
<p>Coverage is read directly from <code>data_coverage.csv</code>. Each candidate must join by its exact <code>problem:candidate</code> key; candidate counts are not inferred from a shorter frozen benchmark.</p>
{_table(coverage, coverage_roster_columns, empty="Coverage artifact unavailable; this is itself a validation blocker.", caption="Joined data roster and exclusions")}
<div class="callout"><strong>Two runs of one valid Gate B contract are retained.</strong> Both start from fresh teacher-forced ordinary final logits, apply the original temperature/top-k/top-p warp, compute top-15 entropy, and compare it with raw <code>token_entropies</code>. Raw backfill Gate B records the sampled backfill-report run; corrected live Gate B repeats that contract with the fixed nested <code>problem:candidate</code> loader. Sidecar lens fidelity is a separate architecture check: unwarped/full-vocabulary sidecar lens quantities are compared with fresh hooked-state lens quantities. A direct raw-<code>token_entropies</code> versus sidecar-<code>lens_H</code> comparison is invalid and is never performed. Only corrected live Gate B controls report promotion.</div>
{_table(coverage, coverage_gate_columns, empty="Neither raw-backfill nor corrected-live Gate B evidence was supplied.", caption="Raw backfill Gate B versus corrected live Gate B")}
</section>

<section aria-labelledby="headline"><h2 id="headline">3. Headline method comparison</h2>
<p>Intervals use the runner's shared grouped-bootstrap draws. The chart does not choose a winner: it preserves the registered summary row order.</p>
{_figure(figures / 'macro_forest.svg', 'Macro AUROC forest plot', 'Equal-cell estimates with grouped-bootstrap intervals when supplied by headline_summary.csv.')}
{_figure(figures / 'per_cell_heatmap.svg', 'Per-cell AUROC heatmap', 'All protocol-eligible architecture/dataset cells remain separate so a macro result cannot hide a concentrated gain or loss.')}
{_table(eligible_per_cell, metric_columns, empty="Eligible per-cell metric artifact unavailable.", caption="Eligible label-free methods by cell")}
<h3>Architecture and continuity cohorts</h3>
<p>The primary macro uses 13 protocol-eligible cells. The original six-Llama cohort preserves continuity; the seven-model GSM8K cohort isolates architecture replication; the 14-cell descriptive cohort includes the rejected CoQA cell and cannot support a claim.</p>
{_table(cohorts, cohort_columns, empty="Cohort summary unavailable.", caption="Equal-cell macro summaries by registered cohort")}
<h3>Comparator fidelity map</h3>
{_table(comparator_fidelity, fidelity_columns, empty="Comparator fidelity artifact unavailable.", caption="Exact implementation and claim boundary for each literature comparator")}
<h3>Label-using diagnostic ceilings — visually and inferentially separate</h3>
<div class="callout warn"><strong>Not eligible for the headline or registered claim.</strong> Balanced grouped-CV logistic regression and best-single-layer curves inspect labels. They are shown only as diagnostic ceilings; their probabilities are scored fold by fold and never concatenated across independently calibrated folds.</div>
{_table(diagnostic_ceiling_rows, metric_columns, empty="No supervised grouped-CV or best-single-layer diagnostic rows were supplied.", caption="Label-using diagnostic ceilings")}
<h3>Protocol-rejected appendix</h3>
<div class="callout warn"><strong>Excluded from every promoted macro.</strong> The CoQA/Llama-1 capture contains the only paper-shaped INSIDE K=10 last-token embeddings, but the project audit rejected that generation cell because of a chat-template defect.</div>
{_table(appendix_rows, metric_columns, empty="No appendix-only rows were supplied.", caption="Rejected CoQA/INSIDE and companion scores")}
</section>

<section aria-labelledby="contrasts"><h2 id="contrasts">4. Pre-registered paired contrasts</h2>
<p>Positive deltas favor the left-hand method. Confidence intervals, win/tie/loss counts, worst-cell loss, and Holm-adjusted tests come from <code>paired_comparisons.csv</code>; this page never recomputes or reorders them by outcome.</p>
{_figure(figures / 'paired_deltas.svg', 'Paired macro deltas', 'The two primary DUFS-LIU contrasts and secondary solver, structure, spacing, and length-sensitivity comparisons.')}
{_table(paired, paired_columns, empty="Paired comparison artifact unavailable.", caption="Pre-registered paired method comparisons")}
</section>

<section aria-labelledby="layers"><h2 id="layers">5. Where layer signal appears</h2>
<div class="callout warn"><strong>Diagnostic only.</strong> Single-layer AUROCs use labels and therefore cannot select the fusion contract, layer subset, signs, or hierarchy.</div>
{_figure(figures / 'layer_curves.svg', 'Layer-by-layer signal curves', 'AUROC by layer, metric, and module position; displayed only as a post-freeze mechanism diagnostic.')}
</section>

<section aria-labelledby="dependence"><h2 id="dependence">6. Dependence and fusion mechanics</h2>
{_figure(figures / 'layer_correlation_heatmap.svg', 'Layer-correlation heatmap', 'An explicit matrix from feature_a/feature_b correlation rows, averaged over supplied duplicate cell/contract pairs without labels.')}
<div class="grid">
{_figure(figures / 'dependence_diagnostics.svg', 'Distance and effective-rank diagnostics', 'Correlation versus layer distance alongside effective rank by cell and feature contract.')}
{_figure(figures / 'weights_diagnostics.svg', 'Weights, gates, graph, and convergence', 'Fusion coefficients, DUFS survival probabilities, graph components/degree/gap, seeds, epochs, and convergence when supplied.')}
</div>
<h3>Dependence diagnostics</h3>{_dynamic_table(dependence[:120], empty="Dependence diagnostics unavailable.", caption="Generic layer-dependence diagnostics", max_columns=14)}
<h3>Fusion weights</h3>{_table(fusion_weight_rows[:120], weight_columns, empty="Fusion-weight diagnostics unavailable.", caption="Fusion coefficients and provenance")}
<h3>DUFS gates</h3>{_table(dufs_gate_rows[:120], gate_columns, empty="DUFS gate diagnostics unavailable.", caption="DUFS survival probabilities and seeds")}
<h3>Graph and convergence health</h3>{_table(graph_rows[:120], graph_columns, empty="Graph or optimizer diagnostics unavailable.", caption="Graph structure and optimizer convergence")}
</section>

<section aria-labelledby="confounds"><h2 id="confounds">7. Token-length confounds and sensitivity</h2>
<p><code>n_gen_tokens</code> is excluded from every fusion input. It appears only as a transparent baseline and in the pre-registered unlabeled length-residualization sensitivity analysis.</p>
{_table(confound_rows, metric_columns, empty="No token-length or length-residualized rows were supplied; the sensitivity result is pending.", caption="Token-length baseline and residualization sensitivity")}
</section>

<section aria-labelledby="limitations"><h2 id="limitations">8. Limitations</h2>
<ul>
<li>The primary evidence covers 13 architecture/dataset cells, but seven architectures occur only on GSM8K; it is not a fully crossed model-by-dataset design.</li>
<li>The capture source is recovered. Hidden projections are still mean-token 256-D Gaussian JL summaries, so HaloScope is a direct-score proxy rather than a full reproduction.</li>
<li><code>cov_eigs</code> overflowed float16 on Phi-3, Phi-3.5, and Qwen3. Covariance-geometry performance is omitted rather than imputed; core lens tensors remain finite.</li>
<li>TriLens uses the saved three-position entropies, but token-mean readout is a frozen approximation because the paper text leaves the fixed token readout unspecified. DoLa uses KL rather than JSD.</li>
<li>Spilled Energy uses Eq. 8 only where the sampled token is present in saved raw top-K and pools the full generated answer because exact-answer spans were not captured.</li>
<li>Label-free fitting is transductive within each cell. Labels are opened only after score hashes are frozen; this is a leakage boundary, not an unseen external test.</li>
<li>Thirteen primary cell-level pairs still provide modest power, and repeated GSM8K architectures are not independent datasets. Confidence intervals and per-cell effects carry more information than a lone p-value.</li>
<li>A supervised grouped-CV ceiling and best-single-layer curves are diagnostics, never eligible headline methods.</li>
</ul>
</section>

<section aria-labelledby="repro"><h2 id="repro">9. Reproducibility and provenance</h2>
<div class="tiles"><div class="tile"><div class="label">Protocol signature</div><div class="foot hash">{html.escape(str(run_protocol))}</div></div>
<div class="tile"><div class="label">Labels seen during fitting</div><div class="value">{html.escape(str(leakage))}</div><div class="foot">must be false for promotion</div></div>
<div class="tile"><div class="label">Report generator</div><div class="foot"><code>scripts/whitebox_layer_fusion_report.py</code></div></div></div>
{_table(provenance_rows, (("artifact", "Input artifact"), ("bytes", "Bytes"), ("sha256", "SHA-256")), empty="No source hashes were recorded.", caption="Input artifact hashes", table_class="provenance")}
<p class="muted">The generated <code>REPORT_MANIFEST.json</code> also hashes this page and every separate SVG. Images above are embedded as base64 copies, so the report has no network or neighboring-file dependency.</p>
</section>
</main>{THEME_SCRIPT}</body></html>"""


def build_report(results_dir: str | Path) -> dict[str, Any]:
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, Any] = {
        "per_cell": _read_csv(results_dir / "per_cell_metrics.csv"),
        "headline": _read_csv(results_dir / "headline_summary.csv"),
        "cohorts": _read_csv(results_dir / "cohort_summary.csv"),
        "comparator_fidelity": _read_csv(results_dir / "comparator_fidelity.csv"),
        "paired": _read_csv(results_dir / "paired_comparisons.csv"),
        "coverage": _read_csv(results_dir / "data_coverage.csv"),
        "layer": _read_csv(results_dir / "layer_diagnostics.csv"),
        "dependence": _read_csv(results_dir / "dependence_diagnostics.csv"),
        "weights": _read_csv(results_dir / "weights_diagnostics.csv"),
        "validation_status": _read_json(results_dir / "validation_status.json"),
        "run_definition": _read_json(results_dir / "RUN_DEFINITION.json"),
        "source_freeze": _read_json(results_dir / "SOURCE_FREEZE_MANIFEST.json"),
        "score_freeze": _read_json(results_dir / "SCORE_FREEZE_MANIFEST.json"),
    }
    validation = derive_validation_status(inputs["validation_status"], results_dir)
    source_hashes = {
        name: sha256_file(results_dir / name)
        for name in SOURCE_FILES
        if (results_dir / name).is_file()
    }

    eligible_headline = [
        row for row in inputs["headline"] if not _is_label_using_diagnostic(row)
    ]
    eligible_per_cell = [
        row for row in inputs["per_cell"]
        if not _is_label_using_diagnostic(row)
        and "appendix" not in str(_first(row, "status", default="")).lower()
    ]
    curated_headline = [row for row in eligible_headline if _is_curated_visual_method(row)]
    curated_per_cell = [row for row in eligible_per_cell if _is_curated_visual_method(row)]
    _plot_forest(curated_headline, figures_dir / "macro_forest.svg")
    _plot_heatmap(curated_per_cell, figures_dir / "per_cell_heatmap.svg")
    _plot_deltas(inputs["paired"], figures_dir / "paired_deltas.svg")
    _plot_layer_curves(inputs["layer"], figures_dir / "layer_curves.svg")
    _plot_correlation_heatmap(
        inputs["dependence"], figures_dir / "layer_correlation_heatmap.svg"
    )
    _plot_dependence(inputs["dependence"], figures_dir / "dependence_diagnostics.svg")
    _plot_weights(inputs["weights"], figures_dir / "weights_diagnostics.svg")

    report_path = results_dir / "REPORT.html"
    report_path.write_text(
        render_html(
            results_dir=results_dir,
            inputs=inputs,
            validation=validation,
            source_hashes=source_hashes,
        ),
        encoding="utf-8",
    )
    generated = {"REPORT.html": sha256_file(report_path)}
    for name in FIGURES:
        generated[f"figures/{name}"] = sha256_file(figures_dir / name)
    manifest = {
        "version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": validation["status"],
        "validated": validation["validated"],
        "blockers": validation["blockers"],
        "input_artifacts": source_hashes,
        "generated_artifacts": generated,
        "self_contained_html": True,
        "external_assets": [],
    }
    _write_json(results_dir / "REPORT_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    manifest = build_report(args.results_dir)
    print(f"wrote {Path(args.results_dir).resolve() / 'REPORT.html'}")
    print(f"status: {manifest['status']}")
    if manifest["blockers"]:
        print("blockers: " + "; ".join(manifest["blockers"]))


if __name__ == "__main__":
    main()
