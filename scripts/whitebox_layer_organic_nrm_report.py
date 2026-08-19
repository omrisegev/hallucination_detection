#!/usr/bin/env python3
"""Render the self-contained report for layer-organic white-box NRM."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "whitebox-layer-organic-nrm-report-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def number(row: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4, *, signed: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:+.{digits}f}" if signed else f"{numeric:.{digits}f}"


def table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]], *, caption: str) -> str:
    numeric = {
        "macro_auroc", "macro_auprc", "macro_auroc_ci_low", "macro_auroc_ci_high",
        "macro_auprc_ci_low", "macro_auprc_ci_high", "delta", "ci_low", "ci_high",
        "worst_cell_delta", "p_holm", "auroc", "auprc", "prevalence",
        "selected_eigenvalue", "distance_from_unit", "correction_scale",
    }
    head = "".join(f'<th scope="col">{html.escape(label)}</th>' for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "—")
            rendered = fmt(value, signed=key in {"delta", "ci_low", "ci_high", "worst_cell_delta"}) if key in numeric else html.escape(str(value))
            cls = ' class="num"' if key in numeric else ""
            cells.append(f"<td{cls}>{rendered}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table>'
        f"<caption>{html.escape(caption)}</caption>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{html.escape(title)}"><style>'
        'text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#20303a}'
        '.label{font-size:12px}.small{font-size:10px;fill:#687782}'
        '.title{font-size:15px;font-weight:700}.grid{stroke:#d8e0e4;stroke-width:1}'
        '.zero{stroke:#25343e;stroke-width:1.5}</style>'
    )


def contract_svg(path: Path) -> None:
    width, height = 940, 235
    body = [svg_header(width, height, "Layer-organic grouping contract"),
            '<text class="title" x="18" y="26">One organic group = one residual transformer layer</text>']
    layers = [(35, "Layer 0"), (350, "Layer 1"), (665, "Layer 31")]
    colors = ("#d9edf7", "#dcefe7", "#f6e7cf")
    labels = ("entropy", "target NLL", "top-1 surprisal")
    for x, layer in layers:
        body.append(f'<rect x="{x}" y="55" width="240" height="125" rx="14" fill="#f8fafb" stroke="#9eb0ba"/>')
        body.append(f'<text x="{x+120}" y="79" text-anchor="middle" class="label" font-weight="700">{layer}</text>')
        for index, (label, color) in enumerate(zip(labels, colors)):
            y = 92 + index * 27
            body.append(f'<rect x="{x+18}" y="{y}" width="204" height="21" rx="7" fill="{color}"/>')
            body.append(f'<text x="{x+120}" y="{y+15}" text-anchor="middle" class="small">{label}</text>')
    body.append('<text x="309" y="122" class="title">→</text><text x="624" y="122" class="title">… →</text>')
    body.append('<text x="18" y="215" class="small">Primary: 32 groups × 3 local metrics = 96 atomic features. KL-to-final is evaluated separately because it is nonlocal.</text>')
    body.append("</svg>")
    path.write_text("".join(body))


def forest_svg(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = list(rows)
    width, left, right, top, row_h = 960, 340, 35, 54, 38
    height = top + row_h * len(rows) + 40
    lo = min(number(row, "macro_auroc_ci_low") for row in rows) - 0.01
    hi = max(number(row, "macro_auroc_ci_high") for row in rows) + 0.01
    span = max(hi - lo, 1e-6)
    x = lambda value: left + (value - lo) / span * (width - left - right)
    body = [svg_header(width, height, "32-layer cohort macro AUROC"),
            '<text class="title" x="15" y="24">Equal-cell macro AUROC · exact 32-layer cohort</text>']
    for tick in range(6):
        value = lo + span * tick / 5
        px = x(value)
        body.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="38" y2="{height-25}"/>')
        body.append(f'<text class="small" x="{px:.1f}" y="{height-8}" text-anchor="middle">{value:.2f}</text>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        point, low, high = (number(row, key) for key in ("macro_auroc", "macro_auroc_ci_low", "macro_auroc_ci_high"))
        method = str(row["method"])
        color = "#14735c" if method.startswith("nrm_") else ("#7d4c9e" if "triad" in method else "#286f9b")
        body.append(f'<text class="label" x="15" y="{y+4}">{html.escape(str(row["display_method"]))}</text>')
        body.append(f'<line x1="{x(low):.1f}" x2="{x(high):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<circle cx="{x(point):.1f}" cy="{y}" r="5" fill="{color}"/>')
        body.append(f'<text class="small" x="{width-right}" y="{y+4}" text-anchor="end">{point:.4f}</text>')
    body.append("</svg>")
    path.write_text("".join(body))


def delta_svg(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = [row for row in rows if row.get("metric") == "auroc"]
    width, left, right, top, row_h = 960, 350, 35, 55, 40
    height = top + row_h * len(rows) + 42
    limit = max(abs(number(row, key)) for row in rows for key in ("ci_low", "ci_high")) * 1.12
    limit = max(limit, 0.005)
    x = lambda value: left + (value + limit) / (2 * limit) * (width - left - right)
    body = [svg_header(width, height, "Paired AUROC contrasts"),
            '<text class="title" x="15" y="24">Paired problem-bootstrap AUROC contrasts</text>',
            f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="38" y2="{height-25}"/>']
    for index, row in enumerate(rows):
        y = top + index * row_h
        point, low, high = (number(row, key) for key in ("delta", "ci_low", "ci_high"))
        color = "#14735c" if low > 0 else ("#ae4743" if high < 0 else "#a46c13")
        label = f"{row['contrast'].replace('_', ' ')} · {row['cohort']}"
        body.append(f'<text class="label" x="15" y="{y+4}">{html.escape(label)}</text>')
        body.append(f'<line x1="{x(low):.1f}" x2="{x(high):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<circle cx="{x(point):.1f}" cy="{y}" r="5" fill="{color}"/>')
        body.append(f'<text class="small" x="{width-right}" y="{y+4}" text-anchor="end">{point:+.4f}</text>')
    body.append("</svg>")
    path.write_text("".join(body))


def cell_delta_svg(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    wanted = ("organic_lodo_minus_atomic_iu", "organic_lomo_minus_atomic_iu")
    data = {
        row["contrast"]: json.loads(row["per_cell_deltas_json"])
        for row in rows if row.get("metric") == "auroc" and row.get("contrast") in wanted
    }
    cells = list(next(iter(data.values())))
    width, left, top, cell_w, row_h = 990, 255, 105, 66, 46
    height = top + row_h * len(wanted) + 45
    maximum = max(abs(value) for values in data.values() for value in values.values()) or 1.0
    body = [svg_header(width, height, "Per-cell organic NRM deltas"),
            '<text class="title" x="15" y="24">Per-cell AUROC delta against atomic IU-PCR</text>']
    for column, cell in enumerate(cells):
        x = left + column * cell_w + cell_w / 2
        body.append(f'<text class="small" x="{x:.1f}" y="{top-8}" text-anchor="end" transform="rotate(-55 {x:.1f} {top-8})">{html.escape(cell)}</text>')
    for row_index, name in enumerate(wanted):
        y = top + row_index * row_h
        body.append(f'<text class="label" x="15" y="{y+28}">{html.escape(name.replace("_", " "))}</text>')
        for column, cell in enumerate(cells):
            value = data[name][cell]
            strength = min(abs(value) / maximum, 1.0)
            color = (
                f"rgb({int(222-90*strength)},{int(241-50*strength)},{int(234-75*strength)})"
                if value >= 0 else
                f"rgb({int(249-35*strength)},{int(230-100*strength)},{int(228-90*strength)})"
            )
            x = left + column * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{row_h-2}" rx="3" fill="{color}"/>')
            body.append(f'<text class="small" x="{x+cell_w/2-1:.1f}" y="{y+28}" text-anchor="middle">{value*100:+.2f}</text>')
    body.append("</svg>")
    path.write_text("".join(body))


def direction_svg(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = [row for row in rows if row.get("contract") == "triad" and row.get("method") == "nrm_layer_lodo"]
    vectors = [json.loads(row["direction_json"]) for row in rows]
    width, left, top, col_w, row_h = 1120, 215, 52, 27, 27
    height = top + len(rows) * row_h + 35
    maximum = max(abs(value) for vector in vectors for value in vector) or 1.0
    body = [svg_header(width, height, "Layer NRM calibration direction heatmap"),
            '<text class="title" x="15" y="23">Leave-dataset-out neutral direction over 32 layer groups</text>']
    for layer in range(32):
        body.append(f'<text class="small" x="{left+layer*col_w+col_w/2:.1f}" y="{top-7}" text-anchor="middle">{layer}</text>')
    for row_index, (row, vector) in enumerate(zip(rows, vectors)):
        y = top + row_index * row_h
        body.append(f'<text class="small" x="15" y="{y+18}">{html.escape(row["target_cell"])}</text>')
        for layer, value in enumerate(vector):
            strength = min(abs(value) / maximum, 1.0)
            color = (
                f"rgb({int(222-90*strength)},{int(241-50*strength)},{int(234-75*strength)})"
                if value >= 0 else
                f"rgb({int(249-35*strength)},{int(230-100*strength)},{int(228-90*strength)})"
            )
            body.append(f'<rect x="{left+layer*col_w}" y="{y}" width="{col_w-1}" height="{row_h-1}" fill="{color}"/>')
    body.append("</svg>")
    path.write_text("".join(body))


def figure(path: Path, title: str, caption: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        '<figure class="panel">'
        f'<img src="data:image/svg+xml;base64,{encoded}" alt="{html.escape(title)}">'
        f'<figcaption><strong>{html.escape(title)}.</strong> {html.escape(caption)}</figcaption>'
        '</figure>'
    )


def build_report(results: Path) -> dict[str, Any]:
    definition = read_json(results / "RUN_DEFINITION.json")
    validation = read_json(results / "validation_status.json")
    freeze = read_json(results / "SCORE_FREEZE_MANIFEST.json")
    cohorts = read_csv(results / "cohort_summary.csv")
    paired = read_csv(results / "paired_comparisons.csv")
    calibration = read_csv(results / "calibration_diagnostics.csv")
    per_cell = read_csv(results / "per_cell_metrics.csv")

    all32_order = (
        "final_nll", "iu_compressed", "iu_layer_triad", "nrm_layer_lodo",
        "nrm_layer_lomo", "nrm_layer_loco", "iu_layer_kl", "nrm_layer_kl_lodo",
    )
    all32_map = {
        row["method"]: row for row in cohorts if row["cohort"] == "all_32layer"
    }
    all32 = [all32_map[method] for method in all32_order]
    focal = [row for row in paired if row["metric"] == "auroc"]
    subgroup = [
        row for row in paired
        if row["metric"] == "auroc" and row["contrast"] in {
            "llama_only_organic_minus_atomic_iu", "gsm8k_only_organic_minus_atomic_iu"
        }
    ]

    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "group_contract.svg": contract_svg,
        "macro_auroc.svg": lambda path: forest_svg(all32, path),
        "paired_auroc.svg": lambda path: delta_svg(focal, path),
        "per_cell_delta.svg": lambda path: cell_delta_svg(paired, path),
        "layer_directions.svg": lambda path: direction_svg(calibration, path),
    }
    for name, render in figure_paths.items():
        render(figures / name)

    lodo = next(row for row in paired if row["contrast"] == "organic_lodo_minus_atomic_iu" and row["metric"] == "auroc")
    lomo = next(row for row in paired if row["contrast"] == "organic_lomo_minus_atomic_iu" and row["metric"] == "auroc")
    atomic = all32_map["iu_layer_triad"]
    best_nrm = max((all32_map[key] for key in ("nrm_layer_lodo", "nrm_layer_lomo", "nrm_layer_loco")), key=lambda row: number(row, "macro_auroc"))

    styles = """
:root{color-scheme:light dark;--bg:#f2f5f4;--card:#fff;--ink:#1c2b33;--muted:#61717b;--line:#d8e0e3;--accent:#14735c;--warn:#9b6417;--bad:#a4423e}
@media(prefers-color-scheme:dark){:root{--bg:#10181d;--card:#182228;--ink:#e8eff1;--muted:#a8b6bd;--line:#36454d;--accent:#63c2a5;--warn:#e4b665;--bad:#ed8d87}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}main{max-width:1180px;margin:auto;padding:28px 20px 70px}h1{font-size:clamp(2rem,5vw,4.2rem);line-height:1.02;margin:.2rem 0 .8rem}h2{margin:2.4rem 0 .8rem;font-size:1.55rem}.eyebrow{letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}.banner{background:color-mix(in srgb,var(--warn) 18%,var(--card));border:1px solid var(--warn);padding:14px 18px;border-radius:12px;font-weight:800}.lead{font-size:1.15rem;max-width:850px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.stat,.panel,.note{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:17px}.stat strong{display:block;font-size:1.75rem}.stat span,.muted,figcaption{color:var(--muted)}figure{margin:16px 0}img{display:block;width:100%;height:auto;background:#fff;border-radius:9px}figcaption{font-size:.9rem;margin-top:10px}.table-wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px;margin:13px 0}table{border-collapse:collapse;min-width:720px;width:100%}caption{text-align:left;font-weight:800;padding:13px 14px}th,td{text-align:left;padding:9px 11px;border-top:1px solid var(--line);vertical-align:top}th{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em}.num{text-align:right;font-variant-numeric:tabular-nums}.callout{border-left:5px solid var(--accent);padding:10px 16px;background:var(--card)}code{font-size:.9em}.split{display:grid;grid-template-columns:1fr 1fr;gap:16px}ul{padding-left:1.3rem}@media(max-width:760px){main{padding:18px 12px 48px}.grid,.split{grid-template-columns:1fr}.stat strong{font-size:1.45rem}h1{font-size:2.4rem}.table-wrap{margin-left:0;margin-right:0}figure.panel{padding:8px}th,td{padding:7px 8px}}
"""
    html_doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Layer-organic white-box NRM</title><style>{styles}</style></head><body><main>
<p class="eyebrow">White-box layer fusion · retrospective structural addendum</p>
<h1>Layer-organic NRM</h1>
<p class="banner">{html.escape(validation['status'])}</p>
<p class="lead">This run tests the user's structural proposal directly: each residual transformer layer is one group, and entropy, target-token NLL, and top-1 surprisal are the three features inside it. Scores were fitted and hashed before correctness labels were opened.</p>
<div class="grid"><div class="stat"><strong>{fmt(atomic['macro_auroc'])}</strong><span>Atomic layer-triad IU-PCR · macro AUROC</span></div><div class="stat"><strong>{fmt(best_nrm['macro_auroc'])}</strong><span>Best NRM transfer arm ({html.escape(best_nrm['method'])})</span></div><div class="stat"><strong>{fmt(lodo['delta'], signed=True)}</strong><span>LODO NRM − atomic IU-PCR · AUROC</span></div></div>
<h2>Decision</h2><p class="callout">The organic grouping is scientifically cleaner, but adoption depends on the paired results below. A positive descriptive point is not enough: transfer must be stable across the leave-dataset and same-model/same-dataset cohorts. This addendum cannot promote the white-box claim while live capture validation remains blocked.</p>
{figure(figures/'group_contract.svg','Registered grouping','The primary contract has 32 organic layer groups and 96 atomic residual features. The KL-to-final quartet is a separate sensitivity, not part of the primary claim.')}
<h2>Headline performance</h2>
{figure(figures/'macro_auroc.svg','Exact-depth macro AUROC','Equal-cell means over ten protocol-eligible 32-layer cells; uncertainty resamples problem groups within each cell and averages identical draws across cells.')}
{table(all32,(("display_method","Method"),("macro_auroc","AUROC"),("macro_auroc_ci_low","CI low"),("macro_auroc_ci_high","CI high"),("macro_auprc","AUPRC"),("macro_auprc_ci_low","CI low"),("macro_auprc_ci_high","CI high")),caption="All exact-32-layer cells")}
<h2>Paired comparisons</h2>
{figure(figures/'paired_auroc.svg','Registered AUROC contrasts','Zero is the no-change line. Intervals use the same grouped bootstrap draws for both methods in every cell.')}
{table(focal,(("contrast","Contrast"),("cohort","Cohort"),("n_cells","Cells"),("delta","Δ AUROC"),("ci_low","CI low"),("ci_high","CI high"),("wins","W"),("ties","T"),("losses","L"),("worst_cell_delta","Worst"),("p_holm","Holm p")),caption="Paired AUROC evidence")}
{figure(figures/'per_cell_delta.svg','Per-cell deltas','Values are AUROC percentage points for the two cross-cell transfer definitions against matched atomic IU-PCR.')}
<h2>The clean controls</h2><div class="split"><div class="note"><strong>Same model, six datasets</strong><p>Llama-3.1-8B is fixed; each target is calibrated from the other five datasets. This isolates dataset transfer while preserving exact layer identity.</p></div><div class="note"><strong>Same dataset, five 32-layer models</strong><p>GSM8K is fixed; each target is calibrated from the other four models. This isolates model transfer without interpolating depth.</p></div></div>
{table(subgroup,(("contrast","Contrast"),("cohort","Cohort"),("n_cells","Cells"),("delta","Δ AUROC"),("ci_low","CI low"),("ci_high","CI high"),("wins","W"),("ties","T"),("losses","L"),("worst_cell_delta","Worst")),caption="Same-model and same-dataset organic NRM contrasts")}
<h2>Mechanism diagnostics</h2>
{figure(figures/'layer_directions.svg','Layer-mode directions','Each row is one target cell. Green and red are opposite signed coefficients after the frozen equal-layer risk orientation; magnitude is normalized only for color.')}
{table(calibration[:20],(("target_cell","Target"),("method","Method"),("contract","Contract"),("source_count","Sources"),("n_groups","Groups"),("selected_eigenvalue","Selected λ"),("distance_from_unit","|λ−1|"),("correction_scale","Correction SD")),caption="Calibration diagnostics (first 20 rows; full CSV is authoritative)")}
<h2>Boundaries and reproducibility</h2><ul><li>Only the ten eligible 32-layer cells enter the primary analysis. The Qwen3 36-layer and two Mistral 40-layer cells are excluded rather than depth-warped.</li><li>KL-to-final is secondary because it relates every layer to the final layer. Its final-layer column is mechanically zero, leaving 127 features.</li><li>The analysis is retrospective: v2 outcomes existed historically before this grouping was proposed.</li><li>Gate B and the architecture-fidelity pilot remain open, so the status is forcibly PRELIMINARY / VALIDATION BLOCKED.</li><li>Every NRM score bundle contains no label-like array; <code>labels_seen_during_fit=false</code>; score hashes are verified before evaluation.</li></ul>
{table([{"item":"Exact-depth roster","value":", ".join(definition['eligible_cells'])},{"item":"Excluded depth mismatch","value":", ".join(definition['excluded_nonmatching_depth_cells'])},{"item":"Primary features","value":"32 layers × 3 local residual metrics = 96"},{"item":"Sensitivity features","value":"32 layers × 4 metrics − final zero KL = 127"},{"item":"Frozen before labels","value":str(freeze['scores_frozen_before_labels'])},{"item":"Bootstrap","value":f"{definition['bootstrap']['draws']} problem-group draws; seed {definition['bootstrap']['seed']}"}],(("item","Inventory item"),("value","Value")),caption="Reproducibility inventory")}
<p class="muted">Generated from CSV/JSON/NPZ artifacts only. Separate SVG files are retained under <code>figures/</code>; identical SVG bytes are embedded here for portability. No network assets.</p>
</main></body></html>"""
    report_path = results / "REPORT.html"
    report_path.write_text(html_doc)

    inputs = (
        "RUN_DEFINITION.json", "FIT_COMPLETE.json", "SCORE_FREEZE_MANIFEST.json",
        "validation_status.json", "cohort_summary.csv", "paired_comparisons.csv",
        "per_cell_metrics.csv", "calibration_diagnostics.csv", "bootstrap_draw_manifest.json",
    )
    manifest = {
        "version": VERSION,
        "status": validation["status"],
        "report_generator_sha256": sha256_file(Path(__file__).resolve()),
        "self_contained": True,
        "network_assets": False,
        "semantic_tables": html_doc.count("<table>"),
        "embedded_svg_count": html_doc.count("data:image/svg+xml;base64,"),
        "inputs": {name: sha256_file(results / name) for name in inputs},
        "generated": {
            "REPORT.html": sha256_file(report_path),
            **{f"figures/{name}": sha256_file(figures / name) for name in figure_paths},
        },
    }
    (results / "REPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.results_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
