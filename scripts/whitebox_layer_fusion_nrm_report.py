#!/usr/bin/env python3
"""Render a self-contained HTML report for the white-box NRM addendum."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


VERSION = "whitebox-layer-fusion-nrm-report-v1"


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


def fmt(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.{digits}f}"


def table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    *,
    caption: str,
) -> str:
    head = "".join(f'<th scope="col">{html.escape(label)}</th>' for _, label in columns)
    body = []
    numeric = {
        "macro_auroc", "macro_auprc", "macro_auroc_ci_low", "macro_auroc_ci_high",
        "macro_auprc_ci_low", "macro_auprc_ci_high", "auroc", "auprc", "delta",
        "ci_low", "ci_high", "worst_cell_delta", "p_holm", "selected_eigenvalue",
        "distance_from_unit", "unit_distance_gap", "correction_scale",
    }
    for row in rows:
        cells = []
        for key, _ in columns:
            value = fmt(row.get(key)) if key in numeric else html.escape(str(row.get(key, "—")))
            cls = ' class="num"' if key in numeric else ""
            cells.append(f"<td{cls}>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table>'
        f"<caption>{html.escape(caption)}</caption>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{html.escape(title)}">'
        '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#24313d}'
        '.label{font-size:12px}.small{font-size:10px;fill:#65717d}.title{font-size:15px;font-weight:700}'
        '.axis{stroke:#9da7b0;stroke-width:1}.grid{stroke:#d9dee3;stroke-width:1}.zero{stroke:#18202a;stroke-width:1.5}'
        '</style>'
    )


def forest_svg(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = list(rows)
    width, left, right = 940, 335, 35
    row_h, top = 37, 54
    height = top + row_h * len(rows) + 40
    lo = min(number(row, "macro_auroc_ci_low") for row in rows) - 0.01
    hi = max(number(row, "macro_auroc_ci_high") for row in rows) + 0.01
    span = max(hi - lo, 1e-6)
    x = lambda value: left + (value - lo) / span * (width - left - right)
    body = [svg_header(width, height, "Equal-cell macro AUROC forest"),
            '<text class="title" x="15" y="24">Equal-cell macro AUROC · 13 eligible cells</text>']
    for tick in range(6):
        value = lo + span * tick / 5
        px = x(value)
        body.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="38" y2="{height-25}"/>')
        body.append(f'<text class="small" x="{px:.1f}" y="{height-8}" text-anchor="middle">{value:.2f}</text>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        point = number(row, "macro_auroc")
        low = number(row, "macro_auroc_ci_low")
        high = number(row, "macro_auroc_ci_high")
        color = "#176f5b" if str(row.get("method", "")).startswith("nrm_") else "#286f9b"
        body.append(f'<text class="label" x="15" y="{y+4}">{html.escape(str(row.get("display_method", row.get("method", ""))))}</text>')
        body.append(f'<line x1="{x(low):.1f}" x2="{x(high):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<circle cx="{x(point):.1f}" cy="{y}" r="5" fill="{color}"/>')
        body.append(f'<text class="small" x="{width-right}" y="{y+4}" text-anchor="end">{point:.4f}</text>')
    body.append("</svg>")
    path.write_text("".join(body))


def delta_svg(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = [row for row in rows if row.get("metric") == "auroc"]
    width, left, right = 940, 310, 35
    row_h, top = 38, 55
    height = top + row_h * len(rows) + 42
    limit = max(
        abs(number(row, key)) for row in rows for key in ("ci_low", "ci_high")
    ) * 1.15
    limit = max(limit, 0.01)
    x = lambda value: left + (value + limit) / (2 * limit) * (width - left - right)
    body = [svg_header(width, height, "Paired NRM AUROC deltas"),
            '<text class="title" x="15" y="24">NRM minus matched baseline · paired grouped bootstrap</text>',
            f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="38" y2="{height-25}"/>']
    for index, row in enumerate(rows):
        y = top + index * row_h
        point, low, high = (number(row, key) for key in ("delta", "ci_low", "ci_high"))
        color = "#176f5b" if low > 0 else ("#a53d39" if high < 0 else "#9b6100")
        label = str(row.get("contrast", "")).replace("_", " ")
        body.append(f'<text class="label" x="15" y="{y+4}">{html.escape(label)}</text>')
        body.append(f'<line x1="{x(low):.1f}" x2="{x(high):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<circle cx="{x(point):.1f}" cy="{y}" r="5" fill="{color}"/>')
        body.append(f'<text class="small" x="{width-right}" y="{y+4}" text-anchor="end">{point:+.4f}</text>')
    body.append("</svg>")
    path.write_text("".join(body))


def per_cell_svg(paired: Sequence[Mapping[str, Any]], path: Path) -> None:
    wanted = ["depth_lodo_minus_iu", "depth_lomo_minus_iu", "lens_lodo_minus_iu"]
    rows = {
        row["contrast"]: json.loads(row["per_cell_deltas_json"])
        for row in paired if row.get("metric") == "auroc" and row.get("contrast") in wanted
    }
    cells = list(next(iter(rows.values())))
    width, left, top = 1000, 230, 78
    cell_w, cell_h = 54, 42
    height = top + len(wanted) * cell_h + 55
    maximum = max(abs(value) for values in rows.values() for value in values.values()) or 1.0
    body = [svg_header(width, height, "Per-cell NRM AUROC delta heatmap"),
            '<text class="title" x="15" y="24">Per-cell AUROC deltas</text>']
    for column, cell in enumerate(cells):
        x = left + column * cell_w + cell_w / 2
        short = cell.replace("_t1.0", "").replace("_t0.5", "").replace("_t0.0", "").replace("_t0.6", "")
        body.append(f'<text class="small" x="{x:.1f}" y="{top-8}" text-anchor="end" transform="rotate(-55 {x:.1f} {top-8})">{html.escape(short)}</text>')
    for row_index, name in enumerate(wanted):
        y = top + row_index * cell_h
        body.append(f'<text class="label" x="15" y="{y+25}">{html.escape(name.replace("_", " "))}</text>')
        for column, cell in enumerate(cells):
            value = rows[name][cell]
            strength = min(abs(value) / maximum, 1.0)
            if value >= 0:
                color = f"rgb({int(225-95*strength)},{int(242-55*strength)},{int(236-85*strength)})"
            else:
                color = f"rgb({int(250-35*strength)},{int(232-105*strength)},{int(230-95*strength)})"
            x = left + column * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" rx="3" fill="{color}"/>')
            body.append(f'<text class="small" x="{x+cell_w/2-1:.1f}" y="{y+25}" text-anchor="middle">{value*100:+.1f}</text>')
    body.append(f'<text class="small" x="{left}" y="{height-12}">values shown in AUROC percentage points; green positive, red negative</text>')
    body.append("</svg>")
    path.write_text("".join(body))


def direction_svg(calibration: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = [
        row for row in calibration
        if row.get("contract") == "depth" and row.get("strategy") in {"lodo", "lomo"}
    ]
    width, left, top = 850, 265, 45
    family_w, row_h = 115, 27
    height = top + len(rows) * row_h + 35
    values = [json.loads(row["direction_json"]) for row in rows]
    maximum = max(abs(value) for vector in values for value in vector) or 1.0
    body = [svg_header(width, height, "Depth NRM calibration directions"),
            '<text class="title" x="15" y="22">Leave-out depth-mode directions</text>']
    for index in range(4):
        body.append(f'<text class="small" x="{left+index*family_w+family_w/2:.1f}" y="{top-8}" text-anchor="middle">depth band {index}</text>')
    for row_index, (row, vector) in enumerate(zip(rows, values)):
        y = top + row_index * row_h
        label = f"{row['target_cell']} · {row['strategy']}"
        body.append(f'<text class="small" x="15" y="{y+18}">{html.escape(label)}</text>')
        for index, value in enumerate(vector):
            strength = min(abs(value) / maximum, 1.0)
            color = (
                f"rgb({int(222-90*strength)},{int(241-50*strength)},{int(234-75*strength)})"
                if value >= 0 else
                f"rgb({int(249-35*strength)},{int(230-100*strength)},{int(228-90*strength)})"
            )
            x = left + index * family_w
            body.append(f'<rect x="{x}" y="{y}" width="{family_w-3}" height="{row_h-2}" rx="3" fill="{color}"/>')
            body.append(f'<text class="small" x="{x+family_w/2-1:.1f}" y="{y+18}" text-anchor="middle">{value:+.3f}</text>')
    body.append("</svg>")
    path.write_text("".join(body))


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def figure(path: Path, title: str, caption: str) -> str:
    return (
        f'<figure><img alt="{html.escape(title)}" src="{data_uri(path)}">'
        f'<figcaption><strong>{html.escape(title)}.</strong> {html.escape(caption)}</figcaption></figure>'
    )


def find_row(rows: Sequence[Mapping[str, Any]], key: str, value: str) -> Mapping[str, Any]:
    return next(row for row in rows if row.get(key) == value)


def build_report(results: Path) -> dict[str, Any]:
    headline = read_csv(results / "headline_summary.csv")
    paired = read_csv(results / "paired_comparisons.csv")
    per_cell = read_csv(results / "per_cell_metrics.csv")
    cohorts = read_csv(results / "cohort_summary.csv")
    calibration = read_csv(results / "calibration_diagnostics.csv")
    validation = read_json(results / "validation_status.json")
    definition = read_json(results / "RUN_DEFINITION.json")
    score_freeze = read_json(results / "SCORE_FREEZE_MANIFEST.json")

    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    forest_svg(headline, figures / "macro_forest.svg")
    delta_svg(paired, figures / "paired_deltas.svg")
    per_cell_svg(paired, figures / "per_cell_deltas.svg")
    direction_svg(calibration, figures / "depth_directions.svg")

    depth = find_row(headline, "method", "nrm_depth_lodo")
    depth_model = find_row(headline, "method", "nrm_depth_lomo")
    lens = find_row(headline, "method", "nrm_lens_lodo")
    depth_delta = next(row for row in paired if row["contrast"] == "depth_lodo_minus_iu" and row["metric"] == "auroc")
    depth_model_delta = next(row for row in paired if row["contrast"] == "depth_lomo_minus_iu" and row["metric"] == "auroc")
    lens_delta = next(row for row in paired if row["contrast"] == "lens_lodo_minus_iu" and row["metric"] == "auroc")
    focal = [row for row in paired if row["focal_addendum_contrast"] == "True"]
    depth_calibration = [row for row in calibration if row["contract"] == "depth"]
    eigen_summary = []
    for contract in ("depth", "lens"):
        for strategy in ("lodo", "lomo", "loco"):
            selected = [row for row in calibration if row["contract"] == contract and row["strategy"] == strategy and row["target_cell"] in {r["cell"] for r in per_cell if r["status"] == "eligible_retrospective"}]
            eigen_summary.append({
                "contract": contract,
                "strategy": strategy,
                "targets": len(selected),
                "source_count_range": f"{min(int(r['source_count']) for r in selected)}–{max(int(r['source_count']) for r in selected)}",
                "selected_eigenvalue": median(number(r, "selected_eigenvalue") for r in selected),
                "distance_from_unit": median(number(r, "distance_from_unit") for r in selected),
                "unit_distance_gap": median(number(r, "unit_distance_gap") for r in selected),
            })

    styles = """
:root{--bg:#f4f3ef;--panel:#fff;--ink:#18202a;--muted:#65717d;--line:#d9dee3;--green:#176f5b;--green-bg:#dceee8;--red:#a53d39;--red-bg:#fbe8e6;--warn:#8b5900;--warn-bg:#fff1d2;--blue:#286f9b;--code:#edf0f2}
:root[data-theme=dark]{--bg:#11161c;--panel:#19212a;--ink:#edf2f6;--muted:#a7b1bc;--line:#34404c;--green:#55c7a9;--green-bg:#183c34;--red:#ff9088;--red-bg:#3d2525;--warn:#ffc35c;--warn-bg:#392d17;--blue:#75b9e2;--code:#252f39}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#11161c;--panel:#19212a;--ink:#edf2f6;--muted:#a7b1bc;--line:#34404c;--green:#55c7a9;--green-bg:#183c34;--red:#ff9088;--red-bg:#3d2525;--warn:#ffc35c;--warn-bg:#392d17;--blue:#75b9e2;--code:#252f39}}
*{box-sizing:border-box}html,body{max-width:100%}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1120px;margin:auto;padding:28px 20px 75px;overflow-wrap:anywhere}h1{font-size:clamp(28px,5vw,46px);line-height:1.08;letter-spacing:-.035em;margin:8px 0}h2{font-size:23px;margin:44px 0 8px;border-top:1px solid var(--line);padding-top:23px}p{max-width:84ch}.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}.lede{font-size:17px;color:var(--muted)}.banner{border:2px solid var(--red);background:var(--red-bg);border-radius:10px;padding:11px 14px;font-weight:750}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr));gap:12px;margin:20px 0}.tile,figure{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}.tile .label{font-size:12px;color:var(--muted)}.tile .value{font-size:27px;font-weight:750;margin:4px 0}.tile .foot{font-size:12px;color:var(--muted)}.callout{border-left:4px solid var(--warn);background:var(--warn-bg);padding:12px 15px;border-radius:0 9px 9px 0;max-width:90ch}.callout.good{border-color:var(--green);background:var(--green-bg)}.callout.bad{border-color:var(--red);background:var(--red-bg)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px}.table-wrap{max-width:100%;overflow:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel);margin:12px 0 18px}table{border-collapse:collapse;width:100%;font-size:13px}caption{text-align:left;padding:9px 10px;font-weight:750}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}figure{margin:14px 0;overflow:hidden}figure img{display:block;width:100%;height:auto;background:#fff;border-radius:5px}figcaption{color:var(--muted);font-size:12px;margin-top:9px}code{background:var(--code);padding:1px 5px;border-radius:4px}.theme{position:fixed;right:12px;top:10px;z-index:3;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:5px 10px;cursor:pointer}.muted{color:var(--muted)}@media(max-width:680px){main{padding:20px 12px 60px}.grid,.tiles{grid-template-columns:minmax(0,1fr)}.theme{position:absolute}.tile .value{font-size:22px}th,td{padding:6px 8px}}
"""
    headline_columns = [
        ("display_method", "Method"), ("macro_auroc", "AUROC"),
        ("macro_auroc_ci_low", "AUROC CI low"), ("macro_auroc_ci_high", "AUROC CI high"),
        ("macro_auprc", "AUPRC"), ("macro_auprc_ci_low", "AUPRC CI low"),
        ("macro_auprc_ci_high", "AUPRC CI high"),
    ]
    paired_columns = [
        ("contrast", "Contrast"), ("metric", "Metric"), ("delta", "Delta"),
        ("ci_low", "CI low"), ("ci_high", "CI high"), ("wins", "W"),
        ("ties", "T"), ("losses", "L"), ("worst_cell_delta", "Worst"),
        ("p_holm", "Holm p"),
    ]
    cohort_selected = [
        row for row in cohorts
        if row["method"] in {"iu_resid", "nrm_depth_lodo", "nrm_depth_lomo", "iu_lens96", "nrm_lens_lodo"}
    ]
    html_text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>White-box NRM Addendum</title><style>{styles}</style></head>
<body><button class="theme" type="button" onclick="let r=document.documentElement;r.dataset.theme=r.dataset.theme==='dark'?'light':'dark'">theme</button><main>
<header><div class="banner">PRELIMINARY / VALIDATION BLOCKED · retrospective post-v2 addendum</div><p class="eyebrow">{VERSION} · NRM-CS-IU transferred to white-box layer contributions</p><h1>Does Neutral Residual Mode repair white-box layer fusion?</h1>
<p class="lede">A label-free cross-cell correction of IU-PCR contributions, evaluated on the frozen 13-cell white-box roster. The original v2 primary remains unchanged. Higher scores mean greater hallucination risk.</p>
<div class="tiles"><div class="tile"><div class="label">Depth NRM · leave-dataset-out</div><div class="value">{fmt(depth['macro_auroc'])}</div><div class="foot">AUROC; Δ vs IU {fmt(depth_delta['delta'])}</div></div><div class="tile"><div class="label">Depth NRM · leave-model-out</div><div class="value">{fmt(depth_model['macro_auroc'])}</div><div class="foot">Δ vs IU {fmt(depth_model_delta['delta'])}</div></div><div class="tile"><div class="label">Lens NRM · leave-dataset-out</div><div class="value">{fmt(lens['macro_auroc'])}</div><div class="foot">Δ vs lens IU {fmt(lens_delta['delta'])}</div></div><div class="tile"><div class="label">Evidence status</div><div class="value">Blocked</div><div class="foot">no promoted improvement claim</div></div></div>
<div class="callout bad"><strong>Bottom line:</strong> NRM does not robustly repair the white-box benchmark. Depth NRM improves IU under leave-model-out by {fmt(depth_model_delta['delta'])} AUROC [{fmt(depth_model_delta['ci_low'])}, {fmt(depth_model_delta['ci_high'])}], but reverses to {fmt(depth_delta['delta'])} [{fmt(depth_delta['ci_low'])}, {fmt(depth_delta['ci_high'])}] under leave-dataset-out. Lens-96 NRM is slightly negative. This transfer sensitivity blocks a general success claim.</div></header>

<section><h2>1. Frozen design and leakage boundary</h2><p>The base is the exact anchor-oriented IU-PCR score from v2. Its feature contributions are summed within four architecture-relative depth quartiles or twelve fixed module×metric families. Source-cell residual covariance is averaged equally by cell; NRM selects the eigenvector closest to eigenvalue one, orients it toward the equal-family risk direction, and adds it at the frozen <code>1/G</code> trust ratio.</p>
<div class="callout"><strong>Why two leave-out rules?</strong> The roster is not fully crossed. A simultaneous dataset-and-model exclusion leaves only one source for GSM8K/Llama, which cannot support the registered covariance fit. This was detected before NRM outcomes were opened, so the analysis reports separate leave-dataset-out and leave-model-out transfers plus LOCO sensitivity.</div>
<p>Fit APIs receive only frozen <code>FeatureMatrix</code> bundles. <code>labels_seen_during_fit=false</code> is attested in every diagnostic; score hashes were frozen before raw correctness fields were opened. Nevertheless, this is retrospective because v2 outcomes were historically visible before the NRM addendum was proposed.</p></section>

<section><h2>2. Equal-cell AUROC and AUPRC</h2>{figure(figures/'macro_forest.svg','Macro AUROC forest','Intervals use 2,000 identical problem-group bootstrap draws per cell; the headline averages cells equally, never candidates.')}{table(headline, headline_columns, caption='Equal-cell macro over 13 protocol-eligible cells')}</section>

<section><h2>3. Paired comparisons</h2>{figure(figures/'paired_deltas.svg','Paired AUROC deltas','Positive values favor NRM. Green intervals exclude zero positively; red intervals exclude zero negatively.')}{table(paired, paired_columns, caption='Retrospective NRM contrasts; focal rows were fixed before the new NRM scores were evaluated')}</section>

<section><h2>4. Transfer instability across cells</h2>{figure(figures/'per_cell_deltas.svg','Per-cell AUROC deltas','Depth leave-dataset-out, depth leave-model-out, and lens leave-dataset-out compared with their matched IU baselines.')}
<p>Depth leave-dataset-out improves several original Llama task cells but loses on most GSM8K architecture-transfer cells. Leave-model-out changes the calibration population and turns the macro positive. That dependence on which nuisance axis is held out is the central negative robustness finding.</p>{table(cohort_selected, [('cohort','Cohort'),('display_method','Method'),('macro_auroc','AUROC'),('macro_auprc','AUPRC')], caption='Cohort sensitivity')}</section>

<section><h2>5. NRM mechanics and direction stability</h2>{figure(figures/'depth_directions.svg','Depth calibration directions','Each row is fitted without target outcomes. Direction changes across leave-dataset and leave-model source populations are visible directly.')}{table(eigen_summary, [('contract','Contract'),('strategy','Strategy'),('targets','Targets'),('source_count_range','Source cells'),('selected_eigenvalue','Median selected λ'),('distance_from_unit','Median |λ−1|'),('unit_distance_gap','Median mode gap')], caption='Neutral-mode diagnostics')}</section>

<section><h2>6. Decision and next test</h2><div class="callout good"><strong>Bounded positive observation:</strong> Depth NRM leave-model-out reaches AUROC {fmt(depth_model['macro_auroc'])}/AUPRC {fmt(depth_model['macro_auprc'])}, improving residual IU by {fmt(depth_model_delta['delta'])} AUROC. This supports the idea that cross-model depth-contribution calibration can carry signal.</div><div class="callout bad"><strong>Decision:</strong> do not adopt NRM as the white-box method yet. The positive observation fails the leave-dataset-out robustness check; the richer lens family version consistently trails its IU baseline; and final-layer NLL remains much stronger. Freeze these exact variants and test them on a new fully crossed model×dataset capture without retuning.</div></section>

<section><h2>7. Validation, limitations, and reproducibility</h2><ul><li>Status remains <strong>{html.escape(validation['status'])}</strong>: corrected live Gate B and the architecture pilot are incomplete.</li><li>The original v2 primary comparison is not replaced or reinterpreted.</li><li>Seven architecture cells are GSM8K-only, so dataset and architecture effects are not orthogonally estimable.</li><li>All new NRM variants are label-free at fit time, but the research hypothesis is post-v2 and therefore retrospective.</li><li>Confidence intervals resample problem groups within each cell; Holm-adjusted Wilcoxon tests are low-power support only.</li></ul>
<p class="muted">Base: <code>{html.escape(definition['base_result'])}</code> · score freeze: <code>{html.escape(score_freeze['fit_complete_sha256'])}</code>. The report embeds every SVG and uses no network assets; separate SVGs remain in <code>figures/</code> for audit.</p></section>
</main></body></html>"""
    report_path = results / "REPORT.html"
    report_path.write_text(html_text)
    generated = {
        "REPORT.html": sha256_file(report_path),
        **{
            str(path.relative_to(results)): sha256_file(path)
            for path in sorted(figures.glob("*.svg"))
        },
    }
    inputs = {
        name: sha256_file(results / name)
        for name in (
            "RUN_DEFINITION.json", "FIT_COMPLETE.json", "SCORE_FREEZE_MANIFEST.json",
            "validation_status.json", "headline_summary.csv", "paired_comparisons.csv",
            "per_cell_metrics.csv", "cohort_summary.csv", "calibration_diagnostics.csv",
            "bootstrap_draw_manifest.json",
        )
    }
    manifest = {
        "version": VERSION,
        "self_contained": True,
        "network_assets": False,
        "semantic_tables": True,
        "preformatted_report_body": False,
        "report_generator_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": inputs,
        "generated": generated,
    }
    (results / "REPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.results_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
