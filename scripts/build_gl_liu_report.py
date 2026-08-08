"""Build the standalone GL-LIU v1 advisor report.

The report reads only frozen CSV artifacts from
``results/ours_only_localization_v1``. It does not refit a score, open a cache,
or modify any experiment artifact.
"""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "ours_only_localization_v1"
OUTPUT = RESULT_DIR / "REPORT.html"

SYSTEM_LABELS = {
    "ours_only": "GL-LIU v1",
    "mindgap_control": "Mind the Gap control",
    "mindgap_detector_ours_locator": "Mind the Gap detector + GL-LIU locator",
}
SYSTEM_COLORS = {
    "ours_only": "#0f766e",
    "mindgap_control": "#94a3b8",
    "mindgap_detector_ours_locator": "#f59e0b",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULT_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def pp(value: float, digits: int = 2) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{100.0 * value:.{digits}f} pp"


def esc(value: object) -> str:
    return html.escape(str(value))


def macro(rows: list[dict[str, str]], systems: list[str]) -> dict[str, dict[str, float]]:
    metrics = ("f1", "sla", "sla_tol1", "acc_correct")
    result: dict[str, dict[str, float]] = {}
    for system in systems:
        selected = [row for row in rows if row["system"] == system]
        result[system] = {
            metric: mean(float(row[metric]) for row in selected) for metric in metrics
        }
    return result


def comparison_table(values: dict[str, dict[str, float]]) -> str:
    ordered = ["mindgap_control", "mindgap_detector_ours_locator", "ours_only"]
    body = []
    for system in ordered:
        item = values[system]
        emphasis = ' class="leader"' if system == "ours_only" else ""
        body.append(
            f"<tr{emphasis}><th>{esc(SYSTEM_LABELS[system])}</th>"
            f"<td>{pct(item['f1'])}</td><td>{pct(item['sla'])}</td>"
            f"<td>{pct(item['sla_tol1'])}</td><td>{pct(item['acc_correct'])}</td></tr>"
        )
    return (
        "<div class='table-scroll'><table><thead><tr><th>System</th>"
        "<th>PB-F1</th><th>Exact</th><th>Within ±1 step</th>"
        "<th>Clean accuracy</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def f1_svg(rows: list[dict[str, str]]) -> str:
    by_cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    split_by_cell: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["model"], row["subset"])
        by_cell[key][row["system"]] = float(row["f1"])
        split_by_cell[key] = row["split"]

    cells = sorted(by_cell, key=lambda k: (k[0], ["gsm8k", "math", "olympiadbench", "omnimath"].index(k[1])))
    width, height = 1120, 500
    left, right, top, bottom = 74, 24, 45, 118
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = 0.18, 0.37
    systems = ["mindgap_control", "mindgap_detector_ours_locator", "ours_only"]
    group_w = plot_w / len(cells)
    bar_w = min(26, group_w / 4.3)

    items = [
        f"<svg class='chart-svg' viewBox='0 0 {width} {height}' role='img' "
        "aria-label='ProcessBench F1 by model and dataset'>"
    ]
    for tick in (0.20, 0.25, 0.30, 0.35):
        y = top + (ymax - tick) / (ymax - ymin) * plot_h
        items.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' class='grid'/>")
        items.append(f"<text x='{left-12}' y='{y+5:.1f}' text-anchor='end' class='axis'>{tick*100:.0f}%</text>")

    for index, cell in enumerate(cells):
        center = left + group_w * (index + 0.5)
        for offset, system in enumerate(systems):
            value = by_cell[cell][system]
            x = center + (offset - 1) * (bar_w + 4) - bar_w / 2
            y = top + (ymax - value) / (ymax - ymin) * plot_h
            h = top + plot_h - y
            items.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' "
                f"rx='3' fill='{SYSTEM_COLORS[system]}'><title>{esc(SYSTEM_LABELS[system])}: {pct(value)}</title></rect>"
            )
        model = "4B" if cell[0] == "qwen3_4b" else "8B"
        dataset = {"olympiadbench": "Olympiad", "omnimath": "OmniMath"}.get(cell[1], cell[1].upper())
        split = "DEV" if split_by_cell[cell] == "development" else "CONFIRM"
        items.append(f"<text x='{center:.1f}' y='{top+plot_h+25}' text-anchor='middle' class='label'>{esc(dataset)}</text>")
        items.append(f"<text x='{center:.1f}' y='{top+plot_h+43}' text-anchor='middle' class='sub-label'>{model} · {split}</text>")

    items.append(f"<line x1='{left}' x2='{width-right}' y1='{top+plot_h}' y2='{top+plot_h}' class='axis-line'/>")
    legend_x = left
    for system in systems:
        items.append(f"<rect x='{legend_x}' y='{height-26}' width='13' height='13' rx='2' fill='{SYSTEM_COLORS[system]}'/>")
        items.append(f"<text x='{legend_x+20}' y='{height-15}' class='legend'>{esc(SYSTEM_LABELS[system])}</text>")
        legend_x += {"mindgap_control": 220, "mindgap_detector_ours_locator": 340, "ours_only": 160}[system]
    items.append("</svg>")
    return "".join(items)


def ranking_bars(rows: list[dict[str, str]], value_key: str, label_key: str, limit: int) -> str:
    chosen = rows[:limit]
    values = [float(row[value_key]) for row in chosen]
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-9)
    pieces = ["<div class='rank-list'>"]
    for row, value in zip(chosen, values):
        width = 30 + 70 * (value - lo) / span
        label = row[label_key].replace("answer_", "").replace("token_", "").replace("__", " / ")
        pieces.append(
            "<div class='rank-row'>"
            f"<div class='rank-label'>{esc(label)}</div>"
            f"<div class='rank-track'><span style='width:{width:.1f}%'></span></div>"
            f"<div class='rank-value'>{value:.4f}</div></div>"
        )
    pieces.append("</div>")
    return "".join(pieces)


def per_cell_table(rows: list[dict[str, str]]) -> str:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["model"], row["subset"])][row["system"]] = row
    order = sorted(grouped, key=lambda k: (k[0], ["gsm8k", "math", "olympiadbench", "omnimath"].index(k[1])))
    body = []
    for key in order:
        systems = grouped[key]
        ours = float(systems["ours_only"]["f1"])
        control = float(systems["mindgap_control"]["f1"])
        body.append(
            "<tr>"
            f"<th>{esc(key[0].replace('_', '-').upper())}</th><td>{esc(key[1])}</td>"
            f"<td>{esc(systems['ours_only']['split'])}</td><td>{pct(control)}</td>"
            f"<td><strong>{pct(ours)}</strong></td><td class='positive'>{pp(ours-control)}</td>"
            f"<td>{pct(float(systems['ours_only']['sla']))}</td>"
            f"<td>{pct(float(systems['ours_only']['acc_correct']))}</td></tr>"
        )
    return (
        "<div class='table-scroll'><table><thead><tr><th>Model</th><th>Dataset</th>"
        "<th>Split</th><th>Mind Gap F1</th><th>GL-LIU F1</th><th>Δ F1</th>"
        "<th>GL-LIU exact</th><th>GL-LIU clean</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def build() -> str:
    final_rows = read_csv("final_systems_per_cell.csv")
    detector_rows = read_csv("development_detector_ranking.csv")
    locator_rows = read_csv("development_locator_ranking.csv")
    systems = ["mindgap_control", "mindgap_detector_ours_locator", "ours_only"]
    all_macro = macro(final_rows, systems)
    non_selection = [row for row in final_rows if not (row["model"] == "qwen3_4b" and row["subset"] in {"gsm8k", "math"})]
    confirmation_macro = macro(non_selection, systems)
    ours = all_macro["ours_only"]
    control = all_macro["mindgap_control"]

    cards = "".join(
        f"<article class='metric'><div class='metric-name'>{esc(name)}</div>"
        f"<div class='metric-value'>{pct(ours[key])}</div>"
        f"<div class='metric-delta'>{pp(ours[key]-control[key])} vs Mind the Gap</div></article>"
        for name, key in [
            ("ProcessBench F1", "f1"),
            ("Exact localization", "sla"),
            ("Within ±1 step", "sla_tol1"),
            ("Clean accuracy", "acc_correct"),
        ]
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GL-LIU v1 — Advisor Research Brief</title>
<style>
:root {{ --ink:#172033; --muted:#5f6b7a; --paper:#f7f8fa; --card:#ffffff; --line:#dce2e8; --teal:#0f766e; --teal2:#0d9488; --amber:#f59e0b; --soft:#ecf7f5; --blue:#0f3d68; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font:16px/1.62 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
a {{ color:var(--teal); }}
.hero {{ padding:72px 28px 56px; color:white; background:linear-gradient(135deg,#10283d 0%,#0f3d68 52%,#0f766e 100%); }}
.wrap {{ width:min(1160px,calc(100% - 40px)); margin:0 auto; }}
.eyebrow {{ letter-spacing:.13em; text-transform:uppercase; font-size:.78rem; font-weight:750; color:#9de3d8; }}
h1 {{ margin:.18em 0 .12em; font-size:clamp(2.5rem,6vw,5.4rem); line-height:1.02; letter-spacing:-.045em; }}
.subtitle {{ max-width:820px; margin:18px 0 0; font-size:clamp(1.05rem,2vw,1.35rem); color:#d8e7ee; }}
.hero-meta {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:28px; }}
.chip {{ border:1px solid #ffffff38; border-radius:999px; padding:7px 12px; background:#ffffff12; font-size:.86rem; }}
nav {{ position:sticky; top:0; z-index:3; overflow:auto; background:#fffffff0; backdrop-filter:blur(14px); border-bottom:1px solid var(--line); }}
nav .wrap {{ display:flex; gap:22px; padding:11px 0; white-space:nowrap; }}
nav a {{ color:var(--muted); text-decoration:none; font-size:.9rem; font-weight:650; }}
main {{ padding:38px 0 80px; }}
section {{ margin:0 0 58px; scroll-margin-top:72px; }}
h2 {{ margin:0 0 14px; font-size:clamp(1.65rem,3vw,2.35rem); letter-spacing:-.025em; line-height:1.2; }}
h3 {{ margin:26px 0 8px; font-size:1.18rem; }}
.lead {{ max-width:850px; color:var(--muted); font-size:1.08rem; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:24px 0; }}
.metric,.card,.stage,.callout {{ background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:0 9px 32px #1831450b; }}
.metric {{ padding:18px; }}
.metric-name {{ color:var(--muted); font-size:.82rem; font-weight:700; text-transform:uppercase; letter-spacing:.055em; }}
.metric-value {{ font-size:2.15rem; font-weight:780; letter-spacing:-.035em; margin:3px 0; }}
.metric-delta,.positive {{ color:var(--teal); font-weight:700; }}
.stage-grid {{ display:grid; grid-template-columns:1fr 70px 1fr; gap:14px; align-items:stretch; margin-top:24px; }}
.stage {{ padding:24px; }}
.stage-number {{ display:inline-grid; place-items:center; width:34px; height:34px; border-radius:50%; color:white; background:var(--teal); font-weight:800; }}
.arrow {{ display:grid; place-items:center; color:var(--teal); font-size:2.3rem; }}
.formula {{ overflow:auto; margin:14px 0; padding:16px 18px; background:#10283d; color:#effaf8; border-radius:12px; font:1rem/1.7 ui-monospace,SFMono-Regular,Menlo,monospace; }}
.callout {{ padding:22px 24px; border-left:5px solid var(--teal); }}
.callout.warning {{ border-left-color:var(--amber); background:#fffaf0; }}
.callout strong {{ display:block; margin-bottom:4px; }}
.chart-card {{ padding:18px; background:white; border:1px solid var(--line); border-radius:16px; overflow:hidden; }}
.chart-svg {{ width:100%; height:auto; min-width:760px; }}
.grid {{ stroke:#e5e9ed; stroke-width:1; }} .axis-line {{ stroke:#9aa6b2; }}
.axis,.label,.sub-label,.legend {{ fill:#536172; font-family:system-ui,sans-serif; }}
.axis,.sub-label {{ font-size:13px; }} .label {{ font-size:14px; font-weight:650; }} .legend {{ font-size:14px; }}
.table-scroll {{ overflow:auto; border:1px solid var(--line); border-radius:14px; background:white; }}
table {{ width:100%; border-collapse:collapse; min-width:760px; }}
th,td {{ padding:12px 14px; border-bottom:1px solid var(--line); text-align:right; vertical-align:top; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3) {{ text-align:left; }}
thead th {{ background:#eef3f6; color:#405064; font-size:.78rem; letter-spacing:.045em; text-transform:uppercase; }}
tbody tr:last-child th,tbody tr:last-child td {{ border-bottom:0; }} .leader {{ background:var(--soft); }}
.rank-list {{ display:grid; gap:11px; margin-top:18px; }}
.rank-row {{ display:grid; grid-template-columns:minmax(180px,280px) 1fr 68px; align-items:center; gap:12px; }}
.rank-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font: .84rem ui-monospace,SFMono-Regular,Menlo,monospace; color:#405064; }}
.rank-track {{ height:11px; background:#e8edf0; border-radius:99px; overflow:hidden; }}
.rank-track span {{ display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,var(--teal2),var(--teal)); }}
.rank-value {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:700; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
.card {{ padding:22px; }}
.status {{ display:inline-block; border-radius:999px; padding:4px 9px; font-size:.76rem; font-weight:800; text-transform:uppercase; letter-spacing:.05em; }}
.supported {{ color:#08665f; background:#dff5f1; }} .fragile {{ color:#8a5800; background:#fff0cb; }}
.refs li {{ margin-bottom:8px; }}
footer {{ padding:28px 0 46px; color:var(--muted); border-top:1px solid var(--line); }}
@media(max-width:850px) {{ .metrics {{ grid-template-columns:1fr 1fr; }} .two-col {{ grid-template-columns:1fr; }} .stage-grid {{ grid-template-columns:1fr; }} .arrow {{ transform:rotate(90deg); }} }}
@media(max-width:520px) {{ .wrap {{ width:min(100% - 24px,1160px); }} .hero {{ padding:52px 12px 42px; }} .metrics {{ grid-template-columns:1fr; }} .rank-row {{ grid-template-columns:1fr 58px; }} .rank-track {{ grid-column:1 / -1; grid-row:2; }} }}
@media print {{ nav {{ display:none; }} body {{ background:white; }} .hero {{ padding:34px 0; }} section {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
  <div class="eyebrow">Advisor research brief · frozen 8 August 2026</div>
  <h1>GL-LIU v1</h1>
  <p class="subtitle">Global–Local Laplacian IU-PCR: one spectral system for deciding whether a reasoning trace contains an error and locating its first failure.</p>
  <div class="hero-meta"><span class="chip">One model generation</span><span class="chip">Token statistics</span><span class="chip">No step-wise feature construction</span><span class="chip">Calibrated unsupervised scoring</span></div>
</div></header>
<nav><div class="wrap"><a href="#result">Result</a><a href="#method">Method</a><a href="#evidence">Evidence</a><a href="#competitors">Competitors</a><a href="#limits">Limits</a><a href="#next">Next step</a></div></nav>
<main class="wrap">
<section id="result">
  <div class="eyebrow">Headline result</div><h2>Our system replaces Mind the Gap end to end</h2>
  <p class="lead">Under the same repeated ProcessBench calibration protocol, GL-LIU v1 has higher F1 in every evaluated model/dataset cell. The largest confirmed contribution is the global DUFS-LIU detector; the temporal locator is still provisional.</p>
  <div class="metrics">{cards}</div>
  <div class="callout"><strong>Six-cell non-selection result</strong>GL-LIU v1 scores {pct(confirmation_macro['ours_only']['f1'])} F1 versus {pct(confirmation_macro['mindgap_control']['f1'])} for Mind the Gap ({pp(confirmation_macro['ours_only']['f1']-confirmation_macro['mindgap_control']['f1'])}). These six cells include model transfer, but only OlympiadBench and OmniMath are new dataset families.</div>
</section>

<section id="method">
  <div class="eyebrow">What we do</div><h2>Separate global detection from local placement</h2>
  <p class="lead">A whole-trace score is better at answering “is there an error?” A continuous token curve is better at answering “where is it?” GL-LIU fits both without correctness labels, then calibrates one final threshold.</p>
  <div class="stage-grid">
    <article class="stage"><span class="stage-number">A</span><h3>Global detector</h3><p>29 full-trace token-statistic features use the frozen mixed contract. DUFS learns feature gates, constructs a sample graph, and a Laplacian penalty enters the two-dimensional IU-PCR solve.</p><div class="formula">R<sub>g</sub> = F<sub>g</sub>L<sub>g</sub>F<sub>g</sub><sup>T</sup>/n<br>w<sub>g</sub> = U[U<sup>T</sup>(C<sub>g</sub> + 0.1R̄<sub>g</sub>)U]<sup>−1</sup>U<sup>T</sup>ρ̂<br>q<sub>i</sub> = −w<sub>g</sub><sup>T</sup>g<sub>i</sub></div><p><strong>Output:</strong> one error-risk score per complete trace.</p></article>
    <div class="arrow">→</div>
    <article class="stage"><span class="stage-number">B</span><h3>Continuous locator</h3><p>Entropy, sliding variance, CUSUM, and spilled-energy signals remain on the original token grid. A temporal-chain Laplacian regularizes token IU-PCR.</p><div class="formula">x<sub>t</sub> = [H, SWVar(H), |CUSUM(H)|,<br>SWVar(S), |CUSUM(S)|]<sub>t</sub><br>r<sub>t</sub> = w<sub>l</sub><sup>T</sup>x<sub>t</sub>, &nbsp; t̂ = argmax<sub>t</sub> r<sub>t</sub></div><p><strong>Output:</strong> the most suspicious token. Step spans are used only later for benchmark scoring.</p></article>
  </div>
  <h3>Decision</h3><div class="formula">if q<sub>i</sub> ≤ τ: return “no error”<br>if q<sub>i</sub> &gt; τ: return predicted token t̂<sub>i</sub></div>
  <p>The threshold τ is selected on the calibration half and tested on the untouched half over 100 repeated splits.</p>
</section>

<section id="evidence">
  <div class="eyebrow">Measured evidence</div><h2>Comparison under one protocol</h2>
  <p><strong>ProcessBench F1</strong> combines exact localization on erroneous traces with correct abstention on clean traces. <strong>Exact</strong> means the predicted token falls in the annotated first wrong step. <strong>Within ±1</strong> also accepts a neighbouring step.</p>
  <h3>All eight cells</h3>{comparison_table(all_macro)}
  <h3>Six cells excluded from component selection</h3>{comparison_table(confirmation_macro)}
  <h3>F1 in every cell</h3><div class="chart-card">{f1_svg(final_rows)}</div>
  <h3>Per-cell audit table</h3>{per_cell_table(final_rows)}
</section>

<section id="components">
  <div class="eyebrow">Why these components</div><h2>Development selection and transfer diagnosis</h2>
  <div class="two-col">
    <article class="card"><span class="status supported">Supported</span><h3>Global DUFS-LIU detector</h3><p>Development AUROC ranks full-trace DUFS-LIU first. Global trace fusion also clearly beats maximum or top-5% aggregation of token-risk curves.</p>{ranking_bars(detector_rows, 'auroc', 'candidate', 7)}<p><strong>Transfer:</strong> mixed DUFS-LIU beats mixed ordinary IU-PCR in all eight cells, by about +0.22 AUROC percentage points on average.</p></article>
    <article class="card"><span class="status fragile">Fragile</span><h3>Temporal-LIU locator</h3><p>The frozen v1 locator wins the declared development macro, but the gain is driven by GSM8K and does not transfer as a universal advantage.</p>{ranking_bars(locator_rows, 'exact', 'candidate', 8)}<p><strong>Six non-selection cells:</strong> temporal LIU averages about 25.14% exact localization, below DUFS feature-graph IU at about 25.78%.</p></article>
  </div>
</section>

<section id="competitors">
  <div class="eyebrow">Comparison boundary</div><h2>What each competitor establishes</h2>
  <div class="table-scroll"><table><thead><tr><th>System</th><th>Role</th><th>Access</th><th>What the comparison tells us</th></tr></thead><tbody>
    <tr><th>Mind the Gap control</th><td>Immediate published-method control</td><td>One generation; token probability dynamics</td><td>Whether our whole end-to-end score and locator improve the reproduced ProcessBench pipeline.</td></tr>
    <tr><th>Mind the Gap detector + GL-LIU locator</th><td>Mechanism ablation</td><td>Mixed</td><td>Separates improvement from local placement from improvement in global error presence.</td></tr>
    <tr><th>Deployed U-PCR / IU-PCR</th><td>Spectral component controls</td><td>Our full-trace or token features</td><td>Whether the Laplacian and DUFS metric add value beyond the original two-component solve.</td></tr>
    <tr><th>Uniform-LIU / stable DUFS-LIU</th><td>Graph and feature-contract controls</td><td>Our full-trace features</td><td>Whether gains come from DUFS geometry, the graph alone, or transformed non-monotone features.</td></tr>
  </tbody></table></div>
  <div class="callout warning" style="margin-top:18px"><strong>Fairness note</strong>The Mind the Gap control receives the same split-local F1-optimized threshold as GL-LIU. This is a fair comparison of scores and locators under one ProcessBench protocol. It is not a reproduction of the paper's original Neyman–Pearson decision operating point.</div>
  <div class="callout warning" style="margin-top:18px"><strong>Current competitor scope</strong>Mind the Gap is the only external published method measured in this exact run. U-PCR, IU-PCR, uniform-LIU, and stable DUFS-LIU are internal mechanism controls. This result is not yet a complete state-of-the-art benchmark.</div>
</section>

<section id="limits">
  <div class="eyebrow">Scientific claim boundary</div><h2>What is confirmed—and what is not</h2>
  <div class="two-col">
    <article class="card"><h3>We can say</h3><ul><li>GL-LIU uses one generation and token statistics.</li><li>No reasoning-step boundary constructs any score.</li><li>Under the shared protocol, it improves F1 in all eight cells.</li><li>The global DUFS-LIU contribution is small but consistent against mixed IU-PCR.</li><li>Native moving-window curves contain useful localization information.</li></ul></article>
    <article class="card"><h3>We cannot yet say</h3><ul><li>The system is fully label-free: labels choose components and calibrate τ.</li><li>The temporal Laplacian universally improves localization.</li><li>Eight cells are eight independent datasets; there are four dataset families.</li><li>The mixed feature contract alone caused the improvement.</li><li>The reported Mind the Gap number is its original deployment policy.</li></ul></article>
  </div>
</section>

<section id="next">
  <div class="eyebrow">Registered next step</div><h2>Confirm the locator without tuning it again here</h2>
  <div class="callout"><strong>External validation</strong>Freeze the global mixed DUFS-LIU detector. On at least one new dataset family—and preferably a new model family—compare the frozen temporal locator with ordinary token IU-PCR and DUFS feature-graph IU-PCR. Use identical generations and telemetry for every method. Report AUROC, exact and ±1 localization, clean abstention, ProcessBench F1, runtime, and each independent family separately.</div>
  <p>A completely label-free deployment is a separate problem. It requires either a threshold fixed on an external calibration source or a new label-free abstention rule.</p>
  <h3>Primary sources</h3><ul class="refs"><li>Tenzer et al., <em>Crowdsourcing Regression: A Spectral Approach</em>, AISTATS 2022.</li><li>Lindenbaum et al., <em>Differentiable Unsupervised Feature Selection based on a Gated Laplacian</em>, NeurIPS 2021.</li><li>Frozen run definition and score tables in <code>results/ours_only_localization_v1/</code>.</li></ul>
</section>
</main>
<footer><div class="wrap">GL-LIU v1 · generated from frozen CSV artifacts by <code>scripts/build_gl_liu_report.py</code> · no experiment was refit while building this page.</div></footer>
</body></html>"""
    return html_text


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
