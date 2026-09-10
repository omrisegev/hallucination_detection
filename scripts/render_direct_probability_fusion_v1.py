"""Render the direct probability fusion benchmark as one self-contained HTML file."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "direct_probability_fusion_v1"

NAMES = {
    "entropy": "Token Entropy",
    "rank_equal": "Direct Probability Fusion - Equal Weights",
    "rank_iu": "Direct Probability Fusion - IU-PCR",
    "rank_joint_lw": "Direct Probability Fusion - Joint Shrinkage",
    "mindgap_paper_locator_common_gate": "Mind the Gap Locator - Common Gate",
    "varentropy": "Token Varentropy",
    "historical_iu_pcr": "Historical IU-PCR",
}
COLORS = {
    "entropy": "#64748b",
    "rank_equal": "#0ea5e9",
    "rank_iu": "#2563eb",
    "rank_joint_lw": "#7c3aed",
    "mindgap_paper_locator_common_gate": "#f59e0b",
    "varentropy": "#14b8a6",
    "historical_iu_pcr": "#dc2626",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def esc(value) -> str:
    return html.escape(str(value))


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def f4(value: float) -> str:
    return f"{float(value):.4f}"


def legend(keys: list[str]) -> str:
    items = "".join(
        f'<span class="legend-item"><i style="background:{COLORS[key]}"></i>{esc(NAMES[key])}</span>'
        for key in keys
    )
    return f'<div class="legend">{items}</div>'


def grouped_bars(
    categories: list[str],
    series: list[tuple[str, list[float]]],
    *,
    maximum: float,
    percent: bool,
    title: str,
) -> str:
    width = 900
    height = 330
    left, right, top, bottom = 70, 20, 42, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / max(len(categories), 1)
    bar_w = min(32.0, group_w * 0.78 / max(len(series), 1))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h - plot_h * tick / 5
        label = f"{100 * value:.0f}%" if percent else f"{value:.2f}"
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{label}</text>')
    for ci, category in enumerate(categories):
        center = left + (ci + 0.5) * group_w
        parts.append(f'<text x="{center:.1f}" y="{height-42}" text-anchor="middle" class="axis-label">{esc(category)}</text>')
        start = center - bar_w * len(series) / 2
        for si, (key, values) in enumerate(series):
            value = max(0.0, min(float(values[ci]), maximum))
            h = plot_h * value / maximum
            x = start + si * bar_w + 1
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{h:.1f}" fill="{COLORS[key]}"><title>{esc(NAMES[key])}: {pct(value) if percent else f4(value)}</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def historical_dumbbell(historical: dict) -> str:
    cells = list(historical["cells"])
    width = 1000
    row_h, top, bottom, left, right = 28, 48, 35, 270, 40
    height = top + bottom + row_h * len(cells)
    plot_w = width - left - right
    lo, hi = 0.30, 1.00

    def xpos(value: float) -> float:
        return left + (float(value) - lo) / (hi - lo) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="24-cell comparison">']
    for value in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        x = xpos(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top-18}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top-26}" text-anchor="middle" class="tick">{value:.1f}</text>')
    for index, cell in enumerate(cells):
        row = historical["cells"][cell]
        direct = float(row["auroc"]["rank_iu"])
        old = float(row["historical_iu_pcr"])
        y = top + index * row_h
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="cell-label">{esc(cell)}</text>')
        parts.append(f'<line x1="{xpos(old):.1f}" y1="{y:.1f}" x2="{xpos(direct):.1f}" y2="{y:.1f}" class="connector"/>')
        parts.append(f'<circle cx="{xpos(old):.1f}" cy="{y:.1f}" r="5" fill="{COLORS["historical_iu_pcr"]}"><title>{esc(NAMES["historical_iu_pcr"])}: {old:.4f}</title></circle>')
        parts.append(f'<circle cx="{xpos(direct):.1f}" cy="{y:.1f}" r="5" fill="{COLORS["rank_iu"]}"><title>{esc(NAMES["rank_iu"])}: {direct:.4f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def write_csvs(out: Path, localization: dict, historical: dict) -> None:
    with (out / "LOCALIZATION_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "display_name", "pb_all8", "pb_q4", "pb_q8", "prm_within", "prm_pooled", "prmscore_q08", "coverage"])
        for key, row in localization["methods"].items():
            writer.writerow([key, NAMES[key], row["pb_all8"], row["pb_q4"], row["pb_q8"], row["prm_within"], row["prm_pooled"], row["prmscore_q08"], row["coverage"]])
        for key, row in localization["comparators"].items():
            writer.writerow([key, NAMES[key], row["pb_all8"], row["pb_q4"], row["pb_q8"], "", "", "", row["coverage"]])
    with (out / "HISTORICAL_24_CELLS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "domain", "n_answers", "historical_iu_pcr", *localization_method_keys(historical)])
        keys = localization_method_keys(historical)
        for cell, row in historical["cells"].items():
            writer.writerow([cell, row["domain"], row["n_answers"], row["historical_iu_pcr"], *[row["auroc"][key] for key in keys]])


def localization_method_keys(historical: dict) -> list[str]:
    first = next(iter(historical["cells"].values()))
    return [key for key in ("entropy", "varentropy", "rank_equal", "rank_iu", "rank_joint_lw") if key in first["auroc"]]


def render(localization: dict, historical: dict) -> str:
    loc_keys = ["entropy", "rank_equal", "rank_iu", "rank_joint_lw", "mindgap_paper_locator_common_gate"]
    all_loc = {**localization["methods"], **localization["comparators"]}
    pb_chart = grouped_bars(
        ["All 8", "Qwen3-4B", "Qwen3-8B"],
        [(key, [all_loc[key]["pb_all8"], all_loc[key]["pb_q4"], all_loc[key]["pb_q8"]]) for key in loc_keys],
        maximum=0.50,
        percent=True,
        title="ProcessBench macro F1",
    )
    prm_keys = ["entropy", "rank_equal", "rank_iu", "rank_joint_lw"]
    prm_chart = grouped_bars(
        ["Within-answer AUC", "Pooled AUC", "PRMScore"],
        [(key, [all_loc[key]["prm_within"], all_loc[key]["prm_pooled"], all_loc[key]["prmscore_q08"]]) for key in prm_keys],
        maximum=0.85,
        percent=False,
        title="PRMBench metrics",
    )
    hist_keys = ["historical_iu_pcr", "entropy", "rank_equal", "rank_iu", "rank_joint_lw"]
    hist_chart = grouped_bars(
        ["All 24", "QA 9", "Math 15"],
        [(key, [historical["macro"][key]["all24"], historical["macro"][key]["qa9"], historical["macro"][key]["math15"]]) for key in hist_keys],
        maximum=0.90,
        percent=False,
        title="Historical 24-cell AUROC",
    )

    best_pb_key = max(loc_keys, key=lambda key: all_loc[key]["pb_all8"])
    best_prm_key = max(prm_keys, key=lambda key: all_loc[key]["prm_within"])
    best_24_key = max(hist_keys, key=lambda key: historical["macro"][key]["all24"])

    loc_rows = []
    for key in loc_keys:
        row = all_loc[key]
        loc_rows.append(
            f'<tr><th>{esc(NAMES[key])}</th><td>{pct(row["pb_all8"])}</td><td>{pct(row["pb_q4"])}</td><td>{pct(row["pb_q8"])}</td>'
            + (f'<td>{f4(row["prm_within"])}</td><td>{f4(row["prm_pooled"])}</td><td>{f4(row["prmscore_q08"])}</td>' if key in localization["methods"] else '<td>n/a</td><td>n/a</td><td>n/a</td>')
            + f'<td>{pct(row["coverage"])}</td></tr>'
        )
    macro_rows = "".join(
        f'<tr><th>{esc(NAMES[key])}</th><td>{f4(historical["macro"][key]["all24"])}</td><td>{f4(historical["macro"][key]["qa9"])}</td><td>{f4(historical["macro"][key]["math15"])}</td></tr>'
        for key in hist_keys + ["varentropy"]
    )
    cell_rows = []
    for cell, row in historical["cells"].items():
        delta = float(row["auroc"]["rank_iu"]) - float(row["historical_iu_pcr"])
        cell_rows.append(
            f'<tr><th>{esc(cell)}</th><td>{esc(row["domain"])}</td><td>{int(row["n_answers"]):,}</td><td>{f4(row["historical_iu_pcr"])}</td><td>{f4(row["auroc"]["rank_iu"])}</td><td class="{("gain" if delta >= 0 else "loss")}">{delta:+.4f}</td><td>[{f4(row["direct_iu_minus_historical_iu_pcr_ci97_5"][0])}, {f4(row["direct_iu_minus_historical_iu_pcr_ci97_5"][1])}]</td><td>{f4(row["auroc"]["rank_joint_lw"])}</td></tr>'
        )

    pb_cell_names = sorted(localization["methods"]["entropy"]["pb_cells"])
    pb_cell_rows = "".join(
        "<tr><th>" + esc(cell) + "</th>" + "".join(
            f'<td>{pct(all_loc[key]["pb_cells"][cell])}</td>' for key in loc_keys
        ) + "</tr>"
        for cell in pb_cell_names
    )
    diagnostic_rows = []
    for key in loc_keys:
        row = all_loc[key]
        fallback_count = sum(row.get("fallbacks", {}).values())
        diagnostic_rows.append(
            f'<tr><th>{esc(NAMES[key])}</th><td>{pct(row["pb_raw_exact"])}</td><td>{pct(row["pb_within_one"])}</td><td>{row["pb_early"]:,}</td><td>{row["pb_late"]:,}</td><td>{pct(row["pb_clean_accuracy"])}</td><td>{fallback_count:,}</td><td>{row.get("fit_seconds",0.0):.1f}s</td></tr>'
        )
    contrast_rows = []
    for key, row in localization["contrasts"].items():
        prm_delta = row.get("prm_within_delta")
        prm_text = "n/a" if prm_delta is None else (
            f'{prm_delta:+.4f} [{row["prm_within_ci_97_5"][0]:+.4f}, {row["prm_within_ci_97_5"][1]:+.4f}]'
        )
        contrast_rows.append(
            f'<tr><th>{esc(key)}</th><td>{100*row["pb_delta"]:+.2f} pp [{100*row["pb_ci_97_5"][0]:+.2f}, {100*row["pb_ci_97_5"][1]:+.2f}]</td><td>{prm_text}</td></tr>'
        )
    reference_names = {
        "token_varentropy": "Token Varentropy (frozen)",
        "token_feature_fusion_iu": "Token Feature Fusion - IU (frozen)",
        "window_feature_fusion_iu": "Window Feature Fusion - IU (frozen)",
        "supervised_math_prm": "Qwen2.5-Math-PRM-7B (supervised reference)",
    }
    reference_rows = []
    for key, row in localization["frozen_references"].items():
        if key == "supervised_math_prm":
            reference_rows.append(
                f'<tr><th>{esc(reference_names[key])}</th><td>n/a</td><td>n/a</td><td>n/a</td><td>{f4(row["prmscore"])}</td></tr>'
            )
        else:
            reference_rows.append(
                f'<tr><th>{esc(reference_names[key])}</th><td>{pct(row["pb_all8"])}</td><td>{f4(row["prm_within"])}</td><td>{f4(row["prm_pooled"])}</td><td>{f4(row["prmscore_q08"])}</td></tr>'
            )
    primary_historical = historical["contrasts"]["rank_iu_minus_historical_iu_pcr"]
    historical_runtime_rows = "".join(
        f'<tr><th>{esc(NAMES[key])}</th><td>{pct(historical["coverage"][key])}</td><td>{sum(historical.get("fallbacks", dict()).get(key, dict()).values()):,}</td><td>{historical.get("fit_seconds", dict()).get(key,0.0):.1f}s</td></tr>'
        for key in ("entropy", "rank_equal", "rank_iu", "rank_joint_lw", "varentropy")
    )

    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Direct Probability Fusion v1</title>
<style>
:root{{--ink:#132238;--muted:#526173;--paper:#f7f9fc;--card:#fff;--line:#dbe3ee;--accent:#2563eb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,Segoe UI,Arial,sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 22px 70px}} h1{{font-size:34px;margin:0 0 8px}} h2{{margin-top:42px;font-size:24px}} h3{{font-size:18px}}
.subtitle,.note{{color:var(--muted)}} .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}
.card,.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 5px 18px #17324d0b}}
.card strong{{display:block;font-size:22px;margin-top:6px}} .panel{{margin:16px 0;overflow:auto}} svg{{width:100%;min-width:720px;height:auto}}
.grid{{stroke:#dbe3ee;stroke-width:1}} .connector{{stroke:#94a3b8;stroke-width:2}} .tick,.axis-label,.cell-label{{fill:#526173;font-size:11px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 14px}} .legend-item{{display:inline-flex;align-items:center;gap:7px}} .legend-item i{{width:13px;height:13px;border-radius:3px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:9px 10px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} thead th{{background:#edf3fa;position:sticky;top:0}} .gain{{color:#047857;font-weight:700}} .loss{{color:#b91c1c;font-weight:700}}
.boundary{{border-left:5px solid #f59e0b;background:#fffbeb;padding:14px 18px;border-radius:8px}} code{{background:#eef2f7;padding:2px 5px;border-radius:4px}} @media(max-width:800px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Direct Probability Fusion v1</h1>
<p class="subtitle">Gray-box experiment. Direct top-15 next-token probabilities are fused without first reducing them to entropy. Generated from the frozen benchmark outputs.</p>
<div class="cards">
 <div class="card">Best ProcessBench score<strong>{esc(NAMES[best_pb_key])}: {pct(all_loc[best_pb_key]["pb_all8"])}</strong></div>
 <div class="card">Best PRMBench within-answer AUC<strong>{esc(NAMES[best_prm_key])}: {f4(all_loc[best_prm_key]["prm_within"])}</strong></div>
 <div class="card">Best historical 24-cell macro<strong>{esc(NAMES[best_24_key])}: {f4(historical["macro"][best_24_key]["all24"])}</strong></div>
</div>
<div class="boundary"><strong>How to read this report.</strong> All ProcessBench methods use the same frozen mean-entropy q=0.3 gate. The gate decides whether an answer contains an error. The fused token scores and top-10 step readout decide where the first error is. PRMBench has no no-error gate; its PRMScore uses the separately frozen q=0.8 cross-fold readout. Mind the Gap is replayed as a locator with the common ProcessBench gate, so this row is not its native paper protocol.</div>

<h2>1. One-answer localization</h2>
<p>Each answer is divided into tokens and reasoning steps. Fusion weights are learned only from that answer. No labels and no other answers enter the fusion fit.</p>
<div class="panel"><h3>ProcessBench: exact first-error or correct no-error decision</h3>{legend(loc_keys)}{pb_chart}</div>
<div class="panel"><table><thead><tr><th>Method</th><th>PB all 8</th><th>PB Qwen3-4B</th><th>PB Qwen3-8B</th><th>PRMB within</th><th>PRMB pooled</th><th>PRMScore</th><th>Coverage</th></tr></thead><tbody>{''.join(loc_rows)}</tbody></table></div>
<div class="panel"><h3>PRMBench: step ranking</h3>{legend(prm_keys)}{prm_chart}</div>
<div class="panel"><h3>ProcessBench by cell</h3><table><thead><tr><th>Cell</th>{''.join(f'<th>{esc(NAMES[key])}</th>' for key in loc_keys)}</tr></thead><tbody>{pb_cell_rows}</tbody></table></div>
<div class="panel"><h3>Location diagnostics, failures and measured fit time</h3><table><thead><tr><th>Method</th><th>Raw exact peak</th><th>Within one step</th><th>Early peaks</th><th>Late peaks</th><th>Clean accuracy</th><th>Fallbacks</th><th>Fit time</th></tr></thead><tbody>{''.join(diagnostic_rows)}</tbody></table></div>
<div class="panel"><h3>Pre-declared paired contrasts (97.5% confidence intervals)</h3><table><thead><tr><th>Contrast ID</th><th>ProcessBench difference</th><th>PRMB within-answer difference</th></tr></thead><tbody>{''.join(contrast_rows)}</tbody></table></div>
<div class="panel"><h3>Frozen continuity and supervised references</h3><table><thead><tr><th>Reference</th><th>PB all 8</th><th>PRMB within</th><th>PRMB pooled</th><th>PRMScore</th></tr></thead><tbody>{''.join(reference_rows)}</tbody></table></div>

<h2>2. Complete-answer hallucination detection</h2>
<p>For every historical cell, each answer becomes one row. The same top-15 probability ranks become the fusion columns. The fixed primary comparison is Direct Probability Fusion - IU-PCR versus Historical IU-PCR.</p>
<div class="panel"><h3>Macro AUROC</h3>{legend(hist_keys)}{hist_chart}</div>
<div class="panel"><table><thead><tr><th>Method</th><th>All 24</th><th>QA 9</th><th>Math 15</th></tr></thead><tbody>{macro_rows}</tbody></table></div>
<div class="panel"><h3>Historical coverage, fallbacks and fit time</h3><table><thead><tr><th>Method</th><th>Coverage</th><th>Fallbacks</th><th>Fusion fit time</th></tr></thead><tbody>{historical_runtime_rows}</tbody></table></div>
<div class="boundary"><strong>Primary matched difference:</strong> Direct Probability Fusion - IU-PCR minus Historical IU-PCR = {primary_historical["delta"]:+.4f}. Hierarchical 97.5% confidence interval: [{primary_historical["hierarchical_group_ci97_5"][0]:+.4f}, {primary_historical["hierarchical_group_ci97_5"][1]:+.4f}]. The bootstrap resamples canonical problem groups inside each cell and cells for the macro.</div>
<div class="panel"><h3>All 24 cells: fixed primary pair</h3>{legend(["historical_iu_pcr","rank_iu"])}{historical_dumbbell(historical)}</div>
<div class="panel"><table><thead><tr><th>Cell</th><th>Domain</th><th>Answers</th><th>Historical IU-PCR</th><th>Direct Probability Fusion - IU-PCR</th><th>Difference</th><th>Difference CI 97.5%</th><th>Direct Probability Fusion - Joint Shrinkage</th></tr></thead><tbody>{''.join(cell_rows)}</tbody></table></div>

<h2>3. What came from Claude's work</h2>
<p>The fixed mean-entropy q=0.3 gate isolates location quality and prevents the old GMM gate from hiding valid peaks. Joint Shrinkage transfers Claude's useful Joint covariance idea into a closed-form IU fit. It is a supporting method because the direct probability ranks do not have the original feature-stream grouping. BOCPD is excluded because it did not improve the frozen localization task.</p>

<h2>4. Limits</h2>
<p>This is development evidence on previously studied datasets. K=15 and the top-10 token readout were frozen before scoring. The historical IU-PCR reference is exactly <code>mixed_v2 / full / iu_pcr</code>; the similarly named U-PCR plus sign heuristic is not used. DEEM and white-box signals are outside this run.</p>
</main></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    out = args.results.resolve()
    localization = read_json(out / "LOCALIZATION.json")
    historical = read_json(out / "HISTORICAL_24.json")
    write_csvs(out, localization, historical)
    report = render(localization, historical)
    (out / "REPORT.html").write_text(report, encoding="utf-8")
    print(out / "REPORT.html")


if __name__ == "__main__":
    main()
