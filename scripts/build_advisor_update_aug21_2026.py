#!/usr/bin/env python3
"""Build the short advisor email, results map and four research briefs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/meetings/advisor_update_aug21_2026"
EMAIL = ROOT / "docs/meetings/Advisor_Update_Aug21_2026.md"

METHOD_CSV = ROOT / "results/method_comparison_table1.csv"
CORE_CSV = ROOT / "results/frozen_24cell_benchmark/headline_summary.csv"
BENCHMARK_STANDING = ROOT / "results/BENCHMARK_STANDING.md"
NRM_JSON = ROOT / "results/neutral_residual_mode_prmbench_v1/RESULT.json"
DEPENDENCE_CSV = ROOT / "results/dependency_fusion_study/contrasts.csv"
DEPENDENCE_ARMS_CSV = ROOT / "results/dependency_fusion_study/arm_summary.csv"
CSTG_JSON = ROOT / "results/global_contextual_stg_router_diagnostic_v1/DECISION.json"
A4_REPORT = ROOT / "results/automatic_group_free_phase_a4_v1/REPORT.md"
A5_REPORT = ROOT / "results/automatic_group_free_phase_a5_v1/REPORT.md"
A6_JSON = ROOT / "results/automatic_group_free_phase_a6_s0a_v1/A6_S0A_BOUNDARY.json"
PROGRESS = ROOT / "PROGRESS.md"
STATUS = ROOT / "docs/research_notes/research_status_consolidated_2026-08-19.md"
TENZER_DIGEST = ROOT / "papers/digests/crowdsourcing-regression-a-spectral-approach.md"
DEEM_DIGEST = ROOT / "papers/digests/unsupervised-ensemble-learning-through-deep-energy-based-models.md"
FAMILY_GRAPH_JSON = ROOT / "results/family_residual_graph_liu_v3/RESULT.json"
FAMILY_GRAPH_SYNTHESIS = ROOT / "results/family_residual_graph_liu_v3/SYNTHESIS.md"
FAMILY_GRAPH_CONTROLS_JSON = ROOT / "results/family_residual_graph_liu_v3/controls/RESULT.json"
FAMILY_GRAPH_PRM_JSON = ROOT / "results/family_residual_graph_liu_prmbench_v3/RESULT.json"
FAMILY_GRAPH_HLE_JSON = ROOT / "results/family_residual_graph_liu_hle_v3/RESULT.json"

WHITE_CSV = ROOT / "results/whitebox_vs_graybox_matched_v1/headline_summary.csv"
WHITE_PAIR_CSV = ROOT / "results/whitebox_vs_graybox_matched_v1/paired_comparisons.csv"
WHITE_AUDIT = ROOT / "results/whitebox_vs_graybox_matched_v1/AUDIT.json"
MULTISAMPLE_JSON = ROOT / "results/repgrid/phase15_followups.json"

# Certified reconstruction snapshot.  The signed releases live in the detached
# science worktree rather than this documentation checkout, so the packet must
# not acquire a mutable runtime dependency on that worktree.  These exact
# values were copied from the immutable A/B-certified release payloads named
# below.  LEASH and RAG now have their own certified snapshots.  The unified
# reporting certificate authenticates their source bindings without creating
# a cross-panel or cross-task estimand.
APPLICATION_RECONSTRUCTION = {
    "localization": {
        "status": "CERTIFIED",
        "release": "results/reconstruction_benchmark_v1/releases/2026-08-24_localization_v1/",
        "evaluation_certificate_payload_sha256": "ad2378954d530510aedc6fca1283460604b361eadaaaab9c988d3c9432b28a4c",
        "processbench": {
            "llama31_8b": {
                "reference": 0.3074396911703836,
                "candidate": 0.31320480146719376,
                "delta": 0.00576511029681015,
                "ci_low": -0.0022187927637086144,
                "ci_high": 0.014034576139930708,
            },
            "qwen3_4b": {
                "reference": 0.3115384618188499,
                "candidate": 0.314804609454725,
                "delta": 0.003266147635875072,
                "ci_low": -0.003264849174958881,
                "ci_high": 0.009996491600528887,
            },
            "qwen3_8b": {
                "reference": 0.30560235288708426,
                "candidate": 0.30936934825470497,
                "delta": 0.003766995367620707,
                "ci_low": -0.00270113593404443,
                "ci_high": 0.010111345714370817,
            },
        },
        "prmbench": {
            "fusion_auroc": 0.5988340566273779,
            "token_only_auroc": 0.6711781515467627,
            "auroc_delta": 0.07234409491938487,
            "auroc_ci_low": 0.06897782793345128,
            "auroc_ci_high": 0.0756929940874809,
            "fusion_auprc": 0.20869011184691225,
            "token_only_auprc": 0.25485485697917315,
            "auprc_delta": 0.046164745132260904,
            "auprc_ci_low": 0.043250134918749863,
            "auprc_ci_high": 0.04922203910841002,
            "supervised_prm_auroc": 0.7983220044539499,
            "supervised_prm_auprc": 0.4634832603965231,
        },
    },
    "prefix": {
        "status": "CERTIFIED",
        "release": "results/reconstruction_benchmark_v1/releases/2026-08-24_prefix_v1/",
        "evaluation_certificate_payload_sha256": "662dfa8ec233f45a1c510726928d285a7eb8e125f80401f5aa15d6dc91ed6d62",
        "budgets": {
            64: {
                "unified_auroc": 0.5629490566083293,
                "step272_auroc": 0.5955176048077226,
                "delta": 0.032568548199393366,
                "ci_low": 0.0034781731852584993,
                "ci_high": 0.06252538248677737,
            },
            256: {
                "unified_auroc": 0.6114099828794719,
                "step272_auroc": 0.6572248773466058,
                "delta": 0.04581489446713394,
                "ci_low": 0.014703364359100406,
                "ci_high": 0.07651129586461866,
                "vs_iu28_no_length_auprc_delta": 0.0184924371314672,
                "vs_iu28_no_length_auprc_ci_low": 0.002327573060248088,
                "vs_iu28_no_length_auprc_ci_high": 0.035166149952487485,
            },
            512: {"status": "METRIC_UNDEFINED_MISSING_REGISTERED_SUBSET", "missing_subset": "gsm8k"},
        },
    },
    "leash": {
        "status": "CERTIFIED",
        "fidelity": "paper-specified-partial",
        "release": "results/reconstruction_benchmark_v1/releases/2026-08-25_leash_v1/",
        "evaluation_certificate_file_sha256": "a98c4084e93d62b88227e1a78a16f86fdab071e39664440a6b609bf911ec3b88",
        "evaluation_certificate_payload_sha256": "3ce4fb378ea308bfdb132d5758fab967567209be3743fef8b6639a723b90af15",
        "actual_callback_stopping_observed": True,
        "ready_cells": 6,
        "blocked_cells": 2,
        "blocked_model": "mistralai/Mistral-7B-v0.1",
        "paper_exact_claim": False,
        "matched_accuracy_claim": False,
        "conceptual_objective_reproduced_as_equation": False,
        "datasets": {
            "aqua": {
                "scope": "equal_model_within_dataset",
                "cot_pass_at_1": 0.3175853018372703,
                "leash_pass_at_1": 0.21784776902887137,
                "pass_at_1_delta": -0.09973753280839892,
                "pass_at_1_delta_ci_low": -0.13910761154855641,
                "pass_at_1_delta_ci_high": -0.06295931758530196,
                "token_reduction": 0.45300007258086084,
                "token_reduction_ci_low": 0.42574140095017116,
                "token_reduction_ci_high": 0.47848829221657,
            },
            "gsm8k": {
                "scope": "equal_model_within_dataset",
                "cot_pass_at_1": 0.6255555555555556,
                "leash_pass_at_1": 0.35999999999999993,
                "pass_at_1_delta": -0.2655555555555557,
                "pass_at_1_delta_ci_low": -0.3055555555555555,
                "pass_at_1_delta_ci_high": -0.22333333333333333,
                "token_reduction": 0.3235232399538347,
                "token_reduction_ci_low": 0.30054100727781813,
                "token_reduction_ci_high": 0.34763118329346965,
            },
        },
        "overall": {
            "scope": "equal_dataset_after_equal_model",
            "pass_at_1_delta": -0.18264654418197732,
            "pass_at_1_delta_ci_low": -0.2120902230971129,
            "pass_at_1_delta_ci_high": -0.15477241907261594,
            "token_reduction": 0.38826165626734777,
            "token_reduction_ci_low": 0.3707788804483397,
            "token_reduction_ci_high": 0.40544379262415875,
        },
    },
    "rag": {
        "status": "CERTIFIED",
        "release": "results/reconstruction_benchmark_v1/releases/2026-08-25_rag_evidence_v1/",
        "evaluation_certificate_file_sha256": "03a7cd721894a3a8965da1cd1f3eaebdfc41ba0a59ef5c42342f14d2634c9845",
        "evaluation_certificate_payload_sha256": "cb241365b5ba31ab9a9fd8eed857e34dae5ca08ae4b3dbdb3d2fc2c71bea40a7",
        "panels": 7,
        "cross_panel_macro_computed": False,
        "scope": "retrospective_application_evidence",
        "historical_labels_opened": True,
        "ragtruth_test_auroc": {
            "answer": 0.7273659490201593,
            "sentence": 0.6891729751478315,
            "token": 0.6586875328346804,
            "data2txt_vs_qa_heterogeneous": True,
        },
        "gasp": {
            "scope": "local_400_response_sample",
            "auroc": 0.6708286136366572,
            "matched_iu_pcr_auroc": 0.6597166889910114,
            "delta": 0.01111192464564581,
            "ci_low": -0.012474012217862887,
            "ci_high": 0.03428946805218775,
            "superiority": False,
        },
        "lettuce": {
            "role": "supervised_ceiling",
            "f1": 0.7928994082840237,
        },
        "refchecker": {
            "settings": ("accurate_context", "noisy_context", "zero_context"),
            "settings_pooled": False,
            "claim_extraction": "OUT_OF_SCOPE",
            "nli_accuracy": {
                "accurate_context": 0.600650976464697,
                "noisy_context": 0.7619883040935672,
                "zero_context": 0.7336547152756855,
            },
            "fixed_binary_transfer_auroc": {
                "accurate_context": 0.6644595346250115,
                "noisy_context": 0.6401515938291364,
                "zero_context": 0.7505681474232213,
            },
        },
    },
    "unified_reporting": {
        "status": "CERTIFIED",
        "release": "results/reconstruction_benchmark_v1/derived/unified_reporting_v1_certified/",
        "ab_certificate_file_sha256": "55b3204c591e022ee5051c9b1d613271881641df0eb099e9f604b4f273e8a045",
        "ab_certificate_payload_sha256": "3b6811afd70bf2e5db028c6d523400f5865e077046be0b7fc44088e002bea945",
        "release_content_sha256": "593f5fcfd00928466ba0db98f01f4a77d0c5ef69183fc577fbfdc8b4ba86c29a",
        "authenticated_source_bindings": 9,
    },
}


STYLE = """
:root {
  color-scheme: light dark;
  --ink: #17243b;
  --muted: #5f6c80;
  --paper: #fbfcfe;
  --panel: #f3f6fb;
  --line: #d8e0eb;
  --blue: #2764c5;
  --blue-soft: #dce9ff;
  --teal: #177b77;
  --teal-soft: #d8f2ef;
  --amber: #aa6b00;
  --amber-soft: #fff0c9;
  --red: #a23b42;
  --red-soft: #f8dfe1;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 16px/1.58 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(980px, calc(100% - 32px)); margin: 0 auto; padding: 46px 0 64px; }
h1 { font-size: clamp(2rem, 5vw, 3.25rem); line-height: 1.05; margin: .35rem 0 1rem; letter-spacing: -.035em; }
h2 { font-size: 1.45rem; line-height: 1.2; margin: 2.1rem 0 .75rem; }
h3 { font-size: 1.08rem; line-height: 1.3; margin: 0 0 .45rem; }
p { margin: .45rem 0 .9rem; }
.eyebrow { color: var(--blue); font-size: .78rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
.lead { max-width: 780px; font-size: 1.13rem; color: var(--muted); }
.guide {
  border-left: 4px solid var(--blue);
  background: var(--panel);
  padding: 14px 18px;
  margin: 1.4rem 0 2rem;
}
.guide strong { color: var(--ink); }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.card {
  border: 1px solid var(--line);
  background: var(--panel);
  padding: 18px;
}
.card p:last-child { margin-bottom: 0; }
.tag {
  display: inline-block;
  color: var(--blue);
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
  margin-bottom: .55rem;
}
.equation {
  overflow-x: auto;
  border-left: 3px solid var(--teal);
  background: var(--teal-soft);
  color: #123a3a;
  font: 1rem/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
  padding: 12px 15px;
  margin: 1rem 0;
}
.equation small { display: block; font: .86rem/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin-top: 6px; }
.plot { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 18px 0 14px; margin: 1rem 0; }
.plot-title { font-weight: 700; margin-bottom: 2px; }
.plot-note { color: var(--muted); font-size: .88rem; margin-bottom: 13px; }
.bar-row { display: grid; grid-template-columns: minmax(145px, 1.5fr) minmax(170px, 3fr) 74px; gap: 10px; align-items: center; margin: 9px 0; }
.bar-label { font-size: .92rem; }
.bar-track { height: 18px; background: var(--blue-soft); position: relative; }
.bar-fill { height: 100%; background: var(--blue); min-width: 2px; }
.bar-fill.teal { background: var(--teal); }
.bar-fill.amber { background: var(--amber); }
.bar-value { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
.axis { display: flex; justify-content: space-between; margin: 7px 84px 0 calc(1.5 / 4.5 * 100%); color: var(--muted); font-size: .75rem; }
.delta-row { display: grid; grid-template-columns: minmax(165px, 1.6fr) minmax(180px, 3fr) 76px; gap: 10px; align-items: center; margin: 10px 0; }
.delta-track { height: 20px; background: var(--panel); position: relative; border: 1px solid var(--line); }
.delta-zero { position: absolute; left: 75%; top: -2px; bottom: -2px; width: 1px; background: var(--ink); }
.delta-neg, .delta-pos { position: absolute; top: 3px; bottom: 3px; }
.delta-neg { right: 25%; background: var(--red); }
.delta-pos { left: 75%; background: var(--teal); }
.interval-row { display: grid; grid-template-columns: minmax(185px, 1.7fr) minmax(210px, 3fr) 92px; gap: 10px; align-items: center; margin: 12px 0; }
.interval-track { height: 24px; background: var(--panel); position: relative; border: 1px solid var(--line); }
.interval-zero { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--ink); }
.interval-whisker { position: absolute; top: 11px; height: 2px; background: var(--teal); }
.interval-point { position: absolute; top: 6px; width: 10px; height: 10px; margin-left: -5px; border-radius: 50%; background: var(--teal); }
.interval-value { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
.result { color: var(--muted); font-size: .92rem; }
.callout { background: var(--amber-soft); color: #4c3507; padding: 13px 16px; margin: 1rem 0; }
.questions { border-top: 3px solid var(--blue); margin-top: 2.25rem; padding-top: 1rem; }
ul { padding-left: 1.2rem; }
li { margin: .4rem 0; }
details { margin-top: 2rem; color: var(--muted); font-size: .86rem; }
summary { cursor: pointer; color: var(--ink); font-weight: 700; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (max-width: 690px) {
  main { width: min(100% - 22px, 980px); padding-top: 28px; }
  .grid { grid-template-columns: 1fr; }
  .bar-row, .delta-row { grid-template-columns: 1fr 62px; gap: 5px 8px; }
  .interval-row { grid-template-columns: 1fr 82px; gap: 5px 8px; }
  .bar-track, .delta-track { grid-row: 2; grid-column: 1 / -1; }
  .interval-track { grid-row: 2; grid-column: 1 / -1; }
  .bar-value { grid-column: 2; grid-row: 1; }
  .interval-value { grid-column: 2; grid-row: 1; }
  .axis { display: none; }
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #edf3ff;
    --muted: #b4bfd0;
    --paper: #111723;
    --panel: #1a2332;
    --line: #344157;
    --blue: #78a9ff;
    --blue-soft: #243b61;
    --teal: #57c8c1;
    --teal-soft: #153b3a;
    --amber: #efbd5b;
    --amber-soft: #493816;
    --red: #ef8990;
    --red-soft: #4a2529;
  }
  .equation { color: #daf9f5; }
  .callout { color: #fff0c9; }
}
@media print {
  body { background: white; color: #17243b; }
  main { width: 100%; padding: 0; }
  .card, .guide, .plot { break-inside: avoid; }
  details { display: block; }
  details > * { display: block; }
}
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def one(rows: list[dict], **wanted):
    matches = [row for row in rows if all(str(row.get(k)) == str(v) for k, v in wanted.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {wanted}, found {len(matches)}")
    return matches[0]


def number(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.I | re.S)
    if not match:
        raise ValueError(f"Pattern not found: {pattern}")
    return float(match.group(1))


def percent_string(value: str) -> float:
    return float(value.rstrip("%")) / 100.0


def page(title: str, eyebrow: str, lead: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
  <div class="eyebrow">{html.escape(eyebrow)}</div>
  <h1>{html.escape(title)}</h1>
  <p class="lead">{lead}</p>
  {body}
</main>
</body>
</html>
"""


def score_plot(
    title: str,
    items: list[tuple[str, float, str]],
    lo: float,
    hi: float,
    note: str,
    axis_label: str,
    axis_precision: int = 2,
    value_precision: int = 4,
) -> str:
    rows = []
    for label, value, tone in items:
        width = max(0.0, min(100.0, 100.0 * (value - lo) / (hi - lo)))
        rows.append(
            f"""<div class="bar-row">
  <div class="bar-label">{html.escape(label)}</div>
  <div class="bar-track"><div class="bar-fill {tone}" style="width:{width:.2f}%" role="img" aria-label="{html.escape(label)}: {value:.4f}"></div></div>
  <div class="bar-value">{value:.{value_precision}f}</div>
</div>"""
        )
    return f"""<figure class="plot">
  <figcaption class="plot-title">{html.escape(title)}</figcaption>
  <div class="plot-note">{note}</div>
  {''.join(rows)}
  <div class="axis"><span>{lo:.{axis_precision}f}</span><span>{html.escape(axis_label)}</span><span>{hi:.{axis_precision}f}</span></div>
</figure>"""


def count_plot(title: str, items: list[tuple[str, int, str]], note: str) -> str:
    maximum = max(value for _, value, _ in items)
    rows = []
    for label, value, tone in items:
        rows.append(
            f"""<div class="bar-row">
  <div class="bar-label">{html.escape(label)}</div>
  <div class="bar-track"><div class="bar-fill {tone}" style="width:{100 * value / maximum:.2f}%" role="img" aria-label="{html.escape(label)}: {value:,} rows"></div></div>
  <div class="bar-value">{value:,}</div>
</div>"""
        )
    return f"""<figure class="plot">
  <figcaption class="plot-title">{html.escape(title)}</figcaption>
  <div class="plot-note">{note}</div>
  {''.join(rows)}
</figure>"""


def delta_plot(title: str, items: list[tuple[str, float]], note: str) -> str:
    rows = []
    for label, value in items:
        width = min(75.0 if value < 0 else 25.0, abs(value) / 8.0 * 100.0)
        cls = "delta-neg" if value < 0 else "delta-pos"
        rows.append(
            f"""<div class="delta-row">
  <div class="bar-label">{html.escape(label)}</div>
  <div class="delta-track"><div class="delta-zero"></div><div class="{cls}" style="width:{width:.2f}%" role="img" aria-label="{html.escape(label)}: {value:+.2f} AUROC percentage points"></div></div>
  <div class="bar-value">{value:+.2f}pp</div>
</div>"""
        )
    return f"""<figure class="plot">
  <figcaption class="plot-title">{html.escape(title)}</figcaption>
  <div class="plot-note">{note}</div>
  {''.join(rows)}
</figure>"""


def interval_plot(
    title: str,
    items: list[tuple[str, float, float, float]],
    lo: float,
    hi: float,
    note: str,
) -> str:
    if not lo < 0 < hi:
        raise ValueError("Interval plot domain must contain zero")

    def position(value: float) -> float:
        return 100.0 * (min(hi, max(lo, value)) - lo) / (hi - lo)

    zero = position(0.0)
    rows = []
    for label, point, low, high in items:
        left = position(low)
        right = position(high)
        rows.append(
            f"""<div class="interval-row">
  <div class="bar-label">{html.escape(label)}</div>
  <div class="interval-track" role="img" aria-label="{html.escape(label)}: {point:+.3f} percentage points; 95% interval {low:+.3f} to {high:+.3f}">
    <div class="interval-zero" style="left:{zero:.2f}%"></div>
    <div class="interval-whisker" style="left:{left:.2f}%;width:{right - left:.2f}%"></div>
    <div class="interval-point" style="left:{position(point):.2f}%"></div>
  </div>
  <div class="interval-value">{point:+.3f}pp</div>
</div>"""
        )
    return f"""<figure class="plot">
  <figcaption class="plot-title">{html.escape(title)}</figcaption>
  <div class="plot-note">{note}</div>
  {''.join(rows)}
  <div class="axis"><span>{lo:+.2f}pp</span><span>Change from IU-PCR</span><span>{hi:+.2f}pp</span></div>
</figure>"""


def sources(items: list[tuple[str, Path]]) -> str:
    missing = [str(path) for _, path in items if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source artifacts: " + ", ".join(missing))
    rows = "".join(
        f"<li><strong>{html.escape(label)}:</strong> <code>{html.escape(str(path.relative_to(ROOT)))}</code></li>"
        for label, path in items
    )
    return f"<details><summary>Sources and exact artifacts</summary><ul>{rows}</ul></details>"


def load_context() -> dict:
    historical = one(read_csv(METHOD_CSV), domain="MACRO_AVG", cell_key="MEAN_OF_DOMAIN_MEANS")
    core_rows = [r for r in read_csv(CORE_CSV) if r["metric"] == "auroc"]
    core = {r["method_key"]: float(r["cell_macro"]) for r in core_rows}
    standing_text = BENCHMARK_STANDING.read_text(encoding="utf-8")
    good6_match = re.search(r"\| GOOD_6 \|[^\n]*\*\*([0-9.]+)\*\*", standing_text)
    if not good6_match:
        raise ValueError("GOOD_6 reference missing from benchmark standing")

    nrm = read_json(NRM_JSON)
    dependence = one(read_csv(DEPENDENCE_CSV), contrast="H2_dependency_weights")
    sparse_dependence = one(read_csv(DEPENDENCE_CSV), contrast="H1_sparse_reliability")
    dependence_arms = {r["arm"]: r for r in read_csv(DEPENDENCE_ARMS_CSV)}
    router = read_json(CSTG_JSON)
    router_lr = one(router["exploratory_core_contrasts"], baseline="global_lr")
    router_gates = {row["gate"]: row["observed"] for row in router["gates"]}

    a4_text = A4_REPORT.read_text(encoding="utf-8")
    a5_text = A5_REPORT.read_text(encoding="utf-8")
    progress_text = PROGRESS.read_text(encoding="utf-8")
    status_text = STATUS.read_text(encoding="utf-8")
    a6 = read_json(A6_JSON)
    family_graph = read_json(FAMILY_GRAPH_JSON)
    family_graph_controls = read_json(FAMILY_GRAPH_CONTROLS_JSON)
    family_graph_prm = read_json(FAMILY_GRAPH_PRM_JSON)
    family_graph_hle = read_json(FAMILY_GRAPH_HLE_JSON)

    white_rows = {r["method"]: r for r in read_csv(WHITE_CSV)}
    white_pairs = {r["contrast"] + ":" + r["metric"]: r for r in read_csv(WHITE_PAIR_CSV)}
    white_audit = read_json(WHITE_AUDIT)
    multisample = read_json(MULTISAMPLE_JSON)

    context = {
        "historical_binary": percent_string(historical["PROD"]),
        "historical_continuous": percent_string(historical["CONT"]),
        "core": core,
        "good6": float(good6_match.group(1)),
        "nrm": nrm,
        "dep_delta_pp": 100.0 * float(dependence["mean_delta"]),
        "dep_ci_low_pp": 100.0 * float(dependence["ci95_low"]),
        "dep_ci_high_pp": 100.0 * float(dependence["ci95_high"]),
        "sparse_dep_delta_pp": 100.0 * float(sparse_dependence["mean_delta"]),
        "sparse_dep_ci_low_pp": 100.0 * float(sparse_dependence["ci95_low"]),
        "sparse_dep_ci_high_pp": 100.0 * float(sparse_dependence["ci95_high"]),
        "dependency_full_iu_auc": float(dependence_arms["full.iu_pcr"]["macro_auc"]),
        "dependency_deem_auc": float(dependence_arms["full.deem_deep_hard_ensemble"]["macro_auc"]),
        "router_delta_pp": 100.0 * float(router_lr["equal_family_delta"]),
        "router_ci_low_pp": 100.0 * float(router_lr["ci_low"]),
        "router_ci_high_pp": 100.0 * float(router_lr["ci_high"]),
        "router_primary_delta_pp": 100.0 * float(router_gates["gain_at_least_0.005"]),
        "router_primary_family_wins": int(router_gates["five_of_eight_family_wins"]),
        "a4_repeatability": number(a4_text, r"repeatability\s+\*\*([0-9.]+)\*\*"),
        "a4_external_corr": number(a4_text, r"external structural correlation\s+\*\*([0-9.]+)\*\*"),
        "a4_length_corr": number(a4_text, r"baseline has\s+Llama correlation\s+\*\*([0-9.]+)\*\*"),
        "a4_delta_pp": 100.0 * number(a4_text, r"observed delta is \*\*(-[0-9.]+)\*\*"),
        "a5_delta_pp": 100.0 * number(a5_text, r"mean (-[0-9.]+); 95% CI"),
        "nonmono_delta_pp": number(progress_text, r"deployable fused\s+macro is \*\*\+([0-9.]+)pp"),
        "a6": a6,
        "family_graph": family_graph,
        "family_graph_controls": family_graph_controls,
        "family_graph_prm": family_graph_prm,
        "family_graph_hle": family_graph_hle,
        "status_text": status_text,
        "gamma3_before": number(status_text, r"pooled cos\(gamma-3, g\*\) falls \+([0-9.]+)"),
        "gamma3_after": number(status_text, r"pooled cos\(gamma-3, g\*\) falls \+[0-9.]+ -> \+([0-9.]+)"),
        "white_rows": white_rows,
        "white_pairs": white_pairs,
        "white_audit": white_audit,
        "multisample_single_auc": float(multisample["f1_k_sweep"]["1"]["auc"]),
        "multisample_five_auc": float(multisample["f1_k_sweep"]["5"]["auc"]),
        "applications": APPLICATION_RECONSTRUCTION,
    }

    assert abs(context["core"]["iu_pcr"] - 0.7740628976408787) < 1e-12
    assert abs(context["good6"] - 0.7733) < 1e-8
    assert nrm["status"] == "PASS" and nrm["n"] == 6966
    assert white_audit["matched_row_ids_exact"] is True
    assert context["applications"]["localization"]["status"] == "CERTIFIED"
    assert context["applications"]["prefix"]["status"] == "CERTIFIED"
    assert context["applications"]["leash"]["status"] == "CERTIFIED"
    assert context["applications"]["leash"]["actual_callback_stopping_observed"] is True
    assert context["applications"]["leash"]["ready_cells"] == 6
    assert context["applications"]["leash"]["blocked_cells"] == 2
    assert context["applications"]["leash"]["paper_exact_claim"] is False
    assert context["applications"]["leash"]["matched_accuracy_claim"] is False
    assert context["applications"]["rag"]["status"] == "CERTIFIED"
    assert context["applications"]["rag"]["panels"] == 7
    assert context["applications"]["rag"]["cross_panel_macro_computed"] is False
    assert context["applications"]["rag"]["gasp"]["superiority"] is False
    assert context["applications"]["rag"]["refchecker"]["settings_pooled"] is False
    assert context["applications"]["unified_reporting"]["status"] == "CERTIFIED"
    assert context["applications"]["unified_reporting"]["authenticated_source_bindings"] == 9
    assert "S0b" in status_text and "scope" in status_text.lower()
    assert family_graph["version"] == "family-residual-graph-liu-v3-2026-08-23"
    assert family_graph["promotion_pass"] is False
    assert family_graph_prm["status"] == "FAIL" and family_graph_hle["status"] == "FAIL"
    return context


def render_fusion(c: dict) -> str:
    core = c["core"]
    nrm = c["nrm"]
    nrm_iu = nrm["metrics"]["iu"]["auroc"]
    nrm_score = nrm["metrics"]["nrm"]["auroc"]
    nrm_ci = nrm["paired_source_group_bootstrap"]

    body = f"""
<div class="guide">
  <strong>Short version.</strong> Historical L-SML showed that continuous inputs help. U-PCR later automated my subset at
  similar accuracy; the DUFS graph added almost nothing; Family-NRM gave one small result that needs manual groups.
</div>

<p>Each method turns about 30 token-probability summaries—entropy, energy, change over time and top-probability shape—into
one score. Fitting uses no answer labels. The corrected development set has 24 dataset–model experiments: 9 question-answering and 15
math. Macro AUROC averages their AUROCs equally; 0.5 is random.</p>

<h2>1. Continuous L-SML</h2>
<p>L-SML (Latent Spectral Meta-Learner, previously “Nadler fusion”) fuses measurements inside correlated groups, then
fuses the group scores. Our historical change kept continuous values instead of only “high” or “low.”</p>
<div class="equation">q<sub>g</sub> = X<sub>g</sub>a<sub>g</sub>;&nbsp;&nbsp;
s = Σ<sub>g</sub> β<sub>g</sub>q<sub>g</sub>
  <small>X<sub>g</sub> is one measurement group; covariance supplies the within-group weights a<sub>g</sub> and final weights β<sub>g</sub>.</small>
</div>
{score_plot(
    "Historical L-SML encoding test",
    [
        ("Binary high/low input", c["historical_binary"], ""),
        ("Continuous standardized input", c["historical_continuous"], "teal"),
    ],
    0.50,
    0.75,
    "Mean of five domain AUROCs over 29 historical cells: 0.652 → 0.701. The five inputs were label-selected, so this isolates the encoding change only.",
    "AUROC",
)}

<h2>2. U-PCR</h2>
<p>U-PCR (Unsupervised Principal Component Regression) estimates each measurement's relation to unknown correctness from
covariance, drops weak measurements and combines the rest. IU-PCR is the keep-all control.</p>
<div class="equation">C<sub>ij</sub> ≈ ρ<sub>i</sub> + ρ<sub>j</sub> − g²;&nbsp;&nbsp; s = Xŵ
  <small>C is observed covariance; ρ estimates each measurement's relation to hidden correctness; ŵ is the final
  weight; g² is a fitted shared-signal term. One pre-chosen orientation feature still tells the method which direction
  means “correct.”</small>
</div>
<h2>3. DUFS inside U-PCR</h2>
<p>Following Bracha's suggestion, DUFS (Differentiable Unsupervised Feature Selection) learns which measurements define
answer similarity; DUFS-LIU then encourages similar answers to receive similar IU-PCR scores.</p>
<div class="equation">R = XLXᵀ/n;&nbsp;&nbsp;
w<sub>λ</sub> = U[Uᵀ(C + λR)U]<sup>−1</sup>Uᵀρ̂
  <small>L is the answer-graph Laplacian; U holds the leading covariance directions; ρ̂ is U-PCR's estimated relation
  to correctness. λ controls the graph, and λ=0 gives ordinary IU-PCR.</small>
</div>
{score_plot(
    "Current methods on the same corrected 24 cells",
    [
        ("GOOD_6 · six hand-picked measurements", c["good6"], "amber"),
        ("U-PCR · drops weak measurements", core["deployed_upcr"], ""),
        ("IU-PCR · keeps all measurements", core["iu_pcr"], ""),
        ("DUFS-LIU · adds answer graph", core["dufs_liu__lambda_0p1"], "teal"),
    ],
    0.50,
    0.80,
    f"GOOD_6 was selected with labels. U-PCR differs from it by {100 * (core['deployed_upcr'] - c['good6']):+.2f} percentage points; DUFS-LIU differs from IU-PCR by {100 * (core['dufs_liu__lambda_0p1'] - core['iu_pcr']):+.3f}. Retrospective development evidence.",
    "AUROC",
)}

<h2>4. Family-NRM</h2>
<p>Family-NRM (Family Neutral Residual Mode) divides the 30 measurements into six types based on how they were produced, removes the shared
IU-PCR part, and adds a repeatable leftover disagreement as a small correction.</p>
<div class="equation">q = Rv*;&nbsp;&nbsp; s<sub>NRM</sub> = z(s<sub>IU</sub>) + q/[G · sd(q)]
  <small>R holds standardized family contributions after removing their shared IU-PCR part; v* is the residual-covariance
  direction closest to unit variance; G is the number of present families; z and sd standardize.</small>
</div>
{score_plot(
    "Family-NRM on the separate PRMBench response test",
    [
        ("IU-PCR", nrm_iu, ""),
        ("IU-PCR + Family-NRM", nrm_score, "teal"),
    ],
    0.50,
    0.80,
    f"The direction used 23 source experiments without reading their labels, although their eligibility rule used label counts. It was then fixed. On {nrm['n']:,} Qwen3-8B responses covering correct solutions and nine math-error types: {nrm['nrm_vs_iu_delta_pp']:+.2f} percentage points, 95% interval [{nrm_ci['low_pp']:+.2f},{nrm_ci['high_pp']:+.2f}]. Whole-response task, not the official step score.",
    "AUROC",
)}
<p class="result">The six groups—entropy level, entropy change, two energy types, top-probability shape and trace
structure—are supplied by hand. Family-NRM does not discover them.</p>
<p class="callout"><strong>For the next comparison:</strong> the direction used unlabeled donors selected by a
label-count eligibility rule, so I treat it as legacy label-touched calibration. Donors are optional: a target-cell-only
version without donors or labels is planned, not tested.</p>

<section class="questions">
  <h2>What I would like to discuss</h2>
  <ul>
    <li>Does U-PCR reaching the hand-picked result without labels count as a useful contribution?</li>
    <li>Is the manual six-family definition acceptable for the small Family-NRM result?</li>
    <li>Should the near-zero DUFS result stay in the main story?</li>
  </ul>
</section>
{sources([
    ("Continuous L-SML ablation", METHOD_CSV),
    ("Corrected 24-cell comparison", CORE_CSV),
    ("GOOD_6 reference standing", BENCHMARK_STANDING),
    ("Family-NRM PRMBench result", NRM_JSON),
])}
"""
    return page(
        "Basic fusion methods",
        "Advisor discussion brief 1 · August 2026",
        "Four methods I think are worth discussing, separated from the much larger set of variants that did not improve the result.",
        body,
    )


def render_graphs(c: dict) -> str:
    core = c["core"]
    family_graph = c["family_graph"]
    family_controls = c["family_graph_controls"]["arms"]
    family_prm = c["family_graph_prm"]
    family_hle = c["family_graph_hle"]
    body = f"""
<div class="guide">
  <strong>Why I tried this.</strong> Bracha suggested adding L-SML-style clustering to U-PCR and using DUFS inside it.
  Both found repeatable structure, but did not tell me whether that structure represented correctness.
</div>

<p>Unless I say otherwise, results use the corrected 24 dataset–model development experiments. Macro AUROC averages
their AUROCs equally; 0.5 is random. Differences are reported in percentage points.</p>

<h2>1. Clustering and answer graphs</h2>
<p>I first used feature clusters inside U-PCR. Because measurements in one cluster may be near-duplicates, the method
estimated weights only from covariance equations joining different clusters. In an earlier 25-cell audit it lost 4.46
percentage points (9 wins/16 losses). After matching pairs by covariance strength, the apparent within/cross-cluster
dependence difference disappeared.</p>
<div class="equation">ρ̂ = argmin<sub>ρ</sub> Σ<sub>g(i)≠g(j)</sub>
[C<sub>ij</sub> − (ρ<sub>i</sub> + ρ<sub>j</sub> − γ²)]²
  <small>C is observed covariance; ρ is the unknown relation to correctness; γ² is the shared-signal scale; g(i) is
  measurement i's cluster. Only cross-cluster pairs enter the fit.</small>
</div>
<p>For the graph tests, IU-PCR fuses all measurements without labels. A graph connects similar answers and asks their scores to be close:</p>
<div class="equation">Ω(s) = ½ Σ<sub>ij</sub> W<sub>ij</sub>(s<sub>i</sub> − s<sub>j</sub>)²
  <small>W<sub>ij</sub> is answer similarity; Ω is the smoothness penalty.</small>
</div>
{score_plot(
    "Graph variants on the same corrected 24 cells",
    [
        ("IU-PCR · no graph", core["iu_pcr"], ""),
        ("DUFS-LIU · learned similarity", core["dufs_liu__lambda_0p1"], "teal"),
        ("Cross-view agreement graph", core["atomic__ca_specrage_alpha_liu__lambda_10"], "teal"),
    ],
    0.50,
    0.80,
    "Macro AUROC on the same 24 dataset–model experiments. DUFS learns which measurements define similarity; cross-view agreement rewards neighbours shared by several measurement views. Differences are below 0.03 percentage points.",
    "AUROC",
)}

<h2>2. Why stable structure may be nuisance</h2>
<div class="equation">X = aY + bN + ε;&nbsp;&nbsp;
Cov(X) = aaᵀVar(Y) + bbᵀVar(N) + Cov(ε)
  <small>This simplified equation assumes Y, N and ε are uncorrelated. Y is correctness and N is nuisance such as
  length; a and b are their effects on the measurements, and ε is remaining variation. Covariance can reveal both
  patterns without naming which one is Y.</small>
</div>
{score_plot(
    "One very repeatable component mostly reproduced answer length",
    [
        ("Learned repeatable component", c["a4_external_corr"], "teal"),
        ("Generated-token count alone", c["a4_length_corr"], ""),
    ],
    0.00,
    1.00,
    "Held-Llama structural correlation on 3,400 fixed ProcessBench responses; no correctness labels were accessed.",
    "Correlation",
)}
<details>
  <summary>Other checks: feature shapes, a label-using router, and the 2022/2026 dependence methods</summary>
<div class="grid">
  <article class="card">
    <h3>Other nuisance checks</h3>
    <p>Symmetric transforms repaired two-tailed measurements, but nearly all recovered signal was already elsewhere.
    A later label-opened search gained +0.242 points; only about +0.022 remained without one MATH cell. In a synthetic
    nuisance test, the objective was {-c["a5_delta_pp"]:.2f} points below IU-PCR on 98 usable runs.</p>
  </article>
  <article class="card">
    <h3>Choosing a rule by context</h3>
    <p>A retrospective, label-using diagnostic asked whether answer types need different rules. Its registered router was
    {c["router_primary_delta_pp"]:+.2f} percentage points versus one global rule and won {c["router_primary_family_wins"]}/8
    dataset families. It is not a deployable label-free method.</p>
  </article>
</div>

<h2>3. Newer dependence papers</h2>
<div class="equation">C = L + S
  <small>Tenzer et al. (AISTATS 2022) extended U-PCR by splitting covariance into shared signal L and sparse correlated
  errors S. I call this sparse-error version SU-PCR.</small>
</div>
<p>SU-PCR changed AUROC by {c["sparse_dep_delta_pp"]:+.2f} points versus IU-PCR; its 95% interval
[{c["sparse_dep_ci_low_pp"]:+.2f},{c["sparse_dep_ci_high_pp"]:+.2f}] crossed zero. An earlier median-split DEEM adapter
also tied. Our graph-free continuous additive DEEM-B3 adapter is inspired by DEEM, not a reproduction of the published
hard multinomial/iRBM method. It completed 24 experiments and met preregistered noninferiority to IU-PCR, but was not
declared better. Its feature inventory and macro differ, so it stays outside this figure pending an aligned
rerun.</p>
</details>

<p class="result"><strong>Latest graph check:</strong> even when I built answer similarity from Family-NRM's six
measurement families, the estimated change was only {family_graph["nested_delta_vs_iu_pp"]:+.3f} percentage points,
with a 95% interval that crossed zero. It did not meet the rule set in advance for continuing it.</p>

<details>
<summary>Details and controls for the Family-NRM graph</summary>
<h2>4. Giving the graph Family-NRM's information</h2>
<p>Family-NRM is the small correction described in Brief 1: it divides the measurements into six manually defined
families and keeps their disagreement after IU-PCR. I used those six residual contributions as graph coordinates.</p>
<div class="equation">u<sub>i</sub> = (r<sub>i1</sub>,…,r<sub>i6</sub>);&nbsp;&nbsp;
W<sub>ij</sub> = exp(−||u<sub>i</sub> − u<sub>j</sub>||²/σ²)
  <small>r<sub>ig</sub> is answer i's residual from family g; nearby patterns receive a stronger edge.</small>
</div>
<p>Labels from seven dataset families chose the graph; the eighth measured transfer. Scores were label-free, but
selection was retrospective. Only 4/8 held-out families were positive.</p>
{interval_plot(
    "Family-residual graph: change from IU-PCR",
    [
        ("Nested transfer · 23 usable cells", family_graph["nested_delta_vs_iu_pp"], family_graph["nested_delta_vs_iu_ci_pp"][0], family_graph["nested_delta_vs_iu_ci_pp"][1]),
        ("PRMBench · math reasoning traces", family_prm["delta_vs_iu_pp"], family_prm["bootstrap"]["finalist"]["low_pp"], family_prm["bootstrap"]["finalist"]["high_pp"]),
        ("Humanity's Last Exam · broad question answering", family_hle["delta_vs_iu_pp"], family_hle["bootstrap"]["finalist"]["low_pp"], family_hle["bootstrap"]["finalist"]["high_pp"]),
    ],
    -0.15,
    0.15,
    "AUROC percentage points with 95% intervals. Both stress-test outcomes were already known; Humanity's Last Exam had only 68 correct answers.",
)}
<p>The fixed graph gave {family_controls["selected"]["equal_family_delta_pp"]:+.3f} percentage points; permuted nodes gave
{family_controls["node_permuted_graph"]["equal_family_delta_pp"]:+.3f}, and random families {family_controls["random_family_graph"]["equal_family_delta_pp"]:+.3f}.</p>
</details>

<p class="callout"><strong>Next comparison:</strong> Family-NRM and PGRD do not require donor data. New target-cell-only
versions enter the main table; unlabeled-donor and donor-label-selected versions stay separate. The local versions are
not yet tested.</p>

<p class="callout"><strong>My current reading:</strong> the repeated problem was not finding structure; it was knowing
which structure represented correctness. I designed a paired prompt experiment to supply that contrast, but only built
and audited the examples. The response-scoring stage never ran after I narrowed the scope, so this is not a negative result.</p>

<section class="questions">
  <h2>What I would like to discuss</h2>
  <ul>
    <li>Is this repeated nuisance result useful enough to be part of the thesis story?</li>
    <li>If you think a target contrast deserves more work, should we reopen the paired-intervention scope or use calibration labels?</li>
  </ul>
</section>
{sources([
    ("Matched graph comparison", CORE_CSV),
    ("Earlier clustered U-PCR audit", ROOT / "results/upcr_study/05_cluster_variant/index.html"),
    ("Non-monotone feature-contract conclusion", ROOT / "docs/research_notes/dufs_liu_mixed_feature_contract_conclusion.md"),
    ("Sparse dependence study", DEPENDENCE_CSV),
    ("Matched dependence arm scores", DEPENDENCE_ARMS_CSV),
    ("SU-PCR paper digest", TENZER_DIGEST),
    ("DEEM paper digest", DEEM_DIGEST),
    ("Context router diagnostic", CSTG_JSON),
    ("Repeatability and trace-length audit", A4_REPORT),
    ("Synthetic nuisance stress test", A5_REPORT),
    ("Family-residual graph synthesis", FAMILY_GRAPH_SYNTHESIS),
    ("Family-graph nested result", FAMILY_GRAPH_JSON),
    ("Family-graph controls", FAMILY_GRAPH_CONTROLS_JSON),
    ("PRMBench stress test", FAMILY_GRAPH_PRM_JSON),
    ("HLE stress test", FAMILY_GRAPH_HLE_JSON),
    ("Paired-intervention construction boundary", A6_JSON),
    ("Current decisions and experiment status", PROGRESS),
])}
"""
    return page(
        "Graphs and nuisance: why the gains kept disappearing",
        "Advisor discussion brief 2 · August 2026",
        "I tried the clustering and DUFS/U-PCR directions from our meeting. This page collects what they found, including the latest Family-NRM graph test.",
        body,
    )


def render_whitebox(c: dict) -> str:
    rows = c["white_rows"]
    pairs = c["white_pairs"]
    audit = c["white_audit"]

    gray = float(rows["gray_mixed_v2_dufs_liu"]["macro_auroc"])
    white = float(rows["white_pure_upcr"]["macro_auroc"])
    hybrid = float(rows["exploratory_equal_z_hybrid"]["macro_auroc"])
    gray_auprc = float(rows["gray_mixed_v2_dufs_liu"]["macro_auprc_hallucination"])
    white_auprc = float(rows["white_pure_upcr"]["macro_auprc_hallucination"])
    white_pair = pairs["white_minus_gray_final:auroc"]
    hybrid_pair = pairs["hybrid_minus_gray_final:auroc"]

    body = f"""
<div class="guide">
  <strong>Why I tried this.</strong> Following Amir's comment about the small feature pool, Bracha suggested either
  generating several answers or reading internal layers. I tested both; the layer experiment still uses one answer per question.
</div>

<div class="callout"><strong>Status: PRELIMINARY / VALIDATION BLOCKED.</strong> The corrected live-capture check and a
small architecture-fidelity pilot are unfinished. The comparisons below are retrospective.</div>

<h2>1. What “white-box” means here</h2>
<p>Gray-box access reads only the final token probabilities produced during generation. White-box access also reads
the hidden state inside each transformer layer and projects it through the model's own output head. This projection,
called a logit lens, asks what token distribution an intermediate layer would produce:</p>
<div class="equation">p<sub>ℓ,t</sub>(v) = softmax(W<sub>out</sub> LN(h<sub>ℓ,t</sub>))<sub>v</sub>;&nbsp;&nbsp;
D<sub>ℓ,t</sub> = −log p<sub>ℓ,t</sub>(y<sub>t</sub>);&nbsp;&nbsp; s<sub>white</sub> = X<sub>depth</sub>w
  <small>h is the hidden state, LN is the model's normalization, W<sub>out</sub> is the output head, and D is the surprise assigned to the generated
  token. X<sub>depth</sub> holds 13 summaries across early, middle and late layers; U-PCR derives w from their covariance
  without correctness labels.</small>
</div>

<div class="grid">
  <article class="card">
    <h3>Several generated answers</h3>
    <p>As I mentioned on August 3, repeated answers at the same temperature helped one MATH-500 experiment:
    {c["multisample_single_auc"]:.3f} → {c["multisample_five_auc"]:.3f} AUROC.
    Mixing temperatures hurt. I kept this outside the main comparison because it needs several answers per question.</p>
  </article>
  <article class="card">
    <h3>Intermediate layers</h3>
    <p>I summarized average and unusually large token surprises across early, middle and late depth. I selected the final
    13 summaries after inspecting earlier outcomes; only the final U-PCR covariance fit is label-free.</p>
  </article>
</div>

<h2>2. What happened</h2>
<p>The comparison below is retrospective and uses {audit["n_common_candidates"]:,} identical answers from 13
dataset–model pairs; one pair means one dataset tested with one model. AUROC measures how well a score ranks wrong above
correct answers; 0.5 is random. Neither method has been tested on untouched data. The gray-box comparison is the
retrospectively selected mixed-feature DUFS-LIU score: it learns answer similarity from 30 final-probability summaries
and smooths the covariance score over that graph.</p>

{score_plot(
    "Exact-row final-answer detection",
    [
        ("Gray-box · selected output-only score", gray, ""),
        ("White-box · internal depth", white, "teal"),
        ("White + gray average · chosen after seeing data", hybrid, "amber"),
    ],
    0.50,
    0.82,
    f"Macro AUROC on {audit['n_common_candidates']:,} exact common answers. The combined score was chosen after inspecting these data.",
    "AUROC",
)}

<p class="result"><strong>White minus gray:</strong> {100 * float(white_pair["macro_delta"]):+.2f} percentage points AUROC,
95% interval [{100 * float(white_pair["cell_bootstrap_ci_low"]):+.2f},
{100 * float(white_pair["cell_bootstrap_ci_high"]):+.2f}]. The interval includes zero.
Hallucination AUPRC (precision–recall area) is {white_auprc:.4f} white versus {gray_auprc:.4f} gray.
<strong>Combined score minus gray:</strong> {100 * float(hybrid_pair["macro_delta"]):+.2f} percentage points,
95% interval [{100 * float(hybrid_pair["cell_bootstrap_ci_low"]):+.2f},
{100 * float(hybrid_pair["cell_bootstrap_ci_high"]):+.2f}].</p>

{count_plot(
    "Answers available in the saved caches",
    [
        ("White-box scorable rows", int(audit["white_candidates"]), "teal"),
        ("Gray-box complete rows", int(audit["gray_candidates"]), ""),
        ("Exact common rows", int(audit["n_common_candidates"]), "amber"),
    ],
    "This is cache availability, not evidence that white-box access inherently covers more answers. Performance uses only common rows.",
)}

<div class="callout"><strong>My current reading:</strong> internal layers recovered nearly the same ranking through a
different route, but required stronger access and did not beat final probabilities. This is why I paused. The combined
score is only a hypothesis because I chose it after seeing these results.</div>

<section class="questions">
  <h2>What I would like to discuss</h2>
  <ul>
    <li>Is similar accuracy from a different measurement route useful enough to justify the extra access?</li>
    <li>Is the combined score worth one test fixed before opening new labels, or should we stop here?</li>
  </ul>
</section>
{sources([
    ("White-box research record", ROOT / "docs/experiments/WHITEBOX_LAYER_FUSION_RESEARCH_RECORD.md"),
    ("Exact-row summary", WHITE_CSV),
    ("Paired intervals", WHITE_PAIR_CSV),
    ("Row-identity and cache audit", WHITE_AUDIT),
    ("Earlier repeated-generation experiment", MULTISAMPLE_JSON),
])}
"""
    return page(
        "Internal layers: more measurements, not yet a gain",
        "Advisor discussion brief 3 · August 2026",
        "Internal model layers gave another way to measure uncertainty. They roughly matched the final-output score, but need more access and new-data validation.",
        body,
    )


def render_results_map(c: dict) -> str:
    report = (
        "../../../results/reconstruction_benchmark_v1/releases/2026-08-24_frozen24_v1/"
        "reporting_v2/2026-08-24_frozen24_v1/07_reports"
    )
    body = f"""<div class="guide">
  <strong>Start here.</strong> The four short briefs are a narrative advisor packet. The
  <a href="{report}/REPORT.html">full interactive benchmark report</a> is the exhaustive visual result browser:
  all 13 methods, all 24 frozen cells, exact rows, intervals, paired contrasts, heatmaps and graph diagnostics.
</div>

<h2>1. What was promised, and where it lives</h2>
<div class="grid">
  <div class="card"><span class="tag">FULL BENCHMARK</span><h3><a href="{report}/REPORT.html">Interactive 13-method report</a></h3><p>Method guide, dataset guide, results explorer, metric forests, cell × method heatmaps, paired contrasts, exact rows and graph checks.</p></div>
  <div class="card"><span class="tag">ADVISOR STORY</span><h3>Four presentation briefs</h3><p><a href="01_basic_fusion_methods.html">Fusion</a> · <a href="02_graphs_and_nuisance.html">graphs and nuisance</a> · <a href="03_whitebox_depth.html">white-box depth</a> · <a href="04_localization_and_early.html">applications</a>.</p></div>
  <div class="card"><span class="tag">MACHINE READABLE</span><h3><a href="{report}/../05_evaluation/benchmark.duckdb">DuckDB and tidy tables</a></h3><p>Predictions, metrics, contrasts and coverage are also available as CSV/Parquet beside the database.</p></div>
  <div class="card"><span class="tag">PLOT CONTRACT</span><h3><a href="{report}/plot_manifest.json">Plot manifest and exact plot data</a></h3><p>The manifest binds 175 plot-data CSVs under <a href="{report}/plot_data/">plot_data/</a>.</p></div>
</div>
<p><strong>Ready-to-open exports:</strong>
<a href="leaderboards/cell_leaderboard.csv">cell</a> ·
<a href="leaderboards/dataset_leaderboard.csv">dataset</a> ·
<a href="leaderboards/task_leaderboard.csv">task</a> ·
<a href="leaderboards/slice_leaderboard.csv">slice</a> ·
<a href="leaderboards/release_leaderboard.csv">release</a> leaderboards, plus the separate
<a href="published_comparators.json">published-comparator context registry</a>. Published values never enter a
common-cohort ranking or delta.</p>

<div class="equation">benchmark = 24 frozen dataset–model cells × 13 registered methods
  <small>Response-level ranking stays separate from localization, prefix, stopping, RAG and white-box estimands.</small>
</div>
{count_plot(
    "Frozen response benchmark coverage",
    [("Dataset–model cells", 24, "teal"), ("Registered methods", 13, "")],
    "The interactive report exposes the complete aligned response-level benchmark, not only the four methods discussed most often in meetings.",
)}
{count_plot(
    "Visual evidence in the full report",
    [
        ("Metric forest panels", 84, "teal"),
        ("Paired-contrast panels", 84, "teal"),
        ("Faceted heatmaps", 2, "amber"),
        ("Graph diagnostic/example panels", 5, "amber"),
    ],
    "There are 175 exact plot-data files in total; every plotted result can be traced back to its CSV.",
)}

<h2>2. Complete method roster</h2>
<div class="grid">
  <div class="card"><h3>Transparent references</h3><p>Equal-feature mean<br>Equal-family mean</p></div>
  <div class="card"><h3>Spectral / PCR core</h3><p>Continuous L-SML<br>IU-PCR<br>U-PCR<br>SU-PCR</p></div>
  <div class="card"><h3>Graph and feature-selection arms</h3><p>DUFS-LIU<br>Parameter-free DUFS→L-SML<br>Stability-selected DUFS→L-SML<br>CA-SpecRaGE atomic</p></div>
  <div class="card"><h3>Reconstructed extensions</h3><p>DEEM-B3<br>Family-NRM-A<br>PGRD-A</p></div>
</div>
<p class="result">All thirteen appear by name in the interactive report. The report keeps the method-level result rows
and direct paired uncertainty visible; it does not manufacture one cross-task winner.</p>

<h2>3. Application and access lanes</h2>
<div class="grid">
  <div class="card"><h3>Localization and prefix</h3><p>ProcessBench first-error localization, PRMBench step scoring and causal prefix prediction are visualized in <a href="04_localization_and_early.html">Brief 4</a>.</p></div>
  <div class="card"><h3>Stopping and RAG</h3><p>LEASH pass@1/token tradeoffs and all seven unpooled RAG panels now have dedicated charts in <a href="04_localization_and_early.html">Brief 4</a>.</p></div>
  <div class="card"><h3>White-box depth</h3><p>Matched white/gray discrimination and the coverage benefit are visualized in <a href="03_whitebox_depth.html">Brief 3</a>.</p></div>
  <div class="card"><h3>Graph mechanism checks</h3><p>Family-NRM, nuisance structure and residual-graph controls are summarized in <a href="02_graphs_and_nuisance.html">Brief 2</a> and expanded in the full report.</p></div>
</div>

<section class="questions">
  <h2>What I would like to discuss</h2>
  <ul>
    <li>Which two or three result panels should lead the advisor meeting?</li>
    <li>Should the full interactive report be shared as background, with the four briefs used as the spoken narrative?</li>
  </ul>
</section>
<details><summary>Sources and exact artifacts</summary><ul>
  <li><strong>Original protocol:</strong> <a href="../../experiments/RECONSTRUCTION_BENCHMARK_V1.md"><code>docs/experiments/RECONSTRUCTION_BENCHMARK_V1.md</code></a>.</li>
  <li><strong>Reporting contract:</strong> <a href="../../../scripts/reconstruction_benchmark/README.md"><code>scripts/reconstruction_benchmark/README.md</code></a>.</li>
  <li><strong>Full report:</strong> <a href="{report}/REPORT.html"><code>07_reports/REPORT.html</code></a>.</li>
  <li><strong>Exact visual manifest:</strong> <a href="{report}/plot_manifest.json"><code>07_reports/plot_manifest.json</code></a>.</li>
</ul></details>
"""
    return page(
        "Reconstruction benchmark: complete results map",
        "Advisor results index · August 2026",
        "One entry point for the full 13-method response benchmark, the four discussion briefs and the separate application lanes.",
        body,
    )


def render_applications(c: dict) -> str:
    applications = c["applications"]
    localization = applications["localization"]
    prefix = applications["prefix"]
    llama = localization["processbench"]["llama31_8b"]
    qwen4 = localization["processbench"]["qwen3_4b"]
    qwen8 = localization["processbench"]["qwen3_8b"]
    prmbench = localization["prmbench"]
    prefix64 = prefix["budgets"][64]
    prefix256 = prefix["budgets"][256]
    leash = applications["leash"]
    leash_aqua = leash["datasets"]["aqua"]
    leash_gsm8k = leash["datasets"]["gsm8k"]
    leash_overall = leash["overall"]
    rag = applications["rag"]
    ragtruth = rag["ragtruth_test_auroc"]
    gasp = rag["gasp"]
    lettuce = rag["lettuce"]
    refchecker = rag["refchecker"]
    unified = applications["unified_reporting"]

    body = f"""
<div class="guide">
  <strong>Why I tried this.</strong> Ofir suggested locating the first wrong reasoning step and shared Mind the Gap.
  I ran that ProcessBench test, then prefix prediction. LEASH stopping is a separate certified reconstruction.
</div>

<h2>1. First-error localization</h2>
<p>ProcessBench has 3,400 math solutions, clean or marked at the first wrong step. The certified reconstruction covers
three saved-telemetry scorers and four subsets. Each of the 13 frozen response-risk methods is combined with one
freshly fitted, label-free token-IU localizer through the same preregistered adapter; response-only and token-only
ablations are kept separate. The historical label-selected 0.75/0.25 blend is explicitly forbidden.</p>
<div class="equation">u<sub>i</sub> = midrank(response risk<sub>i</sub>),&nbsp;&nbsp;
v<sub>ij</sub> = midrank(step risk<sub>ij</sub>),&nbsp;&nbsp;
a<sub>ij</sub> = √(u<sub>i</sub>v<sub>ij</sub>);&nbsp;&nbsp;
ĵ = -1 if max<sub>j</sub>a<sub>ij</sub>≤τ, otherwise argmax<sub>j</sub>a<sub>ij</sub>
  <small>Ranks are computed only inside the exact response cell. The threshold is fitted after score freeze in five
  source-question-grouped folds, jointly over the four subsets for each scorer. The official macro F1 is the harmonic
  mean of first-error accuracy and clean-answer abstention accuracy.</small>
</div>
{score_plot(
    "Certified common-access ProcessBench localization",
    [
        ("Llama / IU adapter", llama["reference"], ""),
        ("Llama / DEEM-B3 adapter", llama["candidate"], "teal"),
        ("Qwen3-4B / IU adapter", qwen4["reference"], ""),
        ("Qwen3-4B / CA-SpecRaGE adapter", qwen4["candidate"], "teal"),
        ("Qwen3-8B / IU adapter", qwen8["reference"], ""),
        ("Qwen3-8B / equal-family adapter", qwen8["candidate"], "teal"),
    ],
    0.25,
    0.35,
    "Equal-subset official macro F1; 20,000 paired source-question bootstrap draws. Each named method is the point leader for that scorer, but every paired interval versus its matched IU adapter includes zero.",
    "Macro F1",
)}
<p class="result">The paired gains over the matched IU adapter are small and unresolved: Llama DEEM-B3 {llama['delta']:+.4f}
[{llama['ci_low']:+.4f},{llama['ci_high']:+.4f}], Qwen3-4B CA-SpecRaGE {qwen4['delta']:+.4f}
[{qwen4['ci_low']:+.4f},{qwen4['ci_high']:+.4f}], and Qwen3-8B equal-family {qwen8['delta']:+.4f}
[{qwen8['ci_low']:+.4f},{qwen8['ci_high']:+.4f}]. On PRMBench's exact step evaluator the token-only IU head instead
beats the IU response+token fusion by {prmbench['auroc_delta']:+.4f} AUROC
[{prmbench['auroc_ci_low']:+.4f},{prmbench['auroc_ci_high']:+.4f}] and {prmbench['auprc_delta']:+.4f} AUPRC
[{prmbench['auprc_ci_low']:+.4f},{prmbench['auprc_ci_high']:+.4f}]. The supervised PRM and large critic remain
high-access context ceilings, not common-access head-to-head competitors.</p>

<h2>2. Predicting from an unfinished answer</h2>
<p>Using only telemetry available at each unfinished prefix, the task is to predict whether the eventual answer will be
wrong. The certified lane recomputes three historical causal scores on CPU rather than copying their ledgers:
Unified-28, IU-28 without elapsed length, and the selected two-head Step272 score. The signed historical ledger is an
anchor only. Results are equal-subset macros over the four ProcessBench subsets.</p>
<div class="equation">s<sub>prefix</sub> = ½z(global risk) + ½z(max local risk)
  <small>z standardizes each score. No future token or elapsed prefix length is used by the registered Step272 arm.
  This is early prediction from a fixed saved trace, not an adaptive stopping policy.</small>
</div>
{score_plot(
    "Certified causal prefix prediction",
    [
        ("64 tokens / Unified-28", prefix64["unified_auroc"], ""),
        ("64 tokens / Step272", prefix64["step272_auroc"], "teal"),
        ("256 tokens / Unified-28", prefix256["unified_auroc"], ""),
        ("256 tokens / Step272", prefix256["step272_auroc"], "teal"),
    ],
    0.50,
    0.70,
    "Equal-subset AUROC; 2,000 paired source-question-within-subset draws. Step272 minus Unified is separated at both shown budgets.",
    "AUROC",
)}
<p class="result">Step272 improves on Unified-28 by {prefix64['delta']:+.4f} AUROC
[{prefix64['ci_low']:+.4f},{prefix64['ci_high']:+.4f}] at 64 tokens and {prefix256['delta']:+.4f}
[{prefix256['ci_low']:+.4f},{prefix256['ci_high']:+.4f}] at 256. Against the stronger IU-28-no-length control, only the
256-token AUPRC contrast is separated: {prefix256['vs_iu28_no_length_auprc_delta']:+.4f}
[{prefix256['vs_iu28_no_length_auprc_ci_low']:+.4f},{prefix256['vs_iu28_no_length_auprc_ci_high']:+.4f}]. At 512 tokens
GSM8K is single-class, so the preregistered four-subset macro is undefined; the evaluator does not silently drop that
subset. These results authorize no warning-time, stopping, token-saving or latency-saving claim.</p>

<details>
<summary>Certified LEASH reconstruction: actual callback stopping</summary>
<h2>3. LEASH stopping — certified paper-specified-partial</h2>
<p><strong>Reconstruction status: {leash['status']}.</strong> Actual callback stopping was observed in six separate ready
Qwen, Llama and Phi dataset–model cells. Two Mistral cells were protocol-gate blocked because the base tokenizer had no
chat template. Fidelity remains paper-specified-partial, not paper-exact: important constants, prompt details and the
GSM8K seed are unspecified.</p>
<div class="equation">min<sub>π</sub> E[T<sub>π</sub>]&nbsp;&nbsp; subject to &nbsp;&nbsp;
pass@1(π) ≥ pass@1(full) − δ
  <small>This is the conceptual optimization objective used to explain the tradeoff, not a reproduced equation from the
  paper. π is the rule, T<sub>π</sub> is token count and δ is allowed accuracy loss.</small>
</div>
<p class="result"><strong>Descriptive results only; brackets are 95% grouped-bootstrap intervals.</strong> AQuA
equal-model CoT → LEASH pass@1 was
{leash_aqua['cot_pass_at_1']:.6f} → {leash_aqua['leash_pass_at_1']:.6f}, Δ {leash_aqua['pass_at_1_delta']:+.6f}
[{leash_aqua['pass_at_1_delta_ci_low']:+.6f},{leash_aqua['pass_at_1_delta_ci_high']:+.6f}], with token reduction versus CoT
{leash_aqua['token_reduction']:.6f} [{leash_aqua['token_reduction_ci_low']:.6f},{leash_aqua['token_reduction_ci_high']:.6f}].
GSM8K CoT → LEASH was {leash_gsm8k['cot_pass_at_1']:.6f} → {leash_gsm8k['leash_pass_at_1']:.6f}, Δ
{leash_gsm8k['pass_at_1_delta']:+.6f} [{leash_gsm8k['pass_at_1_delta_ci_low']:+.6f},{leash_gsm8k['pass_at_1_delta_ci_high']:+.6f}],
with reduction {leash_gsm8k['token_reduction']:.6f}
[{leash_gsm8k['token_reduction_ci_low']:.6f},{leash_gsm8k['token_reduction_ci_high']:.6f}]. Overall equal-dataset-after-equal-model
Δpass@1 was {leash_overall['pass_at_1_delta']:+.6f}
[{leash_overall['pass_at_1_delta_ci_low']:+.6f},{leash_overall['pass_at_1_delta_ci_high']:+.6f}] and reduction
{leash_overall['token_reduction']:.6f} [{leash_overall['token_reduction_ci_low']:.6f},{leash_overall['token_reduction_ci_high']:.6f}].
Pass@1 fell in all six ready cells. This supports no matched-accuracy claim, theorem or cross-task headline.</p>
{score_plot(
    "LEASH trades pass@1 for shorter generations",
    [
        ("AQuA / full CoT", leash_aqua["cot_pass_at_1"], ""),
        ("AQuA / LEASH", leash_aqua["leash_pass_at_1"], "teal"),
        ("GSM8K / full CoT", leash_gsm8k["cot_pass_at_1"], ""),
        ("GSM8K / LEASH", leash_gsm8k["leash_pass_at_1"], "teal"),
    ],
    0.0,
    0.70,
    "Equal-model within-dataset pass@1. The callback stopped generation, but accuracy fell in every ready cell.",
    "pass@1",
)}
{score_plot(
    "Observed token reduction from actual callback stopping",
    [
        ("AQuA", leash_aqua["token_reduction"], "teal"),
        ("GSM8K", leash_gsm8k["token_reduction"], "teal"),
        ("Equal-dataset summary", leash_overall["token_reduction"], "amber"),
    ],
    0.0,
    0.50,
    "Descriptive grouped-bootstrap estimates. The summary combines datasets only after equal-model aggregation.",
    "Fraction fewer tokens",
)}
</details>

<details>
  <summary>Certified RAG evidence reconstruction: seven separate panels</summary>
  <p><strong>Reconstruction status: {rag['status']}.</strong> This is retrospective application evidence with historical
  labels opened, not a pooled cross-task benchmark. RAGTruth test AUROC is {ragtruth['answer']:.4f} by answer,
  {ragtruth['sentence']:.4f} by sentence and {ragtruth['token']:.4f} by token; Data2txt-versus-QA ordering changes with
  granularity. On the local GASP sample, AUROC is {gasp['auroc']:.4f} versus {gasp['matched_iu_pcr_auroc']:.4f}, Δ
  {gasp['delta']:+.4f} [{gasp['ci_low']:+.4f},{gasp['ci_high']:+.4f}]: no superiority. Lettuce is a supervised ceiling
  at F1 {lettuce['f1']:.4f}. RefChecker accurate-, noisy- and zero-context settings stay separate; fixed claims only,
  with claim extraction out of scope. The seven panels have no pooled macro or cross-task headline.</p>
  {score_plot(
      "RAGTruth test: the estimand changes with localization granularity",
      [
          ("Answer-level", ragtruth["answer"], "teal"),
          ("Sentence-level", ragtruth["sentence"], "teal"),
          ("Token-level", ragtruth["token"], "teal"),
      ],
      0.50,
      0.80,
      "Separate answer, sentence and token panels; they are not interchangeable rows of one pooled leaderboard.",
      "AUROC",
  )}
  {score_plot(
      "Local GASP comparison",
      [
          ("Evidence-contrast score", gasp["auroc"], "teal"),
          ("Matched IU-PCR", gasp["matched_iu_pcr_auroc"], ""),
      ],
      0.62,
      0.70,
      "The point difference is +0.0111, but its paired interval crosses zero: no superiority claim.",
      "AUROC",
  )}
  {score_plot(
      "Lettuce supervised ceiling",
      [("Supervised example classifier", lettuce["f1"], "amber")],
      0.0,
      1.0,
      "A separate supervised example-level ceiling, not a common-access comparison with RAGTruth or GASP.",
      "F1",
  )}
  {score_plot(
      "RefChecker NLI detector by context setting",
      [
          ("Accurate context", refchecker["nli_accuracy"]["accurate_context"], "teal"),
          ("Noisy context", refchecker["nli_accuracy"]["noisy_context"], "teal"),
          ("Zero context", refchecker["nli_accuracy"]["zero_context"], "teal"),
      ],
      0.40,
      0.80,
      "Three-way claim accuracy. Settings remain separate because both task conditions and class balance differ.",
      "Accuracy",
  )}
  {score_plot(
      "Fixed binary IU transfer by RefChecker setting",
      [
          ("Accurate context", refchecker["fixed_binary_transfer_auroc"]["accurate_context"], ""),
          ("Noisy context", refchecker["fixed_binary_transfer_auroc"]["noisy_context"], ""),
          ("Zero context", refchecker["fixed_binary_transfer_auroc"]["zero_context"], ""),
      ],
      0.55,
      0.80,
      "Binary fixed-claim AUROC. This is a separate estimand from the three-way NLI accuracy above.",
      "AUROC",
  )}
</details>

<div class="callout"><strong>My current reading:</strong> the certified localization gains are small and unresolved,
while Step272 separates from Unified-28 at 64 and 256 tokens. Certified LEASH stopped generation but lowered pass@1 in
every ready cell. Certified RAG is retrospective evidence across seven unpooled panels, with no cross-task headline.</div>

<section class="questions">
  <h2>What I would like to discuss</h2>
  <ul>
    <li>Should localization and the task-specific scores be the main application story?</li>
    <li>Would new-model and new-question localization be the best final test?</li>
  </ul>
</section>
<details><summary>Sources and exact artifacts</summary><ul>
  <li><strong>Certified localization release:</strong> <code>{localization['release']}</code>; evaluation A/B certificate payload <code>{localization['evaluation_certificate_payload_sha256']}</code>.</li>
  <li><strong>Certified causal-prefix release:</strong> <code>{prefix['release']}</code>; evaluation A/B certificate payload <code>{prefix['evaluation_certificate_payload_sha256']}</code>.</li>
  <li><strong>Historical fixed ProcessBench and RAGTruth context:</strong> <code>results/fixed_application_pipelines_v1/REPORT.md</code>.</li>
  <li><strong>Historical fair-comparison context:</strong> <code>results/fair_paper_exact_comparisons_v1/REPORT.md</code>.</li>
  <li><strong>Certified LEASH release:</strong> <code>{leash['release']}</code>; evaluation A/B certificate file SHA-256 <code>{leash['evaluation_certificate_file_sha256']}</code>; payload <code>{leash['evaluation_certificate_payload_sha256']}</code>.</li>
  <li><strong>Certified RAG release:</strong> <code>{rag['release']}</code>; evaluation A/B certificate file SHA-256 <code>{rag['evaluation_certificate_file_sha256']}</code>; payload <code>{rag['evaluation_certificate_payload_sha256']}</code>.</li>
  <li><strong>Certified unified reporting release:</strong> <code>{unified['release']}</code>; A/B certificate file SHA-256 <code>{unified['ab_certificate_file_sha256']}</code>; payload <code>{unified['ab_certificate_payload_sha256']}</code>; content <code>{unified['release_content_sha256']}</code>; {unified['authenticated_source_bindings']} authenticated source bindings.</li>
</ul></details>
"""
    return page(
        "From flagging an answer to finding where it went wrong",
        "Advisor discussion brief 4 · August 2026",
        "Certified sections cover ProcessBench localization, causal prefix prediction, paper-specified-partial LEASH callback stopping and seven separate RAG evidence panels.",
        body,
    )


def render_email() -> str:
    return """Subject: Update after our July 30 meeting

Hi Ofir, Bracha and Amir,

Since my August 3 email, I finished the clustering/DUFS, internal-layer and ProcessBench experiments. Conformal remains unstarted.

- Basic fusion: cluster-aware U-PCR lost 4.5 points in 25 experiments. DUFS's answer graph added almost zero. Methods remain near 0.774 macro AUROC. One stable pattern followed length.
- Family-NRM uses disagreement among six hand-defined measurement types to correct IU-PCR. On 6,966 PRMBench math traces, AUROC rose 0.7206 → 0.7252: +0.46 points, 95% interval [+0.07,+0.84].
- Internal layers and final-output scores had similar AUROC on 31,440 identical answers. Their combination still needs validation.
- Across the three certified ProcessBench scorer panels, the point-leading adapters added 0.33 to 0.58 F1 points over matched IU, but every interval crossed zero. At 64 and 256 tokens, Step272 exceeded Unified-28 by 3.26 and 4.58 AUROC points. Certified paper-specified-partial LEASH callback stopping cut tokens but lowered pass@1 in all six ready cells; two Mistral cells were blocked. Certified RAG spans seven unpooled retrospective panels; local GASP shows no superiority.

SU-PCR, the 2022 sparse-error extension, did not clearly help. Our graph-free continuous additive DEEM-B3 adapter is inspired by DEEM, not a reproduction of its published hard multinomial/iRBM method. It completed 24 experiments and was not worse than IU-PCR under the preregistered rules, but was not declared better. Its features and macro differ, so I am not ranking it yet.

For the aligned rerun, I plan new Family-NRM and graph-roughness versions that learn inside each dataset–model experiment, without donor data or labels. Cross-dataset variants will be separate unlabeled-donor or donor-label controls. These local versions are not results yet.

I do not yet have one directly comparable general winner. I would value your view on what should lead, and whether a final test should study new-question localization, conformal calibration or internal layers.

Could we meet next week?

Thanks,
Omri
"""


def render_ledger(c: dict) -> str:
    leash = c["applications"]["leash"]
    leash_aqua = leash["datasets"]["aqua"]
    leash_gsm8k = leash["datasets"]["gsm8k"]
    leash_overall = leash["overall"]
    rag = c["applications"]["rag"]
    ragtruth = rag["ragtruth_test_auroc"]
    gasp = rag["gasp"]
    lettuce = rag["lettuce"]
    unified = c["applications"]["unified_reporting"]
    return f"""# Claim ledger for the advisor packet

This is an internal boundary check, not a suggested attachment.

| Topic | Short supported claim | Important boundary |
|---|---|---|
| Continuous L-SML | Continuous input improved the historical matched ablation from {c['historical_binary']:.3f} to {c['historical_continuous']:.3f}. | Historical label-selected five-feature set; 29-cell, five-domain macro; not the current 24 cells. |
| U-PCR | Automatic weak-feature exclusion reaches about the same current macro AUROC as IU-PCR. | Retrospective development cells; one orientation bit remains fixed. |
| Clustered U-PCR | Literal L-SML-style clustered U-PCR lost 4.46pp in the earlier 25-cell audit. | Separate historical population; do not conflate with the current DUFS graph tie. |
| DUFS-LIU | The registered graph changes current macro AUROC by {100 * (c['core']['dufs_liu__lambda_0p1'] - c['core']['iu_pcr']):+.3f}pp versus IU-PCR. | This is not a confirmed graph gain. |
| Non-monotone contract | Retrospective mixed-v2 DUFS-LIU was +0.242pp; leave-one-family-out was +0.123pp. | Fragile: about +0.022pp without one MATH cell; awaits external validation. |
| Family-NRM | Frozen PRMBench response AUROC improves by {c['nrm']['nrm_vs_iu_delta_pp']:+.3f}pp. | Completed method uses a direction from other unlabeled cells and a label-count-derived 23-cell roster; manual six-family prior; new within-cell variant is unrun. |
| Family-NRM/PGRD A-B-C benchmark | A uses the target cell only; B may use unlabeled donors; C uses donor labels for selection. | A variants are new ablations. Donor data are not an inherent method requirement. C is a supervised ceiling. |
| Family-residual graph | Nested leave-dataset-family-out change is {c['family_graph']['nested_delta_vs_iu_pp']:+.3f}pp, 95% CI [{c['family_graph']['nested_delta_vs_iu_ci_pp'][0]:+.3f},{c['family_graph']['nested_delta_vs_iu_ci_pp'][1]:+.3f}]. | Retrospective label-using configuration transfer; graph scores themselves were label-free. PRMBench/HLE were known-outcome stress tests. |
| Graphs and nuisances | Tested methods often found stable structure without a robust matched detection gain. | Do not generalize to all graphs or future measurements. |
| Later dependence work | Tenzer et al. (2022) adds sparse-error SU-PCR. Our continuous additive DEEM-B3 adapter, inspired by DEEM, registered noninferiority in a separate 24-cell run. | DEEM-B3 is our graph-free adaptation, not a reproduction of the published hard multinomial/iRBM DEEM method and not a direct U-PCR extension; its present-inventory/equal-family result is not yet ranked against the fixed-stable/cell-macro table. Residual-Graph DEEM is a separate synthetic-gate closure. |
| PTNI | Construction audit passed; detector stage did not run. | Closed by scope decision; separate gamma-3 orientation control was negative. |
| White-box | White minus selected gray mixed-v2 DUFS-LIU is {100 * float(c['white_pairs']['white_minus_gray_final:auroc']['macro_delta']):+.3f}pp on exact common rows. | The 13-expert white roster and gray contract were retrospectively selected; only final score fitting is label-free. |
| Localization | Across the three certified ProcessBench scorer panels, the point-leading common-access adapters improve official macro F1 by +0.33 to +0.58pp versus matched IU, but every paired interval crosses zero. On PRMBench, token-only IU exceeds IU response+token fusion by +7.23pp AUROC and +4.62pp AUPRC. | ProcessBench uses an adapted common protocol with five-fold source-question-grouped threshold fitting; PRMBench is the exact official step evaluator. Supervised PRM and critic rows are high-access context only. |
| Prefix | Step272 exceeds Unified-28 AUROC by +3.26pp at 64 tokens and +4.58pp at 256, with both paired intervals above zero. | This is causal early scoring on fixed saved traces, not adaptive stopping. Against IU-28-no-length, only 256-token AUPRC separates; the 512-token four-subset macro is undefined because GSM8K is single-class. |
| Stopping | CERTIFIED actual callback stopping in six ready cells. AQuA equal-model CoT → LEASH pass@1 is {leash_aqua['cot_pass_at_1']:.6f} → {leash_aqua['leash_pass_at_1']:.6f}, Δ {leash_aqua['pass_at_1_delta']:+.6f} [{leash_aqua['pass_at_1_delta_ci_low']:+.6f},{leash_aqua['pass_at_1_delta_ci_high']:+.6f}], with token reduction {leash_aqua['token_reduction']:.6f} [{leash_aqua['token_reduction_ci_low']:.6f},{leash_aqua['token_reduction_ci_high']:.6f}]. GSM8K CoT → LEASH is {leash_gsm8k['cot_pass_at_1']:.6f} → {leash_gsm8k['leash_pass_at_1']:.6f}, Δ {leash_gsm8k['pass_at_1_delta']:+.6f} [{leash_gsm8k['pass_at_1_delta_ci_low']:+.6f},{leash_gsm8k['pass_at_1_delta_ci_high']:+.6f}], with reduction {leash_gsm8k['token_reduction']:.6f} [{leash_gsm8k['token_reduction_ci_low']:.6f},{leash_gsm8k['token_reduction_ci_high']:.6f}]. Pass@1 fell in all six. | Paper-specified-partial, not paper-exact; two Mistral cells were blocked. All brackets are 95% grouped-bootstrap intervals. Overall equal-dataset-after-equal-model Δpass@1 {leash_overall['pass_at_1_delta']:+.6f} [{leash_overall['pass_at_1_delta_ci_low']:+.6f},{leash_overall['pass_at_1_delta_ci_high']:+.6f}] and token reduction {leash_overall['token_reduction']:.6f} [{leash_overall['token_reduction_ci_low']:.6f},{leash_overall['token_reduction_ci_high']:.6f}] are descriptive only. No matched-accuracy, theorem or cross-task headline. |
| RAG | CERTIFIED retrospective application evidence across {rag['panels']} separate panels. RAGTruth test AUROC is {ragtruth['answer']:.4f}/{ragtruth['sentence']:.4f}/{ragtruth['token']:.4f} by answer/sentence/token. Local GASP is {gasp['auroc']:.4f} versus {gasp['matched_iu_pcr_auroc']:.4f}, Δ {gasp['delta']:+.4f} [{gasp['ci_low']:+.4f},{gasp['ci_high']:+.4f}]; no superiority. | Historical labels were opened and Data2txt-versus-QA behavior is heterogeneous. Lettuce is a supervised ceiling at F1 {lettuce['f1']:.4f}. RefChecker accurate/noisy/zero settings stay separate; fixed claims only, claim extraction out of scope. No pooling, cross-panel macro or cross-task headline. |
| Unified reporting | CERTIFIED final release with {unified['authenticated_source_bindings']} authenticated source bindings. | Provenance bridge only; it does not create a pooled cross-task estimand. |

Canonical scope is governed by docs/research_notes/research_status_consolidated_2026-08-19.md plus the later amendments and the August 23 Family-residual graph synthesis recorded in PROGRESS.md.
"""


def render_readme() -> str:
    return """# Advisor update packet — August 2026

Suggested attachment order:

0. 00_results_map.html — the complete navigation page and the link to the exhaustive 13-method visual report
1. 01_basic_fusion_methods.html
2. 02_graphs_and_nuisance.html
3. 03_whitebox_depth.html
4. 04_localization_and_early.html

The short email is ../Advisor_Update_Aug21_2026.md.

The results map separates the exhaustive benchmark browser from the shorter advisor story. Each brief stands alone.
It explains the task, gives the smallest useful method equation, shows performance visually,
states the evidence boundary, and ends with questions for the advisors. CLAIM_LEDGER.md is an internal accuracy check
and is not intended as an attachment.

The order follows the previous advisor emails:

- Brief 1: the U-PCR/DUFS fusion line from Bracha's July 30 summary.
- Brief 2: clustering, dependence and nuisance, including the latest Family-NRM graph test.
- Brief 3: Amir's feature-pool question and Bracha's internal-feature access suggestion.
- Brief 4: certified ProcessBench localization, prefix detection, paper-specified-partial LEASH callback stopping and seven separate retrospective RAG evidence panels.

The exhaustive visual result browser is:

    ../../../results/reconstruction_benchmark_v1/releases/2026-08-24_frozen24_v1/reporting_v2/2026-08-24_frozen24_v1/07_reports/REPORT.html

It contains all 13 registered methods across the frozen 24 cells, 84 metric-forest panels, 84 paired-contrast
panels, heatmaps, exact rows, graph diagnostics, a plot manifest and 175 exact plot-data CSVs. It is the place to
inspect every method; the four briefs intentionally emphasize the scientific narrative rather than repeat every row.

For spreadsheet review, `leaderboards/` contains cell, dataset, task, slice and release CSV exports. The historical
single-task macro is stored at release level, so the task and release files are exact aliases; see
`leaderboards/README.md`. `published_comparators.json` is a separate context registry and is never mixed into those
common-cohort rankings.

Conformal calibration remains unstarted and is left as a meeting decision rather than presented as a result.

The packet also preserves a benchmark distinction that is still a plan rather than a result: new within-cell
Family-NRM/PGRD variants use no donors or labels; donor-unsupervised and donor-label-selected variants appear only as
separate controls.

The packet is pinned to the certified unified reporting release with nine authenticated source bindings; this
provenance bridge does not create a pooled cross-task result.

Rebuild:

    python3 scripts/build_advisor_update_aug21_2026.py

Verify without changing files:

    python3 scripts/build_advisor_update_aug21_2026.py --check
"""


def artifacts(c: dict) -> dict[Path, str]:
    return {
        EMAIL: render_email(),
        OUT / "00_results_map.html": render_results_map(c),
        OUT / "01_basic_fusion_methods.html": render_fusion(c),
        OUT / "02_graphs_and_nuisance.html": render_graphs(c),
        OUT / "03_whitebox_depth.html": render_whitebox(c),
        OUT / "04_localization_and_early.html": render_applications(c),
        OUT / "CLAIM_LEDGER.md": render_ledger(c),
        OUT / "README.md": render_readme(),
    }


def expanded_words(document: str) -> int:
    text_only = re.sub(r"<style.*?</style>|<head.*?</head>|<title.*?</title>", " ", document, flags=re.I | re.S)
    text_only = re.sub(r"<[^>]+>", " ", text_only)
    return len(html.unescape(text_only).split())


def visible_words(document: str) -> int:
    initial = re.sub(r"<details.*?</details>", " ", document, flags=re.I | re.S)
    return expanded_words(initial)


def validate(bundle: dict[Path, str]) -> None:
    email_words = len(bundle[EMAIL].split())
    if email_words > 310:
        raise ValueError(f"Email is too long: {email_words} words")

    forbidden = (
        "I recommend",
        "we should close",
        "PTNI failed",
        "all graphs fail",
        "the static fusion search is saturated",
    )
    combined = "\n".join(bundle.values())
    for phrase in forbidden:
        if phrase.lower() in combined.lower():
            raise ValueError(f"Over-decisive or unsafe phrase present: {phrase}")

    for path, content in bundle.items():
        if path.suffix != ".html":
            continue
        for required in ("class=\"equation\"", "class=\"plot\"", "What I would like to discuss", "Sources and exact artifacts"):
            if required not in content:
                raise ValueError(f"{path.name} is missing {required}")
        if "http://" in content or "https://" in content or "<script" in content.lower():
            raise ValueError(f"{path.name} is not self-contained")
        if content.count('<figure class="plot">') < 2:
            raise ValueError(f"{path.name} needs at least two result figures")
        if path.name == "00_results_map.html":
            method_names = (
                "Equal-feature mean",
                "Equal-family mean",
                "Continuous L-SML",
                "IU-PCR",
                "U-PCR",
                "SU-PCR",
                "DUFS-LIU",
                "Parameter-free DUFS→L-SML",
                "Stability-selected DUFS→L-SML",
                "CA-SpecRaGE atomic",
                "DEEM-B3",
                "Family-NRM-A",
                "PGRD-A",
            )
            missing_methods = [name for name in method_names if name not in content]
            if missing_methods:
                raise ValueError(f"results map is missing methods: {missing_methods}")
            if "07_reports/REPORT.html" not in content or "plot_manifest.json" not in content:
                raise ValueError("results map is missing the exhaustive report contract")
        if path.name == "04_localization_and_early.html":
            if content.count('<figure class="plot">') < 9:
                raise ValueError("application brief needs all nine registered result figures")
            for required_panel in (
                "LEASH trades pass@1",
                "RAGTruth test",
                "Local GASP comparison",
                "Lettuce supervised ceiling",
                "RefChecker NLI detector",
                "Fixed binary IU transfer",
            ):
                if required_panel not in content:
                    raise ValueError(f"application brief is missing {required_panel}")
        if visible_words(content) > 650:
            raise ValueError(f"{path.name} is too long: {visible_words(content)} initially visible words")
        expanded_limit = 1250 if path.name in {"00_results_map.html", "04_localization_and_early.html"} else 1000
        if expanded_words(content) > expanded_limit:
            raise ValueError(f"{path.name} is too long when expanded: {expanded_words(content)} words")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify generated files match canonical inputs")
    args = parser.parse_args()

    context = load_context()
    bundle = artifacts(context)
    validate(bundle)

    if args.check:
        changed = []
        for path, expected in bundle.items():
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                changed.append(str(path.relative_to(ROOT)))
        if changed:
            raise SystemExit("Generated files are stale:\n" + "\n".join(changed))
        print("Advisor packet is current.")
        for path, content in bundle.items():
            if path.suffix == ".html":
                print(f"{path.name}: {visible_words(content)} initially visible / {expanded_words(content)} expanded words")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    for path, content in bundle.items():
        path.write_text(content, encoding="utf-8")
    print("Built advisor packet.")
    for path, content in bundle.items():
        if path.suffix == ".html":
            print(f"{path.name}: {visible_words(content)} initially visible / {expanded_words(content)} expanded words")
    print(f"Email: {len(bundle[EMAIL].split())} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
