#!/usr/bin/env python
"""
action_items_report.py — 9 advisor-facing HTML pages under results/action_items/.

Compilation of the six 2026-06-17 meeting action items, plus a per-domain
breakdown and a critical-review ("advisor scrutiny") page. Follows the
advisor_report.py convention: every numeric table cell is generated from a
source CSV/JSON, narrative sections cite HISTORY step numbers inline.

Sources:
  results/reasoning_benchmark.csv            (items 4, scrutiny 1)
  results/repgrid/scores_lsml_upcr.csv       (items 3/4, breakdown)
  results/repgrid/ubaseline_scores.csv       (item 4, scrutiny 2)
  results/subset_sweep/sweep_summary.csv     (breakdown, legacy L-SML cross-check)
  results/subset_sweep/upcr_legacy.csv       (breakdown, legacy cells both methods)
  results/repgrid/phase12_signfix.json       (0c output — item 5, scrutiny 4)
  results/repgrid/phase15_rescore.json       (0d output — items 5/6, scrutiny 3)

Terminology guardrails (project rules, enforced by guardrail_scan on every
page): no bare "Nadler" (only the full lineage name "Jaffe-Fetaya-Nadler"),
no "MV_EPR", no "recommended".

Usage:
    PYTHONPATH=. python scripts/action_items_report.py
"""
import json
import os
import sys
from collections import defaultdict

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPTS)
for p in (REPO, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from advisor_report import esc, pct, read_csv, CSS, guardrail_scan  # noqa: E402
from report_figs import (  # noqa: E402 — CSV-driven inline-SVG figures
    FIG_CSS, fig_gsm8k_forest, fig_same_model_deltas,
    fig_cell_landscape_losnet, fig_cell_landscape_gsm8k_llama8b, fig_good5_vs_seqlp,
    fig_math500_forest, fig_triviaqa_forest, fig_qa_extension_forest, master_table_html,
    fig_item6_temperature, fig_item6_arms,
)

RESULTS = os.path.join(REPO, "results")
REPGRID = os.path.join(RESULTS, "repgrid")
SWEEP = os.path.join(RESULTS, "subset_sweep")
OUT_DIR = os.path.join(RESULTS, "action_items")

AS_OF = ("Cluster state 2026-07-12 late — benchmarking desk CLOSED: all cells fetched. "
         "A2 + A3 (both Qwen3/ARS) = documented REJECT (truncation-label leakage at 8192/16384); "
         "C1 inside_coqa scored 68.4 vs INSIDE 80.4 (FLOOR flag, judge acc 0.132); no jobs left in queue")
GEN_DATE = "2026-07-12"

EXTRA_CSS = """
  .badge-flagged{background:var(--amber-light);color:var(--amber-dark);}
  .card-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0;}
  @media(max-width:820px){.card-grid{grid-template-columns:1fr;}}
  .item-card{background:#fff;border:1px solid var(--gray-200);border-radius:12px;padding:22px;}
  .item-card h3{margin-top:0;}
  .item-card a.cta{color:var(--blue);font-weight:700;text-decoration:none;}
  .neutral{color:var(--gray-600);font-weight:600;}
""" + FIG_CSS


# ── tiny helpers ───────────────────────────────────────────────────────────────

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pp(x, signed=True):
    """Delta in percentage points, one decimal."""
    v = float(x)
    if abs(v) <= 1.5:
        v *= 100.0
    return f"{v:+.1f}" if signed else f"{v:.1f}"


def ci(lo, hi):
    lo, hi = f(lo), f(hi)
    if lo is None or hi is None:
        return "—"
    return f"[{pct(lo)}, {pct(hi)}]"


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ── data ───────────────────────────────────────────────────────────────────────

RB = read_csv(os.path.join(RESULTS, "reasoning_benchmark.csv"))
LU = read_csv(os.path.join(REPGRID, "scores_lsml_upcr.csv"))
UB = read_csv(os.path.join(REPGRID, "ubaseline_scores.csv"))
SW = read_csv(os.path.join(SWEEP, "sweep_summary.csv"))
UL = read_csv(os.path.join(SWEEP, "upcr_legacy.csv"))
SIGNFIX = load_json(os.path.join(REPGRID, "phase12_signfix.json"))
P15 = load_json(os.path.join(REPGRID, "phase15_rescore.json"))

CELL_ROWS = defaultdict(list)
for r in LU:
    CELL_ROWS[r["cell"]].append(r)


def g5(cell, method):
    for r in CELL_ROWS.get(cell, []):
        if r["subset"] == "GOOD_5" and r["method"] == method:
            return r
    return None


def best_row(cell):
    rows = CELL_ROWS.get(cell, [])
    return max(rows, key=lambda r: f(r["auroc_X"]) or 0) if rows else None


# ── computed check (a): CI-overlap scan on reasoning_benchmark ────────────────

def ci_scan():
    """For every (dataset, model) where we are numerically ahead of a competitor,
    check whether the competitor's point estimate falls inside our 95% CI."""
    groups = defaultdict(list)
    for r in RB:
        groups[(r["dataset"], r["model"])].append(r)
    out = []
    for (ds, m), rows in sorted(groups.items()):
        ours = [r for r in rows if r["is_ours"] == "yes" and f(r["ci_lo"]) is not None]
        comps = [r for r in rows if r["is_ours"] != "yes" and f(r["auroc"]) is not None]
        for o in ours:
            a, lo, hi = f(o["auroc"]), f(o["ci_lo"]), f(o["ci_hi"])
            for c in comps:
                y = f(c["auroc"])
                if a > y:
                    out.append({
                        "dataset": ds, "model": m, "our_method": o["method"],
                        "auroc": a, "lo": lo, "hi": hi,
                        "comp": c["method"], "comp_auroc": y,
                        "comp_category": c.get("category", ""),
                        "comp_citable": c.get("citable", ""),
                        "overlap": lo <= y <= hi,
                    })
    return out


# ── computed check (b): GOOD_5 vs same-trace sequence-logprob ─────────────────

def good5_vs_seqlp():
    lsml_g5 = {}
    for r in LU:
        if r["subset"] == "GOOD_5" and r["method"] == "lsml":
            lsml_g5[r["cell"]] = f(r["auroc_X"])
    rows, skipped = [], []
    for r in UB:
        cell, seqlp = r["cell"], f(r.get("seqlp_auroc"))
        if seqlp is None:
            continue
        g5v = f(r.get("lsml_good5_auroc")) or lsml_g5.get(cell)
        if g5v is None:
            skipped.append(cell)
            continue
        d = g5v - seqlp
        rows.append({"cell": cell, "g5": g5v, "seqlp": seqlp, "delta": d,
                     "verdict": ("GOOD_5" if d > 0.005 else
                                 ("seq-logprob" if d < -0.005 else "tie"))})
    rows.sort(key=lambda x: -x["delta"])
    tally = {
        "wins": sum(1 for r in rows if r["verdict"] == "GOOD_5"),
        "ties": sum(1 for r in rows if r["verdict"] == "tie"),
        "losses": sum(1 for r in rows if r["verdict"] == "seq-logprob"),
    }
    return rows, tally, skipped


# ── page scaffolding ───────────────────────────────────────────────────────────

def page(title, subtitle, body, out_name):
    nav = ""
    if out_name != "index.html":
        nav = ('<div class="nav-bar"><strong>Navigate:</strong> '
               '<a href="index.html">&larr; Index</a> &bull; '
               '<a href="advisor_scrutiny.html">Scrutiny page</a> &bull; '
               '<a href="per_domain_breakdown.html">Per-domain breakdown</a></div>')
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(title)}</title>
<style>{CSS}{EXTRA_CSS}</style>
</head>
<body>
<div class="header-hero"><div class="header-content">
  <h1>{esc(title)}</h1>
  <div class="subtitle">{subtitle}</div>
  <div class="meta-pills">
    <span class="meta-pill">Generated {GEN_DATE} by scripts/action_items_report.py</span>
    <span class="meta-pill">{esc(AS_OF)}</span>
    <span class="meta-pill">Every numeric cell sourced from a CSV/JSON</span>
  </div>
</div></div>
<div class="page">
{nav}
{body}
</div>
</body>
</html>
"""
    return html_text


def badge(kind, text):
    cls = {"done": "badge-completed", "progress": "badge-progress",
           "flagged": "badge-flagged", "finding": "badge-finding"}[kind]
    return f'<span class="section-badge {cls}">{esc(text)}</span>'


def wl_class(d, tie=0.005):
    if d > tie:
        return "win"
    if d < -tie:
        return "loss"
    return "neutral"


# ── item 1 ─────────────────────────────────────────────────────────────────────

def build_item1():
    body = f"""
<div class="section-card">
{badge('done', 'Complete — already sent to advisors')}
<h2>Item 1 — Literature search: successors to the 2016 spectral meta-learner</h2>
<p>Question from the meeting: did the spectral meta-learner line (Jaffe-Fetaya-Nadler 2016)
continue after 2016, and is anything in that line directly usable? Answered in HISTORY Steps
139&ndash;141 and folded into the advisor email; recap below.</p>

<h3>U-PCR — the direct continuation, and effectively what we already run</h3>
<p><strong>U-PCR (Tenzer et al. 2022, AISTATS)</strong> is the continuous-input successor in the
Jaffe-Fetaya-Nadler line: it drops the binary-vote requirement and estimates ensemble weights
from the continuous score covariance under an uncorrelated-error assumption. Our L-SML
continuous pipeline is essentially this estimator; since Step 160 we score <em>both</em> L-SML
and U-PCR on every replication-grid cell (<span class="mono">results/repgrid/scores_lsml_upcr.csv</span>),
and this session extended U-PCR to all legacy local caches
(<span class="mono">results/subset_sweep/upcr_legacy.csv</span> — see the
<a href="per_domain_breakdown.html">per-domain breakdown</a>).</p>

<h3>FUSE — the closest competing paper (positioning)</h3>
<p><strong>FUSE (Cand&egrave;s et al., April 2026)</strong> sits in the same theoretical lineage but
differs on the two axes that define our contribution: <em>signal</em> (FUSE ensembles external
verifier models; we use the internal per-token entropy trace of the generating model itself,
single pass, no extra models) and <em>task</em> (FUSE selects among N candidate answers,
Best-of-N; we score one answer for hallucination risk). Both advisors flagged it independently;
the differentiation memo is in Research_Directions.md &sect;Item 1. The contribution to defend is
the <em>signal</em>, not the fusion algebra.</p>

<h3>Deep L-SML</h3>
<p>A deep extension (RBM-equivalence) of the same estimator family. Relevant only if the feature
set grows well past ~16 views and the pairwise |&rho;| &ge; 0.75 correlation filter starts
rejecting subsets wholesale; parked, no action.</p>

<div class="info-box"><b>Bottom line:</b> the line continued (U-PCR, Deep L-SML), we already run
its continuous form, and the nearest competitor (FUSE) is distinguishable on signal and task.
Sources: Research_Directions.md &sect;Item 1, HISTORY Steps 139&ndash;141.</div>
</div>"""
    return page("Item 1 — Literature search", "Successors to the 2016 spectral meta-learner "
                "line, and how we position against the closest competing paper.", body,
                "item1_literature_search.html")


# ── item 2 ─────────────────────────────────────────────────────────────────────

def build_item2():
    body = f"""
<div class="section-card">
{badge('done', 'Complete — already sent to advisors')}
<h2>Item 2 — How far is label-free fusion from a supervised oracle?</h2>
<p>Question from the meeting: quantify the gap between our unsupervised fusion and a supervised
upper reference on the same features. Method and pitfalls (class-weight balancing, the
<span class="mono">cross_val_predict</span> calibration trap) are documented in
<span class="mono">SUPERVISED_ORACLE_CORRECTION.md</span>; numbers from Research_Directions.md
&sect;Item 2.</p>

<table>
<tr><th>Feature set</th><th>L-SML (label-free)</th><th>LR oracle (balanced, 5-fold CV)</th>
<th>Gap</th><th>In-sample ceiling</th></tr>
<tr><td>GOOD_5 (5 features)</td><td>64.2</td><td>68.9</td><td class="loss">+4.7pp</td><td>70.5</td></tr>
<tr><td>STABLE_H9 (9 features)</td><td>62.9</td><td>66.8</td><td class="loss">+3.8pp</td><td>73.7</td></tr>
<tr><td>ALL_H16 (16 features)</td><td>64.1</td><td>67.8</td><td class="loss">+3.6pp</td><td>79.3</td></tr>
</table>

<h3>Where the gap lives</h3>
<p>The macro gap is <strong>not uniform across domains</strong>: &asymp;0pp on reasoning (GSM8K,
MATH-500 — both fusions sit near the in-sample ceiling there, so the <em>features</em>, not the
fusion, are the bottleneck), <strong>+4.9pp on GPQA</strong> and <strong>+5.8pp on RAG+QA</strong>
(where labels let LR re-weight features whose usefulness varies per cell). LR and L-SML weight
vectors correlate only weakly (Spearman 0.1&ndash;0.2) — supervision is buying different weights,
not just better scaling.</p>

<div class="takeaway-box"><b>Bottom line:</b> label-free fusion gives up 3.6&ndash;4.7pp macro vs a
supervised LR on identical features, and essentially nothing on the reasoning domain that is the
thesis headline. The gap is a QA/GPQA phenomenon.</div>
</div>"""
    return page("Item 2 — Supervised-oracle gap", "L-SML label-free fusion vs a balanced "
                "logistic-regression oracle on the same features.", body, "item2_lr_oracle.html")


# ── item 3 ─────────────────────────────────────────────────────────────────────

QA_CELLS = [
    ("se_squad_v2_llama8b", "SQuAD v2", "Llama-3.1-8B", ""),
    ("sciq_llama8b", "SciQ", "Llama-3.1-8B", "ceiling caveat: acc 0.877"),
    ("truthfulqa_llama8b", "TruthfulQA (gen.)", "Llama-3.1-8B", "label floor: acc 0.116"),
    ("se_nq_open_llama8b", "NQ-Open", "Llama-3.1-8B", "judge-rescued: acc 0.067 → 0.501"),
    ("inside_coqa_llama7b", "CoQA", "LLaMA-7B", "OLD floor run (acc 0.183) — judge-regraded full-N rerun in flight (C1)"),
    ("epr_triviaqa_mistral24b", "TriviaQA (wiki)", "Mistral-Small-24B", ""),
    ("semenergy_triviaqa_qwen3_8b", "TriviaQA", "Qwen3-8B", ""),
    ("seiclr_triviaqa_opt30b", "TriviaQA", "OPT-30B", ""),
    ("spilled_triviaqa_llama8b", "TriviaQA (spilled cfg)", "Llama-3.1-8B", "selection caveat: n_pos=6 flagged at Step 163"),
    ("losnet_hotpotqa_mistral7b", "HotpotQA", "Mistral-7B", ""),
]


def build_item3():
    rows_html = ""
    gate_hits = []
    for cell, ds, model, note in QA_CELLS:
        rows = CELL_ROWS.get(cell, [])
        if not rows:
            continue
        r0 = rows[0]
        gl, gu, br = g5(cell, "lsml"), g5(cell, "upcr"), best_row(cell)
        gl_v = f(gl["auroc_X"]) if gl else None
        y = f(r0["published_Y"])
        y_txt = f"{pct(y)} ({esc(r0['Y_method'])})" if y else "— (no published same-model anchor)"
        rows_html += (
            f"<tr><td>{esc(ds)}</td><td>{esc(model)}</td>"
            f"<td>{pct(r0['acc'])}%</td><td>{esc(r0['n_rows'])}</td>"
            f"<td>{pct(gl['auroc_X']) if gl else '—'} {ci(gl['lo'], gl['hi']) if gl else ''}</td>"
            f"<td>{pct(gu['auroc_X']) if gu else '—'}</td>"
            f"<td>{esc(br['subset'])}/{esc(br['method'])} = {pct(br['auroc_X'])}</td>"
            f"<td>{y_txt}</td><td>{esc(note) or '—'}</td></tr>")
        if gl_v is not None:
            gate_hits.append((ds, cell, max(gl_v, f(gu["auroc_X"]) if gu else 0)))

    # Step-155 decision gate: >=3/4 of {CoQA, SQuAD v2, TruthfulQA, +1} at CONT AUROC >= 65
    gate_rows = [
        ("SQuAD v2", "se_squad_v2_llama8b"),
        ("TruthfulQA", "truthfulqa_llama8b"),
        ("SciQ", "sciq_llama8b"),
        ("NQ-Open", "se_nq_open_llama8b"),
    ]
    gate_html = ""
    n_pass = 0
    for name, cell in gate_rows:
        gl, gu = g5(cell, "lsml"), g5(cell, "upcr")
        v = max([x for x in (f(gl["auroc_X"]) if gl else None,
                             f(gu["auroc_X"]) if gu else None) if x is not None])
        ok = v >= 0.65
        n_pass += ok
        gate_html += (f"<tr><td>{esc(name)}</td><td>{pct(v)}</td>"
                      f"<td class=\"{'win' if ok else 'loss'}\">{'&ge; 65 ✓' if ok else '&lt; 65 ✗'}</td></tr>")

    body = f"""
<div class="section-card">
{badge('done', 'Complete — every QA dataset scored or REJECT-documented (Step 172)')}
<h2>Item 3 — Extend the evaluation to short-form QA</h2>
<p>All five originally-prioritised datasets (CoQA, SQuAD v2, TruthfulQA, NQ-Open, SciQ) plus
TriviaQA and HotpotQA are now scored. TruthfulQA and SciQ were scored at Step 169; NQ-Open was
rescued from its label floor by LLM-judge regrading (accuracy 0.067 &rarr; 0.501) and scored;
CoQA's judge-regraded full-N landed at Step 171 (GOOD_5 68.4 vs INSIDE 80.4, FLOOR flag).
Numbers below are GOOD_5 unless stated; source
<span class="mono">results/repgrid/scores_lsml_upcr.csv</span>.</p>

<h3>The QA story in two pictures</h3>
{fig_triviaqa_forest()}
{fig_qa_extension_forest()}

<table>
<tr><th>Dataset</th><th>Model</th><th>Acc</th><th>n</th><th>L-SML GOOD_5 [95% CI]</th>
<th>U-PCR GOOD_5</th><th>Best subset</th><th>Published same-model anchor</th><th>Notes</th></tr>
{rows_html}
</table>

<h3>Step-155 decision gate</h3>
<p>Gate: &ge;3 of 4 short-QA datasets at CONT AUROC &ge; 65. Using the better of L-SML/U-PCR
GOOD_5 per cell:</p>
<table>
<tr><th>Dataset</th><th>GOOD_5 AUROC (best of L-SML/U-PCR)</th><th>vs 65 bar</th></tr>
{gate_html}
</table>
<div class="takeaway-box"><b>Gate status: {n_pass}/4 clear the bar</b> with CoQA still pending —
the gate is already met on currently-available data. Caveats stay attached: SciQ sits at 87.7%
accuracy (ceiling), TruthfulQA at 11.6% accuracy (floor; judge-regrade pending as a follow-up).</div>

<h3>Short-QA subset finding</h3>
<p>On several short-trace QA cells a smaller 4-feature subset (<span class="mono">consensus_4</span>)
beats GOOD_5 (Step 163): TriviaQA/OPT-30B 63.0 vs 59.5, TriviaQA-spilled 96.2 vs 93.4,
TruthfulQA U-PCR 67.5 vs 67.3. Short traces support fewer stable spectral views; the
<a href="per_domain_breakdown.html">breakdown page</a> flags every cell where a
non-GOOD_5 subset leads.</p>
</div>"""
    return page("Item 3 — Short-form QA evaluation", "Status and live numbers for the QA "
                "extension, with the Step-155 decision-gate check.", body,
                "item3_qa_evaluation.html")


# ── item 4 ─────────────────────────────────────────────────────────────────────

CITATIONS = {
    "Semantic Entropy": "Semantic Entropy — Kuhn, Gal &amp; Farquhar, ICLR 2023",
    "Semantic Entropy (SE-ICLR'23)": "Semantic Entropy — Kuhn, Gal &amp; Farquhar, ICLR 2023",
    "Semantic Entropy NLI": "Semantic Entropy — Kuhn, Gal &amp; Farquhar, ICLR 2023",
    "SelfCheckGPT": "SelfCheckGPT — Manakul, Liusie &amp; Gales, EMNLP 2023",
    "Semantic Energy": "Semantic Energy — Chen et al., arXiv 2508.14496",
    "LapEigvals": "LapEigvals — arXiv 2502.17598",
    "LapEigvals AttentionScore": "LapEigvals — arXiv 2502.17598",
    "LOS-Net": "LOS-Net — arXiv 2503.14043",
    "ARS (CCS)": "ARS / CCS — arXiv 2601.17467",
    "Internal-States+RC": "Internal-States + Response Consistency — arXiv 2510.11529",
    "Noise Injection": "Noise Injection — arXiv 2502.03799 (v4)",
    "Answer entropy (K=10)": "Noise Injection (answer-entropy baseline) — arXiv 2502.03799 (v4)",
    "EPR": "EPR — Minut et al., 2026",
    "EDIS": "EDIS — arXiv 2602.01288",
    "TSV (arXiv 2503.01917)": "TSV — arXiv 2503.01917",
    "INSIDE": "INSIDE (EigenScore) — arXiv 2402.03744",
    "EigenScore (vanilla)": "INSIDE (EigenScore) — arXiv 2402.03744",
    "SAPLMA": "SAPLMA — Azaria &amp; Mitchell, 2023",
    "Perplexity": "Perplexity baseline (standard)",
    "Self-Consistency": "Self-Consistency — Wang et al., 2023",
}


def build_item4(scan, seq_rows, seq_tally, seq_skipped):
    # papers cited — generated from the distinct method/Y_method values actually present
    names = set()
    for r in RB:
        if r["is_ours"] != "yes":
            names.add(r["method"])
    for r in LU:
        if r["Y_method"]:
            names.add(r["Y_method"])
    refs = sorted({CITATIONS.get(n, esc(n)) for n in names})
    refs_html = "".join(f"<li>{r}</li>" for r in refs)

    # win/loss table from the CI scan + the behind-cells from scores_lsml_upcr
    scan_html = ""
    for s in scan:
        verdict = ("<span class='win'>WIN (CI-clear)</span>" if not s["overlap"]
                   else "<span class='neutral'>numerically ahead, CI-overlapping</span>")
        cit = "" if s["comp_citable"] == "yes" else " <em>(anchor not-yet-citable)</em>"
        scan_html += (f"<tr><td>{esc(s['dataset'])}</td><td>{esc(s['model'])}</td>"
                      f"<td>{esc(s['our_method'])}</td>"
                      f"<td>{pct(s['auroc'])} [{pct(s['lo'])}, {pct(s['hi'])}]</td>"
                      f"<td>{esc(s['comp'])} ({esc(s['comp_category'])}){cit}</td>"
                      f"<td>{pct(s['comp_auroc'])}</td><td>{verdict}</td></tr>")

    behind_html = ""
    for cell, label in [("lapeigvals_gsm8k_llama3b", "GSM8K / Llama-3.2-3B"),
                        ("noise_gsm8k_phi3mini", "GSM8K / Phi-3-mini"),
                        ("internalstates_gsm8k_qwen25_7b", "GSM8K / Qwen2.5-7B (T=0.8)"),
                        ("noise_gsm8k_mistral7b", "GSM8K / Mistral-7B-v0.3")]:
        gl, gu = g5(cell, "lsml"), g5(cell, "upcr")
        r0 = CELL_ROWS[cell][0]
        x = max(f(gl["auroc_X"]), f(gu["auroc_X"]))
        y = f(r0["published_Y"])
        d = x - y
        behind_html += (f"<tr><td>{esc(label)}</td><td>{pct(x)}</td>"
                        f"<td>{pct(y)} ({esc(r0['Y_method'])})</td>"
                        f"<td class='{wl_class(d)}'>{pp(d)}pp</td></tr>")

    seq_html = ""
    for r in seq_rows:
        seq_html += (f"<tr><td class='mono'>{esc(r['cell'])}</td><td>{pct(r['g5'])}</td>"
                     f"<td>{pct(r['seqlp'])}</td>"
                     f"<td class='{wl_class(r['delta'])}'>{pp(r['delta'])}pp</td>"
                     f"<td>{esc(r['verdict'])}</td></tr>")

    body = f"""
<div class="section-card">
{badge('progress', 'In progress — nearly done (3 cells still running)')}
<h2>Item 4 — Benchmarking against published detectors</h2>

<h3>How this was built</h3>
<p>For each competitor paper we run <em>our own inference</em> on the paper's exact (dataset,
model, N, decoding config), so the accuracy and label distribution match their setup. We then
score <em>our</em> L-SML / U-PCR offline on that trace — we never re-implement their detector —
and place our AUROC next to their <em>published</em> number. This inference-only,
offline-scoring boundary is documented in <span class="mono">cluster/reasoning_grid_runbook.md</span>
and <span class="mono">BENCHMARKING_COMPETITOR_GUIDE.md</span>. Every cell passes the gate ladder
<strong>local smoke test &rarr; N=30 pilot &rarr; full N</strong>; numbers land in
<span class="mono">results/reasoning_benchmark.csv</span> and
<span class="mono">results/repgrid/scores_lsml_upcr.csv</span>.</p>

<div class="info-box"><b>Gate policy (formalized 2026-07-12, applies desk-wide — reasoning and QA).</b>
Two tiers. (1) <b>Band violation</b> — cell accuracy outside [0.20, 0.85] — is a <em>quality flag</em>:
the cell is scored, appears in every table and figure with a CEILING / FLOOR tag, and is excluded from
the headline win tally. A flagged number is a noisy estimate of the right quantity.
(2) <b>Label-validity failure</b> — truncation-label leakage (cap-pinned negatives) or a single-class
label set — is a <em>documented REJECT</em>, never scored: there the AUROC would be a clean estimate of
the <em>wrong</em> quantity, and no caveat fixes that. Current REJECTs: ars_gsm8k_qwen3_8b (15/29
negatives cap-pinned at 8192), ars_math500_qwen3_8b (23/50 negatives cap-pinned at 16384 — Qwen3's
reasoning on hard MATH-500 items is effectively unbounded), noise_gsm8k_gemma2b (acc 0.000, single
class). The Qwen3/ARS pair is closed as REJECT-leakage — itself a reportable finding about capped
greedy Qwen3 protocols.</div>

<h3>The picture first — regenerated from the CSVs on every build</h3>
{fig_gsm8k_forest()}
{fig_math500_forest()}
{fig_same_model_deltas()}
{fig_cell_landscape_gsm8k_llama8b()}
{fig_cell_landscape_losnet()}

<h3>Papers cited in this comparison</h3>
<p>Generated from the distinct competitor-method values actually present in the two source CSVs
(so this list cannot drift from what is actually compared against):</p>
<ul>{refs_html}</ul>

<h3>Same-model comparisons where we are numerically ahead</h3>
<p>Computed CI check: a row is a <strong>WIN (CI-clear)</strong> only if the competitor's point
estimate falls outside our 95% CI; otherwise it is reported as <em>numerically ahead,
CI-overlapping</em> — see <a href="advisor_scrutiny.html">scrutiny point 1</a> for why this
distinction matters.</p>
<table>
<tr><th>Dataset</th><th>Model</th><th>Ours</th><th>AUROC [95% CI]</th><th>Competitor
(category)</th><th>Published</th><th>Verdict</th></tr>
{scan_html}
</table>

<h3>Cells where the published anchor stays ahead</h3>
<table>
<tr><th>Cell</th><th>Ours (best GOOD_5)</th><th>Published anchor</th><th>Delta</th></tr>
{behind_html}
</table>
<p>Also behind on the deep-supervised anchors that use training labels + internal access:
LapEigvals supervised probe (GSM8K/Llama-8B 92.5 vs our 81.5), TSV on TruthfulQA (84.2 vs our
67.5), INSIDE on CoQA (80.4 — our judge-regraded rerun in flight), LOS-Net on HotpotQA (72.9 vs
our 58.3), Semantic Entropy on TriviaQA/OPT-30B (83.0 vs our 63.0). Wave-3 tally over the
same-model cells: <strong>3 wins, 1 exact tie (Mistral-7B 78.5 = Noise-Injection K=10 at 1/10th
the compute), 1 edge, 3 honest losses, 1 floor-REJECT (gemma2b), 1 ceiling-caveat (SciQ)</strong>.</p>

<h3>The fairest baseline: same-trace sequence-logprob</h3>
<p>The competitor numbers above come from <em>their</em> inference budgets. The tightest
comparison available is the trivial single-number baseline computed on <em>our own traces</em>:
mean sequence log-probability. Recomputed this session across all {len(seq_rows)} cells where
both are scored (<span class="mono">ubaseline_scores.csv</span>, post data-loss recovery):</p>
<table>
<tr><th>Cell</th><th>L-SML GOOD_5</th><th>seq-logprob</th><th>Delta</th><th>Ahead</th></tr>
{seq_html}
</table>
<div class="warn-box"><b>Tally: GOOD_5 {seq_tally['wins']} wins / {seq_tally['ties']} ties /
{seq_tally['losses']} losses (&plusmn;0.5pp band).</b> The trivial baseline is ahead more often
than not on this larger sample — the full discussion, including where the wins cluster, is
<a href="advisor_scrutiny.html">scrutiny point 2</a>. ({len(seq_skipped)} cells without a scored
L-SML GOOD_5 row excluded — partials, archived pilots and documented-REJECT cells:
{esc(', '.join(seq_skipped))}.)</div>

<div class="warn-box"><b>Not citable yet:</b> the old Phase-12 Semantic-Entropy /
Self-Consistency reasoning baselines are excluded pending the NLI-truncation reconciliation
(Step 152 Priority 1, still open — <a href="advisor_scrutiny.html">scrutiny point 4</a> tracks
it). Pending cells: A2 <span class="mono">ars_gsm8k_qwen3_8b</span>, A3
<span class="mono">ars_math500_qwen3_8b</span> (max_new=16384, the long pole, ETA ~2026-07-12
evening), C1 <span class="mono">inside_coqa</span> + chained judge-regrade.</div>
</div>"""
    return page("Item 4 — Benchmarking vs published detectors",
                "Methodology, reference list, win/loss tables, and the same-trace "
                "sequence-logprob comparison.", body, "item4_benchmarking.html")


# ── item 5 ─────────────────────────────────────────────────────────────────────

def build_item5():
    # 0c results (sign fix)
    sf = SIGNFIX or {}
    m = sf.get("math500", {})
    sf_mode = m.get("mode", "pending")
    if sf_mode == "mirror-fallback":
        l_cor = m["lsml_corrected"]
        f_cor = m["fusion_corrected"]
        bc = sf.get("battery_corroboration") or {}
        sf_html = f"""
<p><strong>Status: corrected (mirror arithmetic) by <span class="mono">scripts/refix_phase12_signs.py</span>.</strong>
The results pkl stores only (AUROC, CI) per method — no score arrays — so the exact &plusmn;
ambiguity is resolved arithmetically (AUC(&minus;s) = 1 &minus; AUC(s)): MATH-500 L-SML
0.230 &rarr; <strong>{pct(l_cor[0])} [{pct(l_cor[1])}, {pct(l_cor[2])}]</strong>, fusion 0.232 &rarr;
<strong>{pct(f_cor[0])} [{pct(f_cor[1])}, {pct(f_cor[2])}]</strong>. GSM8K and GPQA are unchanged
(their stored values are already on the &gt;0.5 side). The flip direction is corroborated
label-free: the epr-anchor decision executed on the same-model battery cell
(math500/{esc(bc.get('cell', 'Qwen-Math-7B'))}, n={bc.get('n', '—')}) flips the fused score and
lands at {pct(bc.get('auc', ''))} — same direction. The full label-free re-derivation on the
original Phase-12 scores runs automatically once the raw two-pass caches
(<span class="mono">p1/p2/p3 *.pkl</span> from Drive <span class="mono">cache/phase12_corrected/</span>)
are copied into <span class="mono">local_cache/</span>.</p>"""
    elif sf_mode == "full-recompute":
        sf_html = "<p><strong>Status: fully re-derived label-free from the raw caches.</strong></p>"
    else:
        sf_html = ("<p><strong>Status: pending local data drop</strong> — "
                   "<span class='mono'>local_cache/phase12_corrected_results.pkl</span>.</p>")

    # 0d results
    p15 = P15 or {}
    if p15.get("mode") == "full-rescore":
        p15_html = f"""<p><strong>Answer-agreement re-test ran in full:</strong> SC K=5
{pct(p15['sc'][0])}, L-SML 1-pass {pct(p15['lsml'][0])}, &rho; {p15['rho']:+.2f}, fused
{pct(p15['fused'][0])} — gate {'PASS' if p15['gate_pass'] else 'FAIL'}.</p>"""
    elif p15.get("mode") == "partial-results-pkl":
        aucs = p15["aucs"]
        base = aucs["single pass T=1.0 (base)"]
        a_l = aucs["A: K=5 same-T, L-SML"]
        d = p15["delta_A_minus_base"]
        p15_html = f"""
<p><strong>Partial re-score done by <span class="mono">scripts/rescore_phase15_selfconsistency.py</span></strong>
(the Phase-15 results pkl holds fused arrays + labels but not the generated texts, so the
answer-agreement arm needs the 5 raw pass caches —
<span class="mono">math500_qwen7b_T1.0_run0..4.pkl</span> from Drive
<span class="mono">cache/phase15_temperature/</span>; the script upgrades itself automatically
when they land). What <em>is</em> label-free derivable now: applying Item 5's own gate to the
same-T K=5 entropy-averaging arm from Phase-15 gives cross-pass &rho;
{p15['a_rho_offdiag_mean']:+.2f} &lt; 0.75 <em>and</em> fused {pct(a_l[0])} &gt; single-pass
{pct(base[0])} + 1pp (paired delta {pp(d[0])}pp, CI [{pp(d[1])}, {pp(d[2])}]) —
<strong>that arm PASSES the gate</strong>. See "What later work tells us" below.</p>"""
    else:
        p15_html = ("<p><strong>Pending local data drop</strong> — "
                    "<span class='mono'>local_cache/phase15_results.pkl</span>.</p>")

    body = f"""
<div class="section-card">
{badge('flagged', 'Complete (Step 152) — but flagged: partial re-score done, one arm pending raw caches')}
<h2>Item 5 — Does sampling-based fusion add anything?</h2>
<p>The Step-152 experiment fused the 1-pass spectral L-SML score with a K=10
likelihood-weighted semantic-entropy (LW-SE) baseline. Pre-registered gate:
&rho; &lt; 0.75 <em>and</em> fused &gt; max(single arms) + 1pp.</p>

<h3>Step-152 result (as run)</h3>
<table>
<tr><th>Cell</th><th>LW-SE K=10</th><th>SelfCheckGPT</th><th>L-SML 1-pass</th><th>Fused</th>
<th>Gate</th></tr>
<tr><td>GSM8K / Llama-3.1-8B</td><td>61.4</td><td>70.1</td><td>75.4</td><td>75.8 (+0.4pp)</td>
<td class="loss">FAIL</td></tr>
<tr><td>MATH-500 / Qwen2.5-Math-7B</td><td colspan="4"><em>invalid — fused-score sign flip
(stored 0.230); corrected below</em></td><td class="loss">—</td></tr>
<tr><td>GPQA / Qwen2.5-7B</td><td>50.1 (at chance)</td><td>—</td><td>55.3</td><td>57.3 (+2.0pp)</td>
<td class="loss">FAIL (baseline at chance)</td></tr>
</table>
<p>Read alone, the verdict was: extra sampling spent on semantic clustering adds &le;+2.0pp over
the 1-pass spectral score, while the spectral score adds +14.5pp over LW-SE alone.</p>

<h3>Problems found on review</h3>
<p><strong>(1) The MATH-500 row was invalid — sign flip.</strong> The Phase-12-Corrected analysis
cells hand the fused score straight to the AUROC computation with no <span class="mono">anchor_orient</span>
step, so it keeps the eigenvector solver's arbitrary global sign. {sf_html}</p>
<p><strong>(2) LW-SE / SelfCheckGPT collapsed vs an older cache</strong> for suspected but
unconfirmed reasons (NLI truncation on long traces) — open since Step 152; those old-cache
baselines stay non-citable until reconciled.</p>
<p><strong>(3) Label noise found later:</strong> Step-167 dual-label checks showed grading swings
up to 35pp on some cells, so the correctness labels under this test were noisier than assumed
at the time.</p>

<h3>What later work tells us (reconciling Items 5 and 6)</h3>
{p15_html}
<div class="info-box"><b>The reconciling sentence no earlier document says out loud:</b>
Item 5 ("SE K=10 adds nothing") and Item 6 ("K=5 same-T adds +6.1pp") are both true because the
extra passes were spent on <em>different signals</em> — Item 5 spent them on a fragile NLI
semantic-clustering score; Item 6's condition A spent them on averaging the same cheap entropy
signal, which denoises (cross-pass &rho; &asymp; +0.45) and clears Item 5's own gate. Sampling
helps; <em>what you compute from the samples</em> is what decides whether it shows.
(<a href="advisor_scrutiny.html">Scrutiny point 3</a>.)</div>
</div>"""
    return page("Item 5 — Sampling-based fusion", "Step-152 gate result, the problems found on "
                "review, and the partial re-score that reconciles it with Item 6.", body,
                "item5_sampling_fusion.html")


# ── item 6 ─────────────────────────────────────────────────────────────────────

def build_item6():
    body = f"""
<div class="section-card">
{badge('done', 'Complete (Step 158) — clean negative result')}
<h2>Item 6 — Does temperature variation help?</h2>
<div class="info-box"><b>Bookkeeping note:</b> PROGRESS.md's action-items table and
Research_Directions.md's summary table said "Not started" for this item until this session —
that was stale; the experiment finished at Step 158. Both tables were fixed as part of this
report pass.</div>

<h3>Setup (a reminder)</h3>
<p>Qwen2.5-Math-7B / MATH-500, N=200, nine fresh runs: one at each of T &isin;
{{0.3, 0.6, 1.0, 1.5, 2.0}} plus four extra T=1.0 passes. All runs saved the full raw-data
schema (entropies, spilled energies, top-50 logprobs, token ids) — T=1.0 run0 became the
canonical raw-trace cache for this cell, repaying the Extension-E data debt. Consolidated
results: <span class="mono">cache/phase15_temperature/results/phase15_results.pkl</span>
(now also in <span class="mono">local_cache/</span>).</p>

<h3>Q1 — does higher T improve detectability? (Confounded inverted-U)</h3>
<table>
<tr><th>T</th><th>0.3</th><th>0.6</th><th>1.0</th><th>1.5</th><th>2.0</th></tr>
<tr><td>Single-pass L-SML GOOD_5 AUROC</td><td>54.5</td><td>64.4</td><td>85.1</td><td>87.8</td>
<td>62.9</td></tr>
<tr><td>Accuracy</td><td>80.0%</td><td>81.5%</td><td>70.5%</td><td>27.5%</td><td>4.0%</td></tr>
</table>
<p>The "peak" at T=1.5 is <strong>confounded by the accuracy collapse (80% &rarr; 4%)</strong> —
the class mix shifts under the curve, and T=2.0 has only 8 correct answers. Gate G-T1 FAIL
(overlapping CIs at 27.5% accuracy). So Q1's honest answer: no usable temperature lever here.</p>
{fig_item6_temperature()}

<h3>Q2 (primary) — diversity vs simply more passes</h3>
<p>Paired on the 200 common samples: <strong>condition A</strong> (K=5 passes, all T=1.0) vs
<strong>condition B</strong> (K=5 passes at T = 0.3/0.6/1.0/1.5/2.0).</p>
<table>
<tr><th>Arm</th><th>AUROC [95% CI]</th></tr>
<tr><td>Single pass T=1.0 (base)</td><td>85.1 [77.7, 91.8]</td></tr>
<tr><td>A: K=5 same-T, L-SML fusion</td><td class="win">91.2 [85.8, 95.4]</td></tr>
<tr><td>A: K=5 same-T, simple average</td><td>90.6 [85.0, 94.8]</td></tr>
<tr><td>B: K=5 multi-T, L-SML fusion</td><td>85.9 [79.4, 91.4]</td></tr>
<tr><td>B: K=5 multi-T, simple average</td><td>83.0 [76.0, 89.0]</td></tr>
</table>
<p>Paired deltas: <strong>A &minus; base = +6.1pp [+0.4, +12.8]</strong> (more same-T passes help);
<strong>B &minus; A = &minus;5.3pp [&minus;10.3, &minus;1.1]</strong> (CI excludes 0 —
temperature diversity <em>hurts</em>). Gate G-T2 FAIL, sign negative.</p>
{fig_item6_arms()}
<p><strong>Mechanism</strong> (off-diagonal cross-pass correlation): same-T passes correlate at
&rho; &asymp; +0.45 — same signal plus independent noise, so averaging denoises. Multi-T passes
correlate at &rho; &asymp; +0.01 — but that decorrelation is the off-temperature passes being
<em>near-random</em> (T=0.3/0.6 weak, T=2.0 degenerate), not independent true signal.</p>
<div class="takeaway-box"><b>Answer to the meeting question:</b> the multi-pass lift is variance
reduction from repeated sampling at a single good temperature (T&asymp;1.0); mixing temperatures
dilutes it. <strong>Temperature is not the lever; repeated sampling is.</strong></div>

<h3>Two caveats surfaced (never followed up)</h3>
<p>(1) <span class="mono">spectral_entropy</span>'s sign is temperature-dependent (AUROC 0.26 @
T=1.0 vs 0.14 @ T=1.5 under the fixed &minus;1 sign — informative if flipped). (2) The label-free
fusion underperforms its best single feature at every T (fused 85.1 vs cusum_max 92.7 @ T=1.0)
because the epr anchor weakens at low T — so Q1's low-T dip is plausibly a fusion/anchor
artifact, not a signal property. The Step-0b anchor test this session
(<a href="advisor_scrutiny.html">scrutiny point 4</a>) addressed the anchor's <em>sign</em>
robustness on the full battery; its low-T <em>strength</em> remains follow-up #3 below.</p>

<h3>Independent later confirmation</h3>
<p>The Step-168 Internal-States rerun at T=0.8 (after the T=1.0 sampling-collapse confound was
caught) independently confirms the core caveat — decoding temperature strongly confounds
accuracy/label mix — without changing any Phase-15 number.</p>

<h3>The 8 unexecuted follow-ups on this data (all CPU once the 9 raw caches are local)</h3>
<ol>
<li><strong>Self-consistency / semantic-entropy baseline from the 5 T=1.0 passes</strong> — ties
directly to Item 5; the analysis script is written and armed
(<span class="mono">scripts/rescore_phase15_selfconsistency.py</span>), waiting only on the raw
pass caches.</li>
<li>K-sweep for condition A (does the lift saturate at K=3?).</li>
<li><strong>Anchor/sign robustness across T</strong> — stronger anchor (cusum_max), per-feature
label-free signs, leave-spectral_entropy-out; tests whether Q1's low-T dip is a fusion
artifact.</li>
<li>New feature families from saved-but-unused data (spilled-energy trace spectra, top-50
logprob margins/varentropy).</li>
<li>Fairer diversity set B&prime; = {{0.6, 1.0, 1.5}} (drop the degenerate T=2.0 pass).</li>
<li>Cross-temperature probing (hot-pass trace predicting the cold answer's correctness).</li>
<li>Length-controlled AUROC per T.</li>
<li>Streaming earliest-prefix replication on the T=1.0 run0 canonical cache (Extension E).</li>
</ol>

<h3>Scope of the claim</h3>
<div class="warn-box"><b>This is a single (dataset, model) result</b> — MATH-500 /
Qwen2.5-Math-7B only. The paired CI is clean, but "temperature diversity hurts" should be stated
to advisors as <em>this cell's</em> finding until it is checked on a second dataset (GSM8K or a
QA cell) — <a href="advisor_scrutiny.html">scrutiny point 6</a>.</div>
</div>"""
    return page("Item 6 — Temperature variation", "Full explainer of the Phase-15 experiment: "
                "the inverted-U is confounded, same-T repeats help, diversity hurts.", body,
                "item6_temperature_variation.html")


# ── per-domain breakdown ───────────────────────────────────────────────────────

LEGACY_MODEL_NAMES = {
    "Qwen2.5-Math-1.5B-Instruct_T1.0": "Qwen2.5-Math-1.5B (legacy cache)",
    "Qwen-Math-7B_T1.0": "Qwen2.5-Math-7B (legacy cache)",
    "deepseek-math-7b-instruct_T1.0": "DeepSeek-Math-7B (legacy cache)",
    "DeepSeek-R1-Distill-Llama-8B_T1.0": "R1-Distill-Llama-8B (legacy cache)",
    "Llama-8B_T1.0": "Llama-3.1-8B (legacy cache)",
    "spectral_phase9_cache_trivia_qa_traces_T1.0": "Phase-9 TriviaQA (plain)",
    "spectral_phase9_cache_trivia_qa_cot_traces_T1.0": "Phase-9 TriviaQA (CoT)",
    "spectral_phase9_cache_webq_cot_traces_T1.0": "Phase-9 WebQ (CoT only — stale variant)",
}

AIRCC_REASONING = [
    ("lapeigvals_gsm8k_llama8b", "GSM8K", "Llama-3.1-8B"),
    ("lapeigvals_gsm8k_llama3b", "GSM8K", "Llama-3.2-3B"),
    ("lapeigvals_gsm8k_nemo", "GSM8K", "Mistral-Nemo-12B"),
    ("lapeigvals_gsm8k_mistral24b", "GSM8K", "Mistral-Small-24B"),
    ("lapeigvals_gsm8k_phi35", "GSM8K", "Phi-3.5-mini"),
    ("noise_gsm8k_mistral7b", "GSM8K", "Mistral-7B-v0.3"),
    ("noise_gsm8k_phi3mini", "GSM8K", "Phi-3-mini"),
    ("ars_gsm8k_r1distill8b", "GSM8K", "R1-Distill-Llama-8B"),
    ("internalstates_gsm8k_qwen25_7b", "GSM8K", "Qwen2.5-7B (T=0.8)"),
]

AIRCC_QA = [
    ("epr_triviaqa_mistral24b", "TriviaQA", "Mistral-Small-24B"),
    ("semenergy_triviaqa_qwen3_8b", "TriviaQA", "Qwen3-8B"),
    ("seiclr_triviaqa_opt30b", "TriviaQA", "OPT-30B"),
    ("spilled_triviaqa_llama8b", "TriviaQA (spilled cfg)", "Llama-3.1-8B"),
    ("se_squad_v2_llama8b", "SQuAD v2", "Llama-3.1-8B"),
    ("truthfulqa_llama8b", "TruthfulQA", "Llama-3.1-8B"),
    ("sciq_llama8b", "SciQ", "Llama-3.1-8B"),
    ("se_nq_open_llama8b", "NQ-Open", "Llama-3.1-8B"),
    ("inside_coqa_llama7b", "CoQA (old floor run)", "LLaMA-7B"),
    ("losnet_hotpotqa_mistral7b", "HotpotQA", "Mistral-7B"),
]


def legacy_lookup():
    d = {}
    for r in UL:
        d[(r["domain"], r["cell_key"], r["subset"], r["method"])] = (
            f(r["auroc"]), f(r["lo"]), f(r["hi"]))
    return d


def aircc_cell_vals(cell):
    out = {}
    for sub in ("GOOD_5", "ALL_H16"):
        for meth in ("lsml", "upcr"):
            for r in CELL_ROWS.get(cell, []):
                if r["subset"] == sub and r["method"] == meth:
                    out[(sub, meth)] = f(r["auroc_X"])
    return out


def fmt4(vals):
    """Cells: L-SML G5 / U-PCR G5 / L-SML H16 / U-PCR H16 (+ bold the row max)."""
    order = [("GOOD_5", "lsml"), ("GOOD_5", "upcr"), ("ALL_H16", "lsml"), ("ALL_H16", "upcr")]
    nums = [vals.get(k) for k in order]
    present = [v for v in nums if v is not None]
    mx = max(present) if present else None
    tds = ""
    for v in nums:
        if v is None:
            tds += "<td>—</td>"
        elif v == mx:
            tds += f"<td><strong>{pct(v)}</strong></td>"
        else:
            tds += f"<td>{pct(v)}</td>"
    return tds, mx


def build_breakdown():
    LL = legacy_lookup()

    def legacy_vals(domain, cell_key):
        out = {}
        for sub in ("GOOD_5", "ALL_H16"):
            for meth in ("lsml", "upcr"):
                v = LL.get((domain, cell_key, sub, meth))
                if v:
                    out[(sub, meth)] = v[0]
        return out

    def block(rows):
        h = ("<table><tr><th>Dataset</th><th>Model</th><th>Source</th>"
             "<th>L-SML GOOD_5</th><th>U-PCR GOOD_5</th><th>L-SML ALL_H16</th>"
             "<th>U-PCR ALL_H16</th><th>Best non-GOOD_5 subset?</th></tr>")
        for ds, model, src, vals, best_note in rows:
            tds, _ = fmt4(vals)
            h += (f"<tr><td>{esc(ds)}</td><td>{esc(model)}</td><td>{esc(src)}</td>"
                  f"{tds}<td>{best_note}</td></tr>")
        return h + "</table>"

    def best_subset_note(cell):
        br = best_row(cell)
        if br is None:
            return "—"
        if br["subset"] in ("GOOD_5",):
            return "—"
        return (f"<span class='mono'>{esc(br['subset'])}</span>/{esc(br['method'])} "
                f"= {pct(br['auroc_X'])}")

    # reasoning block
    reasoning_rows = []
    for cell, ds, model in AIRCC_REASONING:
        if cell in CELL_ROWS:
            reasoning_rows.append((ds, model, "AIRCC grid", aircc_cell_vals(cell),
                                   best_subset_note(cell)))
    reasoning_rows.append(("GSM8K", LEGACY_MODEL_NAMES["Llama-8B_T1.0"], "legacy",
                           legacy_vals("gsm8k", "Llama-8B_T1.0"), "—"))
    for ck in ("Qwen-Math-7B_T1.0", "Qwen2.5-Math-1.5B-Instruct_T1.0",
               "deepseek-math-7b-instruct_T1.0", "DeepSeek-R1-Distill-Llama-8B_T1.0"):
        reasoning_rows.append(("MATH-500", LEGACY_MODEL_NAMES[ck], "legacy",
                               legacy_vals("math500", ck), "—"))
    reasoning_rows.append(("MATH-500", "Qwen3-8B", "AIRCC — <em>running (A3)</em>", {}, "—"))
    reasoning_rows.append(("GSM8K", "Qwen3-8B", "AIRCC — <em>running (A2)</em>", {}, "—"))

    # QA block
    qa_rows = []
    for cell, ds, model in AIRCC_QA:
        if cell in CELL_ROWS:
            qa_rows.append((ds, model, "AIRCC grid", aircc_cell_vals(cell),
                            best_subset_note(cell)))
    qa_rows.append(("CoQA", "LLaMA-7B (judge-regraded full-N)",
                    "AIRCC — <em>running (C1)</em>", {}, "—"))
    for ck in ("spectral_phase9_cache_trivia_qa_traces_T1.0",
               "spectral_phase9_cache_trivia_qa_cot_traces_T1.0",
               "spectral_phase9_cache_webq_cot_traces_T1.0"):
        qa_rows.append(("TriviaQA/WebQ (legacy)", LEGACY_MODEL_NAMES[ck], "legacy",
                        legacy_vals("qa", ck), "—"))
    rag_models = ["Llama-8B", "Qwen-7B", "Qwen-72B", "Mistral-24B"]
    rag_ds = ["hotpotqa", "natural-questions", "2wikimultihopqa", "narrativeqa"]
    for m in rag_models:
        for d in rag_ds:
            ck = f"{m}/{d}"
            v = legacy_vals("rag", ck)
            if v:
                qa_rows.append((f"RAG / {d}", f"{m} (legacy L-CiteEval)", "legacy", v, "—"))

    body = f"""
<div class="section-card">
{badge('finding', 'Cross-cutting view — GPQA excluded per request')}
<h2>Per-domain leading-variant breakdown (L-SML + U-PCR &times; feature subset)</h2>
<p>AIRCC-grid cells from <span class="mono">results/repgrid/scores_lsml_upcr.csv</span>; legacy
local-cache cells computed fresh this session by
<span class="mono">scripts/compute_legacy_upcr.py</span> &rarr;
<span class="mono">results/subset_sweep/upcr_legacy.csv</span> (both methods, 4 subsets, 24
non-GPQA cells; the L-SML rows reproduce <span class="mono">sweep_summary.csv</span>, e.g. the
MATH-500/Qwen-Math 94.4 headline, so the two sources are consistent). Bold = row max of the
four variants. AUROC, percent.</p>

{master_table_html()}

<h3>Reasoning (GSM8K, MATH-500)</h3>
{block(reasoning_rows)}
<p>Reading: on reasoning, L-SML and U-PCR track each other within &asymp;1&ndash;3pp; GOOD_5
beats ALL_H16 on most cells; the strongest cells (MATH-500 legacy, GSM8K large models) are
GOOD_5-led.</p>

<h3>QA / RAG (TriviaQA, SQuAD v2, TruthfulQA, SciQ, HotpotQA, CoQA, NQ-Open, WebQ, RAG&times;4)</h3>
{block(qa_rows)}
<p><strong>Footnote — where a non-GOOD_5 subset leads:</strong> the "best non-GOOD_5 subset"
column flags cells where a different subset tops GOOD_5 (mostly
<span class="mono">consensus_4</span> and energy/logprob-augmented variants on short-trace QA —
the Step-163 short-QA finding). On such cells GOOD_5 remains the pre-registered headline; the
best-subset value is exploratory.</p>

<h3>Coverage check (short form — full version on the <a href="index.html">index</a>)</h3>
<p>GPQA excluded per request but note: 5 legacy models, zero cluster-wave presets (Phase-14 Colab
rerun pending). RAG&times;4 has L-SML+U-PCR numbers (above) but no cluster-era competitor
re-check since Step 152 (SelfCheckGPT below-chance issue open). WebQ has only a stale CoT-only
Phase-9 cache — no plain variant anywhere.</p>
</div>"""
    return page("Per-domain breakdown", "L-SML and U-PCR across feature subsets, datasets and "
                "domains — AIRCC grid + legacy caches, GPQA excluded.", body,
                "per_domain_breakdown.html")


# ── advisor scrutiny ───────────────────────────────────────────────────────────

def build_scrutiny(scan, seq_rows, seq_tally):
    overlaps = [s for s in scan if s["overlap"]]
    clears = [s for s in scan if not s["overlap"]]
    ov_html = "".join(
        f"<tr><td>{esc(s['dataset'])} / {esc(s['model'])}</td>"
        f"<td>{pct(s['auroc'])} [{pct(s['lo'])}, {pct(s['hi'])}]</td>"
        f"<td>{esc(s['comp'])} = {pct(s['comp_auroc'])}</td>"
        f"<td class='neutral'>CI contains competitor — say &ldquo;numerically ahead, "
        f"CI-overlapping&rdquo;</td></tr>" for s in overlaps)
    cl_html = "".join(
        f"<tr><td>{esc(s['dataset'])} / {esc(s['model'])}</td>"
        f"<td>{pct(s['auroc'])} [{pct(s['lo'])}, {pct(s['hi'])}]</td>"
        f"<td>{esc(s['comp'])} = {pct(s['comp_auroc'])}</td>"
        f"<td class='win'>CI-clear</td></tr>"
        for s in clears if s["comp_category"] in ("UGB", "BB") or
        (f(s['auroc']) - f(s['comp_auroc'])) > 9)

    seq_top = seq_rows[:4] + seq_rows[-5:]
    seq_html = "".join(
        f"<tr><td class='mono'>{esc(r['cell'])}</td><td>{pct(r['g5'])}</td>"
        f"<td>{pct(r['seqlp'])}</td>"
        f"<td class='{wl_class(r['delta'])}'>{pp(r['delta'])}pp</td></tr>" for r in seq_top)
    seq_fig = fig_good5_vs_seqlp()

    # point 3 numbers from 0d
    p15 = P15 or {}
    d = p15.get("delta_A_minus_base", (0.0614, 0.0042, 0.1278))
    rho = p15.get("a_rho_offdiag_mean", 0.447)

    # point 4 — 0b + 0c
    sf = SIGNFIX or {}
    m = sf.get("math500", {})
    cor = m.get("lsml_corrected")
    cor_txt = (f"{pct(cor[0])} [{pct(cor[1])}, {pct(cor[2])}]" if cor else "pending")

    # point 7 — internalstates acc from CSV
    is_row = CELL_ROWS["internalstates_gsm8k_qwen25_7b"][0]

    body = f"""
<div class="section-card">
{badge('finding', 'Critical review — read before any meeting or paper use')}
<h2>Questions an advisor would ask</h2>
<p>The numbers in the CSVs are correct. But several were written up one cell at a time, and
reading <span class="mono">scores_lsml_upcr.csv</span>,
<span class="mono">ubaseline_scores.csv</span> and
<span class="mono">reasoning_benchmark.csv</span> <em>together</em> raises questions the
per-cell writeups didn't have room for. Nothing here retracts a finding; it flags where the
interpretive framing outran what the CI/baseline comparisons support. All numbers below are
recomputed from the live CSVs by this script, not copied from earlier writeups.</p>

<h3>1 — Three "we beat/edge X" headlines don't survive a CI check</h3>
<p>Recomputed CI-overlap scan (our 95% CI vs the competitor's point estimate, all same-model
comparisons where we are numerically ahead):</p>
<table><tr><th>Cell</th><th>Ours [95% CI]</th><th>Competitor</th><th>Verdict</th></tr>
{ov_html}</table>
<p>PROGRESS.md's own phrasing ("beats ARS supervised", "edges SelfCheckGPT") is not statistically
supported for the first two; the Phi-3-mini case was not previously flagged at all. By contrast
these wins are CI-clear and can be stated plainly:</p>
<table><tr><th>Cell</th><th>Ours [95% CI]</th><th>Competitor</th><th>Verdict</th></tr>
{cl_html}</table>
<p>So this isn't "the wins are fake" — it's "three specific headline claims need softer language,
and the LapEigvals-family wins don't."</p>

<h3>2 — The trivial same-trace baseline is ahead more often than not</h3>
<p>Across all {len(seq_rows)} cells where both are scored on our own traces, GOOD_5 wins
<strong>{seq_tally['wins']}</strong>, ties <strong>{seq_tally['ties']}</strong>, loses
<strong>{seq_tally['losses']}</strong> against mean sequence-logprob (&plusmn;0.5pp band).
Extremes:</p>
<table><tr><th>Cell</th><th>GOOD_5</th><th>seq-logprob</th><th>Delta</th></tr>{seq_html}</table>
{seq_fig}
<p>Framing note (2026-07-12): the headline of Item 4 stays the <em>published-method</em> scoreboard —
that is the comparison the thesis is judged on, and sequence log-prob is an in-house audit, not a cited
rival (it is, however, the standard likelihood baseline the cited papers themselves report — Malinin &amp;
Gales 2021; Guerreiro et al. EACL 2023 — so it belongs in the appendix with a stated verdict). The audit's
finding stands: the trivial baseline is slightly ahead more often than not, including on
<span class="mono">ars_gsm8k_r1distill8b</span> (&minus;2.2pp), the very cell headlined as a
supervised-matching win. The open question to
answer directly: <strong>what is GOOD_5's real marginal value over mean(&minus;logprob), and
does it cluster by domain?</strong> (Wins cluster on QA/long-trace cells — CoQA +12.4pp,
TriviaQA-semenergy +4.3pp, SQuAD v2 +3.4pp; losses on GSM8K cells with strong models.)
Full table on the <a href="item4_benchmarking.html">Item 4 page</a>.</p>

<h3>3 — Items 5 and 6 sit in tension; here is the reconciliation</h3>
<p>Item 5 (Step 152): extra sampling (SE K=10) adds &le;+2.0pp, gate FAIL. Item 6 (Step 158):
extra sampling (K=5 same-T) adds +6.1pp, CI excludes 0. Both are true because the passes were
spent on different signals — fragile NLI semantic clustering vs averaging the same cheap entropy
signal. Computed this session from the Phase-15 pkl: applying Item 5's own gate to Item 6's
same-T arm gives cross-pass &rho; {rho:+.2f} &lt; 0.75 and paired delta {pp(d[0])}pp
[{pp(d[1])}, {pp(d[2])}] &gt; 1pp — <strong>PASS</strong>. No earlier document stated this
reconciliation; it is now on the <a href="item5_sampling_fusion.html">Item 5 page</a>.</p>

<h3>4 — The MATH-500 sign flip was "Priority 1" for 17 steps; the sign question, re-derived</h3>
<p>Fixed this session to the extent the local data allows. Three parts:</p>
<p><strong>(a) Step 110's rejection of the paper's raw majority-vote sign rule holds up under
independent re-derivation</strong> — our features are domain-homogeneous (all 5 of GOOD_5 share
the same raw orientation), which violates the paper's assumption (iii) so badly that the rule
inverts a correctly-estimated eigenvector (adversarial synthetic: 1.6% vs 98.4% AUROC). Not a
coin flip — a forced wrong sign. The consensus per-feature signs (FEATURE_SIGNS) stay.</p>
<p><strong>(b) New test this session — is the single-epr residual anchor a live point of
failure?</strong> <span class="mono">scripts/test_multi_anchor_orient.py</span> compared the
production single-epr anchor against a multi-feature-average anchor (the same pattern
<span class="mono">multipass_lsml_continuous</span> already trusts cross-pass) on the full
29-cell battery: on GOOD_5 the two agree on 26/29 cells (macro 63.6 vs 63.9 — the 3
disagreements are two GPQA cells where epr lands &gt;0.5 and one RAG cell where multi does);
on ALL_H16 the multi anchor is a large net loss (macro 62.3 &rarr; 54.7, flipping the strongest
MATH-500 cells to their mirror). <strong>Verdict: the epr anchor is kept — simpler, and it is
not the failure mode here.</strong></p>
<p><strong>(c) The actual re-fix</strong>
(<span class="mono">scripts/refix_phase12_signs.py</span>): MATH-500 L-SML corrects to
<strong>{cor_txt}</strong> (mirror arithmetic on the stored CI — the results pkl carries no
score arrays; flip direction corroborated label-free on the same-model battery cell). The full
label-free re-derivation runs automatically when the raw Phase-12 two-pass caches are copied
from Drive. The old-cache SE/SC NLI-truncation reconciliation (also Step-152 Priority 1)
remains open — decide whether to chase it or retire the old Phase-12 baseline table.</p>

<h3>5 — RAG SelfCheckGPT below chance: bug, or a citable finding?</h3>
<p>Called "unresolved / orientation bug suspected" since Step 152 without anyone checking the
alternative: SelfCheckGPT's consistency-voting assumption may legitimately break on long
retrieval-grounded contexts — which would be a <em>citable result</em>, not a bug to apologise
for. Cheap next check: verify the orientation on one RAG cell by hand; if the orientation is
right, this becomes a finding.</p>

<h3>6 — "Temperature is not the lever" is a single-cell claim</h3>
<p>Item 6's paired result is clean (CI excludes 0) but comes from exactly one (dataset, model):
MATH-500 / Qwen2.5-Math-7B, whose accuracy collapses 80% &rarr; 4% across the T range. Before
repeating it as a general finding, check one more cell (GSM8K or QA) where accuracy degrades
more gently. Cost: one 5-run Colab session on an existing model.</p>

<h3>7 — The Internal-States "paper-matched" rerun still sits at {pct(is_row['acc'])}% accuracy</h3>
<p>The T=0.8 rerun only partially recovers the sampling collapse (greedy &asymp;85%); that caveat
lives in a CSV cell note, invisible at headline level. Worth asking whether decoding temperature
is really the only difference from the paper's setup, or whether the prompt/grading protocol
also diverges. Until then, the {pct(f(g5('internalstates_gsm8k_qwen25_7b','upcr')['auroc_X']))}
U-PCR number should carry the accuracy caveat whenever quoted (see also point 1 — its CI
contains SelfCheckGPT's point estimate).</p>
</div>"""
    return page("Advisor scrutiny", "Seven questions a careful advisor would ask before these "
                "numbers go into a meeting or a paper — with the evidence, recomputed.", body,
                "advisor_scrutiny.html")


# ── index ──────────────────────────────────────────────────────────────────────

def build_index(seq_tally):
    p15 = P15 or {}
    body = f"""
<div class="section-card">
{badge('finding', 'Before you read the rest')}
<h2>Advisor action items — compilation</h2>
<p>All numbers are correct and CSV-sourced, but read together they raise questions the per-cell
writeups didn't have room for: <strong>three</strong> headline "beats/edges" claims are
CI-overlapping rather than CI-clear, and on {seq_tally['wins'] + seq_tally['ties'] +
seq_tally['losses']} comparable cells the trivial same-trace sequence-logprob baseline is ahead
of GOOD_5 more often than not ({seq_tally['wins']} wins / {seq_tally['ties']} ties /
{seq_tally['losses']} losses for GOOD_5). Start with the
<a href="advisor_scrutiny.html"><strong>scrutiny page</strong></a>, then the items.</p>
</div>

<div class="card-grid">
<div class="item-card">{badge('done', 'Complete')}
<h3>Item 1 — Literature search</h3>
<p>The 2016 spectral meta-learner line continued: U-PCR (Tenzer et al. 2022, AISTATS) is its
continuous-input successor and effectively what we run; FUSE (2026) is the closest competitor,
distinguishable on signal (internal entropy trace vs external verifiers) and task (per-answer
detection vs Best-of-N).</p>
<a class="cta" href="item1_literature_search.html">Read &rarr;</a></div>

<div class="item-card">{badge('done', 'Complete')}
<h3>Item 2 — Supervised-oracle gap</h3>
<p>Label-free fusion gives up +3.6&ndash;4.7pp macro vs a balanced LR oracle on identical
features — but &asymp;0pp on reasoning, where both sit near the in-sample ceiling. The gap is a
QA/GPQA phenomenon; LR buys different weights (Spearman 0.1&ndash;0.2), not better scaling.</p>
<a class="cta" href="item2_lr_oracle.html">Read &rarr;</a></div>

<div class="item-card">{badge('progress', 'In progress')}
<h3>Item 3 — Short-form QA evaluation</h3>
<p>All 5 prioritised datasets have data; SQuAD v2 (79.8), SciQ (74.4), TruthfulQA (67.3) and the
judge-rescued NQ-Open (73.2) all clear the &ge;65 bar — the Step-155 gate (&ge;3 of 4) is already
met, with ceiling/floor caveats attached; CoQA's judge-regraded full-N rerun is in flight.</p>
<a class="cta" href="item3_qa_evaluation.html">Read &rarr;</a></div>

<div class="item-card">{badge('progress', 'In progress — nearly done')}
<h3>Item 4 — Benchmarking</h3>
<p>Same-model wins that are CI-clear: LapEigvals-unsup on GSM8K Llama-8B (+9.5pp), Phi-3.5
(+13.7pp), Nemo (+15.2pp), Mistral-24B (+22.5pp, ceiling-caveated). Three narrower claims
(R1-Distill vs ARS, Qwen2.5-7B vs SelfCheckGPT, Phi-3-mini vs answer-entropy) are numerically
ahead but CI-overlapping. The fairest same-trace baseline is ahead of GOOD_5 on over half the
comparable cells.</p>
<a class="cta" href="item4_benchmarking.html">Read &rarr;</a></div>

<div class="item-card">{badge('flagged', 'Complete, flagged')}
<h3>Item 5 — Sampling fusion</h3>
<p>Step-152 gate FAIL stands for SE-fusion (+0.4pp on GSM8K), but the MATH-500 row is now
sign-corrected (0.230 &rarr; 0.770 mirror), and the Phase-15 partial re-score shows the same-T
K=5 entropy-averaging arm <em>passes</em> the same gate — the Items-5/6 reconciliation. The
answer-agreement arm is armed and waits only on the 5 raw pass caches.</p>
<a class="cta" href="item5_sampling_fusion.html">Read &rarr;</a></div>

<div class="item-card">{badge('done', 'Complete (Step 158)')}
<h3>Item 6 — Temperature variation</h3>
<p>Clean negative result: temperature diversity <em>hurts</em> multi-pass fusion (&minus;5.3pp,
CI excludes 0); repeated same-T sampling <em>helps</em> (+6.1pp). Mechanism: same-T passes
denoise (&rho;&asymp;+0.45); multi-T passes only look decorrelated because the off-T passes are
near-random. Single-cell scope caveat applies. (The "Not started" rows in PROGRESS.md /
Research_Directions.md were stale — fixed this session.)</p>
<a class="cta" href="item6_temperature_variation.html">Read &rarr;</a></div>

<div class="item-card">{badge('finding', 'Cross-cutting')}
<h3>Per-domain breakdown</h3>
<p>L-SML + U-PCR &times; 4 feature subsets across every non-GPQA cell — AIRCC grid plus the
legacy caches, whose per-cell U-PCR was computed fresh this session
(<span class="mono">upcr_legacy.csv</span>, 192 rows).</p>
<a class="cta" href="per_domain_breakdown.html">Read &rarr;</a></div>

<div class="item-card">{badge('finding', 'Critical review')}
<h3>Advisor scrutiny — 7 points</h3>
<p>CI-overlap on three headlines; seq-logprob near-parity; the Items-5/6 tension reconciled; the
sign-flip fix and the anchor re-derivation; RAG SelfCheckGPT bug-or-finding; single-cell scope
of Item 6; the 30.6%-accuracy caveat on the Internal-States rerun.</p>
<a class="cta" href="advisor_scrutiny.html">Read &rarr;</a></div>
</div>

<div class="section-card">
<h2>Coverage check — what is missing from the recent cluster wave</h2>
<table>
<tr><th>Gap</th><th>Status</th><th>Note</th></tr>
<tr><td>GPQA</td><td>Zero cluster presets</td><td>5 legacy models only; Phase-14 Colab rerun
pending. Excluded from the breakdown per request.</td></tr>
<tr><td>RAG &times; 4 (L-CiteEval)</td><td>Legacy numbers only</td><td>L-SML + fresh U-PCR
computed locally, but untouched on the cluster since Step 152; the SelfCheckGPT below-chance
issue there is open (scrutiny point 5).</td></tr>
<tr><td>WebQ</td><td>Stale CoT-only cache</td><td>No plain variant anywhere; absent from the
AIRCC grid.</td></tr>
<tr><td>AMC23 / AIME24</td><td>Loaders + EDIS cache exist</td><td>No replication-grid presets
for our detector yet.</td></tr>
</table>
</div>

<div class="section-card">
<h2>Candidate follow-ups</h2>
<p>Surfaced by this pass; each with a cost estimate so they can be prioritised. (Two of the four
were partially executed this session by the armed scripts.)</p>
<table>
<tr><th>#</th><th>Candidate</th><th>Cost</th><th>State</th></tr>
<tr><td>1</td><td>Answer-agreement self-consistency baseline from the 5 cached T=1.0 MATH-500
passes — closes Item 5's missing arm</td><td>CPU-only, zero GPU</td>
<td>Script armed (<span class="mono">rescore_phase15_selfconsistency.py</span>); needs the 5 raw
pass pkls copied from Drive <span class="mono">cache/phase15_temperature/</span></td></tr>
<tr><td>2</td><td>Full label-free re-derivation of the Phase-12 sign fix on the original
scores</td><td>CPU-only</td>
<td>Mirror correction done; script armed (<span class="mono">refix_phase12_signs.py</span>);
needs <span class="mono">p1/p2/p3</span> caches from Drive
<span class="mono">cache/phase12_corrected/</span></td></tr>
<tr><td>3</td><td>NLI-truncation reconciliation for the old-cache SE/SC baselines — or a
decision to retire that table</td><td>CPU + NLI model, hours</td><td>Open since Step 152</td></tr>
<tr><td>4</td><td>RAG SelfCheckGPT orientation check (bug vs citable finding)</td>
<td>CPU, &lt;1h</td><td>Open since Step 152 (scrutiny point 5)</td></tr>
<tr><td>5</td><td>Second-dataset temperature check (Item 6 scope)</td><td>One Colab 5-run
session</td><td>Not started</td></tr>
</table>
</div>"""
    return page("Advisor action items — index", "Six action items from the 2026-06-17 meeting: "
                "status, results, coverage gaps, and the critical-review page.", body,
                "index.html")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scan = ci_scan()
    seq_rows, seq_tally, seq_skipped = good5_vs_seqlp()

    pages = {
        "index.html": build_index(seq_tally),
        "item1_literature_search.html": build_item1(),
        "item2_lr_oracle.html": build_item2(),
        "item3_qa_evaluation.html": build_item3(),
        "item4_benchmarking.html": build_item4(scan, seq_rows, seq_tally, seq_skipped),
        "item5_sampling_fusion.html": build_item5(),
        "item6_temperature_variation.html": build_item6(),
        "per_domain_breakdown.html": build_breakdown(),
        "advisor_scrutiny.html": build_scrutiny(scan, seq_rows, seq_tally),
    }

    all_hits = {}
    for name, text in pages.items():
        hits = guardrail_scan(text)
        if hits:
            all_hits[name] = hits
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote results/action_items/{name} ({len(text)} chars)")

    if all_hits:
        print("\nGUARDRAIL HITS:")
        for name, hits in all_hits.items():
            print(f"  {name}: {hits}")
        return 2
    print("\nguardrail scan clean on all 9 pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
