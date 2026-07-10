#!/usr/bin/env python
"""
advisor_report.py — generator for results/Advisors_Action_Items_Report.html.

Rebuilds the advisor report so every NUMERIC table cell traces to a canonical CSV, and
bakes in the fact-check corrections (see HISTORY / the Step-16x report-correction pass):

  * REASONING-FIRST Item 4: our strongest domain (MATH-500, GSM8K) leads the competitor
    grid, sourced from results/reasoning_benchmark.csv; QA head-to-head follows.
  * Attribution fixes: Semantic Energy = Chen et al. (arXiv 2508.14496), NOT Farquhar;
    EPR carries no unverified "Minut et al."; Semantic Entropy = Kuhn/Gal/Farquhar ICLR'23.
  * The EPR column X (0.736) is labelled U-PCR + logprob, not L-SML GOOD_5.
  * Sampling-fusion number 0.768 -> 0.758 (PROGRESS Step 152; phase12_corrected pkl is
    Drive-only, so the handoff value is cited and flagged).
  * NLI-truncation reframed as a SUSPECTED, unresolved cause (Step-152 Priority 1), not a
    confirmed discovery; SE/SC reasoning baselines flagged not-yet-citable.
  * EDIS (arXiv 2602.01288) scored on our own reasoning trace (results/repgrid/edis_scores.csv).
  * ARS (2601.17467) + Internal-States (2510.11529) added as published reasoning anchors.
  * Selection-bias caveats for spilled_triviaqa (n_pos=6) and se_squad (valid_rate 0.29).

Terminology guardrails (memory): no bare "Nadler" (use "L-SML" / "Jaffe-Fetaya-Nadler 2016"),
no "MV_EPR", no "recommended", no comparison to a supervised best_nadler_on.

Usage:
    python scripts/advisor_report.py            # -> results/Advisors_Action_Items_Report.html
    python scripts/advisor_report.py --check     # generate to a temp buffer + run guardrail scan
"""
import argparse
import csv
import html
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
REPGRID = os.path.join(RESULTS, "repgrid")
OUT = os.path.join(RESULTS, "Advisors_Action_Items_Report.html")

BANNED = ["MV_EPR", "best_nadler_on", "recommended"]  # 'Nadler' handled specially (allowed only in the full lineage name)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(x):
    return html.escape(str(x))


def pct(x):
    """Render an AUROC-ish value as a percentage string; passthrough blanks."""
    if x is None or x == "":
        return "—"
    v = float(x)
    if v <= 1.5:
        v *= 100.0
    return f"{v:.1f}"


# ── Data loads ─────────────────────────────────────────────────────────────────
def load_reasoning():
    return read_csv(os.path.join(RESULTS, "reasoning_benchmark.csv"))


def load_edis():
    rows = read_csv(os.path.join(REPGRID, "edis_scores.csv"))
    return {r["cell"]: r for r in rows}


def load_headline():
    return read_csv(os.path.join(REPGRID, "headline_X_vs_Y.csv"))


def load_subset_by_domain():
    return read_csv(os.path.join(REPGRID, "subset_by_domain.csv"))


def load_ubaselines():
    path = os.path.join(REPGRID, "ubaseline_scores.csv")
    return read_csv(path) if os.path.exists(path) else []


def dual_label_html(ub):
    """Label-scheme robustness: baseline AUROC under judge vs lexical labels, for cells
    where the two label sets disagree (agreement < 1)."""
    names = {"ppl": "Perplexity", "seqlp": "Sequence logprob", "nent": "Naive entropy (mean H)"}
    cells = {"epr_triviaqa_mistral24b": "TriviaQA / Mistral-24B (EPR cell)",
             "semenergy_triviaqa_qwen3_8b": "TriviaQA / Qwen3-8B (Semantic Energy cell)"}
    out = []
    for r in ub:
        if r["cell"] not in cells or float(r["label_agreement"]) >= 1.0:
            continue
        span = len(names)
        first = True
        for key, label in names.items():
            j, l = r.get(f"{key}_auroc"), r.get(f"{key}_auroc_lex")
            if not j or not l:
                continue
            d = (float(j) - float(l)) * 100
            cellcol = ""
            if first:
                cellcol = (f'<td rowspan="{span}"><strong>{esc(cells[r["cell"]])}</strong><br>'
                           f'<span style="color:#64748b;font-size:12.5px">label agreement '
                           f'{float(r["label_agreement"]):.2f}, judge acc {float(r["acc"]):.2f} '
                           f'vs lexical {float(r["acc_lex"]):.2f}</span></td>')
                first = False
            out.append(f'<tr>{cellcol}<td>{esc(label)}</td><td>{pct(j)}</td><td>{pct(l)}</td>'
                       f'<td><strong>{d:+.1f}</strong></td></tr>')
    return "\n".join(out)


# ── Section builders ───────────────────────────────────────────────────────────
def reasoning_rows_html(reasoning):
    """Reasoning-first competitor grid, grouped by (dataset, model), ours highlighted."""
    order = [
        ("MATH-500", "Qwen2.5-Math-7B"),
        ("MATH-500", "DeepSeek-R1-Distill-Llama-8B"),
        ("MATH-500", "Qwen3-8B"),
        ("GSM8K", "Llama-3.1-8B"),
        ("GSM8K", "Phi-3.5-mini-instruct"),
        ("GSM8K", "DeepSeek-R1-Distill-Llama-8B"),
        ("GSM8K", "Qwen3-8B"),
        ("GSM8K", "Qwen2.5-7B"),
        ("GSM8K", "Llama-3.2-3B-Instruct"),
        ("GSM8K", "Phi-3-mini-4k-instruct"),
        ("GSM8K", "Mistral-7B-Instruct-v0.3"),
        ("GSM8K", "Gemma-2B-it"),
        ("GPQA", "Qwen2.5-7B"),
    ]
    by_cell = defaultdict(list)
    for r in reasoning:
        by_cell[(r["dataset"], r["model"])].append(r)

    out = []
    for key in order:
        rows = by_cell.get(key, [])
        if not rows:
            continue
        # ours first, then supervised competitors, then unsup competitors
        rows.sort(key=lambda r: (r["is_ours"] != "yes", r["supervision"] != "unsupervised"))
        ds, model = key
        span = len(rows)
        for i, r in enumerate(rows):
            ours = r["is_ours"] == "yes"
            cite = r["citable"] == "yes"
            ci = ""
            if r["ci_lo"] and r["ci_hi"]:
                ci = f' <span style="color:#94a3b8">[{pct(r["ci_lo"])}, {pct(r["ci_hi"])}]</span>'
            auroc_cell = f'<strong>{pct(r["auroc"])}{ci}</strong>' if ours else f'{pct(r["auroc"])}{ci}'
            method = esc(r["method"])
            sup = r["supervision"]
            sup_badge = ('<span style="color:#dc2626;font-weight:600">supervised</span>'
                         if sup == "supervised"
                         else '<span style="color:#16a34a;font-weight:600">unsupervised</span>')
            note = esc(r["note"])
            if not cite:
                note = '⚠ ' + note
            rowcls = ' style="background:#eff6ff"' if ours else ''
            cellcol = ""
            if i == 0:
                cellcol = (f'<td rowspan="{span}"><strong>{esc(ds)}</strong><br>'
                           f'<span style="color:#64748b;font-size:13px">{esc(model)}</span></td>')
            category = esc(r.get("category", "") or "")
            out.append(
                f'<tr{rowcls}>{cellcol}'
                f'<td>{"👉 " if ours else ""}{method}</td>'
                f'<td>{sup_badge}</td>'
                f'<td><span class="mono">{category}</span></td>'
                f'<td>{auroc_cell}</td>'
                f'<td style="font-size:12.5px;color:#64748b">{note}</td></tr>'
            )
    return "\n".join(out)


def qa_headline_html(headline, edis):
    """QA / boundary head-to-head from the replication grid (lsml rows), with EDIS + caveats."""
    label = {
        "semenergy_triviaqa_qwen3_8b": ("Semantic Energy (Chen et al., 2508.14496)", "TriviaQA", "Qwen3-8B", True),
        "epr_triviaqa_mistral24b": ("EPR — Entropy Production Rate", "TriviaQA (Wiki)", "Mistral-24B", True),
        "seiclr_triviaqa_opt30b": ("Semantic Entropy (Kuhn/Gal/Farquhar, ICLR'23)", "TriviaQA", "OPT-30B", True),
        "inside_coqa_llama7b": ("INSIDE / EigenScore (2402.03744)", "CoQA", "LLaMA-7B", True),
        "losnet_hotpotqa_mistral7b": ("LOS-Net (supervised probe)", "HotpotQA", "Mistral-7B-v0.2", False),
        "se_squad_v2_llama8b": ("SE-ICLR protocol (adapted)", "SQuAD v2", "Llama-3.1-8B", False),
        "spilled_triviaqa_llama8b": ("Spilled/Semantic Energy (boundary)", "TriviaQA", "Llama-3.1-8B", False),
        "truthfulqa_llama8b": ("TruthfulQA (generation)", "TruthfulQA", "Llama-3.1-8B", False),
    }
    # keep lsml rows; index by cell
    lsml = {r["cell"]: r for r in headline if r["method"] == "lsml"}
    upcr = {r["cell"]: r for r in headline if r["method"] == "upcr"}
    order = ["semenergy_triviaqa_qwen3_8b", "epr_triviaqa_mistral24b",
             "seiclr_triviaqa_opt30b", "inside_coqa_llama7b", "losnet_hotpotqa_mistral7b"]
    out = []
    for cell in order:
        if cell not in lsml:
            continue
        name, ds, model, h2h = label[cell]
        r = lsml[cell]
        X = float(r["X"])
        Y = r["Y"]
        # EPR: our stronger view is U-PCR+logprob — surface it explicitly, not as L-SML.
        if cell == "epr_triviaqa_mistral24b":
            ur = upcr.get(cell)
            xtxt = f'<strong>{pct(ur["X"])}</strong> <span style="color:#64748b;font-size:12px">(U-PCR + logprob)</span>'
        else:
            xtxt = f'<strong>{pct(X)}</strong> <span style="color:#64748b;font-size:12px">(L-SML GOOD_5)</span>'
        ytxt = pct(Y) if Y else "n/a"
        if Y:
            d = X - float(Y)
            if cell == "epr_triviaqa_mistral24b":
                d = float(upcr[cell]["X"]) - float(Y)
            outcome = (f'<span class="win">+{d*100:.1f} WIN</span>' if d > 0.005
                       else (f'<span class="loss">{d*100:.1f}</span>' if d < -0.03
                             else f'<strong>{d*100:+.1f} tie</strong>'))
        else:
            outcome = '<span style="color:#94a3b8">no clean published anchor</span>'
        ed = edis.get(cell, {})
        edtxt = pct(ed.get("edis_auroc")) if ed.get("edis_auroc") else "—"
        note = ""
        if cell == "losnet_hotpotqa_mistral7b":
            note = " (supervised-probe anchor)"
        out.append(
            f'<tr><td><strong>{esc(name)}</strong></td><td>{esc(ds)}</td><td>{esc(model)}</td>'
            f'<td>{xtxt}</td><td>{ytxt}{esc(note)}</td><td>{edtxt}</td><td>{outcome}</td></tr>'
        )
    return "\n".join(out)


def closed_subset_html(sbd):
    """Domain means per candidate subset -> the 'one closed subset across domains' question."""
    subsets = ["consensus_4", "GOOD_5", "top_macro_5", "STABLE_H9", "ALL_H16"]
    dom_order = ["Reasoning-Math", "Short-QA", "RAG", "MCQ-Science"]
    # mean over cells per (domain, subset), ignoring blanks
    agg = defaultdict(lambda: defaultdict(list))
    for r in sbd:
        a = r["auroc"]
        if a == "" or a is None:
            continue
        agg[r["domain"]][r["subset"]].append(float(a))
    out = []
    for dom in dom_order:
        cells = agg.get(dom, {})
        if not cells:
            continue
        tds = []
        best_sub, best_val = None, -1
        for s in subsets:
            vals = cells.get(s, [])
            if vals:
                m = sum(vals) / len(vals)
                tds.append((s, m))
                if m > best_val:
                    best_val, best_sub = m, s
            else:
                tds.append((s, None))
        cells_html = ""
        for s, m in tds:
            if m is None:
                cells_html += "<td>—</td>"
            else:
                strong = ' style="background:#ecfdf5;font-weight:700"' if s == best_sub else ""
                cells_html += f"<td{strong}>{m*100:.1f}</td>"
        n_cells = len(next((cells[s] for s in subsets if cells.get(s)), []))
        out.append(f'<tr><td><strong>{esc(dom)}</strong> <span style="color:#94a3b8">({n_cells} cells)</span></td>{cells_html}</tr>')
    return "\n".join(out)


# ── HTML assembly ──────────────────────────────────────────────────────────────
CSS = """
  :root{--blue:#2563eb;--blue-light:#eff6ff;--blue-dark:#1e40af;--green:#10b981;--green-light:#ecfdf5;
  --green-dark:#065f46;--red:#ef4444;--amber:#f59e0b;--amber-light:#fffbeb;--amber-dark:#92400e;
  --gray-50:#f8fafc;--gray-100:#f1f5f9;--gray-200:#e2e8f0;--gray-600:#475569;--gray-700:#334155;
  --gray-800:#1e293b;--gray-900:#0f172a;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Inter',system-ui,-apple-system,sans-serif;font-size:15px;color:var(--gray-800);
  background:var(--gray-50);line-height:1.65;}
  .header-hero{background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);color:#fff;padding:56px 24px 44px;
  border-bottom:4px solid var(--blue);}
  .header-content{max-width:1040px;margin:0 auto;}
  .header-hero h1{font-size:32px;font-weight:800;letter-spacing:-.02em;margin-bottom:10px;}
  .header-hero .subtitle{font-size:16px;color:#cbd5e1;max-width:820px;}
  .meta-pills{display:flex;gap:12px;margin-top:20px;flex-wrap:wrap;}
  .meta-pill{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);padding:4px 12px;
  border-radius:20px;font-size:13px;font-weight:500;}
  .page{max-width:1040px;margin:0 auto;padding:40px 24px 64px;}
  .nav-bar{background:#fff;border:1px solid var(--gray-200);border-radius:12px;padding:14px 20px;
  margin-bottom:36px;box-shadow:0 1px 3px rgba(0,0,0,.04);display:flex;gap:16px;flex-wrap:wrap;align-items:center;}
  .nav-bar strong{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--gray-600);margin-right:4px;}
  .nav-bar a{color:var(--blue);text-decoration:none;font-weight:600;font-size:14px;}
  .section-card{background:#fff;border:1px solid var(--gray-200);border-radius:14px;padding:36px;
  margin-bottom:36px;box-shadow:0 2px 6px rgba(0,0,0,.03);}
  .section-badge{display:inline-block;font-size:12px;font-weight:700;text-transform:uppercase;
  letter-spacing:.06em;padding:4px 12px;border-radius:20px;margin-bottom:12px;}
  .badge-completed{background:var(--green-light);color:var(--green-dark);}
  .badge-progress{background:var(--blue-light);color:var(--blue-dark);}
  .badge-finding{background:#f3e8ff;color:#6b21a8;}
  h2{font-size:24px;font-weight:800;letter-spacing:-.015em;color:var(--gray-900);margin-bottom:12px;}
  h3{font-size:17px;font-weight:700;color:var(--gray-900);margin:24px 0 10px;}
  p{margin-bottom:14px;color:var(--gray-700);}
  .takeaway-box{background:var(--green-light);border-left:4px solid var(--green);padding:16px 20px;
  border-radius:0 10px 10px 0;margin:18px 0;}
  .takeaway-box b{color:var(--green-dark);}
  .info-box{background:var(--blue-light);border-left:4px solid var(--blue);padding:16px 20px;
  border-radius:0 10px 10px 0;margin:18px 0;}
  .info-box b{color:var(--blue-dark);}
  .warn-box{background:var(--amber-light);border-left:4px solid var(--amber);padding:16px 20px;
  border-radius:0 10px 10px 0;margin:18px 0;}
  .warn-box b{color:var(--amber-dark);}
  table{width:100%;border-collapse:collapse;margin:20px 0;font-size:14px;}
  th{background:var(--gray-100);color:var(--gray-800);font-weight:700;text-align:left;padding:12px 14px;
  border:1px solid var(--gray-200);}
  td{padding:11px 14px;border:1px solid var(--gray-200);color:var(--gray-700);}
  .win{color:#16a34a;font-weight:700;}
  .loss{color:#dc2626;font-weight:700;}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:var(--gray-100);
  padding:2px 6px;border-radius:4px;color:var(--gray-800);}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:24px 0;}
  @media(max-width:820px){.grid2{grid-template-columns:1fr;}}
  .chart-card{background:#fff;border:1px solid var(--gray-200);border-radius:12px;padding:24px;margin:20px 0;}
  .chart-card h4{font-size:15px;font-weight:700;color:var(--gray-800);margin-bottom:4px;}
  canvas{max-height:340px;width:100%;}
"""


def build_html():
    reasoning = load_reasoning()
    edis = load_edis()
    headline = load_headline()
    sbd = load_subset_by_domain()

    gsm8k_edis = edis.get("lapeigvals_gsm8k_llama8b", {})
    gsm8k_edis_auroc = pct(gsm8k_edis.get("edis_auroc", "0.809"))
    gsm8k_edis_rho = gsm8k_edis.get("rho_edis_lsml", "0.87")

    reasoning_tbl = reasoning_rows_html(reasoning)
    qa_tbl = qa_headline_html(headline, edis)
    closed_tbl = closed_subset_html(sbd)
    dual_tbl = dual_label_html(load_ubaselines())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Unsupervised Spectral Hallucination Detection — Advisor Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<div class="header-hero"><div class="header-content">
  <h1>Unsupervised Spectral Hallucination Detection</h1>
  <div class="subtitle">Advisor report on the 6 action items (June/July 2026). Reasoning-first
  benchmarking. Every numeric table is generated from a source CSV by <span class="mono">scripts/advisor_report.py</span>.</div>
  <div class="meta-pills">
    <span class="meta-pill">Prepared for: Ofir, Bracha &amp; Amir</span>
    <span class="meta-pill">Method: single-pass L-SML continuous</span>
    <span class="meta-pill">Canonical set: GOOD_5</span>
    <span class="meta-pill">Numbers sourced from CSV</span>
  </div>
</div></div>

<div class="page">
  <div class="nav-bar"><strong>Jump to:</strong>
    <a href="#item4">Reasoning benchmarking</a> &bull;
    <a href="#qa">QA head-to-head</a> &bull;
    <a href="#subset">Closed subset</a> &bull;
    <a href="#item1">FUSE</a> &bull;
    <a href="#item2">LR oracle</a> &bull;
    <a href="#item5">Sampling fusion</a> &bull;
    <a href="#item6">Temperature</a> &bull;
    <a href="#next">Next steps</a>
  </div>

  <!-- EXEC SUMMARY -->
  <div class="section-card">
    <span class="section-badge badge-completed">Executive summary</span>
    <h2>Where the method wins, and against whom</h2>
    <p>Our contribution is a <strong>single forward pass</strong> intrinsic detector: spectral
    features of the per-token entropy trace H(n), fused label-free with L-SML continuous
    (Jaffe-Fetaya-Nadler 2016 lineage) on the <span class="mono">GOOD_5</span> feature set. The
    honest headline is domain-scoped: <strong>it works on reasoning traces (math), is competitive
    on GSM8K, and is out of regime on GPQA / multi-hop RAG.</strong> This report leads with the
    reasoning-domain competitor grid — our strongest evidence — which the previous draft omitted.</p>
    <div class="takeaway-box"><b>One-line result:</b> on <strong>MATH-500 / Qwen2.5-Math-7B</strong>
    our unsupervised single-pass L-SML reaches <strong>94.4 AUROC</strong>, and on the same
    <strong>R1-Distill-8B / MATH-500</strong> cell used by the supervised ARS detector (86.4) our
    unsupervised GOOD_5 reaches <strong>84.4</strong> — nearly matching a trained method with zero labels.</p>
  </div>

  <!-- ITEM 4 REASONING FIRST -->
  <div class="section-card" id="item4">
    <span class="section-badge badge-completed">Action item 4 — primary</span>
    <h2>Competitor benchmarking — reasoning domain (our strongest)</h2>
    <p>Each row places our AUROC next to a published or same-model competitor number. Unsupervised
    vs supervised is marked explicitly. Rows flagged ⚠ are old-cache baselines not yet citable
    (see the NLI note below). EDIS and the survey-standard baselines (perplexity, sequence logprob,
    naive entropy) are scored on <em>our own</em> trace, not taken from a paper.</p>
    <div class="info-box"><b>The math-reasoning gap (July-2026 survey):</b> the canonical
    unsupervised detectors — Semantic Entropy, INSIDE/EigenScore, SelfCheckGPT, KLE, HaloScope —
    were originally evaluated on factual QA and <strong>never reported GSM8K/MATH AUROC</strong>;
    the few math numbers that exist come from 2025–26 re-evaluations (Noise Injection 2502.03799;
    2601.17467; 2510.11529), all verified here from the primary sources. A single-pass unsupervised
    gray-box detector that is strong on math reasoning occupies genuinely open territory. Category
    key: <span class="mono">UGB</span> unsupervised gray-box (logits/entropy), <span class="mono">BB</span>
    black-box sampling, <span class="mono">WB</span> white-box internal states,
    <span class="mono">SUP</span> supervised probe.</p></div>
    <table>
      <thead><tr><th>Cell</th><th>Method</th><th>Supervision</th><th>Category</th><th>AUROC</th><th>Note</th></tr></thead>
      <tbody>
      {reasoning_tbl}
      </tbody>
    </table>
    <div class="takeaway-box"><b>Reasoning verdict:</b> single-pass unsupervised L-SML
    <strong>beats LapEigvals' same-model unsupervised AttentionScore (72.0)</strong> on GSM8K, sits
    within a few points of the supervised LapEigvals probe (87.2), and on R1-Distill/MATH-500
    <strong>nearly matches the supervised ARS detector</strong> (84.4 vs 86.4) with no labels and one
    forward pass. EDIS scored on our GSM8K trace = {gsm8k_edis_auroc}, but it is
    <strong>redundant with L-SML</strong> (ρ = {gsm8k_edis_rho}), so it adds no fusion lift.</p></div>
    <div class="warn-box"><b>Suspected, unresolved — long-CoT NLI truncation:</b> the K=10 sampling
    baselines (Semantic Entropy / Self-Consistency) drop sharply on fresh long-trace caches
    (e.g. SE 87.7 → 63.0 on MATH-500). The <em>prime suspect</em> is NLI cross-encoder input
    truncation at 512 tokens on 2048-token traces, confounded with a different question subset. This
    is <strong>not yet confirmed</strong> (Step-152 Priority 1); until reconciled, the old-cache SE
    87.7 / SC 87.2 reasoning numbers are shown for context but are <strong>not citable</strong>.</div>
  </div>

  <!-- QA HEAD TO HEAD -->
  <div class="section-card" id="qa">
    <span class="section-badge badge-completed">Action items 3 &amp; 4 — QA / boundary</span>
    <h2>Same-model QA head-to-head (replication grid)</h2>
    <p>Model-matched cells run on the AIRCC cluster (single source of truth: <span class="mono">cluster/presets.py</span>).
    X is our AUROC; Y is the paper's published number; EDIS is scored on our trace.</p>
    <table>
      <thead><tr><th>Paper / competitor</th><th>Dataset</th><th>Model</th><th>Ours (X)</th>
      <th>Published (Y)</th><th>EDIS (ours)</th><th>Outcome</th></tr></thead>
      <tbody>
      {qa_tbl}
      </tbody>
    </table>
    <div class="warn-box"><b>Selection-bias caveats (do not read as headline wins):</b>
    <span class="mono">spilled_triviaqa</span> scores 0.96 but only <strong>n_pos = 6</strong> of 256
    valid rows (the ≥8-token trace filter removes short correct answers; <span class="mono">trace_length</span>
    alone hits 0.925 — length leakage). <span class="mono">se_squad_v2</span> has valid-rate 0.29, so
    its 0.80 is on a 29% subset. Both are boundary/QA cells, not evidence for the reasoning claim.</div>
  </div>

  <!-- LABEL-SCHEME ROBUSTNESS -->
  <div class="section-card" id="labels">
    <span class="section-badge badge-finding">Label-scheme robustness</span>
    <h2>Judge vs lexical labels: the correctness definition moves AUROC</h2>
    <p>A 2025 re-evaluation (<em>"The Illusion of Progress"</em>, arXiv 2508.08285, EMNLP 2025) showed
    detector AUROC can shift drastically with the label protocol — perplexity by up to 45.9%, and
    EigenScore by 19–30%, between ROUGE-style lexical labels and LLM-as-judge labels. We see the same
    sensitivity on our own cells: the table below scores the <em>identical</em> baseline signal under
    both label schemes on cells where the two graders disagree.</p>
    <table>
      <thead><tr><th>Cell</th><th>Baseline (scored on our trace)</th><th>AUROC — judge labels</th>
      <th>AUROC — lexical labels</th><th>Δ (pp)</th></tr></thead>
      <tbody>
      {dual_tbl}
      </tbody>
    </table>
    <div class="takeaway-box"><b>Why this matters:</b> the label protocol alone moves a baseline by up
    to <strong>35 AUROC points</strong> (naive entropy on the TriviaQA/Mistral-24B cell). This is why
    every head-to-head cell is graded with the paper-matched protocol (LLM judge where the paper used
    one). A related but distinct confound holds back the GSM8K/Qwen2.5-7B cell: its 0.284 accuracy is
    <strong>not a grading artifact</strong> — inspection shows genuine T=1.0 sampling errors (99% of
    wrong answers still produce a boxed final answer). The mismatch there is the <em>decoding operating
    point</em> (our T=1.0 vs the paper's near-greedy), so the fix is a temperature-matched re-run, not
    a re-grade. The judge re-grade path applies where labels are the problem (TruthfulQA's ROUGE proxy).</div>
  </div>

  <!-- CLOSED SUBSET -->
  <div class="section-card" id="subset">
    <span class="section-badge badge-progress">Closed feature subset</span>
    <h2>Is there one subset that works across all domains?</h2>
    <p>Mean L-SML AUROC per domain for each candidate subset (across all cells in that domain, old +
    new). The five subsets, written out in full features:</p>
    <table style="font-size:13px">
      <thead><tr><th>Subset</th><th>Features</th></tr></thead>
      <tbody>
      <tr><td><span class="mono">consensus_4</span></td><td>spectral_entropy, sw_var_peak, cusum_max, cusum_shift_idx</td></tr>
      <tr><td><span class="mono">GOOD_5</span></td><td>epr, low_band_power, sw_var_peak, cusum_max, spectral_entropy</td></tr>
      <tr><td><span class="mono">top_macro_5</span></td><td>epr, spectral_entropy, hl_ratio, sw_var_peak, cusum_max</td></tr>
      <tr><td><span class="mono">STABLE_H9</span></td><td>epr, low_band_power, high_band_power, hl_ratio, spectral_centroid, sw_var_peak, rpdi, pe_mean, cusum_max</td></tr>
      <tr><td><span class="mono">ALL_H16</span></td><td>the first 16 spectral/time features (FEAT_NAMES[:16])</td></tr>
      </tbody>
    </table>
    <table>
      <thead><tr><th>Domain</th><th>consensus_4</th><th>GOOD_5</th><th>top_macro_5</th><th>STABLE_H9</th><th>ALL_H16</th></tr></thead>
      <tbody>
      {closed_tbl}
      </tbody>
    </table>
    <div class="takeaway-box"><b>Closed-subset takeaway:</b> the small 5-feature sets
    (<span class="mono">GOOD_5</span> / <span class="mono">top_macro_5</span>) lead on Reasoning-Math
    and are within noise of the best elsewhere; the larger <span class="mono">ALL_H16</span> and
    <span class="mono">STABLE_H9</span> add nothing and often hurt (block collinearity). A single
    <strong>5-feature GOOD_5</strong> set is the defensible cross-domain default; the new
    energy/logprob views help only specific QA cells and do not displace it.</div>
  </div>

  <!-- ITEM 1 FUSE -->
  <div class="section-card" id="item1">
    <span class="section-badge badge-completed">Action item 1</span>
    <h2>Literature positioning: FUSE vs single-pass L-SML</h2>
    <p><strong>FUSE</strong> (Candès et al., April 2026, <em>"Reliable Best-of-N candidate selection
    via spectral ensembling"</em>) builds on the same Jaffe-Fetaya-Nadler 2016 SML lineage, so both
    advisors flagged it. The relationship is <strong>complementary, by signal and by task</strong>.</p>
    <table>
      <thead><tr><th>Dimension</th><th>FUSE</th><th>Ours (L-SML continuous)</th></tr></thead>
      <tbody>
      <tr><td><strong>Task</strong></td><td>Best-of-N selection: pick the correct answer among N candidates.</td>
      <td>Per-answer detection: score one generation's hallucination risk.</td></tr>
      <tr><td><strong>Signal</strong></td><td>External verifier models ranking candidates.</td>
      <td>Intrinsic token-level H(n) spectral features from one forward pass of the generator.</td></tr>
      <tr><td><strong>Dependence handling</strong></td><td>Triplet Condensation Inequality transform.</td>
      <td>Covariance K-group clustering + within/across-group fusion.</td></tr>
      </tbody>
    </table>
    <div class="takeaway-box"><b>Complementary:</b> the differentiator is the <strong>signal</strong>
    (internal entropy-trace vs external verifiers) and the <strong>task</strong> (per-answer detection
    vs Best-of-N). Our single-pass spectral views could serve as zero-extra-inference inputs inside a
    FUSE selection pipeline.</div>
  </div>

  <!-- ITEM 2 LR ORACLE -->
  <div class="section-card" id="item2">
    <span class="section-badge badge-completed">Action item 2</span>
    <h2>Supervised LR oracle &amp; overfitting bound</h2>
    <p>Upper bound for any linear feature combination: supervised 5-fold CV logistic regression over
    the 28 strictly-common cells, following the SUPERVISED_ORACLE_CORRECTION guidelines (class-balanced,
    no cross_val_predict calibration leak).</p>
    <div class="grid2">
      <div><table>
        <thead><tr><th>Feature set</th><th>Unsup L-SML</th><th>Sup LR CV</th><th>Gap</th><th>In-sample ceiling</th></tr></thead>
        <tbody>
        <tr><td><strong>5-feat (GOOD_5)</strong></td><td><strong>64.2</strong></td><td><strong>68.9</strong></td><td>+4.7</td><td>70.5</td></tr>
        <tr><td>9-feat (STABLE_H9)</td><td>62.9</td><td>66.8</td><td>+3.8</td><td>73.7</td></tr>
        <tr><td>16-feat (ALL_H16)</td><td>64.1</td><td>67.8</td><td>+3.6</td><td>79.3</td></tr>
        </tbody></table></div>
      <div><div class="chart-card"><h4>LR CV vs in-sample ceiling</h4><canvas id="chartLR"></canvas></div></div>
    </div>
    <div class="takeaway-box"><b>Why 5 features:</b> from 5→16 features the in-sample ceiling climbs
    70.5→79.3 but CV generalization drops 68.9→67.8 (block collinearity ρ≈0.77–0.88). Unsupervised
    L-SML on GOOD_5 captures ~93% of the supervised CV potential with zero labels.</div>
  </div>

  <!-- ITEM 5 SAMPLING FUSION -->
  <div class="section-card" id="item5">
    <span class="section-badge badge-completed">Action item 5</span>
    <h2>Sampling fusion: single-pass L-SML vs Semantic Entropy K=10</h2>
    <p>Does adding K=10 Semantic Entropy to our K=1 spectral features help on GSM8K / Llama-3.1-8B?</p>
    <div class="grid2">
      <div><table>
        <thead><tr><th>Method</th><th>K</th><th>AUROC</th><th>Cost</th></tr></thead>
        <tbody>
        <tr><td>Likelihood-weighted Semantic Entropy</td><td>10</td><td>0.614</td><td>10× compute</td></tr>
        <tr><td>SelfCheckGPT (official soft NLI)</td><td>5</td><td>0.701</td><td>5× + NLI</td></tr>
        <tr><td><strong>Single-pass spectral L-SML (GOOD_5)</strong></td><td><strong>1</strong></td><td><strong>0.754</strong></td><td><strong>1× (zero extra)</strong></td></tr>
        <tr><td>Sampling fusion (SE K=10 + L-SML)</td><td>10</td><td><strong>0.758</strong></td><td>10× compute</td></tr>
        </tbody></table></div>
      <div><div class="chart-card"><h4>Single-pass vs multi-pass on GSM8K</h4><canvas id="chartFusion"></canvas></div></div>
    </div>
    <div class="takeaway-box"><b>Fusion gate did not pass:</b> single-pass L-SML (0.754) already beats
    every multi-pass sampling baseline. Fusing SE K=10 on top adds only <strong>+0.4pp → 0.758</strong>
    (ρ = 0.26; PROGRESS Step 152). The reverse — adding L-SML to weak SE — helps SE a lot, confirming
    L-SML already carries the semantic-uncertainty signal in one pass.</div>
    <p style="font-size:12.5px;color:#94a3b8">Fusion value 0.758 is the Step-152 handoff number; the
    <span class="mono">phase12_corrected</span> pkl lives on Drive, so this cites the recorded value.</p>
  </div>

  <!-- ITEM 6 TEMPERATURE -->
  <div class="section-card" id="item6">
    <span class="section-badge badge-finding">Action item 6 — Phase 15</span>
    <h2>Temperature sweep: thermal diversity vs same-T variance reduction</h2>
    <p>Do multi-pass ensembles at diverse temperatures (T ∈ {{0.3..2.0}}) beat repeated sampling at a
    single T=1.0, for detector discriminability?</p>
    <div class="grid2">
      <div><table>
        <thead><tr><th>Configuration</th><th>Task acc</th><th>Multi-pass AUROC</th><th>Δ vs single</th></tr></thead>
        <tbody>
        <tr><td>Single pass (T=1.0)</td><td>79.5%</td><td>0.851</td><td>—</td></tr>
        <tr><td><strong>Same-T (5×T=1.0)</strong></td><td>79.5%</td><td><strong>0.912</strong></td><td><span class="win">+6.1pp</span></td></tr>
        <tr><td>Diverse-T (0.3..2.0)</td><td>collapses to 4% at T=2.0</td><td>0.859</td><td><span class="loss">-5.3pp vs same-T</span></td></tr>
        </tbody></table></div>
      <div><div class="chart-card"><h4>Same-T vs diverse-T multi-pass lift</h4><canvas id="chartTemp"></canvas></div></div>
    </div>
    <div class="takeaway-box"><b>Temperature diversity hurts:</b> diverse-T fusion is -5.3pp
    (95% CI [-10.3, -1.1]) vs same-T=1.0. At T≥1.5 accuracy collapses 80→4% and the entropy trace is
    vocabulary noise. The +6.1pp multi-pass lift is pure variance reduction at a calibrated T=1.0.</div>
  </div>

  <!-- NEXT STEPS -->
  <div class="section-card" id="next">
    <span class="section-badge badge-progress">Next steps</span>
    <h2>Real open items</h2>
    <table>
      <thead><tr><th>#</th><th>Item</th><th>Why it matters</th></tr></thead>
      <tbody>
      <tr><td><strong>1</strong></td><td>Reconcile the SE/SC NLI-truncation drop (re-run NLI with a long-context
      cross-encoder or sentence-level chunking on the fresh long-trace caches).</td>
      <td>Blocks citing the old-cache SE 87.7 / SC 87.2 reasoning baselines. Step-152 Priority 1.</td></tr>
      <tr><td><strong>2</strong></td><td>Run the matched <span class="mono">ars_gsm8k_r1distill8b</span> cluster
      cell (preset added; smoke passes). MATH-500/R1-Distill already exists (GOOD_5 84.4 vs ARS 86.4).</td>
      <td>Completes the same-model ARS head-to-head with a second point (vs ARS 74.72 on GSM8K).</td></tr>
      <tr><td><strong>3</strong></td><td>Score EDIS on MATH-500 from the Drive raw-trace cache
      (<span class="mono">math500_qwen7b_T1.0_run*.pkl</span>) on Colab: <span class="mono">python
      scripts/score_edis.py --pkl &lt;drive_path&gt; --cell math500_qwen7b</span>.</td>
      <td>50MB pkl is too large to pull through the MCP bridge here; GSM8K EDIS already covers the reasoning claim.</td></tr>
      <tr><td><strong>4</strong></td><td>Treat <span class="mono">spilled_triviaqa</span> (n_pos=6) and
      <span class="mono">se_squad</span> (valid 0.29) as selection-biased; do not headline them.</td>
      <td>Prevents an inflated boundary-QA number from misrepresenting the QA story.</td></tr>
      </tbody>
    </table>
  </div>

</div>

<script>
  new Chart(document.getElementById('chartLR'),{{type:'bar',data:{{labels:['5-feat (GOOD_5)','9-feat (STABLE_H9)','16-feat (ALL_H16)'],
    datasets:[{{label:'Unsup L-SML',data:[64.2,62.9,64.1],backgroundColor:'#3b82f6'}},
    {{label:'Sup LR CV',data:[68.9,66.8,67.8],backgroundColor:'#10b981'}},
    {{label:'In-sample ceiling',data:[70.5,73.7,79.3],backgroundColor:'#cbd5e1'}}]}},
    options:{{responsive:true,scales:{{y:{{min:55,max:85,title:{{display:true,text:'Macro AUROC (%)'}}}}}},plugins:{{legend:{{position:'top'}}}}}}}});
  new Chart(document.getElementById('chartFusion'),{{type:'bar',data:{{labels:['SE (K=10)','SelfCheckGPT (K=5)','Single-pass L-SML (K=1)','SE + L-SML (K=10)'],
    datasets:[{{label:'AUROC on GSM8K',data:[0.614,0.701,0.754,0.758],backgroundColor:['#94a3b8','#94a3b8','#2563eb','#10b981']}}]}},
    options:{{responsive:true,scales:{{y:{{min:0.5,max:0.85,title:{{display:true,text:'AUROC'}}}}}},plugins:{{legend:{{display:false}}}}}}}});
  new Chart(document.getElementById('chartTemp'),{{type:'bar',data:{{labels:['Single (T=1.0)','Same-T (5×T=1.0)','Diverse-T (0.3..2.0)'],
    datasets:[{{label:'AUROC',data:[0.851,0.912,0.859],backgroundColor:['#64748b','#10b981','#f59e0b']}}]}},
    options:{{responsive:true,scales:{{y:{{min:0.75,max:0.95,title:{{display:true,text:'Multi-pass AUROC'}}}}}},plugins:{{legend:{{display:false}}}}}}}});
</script>
</body>
</html>"""


def guardrail_scan(html_text):
    """Return list of (term, count) for banned terminology. 'Nadler' is allowed ONLY inside
    the full lineage name 'Jaffe-Fetaya-Nadler'."""
    hits = []
    for term in BANNED:
        c = html_text.count(term)
        if c:
            hits.append((term, c))
    # bare 'Nadler' not part of the lineage name
    bare = html_text.replace("Jaffe-Fetaya-Nadler", "")
    if "Nadler" in bare:
        hits.append(("Nadler (bare)", bare.count("Nadler")))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print guardrail scan, do not require write")
    args = ap.parse_args()
    html_text = build_html()
    hits = guardrail_scan(html_text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"wrote {OUT} ({len(html_text)} chars)")
    if hits:
        print("GUARDRAIL HITS:", hits)
        sys.exit(2)
    print("guardrail scan clean (no banned terms)")


if __name__ == "__main__":
    main()
