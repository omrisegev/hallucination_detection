#!/usr/bin/env python
"""
report_figs.py — CSV-driven inline-SVG figures for the advisor-facing HTML reports.

Every number is read from the canonical CSVs at build time, so regenerating a report
after new cells are scored refreshes the figures automatically (same convention as
scripts/advisor_report.py). No JS, no CDN — pages stay self-contained.

Data sources:
    results/repgrid/scores_lsml_upcr.csv     ours (L-SML GOOD_5 primary), published_Y
    results/reasoning_benchmark.csv          published same-model anchors + ceilings
    results/repgrid/ubaseline_scores.csv     standard baselines on our own traces
    results/repgrid/published_baselines.csv  published per-cell baseline tables
                                             (e.g. LOS-Net arXiv 2503.14043 Table 1)

Regen-chain rule (see PROGRESS / report_regen_chain): a NEW GSM8K model needs one
entry in GSM8K_SPEC below (exact reasoning_benchmark.csv model string + cell id),
exactly like advisor_report.py's order list — otherwise the row silently drops.
"""
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
REPGRID = os.path.join(RESULTS, "repgrid")


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _lin(v, d0, d1, r0, r1):
    return r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def _load_all():
    lu = _read(os.path.join(REPGRID, "scores_lsml_upcr.csv"))
    rb = _read(os.path.join(RESULTS, "reasoning_benchmark.csv"))
    ub = _read(os.path.join(REPGRID, "ubaseline_scores.csv"))
    pb = _read(os.path.join(REPGRID, "published_baselines.csv"))
    return lu, rb, ub, pb


# ── Gate-status policy (desk rule, formalized 2026-07-12) ─────────────────────
# Two tiers, applied desk-wide (reasoning + QA):
#   1. BAND VIOLATION (accuracy outside [0.20, 0.85]) = quality flag. The cell is
#      SCORED, shown everywhere with a CEILING/FLOOR flag, and excluded from the
#      headline win tally. A flag marks a noisy estimate of the right quantity.
#   2. LABEL-VALIDITY FAILURE = documented REJECT, never scored: truncation-label
#      leakage (cap-pinned negatives) or a single-class label set make AUROC a
#      clean estimate of the WRONG quantity — no caveat rescues that.
ACC_BAND = (0.20, 0.85)
REJECT_REGISTRY = {
    "ars_gsm8k_qwen3_8b": "truncation leakage: 15/29 negatives cap-pinned at 8192 (+ ceiling acc 0.942)",
    "ars_gsm8k_qwen3_8b_reject": "same cell, archived dir name",
    "ars_math500_qwen3_8b": "truncation leakage: 23/50 negatives cap-pinned at 16384 (acc 0.900; p95 trace = cap — unbounded reasoning)",
    "ars_math500_qwen3_8b_reject": "same cell, archived dir name",
    "noise_gsm8k_gemma2b": "single-class labels: acc 0.000 (0/30 pilot) — AUROC undefined",
}


def gate_flag(acc):
    """'' if in-band, else 'CEILING'/'FLOOR'. Feed it any cell's accuracy."""
    a = _f(acc)
    if a is None:
        return ""
    if a < ACC_BAND[0]:
        return "FLOOR"
    if a > ACC_BAND[1]:
        return "CEILING"
    return ""


# Scoped CSS: light-theme values matched to advisor_report.py's palette.
FIG_CSS = """
  .rf-fig{background:#fff;border:1px solid var(--gray-200);border-radius:12px;
    padding:18px 20px 10px;margin:20px 0;}
  .rf-fig .rf-cap{font-size:14px;font-weight:700;color:var(--gray-900);margin-bottom:2px;}
  .rf-fig .rf-sub{font-size:12.5px;color:var(--gray-600);margin-bottom:10px;}
  .rf-chart{width:100%;height:auto;display:block;}
  .rf-legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0 0 10px;align-items:center;}
  .rf-legend .rf-li{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--gray-700);}
  .rf-fnote{font-size:11.5px;color:var(--gray-600);margin:8px 0 4px;line-height:1.5;}
  .rf-grid{stroke:var(--gray-200);stroke-width:1;}
  .rf-zero{stroke:#94a3b8;stroke-width:1.5;}
  .rf-tick{fill:var(--gray-600);font-size:11px;}
  .rf-axname{fill:var(--gray-600);font-size:11.5px;}
  .rf-rowlbl{fill:var(--gray-800);font-size:12px;font-weight:600;}
  .rf-rowsub{fill:var(--gray-600);font-size:10px;}
  .rf-vlbl{fill:var(--gray-700);font-size:11px;font-weight:600;}
  .rf-barlbl{fill:var(--gray-700);font-size:11px;}
  .rf-dlbl{fill:var(--gray-600);font-size:10.5px;}
  .rf-dlbl-ours{fill:var(--blue);font-weight:700;}
  .rf-region{fill:var(--gray-600);font-size:11px;font-style:italic;}
  .rf-albl{fill:var(--gray-700);font-size:11px;font-weight:600;}
  .rf-leader{stroke:#94a3b8;stroke-width:1;}
  .rf-ci{stroke:var(--blue);stroke-width:2;stroke-linecap:round;}
  .rf-ours{fill:var(--blue);stroke:#fff;stroke-width:2;}
  .rf-anchor{fill:var(--green);stroke:#fff;stroke-width:2;}
  .rf-anchor-open{fill:none;stroke:var(--green);stroke-width:2;}
  .rf-sup{fill:none;stroke:#475569;stroke-width:2;}
  .rf-seqlp{stroke:var(--gray-700);stroke-width:1.5;opacity:.85;}
  .rf-bar-pos{fill:var(--blue);}
  .rf-bar-neg{fill:var(--red);}
  .rf-bar-ctx{opacity:.42;}
  .rf-bar-ours{fill:var(--blue);}
  .rf-bar-gray{fill:#94a3b8;}
  .rf-bar-gray-sup{fill:#475569;}
  .rf-bar-gray-flag{fill:#cbd5e1;}
  .rf-pt-qa{fill:var(--green);stroke:#fff;stroke-width:2;}
  .rf-line{stroke:var(--blue);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round;}
  .rf-line-acc{stroke:var(--green);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round;}
  .rf-dot-acc{fill:var(--green);stroke:#fff;stroke-width:2;}
  .rf-panes{display:flex;gap:24px;flex-wrap:wrap;}
  .rf-pane{flex:1 1 340px;min-width:300px;}
  .rf-pane .rf-ptitle{font-size:13px;font-weight:700;color:var(--gray-900);margin-bottom:6px;}
  .rf-diag{stroke:#94a3b8;stroke-width:1.5;}
  .rf-wash{fill:rgba(37,99,235,.06);}
"""


def _svg(w, h):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" class="rf-chart" '
            f'preserveAspectRatio="xMidYMid meet">')


def _fig(cap, sub, legend, svg, fnote):
    leg = f'<div class="rf-legend">{legend}</div>' if legend else ""
    fn = f'<p class="rf-fnote">{fnote}</p>' if fnote else ""
    return (f'<div class="rf-fig"><div class="rf-cap">{cap}</div>'
            f'<div class="rf-sub">{sub}</div>{leg}{svg}{fn}</div>')


# ── Figure 1: GSM8K forest plot across models ─────────────────────────────────
# (display label, RB model string, cell id, anchor method substr, sup method substr, note)
GSM8K_SPEC = [
    ("Llama-3.1-8B",        "Llama-3.1-8B",                    "lapeigvals_gsm8k_llama8b",       "LapEigvals AttentionScore", "LapEigvals probe", ""),
    ("Phi-3.5-mini",        "Phi-3.5-mini-instruct",           "lapeigvals_gsm8k_phi35",         "LapEigvals AttentionScore", "LapEigvals probe", ""),
    ("Mistral-Small-24B",   "Mistral-Small-24B-Instruct-2501", "lapeigvals_gsm8k_mistral24b",    "LapEigvals AttentionScore", "LapEigvals probe", ""),
    ("Mistral-7B-v0.3",     "Mistral-7B-Instruct-v0.3",        "noise_gsm8k_mistral7b",          "Noise Injection",           None,               "¶"),
    ("Mistral-Nemo-12B",    "Mistral-Nemo-Instruct-2407",      "lapeigvals_gsm8k_nemo",          "LapEigvals AttentionScore", "LapEigvals probe", ""),
    ("R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Llama-8B",    "ars_gsm8k_r1distill8b",          "Semantic Entropy",          "ARS (CCS)",        ""),
    ("Llama-3.2-3B",        "Llama-3.2-3B-Instruct",           "lapeigvals_gsm8k_llama3b",       "LapEigvals AttentionScore", "LapEigvals probe", "§"),
    ("Phi-3-mini",          "Phi-3-mini-4k-instruct",          "noise_gsm8k_phi3mini",           "Noise Injection",           None,               "¶"),
    ("Qwen2.5-7B (T=0.8)",  "Qwen2.5-7B",                      "internalstates_gsm8k_qwen25_7b", "SelfCheckGPT",              "Internal-States + Reasoning-Consistency", "‡"),
]


def _rb_val(rb, dataset, model, method_substr):
    for r in rb:
        if (r["dataset"] == dataset and r["model"] == model
                and method_substr in r["method"] and r["is_ours"] != "yes"):
            return _f(r["auroc"])
    return None


def _lu_g5(lu, cell, method="lsml"):
    for r in lu:
        if r["cell"] == cell and r["subset"] == "GOOD_5" and r["method"] == method:
            return r
    return None


def _ub_row(ub, cell):
    for r in ub:
        if r["cell"] == cell:
            return r
    return None


def fig_gsm8k_forest():
    lu, rb, ub, _ = _load_all()
    rows = []
    for (lbl, model, cell, anch_m, sup_m, note) in GSM8K_SPEC:
        if cell in REJECT_REGISTRY:
            continue  # label-validity REJECT — documented, never plotted as a score
        g = _lu_g5(lu, cell)
        if g is None:
            continue  # cell not scored yet — appears automatically once it is
        v, lo, hi = _f(g["auroc_X"]) * 100, _f(g["lo"]) * 100, _f(g["hi"]) * 100
        av = _rb_val(rb, "GSM8K", model, anch_m)
        sv = _rb_val(rb, "GSM8K", model, sup_m) if sup_m else None
        u = _ub_row(ub, cell)
        sq = _f(u["seqlp_auroc"]) * 100 if u and _f(u.get("seqlp_auroc")) else None
        flag = gate_flag(g.get("acc"))
        tag = (" †" if flag == "CEILING" else " ▿" if flag == "FLOOR" else "")
        rows.append((lbl + tag + (" " + note if note else ""), v, lo, hi, av, anch_m, sv, sup_m, sq))
    rows.sort(key=lambda r: -r[1])
    if not rows:
        return ""
    X0, X1, D0, D1, ROW, TOP = 210, 850, 48, 96, 38, 14
    n = len(rows)
    H = TOP + n * ROW + 34
    x = lambda v: _lin(v, D0, D1, X0, X1)
    s = [_svg(900, H)]
    for gv in (50, 60, 70, 80, 90):
        s.append(f'<line x1="{x(gv):.1f}" y1="{TOP}" x2="{x(gv):.1f}" y2="{TOP+n*ROW}" class="rf-grid"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{TOP+n*ROW+16}" class="rf-tick" text-anchor="middle">{gv}</text>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-2}" class="rf-axname" text-anchor="middle">AUROC (%) · GSM8K, same model per row</text>')
    for i, (lbl, v, lo, hi, av, anch_m, sv, sup_m, sq) in enumerate(rows):
        cy = TOP + i * ROW + ROW / 2
        s.append(f'<text x="200" y="{cy+4:.1f}" class="rf-rowlbl" text-anchor="end">{lbl}</text>')
        s.append(f'<line x1="{x(lo):.1f}" y1="{cy:.1f}" x2="{x(hi):.1f}" y2="{cy:.1f}" class="rf-ci"/>')
        if sq is not None:
            s.append(f'<line x1="{x(sq):.1f}" y1="{cy-8:.1f}" x2="{x(sq):.1f}" y2="{cy+8:.1f}" class="rf-seqlp"><title>Sequence log-prob on our traces: {sq:.1f}</title></line>')
        if sv is not None:
            tx = x(sv)
            s.append(f'<path d="M {tx:.1f} {cy-6:.1f} L {tx+6:.1f} {cy+5:.1f} L {tx-6:.1f} {cy+5:.1f} Z" class="rf-sup"><title>{sup_m}: {sv} (supervised, same model)</title></path>')
        if av is not None:
            dx = x(av)
            s.append(f'<path d="M {dx:.1f} {cy-6.5:.1f} L {dx+6.5:.1f} {cy:.1f} L {dx:.1f} {cy+6.5:.1f} L {dx-6.5:.1f} {cy:.1f} Z" class="rf-anchor"><title>{anch_m}: {av} (published, unsupervised, same model)</title></path>')
        s.append(f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="5.5" class="rf-ours"><title>L-SML GOOD_5 (ours): {v:.1f} [{lo:.1f}, {hi:.1f}]</title></circle>')
        if i == 0:
            s.append(f'<text x="{x(v):.1f}" y="{cy-10:.1f}" class="rf-dlbl rf-dlbl-ours" text-anchor="middle">{v:.1f}</text>')
            if av is not None:
                s.append(f'<text x="{x(av):.1f}" y="{cy+20:.1f}" class="rf-dlbl" text-anchor="middle">{av:.1f}</text>')
            if sv is not None:
                s.append(f'<text x="{x(sv):.1f}" y="{cy-10:.1f}" class="rf-dlbl" text-anchor="middle">{sv:.1f}</text>')
    s.append("</svg>")
    legend = (
        '<span class="rf-li"><svg width="30" height="14"><line x1="2" y1="7" x2="28" y2="7" class="rf-ci"/><circle cx="15" cy="7" r="5.5" class="rf-ours"/></svg> L-SML GOOD_5 — ours, K=1, unsupervised (95% CI)</span>'
        '<span class="rf-li"><svg width="16" height="14"><path d="M 8 1 L 14 7 L 8 13 L 2 7 Z" class="rf-anchor"/></svg> published unsupervised anchor, same model</span>'
        '<span class="rf-li"><svg width="16" height="14"><path d="M 8 2 L 14 12 L 2 12 Z" class="rf-sup"/></svg> published supervised probe (ceiling)</span>'
        '<span class="rf-li"><svg width="10" height="14"><line x1="5" y1="1" x2="5" y2="13" class="rf-seqlp"/></svg> sequence log-prob on our own traces</span>')
    fnote = ("Gate flags derived from each cell's accuracy at build time: † CEILING (acc &gt; 0.85), ▿ FLOOR (acc &lt; 0.20) "
             "— flagged cells are scored and shown but excluded from the headline win tally; label-validity REJECTs "
             "(truncation leakage, single-class) are never plotted. "
             "¶ Noise Injection anchor is K=10 with question-level majority-vote labels. "
             "§ the NI anchor for Llama-3.2-3B (82.7) is protocol-mismatched and omitted; the plotted anchor is LapEigvals AttentionScore. "
             "‡ run at the Internal-States paper's T=0.8; the U-PCR GOOD_5 variant on this cell is the number quoted in the CSV notes. "
             "Ours = L-SML GOOD_5 from scores_lsml_upcr.csv (fixed subset, primary method) on every row.")
    return _fig("GSM8K, single-pass unsupervised detection across models",
                "Our L-SML on the fixed GOOD_5 subset vs published same-model numbers. Rows sorted by our AUROC; hover any mark for value and source.",
                legend, "".join(s), fnote)


# ── Figure 2: same-model head-to-head deltas ──────────────────────────────────
SUP_Y = {"ARS (CCS)": "supervised", "Internal-States+RC": "supervised",
         "TSV (arXiv 2503.01917)": "semi-supervised", "LOS-Net": "supervised probe",
         "INSIDE": ""}
NOTE_Y = {"Noise Injection": "K=10, question-level labels",
          "Semantic Entropy (SE-ICLR'23)": "protocol mismatch ¶"}
# Manifest-Y override: the lapeigvals_gsm8k_llama8b manifest still carries the
# supervised cross-model probe (92.5); the fair same-model unsup anchor is 72.0
# (Step 165 correction, reasoning_benchmark.csv).
OVERRIDE_Y = {"lapeigvals_gsm8k_llama8b": (0.720, "LapEigvals AttentionScore", True)}
DS_PRETTY = {"gsm8k": "GSM8K", "math500": "MATH-500", "trivia_qa": "TriviaQA",
             "trivia_qa_wiki": "TriviaQA", "trivia_qa_rougel": "TriviaQA",
             "hotpotqa": "HotpotQA", "truthfulqa": "TruthfulQA", "coqa": "CoQA",
             "nq_open": "NQ-Open", "squad_v2": "SQuAD v2", "sciq": "SciQ"}


def fig_same_model_deltas():
    lu, _, _, _ = _load_all()
    bars = []
    for r in lu:
        if r["subset"] != "GOOD_5" or r["method"] != "lsml":
            continue
        cell = r["cell"]
        if cell.endswith(("_partial", "_pilot")) or cell in REJECT_REGISTRY:
            continue
        y_val, y_m, fair_override = OVERRIDE_Y.get(cell, (None, None, None))
        if y_val is None:
            if r.get("head_to_head") != "SAME-MODEL" or not _f(r.get("published_Y")):
                continue
            y_val, y_m = _f(r["published_Y"]), r.get("Y_method", "")
        x_val = _f(r["auroc_X"])
        d = (x_val - y_val) * 100
        model = r["model"].split("/")[-1].replace("-Instruct", "").replace("-instruct", "")
        ds = DS_PRETTY.get(r["dataset"], r["dataset"])
        sup = SUP_Y.get(y_m, "")
        note = NOTE_Y.get(y_m, "")
        flag = gate_flag(r.get("acc"))
        sub = (f"vs {y_m}" + (f" — {sup}" if sup else "") + (f" ({note})" if note else "")
               + (f" [{flag}]" if flag else ""))
        # out-of-band cells are shown but never counted as clean wins: fade them
        fair = (fair_override if fair_override is not None
                else (not sup and "mismatch" not in note)) and not flag
        bars.append((d, f"{model} · {ds}", sub, fair))
    bars.sort(key=lambda b: -b[0])
    if not bars:
        return ""
    ROW, TOP = 34, 12
    n = len(bars)
    X0, X1 = 330, 880
    dmax = max(24.0, max(abs(b[0]) for b in bars) + 2)
    D0, D1 = -dmax, dmax
    H = TOP + n * ROW + 34
    x = lambda v: _lin(v, D0, D1, X0, X1)
    zx = x(0)
    s = [_svg(900, H)]
    for gv in (-20, -10, 10, 20):
        s.append(f'<line x1="{x(gv):.1f}" y1="{TOP}" x2="{x(gv):.1f}" y2="{TOP+n*ROW}" class="rf-grid"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{TOP+n*ROW+16}" class="rf-tick" text-anchor="middle">{gv:+d}</text>')
    s.append(f'<line x1="{zx:.1f}" y1="{TOP}" x2="{zx:.1f}" y2="{TOP+n*ROW}" class="rf-zero"/>')
    s.append(f'<text x="{zx:.1f}" y="{TOP+n*ROW+16}" class="rf-tick" text-anchor="middle">0</text>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-2}" class="rf-axname" text-anchor="middle">Δ AUROC (pp) — our L-SML GOOD_5 minus the published same-model number</text>')
    for i, (d, lbl, sub, fair) in enumerate(bars):
        cy = TOP + i * ROW + ROW / 2
        s.append(f'<text x="320" y="{cy-2:.1f}" class="rf-rowlbl" text-anchor="end">{lbl}</text>')
        s.append(f'<text x="320" y="{cy+11:.1f}" class="rf-rowsub" text-anchor="end">{sub}</text>')
        bx = x(d)
        h = 16
        op = "" if fair else " rf-bar-ctx"
        if abs(d) < 0.05:
            s.append(f'<rect x="{zx-1:.1f}" y="{cy-h/2:.1f}" width="2" height="{h}" class="rf-bar-pos{op}"/>')
            s.append(f'<text x="{zx+8:.1f}" y="{cy+4:.1f}" class="rf-vlbl" text-anchor="start">+0.0</text>')
            continue
        cls = "rf-bar-pos" if d > 0 else "rf-bar-neg"
        rr = 4
        if d > 0:
            path = (f'M {zx:.1f} {cy-h/2:.1f} H {bx-rr:.1f} Q {bx:.1f} {cy-h/2:.1f} {bx:.1f} {cy-h/2+rr:.1f} '
                    f'V {cy+h/2-rr:.1f} Q {bx:.1f} {cy+h/2:.1f} {bx-rr:.1f} {cy+h/2:.1f} H {zx:.1f} Z')
        else:
            path = (f'M {zx:.1f} {cy-h/2:.1f} H {bx+rr:.1f} Q {bx:.1f} {cy-h/2:.1f} {bx:.1f} {cy-h/2+rr:.1f} '
                    f'V {cy+h/2-rr:.1f} Q {bx:.1f} {cy+h/2:.1f} {bx+rr:.1f} {cy+h/2:.1f} H {zx:.1f} Z')
        s.append(f'<path d="{path}" class="{cls}{op}"><title>{lbl} {sub}: {d:+.1f}pp</title></path>')
        tx, ta = (bx + 8, "start") if d > 0 else (bx - 8, "end")
        s.append(f'<text x="{tx:.1f}" y="{cy+4:.1f}" class="rf-vlbl" text-anchor="{ta}">{d:+.1f}</text>')
    s.append("</svg>")
    legend = (
        '<span class="rf-li"><svg width="16" height="12"><rect x="1" y="2" width="14" height="8" rx="2" class="rf-bar-pos"/></svg> we are ahead</span>'
        '<span class="rf-li"><svg width="16" height="12"><rect x="1" y="2" width="14" height="8" rx="2" class="rf-bar-neg"/></svg> published number ahead</span>'
        '<span class="rf-li"><svg width="16" height="12"><rect x="1" y="2" width="14" height="8" rx="2" class="rf-bar-pos rf-bar-ctx"/></svg> faded = supervised / protocol-mismatched anchor, or out-of-band cell [CEILING/FLOOR] — shown, never counted as a clean win</span>')
    fnote = ("Ours = L-SML GOOD_5 (fixed subset, primary method) throughout — per-cell best variants are ablations, not headlines. "
             "Gate policy: band violations (acc outside [0.20, 0.85]) are scored + flagged; label-validity REJECTs "
             "(truncation leakage, single-class) are excluded entirely. "
             "¶ SE-ICLR evaluates per-question K=10 semantic sampling — different units. "
             "Source: scores_lsml_upcr.csv (published_Y verified from each paper's table).")
    return _fig("Every same-model head-to-head, one picture",
                "Positive = our unsupervised single-pass score is ahead of the published same-model number.",
                legend, "".join(s), fnote)


# ── Figure 3: one-cell landscape (all methods on the identical cell) ──────────
def _bar_panel(rows, d0=35, d1=100, W=880):
    """rows: (label, value_pct, kind, note); kinds: ours/ctx/ctx-sup/ctx-flag."""
    ROW, TOP = 38, 8
    n = len(rows)
    X0, X1 = 10, W - 20
    H = TOP + n * ROW + 30
    x = lambda v: _lin(v, d0, d1, X0, X1)
    zx = x(50)
    s = [_svg(W, H)]
    for gv in (40, 60, 70, 80, 90, 100):
        s.append(f'<line x1="{x(gv):.1f}" y1="{TOP}" x2="{x(gv):.1f}" y2="{TOP+n*ROW}" class="rf-grid"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{TOP+n*ROW+15}" class="rf-tick" text-anchor="middle">{gv}</text>')
    s.append(f'<line x1="{zx:.1f}" y1="{TOP}" x2="{zx:.1f}" y2="{TOP+n*ROW}" class="rf-zero"/>')
    s.append(f'<text x="{zx:.1f}" y="{TOP+n*ROW+15}" class="rf-tick" text-anchor="middle">50 = chance</text>')
    for i, (lbl, v, kind, note) in enumerate(rows):
        y0 = TOP + i * ROW
        by, h = y0 + 16, 14
        bx = x(v)
        cls = {"ours": "rf-bar-ours", "ctx": "rf-bar-gray",
               "ctx-sup": "rf-bar-gray-sup", "ctx-flag": "rf-bar-gray-flag"}[kind]
        rr = 4
        if v >= 50:
            path = (f'M {zx:.1f} {by:.1f} H {bx-rr:.1f} Q {bx:.1f} {by:.1f} {bx:.1f} {by+rr:.1f} '
                    f'V {by+h-rr:.1f} Q {bx:.1f} {by+h:.1f} {bx-rr:.1f} {by+h:.1f} H {zx:.1f} Z')
            vx, va = bx + 6, "start"
        else:
            path = (f'M {zx:.1f} {by:.1f} H {bx+rr:.1f} Q {bx:.1f} {by:.1f} {bx:.1f} {by+rr:.1f} '
                    f'V {by+h-rr:.1f} Q {bx:.1f} {by+h:.1f} {bx+rr:.1f} {by+h:.1f} H {zx:.1f} Z')
            vx, va = bx - 6, "end"
        s.append(f'<path d="{path}" class="{cls}"><title>{lbl}: {v:.1f} {note}</title></path>')
        s.append(f'<text x="{X0+2}" y="{y0+10:.1f}" class="rf-barlbl">{lbl}</text>')
        s.append(f'<text x="{vx:.1f}" y="{by+h-3:.1f}" class="rf-vlbl" text-anchor="{va}">{v:.1f}</text>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-1}" class="rf-axname" text-anchor="middle">AUROC (%), bars grow from chance</text>')
    s.append("</svg>")
    return "".join(s)


def fig_cell_landscape_losnet():
    """HotpotQA / Mistral-7B-v0.2: LOS-Net Table 1 (published) + ours + our baselines."""
    lu, _, ub, pb = _load_all()
    cell = "losnet_hotpotqa_mistral7b"
    rows = []
    for r in pb:
        if r["cell"] != cell:
            continue
        kind = "ctx-sup" if r["supervision"].startswith("sup") else "ctx"
        sup_tag = " — supervised" if kind == "ctx-sup" else ""
        rows.append((f'{r["method"]}{sup_tag} (published)', _f(r["auroc"]), kind, r.get("note", "")))
    g = _lu_g5(lu, cell)
    if g:
        rows.append(("L-SML GOOD_5 — ours, K=1, unsupervised", _f(g["auroc_X"]) * 100, "ours",
                     f'[{_f(g["lo"])*100:.1f}, {_f(g["hi"])*100:.1f}]'))
    u = _ub_row(ub, cell)
    if u:
        for key, lbl in (("seqlp_auroc", "Sequence log-prob — our traces"),
                         ("pmean_auroc", "Probas-mean — our traces")):
            v = _f(u.get(key))
            if v:
                rows.append((lbl, v * 100, "ctx", "score_ubaselines.py"))
    if not rows:
        return ""
    rows.sort(key=lambda r: -r[1])
    fnote = ("Published rows: LOS-Net arXiv 2503.14043 Table 1, Mistral-7b-instruct-v0.2 + HotpotQA column "
             "(includes p(True) — Kadavath et al. 2022 — and the Logits/Probas aggregation baselines). "
             "Our rows are scored on our own traces of the same (model, dataset) cell; labels/splits differ from theirs, "
             "so treat cross-block comparisons as indicative. This cell is out-of-regime for the spectral method "
             "(multi-hop QA) and is reported as an honest loss.")
    return _fig("HotpotQA · Mistral-7B-v0.2 — the LOS-Net comparison, their full baseline table",
                "Who LOS-Net compares against, plus us: supervised probes on top, trivial aggregations at the bottom.",
                "", _bar_panel(rows, d0=35, d1=100), fnote)


def fig_cell_landscape_gsm8k_llama8b():
    """GSM8K / Llama-3.1-8B: every method in reasoning_benchmark.csv + our baselines."""
    lu, rb, ub, _ = _load_all()
    cell = "lapeigvals_gsm8k_llama8b"
    rows = []
    for r in rb:
        if r["dataset"] != "GSM8K" or r["model"] != "Llama-3.1-8B":
            continue
        if r["is_ours"] == "yes":
            continue  # ours comes from LU below (single primary row)
        v = _f(r["auroc"])
        if v is None:
            continue
        m = r["method"]
        if m in ("Sequence logprob", "Perplexity", "Naive entropy (mean H)"):
            kind, tag = "ctx", " — our traces"
        elif r.get("supervision") == "supervised":
            kind, tag = "ctx-sup", " — supervised"
        elif r.get("citable") == "no":
            kind, tag = "ctx-flag", " — old cache, under reconciliation"
        else:
            kind, tag = "ctx", ""
        rows.append((m + tag, v, kind, r.get("note", "")[:80]))
    g = _lu_g5(lu, cell)
    if g:
        rows.append(("L-SML GOOD_5 — ours, K=1, unsupervised", _f(g["auroc_X"]) * 100, "ours",
                     f'[{_f(g["lo"])*100:.1f}, {_f(g["hi"])*100:.1f}]'))
    if not rows:
        return ""
    rows.sort(key=lambda r: -r[1])
    fnote = ("All published rows from reasoning_benchmark.csv (sources inline there); baselines rows are scored on our own "
             "traces (score_ubaselines.py). Ours 81.5 [76.8, 85.4]: the CI excludes the published unsupervised anchor "
             "(LapEigvals AttentionScore 72.0) and does not reach the supervised probe (87.2).")
    return _fig("GSM8K · Llama-3.1-8B — every method we cite, one cell",
                "The most-covered cell in the project: published anchors, competitor methods scored on our traces, and standard baselines.",
                "", _bar_panel(rows, d0=45, d1=100), fnote)


# ── Figure 4: GOOD_5 vs sequence log-prob scatter ─────────────────────────────
MATH_DS = {"gsm8k", "math500"}


def fig_good5_vs_seqlp():
    lu, _, ub, _ = _load_all()
    cell_ds = {r["cell"]: r["dataset"] for r in lu}
    pts = []
    for r in ub:
        cell = r["cell"]
        if cell.endswith(("_partial", "_pilot")):
            continue
        sq = _f(r.get("seqlp_auroc"))
        g = _lu_g5(lu, cell)
        g5 = _f(g["auroc_X"]) if g else _f(r.get("lsml_good5_auroc"))
        if sq is None or g5 is None:
            continue
        is_math = cell_ds.get(cell, "") in MATH_DS
        pts.append((sq * 100, g5 * 100, cell, is_math))
    if not pts:
        return ""
    SZ, M, D0, D1 = 520, 54, 38, 98
    x = lambda v: _lin(v, D0, D1, M, SZ - 14)
    y = lambda v: _lin(v, D0, D1, SZ - M, 14)
    s = [_svg(SZ, SZ)]
    s.append(f'<path d="M {x(D0):.1f} {y(D0):.1f} L {x(D1):.1f} {y(D1):.1f} L {x(D0):.1f} {y(D1):.1f} Z" class="rf-wash"/>')
    for gv in (40, 50, 60, 70, 80, 90):
        s.append(f'<line x1="{x(gv):.1f}" y1="{y(D0):.1f}" x2="{x(gv):.1f}" y2="{y(D1):.1f}" class="rf-grid"/>')
        s.append(f'<line x1="{x(D0):.1f}" y1="{y(gv):.1f}" x2="{x(D1):.1f}" y2="{y(gv):.1f}" class="rf-grid"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{y(D0)+16:.1f}" class="rf-tick" text-anchor="middle">{gv}</text>')
        s.append(f'<text x="{x(D0)-8:.1f}" y="{y(gv)+4:.1f}" class="rf-tick" text-anchor="end">{gv}</text>')
    s.append(f'<line x1="{x(D0):.1f}" y1="{y(D0):.1f}" x2="{x(D1):.1f}" y2="{y(D1):.1f}" class="rf-diag"/>')
    s.append(f'<text x="{x(55):.1f}" y="{y(89):.1f}" class="rf-region">spectral fusion ahead</text>')
    s.append(f'<text x="{x(70):.1f}" y="{y(44):.1f}" class="rf-region">sequence log-prob ahead</text>')
    for (px, py, cell, is_math) in pts:
        cls = "rf-ours" if is_math else "rf-pt-qa"
        s.append(f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="5.5" class="{cls}"><title>{cell}: GOOD_5 {py:.1f} vs seq-logprob {px:.1f}</title></circle>')
    # annotate the extreme cells only (largest |delta| each side), label toward free space
    deltas = sorted(pts, key=lambda p: p[1] - p[0])
    for p, up in ((deltas[0], False), (deltas[-1], True)):
        px, py, cell, _m = p
        ax_, ay_ = x(px) + (12 if not up else 12), y(py) + (16 if not up else -12)
        s.append(f'<line x1="{x(px):.1f}" y1="{y(py)+(4 if not up else -4):.1f}" x2="{ax_-2:.1f}" y2="{ay_-3:.1f}" class="rf-leader"/>')
        s.append(f'<text x="{ax_:.1f}" y="{ay_:.1f}" class="rf-albl" text-anchor="start">{cell.replace("_", " ")} {py-px:+.1f}</text>')
    s.append(f'<text x="{(x(D0)+x(D1))//2}" y="{SZ-4}" class="rf-axname" text-anchor="middle">Sequence log-prob AUROC (%) — same traces, same labels</text>')
    s.append(f'<text x="14" y="{(y(D0)+y(D1))//2}" class="rf-axname" text-anchor="middle" transform="rotate(-90 14 {(y(D0)+y(D1))//2})">L-SML GOOD_5 AUROC (%)</text>')
    s.append("</svg>")
    legend = (
        '<span class="rf-li"><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" class="rf-ours"/></svg> math CoT cells (GSM8K / MATH-500)</span>'
        '<span class="rf-li"><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" class="rf-pt-qa"/></svg> QA / other cells</span>'
        '<span class="rf-li"><svg width="24" height="14"><line x1="2" y1="12" x2="22" y2="2" class="rf-diag"/></svg> y = x (no gain over the baseline)</span>')
    fnote = ("Standard-baselines audit (appendix): sequence log-prob is the classic likelihood baseline "
             "(Malinin &amp; Gales 2021; Guerreiro et al. EACL 2023). Pilot/partial cells excluded. "
             "Source: ubaseline_scores.csv × scores_lsml_upcr.csv.")
    return _fig("L-SML GOOD_5 vs sequence log-prob, scored on identical traces and labels",
                "Each dot is one (dataset, model) cell. Above the diagonal = the spectral fusion beats the one-number baseline.",
                legend, "".join(s), fnote)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-dataset family (2026-07-13): the EDIS/EPR-paper-style comparisons —
# same figure per dataset across models, plus the master per-domain table.
# Legacy cells come from results/subset_sweep/sweep_summary.csv (no CIs there).

SWEEP = os.path.join(RESULTS, "subset_sweep", "sweep_summary.csv")


def _sweep():
    return _read(SWEEP)


def _sweep_val(sw, domain, key_substr):
    for r in sw:
        if r["domain"] == domain and key_substr in r["cell_key"]:
            return _f(r["good5_auroc"]), _f(r["pos_rate"]), r["cell_key"], _f(r["n"])
    return None, None, None, None


def _generic_forest(rows, D0, D1, axname, teach=True, ticks=None):
    """rows: (label, v, lo, hi, anchors[(val,name,kind)], seqlp)
    kind: 'unsup' filled diamond | 'sup' open triangle | 'flag' open diamond
    lo/hi None -> no whisker (legacy cells)."""
    X0, X1, ROW, TOP = 235, 850, 38, 14
    n = len(rows)
    H = TOP + n * ROW + 34
    x = lambda v: _lin(v, D0, D1, X0, X1)
    s = [_svg(900, H)]
    gticks = [g for g in (ticks or (40, 50, 60, 70, 80, 90)) if D0 <= g <= D1]
    for gv in gticks:
        s.append(f'<line x1="{x(gv):.1f}" y1="{TOP}" x2="{x(gv):.1f}" y2="{TOP+n*ROW}" class="rf-grid"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{TOP+n*ROW+16}" class="rf-tick" text-anchor="middle">{gv}</text>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-2}" class="rf-axname" text-anchor="middle">{axname}</text>')
    for i, (lbl, v, lo, hi, anchors, sq) in enumerate(rows):
        cy = TOP + i * ROW + ROW / 2
        s.append(f'<text x="225" y="{cy+4:.1f}" class="rf-rowlbl" text-anchor="end">{lbl}</text>')
        if lo is not None and hi is not None:
            s.append(f'<line x1="{x(lo):.1f}" y1="{cy:.1f}" x2="{x(hi):.1f}" y2="{cy:.1f}" class="rf-ci"/>')
        if sq is not None:
            s.append(f'<line x1="{x(sq):.1f}" y1="{cy-8:.1f}" x2="{x(sq):.1f}" y2="{cy+8:.1f}" class="rf-seqlp"><title>Sequence log-prob on our traces: {sq:.1f}</title></line>')
        for (av, an, kind) in anchors:
            ax = x(av)
            if kind == "sup":
                s.append(f'<path d="M {ax:.1f} {cy-6:.1f} L {ax+6:.1f} {cy+5:.1f} L {ax-6:.1f} {cy+5:.1f} Z" class="rf-sup"><title>{an}: {av} (supervised, same model)</title></path>')
            elif kind == "flag":
                s.append(f'<path d="M {ax:.1f} {cy-6.5:.1f} L {ax+6.5:.1f} {cy:.1f} L {ax:.1f} {cy+6.5:.1f} L {ax-6.5:.1f} {cy:.1f} Z" class="rf-anchor-open"><title>{an}: {av} (published — caveated, see footnote)</title></path>')
            else:
                s.append(f'<path d="M {ax:.1f} {cy-6.5:.1f} L {ax+6.5:.1f} {cy:.1f} L {ax:.1f} {cy+6.5:.1f} L {ax-6.5:.1f} {cy:.1f} Z" class="rf-anchor"><title>{an}: {av} (published, unsupervised, same model)</title></path>')
        s.append(f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="5.5" class="rf-ours"><title>L-SML GOOD_5 (ours): {v:.1f}' + (f" [{lo:.1f}, {hi:.1f}]" if lo is not None else " (legacy, no CI in summary CSV)") + '</title></circle>')
        if teach and i == 0:
            s.append(f'<text x="{x(v):.1f}" y="{cy-10:.1f}" class="rf-dlbl rf-dlbl-ours" text-anchor="middle">{v:.1f}</text>')
    s.append("</svg>")
    return "".join(s)


FOREST_LEGEND = (
    '<span class="rf-li"><svg width="30" height="14"><line x1="2" y1="7" x2="28" y2="7" class="rf-ci"/><circle cx="15" cy="7" r="5.5" class="rf-ours"/></svg> L-SML GOOD_5 — ours, K=1, unsupervised (95% CI where scored fresh)</span>'
    '<span class="rf-li"><svg width="16" height="14"><path d="M 8 1 L 14 7 L 8 13 L 2 7 Z" class="rf-anchor"/></svg> published unsupervised anchor, same model</span>'
    '<span class="rf-li"><svg width="16" height="14"><path d="M 8 1 L 14 7 L 8 13 L 2 7 Z" class="rf-anchor-open"/></svg> published but caveated (old-cache / protocol mismatch)</span>'
    '<span class="rf-li"><svg width="16" height="14"><path d="M 8 2 L 14 12 L 2 12 Z" class="rf-sup"/></svg> published supervised (ceiling)</span>'
    '<span class="rf-li"><svg width="10" height="14"><line x1="5" y1="1" x2="5" y2="13" class="rf-seqlp"/></svg> sequence log-prob on our own traces</span>')

MATH500_SPEC = [
    # display label, sweep cell_key substring, RB model string (for anchors)
    ("Qwen2.5-Math-7B",      "Qwen-Math-7B",                 "Qwen2.5-Math-7B"),
    ("Qwen2.5-Math-1.5B",    "Qwen2.5-Math-1.5B",            None),
    ("R1-Distill-Llama-8B",  "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Llama-8B"),
    ("DeepSeek-Math-7B",     "deepseek-math-7b",             None),
]


def _rb_anchors(rb, dataset, model):
    out = []
    if not model:
        return out
    for r in rb:
        if r["dataset"] != dataset or r["model"] != model or r["is_ours"] == "yes":
            continue
        v = _f(r["auroc"])
        if v is None:
            continue
        kind = ("sup" if r.get("supervision") == "supervised"
                else "flag" if r.get("citable") == "no" else "unsup")
        out.append((v, r["method"], kind))
    return out


def fig_math500_forest():
    _, rb, _, _ = _load_all()
    sw = _sweep()
    rows = []
    for (lbl, key, rbm) in MATH500_SPEC:
        v, pos, _ck, _n = _sweep_val(sw, "math500", key)
        if v is None:
            continue
        flag = gate_flag(pos)
        tag = " †" if flag == "CEILING" else " ▿" if flag == "FLOOR" else ""
        rows.append((lbl + tag, v * 100, None, None, _rb_anchors(rb, "MATH-500", rbm), None))
    rows.sort(key=lambda r: -r[1])
    if not rows:
        return ""
    fnote = ("Ours = legacy subset-sweep cells (N=300, T=1.0; sweep_summary.csv carries no CIs — the per-cell bootstrap "
             "lives in the npz manifests). R1-Distill anchors from ARS arXiv 2601.17467 Tables 1–2 (same model): the supervised "
             "probe at 86.4 vs our unsupervised 84.4, with every published unsupervised baseline 8–43pp below us — Semantic Entropy "
             "and Perplexity collapse below chance on long R1 traces. Qwen2.5-Math-7B's old-cache anchors (open diamonds) are "
             "under the NLI-truncation reconciliation (Step 152 P1) and not yet citable.")
    return _fig("MATH-500, single-pass unsupervised detection across four models",
                "Same story as the GSM8K sweep, on the second reasoning dataset. Rows sorted by our AUROC.",
                FOREST_LEGEND, _generic_forest(rows, 38, 98, "AUROC (%) · MATH-500, same model per row"), fnote)


def fig_triviaqa_forest():
    lu, _, ub, pb = _load_all()
    spec = [
        # label, cell, note
        ("Qwen3-8B",                 "semenergy_triviaqa_qwen3_8b", ""),
        ("Llama-3.1-8B §",           "spilled_triviaqa_llama8b",    "energy-capture subset, valid 0.51"),
        ("Mistral-Small-3.1-24B",    "epr_triviaqa_mistral24b",     ""),
        ("OPT-30B (base) ¶",         "seiclr_triviaqa_opt30b",      "SE-ICLR per-question K=10 protocol"),
    ]
    rows = []
    for (lbl, cell, note) in spec:
        g = _lu_g5(lu, cell)
        if g is None:
            continue
        v, lo, hi = _f(g["auroc_X"]) * 100, _f(g["lo"]) * 100, _f(g["hi"]) * 100
        anchors = [(_f(r["auroc"]), r["method"],
                    "sup" if r["supervision"].startswith("sup") else "unsup")
                   for r in pb if r["cell"] == cell]
        if not anchors:
            y, ym = _f(g.get("published_Y")), g.get("Y_method", "")
            if y:
                kind = "flag" if "ICLR" in ym else "unsup"
                anchors = [(y * 100, ym, kind)]
        u = _ub_row(ub, cell)
        sq = _f(u["seqlp_auroc"]) * 100 if u and _f(u.get("seqlp_auroc")) else None
        flag = gate_flag(g.get("acc"))
        tag = " †" if flag == "CEILING" else " ▿" if flag == "FLOOR" else ""
        rows.append((lbl + tag, v, lo, hi, anchors, sq))
    rows.sort(key=lambda r: -r[1])
    if not rows:
        return ""
    fnote = ("The EPR-paper-style comparison (their Table 1) on the models we share with the literature. "
             "Mistral-Small-3.1-24B anchors are the EPR paper's own same-model TriviaQA row: SelfCheckGPT 79.0 and EPR 74.6 "
             "(unsupervised, diamonds) plus HalluDetect 78.7 and WEPR 82.0 (supervised, triangles) — we sit below their "
             "unsupervised pair on this cell (our best variant, U-PCR GOOD_5+logprob, reaches 73.6). "
             "Qwen3-8B: we beat Semantic Energy's published 74.8 by +5.3pp, CI-clear. "
             "§ energy-capture cell — only 51% of traces carry the energy fields (selection caveat). "
             "¶ SE-ICLR's 83 is per-question K=10 semantic sampling — different units, caveated open diamond. "
             "The EPR paper's other models (Falcon-3-10B, Phi-4, Ministral-8B) and ArGiMi were not run — no same-model row exists for them.")
    return _fig("TriviaQA, single-pass unsupervised detection across four models",
                "The QA flagship dataset, model-by-model against every published same-model number we track.",
                FOREST_LEGEND, _generic_forest(rows, 48, 98, "AUROC (%) · TriviaQA, same model per row"), fnote)


def fig_qa_extension_forest():
    lu, _, ub, pb = _load_all()
    sw = _sweep()
    spec = [
        ("SQuAD v2 · Llama-8B",          "se_squad_v2_llama8b"),
        ("SciQ (MCQ) · Llama-8B",        "sciq_llama8b"),
        ("NQ-Open · Llama-8B",           "se_nq_open_llama8b"),
        ("CoQA · LLaMA-7B",              "inside_coqa_llama7b"),
        ("TruthfulQA · Llama-8B",        "truthfulqa_llama8b"),
        ("HotpotQA · Mistral-7B-v0.2",   "losnet_hotpotqa_mistral7b"),
    ]
    rows = []
    for (lbl, cell) in spec:
        g = _lu_g5(lu, cell)
        if g is None:
            continue
        v, lo, hi = _f(g["auroc_X"]) * 100, _f(g["lo"]) * 100, _f(g["hi"]) * 100
        anchors = []
        y, ym = _f(g.get("published_Y")), g.get("Y_method", "")
        if y:
            kind = "sup" if any(k in ym for k in ("LOS-Net", "TSV")) else "unsup"
            anchors.append((y * 100, ym, kind))
        for r in pb:  # add the published unsup SE row on the LOS-Net cell
            if r["cell"] == cell and r["method"] == "Semantic Entropy":
                anchors.append((_f(r["auroc"]), "Semantic Entropy (LOS-Net T1)", "unsup"))
        u = _ub_row(ub, cell)
        sq = _f(u["seqlp_auroc"]) * 100 if u and _f(u.get("seqlp_auroc")) else None
        flag = gate_flag(g.get("acc"))
        tag = " †" if flag == "CEILING" else " ▿" if flag == "FLOOR" else ""
        rows.append((lbl + tag, v, lo, hi, anchors, sq))
    # WebQuestions from the legacy Phase-9 cache (no CI, no seqlp)
    v, pos, _ck, _n = _sweep_val(sw, "qa", "webq_cot")
    if v is not None:
        flag = gate_flag(pos)
        tag = " ▿" if flag == "FLOOR" else " †" if flag == "CEILING" else ""
        rows.append(("WebQuestions · Phase-9 legacy" + tag, v * 100, None, None, [], None))
    rows.sort(key=lambda r: -r[1])
    if not rows:
        return ""
    fnote = ("The Item-3 QA extension, complete: every dataset ends scored (gate flags: † CEILING, ▿ FLOOR — scored, "
             "flagged, out of the win tally). CoQA and TruthfulQA carry published same-model anchors (INSIDE 80.4; TSV 84.2 "
             "semi-supervised) — both honest losses on floor-flagged cells. HotpotQA shows LOS-Net's supervised probe (triangle) "
             "and its own published unsupervised Semantic Entropy row (diamond). SQuAD v2 / NQ-Open / SciQ have no published "
             "same-model detection anchor — our rows stand with the seq-logprob audit tick. WebQuestions is the legacy Phase-9 "
             "CoT cache (model differs from the EPR paper's WebQ models, so their anchors do not transfer; no CI in the sweep CSV). "
             "TriviaQA rows live in the dedicated TriviaQA figure.")
    return _fig("The QA extension, dataset by dataset",
                "Seven short-answer / open-domain QA datasets beyond TriviaQA — ours vs every published same-model anchor that exists.",
                FOREST_LEGEND, _generic_forest(rows, 40, 96, "AUROC (%) · one QA dataset per row"), fnote)


# ── Master per-domain table (EPR-Table-1 style): every dataset × model we ran ──
def master_table_html():
    lu, rb, ub, _ = _load_all()
    sw = _sweep()
    ubx = {r["cell"]: r for r in ub}

    def _row(domain, ds, model, n, acc, ours, ci, seqlp, y, ym, flag):
        d = f"{(ours - y):+.1f}" if (y is not None and ours is not None) else "—"
        fl = f'<span class="loss">{flag}</span>' if flag else ""
        return (f"<tr><td>{domain}</td><td>{ds}</td><td>{model}</td><td>{n or '—'}</td>"
                f"<td>{acc if acc is not None else '—'}</td><td><strong>{ours:.1f}</strong>{ci}</td>"
                f"<td>{seqlp if seqlp else '—'}</td><td>{ym + ' ' + format(y, '.1f') if y else '—'}</td>"
                f"<td>{d}</td><td>{fl}</td></tr>")

    rows = []
    # repgrid cells (fresh, with CI + seqlp + anchor)
    for r in lu:
        if r["subset"] != "GOOD_5" or r["method"] != "lsml":
            continue
        cell = r["cell"]
        if any(s in cell for s in ("_partial", "_pilot", "_reject")):
            continue
        ds = DS_PRETTY.get(r["dataset"], r["dataset"])
        model = r["model"].split("/")[-1]
        ours = _f(r["auroc_X"]) * 100
        ci = f' <span class="mono" style="font-size:11px">[{_f(r["lo"])*100:.1f}, {_f(r["hi"])*100:.1f}]</span>'
        u = ubx.get(cell)
        sq = f"{_f(u['seqlp_auroc'])*100:.1f}" if u and _f(u.get("seqlp_auroc")) else None
        y = _f(r.get("published_Y"))
        y = y * 100 if y else None
        domain = "Math CoT" if r["dataset"] in MATH_DS else "QA"
        rows.append(("0" + domain, _row(domain, ds, model, r.get("n_problems"), r.get("acc"),
                                        ours, ci, sq, y, r.get("Y_method", ""), gate_flag(r.get("acc")))))
    # legacy sweep cells (no CI in summary csv)
    LEG = {"math500": ("Math CoT", "MATH-500"), "gsm8k": ("Math CoT", "GSM8K (legacy)"),
           "gpqa": ("MCQ", "GPQA"), "qa": ("QA (legacy)", None), "rag": ("RAG", None)}
    for r in sw:
        if r["domain"] not in LEG:
            continue
        dom, dsfix = LEG[r["domain"]]
        key = r["cell_key"]
        ds = dsfix or key.replace("spectral_phase9_cache_", "").replace("_traces_T1.0", "")
        model = key.replace("_T1.0", "")
        if r["domain"] == "rag":
            model, ds = key.split("/")[0], key.split("/")[1]
        elif r["domain"] == "qa":
            model = "Phase-9 legacy"
        ours = _f(r["good5_auroc"])
        if ours is None:
            continue
        rows.append(("1" + dom, _row(dom + " (legacy)", ds, model, r.get("n"), r.get("pos_rate"),
                                     ours * 100, "", None, None, None, gate_flag(r.get("pos_rate")))))
    rows.sort(key=lambda t: t[0])
    body = "".join(t[1] for t in rows)
    return f"""
<h3>Every (dataset, model) cell we ran — one table</h3>
<p>The EPR-paper-Table-1 view of the whole project: fresh replication-grid cells first (with CIs,
the same-trace seq-logprob audit, and the published same-model anchor where one exists), then the
legacy subset-sweep battery (MATH-500, GPQA, WebQ/TriviaQA CoT, and the 4-model × 4-dataset RAG
grid). Gate flags per the two-tier policy; flagged cells are scored but out of the win tally.</p>
<div style="overflow-x:auto"><table>
<tr><th>Domain</th><th>Dataset</th><th>Model</th><th>N</th><th>Acc</th><th>Ours GOOD_5</th>
<th>seq-logprob</th><th>Published same-model Y</th><th>Δ</th><th>Flag</th></tr>
{body}
</table></div>
<p class="rf-fnote">Legacy rows: subset-sweep summary carries point estimates only (per-cell bootstrap
lives in the npz manifests); Acc column shows the positive-class rate. RAG and GPQA are the documented
out-of-regime domains (thesis scope: spectral features of H(n) live on reasoning traces) — shown in
full rather than hidden, many at FLOOR. Phase-9 QA legacy cells predate the model-matched protocol.</p>
"""


# ── Item 6 figures (Phase-15 temperature experiment, Step 158) ────────────────
# Values are the same hardcoded Step-158 numbers the item6 tables show
# (source: HISTORY Step 158 / phase15 results summary — no CSV exists for them).
P15_T = [0.3, 0.6, 1.0, 1.5, 2.0]
P15_AUROC = [54.5, 64.4, 85.1, 87.8, 62.9]
P15_ACC = [80.0, 81.5, 70.5, 27.5, 4.0]
P15_ARMS = [
    ("Single pass T=1.0 (base)",        85.1, 77.7, 91.8),
    ("A · K=5 same-T — L-SML fusion",   91.2, 85.8, 95.4),
    ("A · K=5 same-T — simple average", 90.6, 85.0, 94.8),
    ("B · K=5 multi-T — L-SML fusion",  85.9, 79.4, 91.4),
    ("B · K=5 multi-T — simple average", 83.0, 76.0, 89.0),
]


def _line_panel(ys, cls_line, cls_dot, ylab, y0, y1, band=None):
    """Single-series line over T (x linear in temperature). All 5 points labeled —
    the chart replaces a 5-value table row."""
    W, H, ML, MB = 430, 260, 46, 34
    x = lambda t: _lin(t, 0.2, 2.1, ML, W - 14)
    y = lambda v: _lin(v, y0, y1, H - MB, 12)
    s = [_svg(W, H)]
    if band:  # accuracy gate band [20, 85]
        s.append(f'<rect x="{ML}" y="{y(band[1]):.1f}" width="{W-14-ML}" height="{y(band[0])-y(band[1]):.1f}" class="rf-wash"/>')
        s.append(f'<text x="{ML+6}" y="{y(band[1])+14:.1f}" class="rf-region">gate band [{band[0]}, {band[1]}]</text>')
    for gv in range(int(y0), int(y1) + 1, 20):
        s.append(f'<line x1="{ML}" y1="{y(gv):.1f}" x2="{W-14}" y2="{y(gv):.1f}" class="rf-grid"/>')
        s.append(f'<text x="{ML-6}" y="{y(gv)+4:.1f}" class="rf-tick" text-anchor="end">{gv}</text>')
    for t in P15_T:
        s.append(f'<text x="{x(t):.1f}" y="{H-MB+16}" class="rf-tick" text-anchor="middle">{t}</text>')
    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(P15_T, ys))
    s.append(f'<polyline points="{pts}" class="{cls_line}"/>')
    for t, v in zip(P15_T, ys):
        s.append(f'<circle cx="{x(t):.1f}" cy="{y(v):.1f}" r="4.5" class="{cls_dot}"><title>T={t}: {v}</title></circle>')
        dy = -9 if v < (y0 + y1) / 2 or v > y1 - 8 else -9
        s.append(f'<text x="{x(t):.1f}" y="{y(v)+dy:.1f}" class="rf-vlbl" text-anchor="middle">{v:g}</text>')
    s.append(f'<text x="{(ML+W-14)//2}" y="{H-2}" class="rf-axname" text-anchor="middle">sampling temperature T</text>')
    s.append(f'<text x="12" y="{(y(y0)+y(y1))//2:.0f}" class="rf-axname" text-anchor="middle" transform="rotate(-90 12 {(y(y0)+y(y1))//2:.0f})">{ylab}</text>')
    s.append("</svg>")
    return "".join(s)


def fig_item6_temperature():
    p1 = _line_panel(P15_AUROC, "rf-line", "rf-ours", "single-pass AUROC (%)", 40, 100)
    p2 = _line_panel(P15_ACC, "rf-line-acc", "rf-dot-acc", "accuracy (%)", 0, 100, band=(20, 85))
    body = (f'<div class="rf-panes"><div class="rf-pane"><div class="rf-ptitle">Detection AUROC vs T</div>{p1}</div>'
            f'<div class="rf-pane"><div class="rf-ptitle">Task accuracy vs T (the confound)</div>{p2}</div></div>')
    fnote = ("Two panels, one axis each (never a dual-axis chart): the left panel's apparent peak at T=1.5 must be read "
             "against the right panel — accuracy collapses 80% → 4%, leaving the gate band above T=1.0, so the class mix "
             "under the AUROC shifts and T=2.0 has only 8 correct answers. That is why gate G-T1 FAILS and Q1's honest "
             "answer is: no usable temperature lever. Values = the Q1 table above (Step 158, MATH-500 / Qwen2.5-Math-7B, N=200).")
    return _fig("Q1 as a picture — the inverted-U and its confound",
                "Same numbers as the Q1 table: single-pass L-SML GOOD_5 AUROC per temperature (left), task accuracy per temperature (right).",
                "", body, fnote)


def fig_item6_arms():
    rows = [(lbl, v, lo, hi, [], None) for (lbl, v, lo, hi) in P15_ARMS]
    svg = _generic_forest(rows, 74, 97, "AUROC (%) · paired on the same 200 MATH-500 questions",
                          teach=False, ticks=(75, 80, 85, 90, 95))
    fnote = ("Same numbers as the Q2 table. The paired deltas are the result: A − base = <b>+6.1pp [+0.4, +12.8]</b> "
             "(more same-T passes help) and B − A = <b>−5.3pp [−10.3, −1.1]</b>, CI excluding zero — temperature "
             "diversity hurts. Mechanism: same-T passes correlate ρ ≈ +0.45 (averaging denoises one good signal); "
             "multi-T passes decorrelate (ρ ≈ +0.01) only because the off-temperature passes are near-random. "
             "Single (dataset, model) result — MATH-500 / Qwen2.5-Math-7B.")
    return _fig("Q2 as a picture — five arms, one axis",
                "Single pass vs K=5 same-temperature vs K=5 mixed-temperature, each with its 95% CI.",
                "", svg, fnote)


if __name__ == "__main__":
    # quick self-check: render all figures, report sizes
    for name, fn in (("gsm8k_forest", fig_gsm8k_forest),
                     ("same_model_deltas", fig_same_model_deltas),
                     ("losnet_landscape", fig_cell_landscape_losnet),
                     ("gsm8k_llama8b_landscape", fig_cell_landscape_gsm8k_llama8b),
                     ("good5_vs_seqlp", fig_good5_vs_seqlp),
                     ("math500_forest", fig_math500_forest),
                     ("triviaqa_forest", fig_triviaqa_forest),
                     ("qa_extension_forest", fig_qa_extension_forest),
                     ("master_table", master_table_html)):
        h = fn()
        print(f"{name}: {len(h)} chars, svg={'<svg' in h}")
