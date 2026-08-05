#!/usr/bin/env python
"""
labelfree_standing_report.py — one page combining the QA-evaluation desk (old
`item3_qa_evaluation.html`) and the published-benchmark desk (old
`item4_benchmarking.html`), rescored for the two **label-free** arms.

WHY THIS EXISTS
---------------
Both source pages report `L-SML GOOD_5` — a five-view subset that was chosen with
answer keys, on a 16-view pool, over a cell roster that still included RAG and
GPQA. None of those three things is true of the current pipeline:

  * the shortlist is now the two arms that need nothing hand-picked beyond the
    single global anchor bit — `a2.dufs_pf` (DUFS parameter-free + L-SML) and
    U-PCR with sign(rho) orientation;
  * the pool is the 30-view CANONICAL_POOL from the cluster re-runs;
  * scope is the 24 in-scope cells (9 QA + 15 math). RAG and GPQA are OUT
    (Step 191), so no cell from either appears anywhere on this page; and
    `inside_coqa_llama7b` is REJECTED (Step 216) for a generation defect, so it
    is out of every macro here until it is re-generated.

Everything is recomputed here from the canonical path rather than copied from the
old pages, and each arm is cross-checked against its recorded value before any of
it is written out.

SOURCES
  scripts/inscope_bench_common.load_cells        the 24 cells, z-scored over
                                                 CANONICAL_POOL, per-cell anchor
  results/selector_bench/a2_groupfs__c46.csv     `a2.dufs_pf` selected view sets
  results/upcr_study/06_orientation/per_cell.csv recorded U-PCR + sign(rho) AUROC
  results/advisor_inscope/competitors_verified.csv  published numbers + cost class
  results/repgrid/scores_lsml_upcr.csv           per-cell task accuracy / N
  results/repgrid/ubaseline_scores.csv           seq-logprob + perplexity floors

OUTPUT
  results/action_items/labelfree_standing.html

Usage:
    python scripts/labelfree_standing_report.py
"""
import csv
import html
import os
import statistics as st
import sys

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import INSCOPE, QA_CELLS, MATH_CELLS              # noqa: E402
from inscope_bench_common import (load_cells, assert_good6, good6_cols,  # noqa: E402
                                  GOOD6_EXPECTED)
from report_figs import FIG_CSS, _svg, _fig, _lin                    # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous, boot_auc    # noqa: E402
from spectral_utils.streaming_utils import anchor_orient             # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                    # noqa: E402
from spectral_utils.upcr import upcr_fit                             # noqa: E402

OUT = os.path.join(REPO, "results", "action_items", "labelfree_standing.html")

# exp06's fitted configuration, verbatim — this page must not silently re-tune it.
FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

ACC_BAND = (0.20, 0.85)

# Bars are cost classes, not quality tiers: a number is only comparable to ours if
# it was bought with the same budget. Ordered from our exact class outward.
BARS = [
    ("A", "our exact class: unsupervised, one forward pass, grey-box (logprobs only)",
     lambda r: r["supervision"] == "unsupervised" and r["passes"] == "1"
     and r["access"] == "grey-box"),
    ("B", "unsupervised, one forward pass, any access (adds white-box internals)",
     lambda r: r["supervision"] == "unsupervised" and r["passes"] == "1"),
    ("C", "any unsupervised method, including 10-pass / K-sample",
     lambda r: r["supervision"] == "unsupervised"),
    ("D", "the best method in each paper, including supervised probes",
     lambda r: True),
]

PRETTY = {
    "epr_triviaqa_mistral24b": ("TriviaQA (wiki)", "Mistral-Small-3.1-24B"),
    "inside_coqa_llama7b": ("CoQA", "LLaMA-7B (base)"),
    "losnet_hotpotqa_mistral7b": ("HotpotQA", "Mistral-7B-v0.2"),
    "sciq_llama8b": ("SciQ", "Llama-3.1-8B"),
    "se_nq_open_llama8b": ("NQ-Open", "Llama-3.1-8B"),
    "se_squad_v2_llama8b": ("SQuAD v2", "Llama-3.1-8B"),
    "seiclr_triviaqa_opt30b": ("TriviaQA", "OPT-30B (base)"),
    "semenergy_triviaqa_qwen3_8b": ("TriviaQA", "Qwen3-8B"),
    "spilled_triviaqa_llama8b": ("TriviaQA (spilled cfg)", "Llama-3.1-8B"),
    "truthfulqa_llama8b": ("TruthfulQA (gen.)", "Llama-3.1-8B"),
    "ars_gsm8k_r1distill8b": ("GSM8K", "R1-Distill-Llama-8B"),
    "internalstates_gsm8k_qwen25_7b": ("GSM8K (T=0.8)", "Qwen2.5-7B"),
    "lapeigvals_gsm8k_llama3b": ("GSM8K", "Llama-3.2-3B"),
    "lapeigvals_gsm8k_llama8b": ("GSM8K", "Llama-3.1-8B"),
    "lapeigvals_gsm8k_mistral24b": ("GSM8K", "Mistral-Small-24B"),
    "lapeigvals_gsm8k_nemo": ("GSM8K", "Mistral-Nemo-12B"),
    "lapeigvals_gsm8k_phi35": ("GSM8K", "Phi-3.5-mini"),
    "noise_gsm8k_mistral7b": ("GSM8K", "Mistral-7B-v0.3"),
    "noise_gsm8k_phi3mini": ("GSM8K", "Phi-3-mini"),
    "math500_dsmath7b": ("MATH-500", "DeepSeek-Math-7B"),
    "math500_qwenmath7b": ("MATH-500", "Qwen2.5-Math-7B"),
    "math500_r1distill8b": ("MATH-500", "R1-Distill-Llama-8B"),
    "math500_r1distill8b_mn4096": ("MATH-500 (4096 tok)", "R1-Distill-Llama-8B"),
    "trace_gsm8k_llama8b_k10": ("GSM8K (K=10 traces)", "Llama-3.1-8B"),
    "trace_math500_qwenmath15b_k10": ("MATH-500 (K=10 traces)", "Qwen2.5-Math-1.5B"),
}


def esc(s):
    return html.escape(str(s), quote=True)


def flag_of(pos_rate):
    if pos_rate < ACC_BAND[0]:
        return "FLOOR"
    if pos_rate > ACC_BAND[1]:
        return "CEILING"
    return ""


# ── scoring ───────────────────────────────────────────────────────────────────
def lsml_oriented(cell, cols):
    V = cell["V"]
    fused, _ = lsml_continuous(*[V[:, c] for c in sorted(set(cols))])
    s, _ = anchor_orient(np.asarray(fused, dtype=float), cell["anchor"])
    return s


def upcr_rho_oriented(cell):
    """U-PCR with polarity from sign(rho-hat), global sign from the anchor.

    `prepare_cell` has already applied ALL_SIGNS, so V must be multiplied by the
    hand signs to RECOVER the unoriented feature before rho gets to derive the
    polarity itself. Getting this backwards silently scores the incumbent arm."""
    V, pool = cell["V"], cell["pool"]
    hand = np.array([ALL_SIGNS.get(f, +1) for f in pool], dtype=float)
    V_un = V * hand
    derived = np.sign(upcr_fit(V_un.T, **FIT).rho_hat_full)
    derived[derived == 0] = 1.0
    F = (V_un * derived).T
    res = upcr_fit(F, **FIT)
    s, _ = anchor_orient(res.w @ F, cell["anchor"])
    return s, int(res.keep.sum())


def build_rows():
    cells = load_cells()
    ok, macro = assert_good6(cells, verbose=True)
    if not ok:
        raise SystemExit(f"validity check failed: GOOD_6 macro {macro:.4f} != "
                         f"{GOOD6_EXPECTED}. Refusing to publish numbers off data "
                         "that is not ours.")

    bench = list(csv.DictReader(open(
        os.path.join(REPO, "results/selector_bench/a2_groupfs__c46.csv"), newline="")))
    pf_rec = {r["cell"]: r for r in bench if r["variant"] == "a2.dufs_pf"}
    up_rec = {r["cell"]: r for r in csv.DictReader(open(
        os.path.join(REPO, "results/upcr_study/06_orientation/per_cell.csv"), newline=""))}

    task = {}
    for r in csv.DictReader(open(
            os.path.join(REPO, "results/repgrid/scores_lsml_upcr.csv"), newline="")):
        task.setdefault(r["cell"], r)
    ub = {r["cell"]: r for r in csv.DictReader(open(
        os.path.join(REPO, "results/repgrid/ubaseline_scores.csv"), newline=""))}

    rows, drift = {}, []
    for ck in INSCOPE:
        cell = cells[ck]
        y = cell["labels"]
        pool = cell["pool"]

        chosen = pf_rec[ck]["chosen"].split("|")
        cols = [pool.index(f) for f in chosen if f in pool]
        if len(cols) != len(chosen):
            raise SystemExit(f"{ck}: selected views missing from pool: "
                             f"{sorted(set(chosen) - set(pool))}")
        s_pf = lsml_oriented(cell, cols)
        a_pf, lo_pf, hi_pf = boot_auc(y, s_pf)

        s_up, kept = upcr_rho_oriented(cell)
        a_up, lo_up, hi_up = boot_auc(y, s_up)

        a_g6 = float(roc_auc_score(y, lsml_oriented(cell, good6_cols(cell))))

        # reproduction gate: both arms must land on their recorded value.
        for name, got, want in (("a2.dufs_pf", a_pf, float(pf_rec[ck]["auroc"])),
                                ("upcr+sign(rho)", a_up,
                                 float(up_rec[ck]["auroc_rho_anchor"]))):
            if abs(got - want) > 5e-4:
                drift.append(f"{ck}/{name}: {got:.4f} vs recorded {want:.4f}")

        t = task.get(ck, {})
        pos = float(np.mean(y))
        u = ub.get(ck)
        rows[ck] = dict(
            cell=ck, domain="QA" if ck in QA_CELLS else "math",
            dataset=PRETTY[ck][0], model=PRETTY[ck][1],
            n=int(len(y)), pos=pos, n_pos=int(np.sum(y)),
            task_acc=float(t["acc"]) if t.get("acc") else None,
            flag=flag_of(pos), pool=len(pool),
            pf=a_pf, pf_lo=lo_pf, pf_hi=hi_pf, pf_size=len(cols),
            pf_chosen=chosen,
            up=a_up, up_lo=lo_up, up_hi=hi_up, up_kept=kept,
            g6=a_g6,
            seqlp=float(u["seqlp_auroc"]) if u and u.get("seqlp_auroc") else None,
            ppl=float(u["ppl_auroc"]) if u and u.get("ppl_auroc") else None,
        )
        print(f"  {ck:34s} dufs_pf {a_pf:.4f}  upcr {a_up:.4f}  GOOD_6 {a_g6:.4f}",
              flush=True)

    if drift:
        raise SystemExit("reproduction gate failed:\n  " + "\n  ".join(drift))

    comp = [r for r in csv.DictReader(open(
        os.path.join(REPO, "results/advisor_inscope/competitors_verified.csv"),
        newline="")) if r["cell"] in INSCOPE and r["auroc"]]
    for r in comp:
        r["auroc"] = float(r["auroc"])
    return rows, comp, macro


def bar_best(comp, pred):
    best = {}
    for r in comp:
        if not pred(r):
            continue
        if r["cell"] not in best or r["auroc"] > best[r["cell"]]["auroc"]:
            best[r["cell"]] = r
    return best


def paired(rows, cells, a, b):
    d = [rows[c][b] - rows[c][a] for c in cells]
    return dict(n=len(d), mean=st.mean(d), median=st.median(d),
                w=sum(x > 0 for x in d), l=sum(x < 0 for x in d),
                p=wilcoxon(d).pvalue)


def fmt_pair(s):
    return (f"{s['mean']*100:+.2f}pp mean / {s['median']*100:+.2f}pp median, "
            f"{s['w']}W/{s['l']}L of {s['n']}, p={s['p']:.3f}")


# ── figures ───────────────────────────────────────────────────────────────────
FOREST_LEGEND = (
    '<span class="rf-li"><svg width="30" height="14">'
    '<line x1="2" y1="7" x2="28" y2="7" class="rf-ci"/>'
    '<circle cx="15" cy="7" r="5.5" class="rf-ours"/></svg> '
    'U-PCR + sign(rho) — ours, label-free, K=1 (95% CI)</span>'
    '<span class="rf-li"><svg width="30" height="14">'
    '<line x1="2" y1="7" x2="28" y2="7" class="rf-ci2"/>'
    '<rect x="10" y="2" width="10" height="10" class="rf-ours2"/></svg> '
    'DUFS parameter-free + L-SML — ours, label-free, K=1 (95% CI)</span>'
    '<span class="rf-li"><svg width="10" height="14">'
    '<line x1="5" y1="1" x2="5" y2="13" class="rf-g6"/></svg> '
    'GOOD_6, the hand-picked subset (the bar to clear)</span>'
    '<span class="rf-li"><svg width="16" height="14">'
    '<path d="M 8 1 L 14 7 L 8 13 L 2 7 Z" class="rf-anchor"/></svg> '
    'published unsupervised, same model</span>'
    '<span class="rf-li"><svg width="16" height="14">'
    '<path d="M 8 2 L 14 12 L 2 12 Z" class="rf-sup"/></svg> '
    'published supervised (ceiling)</span>')

EXTRA_CSS = """
  .rf-ci2{stroke:#a855f7;stroke-width:2;stroke-linecap:round;}
  .rf-ours2{fill:#a855f7;stroke:#fff;stroke-width:1.5;}
  .rf-g6{stroke:#94a3b8;stroke-width:1.5;stroke-dasharray:2 2;}
  .arm{font-weight:700;}
  .arm-up{color:#2563eb;}
  .arm-pf{color:#7e22ce;}
  .flagpill{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 6px;
    border-radius:10px;background:var(--amber-light);color:var(--amber-dark);
    margin-left:6px;vertical-align:middle;}
  .kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:18px 0;}
  .kv .kvi{background:var(--gray-50);border:1px solid var(--gray-200);border-radius:10px;padding:14px 16px;}
  .kv .kvi .kvn{font-size:26px;font-weight:800;color:var(--gray-900);letter-spacing:-.02em;}
  .kv .kvi .kvl{font-size:12px;color:var(--gray-600);margin-top:2px;}
  td.num,th.num{text-align:right;}
  .ci{font-size:11px;color:var(--gray-600);font-weight:400;}
"""


def forest(rows, order, comp_by_cell, d0, d1, axname):
    """One row per cell: both label-free arms with CIs, GOOD_6 tick, published marks."""
    # Labels are two lines (dataset, then model) because a single line at this font
    # runs off the left edge on the longest cells, e.g. MATH-500 (K=10 traces).
    LBL, X0, X1, ROW, TOP = 200, 212, 860, 42, 16
    n = len(order)
    H = TOP + n * ROW + 42          # +42 so the axis title clears the tick row
    x = lambda v: _lin(v * 100, d0, d1, X0, X1)
    s = [_svg(900, H)]
    for g in range(int(d0), int(d1) + 1, 10):
        if not (d0 <= g <= d1):
            continue
        s.append(f'<line x1="{x(g/100):.1f}" y1="{TOP}" x2="{x(g/100):.1f}" '
                 f'y2="{TOP+n*ROW}" class="rf-grid"/>')
        s.append(f'<text x="{x(g/100):.1f}" y="{TOP+n*ROW+16}" class="rf-tick" '
                 f'text-anchor="middle">{g}</text>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-8}" class="rf-axname" '
             f'text-anchor="middle">{axname}</text>')

    for i, ck in enumerate(order):
        r = rows[ck]
        cy = TOP + i * ROW + ROW / 2
        top = r["dataset"] + (" " + ("▿" if r["flag"] == "FLOOR" else "†")
                              if r["flag"] else "")
        s.append(f'<text x="{LBL}" y="{cy-1:.1f}" class="rf-rowlbl" '
                 f'text-anchor="end">{esc(top)}</text>')
        s.append(f'<text x="{LBL}" y="{cy+11:.1f}" class="rf-rowsub" '
                 f'text-anchor="end">{esc(r["model"])}</text>')

        # published marks first, so our dots sit on top of them
        seen = set()
        for c in sorted(comp_by_cell.get(ck, []), key=lambda z: -z["auroc"]):
            key = round(c["auroc"], 4)
            if key in seen:
                continue
            seen.add(key)
            ax = x(c["auroc"])
            sup = c["supervision"] == "supervised"
            tip = (f"{esc(c['method'])}: {c['auroc']*100:.1f} "
                   f"({'supervised' if sup else 'unsupervised'}, "
                   f"{esc(c['passes'])} pass, {esc(c['access'])})")
            if sup:
                s.append(f'<path d="M {ax:.1f} {cy-6:.1f} L {ax+6:.1f} {cy+5:.1f} '
                         f'L {ax-6:.1f} {cy+5:.1f} Z" class="rf-sup">'
                         f'<title>{tip}</title></path>')
            else:
                s.append(f'<path d="M {ax:.1f} {cy-6.5:.1f} L {ax+6.5:.1f} {cy:.1f} '
                         f'L {ax:.1f} {cy+6.5:.1f} L {ax-6.5:.1f} {cy:.1f} Z" '
                         f'class="rf-anchor"><title>{tip}</title></path>')

        s.append(f'<line x1="{x(r["g6"]):.1f}" y1="{cy-9:.1f}" '
                 f'x2="{x(r["g6"]):.1f}" y2="{cy+9:.1f}" class="rf-g6">'
                 f'<title>GOOD_6 (hand-picked subset): {r["g6"]*100:.1f}</title></line>')

        yb, yt = cy + 7, cy - 7
        s.append(f'<line x1="{x(r["pf_lo"]):.1f}" y1="{yb:.1f}" '
                 f'x2="{x(r["pf_hi"]):.1f}" y2="{yb:.1f}" class="rf-ci2"/>')
        s.append(f'<rect x="{x(r["pf"])-5:.1f}" y="{yb-5:.1f}" width="10" height="10" '
                 f'class="rf-ours2"><title>DUFS parameter-free + L-SML: '
                 f'{r["pf"]*100:.1f} [{r["pf_lo"]*100:.1f}, {r["pf_hi"]*100:.1f}], '
                 f'{r["pf_size"]} of {r["pool"]} views kept</title></rect>')
        s.append(f'<line x1="{x(r["up_lo"]):.1f}" y1="{yt:.1f}" '
                 f'x2="{x(r["up_hi"]):.1f}" y2="{yt:.1f}" class="rf-ci"/>')
        s.append(f'<circle cx="{x(r["up"]):.1f}" cy="{yt:.1f}" r="5.5" class="rf-ours">'
                 f'<title>U-PCR + sign(rho): {r["up"]*100:.1f} '
                 f'[{r["up_lo"]*100:.1f}, {r["up_hi"]*100:.1f}], '
                 f'{r["up_kept"]} of {r["pool"]} views kept</title></circle>')
    s.append("</svg>")
    return "".join(s)


def bars_fig(stats):
    """Delta vs each cost-class bar, both arms, diverging from zero."""
    X0, X1, ROW, TOP = 250, 850, 54, 16
    d0, d1 = -14.0, 14.0
    n = len(stats)
    H = TOP + n * ROW + 42          # +42 so the axis title clears the tick row
    x = lambda v: _lin(v, d0, d1, X0, X1)
    s = [_svg(900, H)]
    for g in range(-12, 13, 4):
        s.append(f'<line x1="{x(g):.1f}" y1="{TOP}" x2="{x(g):.1f}" '
                 f'y2="{TOP+n*ROW}" class="rf-grid"/>')
        s.append(f'<text x="{x(g):.1f}" y="{TOP+n*ROW+16}" class="rf-tick" '
                 f'text-anchor="middle">{g:+d}</text>')
    s.append(f'<line x1="{x(0):.1f}" y1="{TOP}" x2="{x(0):.1f}" '
             f'y2="{TOP+n*ROW}" class="rf-zero"/>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-8}" class="rf-axname" text-anchor="middle">'
             f'mean AUROC difference vs the bar (pp) &#183; positive = ours ahead</text>')
    for i, (tag, desc, ncov, per) in enumerate(stats):
        cy = TOP + i * ROW
        s.append(f'<text x="240" y="{cy+20:.1f}" class="rf-rowlbl" text-anchor="end">'
                 f'Bar {tag}</text>')
        s.append(f'<text x="240" y="{cy+34:.1f}" class="rf-rowsub" text-anchor="end">'
                 f'{ncov} cells</text>')
        for j, (arm, cls) in enumerate((("up", "rf-bar-ours"), ("pf", "rf-bar-pf"))):
            v = per[arm]["mean"] * 100
            by = cy + 10 + j * 17
            x0, x1 = min(x(0), x(v)), max(x(0), x(v))
            s.append(f'<rect x="{x0:.1f}" y="{by:.1f}" width="{max(x1-x0,1):.1f}" '
                     f'height="13" class="{cls}"><title>'
                     f'{"U-PCR + sign(rho)" if arm=="up" else "DUFS parameter-free + L-SML"}: '
                     f'{v:+.2f}pp mean, {per[arm]["w"]}W/{per[arm]["l"]}L, '
                     f'p={per[arm]["p"]:.3f}</title></rect>')
            tx = x1 + 6 if v >= 0 else x0 - 6
            anc = "start" if v >= 0 else "end"
            s.append(f'<text x="{tx:.1f}" y="{by+10:.1f}" class="rf-barlbl" '
                     f'text-anchor="{anc}">{v:+.1f}pp, p={per[arm]["p"]:.3f}</text>')
    s.append("</svg>")
    return "".join(s)


# ── page ──────────────────────────────────────────────────────────────────────
def macro_row(rows, cells, key):
    return st.mean(rows[c][key] for c in cells)


def main():
    rows, comp, g6macro = build_rows()

    comp_by_cell = {}
    for r in comp:
        comp_by_cell.setdefault(r["cell"], []).append(r)

    inband = [c for c in INSCOPE if not rows[c]["flag"]]
    arms = [("up", "U-PCR + sign(rho)", "arm-up"),
            ("pf", "DUFS parameter-free + L-SML", "arm-pf"),
            ("g6", "GOOD_6 (hand-picked 6-view subset)", "")]

    P = []
    A = P.append

    A('<div class="header-hero"><div class="header-content">')
    A('<h1>Label-free selection, scored on the in-scope grid</h1>')
    A('<div class="subtitle">The QA-evaluation desk and the published-benchmark desk '
      'on one page, rescored for the two arms that need nothing hand-picked: '
      'DUFS parameter-free + L-SML, and U-PCR with sign(rho) orientation.</div>')
    A('<div class="meta-pills">'
      '<span class="meta-pill">Generated by scripts/labelfree_standing_report.py</span>'
      f'<span class="meta-pill">{len(INSCOPE)} in-scope cells &#183; {len(QA_CELLS)} QA + {len(MATH_CELLS)} math</span>'
      '<span class="meta-pill">inside_coqa_llama7b rejected (Step 216)</span>'
      '<span class="meta-pill">RAG and GPQA excluded (Step 191)</span>'
      f'<span class="meta-pill">GOOD_6 validity check {g6macro:.4f}, expected {GOOD6_EXPECTED}</span>'
      '</div></div></div>')
    A('<div class="page">')

    # ---- scope + what changed --------------------------------------------
    A('<div class="section-card">')
    A('<span class="section-badge badge-finding">Scope and what this replaces</span>')
    A('<h2>What is on this page</h2>')
    A('<p>The two source pages both reported <span class="mono">L-SML GOOD_5</span>, a '
      'five-view subset picked with answer keys off a 16-view pool, over a cell roster '
      'that still included RAG and GPQA. This page rescores the same two questions for '
      'the current pipeline: which cells the method works on, and where it stands '
      'against published detectors, using only arms that never see a label.</p>')
    A('<div class="grid2">')
    A('<div><h3>The two arms</h3><ul>'
      '<li><b class="arm arm-up">U-PCR + sign(rho)</b> &mdash; each view\'s polarity comes '
      'from the sign of its estimated covariance with the latent target, and U-PCR\'s own '
      'exclusion step decides which views survive. No subset is supplied.</li>'
      '<li><b class="arm arm-pf">DUFS parameter-free + L-SML</b> &mdash; per-feature gates '
      'trained on the gated-Laplacian objective in its ratio form (DUFS Eq. 7), so there '
      'is no sparsity strength to tune, then continuous L-SML over the surviving views.</li>'
      '</ul><p>Both carry exactly one hand-set prior: the global anchor bit that says '
      'which direction means correct. Dropping it inverts every cell, and it is not '
      'recoverable from the covariance, since flipping every view leaves the covariance '
      'bit-identical.</p></div>')
    A(f'<div><h3>Scope</h3><p>{len(INSCOPE)} cells: {len(QA_CELLS)} short-answer QA and '
      f'{len(MATH_CELLS)} math reasoning. '
      'GPQA is out because every view sits at chance there (0.51 to 0.55, nothing to '
      'orient), and RAG is out because the signal is confined to one sub-dataset and is '
      'bottlenecked by view sign rather than pool size. Neither contributes a cell, a '
      'macro, or a win to anything below. The one HotpotQA cell that does appear is a '
      'plain multi-hop QA trace on Mistral-7B, not part of the excluded retrieval '
      'battery; it is kept because the canonical roster keeps it, and it is an honest '
      'loss.</p>'
      '<p>Every number is recomputed here through the canonical scoring path and gated '
      'against its recorded value before the page is written; a drift above 0.0005 AUROC '
      'aborts the build.</p></div>')
    A('</div>')

    A('<div class="kv">')
    for key, lab, _ in arms[:2]:
        A(f'<div class="kvi"><div class="kvn">{macro_row(rows, INSCOPE, key)*100:.1f}%</div>'
          f'<div class="kvl">{esc(lab)} &#183; macro over {len(INSCOPE)} cells</div></div>')
    A(f'<div class="kvi"><div class="kvn">{macro_row(rows, INSCOPE, "g6")*100:.1f}%</div>'
      '<div class="kvl">GOOD_6, the hand-picked bar to clear</div></div>')
    d = paired(rows, INSCOPE, "up", "g6")
    A(f'<div class="kvi"><div class="kvn">{d["mean"]*100:+.2f}pp</div>'
      f'<div class="kvl">GOOD_6 over U-PCR + sign(rho), p={d["p"]:.3f}</div></div>')
    A('</div>')
    A('</div>')

    # ---- headline table ---------------------------------------------------
    A('<div class="section-card">')
    A('<span class="section-badge badge-completed">Where the two arms land</span>')
    A('<h2>Macro standing</h2>')
    A('<table><tr><th>Method</th><th>Prior it needs</th><th class="num">Macro (25)</th>'
      '<th class="num">QA (10)</th><th class="num">Math (15)</th>'
      '<th class="num">In-band only (19)</th></tr>')
    priors = {"up": "global anchor bit only", "pf": "global anchor bit only",
              "g6": "hand-picked 6-view subset"}
    for key, lab, cls in arms:
        bold = ("<b>", "</b>") if key != "g6" else ("", "")
        A(f'<tr><td class="{cls}">{bold[0]}{esc(lab)}{bold[1]}</td>'
          f'<td>{priors[key]}</td>'
          f'<td class="num">{bold[0]}{macro_row(rows, INSCOPE, key)*100:.2f}{bold[1]}</td>'
          f'<td class="num">{macro_row(rows, QA_CELLS, key)*100:.2f}</td>'
          f'<td class="num">{macro_row(rows, MATH_CELLS, key)*100:.2f}</td>'
          f'<td class="num">{macro_row(rows, inband, key)*100:.2f}</td></tr>')
    A('</table>')

    A('<h3>Paired, cell by cell</h3>')
    A(f'<table><tr><th>Contrast</th><th>Effect over the {len(INSCOPE)} cells</th></tr>')
    for a, b, lab in (("pf", "up", "U-PCR + sign(rho) &minus; DUFS parameter-free"),
                      ("pf", "g6", "GOOD_6 &minus; DUFS parameter-free"),
                      ("up", "g6", "GOOD_6 &minus; U-PCR + sign(rho)")):
        A(f'<tr><td>{lab}</td><td>{fmt_pair(paired(rows, INSCOPE, a, b))}</td></tr>')
    A('</table>')
    A('<div class="takeaway-box"><b>Neither gap to the hand-picked subset is '
      'significant.</b> The claim is not that a label-free selector wins. It is that '
      'both reach the hand-curated bar without ever seeing an answer key, which is the '
      'thing GOOD_6 cannot say about itself. The two arms are also statistically '
      'indistinguishable from each other.</div>')
    A('</div>')

    # ---- QA figure + table ------------------------------------------------
    A('<div class="section-card">')
    A('<span class="section-badge badge-completed">The QA desk, rescored</span>')
    A('<h2>Short-form QA, ten cells</h2>')
    qa_order = sorted(QA_CELLS, key=lambda c: -rows[c]["up"])
    A(_fig("Short-answer QA: both label-free arms against every published same-model number",
           "One row per cell, sorted by U-PCR + sign(rho). Hover any mark for its value, "
           "cost class and source.",
           FOREST_LEGEND,
           forest(rows, qa_order, comp_by_cell, 40, 100, "AUROC (%) · one QA cell per row"),
           "&#9663; FLOOR (positive rate &lt; 0.20) and &#8224; CEILING (&gt; 0.85) mark "
           "cells where the label distribution makes the AUROC a noisy estimate of the "
           "right quantity. They are scored and shown, and excluded from the in-band "
           "macro column only. Published marks are same-model numbers verified against "
           "each paper's own table; a supervised triangle is a ceiling, not a peer."))

    A('<h3>Per cell</h3>')
    A('<table><tr><th>Dataset</th><th>Model</th><th class="num">n</th>'
      '<th class="num">pos rate</th><th class="num">U-PCR + sign(rho)</th>'
      '<th class="num">DUFS param-free</th><th class="num">GOOD_6</th>'
      '<th>Strongest published, unsupervised single-pass (Bar B)</th></tr>')
    barB = bar_best(comp, BARS[1][2])
    for ck in qa_order:
        r = rows[ck]
        pub = barB.get(ck)
        pubs = (f'{pub["auroc"]*100:.1f} &#183; {esc(pub["method"])}' if pub
                else '<span class="neutral">no published number in this class</span>')
        fl = f'<span class="flagpill">{r["flag"]}</span>' if r["flag"] else ""
        A(f'<tr><td>{esc(r["dataset"])}{fl}</td><td>{esc(r["model"])}</td>'
          f'<td class="num">{r["n"]}</td><td class="num">{r["pos"]:.3f}</td>'
          f'<td class="num arm arm-up">{r["up"]*100:.1f} '
          f'<span class="ci">[{r["up_lo"]*100:.1f}, {r["up_hi"]*100:.1f}]</span></td>'
          f'<td class="num arm arm-pf">{r["pf"]*100:.1f} '
          f'<span class="ci">[{r["pf_lo"]*100:.1f}, {r["pf_hi"]*100:.1f}]</span></td>'
          f'<td class="num">{r["g6"]*100:.1f}</td><td>{pubs}</td></tr>')
    A('</table>')
    qa_gap = macro_row(rows, QA_CELLS, "g6") - macro_row(rows, QA_CELLS, "up")
    math_gap = macro_row(rows, MATH_CELLS, "g6") - macro_row(rows, MATH_CELLS, "up")
    # Largest single contributor, measured rather than named — the cell that used to
    # carry this paragraph (CoQA) is no longer on the roster.
    worst = max(QA_CELLS, key=lambda c: rows[c]["g6"] - rows[c]["up"])
    w = rows[worst]
    qa_no_worst = [c for c in QA_CELLS if c != worst]
    qa_gap_nw = macro_row(rows, qa_no_worst, "g6") - macro_row(rows, qa_no_worst, "up")
    A('<div class="warn-box"><b>The QA desk changed shape in Step 216, and the previous '
      'reading of it was measuring two broken cells.</b> '
      f'GOOD_6 now leads QA by {qa_gap*100:+.2f}pp against {math_gap*100:+.2f}pp on math. '
      'The two cells that used to dominate this paragraph were both <b>base checkpoints '
      'generating malformed traces</b>, not method failures: '
      '<code>inside_coqa_llama7b</code> is <b>REJECTED</b> (a chat template applied to a '
      'LLaMA-7B base checkpoint; 45.1% of its answer spans are <code>[/INST]</code> '
      'echoes, fabricated <code>Question:</code> turns or empty, and its positive rate is '
      '0.002 on the broken rows against 0.239 on the usable ones &mdash; a degenerate '
      'sub-population, so it is out of every macro until it is re-generated), and '
      '<code>seiclr_triviaqa_opt30b</code> is <b>REPAIRED</b> by cropping its features to '
      'the answer span its grader already reads, moving it from 0.5614 to '
      f'{rows["seiclr_triviaqa_opt30b"]["pf"]*100:.2f} / '
      f'{rows["seiclr_triviaqa_opt30b"]["up"]*100:.2f} on the two arms. '
      f'On the nine surviving QA cells the largest single contributor is now '
      f'<b>{esc(w["dataset"])} / {esc(w["model"])}</b> at '
      f'{(w["g6"] - w["up"])*100:+.2f}pp; dropping it moves the QA gap to '
      f'{qa_gap_nw*100:+.2f}pp over the remaining eight. There is no longer one cell '
      'carrying this desk.</div>')

    # The Step-155 gate, re-run under the label-free arms rather than GOOD_5.
    A('<h3>The Step-155 QA decision gate, re-run</h3>')
    A('<p>The gate that opened the QA extension was: at least 3 of the 4 short-QA '
      'datasets reach AUROC 65 or better. It was written for the hand-picked subset. '
      'Re-run against the better of the two label-free arms per cell:</p>')
    GATE = ["se_squad_v2_llama8b", "truthfulqa_llama8b", "sciq_llama8b",
            "se_nq_open_llama8b"]
    A('<table><tr><th>Dataset</th><th class="num">Best label-free arm</th>'
      '<th>Which arm</th><th>vs the 65 bar</th></tr>')
    npass = 0
    for ck in GATE:
        r = rows[ck]
        best, which = ((r["up"], "U-PCR + sign(rho)") if r["up"] >= r["pf"]
                       else (r["pf"], "DUFS parameter-free"))
        ok_ = best >= 0.65
        npass += ok_
        A(f'<tr><td>{esc(r["dataset"])}</td><td class="num">{best*100:.1f}</td>'
          f'<td>{which}</td><td class="{"win" if ok_ else "loss"}">'
          f'{"&ge; 65 &#10003;" if ok_ else "&lt; 65 &#10007;"}</td></tr>')
    A('</table>')
    A(f'<div class="takeaway-box"><b>Gate status: {npass} of 4 clear the bar</b> without '
      'a hand-picked subset anywhere in the path, and every caveat that was attached the '
      'first time still applies. <b>This gate is not a clean sweep of short-form QA.</b> '
      'It was written over four datasets, and <b>CoQA &mdash; the top-priority dataset of '
      'the QA extension &mdash; is deliberately not one of them</b>. That omission now '
      'matters more, not less: the CoQA cell is REJECTED as of Step 216 for a generation '
      'defect, so this page carries no CoQA evidence at all in either direction. Reading '
      'the gate as coverage of short-form QA overstates it by exactly that dataset. Of '
      'the four that are in the gate, TruthfulQA is floor-flagged at a '
      f'{rows["truthfulqa_llama8b"]["pos"]*100:.1f}% positive rate, and SciQ carries both '
      f'a ceiling caveat at {rows["sciq_llama8b"]["task_acc"]:.1%} task accuracy and only '
      f'{rows["sciq_llama8b"]["n"]} scored rows, so its interval is wide.</div>')
    A('</div>')

    # ---- math figure + table ----------------------------------------------
    A('<div class="section-card">')
    A('<span class="section-badge badge-completed">The reasoning desk, rescored</span>')
    A('<h2>Math reasoning, fifteen cells</h2>')
    math_order = sorted(MATH_CELLS, key=lambda c: -rows[c]["up"])
    A(_fig("GSM8K and MATH-500: both label-free arms against every published same-model number",
           "One row per cell, sorted by U-PCR + sign(rho).",
           FOREST_LEGEND,
           forest(rows, math_order, comp_by_cell, 40, 100,
                  "AUROC (%) · one math cell per row"),
           "The six cells with no published mark are the cluster-era MATH-500 and "
           "K=10 trace cells: no paper reports a same-model number on them, so they "
           "carry our numbers only. On math the gap to the hand-picked subset closes and "
           f"reverses for one arm: U-PCR + sign(rho) is "
           f"{(macro_row(rows, MATH_CELLS, 'up') - macro_row(rows, MATH_CELLS, 'g6'))*100:+.2f}pp "
           "against GOOD_6 here and DUFS parameter-free is "
           f"{(macro_row(rows, MATH_CELLS, 'pf') - macro_row(rows, MATH_CELLS, 'g6'))*100:+.2f}pp, "
           "both well inside noise."))

    A('<h3>Per cell</h3>')
    A('<table><tr><th>Dataset</th><th>Model</th><th class="num">n</th>'
      '<th class="num">pos rate</th><th class="num">U-PCR + sign(rho)</th>'
      '<th class="num">DUFS param-free</th><th class="num">GOOD_6</th>'
      '<th>Strongest published, unsupervised single-pass (Bar B)</th></tr>')
    for ck in math_order:
        r = rows[ck]
        pub = barB.get(ck)
        pubs = (f'{pub["auroc"]*100:.1f} &#183; {esc(pub["method"])}' if pub
                else '<span class="neutral">no published number in this class</span>')
        fl = f'<span class="flagpill">{r["flag"]}</span>' if r["flag"] else ""
        A(f'<tr><td>{esc(r["dataset"])}{fl}</td><td>{esc(r["model"])}</td>'
          f'<td class="num">{r["n"]}</td><td class="num">{r["pos"]:.3f}</td>'
          f'<td class="num arm arm-up">{r["up"]*100:.1f} '
          f'<span class="ci">[{r["up_lo"]*100:.1f}, {r["up_hi"]*100:.1f}]</span></td>'
          f'<td class="num arm arm-pf">{r["pf"]*100:.1f} '
          f'<span class="ci">[{r["pf_lo"]*100:.1f}, {r["pf_hi"]*100:.1f}]</span></td>'
          f'<td class="num">{r["g6"]*100:.1f}</td><td>{pubs}</td></tr>')
    A('</table>')
    A('</div>')

    # ---- external standing -------------------------------------------------
    A('<div class="section-card">')
    A('<span class="section-badge badge-completed">Against the published roster</span>')
    A('<h2>External standing, by cost class</h2>')
    A('<p>A published number is only comparable to ours if it was bought with the same '
      'budget, so the roster is binned by what each method is allowed to use rather than '
      'pooled into one scoreboard. Every entry is a same-model number verified against '
      'the paper\'s own table. Within a bar, each cell contributes its single strongest '
      'published number, which is the hardest version of the comparison.</p>')

    stats = []
    for tag, desc, pred in BARS:
        B = bar_best(comp, pred)
        cov = [c for c in INSCOPE if c in B]
        per = {}
        for key in ("up", "pf", "g6"):
            dl = [rows[c][key] - B[c]["auroc"] for c in cov]
            per[key] = dict(mean=st.mean(dl), median=st.median(dl),
                            w=sum(v > 0 for v in dl), l=sum(v < 0 for v in dl),
                            p=wilcoxon(dl).pvalue)
        stats.append((tag, desc, len(cov), per))

    A(_fig("Where each arm sits against each cost class",
           "Mean AUROC difference against the strongest published number in that class, "
           "per cell, over the cells that have one.",
           '<span class="rf-li"><svg width="16" height="12">'
           '<rect x="0" y="1" width="16" height="10" class="rf-bar-ours"/></svg> '
           'U-PCR + sign(rho)</span>'
           '<span class="rf-li"><svg width="16" height="12">'
           '<rect x="0" y="1" width="16" height="10" class="rf-bar-pf"/></svg> '
           'DUFS parameter-free + L-SML</span>',
           bars_fig(stats),
           "Bar A is our exact cost class, grey-box and single-pass, and it covers only "
           "5 cells, so its p-value cannot reach significance whatever the effect. Bar B "
           "keeps the single-pass unsupervised budget but lets the competitor read model "
           "internals we never touch, so it is a harder comparison than Bar A on access "
           "and an easier one on coverage; it is where the significant result lives, and "
           "it should be quoted as Bar B rather than as our own class. Bar C adds methods "
           "that pay 10 forward passes for their score, Bar D adds supervised probes "
           "trained on labelled data, and we are behind on both by construction rather "
           "than by accident."))

    A('<table><tr><th>Bar</th><th>What it contains</th><th class="num">Cells</th>'
      '<th>U-PCR + sign(rho)</th><th>DUFS parameter-free</th></tr>')
    for tag, desc, ncov, per in stats:
        def cell(pp):
            cls = "win" if pp["mean"] > 0 else "loss"
            return (f'<td class="{cls}">{pp["mean"]*100:+.2f}pp mean, '
                    f'{pp["w"]}W/{pp["l"]}L, p={pp["p"]:.3f}</td>')
        A(f'<tr><td><b>{tag}</b></td><td>{esc(desc)}</td>'
          f'<td class="num">{ncov}</td>{cell(per["up"])}{cell(per["pf"])}</tr>')
    A('</table>')
    S = {t: (n, p) for t, _, n, p in stats}
    A('<div class="takeaway-box"><b>The one-sentence version, said in the right order.</b> '
      'Our exact class is Bar A, grey-box and single-pass, and there we are '
      f'{S["A"][1]["up"]["mean"]*100:+.2f}pp (U-PCR) and {S["A"][1]["pf"]["mean"]*100:+.2f}pp '
      f'(DUFS) over {S["A"][0]} cells, which is far too few to be significant '
      f'(p={S["A"][1]["up"]["p"]:.3f}). Widen to every unsupervised single-pass method, '
      'including ones that read internals we never touch, and both arms are around '
      f'{S["B"][1]["up"]["mean"]*100:.1f}pp ahead on {S["B"][0]} cells at '
      f'p={S["B"][1]["up"]["p"]:.3f}. That is the strongest defensible claim on this '
      'page, and it is a claim about Bar B, not about our own narrower class. Against '
      'methods that sample ten times we are level. Against supervised probes we are '
      f'behind by {abs(S["D"][1]["up"]["mean"])*100:.1f} to '
      f'{abs(S["D"][1]["pf"]["mean"])*100:.1f}pp.</div>')

    A('<h3>Per cell, against the strongest unsupervised single-pass number (Bar B)</h3>')
    A('<table><tr><th>Dataset</th><th>Model</th>'
      '<th class="num">U-PCR + sign(rho)</th><th class="num">DUFS param-free</th>'
      '<th>Published</th><th class="num">Best of ours &minus; them</th></tr>')
    for ck in sorted((c for c in INSCOPE if c in barB),
                     key=lambda c: -(max(rows[c]["up"], rows[c]["pf"]) - barB[c]["auroc"])):
        r, pub = rows[ck], barB[ck]
        dl = max(r["up"], r["pf"]) - pub["auroc"]
        cls = "win" if dl > 0 else "loss"
        A(f'<tr><td>{esc(r["dataset"])}</td><td>{esc(r["model"])}</td>'
          f'<td class="num">{r["up"]*100:.1f}</td><td class="num">{r["pf"]*100:.1f}</td>'
          f'<td>{pub["auroc"]*100:.1f} &#183; {esc(pub["method"])} '
          f'<span class="ci">({esc(pub["access"])})</span></td>'
          f'<td class="num {cls}">{dl*100:+.2f}pp</td></tr>')
    A('</table>')
    A('</div>')

    # ---- appendix ----------------------------------------------------------
    A('<div class="section-card">')
    A('<span class="section-badge badge-flagged">Appendix</span>')
    A('<h2>Supporting detail</h2>')

    A('<h3>What each arm actually selected</h3>')
    A('<table><tr><th>Dataset</th><th>Model</th>'
      '<th class="num">Pool</th><th class="num">DUFS param-free kept</th>'
      '<th class="num">U-PCR kept</th></tr>')
    for ck in INSCOPE:
        r = rows[ck]
        A(f'<tr><td>{esc(r["dataset"])}</td><td>{esc(r["model"])}</td>'
          f'<td class="num">{r["pool"]}</td><td class="num">{r["pf_size"]}</td>'
          f'<td class="num">{r["up_kept"]}</td></tr>')
    A('</table>')
    pf_mean = st.mean(rows[c]["pf_size"] for c in INSCOPE)
    up_mean = st.mean(rows[c]["up_kept"] for c in INSCOPE)
    A(f'<p>Averaged over the grid, the DUFS gates keep {pf_mean:.1f} views and U-PCR\'s '
      f'exclusion step keeps {up_mean:.1f}, out of a pool that is 30 views where every '
      'field was captured and slightly smaller where a cell is missing one. Neither '
      'number was chosen; both fall out of the respective objective.</p>')

    A('<h3>The trivial single-number floors, on our own traces</h3>')
    A('<p>Published numbers come from other people\'s inference budgets. The tightest '
      'available control is the trivial baseline computed on the exact same traces: '
      'mean sequence log-probability, and perplexity. This is a floor check, not a '
      'headline comparison.</p>')
    A('<table><tr><th>Dataset</th><th>Model</th><th class="num">U-PCR + sign(rho)</th>'
      '<th class="num">DUFS param-free</th><th class="num">seq-logprob</th>'
      '<th class="num">perplexity</th><th class="num">Best of ours &minus; seq-logprob</th></tr>')
    have = [c for c in INSCOPE if rows[c]["seqlp"] is not None]
    for ck in sorted(have, key=lambda c: -(max(rows[c]["up"], rows[c]["pf"]) - rows[c]["seqlp"])):
        r = rows[ck]
        dl = max(r["up"], r["pf"]) - r["seqlp"]
        cls = "win" if dl > 0.005 else ("loss" if dl < -0.005 else "neutral")
        A(f'<tr><td>{esc(r["dataset"])}</td><td>{esc(r["model"])}</td>'
          f'<td class="num">{r["up"]*100:.1f}</td><td class="num">{r["pf"]*100:.1f}</td>'
          f'<td class="num">{r["seqlp"]*100:.1f}</td>'
          f'<td class="num">{r["ppl"]*100:.1f}</td>'
          f'<td class="num {cls}">{dl*100:+.2f}pp</td></tr>')
    A('</table>')
    dl = [max(rows[c]["up"], rows[c]["pf"]) - rows[c]["seqlp"] for c in have]
    A(f'<div class="warn-box"><b>Against seq-logprob: {sum(v > 0.005 for v in dl)} wins, '
      f'{sum(abs(v) <= 0.005 for v in dl)} ties, {sum(v < -0.005 for v in dl)} losses '
      f'over {len(have)} cells ({st.mean(dl)*100:+.2f}pp mean, '
      f'p={wilcoxon(dl).pvalue:.3f}).</b> The six cluster-era MATH-500 and K=10 trace '
      'cells have no scored seq-logprob row and are excluded. This baseline is close '
      'and it is meant to be reported that way: it is the honest floor for any '
      'single-pass grey-box detector, and beating the published roster while only '
      'edging the trivial baseline is a real limitation of the current numbers.</div>')

    A('<h3>Positive rate versus task accuracy</h3>')
    starred = [c for c in INSCOPE if rows[c]["task_acc"] is not None
               and abs(rows[c]["task_acc"] - rows[c]["pos"]) > 0.05]
    A('<p>The flag on each cell comes from the positive rate of the labels the AUROC is '
      'actually computed over, not from the task accuracy across all problems. On '
      f'{len(starred)} cells the two differ by more than 5pp, because only part of that '
      'cell\'s traces carry every field the pool needs, so the scored subset is not a '
      'uniform sample. Both numbers are shown rather than picking one.</p>')
    A('<table><tr><th>Dataset</th><th>Model</th><th class="num">Scored rows</th>'
      '<th class="num">Positives</th><th class="num">Positive rate</th>'
      '<th class="num">Task accuracy</th><th>Flag</th></tr>')
    for ck in INSCOPE:
        r = rows[ck]
        ta = f'{r["task_acc"]:.3f}' if r["task_acc"] is not None else "&mdash;"
        diff = (r["task_acc"] is not None and abs(r["task_acc"] - r["pos"]) > 0.05)
        A(f'<tr><td>{esc(r["dataset"])}</td><td>{esc(r["model"])}</td>'
          f'<td class="num">{r["n"]}</td><td class="num">{r["n_pos"]}</td>'
          f'<td class="num">{r["pos"]:.3f}</td>'
          f'<td class="num">{ta}{" *" if diff else ""}</td>'
          f'<td>{r["flag"] or "&mdash;"}</td></tr>')
    A('</table>')
    A('<p class="rf-fnote">* the two differ by more than 5pp because only part of that '
      'cell\'s traces carry every captured field, so the scored subset is not a uniform '
      'sample of the cell.</p>')

    A('<h3>Papers behind the published marks</h3>')
    bypaper = {}
    for r in comp:
        bypaper.setdefault(r["paper_slug"], set()).add(r["method"])
    # The CSV carries the same method under two strings on some rows, e.g. "HCPD" and
    # "HCPD (arXiv 2606.12900)". Keep the cited form, or the list names a method that no
    # mark on this page is labelled with. Match ONLY an arXiv-id suffix: a parenthetical
    # in general distinguishes real methods, e.g. "Semantic Entropy" and "Semantic
    # Entropy (SE-ICLR'23)" are two different published numbers in the same paper.
    for slug, names in bypaper.items():
        bypaper[slug] = {n for n in names
                         if not any(o != n and o.startswith(n + " (arXiv") for o in names)}
    A('<ul>')
    for slug in sorted(bypaper):
        A(f'<li><span class="mono">{esc(slug)}</span> &mdash; '
          f'{esc(", ".join(sorted(bypaper[slug])))}</li>')
    A('</ul>')
    A('<p class="rf-fnote">Distinct published numbers can coincide to the plotted '
      'precision and then render as a single mark: Logits-min and Logits-mean are both '
      '0.61 on the HotpotQA cell, so only one diamond is drawn there.</p>')

    A('<h3>How a cell gets on this page</h3>')
    A('<p>For each competitor paper we ran <em>our own</em> inference on that paper\'s '
      'exact dataset, model, N and decoding configuration, so the accuracy and label '
      'distribution match their setup, then scored our own detector offline on that '
      'trace. We never re-implement a competitor. Every cell passed the gate ladder of '
      'local smoke test, then an N=30 pilot, then full N. Cells whose labels are invalid '
      'rather than merely skewed, meaning truncation leakage or a single-class label set, '
      'are rejected outright and appear nowhere here, because there the AUROC would be a '
      'clean estimate of the wrong quantity.</p>')
    A('</div>')

    A('</div>')

    page = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            "<title>Label-free selection, scored on the in-scope grid</title>\n"
            "<style>\n" + BASE_CSS + FIG_CSS + EXTRA_CSS + "\n</style>\n</head>\n<body>\n"
            + "\n".join(P) + "\n</body>\n</html>\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"\nwrote {OUT}  ({len(page)/1024:.0f} KB)")
    return 0


BASE_CSS = """
  :root{--blue:#2563eb;--blue-light:#eff6ff;--blue-dark:#1e40af;--green:#10b981;--green-light:#ecfdf5;
  --green-dark:#065f46;--red:#ef4444;--amber:#f59e0b;--amber-light:#fffbeb;--amber-dark:#92400e;
  --gray-50:#f8fafc;--gray-100:#f1f5f9;--gray-200:#e2e8f0;--gray-600:#475569;--gray-700:#334155;
  --gray-800:#1e293b;--gray-900:#0f172a;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Inter',system-ui,-apple-system,sans-serif;font-size:15px;color:var(--gray-800);
  background:var(--gray-50);line-height:1.65;}
  .header-hero{background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);color:#fff;padding:56px 24px 44px;
  border-bottom:4px solid var(--blue);}
  .header-content{max-width:1120px;margin:0 auto;}
  .header-hero h1{font-size:32px;font-weight:800;letter-spacing:-.02em;margin-bottom:10px;}
  .header-hero .subtitle{font-size:16px;color:#cbd5e1;max-width:860px;}
  .meta-pills{display:flex;gap:12px;margin-top:20px;flex-wrap:wrap;}
  .meta-pill{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);padding:4px 12px;
  border-radius:20px;font-size:13px;font-weight:500;}
  .page{max-width:1120px;margin:0 auto;padding:40px 24px 64px;}
  .section-card{background:#fff;border:1px solid var(--gray-200);border-radius:14px;padding:36px;
  margin-bottom:36px;box-shadow:0 2px 6px rgba(0,0,0,.03);}
  .section-badge{display:inline-block;font-size:12px;font-weight:700;text-transform:uppercase;
  letter-spacing:.06em;padding:4px 12px;border-radius:20px;margin-bottom:12px;}
  .badge-completed{background:var(--green-light);color:var(--green-dark);}
  .badge-finding{background:#f3e8ff;color:#6b21a8;}
  .badge-flagged{background:var(--amber-light);color:var(--amber-dark);}
  h2{font-size:24px;font-weight:800;letter-spacing:-.015em;color:var(--gray-900);margin-bottom:12px;}
  h3{font-size:17px;font-weight:700;color:var(--gray-900);margin:24px 0 10px;}
  p{margin-bottom:14px;color:var(--gray-700);}
  ul{margin:0 0 14px 20px;color:var(--gray-700);}
  li{margin-bottom:6px;}
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
  th{background:var(--gray-100);color:var(--gray-800);font-weight:700;text-align:left;padding:11px 12px;
  border:1px solid var(--gray-200);}
  td{padding:10px 12px;border:1px solid var(--gray-200);color:var(--gray-700);}
  .win{color:#16a34a;font-weight:700;}
  .loss{color:#dc2626;font-weight:700;}
  .neutral{color:var(--gray-600);font-weight:600;}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:var(--gray-100);
  padding:2px 6px;border-radius:4px;color:var(--gray-800);}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin:20px 0;}
  @media(max-width:820px){.grid2{grid-template-columns:1fr;}}
  .rf-bar-pf{fill:#a855f7;}
"""


if __name__ == "__main__":
    sys.exit(main())
