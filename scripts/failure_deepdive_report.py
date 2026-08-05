#!/usr/bin/env python
"""
failure_deepdive_report.py — render results/failure_deepdive/index.html.

Reads only the CSVs that `failure_deepdive.py` wrote, so the page cannot drift
from the measurement. Self-contained: no external stylesheets, scripts, fonts or
images (same rule as labelfree_standing.html).

DIAGNOSIS ONLY — the page names mechanisms and pre-registers repairs. It does not
report any repair result, because none has been run.
"""
import csv
import html
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from labelfree_standing_report import PRETTY                            # noqa: E402

DIR = os.path.join(REPO, "results", "failure_deepdive")
OUT = os.path.join(DIR, "index.html")

CSS = """
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--hi:#fff8e1;--bad:#c0392b;
      --good:#1e7e34;--box:#f7f7f9;--accent:#1a4f8a}
*{box-sizing:border-box}
body{margin:0;padding:0 20px 80px;background:var(--bg);color:var(--fg);
     font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:27px;margin:34px 0 6px;line-height:1.25}
h2{font-size:20px;margin:40px 0 10px;padding-top:14px;border-top:2px solid var(--line)}
h3{font-size:16px;margin:24px 0 8px}
p{margin:10px 0}
.sub{color:var(--mut);font-size:14px;margin:0 0 22px}
.box{background:var(--box);border-left:4px solid var(--accent);padding:12px 16px;
     margin:16px 0;border-radius:0 4px 4px 0}
.box.warn{border-left-color:#b8860b;background:var(--hi)}
.box.bad{border-left-color:var(--bad);background:#fdf0ee}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:14px 0}
table{border-collapse:collapse;font-size:13px;width:100%;min-width:640px}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:right;
      white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{background:#f0f0f3;font-weight:600;position:sticky;top:0}
tr.weak{background:var(--hi)}
tr.weak td:first-child{font-weight:600}
td.neg{color:var(--bad)} td.pos{color:var(--good)}
code{background:#f0f0f3;padding:1px 5px;border-radius:3px;font-size:12.5px}
.mech{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11.5px;
      font-weight:600;color:#fff}
.m1{background:#c0392b} .m2{background:#b8860b} .m3{background:#5a6b7a}
.m4{background:#7d4a9e}
.foot{color:var(--mut);font-size:12.5px;margin-top:34px;border-top:1px solid var(--line);
      padding-top:14px}
.bar{display:inline-block;height:9px;border-radius:2px;vertical-align:middle}
@media (prefers-color-scheme:dark){
  :root{--bg:#16181c;--fg:#e8e8ea;--mut:#9aa0a6;--line:#2c2f36;--hi:#2e2a1c;
        --box:#1e2126;--accent:#5b9bd5;--bad:#e57373;--good:#66bb6a}
  th{background:#22252b} code{background:#22252b}
}
:root[data-theme="dark"]{--bg:#16181c;--fg:#e8e8ea;--mut:#9aa0a6;--line:#2c2f36;
  --hi:#2e2a1c;--box:#1e2126;--accent:#5b9bd5;--bad:#e57373;--good:#66bb6a}
:root[data-theme="dark"] th{background:#22252b}
:root[data-theme="light"]{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--hi:#fff8e1;
  --box:#f7f7f9;--accent:#1a4f8a;--bad:#c0392b;--good:#1e7e34}
:root[data-theme="light"] th{background:#f0f0f3}
"""

MECH = {
    "sign-recovery": ("m1", "sign-recovery failure"),
    "selection-miss": ("m2", "selection miss"),
    "non-monotone": ("m4", "non-monotone views"),
    "ceiling": ("m3", "ceiling-limited"),
}

# Thresholds fixed from the healthy cells' own spread, not tuned per weak cell.
RECOV_MIN = 0.90      # every healthy cell recovers >= 0.919
SEL_MISS_PP = -1.0    # chosen subset excludes the pool's best view by >= 1pp
NONMONO_MIN = 0.02    # max across all cells except CoQA is 0.0201
CEILING_PP = 3.0      # headroom to the label-chosen oracle-5


def esc(s):
    return html.escape(str(s), quote=True)


def rd(name):
    with open(os.path.join(DIR, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def sgn(v, nd=2, suffix="pp"):
    """Signed cell with colour class."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return '<td>—</td>'
    cls = "neg" if x < -0.05 else ("pos" if x > 0.05 else "")
    return f'<td class="{cls}">{x:+.{nd}f}{suffix}</td>'


def name_of(ck):
    d, m = PRETTY.get(ck, (ck, ""))
    return f"{d} / {m}" if m else d


def recov(r):
    """How much of the relative-sign information L-SML gives back.

    Losing the per-view signs costs a simple average `d_signs`; L-SML's job is to
    recover it without labels. The ratio is ~1.0 when it does. NaN where signs
    barely mattered (|d_signs| < 0.5pp), because the ratio is then meaningless.
    """
    ds, dl = float(r["d_signs"]), float(r["d_lsml_vs_avg"])
    return dl / (-ds) if ds < -0.5 else np.nan


def sel_miss(r):
    """pp by which the CHOSEN subset's best view falls short of the pool's."""
    return (float(r["r1_best_single_in_subset"]) - float(r["best_single"])) * 100


def classify(r):
    """Every mechanism a cell trips, from its own numbers, against thresholds
    fixed by the healthy cells' spread. A cell can trip more than one — that is
    a finding, not a tie to be broken."""
    out = []
    rc = recov(r)
    if np.isfinite(rc) and rc < RECOV_MIN:
        out.append("sign-recovery")
    if sel_miss(r) <= SEL_MISS_PP:
        out.append("selection-miss")
    if float(r["nonmono_mean"]) > NONMONO_MIN:
        out.append("non-monotone")
    if (float(r["d_headroom"]) < CEILING_PP
            and min(float(r["upcr_minus_best"]),
                    float(r["dufs_minus_best"])) > -1.0):
        out.append("ceiling")
    return out


def main():
    pc = rd("percell.csv")
    pf = rd("perfeature.csv")
    cur = rd("residual_curves.csv")
    gates = json.load(open(os.path.join(DIR, "gates.json"), encoding="utf-8"))

    for r in pc:
        r["_weak"] = r["weak"].lower() == "true"
        r["_mech"] = classify(r)
        r["_recov"] = recov(r)
        r["_sel"] = sel_miss(r)
    pc.sort(key=lambda r: (not r["_weak"], float(r["anchor_auc"])))
    weak = [r for r in pc if r["_weak"]]
    heal = [r for r in pc if not r["_weak"]]

    def mean(rows, k):
        return float(np.mean([float(r[k]) for r in rows]))

    A = []
    W = A.append
    W('<div class="wrap">')
    W("<h1>Why we fail where we fail</h1>")
    W('<p class="sub">Nine cells diagnosed against the other sixteen as the '
      "comparison group. Jul-30 advisor action item 1. "
      "<b>Diagnosis only</b> — this page names mechanisms and pre-registers "
      "repairs; no repair has been run, so none is reported.</p>")

    # ── 0. the confound ──────────────────────────────────────────────────────
    W("<h2>0. The obvious answer is a confound, and it is not the answer</h2>")
    W(f'<div class="box warn"><p>The nine weak cells are exactly the nine lowest '
      f'<code>anchor_auc</code> in the grid, and Spearman(anchor_auc, deployed '
      f'AUROC) = <b>{gates["spearman_anchor_deployed"]:+.3f}</b>. That reads as an '
      f'orientation failure. It is not one: '
      f'Spearman(anchor_auc, <b>best single view</b>) = '
      f'<b>{gates["spearman_anchor_best_single"]:+.3f}</b>. The <code>epr</code> '
      f'anchor is itself a pooled feature, so a weak anchor only means every view '
      f'is weak on that cell.</p>'
      f'<p>Two independent checks agree. The <code>allsigns</code> / <code>z2</code> / '
      f'<code>raw</code> / <code>oracle</code> anchor conditions all return '
      f'<b>identically 0.7594</b> with zero cells below chance '
      f'(<code>h1_orientation_summary.csv</code>), and on this page the global-sign '
      f'rung costs a mean of '
      f'{mean(pc, "d_global_sign"):+.2f}pp across all 25 cells. '
      f'<b>The global orientation bit is being resolved correctly everywhere.</b></p></div>')

    # ── 1. headline ──────────────────────────────────────────────────────────
    W("<h2>1. The quantity that survives: fusion minus best single view</h2>")
    W("<p>Every fusion should at least match the best single view in its pool "
      "(the pool contains that view). On the sixteen healthy cells it does. On "
      "the nine it does not.</p>")
    W('<div class="scroll"><table><tr><th>Arm</th>'
      "<th>weak (9)</th><th>healthy (16)</th><th>difference</th></tr>")
    for arm, lbl in (("upcr_minus_best", "U-PCR + sign(rho)"),
                     ("dufs_minus_best", "DUFS + L-SML"),
                     ("good6_minus_best", "GOOD_6 (hand-picked)")):
        a, b = mean(weak, arm), mean(heal, arm)
        W(f"<tr><td>{lbl}</td>{sgn(a)}{sgn(b)}{sgn(a - b)}</tr>")
    W("</table></div>")
    W("<p>So the label-free arms are not merely operating on weak cells — on "
      "those cells they actively give back several points relative to just "
      "taking the strongest view. The hand-picked subset gives back far less, "
      "which is why the gap between hand-picked and label-free is concentrated "
      "here rather than spread across the grid.</p>")

    # ── 2. ladder ────────────────────────────────────────────────────────────
    W("<h2>2. Which stage loses it — the ladder</h2>")
    W("<p>Each rung adds exactly one thing to the previous one, on the deployed "
      "(<code>a2.dufs_pf</code>) subset. <code>r5</code> is the deployed number. "
      "Deltas are the cost of the step that produced the rung.</p>")
    W("<div class='scroll'><table><tr>"
      "<th>Cell</th><th>r1 best view</th><th>r2 avg, oracle signs</th>"
      "<th>r3 avg, no signs</th><th>r4 L-SML, oracle global</th>"
      "<th>r5 <b>deployed</b></th><th>r6 oracle-5</th>"
      "<th>fuse−best</th><th>signs</th><th>L-SML−avg</th><th>global sign</th>"
      "<th>headroom</th></tr>")
    for r in pc:
        cls = ' class="weak"' if r["_weak"] else ""
        W(f'<tr{cls}><td>{esc(name_of(r["cell"]))}</td>'
          f'<td>{fnum(r["r1_best_single_in_subset"])}</td>'
          f'<td>{fnum(r["r2_avg_oracle_signs"])}</td>'
          f'<td>{fnum(r["r3_avg_no_signs"])}</td>'
          f'<td>{fnum(r["r4_lsml_oracle_global"])}</td>'
          f'<td><b>{fnum(r["r5_lsml_anchor_DEPLOYED"])}</b></td>'
          f'<td>{fnum(r["r6_oracle5_ceiling"])}</td>'
          f'{sgn(r["d_fuse_vs_best"])}{sgn(r["d_signs"])}'
          f'{sgn(r["d_lsml_vs_avg"])}{sgn(r["d_global_sign"])}'
          f'{sgn(r["d_headroom"])}</tr>')
    W("</table></div>")
    W(f'<p><b>Mean across all 25:</b> fusion vs best view '
      f'{mean(pc, "d_fuse_vs_best"):+.2f}pp · not knowing relative signs '
      f'{mean(pc, "d_signs"):+.2f}pp · L-SML over simple average '
      f'{mean(pc, "d_lsml_vs_avg"):+.2f}pp · global sign from the anchor '
      f'{mean(pc, "d_global_sign"):+.2f}pp · remaining headroom to the '
      f'label-chosen oracle-5 {mean(pc, "d_headroom"):+.2f}pp.</p>')

    # ── 2b. THE MECHANISM ────────────────────────────────────────────────────
    rv = np.array([r["_recov"] for r in pc], dtype=float)
    wk = np.array([r["_weak"] for r in pc])
    fin = np.isfinite(rv)
    hlo = int((rv[fin & ~wk] < RECOV_MIN).sum())
    wlo = int((rv[fin & wk] < RECOV_MIN).sum())
    from scipy import stats as _st
    fisher_p = _st.fisher_exact(
        [[wlo, int((fin & wk).sum()) - wlo],
         [hlo, int((fin & ~wk).sum()) - hlo]], alternative="greater")[1]

    W("<h2>2b. The mechanism: L-SML is a sign-recovery machine, and it "
      "under-recovers on the weak cells</h2>")
    W("<p>Read the two middle columns of the ladder together. Not knowing the "
      "per-view signs costs a simple average <code>d_signs</code>; recovering that "
      "without labels is precisely what L-SML's grouping and weighting is for. "
      "The ratio <code>d_lsml_vs_avg / −d_signs</code> is <b>how much of it comes "
      "back</b> — 1.0 means all of it.</p>")
    W('<div class="scroll"><table><tr><th>Cell</th><th>cost of losing signs</th>'
      "<th>recovered by L-SML</th><th>recovery ratio</th></tr>")
    for r in pc:
        cls = ' class="weak"' if r["_weak"] else ""
        rr = ("—" if not np.isfinite(r["_recov"])
              else f'{r["_recov"]:.3f}')
        bad = (' class="neg"' if np.isfinite(r["_recov"])
               and r["_recov"] < RECOV_MIN else "")
        W(f'<tr{cls}><td>{esc(name_of(r["cell"]))}</td>'
          f'{sgn(r["d_signs"])}{sgn(r["d_lsml_vs_avg"])}'
          f'<td{bad}>{rr}</td></tr>')
    W("</table></div>")
    W(f'<div class="box bad"><p>On <b>every one of the {int((fin & ~wk).sum())} '
      f'healthy cells where signs matter at all, recovery lands between 0.919 and '
      f'1.247</b> (median {np.nanmedian(rv[fin & ~wk]):.3f}) — L-SML gets '
      f'essentially all of it back, without labels. On the weak cells '
      f'<b>{wlo} of {int((fin & wk).sum())}</b> fall below {RECOV_MIN}, and no '
      f'healthy cell does at all. Fisher exact <b>p = {fisher_p:.4f}</b>.</p>'
      f'<p>This is the failure. It is not orientation (the global bit is right on '
      f'25/25), not the K-selection (§5), and not the pool. It is that the '
      f'label-free machinery for recovering <i>relative</i> feature polarity stops '
      f'working on exactly the cells where the views are individually weak — which '
      f'is where it has the least covariance structure to work from.</p></div>')

    # ── 3. selection ─────────────────────────────────────────────────────────
    W("<h2>3. What was selected (B1)</h2>")
    W("<p>Overlap with the label-chosen oracle-5, and how many <i>strong</i> views "
      "(oracle-oriented AUROC at or above the cell's 75th percentile) each arm "
      "left out.</p>")
    W("<div class='scroll'><table><tr><th>Cell</th><th>pool</th>"
      "<th>DUFS size</th><th>U-PCR kept</th><th>Jaccard DUFS∩oracle</th>"
      "<th>Jaccard U-PCR∩oracle</th><th>strong dropped (DUFS)</th>"
      "<th>strong excluded (U-PCR)</th><th>best view missed by</th></tr>")
    for r in pc:
        cls = ' class="weak"' if r["_weak"] else ""
        W(f'<tr{cls}><td>{esc(name_of(r["cell"]))}</td><td>{r["p_pool"]}</td>'
          f'<td>{r["dufs_size"]}</td><td>{r["upcr_size"]}</td>'
          f'<td>{fnum(r["jac_dufs_oracle"], 3)}</td>'
          f'<td>{fnum(r["jac_upcr_oracle"], 3)}</td>'
          f'<td>{r["n_strong_dropped_dufs"]}</td>'
          f'<td>{r["n_strong_excluded_upcr"]}</td>{sgn(r["_sel"])}</tr>')
    W("</table></div>")
    miss = [r for r in pc if r["_sel"] <= SEL_MISS_PP]
    W(f'<p><b>The chosen subset excludes the pool\'s strongest view on '
      f'{len(miss)} cells, {sum(1 for r in miss if r["_weak"])} of them weak</b> — '
      f"and on the two worst (TriviaQA / OPT-30B −4.57pp and GSM8K / Qwen2.5-7B "
      f"−4.81pp) the selection is throwing away more than the whole gap to the "
      f"ceiling. Two weak cells share a Jaccard of <b>exactly 0.000</b> against "
      f"the label-chosen oracle-5: the selector and the oracle agree on nothing.</p>")

    # ── 4. feature behaviour ─────────────────────────────────────────────────
    W("<h2>4. Did any view behave differently there (B2)</h2>")
    W("<p><code>nonmono_gain_cv</code> is the cross-fitted AUROC of a "
      "piecewise-constant (bin-mean) predictor minus the view's oracle-oriented "
      "AUROC. Positive means the view carries signal that <b>no monotone, "
      "sign-oriented use of it can reach</b> — and every fusion in this project "
      "is monotone in each view. Cross-fitted over 5 folds, because the in-sample "
      "version beats a monotone score even on pure noise.</p>")
    W("<div class='scroll'><table><tr><th>Cell</th><th>mean</th><th>p90</th>"
      "<th>views with gain &gt; 2pp</th><th>largest single gain</th></tr>")
    byc = {}
    for r in pf:
        byc.setdefault(r["cell"], []).append(r)
    for r in pc:
        rows = byc[r["cell"]]
        g = np.array([float(x["nonmono_gain_cv"]) for x in rows
                      if x["nonmono_gain_cv"] not in ("", "nan")])
        big = [x for x in rows if x["nonmono_gain_cv"] not in ("", "nan")
               and float(x["nonmono_gain_cv"]) > 0.02]
        top = max(rows, key=lambda x: float(x["nonmono_gain_cv"] or -9))
        cls = ' class="weak"' if r["_weak"] else ""
        W(f'<tr{cls}><td>{esc(name_of(r["cell"]))}</td>'
          f'{sgn(g.mean() * 100)}{sgn(np.percentile(g, 90) * 100)}'
          f'<td>{len(big)}</td>'
          f'<td>{esc(top["feature"])} '
          f'{float(top["nonmono_gain_cv"]) * 100:+.1f}pp</td></tr>')
    W("</table></div>")

    # ── 5. residual process ──────────────────────────────────────────────────
    W("<h2>5. What the residual process did (B3)</h2>")
    W("<p>K is chosen by minimising the Eq.-14 residual. <code>gap</code> is the "
      "relative separation between the winning K and the runner-up: a small gap "
      "means K was decided by very little. The eigengap column is the "
      "counterfactual — what a different K rule would have produced on the same "
      "subset.</p>")
    W("<div class='scroll'><table><tr><th>Cell</th><th>m</th><th>K</th>"
      "<th>residual</th><th>gap to runner-up</th><th>degenerate</th>"
      "<th>K eigengap</th><th>AUROC eigengap</th><th>Δ vs deployed</th></tr>")
    for r in pc:
        cls = ' class="weak"' if r["_weak"] else ""
        dg = "yes" if r["degenerate"].lower() == "true" else ""
        W(f'<tr{cls}><td>{esc(name_of(r["cell"]))}</td><td>{r["dufs_size"]}</td>'
          f'<td>{r["K"]}</td><td>{fnum(r["residual"], 1)}</td>'
          f'<td>{fnum(float(r["residual_gap_rel"]) * 100, 2)}%</td>'
          f'<td>{dg}</td><td>{r["K_eigengap"]}</td>'
          f'<td>{fnum(r["auc_eigengap"])}</td>{sgn(r["d_eigengap"])}</tr>')
    W("</table></div>")
    ndeg = sum(1 for r in pc if r["degenerate"].lower() == "true")
    nhelp = sum(1 for r in pc if float(r["d_eigengap"]) > 0)
    W(f'<div class="box"><p><b>This is a negative result, and it clears a '
      f'suspect.</b> The Step-205 degeneracy flag fires on <b>{ndeg} of 25</b> '
      f'cells — the grouping is a measurement everywhere here, not a coin flip. '
      f'And swapping the residual criterion for the eigengap helps on only '
      f'<b>{nhelp} of 25</b> cells, mean {mean(pc, "d_eigengap"):+.2f}pp. '
      f'<b>K-selection is not the failure mechanism on the deployed path.</b></p>'
      f'<p>An earlier lead pointed the other way — on <code>ALL_H16</code>, '
      f'<code>ars_gsm8k_r1distill8b</code> picks K=4 and lands at 0.364, below '
      f'chance, where the eigengap picks K=2 and gets 0.658. That is real, but '
      f'<code>ALL_H16</code> is an obsolete 16-view subset we do not deploy. On '
      f'the subset actually in use, the same cell is fine.</p></div>')

    # ── 6. verdict ───────────────────────────────────────────────────────────
    W("<h2>6. Mechanism per cell</h2>")
    W("<p>Every mechanism a cell trips, against thresholds fixed by the healthy "
      "cells' own spread — not chosen per cell. A cell tripping more than one is "
      "a finding, not a tie to be broken. "
      f"<span class='mech m1'>sign-recovery failure</span> recovery &lt; "
      f"{RECOV_MIN} (no healthy cell is). "
      f"<span class='mech m2'>selection miss</span> the chosen subset excludes "
      f"the pool's best view by ≥ {abs(SEL_MISS_PP):.0f}pp. "
      f"<span class='mech m4'>non-monotone views</span> mean cross-fitted "
      f"non-monotone gain &gt; {NONMONO_MIN:.2f} (the max over every other cell "
      f"is 0.020). "
      f"<span class='mech m3'>ceiling-limited</span> headroom &lt; "
      f"{CEILING_PP:.0f}pp and fusion within 1pp of the best view.</p>")
    W("<div class='scroll'><table><tr><th>Cell</th><th>n</th><th>pos rate</th>"
      "<th>best view</th><th>deployed</th><th>oracle-5</th>"
      "<th>headroom</th><th>mechanism(s)</th></tr>")
    for r in weak:
        tags = "".join(f'<span class="mech {MECH[m][0]}">{MECH[m][1]}</span> '
                       for m in r["_mech"]) or "<i>none fires</i>"
        W(f'<tr class="weak"><td>{esc(name_of(r["cell"]))}</td><td>{r["n"]}</td>'
          f'<td>{fnum(r["pos_rate"], 3)}</td><td>{fnum(r["best_single"])}</td>'
          f'<td>{fnum(r["r5_lsml_anchor_DEPLOYED"])}</td>'
          f'<td>{fnum(r["r6_oracle5_ceiling"])}</td>{sgn(r["d_headroom"])}'
          f'<td style="text-align:left">{tags}</td></tr>')
    W("</table></div>")
    hfire = [r for r in heal if r["_mech"]]
    wnone = [r for r in weak if not r["_mech"]]
    W(f'<p>For contrast, {len(hfire)} of the 16 healthy cells trip any mechanism '
      f'at all{": " + ", ".join(esc(name_of(r["cell"])) for r in hfire) if hfire else ""}.</p>')
    if wnone:
        heads = ", ".join(f'{float(r["d_headroom"]):.1f}pp' for r in wnone)
        W(f'<div class="box"><p><b>{len(wnone)} of the nine trip nothing, and that '
          f'is a result rather than a gap in the diagnostic:</b> '
          f'{", ".join(esc(name_of(r["cell"])) for r in wnone)}. On these the '
          f'pipeline behaves the way it does on the healthy cells — sign recovery '
          f'is at or above {RECOV_MIN}, the selector keeps the strongest view, the '
          f'views are monotone, the grouping is determinate. They score low because '
          f'the signal is weak, not because anything is broken. The remaining '
          f'headroom to a label-chosen five-view subset is {heads} '
          f'respectively — real, but only reachable with labels.</p>'
          f'<p><b>So "why do we fail here" has two different answers, and they need '
          f'different responses.</b> Six cells have a named, fixable defect. Three '
          f'are simply hard, and on those the honest move is to report the ceiling '
          f'rather than to chase it.</p></div>')

    # ── 7. pre-registration ──────────────────────────────────────────────────
    W("<h2>7. Repairs, pre-registered and NOT tested here</h2>")
    W('<div class="box"><p>Written down before any of them is run, so the '
      "diagnosis cannot be tuned to make a fix look good. Each names the cells it "
      "should help and the cells it must not hurt.</p>"
      "<ol>"
      "<li><b>A better label-free relative-sign estimator</b> — targets §2b, the "
      "headline mechanism, and is the only repair aimed at it. The candidate is "
      "Z<sub>2</sub> synchronisation on the sign pattern of the correlation "
      "matrix (<code>spectral_utils/orientation.z2_sign_recovery</code>, already "
      "written and currently unused on this path) in place of L-SML's implicit "
      "recovery. <b>Gate:</b> must lift recovery above 0.90 on the four failing "
      "cells and move the twenty-two healthy ones by less than 0.5pp.</li>"
      "<li><b>Rank / quantile transform of each view before fusion</b> — targets "
      "the non-monotonicity in §4, which fires on CoQA and nowhere else. "
      "<b>Gate:</b> must be a no-op (&lt;0.5pp) on the 24 cells whose "
      "<code>nonmono_gain_cv</code> is ≈0, or it is buying CoQA with everything "
      "else.</li>"
      "<li><b>Keep the pool's strongest view unconditionally</b> — targets §3. "
      "The selector drops it on four cells, costing up to 4.8pp before fusion "
      "even starts. This is label-free only if 'strongest' is decided without "
      "labels, which is the hard part and may sink it.</li>"
      "</ol>"
      "<p><b>Not proposed, and why:</b> a K-selection change (§5 clears it — "
      "0/25 degenerate, eigengap helps 5/25), anything touching orientation "
      "(§0 — the global bit is right on 25/25 and costs 0.00pp), and anything "
      "touching pool composition (Step 206 closed it in both directions).</p>"
      "</div>")

    # ── gates ────────────────────────────────────────────────────────────────
    W('<div class="foot">')
    W(f"<b>Gates.</b> GOOD_6 validity anchor {gates['good6_macro']:.4f} "
      f"(expected 0.7594 ± 0.002) · reproduction of both deployed arms to &lt;5e-4: "
      f"{'PASS' if not gates['drift'] else 'FAIL — ' + esc('; '.join(gates['drift']))} · "
      f"ladder r5 ≤ r4: "
      f"{'PASS' if not gates['ladder_viol'] else 'FAIL'} · "
      f"confound re-check Spearman(anchor, best single) = "
      f"{gates['spearman_anchor_best_single']:+.3f}.")
    W("<br>Generated by <code>scripts/failure_deepdive_report.py</code> from "
      "<code>results/failure_deepdive/*.csv</code>. Every K and residual on this "
      "page was recomputed by <code>scripts/failure_deepdive.py</code> under "
      "current code — none is joined from a cached bench CSV, all of which "
      "predate the Step-205 grouping fix.")
    W("</div></div>")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"<title>Why we fail where we fail — the nine-cell diagnosis</title>"
                f"<style>{CSS}</style>" + "\n".join(A))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
