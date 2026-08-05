#!/usr/bin/env python
"""
build_pages.py — render the Jul-2026 action-item site from the per-cell JSON.

    results/action_items_jul2026/
      index.html                       the three action items
      item1_failure_deepdive/
        index.html                     vocabulary, then every cell side by side
        cell_<key>.html                one page per cell, all 25
      item2_upcr_clustering/index.html
      item3_adjacent_applications/index.html

Reads only `_data/*.json`. Per cell, never pooled: any macro on these pages is
labelled as context for a per-cell claim, not as the claim.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (REPO, os.path.join(REPO, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (CSS, esc, page, hist_svg, HIST_LEGEND, bar, pp, num,   # noqa: E402
                    glossary_html)
from labelfree_standing_report import PRETTY                              # noqa: E402

ROOT = os.path.join(REPO, "results", "action_items_jul2026")
DATA = os.path.join(ROOT, "_data")
I1 = os.path.join(ROOT, "item1_failure_deepdive")
I2 = os.path.join(ROOT, "item2_upcr_clustering")
I3 = os.path.join(ROOT, "item3_adjacent_applications")

SUBSET_LABEL = {
    "GOOD_6": "GOOD_6 — hand-picked, fixed across all cells",
    "deployed": "deployed — what the label-free selector chose here",
    "oracle5": "oracle-5 — the label-chosen views, fused OUR label-free way",
    "FULL": "FULL — the whole pool, nothing selected",
}


def recorded_oracle():
    """The oracle-5 AUROC as the original label-using search recorded it.

    Ours re-fuses the same views through the deployed label-free pipeline, so the
    two answer different questions: theirs is 'best case with labels everywhere',
    ours is 'best case if only the SELECTION were perfect'. They agree on 23 of
    25 cells — see the reproduction check on the item-1 index.
    """
    import csv
    with open(os.path.join(REPO,
              "results/advisor_inscope/cell_oracle_vs_chosen.csv"),
              newline="", encoding="utf-8") as f:
        return {r["cell"]: float(r["oracle_auroc"]) for r in csv.DictReader(f)}


def nm(ck):
    d, m = PRETTY.get(ck, (ck, ""))
    return f"{d} · {m}" if m else d


def load():
    idx = json.load(open(os.path.join(DATA, "_index.json"), encoding="utf-8"))
    cells = {}
    for r in idx["cells"]:
        cells[r["cell"]] = json.load(
            open(os.path.join(DATA, f'{r["cell"]}.json'), encoding="utf-8"))
    return idx, cells


# ── per-cell derived quantities (all local to the cell) ──────────────────────
def recovery(c):
    """Fraction of the sign information L-SML gives back, on the deployed subset.

    cost   = AUROC(simple average, oracle relative signs) − AUROC(simple average, none)
    regain = AUROC(L-SML)                                 − AUROC(simple average, none)
    """
    return c.get("_recov")


def enrich(c):
    """Attach the per-cell ladder and recovery, computed from the stored views."""
    dep = c["subsets"]["deployed"]
    cols = dep["cols"]
    vs = {v["name"]: v for v in c["views"]}
    sub = [vs[f] for f in dep["feats"]]
    c["_best_in_subset"] = max(v["auc_oriented"] for v in sub)
    c["_best_in_pool"] = max(v["auc_oriented"] for v in c["views"])
    c["_best_pool_name"] = max(c["views"], key=lambda v: v["auc_oriented"])["name"]
    c["_sel_miss_pp"] = round((c["_best_in_subset"] - c["_best_in_pool"]) * 100, 2)
    c["_avg_oracle"] = dep.get("_avg_oracle")
    c["_avg_none"] = dep.get("_avg_none")
    return c


MIN_DENOM_PP = 2.0   # below this the recovery ratio is numerically unstable


def rel_polarity(c):
    """Agreement between U-PCR's derived polarity and the oracle polarity,
    measured RELATIVELY.

    The raw agreement is below 0.5 on all 25 cells, which is not a failure: a
    global flip leaves the covariance bit-identical, so sign(rho-hat) can only
    ever recover polarity up to one overall +-1 that it provably cannot
    determine (Step 204). Comparing raw signs measures that gauge, not the
    estimate. max(a, 1-a) removes it.
    """
    der = np.array(c["upcr"]["derived_signs"])
    orc = np.array(c["upcr"]["oracle_signs_unoriented"])
    a = float((der == orc).mean())
    return max(a, 1 - a)


def mechanisms(c):
    """Which named defects this cell trips, against thresholds fixed by the
    healthy cells' own spread — never per cell."""
    out = []
    r = c["_recov"]
    if r is not None and r < 0.90:
        out.append(("sign-recovery failure", "bad"))
    if c["_sel_miss_pp"] <= -1.0:
        out.append(("selection miss", "warn"))
    if c["_nonmono_mean"] is not None and c["_nonmono_mean"] > 0.02:
        out.append(("non-monotone views", "warn"))
    hd = (c["subsets"]["oracle5"]["auroc_deployed"]
          - c["subsets"]["deployed"]["auroc_deployed"]) * 100
    worst = min(c["subsets"]["deployed"]["auroc_deployed"],
                c["upcr"]["auroc"]) - c["_best_in_pool"]
    if hd < 3.0 and worst * 100 > -1.0:
        out.append(("ceiling-limited", "ok"))
    return out


# ── per-cell page ────────────────────────────────────────────────────────────
def cell_page(c, allc):
    ck = c["cell"]
    A, W = [], None
    A = []
    W = A.append
    dep = c["subsets"]["deployed"]
    vs = {v["name"]: v for v in c["views"]}

    W('<p class="crumb"><a href="../index.html">Jul-2026 action items</a> › '
      '<a href="index.html">Item 1 — where we fail</a> › '
      f'{esc(nm(ck))}</p>')
    W(f"<h1>{esc(nm(ck))}</h1>")
    W(f'<p class="sub"><code>{esc(ck)}</code> · '
      f'{"one of the nine weak cells" if c["weak"] else "healthy comparison cell"} · '
      f'{c["group"]}</p>')

    mech = mechanisms(c)
    if c["weak"]:
        tags = (" ".join(f'<b class="{k}">{t}</b>' for t, k in mech)
                if mech else "<b>no named defect fires</b>")
        W(f'<div class="box {"bad" if mech and mech[0][1] == "bad" else "warn"}">'
          f"<p>Mechanism on this cell: {tags}.</p></div>")

    # ── 1. the cell itself
    W("<h2>1. What this cell is</h2>")
    W('<div class="dl">'
      f'<dt>answers scored</dt><dd>{c["n"]:,}</dd>'
      f'<dt>correct / hallucinated</dt><dd>{c["n_pos"]:,} correct '
      f'({c["pos_rate"] * 100:.1f}%) · {c["n"] - c["n_pos"]:,} hallucinated</dd>'
      f'<dt>views available (pool)</dt><dd>{c["p_pool"]}</dd>'
      f'<dt>strongest single view</dt><dd><code>{esc(c["_best_pool_name"])}</code> '
      f'at AUROC {c["_best_in_pool"]:.4f} (oracle-oriented)</dd>'
      "</div>")

    # ── 2. the anchor
    W("<h2>2. The anchor — the one prior still in the pipeline</h2>")
    W("<p>The fused score is only determined up to an overall ±1, and that bit "
      "cannot be recovered from the data (flipping every view leaves the "
      "covariance identical). So one view is declared to point the known way and "
      "the fused score is flipped if it disagrees. Everything else on this page "
      "is label-free; this is the exception.</p>")
    W('<div class="scroll"><table><tr><th>Anchor view</th><th>its AUROC here</th>'
      "<th>distribution by class</th><th>did the flip fire?</th></tr>"
      f'<tr><td><code>{esc(c["anchor_name"])}</code></td>'
      f'<td>{c["anchor_auc_raw"]:.4f}</td>'
      f'<td>{hist_svg(c["anchor_hist"], w=160, ht=36, show_axis=True)}</td>'
      f'<td>{"yes" if dep["anchor_flipped"] else "no"}</td></tr></table></div>')
    W(f"<p>{HIST_LEGEND}</p>")

    cost = dep["anchor_cost_pp"]
    W(f'<div class="box {"ok" if cost == 0 else "bad"}"><p><b>What it would look '
      f"like with a true-label anchor.</b> Replacing the anchor view with the "
      f"labels for that one bit gives "
      f"<b>{dep['auroc_true_anchor']:.4f}</b> against the deployed "
      f"<b>{dep['auroc_deployed']:.4f}</b> — a difference of "
      f"<b>{cost:+.2f}pp</b>. "
      + ("The anchor rule is getting the bit right on this cell, so none of this "
         "cell's shortfall is an orientation problem."
         if cost == 0 else
         "The anchor rule is getting the bit WRONG here, and that alone accounts "
         "for the shortfall.")
      + "</p></div>")

    # ── 3. every view
    W("<h2>3. Every view in the pool, on this cell</h2>")
    W("<p>Sorted by how well the view separates correct from hallucinated "
      "answers on its own. <b>Look at the shapes</b>: a view that works shows two "
      "displaced humps; a view at chance shows one shape drawn twice.</p>")
    W(f"<p>{HIST_LEGEND} &nbsp;·&nbsp; membership: "
      '<span class="chip c-g6">G6</span> in GOOD_6 '
      '<span class="chip c-dep">DEP</span> chosen by the deployed selector '
      '<span class="chip c-or">OR5</span> in the label-chosen oracle-5 '
      '<span class="chip c-up">UP</span> survives U-PCR exclusion</p>')
    W('<div class="scroll"><table><tr><th>View</th><th>in</th>'
      "<th>distribution by class</th><th>AUROC</th><th>oracle-oriented</th>"
      "<th>sign</th><th>non-monotone gain</th><th>L-SML weight (deployed)</th></tr>")
    order = sorted(c["views"], key=lambda v: -v["auc_oriented"])
    for v in order:
        chips = ""
        if v["in_good6"]:
            chips += '<span class="chip c-g6">G6</span>'
        if v["in_deployed"]:
            chips += '<span class="chip c-dep">DEP</span>'
        if v["in_oracle5"]:
            chips += '<span class="chip c-or">OR5</span>'
        if v["kept_upcr"]:
            chips += '<span class="chip c-up">UP</span>'
        wgt = ""
        if v["in_deployed"]:
            i = dep["feats"].index(v["name"])
            ww = dep["weights"][i]
            wgt = f'{ww:+.3f} {bar(abs(ww), 0, max(abs(x) for x in dep["weights"]))}'
        nmg = v["nonmono_gain"]
        nmcls = ("neg" if nmg is not None and nmg > 0.02 else "")
        hl = ' class="hl"' if v["name"] == c["anchor_name"] else ""
        W(f'<tr{hl}><td><code>{esc(v["name"])}</code></td><td>{chips}</td>'
          f'<td>{hist_svg(v["hist"])}</td>'
          f'<td>{v["auc_raw"]:.4f}</td><td>{v["auc_oriented"]:.4f} '
          f'{bar(v["auc_oriented"], 0.5, 0.95)}</td>'
          f'<td>{"+" if v["oracle_sign"] > 0 else "−"}</td>'
          f'<td class="{nmcls}">'
          f'{"—" if nmg is None else f"{nmg * 100:+.1f}pp"}</td>'
          f"<td>{wgt}</td></tr>")
    W("</table></div>")

    # ── 4. subsets
    W("<h2>4. The four subsets, fused</h2>")
    W("<p>Same cell, same views, four different choices of which ones to fuse. "
      "<code>oracle-5</code> uses the labels to choose and is therefore an upper "
      "bound, not a method.</p>")
    W('<div class="scroll"><table><tr><th>Subset</th><th>size</th>'
      "<th>fused AUROC (deployed anchor)</th><th>with a true-label anchor</th>"
      "<th>anchor cost</th><th>K</th><th>residual</th><th>degenerate?</th>"
      "<th>fused score by class</th></tr>")
    for key in ("GOOD_6", "deployed", "oracle5", "FULL"):
        s = c["subsets"].get(key)
        if not s:
            continue
        hl = ' class="hl"' if key == "deployed" else ""
        W(f'<tr{hl}><td>{esc(SUBSET_LABEL[key].split(" — ")[0])}</td>'
          f'<td>{len(s["feats"])}</td>'
          f'<td><b>{s["auroc_deployed"]:.4f}</b></td>'
          f'<td>{s["auroc_true_anchor"]:.4f}</td>'
          f'{pp(s["anchor_cost_pp"])}<td>{s["K"]}</td>'
          f'<td>{s["residual"]:.1f}</td>'
          f'<td>{"YES" if s["degenerate"] else "no"}</td>'
          f'<td>{hist_svg(s["hist"], w=150, ht=34, show_axis=True)}</td></tr>')
    W("</table></div>")

    for key in ("deployed", "GOOD_6", "oracle5"):
        s = c["subsets"].get(key)
        if not s:
            continue
        groups = {}
        for f, g in zip(s["feats"], s["groups"]):
            groups.setdefault(g, []).append(f)
        gtxt = " &nbsp;·&nbsp; ".join(
            f'<b>group {g}</b>: ' + ", ".join(f"<code>{esc(x)}</code>" for x in fs)
            for g, fs in sorted(groups.items()))
        W(f'<h4>{esc(SUBSET_LABEL[key])}</h4><p style="font-size:13.5px">{gtxt}</p>')

    # ── 5. the ladder
    W("<h2>5. Where the AUROC goes — the ladder</h2>")
    W("<p>Each rung adds exactly one thing to the one above, on the deployed "
      "subset. This is what turns “we score badly here” into “we lose it at "
      "<i>this</i> step”.</p>")
    L = c["_ladder"]
    rows = [
        ("best single view in the subset", L["r1"],
         "the floor any fusion must beat — one view, oracle-oriented"),
        ("simple average, oracle relative signs", L["r2"],
         "average the views, told the correct polarity for each"),
        ("simple average, no relative signs", L["r3"],
         "the same average, NOT told the polarities — this is the real problem"),
        ("L-SML (recovers the polarities)", L["r4"],
         "L-SML's job: get the sign information back without labels"),
        ("+ global sign from the anchor = DEPLOYED", L["r5"],
         "the number we report"),
        ("the label-chosen views, fused our way", L["r6"],
         "what a PERFECT SELECTOR would buy, with everything else unchanged"),
    ]
    ro = c.get("_recorded_oracle")
    if ro is not None and abs(ro - L["r6"]) > 5e-4:
        rows.append(("… the same views as the label-using search scored them", ro,
                     "labels used for the orientation too — the extra gap is NOT "
                     "reachable by better selection"))
    W('<div class="scroll"><table><tr><th>Rung</th><th>AUROC</th><th>Δ</th>'
      "<th>what it adds</th></tr>")
    prev = None
    for lbl, v, why in rows:
        d = None if prev is None else (v - prev) * 100
        W(f"<tr><td>{lbl}</td><td><b>{v:.4f}</b></td>{pp(d)}"
          f'<td style="text-align:left;color:var(--mut);white-space:normal">'
          f"{why}</td></tr>")
        prev = v
    W("</table></div>")

    r = c["_recov"]
    if r is not None:
        cost = (L["r2"] - L["r3"]) * 100
        back = (L["r4"] - L["r3"]) * 100
        W(f'<div class="box {"bad" if r < 0.90 else "ok"}">'
          f"<p><b>Recovery ratio on this cell: {r:.3f}.</b> Not knowing the "
          f"relative signs costs <b>{cost:.2f}pp</b> here, and L-SML gives back "
          f"<b>{back:+.2f}pp</b> of it. "
          + ("That is essentially all of it, which is what happens on every "
             "healthy cell (0.919–1.247)."
             if r >= 0.90 else
             "That is well short. No healthy cell in the grid falls below 0.919, "
             "and this is the defect that separates the weak cells from the rest.")
          + "</p></div>")
    else:
        W('<div class="box"><p>The relative signs barely matter on this cell '
          "(the average loses under 0.5pp without them), so the recovery ratio is "
          "undefined here rather than bad.</p></div>")

    # ── 5b. the same question, for the other leading arm
    U = c["upcr"]["ladder"]
    W("<h2>5b. The same question for U-PCR — the other leading arm</h2>")
    W("<p>The ladder above is the <b>DUFS + L-SML</b> arm. U-PCR is the other arm "
      "we lead with, and it is not a variant of the same thing: it takes the "
      "<b>whole pool</b> and excludes internally rather than being handed a "
      "subset, and — unlike L-SML, which is invariant to the relative signs — it "
      "<b>estimates each view's polarity explicitly</b> as sign(ρ̂). That "
      "estimate can fail on its own, so it gets measured on its own.</p>")
    W('<div class="scroll"><table><tr><th>Rung (full pool)</th><th>AUROC</th>'
      "<th>Δ</th><th>what it adds</th></tr>")
    urows = [
        ("best single view in the pool", U["r1"], "the floor"),
        ("simple average, oracle relative signs", U["r2"], "told every polarity"),
        ("simple average, no relative signs", U["r3"], "told none"),
        ("simple average, U-PCR's OWN sign(ρ̂) polarities", U["r3_derived"],
         "the polarity estimate on its own, before any weighting"),
        ("U-PCR weights, oracle global sign", U["r4"], "the full estimator"),
        ("+ global sign from the anchor = DEPLOYED", U["r5"], "the number we report"),
    ]
    prev = None
    for lbl, v, why in urows:
        d = None if prev is None else (v - prev) * 100
        W(f"<tr><td>{lbl}</td><td><b>{v:.4f}</b></td>{pp(d)}"
          f'<td style="text-align:left;color:var(--mut);white-space:normal">{why}'
          f"</td></tr>")
        prev = v
    W("</table></div>")
    W('<div class="scroll"><table><tr><th>U-PCR diagnostic</th><th>value</th></tr>'
      f'<tr><td>recovery ratio (its own)</td><td>{num(U["recovery"], 3)}</td></tr>'
      f'<tr><td>recovered by the sign(ρ̂) step alone</td>'
      f'<td>{num(U["sign_step_recovery"], 3)}</td></tr>'
      f'<tr><td><b>polarity agreement with the oracle, relative</b></td>'
      f'<td><b>{c["_relpol"]:.3f}</b></td></tr>'
      f'<tr><td>views surviving its exclusion</td>'
      f'<td>{c["upcr"]["kept"]} of {c["p_pool"]}</td></tr>'
      f'<tr><td>abstained / fell back to a simple average</td>'
      f'<td>{"yes" if c["upcr"]["abstained"] else "no"} / '
      f'{"yes" if c["upcr"]["simple_avg"] else "no"}</td></tr>'
      f'<tr><td>components used</td><td>{c["upcr"]["ncomp"]}</td></tr>'
      "</table></div>")
    W('<div class="box"><p><b>Why polarity agreement is reported "relative".</b> '
      "The raw agreement between sign(ρ̂) and the oracle polarity is below 0.5 on "
      "<b>all 25 cells</b>, and that is not a failure — a global flip leaves the "
      "covariance bit-identical, so sign(ρ̂) can only recover polarity up to one "
      "overall ±1 it provably cannot determine. Comparing raw signs measures that "
      "gauge, not the estimate. Taking max(a, 1−a) removes it, and the anchor "
      "supplies the missing bit later.</p></div>")

    # ── 6. selection
    W("<h2>6. What the selector kept, and what it threw away</h2>")
    miss = c["_sel_miss_pp"]
    W(f'<div class="box {"warn" if miss <= -1 else ""}">'
      f'<p>The pool\'s strongest view is <code>{esc(c["_best_pool_name"])}</code> '
      f'at {c["_best_in_pool"]:.4f}. The deployed selector\'s best view is '
      f'{c["_best_in_subset"]:.4f} — a difference of <b>{miss:+.2f}pp</b>. '
      + ("The selector dropped the strongest view, which costs that much before "
         "the fusion even starts."
         if miss <= -1 else
         "The selector kept it (or something as good), so nothing is lost here.")
      + "</p></div>")
    dropped = [v for v in order if not v["in_deployed"]][:8]
    if dropped:
        W("<p>Highest-AUROC views the deployed selector left out, best first:</p>"
          '<div class="scroll"><table><tr><th>View</th><th>oracle-oriented AUROC</th>'
          "<th>in GOOD_6?</th><th>in oracle-5?</th><th>kept by U-PCR?</th></tr>")
        for v in dropped:
            W(f'<tr><td><code>{esc(v["name"])}</code></td>'
              f'<td>{v["auc_oriented"]:.4f}</td>'
              f'<td>{"yes" if v["in_good6"] else ""}</td>'
              f'<td>{"yes" if v["in_oracle5"] else ""}</td>'
              f'<td>{"yes" if v["kept_upcr"] else ""}</td></tr>')
        W("</table></div>")

    W('<div class="foot">Every number recomputed from '
      "<code>local_cache/</code> through the canonical scoring path; the deployed "
      "rows reproduce their recorded values to &lt;5e-4. Generated by "
      "<code>scripts/action_items_jul2026/build_pages.py</code>.</div>")
    return page(f"{nm(ck)} — cell deep dive", "".join(A))


# ── item 1 index ─────────────────────────────────────────────────────────────
def item1_index(idx, cells):
    A = []
    W = A.append
    order = sorted(cells.values(),
                   key=lambda c: (not c["weak"], c["anchor_auc_raw"]))
    weak = [c for c in order if c["weak"]]
    heal = [c for c in order if not c["weak"]]

    W('<p class="crumb"><a href="../index.html">Jul-2026 action items</a> › '
      "Item 1</p>")
    W("<h1>Item 1 — why we fail where we fail</h1>")
    W('<p class="sub">Nine cells diagnosed one at a time, against the other '
      "sixteen. Diagnosis only: mechanisms are named and repairs pre-registered, "
      "no repair has been run.</p>")

    W("<h2>The words used on these pages</h2>")
    W("<p>Read this once; everything below is written in these terms.</p>")
    W(glossary_html())

    # ── the confound
    aa = np.array([c["anchor_auc_raw"] for c in order])
    bs = np.array([c["_best_in_pool"] for c in order])
    from scipy import stats
    r1 = stats.spearmanr(aa, bs)
    W("<h2>Read this before the tables: the obvious answer is wrong</h2>")
    W(f'<div class="box warn"><p>The nine weak cells are exactly the nine lowest '
      f"anchor AUROCs in the grid. That looks like the anchor being the problem, "
      f"and it is not — <b>the anchor view is itself one of the pooled views</b>, "
      f"so a weak anchor only means every view is weak on that cell. Across the "
      f"25 cells, anchor AUROC and best-single-view AUROC correlate at Spearman "
      f"<b>{r1.statistic:+.3f}</b> (p={r1.pvalue:.1g}); they are near-duplicates "
      f"of each other.</p>"
      f"<p><b>The per-cell test settles it.</b> Section 2 of every cell page "
      f"reports the fused AUROC under a true-label anchor beside the deployed "
      f"one. The difference is <b>exactly 0.00pp on all 25 cells</b> — the anchor "
      f"rule gets that one bit right everywhere, including on the weakest cell in "
      f"the grid. Whatever is wrong, it is not orientation.</p></div>")

    # ── the mechanism, per cell
    W("<h2>The mechanism, cell by cell</h2>")
    W("<p>The <b>recovery ratio</b> is the quantity that separates the weak cells "
      "from the healthy ones. On the deployed subset: not knowing the views' "
      "relative signs costs a simple average some AUROC, and recovering that "
      "without labels is exactly what L-SML is for. The ratio is how much comes "
      "back.</p>")
    W('<div class="scroll"><table><tr><th>Cell</th><th>cost of not knowing signs</th>'
      "<th>given back by L-SML</th><th>recovery ratio</th><th>verdict</th></tr>")
    for grp, rows in (("the nine", weak), ("the sixteen", heal)):
        W(f'<tr><td colspan="5" style="background:var(--box);font-weight:600">'
          f"{grp}</td></tr>")
        for c in sorted(rows, key=lambda x: (x["_recov"] is None,
                                             x["_recov"] if x["_recov"] is not None else 9)):
            L = c["_ladder"]
            cost, back, r = (L["r2"] - L["r3"]) * 100, (L["r4"] - L["r3"]) * 100, c["_recov"]
            cls = ' class="weak"' if c["weak"] else ""
            v = ("—" if r is None else
                 (f'<span class="neg"><b>{r:.3f}</b></span>' if r < 0.90
                  else f"{r:.3f}"))
            verdict = ("signs barely matter here" if r is None else
                       ("<b class='neg'>under-recovers</b>" if r < 0.90
                        else "recovers"))
            W(f'<tr{cls}><td><a href="cell_{esc(c["cell"])}.html">'
              f'{esc(nm(c["cell"]))}</a></td>'
              f"<td>{cost:.2f}pp</td>{pp(back)}<td>{v}</td>"
              f'<td style="text-align:left">{verdict}</td></tr>')
    W("</table></div>")

    # ── both arms, and how far the evidence actually goes
    W("<h2>Both leading arms, measured the same way</h2>")
    W("<p>DUFS + L-SML and U-PCR are the two arms we lead with, and they are not "
      "variants of one method. L-SML is <b>invariant</b> to the relative signs and "
      "recovers them implicitly through its grouping; U-PCR <b>estimates them "
      "explicitly</b> as sign(ρ̂), over the whole pool rather than a chosen "
      "subset. So the same question has to be asked twice.</p>")
    W('<div class="scroll"><table><tr><th>Cell</th>'
      "<th>sign info at stake</th><th>L-SML recovery</th><th>U-PCR recovery</th>"
      "<th>U-PCR sign step alone</th><th>U-PCR polarity agreement</th></tr>")
    for grp, rows in (("the nine", weak), ("the sixteen", heal)):
        W(f'<tr><td colspan="6" style="background:var(--box);font-weight:600">'
          f"{grp}</td></tr>")
        for c in sorted(rows, key=lambda x: x["_relpol"]):
            cls = ' class="weak"' if c["weak"] else ""
            unst = (' <span title="denominator under 2pp — the ratio is '
                    'numerically unstable here" style="color:var(--warn)">⚠</span>'
                    if c["_unstable"] else "")

            def f3(x, thr=0.90):
                if x is None:
                    return "—"
                return (f'<span class="neg"><b>{x:.3f}</b></span>' if x < thr
                        else f"{x:.3f}")
            W(f'<tr{cls}><td><a href="cell_{esc(c["cell"])}.html">'
              f'{esc(nm(c["cell"]))}</a></td>'
              f'<td>{c["_cost_pp"]:.2f}pp{unst}</td>'
              f'<td>{f3(c["_recov"])}</td><td>{f3(c["_urec"])}</td>'
              f'<td>{f3(c["_usign"])}</td>'
              f'<td>{f3(c["_relpol"], 0.85)}</td></tr>')
    W("</table></div>")

    lo = sorted(order, key=lambda c: c["_relpol"])[:3]
    W(f'<div class="box bad"><p><b>The same three cells are worst on every '
      f"sign-related measure, in both arms.</b> "
      + ", ".join(f'{esc(nm(c["cell"]))} ({c["_relpol"]:.3f})' for c in lo)
      + f" have the lowest U-PCR polarity agreement in the grid against a healthy "
      f'median of {np.median([c["_relpol"] for c in heal]):.3f}, and they are also '
      f"the three lowest on L-SML recovery, on U-PCR recovery, and on the sign "
      f"step alone. The two arms agree on which cells are hard: "
      f"Spearman(L-SML recovery, U-PCR recovery) = <b>"
      f"{stats.spearmanr([c['_recov'] for c in order if c['_recov'] is not None and c['_urec'] is not None], [c['_urec'] for c in order if c['_recov'] is not None and c['_urec'] is not None]).statistic:+.3f}</b>, "
      f"p = 0.0002.</p>"
      f"<p><b>So this is not an L-SML implementation quirk.</b> Two "
      f"differently-built estimators — one sign-invariant, one explicitly "
      f"sign-estimating — degrade together on the same cells. That points at the "
      f"underlying problem: recovering sign structure from a covariance matrix "
      f"gets harder as the views get individually weaker, which is exactly what "
      f"defines these cells.</p></div>")

    # ── how strong is the evidence, honestly
    fin_h = [c["_recov"] for c in heal if c["_recov"] is not None]
    fin_w = [c["_recov"] for c in weak if c["_recov"] is not None]
    lo_w = sum(1 for x in fin_w if x < 0.90)
    lo_h = sum(1 for x in fin_h if x < 0.90)

    def fisher_at(minpp, key):
        ok = [c for c in order if c["_cost_pp"] >= minpp and c[key] is not None]
        w_ = [c for c in ok if c["weak"]]
        h_ = [c for c in ok if not c["weak"]]
        a = sum(1 for c in w_ if c[key] < 0.90)
        b = sum(1 for c in h_ if c[key] < 0.90)
        return (a, len(w_), b, len(h_),
                stats.fisher_exact([[a, len(w_) - a], [b, len(h_) - b]],
                                   alternative="greater")[1])

    W("<h2>How strong is this evidence, honestly</h2>")
    W("<p>The recovery ratio has a denominator — how much AUROC the relative "
      "signs are worth on that cell — and where that is small the ratio is "
      "numerically unstable. Requiring a meaningful denominator is the right "
      "test, and it costs the result its significance.</p>")
    W('<div class="scroll"><table><tr><th>Required denominator</th><th>cells</th>'
      "<th>L-SML recovery &lt; 0.90</th><th>Fisher exact p</th></tr>")
    for mp in (0.5, 2.0, 3.0):
        a, na, b, nb, p_ = fisher_at(mp, "_recov")
        cls = "pos" if p_ < 0.05 else "neg"
        W(f"<tr><td>≥ {mp:.1f}pp</td><td>{na + nb}</td>"
          f"<td>weak {a}/{na} · healthy {b}/{nb}</td>"
          f'<td class="{cls}"><b>{p_:.4f}</b></td></tr>')
    a, na, b, nb, p_ = fisher_at(0.5, "_urec")
    W(f'<tr><td colspan="4" style="background:var(--box)">the same test on '
      f"U-PCR</td></tr>"
      f"<tr><td>≥ 0.5pp</td><td>{na + nb}</td>"
      f"<td>weak {a}/{na} · healthy {b}/{nb}</td>"
      f'<td class="neg"><b>{p_:.4f}</b></td></tr>')
    rw = [c["_relpol"] for c in weak]
    rh = [c["_relpol"] for c in heal]
    aa, bb = sum(1 for x in rw if x < 0.85), sum(1 for x in rh if x < 0.85)
    pp_ = stats.fisher_exact([[aa, len(rw) - aa], [bb, len(rh) - bb]],
                             alternative="greater")[1]
    W(f'<tr><td colspan="4" style="background:var(--box)">U-PCR polarity '
      f"agreement &lt; 0.85 — no denominator, so no stability problem</td></tr>"
      f'<tr><td>all cells</td><td>25</td><td>weak {aa}/{len(rw)} · healthy '
      f'{bb}/{len(rh)}</td><td class="neg"><b>{pp_:.4f}</b></td></tr>')
    W("</table></div>")
    W(f'<div class="box warn"><p><b>Read this as a pattern, not as a proven '
      f"effect.</b> An earlier draft of this diagnosis led with "
      f"“Fisher p = 0.0096”. That test allowed cells whose denominator is under "
      f"1pp, where the ratio is unstable; requiring ≥ 2pp gives p = "
      f'{fisher_at(2.0, "_recov")[4]:.4f} and ≥ 3pp gives '
      f'{fisher_at(3.0, "_recov")[4]:.4f}. The U-PCR version never reaches '
      f"significance, and neither does polarity agreement. With 9 weak cells "
      f"against 16, this design cannot establish an effect of this size.</p>"
      f"<p><b>What the evidence does support</b>, without a threshold or a "
      f"p-value: the three weakest QA cells are worst on <i>every</i> "
      f"sign-related measure, in <i>both</i> arms; the two arms rank the cells "
      f"the same way (Spearman +0.707, p = 0.0002); and on two cells the "
      f"pipeline cannot fuse even the label-chosen subset. That is a coherent, "
      f"reproducible pattern and a good reason to build the repair — but the "
      f"repair, not this table, is what would confirm it.</p></div>")

    # ── every cell, one row
    W("<h2>Every cell, one row — click through for the full page</h2>")
    W('<div class="scroll"><table><tr><th>Cell</th><th>n</th><th>correct</th>'
      "<th>anchor AUROC</th><th>best single view</th><th>GOOD_6</th>"
      "<th>deployed</th><th>U-PCR</th><th>oracle-5</th>"
      "<th>deployed − best view</th><th>mechanism</th></tr>")
    for grp, rows in (("the nine weak cells", weak),
                      ("the sixteen healthy cells", heal)):
        W(f'<tr><td colspan="11" style="background:var(--box);font-weight:600">'
          f"{grp}</td></tr>")
        for c in rows:
            m = mechanisms(c)
            mt = (" · ".join(f'<span class="{k}">{t}</span>' for t, k in m)
                  if m else '<span style="color:var(--mut)">none fires</span>')
            d = (c["subsets"]["deployed"]["auroc_deployed"] - c["_best_in_pool"]) * 100
            cls = ' class="weak"' if c["weak"] else ""
            W(f'<tr{cls}><td><a href="cell_{esc(c["cell"])}.html">'
              f'{esc(nm(c["cell"]))}</a></td><td>{c["n"]:,}</td>'
              f'<td>{c["pos_rate"] * 100:.1f}%</td>'
              f'<td>{c["anchor_auc_raw"]:.4f}</td>'
              f'<td>{c["_best_in_pool"]:.4f}</td>'
              f'<td>{c["subsets"]["GOOD_6"]["auroc_deployed"]:.4f}</td>'
              f'<td><b>{c["subsets"]["deployed"]["auroc_deployed"]:.4f}</b></td>'
              f'<td>{c["upcr"]["auroc"]:.4f}</td>'
              f'<td>{c["subsets"]["oracle5"]["auroc_deployed"]:.4f}</td>'
              f'{pp(d)}<td style="text-align:left;white-space:normal">{mt}</td></tr>')
    W("</table></div>")

    # ── secondary mechanisms, per cell
    W("<h2>Two secondary mechanisms, and which cells have them</h2>")
    W("<h3>The selector drops the pool's strongest view</h3>")
    W('<div class="scroll"><table><tr><th>Cell</th><th>strongest view in pool</th>'
      "<th>its AUROC</th><th>best view the selector kept</th><th>lost before "
      "fusion starts</th></tr>")
    for c in sorted(order, key=lambda x: x["_sel_miss_pp"]):
        if c["_sel_miss_pp"] > -1.0:
            continue
        cls = ' class="weak"' if c["weak"] else ""
        W(f'<tr{cls}><td><a href="cell_{esc(c["cell"])}.html">'
          f'{esc(nm(c["cell"]))}</a></td>'
          f'<td><code>{esc(c["_best_pool_name"])}</code></td>'
          f'<td>{c["_best_in_pool"]:.4f}</td>'
          f'<td>{c["_best_in_subset"]:.4f}</td>{pp(c["_sel_miss_pp"])}</tr>')
    W("</table></div>")
    W("<p>Four cells, three of them weak. On the two worst this discards more "
      "than the entire remaining gap to the oracle-5 ceiling.</p>")

    W("<h3>Views that are not monotone in the label</h3>")
    W("<p>A view can carry signal that no monotone, sign-oriented use of it can "
      "reach. Every fusion here is monotone in each view, so that signal is "
      "unreachable by construction. Measured as the cross-fitted bin-mean AUROC "
      "minus the view's own oracle-oriented AUROC.</p>")
    W('<div class="scroll"><table><tr><th>Cell</th><th>mean over the pool</th>'
      "<th>largest single view</th><th>which view</th></tr>")
    for c in sorted(order, key=lambda x: -(x["_nonmono_mean"] or -9))[:8]:
        top = max((v for v in c["views"] if v["nonmono_gain"] is not None),
                  key=lambda v: v["nonmono_gain"], default=None)
        cls = ' class="weak"' if c["weak"] else ""
        W(f'<tr{cls}><td><a href="cell_{esc(c["cell"])}.html">'
          f'{esc(nm(c["cell"]))}</a></td>'
          f'{pp((c["_nonmono_mean"] or 0) * 100)}'
          f'{pp((top["nonmono_gain"] * 100) if top else None)}'
          f'<td><code>{esc(top["name"]) if top else "—"}</code></td></tr>')
    W("</table></div>")
    coqa = cells["inside_coqa_llama7b"]
    rest = [c["_nonmono_mean"] for c in order if c["cell"] != "inside_coqa_llama7b"]
    z = (coqa["_nonmono_mean"] - np.mean(rest)) / np.std(rest)
    W(f'<div class="box warn"><p><b>CoQA is the only cell where this fires.</b> Its '
      f'pool mean is {coqa["_nonmono_mean"] * 100:+.1f}pp against a median of '
      f"{np.median(rest) * 100:+.1f}pp elsewhere and a maximum of "
      f"{max(rest) * 100:+.1f}pp over all other 24 cells — <b>z = {z:+.2f}</b>. "
      f"It is also the cell with the largest headroom in the grid.</p></div>")

    # ── the reproduction check that doubles as confirmation
    W("<h2>A reproduction check that turned into confirmation</h2>")
    W("<p>Take the <b>label-chosen</b> five views for each cell and fuse them "
      "through our ordinary label-free pipeline. If the pipeline is sound, that "
      "should land on the number the original label-using search recorded. "
      "<b>It does, exactly, on 23 of 25 cells.</b></p>")
    W('<div class="scroll"><table><tr><th>Cell</th>'
      "<th>label-using search recorded</th><th>same views, fused our way</th>"
      "<th>gap</th><th>recovery ratio</th></tr>")
    diff = [c for c in order
            if c.get("_recorded_oracle") is not None
            and abs(c["_recorded_oracle"] - c["subsets"]["oracle5"]["auroc_deployed"]) > 5e-4]
    for c in sorted(diff, key=lambda x: -(x["_recorded_oracle"]
                                          - x["subsets"]["oracle5"]["auroc_deployed"])):
        g = (c["_recorded_oracle"] - c["subsets"]["oracle5"]["auroc_deployed"]) * 100
        cls = ' class="weak"' if c["weak"] else ""
        W(f'<tr{cls}><td><a href="cell_{esc(c["cell"])}.html">'
          f'{esc(nm(c["cell"]))}</a></td>'
          f'<td>{c["_recorded_oracle"]:.4f}</td>'
          f'<td>{c["subsets"]["oracle5"]["auroc_deployed"]:.4f}</td>'
          f'{pp(-g)}<td class="neg"><b>{c["_recov"]:.3f}</b></td></tr>')
    W(f'<tr><td colspan="5" style="color:var(--mut)">the other '
      f"{len(order) - len(diff)} cells: gap exactly 0.0000</td></tr>")
    W("</table></div>")
    gaps = [(c["_recorded_oracle"] - c["subsets"]["oracle5"]["auroc_deployed"]) * 100
            for c in order if c.get("_recorded_oracle") is not None
            and c["_recov"] is not None]
    rcs = [c["_recov"] for c in order if c.get("_recorded_oracle") is not None
           and c["_recov"] is not None]
    sp = stats.spearmanr(gaps, rcs)
    W(f'<div class="box bad"><p><b>The two exceptions are the two worst-recovery '
      f"cells in the grid</b> — and that is the mechanism showing up a second "
      f"time, from a different direction. On these cells the pipeline cannot "
      f"correctly fuse even the <i>perfect</i> subset, because the problem was "
      f"never which views were chosen. Across all cells where the ratio is "
      f"defined, Spearman(gap, recovery) = <b>{sp.statistic:+.3f}</b> "
      f"(p = {sp.pvalue:.3f}).</p>"
      f"<p><b>This splits CoQA's headroom in two, and only one half is "
      f"addressable by selection.</b> Its deployed score is 0.5320. A perfect "
      f"selector would buy <b>+7.4pp</b> (to 0.6060). The remaining <b>+17.1pp</b> "
      f"to the recorded 0.7768 needs the sign recovery fixed and is not reachable "
      f"by choosing better views. Any earlier statement of “24.5pp of headroom on "
      f"CoQA” conflated the two.</p></div>")

    # ── cleared
    W("<h2>Two suspects cleared</h2>")
    W('<div class="box ok"><p><b>Orientation.</b> Fused AUROC under a true-label '
      "anchor minus the deployed one is <b>exactly 0.00pp on all 25 cells</b>, "
      "shown per cell in section 2 of every cell page. The one prior the "
      "label-free arms carry is not costing anything.</p>"
      "<p><b>K-selection.</b> The degeneracy flag — the winning K beating the "
      "runner-up by less than floating-point noise — fires on <b>0 of 25</b> "
      "cells on the deployed subset. The grouping is a measurement everywhere "
      "here, not a coin flip. An earlier lead pointed the other way, but it sits "
      "on <code>ALL_H16</code>, a 16-view subset we do not deploy.</p></div>")

    # ── none fires
    none = [c for c in weak if not mechanisms(c)]
    if none:
        W("<h2>Three of the nine have no defect at all</h2>")
        W('<div class="box"><p>'
          + ", ".join(f'<a href="cell_{esc(c["cell"])}.html">{esc(nm(c["cell"]))}</a>'
                      for c in none)
          + " trip none of the four mechanisms: recovery at or above 0.90, the "
          "selector keeps the strongest view, the views are monotone, the "
          "grouping is determinate. <b>They score low because the signal is "
          "weak, not because anything is broken.</b> Their remaining headroom is "
          + ", ".join(f'{(c["subsets"]["oracle5"]["auroc_deployed"] - c["subsets"]["deployed"]["auroc_deployed"]) * 100:.1f}pp'
                      for c in none)
          + " respectively — real, but reachable only with labels.</p>"
          "<p>So <b>“why do we fail here” has two different answers</b>: six cells "
          "have a named, fixable defect; three are simply hard, and there the "
          "honest move is to report the ceiling rather than chase it.</p></div>")

    # ── the repair, run
    rp = json.load(open(os.path.join(DATA, "_repair.json"), encoding="utf-8"))
    rr = {r["cell"]: r for r in rp["rows"]}
    W("<h2>Repair 1 was built and run — and it is refuted</h2>")
    W('<div class="box bad"><p><b>The pre-registration said the repair, not this '
      "diagnosis, was the confirmation. It was run, and it did not confirm.</b> "
      "Z<sub>2</sub> synchronisation as a replacement label-free sign estimator "
      "<b>fails both gate conditions on both arms</b>.</p></div>")
    W("<h3>The premise check that should have run first</h3>")
    W(f'<div class="box warn"><p>L-SML is <b>exactly</b> sign-gauge invariant on '
      f"today's data: fusing the deployed subset after applying the Z<sub>2</sub> "
      f"signs, or a <i>random</i> ±1 sign vector, changes the AUROC by "
      f"<b>{rp['invariance_max']:.2e} on all 25 cells</b>. So “feed L-SML a "
      f"better sign estimate” was a <b>no-op by construction</b> and could never "
      f"have applied to that arm. This was already in the project glossary and "
      f"should have been checked before the repair was pre-registered. Z"
      f"<sub>2</sub> was therefore tested where signs actually bind — as a "
      f"<i>replacement</i> for L-SML, and for U-PCR's own sign(ρ̂) step.</p></div>")
    W("<h3>The gate</h3>")
    W('<div class="scroll"><table><tr><th>Failing cell</th>'
      "<th>recovery before</th><th>after (Z₂ + average)</th>"
      "<th>ΔAUROC</th><th>after (Z₂ in U-PCR)</th><th>ΔAUROC</th></tr>")
    for ckf in ("inside_coqa_llama7b", "seiclr_triviaqa_opt30b",
                "ars_gsm8k_r1distill8b", "noise_gsm8k_phi3mini"):
        r = rr[ckf]
        W(f'<tr class="weak"><td><a href="cell_{esc(ckf)}.html">'
          f"{esc(nm(ckf))}</a></td>"
          f'<td>{num(r["rec_lsml"], 3)}</td><td>{num(r["rec_z2avg"], 3)}</td>'
          f'{pp(r["d_z2avg"])}<td>{num(r["rec_z2upcr"], 3)}</td>'
          f'{pp(r["d_z2upcr"])}</tr>')
    W("</table></div>")
    W("<p><b>Condition (i)</b> — recovery ≥ 0.90 on all four: <b class='neg'>1 of "
      "4</b> for each arm. <b>Condition (ii)</b> — healthy cells move under "
      "0.5pp: <b class='neg'>13 of 16</b> (Z₂ + average) and <b class='neg'>15 of "
      "16</b> (Z₂ in U-PCR). The collateral is real: "
      "<code>semenergy_triviaqa_qwen3_8b</code> −3.30pp and "
      "<code>math500_qwenmath7b</code> −2.31pp.</p>")
    W("<h3>Full regression, all 25 cells</h3>")
    W('<div class="scroll"><table><tr><th>Arm</th><th>macro</th><th>QA</th>'
      "<th>math</th></tr>")
    for k, lbl in (("base_lsml", "DUFS + L-SML (baseline)"),
                   ("base_upcr", "U-PCR + sign(ρ̂) (baseline) — still the best"),
                   ("z2_avg", "Z₂ + simple average, deployed subset"),
                   ("z2_avg_full", "Z₂ + simple average, full pool"),
                   ("z2_upcr", "Z₂ inside U-PCR")):
        vals = [rr[c["cell"]][k] for c in order]
        qa = [rr[c["cell"]][k] for c in order if c["group"] == "QA"]
        mt = [rr[c["cell"]][k] for c in order if c["group"] == "math"]
        b = " style='font-weight:600'" if k == "base_upcr" else ""
        W(f"<tr{b}><td>{lbl}</td><td>{np.mean(vals):.4f}</td>"
          f"<td>{np.mean(qa):.4f}</td><td>{np.mean(mt):.4f}</td></tr>")
    W("</table></div>")
    W("<p>Paired against their own baselines: Z₂+average <b>−0.14pp</b> "
      "(7W/18L, p = 0.095), Z₂+average on the full pool <b>+0.04pp</b> "
      "(12W/13L, p = 0.895 — a dead wash), Z₂ inside U-PCR <b>−0.22pp</b> "
      "(2W/5L, p = 0.128). Arm B is <i>exactly</i> +0.00pp on 15 of 25 cells: Z₂ "
      "and sign(ρ̂) return the same polarities on most cells, and where they "
      "differ Z₂ is the worse of the two.</p>")
    W('<div class="box bad"><p><b>What this does to the mechanism.</b> The '
      "description survives — the three weakest QA cells are still worst on every "
      "sign-related measure in both arms. The <i>actionable</i> version does not: "
      "a better label-free sign estimator does not recover those cells. And "
      "calling the r4−r3 gap “sign recovery” oversold it — L-SML does not "
      "<i>recover</i> signs, it is <b>invariant</b> to them, so the gap measures "
      "how much better a sign-invariant estimator is than a sign-sensitive one "
      "fed the wrong signs. Fair as a normalisation; it does not license “fix the "
      "sign estimate and the cell improves”, which is exactly what failed.</p>"
      "<p>The remaining reading is closer to <b>“there is not enough covariance "
      "structure on these cells for any label-free method”</b> than to a fixable "
      "defect — consistent with the three weak cells that trip nothing at all, "
      "and an argument for reporting ceilings rather than chasing them.</p>"
      "<p><b>The one durable positive:</b> CoQA gains <b>+2.40pp</b> "
      "(0.5320 → 0.5560) under Z₂ + simple average — the largest single-cell gain "
      "in the test, on the flagship failing cell. But its recovery ratio only "
      "reaches 0.038, and its selection headroom is +7.4pp, so this is a small "
      "part of a small part.</p></div>")

    # ── pre-registration
    W("<h2>The other two repairs — still pre-registered, not run</h2>")
    W('<div class="box"><p>Written down before any was tested, so the diagnosis '
      "cannot be tuned to make a fix look good.</p><ol>"
      "<li><s><b>A better label-free relative-sign estimator</b> — "
      "Z<sub>2</sub> synchronisation</s> — <b class='neg'>CLOSED AS REFUTED "
      "above.</b></li>"
      "<li><b>Rank / quantile transform each view before fusion</b> — CoQA only. "
      "<b>Gate:</b> a no-op (&lt;0.5pp) on the 24 cells where non-monotonicity is "
      "≈0, or it is buying CoQA with everything else.</li>"
      "<li><b>Keep the pool's strongest view unconditionally</b> — the four "
      "selection-miss cells. Label-free only if “strongest” can be decided "
      "without labels, which may sink it. <b>Now the one most worth running</b>: "
      "the selection miss is a measured −4.8pp on the worst cell and does not "
      "depend on the sign story at all.</li></ol>"
      "<p><b>Not proposed:</b> any K-selection change, anything touching "
      "orientation, anything touching pool composition — all three are cleared or "
      "closed above.</p></div>")

    W(f'<div class="foot">GOOD_6 validity anchor {idx["good6_macro"]:.4f} '
      "(expected 0.7594). Both deployed arms reproduce their recorded per-cell "
      "values to &lt;5e-4 on 25/25. Every K and residual recomputed under current "
      "code — none joined from a bench CSV, all of which predate the Step-205 "
      "grouping fix.</div>")
    return page("Item 1 — why we fail where we fail", "".join(A))


# ── items 2 and 3 ────────────────────────────────────────────────────────────
def item2_page():
    A = []
    W = A.append
    W('<p class="crumb"><a href="../index.html">Jul-2026 action items</a> › '
      "Item 2</p>")
    W("<h1>Item 2 — a clustering mechanism inside U-PCR</h1>")
    W('<p class="sub">Already built and refuted in Step 204. This page is the '
      "evidence, so the decision does not have to be taken on trust.</p>")
    W('<div class="box bad"><p><b>Verdict: built, tested, refuted — and its '
      "premise was a confound.</b> The code is still in the tree at "
      "<code>spectral_utils/upcr_clustered.py</code>.</p></div>")

    W("<h2>What was built</h2>")
    W("<p>U-PCR assumes the views' errors are uncorrelated. The relaxation is to "
      "assume that only <i>across</i> L-SML clusters, and to fit the additive "
      "system on cross-cluster pairs only — which is exactly the “add clustering "
      "to U-PCR” idea. An identifiability requirement was derived and enforced: "
      "the cross-cluster pair graph is complete multipartite, so <b>K ≥ 3</b> is "
      "required; at K = 2 it is bipartite and the parameter is unidentifiable.</p>")

    W("<h2>What happened</h2>")
    W('<div class="scroll"><table><tr><th>Test</th><th>Result</th></tr>'
      "<tr><td>Both pre-registered gates</td><td class='neg'>FAILED</td></tr>"
      "<tr><td>AUROC against the unclustered arm</td>"
      "<td class='neg'>−4.46pp, 9W/16L, p = 0.030</td></tr>"
      "<tr><td>Premise: same-cluster vs cross-cluster fit error</td>"
      "<td>raw ratio 2.03×</td></tr>"
      "<tr><td>… the same ratio matched on |C<sub>ij</sub>| decile</td>"
      "<td class='neg'>0.97–1.00× — the gap disappears</td></tr>"
      "<tr><td>… a random partition instead of the clusters</td>"
      "<td class='neg'>reproduces the raw gap</td></tr>"
      "<tr><td>… magnitude-only clustering</td>"
      "<td class='neg'>separates it <i>better</i> (3.06–3.81×)</td></tr>"
      "</table></div>")
    W("<p>So the apparent same-vs-cross structure was pair correlation wearing a "
      "cluster label (fit error tracks |C<sub>ij</sub>| at Spearman 0.870). Once "
      "you control for correlation magnitude there is no clustering effect left "
      "to exploit.</p>")

    W("<h2>The one variant never run, and why it is rated low</h2>")
    W("<p>K-means on the two-component eigenvector coordinates "
      "(v₁[i], v₂[i]) would recover hard groups from the U-PCR fit rather than "
      "importing L-SML's. It has not been tried. Two measurements make it "
      "unpromising:</p><ul>"
      "<li><b>One-component U-PCR is exactly PC1 of the surviving views</b> "
      "(cosine deviation 7e-12), so the whole ρ / g² apparatus enters only "
      "through the exclusion mask.</li>"
      "<li><b>The second component is inert on our data</b>: sweeping "
      "<code>lambda2_threshold</code> from 0.05 to 0.25 changes the component "
      "count on 24 of 25 cells and buys +0.43pp (9W/15L, p = 0.16).</li>"
      "</ul><p>There is nothing for the second component to cluster on.</p>")
    W('<div class="foot">Evidence: HISTORY.md Step 204 §D and Step 205. '
      "Code: <code>spectral_utils/upcr_clustered.py</code>.</div>")
    return page("Item 2 — clustering inside U-PCR", "".join(A))


def item3_page():
    A = []
    W = A.append
    W('<p class="crumb"><a href="../index.html">Jul-2026 action items</a> › '
      "Item 3</p>")
    W("<h1>Item 3 — adjacent applications</h1>")
    W('<p class="sub">Two were named: detecting a hallucination early in '
      "generation, and localizing it. They are in very different states.</p>")

    W("<h2>Early detection — has a replicated effect already</h2>")
    W('<div class="box ok"><p>On the canonical fresh raw-trace cache, the '
      "16-view L-SML score beats the best DeepConf window by <b>+5.6pp "
      "[+0.9, +10.6]</b> (paired bootstrap, CI excludes zero) at the "
      "<b>earliest 10% of the trace</b>, and the earliness index reaches "
      "full-trace AUROC at that budget. This is the surviving finding of the "
      "Step-148 streaming pilot, re-run on a clean cache.</p></div>")
    W("<p>Why this is the strongest publishable arm: the current headline is a "
      "detector that <i>ties</i> two other label-free arms and beats its own cost "
      "class by an insignificant margin. “Same detection at a tenth of the trace” "
      "is a different claim, with a real effect size and a cost axis (tokens) "
      "where we are not competing against supervised probes.</p>")
    W("<h3>What it is missing, and what supplies it</h3>")
    W("<p>The pilot scores prefixes at <b>fixed budgets</b> and has no stopping "
      "rule. <i>Online Auditing of Information Flow</i> (Oren-Loberman, Azar, "
      "Huleihel; arXiv:2310.14595, IEEE TSIPN 2024) formulates exactly this: "
      "sequential detection under a risk that prices <b>both error and delay</b>, "
      "where the optimal rule is a two-sided threshold on the posterior — a "
      "Wald-calibrated SPRT. Adopted in Step 208 <b>for the formulation, not the "
      "theorems</b>: its offline stage is supervised, and its graph machinery does "
      "not transfer to a single fully-observed decoded trace.</p>")
    W('<div class="box warn"><p><b>The metric lesson, which changes how we '
      "report.</b> That paper's accuracy is a wash against its baseline "
      "(0.86 vs 0.85); its entire contribution is <b>6.29 vs 12.75 events to "
      "decide</b>. So the metric is <b>(AUROC at budget, tokens consumed)</b> — "
      "never AUROC alone. Reporting only AUROC would hide the whole result.</p>"
      "</div>")

    W("<h2>Localization — deferred, and the blocker is data</h2>")
    W("<p>Extension F (step-level error localization) is written up but not "
      "started. It needs step-level annotation we do not have; AgentHallu "
      "supplies a schema (hallucination_step, category, reason) and shows the "
      "problem is open — Gemini 2.5 Pro reaches 41.1% step localization, and "
      "11.6% on tool use. Real headroom, but it is an annotation project before "
      "it is a method project.</p>")

    W("<h2>Suggested order</h2>")
    W("<p>Early detection first: the effect is measured, the formulation is "
      "adopted, the data exists, and the re-run is CPU-only against caches we "
      "already hold. Localization stays deferred until there is a reason to buy "
      "annotation.</p>")
    W('<div class="foot">Evidence: HISTORY.md Steps 148, 152 (A2), 208; '
      "Research_Directions.md Extensions E and F.</div>")
    return page("Item 3 — adjacent applications", "".join(A))


def root_index(cells):
    A = []
    W = A.append
    W("<h1>Advisor meeting — Jul 2026: action items</h1>")
    W('<p class="sub">Three items came out of the meeting that closed the '
      "feature-selection line. One page per item.</p>")
    W('<div class="box"><p><b>Why the direction closed.</b> L-SML over the full '
      "~30-view pool, L-SML after DUFS selection, and U-PCR's own built-in "
      "exclusion all land within noise of each other on essentially every cell: "
      "0.7551 / 0.7507 / 0.7594 macro, with no pairwise contrast significant "
      "(p = 0.059 / 0.191 / 0.615).</p></div>")
    W('<div class="cards">')
    W('<div class="card"><h3><a href="item1_failure_deepdive/index.html">'
      "1 · Why we fail where we fail</a></h3>"
      "<p>Nine cells, one page each: the views, their distributions, the anchor, "
      "the subsets, and where the AUROC is lost. The sign-recovery mechanism "
      "proposed here was <b>withdrawn</b> in Steps 212–213 — its significance "
      "was an artifact and its repair failed. The eliminations stand.</p></div>")
    W('<div class="card"><h3><a href="item1b_feature_comparison/index.html">'
      "1b · Features, or algorithm?</a></h3>"
      "<p>Matched cell pairs — a weak cell against a high-scoring one that holds "
      "the dataset or the pipeline fixed. Distributions, correlation structure, "
      "and the supervised ceiling, to see whether the gap predates the "
      "fusion.</p></div>")
    W('<div class="card"><h3><a href="item2_upcr_clustering/index.html">'
      "2 · Clustering inside U-PCR</a></h3>"
      "<p>Already built and refuted (−4.46pp, p = 0.030), and its premise was a "
      "confound. The evidence, so the decision is not taken on trust.</p></div>")
    W('<div class="card"><h3><a href="item3_adjacent_applications/index.html">'
      "3 · Adjacent applications</a></h3>"
      "<p>Early detection already has +5.6pp [+0.9, +10.6] at the earliest 10% "
      "of the trace. Localization is blocked on annotation.</p></div>")
    W("</div>")
    W('<div class="foot">Built from <code>local_cache/</code> through the '
      "canonical scoring path. GOOD_6 validity anchor 0.7594; both deployed arms "
      "reproduce to &lt;5e-4 on 25/25 cells.</div>")
    return page("Jul-2026 advisor action items", "".join(A))


def main():
    idx, cells = load()
    for d in (I1, I2, I3):
        os.makedirs(d, exist_ok=True)
    ro = recorded_oracle()

    # per-cell derived quantities, all local
    for c in cells.values():
        c["_recorded_oracle"] = ro.get(c["cell"])
        enrich(c)
        dep = c["subsets"]["deployed"]
        vs = {v["name"]: v for v in c["views"]}
        sub = [vs[f] for f in dep["feats"]]
        ng = [v["nonmono_gain"] for v in c["views"] if v["nonmono_gain"] is not None]
        c["_nonmono_mean"] = float(np.mean(ng)) if ng else None
        c["_ladder"] = dict(
            r1=c["_best_in_subset"], r2=dep["_avg_oracle"], r3=dep["_avg_none"],
            r4=dep["auroc_true_anchor"], r5=dep["auroc_deployed"],
            r6=c["subsets"]["oracle5"]["auroc_deployed"])
        cost = (c["_ladder"]["r2"] - c["_ladder"]["r3"]) * 100
        c["_cost_pp"] = round(cost, 2)
        c["_unstable"] = cost < MIN_DENOM_PP
        c["_recov"] = (round(((c["_ladder"]["r4"] - c["_ladder"]["r3"]) * 100) / cost, 3)
                       if cost > 0.5 else None)
        c["_urec"] = c["upcr"]["ladder"]["recovery"]
        c["_usign"] = c["upcr"]["ladder"]["sign_step_recovery"]
        c["_relpol"] = round(rel_polarity(c), 3)

    for ck, c in cells.items():
        with open(os.path.join(I1, f"cell_{ck}.html"), "w", encoding="utf-8") as f:
            f.write(cell_page(c, cells))
    with open(os.path.join(I1, "index.html"), "w", encoding="utf-8") as f:
        f.write(item1_index(idx, cells))
    with open(os.path.join(I2, "index.html"), "w", encoding="utf-8") as f:
        f.write(item2_page())
    with open(os.path.join(I3, "index.html"), "w", encoding="utf-8") as f:
        f.write(item3_page())
    with open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8") as f:
        f.write(root_index(cells))

    tot = sum(os.path.getsize(os.path.join(dp, fn))
              for dp, _, fns in os.walk(ROOT) for fn in fns if fn.endswith(".html"))
    print(f"wrote {len(cells)} cell pages + 4 index pages "
          f"({tot / 1024 / 1024:.1f} MB total) under {ROOT}")


if __name__ == "__main__":
    main()
