#!/usr/bin/env python
"""
localization_report.py — assemble the U-PCR vs "Mind the Gap" report page.

Reads only files that other scripts wrote, and renders whatever exists. Anything not yet run is
shown as PENDING with what it is waiting on, so the page is publishable at every stage instead of
only at the end.

  results/localization/evdrop/*__evdrop.csv          answer-level (their Tables 1 and 2)
  results/localization/evdrop/*__diagnostics.json    operating point vs their reported accuracy
  results/localization/processbench/*__processbench.csv   step-level (their Table 3)
  results/localization/processbench/*__diagnostics.json   step lengths + per-view availability
  results/localization/*__examples.json              the marked-up traces

Usage:
    python scripts/localization/localization_report.py [--out results/localization/report.html]
"""
import argparse
import csv
import glob
import html
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from our_arm import _main_tree_root  # noqa: E402

# Results live in the MAIN tree, not the worktree. Every other script in this project writes
# under `<main tree>/results`, `sync_code.sh` excludes that directory from the cluster push, and
# a worktree-local copy would silently fork the numbers. When this file already IS the main tree
# the two resolve to the same path.
RES = os.path.join(_main_tree_root(), "results", "localization")

# Their Table 3, best column (Shannon Drop). The targets this whole exercise is measured against.
PAPER_SLA = {
    ("Qwen3-8B", "gsm8k"): 46.11, ("Qwen3-8B", "math"): 32.90,
    ("Qwen3-8B", "olympiadbench"): 41.52, ("Qwen3-8B", "omnimath"): 37.04,
    ("Qwen3-4B", "gsm8k"): 43.42, ("Qwen3-4B", "math"): 32.03,
    ("Qwen3-4B", "olympiadbench"): 43.06, ("Qwen3-4B", "omnimath"): 38.04,
}

PRETTY = {
    "ours_UPCR_fullpool": ("U-PCR (full pool)", "ours"),
    "upcr_positional": ("U-PCR — positional series", "ours"),
    "upcr_step": ("U-PCR — step as item", "ours"),
    "upcr_token": ("U-PCR — token trace", "ours"),
    "a2_dufs_pf": ("DUFS + L-SML", "ours"),
    "ref_lsml_GOOD_6": ("L-SML GOOD_6", "ref"),
    "ref_lsml_GOOD_5": ("L-SML GOOD_5", "ref"),
    "ref_lsml_H16": ("L-SML H16", "ref"),
    "shannon_avg": ("Shannon Avg", "paper"), "shannon_drop": ("Shannon Drop", "paper"),
    "logtoku_avg": ("LogTokU Avg (Eq. 47 verbatim)", "paper"),
    "logtoku_oriented_avg": ("LogTokU Avg (sign repaired)", "paper"),
    "logtoku_drop": ("LogTokU Drop", "paper"),
    "logtoku_oriented_drop": ("LogTokU Drop (sign repaired)", "paper"),
    "ln_s_avg": ("LN-S Avg", "paper"), "ln_s_drop": ("LN-S Drop", "paper"),
}

# Cells whose accuracy sits outside the project's own ACC_BAND (0.20, 0.85). Every one of these
# is a CEILING cell: at 79-94% accuracy there are few negatives, every selective metric is
# compressed, and the CIs are wide. Flagged in the tables rather than mentioned in a caption.
CEILING = {"evdrop_gsm8k_qwen3_8b", "evdrop_gsm8k_qwen3_4b"}
CELL_PRETTY = {
    "evdrop_gsm8k_qwen3_8b": ("GSM8K", "Qwen3-8B"),
    "evdrop_gsm8k_qwen3_4b": ("GSM8K", "Qwen3-4B"),
    "evdrop_math_qwen3_8b": ("MATH", "Qwen3-8B"),
    "evdrop_math_qwen3_4b": ("MATH", "Qwen3-4B"),
}


def esc(s):
    return html.escape(str(s), quote=True)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x, nd=2, dash="—"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return dash if not np.isfinite(v) else f"{v:.{nd}f}"


# ── figure primitives ────────────────────────────────────────────────────────

def svg_open(w, h, cls="fig"):
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img">')


def lin(v, d0, d1, r0, r1):
    if d1 == d0:
        return (r0 + r1) / 2
    return r0 + (float(v) - d0) * (r1 - r0) / (d1 - d0)


def evidence_svg(panel, w=880, h=210):
    """The token-level fused evidence curve, with step boundaries and the worst drops marked."""
    ev = [v for v in panel["smoothed"] if v is not None]
    if len(ev) < 3:
        return "<p class='note'>trace too short to plot</p>"
    full = [np.nan if v is None else float(v) for v in panel["smoothed"]]
    n = len(full)
    lo, hi = float(np.nanmin(full)), float(np.nanmax(full))
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 46, 14, 16, 30
    x = lambda i: lin(i, 0, n - 1, L, w - R)          # noqa: E731
    y = lambda v: lin(v, lo, hi, h - B, T)            # noqa: E731

    out = [svg_open(w, h, "fig trace")]
    # step boundaries — faint, so the curve is read against the text above it
    for span in panel["step_token_spans"]:
        if span:
            out.append(f'<line class="stepline" x1="{x(span[0]):.1f}" y1="{T}" '
                       f'x2="{x(span[0]):.1f}" y2="{h-B}"/>')
    out.append(f'<line class="axis" x1="{L}" y1="{h-B}" x2="{w-R}" y2="{h-B}"/>')
    out.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{h-B}"/>')

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(full) if np.isfinite(v))
    out.append(f'<polyline class="curve" points="{pts}"/>')

    for d in panel["drops"]:
        j = int(d["token"])
        if not (0 <= j < n) or not np.isfinite(full[j]):
            continue
        out.append(f'<line class="dropline" x1="{x(j):.1f}" y1="{T}" '
                   f'x2="{x(j):.1f}" y2="{h-B}"/>')
        out.append(f'<circle class="dropdot" cx="{x(j):.1f}" cy="{y(full[j]):.1f}" r="3.5"/>')
        anchor = "end" if x(j) > w * 0.72 else "start"
        dx = -6 if anchor == "end" else 6
        step_txt = "—" if d["step"] is None else f"step {d['step']}"
        out.append(f'<text class="droplab" x="{x(j)+dx:.1f}" y="{T+13}" text-anchor="{anchor}">'
                   f'Δ {d["flux"]:.2f} · tok {j} · {step_txt}</text>')

    out.append(f'<text class="axlab" x="{L-8}" y="{T+9}" text-anchor="end">{hi:.1f}</text>')
    out.append(f'<text class="axlab" x="{L-8}" y="{h-B}" text-anchor="end">{lo:.1f}</text>')
    out.append(f'<text class="axlab" x="{L}" y="{h-8}">token 0</text>')
    out.append(f'<text class="axlab" x="{w-R}" y="{h-8}" text-anchor="end">token {n-1}</text>')
    out.append('<text class="axttl" x="12" y="'
               f'{(T+h-B)/2:.0f}" transform="rotate(-90 12 {(T+h-B)/2:.0f})" '
               'text-anchor="middle">fused evidence</text>')
    out.append("</svg>")
    return "".join(out)


def steps_html(panel):
    """The reasoning chain, each step tinted by its own fused risk."""
    risk = [np.nan if v is None else float(v) for v in panel["step_risk"]]
    finite = [v for v in risk if np.isfinite(v)]
    top = max(finite) if finite else 1.0
    peak = panel["peak_step"]
    gt = panel.get("label", -1)
    rows = []
    for i, (txt, r) in enumerate(zip(panel["steps"], risk)):
        alpha = 0.0 if not np.isfinite(r) or top <= 0 else min(1.0, (r / top) ** 0.65)
        marks = []
        if i == peak:
            marks.append('<span class="tag tag-pick">detector&nbsp;pick</span>')
        if gt >= 0 and i == gt:
            marks.append('<span class="tag tag-gt">annotated&nbsp;error</span>')
        cls = "step" + (" step-peak" if i == peak else "") + (" step-gt" if gt >= 0 and i == gt else "")
        rows.append(
            f'<div class="{cls}" style="--tint:{alpha:.3f}">'
            f'<div class="step-meta"><span class="step-i">{i}</span>'
            f'<span class="step-r">{"—" if not np.isfinite(r) else f"{r:.3f}"}</span></div>'
            f'<div class="step-body">{esc(txt)}{"".join(marks)}</div></div>')
    return '<div class="chain">' + "".join(rows) + "</div>"


def separability_svg(ex, w=880, h=230):
    """Peak step risk, incorrect vs correct answers — their Figure 3, on our arm."""
    inc = np.array([v for v in ex["separability"]["incorrect"] if v is not None], dtype=float)
    cor = np.array([v for v in ex["separability"]["correct"] if v is not None], dtype=float)
    if inc.size < 3 or cor.size < 3:
        return "<p class='note'>not enough rows to plot</p>"
    hi = float(np.percentile(np.concatenate([inc, cor]), 99))
    bins = np.linspace(0, hi, 34)
    hc, _ = np.histogram(cor.clip(0, hi), bins=bins)
    hi_, _ = np.histogram(inc.clip(0, hi), bins=bins)
    hc = hc / max(hc.max(), 1)
    hi_ = hi_ / max(hi_.max(), 1)
    L, R, T, B = 46, 14, 18, 34
    x = lambda v: lin(v, 0, hi, L, w - R)            # noqa: E731
    y = lambda v: lin(v, 0, 1, h - B, T)             # noqa: E731

    out = [svg_open(w, h, "fig hist")]
    out.append(f'<line class="axis" x1="{L}" y1="{h-B}" x2="{w-R}" y2="{h-B}"/>')
    for arr, cls in ((hc, "barok"), (hi_, "barbad")):
        pts = [f"{x(bins[0]):.1f},{y(0):.1f}"]
        for k, v in enumerate(arr):
            pts.append(f"{x(bins[k]):.1f},{y(v):.1f}")
            pts.append(f"{x(bins[k+1]):.1f},{y(v):.1f}")
        pts.append(f"{x(bins[-1]):.1f},{y(0):.1f}")
        out.append(f'<polygon class="{cls}" points="{" ".join(pts)}"/>')
    for arr, cls, lab in ((cor, "medok", "correct"), (inc, "medbad", "incorrect")):
        m = float(np.median(arr))
        out.append(f'<line class="{cls}" x1="{x(m):.1f}" y1="{T}" x2="{x(m):.1f}" y2="{h-B}"/>')
        out.append(f'<text class="medlab {cls}" x="{x(m):.1f}" y="{T+11}" '
                   f'text-anchor="middle">{lab} median {m:.2f}</text>')
    for t in np.linspace(0, hi, 5):
        out.append(f'<text class="axlab" x="{x(t):.1f}" y="{h-14}" text-anchor="middle">'
                   f'{t:.1f}</text>')
    out.append(f'<text class="axttl" x="{(L+w-R)/2:.0f}" y="{h-2}" text-anchor="middle">'
               'peak step risk within an answer</text>')
    out.append("</svg>")
    return "".join(out)


# ── tables ───────────────────────────────────────────────────────────────────

def answer_tables(rows_by_cell, diags):
    if not rows_by_cell:
        return pending("Answer-level tables",
                       "no <code>*__evdrop.csv</code> yet — run "
                       "<code>score_evdrop.py</code> on a fetched cell")
    cells = list(rows_by_cell)
    methods, seen = [], set()
    for c in cells:
        for r in rows_by_cell[c]:
            if r["method"] not in seen:
                seen.add(r["method"])
                methods.append(r["method"])
    methods.sort(key=lambda m: ({"ours": 0, "paper": 1, "ref": 2}.get(PRETTY.get(m, ("", "x"))[1], 3),
                                PRETTY.get(m, (m,))[0]))

    def table(metric, nd, better_low, cap):
        head = "".join(f'<th colspan="1">{esc(CELL_PRETTY.get(c, (c, ""))[0])}<br>'
                       f'<span class="sub">{esc(CELL_PRETTY.get(c, ("", c))[1])}</span></th>'
                       for c in cells)
        body = []
        best = {}
        for c in cells:
            vals = [(float(r[metric]), r["method"]) for r in rows_by_cell[c]
                    if r.get(metric) not in (None, "") and np.isfinite(float(r[metric]))]
            if vals:
                best[c] = (min if better_low else max)(vals)[1]
        for m in methods:
            kind = PRETTY.get(m, ("", "other"))[1]
            tds = []
            for c in cells:
                hit = [r for r in rows_by_cell[c] if r["method"] == m]
                if not hit:
                    tds.append('<td class="num">—</td>')
                    continue
                v = fnum(hit[0][metric], nd)
                cls = "num" + (" best" if best.get(c) == m else "")
                tds.append(f'<td class="{cls}">{v}</td>')
            body.append(f'<tr class="k-{kind}"><td class="mname">{esc(PRETTY.get(m, (m,))[0])}'
                        f'<span class="kind k-{kind}">{kind}</span></td>{"".join(tds)}</tr>')
        return (f'<figure class="tbl"><div class="tscroll"><table>'
                f'<thead><tr><th>method</th>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>'
                f'<figcaption>{cap}</figcaption></figure>')

    op = []
    for c in cells:
        d = diags.get(c, {})
        dg, mf = d.get("diagnostics", {}), d.get("manifest", {})
        pa = (mf.get("published") or {}).get("pretrained_accuracy")
        acc = dg.get("accuracy")
        gap = (acc * 100 - pa) if (acc and pa) else None
        flag = "" if gap is None else ("ok" if abs(gap) < 3 else "warn")
        op.append(
            f'<tr><td>{esc(CELL_PRETTY.get(c, (c, ""))[0])} · '
            f'{esc(CELL_PRETTY.get(c, ("", c))[1])}</td>'
            f'<td class="num">{dg.get("n", "—")}</td>'
            f'<td class="num">{fnum((acc or 0)*100, 2)}</td>'
            f'<td class="num">{fnum(pa, 2)}</td>'
            f'<td class="num {flag}">{"—" if gap is None else f"{gap:+.2f}"}</td>'
            f'<td class="num">{dg.get("n_incorrect", "—")}</td>'
            f'<td class="num">{fnum((dg.get("frac_pinned_in_negatives") or 0)*100, 1)}%</td></tr>')

    return f"""
<figure class="tbl"><div class="tscroll"><table>
<thead><tr><th>cell</th><th>n</th><th>our accuracy</th><th>their accuracy</th><th>gap (pp)</th>
<th>n incorrect</th><th>capped, within incorrect</th></tr></thead>
<tbody>{''.join(op)}</tbody></table></div>
<figcaption>Operating point. Selective accuracy and AURC are both monotone in the base error
rate, so a cell whose accuracy differs from theirs is not directly comparable no matter how
faithful the estimator is. Under 3&nbsp;pp is treated as comparable.</figcaption></figure>
{table("aurc_x1000", 1, True,
       "AURC &times;1000, lower is better — their Table 2. Bold is the best method in that cell.")}
{sel_acc_table(rows_by_cell, cells, methods)}
"""


def sel_acc_table(rows_by_cell, cells, methods):
    """Their Table 1: selective accuracy at all three alphas, cell x alpha."""
    alphas = ("0.05", "0.1", "0.5")
    head = "".join(
        f'<th>{esc(CELL_PRETTY.get(c, (c, ""))[0])} · {esc(CELL_PRETTY.get(c, ("", c))[1])}'
        f'<br><span class="sub">&alpha;={a}</span></th>'
        for c in cells for a in alphas)
    best = {}
    for c in cells:
        for a in alphas:
            vals = [(float(r[f"sel_acc@{a}"]), r["method"]) for r in rows_by_cell[c]
                    if r.get(f"sel_acc@{a}") not in (None, "")
                    and np.isfinite(float(r[f"sel_acc@{a}"]))]
            if vals:
                best[(c, a)] = max(vals)[1]
    body = []
    for m in methods:
        kind = PRETTY.get(m, ("", "other"))[1]
        tds = []
        for c in cells:
            hit = [r for r in rows_by_cell[c] if r["method"] == m]
            for a in alphas:
                if not hit:
                    tds.append('<td class="num">—</td>')
                    continue
                sd = fnum(hit[0].get(f"sel_acc_sd@{a}"), 2, "")
                cls = "num" + (" best" if best.get((c, a)) == m else "")
                tds.append(f'<td class="{cls}">{fnum(hit[0][f"sel_acc@{a}"], 2)}'
                           f'<span class="sd">±{sd}</span></td>')
        body.append(f'<tr class="k-{kind}"><td class="mname">{esc(PRETTY.get(m, (m,))[0])}'
                    f'<span class="kind k-{kind}">{kind}</span></td>{"".join(tds)}</tr>')
    return (f'<figure class="tbl"><div class="tscroll"><table>'
            f'<thead><tr><th>method</th>{head}</tr></thead><tbody>{"".join(body)}</tbody>'
            f'</table></div><figcaption>Selective accuracy (%), higher is better &mdash; their '
            f'Table 1. The threshold is the &alpha;-quantile of the risk distribution of the '
            f'INCORRECT calibration samples only, averaged over 200 random 50/50 splits; the '
            f'&plusmn; is the spread across those splits, which with greedy decoding is the only '
            f'source of variance. At these base accuracies (91&ndash;94%) the top methods are '
            f'separated by less than one split-to-split standard deviation, so AURC above is the '
            f'discriminating column and this one is the compatibility view.</figcaption></figure>')


def processbench_tables(rows_by_subset, diags):
    if not rows_by_subset:
        return pending("Step-level localization (their Table 3)",
                       "ProcessBench inference has not produced a cell yet — needs "
                       "<code>submit_teacher_forced.sbatch</code> to run past Gate&nbsp;B")
    subsets = list(rows_by_subset)
    methods, seen = [], set()
    for s in subsets:
        for r in rows_by_subset[s]:
            if r["method"] not in seen:
                seen.add(r["method"])
                methods.append(r["method"])
    methods.sort(key=lambda m: ({"ours": 0, "paper": 1}.get(PRETTY.get(m, ("", "x"))[1], 3),
                                PRETTY.get(m, (m,))[0]))

    # One grid per subset: COLUMNS are methods, ROWS are metrics, the direction is in the row
    # label, and the winning cell is marked in the cell itself rather than left to be inferred.
    METRICS = [("sla", "SLA — exact step", "↑", 100.0, True),
               ("sla_tol1", "SLA — within ±1 step", "↑", 100.0, True),
               ("f1", "ProcessBench official F1", "↑", 100.0, True),
               ("acc_erroneous", "accuracy on erroneous rows", "↑", 100.0, True),
               ("acc_correct", "accuracy on error-free rows", "↑", 100.0, True)]

    out = []
    for s in subsets:
        by_m = {r["method"]: r for r in rows_by_subset[s]}
        cols = [m for m in methods if m in by_m]
        head = "".join(
            f'<th>{esc(PRETTY.get(m, (m,))[0])}'
            f'<br><span class="sub">{PRETTY.get(m, ("", "other"))[1]}</span></th>' for m in cols)
        pub = PAPER_SLA.get(("Qwen3-8B", s))
        body = []
        for key, label, arrow, scale, higher in METRICS:
            vals = []
            for m in cols:
                try:
                    vals.append(float(by_m[m][key]) * scale)
                except (KeyError, TypeError, ValueError):
                    vals.append(float("nan"))
            fin = [v for v in vals if np.isfinite(v)]
            best = (max if higher else min)(fin) if fin else None
            tds = []
            for m, v in zip(cols, vals):
                win = best is not None and np.isfinite(v) and abs(v - best) < 1e-9
                sd = by_m[m].get(f"{key}_sd")
                sdtxt = f'<span class="sd">±{fnum(float(sd)*scale, 1, "")}</span>' if sd else ""
                tds.append(f'<td class="num{" best" if win else ""}">{fnum(v, 2)}{sdtxt}'
                           f'{" ◄" if win else ""}</td>')
            body.append(f'<tr><td class="mname">{esc(label)} '
                        f'<span class="dir">{arrow}</span></td>{"".join(tds)}</tr>')
        pubrow = ""
        if pub is not None:
            pubrow = (f'<tr class="k-target"><td class="mname">their published SLA '
                      f'<span class="kind k-target">Table 3</span></td>'
                      f'<td class="num" colspan="{len(cols)}">Shannon Drop, Qwen3-8B: '
                      f'<b>{pub:.2f}</b> — the number to beat</td></tr>')
        out.append(
            f'<h3>{esc(s)}</h3><figure class="tbl"><div class="tscroll"><table>'
            f'<thead><tr><th>metric</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}{pubrow}</tbody></table></div>'
            f'<figcaption>&#9668; marks the best method for that metric. &plusmn; is the spread '
            f'over 100 repeated 50/50 calibration splits. Rows with <code>label = -1</code> '
            f'(no erroneous step) are excluded from SLA — there is nothing to localize — but '
            f'are scored by the official F1, which is why both are reported.</figcaption>'
            f'</figure>')
    return "".join(out)


def pending(title, why):
    return (f'<div class="pending"><span class="ptag">pending</span>'
            f'<b>{esc(title)}</b><p>{why}</p></div>')


# ── page ─────────────────────────────────────────────────────────────────────

CSS = """
:root{
  --ink:#141922; --ink-2:#4A5566; --ink-3:#727E90;
  --paper:#F6F7F9; --card:#FFFFFF; --rule:#DCE1E8;
  --signal:#1D5E8C; --signal-soft:#D6E6F1;
  --risk:#B23B31; --risk-soft:#F2DAD6;
  --ok:#2C7355; --accentbg:#EEF3F7;
}
@media (prefers-color-scheme: dark){
  :root{
    --ink:#E4E8EF; --ink-2:#A3AEBF; --ink-3:#78849A;
    --paper:#0E1218; --card:#151B24; --rule:#26303D;
    --signal:#63A9D8; --signal-soft:#17324A; --risk:#E07C71; --risk-soft:#3E2320;
    --ok:#63B694; --accentbg:#141C26;
  }
}
:root[data-theme="dark"]{
  --ink:#E4E8EF; --ink-2:#A3AEBF; --ink-3:#78849A;
  --paper:#0E1218; --card:#151B24; --rule:#26303D;
  --signal:#63A9D8; --signal-soft:#17324A; --risk:#E07C71; --risk-soft:#3E2320;
  --ok:#63B694; --accentbg:#141C26;
}
:root[data-theme="light"]{
  --ink:#141922; --ink-2:#4A5566; --ink-3:#727E90;
  --paper:#F6F7F9; --card:#FFFFFF; --rule:#DCE1E8;
  --signal:#1D5E8C; --signal-soft:#D6E6F1; --risk:#B23B31; --risk-soft:#F2DAD6;
  --ok:#2C7355; --accentbg:#EEF3F7;
}
*{box-sizing:border-box;}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;}
.wrap{max-width:1040px;margin:0 auto;padding:56px 24px 96px;}
.mono{font-family:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;}

header.top{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:8px;}
.eyebrow{font-family:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
  font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--signal);
  margin:0 0 14px;}
h1{font-family:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  font-weight:600;font-size:clamp(30px,4.4vw,44px);line-height:1.12;letter-spacing:-.015em;
  margin:0 0 12px;text-wrap:balance;}
.standfirst{font-size:17.5px;color:var(--ink-2);max-width:66ch;margin:0;}
.meta{display:flex;flex-wrap:wrap;gap:8px 26px;margin-top:20px;font-size:12.5px;
  color:var(--ink-3);font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}

section{margin-top:52px;}
h2{font-family:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  font-size:25px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px;text-wrap:balance;
  padding-left:14px;border-left:3px solid var(--signal);}
h3{font-size:15px;font-weight:650;letter-spacing:.01em;margin:30px 0 8px;color:var(--ink);}
p{max-width:70ch;color:var(--ink-2);}
p.lede{color:var(--ink);}
code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.88em;
  background:var(--accentbg);padding:1px 5px;border-radius:3px;color:var(--ink);}

.tbl{margin:22px 0 0;}
.tscroll{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--card);}
table{border-collapse:collapse;width:100%;font-size:13.5px;
  font-variant-numeric:tabular-nums;}
th,td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--rule);}
thead th{background:var(--accentbg);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-2);font-weight:650;white-space:nowrap;}
thead th .sub{font-family:ui-monospace,Menlo,monospace;text-transform:none;letter-spacing:0;
  color:var(--ink-3);font-weight:400;}
tbody tr:last-child td{border-bottom:none;}
td.num{text-align:right;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}
td.num.best{font-weight:700;color:var(--signal);}
td.num.ok{color:var(--ok);} td.num.warn{color:var(--risk);}
.mname{white-space:nowrap;}
.sd{color:var(--ink-3);font-size:10.5px;margin-left:3px;}
.dir{color:var(--signal);font-weight:700;margin-left:4px;}
td.num.best{font-weight:700;color:var(--signal);background:color-mix(in srgb,var(--signal) 9%,transparent);}
.kind{display:inline-block;margin-left:9px;font-size:9.5px;letter-spacing:.08em;
  text-transform:uppercase;padding:2px 6px;border-radius:3px;vertical-align:1px;
  font-family:ui-monospace,Menlo,monospace;}
.kind.k-ours{background:var(--signal);color:var(--card);}
.kind.k-paper{background:var(--accentbg);color:var(--ink-2);border:1px solid var(--rule);}
.kind.k-ref{background:transparent;color:var(--ink-3);border:1px dashed var(--rule);}
.kind.k-target{background:var(--risk-soft);color:var(--risk);}
tr.k-ours{background:color-mix(in srgb,var(--signal) 6%,transparent);}
figcaption{font-size:12.5px;color:var(--ink-3);margin-top:9px;max-width:76ch;line-height:1.55;}

.fig{display:block;margin:0;}
.axis{stroke:var(--rule);stroke-width:1;}
.stepline{stroke:var(--rule);stroke-width:1;stroke-dasharray:2 4;opacity:.75;}
.curve{fill:none;stroke:var(--signal);stroke-width:1.7;stroke-linejoin:round;}
.dropline{stroke:var(--risk);stroke-width:1;stroke-dasharray:3 3;opacity:.8;}
.dropdot{fill:var(--risk);}
.droplab{fill:var(--risk);font-size:10.5px;font-family:ui-monospace,Menlo,monospace;}
.axlab{fill:var(--ink-3);font-size:10px;font-family:ui-monospace,Menlo,monospace;}
.axttl{fill:var(--ink-3);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
  font-family:ui-monospace,Menlo,monospace;}
.barok{fill:color-mix(in srgb,var(--ok) 26%,transparent);stroke:var(--ok);stroke-width:1.2;}
.barbad{fill:color-mix(in srgb,var(--risk) 26%,transparent);stroke:var(--risk);stroke-width:1.2;}
.medok{stroke:var(--ok);stroke-width:1.4;stroke-dasharray:4 3;}
.medbad{stroke:var(--risk);stroke-width:1.4;stroke-dasharray:4 3;}
.medlab{font-size:10.5px;font-family:ui-monospace,Menlo,monospace;stroke:none;}
.medlab.medok{fill:var(--ok);} .medlab.medbad{fill:var(--risk);}

.panel{border:1px solid var(--rule);border-radius:8px;background:var(--card);padding:18px 20px;
  margin:20px 0 0;overflow:hidden;}
.panel-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 16px;
  border-bottom:1px solid var(--rule);padding-bottom:12px;margin-bottom:14px;}
.panel-title{font-weight:650;font-size:15px;}
.panel-note{font-size:12.5px;color:var(--ink-3);font-family:ui-monospace,Menlo,monospace;}
.verdict{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;padding:3px 8px;
  border-radius:3px;font-family:ui-monospace,Menlo,monospace;font-weight:650;}
.v-caught{background:var(--risk);color:var(--card);}
.v-clean{background:var(--ok);color:var(--card);}
.v-missed{background:var(--accentbg);color:var(--ink-2);border:1px solid var(--rule);}

.chain{display:flex;flex-direction:column;gap:3px;max-height:430px;overflow-y:auto;
  border:1px solid var(--rule);border-radius:5px;padding:8px;background:var(--paper);}
.step{display:grid;grid-template-columns:64px 1fr;gap:12px;padding:6px 8px;border-radius:4px;
  background:color-mix(in srgb,var(--risk) calc(var(--tint) * 42%),transparent);}
.step-meta{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--ink-3);
  display:flex;gap:8px;justify-content:space-between;padding-top:2px;}
.step-r{font-variant-numeric:tabular-nums;}
.step-body{font-size:13.5px;line-height:1.55;white-space:pre-wrap;word-break:break-word;
  color:var(--ink);}
.step-peak{outline:2px solid var(--risk);outline-offset:-1px;}
.step-gt{box-shadow:inset 3px 0 0 var(--signal);}
.tag{display:inline-block;margin-left:8px;font-size:9.5px;letter-spacing:.07em;
  text-transform:uppercase;padding:1px 6px;border-radius:3px;font-family:ui-monospace,Menlo,monospace;
  vertical-align:2px;}
.tag-pick{background:var(--risk);color:var(--card);}
.tag-gt{background:var(--signal);color:var(--card);}

.pending{border:1px dashed var(--rule);border-radius:8px;padding:18px 20px;margin:22px 0 0;
  background:var(--card);}
.pending .ptag{display:inline-block;font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);border:1px solid var(--rule);border-radius:3px;padding:2px 7px;
  margin-right:10px;font-family:ui-monospace,Menlo,monospace;}
.pending p{margin:8px 0 0;font-size:13.5px;}
.note{font-size:13px;color:var(--ink-3);}
ul{max-width:70ch;color:var(--ink-2);padding-left:20px;}
li{margin:6px 0;}
.callout{border-left:3px solid var(--risk);background:color-mix(in srgb,var(--risk) 6%,transparent);
  padding:14px 18px;border-radius:0 6px 6px 0;margin:20px 0;}
.callout p{margin:0;max-width:none;color:var(--ink);}
footer{margin-top:70px;padding-top:20px;border-top:1px solid var(--rule);font-size:12.5px;
  color:var(--ink-3);font-family:ui-monospace,Menlo,monospace;}
@media (max-width:640px){.step{grid-template-columns:1fr;gap:2px;}}
"""


def build(out_path):
    ev_rows, ev_diags, skipped = {}, {}, []
    for p in sorted(glob.glob(os.path.join(RES, "evdrop", "*__evdrop.csv"))):
        c = os.path.basename(p).replace("__evdrop.csv", "")
        # N=30 pilots are excluded from the tables, not because they disagree but because they
        # cannot disagree: with a single incorrect answer every method scores 100.00 +/- 0.00 at
        # every alpha, and the alpha-quantile of one calibration sample is that sample. Their
        # only job was to prove the pipeline ran.
        if c.endswith("_pilot"):
            skipped.append(c)
            continue
        ev_rows[c] = read_csv(p)
        dp = p.replace("__evdrop.csv", "__diagnostics.json")
        if os.path.exists(dp):
            ev_diags[c] = json.load(open(dp))

    pb_rows, pb_diags = {}, {}
    for p in sorted(glob.glob(os.path.join(RES, "processbench", "*__processbench.csv"))):
        s = os.path.basename(p).replace("__processbench.csv", "")
        pb_rows[s] = read_csv(p)
        dp = p.replace("__processbench.csv", "__diagnostics.json")
        if os.path.exists(dp):
            pb_diags[s] = json.load(open(dp))

    exs = {}
    for p in sorted(glob.glob(os.path.join(RES, "*__examples.json"))):
        exs[os.path.basename(p).replace("__examples.json", "")] = json.load(open(p))

    # ── traceability panels ──────────────────────────────────────────────────
    panels_html = []
    for cell, ex in exs.items():
        arm = ex["arm"]
        pr = ex["peak_risk"]
        titles = {
            "caught": ("Caught", "v-caught",
                       "wrong answer, highest peak step risk in the cell"),
            "clean": ("Clean", "v-clean",
                      "right answer, lowest peak step risk in the cell"),
            "missed": ("Missed", "v-missed",
                       "wrong answer whose peak risk sits below the correct-answer median"),
        }
        for name in ("caught", "missed", "clean"):
            if name not in ex["panels"]:
                continue
            p = ex["panels"][name]
            t, vcls, why = titles[name]
            src = p["source"]
            panels_html.append(f"""
<div class="panel">
  <div class="panel-head">
    <span class="verdict {vcls}">{t}</span>
    <span class="panel-title">{esc(why)}</span>
    <span class="panel-note">{esc(cell)} · row {src['idx']} ·
      {len(p['steps'])} steps · {len(p['evidence'])} tokens ·
      answer {'correct' if p['answer_correct'] else 'incorrect'} ·
      peak step {p['peak_step']}</span>
  </div>
  {steps_html(p)}
  {evidence_svg(p)}
  <figcaption>Each step is tinted by its own U-PCR risk (deeper = more suspect) and the number
  beside it is that risk. The curve below is the same detector read at token resolution, over
  32-token sliding windows using the step fit's frozen parameters; the marked points are the
  three worst first-differences &Delta; of the smoothed evidence, with the token they land on and
  the step that token belongs to.</figcaption>
</div>""")
        panels_html.append(f"""
<h3>Separability of the step-level risk</h3>
{separability_svg(ex)}
<figcaption>Peak step risk within an answer, over all {pr['n_correct'] + pr['n_incorrect']}
answers of <code>{esc(cell)}</code>. Incorrect answers sit at median
{pr['incorrect_median']:.3f} against {pr['correct_median']:.3f} for correct ones; only
{pr['n_missed_below_correct_median']} of {pr['n_incorrect']} incorrect answers fall below the
correct-answer median. The arm was fitted label-free on all {arm['n_steps_fit']:,} pooled steps
&mdash; {arm['n_kept']} of {len(arm['pool'])} views survived its exclusion step, anchored on
<code>{esc(arm['anchor'])}</code>.</figcaption>""")

    if not panels_html:
        panels_html = [pending("Worked examples",
                               "run <code>build_examples.py</code> on a fetched cell")]

    body = f"""
<div class="wrap">
<header class="top">
  <p class="eyebrow">Thesis experiment &middot; step-level localization</p>
  <h1>Does U-PCR find <em>where</em> the reasoning goes wrong?</h1>
  <p class="standfirst">Our label-free arm, put through the protocol of &ldquo;Mind the Gap:
  Catching Hallucinations via Evidence Drop&rdquo; (ICML 2026) &mdash; the same metrics, the same
  calibration, the same baselines &mdash; at the answer level and at the step level.</p>
  <div class="meta">
    <span>arm: U-PCR over the canonical pool</span>
    <span>polarity: sign(&rho;&#770;), label-free</span>
    <span>orientation: cell anchor</span>
    <span>no subset, no fixed K, no hand-picked views</span>
  </div>
</header>

<section>
<h2>What is being measured</h2>
<p class="lede">The paper's contribution is a detector: evidence
<code>&Ecirc;&nbsp;=&nbsp;&minus;H(P&#771;)</code> over the renormalized top-20, smoothed with an
EMA, and a risk score built from the <em>worst first-differences</em> of that curve &mdash; the
&ldquo;drops&rdquo;. Ours is a different signal fused a different way: a pool of spectral,
temporal and log-probability views over the entropy trace, combined by U-PCR with no labels, no
hand-picked feature list and no chosen subset size.</p>
<p>Both are one forward pass, both are unsupervised, so they are directly comparable on cost. The
question is whether ours also localizes &mdash; whether the fused score points at the step where
the reasoning broke, not just at the answer being wrong.</p>
<div class="callout"><p><b>Which arm this is, and exactly how far the verification reaches.</b>
U-PCR here means <code>upcr_fit</code> over the full canonical pool with the fitted
<code>exp06</code> config and polarity derived from <code>sign(&rho;&#770;)</code>. It is not
<code>upcr_pipeline</code> and not a fixed subset. An independent audit confirmed it reproduces
the maintained scorer <b>bit-for-bit on all 24 in-scope roster cells</b> (max score difference
0.0, identical kept-view counts, matching the recorded <code>auroc_rho_anchor</code> to 5e-4).</p>
<p>The scope of that check, stated precisely: <code>assert_upcr_mirrors_canonical</code> runs on
the <b>answer-level</b> cells only. The step-level and token-level arms fit their own detector on
a different population and are <b>not</b> covered by it &mdash; and the step-level pool is
structurally smaller (spectral and time-domain views only; the log-probability and anomaly views
have no per-step analogue). Read the step-level numbers with that in mind.</p>
<p>The hand-picked L-SML subsets appear only as clearly-labelled reference rows. The label-free
peer to compare against is <b>DUFS + L-SML</b>, which sits within noise of U-PCR across the
roster (macro 0.7687 vs 0.7741, 15W/9L of 24, p=0.069) &mdash; not the GOOD_* rows.</p></div>
</section>

<section>
<h2>Answer level &mdash; their Tables 1 and 2</h2>
<p>Read the operating point first. Selective accuracy and AURC are both monotone in the base error
rate, so a cell that is more accurate than theirs will look better on both no matter what the
detector does.</p>
{answer_tables(ev_rows, ev_diags)}
</section>

<section>
<h2>Step level &mdash; their Table 3</h2>
<p>ProcessBench, teacher-forced: a single forward pass over a reasoning chain someone else wrote,
so the signal is <em>our model's surprise at another model's text</em>. That is their protocol,
and it is not the same quantity the answer-level cells measure.</p>
{processbench_tables(pb_rows, pb_diags)}
</section>

<section>
<h2>Where it marks the hallucination</h2>
<p>The panels below are real traces from <code>evdrop_gsm8k_qwen3_8b</code>, chosen by a rule
fixed before any score was looked at: the most confident catch, the case it misses, and a clean
answer for contrast. Showing only the catch would be picking the evidence.</p>
<div class="callout"><p><b>What these panels can and cannot show.</b> Our own generated answers
carry a correctness label for the <em>answer</em> and no per-step annotation anywhere. So these
show where the detector fires and how a wrong answer differs from a right one &mdash; but there
is no ground-truth step to check the pick against. That comparison needs ProcessBench, whose rows
carry an annotated first-erroneous step; those panels arrive with the table above.</p></div>

<h3>The step-level pool is smaller than the answer-level one</h3>
<p>Two reasons, both structural rather than incidental, and both worth stating before reading any
step-level number:</p>
<ul>
<li><b>Most views need whole-trace context.</b> A step is scored by running the ordinary feature
extractor on the slice of the entropy trace belonging to it, which yields the spectral and
time-domain views only. The log-probability views and the anomaly views are computed per trace
elsewhere in the pipeline and have no per-step analogue, so they are simply absent at step level.
That is why the answer-level arm fits on tens of views and the step-level arm on around a
dozen.</li>
<li><b>Short steps lose more.</b> <code>compute_spectral_features</code> returns nothing below 8
tokens and <code>compute_stft_features</code> returns <em>0.0 rather than NaN</em> below 32
&mdash; a constant that passes a finiteness check and would enter the fusion as an
information-free column. Those views are explicitly marked missing per step instead, so the pool
shrinks honestly rather than filling with zeros.</li>
</ul>
<p>The token trace exists because of the second point: a 32-token sliding window spans step
boundaries, so it has no step-length floor at all and keeps the full extractable pool no matter
how short the steps are. Where the step arm cannot be fitted, the token arm still can.</p>
{''.join(panels_html)}
</section>

<section>
<h2>What is ours and not theirs</h2>
<p>The paper defines its risk at the token level but reports localization at the step level, and
never says how one becomes the other. Those choices are ours, fixed before any number was
computed, and a gap against their table may be a rule difference rather than a method difference:</p>
<ul>
<li><b>Token &rarr; step.</b> A step's risk is the <em>worst</em> first-difference inside it, since
the method is explicitly a worst-case drop detector. The mean is computed as a sensitivity, never
as the headline.</li>
<li><b>Flux attribution.</b> The difference between tokens <code>j</code> and <code>j+1</code>
belongs to the step containing <code>j+1</code> &mdash; a drop that happens on the way into a bad
step belongs to that step.</li>
<li><b>Smoothing is global.</b> One EMA over the whole chain, then partition by step. Smoothing
each step separately would reset the filter at every boundary and manufacture a drop there.</li>
<li><b>Rows with no erroneous step</b> are excluded from SLA (there is nothing to localize) but
kept for ProcessBench's official F1, which scores exactly that abstention.</li>
<li><b>The step threshold</b> is the &alpha;-quantile of per-step risk on annotated erroneous
steps in a calibration split &mdash; the same Neyman&ndash;Pearson construction they use at
sequence level. They give no rule at all here.</li>
<li><b>Window and stride.</b> W&nbsp;=&nbsp;32 tokens is not tuned: it is exactly the minimum
length at which the full feature pool is measurable.</li>
</ul>
</section>

<footer>
Generated by <span class="mono">scripts/localization/localization_report.py</span> from the CSVs
and JSON beside it. Every table is diffable; nothing on this page is typed by hand.
</footer>
</div>
"""
    html_out = f"<title>U-PCR on step-level localization</title>\n<style>{CSS}</style>\n{body}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"answer-level cells : {len(ev_rows)}  ({', '.join(ev_rows) or 'none'})")
    print(f"processbench subsets: {len(pb_rows)}  ({', '.join(pb_rows) or 'none — PENDING'})")
    print(f"example sets       : {len(exs)}  ({', '.join(exs) or 'none'})")
    print(f"-> {out_path}  ({len(html_out)/1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=os.path.join(RES, "localization_report.html"))
    build(ap.parse_args().out)
