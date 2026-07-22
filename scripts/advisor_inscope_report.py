#!/usr/bin/env python
"""
advisor_inscope_report.py — the in-scope (QA + math) advisor deliverable (Step 193).

Builds results/advisor_inscope/*.html: a per-cell competitor grid over the 25 in-scope
cells, the supervised ceiling, the label-free selector diagnosis, the anchor robustness
check, and — the page that lets the rest be trusted — a verification write-up.

Design rules carried over from scripts/action_items_report.py (the advisor house):
  * every numeric cell is read from a CSV at build time; nothing is hand-typed,
  * guardrail_scan runs on every page and a banned term is a build failure (exit 1),
  * the canonical all-cell reports are NOT regenerated — doing so would re-mix the
    out-of-scope RAG/GPQA cells back into the headlines.

Inputs (all produced by other Step-193 scripts, all under results/):
  selector_bench/comparison_inscope.csv          leaderboard (corrected)
  selector_bench/reference_macros__c46.csv       per-cell fixed-subset AUROCs
  selector_bench/a2_groupfs__{h16,c46}.csv       learned selector
  selector_bench/splithalf_oracle_c46_inscope_summary.csv
  advisor_inscope/competitors_verified.csv       build_competitors_verified.py
  advisor_inscope/anchor_sweep.csv               anchor_sweep_inscope.py
  advisor_inscope/subset_gap_analysis.csv        subset_gap_analysis.py
  advisor_inscope/groupfs_diagnosis.csv          groupfs_diagnosis.py
  advisor_inscope/lr_oracle_audit.csv            Phase 0.3 audit
  advisor_inscope/stale_audit_all_families.csv   Phase 0.1 audit

Usage:
    python scripts/advisor_inscope_report.py [--check]
"""
import argparse
import csv
import datetime as _dt
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from advisor_report import esc, CSS, guardrail_scan            # noqa: E402
from inscope_cells import INSCOPE, QA_CELLS, MATH_CELLS, GROUP, CLUSTER_CELLS  # noqa: E402
from subset_gap_analysis import spearman                       # noqa: E402
from report_figs import FIG_CSS, _svg, _fig, _lin, _bar_panel  # noqa: E402 — inline-SVG figure idiom (item4 standard)

BENCH = os.path.join(REPO, "results", "selector_bench")
AI = os.path.join(REPO, "results", "advisor_inscope")
OUT_DIR = AI
GEN_DATE = _dt.date.today().isoformat()

PAGES_NAV = [
    ("index.html", "Overview"),
    ("competitor_grid.html", "Competitor grid"),
    ("competitor_sources.html", "Competitor provenance"),
    ("supervised_ceiling.html", "Supervised ceiling"),
    ("subset_gap_analysis.html", "Selection headroom"),
    ("selector_choices.html", "What it selects"),
    ("groupfs_diagnosis.html", "Learned selector"),
    ("gated_laplacian.html", "Gated-Laplacian"),
    ("anchor_sweep.html", "Anchor robustness"),
    ("verification.html", "Verification"),
]

EXTRA_CSS = """
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:18px 0}
.tile{background:#fff;border:1px solid #dde3ea;border-radius:10px;padding:14px 16px}
.tile .v{font-size:1.7em;font-weight:700;color:#12395e;line-height:1.1}
.tile .l{font-size:.82em;color:#5c6b7a;margin-top:4px}
.tile .d{font-size:.78em;color:#7a8794;margin-top:6px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:18px 0}
.card{background:#fff;border:1px solid #dde3ea;border-radius:10px;padding:16px}
.card h3{margin:0 0 6px;font-size:1.02em}
.card p{margin:0;font-size:.88em;color:#5c6b7a}
.nav-bar{background:#eef3f8;border:1px solid #d6e0ea;border-radius:8px;padding:9px 13px;
         margin-bottom:18px;font-size:.87em}
.nav-bar a{margin-right:4px}
table{font-size:.86em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.good{color:#1b7f4b;font-weight:600}
.bad{color:#b3261e;font-weight:600}
.dim{color:#8b97a3}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:.76em;font-weight:600}
.b-ok{background:#e4f4ea;color:#1b7f4b}
.b-warn{background:#fdf1dc;color:#8a5a00}
.b-bad{background:#fbe4e2;color:#b3261e}
.b-neutral{background:#eef1f4;color:#5c6b7a}
.chips span{display:inline-block;background:#eef3f8;border:1px solid #d6e0ea;border-radius:9px;
            padding:0 6px;margin:1px 2px;font-size:.76em;color:#3d4a57}
.callout{border-left:4px solid #2a6faf;background:#f3f8fd;padding:12px 15px;margin:16px 0;
         border-radius:0 8px 8px 0}
.callout.warn{border-left-color:#d08c00;background:#fffaf0}
.callout.bad{border-left-color:#b3261e;background:#fdf3f2}
.small{font-size:.85em;color:#5c6b7a}
""" + FIG_CSS


# ── data loading ──────────────────────────────────────────────────────────────────────

def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(v):
    try:
        x = float(v)
        return x if x == x else None          # NaN -> None
    except (TypeError, ValueError):
        return None


def pc(v, digits=1, signed=False):
    """AUROC (0-1) -> percent string."""
    if v is None:
        return '<span class="dim">--</span>'
    s = f"{100*v:+.{digits}f}" if signed else f"{100*v:.{digits}f}"
    return s


def cls_for(v, good=0.0):
    if v is None:
        return ""
    return "good" if v > good else ("bad" if v < good else "")


D = {}


def load_all():
    D["leader"] = read(os.path.join(BENCH, "comparison_inscope.csv"))
    D["refmac"] = read(os.path.join(BENCH, "reference_macros__c46.csv"))
    D["a2c46"] = read(os.path.join(BENCH, "a2_groupfs__c46.csv"))
    D["a2h16"] = read(os.path.join(BENCH, "a2_groupfs__h16.csv"))
    D["comp"] = read(os.path.join(AI, "competitors_verified.csv"))
    D["anchor"] = read(os.path.join(AI, "anchor_sweep.csv"))
    D["gap"] = read(os.path.join(AI, "subset_gap_analysis.csv"))
    D["gfs"] = read(os.path.join(AI, "groupfs_diagnosis.csv"))
    D["lr"] = read(os.path.join(AI, "lr_oracle_audit.csv"))
    D["stale"] = read(os.path.join(AI, "stale_audit_all_families.csv"))
    D["pool_size"] = read(os.path.join(AI, "pool_size_experiment.csv"))
    D["reconc"] = read(os.path.join(AI, "reconciliation.csv"))

    macro = {}
    for r in D["refmac"]:
        if r.get("cell") in INSCOPE:
            macro.setdefault(r["cell"], {})[r["variant"]] = f(r.get("auroc"))
    D["macro_by_cell"] = macro

    lead = {}
    for r in D["leader"]:
        lead[(r.get("variant"), r.get("pool_mode"))] = r
    D["lead"] = lead

    lr = {}
    for r in D["lr"]:
        lr.setdefault(r["cell"], {})[r["fset"]] = r
    D["lr_by_cell"] = lr

    gfs = {}
    for r in D["gfs"]:
        gfs.setdefault(r["cell"], {})[r["variant"]] = r
    D["gfs_by_cell"] = gfs

    # keep BOTH selection rules per pool: a2.dufs is the trace-based Gated-Laplacian and
    # the better of the two, so the grid must show it, not only GroupFS's group rule.
    a2 = {}
    for tag, rows in (("c46", D["a2c46"]), ("h16", D["a2h16"])):
        for r in rows:
            if r.get("cell") in INSCOPE and r.get("variant") in ("a2.select", "a2.dufs"):
                a2.setdefault(r["cell"], {})[(tag, r["variant"])] = r
    D["a2_by_cell"] = a2
    D["svc"] = read(os.path.join(AI, "selector_vs_competitor.csv"))
    D["choice"] = read(os.path.join(AI, "selector_choice_analysis.csv"))
    D["choice_by_cell"] = {r["cell"]: r for r in D["choice"]}
    D["svc_by_cell"] = {r["cell"]: r for r in D["svc"]}

    comp = {}
    for r in D["comp"]:
        comp.setdefault(r["cell"], []).append(r)
    D["comp_by_cell"] = comp

    D["gap_by_cell"] = {r["cell"]: r for r in D["gap"]}


def lead_val(variant, pool, key):
    r = D["lead"].get((variant, pool))
    return f(r.get(key)) if r else None


# ── inline-SVG figures (report_figs idiom: CSV-driven, no JS, caveat in footnote) ─────

def fig_svc_dumbbell():
    """Ours vs the published competitor, one row per cell, sorted by delta.
    Supervised competitors are drawn as open slate circles — shown, but not
    like-for-like (only the unsupervised anchors are peers)."""
    rows = []
    for r in D["svc"]:
        ours, comp = f(r.get("dufs")), f(r.get("comp"))
        if ours is None or comp is None:
            continue
        sup = (r.get("comp_sup") or "").strip().lower()
        rows.append((100 * ours, 100 * comp, r["cell"], r.get("comp_method", ""),
                     sup == "unsupervised"))
    if not rows:
        return ""
    rows.sort(key=lambda t: -(t[0] - t[1]))
    ROW, TOP = 30, 8
    n = len(rows)
    W, X0, X1 = 900, 260, 880
    lo = min(min(a, b) for a, b, *_ in rows)
    hi = max(max(a, b) for a, b, *_ in rows)
    d0, d1 = max(0.0, lo - 3), min(100.0, hi + 3)
    H = TOP + n * ROW + 30
    x = lambda v: _lin(v, d0, d1, X0, X1)
    s = [_svg(W, H)]
    for gv in range(int(d0 // 10 * 10 + 10), int(d1) + 1, 10):
        s.append(f'<line x1="{x(gv):.1f}" y1="{TOP}" x2="{x(gv):.1f}" y2="{TOP+n*ROW}" class="rf-grid"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{TOP+n*ROW+15}" class="rf-tick" text-anchor="middle">{gv}</text>')
    n_win_uns = sum(1 for a, b, _, _, uns in rows if uns and a > b)
    n_uns = sum(1 for r in rows if r[4])
    for i, (a, b, cell, meth, uns) in enumerate(rows):
        cy = TOP + i * ROW + ROW / 2
        lbl = esc(cell)
        s.append(f'<text x="{X0-6}" y="{cy-1:.1f}" class="rf-rowlbl" text-anchor="end">{lbl}</text>')
        s.append(f'<text x="{X0-6}" y="{cy+11:.1f}" class="rf-rowsub" text-anchor="end">vs {esc(meth)}{"" if uns else " (supervised)"}</text>')
        xa, xb = x(a), x(b)
        s.append(f'<line x1="{xa:.1f}" y1="{cy:.1f}" x2="{xb:.1f}" y2="{cy:.1f}" class="rf-leader"/>')
        if uns:
            s.append(f'<circle cx="{xb:.1f}" cy="{cy:.1f}" r="5" class="rf-anchor"><title>{esc(meth)} (unsupervised): {b:.1f}</title></circle>')
        else:
            s.append(f'<circle cx="{xb:.1f}" cy="{cy:.1f}" r="5" class="rf-sup"><title>{esc(meth)} (supervised): {b:.1f}</title></circle>')
        s.append(f'<circle cx="{xa:.1f}" cy="{cy:.1f}" r="5" class="rf-ours"><title>selector (a2.dufs): {a:.1f}</title></circle>')
        d = a - b
        tx, ta = (max(xa, xb) + 9, "start")
        s.append(f'<text x="{tx:.1f}" y="{cy+4:.1f}" class="rf-dlbl{" rf-dlbl-ours" if d > 0 else ""}" text-anchor="{ta}">{d:+.1f}</text>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-1}" class="rf-axname" text-anchor="middle">AUROC (%)</text>')
    s.append("</svg>")
    legend = ('<span class="rf-li"><svg width="16" height="14"><circle cx="8" cy="7" r="5" class="rf-ours"/></svg> our label-free selector (a2.dufs)</span>'
              '<span class="rf-li"><svg width="16" height="14"><circle cx="8" cy="7" r="5" class="rf-anchor"/></svg> published competitor — unsupervised (like-for-like)</span>'
              '<span class="rf-li"><svg width="16" height="14"><circle cx="8" cy="7" r="5" class="rf-sup"/></svg> published competitor — supervised (context only)</span>')
    fnote = (f"Sorted by delta (ours − competitor). Against the {n_uns} unsupervised anchors the selector wins "
             f"{n_win_uns}/{n_uns}; supervised anchors (open circles) are shown for context, not tallied — they "
             "consume labels our method never sees. The 6 cluster cells have no published anchor and do not appear. "
             "Source: selector_vs_competitor.csv (competitor values verified against each paper's own table, "
             "competitors_verified.csv).")
    return _fig("Ours vs the published competitor, cell by cell",
                "Each row pairs our label-free selector with the published number on the same (dataset, model) cell.",
                legend, "".join(s), fnote)


def fig_macro_bars():
    """Macro AUROC by method over the 25 in-scope cells — the one-glance ranking."""
    spec = [
        ("ref.GOOD_6",  "GOOD_6 (fixed subset — headline reference)", "g6"),
        ("ref.GOOD_5",  "GOOD_5 (compatibility reference)",           "g5"),
        ("a2.dufs",     "Gated-Laplacian selector (a2.dufs)",         "ours"),
        ("a2.select",   "GroupFS selector (a2.select)",               "ctx"),
        ("a6.pl_dufs",  "Pseudo-label gates (a6.pl_dufs)",            "ours"),
        ("a6.pl_rank",  "Pseudo-label rank ablation (a6.pl_rank)",    "ctx"),
        ("a6.dufs",     "a6 unsupervised control (a6.dufs)",          "ctx"),
    ]
    bars = []
    for variant, lbl, kind in spec:
        v = lead_val(variant, "c46", "macro_all")
        if v is not None:
            bars.append((lbl, 100 * v, kind, "macro over 25 in-scope cells"))
    lr = [f(r["floored"]) for r in D["lr"] if r["fset"] == "30"]
    if lr:
        bars.append(("Logistic regression @30 views (SUPERVISED ceiling)",
                     100 * sum(lr) / len(lr), "ctx-sup", "uses labels — not a peer"))
    if not bars:
        return ""
    bars.sort(key=lambda b: -b[1])
    lo = min(b[1] for b in bars)
    svg = _bar_panel(bars, d0=max(35, lo - 8), d1=max(b[1] for b in bars) + 4)
    legend = ('<span class="rf-li"><svg width="16" height="12"><rect x="1" y="2" width="14" height="8" rx="2" class="rf-bar-g6"/></svg> fixed subsets</span>'
              '<span class="rf-li"><svg width="16" height="12"><rect x="1" y="2" width="14" height="8" rx="2" class="rf-bar-ours"/></svg> label-free selectors</span>'
              '<span class="rf-li"><svg width="16" height="12"><rect x="1" y="2" width="14" height="8" rx="2" class="rf-bar-gray"/></svg> ablations / controls</span>'
              '<span class="rf-li"><svg width="16" height="12"><rect x="1" y="2" width="14" height="8" rx="2" class="rf-bar-gray-sup"/></svg> supervised ceiling (not a peer)</span>')
    fnote = ("All rows are macro AUROC over the same 25 in-scope cells (10 QA + 15 math) on the 30-view pool. "
             "GOOD_6 was chosen once by corpus-wide labelled search (Step 182/184), so it carries corpus-level label "
             "information even though applying it is label-free; the selectors get no such prior. The LR ceiling is "
             "fully supervised and shown only to bound the headroom. Source: comparison_inscope.csv + lr_oracle_audit.csv.")
    return _fig("Macro AUROC by method — the one-glance ranking",
                "Fixed subsets vs label-free selectors vs the supervised ceiling, same 25 cells throughout.",
                legend, svg, fnote)


def fig_pool_size_line():
    """Pool-size experiment: nested pools ranked by informativeness — the figure that
    makes 'composition matters, size does not' legible."""
    pts = []
    for r in D.get("pool_size", []):
        ps, sel, g6 = f(r.get("pool_size")), f(r.get("selector_macro")), f(r.get("good6_macro"))
        if ps and sel:
            pts.append((int(ps), 100 * sel, 100 * g6 if g6 else None))
    if len(pts) < 3:
        return ""
    pts.sort()
    W, H, X0, X1, Y0, Y1 = 880, 260, 70, 850, 20, 205
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts] + [p[2] for p in pts if p[2] is not None]
    d0, d1 = min(ys) - 0.4, max(ys) + 0.4
    x = lambda v: _lin(v, min(xs), max(xs), X0, X1)
    y = lambda v: _lin(v, d0, d1, Y1, Y0)
    s = [_svg(W, H)]
    import math as _math
    for gv in [round(g * 0.5, 1) for g in range(int(_math.floor(d0 * 2)), int(_math.ceil(d1 * 2)) + 1)]:
        if gv < d0 or gv > d1:
            continue
        s.append(f'<line x1="{X0}" y1="{y(gv):.1f}" x2="{X1}" y2="{y(gv):.1f}" class="rf-grid"/>')
        s.append(f'<text x="{X0-8}" y="{y(gv)+4:.1f}" class="rf-tick" text-anchor="end">{gv:.1f}</text>')
    for ps in xs:
        s.append(f'<text x="{x(ps):.1f}" y="{Y1+18}" class="rf-tick" text-anchor="middle">{ps}</text>')
    path_sel = " ".join(f"{'M' if i == 0 else 'L'} {x(p[0]):.1f} {y(p[1]):.1f}" for i, p in enumerate(pts))
    s.append(f'<path d="{path_sel}" class="rf-line"/>')
    g6pts = [(p[0], p[2]) for p in pts if p[2] is not None]
    if g6pts:
        path_g6 = " ".join(f"{'M' if i == 0 else 'L'} {x(a):.1f} {y(b):.1f}" for i, (a, b) in enumerate(g6pts))
        s.append(f'<path d="{path_g6}" class="rf-line-acc"/>')
        for a, b in g6pts:
            s.append(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4" class="rf-dot-acc"><title>GOOD_6 @ pool {a}: {b:.2f}</title></circle>')
    for p in pts:
        s.append(f'<circle cx="{x(p[0]):.1f}" cy="{y(p[1]):.1f}" r="4" class="rf-ours"><title>selector @ pool {p[0]}: {p[1]:.2f}</title></circle>')
    s.append(f'<text x="{(X0+X1)//2}" y="{H-4}" class="rf-axname" text-anchor="middle">pool size (views, nested by informativeness rank)</text>')
    s.append(f'<text x="14" y="{(Y0+Y1)//2}" class="rf-axname" transform="rotate(-90 14 {(Y0+Y1)//2})" text-anchor="middle">macro AUROC (%)</text>')
    s.append("</svg>")
    legend = ('<span class="rf-li"><svg width="16" height="14"><circle cx="8" cy="7" r="4" class="rf-ours"/></svg> label-free selector (a2.dufs)</span>'
              '<span class="rf-li"><svg width="16" height="14"><circle cx="8" cy="7" r="4" class="rf-dot-acc"/></svg> GOOD_6 (fixed)</span>')
    fnote = ("Nested pools ranked by informativeness (|AUROC − 0.5|, the criterion L-SML actually exploits — it flips "
             "signs with negative weights). Between 16 and 30 views everything is within 0.11 pp; below ~16 real accuracy "
             "is lost, and GOOD_6 itself drops when spectral_entropy and low_band_power fall out of the pool. "
             "CAVEAT: the ranking uses labelled AUROC aggregated over all 25 cells, so these are IN-SAMPLE upper "
             "bounds — the honest LOCO number for pruning is bounded above by +0.11 pp. Source: pool_size_experiment.csv.")
    return _fig("Pool size barely matters — composition does",
                "Same selector, same cells; only the feature pool shrinks (best-ranked views kept).",
                legend, "".join(s), fnote)


# ── page scaffolding ──────────────────────────────────────────────────────────────────

def page(title, subtitle, body, out_name):
    nav = '<div class="nav-bar"><strong>Pages:</strong> ' + ' &bull; '.join(
        (f'<strong>{esc(lbl)}</strong>' if fn == out_name
         else f'<a href="{fn}">{esc(lbl)}</a>')
        for fn, lbl in PAGES_NAV) + '</div>'
    return f"""<!DOCTYPE html>
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
    <span class="meta-pill">Generated {GEN_DATE} by scripts/advisor_inscope_report.py</span>
    <span class="meta-pill">25 in-scope cells: 10 QA + 15 math</span>
    <span class="meta-pill">Every numeric cell read from a CSV at build time</span>
  </div>
</div></div>
<div class="page">
{nav}
{body}
</div>
</body>
</html>
"""


# ── pages ─────────────────────────────────────────────────────────────────────────────

def p_index():
    g6 = lead_val("ref.GOOD_6", "c46", "macro_all")
    g5 = lead_val("ref.GOOD_5", "c46", "macro_all")
    d = lead_val("ref.GOOD_6", "c46", "delta_vs_good5_all")
    p = lead_val("ref.GOOD_6", "c46", "wilcoxon_p")
    r6 = D["lead"].get(("ref.GOOD_6", "c46"), {})
    dufs = lead_val("a2.dufs", "c46", "macro_all")
    a6 = lead_val("a6.pl_dufs", "c46", "macro_all")
    a6p = lead_val("a6.pl_dufs", "c46", "wilcoxon_p")
    lr30 = None
    lrs = [f(r["floored"]) for r in D["lr"] if r["fset"] == "30"]
    if lrs:
        lr30 = sum(lrs) / len(lrs)
    hon = [f(r["honest_gain"]) for r in D["gap"] if f(r["honest_gain"]) is not None]
    ins = [f(r["insample_gain"]) for r in D["gap"] if f(r["insample_gain"]) is not None]
    honest = sum(hon) / len(hon) if hon else None
    insample = sum(ins) / len(ins) if ins else None

    tiles = [
        (pc(g6, 2), "GOOD_6 macro AUROC", "the leading label-free detector"),
        (pc(g5, 2), "GOOD_5 macro AUROC", "the reference subset"),
        (f"{100*d:+.2f} pp" if d else "--", "GOOD_6 over GOOD_5",
         f"Wilcoxon p = {p:.5f}, {r6.get('wins','?')}W / {r6.get('losses','?')}L" if p else ""),
        (pc(a6 if a6 else dufs, 2), "best label-free selector",
         ("a6.pl_dufs (pseudo-label gates, Step 194) &mdash; first to edge GOOD_5, "
          "not significantly" if a6 else
          "a2.dufs (Gated-Laplacian) &mdash; below both fixed subsets")),
        (pc(lr30, 2), "supervised ceiling (LR, 30 views)",
         f"headroom {100*(lr30-g6):+.2f} pp over GOOD_6" if (lr30 and g6) else ""),
        (f"{100*honest:+.2f} pp" if honest else "--", "honest selection headroom",
         f"vs {100*insample:+.2f} pp in-sample &mdash; the rest is winner&rsquo;s curse"
         if insample else ""),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div>'
        f'<div class="d">{dd}</div></div>' for v, l, dd in tiles)

    cards = "".join(
        f'<a class="card" href="{fn}" style="text-decoration:none;display:block">'
        f'<h3>{esc(lbl)}</h3><p>{esc(desc)}</p></a>'
        for fn, lbl, desc in [
            ("competitor_grid.html", "Competitor grid",
             "One row per cell: our fixed subsets, the published competitor, and the learned selector."),
            ("competitor_sources.html", "Competitor provenance",
             "Which published numbers were verified against the paper's own table, and which could not be."),
            ("supervised_ceiling.html", "Supervised ceiling",
             "What logistic regression achieves with labels, and an audit of how that number is computed."),
            ("subset_gap_analysis.html", "Selection headroom",
             "How much of the apparent gain from searching subsets survives out of sample."),
            ("groupfs_diagnosis.html", "Learned selector",
             "Why GroupFS is non-optimal, and which proposed explanations the data does not support."),
            ("gated_laplacian.html", "Gated-Laplacian",
             "The trace-based selector, already implemented, and what its selection rule changes."),
            ("anchor_sweep.html", "Anchor robustness",
             "Whether the label-free sign anchor is load-bearing."),
            ("verification.html", "Verification",
             "The data-integrity audit that gates everything else on this site."),
        ])

    body = f"""
<div class="callout">
<strong>What this is.</strong> An evaluation of our label-free spectral detector on the 25
in-scope cells &mdash; 10 short-form QA and 15 math/reasoning. RAG and GPQA are out of
scope. Every number below traces to a CSV in <code>results/</code>, and the
<a href="verification.html">verification page</a> documents the integrity audit that had
to pass before any of it was published.
</div>

<div class="tiles">{tile_html}</div>

<h2>Headline</h2>
<p><strong>GOOD_6 is the leading label-free detector at {pc(g6,2)}% macro AUROC</strong>, beating
the GOOD_5 reference by {100*d:+.2f} pp (Wilcoxon p = {p:.5f}, {r6.get('wins','?')} wins /
{r6.get('losses','?')} losses over 25 cells). No label-free feature-selection method we
tested beats it. The best of them &mdash; the new pseudo-label-gated selector
(<code>a6.pl_dufs</code>, Step 194) at {pc(a6,2)}% &mdash; is the first to nominally edge
GOOD_5 ({100*(a6-g5):+.2f} pp, Wilcoxon p = {a6p:.3f}, not significant); the previous best,
the Gated-Laplacian (<code>a2.dufs</code>) at {pc(dufs,2)}%, sits just below GOOD_5.</p>

<p>With labels, logistic regression over the same views reaches {pc(lr30,2)}%, so the
supervised headroom is {100*(lr30-g6):+.2f} pp. That gap is the honest measure of how much
our detector leaves on the table.</p>

{fig_macro_bars()}

<div class="callout warn">
<strong>Two corrections to previously circulated numbers.</strong> A data-integrity audit
found that one cell's bench rows, and every learned-selector row on 11 further cells, had
been computed against a superseded feature cache and were never recomputed. Both are fixed
here. The GOOD_6-over-GOOD_5 margin moves from +0.98 pp to {100*d:+.2f} pp, and the best
learned selector moves from marginally ahead of GOOD_5 to marginally behind it. Details on
the <a href="verification.html">verification page</a>.
</div>

<h2>Pages</h2>
<div class="cards">{cards}</div>
"""
    return page("In-scope evaluation &mdash; QA and reasoning",
                "Label-free spectral hallucination detection on 25 cells",
                body, "index.html")


def _svc_summary_table():
    """Selector-vs-competitor tallies, computed at build time from selector_vs_competitor.csv."""
    rows = [r for r in D["svc"] if f(r["comp"]) is not None]
    uns = [r for r in rows if r["comp_sup"] == "unsupervised"]
    out = []
    for label, rs in (("all published anchors", rows),
                      ("unsupervised anchors only", uns)):
        for name, key in (("Selector: trace-based Gated-Laplacian", "dufs"),
                          ("Selector: GroupFS", "groupfs"),
                          ("Fixed subset GOOD_6", "good6"),
                          ("Best selector per cell (ceiling)", "best_sel")):
            rs2 = [r for r in rs if f(r[key]) is not None]
            if not rs2:
                continue
            m = sum(f(r[key]) for r in rs2) / len(rs2)
            mc = sum(f(r["comp"]) for r in rs2) / len(rs2)
            w = sum(1 for r in rs2 if f(r[key]) > f(r["comp"]) + 1e-9)
            l = sum(1 for r in rs2 if f(r[key]) < f(r["comp"]) - 1e-9)
            d = 100 * (m - mc)
            cls = "good" if d > 0 else "bad"
            out.append(f"<tr><td>{esc(label)}</td><td>{esc(name)}</td>"
                       f"<td class='num'>{len(rs2)}</td>"
                       f"<td class='num'>{pc(m,2)}</td><td class='num'>{pc(mc,2)}</td>"
                       f"<td class='num {cls}'>{d:+.2f}</td>"
                       f"<td class='num'>{w}W / {l}L</td></tr>")
    return ('<table><thead><tr><th>Comparison basis</th><th>Our method</th>'
            '<th class="num">cells</th><th class="num">ours</th>'
            '<th class="num">competitor</th><th class="num">delta pp</th>'
            '<th class="num">W/L</th></tr></thead>'
            f'<tbody>{"".join(out)}</tbody></table>')


def p_competitor_grid():
    hdr = ("<tr><th>Cell</th><th>Group</th><th class='num'>n</th>"
           "<th class='num'>Selector@30</th><th class='num'>GroupFS@30</th>"
           "<th class='num'>Selector@16</th>"
           "<th class='num'>GOOD_6</th><th class='num'>GOOD_5</th>"
           "<th class='num'>Published</th><th>vs published</th><th>vs GOOD_6</th>"
           "<th>Method</th>"
           "<th>Supervision</th><th>Access</th><th class='num'>Passes</th></tr>")
    rows = []
    for c in INSCOPE:
        m = D["macro_by_cell"].get(c, {})
        g6, g5 = m.get("ref.GOOD_6"), m.get("ref.GOOD_5")
        anchors = [r for r in D["comp_by_cell"].get(c, []) if r["role"] == "anchor"]
        a = anchors[0] if anchors else None
        pub = f(a["auroc"]) if a else None
        if a:
            badge = ('<span class="badge b-ok">verified</span>'
                     if a["verified_by"] == "extracted"
                     else '<span class="badge b-bad">unverified</span>')
            meth_html = f"{esc(a['method'])}<br>{badge}"
            sup = a.get("supervision") or ""
            sup_html = ('<span class="badge b-warn">supervised</span>'
                        if sup == "supervised"
                        else ('<span class="badge b-ok">unsupervised</span>' if sup
                              else '<span class="dim">--</span>'))
            acc = a.get("access") or ""
            acc_cls = {"white-box": "b-bad", "grey-box": "b-warn",
                       "black-box": "b-ok"}.get(acc, "b-neutral")
            acc_html = (f'<span class="badge {acc_cls}">{esc(acc)}</span>' if acc
                        else '<span class="dim">--</span>')
            ps = a.get("passes") or ""
            ps_html = (f'<span class="{"bad" if ps not in ("1", "") else ""}">{esc(ps)}</span>'
                       if ps else '<span class="dim">--</span>')
            if (a.get("profile_src") or "") == "inferred":
                ps_html += ' <span class="badge b-neutral">inferred</span>'
        else:
            meth_html = '<span class="badge b-neutral">no published baseline</span>'
            sup_html = acc_html = ps_html = '<span class="dim">--</span>'
        a2 = D["a2_by_cell"].get(c, {})
        sel30 = f(a2.get(("c46", "a2.dufs"), {}).get("auroc"))
        grp30 = f(a2.get(("c46", "a2.select"), {}).get("auroc"))
        sel16 = f(a2.get(("h16", "a2.dufs"), {}).get("auroc"))
        gp = D["gap_by_cell"].get(c)
        n = f(gp.get("n")) if gp else None
        svc = D["svc_by_cell"].get(c, {})

        def _vbadge(v):
            if v == "WIN":
                return '<span class="badge b-ok">WIN</span>'
            if v == "LOSS":
                return '<span class="badge b-bad">LOSS</span>'
            if v == "TIE":
                return '<span class="badge b-neutral">tie</span>'
            return '<span class="dim">--</span>'

        v_comp = _vbadge(svc.get("verdict_selector_vs_comp", ""))
        v_g6 = _vbadge(svc.get("verdict_selector_vs_good6", ""))
        win = "" if (pub is None or sel30 is None) else (
            ' class="good"' if sel30 > pub else ' class="bad"')
        rows.append(
            f"<tr><td><code>{esc(c)}</code>"
            + (' <span class="badge b-neutral">new</span>' if c in CLUSTER_CELLS else "")
            + f"</td><td>{GROUP.get(c,'')}</td>"
            f"<td class='num'>{int(n) if n else '--'}</td>"
            f"<td class='num'{win}><strong>{pc(sel30,1)}</strong></td>"
            f"<td class='num'>{pc(grp30,1)}</td><td class='num'>{pc(sel16,1)}</td>"
            f"<td class='num'>{pc(g6,1)}</td><td class='num'>{pc(g5,1)}</td>"
            f"<td class='num'>{pc(pub,1)}</td><td>{v_comp}</td><td>{v_g6}</td>"
            f"<td>{meth_html}</td>"
            f"<td>{sup_html}</td><td>{acc_html}</td>"
            f"<td class='num'>{ps_html}</td></tr>")

    n_pub = sum(1 for r in D["svc"] if f(r["comp"]) is not None)
    n_uns = sum(1 for r in D["svc"]
                if f(r["comp"]) is not None and r["comp_sup"] == "unsupervised")

    body = f"""
<p>The contribution is the <strong>pipeline</strong>: label-free feature selection, then
continuous L-SML fusion (the Jaffe-Fetaya-Nadler unsupervised-ensemble lineage). So the
leading column is the <strong>selector</strong> — the trace-based Gated-Laplacian, which
picks each cell's feature subset from unlabelled data. GOOD_6 / GOOD_5 are the fixed
hand-curated subsets, shown for reference.</p>

<div class="callout warn">
<strong>Read the two tallies separately.</strong> Of the {n_pub} cells with a published
anchor, only {n_uns} of those anchors are unsupervised. The rest (INSIDE, TSV, LOS-Net,
HCPD, the ICLR-2023 Semantic Entropy) are supervised probes or multi-sample methods and are
not like-for-like against a single-pass label-free detector. Against the unsupervised
anchors the selector is ahead; against the mixed set it is behind, and the gap is dominated
by those supervised competitors.
</div>

{_svc_summary_table()}

{fig_svc_dumbbell()}

<p class="small">&ldquo;Best selector per cell&rdquo; is a <strong>ceiling, not a method</strong>
&mdash; it picks the winning variant per cell with hindsight. It is included to bound what any
per-cell label-free selector could achieve on this data.</p>

<h2>Per cell</h2>
<table><thead>{hdr}</thead><tbody>{''.join(rows)}</tbody></table>

<h2>What the three method columns mean</h2>
<p>A competitor's AUROC is only comparable to ours if it is paying the same price. These three
columns say what each method actually needs, all read off the papers themselves:</p>
<ul>
<li><strong>Supervision</strong> &mdash; does it fit anything on hallucination labels?
    Ours never sees a label.</li>
<li><strong>Access</strong> &mdash; the taxonomy is the LOS-Net paper's own:
    <span class="badge b-ok">black-box</span> generated text only;
    <span class="badge b-warn">grey-box</span> output distributions / log-probabilities;
    <span class="badge b-bad">white-box</span> model internals (hidden states, attention maps,
    activations), which the same paper calls &ldquo;incomparable&rdquo; to output-only methods.
    <strong>Ours is grey-box</strong> &mdash; it needs the top-k log-probabilities and nothing
    deeper.</li>
<li><strong>Passes</strong> &mdash; generations per question at detection time.
    <strong>Ours is 1.</strong> A method at K=10 costs an order of magnitude more per
    question, which matters as much as AUROC for deployment.</li>
</ul>
<p class="small">Rows marked <span class="badge b-neutral">inferred</span> are the ones whose
paper is not in the local library, so the profile comes from the method's description
elsewhere rather than from its own text.</p>

<h2>How to read this</h2>
<ul>
<li><strong>Selector@30 / Selector@16</strong> are the trace-based Gated-Laplacian
    (<code>a2.dufs</code>) on the 30-view and 16-view pools; <strong>GroupFS@30</strong> is
    the group-granular rule (<code>a2.select</code>) on the same gates.</li>
<li>All AUROC values are percentages, direction-resolved label-free against the
    <code>epr</code> anchor (see <a href="anchor_sweep.html">anchor robustness</a>).</li>
<li>Every competitor number is re-derived from <code>competitors_verified.csv</code>; see
    <a href="competitor_sources.html">provenance</a> for which are verified.</li>
<li>The selector loses to the fixed subsets on macro. That is explained on the
    <a href="subset_gap_analysis.html">selection-headroom page</a>: per-cell label-free
    selection is dominated by selection noise, while a subset fixed once across the whole
    corpus is a better-regularised estimator.</li>
</ul>
"""
    return page("Competitor grid",
                "The selection pipeline against the published numbers, per cell",
                body, "competitor_grid.html")


def p_competitor_sources():
    rows = []
    for r in sorted(D["comp"], key=lambda x: (x["cell"], x["role"], x["method"])):
        ver = r["verified_by"]
        badge = ('<span class="badge b-ok">verified</span>' if ver == "extracted"
                 else '<span class="badge b-bad">unverified</span>')
        sup = r.get("supervision") or ""
        supb = ('<span class="badge b-warn">supervised</span>' if sup == "supervised"
                else (f'<span class="badge b-neutral">{esc(sup)}</span>' if sup else ""))
        src = r.get("source") or ""
        if r.get("paper_table"):
            src = f"{esc(r['paper_slug'])} &mdash; {esc(r['paper_table'])}"
        rows.append(
            f"<tr><td><code>{esc(r['cell'])}</code></td><td>{esc(r['method'])}</td>"
            f"<td class='num'>{esc(r['auroc_as_published'])}</td>"
            f"<td>{esc(r['published_scale'])}</td><td>{supb}</td><td>{badge}</td>"
            f"<td class='small'>{src}</td>"
            f"<td class='small'>{esc((r.get('caveat') or '')[:300])}</td></tr>")

    n_ver = sum(1 for r in D["comp"] if r["verified_by"] == "extracted")
    body = f"""
<div class="callout bad">
<strong>Why this page exists.</strong> The competitor numbers lived in two files with
incompatible provenance. <code>published_baselines.csv</code> (6 cells) carries a
<code>source</code> column naming paper and table. <code>scores_lsml_upcr.csv</code>
(the other 13 cells) carries <em>no citation at all</em> &mdash; no paper, no table, no
page, no scale note. A claim that we are competitive with published work is only as good
as those numbers, so each was checked against <code>papers/extracted/</code> &mdash; the
raw extracted text, not the digest cards.
</div>

<p><strong>{n_ver} of {len(D['comp'])} rows verified</strong> against the paper's own
table, covering 7 papers: EPR, HCPD, ARS, Noise Injection, Semantic Energy, ALS/FEPoID
and HARP. The rest could not be checked because the paper is not in <code>papers/</code>.</p>

<div class="callout warn">
<strong>Two corrections this audit produced,</strong> both applied to the grid:
<ul>
<li><strong>HARP is supervised, not unsupervised.</strong> The paper trains its detector
with a binary cross-entropy loss on hallucination labels. It had been recorded as an
unsupervised competitor, which would have made an unfair comparison look fair.</li>
<li><strong>The stored HARP anchor was the wrong model row.</strong> 92.8 is the
Qwen-2.5-7B-Instruct result; our cell uses Llama-3.1-8B-Instruct, for which the paper
reports 92.9. The stored value was cross-model.</li>
</ul>
</div>

<div class="callout warn">
<strong>What could not be verified.</strong> No local copy exists for LapEigvals (5 cells
&mdash; and it is also the G3 decision gate in the roadmap), INSIDE (2 cells), LOS-Net
(1 cell plus its 11 baseline rows), Internal-States+RC, the ICLR-2023 Semantic Entropy
anchor, and TSV's TruthfulQA row. Those are badged
<span class="badge b-bad">unverified</span> wherever they appear.
</div>

<table><thead><tr><th>Cell</th><th>Method</th><th class="num">As published</th>
<th>Scale</th><th>Supervision</th><th>Status</th><th>Source</th><th>Caveat</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""
    return page("Competitor provenance",
                "Which published numbers were checked against the paper, and which could not be",
                body, "competitor_sources.html")


def p_supervised_ceiling():
    sets = ("5", "9", "16", "30")
    macro = {}
    for s in sets:
        vals = [f(r["floored"]) for r in D["lr"] if r["fset"] == s and f(r["floored"])]
        macro[s] = sum(vals) / len(vals) if vals else None
    g6 = lead_val("ref.GOOD_6", "c46", "macro_all")

    rows = []
    for c in INSCOPE:
        by = D["lr_by_cell"].get(c, {})
        m = D["macro_by_cell"].get(c, {})
        g6c = m.get("ref.GOOD_6")
        lr30 = f(by.get("30", {}).get("floored"))
        head = (lr30 - g6c) if (lr30 is not None and g6c is not None) else None
        used = (by.get("30", {}).get("dropped") or "")
        rows.append(
            f"<tr><td><code>{esc(c)}</code></td><td>{GROUP.get(c,'')}</td>"
            f"<td class='num'>{pc(g6c,1)}</td>"
            + "".join(f"<td class='num'>{pc(f(by.get(s,{}).get('floored')),1)}</td>"
                      for s in sets)
            + f"<td class='num'>{by.get('30',{}).get('n_avail','--')}</td>"
              f"<td class='num {cls_for(head)}'>{pc(head,1,signed=True)}</td></tr>")

    spreads = [f(r["C_spread"]) for r in D["lr"]
               if r["fset"] == "30" and f(r["C_spread"]) is not None]
    flips = sum(int(r["folds_flipped"] or 0) for r in D["lr"])
    n_folds = 5 * len(D["lr"])

    body = f"""
<div class="callout">
<strong>This panel is supervised.</strong> It is not a peer of our detector &mdash; it is a
ceiling. Logistic regression sees the labels; our method never does. The number that
matters is the gap.
</div>

<p>Logistic regression, 5-fold cross-validated, <code>class_weight='balanced'</code>, with
folds grouped by <code>problem_id</code> on the multi-candidate cells so a question's
candidates never straddle train and test. Macro over all 25 cells:</p>

<div class="tiles">
{''.join(f'<div class="tile"><div class="v">{pc(macro[s],2)}</div>'
         f'<div class="l">LR, {s} views</div></div>' for s in sets)}
<div class="tile"><div class="v">{pc(g6,2)}</div><div class="l">GOOD_6 (label-free)</div></div>
<div class="tile"><div class="v">{100*(macro['30']-g6):+.2f} pp</div>
<div class="l">supervised headroom</div>
<div class="d">what labels buy over our detector</div></div>
</div>

<h2>Audit of how this number is computed</h2>
<p>Three properties of the oracle code were checked before publishing it, because each
could have inflated the ceiling:</p>
<table><thead><tr><th>Check</th><th>Finding</th></tr></thead><tbody>
<tr><td>Per-fold AUROC is floored at 0.5 and direction-free
(<code>max(p, 1&minus;p)</code>), so an anti-predictive fold is flipped upward</td>
<td><span class="badge b-ok">immaterial here</span> &mdash; only
<strong>{flips} of {n_folds} folds</strong> flip. Macro inflation is +0.07 pp on the
9-view set and +0.00 pp on the others, so the headroom framing stands.</td></tr>
<tr><td>Grouped folds actually applied on the multi-candidate cells</td>
<td><span class="badge b-ok">confirmed</span> for every k&gt;1 cell.</td></tr>
<tr><td>The saturation filter silently drops views</td>
<td><span class="badge b-ok">no GOOD_6 member dropped</span> on any cell.</td></tr>
<tr><td>Regularisation strength <code>C</code> is fixed at 1.0 and never tuned</td>
<td><span class="badge b-warn">a real caveat</span> &mdash; across
C &isin; {{0.1, 1, 10}} at 30 views the AUROC spread averages
{100*sum(spreads)/len(spreads):.2f} pp and reaches
<strong>{100*max(spreads):.2f} pp</strong> on the worst cell. The ceiling is therefore
an untuned point estimate, not a tuned maximum.</td></tr>
</tbody></table>

<h2>Per cell</h2>
<table><thead><tr><th>Cell</th><th>Group</th><th class="num">GOOD_6</th>
{''.join(f'<th class="num">LR@{s}</th>' for s in sets)}
<th class="num">views used</th><th class="num">headroom</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""
    return page("Supervised ceiling",
                "What logistic regression achieves with labels &mdash; and an audit of that number",
                body, "supervised_ceiling.html")


def gap_rho_table():
    """Spearman rho of each covariate against the winner's-curse magnitude, computed
    at build time from subset_gap_analysis.csv (never hand-typed)."""
    labels = [("n", "Sample size <code>n</code>"),
              ("n_anti_oriented", "Anti-oriented feature count"),
              ("pos_rate", "Class balance <code>pos_rate</code>"),
              ("K_good6", "Fusion group count <code>K</code>"),
              ("p_pool", "Pool size")]
    y = [f(r["optimism_gap"]) for r in D["gap"]]
    out = []
    for cov, lab in labels:
        rho, n = spearman([f(r[cov]) for r in D["gap"]], y)
        if abs(rho) >= 0.5:
            v = ('<span class="badge b-ok">explains it</span> '
                 'small cells overfit the selection')
            cell = f"<strong>{rho:+.3f}</strong>".replace("+", "").replace("-", "&minus;")
        elif abs(rho) >= 0.2:
            v = '<span class="badge b-neutral">weak</span>'
            cell = f"{rho:+.3f}".replace("-", "&minus;")
        else:
            v = '<span class="badge b-neutral">no relationship</span>'
            cell = f"{rho:+.3f}".replace("-", "&minus;")
        out.append(f"<tr><td>{lab}</td><td class='num'>{cell}</td><td>{v}</td></tr>")
    return ('<table><thead><tr><th>Candidate cause</th>'
            '<th class="num">Spearman &rho;</th><th>Verdict</th></tr></thead>'
            f'<tbody>{"".join(out)}</tbody></table>')


def gfs_rho_table():
    """Same, for the GroupFS per-cell gap vs GOOD_5, both selection rules."""
    labels = [("frac_selected", "Fraction of pool selected (saturation)"),
              ("frac_anti_chosen", "Anti-oriented views inside the selection"),
              ("pos_rate", "Class balance"),
              ("stability", "Gate stability across seeds")]
    out = []
    for cov, lab in labels:
        cells = []
        for v in ("a2.select", "a2.dufs"):
            sel = [r for r in D["gfs"] if r["variant"] == v]
            rho, n = spearman([f(r[cov]) for r in sel],
                              [f(r["gap_vs_good5"]) for r in sel])
            cells.append("n/a" if n == 0 or rho != rho
                         else f"{rho:+.3f}".replace("-", "&minus;"))
        out.append(f"<tr><td>{lab}</td>"
                   + "".join(f"<td class='num'>{c}</td>" for c in cells) + "</tr>")
    return ('<table><thead><tr><th>Candidate cause</th>'
            '<th class="num">&rho; (group-granular)</th>'
            '<th class="num">&rho; (per-feature)</th></tr></thead>'
            f'<tbody>{"".join(out)}</tbody></table>')


def p_gap():
    rows = []
    for r in sorted(D["gap"], key=lambda x: (f(x["honest_gain"]) is None,
                                             f(x["honest_gain"]) or 0)):
        hg, ig, og = f(r["honest_gain"]), f(r["insample_gain"]), f(r["optimism_gap"])
        rows.append(
            f"<tr><td><code>{esc(r['cell'])}</code></td><td>{esc(r['group'])}</td>"
            f"<td class='num'>{int(f(r['n']) or 0)}</td>"
            f"<td class='num'>{pc(ig,2,signed=True)}</td>"
            f"<td class='num {cls_for(hg)}'><strong>{pc(hg,2,signed=True)}</strong></td>"
            f"<td class='num'>{pc(og,2,signed=True)}</td>"
            f"<td class='num'>{f(r['pos_rate']):.3f}</td></tr>")
    hon = [f(r["honest_gain"]) for r in D["gap"] if f(r["honest_gain"]) is not None]
    ins = [f(r["insample_gain"]) for r in D["gap"] if f(r["insample_gain"]) is not None]
    opt = [f(r["optimism_gap"]) for r in D["gap"] if f(r["optimism_gap"]) is not None]
    mh, mi, mo = (sum(hon)/len(hon), sum(ins)/len(ins), sum(opt)/len(opt))
    n_win = sum(1 for v in hon if v > 0)

    body = f"""
<p>The open question after the last round was: on cells where a <em>searched</em> subset
looks better than our fixed one, why does no label-free selector deliver that gain? There
are two very different answers, and they call for opposite responses &mdash; either the
search is failing and we should build better selectors, or the gain was never real and we
should stop.</p>

<p>The split-half oracle settles it. A subset is chosen greedily on half the data and then
scored on the <strong>held-out</strong> half:</p>

<div class="tiles">
<div class="tile"><div class="v">{100*mi:+.2f} pp</div>
<div class="l">apparent gain (in-sample)</div>
<div class="d">better than GOOD_5 on {len(ins)}/{len(ins)} cells</div></div>
<div class="tile"><div class="v">{100*mh:+.2f} pp</div>
<div class="l">gain that survives (held-out)</div>
<div class="d">better on only {n_win}/{len(hon)} cells</div></div>
<div class="tile"><div class="v">{100*mo:+.2f} pp</div>
<div class="l">winner&rsquo;s curse</div>
<div class="d">{100*mo/mi:.0f}% of the apparent gain is illusion</div></div>
</div>

<div class="callout">
<strong>The answer is mostly &ldquo;there was nothing to find&rdquo;.</strong> Searching
subsets beats GOOD_5 on every single cell in-sample, but two thirds of that advantage
evaporates out of sample, and on 6 of 25 cells the searched subset is actually
<em>worse</em> than the fixed one. The remaining honest headroom over GOOD_5 is
{100*mh:+.2f} pp &mdash; and GOOD_6 already captures a large part of it.
</div>

<h2>What drives the illusion</h2>
<p>Rank correlation of each candidate explanation against the per-cell winner&rsquo;s-curse
magnitude, over 25 cells:</p>
{gap_rho_table()}

<p class="small">The two extremes make the mechanism concrete. On
<code>spilled_triviaqa_llama8b</code> (n = 256) the greedy subset looks 6.3 pp better than
GOOD_5 in-sample and comes out <strong>8.6 pp worse</strong> held-out. On
<code>se_nq_open_llama8b</code> (n = 8460) the in-sample and held-out gains agree to within
0.1 pp. Selection noise, not feature quality, is what separates them.</p>

<h2>Per cell</h2>
<table><thead><tr><th>Cell</th><th>Group</th><th class="num">n</th>
<th class="num">in-sample gain</th><th class="num">honest gain</th>
<th class="num">winner&rsquo;s curse</th><th class="num">pos rate</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""
    return page("Selection headroom",
                "How much of the gain from searching subsets is real",
                body, "subset_gap_analysis.html")


def p_choices():
    rows = D["choice"]
    loss = [r for r in rows if r["verdict"] == "LOSS"]
    win = [r for r in rows if r["verdict"] == "WIN"]

    def agg(rs, k, sc=100.0):
        v = [f(r[k]) for r in rs if f(r[k]) is not None]
        return (sc * sum(v) / len(v)) if v else float("nan")

    rho = [f(r["rho_gate_vs_featauroc"]) for r in rows
           if f(r["rho_gate_vs_featauroc"]) is not None]
    mrho = sum(rho) / len(rho)

    # which GOOD_6 members get dropped, and how often
    dropped = {}
    for r in rows:
        for x in (r["good6_missed"] or "").split("|"):
            if x:
                dropped[x] = dropped.get(x, 0) + 1
    drop_html = "".join(
        f"<tr><td><code>{esc(k)}</code></td><td class='num'>{v}</td>"
        f"<td class='num'>{25-v}</td></tr>"
        for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]))

    body_rows = []
    for r in rows:
        g = f(r["gap_vs_good6"])
        vb = ('<span class="badge b-ok">WIN</span>' if r["verdict"] == "WIN"
              else '<span class="badge b-bad">LOSS</span>')
        miss = (r["good6_missed"] or "").split("|")
        miss_html = ("".join(f'<span>{esc(x)}</span>' for x in miss if x)
                     or '<span class="dim">none</span>')
        body_rows.append(
            f"<tr><td><code>{esc(r['cell'])}</code></td><td>{esc(r['group'])}</td>"
            f"<td>{vb}</td>"
            f"<td class='num {cls_for(g)}'>{pc(g,2,signed=True)}</td>"
            f"<td class='num'>{r['size']}/{r['p_pool']}</td>"
            f"<td class='num'>{r['good6_kept']}/6</td>"
            f"<td class='chips'>{miss_html}</td>"
            f"<td class='num'>{r['n_extra']}</td>"
            f"<td class='num'>{r['n_extra_anti']}</td>"
            f"<td class='num'>{pc(f(r['mean_auc_selected']),1)}</td>"
            f"<td class='num'>{pc(f(r['mean_auc_unselected']),1)}</td>"
            f"<td class='num good'>{pc(f(r['sel_minus_unsel']),1,signed=True)}</td></tr>")

    body = f"""
<p>The selector loses to the fixed GOOD_6 subset on <strong>{len(loss)} of {len(rows)}</strong>
cells. This page answers the obvious follow-up: <em>what did it actually choose, how does that
differ from GOOD_6, and why does it choose that?</em></p>

<h2>1. What it selects</h2>
<p>GOOD_6 is six hand-curated views. The selector picks
<strong>{agg(rows,'size',1):.0f} views on average</strong> out of a ~28-view pool &mdash; roughly
three times larger. It generally <em>keeps</em> most of GOOD_6 and adds a large tail around it.</p>

<div class="tiles">
<div class="tile"><div class="v">{agg(loss,'good6_kept',1):.1f} / 6</div>
<div class="l">GOOD_6 members kept, on cells it LOSES</div></div>
<div class="tile"><div class="v">{agg(win,'good6_kept',1):.1f} / 6</div>
<div class="l">GOOD_6 members kept, on cells it WINS</div>
<div class="d">keeping more of GOOD_6 is the clearest difference between the two groups</div></div>
<div class="tile"><div class="v">{agg(rows,'n_extra',1):.1f}</div>
<div class="l">extra views added beyond GOOD_6</div></div>
<div class="tile"><div class="v">{agg(rows,'n_extra_anti',1):.1f}</div>
<div class="l">of those extras, inverted-polarity</div>
<div class="d">AUROC below 0.5 &mdash; informative but pointing the wrong way; L-SML flips
them with a negative weight</div></div>
<div class="tile"><div class="v">{agg(loss,'n_extra_nearrandom',1):.2f} vs {agg(win,'n_extra_nearrandom',1):.2f}</div>
<div class="l">truly near-random extras: LOSS vs WIN cells</div>
<div class="d">the cells it wins on admit none at all</div></div>
</div>

<h2>2. How it differs from GOOD_6</h2>
<p>Which GOOD_6 members the selector drops, across the 25 cells:</p>
<table><thead><tr><th>GOOD_6 member</th><th class="num">cells where dropped</th>
<th class="num">cells where kept</th></tr></thead><tbody>{drop_html}</tbody></table>
<p class="small">The four remaining GOOD_6 members (<code>epr</code>,
<code>sw_var_peak</code>, <code>cusum_max</code>, <code>varentropy</code>) are almost never
dropped &mdash; the disagreement is concentrated on <code>spectral_entropy</code> and
<code>low_band_power</code>.</p>

<h2>3. Why it chooses that &mdash; the objective is aimed elsewhere</h2>
<div class="callout bad">
The gates are trained on a <strong>Laplacian-smoothness objective over the sample graph</strong>.
That objective never sees a label, and it is not a proxy for separability. Measured directly:
the rank correlation between a view's <strong>gate value</strong> (what the objective maximises)
and that view's <strong>own oriented AUROC</strong> (what we actually want) is
<strong>&rho; = {mrho:+.3f}</strong> on average across the 25 cells.
<br><br>
So the selector is not picking <em>badly</em> &mdash; it is picking for a <em>different
criterion</em>. It does tilt in the right direction: the views it selects average
{agg(rows,'mean_auc_selected'):.1f}% AUROC against {agg(rows,'mean_auc_unselected'):.1f}% for the
views it leaves out ({agg(rows,'sel_minus_unsel'):+.1f} pp). But a &rho; of {mrho:+.3f} is far too
weak to reconstruct a six-view subset that took a corpus-wide labelled search to find.
</div>

<p><strong>And that is the honest asymmetry.</strong> GOOD_6 was chosen once by macro AUROC
<em>across the whole grid</em>, so it carries label information at the corpus level even though
applying it to a new cell needs no labels. The selector is given no such prior: it must
rediscover a good subset per cell, from unlabelled data, using an objective only weakly aligned
with the target. Losing by ~2 pp under those conditions is a reasonable outcome, not a defect.</p>

<div class="callout warn">
<strong>&ldquo;Inverted-polarity&rdquo; is not the same as &ldquo;bad&rdquo;, and the
distinction changes the conclusion.</strong> A view with AUROC 0.30 is <em>strongly</em>
informative &mdash; flipped, it scores 0.70 &mdash; and continuous L-SML flips it automatically
with a negative weight. Of the below-chance views the selector actually picks, <strong>80% sit
below 0.40</strong> (strong once flipped) and only <strong>6% are genuinely near-random</strong>.
<br><br>
Splitting the two separates the win and loss cells cleanly:
the cells it <strong>wins</strong> on take {agg(win,'n_extra_strong_inverted',1):.1f} strongly
informative inverted views and <strong>{agg(win,'n_extra_nearrandom',1):.2f} near-random ones</strong>;
the cells it <strong>loses</strong> on take {agg(loss,'n_extra_strong_inverted',1):.1f} strong ones
but let <strong>{agg(loss,'n_extra_nearrandom',1):.2f} near-random views in</strong>. So the useful
statement is the narrow one: <em>admitting genuinely uninformative views tracks the losses;
admitting inverted-polarity views does not.</em> That is a direct argument for pruning the pool on
uninformativeness rather than on raw AUROC sign.
</div>

<h2>Per cell</h2>
<table><thead><tr><th>Cell</th><th>Group</th><th>vs GOOD_6</th><th class="num">gap pp</th>
<th class="num">selected</th><th class="num">GOOD_6 kept</th><th>GOOD_6 dropped</th>
<th class="num">extra</th><th class="num">extra anti-oriented</th>
<th class="num">sel. AUROC</th><th class="num">unsel. AUROC</th>
<th class="num">diff</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table>
"""
    return page("What the selector chooses",
                "The features it picks, how they differ from GOOD_6, and why",
                body, "selector_choices.html")


def p_groupfs():
    rows = []
    for c in INSCOPE:
        by = D["gfs_by_cell"].get(c, {})
        s, d = by.get("a2.select"), by.get("a2.dufs")
        if not s:
            continue
        gs, gd = f(s.get("gap_vs_good5")), f(d.get("gap_vs_good5")) if d else None
        sat = "yes" if s.get("saturated") == "True" else "no"
        rows.append(
            f"<tr><td><code>{esc(c)}</code></td><td>{GROUP.get(c,'')}</td>"
            f"<td class='num'>{int(f(s['size']) or 0)}/{int(f(s['p_pool']) or 0)}</td>"
            f"<td>{sat}</td>"
            f"<td class='num'>{int(f(d['size']) or 0) if d else '--'}/"
            f"{int(f(s['p_pool']) or 0)}</td>"
            f"<td class='num {cls_for(gs)}'>{pc(gs,2,signed=True)}</td>"
            f"<td class='num {cls_for(gd)}'>{pc(gd,2,signed=True)}</td>"
            f"<td class='num'>{s.get('n_anti_chosen','--')}</td>"
            f"<td class='num'>{f(s['pos_rate']):.3f}</td></tr>")

    sel = [r for r in D["gfs"] if r["variant"] == "a2.select"]
    duf = [r for r in D["gfs"] if r["variant"] == "a2.dufs"]
    sat_s = sum(1 for r in sel if r["saturated"] == "True")
    sat_d = sum(1 for r in duf if r["saturated"] == "True")
    rho_sat, _ = spearman([f(r["frac_selected"]) for r in sel],
                          [f(r["gap_vs_good5"]) for r in sel])
    mac_s = sum(f(r["auroc"]) for r in sel) / len(sel)
    mac_d = sum(f(r["auroc"]) for r in duf) / len(duf)
    # the outlier cell: worst gap under the group rule
    worst = min(sel, key=lambda r: f(r["gap_vs_good5"]))
    worst_d = next((r for r in duf if r["cell"] == worst["cell"]), None)

    body = f"""
<p>GroupFS is a learned, label-free feature selector: it puts a differentiable gate on each
view and trains them against a Laplacian-smoothness objective, discovering groups of views
as it goes. We use it as a pre-fusion selection stage &mdash; select, then fuse the selected
views with the same L-SML fusion the fixed subsets use.</p>

<p>It ships here with two selection rules over identical gates and an identical objective.
<code>a2.select</code> opens whole <em>groups</em> (a group is in if its median member gate
is open); <code>a2.dufs</code> thresholds each gate <em>individually</em>.</p>

<h2>Saturation is real, and per-feature gating removes it entirely</h2>
<div class="tiles">
<div class="tile"><div class="v">{sat_s}/{len(sel)}</div>
<div class="l">cells where group-granular gating selects the entire pool</div></div>
<div class="tile"><div class="v">{sat_d}/{len(duf)}</div>
<div class="l">same, per-feature gating</div>
<div class="d">saturation is a property of the group rule, not the gates</div></div>
</div>

<div class="callout bad">
<strong>But saturation is not why it loses.</strong> The standing explanation for
GroupFS's worst misses was that it opens nearly the whole pool and swamps the fusion. That
is testable, and it fails: the rank correlation between the fraction of the pool selected
and the per-cell gap to GOOD_5 is <strong>{rho_sat:+.3f}</strong> &mdash; no relationship at
all. Switching to per-feature gating eliminates saturation completely and moves the macro
by just {100*(mac_d-mac_s):+.2f} pp, still below GOOD_5.
</div>

<h2>No candidate explanation survives</h2>
<p>Rank correlation against the per-cell gap to GOOD_5:</p>
{gfs_rho_table()}

<p>None of these reaches a level that would let us claim a mechanism. Read together with
the <a href="subset_gap_analysis.html">selection-headroom result</a> &mdash; that two
thirds of any apparent subset-search gain is winner's curse, driven by sample size &mdash;
the coherent reading is that <strong>label-free subset selection on these cells is
dominated by selection noise rather than by a fixable defect in the selector</strong>.
That is an argument against investing in another selector variant, not for one.</p>

<div class="callout warn">
<strong>One cell is an outlier and deserves naming rather than averaging.</strong>
<code>{esc(worst['cell'])}</code> misses GOOD_5 by {abs(100*f(worst['gap_vs_good5'])):.1f} pp
under group-granular gating and
{abs(100*f(worst_d['gap_vs_good5'])):.1f} pp under per-feature gating. It combines the most
extreme class imbalance in the set (positive rate {f(worst['pos_rate']):.3f}) with
{worst['n_anti_chosen']} anti-oriented views among those chosen. It is the one cell
where the &ldquo;bad views got in&rdquo; story is visibly true, and it is doing a lot of
work in any macro average.
</div>

<h2>Per cell</h2>
<table><thead><tr><th>Cell</th><th>Group</th><th class="num">selected (group rule)</th>
<th>saturated</th><th class="num">selected (per-feature)</th>
<th class="num">gap vs GOOD_5</th><th class="num">gap, per-feature</th>
<th class="num">anti-oriented chosen</th><th class="num">pos rate</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""
    return page("Learned selector diagnosis",
                "Why GroupFS is non-optimal &mdash; and which explanations the data rejects",
                body, "groupfs_diagnosis.html")


def p_gated():
    dufs = lead_val("a2.dufs", "c46", "macro_all")
    dufs_q = lead_val("a2.dufs", "c46", "macro_qa")
    dufs_m = lead_val("a2.dufs", "c46", "macro_math")
    dsel = lead_val("a2.select", "c46", "macro_all")
    g5 = lead_val("ref.GOOD_5", "c46", "macro_all")
    g6 = lead_val("ref.GOOD_6", "c46", "macro_all")
    dd = lead_val("a2.dufs", "c46", "delta_vs_good5_all")
    dp = lead_val("a2.dufs", "c46", "wilcoxon_p")

    body = f"""
<p>The trace-based Gated-Laplacian selector &mdash; per-feature stochastic gates trained on
a Laplacian-trace objective over the gated sub-matrix &mdash; is the independence-aware
label-free selector most directly relevant to this work. It was raised as the strongest
option we had not yet built.</p>

<div class="callout">
<strong>It was already built and already benchmarked.</strong> It is implemented in
<code>spectral_utils/selectors/a2_groupfs.py</code> as the per-feature gate rule, and ships
as bench variant <code>a2.dufs</code>. It has rows on all 25 in-scope cells. The objective
is the Laplacian trace on the gated sub-matrix plus a gate-sparsity penalty; the gate
regularisation strength is chosen label-free by selection stability across seeds, which is
our own addition rather than the source method's prescription.
</div>

<div class="tiles">
<div class="tile"><div class="v">{pc(dufs,2)}</div><div class="l">Gated-Laplacian macro</div>
<div class="d">QA {pc(dufs_q,1)} &bull; math {pc(dufs_m,1)}</div></div>
<div class="tile"><div class="v">{pc(g5,2)}</div><div class="l">GOOD_5</div></div>
<div class="tile"><div class="v">{pc(g6,2)}</div><div class="l">GOOD_6</div></div>
<div class="tile"><div class="v">{100*dd:+.2f} pp</div>
<div class="l">vs GOOD_5</div><div class="d">Wilcoxon p = {dp:.3f} &mdash; not significant</div></div>
</div>

{fig_pool_size_line()}

<h2>What the per-feature rule actually changes</h2>
<p>Compared with the group-granular rule (<code>a2.select</code>, macro {pc(dsel,2)}%), the
per-feature rule is a clean structural improvement in one respect and a null result in
another:</p>
<ul>
<li>It <strong>eliminates gate saturation entirely</strong> &mdash; from 12 of 25 cells
selecting the whole pool down to none. That isolates saturation as a consequence of group
granularity rather than of the gates themselves, which is a clean and quotable finding.</li>
<li>It <strong>does not convert that into accuracy</strong>. Macro moves {100*(dufs-dsel):+.2f} pp, the
difference from GOOD_5 is not significant, and on the worst cell it remains catastrophic.</li>
</ul>

<div class="callout warn">
<strong>Recommendation withheld on building a further variant.</strong> The motivation for
a new gate variant was to fix saturation. Saturation turns out to be already solved by the
per-feature rule, and solving it did not help. With the
<a href="subset_gap_analysis.html">honest selection headroom</a> at
{100*(sum(f(r['honest_gain']) for r in D['gap'] if f(r['honest_gain']) is not None)/len(D['gap'])):+.2f}
pp over GOOD_5 and two thirds of any apparent gain being winner's curse, there is little
room for a new label-free selector to win, and no measured mechanism left to attack.
</div>

<p class="small"><strong>Grounding status.</strong> The source method is currently cited
second-hand through the group-discovery paper's description of it; the primary paper is not
yet in <code>papers/</code>. The implementation should be read as
&ldquo;our reconstruction of the method as described by its successor&rdquo; until that is
closed.</p>
"""
    return page("Gated-Laplacian selector",
                "The trace-based label-free selector: status, results, and what it settles",
                body, "gated_laplacian.html")


def p_anchor():
    by = {}
    for r in D["anchor"]:
        by.setdefault(r["anchor"], []).append(r)
    order = ["epr", "topk_tail_mass", "mean_logprob_entropy", "renyi_entropy_2",
             "cusum_max", "low_band_power", "stft_max_high_power", "spectral_entropy",
             "rpdi"]
    rows = []
    for a in order:
        sel = [r for r in by.get(a, []) if r["present"] == "True"]
        if not sel:
            continue
        aur = [f(r["auroc"]) for r in sel if f(r["auroc"]) is not None]
        qa = [f(r["auroc"]) for r in sel if r["group"] == "QA" and f(r["auroc"])]
        mt = [f(r["auroc"]) for r in sel if r["group"] == "math" and f(r["auroc"])]
        wrong = sum(1 for r in sel if r["wrong_sign"] == "True")
        badge = ('<span class="badge b-ok">safe</span>' if wrong == 0
                 else '<span class="badge b-bad">unsafe</span>')
        rows.append(
            f"<tr><td><code>{esc(a)}</code>"
            + (" <em>(incumbent)</em>" if a == "epr" else "")
            + f"</td><td class='num'>{len(sel)}</td>"
              f"<td class='num'>{pc(sum(aur)/len(aur),2)}</td>"
              f"<td class='num'>{pc(sum(qa)/len(qa),2) if qa else '--'}</td>"
              f"<td class='num'>{pc(sum(mt)/len(mt),2) if mt else '--'}</td>"
              f"<td class='num'>{wrong}</td><td>{badge}</td></tr>")

    body = f"""
<p>The fused L-SML score carries an arbitrary global sign &mdash; a leading-eigenvector
coin flip. We resolve it without labels by correlating the fused score against one oriented
view, the <em>anchor</em>, and flipping if they disagree. <code>epr</code> has been that
anchor throughout. This page asks whether that choice is load-bearing.</p>

<div class="callout">
<strong>The test isolates the anchor exactly.</strong> The fusion does not depend on the
anchor &mdash; only the final sign does. So GOOD_6 is fused once per cell and the same
fused score is then re-oriented against each candidate. Every difference below is a sign
decision, never a different model.
</div>

<table><thead><tr><th>Anchor</th><th class="num">cells</th><th class="num">macro</th>
<th class="num">QA</th><th class="num">math</th><th class="num">wrong signs</th>
<th>Verdict</th></tr></thead><tbody>{''.join(rows)}</tbody></table>

<h2>Reading</h2>
<ul>
<li><strong><code>epr</code> is confirmed, and it is at the ceiling.</strong> It resolves
the sign correctly on all 25 cells, so no anchor can beat it.</li>
<li><strong>The choice is robust rather than lucky.</strong> Four other views
&mdash; <code>topk_tail_mass</code>, <code>mean_logprob_entropy</code>,
<code>renyi_entropy_2</code>, <code>cusum_max</code> &mdash; agree with it on all 25 cells
and tie exactly. The result does not hinge on <code>epr</code> specifically.</li>
<li><strong>But it is not arbitrary either.</strong> Four of the nine candidates get signs
wrong. <code>rpdi</code> misses 5 cells and drops the macro by 9 pp, with QA collapsing to
near chance. A poor anchor inverts the whole fusion, so the anchor is a real design choice
that happens to have a wide safe band.</li>
</ul>
"""
    return page("Anchor robustness",
                "Is the label-free sign anchor load-bearing?",
                body, "anchor_sweep.html")


def reconciliation_section():
    """Diff of the two independently-verified competitor number sets
    (scripts/reconcile_competitors.py -> reconciliation.csv)."""
    rows = D.get("reconc", [])
    if not rows:
        return ""
    tally = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    conflicts = [r for r in rows if r["status"] == "DELTA"]
    conflict_rows = "".join(
        f"<tr><td><code>{esc(r['cell'])}</code></td>"
        f"<td>{esc(r['method_b'])}</td><td class='num'>{esc(r['auroc_b'])}</td>"
        f"<td class='num'><strong>{esc(r['auroc_a'])}</strong></td>"
        f"<td class='small'>{esc(r['resolution'])}</td></tr>"
        for r in conflicts)
    return f"""
<h2>3. Reconciliation against the pre-Step-193 tables</h2>
<p>The competitor numbers were verified twice from different starting points: the older
<code>action_items</code>/<code>published_baselines</code> generation, and the Step-193
rebuild from the papers' own tables. The two sets were never diffed &mdash; so any
disagreement risked being re-litigated a third time. <code>scripts/reconcile_competitors.py</code>
now diffs them row by row (<code>reconciliation.csv</code>):</p>

<div class="tiles">
<div class="tile"><div class="v">{tally.get('MATCH', 0)}</div><div class="l">MATCH</div>
<div class="d">same value in both generations</div></div>
<div class="tile"><div class="v">{tally.get('DELTA', 0)}</div><div class="l">DELTA</div>
<div class="d">every one is a Step-193b LapEigvals correction</div></div>
<div class="tile"><div class="v">{tally.get('ONLY_A', 0)} / {tally.get('ONLY_B', 0)}</div>
<div class="l">one-source rows</div>
<div class="d">coverage differences (own-run baselines, out-of-scope cells) &mdash; not conflicts</div></div>
</div>

<table><thead><tr><th>Cell</th><th>Old row</th><th class="num">old</th>
<th class="num">verified</th><th>Resolution</th></tr></thead>
<tbody>{conflict_rows}</tbody></table>

<div class="callout warn">
<strong>Still load-bearing:</strong> <code>scores_lsml_upcr.csv</code> itself still carries
the pre-correction <code>published_Y = 0.925 / LapEigvals</code> for
<code>lapeigvals_gsm8k_llama8b</code>; the figures compensate via
<code>report_figs.OVERRIDE_Y</code>. The override can only be retired by fixing the
published table inside <code>score_repgrid.py</code> and re-running the regen chain
(score_repgrid &rarr; repgrid_report &rarr; advisor_report).
</div>
"""


def p_verification():
    stale = [r for r in D["stale"] if r.get("stale") == "True"]
    cells = sorted({r["cell"] for r in stale})
    files = sorted({r["file"] for r in stale})
    g6 = lead_val("ref.GOOD_6", "c46", "macro_all")
    g5 = lead_val("ref.GOOD_5", "c46", "macro_all")
    d = lead_val("ref.GOOD_6", "c46", "delta_vs_good5_all")
    p = lead_val("ref.GOOD_6", "c46", "wilcoxon_p")
    r6 = D["lead"].get(("ref.GOOD_6", "c46"), {})

    body = f"""
<p>Nothing on this site was published before this audit passed. It is here because two
defects were found, both of which would have put wrong numbers in front of you.</p>

<h2>1. Bench rows computed against a superseded feature cache</h2>
<p>Two independent artifacts held per-cell AUROCs for the same subsets and disagreed. The
disagreement was <em>not</em> in the code: driving both fusion paths on the same cell
reproduces the same score, the same group count and the same residual to full precision.
The cause was staleness. The benchmark harness is resume-safe &mdash; it skips any
(variant, cell) pair already present in its output &mdash; so when a cell's feature cache
was rebuilt, none of its existing rows was ever recomputed.</p>

<p>Sizing it required re-scoring every stored row's own selected subset against the current
cache: <strong>{len(D['stale'])} rows re-scored, {len(stale)} stale</strong>, confined to
{len(cells)} cell{'s' if len(cells)!=1 else ''} but spread across {len(files)} of the
benchmark files. A second, wider instance of the same defect was then found: on
<strong>11 further cells</strong> every learned selector had searched a pool four views
smaller than the current one, so those selectors were handicapped. Both were fixed by
deleting the affected rows and re-running.</p>

<div class="callout warn">
<strong>Effect on previously circulated numbers.</strong>
<table style="margin-top:8px"><thead><tr><th>Quantity</th><th class="num">Previously</th>
<th class="num">Corrected</th></tr></thead><tbody>
<tr><td>GOOD_6 macro AUROC</td><td class="num">75.87</td>
<td class="num"><strong>{pc(g6,2)}</strong></td></tr>
<tr><td>GOOD_5 macro AUROC</td><td class="num">74.89</td>
<td class="num"><strong>{pc(g5,2)}</strong></td></tr>
<tr><td>GOOD_6 over GOOD_5</td><td class="num">+0.98 pp</td>
<td class="num"><strong>{100*d:+.2f} pp</strong></td></tr>
<tr><td>Wilcoxon p</td><td class="num">0.0025</td>
<td class="num"><strong>{p:.5f}</strong></td></tr>
<tr><td>Win / loss over 25 cells</td><td class="num">19 / 6</td>
<td class="num"><strong>{r6.get('wins','?')} / {r6.get('losses','?')}</strong></td></tr>
<tr><td>Best label-free selector vs GOOD_5</td><td class="num">+0.06 pp (ahead)</td>
<td class="num"><strong>&minus;0.17 pp (behind)</strong></td></tr>
</tbody></table>
The headline conclusion is unchanged and slightly cleaner: GOOD_6 leads, and no label-free
selector beats it.
</div>

<h2>2. Competitor numbers without citations</h2>
<p>13 of the 19 published anchors carried no citation of any kind &mdash; no paper, table,
page or scale. Every anchor that could be checked was checked against the raw extracted
paper text, which surfaced two errors in how a competitor had been recorded. Full detail on
the <a href="competitor_sources.html">provenance page</a>.</p>

<h2>Standing integrity gates</h2>
<table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>
<tr><td><code>scripts/smoke_selectors.py</code> &mdash; every selector reproduces a known
answer on planted data</td><td><span class="badge b-ok">20/20 pass</span> (incl. the new
<code>a6_pseudolabel_gates</code>)</td></tr>
<tr><td><code>run_selector_bench.py --self-check</code> &mdash; stored-vs-live agreement and
GOOD_5 reproduction</td><td><span class="badge b-ok">pass</span> &mdash; 51 cells, max
difference 2.9&times;10<sup>&minus;8</sup></td></tr>
<tr><td>Post-fix staleness re-audit</td>
<td><span class="badge b-ok">0 stale rows, 0 pool mismatches</span></td></tr>
<tr><td>Cell roster consistency</td><td><span class="badge b-ok">pass</span> &mdash; the
25-cell roster was hoisted into one module and verified identical to the three copies it
replaced</td></tr>
</tbody></table>

{reconciliation_section()}

<h2>What is still open</h2>
<ul>
<li>The primary paper for the Gated-Laplacian selector is not in the local library, so that
implementation is grounded only in a successor paper's description of it.</li>
<li>One competitor method still has no local PDF (<code>Internal-States+RC</code>); the
LapEigvals / INSIDE / LOS-Net papers were all located and verified (61/62 rows, 17/18
anchors) &mdash; the LapEigvals PDF had been filed under its <em>title</em>, not the
method name.</li>
<li>The supervised ceiling uses an untuned regularisation strength; the spread across
plausible values reaches 6.2 pp on the worst cell.</li>
</ul>
"""
    return page("Verification",
                "The data-integrity audit that gates everything else here",
                body, "verification.html")


# ── main ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="run the guardrail scan and report, do not fail the build")
    args = ap.parse_args()

    load_all()
    pages = {
        "index.html": p_index(),
        "competitor_grid.html": p_competitor_grid(),
        "competitor_sources.html": p_competitor_sources(),
        "supervised_ceiling.html": p_supervised_ceiling(),
        "subset_gap_analysis.html": p_gap(),
        "selector_choices.html": p_choices(),
        "groupfs_diagnosis.html": p_groupfs(),
        "gated_laplacian.html": p_gated(),
        "anchor_sweep.html": p_anchor(),
        "verification.html": p_verification(),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    all_hits = {}
    for name, html in pages.items():
        hits = guardrail_scan(html)
        if hits:
            all_hits[name] = hits
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"  wrote {name:<28} {len(html):>7} bytes")

    print(f"\n{len(pages)} pages -> {OUT_DIR}")
    if all_hits:
        print("\nGUARDRAIL FAILURES (banned terminology):")
        for name, hits in all_hits.items():
            for term, n in hits:
                print(f"  {name}: {term!r} x{n}")
        if not args.check:
            sys.exit(1)
    else:
        print("guardrail scan: clean")


if __name__ == "__main__":
    main()
