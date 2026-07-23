#!/usr/bin/env python
"""
selector_vs_competitor.py — the SELECTOR against the published competitors (Step 193c).

The thesis contribution is the *pipeline*: label-free feature selection, then L-SML fusion.
The Step-193 competitor grid led with the fixed hand-curated subsets (GOOD_5/GOOD_6) and
buried the selector, which answers the wrong question. This script puts the selector first.

Two tallies are reported and they differ a lot, so read the right one:
  * vs ALL anchors        — mixes supervised competitors (INSIDE, TSV, LOS-Net, HCPD,
                            LapEigvals) with unsupervised ones. Not like-for-like.
  * vs UNSUPERVISED only  — the fair comparison for a label-free detector.

`best selector on that cell` is a CEILING, not a method: it picks the best variant per cell
with hindsight. It is shown to bound what any per-cell selector could achieve here.

Usage:
    python scripts/selector_vs_competitor.py
Writes: results/advisor_inscope/selector_vs_competitor.csv
"""
import os, sys, csv, glob
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from inscope_cells import INSCOPE, GROUP

BENCH = os.path.join(REPO, "results", "selector_bench")
AI = os.path.join(REPO, "results", "advisor_inscope")


def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig"))) if os.path.exists(p) else []


def f(v):
    try:
        x = float(v); return x if x == x else None
    except (TypeError, ValueError):
        return None


# fixed subsets
mac = {}
for r in read(os.path.join(BENCH, "reference_macros__c46.csv")):
    if r["cell"] in INSCOPE:
        mac.setdefault(r["cell"], {})[r["variant"]] = f(r["auroc"])

# every learned-selector variant, c46 pool
sel = {}
for p in sorted(glob.glob(os.path.join(BENCH, "*__c46.csv"))):
    if "reference_macros" in p:
        continue
    for r in read(p):
        c, v, a = r.get("cell"), r.get("variant"), f(r.get("auroc"))
        if c in INSCOPE and a is not None:
            sel.setdefault(c, {})[v] = a

# competitor anchors (unsupervised only = like-for-like)
comp = {}
for r in read(os.path.join(AI, "competitors_verified.csv")):
    if r["role"] == "anchor" and r["cell"] in INSCOPE:
        comp[r["cell"]] = (f(r["auroc"]), r["method"], r["supervision"], r["verified_by"])

KEY = "a2.dufs"          # the trace-based Gated-Laplacian: best label-free selector
ALT = "a2.select"        # GroupFS group-granular

print(f"{'cell':<30}{'grp':>5}{'dufs':>8}{'GrpFS':>8}{'bestSel':>8}{'GOOD_6':>8}"
      f"{'comp':>8}  {'d(dufs-comp)':>12}  method")
print("-" * 118)
rows = []
for c in INSCOPE:
    s = sel.get(c, {})
    d = s.get(KEY)
    g = s.get(ALT)
    best_v, best_a = (None, None)
    if s:
        best_v = max(s, key=lambda k: s[k]); best_a = s[best_v]
    g6 = mac.get(c, {}).get("ref.GOOD_6")
    cm = comp.get(c)
    cv = cm[0] if cm else None
    delta = (d - cv) if (d is not None and cv is not None) else None
    def verdict(v, ref, tol=1e-9):
        if v is None or ref is None:
            return ""
        return "WIN" if v > ref + tol else ("LOSS" if v < ref - tol else "TIE")

    rows.append(dict(cell=c, group=GROUP[c], dufs=d, groupfs=g, best_sel=best_a,
                     best_sel_variant=best_v, good6=g6, comp=cv,
                     comp_method=cm[1] if cm else "", comp_sup=cm[2] if cm else "",
                     delta_dufs_comp=delta,
                     verdict_selector_vs_comp=verdict(d, cv),
                     verdict_good6_vs_comp=verdict(g6, cv),
                     verdict_selector_vs_good6=verdict(d, g6)))
    fmt = lambda v: f"{100*v:>8.1f}" if v is not None else f"{'--':>8}"
    print(f"{c:<30}{GROUP[c]:>5}{fmt(d)}{fmt(g)}{fmt(best_a)}{fmt(g6)}{fmt(cv)}"
          f"{(f'{100*delta:>+12.1f}' if delta is not None else f'{chr(45)*2:>12}')}"
          f"  {(cm[1][:34] if cm else '(none)')}")

print("-" * 118)
have = [r for r in rows if r["comp"] is not None and r["dufs"] is not None]
uns = [r for r in have if r["comp_sup"] == "unsupervised"]


def tally(rs, key):
    w = sum(1 for r in rs if r[key] > r["comp"] + 1e-9)
    l = sum(1 for r in rs if r[key] < r["comp"] - 1e-9)
    return w, l


print(f"\ncells with a published anchor: {len(have)}")
for label, rs in (("ALL anchors", have), ("UNSUPERVISED anchors only", uns)):
    print(f"\n-- {label} (n={len(rs)}) --")
    for name, key in (("a2.dufs (trace-based selector)", "dufs"),
                      ("a2.select (GroupFS)", "groupfs"),
                      ("best selector on that cell", "best_sel"),
                      ("GOOD_6 (fixed)", "good6")):
        rs2 = [r for r in rs if r[key] is not None]
        w, l = tally(rs2, key)
        m = np.mean([r[key] for r in rs2])
        mc = np.mean([r["comp"] for r in rs2])
        print(f"   {name:<34} macro {m:.4f} vs comp {mc:.4f}  "
              f"({100*(m-mc):+.2f} pp)   {w}W / {l}L")

out = os.path.join(AI, "selector_vs_competitor.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nwrote {out}")
