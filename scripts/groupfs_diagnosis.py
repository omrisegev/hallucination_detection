#!/usr/bin/env python
"""
groupfs_diagnosis.py — why is the learned selector non-optimal? (Step 193, Phase 3.3)

GroupFS (arXiv:2511.09166) ships in this repo as two selection rules over the SAME gates
and the SAME Laplacian-trace objective (spectral_utils/selectors/a2_groupfs.py):

    a2.select  — GROUP granular: a group is open iff its median member gate is open, then
                 take the union of open groups. Amplifies: one open member can carry a
                 whole group in.
    a2.dufs    — PER-FEATURE: threshold each gate independently. This is the Gated-
                 Laplacian / DUFS rule (Lindenbaum et al., arXiv:2007.04728), i.e. the
                 predecessor GroupFS builds on.

The standing explanation for GroupFS's worst misses was "gate saturation" — the selector
opens ~100% of the pool, so L-SML's own clustering is swamped. This script tests that
claim rather than assuming it, and regresses the per-cell gap vs GOOD_5 on the three
candidate mechanisms (saturation, anti-oriented content, class imbalance).

IMPORTANT: run this only on bench rows whose p_pool matches the current cache. The
Step-193 audit found 11 cells whose stored rows predated a pool enlargement; the
saturation counts computed from those rows were wrong.

Usage:
    python scripts/groupfs_diagnosis.py
Writes: results/advisor_inscope/groupfs_diagnosis.csv
"""
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inscope_cells import INSCOPE, GROUP
from subset_gap_analysis import read, fnum, spearman

BENCH = os.path.join(REPO, "results", "selector_bench")
A2 = os.path.join(BENCH, "a2_groupfs__c46.csv")
REFMAC = os.path.join(BENCH, "reference_macros__c46.csv")
ORIENT = os.path.join(BENCH, "inscope_feature_orientation.csv")
SCORES = os.path.join(REPO, "results", "repgrid", "scores_lsml_upcr.csv")
OUT = os.path.join(REPO, "results", "advisor_inscope", "groupfs_diagnosis.csv")

VARIANTS = ("a2.select", "a2.dufs")


def main():
    # per-feature orientation, so we can count anti-oriented views INSIDE each selection
    anti = {}
    for r in read(ORIENT):
        a = fnum(r.get("oriented_auroc"))
        if r.get("cell") in INSCOPE and a is not None:
            anti.setdefault(r["cell"], {})[r["feature"]] = a < 0.5

    pos_rate = {}
    for r in read(SCORES):
        if r["cell"] in INSCOPE and r["cell"] not in pos_rate:
            pos_rate[r["cell"]] = fnum(r.get("acc"))

    good5, good6 = {}, {}
    for r in read(REFMAC):
        if r.get("cell") in INSCOPE:
            if r.get("variant") == "ref.GOOD_5":
                good5[r["cell"]] = fnum(r.get("auroc"))
            elif r.get("variant") == "ref.GOOD_6":
                good6[r["cell"]] = fnum(r.get("auroc"))

    rows = []
    for r in read(A2):
        c = r.get("cell")
        if c not in INSCOPE or r.get("variant") not in VARIANTS:
            continue
        chosen = [x for x in (r.get("chosen") or "").split("|") if x]
        size, pool = fnum(r.get("size")), fnum(r.get("p_pool"))
        auroc = fnum(r.get("auroc"))
        try:
            diag = json.loads(r.get("diag_json") or "{}")
        except Exception:
            diag = {}
        cell_anti = anti.get(c, {})
        n_anti_chosen = sum(1 for f in chosen if cell_anti.get(f))
        rows.append(dict(
            cell=c, group=GROUP.get(c), variant=r["variant"],
            size=size, p_pool=pool,
            frac_selected=(size / pool) if (size and pool) else None,
            saturated=bool(size and pool and size == pool),
            auroc=auroc,
            gap_vs_good5=(auroc - good5[c]) if (auroc is not None and c in good5) else None,
            gap_vs_good6=(auroc - good6[c]) if (auroc is not None and c in good6) else None,
            n_anti_chosen=n_anti_chosen,
            frac_anti_chosen=(n_anti_chosen / len(chosen)) if chosen else None,
            pos_rate=pos_rate.get(c),
            K_groups=diag.get("K_groups"), stability=diag.get("stability"),
        ))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── saturation ────────────────────────────────────────────────────────────────────
    print("Same gates, same objective, same lambda — only the selection RULE differs.\n")
    print(f"{'variant':<12} {'cells':>6} {'saturated':>10} {'mean size':>10} "
          f"{'mean frac':>10} {'macro':>8} {'worst gap':>10}")
    print("-" * 72)
    for v in VARIANTS:
        sel = [d for d in rows if d["variant"] == v]
        sat = sum(1 for d in sel if d["saturated"])
        gaps = [d["gap_vs_good5"] for d in sel if d["gap_vs_good5"] is not None]
        print(f"{v:<12} {len(sel):>6} {sat:>7}/{len(sel):<2} "
              f"{np.mean([d['size'] for d in sel]):>10.1f} "
              f"{np.mean([d['frac_selected'] for d in sel]):>10.3f} "
              f"{np.mean([d['auroc'] for d in sel]):>8.4f} {min(gaps):>10.4f}")

    # ── does saturation predict the loss? ─────────────────────────────────────────────
    print("\nWhat predicts the per-cell gap vs GOOD_5? (Spearman rho)")
    print(f"{'variant':<12} {'covariate':<20} {'rho':>8} {'n':>4}")
    print("-" * 48)
    for v in VARIANTS:
        sel = [d for d in rows if d["variant"] == v]
        y = [d["gap_vs_good5"] for d in sel]
        for cov in ("frac_selected", "frac_anti_chosen", "n_anti_chosen",
                    "pos_rate", "stability"):
            rho, n = spearman([d[cov] for d in sel], y)
            print(f"{v:<12} {cov:<20} {rho:>8.3f} {n:>4}")
        print()

    # ── the loss cells ────────────────────────────────────────────────────────────────
    sel_rows = {d["cell"]: d for d in rows if d["variant"] == "a2.select"}
    duf_rows = {d["cell"]: d for d in rows if d["variant"] == "a2.dufs"}
    losers = sorted([c for c, d in sel_rows.items()
                     if d["gap_vs_good5"] is not None and d["gap_vs_good5"] < -0.01],
                    key=lambda c: sel_rows[c]["gap_vs_good5"])
    print(f"cells where a2.select loses >1pp to GOOD_5: {len(losers)}")
    print(f"{'cell':<32} {'select gap':>11} {'dufs gap':>10} {'sel size':>9} "
          f"{'dufs size':>10} {'pool':>5} {'anti':>5} {'pos':>6}")
    print("-" * 96)
    repaired = 0
    for c in losers:
        s, d = sel_rows[c], duf_rows.get(c)
        if d and d["gap_vs_good5"] > s["gap_vs_good5"]:
            repaired += 1
        print(f"{c:<32} {s['gap_vs_good5']:>+11.4f} "
              f"{(d['gap_vs_good5'] if d else float('nan')):>+10.4f} "
              f"{int(s['size']):>9}/{int(s['p_pool'])} "
              f"{(int(d['size']) if d else 0):>6}/{int(d['p_pool']) if d else 0} "
              f"{int(s['p_pool']):>5} {s['n_anti_chosen']:>5} {s['pos_rate']:>6.3f}")
    print(f"\nper-feature gating (a2.dufs) repairs {repaired}/{len(losers)} of them")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
