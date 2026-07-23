#!/usr/bin/env python
"""
selector_choice_analysis.py — WHAT does the selector pick, and why does that lose to GOOD_6?

The open question after Step 193: on the cells where the label-free selector
(`a2.dufs`, the trace-based Gated-Laplacian) scores below the fixed GOOD_6 subset, what did
it actually choose, how does that differ from GOOD_6, and what drives the difference?

Three things are measured per cell:

  1. COMPOSITION — which GOOD_6 members the selector dropped, and what it added instead.
  2. QUALITY OF THE PICK — mean per-feature oriented AUROC of the views it selected vs the
     views it left out. If selection were informative, selected >> unselected.
  3. WHY — Spearman rho between each view's GATE VALUE (what the objective maximises) and
     that view's own oriented AUROC (what we actually want). The gates are trained on a
     Laplacian-smoothness objective that never sees a label, so this correlation is the
     direct test of whether the objective is even aiming at the right target.

Per-feature oriented AUROC comes from the Step-192 orientation audit
(results/selector_bench/inscope_feature_orientation.csv).

Usage:
    python scripts/selector_choice_analysis.py
Writes: results/advisor_inscope/selector_choice_analysis.csv
"""
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inscope_cells import INSCOPE, GROUP
from subset_gap_analysis import read, fnum, spearman

BENCH = os.path.join(REPO, "results", "selector_bench")
AI = os.path.join(REPO, "results", "advisor_inscope")
OUT = os.path.join(AI, "selector_choice_analysis.csv")

GOOD_6 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy",
          "varentropy"]
VARIANT = "a2.dufs"


def main():
    # per-feature oriented AUROC, per cell
    fauc = {}
    for r in read(os.path.join(BENCH, "inscope_feature_orientation.csv")):
        a = fnum(r.get("oriented_auroc"))
        if r.get("cell") in INSCOPE and a is not None:
            fauc.setdefault(r["cell"], {})[r["feature"]] = a

    # fixed-subset scores
    good6 = {}
    for r in read(os.path.join(BENCH, "reference_macros__c46.csv")):
        if r.get("cell") in INSCOPE and r.get("variant") == "ref.GOOD_6":
            good6[r["cell"]] = fnum(r.get("auroc"))

    rows = []
    for r in read(os.path.join(BENCH, "a2_groupfs__c46.csv")):
        c = r.get("cell")
        if c not in INSCOPE or r.get("variant") != VARIANT:
            continue
        chosen = [x for x in (r.get("chosen") or "").split("|") if x]
        auroc = fnum(r.get("auroc"))
        g6 = good6.get(c)
        af = fauc.get(c, {})
        try:
            diag = json.loads(r.get("diag_json") or "{}")
        except Exception:
            diag = {}
        gates = diag.get("feat_gate_means")

        missed = [x for x in GOOD_6 if x in af and x not in chosen]
        extra = [x for x in chosen if x not in GOOD_6]
        kept = [x for x in GOOD_6 if x in chosen]

        sel_auc = [af[x] for x in chosen if x in af]
        pool = sorted(af)
        unsel = [af[x] for x in pool if x not in chosen]
        # "anti-oriented" means AUROC < 0.5, i.e. INFORMATIVE BUT INVERTED — L-SML flips it
        # with a negative weight. That is NOT the same as uninformative. Split the two, or
        # the count reads as "bad features" when most of them are strong once flipped.
        anti_extra = sum(1 for x in extra if af.get(x, 1.0) < 0.5)
        nearrandom_extra = sum(1 for x in extra if 0.45 <= af.get(x, 1.0) <= 0.55)
        strong_inverted = sum(1 for x in extra if af.get(x, 1.0) < 0.40)

        # gate value vs that view's own oriented AUROC (pool order == prepare_cell order,
        # which is CANONICAL_POOL order; align by taking the pool the audit saw)
        rho_gate = float("nan")
        if gates and len(gates) == len(pool):
            rho_gate, _ = spearman(list(gates), [af[x] for x in pool])

        rows.append(dict(
            cell=c, group=GROUP.get(c), auroc=auroc, good6=g6,
            gap_vs_good6=(auroc - g6) if (auroc is not None and g6 is not None) else None,
            verdict=("WIN" if (auroc is not None and g6 is not None and auroc > g6)
                     else "LOSS"),
            size=len(chosen), p_pool=len(pool),
            good6_kept=len(kept), good6_missed="|".join(missed),
            n_extra=len(extra), n_extra_anti=anti_extra,
            n_extra_nearrandom=nearrandom_extra, n_extra_strong_inverted=strong_inverted,
            mean_auc_selected=(np.mean(sel_auc) if sel_auc else None),
            mean_auc_unselected=(np.mean(unsel) if unsel else None),
            sel_minus_unsel=((np.mean(sel_auc) - np.mean(unsel))
                             if (sel_auc and unsel) else None),
            rho_gate_vs_featauroc=rho_gate,
            chosen="|".join(chosen),
        ))

    rows.sort(key=lambda d: (d["gap_vs_good6"] is None, d["gap_vs_good6"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    losses = [d for d in rows if d["verdict"] == "LOSS"]
    wins = [d for d in rows if d["verdict"] == "WIN"]

    print(f"{VARIANT} vs GOOD_6 over {len(rows)} cells: "
          f"{len(wins)} WIN / {len(losses)} LOSS\n")
    print(f"{'cell':<32}{'gap':>8}{'size':>6}{'G6 kept':>8}{'extra':>7}"
          f"{'anti':>6}{'selAUC':>8}{'unselAUC':>9}{'diff':>7}  GOOD_6 members dropped")
    print("-" * 122)
    for d in rows:
        g = d["gap_vs_good6"]
        print(f"{d['cell']:<32}{100*g:>+8.2f}{d['size']:>6}"
              f"{d['good6_kept']:>6}/6{d['n_extra']:>7}{d['n_extra_anti']:>6}"
              f"{100*d['mean_auc_selected']:>8.1f}{100*d['mean_auc_unselected']:>9.1f}"
              f"{100*d['sel_minus_unsel']:>+7.1f}  {d['good6_missed'] or '-'}")
    print("-" * 122)

    def agg(rs, k):
        v = [d[k] for d in rs if d[k] is not None and np.isfinite(d[k])]
        return np.mean(v) if v else float("nan")

    print(f"\n{'':<26}{'LOSS cells':>12}{'WIN cells':>12}")
    for label, k in (("mean gap vs GOOD_6 (pp)", "gap_vs_good6"),
                     ("GOOD_6 members kept", "good6_kept"),
                     ("extra views added", "n_extra"),
                     ("inverted-polarity extras", "n_extra_anti"),
                     ("  of which strongly informative", "n_extra_strong_inverted"),
                     ("  truly near-random extras", "n_extra_nearrandom"),
                     ("selected mean AUROC", "mean_auc_selected"),
                     ("unselected mean AUROC", "mean_auc_unselected"),
                     ("selected - unselected", "sel_minus_unsel")):
        a, b = agg(losses, k), agg(wins, k)
        sc = 100 if k not in ("good6_kept", "n_extra", "n_extra_anti",
                              "n_extra_nearrandom", "n_extra_strong_inverted") else 1
        print(f"{label:<26}{sc*a:>12.2f}{sc*b:>12.2f}")

    r_all = [d["rho_gate_vs_featauroc"] for d in rows
             if d["rho_gate_vs_featauroc"] is not None
             and np.isfinite(d["rho_gate_vs_featauroc"])]
    print(f"\nrho(gate value, that view's own oriented AUROC), per cell:")
    print(f"   mean {np.mean(r_all):+.3f}   median {np.median(r_all):+.3f}   "
          f"range [{np.min(r_all):+.3f}, {np.max(r_all):+.3f}]   cells={len(r_all)}")
    print("   -> the gates are trained on a label-free smoothness objective; this is how")
    print("      strongly what they maximise lines up with what actually separates.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
