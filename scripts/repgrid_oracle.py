#!/usr/bin/env python
"""
repgrid_oracle.py — LR oracle (supervised headroom) on the replication-grid cells.

Phase-2 item 2 of HANDOFF_punchlist_and_reruns.md. Re-runs the Step-142/147 question
("are features the bottleneck, not fusion?" — supervised LR beats unsupervised L-SML by
+3.6..4.7pp on the original 32-cell battery) on the 19 newer, more domain-diverse repgrid
cells the oracle has never seen (short-trace QA especially).

Reuses the canonical, correction-compliant machinery from logistic_oracle.py:
  build_X                 — feature matrix (median-imputed, saturation filter),
  lr_oracle_auc_variants  — 5-fold CV, per-fold AUROC averaged, class_weight='balanced'
                            (SUPERVISED_ORACLE_CORRECTION.md), NOW with grouped folds,
  FEATURE_SETS            — GOOD_5 / STABLE_H9 / ALL_H16.

Leakage fix: cells with k>1 (se_squad_v2, truthfulqa, se_nq_open, semenergy: K=10) pass
problem_id as `groups` so StratifiedGroupKFold keeps a question's candidates in one fold.

CONT (unsupervised L-SML) column = the GOOD_5/STABLE_H9/ALL_H16 lsml AUROCs from
results/repgrid/scores_lsml_upcr.csv (carried in local_cache/repgrid_cont.pkl by the
featcache builder) — the same recipe that produced the canonical numbers.

Inputs (from scripts/build_repgrid_featcache.py): local_cache/repgrid_cells.pkl + repgrid_cont.pkl
Outputs: results/repgrid/oracle_repgrid.csv + .pkl

Usage:
    python scripts/repgrid_oracle.py
"""
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from logistic_oracle import (  # scripts/ is on sys.path via REPO/scripts import below
    build_X, lr_oracle_auc_variants, FEATURE_SETS,
)

CELLS_PKL = os.path.join(REPO, "local_cache", "repgrid_cells.pkl")
CONT_PKL = os.path.join(REPO, "local_cache", "repgrid_cont.pkl")
OUT_CSV = os.path.join(REPO, "results", "repgrid", "oracle_repgrid.csv")
OUT_PKL = os.path.join(REPO, "results", "repgrid", "oracle_repgrid.pkl")


def main():
    with open(CELLS_PKL, "rb") as f:
        cells = pickle.load(f)
    cont = {}
    if os.path.exists(CONT_PKL):
        with open(CONT_PKL, "rb") as f:
            for r in pickle.load(f):
                cont[r["cell"].split("/", 1)[-1]] = r   # keyed by preset_id

    results = []
    print(f"{'cell':<32} {'k':>2} {'n':>5} {'acc':>5} | "
          + " | ".join(f"{'CONT-'+s:>8} {'LR-'+s:>7} {'Δ-'+s:>6}" for s in ("5", "9", "16")))
    print("-" * 120)
    accum = {s: {"cont": [], "lr": []} for s in ("5", "9", "16")}

    for pid in sorted(cells):
        cell = cells[pid]
        y = np.asarray(cell["labels"], dtype=int)
        n = len(y)
        k = cell.get("k", 1)
        groups = np.asarray(cell["problem_id"]) if k > 1 else None
        row = {"cell": pid, "k": k, "n": n, "acc": float(y.mean()),
               "dataset": cell.get("dataset"), "model": cell.get("model")}
        cells_parts = []
        for s, feats in FEATURE_SETS.items():
            X, avail = build_X(cell["feats"], feats, n)
            c = (cont.get(pid) or {}).get(f"cont_{s}")
            row[f"cont_{s}"] = c
            row[f"n_avail_{s}"] = len(avail)
            lr = None
            if X is not None and y.sum() >= 5 and (n - y.sum()) >= 5:
                try:
                    v = lr_oracle_auc_variants(X, y, groups=groups)
                    lr = v["bal_cv"][0]
                    row[f"lr_{s}"] = lr
                    row[f"lr_ci_{s}"] = [v["bal_cv"][1], v["bal_cv"][2]]
                    row[f"lr_in_{s}"] = v["bal_in"]
                except Exception as e:
                    row[f"lr_{s}"] = None
                    print(f"  [{pid} feat{s}] LR failed: {type(e).__name__}: {e}")
            d = (lr - c) if (lr is not None and c is not None) else None
            row[f"delta_{s}"] = d
            if c is not None and lr is not None:
                accum[s]["cont"].append(c)
                accum[s]["lr"].append(lr)
            cs = f"{100*c:.1f}" if c is not None else "  N/A"
            ls = f"{100*lr:.1f}" if lr is not None else "  N/A"
            ds = f"{100*d:+.1f}" if d is not None else " N/A"
            cells_parts.append(f"{cs:>8} {ls:>7} {ds:>6}")
        results.append(row)
        gtag = "G" if groups is not None else " "
        print(f"{pid:<32} {k:>2}{gtag} {n:>5} {y.mean():>5.2f} | " + " | ".join(cells_parts))

    print("-" * 120)
    macro_parts = []
    for s in ("5", "9", "16"):
        mc = np.mean(accum[s]["cont"]) if accum[s]["cont"] else None
        ml = np.mean(accum[s]["lr"]) if accum[s]["lr"] else None
        md = (ml - mc) if (mc is not None and ml is not None) else None
        macro_parts.append(f"{100*mc:>8.1f} {100*ml:>7.1f} {100*md:>+6.1f}"
                           if md is not None else f"{'N/A':>8} {'N/A':>7} {'N/A':>6}")
    print(f"{'MACRO (common cells)':<32} {'':>2} {'':>5} {'':>5} | " + " | ".join(macro_parts))
    print(f"\n(Δ = supervised LR balanced-CV headroom over unsupervised L-SML CONT; "
          f"'G' = grouped folds by problem_id)")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)
    import csv as _csv
    keys = sorted({k for r in results for k in r})
    with open(OUT_CSV, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"wrote {len(results)} cells -> {OUT_CSV}")


if __name__ == "__main__":
    main()
