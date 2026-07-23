#!/usr/bin/env python
"""
truncation_confound_audit.py — how much of a cell's AUROC is the max_new cap?

Motivation (2026-07-22): cap_pilot_audit.py showed the R1-distill cells run hard into
their max_new cap (GPQA 95.6% / 96.2% of traces pinned at exactly 2048; math500 63.3% at
2048 and still 51.0% at 4096), and that truncated traces are far less accurate than
completed ones (acc|truncated 0.058 vs acc|complete 0.682 on math500_r1distill8b). A
truncated trace usually has no final answer, so it is graded wrong almost by construction.
That makes "was this trace cut off" a near-label — and every length-sensitive spectral
feature can read it.

This is a confound rather than a signal because the cap is an EXPERIMENTER CHOICE. The
measured AUROC moves when we move max_new, so it is not a property of the detector, and it
breaks comparison against published baselines that used a different cap.

The test: recompute the subset's L-SML AUROC on the complete traces only (trace_length <
cap) and compare to the all-rows number. A cell whose score survives is measuring reasoning
structure; a cell that collapses to chance was measuring the cap.

Read the two caveats before quoting the delta:
  * the complete-only subpopulation has a different prevalence (truncation is correlated
    with the label, so removing truncated rows raises accuracy) and a smaller n. It is a
    different, easier population — not a like-for-like re-scoring.
  * a collapse does NOT prove the features are worthless; it proves the headline number on
    THAT cell is not attributable to spectral structure.

Usage:
    python scripts/truncation_confound_audit.py                       # local_cache/regen
    python scripts/truncation_confound_audit.py --featcache <pkl> --subset GOOD_6
    python scripts/truncation_confound_audit.py --csv results/truncation_confound.csv
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.streaming_utils import FEATURE_SIGNS, anchor_orient

GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]
SUBSETS = {"GOOD_5": GOOD_5, "GOOD_6": GOOD_5 + ["varentropy"]}
EXTRA_SIGNS = {"varentropy": -1}
MIN_ROWS = 30


def lsml_auroc(fd, y, names, mask=None):
    """Canonical recipe, mirroring build_repgrid_featcache.good5_lsml_auroc:
    continuous L-SML over the present members, then epr-anchored global orientation."""
    present = [f for f in names if f in fd]
    if len(present) < 3:
        return None
    sub = {f: np.asarray(fd[f], dtype=float) for f in present}
    if mask is not None:
        sub = {f: v[mask] for f, v in sub.items()}
        y = y[mask]
    if len(y) < MIN_ROWS or len(set(y.tolist())) < 2:
        return None
    signs = {**FEATURE_SIGNS, **EXTRA_SIGNS}
    score, _ = lsml_continuous_pipeline(sub, present, signs)
    anchor_feat = "epr" if "epr" in present else present[0]
    anchor = zscore(sub[anchor_feat] * signs.get(anchor_feat, +1))
    score, _ = anchor_orient(np.asarray(score, dtype=float), anchor)
    auc, _, _ = boot_auc(y, score, n=500)
    return float(auc)


def oriented_auroc(y, x):
    """AUROC of a single view, reported oriented (>= 0.5) so magnitude reads directly."""
    m = np.isfinite(x)
    if m.sum() < MIN_ROWS or len(set(y[m].tolist())) < 2:
        return None
    auc, _, _ = boot_auc(y[m], x[m], n=500)
    return max(float(auc), 1.0 - float(auc))


def main():
    ap = argparse.ArgumentParser(description="Quantify the max_new truncation confound.")
    ap.add_argument("--featcache", default=os.path.join("local_cache", "regen", "repgrid_cells.pkl"))
    ap.add_argument("--subset", default="GOOD_5", choices=sorted(SUBSETS))
    ap.add_argument("--csv", default=None, help="also write the table here")
    args = ap.parse_args()

    path = args.featcache if os.path.isabs(args.featcache) else os.path.join(REPO, args.featcache)
    with open(path, "rb") as f:
        cells = pickle.load(f)

    names = SUBSETS[args.subset]
    rows = []
    for cell, v in sorted(cells.items()):
        fd = v["feats"]
        y = np.asarray(v["labels"]).astype(int)
        if "trace_length" not in fd:
            continue
        tl = np.asarray(fd["trace_length"], dtype=float)
        # The cap is not stored per cell in the featcache; on a pinned cell the max IS
        # the cap, and on an unpinned cell nothing is excluded either way.
        cap = np.nanmax(tl)
        complete = tl < cap
        a_all = lsml_auroc(fd, y, names)
        if a_all is None:
            continue
        a_cmp = lsml_auroc(fd, y, names, complete)
        rows.append({
            "cell": cell,
            "pinned_frac": float(np.mean(tl >= cap)),
            "cap_observed": int(cap),
            "n_all": int(len(y)),
            "n_complete": int(complete.sum()),
            "acc_all": float(y.mean()),
            "acc_complete": float(y[complete].mean()) if complete.any() else float("nan"),
            "trace_length_auroc": oriented_auroc(y, tl),
            f"{args.subset}_all": a_all,
            f"{args.subset}_complete": a_cmp,
            "delta": (a_cmp - a_all) if a_cmp is not None else None,
        })

    rows.sort(key=lambda r: -r["pinned_frac"])
    sk, ck = f"{args.subset}_all", f"{args.subset}_complete"
    print(f"{'cell':<36} {'pin%':>6} {'len_auc':>8} {sk:>11} {ck:>13} {'n_cmpl':>7} {'delta':>8}")
    for r in rows:
        cs = f"{r[ck]:.4f}" if r[ck] is not None else "n/a"
        ds = f"{r['delta']:+.4f}" if r["delta"] is not None else "n/a"
        la = f"{r['trace_length_auroc']:.4f}" if r["trace_length_auroc"] is not None else "n/a"
        print(f"{r['cell']:<36} {r['pinned_frac']:>6.1%} {la:>8} {r[sk]:>11.4f} "
              f"{cs:>13} {r['n_complete']:>7} {ds:>8}")

    heavy = [r for r in rows if r["pinned_frac"] >= 0.20 and r["delta"] is not None]
    clean = [r for r in rows if r["pinned_frac"] < 0.05 and r["delta"] is not None]
    print()
    if heavy:
        print(f"   heavily-capped cells (pinned >= 20%): n={len(heavy)}  "
              f"mean delta={np.mean([r['delta'] for r in heavy]):+.4f}")
    if clean:
        print(f"   clean cells        (pinned <  5%): n={len(clean)}  "
              f"mean delta={np.mean([r['delta'] for r in clean]):+.4f}")
    print("\n   delta = complete-only AUROC - all-rows AUROC. Near 0 on clean cells means the\n"
          "   cap is not doing the work. See the module docstring for why a large negative\n"
          "   delta is evidence about the CELL's headline number, not proof about the method.")

    if args.csv:
        out = args.csv if os.path.isabs(args.csv) else os.path.join(REPO, args.csv)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n   wrote {out}")


if __name__ == "__main__":
    main()
