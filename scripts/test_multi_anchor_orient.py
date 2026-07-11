#!/usr/bin/env python
"""
test_multi_anchor_orient.py — single-epr anchor vs multi-feature-average anchor (Step 0b).

After lsml_continuous_pipeline fuses the (already FEATURE_SIGNS-oriented) views, the
fused score still carries the leading-eigenvector's arbitrary global +- sign. The
production fix (repgrid_scoring.score_subset, subset_sweep) resolves it label-free with
anchor_orient against ONE view: oriented z-scored `epr`. Step 158 flagged that single
anchor as fragile at low T. multipass_lsml_continuous (fusion_utils.py:691) already
trusts a stronger pattern for its CROSS-pass step — anchoring against the equal-weight
mean of the oriented score views. This script tests that same pattern WITHIN-pass:

    arm A (current): anchor_orient(fused, zscore(oriented epr))          [epr-or-first]
    arm B (new):     anchor_orient(fused, mean of ALL oriented z-scored subset views)

Both arms share the identical fused score, so per cell they either agree exactly
(delta = 0) or one arm mirrors the other (aucB = 1 - aucA). The interesting cells are
the disagreements: there, whichever arm lands above 0.5 got the sign right (assuming
the fused signal is genuinely informative).

Battery: the full local_cache cell set (ALL domains, gpqa included — this is the sign
battery from Steps 110/131/134/154, not the GPQA-excluded breakdown). AUROC is raw
(never max(a, 1-a)), boot_auc(labels, score) with 95% CI semantics as everywhere else.

Usage:
    PYTHONPATH=. python scripts/test_multi_anchor_orient.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.feature_utils import FEAT_NAMES
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.repgrid_scoring import ALL_SIGNS
from spectral_utils import subset_sweep

H16 = list(FEAT_NAMES[:16])
SUBSETS = {
    "GOOD_5":  ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"],
    "ALL_H16": H16,
}


def fuse_and_orient_both(feats_dict, labels, feat_names, n_boot=1000):
    """One lsml_continuous_pipeline fuse, oriented two ways. Mirrors
    repgrid_scoring.score_subset row-validity semantics (all subset features finite,
    >=20 valid rows, both classes, >=3 features). Returns dict or None."""
    present = [f for f in feat_names if f in feats_dict]
    if len(present) < 3:
        return None
    X = np.column_stack([np.asarray(feats_dict[f], dtype=float) for f in present])
    valid = np.isfinite(X).all(axis=1)
    y = np.asarray(labels, dtype=int)[valid]
    if valid.sum() < 20 or y.sum() in (0, len(y)):
        return None
    fd = {f: X[valid, j] for j, f in enumerate(present)}

    fused, _ = lsml_continuous_pipeline(fd, present, ALL_SIGNS)
    fused = np.asarray(fused, dtype=float)

    # Arm A — production rule: single oriented epr view (first feature if epr absent).
    anchor_feat = "epr" if "epr" in present else present[0]
    anchor_a = zscore(np.asarray(fd[anchor_feat], dtype=float)
                      * ALL_SIGNS.get(anchor_feat, +1))
    score_a, flip_a = anchor_orient(fused, anchor_a)

    # Arm B — cross-pass pattern applied within-pass: equal-weight mean of every
    # oriented z-scored view in the subset (the avg_fused analogue).
    views = [zscore(np.asarray(fd[f], dtype=float) * ALL_SIGNS.get(f, +1))
             for f in present]
    anchor_b = np.mean(views, axis=0)
    score_b, flip_b = anchor_orient(fused, anchor_b)

    auc_a, lo_a, hi_a = boot_auc(y, score_a, n=n_boot)
    auc_b, lo_b, hi_b = boot_auc(y, score_b, n=n_boot)
    return {
        "n": int(valid.sum()), "n_feats": len(present),
        "auc_epr": float(auc_a), "auc_multi": float(auc_b),
        "delta": float(auc_b) - float(auc_a),
        "disagree": bool(flip_a != flip_b),
    }


def main():
    data_dir = "local_cache"
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} not found", file=sys.stderr)
        return 1

    per_subset = {name: [] for name in SUBSETS}
    for domain, cell_key, fd, labels in subset_sweep.iter_cells(data_dir):
        for sub_name, sub_feats in SUBSETS.items():
            r = fuse_and_orient_both(fd, labels, sub_feats)
            if r is None:
                continue
            r.update({"domain": domain, "cell": cell_key})
            per_subset[sub_name].append(r)
            tag = "  DISAGREE" if r["disagree"] else ""
            print(f"[{sub_name:7s}] {domain:8s} {cell_key:45s} n={r['n']:5d} "
                  f"epr={r['auc_epr']:.4f} multi={r['auc_multi']:.4f}{tag}", flush=True)

    for sub_name, rows in per_subset.items():
        if not rows:
            print(f"\n== {sub_name}: no scorable cells ==")
            continue
        a = np.array([r["auc_epr"] for r in rows])
        b = np.array([r["auc_multi"] for r in rows])
        d = b - a
        n_dis = sum(r["disagree"] for r in rows)
        wins = int((d > 0.01).sum())
        losses = int((d < -0.01).sum())
        ties = len(d) - wins - losses
        print(f"\n== {sub_name}: {len(rows)} cells ==")
        print(f"   macro AUROC   epr-anchor={a.mean():.4f}   multi-anchor={b.mean():.4f}"
              f"   delta={d.mean():+.4f}")
        print(f"   win/tie/loss (multi vs epr, +-0.01): {wins}/{ties}/{losses}"
              f"   |   sign disagreements: {n_dis}/{len(rows)}")
        if n_dis:
            print("   disagreement cells (aucB = 1 - aucA by construction):")
            for r in rows:
                if r["disagree"]:
                    if max(r["auc_multi"], r["auc_epr"]) > 0.5:
                        verdict = ("multi side is >0.5" if r["auc_multi"] > r["auc_epr"]
                                   else "epr side is >0.5")
                    else:
                        verdict = "both sides <=0.5"
                    print(f"     {r['domain']:8s} {r['cell']:45s} "
                          f"epr={r['auc_epr']:.4f} multi={r['auc_multi']:.4f}  -> {verdict}")

    g5 = per_subset.get("GOOD_5", [])
    if g5:
        macro = np.mean([r["auc_epr"] for r in g5])
        print(f"\nSanity: GOOD_5 epr-anchor macro = {macro:.4f} "
              f"(known battery reference ~0.65)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
