#!/usr/bin/env python
"""
compute_legacy_upcr.py — per-cell U-PCR (+ L-SML cross-check) for the LEGACY local caches.

The AIRCC replication-grid cells carry both L-SML and U-PCR per feature subset
(results/repgrid/scores_lsml_upcr.csv, from score_repgrid.py). The legacy local caches
(local_cache/{gsm8k,math500,rag,qa}_res.pkl) only ever got per-cell L-SML
(results/subset_sweep/sweep_summary.csv); U-PCR existed only as an aggregated macro
(results/upcr_comparison.pkl). This closes the gap using the EXACT recipe of
spectral_utils.repgrid_scoring.score_subset, so legacy and AIRCC numbers are comparable:

    subset-valid rows (every subset feature finite) -> lsml_continuous_pipeline /
    upcr_pipeline over ALL_SIGNS -> anchor_orient against the oriented `epr` view
    (first subset feature if epr absent) -> raw boot_auc(labels, score), never max(a,1-a).

L-SML rows are emitted too, as a consistency cross-check against sweep_summary.csv (the
sweep median-imputes rare NaNs instead of dropping rows; on these caches the difference
is nil-to-negligible). Downstream consumers should filter on `method`.

GPQA is skipped — excluded from the per-domain breakdown per Omri's request.

Usage:
    PYTHONPATH=. python scripts/compute_legacy_upcr.py
Output:
    results/subset_sweep/upcr_legacy.csv
    (domain, cell_key, n, subset, method, n_feats, auroc, lo, hi, flipped)
"""
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.feature_utils import FEAT_NAMES
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline, upcr_pipeline
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.repgrid_scoring import ALL_SIGNS
from spectral_utils import subset_sweep

# Same reference subsets score_repgrid.py runs on the AIRCC cells (Step-154 sweep).
H16 = list(FEAT_NAMES[:16])
STABLE_H9 = ["epr", "low_band_power", "high_band_power", "hl_ratio",
             "spectral_centroid", "sw_var_peak", "rpdi", "pe_mean", "cusum_max"]
SUBSETS = {
    "consensus_4": ["spectral_entropy", "sw_var_peak", "cusum_max", "cusum_shift_idx"],
    "GOOD_5":      ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"],
    "STABLE_H9":   STABLE_H9,
    "ALL_H16":     H16,
}

OUT_CSV = "results/subset_sweep/upcr_legacy.csv"


def score_subset_arrays(feats_dict, labels, feat_names, method, n_boot=1000):
    """repgrid_scoring.score_subset semantics on iter_cells (feats_dict, labels) arrays."""
    present = [f for f in feat_names if f in feats_dict]
    if len(present) < 3:
        return None
    X = np.column_stack([np.asarray(feats_dict[f], dtype=float) for f in present])
    valid = np.isfinite(X).all(axis=1)
    y = np.asarray(labels, dtype=int)[valid]
    if valid.sum() < 20 or y.sum() in (0, len(y)):
        return None
    fd = {f: X[valid, j] for j, f in enumerate(present)}

    if method == "lsml":
        score, _ = lsml_continuous_pipeline(fd, present, ALL_SIGNS)
    else:
        score, *_ = upcr_pipeline(fd, present, ALL_SIGNS)

    anchor_feat = "epr" if "epr" in present else present[0]
    anchor_view = zscore(np.asarray(fd[anchor_feat], dtype=float)
                         * ALL_SIGNS.get(anchor_feat, +1))
    score, flipped = anchor_orient(np.asarray(score, dtype=float), anchor_view)
    auc, lo, hi = boot_auc(y, score, n=n_boot)
    if not np.isfinite(auc):
        return None
    return {
        "n": int(valid.sum()), "n_feats": len(present),
        "auroc": round(float(auc), 4), "lo": round(float(lo), 4),
        "hi": round(float(hi), 4), "flipped": bool(flipped),
    }


def main():
    data_dir = "local_cache"
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} not found", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows, n_cells = [], 0
    for domain, cell_key, fd, labels in subset_sweep.iter_cells(data_dir):
        if domain == "gpqa":
            continue  # excluded from the per-domain breakdown per Omri's request
        n_cells += 1
        line = [f"{domain:8s} {cell_key:45s}"]
        for sub_name, sub_feats in SUBSETS.items():
            for method in ("lsml", "upcr"):
                r = score_subset_arrays(fd, labels, sub_feats, method)
                if r is None:
                    continue
                r.update({"domain": domain, "cell_key": cell_key,
                          "subset": sub_name, "method": method})
                rows.append(r)
                if sub_name == "GOOD_5":
                    line.append(f"{method}_G5={r['auroc']:.4f}")
        print(" ".join(line), flush=True)

    if not rows:
        print("No scorable cells — nothing written.", file=sys.stderr)
        return 1

    fieldnames = ["domain", "cell_key", "n", "subset", "method",
                  "n_feats", "auroc", "lo", "hi", "flipped"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows ({n_cells} cells x subsets x methods) -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
