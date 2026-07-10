#!/usr/bin/env python
"""
score_edis.py — score EDIS (arXiv 2602.01288) as a stand-alone unsupervised confidence
signal on replication-grid cells, and measure its complementarity to our L-SML GOOD_5.

EDIS is a competitor entropy-dynamics baseline: it counts burst/rebound spikes in the
per-token entropy trace H(n) and scales by trace variance. As a CONFIDENCE feature we use
-compute_edis(H) (higher EDIS = more instability = less likely correct), matching the
Phase-14 GPQA comparison convention.

For each cell we report:
    EDIS AUROC (raw, 95% bootstrap CI)  — the competitor number on OUR trace
    L-SML GOOD_5 AUROC                  — our method on the same rows
    rho(EDIS, L-SML)                    — Spearman correlation of the two confidence signals
                                          (low |rho| => complementary => fusable)

Usage:
    PYTHONPATH=. python scripts/score_edis.py --pkl cache/repgrid/lapeigvals_gsm8k_llama8b/raw_gsm8k_T1.0.pkl --cell gsm8k_llama8b
    PYTHONPATH=. python scripts/score_edis.py --cache-dir cache/repgrid           # all cells with raw H(n)
"""
import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.feature_utils import compute_edis
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.repgrid_scoring import load_repgrid_cell, subset_matrix, ALL_SIGNS
from spectral_utils.streaming_utils import anchor_orient

GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]


def edis_per_candidate(pkl_path):
    """-EDIS(H) confidence per candidate, in the SAME order load_repgrid_cell walks
    (sorted problem ids, candidates in list order) so it aligns row-for-row."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    scores, have_trace = [], []
    for idx in sorted(data.keys()):
        for c in data[idx]["candidates"]:
            H = c.get("token_entropies")
            if H is None or len(H) < 2:
                scores.append(np.nan)
                have_trace.append(False)
            else:
                scores.append(-compute_edis(H))   # confidence: higher -> more likely correct
                have_trace.append(True)
    return np.asarray(scores, dtype=float), np.asarray(have_trace, dtype=bool)


def lsml_good5_score(cell):
    """L-SML continuous GOOD_5 confidence vector + valid mask (label-free anchor_orient)."""
    rows, labels = cell["rows"], cell["labels"]
    present = [f for f in GOOD_5 if f in set(cell["available"])]
    if len(present) < 3:
        return None, None, present
    X, valid = subset_matrix(rows, present)
    feats_dict = {f: X[valid, j] for j, f in enumerate(present)}
    score, _ = lsml_continuous_pipeline(feats_dict, present, ALL_SIGNS)
    anchor_feat = "epr" if "epr" in present else present[0]
    anchor_view = zscore(np.asarray(feats_dict[anchor_feat], dtype=float)
                         * ALL_SIGNS.get(anchor_feat, +1))
    score, _ = anchor_orient(np.asarray(score, dtype=float), anchor_view)
    return score, valid, present


def score_one(pkl_path, cell_id, n_boot=1000):
    cell = load_repgrid_cell(pkl_path)
    labels = cell["labels"]
    edis, edis_ok = edis_per_candidate(pkl_path)
    assert len(edis) == len(labels), f"row mismatch {len(edis)} vs {len(labels)}"

    # EDIS AUROC on rows with a usable trace + both classes present.
    m = edis_ok & np.isfinite(edis)
    y = labels[m].astype(int)
    res = {"cell": cell_id, "n_rows": int(m.sum()), "acc": round(float(labels.mean()), 4),
           "edis_auroc": None, "edis_lo": None, "edis_hi": None,
           "lsml_good5_auroc": None, "rho_edis_lsml": None, "n_overlap": None}
    if m.sum() >= 20 and 0 < y.sum() < len(y):
        auc, lo, hi = boot_auc(y, edis[m], n=n_boot)
        res.update(edis_auroc=round(float(auc), 4), edis_lo=round(float(lo), 4),
                   edis_hi=round(float(hi), 4))

    lsml, lsml_valid, present = lsml_good5_score(cell)
    if lsml is not None:
        yv = labels[lsml_valid].astype(int)
        if lsml_valid.sum() >= 20 and 0 < yv.sum() < len(yv):
            auc, _, _ = boot_auc(yv, lsml, n=n_boot)
            res["lsml_good5_auroc"] = round(float(auc), 4)
        # rho on the overlap: rows valid for GOOD_5 AND with an EDIS value.
        edis_on_valid = edis[lsml_valid]
        both = np.isfinite(edis_on_valid) & np.isfinite(lsml)
        if both.sum() >= 20:
            rho, _ = spearmanr(edis_on_valid[both], np.asarray(lsml)[both])
            res["rho_edis_lsml"] = round(float(rho), 4)
            res["n_overlap"] = int(both.sum())
    print(f"== {cell_id}: EDIS={res['edis_auroc']} CI[{res['edis_lo']},{res['edis_hi']}] "
          f"| L-SML GOOD_5={res['lsml_good5_auroc']} | rho={res['rho_edis_lsml']} "
          f"(n={res['n_rows']}, overlap={res['n_overlap']}, feats={present})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None, help="single raw pkl to score")
    ap.add_argument("--cell", default=None, help="label for --pkl mode")
    ap.add_argument("--cache-dir", default="cache/repgrid", help="scan for raw_*.pkl cells")
    ap.add_argument("--out", default="results/repgrid/edis_scores.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    jobs = []
    if args.pkl:
        jobs.append((args.pkl, args.cell or os.path.basename(os.path.dirname(args.pkl))))
    else:
        for man in sorted(glob.glob(os.path.join(args.cache_dir, "*", "manifest.json"))):
            cell_dir = os.path.dirname(man)
            pkls = sorted(glob.glob(os.path.join(cell_dir, "raw_*.pkl")))
            if pkls:
                jobs.append((pkls[0], os.path.basename(cell_dir)))

    rows = [score_one(p, c, n_boot=args.n_boot) for p, c in jobs]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
