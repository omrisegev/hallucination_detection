#!/usr/bin/env python
"""
eigengap_vs_residual.py — K-selection method comparison on CURRENT cells (WS5).

Omri's question (2026-07-22): the deployed L-SML picks K by minimizing the
Eq-14 residual; the eigengap heuristic exists (`_eigengap_K`) but lost on
Step-106/107-era data (synthetic K=3: residual 0.869 vs eigengap 0.814;
eigengap systematically collapses to K=2 on real data). That evidence predates
the current 25-cell in-scope grid — this script re-runs the comparison so the
advisor-facing claim ("the K-selection choice was empirical") is measured on
the data we actually report.

Per (cell, subset): fuse the same oriented z-scored views under
method='residual' and method='eigengap' (everything else identical — same
anchor_orient(epr) global-sign resolution as the canonical score_subset
recipe), record chosen K and raw AUROC.

Output: results/advisor_inscope/eigengap_vs_residual.csv + printed summary.
Usage:  python scripts/eigengap_vs_residual.py [--subsets GOOD_6,LOCO_5,ALL_H16]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spectral_utils.subset_sweep import (   # noqa: E402
    iter_cells, ALL_SIGNS, REFERENCE_SUBSETS)
from spectral_utils.fusion_utils import lsml_continuous_pipeline, zscore  # noqa: E402
from spectral_utils.streaming_utils import anchor_orient  # noqa: E402
from inscope_cells import INSCOPE  # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")


def score_method(feats, feat_names, anchor_view, y, method):
    s, meta = lsml_continuous_pipeline(feats, feat_names, ALL_SIGNS,
                                       method=method)
    s, _ = anchor_orient(np.asarray(s, dtype=float), anchor_view)
    return float(roc_auc_score(y, s)), int(meta["K"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", default="GOOD_6,LOCO_5,ALL_H16")
    args = ap.parse_args()
    subsets = {s: REFERENCE_SUBSETS[s] for s in args.subsets.split(",")}

    data_dir = os.path.join(REPO, "local_cache")
    rows = []
    for domain, cell_key, fd, labels in iter_cells(
            data_dir, domains=["repgrid"],
            derived_views_pkl=os.path.join(data_dir, "derived_views.pkl"),
            trace_cells_pkl=os.path.join(data_dir, "trace_cells.pkl")):
        if cell_key not in INSCOPE:
            continue
        n = len(labels)
        y_all = np.asarray(labels, dtype=int)
        for sub_name, feat_names in subsets.items():
            X = np.full((n, len(feat_names)), np.nan)
            for j, f in enumerate(feat_names):
                arr = np.asarray(fd.get(f, np.full(n, np.nan)), dtype=float)
                X[:, j] = arr
            valid = np.isfinite(X).all(axis=1)
            if valid.sum() < 20:
                continue
            y = y_all[valid]
            if y.sum() in (0, len(y)):
                continue
            feats = {f: X[valid, j] for j, f in enumerate(feat_names)}
            anchor_feat = "epr" if "epr" in feat_names else feat_names[0]
            anchor_view = zscore(np.asarray(feats[anchor_feat], dtype=float)
                                 * ALL_SIGNS.get(anchor_feat, +1))
            a_res, k_res = score_method(feats, feat_names, anchor_view, y,
                                        "residual")
            a_eig, k_eig = score_method(feats, feat_names, anchor_view, y,
                                        "eigengap")
            rows.append(dict(cell=cell_key, subset=sub_name,
                             n=int(valid.sum()),
                             K_residual=k_res, K_eigengap=k_eig,
                             auroc_residual=a_res, auroc_eigengap=a_eig,
                             delta=a_res - a_eig))
        print(f"  {cell_key}: done")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "eigengap_vs_residual.csv")
    df.to_csv(out, index=False)

    print()
    for sub_name in subsets:
        d = df[df["subset"] == sub_name]
        if not len(d):
            continue
        wins = int((d["delta"] > 0).sum())
        losses = int((d["delta"] < 0).sum())
        k2 = int((d["K_eigengap"] == 2).sum())
        print(f"=== {sub_name} ({len(d)} cells) ===")
        print(f"  residual macro {d['auroc_residual'].mean():.4f} | "
              f"eigengap macro {d['auroc_eigengap'].mean():.4f} | "
              f"delta {d['delta'].mean()*100:+.2f}pp ({wins}W/{losses}L)")
        print(f"  eigengap picked K=2 on {k2}/{len(d)} cells; "
              f"residual K values: {sorted(d['K_residual'].unique())}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
