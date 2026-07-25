#!/usr/bin/env python
"""
eigsign_inscope_test.py — does the ORIGINAL L-SML sign rule hold in-scope? (WS2)

Omri's question (2026-07-22): the project abandoned the original L-SML
orientation — Paper-2 assumption (iii), "majority of classifiers beat random
in the +1 direction", which resolves the leading-eigenvector's global ± — in
favor of anchor_orient(epr). But the headline sign failures were measured on
RAG/GPQA, which are now out of scope. Does the original assumption hold on the
25 in-scope QA+math cells?

Mechanical fact this test exploits: lsml_continuous applies the Paper-2
majority flip INSIDE sml_fuse_signed (fusion_utils.py:347-348) at both fusion
levels — so the pipeline's raw output IS the original method's verdict, and
anchor_orient is a correction applied on top. Conditions per (cell, subset):

  paper2_signed   ALL_SIGNS-oriented features, fused, NO anchor_orient
                  = original majority rule, given correct per-feature signs
  current         + anchor_orient vs oriented epr (canon, score_subset recipe)
  orig_nosigns    RAW features (all signs +1), fused, NO anchor_orient
                  = the true original method with zero offline sign knowledge
                  (Step-110 predicts this fails: entropy-dominated pool)
  nosigns_anchor  RAW features + anchor_orient — offline knowledge for ONE
                  view (the anchor) only
  ceiling         max(a, 1-a) of the current-condition score — sign-oracle

AUROC is raw (never max(a,1-a)) except the explicit ceiling row.

Output: results/advisor_inscope/eigsign_inscope.csv + printed summary.
Usage:  python scripts/eigsign_inscope_test.py [--subset GOOD_6]
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

OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")

from inscope_cells import INSCOPE  # noqa: E402

from spectral_utils.orientation import distributional_orient, z2_sign_recovery  # noqa: E402

NO_SIGNS = {}          # signs.get(f, +1) -> every feature enters raw


def score_conditions(fd, labels, feat_names):
    """Mirror repgrid_scoring.score_subset validity rules, then run the
    sign-rule conditions on the same valid rows."""
    n = len(labels)
    X = np.full((n, len(feat_names)), np.nan)
    for j, f in enumerate(feat_names):
        arr = np.asarray(fd.get(f, np.full(n, np.nan)), dtype=float)
        X[:, j] = arr
    valid = np.isfinite(X).all(axis=1)
    if valid.sum() < 20 or len(feat_names) < 3:
        return None
    y = np.asarray(labels, dtype=int)[valid]
    if y.sum() in (0, len(y)):
        return None
    feats = {f: X[valid, j] for j, f in enumerate(feat_names)}

    def auc(s):
        return float(roc_auc_score(y, s))

    anchor_feat = "epr" if "epr" in feat_names else feat_names[0]
    anchor_view = zscore(np.asarray(feats[anchor_feat], dtype=float)
                         * ALL_SIGNS.get(anchor_feat, +1))

    # oriented-features arm
    s_signed, _ = lsml_continuous_pipeline(feats, feat_names, ALL_SIGNS)
    s_signed = np.asarray(s_signed, dtype=float)
    s_cur, flip_cur = anchor_orient(s_signed, anchor_view)
    s_distr, flip_distr = distributional_orient(s_signed)

    # Z2 sign recovery arm (data-driven relative signs)
    v_matrix = np.column_stack([feats[f] for f in feat_names])
    z2_rel_signs = z2_sign_recovery(v_matrix)
    z2_signs_dict = {f: z2_rel_signs[j] for j, f in enumerate(feat_names)}
    s_z2, _ = lsml_continuous_pipeline(feats, feat_names, z2_signs_dict)
    s_z2 = np.asarray(s_z2, dtype=float)
    s_z2_cur, flip_z2_cur = anchor_orient(s_z2, anchor_view)
    s_z2_distr, flip_z2_distr = distributional_orient(s_z2)

    # raw-features arm (original method: no offline sign knowledge)
    s_raw, _ = lsml_continuous_pipeline(feats, feat_names, NO_SIGNS)
    s_raw = np.asarray(s_raw, dtype=float)
    s_raw_anch, flip_raw = anchor_orient(s_raw, anchor_view)

    a_cur = auc(s_cur)
    return {
        "n": int(valid.sum()),
        "paper2_signed": auc(s_signed),
        "current": a_cur,
        "distr_orient": auc(s_distr),
        "z2_anchor": auc(s_z2_cur),
        "z2_distr": auc(s_z2_distr),
        "orig_nosigns": auc(s_raw),
        "nosigns_anchor": auc(s_raw_anch),
        "ceiling": max(a_cur, 1.0 - a_cur),
        "anchor_flipped_signed": bool(flip_cur),
        "anchor_flipped_raw": bool(flip_raw),
        "distr_flipped": bool(flip_distr),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default="GOOD_6",
                    choices=sorted(list(REFERENCE_SUBSETS) + ["FULL_POOL"]))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = os.path.join(REPO, "local_cache")
    rows = []
    for domain, cell_key, fd, labels in iter_cells(
            data_dir, domains=["repgrid"],
            derived_views_pkl=os.path.join(data_dir, "derived_views.pkl"),
            trace_cells_pkl=os.path.join(data_dir, "trace_cells.pkl")):
        if cell_key not in INSCOPE:
            continue
        if args.subset == "FULL_POOL":
            feat_names = [f for f in fd.keys() if not f.startswith("_")]
        else:
            feat_names = REFERENCE_SUBSETS[args.subset]

        r = score_conditions(fd, labels, feat_names)
        if r is None:
            print(f"  {cell_key}: not scorable on {args.subset} — skipped")
            continue
        r.update(cell=cell_key, subset=args.subset)
        rows.append(r)
        print(f"  {cell_key}: paper2 {r['paper2_signed']:.3f} | "
              f"current {r['current']:.3f} | distr {r['distr_orient']:.3f} | orig-nosigns "
              f"{r['orig_nosigns']:.3f}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(OUT_DIR,
                                   f"eigsign_inscope_{args.subset}.csv")
    df.to_csv(out, index=False)

    conds = ["paper2_signed", "current", "distr_orient", "z2_anchor", "z2_distr", "orig_nosigns", "nosigns_anchor",
             "ceiling"]
    print(f"\n=== {args.subset}, {len(df)} in-scope cells ===")
    for c in conds:
        wrong = int((df[c] < 0.5).sum())
        print(f"  {c:16s} macro {df[c].mean():.4f}   cells<0.5: {wrong}")
    # the verdict Omri asked for: does the majority rule alone get the sign
    # right where the anchor also does?
    sign_ok = int(((df["paper2_signed"] - 0.5) * (df["current"] - 0.5) > 0).sum())
    print(f"  paper2 majority rule agrees with anchor sign on "
          f"{sign_ok}/{len(df)} cells")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
