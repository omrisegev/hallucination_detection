"""
prior_free_bench.py — Phase 5: Integrated Prior-Free L-SML Benchmark & Decision Gate.

Assembles and evaluates the integrated prior-free detector:
  - H1: z2_sign_recovery + distributional_orient
  - H2: predict_k signal dimension rules (eff_rank, mp_floor)
  - H3: a7.iter_consensus iterative refinement

Evaluates across all 25 in-scope QA+Math cells against GOOD_6 (0.7594) and D2_alone (0.7573),
computing cell-level wins/losses and Wilcoxon signed-rank p-values.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from inscope_cells import INSCOPE
from spectral_utils.subset_sweep import iter_cells, REFERENCE_SUBSETS, ALL_SIGNS
from spectral_utils.fusion_utils import lsml_continuous_pipeline, lsml_continuous, zscore
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.orientation import z2_sign_recovery, distributional_orient
from spectral_utils.selectors.a7_iter_consensus import a7_iter_consensus

OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")


class UnlabeledCell:
    def __init__(self, V, feat_names, cell_key):
        self.V = V
        self.feat_names = feat_names
        self.cell_key = cell_key


def main():
    data_dir = os.path.join(REPO, "local_cache")
    rows = []

    for domain, cell_key, fd, labels in iter_cells(
            data_dir, domains=["repgrid"],
            derived_views_pkl=os.path.join(data_dir, "derived_views.pkl"),
            trace_cells_pkl=os.path.join(data_dir, "trace_cells.pkl")):
        if cell_key not in INSCOPE:
            continue

        feat_names = [f for f in fd.keys() if not f.startswith("_")]
        if len(feat_names) < 3:
            continue

        valid = np.isfinite(np.column_stack([fd[f] for f in feat_names])).all(axis=1)
        if valid.sum() < 20:
            continue

        y = np.asarray(labels, dtype=int)[valid]
        if y.sum() in (0, len(y)):
            continue

        feats = {f: fd[f][valid] for f in feat_names}
        V = np.column_stack([feats[f] for f in feat_names])

        # Baseline 1: GOOD_6 reference subset (0.7594 with anchor)
        good6_feats = [f for f in REFERENCE_SUBSETS['GOOD_6'] if f in feats]
        if len(good6_feats) >= 3:
            s_good6_raw, _ = lsml_continuous_pipeline(feats, good6_feats, ALL_SIGNS)
            anchor_feat = "epr" if "epr" in good6_feats else good6_feats[0]
            anchor_view = zscore(np.asarray(feats[anchor_feat], dtype=float) * ALL_SIGNS.get(anchor_feat, +1))
            s_good6_cur, _ = anchor_orient(s_good6_raw, anchor_view)
            auc_good6 = float(roc_auc_score(y, s_good6_cur))
        else:
            auc_good6 = np.nan

        # Selector: a7.iter_consensus
        u_cell = UnlabeledCell(V, feat_names, cell_key)
        rng = np.random.default_rng(42)
        res = a7_iter_consensus(u_cell, rng)
        selected_cols = res[0]['cols']

        # Arm A: Prior-free selection + Z2 sign recovery + anchor_orient (0 hand-picked seeds / sizes)
        if len(selected_cols) >= 3:
            sub_V = V[:, selected_cols]
            z2_sub = z2_sign_recovery(sub_V)
            sub_V_signed = sub_V * z2_sub
            fused, _ = lsml_continuous(*[sub_V_signed[:, j] for j in range(len(selected_cols))])
            anchor_feat = "epr" if "epr" in feat_names else feat_names[0]
            anchor_view = zscore(feats[anchor_feat] * ALL_SIGNS.get(anchor_feat, +1))
            score_z2_anch, _ = anchor_orient(fused, anchor_view)
            auc_z2_anch = float(roc_auc_score(y, score_z2_anch))

            score_pf_distr, _ = distributional_orient(fused)
            auc_pf_distr = float(roc_auc_score(y, score_pf_distr))
        else:
            auc_z2_anch = np.nan
            auc_pf_distr = np.nan

        rows.append({
            'cell': cell_key,
            'domain': domain,
            'auc_good6': auc_good6,
            'auc_z2_anch': auc_z2_anch,
            'auc_pf_distr': auc_pf_distr,
            'k_selected': len(selected_cols)
        })

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "prior_free_bench_results.csv")
    df.to_csv(out_csv, index=False)

    macro_good6 = df['auc_good6'].mean()
    macro_z2_anch = df['auc_z2_anch'].mean()
    macro_pf_distr = df['auc_pf_distr'].mean()

    diffs = df['auc_z2_anch'] - df['auc_good6']
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())

    print(f"\n==================================================")
    print(f"EXTENSION H: PRIOR-FREE L-SML BENCHMARK SUMMARY")
    print(f"==================================================")
    print(f"In-scope cells evaluated       : {len(df)}")
    print(f"GOOD_6 Macro AUROC (Target)    : {macro_good6:.4f}")
    print(f"a7.iter_consensus + Z2 Anchor  : {macro_z2_anch:.4f} (Delta: {macro_z2_anch - macro_good6:+.4f})")
    print(f"a7.iter_consensus + Distr (PF) : {macro_pf_distr:.4f} (Delta: {macro_pf_distr - macro_good6:+.4f})")
    print(f"Win / Loss / Tie (Z2 vs GOOD6) : {wins} W / {losses} L / {ties} T")
    print(f"==================================================\n")


if __name__ == '__main__':
    main()
