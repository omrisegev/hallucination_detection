"""
test_user_pipeline.py — Evaluates user's proposed prior-free selection pipeline:
1. Full 30-feature Z2-synchronized L-SML fusion -> Pseudo-label S_30
2. Adaptive size K* from eff_rank / mp_floor
3. mRMR feature selection of top K* features guided by S_30
4. Final Z2-synchronized L-SML fusion of selected K* features
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from inscope_cells import INSCOPE
from spectral_utils.subset_sweep import iter_cells, ALL_SIGNS, REFERENCE_SUBSETS
from spectral_utils.orientation import z2_sign_recovery
from spectral_utils.fusion_utils import lsml_continuous, zscore
from spectral_utils.selectors.adaptive_k import predict_k
from spectral_utils.streaming_utils import anchor_orient

def run_mrmr_selection(V, target, K):
    """
    Greedy mRMR feature selection: max relevance to target, min redundancy among selected.
    """
    n, p = V.shape
    if K >= p:
        return list(range(p))
    
    # Compute relevance r_j = corr(V_j, target)
    target_std = np.std(target)
    if target_std < 1e-8:
        return list(range(min(K, p)))
    
    rel = np.array([abs(np.corrcoef(V[:, j], target)[0, 1]) for j in range(p)])
    rel = np.nan_to_num(rel, nan=0.0)
    
    # Compute correlation matrix among features
    cov = np.cov(V, rowvar=False)
    std = np.sqrt(np.diag(cov)) + 1e-12
    corr = np.abs(cov / np.outer(std, std))
    corr = np.nan_to_num(corr, nan=0.0)
    
    selected = [int(np.argmax(rel))]
    candidates = set(range(p)) - set(selected)
    
    for _ in range(min(K - 1, p - 1)):
        best_score = -1e9
        best_idx = None
        for c in candidates:
            redundancy = np.mean([corr[c, s] for s in selected])
            score = rel[c] - redundancy
            if score > best_score:
                best_score = score
                best_idx = c
        if best_idx is not None:
            selected.append(best_idx)
            candidates.remove(best_idx)
        else:
            break
            
    return selected

def main():
    data_dir = os.path.join(REPO, "local_cache")
    results = []
    
    for domain, cell_key, fd, labels in iter_cells(
        data_dir, ["repgrid"],
        os.path.join(data_dir, "derived_views.pkl"),
        os.path.join(data_dir, "trace_cells.pkl")
    ):
        if cell_key not in INSCOPE:
            continue
            
        feat_names = [f for f in fd.keys() if not f.startswith("_")]
        valid_cols = [f for f in feat_names if np.isfinite(fd[f]).mean() > 0.9]
        if len(valid_cols) < 3:
            continue
            
        # Orient using Z2 sync / signed features
        feats = {f: fd[f] * ALL_SIGNS.get(f, +1) for f in valid_cols}
        X = np.column_stack([feats[f] for f in valid_cols])
        valid = np.isfinite(X).all(axis=1)
        if valid.sum() < 20:
            continue
            
        V = X[valid, :]
        y = np.asarray(labels, dtype=int)[valid]
        if y.sum() in (0, len(y)):
            continue
            
        n, p = V.shape
        
        # Step 1: Z2-synchronized L-SML fusion on full pool -> Pseudo-label S_30
        z2_full = z2_sign_recovery(V)
        V_signed_full = V * z2_full + 1e-6 * np.random.RandomState(42).randn(n, p)
        fused_full, _ = lsml_continuous(*[V_signed_full[:, c] for c in range(p)])
        
        # Anchor orient for evaluation
        anchor_col = V[:, 0]
        fused_full_oriented, _ = anchor_orient(fused_full, zscore(anchor_col))
        auc_full_30 = roc_auc_score(y, fused_full_oriented)
        auc_full_30 = max(auc_full_30, 1.0 - auc_full_30)
        
        # Step 2: Adaptive K* from eff_rank and mp_floor
        k_eff = predict_k(V, list(range(p)), rule='eff_rank')
        k_mp = predict_k(V, list(range(p)), rule='mp_floor')
        
        # Step 3: mRMR selection guided by fused_full pseudo-label
        sel_eff = run_mrmr_selection(V_signed_full, fused_full, k_eff)
        sel_mp = run_mrmr_selection(V_signed_full, fused_full, k_mp)
        
        # Step 4: Final Z2 L-SML fusion on selected K* features
        # For eff_rank
        sub_eff = V[:, sel_eff]
        z2_eff = z2_sign_recovery(sub_eff)
        sub_eff_signed = sub_eff * z2_eff + 1e-6 * np.random.RandomState(42).randn(n, len(sel_eff))
        fused_eff, _ = lsml_continuous(*[sub_eff_signed[:, c] for c in range(len(sel_eff))])
        fused_eff_oriented, _ = anchor_orient(fused_eff, zscore(sub_eff[:, 0]))
        auc_eff = roc_auc_score(y, fused_eff_oriented)
        auc_eff = max(auc_eff, 1.0 - auc_eff)
        
        # For mp_floor
        sub_mp = V[:, sel_mp]
        z2_mp = z2_sign_recovery(sub_mp)
        sub_mp_signed = sub_mp * z2_mp + 1e-6 * np.random.RandomState(42).randn(n, len(sel_mp))
        fused_mp, _ = lsml_continuous(*[sub_mp_signed[:, c] for c in range(len(sel_mp))])
        fused_mp_oriented, _ = anchor_orient(fused_mp, zscore(sub_mp[:, 0]))
        auc_mp = roc_auc_score(y, fused_mp_oriented)
        auc_mp = max(auc_mp, 1.0 - auc_mp)
        
        # GOOD_6 reference AUC
        good6_cols = [f for f in REFERENCE_SUBSETS['GOOD_6'] if f in valid_cols]
        good6_idx = [valid_cols.index(f) for f in good6_cols]
        sub_g6 = V[:, good6_idx]
        z2_g6 = z2_sign_recovery(sub_g6)
        sub_g6_signed = sub_g6 * z2_g6 + 1e-6 * np.random.RandomState(42).randn(n, len(good6_idx))
        fused_g6, _ = lsml_continuous(*[sub_g6_signed[:, c] for c in range(len(good6_idx))])
        fused_g6_oriented, _ = anchor_orient(fused_g6, zscore(sub_g6[:, 0]))
        auc_good6 = roc_auc_score(y, fused_g6_oriented)
        auc_good6 = max(auc_good6, 1.0 - auc_good6)
        
        results.append({
            'cell': cell_key,
            'domain': domain,
            'k_eff': k_eff,
            'k_mp': k_mp,
            'auc_full_30': auc_full_30,
            'auc_mrmr_eff': auc_eff,
            'auc_mrmr_mp': auc_mp,
            'auc_good6': auc_good6
        })
        
    df = pd.DataFrame(results)
    out_csv = os.path.join(REPO, "results", "advisor_inscope", "user_pipeline_results.csv")
    df.to_csv(out_csv, index=False)
    
    print("=== User Proposed Pipeline Benchmark Results ===")
    print(f"Total cells evaluated: {len(df)}")
    print(f"Full 30-feature Z2 L-SML Macro AUROC : {df['auc_full_30'].mean():.4f}")
    print(f"User Pipeline (eff_rank, K*={df['k_eff'].mean():.1f}) AUROC : {df['auc_mrmr_eff'].mean():.4f}")
    print(f"User Pipeline (mp_floor, K*={df['k_mp'].mean():.1f}) AUROC : {df['auc_mrmr_mp'].mean():.4f}")
    print(f"GOOD_6 Reference Baseline AUROC       : {df['auc_good6'].mean():.4f}")
    print("\nSaved detailed results to:", out_csv)

if __name__ == '__main__':
    main()
