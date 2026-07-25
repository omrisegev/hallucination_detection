"""
test_iterative_lsml_pruning.py — Evaluates Iterative L-SML Backward Elimination variants:
1. Iterative Feature Pruning (prune feature with min |w_j| until K = K_eff)
2. Iterative Group Pruning (GroupFS cluster centroids, prune min cluster weight)
3. Residual Elbow Pruning (prune feature that minimizes L-SML rank-1 residual)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.cluster import AgglomerativeClustering

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

def compute_lsml_weights(V):
    """
    Computes Nadler L-SML feature weights w_j from covariance matrix.
    """
    cov = np.cov(V, rowvar=False)
    # Add tiny diagonal regularization for numerical stability
    cov += 1e-6 * np.eye(cov.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    v1 = eigvecs[:, -1]
    lam1 = eigvals[-1]
    # L-SML weight w_j ~ v1_j / std_j
    std = np.sqrt(np.diag(cov)) + 1e-12
    weights = v1 / std
    return np.abs(weights), cov, v1, lam1

def compute_lsml_residual(V):
    """
    Computes L-SML rank-1 covariance residual || Cov * v1 - lambda1 * v1 ||
    """
    cov = np.cov(V, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v1 = eigvecs[:, -1]
    lam1 = eigvals[-1]
    res = np.linalg.norm(cov @ v1 - lam1 * v1)
    return res

def iterative_feature_pruning(V, target_k):
    """
    Iteratively prunes the feature with smallest L-SML weight until len == target_k.
    """
    remaining = list(range(V.shape[1]))
    while len(remaining) > target_k:
        sub_V = V[:, remaining]
        weights, _, _, _ = compute_lsml_weights(sub_V)
        min_idx = np.argmin(weights)
        remaining.pop(min_idx)
    return remaining

def iterative_residual_pruning(V, target_k):
    """
    Iteratively prunes the feature whose removal yields smallest L-SML residual.
    """
    remaining = list(range(V.shape[1]))
    while len(remaining) > target_k:
        best_res = 1e9
        best_drop = None
        for i, idx in enumerate(remaining):
            cand = [r for r in remaining if r != idx]
            res = compute_lsml_residual(V[:, cand])
            if res < best_res:
                best_res = res
                best_drop = i
        if best_drop is not None:
            remaining.pop(best_drop)
        else:
            break
    return remaining

def iterative_group_pruning(V, initial_C, target_C):
    """
    Clusters features into C groups, computes cluster centroid weights, 
    and iteratively prunes lowest weight cluster.
    """
    cov = np.cov(V, rowvar=False)
    std = np.sqrt(np.diag(cov)) + 1e-12
    corr = cov / np.outer(std, std)
    abs_corr = np.clip(np.abs(corr), 0.0, 1.0)
    dist_matrix = np.clip(1.0 - abs_corr, 0.0, 2.0)
    np.fill_diagonal(dist_matrix, 0.0)
    
    p = V.shape[1]
    ac = AgglomerativeClustering(n_clusters=min(initial_C, p), metric='precomputed', linkage='complete')
    clusters = ac.fit_predict(dist_matrix)
    
    active_clusters = list(range(min(initial_C, p)))
    while len(active_clusters) > target_C:
        # Build centroids for active clusters
        centroids = []
        for c in active_clusters:
            c_cols = np.where(clusters == c)[0]
            centroids.append(V[:, c_cols].mean(axis=1))
        C_mat = np.column_stack(centroids)
        w, _, _, _ = compute_lsml_weights(C_mat)
        min_c_idx = np.argmin(w)
        active_clusters.pop(min_c_idx)
        
    selected_features = []
    for c in active_clusters:
        c_cols = np.where(clusters == c)[0]
        selected_features.extend(c_cols[:2]) # readout top 2 features per cluster
    return list(dict.fromkeys(selected_features))

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
        k_eff = predict_k(V, list(range(p)), rule='eff_rank')
        
        # Method 1: Iterative Weight Pruning
        sel_weight = iterative_feature_pruning(V, k_eff)
        sub_w = V[:, sel_weight]
        z2_w = z2_sign_recovery(sub_w)
        sub_w_signed = sub_w * z2_w + 1e-6 * np.random.RandomState(42).randn(n, len(sel_weight))
        fused_w, _ = lsml_continuous(*[sub_w_signed[:, c] for c in range(len(sel_weight))])
        fused_w_orient, _ = anchor_orient(fused_w, zscore(sub_w[:, 0]))
        auc_weight = roc_auc_score(y, fused_w_orient)
        auc_weight = max(auc_weight, 1.0 - auc_weight)
        
        # Method 2: Iterative Residual Pruning
        sel_res = iterative_residual_pruning(V, k_eff)
        sub_r = V[:, sel_res]
        z2_r = z2_sign_recovery(sub_r)
        sub_r_signed = sub_r * z2_r + 1e-6 * np.random.RandomState(42).randn(n, len(sel_res))
        fused_r, _ = lsml_continuous(*[sub_r_signed[:, c] for c in range(len(sel_res))])
        fused_r_orient, _ = anchor_orient(fused_r, zscore(sub_r[:, 0]))
        auc_res = roc_auc_score(y, fused_r_orient)
        auc_res = max(auc_res, 1.0 - auc_res)
        
        # Method 3: Iterative GroupFS Cluster Pruning (Initial C=8 -> target C=k_eff)
        sel_grp = iterative_group_pruning(V, initial_C=8, target_C=k_eff)
        sub_g = V[:, sel_grp]
        z2_g = z2_sign_recovery(sub_g)
        sub_g_signed = sub_g * z2_g + 1e-6 * np.random.RandomState(42).randn(n, len(sel_grp))
        fused_g, _ = lsml_continuous(*[sub_g_signed[:, c] for c in range(len(sel_grp))])
        fused_g_orient, _ = anchor_orient(fused_g, zscore(sub_g[:, 0]))
        auc_grp = roc_auc_score(y, fused_g_orient)
        auc_grp = max(auc_grp, 1.0 - auc_grp)
        
        # GOOD_6 reference
        good6_cols = [f for f in REFERENCE_SUBSETS['GOOD_6'] if f in valid_cols]
        good6_idx = [valid_cols.index(f) for f in good6_cols]
        sub_g6 = V[:, good6_idx]
        z2_g6 = z2_sign_recovery(sub_g6)
        sub_g6_signed = sub_g6 * z2_g6 + 1e-6 * np.random.RandomState(42).randn(n, len(good6_idx))
        fused_g6, _ = lsml_continuous(*[sub_g6_signed[:, c] for c in range(len(good6_idx))])
        fused_g6_orient, _ = anchor_orient(fused_g6, zscore(sub_g6[:, 0]))
        auc_good6 = roc_auc_score(y, fused_g6_orient)
        auc_good6 = max(auc_good6, 1.0 - auc_good6)
        
        results.append({
            'cell': cell_key,
            'domain': domain,
            'k_eff': k_eff,
            'auc_iter_weight': auc_weight,
            'auc_iter_residual': auc_res,
            'auc_iter_group': auc_grp,
            'auc_good6': auc_good6
        })
        
    df = pd.DataFrame(results)
    out_csv = os.path.join(REPO, "results", "advisor_inscope", "iterative_pruning_results.csv")
    df.to_csv(out_csv, index=False)
    
    print("=== Iterative L-SML Pruning Benchmark Results ===")
    print(f"Total cells evaluated: {len(df)}")
    print(f"1. Iterative Weight Pruning AUROC   : {df['auc_iter_weight'].mean():.4f}")
    print(f"2. Iterative Residual Pruning AUROC : {df['auc_iter_residual'].mean():.4f}")
    print(f"3. Iterative Group Pruning AUROC    : {df['auc_iter_group'].mean():.4f}")
    print(f"GOOD_6 Reference Baseline AUROC     : {df['auc_good6'].mean():.4f}")
    print("\nSaved detailed results to:", out_csv)

if __name__ == '__main__':
    main()
