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

from inscope_cells import GROUP
from spectral_utils.fusion_utils import lsml_continuous
from spectral_utils.selectors.adaptive_k import predict_k

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
    # Step 201 rewrite (defects 8 + 9): the previous version built the pool from
    # `fd.keys()`, anchored orientation on `sub[:, 0]` (an arbitrary column),
    # added undocumented `1e-6 * randn` jitter, and applied `max(auc, 1 - auc)`
    # to every arm — a label-peeking sign oracle that floored each number at 0.5
    # and made none of these arms label-free. Its GOOD_6 reference came out
    # 0.7273 instead of the canonical 0.7594, so every comparison was against a
    # mis-computed baseline. Now: canonical loading + scoring, raw AUROC.
    from inscope_bench_common import (load_cells, score_cols, good6_score,
                                      assert_good6)
    from spectral_utils.selectors.adaptive_k import raw_k

    cells = load_cells()
    ok, macro_g6 = assert_good6(cells)
    if not ok:
        print("FAIL — refusing to report (SPEC_gap_ladder §8).")
        sys.exit(1)

    results = []
    for cell_key, cell in sorted(cells.items()):
        V = cell["V"]
        p = V.shape[1]
        k_eff = predict_k(V, list(range(p)), rule='eff_rank')

        sel_weight = iterative_feature_pruning(V, k_eff)
        sel_res = iterative_residual_pruning(V, k_eff)
        sel_grp = iterative_group_pruning(V, initial_C=8, target_C=k_eff)

        results.append({
            'cell': cell_key,
            'group': GROUP.get(cell_key, '?'),
            'k_eff': k_eff,
            'raw_eff_rank': round(float(raw_k(V, list(range(p)), 'eff_rank')), 3),
            'auc_iter_weight': score_cols(cell, sel_weight),
            'auc_iter_residual': score_cols(cell, sel_res),
            'auc_iter_group': score_cols(cell, sel_grp),
            'auc_good6': good6_score(cell),
        })
        r = results[-1]
        print(f"  {cell_key:34s} K={k_eff} | weight {r['auc_iter_weight']:.4f} | "
              f"resid {r['auc_iter_residual']:.4f} | group {r['auc_iter_group']:.4f} "
              f"| good6 {r['auc_good6']:.4f}", flush=True)
        
    df = pd.DataFrame(results)
    out_csv = os.path.join(REPO, "results", "advisor_inscope", "iterative_pruning_results.csv")
    df.to_csv(out_csv, index=False)
    
    from scipy.stats import wilcoxon
    print("\n=== Iterative L-SML pruning (canonical scoring, raw AUROC) ===")
    print(f"cells: {len(df)}   GOOD_6 macro: {macro_g6:.4f} (canonical 0.7594)")
    print(f"k_eff distribution: {df['k_eff'].value_counts().to_dict()}  "
          f"(raw eff_rank median {df['raw_eff_rank'].median():.2f})")
    base = df['auc_good6'].to_numpy()
    print(f"\n{'arm':22s} {'macro':>8s} {'vs GOOD_6':>11s} {'W/L':>8s} {'p':>9s}")
    print("-" * 64)
    for arm in ('auc_iter_weight', 'auc_iter_residual', 'auc_iter_group',
                'auc_good6'):
        v = df[arm].to_numpy()
        d = v - base
        try:
            pv = float(wilcoxon(v, base).pvalue) if np.any(d != 0) else float('nan')
        except Exception:
            pv = float('nan')
        print(f"{arm:22s} {v.mean():8.4f} {d.mean()*100:+10.2f}pp "
              f"{int((d>0).sum()):3d}/{int((d<0).sum()):<3d} {pv:9.4f}")
    print("\nSaved detailed results to:", out_csv)

if __name__ == '__main__':
    main()
