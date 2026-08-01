#!/usr/bin/env python3
import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from scripts.run_lsml_experiments import gather_all_cells, FEATURE_SIGNS
from spectral_utils.fusion_utils import boot_auc, lsml_continuous_pipeline

NON_MONOTONE_FEATURES = ['pe_mean', 'rpdi', 'dominant_freq', 'spectral_entropy', 'epr_energy']

def safe_auc(lbl, scores):
    lbl = np.asarray(lbl, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(set(lbl.tolist())) < 2 or np.all(scores == scores[0]):
        return 0.5
    p, pl, ph = boot_auc(lbl, scores)
    n, nl, nh = boot_auc(lbl, -scores)
    return p if p >= n else n

def run_evaluation(all_cells, feature_config=None):
    if feature_config is None:
        feature_config = {}
        
    aucs = []
    cell_types = []
    
    for fname, ck, fd, lb in all_cells:
        cur_fd = copy.deepcopy(fd)
        cur_signs = FEATURE_SIGNS.copy()
        
        for f, strat in feature_config.items():
            if f in cur_fd and strat != 'Baseline':
                x = np.array(cur_fd[f])
                t_x = None
                
                if 'Squared' in strat:
                    t_x = x ** 2
                    t_sign = cur_signs.get(f, -1)
                elif 'Dist-Median' in strat:
                    t_x = np.abs(x - np.median(x))
                    t_sign = -1
                elif 'Rank' in strat:
                    ranks = rankdata(x)
                    pct = (ranks - 0.5) / len(x)
                    pct = np.clip(pct, 1e-6, 1 - 1e-6)
                    t_x = np.abs(norm.ppf(pct))
                    t_sign = -1
                    
                if t_x is not None:
                    if 'Replace' in strat:
                        del cur_fd[f]
                    cur_fd[f + '_t'] = t_x
                    cur_signs[f + '_t'] = t_sign

        feat_names = [fn for fn in list(cur_signs.keys()) if fn in cur_fd]
        if len(feat_names) < 3:
            continue
            
        scores, _ = lsml_continuous_pipeline(cur_fd, feat_names, cur_signs)
        auc = safe_auc(lb, scores)
        aucs.append(auc)
        
        is_prob = 'math500' in fname or 'qa' in fname
        cell_types.append('Problematic' if is_prob else 'Other')
        
    return np.mean(aucs), np.mean([a for a, t in zip(aucs, cell_types) if t == 'Problematic'])

def main():
    data_dir = os.path.join(REPO_DIR, 'local_cache')
    all_cells = gather_all_cells(data_dir)
    print(f"Loaded {len(all_cells)} cells.")
    
    strategies = [
        'Baseline',
        'Add Squared', 'Replace Squared',
        'Add Dist-Median', 'Replace Dist-Median',
        'Add Rank', 'Replace Rank'
    ]
    
    base_macro, base_prob = run_evaluation(all_cells, {})
    print(f"Baseline -> Macro: {base_macro:.4f}, Prob: {base_prob:.4f}")
    
    results = []
    best_config = {}
    
    for f in NON_MONOTONE_FEATURES:
        best_strat = 'Baseline'
        best_prob_auc = base_prob
        print(f"\nEvaluating {f}...")
        for strat in strategies:
            if strat == 'Baseline':
                continue
            config = {f: strat}
            macro, prob = run_evaluation(all_cells, config)
            print(f"  {strat:20s}: Macro={macro:.4f}, Prob={prob:.4f}")
            results.append({'Feature': f, 'Strategy': strat, 'Macro': macro, 'Prob': prob})
            
            if prob > best_prob_auc:
                best_prob_auc = prob
                best_strat = strat
                
        best_config[f] = best_strat
        print(f"  -> Best for {f}: {best_strat} (Prob: {best_prob_auc:.4f})")
        
    opt_macro, opt_prob = run_evaluation(all_cells, best_config)
    print(f"\nOptimized Configuration:")
    for f, strat in best_config.items():
        print(f"  {f}: {strat}")
    print(f"Result -> Macro: {opt_macro:.4f}, Prob: {opt_prob:.4f}")
    
    global_sq_macro, global_sq_prob = run_evaluation(all_cells, {f: 'Add Squared' for f in NON_MONOTONE_FEATURES})
    
    df = pd.DataFrame(results)
    df['Delta Prob'] = df['Prob'] - base_prob
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap of deltas
    pivot = df.pivot(index='Feature', columns='Strategy', values='Delta Prob')
    im = axes[0].imshow(pivot, cmap='RdBu', aspect='auto', origin='upper')
    
    # Add colorbar
    cbar = axes[0].figure.colorbar(im, ax=axes[0])
    
    # Set ticks and labels
    axes[0].set_xticks(np.arange(len(pivot.columns)))
    axes[0].set_yticks(np.arange(len(pivot.index)))
    axes[0].set_xticklabels(pivot.columns, rotation=45, ha="right")
    axes[0].set_yticklabels(pivot.index)
    
    # Annotate text
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            color = "white" if abs(val) > np.max(np.abs(pivot.values))/2 else "black"
            axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", color=color)
            
    axes[0].set_title('Δ AUROC on Problematic Cells vs Baseline')
    
    names = ['Baseline', 'Global Add Squared', 'Optimized Set']
    macro_vals = [base_macro, global_sq_macro, opt_macro]
    prob_vals = [base_prob, global_sq_prob, opt_prob]
    
    x = np.arange(3)
    width = 0.35
    axes[1].bar(x - width/2, macro_vals, width, label='Macro Avg')
    axes[1].bar(x + width/2, prob_vals, width, label='Problematic Cells')
    
    axes[1].set_ylabel('AUROC')
    axes[1].set_title('Overall Performance Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].legend()
    axes[1].set_ylim([0.5, 0.75])
    
    plt.tight_layout()
    out_path = os.path.join(REPO_DIR, 'results', 'feature_transform_sweep.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    main()
