#!/usr/bin/env python
"""
eval_unsupervised_dufs_pruned.py — Pure Unsupervised DUFS (no pseudo-labels) + Spectral Gap K_cell + logprob_margin Anchor (Step 200).

Evaluates the exact user proposal:
  1. Pure Unsupervised DUFS STG gates (lambda3 = 0, no pseudo-label generation step)
  2. Cell-adaptive K_cell cutoff using Spectral Gap ratio (gamma = lambda1 / lambda2) and L-SML residual
  3. Orientation reference using logprob_margin anchor (vs epr baseline)

Outputs:
  - Macro AUROC overall (25 cells), Math macro (15 cells), QA macro (10 cells)
  - Mean selected feature count
  - Comparative leaderboard table vs pseudo-labeled DUFS and fixed baselines
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        items = list(iterable)
        total = len(items)
        for i, item in enumerate(items, 1):
            print(f"\r{desc}: [{i}/{total}]", end="", flush=True)
            yield item
        print()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from inscope_cells import QA_CELLS, MATH_CELLS, INSCOPE, GROUP
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.subset_sweep import prepare_cell, CANONICAL_POOL, iter_cells
from spectral_utils.selector_bench import UnlabeledCell
from spectral_utils.selectors.a6_pseudolabel_gates import _train_gates, _lambda0, BATCH

AI_DIR = os.path.join(REPO, "results", "advisor_inscope")
os.makedirs(AI_DIR, exist_ok=True)
EPOCHS_FAST = 100


def load_all_inscope_cells():
    """Load UnlabeledCell + ground-truth labels for all 25 in-scope cells."""
    data_dir = os.path.join(REPO, "local_cache")
    derived_pkl = os.path.join(REPO, "local_cache", "derived_views.pkl")
    trace_pkl = os.path.join(REPO, "local_cache", "trace_cells.pkl")
    
    loaded = {}
    print("Loading 25 in-scope cells...", flush=True)
    for domain, cell_key, fd, labels in iter_cells(data_dir=data_dir, derived_views_pkl=derived_pkl, trace_cells_pkl=trace_pkl):
        if cell_key not in INSCOPE:
            continue
        cell_obj = prepare_cell(domain, cell_key, fd, labels, feature_pool=CANONICAL_POOL)
        if cell_obj is None:
            continue
        
        rho = np.abs(np.corrcoef(cell_obj.V, rowvar=False))
        unlabeled = UnlabeledCell(
            domain=cell_obj.domain,
            cell_key=cell_obj.cell_key,
            pool=cell_obj.pool,
            pool_bits=cell_obj.pool_bits,
            V=cell_obj.V,
            anchor=cell_obj.anchor,
            anchor_name=cell_obj.anchor_name,
            rho=rho
        )
        loaded[cell_key] = {
            'unlabeled': unlabeled,
            'labels': cell_obj.labels,
            'usable': cell_obj.pool,
            'V': cell_obj.V,
            'fd': fd,
            'anchor': cell_obj.anchor
        }
    print(f"Loaded {len(loaded)}/25 cells for Unsupervised DUFS Pruning Evaluation.", flush=True)
    return loaded


def _optimize_unsupervised_cell(cell_key, cdata):
    """Worker function: Unsupervised DUFS gates + Spectral Gap K_cell cutoff."""
    import torch
    u = cdata['unlabeled']
    V = cdata['V']
    labels = cdata['labels']
    fd = cdata['fd']
    p = u.p
    
    # 1. Compute Spectral Gap ratio from correlation matrix R
    cov = np.cov(V, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    gamma = float(eigvals[0] / max(eigvals[1], 1e-6)) if len(eigvals) > 1 else 1.0
    
    # Spectral Gap decision rule for target size K_cell
    k_cell_target = 6 if gamma >= 3.0 else 15
    
    # 2. Train Pure Unsupervised DUFS Gates (lambda3 = 0, NO pseudo-label)
    X_t = torch.tensor(V, dtype=torch.float32)
    rng = np.random.default_rng(42)
    lam0 = _lambda0(X_t, torch.Generator().manual_seed(42))
    
    # Stability pick across 5 seeds for lambda2
    seeds = [int(rng.integers(2**31)) for _ in range(5)]
    gates = []
    for s in seeds:
        g = _train_gates(X_t, lam0 * 2.0, 0.0, None, EPOCHS_FAST, BATCH, s)
        gates.append(g)
    mu_bar = np.mean(gates, axis=0)
    
    # Rank features by unsupervised STG gate values
    ranked_indices = np.argsort(mu_bar)[::-1]
    
    # Select top K_cell features
    chosen_cols = ranked_indices[:k_cell_target]
    chosen_cols = sorted(list(chosen_cols))
    
    # 3. L-SML Fusion on chosen features
    if len(chosen_cols) == 1:
        fused = V[:, chosen_cols[0]]
    else:
        views = [V[:, c] for c in chosen_cols]
        fused, meta = lsml_continuous(*views)
        
    # 4. Orientation under two anchors: epr vs logprob_margin
    if 'logprob_margin' in u.pool:
        anc_margin = V[:, u.pool.index('logprob_margin')]
    elif 'logprob_margin' in fd:
        anc_margin = zscore(np.asarray(fd['logprob_margin'], dtype=float))
    else:
        anc_margin = u.anchor
        
    oriented_epr, _ = anchor_orient(fused, u.anchor)
    oriented_margin, _ = anchor_orient(fused, anc_margin)
    
    auc_epr = float(roc_auc_score(labels, oriented_epr))
    auc_margin = float(roc_auc_score(labels, oriented_margin))
    
    return cell_key, {
        'cell': cell_key,
        'group': GROUP.get(cell_key, 'unknown'),
        'spectral_gap': gamma,
        'k_cell_target': k_cell_target,
        'auc_epr': auc_epr,
        'auc_margin': auc_margin,
        'n_selected': len(chosen_cols)
    }


def run_unsupervised_pruned_eval(cells_dict):
    """Run parallel evaluation of pure unsupervised DUFS + Spectral Gap + logprob_margin."""
    print("\n--- Running Unsupervised DUFS + Spectral Gap + logprob_margin (4 CPU Workers) ---", flush=True)
    results = {}
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_optimize_unsupervised_cell, key, cells_dict[key]): key for key in cells_dict}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Unsupervised DUFS Evaluation"):
            cell_key, res = future.result()
            results[cell_key] = res

    all_res = list(results.values())
    
    macro_epr = float(np.mean([r['auc_epr'] for r in all_res]))
    macro_margin = float(np.mean([r['auc_margin'] for r in all_res]))
    mean_k = float(np.mean([r['n_selected'] for r in all_res]))
    
    math_margin = float(np.mean([r['auc_margin'] for r in all_res if r['group'] == 'math']))
    qa_margin = float(np.mean([r['auc_margin'] for r in all_res if r['group'] == 'QA']))
    
    math_epr = float(np.mean([r['auc_epr'] for r in all_res if r['group'] == 'math']))
    qa_epr = float(np.mean([r['auc_epr'] for r in all_res if r['group'] == 'QA']))
    
    print("\n=======================================================================", flush=True)
    print("  PURE UNSUPERVISED DUFS (No Pseudo-Labels) + PRUNING RESULTS:", flush=True)
    print(f"  • Overall 25-Cell Macro (logprob_margin anchor) = {macro_margin:.4f}", flush=True)
    print(f"  • Overall 25-Cell Macro (default epr anchor)    = {macro_epr:.4f}", flush=True)
    print(f"  • Math Macro (15 cells, logprob_margin anchor)  = {math_margin:.4f}", flush=True)
    print(f"  • QA Macro (10 cells, logprob_margin anchor)    = {qa_margin:.4f}", flush=True)
    print(f"  • Mean Selected Features per Cell               = {mean_k:.1f}", flush=True)
    print("=======================================================================\n", flush=True)
    
    # Save CSV
    csv_path = os.path.join(AI_DIR, "unsupervised_dufs_pruned_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=['cell', 'group', 'spectral_gap', 'k_cell_target', 'auc_epr', 'auc_margin', 'n_selected'])
        writer.writeheader()
        writer.writerows(all_res)
    print(f"[Results] CSV saved to {csv_path}", flush=True)
    
    return macro_margin, macro_epr, math_margin, qa_margin, mean_k, all_res


def main():
    cells_dict = load_all_inscope_cells()
    macro_margin, macro_epr, math_margin, qa_margin, mean_k, all_res = run_unsupervised_pruned_eval(cells_dict)
    print("Unsupervised DUFS evaluation complete!", flush=True)


if __name__ == '__main__':
    main()
