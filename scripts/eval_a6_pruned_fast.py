#!/usr/bin/env python
"""
eval_a6_pruned_fast.py — Parallel (4 workers) benchmark of fixed a6.pruned_dufs across 25 in-scope cells.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import pandas as pd
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from inscope_cells import INSCOPE, QA_CELLS, MATH_CELLS
from spectral_utils.subset_sweep import prepare_cell, CANONICAL_POOL, iter_cells
from spectral_utils.selector_bench import UnlabeledCell, eval_subset_flex
from spectral_utils.selectors.a6_pseudolabel_gates import a6_pseudolabel_gates

DATA_DIR = os.path.join(REPO, "local_cache")
DERIVED_PKL = os.path.join(REPO, "local_cache", "derived_views.pkl")
TRACE_PKL = os.path.join(REPO, "local_cache", "trace_cells.pkl")


def load_cells():
    loaded = {}
    for domain, cell_key, fd, labels in iter_cells(data_dir=DATA_DIR, derived_views_pkl=DERIVED_PKL, trace_cells_pkl=TRACE_PKL):
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
        loaded[cell_key] = {'unlabeled': unlabeled, 'cell_obj': cell_obj}
    return loaded


def eval_one_cell(cell_key, cdata):
    unlabeled = cdata['unlabeled']
    cell_obj = cdata['cell_obj']
    rng = np.random.default_rng(42)
    
    variants = a6_pseudolabel_gates(unlabeled, rng=rng)
    res_list = []
    for vres in variants:
        if vres['variant'] == 'a6.pruned_dufs':
            er = eval_subset_flex(cell_obj, vres['cols'])
            res_list.append({
                'cell': cell_key,
                'group': 'math' if cell_key in MATH_CELLS else 'QA',
                'variant': 'a6.pruned_dufs',
                'n_selected': len(vres['cols']),
                'auroc': er['auroc'],
                'fallback': vres['diag'].get('fallback', False)
            })
    return res_list


def main():
    cells = load_cells()
    print(f"Loaded {len(cells)} cells. Running fast 4-worker evaluation...", flush=True)
    all_rows = []
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(eval_one_cell, k, cells[k]): k for k in cells}
        for future in as_completed(futures):
            res_list = future.result()
            all_rows.extend(res_list)
            
    df = pd.DataFrame(all_rows)
    print("\n=======================================================================", flush=True)
    print("  VERIFIED POST-FIX a6.pruned_dufs BENCHMARK RESULTS:", flush=True)
    macro_all = df['auroc'].mean()
    macro_math = df[df['group']=='math']['auroc'].mean()
    macro_qa = df[df['group']=='QA']['auroc'].mean()
    mean_k = df['n_selected'].mean()
    fallbacks = df['fallback'].sum()
    
    print(f"  • Overall 25-Cell Macro AUROC = {macro_all:.4f}", flush=True)
    print(f"  • Math Macro AUROC (15 cells)  = {macro_math:.4f}", flush=True)
    print(f"  • QA Macro AUROC (10 cells)    = {macro_qa:.4f}", flush=True)
    print(f"  • Mean Selected Features       = {mean_k:.2f}", flush=True)
    print(f"  • Fallbacks Count              = {fallbacks}/25", flush=True)
    print("=======================================================================\n", flush=True)
    
    # Print per-cell table
    print(df[['cell', 'group', 'n_selected', 'auroc', 'fallback']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
