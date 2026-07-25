#!/usr/bin/env python
"""
bench_a6_pruned_fix.py — Benchmarks fixed a6.pruned_dufs across all 25 in-scope cells
and appends/updates results in results/selector_bench/a6_pseudolabel_gates__c46.csv.
"""

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


def main():
    print("Running fixed a6.pruned_dufs benchmark across 25 in-scope cells...", flush=True)
    rows = []
    rng = np.random.default_rng(42)
    
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
        
        # Run selector
        variants = a6_pseudolabel_gates(unlabeled, rng=rng)
        for vres in variants:
            if vres['variant'] == 'a6.pruned_dufs':
                # Evaluate subset using standard eval_subset_flex
                er = eval_subset_flex(cell_obj, vres['cols'])
                auroc = er['auroc']
                rows.append({
                    'cell': cell_key,
                    'group': 'math' if cell_key in MATH_CELLS else 'QA',
                    'variant': 'a6.pruned_dufs',
                    'n_selected': len(vres['cols']),
                    'auroc': auroc
                })

    df = pd.DataFrame(rows)
    print("\n=======================================================================", flush=True)
    print("  FIXED a6.pruned_dufs IN-SCOPE BENCHMARK RESULTS:", flush=True)
    macro_all = df['auroc'].mean()
    macro_math = df[df['group']=='math']['auroc'].mean()
    macro_qa = df[df['group']=='QA']['auroc'].mean()
    mean_k = df['n_selected'].mean()
    
    print(f"  • Overall 25-Cell Macro AUROC = {macro_all:.4f}", flush=True)
    print(f"  • Math Macro AUROC (15 cells)  = {macro_math:.4f}", flush=True)
    print(f"  • QA Macro AUROC (10 cells)    = {macro_qa:.4f}", flush=True)
    print(f"  • Mean Selected Features       = {mean_k:.1f}", flush=True)
    print("=======================================================================\n", flush=True)


if __name__ == '__main__':
    main()
