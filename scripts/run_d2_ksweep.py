#!/usr/bin/env python
import sys
import os
import numpy as np

repo = 'C:/Users/omris/TAU/hallucination_detection'
if repo not in sys.path:
    sys.path.insert(0, repo)
if os.path.join(repo, 'scripts') not in sys.path:
    sys.path.insert(0, os.path.join(repo, 'scripts'))

from inscope_cells import GROUP, QA_CELLS, MATH_CELLS, INSCOPE
from spectral_utils.selector_bench import eval_subset_flex
from spectral_utils.subset_sweep import GOOD_5, GOOD_6
from spectral_utils.selectors.a6_pseudolabel_gates import (
    _seed_cols, _pseudo_label, _corr_with, _plmrmr_order, MRMR_ALPHA
)
from compare_anchor_quality import load_all_inscope_cells

class Ctx:
    def __init__(self, u, labels):
        self.V = np.asarray(u.V, dtype=np.float64)
        self.anchor = u.anchor
        self.labels = labels
        self.pool = u.pool
        self.pool_bits = u.pool_bits

def main():
    cells_dict = load_all_inscope_cells()
    k_list = [5, 6, 8, 10, 12, 15, 18, 20]

    results = {k: {} for k in k_list}
    good5_res = {}
    good6_res = {}

    for ck, cdata in cells_dict.items():
        u = cdata['unlabeled']
        labels = cdata['labels']
        ctx = Ctx(u, labels)
        V = ctx.V
        p = V.shape[1]
        
        # GOOD_5 and GOOD_6
        g5_cols = [u.pool.index(f) for f in GOOD_5 if f in u.pool]
        g6_cols = [u.pool.index(f) for f in GOOD_6 if f in u.pool]
        good5_res[ck] = float(eval_subset_flex(ctx, sorted(set(g5_cols)))['auroc'])
        good6_res[ck] = float(eval_subset_flex(ctx, sorted(set(g6_cols)))['auroc'])
        
        # D2 PL-MRMR ranking
        s_cols, _ = _seed_cols(u)
        y_hat, _ = _pseudo_label(u, s_cols)
        sel = np.array([c for c in range(p) if c not in set(s_cols)], dtype=np.int64)
        if len(sel) >= 3:
            agree = _corr_with(V[:, sel], y_hat)
            order = _plmrmr_order(V[:, sel], agree, alpha=MRMR_ALPHA)
            mrank = [int(c) for c in s_cols] + [int(sel[j]) for j in order]
            
            for k in k_list:
                cols_k = mrank[:min(k, len(mrank))]
                res = float(eval_subset_flex(ctx, sorted(set(cols_k)))['auroc'])
                results[k][ck] = res

    g5_all = np.mean([good5_res[c] for c in INSCOPE])
    g5_qa = np.mean([good5_res[c] for c in QA_CELLS])
    g5_m = np.mean([good5_res[c] for c in MATH_CELLS])

    g6_all = np.mean([good6_res[c] for c in INSCOPE])
    g6_qa = np.mean([good6_res[c] for c in QA_CELLS])
    g6_m = np.mean([good6_res[c] for c in MATH_CELLS])

    print("\n" + "="*80)
    print("D2 (PL-MRMR) BUDGET K-SWEEP SUMMARY ACROSS 25 CANONICAL IN-SCOPE CELLS")
    print("="*80)
    header = f"{'Arm / Budget K':18s} | {'Overall Macro':13s} | {'QA Macro':10s} | {'Math Macro':10s} | {'vs GOOD_5':10s} | {'vs GOOD_6':10s}"
    print(header)
    print("-" * len(header))

    print(f"{'ref.GOOD_5 (K=5)':18s} | {g5_all:13.4f} | {g5_qa:10.4f} | {g5_m:10.4f} | {'—':10s} | {g5_all-g6_all:+10.4f}")
    print(f"{'ref.GOOD_6 (K=6)':18s} | {g6_all:13.4f} | {g6_qa:10.4f} | {g6_m:10.4f} | {g6_all-g5_all:+10.4f} | {'—':10s}")
    print("-" * len(header))

    for k in k_list:
        all_m = np.mean([results[k][c] for c in INSCOPE])
        qa_m = np.mean([results[k][c] for c in QA_CELLS])
        math_m = np.mean([results[k][c] for c in MATH_CELLS])
        d_g5 = all_m - g5_all
        d_g6 = all_m - g6_all
        print(f"D2 (K={k:2d})           | {all_m:13.4f} | {qa_m:10.4f} | {math_m:10.4f} | {d_g5:+10.4f} | {d_g6:+10.4f}")

    print("="*80 + "\n")

    # Domain specific best budget combinations
    best_qa_k = max(k_list, key=lambda k: np.mean([results[k][c] for c in QA_CELLS]))
    best_math_k = max(k_list, key=lambda k: np.mean([results[k][c] for c in MATH_CELLS]))
    best_overall_k = max(k_list, key=lambda k: np.mean([results[k][c] for c in INSCOPE]))

    print(f"Best K for QA alone:     K={best_qa_k}  (QA Macro = {np.mean([results[best_qa_k][c] for c in QA_CELLS]):.4f})")
    print(f"Best K for Math alone:   K={best_math_k}  (Math Macro = {np.mean([results[best_math_k][c] for c in MATH_CELLS]):.4f})")
    print(f"Best K Overall:          K={best_overall_k}  (Overall Macro = {np.mean([results[best_overall_k][c] for c in INSCOPE]):.4f})")
    
    # Domain-specific budget rule (e.g. K=8 for QA, K=12 for Math)
    domain_adapted = [results[8][c] if GROUP[c]=='QA' else results[12][c] for c in INSCOPE]
    print(f"\nDomain-Split Rule (QA K=8, Math K=12): Overall Macro = {np.mean(domain_adapted):.4f} | QA = {np.mean([results[8][c] for c in QA_CELLS]):.4f} | Math = {np.mean([results[12][c] for c in MATH_CELLS]):.4f}")

if __name__ == '__main__':
    main()
