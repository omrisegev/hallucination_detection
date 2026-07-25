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

from inscope_cells import GROUP
from spectral_utils.fusion_utils import lsml_continuous
from spectral_utils.selectors.adaptive_k import predict_k, raw_k

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
    """Step 201 rewrite (defects 8 + 9).

    The previous version had four problems that made every number here unusable:
      1. pool built from `fd.keys()` rather than CANONICAL_POOL;
      2. orientation anchored on `V[:, 0]` / `sub[:, 0]` — an arbitrary column;
      3. `max(auc, 1 - auc)` applied to EVERY arm — a label-peeking sign oracle,
         so none of these arms were label-free and all were floored at 0.5;
      4. undocumented `1e-6 * randn` jitter added to the features.
    Together these produced a GOOD_6 reference of 0.7273 instead of the canonical
    0.7594 (differing on 25/25 cells), i.e. every comparison was against a
    mis-computed baseline.

    Now: canonical loading + scoring via `inscope_bench_common`, raw AUROC, and
    the GOOD_6 validity anchor enforced before anything is reported.
    """
    from inscope_bench_common import (load_cells, score_cols, good6_score,
                                      assert_good6)
    from scipy.stats import wilcoxon

    cells = load_cells()
    ok, macro_g6 = assert_good6(cells)
    if not ok:
        print("FAIL — refusing to report (SPEC_gap_ladder §8).")
        sys.exit(1)

    results = []
    for cell_key, cell in sorted(cells.items()):
        V, y = cell["V"], cell["labels"]
        p = V.shape[1]

        # full-pool consensus as the pseudo-label target (sign is a gauge for
        # L-SML, so no Z2 pre-signing is applied — it provably cannot matter)
        fused_full, _ = lsml_continuous(*[V[:, c] for c in range(p)])
        auc_full_30 = score_cols(cell, range(p))

        k_eff = predict_k(V, list(range(p)), rule='eff_rank')
        k_mp = predict_k(V, list(range(p)), rule='mp_floor')
        raw_eff = raw_k(V, list(range(p)), 'eff_rank')
        raw_mp = raw_k(V, list(range(p)), 'mp_floor')

        sel_eff = run_mrmr_selection(V, fused_full, k_eff)
        sel_mp = run_mrmr_selection(V, fused_full, k_mp)

        results.append({
            'cell': cell_key,
            'group': GROUP.get(cell_key, '?'),
            'k_eff': k_eff, 'k_mp': k_mp,
            'raw_eff_rank': round(float(raw_eff), 3),
            'raw_mp_floor': round(float(raw_mp), 3),
            'auc_full_30': auc_full_30,
            'auc_mrmr_eff': score_cols(cell, sel_eff),
            'auc_mrmr_mp': score_cols(cell, sel_mp),
            'auc_good6': good6_score(cell),
        })
        print(f"  {cell_key:34s} full {auc_full_30:.4f} | "
              f"eff(K={k_eff}) {results[-1]['auc_mrmr_eff']:.4f} | "
              f"mp(K={k_mp}) {results[-1]['auc_mrmr_mp']:.4f} | "
              f"good6 {results[-1]['auc_good6']:.4f}", flush=True)

    df = pd.DataFrame(results)
    out_csv = os.path.join(REPO, "results", "advisor_inscope",
                           "user_pipeline_results.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== User-proposed pipeline (canonical scoring, raw AUROC) ===")
    print(f"cells: {len(df)}   GOOD_6 macro: {macro_g6:.4f} (canonical 0.7594)")
    print(f"k_eff distribution : {df['k_eff'].value_counts().to_dict()}  "
          f"(raw eff_rank median {df['raw_eff_rank'].median():.2f})")
    print(f"k_mp  distribution : {df['k_mp'].value_counts().to_dict()}  "
          f"(raw mp_floor median {df['raw_mp_floor'].median():.2f})")
    base = df['auc_good6'].to_numpy()
    print(f"\n{'arm':16s} {'macro':>8s} {'vs GOOD_6':>11s} {'W/L':>8s} {'p':>9s}")
    print("-" * 60)
    for arm in ('auc_full_30', 'auc_mrmr_eff', 'auc_mrmr_mp', 'auc_good6'):
        v = df[arm].to_numpy()
        d = v - base
        try:
            pv = float(wilcoxon(v, base).pvalue) if np.any(d != 0) else float('nan')
        except Exception:
            pv = float('nan')
        print(f"{arm:16s} {v.mean():8.4f} {d.mean()*100:+10.2f}pp "
              f"{int((d>0).sum()):3d}/{int((d<0).sum()):<3d} {pv:9.4f}")
    print("\nSaved detailed results to:", out_csv)

if __name__ == '__main__':
    main()
