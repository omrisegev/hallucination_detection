#!/usr/bin/env python
"""In-scope (QA + math) per-feature orientation audit — Step 192.

Step 191 ran this audit on the out-of-scope RAG/GPQA cells and found ~10pp of
RAG's deficit was pure global-sign error (Step-187 domain-dependent polarity).
Omri's Jul-20 scope call demoted the Step-187 offline sign-fix from "headline
next win" to a targeted check: it only matters if the IN-SCOPE QA/math cells
also show systematic anti-orientation. This script answers that.

For every in-scope cell it computes, over the wide (c46) pool, the oriented
AUROC of each feature exactly as the fusion consumes it (ctx.V is the
sign-oriented z-scored matrix built by prepare_cell using ALL_SIGNS). A feature
with oriented AUROC < 0.5 is anti-oriented: informative but carried with the
wrong fixed offline sign on this cell. Splits QA vs math and writes a per-cell
per-feature CSV plus a per-feature summary.

CPU-only, seconds. Read-only w.r.t. all caches.
"""
import os
import sys
import csv

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.selector_bench import iter_prepared_cells  # noqa: E402

# The 25 in-scope cells (QA + reasoning/math). RAG + GPQA + trace_gpqa excluded.
QA_CELLS = {
    'epr_triviaqa_mistral24b', 'inside_coqa_llama7b', 'losnet_hotpotqa_mistral7b',
    'sciq_llama8b', 'se_nq_open_llama8b', 'se_squad_v2_llama8b',
    'seiclr_triviaqa_opt30b', 'semenergy_triviaqa_qwen3_8b',
    'spilled_triviaqa_llama8b', 'truthfulqa_llama8b',
}
MATH_CELLS = {
    'ars_gsm8k_r1distill8b', 'internalstates_gsm8k_qwen25_7b',
    'lapeigvals_gsm8k_llama3b', 'lapeigvals_gsm8k_llama8b',
    'lapeigvals_gsm8k_mistral24b', 'lapeigvals_gsm8k_nemo',
    'lapeigvals_gsm8k_phi35', 'noise_gsm8k_mistral7b', 'noise_gsm8k_phi3mini',
    'math500_dsmath7b', 'math500_qwenmath7b', 'math500_r1distill8b',
    'math500_r1distill8b_mn4096', 'trace_gsm8k_llama8b_k10',
    'trace_math500_qwenmath15b_k10',
}
INSCOPE = QA_CELLS | MATH_CELLS
OUT_DIR = os.path.join(REPO_DIR, 'results', 'selector_bench')


def group_of(cell):
    if cell in QA_CELLS:
        return 'QA'
    if cell in MATH_CELLS:
        return 'math'
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per_cell = []   # long rows: cell, group, feature, oriented_auroc, n
    seen = set()
    for ctx in iter_prepared_cells(REPO_DIR, pool_mode='c46', domains=['repgrid']):
        if ctx.cell_key not in INSCOPE:
            continue
        seen.add(ctx.cell_key)
        grp = group_of(ctx.cell_key)
        for j, f in enumerate(ctx.pool):
            auc = roc_auc_score(ctx.labels, ctx.V[:, j])
            per_cell.append(dict(cell=ctx.cell_key, group=grp, feature=f,
                                 oriented_auroc=round(float(auc), 4),
                                 n=int(ctx.V.shape[0])))

    missing = INSCOPE - seen
    if missing:
        print(f"[warn] {len(missing)} in-scope cells not iterated: {sorted(missing)}")

    long = pd.DataFrame(per_cell)
    long_path = os.path.join(OUT_DIR, 'inscope_feature_orientation.csv')
    long.to_csv(long_path, index=False)

    # Per-feature summary, split by group and overall.
    recs = []
    for f, sub in long.groupby('feature'):
        rec = {'feature': f, 'n_cells': len(sub)}
        for grp in ('QA', 'math', 'all'):
            g = sub if grp == 'all' else sub[sub.group == grp]
            v = g['oriented_auroc']
            if len(v):
                rec[f'{grp}_mean'] = round(v.mean(), 4)
                rec[f'{grp}_n_anti'] = int((v < 0.5).sum())
                rec[f'{grp}_n'] = len(v)
            else:
                rec[f'{grp}_mean'] = np.nan
                rec[f'{grp}_n_anti'] = 0
                rec[f'{grp}_n'] = 0
        recs.append(rec)
    summ = pd.DataFrame(recs).sort_values('all_mean', ascending=False)
    summ_path = os.path.join(OUT_DIR, 'inscope_feature_orientation_summary.csv')
    summ.to_csv(summ_path, index=False)

    # Console verdict.
    def anti_share(df):
        return float((df['oriented_auroc'] < 0.5).mean())

    print(f"\nCells iterated: {len(seen)}/{len(INSCOPE)}  "
          f"(QA {len(seen & QA_CELLS)}, math {len(seen & MATH_CELLS)})")
    print(f"Total (cell,feature) pairs: {len(long)}")
    print(f"\nAnti-oriented share (oriented AUROC < 0.5):")
    print(f"  QA   : {anti_share(long[long.group=='QA']):.3f}")
    print(f"  math : {anti_share(long[long.group=='math']):.3f}")
    print(f"  all  : {anti_share(long):.3f}")

    print(f"\nPer-cell anti-oriented feature count (of pool size):")
    for cell, sub in long.groupby('cell'):
        n_anti = int((sub['oriented_auroc'] < 0.5).sum())
        print(f"  [{group_of(cell):4s}] {cell:38s} {n_anti:2d}/{len(sub):2d} anti")

    # Features consistently anti-oriented on QA (the Step-187 candidates).
    qa_anti = summ[(summ['QA_mean'] < 0.5) & (summ['QA_n'] >= 5)]
    math_anti = summ[(summ['math_mean'] < 0.5) & (summ['math_n'] >= 5)]
    print(f"\nFeatures with QA mean oriented AUROC < 0.5 (>=5 cells): "
          f"{list(qa_anti['feature'])}")
    print(f"Features with math mean oriented AUROC < 0.5 (>=5 cells): "
          f"{list(math_anti['feature'])}")
    print(f"\nWrote {long_path}\n      {summ_path}")


if __name__ == '__main__':
    main()
