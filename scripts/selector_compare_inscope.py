#!/usr/bin/env python
"""In-scope (QA + math) selector leaderboard — Step 192.

The canonical `selector_compare.py` aggregates over ALL repgrid-domain cells,
which after the Step-190 regen wave includes the now-out-of-scope RAG + GPQA
cells (Omri's Jul-20 scope call: thesis focus is QA + reasoning/math). This
script produces the same leaderboard restricted to the 25 in-scope cells, with
QA / math macro splits, WITHOUT touching the canonical comparison.csv.

Two things differ from selector_compare.py and are deliberate:
  1. The GOOD_5 (and oracle) comparator per cell is taken from the bench's own
     `ref.GOOD_5` / grid-search rows, not from sweep_summary.csv — the 6 new
     cluster cells (4 math500 + 2 trace) were never exhaustively swept, so they
     have no sweep_summary row. ref.GOOD_5 is scored on every cell by the bench.
  2. macro is reported three ways: all-25, QA-only (10), math-only (15) — the
     canonical file only has a single repgrid macro that would now be polluted.

Honest headroom (the label-peeking oracle ceiling is a winner's-curse artifact,
Step 189) comes from the split-half oracle CSVs, folded in if present.

Reuses spectral_utils.selector_bench.summarize_bench for every metric (no
hand-typed numbers, Step-184 discipline).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.selector_bench import summarize_bench   # noqa: E402

QA_CELLS = [
    'epr_triviaqa_mistral24b', 'inside_coqa_llama7b', 'losnet_hotpotqa_mistral7b',
    'sciq_llama8b', 'se_nq_open_llama8b', 'se_squad_v2_llama8b',
    'seiclr_triviaqa_opt30b', 'semenergy_triviaqa_qwen3_8b',
    'spilled_triviaqa_llama8b', 'truthfulqa_llama8b',
]
MATH_CELLS = [
    'ars_gsm8k_r1distill8b', 'internalstates_gsm8k_qwen25_7b',
    'lapeigvals_gsm8k_llama3b', 'lapeigvals_gsm8k_llama8b',
    'lapeigvals_gsm8k_mistral24b', 'lapeigvals_gsm8k_nemo',
    'lapeigvals_gsm8k_phi35', 'noise_gsm8k_mistral7b', 'noise_gsm8k_phi3mini',
    'math500_dsmath7b', 'math500_qwenmath7b', 'math500_r1distill8b',
    'math500_r1distill8b_mn4096', 'trace_gsm8k_llama8b_k10',
    'trace_math500_qwenmath15b_k10',
]
INSCOPE = QA_CELLS + MATH_CELLS


def good5_comparator(df):
    """Per-cell GOOD_5 AUROC from the bench's own ref.GOOD_5 rows (c46 pref,
    h16 fallback). oracle_auroc left NaN — the honest ceiling is the split-half
    number, not the label-peeking sweep oracle (Step 189)."""
    g5 = df[df['variant'] == 'ref.GOOD_5'].copy()
    g5['auroc'] = pd.to_numeric(g5['auroc'], errors='coerce')
    # prefer c46 GOOD_5 (varentropy-free GOOD_5 is identical across pools for
    # the shared 5 features; c46 exists on all in-scope cells)
    rows = {}
    for _, r in g5.iterrows():
        key = (r['domain'], r['cell'])
        if key not in rows or r['pool_mode'] == 'c46':
            rows[key] = r['auroc']
    comp = pd.DataFrame([{'domain': d, 'cell': c, 'good5_auroc': a,
                          'oracle_auroc': np.nan}
                         for (d, c), a in rows.items()])
    return comp


def leaderboard_for(df, comp, cells):
    sub = df[df['cell'].isin(cells)].copy()
    lb = summarize_bench(sub, comp[['domain', 'cell', 'good5_auroc',
                                    'oracle_auroc']])
    return lb.set_index(['variant', 'pool_mode'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bench-dir',
                    default=os.path.join(REPO_DIR, 'results', 'selector_bench'))
    ap.add_argument('--out',
                    default=os.path.join(REPO_DIR, 'results', 'selector_bench',
                                         'comparison_inscope.csv'))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.bench_dir, '*__h16.csv')) +
                   glob.glob(os.path.join(args.bench_dir, '*__c46.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df['cell'].isin(INSCOPE)].copy()
    print(f"loaded {len(df)} in-scope bench rows from {len(files)} files")

    comp = good5_comparator(df)
    lb_all = leaderboard_for(df, comp, INSCOPE)
    lb_qa = leaderboard_for(df, comp, QA_CELLS)
    lb_math = leaderboard_for(df, comp, MATH_CELLS)

    out = lb_all[['n_cells', 'macro_auroc', 'delta_vs_good5', 'wilcoxon_p',
                  'wins', 'ties', 'losses', 'G_macro']].copy()
    out = out.rename(columns={'macro_auroc': 'macro_all',
                              'delta_vs_good5': 'delta_vs_good5_all'})
    out['macro_qa'] = lb_qa['macro_auroc']
    out['macro_math'] = lb_math['macro_auroc']
    out = out.reset_index().sort_values(['pool_mode', 'macro_all'],
                                        ascending=[True, False])
    cols = ['variant', 'pool_mode', 'n_cells', 'macro_all', 'macro_qa',
            'macro_math', 'delta_vs_good5_all', 'wilcoxon_p', 'wins', 'ties',
            'losses', 'G_macro']
    out = out[cols]
    out.to_csv(args.out, index=False)

    for pool in ('c46', 'h16'):
        p = out[out['pool_mode'] == pool]
        if not len(p):
            continue
        print(f"\n=== in-scope leaderboard ({pool}) — top 12 by macro_all ===")
        print(p.head(12).to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
