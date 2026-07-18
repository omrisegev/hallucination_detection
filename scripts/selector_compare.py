#!/usr/bin/env python
"""
selector_compare — Stage-3 comparison of all benched selectors (Step 186).

Merges every results/selector_bench/*__{h16,c46}.csv with the per-cell
comparators (GOOD_5 + label-peeking oracle from sweep_summary.csv, LOCO from
loco.csv, the size-5 random median from landscape.csv) and emits:

  results/selector_bench/comparison.csv          leaderboard (summarize_bench)
  results/selector_bench/baselines.csv           per-cell comparator table
  docs/research_notes/selector_bench_results.md  auto-generated research note

Every number is produced by spectral_utils.selector_bench.summarize_bench —
none hand-typed (Step-184 discipline). Baselines rendered alongside:
GOOD_5 (fixed macro), LOCO (honest cross-cell selection), random-median
(size-5), oracle (label-peeking ceiling — a ceiling, never a result).
"""

import argparse
import glob
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.selector_bench import summarize_bench   # noqa: E402


def load_comparators(sweep_dir):
    import pandas as pd
    ss = pd.read_csv(os.path.join(sweep_dir, 'sweep_summary.csv'))
    comp = ss.rename(columns={'cell_key': 'cell',
                              'best_auroc': 'oracle_auroc'})[
        ['domain', 'cell', 'n', 'pos_rate', 'good5_auroc', 'oracle_auroc',
         'all16_auroc', 'epr_auroc']].copy()
    loco_path = os.path.join(sweep_dir, 'loco.csv')
    if os.path.exists(loco_path):
        loco = pd.read_csv(loco_path).rename(columns={'loco_auroc': 'loco'})
        comp = comp.merge(loco[['domain', 'cell', 'loco']],
                          on=['domain', 'cell'], how='left')
    land_path = os.path.join(sweep_dir, 'landscape.csv')
    if os.path.exists(land_path):
        land = pd.read_csv(land_path)
        rnd = (land[land['size'] == 5]
               .rename(columns={'p50': 'random_median_s5'})
               [['domain', 'cell', 'random_median_s5']])
        comp = comp.merge(rnd, on=['domain', 'cell'], how='left')
    return comp


def baseline_macros(comp):
    """Macro rows for the fixed comparators, on the same 51-cell basis."""
    rows = []
    for name, col in [('GOOD_5 (fixed)', 'good5_auroc'),
                      ('ALL_H16 (fixed)', 'all16_auroc'),
                      ('epr single', 'epr_auroc'),
                      ('LOCO (honest transfer)', 'loco'),
                      ('random median (size 5)', 'random_median_s5'),
                      ('oracle (LABEL-PEEKING CEILING)', 'oracle_auroc')]:
        if col not in comp.columns:
            continue
        v = comp[col].astype(float)
        rg = comp['domain'].isin(['rag', 'gpqa'])
        rep = comp['domain'] == 'repgrid'
        rows.append({
            'baseline': name, 'n_cells': int(v.notna().sum()),
            'macro_auroc': round(float(v.mean()), 4),
            'macro_repgrid': round(float(v[rep].mean()), 4) if rep.any() else np.nan,
            'macro_rag_gpqa': round(float(v[rg].mean()), 4) if rg.any() else np.nan,
        })
    import pandas as pd
    return pd.DataFrame(rows)


def epr_alone_rows(comp):
    """Synthetic bench-shaped rows for 'epr.alone' — the raw single-feature
    AUROC (L-SML needs >=3 views to run fusion at all, so this is scored
    directly, not through eval_subset_flex). Derived from comp['epr_auroc']
    (already computed by the Step-153 exhaustive sweep — never hand-typed),
    so it appears as a first-class LEADERBOARD row next to the learned/
    curated subsets instead of only in the separate baselines table above it
    — Step-186/187 found several learned selectors are statistically
    indistinguishable from this single feature, which only reads clearly
    when it sits in the same table. c46 rows are repgrid-only (the 46-view
    pool doesn't exist elsewhere), matching every other c46 leaderboard row.
    """
    import pandas as pd
    base = dict(selector='epr_alone', variant='epr.alone', size=1, fusion='none',
               flipped=False, fallback=False, seconds=0.0, seed=0, K=-1,
               p_pool=1, residual=float('nan'), chosen='epr', eval_mode='raw',
               pctile_within_size=float('nan'), rand_med=float('nan'),
               diag_json='{}')
    rows = []
    for _, r in comp.iterrows():
        if not np.isfinite(r['epr_auroc']):
            continue
        common = dict(base, domain=r['domain'], cell=r['cell'], n=int(r['n']),
                      auroc=float(r['epr_auroc']))
        rows.append(dict(common, pool_mode='h16'))
        if r['domain'] == 'repgrid':
            rows.append(dict(common, pool_mode='c46'))
    return pd.DataFrame(rows)


def _md_table(df):
    """Minimal DataFrame->GitHub-markdown (avoids the tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |',
             '| ' + ' | '.join('---' for _ in cols) + ' |']
    for _, row in df.iterrows():
        cells = ['' if v is None or (isinstance(v, float) and np.isnan(v))
                 else str(v) for v in row.tolist()]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def render_note(leader, bases, out_md, bench_dir):
    import pandas as pd
    lines = [
        '# Selector Bench — Results (auto-generated)',
        '',
        '**Generated by**: `python scripts/selector_compare.py` — regenerate, never hand-edit.',
        '**Protocol**: docs/research_notes plan of Step 186; pre-registered metrics/gates in',
        '`spectral_utils/selector_bench.py::summarize_bench`. All AUROCs raw, anchor-oriented,',
        'label-free selection; NaN AUROC counts as a loss; fallback rows flagged, never dropped.',
        '',
        '## Gate definitions',
        '- **G-floor**: mean exact percentile-within-size > 50 AND above the random median on >50% of cells.',
        '- **G-macro**: PASS = macro ≥ GOOD_5 − 0.5pp; SUCCESS = > GOOD_5 + 1pp with Wilcoxon p < 0.05.',
        '- **G-domain**: ≥ 25% of the RAG+GPQA (oracle − GOOD_5) gap captured.',
        '',
        '## Baselines (same cell basis)',
        '',
        _md_table(bases),
        '',
        '## Leaderboard',
        '',
        _md_table(leader),
        '',
    ]
    adm_path = os.path.join(bench_dir, 'admissibility_summary.csv')
    if os.path.exists(adm_path):
        adm = pd.read_csv(adm_path)
        lines += ['## Objective admissibility (A1.0, pre-registered)', '',
                  _md_table(adm[adm["domain"] == "ALL"]), '',
                  'Per-domain detail: `results/selector_bench/admissibility_summary.csv`.', '']
    rt_path = os.path.join(bench_dir, 'admissibility_router_summary.csv')
    if os.path.exists(rt_path):
        rt = pd.read_csv(rt_path)
        lines += ['## Structural-model router validity (memo §2.4)', '',
                  _md_table(rt), '']
    sh_path = os.path.join(bench_dir, 'splithalf_oracle_summary.csv')
    if os.path.exists(sh_path):
        sh = pd.read_csv(sh_path)
        macro = sh[['greedy_halfA', 'greedy_halfB', 'good5_halfB',
                    'fulloracle_halfB', 'fulloracle_insample',
                    'optimism_gap']].mean().round(4)
        lines += [
            '## Split-half honest oracle (Step 189)', '',
            'Addresses the winner\'s-curse flag on the exhaustive-sweep oracle: '
            'that number is best-of-up-to-65k picked by FULL-DATASET AUROC. This '
            'is the out-of-sample counterpart — bounded greedy forward search '
            '(H16 pool, sizes 3-6, eigengap K-selection) on held-out half A '
            'ONLY, refit and scored on half B; R=10 random 50/50 splits/cell, '
            f'macro over {len(sh)} cells with n>=40.', '',
            f'| metric | macro (mean over cells of per-split mean) |',
            f'| --- | --- |',
            f'| greedy search, in-sample (half A, label-peeking within the half) | {macro["greedy_halfA"]} |',
            f'| **greedy search, held out (half B) — the honest ceiling** | **{macro["greedy_halfB"]}** |',
            f'| GOOD_5, held out (half B, same split) | {macro["good5_halfB"]} |',
            f'| full-data exhaustive-sweep oracle subset, refit + scored on held-out half B | {macro["fulloracle_halfB"]} |',
            f'| full-data exhaustive-sweep oracle, in-sample (reference — the 0.7472-macro number) | {macro["fulloracle_insample"]} |',
            f'| optimism gap (half A in-sample minus half B held-out, greedy search) | {macro["optimism_gap"]} |',
            '',
            'Per-cell detail: `results/selector_bench/splithalf_oracle_summary.csv`. '
            'Full per-split rows: `results/selector_bench/splithalf_oracle.csv`.', '',
        ]
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bench-dir',
                    default=os.path.join(REPO_DIR, 'results', 'selector_bench'))
    ap.add_argument('--sweep-dir',
                    default=os.path.join(
                        os.environ.get('HD_DATA_ROOT', REPO_DIR),
                        'results', 'subset_sweep'))
    ap.add_argument('--out-md',
                    default=os.path.join(REPO_DIR, 'docs', 'research_notes',
                                         'selector_bench_results.md'))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.bench_dir, '*__h16.csv')) +
                   glob.glob(os.path.join(args.bench_dir, '*__c46.csv')))
    if not files:
        sys.exit(f"no bench CSVs under {args.bench_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} bench rows from {len(files)} files")

    comp = load_comparators(args.sweep_dir)
    epr_rows = epr_alone_rows(comp)
    df = pd.concat([df, epr_rows], ignore_index=True)
    print(f"added {len(epr_rows)} epr.alone rows (h16 51-cell + c46 repgrid-19)")

    leader = summarize_bench(df, comp[['domain', 'cell', 'good5_auroc',
                                       'oracle_auroc']])
    bases = baseline_macros(comp)

    leader.to_csv(os.path.join(args.bench_dir, 'comparison.csv'), index=False)
    comp.to_csv(os.path.join(args.bench_dir, 'baselines.csv'), index=False)
    render_note(leader, bases, args.out_md, args.bench_dir)

    print('\n=== baselines ===')
    print(bases.to_string(index=False))
    print('\n=== leaderboard (top 15) ===')
    show = ['variant', 'pool_mode', 'n_cells', 'macro_auroc', 'delta_vs_good5',
            'mean_pctile', 'gap_captured_rag_gpqa', 'G_floor', 'G_macro', 'G_domain']
    print(leader[show].head(15).to_string(index=False))
    print(f"\nwrote comparison.csv, baselines.csv, {args.out_md}")


if __name__ == '__main__':
    main()
