#!/usr/bin/env python
"""
selector_splithalf_oracle — honest per-cell subset-selection ceiling (Step 189).

Addresses the winner's-curse flag on the 0.7472 exhaustive-sweep oracle
(Step 186/187 notes): that number is best-of-up-to-65k picked by FULL-DATASET
AUROC, i.e. selection and evaluation share the same labels — a huge multiple-
comparisons search is guaranteed to find some subset that overfits the noise
in a single finite sample, even under the null. It should never be adopted as
"the achievable ceiling" without an honest, out-of-sample counterpart.

Method: for R random 50/50 splits per H16-pool cell,
  1. Half A ONLY: bounded greedy forward selection (continuous L-SML,
     eigengap K-selection for speed) over sizes 3..GREEDY_MAX_SIZE, picking
     the size/feature that maximizes AUROC on half A at each step. This is
     itself label-peeking, but ONLY on half A — the labels used for the
     selection decision are structurally disjoint from the labels used to
     score it.
  2. Refit the resulting subset transductively on held-out half B (residual
     K-selection, matching the project's standard method) and record its
     AUROC there. This is the honest split-half oracle number.
  3. For comparison, on the SAME half B: (a) GOOD_5 refit fresh, (b) the
     full-data exhaustive-sweep's best subset (sweep_summary.csv `best_feats`
     — chosen using ALL n, i.e. including half B's own labels) refit fresh.
     (b) is the "how much does the in-sample oracle collapse when some of
     its own selection evidence disappears" comparator, not a fully separate
     control — the honest zero-leakage number is (2).

Z-scoring is redone independently within each half (not inherited from the
full-cell prepare_cell z-scoring) so half B contributes nothing to how half A
features are standardized, and vice versa — avoids a subtler, second form of
cross-half leakage beyond the label-peeking question this script targets.

Output: results/selector_bench/splithalf_oracle.csv (one row per (cell,
split)), resume-safe; results/selector_bench/splithalf_oracle_summary.csv
(macro table); a short section appended to the research note by
selector_compare.py (read from the summary CSV, not hand-typed).
"""

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.fusion_utils import lsml_continuous, zscore   # noqa: E402
from spectral_utils.streaming_utils import anchor_orient          # noqa: E402
from spectral_utils.subset_sweep import GOOD_5                    # noqa: E402
from spectral_utils.selector_bench import iter_prepared_cells      # noqa: E402

GREEDY_MAX_SIZE = 6
FIELDS = ['domain', 'cell', 'n', 'split', 'seed',
          'greedy_subset', 'greedy_size', 'greedy_halfA_auroc', 'greedy_halfB_auroc',
          'good5_halfB_auroc',
          'fulloracle_subset', 'fulloracle_halfB_auroc', 'fulloracle_insample_auroc']


def _zsc(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def _half_view(ctx, idx):
    """(V, anchor, labels) for row-subset `idx`, re-zscored within the subset."""
    V = np.column_stack([_zsc(ctx.V[idx, j]) for j in range(ctx.V.shape[1])])
    anchor = _zsc(ctx.anchor[idx])
    labels = ctx.labels[idx]
    return V, anchor, labels


def _score_cols(V, anchor, labels, cols, method='residual'):
    cols = np.asarray(sorted(int(j) for j in cols), dtype=np.int64)
    if len(cols) < 3 or len(np.unique(labels)) < 2:
        return float('nan')
    fused, _meta = lsml_continuous(*[V[:, j] for j in cols], method=method)
    oriented, _flip = anchor_orient(fused, anchor)
    if np.std(oriented) < 1e-12:
        return float('nan')
    return float(roc_auc_score(labels, oriented))


def _greedy_forward(V, anchor, labels, max_size=GREEDY_MAX_SIZE, method='eigengap'):
    """Forward selection on (V, anchor, labels) ONLY — maximizes AUROC on
    this data. Fusion needs >=3 views to run at all, so this bootstraps with
    the 3 individually-strongest (raw single-feature AUROC) columns — cheap,
    no fusion fit required — then greedily adds one feature at a time up to
    max_size. Returns (best_cols, best_auroc) across all sizes visited."""
    p = V.shape[1]
    single_auc = np.array([
        roc_auc_score(labels, V[:, j]) if len(np.unique(labels)) > 1 else 0.5
        for j in range(p)
    ])
    order = np.argsort(single_auc)[::-1]
    cols = sorted(order[:3].tolist())
    cur_auc = _score_cols(V, anchor, labels, cols, method=method)
    best_overall = (list(cols), cur_auc if np.isfinite(cur_auc) else -np.inf)

    while len(cols) < min(max_size, p):
        best_j, best_a = None, best_overall[1]
        for j in range(p):
            if j in cols:
                continue
            a = _score_cols(V, anchor, labels, cols + [j], method=method)
            if np.isfinite(a) and a > best_a:
                best_a, best_j = a, j
        if best_j is None:
            break
        cols = sorted(cols + [best_j])
        if best_a > best_overall[1]:
            best_overall = (list(cols), best_a)
    return best_overall


def run_cell(ctx, R, seed, sweep_row):
    rows = []
    n = ctx.V.shape[0]
    if n < 40:
        return rows  # too few per half for a stable AUROC estimate
    g5_cols = np.asarray([ctx.pool.index(f) for f in GOOD_5 if f in ctx.pool],
                         dtype=np.int64)
    fo_names = (sweep_row['best_feats'].split('|') if sweep_row is not None
                else [])
    fo_cols = np.asarray([ctx.pool.index(f) for f in fo_names if f in ctx.pool],
                         dtype=np.int64)
    fo_insample = float(sweep_row['best_auroc']) if sweep_row is not None else float('nan')

    for split in range(R):
        rng = np.random.default_rng([seed, split,
                                     abs(hash((ctx.domain, ctx.cell_key))) % (2**31)])
        perm = rng.permutation(n)
        half = n // 2
        idxA, idxB = perm[:half], perm[half:]

        VA, aA, lA = _half_view(ctx, idxA)
        VB, aB, lB = _half_view(ctx, idxB)
        if len(np.unique(lA)) < 2 or len(np.unique(lB)) < 2:
            continue

        g_cols, g_aucA = _greedy_forward(VA, aA, lA)
        if g_cols is None:
            continue
        g_aucB = _score_cols(VB, aB, lB, g_cols, method='residual')
        good5_aucB = (_score_cols(VB, aB, lB, g5_cols, method='residual')
                      if len(g5_cols) >= 3 else float('nan'))
        fo_aucB = (_score_cols(VB, aB, lB, fo_cols, method='residual')
                  if len(fo_cols) >= 3 else float('nan'))

        rows.append({
            'domain': ctx.domain, 'cell': ctx.cell_key, 'n': n,
            'split': split, 'seed': seed,
            'greedy_subset': '|'.join(ctx.pool[j] for j in g_cols),
            'greedy_size': len(g_cols),
            'greedy_halfA_auroc': round(g_aucA, 4),
            'greedy_halfB_auroc': round(g_aucB, 4) if np.isfinite(g_aucB) else '',
            'good5_halfB_auroc': round(good5_aucB, 4) if np.isfinite(good5_aucB) else '',
            'fulloracle_subset': '|'.join(fo_names),
            'fulloracle_halfB_auroc': round(fo_aucB, 4) if np.isfinite(fo_aucB) else '',
            'fulloracle_insample_auroc': round(fo_insample, 4) if np.isfinite(fo_insample) else '',
        })
    return rows


def _existing_keys(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                done.add((r['domain'], r['cell'], r['split']))
    return done


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', default=os.environ.get('HD_DATA_ROOT', REPO_DIR))
    ap.add_argument('--sweep-summary',
                    default=os.path.join(REPO_DIR, 'results', 'subset_sweep',
                                         'sweep_summary.csv'))
    ap.add_argument('--out',
                    default=os.path.join(REPO_DIR, 'results', 'selector_bench',
                                         'splithalf_oracle.csv'))
    ap.add_argument('--summary-out',
                    default=os.path.join(REPO_DIR, 'results', 'selector_bench',
                                         'splithalf_oracle_summary.csv'))
    ap.add_argument('--R', type=int, default=10, help='random 50/50 splits per cell')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--domains', default=None)
    ap.add_argument('--cells', default=None)
    ap.add_argument('--pool-mode', default='h16',
                    help="feature pool: 'h16' (16 H(n)-derived views, the Step-189 "
                         "baseline) or 'c46' (the full 46-view CANONICAL_POOL — only "
                         "meaningful on cells with genuine 46-view coverage, e.g. the "
                         "Step-190 domain='repgrid' rag_/gpqa_ cells). Write c46 runs to "
                         "a separate --out so the h16 baseline CSV is not clobbered "
                         "(resume dedup keys on (domain, cell, split), not pool).")
    args = ap.parse_args()

    ss = pd.read_csv(args.sweep_summary)
    ss_idx = {(r['domain'], r['cell_key']): r for _, r in ss.iterrows()}

    domains = args.domains.split(',') if args.domains else None
    cells = args.cells.split(',') if args.cells else None

    done_splits = _existing_keys(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new_file = not os.path.exists(args.out)
    n_written = 0

    with open(args.out, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for ctx in iter_prepared_cells(args.data_root, args.pool_mode, domains, cells):
            key0 = (ctx.domain, ctx.cell_key, 0)
            if key0 in done_splits:
                print(f"[splithalf] {ctx.domain}/{ctx.cell_key}: already done — skip")
                continue
            sweep_row = ss_idx.get((ctx.domain, ctx.cell_key))
            rows = run_cell(ctx, args.R, args.seed, sweep_row)
            for row in rows:
                writer.writerow(row)
                n_written += 1
            f.flush()
            print(f"[splithalf] {ctx.domain}/{ctx.cell_key}: {len(rows)} splits written")

    print(f"wrote {n_written} rows -> {args.out}")

    # summary
    df = pd.read_csv(args.out)
    for col in ('greedy_halfB_auroc', 'good5_halfB_auroc', 'fulloracle_halfB_auroc',
               'fulloracle_insample_auroc', 'greedy_halfA_auroc'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    per_cell = df.groupby(['domain', 'cell']).agg(
        n=('n', 'first'),
        n_splits=('split', 'count'),
        greedy_halfA=('greedy_halfA_auroc', 'mean'),
        greedy_halfB=('greedy_halfB_auroc', 'mean'),
        good5_halfB=('good5_halfB_auroc', 'mean'),
        fulloracle_halfB=('fulloracle_halfB_auroc', 'mean'),
        fulloracle_insample=('fulloracle_insample_auroc', 'mean'),
    ).reset_index()
    per_cell['optimism_gap'] = per_cell['greedy_halfA'] - per_cell['greedy_halfB']
    per_cell.to_csv(args.summary_out, index=False)

    macro = per_cell[['greedy_halfA', 'greedy_halfB', 'good5_halfB',
                      'fulloracle_halfB', 'fulloracle_insample',
                      'optimism_gap']].mean()
    print('\n=== macro (per-cell mean of per-split means) ===')
    print(macro.round(4).to_string())
    print(f"\nwrote {args.summary_out}")


if __name__ == '__main__':
    main()
