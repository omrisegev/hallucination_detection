#!/usr/bin/env python3
"""
run_lsml_experiments.py — Compare three L-SML fusion strategies across all
cached cells in local_cache.

Strategies tested on every cell that has >= 3 known features:

  Baseline   simple_average_fusion of z-scored + oriented continuous features.

  Current    binarize_classifiers(FEATURE_SIGNS) -> lsml_fuse  (production pipeline)
             Binarises at median, uses external sign knowledge.
             K_range default is now fixed (K < M always).

  Exp1       sml_unsupervised  (paper-aligned, no FEATURE_SIGNS)
             Binarises at median with no sign orientation; sign resolved
             internally by sml_fuse_signed assumption (iii): majority-positive
             flip.  Matches the fully-unsupervised paper setting.

  Exp2       lsml_continuous_pipeline(FEATURE_SIGNS)
             Z-scores + orients with FEATURE_SIGNS, but keeps features
             continuous (no binarisation).  No theoretical guarantee, but
             preserves the signal that binarisation discards (~4 pp on math500).

Usage:
    python scripts/run_lsml_experiments.py
    python scripts/run_lsml_experiments.py --data-dir ./local_cache --verbose
"""

import argparse
import os
import pickle
import sys
import warnings

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils.fusion_utils import (
    boot_auc as _boot_auc,
    binarize_classifiers,
    lsml_continuous_pipeline,
    lsml_fuse,
    simple_average_fusion,
    sml_unsupervised,
    zscore,
)

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
}

GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

# ── AUROC helper ──────────────────────────────────────────────────────────────

def best_auc(labels, scores):
    a_p, lo_p, hi_p = _boot_auc(labels, scores)
    a_n, lo_n, hi_n = _boot_auc(labels, -scores)
    if a_p >= a_n:
        return a_p, lo_p, hi_p
    return a_n, lo_n, hi_n


# ── Cache loading ─────────────────────────────────────────────────────────────

def load_pkl_cells(path):
    """
    Load cells from a local_cache pkl file.
    Handles two formats:
      - {feats: {cell_key: (feats_dict, labels)}, ...}   (math500_res.pkl etc.)
      - {cell_key: (feats_dict, labels)}                 (rag_feats_all.pkl etc.)
    Returns: list of (cell_key_str, feats_dict, labels_array)
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        return []

    # Try nested 'feats' key first
    raw = obj.get('feats', obj)

    cells = []
    for k, v in raw.items():
        if not isinstance(v, (tuple, list)) or len(v) < 2:
            continue
        fd, lb = v[0], v[1]
        if not isinstance(fd, dict):
            continue
        lb = np.array(lb, dtype=int)
        if len(np.unique(lb)) < 2 or len(lb) < 10:
            continue
        cells.append((str(k), fd, lb))
    return cells


def gather_all_cells(data_dir):
    """
    Collect all cells from all known pkl files.
    Returns list of (source_file, cell_key, feats_dict, labels).
    """
    all_cells = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.pkl'):
            continue
        # Skip pre-computed L-SML results (they don't have raw features)
        if 'lsml' in fname or 'results' in fname or 'ablation' in fname:
            continue
        path = os.path.join(data_dir, fname)
        try:
            cells = load_pkl_cells(path)
            for ck, fd, lb in cells:
                all_cells.append((fname, ck, fd, lb))
        except Exception as e:
            print(f'  [skip] {fname}: {e}')
    return all_cells


# ── Per-cell experiment runner ────────────────────────────────────────────────

def run_cell(cell_key, feats_dict, labels, verbose=False):
    """
    Run all four strategies on one cell.

    Returns a result dict with keys:
        feat_names, n, pos_frac,
        best_individual,
        baseline_auc, current_auc, current_K,
        exp1_auc, exp1_K,
        exp2_auc, exp2_K
    """
    # Use known features present in this cell, in a stable order
    feat_names = [f for f in FEATURE_SIGNS if f in feats_dict]
    if len(feat_names) < 3:
        return None

    n = len(labels)
    pos_frac = labels.mean()

    # ── Best individual feature AUROC (continuous) ────────────────────────────
    indiv_aucs = {}
    for f in feat_names:
        auc, _, _ = best_auc(labels, np.array(feats_dict[f]))
        indiv_aucs[f] = auc
    best_indiv = max(indiv_aucs.values())
    best_feat  = max(indiv_aucs, key=indiv_aucs.get)

    # ── Baseline: simple average of z-scored + oriented continuous features ───
    cont_views = [zscore(np.array(feats_dict[f]) * FEATURE_SIGNS[f])
                  for f in feat_names]
    avg_scores, _ = simple_average_fusion(*cont_views)
    baseline_auc, _, _ = best_auc(labels, avg_scores)

    # ── Current: binarize(FEATURE_SIGNS) + lsml_fuse ─────────────────────────
    binary = binarize_classifiers(
        {f: feats_dict[f] for f in feat_names}, FEATURE_SIGNS,
    )
    bin_views = [binary[f] for f in feat_names]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cur_scores, cur_meta = lsml_fuse(*bin_views)
    current_auc, _, _ = best_auc(labels, cur_scores)
    current_K = cur_meta['K']

    # ── Exp1: sml_unsupervised — binarize at median, no FEATURE_SIGNS ─────────
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exp1_scores, exp1_meta = sml_unsupervised(feats_dict, feat_names)
    exp1_auc, _, _ = best_auc(labels, exp1_scores)
    exp1_K = exp1_meta['K']

    # ── Exp2: lsml_continuous_pipeline — z-scored continuous, FEATURE_SIGNS ──
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exp2_scores, exp2_meta = lsml_continuous_pipeline(
            feats_dict, feat_names, FEATURE_SIGNS,
        )
    exp2_auc, _, _ = best_auc(labels, exp2_scores)
    exp2_K = exp2_meta['K']

    if verbose:
        print(f'  {cell_key}  n={n}  M={len(feat_names)}')
        print(f'    Best individual ({best_feat}): {best_indiv:.3f}')
        print(f'    Baseline (avg continuous):    {baseline_auc:.3f}')
        print(f'    Current  (binarized+signs):   {current_auc:.3f}  K={current_K}')
        print(f'    Exp1     (paper-aligned):     {exp1_auc:.3f}  K={exp1_K}')
        print(f'    Exp2     (continuous lsml):   {exp2_auc:.3f}  K={exp2_K}')

    return {
        'feat_names': feat_names, 'n': n, 'pos_frac': pos_frac,
        'best_individual': best_indiv, 'best_feat': best_feat,
        'baseline_auc': baseline_auc,
        'current_auc': current_auc, 'current_K': current_K,
        'exp1_auc': exp1_auc, 'exp1_K': exp1_K,
        'exp2_auc': exp2_auc, 'exp2_K': exp2_K,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_table(rows):
    if not rows:
        print('No results.')
        return

    col_w = 38
    hdr = (f'{"Cell":<{col_w}} {"n":>5}  {"BestInd":>7}  {"Baseline":>8}  '
           f'{"Current":>9}  {"Exp1":>9}  {"Exp2":>9}')
    print()
    print(hdr)
    print('-' * len(hdr))

    totals = {k: [] for k in ('best_individual', 'baseline_auc',
                               'current_auc', 'exp1_auc', 'exp2_auc')}

    for src, ck, r in rows:
        tag = f'{src}/{ck}'[:col_w]
        print(f'{tag:<{col_w}} {r["n"]:>5}  {r["best_individual"]:>7.3f}  '
              f'{r["baseline_auc"]:>8.3f}  '
              f'{r["current_auc"]:>8.3f}K{r["current_K"]}  '
              f'{r["exp1_auc"]:>8.3f}K{r["exp1_K"]}  '
              f'{r["exp2_auc"]:>8.3f}K{r["exp2_K"]}')
        for k in totals:
            totals[k].append(r[k])

    print('-' * len(hdr))
    means = {k: np.mean(v) for k, v in totals.items()}
    print(f'{"MEAN":<{col_w}} {"":>5}  {means["best_individual"]:>7.3f}  '
          f'{means["baseline_auc"]:>8.3f}  '
          f'{means["current_auc"]:>8.3f}      '
          f'{means["exp1_auc"]:>8.3f}      '
          f'{means["exp2_auc"]:>8.3f}')
    print()

    # Lift summary (vs best individual)
    print('Lift vs best individual:')
    for label, key in [('Baseline (avg continuous)', 'baseline_auc'),
                        ('Current  (binarized+signs)', 'current_auc'),
                        ('Exp1     (paper-aligned)', 'exp1_auc'),
                        ('Exp2     (continuous lsml)', 'exp2_auc')]:
        deltas = [r[key] - r['best_individual'] for _, _, r in rows]
        mean_d = np.mean(deltas) * 100
        wins = sum(1 for d in deltas if d > 0.005)
        print(f'  {label:<32}: {mean_d:+.2f} pp mean  ({wins}/{len(rows)} cells beat best individual)')

    print()
    # Head-to-head: Exp2 vs Current
    print('Exp2 (continuous) vs Current (binarized):')
    deltas2 = [(r['exp2_auc'] - r['current_auc']) * 100 for _, _, r in rows]
    print(f'  Mean delta:  {np.mean(deltas2):+.2f} pp')
    print(f'  Exp2 wins:   {sum(1 for d in deltas2 if d > 0.005)}/{len(rows)} cells')
    print(f'  Exp2 loses:  {sum(1 for d in deltas2 if d < -0.005)}/{len(rows)} cells')

    print()
    # Head-to-head: Exp1 vs Current
    print('Exp1 (paper-aligned) vs Current (binarized+signs):')
    deltas1 = [(r['exp1_auc'] - r['current_auc']) * 100 for _, _, r in rows]
    print(f'  Mean delta:  {np.mean(deltas1):+.2f} pp')
    print(f'  Exp1 wins:   {sum(1 for d in deltas1 if d > 0.005)}/{len(rows)} cells')
    print(f'  Exp1 loses:  {sum(1 for d in deltas1 if d < -0.005)}/{len(rows)} cells')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data-dir', default='./local_cache',
                        help='local_cache directory (default: ./local_cache)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-cell details')
    args = parser.parse_args()

    print('L-SML Experiment Comparison')
    print('Branch: experiment/lsml-variants')
    print()
    print('Methods:')
    print('  Baseline  simple average of continuous z-scored + FEATURE_SIGNS features')
    print('  Current   binarize(FEATURE_SIGNS) + lsml_fuse  (K_range fix applied)')
    print('  Exp1      sml_unsupervised  (paper-aligned, no FEATURE_SIGNS)')
    print('  Exp2      lsml_continuous_pipeline  (continuous z-score, FEATURE_SIGNS)')
    print()

    all_cells = gather_all_cells(args.data_dir)
    if not all_cells:
        print(f'No cells found in {args.data_dir}')
        return

    print(f'Found {len(all_cells)} cells across {len(set(s for s,_,_,_ in all_cells))} files.')
    print()

    rows = []
    for src, ck, fd, lb in all_cells:
        result = run_cell(ck, fd, lb, verbose=args.verbose)
        if result is not None:
            rows.append((src, ck, result))
        else:
            print(f'  [skip] {src}/{ck}: fewer than 3 known features')

    print_table(rows)


if __name__ == '__main__':
    main()
