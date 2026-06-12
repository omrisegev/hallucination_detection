"""
feature_correlation_full.py — Full 16-feature Spearman-|rho| dependence matrix.

method_comparison.py Table 3 only reports the 10 GOOD_5 pairs. The L-SML report
needs the dependence structure across ALL 16 H(n) spectral features (120 pairs) to
motivate clustering: flat SML assumes independence, L-SML does not. This script
loads every cached cell, computes mean |Spearman rho| for every pair of the 16
features (sign-oriented with FEATURE_SIGNS so correlations are on the deployed
orientation), and writes a 16x16 matrix to results/feature_correlation_16.csv.

Usage:
    python scripts/feature_correlation_full.py --data-dir ./local_cache
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import spearmanr

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import FEAT_NAMES  # noqa: E402

# Reuse the loaders + constants from method_comparison so the cell set matches
# exactly what the report's other tables are computed from.
from scripts.method_comparison import (  # noqa: E402
    ALL_H16, FEATURE_SIGNS, PKL_NAMES, load_cached_feats,
)

RESULTS_DIR = os.path.join(REPO_DIR, 'results')


def collect_pair_rhos(data_dir):
    """Return {(fi, fj): [|rho| per cell]} over every cached cell, for ALL_H16 pairs."""
    pair_rhos = {}
    n_cells = 0
    for domain, pkl_name in PKL_NAMES.items():
        feats = load_cached_feats(os.path.join(data_dir, pkl_name))
        if feats is None:
            print(f'[MISSING] {pkl_name}')
            continue
        for cell_key, payload in feats.items():
            fd, lbl = payload
            lbl_arr = np.asarray(lbl, dtype=int)
            if len(set(lbl_arr.tolist())) < 2:
                continue
            n_cells += 1
            for i, fi in enumerate(ALL_H16):
                if fi not in fd:
                    continue
                ai = np.array(fd[fi], dtype=float) * FEATURE_SIGNS.get(fi, 1)
                for fj in ALL_H16[i + 1:]:
                    if fj not in fd:
                        continue
                    aj = np.array(fd[fj], dtype=float) * FEATURE_SIGNS.get(fj, 1)
                    if np.all(ai == ai[0]) or np.all(aj == aj[0]):
                        continue
                    try:
                        rho, _ = spearmanr(ai, aj)
                    except Exception:
                        continue
                    if rho == rho:  # not NaN
                        pair_rhos.setdefault((fi, fj), []).append(abs(float(rho)))
    return pair_rhos, n_cells


def write_matrix(pair_rhos, out_path):
    """Write a 16x16 symmetric mean-|rho| matrix CSV (diagonal = 1.0)."""
    mean_rho = {pair: float(np.mean(v)) for pair, v in pair_rhos.items()}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature'] + ALL_H16)
        for fi in ALL_H16:
            row = [fi]
            for fj in ALL_H16:
                if fi == fj:
                    row.append('1.000')
                else:
                    key = (fi, fj) if (fi, fj) in mean_rho else (fj, fi)
                    val = mean_rho.get(key)
                    row.append(f'{val:.3f}' if val is not None else '')
            w.writerow(row)
    print(f'Matrix -> {out_path}')

    # Also emit a long-form ranked pair list for quick inspection.
    long_path = out_path.replace('.csv', '_pairs.csv')
    rows = sorted(mean_rho.items(), key=lambda kv: -kv[1])
    with open(long_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature_pair', 'mean_abs_rho', 'n_cells'])
        for (fi, fj), m in rows:
            w.writerow([f'{fi}|{fj}', f'{m:.4f}', len(pair_rhos[(fi, fj)])])
    print(f'Pairs  -> {long_path}')

    # Summary stats used as captions in the report.
    vals = list(mean_rho.values())
    n_high = sum(1 for v in vals if v >= 0.75)
    n_mid = sum(1 for v in vals if 0.4 <= v < 0.75)
    print(f'\n{len(vals)} pairs | mean |rho| = {np.mean(vals):.3f} | '
          f'median = {np.median(vals):.3f} | >=0.75: {n_high} | 0.40-0.75: {n_mid}')


def main():
    parser = argparse.ArgumentParser(description='Full 16-feature Spearman correlation matrix.')
    parser.add_argument('--data-dir', default='./local_cache')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print(f'data_dir : {data_dir}')
    pair_rhos, n_cells = collect_pair_rhos(data_dir)
    print(f'cells used: {n_cells} | pairs with data: {len(pair_rhos)}')
    write_matrix(pair_rhos, os.path.join(RESULTS_DIR, 'feature_correlation_16.csv'))


if __name__ == '__main__':
    main()
