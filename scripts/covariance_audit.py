#!/usr/bin/env python3
"""
covariance_audit.py — Inspect the covariance matrix structure of L-SML inputs.

For each selected cell this script prints:

  1. R (M×M covariance matrix of z-scored continuous features)
     - Diagonal should be ~1.0 (unit-variance after z-score)
     - Off-diagonal = Pearson correlation between features

  2. R_theory (rank-1 naive-SML prediction assuming all classifiers independent)
     R_ij^theory = (2*AUC_i - 1)(2*AUC_j - 1)
     When classifiers are truly conditionally independent given y, R_off should
     be exactly this rank-1 matrix.  Deviation = evidence of group structure.

  3. Residual R - diag - R_theory
     Shows which pairs are MORE correlated than the rank-1 model expects
     (candidates for within-group fusion) vs less (cross-group).

  4. Score matrix s_ij (Paper 1 Eq. 15) — what L-SML actually clusters on.
     Large s_ij means i,j are likely in the same dependent group.

  5. After L-SML: within-group vs cross-group mean off-diagonal values in R.
     Paper's key claim: within > cross.  If within < cross, group detection failed.

Usage:
    python scripts/covariance_audit.py                       # all cells
    python scripts/covariance_audit.py --filter math500      # only cells whose
                                                             # source file contains
                                                             # the filter string
    python scripts/covariance_audit.py --filter math500 --plot  # save PNG heatmaps
    python scripts/covariance_audit.py --features good       # only GOOD_FEATURES (5)
    python scripts/covariance_audit.py --features all        # all 16 features
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
    detect_dependent_groups,
    lsml_continuous_pipeline,
    lsml_fuse,
    zscore,
    _score_matrix_lsml,
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


def best_auc_oriented(labels, scores):
    """Return (AUROC, sign) where sign is +1 if higher score = more likely correct."""
    ap, *_ = _boot_auc(labels, scores)
    an, *_ = _boot_auc(labels, -scores)
    if ap >= an:
        return ap, +1
    return an, -1


# ── Matrix helpers ─────────────────────────────────────────────────────────────

def format_matrix(M, names, fmt='{:+.3f}', width=9):
    """Print a matrix with named rows/cols."""
    max_name = max(len(n) for n in names)
    header = ' ' * (max_name + 2) + ''.join(f'{n:>{width}}' for n in names)
    print(header)
    for i, row_name in enumerate(names):
        row = f'{row_name:<{max_name}}  ' + ''.join(fmt.format(M[i, j]).rjust(width)
                                                      for j in range(len(names)))
        print(row)


def _rank1_residual_fraction(M_off):
    """What fraction of variance in M_off is explained by the leading eigenvector?"""
    if M_off.shape[0] < 2:
        return float('nan')
    eigvals = np.linalg.eigvalsh(M_off)
    total = np.sum(eigvals ** 2)
    if total < 1e-12:
        return float('nan')
    return float(eigvals[-1] ** 2 / total)


# ── Cache loading (same as run_lsml_experiments.py) ───────────────────────────

def load_pkl_cells(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        return []
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


def gather_cells(data_dir, filter_str=''):
    all_cells = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.pkl'):
            continue
        if 'lsml' in fname or 'results' in fname or 'ablation' in fname:
            continue
        if filter_str and filter_str not in fname:
            continue
        path = os.path.join(data_dir, fname)
        try:
            for ck, fd, lb in load_pkl_cells(path):
                all_cells.append((fname, ck, fd, lb))
        except Exception as e:
            print(f'  [skip] {fname}: {e}')
    return all_cells


# ── Per-cell audit ─────────────────────────────────────────────────────────────

def audit_cell(src, cell_key, feats_dict, labels, feat_names, plot=False):
    n = len(labels)
    m = len(feat_names)

    print(f'\n{"=" * 72}')
    print(f'Cell: {src} / {cell_key}   n={n}  M={m}')
    print(f'{"=" * 72}')

    # ── 1. Individual AUROCs and sign orientation ─────────────────────────────
    aucs = {}
    for f in feat_names:
        auc, _ = best_auc_oriented(labels, np.array(feats_dict[f]))
        aucs[f] = auc

    print('\n--- Individual feature AUROCs (vs labels) ---')
    for f in feat_names:
        bar = '#' * int(round((aucs[f] - 0.5) * 40))
        print(f'  {f:<25}  {aucs[f]:.3f}  {bar}')

    # ── 2. Compute continuous z-scored R matrix ──────────────────────────────
    views_cont = []
    for f in feat_names:
        arr = np.array(feats_dict[f], dtype=float)
        s = FEATURE_SIGNS.get(f, +1)
        views_cont.append(zscore(arr * s))

    X_cont = np.column_stack(views_cont)
    R_cont = np.cov(X_cont.T)
    if R_cont.ndim == 0:
        R_cont = np.array([[float(R_cont)]])

    print('\n--- R: covariance matrix of z-scored oriented features ---')
    print('(diagonal = variance; off-diagonal = Pearson correlation)')
    format_matrix(R_cont, feat_names)

    diag_vals = np.diag(R_cont)
    print(f'\n  Diagonal: min={diag_vals.min():.3f}  max={diag_vals.max():.3f}  '
          f'mean={diag_vals.mean():.3f}  (theory: all 1.0)')

    off_mask = ~np.eye(m, dtype=bool)
    off_vals = R_cont[off_mask]
    print(f'  Off-diag: min={off_vals.min():.3f}  max={off_vals.max():.3f}  '
          f'mean={off_vals.mean():.3f}  abs_mean={np.abs(off_vals).mean():.3f}')

    # ── 3. Rank-1 naive-SML theoretical prediction ────────────────────────────
    # R_ij^theory = (2*AUC_i - 1)(2*AUC_j - 1)
    # Under conditional independence given y, the off-diagonal is exactly this.
    alpha = np.array([2 * aucs[f] - 1 for f in feat_names])
    R_theory_off = np.outer(alpha, alpha)  # rank-1, zeros on diagonal by convention
    np.fill_diagonal(R_theory_off, 0.0)

    print('\n--- R_theory (rank-1 naive-SML prediction: outer product of (2*AUC-1)) ---')
    print('If features are conditionally independent given y, R_off should match this.')
    format_matrix(R_theory_off, feat_names)

    # ── 4. Residual: how much does R deviate from rank-1? ────────────────────
    R_off_actual = R_cont.copy()
    np.fill_diagonal(R_off_actual, 0.0)

    residual_matrix = R_off_actual - R_theory_off
    print('\n--- Residual = R_off - R_theory_off ---')
    print('Positive: more correlated than rank-1 predicts  (within-group evidence)')
    print('Negative: less correlated than rank-1 predicts  (cross-group evidence)')
    format_matrix(residual_matrix, feat_names)

    # Rank-1 fraction of R_off_actual
    r1_frac = _rank1_residual_fraction(R_off_actual)
    print(f'\n  Leading eigenvector explains {100 * r1_frac:.1f}% of R_off variance')
    print(f'  (naive SML assumes 100%; lower = more group structure present)')

    # ── 5. Binarized R for comparison ─────────────────────────────────────────
    binary = binarize_classifiers(
        {f: feats_dict[f] for f in feat_names}, FEATURE_SIGNS,
    )
    X_bin = np.column_stack([binary[f] for f in feat_names])
    R_bin = np.cov(X_bin.T)
    if R_bin.ndim == 0:
        R_bin = np.array([[float(R_bin)]])

    print('\n--- R_bin: covariance matrix of binarized (±1) oriented classifiers ---')
    format_matrix(R_bin, feat_names)

    diag_bin = np.diag(R_bin)
    off_bin = R_bin[off_mask]
    print(f'\n  Diagonal: min={diag_bin.min():.3f}  max={diag_bin.max():.3f}  '
          f'mean={diag_bin.mean():.3f}  (theory: all 1.0 for binary ±1)')
    print(f'  Off-diag: min={off_bin.min():.3f}  max={off_bin.max():.3f}  '
          f'mean={off_bin.mean():.3f}  abs_mean={np.abs(off_bin).mean():.3f}')

    # ── 6. L-SML score matrix and group detection ─────────────────────────────
    s_mat = _score_matrix_lsml(R_bin)
    print('\n--- Score matrix s (Paper 1 Eq.15, computed on R_bin) ---')
    print('Large s_ij = strong dependence evidence (i,j likely in same group).')
    format_matrix(s_mat, feat_names, fmt='{:.3f}', width=9)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, meta_bin = lsml_fuse(*[binary[f] for f in feat_names])
    K_bin = meta_bin['K']
    c_bin = meta_bin['c']

    print(f'\n--- L-SML group detection (K={K_bin}) ---')
    for g in np.unique(c_bin):
        members = [feat_names[i] for i in range(m) if c_bin[i] == g]
        print(f'  Group {g}: {members}')

    # Within-group vs cross-group mean off-diagonal
    within_vals, cross_vals = [], []
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if c_bin[i] == c_bin[j]:
                within_vals.append(R_bin[i, j])
            else:
                cross_vals.append(R_bin[i, j])

    print('\n  Off-diagonal R_bin values by group membership:')
    if within_vals:
        wm, wabs = np.mean(within_vals), np.mean(np.abs(within_vals))
        print(f'    Within-group  ({len(within_vals)} pairs): mean={wm:.4f}  abs_mean={wabs:.4f}')
    if cross_vals:
        cm, cabs = np.mean(cross_vals), np.mean(np.abs(cross_vals))
        print(f'    Cross-group   ({len(cross_vals)} pairs): mean={cm:.4f}  abs_mean={cabs:.4f}')
    if within_vals and cross_vals:
        ratio = wabs / (cabs + 1e-12)
        meets_theory = 'YES (within > cross)' if ratio > 1.0 else 'NO  (cross >= within)'
        print(f'    Abs-mean ratio within/cross = {ratio:.3f}  -- paper predicts > 1 -- {meets_theory}')

    # ── 7. Continuous pipeline group detection ───────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, meta_cont = lsml_continuous_pipeline(feats_dict, feat_names, FEATURE_SIGNS)
    K_cont = meta_cont['K']
    c_cont = meta_cont['c']

    if K_cont != K_bin or not np.array_equal(c_cont, c_bin):
        print(f'\n  [NOTE] Continuous pipeline detects K={K_cont} vs binarized K={K_bin}')
        for g in np.unique(c_cont):
            members = [feat_names[i] for i in range(m) if c_cont[i] == g]
            print(f'    Group {g}: {members}')

    # ── 8. Optional: save heatmap PNG ────────────────────────────────────────
    if plot:
        _save_heatmaps(src, cell_key, feat_names, R_cont, R_theory_off,
                       residual_matrix, s_mat, c_bin)


def _save_heatmaps(src, cell_key, feat_names, R, R_theory, residual, s, c):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [plot] matplotlib not available; skipping PNG')
        return

    short_names = [f[:8] for f in feat_names]
    m = len(feat_names)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'{src} / {cell_key}', fontsize=10)

    def heatmap(ax, M, title, vmin=None, vmax=None, cmap='RdBu_r'):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(m))
        ax.set_yticks(range(m))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Color rows/cols by detected group
        for i in range(m):
            col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                   'tab:purple'][c[i] % 5]
            ax.get_xticklabels()[i].set_color(col)
            ax.get_yticklabels()[i].set_color(col)

    abs_max = max(np.abs(R).max(), 0.01)
    heatmap(axes[0], R,        'R (continuous z-scored)', -abs_max, abs_max)
    heatmap(axes[1], R_theory, 'R_theory (rank-1)',       -abs_max, abs_max)
    heatmap(axes[2], residual, 'Residual (R - R_theory)', -abs_max, abs_max)
    heatmap(axes[3], s,        'Score matrix s (L-SML)',
            vmin=0, vmax=s.max() + 1e-12, cmap='YlOrRd')

    plt.tight_layout()
    safe_cell = cell_key.replace('/', '_').replace(' ', '_')[:40]
    out_path = os.path.join('local_cache', f'cov_audit_{safe_cell}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [plot] saved {out_path}')


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data-dir', default='./local_cache',
                        help='local_cache directory (default: ./local_cache)')
    parser.add_argument('--filter', default='',
                        help='Only process cells whose source filename contains this string')
    parser.add_argument('--features', default='good',
                        choices=['good', 'all'],
                        help='"good" = GOOD_FEATURES (5), "all" = all 16 (default: good)')
    parser.add_argument('--plot', action='store_true',
                        help='Save PNG heatmaps to local_cache/')
    args = parser.parse_args()

    feat_names = GOOD_FEATURES if args.features == 'good' else list(FEATURE_SIGNS.keys())

    cells = gather_cells(args.data_dir, filter_str=args.filter)
    if not cells:
        print(f'No matching cells found in {args.data_dir} (filter={args.filter!r})')
        return

    print(f'Found {len(cells)} cells. Feature set: {args.features} ({len(feat_names)} features)')
    print(f'Features: {feat_names}')

    for src, ck, fd, lb in cells:
        available = [f for f in feat_names if f in fd]
        if len(available) < 3:
            print(f'\n[skip] {src}/{ck}: only {len(available)} of {len(feat_names)} features present')
            continue
        audit_cell(src, ck, fd, lb, available, plot=args.plot)


if __name__ == '__main__':
    main()
