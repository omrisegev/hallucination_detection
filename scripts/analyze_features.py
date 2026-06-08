"""
analyze_features.py — Two local diagnostics using cached feature pkls.

1. L-SML cluster visualization: which features co-cluster, how strong is
   the pairwise dependency signal, and what effective weight does each
   feature receive? Runs on GOOD_FEATURES (5) by default or all 16 with
   --all-features.

2. trace_length binarization investigation: shows why trace_length is
   excluded from GOOD_FEATURES — right-censoring at max_new_tokens causes
   the strict-median split to produce a heavily imbalanced binary classifier.

Usage:
    python scripts/analyze_features.py                             # GOOD_FEATURES (5)
    python scripts/analyze_features.py --features all             # all 16 features
    python scripts/analyze_features.py --features epr sw_var_peak cusum_max spectral_entropy
    python scripts/analyze_features.py --analysis trace_length    # trace_length only
"""

import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import FEAT_NAMES, boot_auc, binarize_classifiers, lsml_fuse

# ── Constants (same as PROGRESS.md / Optimized notebook) ──────────────────────
GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
}

MIN_IND_AUC_THRESHOLD = 0.53

DOMAIN_PKLS = {
    'math500': 'math500_res.pkl',
    'gsm8k':   'gsm8k_res.pkl',
    'gpqa':    'gpqa_res.pkl',
    'rag':     'rag_feats_all.pkl',
    'qa':      'qa_res.pkl',
}

DOMAIN_COLORS = {
    'math500': '#2196F3', 'gsm8k': '#4CAF50', 'gpqa': '#FF9800',
    'rag': '#9C27B0', 'qa': '#F44336',
}


# ── Data loading (mirrors the Optimized notebook's Section 3) ──────────────────

def load_all_cells(data_dir):
    """Return {cell_key: (feats_dict, labels_array)} from the 5 domain pkls."""
    consolidated = os.path.join(data_dir, 'consolidated_results')
    search_dirs = [consolidated, data_dir]

    all_cells = {}
    for domain, fname in DOMAIN_PKLS.items():
        path = None
        for d in search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            print(f'  WARNING: {fname} not found in {search_dirs}')
            continue

        with open(path, 'rb') as f:
            obj = pickle.load(f)

        # Two on-disk formats: plain dict or {'results': ..., 'feats': ...}
        feats_map = obj.get('feats', obj) if isinstance(obj, dict) else {}

        for key, val in feats_map.items():
            if not (isinstance(val, tuple) and len(val) == 2):
                continue
            fd, lbl = val
            if not isinstance(fd, dict) or len(lbl) == 0:
                continue
            all_cells[f'{domain}/{key}'] = (fd, np.array(lbl, dtype=int))

    print(f'Loaded {len(all_cells)} cells from {data_dir}')
    return all_cells


# ── Analysis 1: L-SML cluster visualization ───────────────────────────────────

def analyze_clusters(all_cells, out_dir, features=None):
    if features is None:
        features = GOOD_FEATURES
    m = len(features)
    if features == FEAT_NAMES:
        tag = 'all16'
    elif features == GOOD_FEATURES:
        tag = 'good5'
    else:
        tag = f'custom{m}'

    co_cluster_counts = np.zeros((m, m), dtype=int)
    score_mat_sum     = np.zeros((m, m), dtype=float)
    all_records = []   # (cell_key, meta, lbl, fused_score)

    for cell_key, (fd, lbl) in all_cells.items():
        if len(set(lbl.tolist())) < 2:
            continue
        if any(fn not in fd for fn in features):
            continue
        try:
            binary = binarize_classifiers(
                {fn: fd[fn] for fn in features}, FEATURE_SIGNS
            )
            fused, meta = lsml_fuse(*[binary[fn] for fn in features])
            c = meta['c']
            score_mat_sum += meta['score_matrix']
            for i in range(m):
                for j in range(m):
                    if c[i] == c[j]:
                        co_cluster_counts[i, j] += 1
            all_records.append((cell_key, meta, lbl, fused))
        except Exception as e:
            print(f'  [{cell_key}] skipped: {e}')

    n_valid = len(all_records)
    if n_valid == 0:
        print('No valid cells for cluster analysis.')
        return

    score_mat_mean = score_mat_sum / n_valid
    co_freq = co_cluster_counts / n_valid

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f'\n=== L-SML cluster analysis — {m} features ({n_valid} cells) ===')
    print(f'\nK selection (residual-minimisation over K=2..min({m},8)):')
    for k, cnt in sorted(Counter(meta['K'] for _, meta, _, _ in all_records).items()):
        print(f'  K={k}: {cnt} cells ({100*cnt/n_valid:.0f}%)')

    print(f'\nCo-clustering frequency (fraction of cells where pair is in same group):')
    col_w = max(len(fn) for fn in features) + 2
    header = f'{"":>{col_w}}' + ''.join(f'{fn:>{col_w}}' for fn in features)
    print(header)
    for i, fn_i in enumerate(features):
        row = f'{fn_i:>{col_w}}' + ''.join(f'{co_freq[i,j]:>{col_w}.0%}' for j in range(m))
        print(row)

    # ── Per-group AUROC (averaged across all cells) ───────────────────────────
    # virtual_classifiers[:,g] is the binary ±1 fused score for group g.
    # We take max(AUC, 1-AUC) to handle sign ambiguity.
    # Groups are matched by their feature membership (sorted tuple of names).
    group_aucs = {}   # frozenset(feat_names) -> list of AUROCs
    lsml_aucs  = []
    for cell_key, meta, lbl, fused in all_records:
        c = meta['c']
        vc = meta['virtual_classifiers']   # n_samples × K
        for g in range(meta['K']):
            xi = vc[:, g]
            a_pos, *_ = boot_auc(lbl,  xi)
            a_neg, *_ = boot_auc(lbl, -xi)
            auc = max(float(a_pos), float(a_neg)) if not np.isnan(a_pos) else float('nan')
            key = frozenset(features[i] for i in range(m) if c[i] == g)
            group_aucs.setdefault(key, []).append(auc)
        a_pos, *_ = boot_auc(lbl,  fused)
        a_neg, *_ = boot_auc(lbl, -fused)
        lsml_aucs.append(max(float(a_pos), float(a_neg)))

    print(f'\nPer-group mean AUROC (virtual classifier xi_g, across {n_valid} cells):')
    sorted_groups = sorted(group_aucs.items(), key=lambda kv: -np.nanmean(kv[1]))
    for key, aucs in sorted_groups:
        valid = [a for a in aucs if not np.isnan(a)]
        feat_str = ', '.join(sorted(key))
        print(f'  [{len(valid):>2}x] {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}  '
              f'n_cells={len(valid)}  features: {feat_str}')
    print(f'  Full L-SML output:          '
          f'{np.nanmean(lsml_aucs):.3f} +/- {np.nanstd(lsml_aucs):.3f}  (n={len(lsml_aucs)})')

    # ── Pick representative cell ──────────────────────────────────────────────
    rep_cell = 'math500/Qwen-Math-7B_T1.0'
    rep_rec  = next((r for r in all_records if r[0] == rep_cell), None)
    if rep_rec is None:
        rep_rec = all_records[0]
        print(f'\nFallback representative cell: {rep_rec[0]}')
    else:
        print(f'\nRepresentative cell: {rep_cell}')
    rep_cell, rep_meta, rep_lbl, rep_fused = rep_rec

    c_rep = rep_meta['c']
    K_rep = rep_meta['K']
    group_weights = rep_meta['group_weights']
    cross_weights = rep_meta['cross_weights']
    vc_rep = rep_meta['virtual_classifiers']

    print(f'\nK={K_rep} groups for {rep_cell}:')
    for g_idx, (idx_arr, w_g) in enumerate(group_weights):
        cw = float(np.abs(cross_weights[g_idx])) if K_rep > 1 else 1.0
        g_feats = [features[i] for i in idx_arr]
        xi = vc_rep[:, g_idx]
        ga_pos, *_ = boot_auc(rep_lbl,  xi)
        ga_neg, *_ = boot_auc(rep_lbl, -xi)
        g_auc = max(float(ga_pos), float(ga_neg))
        print(f'  Group {g_idx}  cross-weight={cw:.3f}  group-AUROC={g_auc:.3f}  features={g_feats}')
        for rank, fi in enumerate(idx_arr):
            print(f'    {features[fi]:<24} within-weight={np.abs(w_g[rank]):.3f}')
    a_pos, *_ = boot_auc(rep_lbl,  rep_fused)
    a_neg, *_ = boot_auc(rep_lbl, -rep_fused)
    print(f'  Full L-SML AUROC: {max(float(a_pos), float(a_neg)):.3f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    palette = plt.cm.tab10.colors   # 10 colors — enough for up to 10 groups
    fs = 7 if m > 8 else 9          # smaller font for 16-feature labels
    fig_w = max(22, m * 0.9)
    fig, axes = plt.subplots(1, 4, figsize=(fig_w, max(5, m * 0.45)))

    # Panel 1: Co-clustering frequency
    ax = axes[0]
    im = ax.imshow(co_freq, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(m)); ax.set_xticklabels(features, rotation=45, ha='right', fontsize=fs)
    ax.set_yticks(range(m)); ax.set_yticklabels(features, fontsize=fs)
    if m <= 8:
        for i in range(m):
            for j in range(m):
                val = co_freq[i, j]
                ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=6,
                        color='white' if val > 0.55 else 'black')
    ax.set_title(f'Co-clustering frequency\n({n_valid} cells, same group)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Mean score matrix (diagonal zeroed for readability)
    ax = axes[1]
    disp_s = score_mat_mean.copy()
    np.fill_diagonal(disp_s, 0)
    im2 = ax.imshow(disp_s, cmap='OrRd', aspect='auto')
    ax.set_xticks(range(m)); ax.set_xticklabels(features, rotation=45, ha='right', fontsize=fs)
    ax.set_yticks(range(m)); ax.set_yticklabels(features, fontsize=fs)
    ax.set_title('Mean pairwise dependency score\n(Eq.15; higher = more dependent)')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Effective weights for representative cell
    ax = axes[2]
    feat_weights = np.zeros(m)
    for g_idx, (idx_arr, w_g) in enumerate(group_weights):
        cw = float(np.abs(cross_weights[g_idx])) if K_rep > 1 else 1.0
        for rank, feat_idx in enumerate(idx_arr):
            feat_weights[feat_idx] = cw * float(np.abs(w_g[rank]))
    feat_weights /= feat_weights.sum() + 1e-12

    bar_colors = [palette[int(c_rep[i]) % len(palette)] for i in range(m)]
    ax.bar(range(m), feat_weights, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(m)); ax.set_xticklabels(features, rotation=45, ha='right', fontsize=fs)
    ax.set_ylabel('Effective weight (cross × within, L1-normalized)')
    ax.set_title(f'Feature weights — {rep_cell.split("/")[-1]}\n(K={K_rep} groups, color=group)')
    for g in range(K_rep):
        g_feats = [features[i] for i in range(m) if c_rep[i] == g]
        cw = float(np.abs(cross_weights[g])) if K_rep > 1 else 1.0
        ax.bar([], [], color=palette[g % len(palette)], alpha=0.85,
               label=f'G{g} ({cw:.2f}): {", ".join(g_feats)}')
    ax.legend(fontsize=6, loc='upper right')

    # Panel 4: Per-group mean AUROC (across all cells), ordered by mean AUROC
    ax = axes[3]
    sorted_gk = sorted(group_aucs.items(), key=lambda kv: np.nanmean(kv[1]))
    bar_labels, bar_means, bar_stds, bar_cols_g = [], [], [], []
    lsml_mean = np.nanmean(lsml_aucs)
    for gi, (key, aucs) in enumerate(sorted_gk):
        valid = [a for a in aucs if not np.isnan(a)]
        bar_labels.append('\n'.join(sorted(key)))
        bar_means.append(np.nanmean(aucs))
        bar_stds.append(np.nanstd(aucs))
        bar_cols_g.append(palette[gi % len(palette)])
    y_pos = range(len(bar_labels))
    ax.barh(list(y_pos), bar_means, xerr=bar_stds, color=bar_cols_g,
            alpha=0.85, capsize=3, align='center')
    ax.axvline(0.5, color='gray', linestyle=':', lw=1, label='Chance')
    ax.axvline(lsml_mean, color='black', linestyle='--', lw=1.5,
               label=f'L-SML full ({lsml_mean:.3f})')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(bar_labels, fontsize=6)
    ax.set_xlabel('Mean AUROC (+/- 1 std, across cells)')
    ax.set_title(f'Per-group AUROC\n(virtual classifier xi_g, {n_valid} cells)')
    ax.legend(fontsize=7)
    ax.set_xlim(0.45, 1.0)

    plt.suptitle(f'L-SML Feature Cluster Analysis ({m} features)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'lsml_cluster_{tag}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved -> {out_path}')


# ── Analysis 2: trace_length binarization investigation ───────────────────────

def analyze_trace_length(all_cells, data_dir, out_dir):
    # Load quantile optimization curves saved by the Optimized notebook Cell 5
    quant_curves = None
    quant_all = {}
    for search in [os.path.join(data_dir, 'consolidated_results'), data_dir]:
        path = os.path.join(search, 'lsml_opt_quantiles.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                saved = pickle.load(f)
            quant_curves = saved.get('curves', {})
            quant_all = saved.get('quantiles', {})
            print(f'Loaded quantile curves from {path}')
            print(f'  trace_length optimal quantile: {quant_all.get("trace_length", "N/A")}')
            break
    if quant_curves is None:
        print('WARNING: lsml_opt_quantiles.pkl not found — Panel 3 will be empty.')

    # Per-cell trace_length stats
    tl_stats = {}
    for cell_key, (fd, lbl) in all_cells.items():
        if 'trace_length' not in fd or len(set(lbl.tolist())) < 2:
            continue
        tl = np.array(fd['trace_length'], dtype=float)
        med = np.median(tl)
        binary = np.where(tl > med, 1.0, -1.0)
        frac_pos = float((binary == 1.0).mean())
        n_at_med = int(np.sum(tl == med))

        raw_a, *_  = boot_auc(lbl, tl)
        bin_a, *_  = boot_auc(lbl, binary)

        opt_q = quant_all.get('trace_length', 0.5)
        tl_opt = np.where(tl > np.quantile(tl, opt_q), 1.0, -1.0)
        opt_a, *_ = boot_auc(lbl, tl_opt)

        tl_stats[cell_key] = dict(
            median=med, max=tl.max(), std=tl.std(),
            frac_pos=frac_pos, pct_at_med=n_at_med / len(tl),
            raw_auc=float(raw_a), bin_auc=float(bin_a),
            opt_q=opt_q, opt_auc=float(opt_a),
        )

    # Print table
    print(f'\n=== trace_length binarization investigation ({len(tl_stats)} cells) ===')
    print(f'{"Cell":<52} {"Med":>5} {"Pct@med":>8} {"Frac+":>6} '
          f'{"RawAUC":>8} {"BinAUC(0.5)":>12} {"BinAUC(q*)":>12}')
    print('-' * 110)
    for ck, s in sorted(tl_stats.items()):
        print(f'{ck:<52} {s["median"]:>5.0f} {s["pct_at_med"]:>8.1%} '
              f'{s["frac_pos"]:>6.2f} {s["raw_auc"]:>8.3f} '
              f'{s["bin_auc"]:>12.3f} {s["opt_auc"]:>12.3f}  (q*={s["opt_q"]:.2f})')

    raw_m  = np.mean([s['raw_auc']  for s in tl_stats.values()])
    bin_m  = np.mean([s['bin_auc']  for s in tl_stats.values()])
    opt_m  = np.mean([s['opt_auc']  for s in tl_stats.values()])
    frac_m = np.mean([s['frac_pos'] for s in tl_stats.values()])
    frac_s = np.std([s['frac_pos']  for s in tl_stats.values()])
    print(f'\nMeans across {len(tl_stats)} cells:')
    print(f'  Raw (continuous) AUROC     : {raw_m:.3f}')
    print(f'  Binarized at median AUROC  : {bin_m:.3f}')
    print(f'  Binarized at q* AUROC      : {opt_m:.3f}  (q*={quant_all.get("trace_length", 0.5):.2f})')
    print(f'  Mean fraction +1 (want 0.50): {frac_m:.3f} ± {frac_s:.3f}')
    opt_q = quant_all.get('trace_length', 0.5)
    rescued = opt_m >= MIN_IND_AUC_THRESHOLD
    print(f'\nConclusion: at q*={opt_q:.2f}, mean binarized AUROC = {opt_m:.3f} — '
          f'{"rescue works" if rescued else f"still below {MIN_IND_AUC_THRESHOLD:.2f}; feature is not recoverable with a quantile shift alone"}.')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # Panel 1: Histograms for representative cells
    ax = axes[0]
    rep_cells = [
        ('math500/Qwen-Math-7B_T1.0', '#2196F3'),
        ('gsm8k/Llama-8B_T1.0',       '#4CAF50'),
        ('gpqa/Mistral-7B_T1.0',       '#FF9800'),
        ('rag/Qwen-7B/hotpotqa',       '#9C27B0'),
    ]
    for ck, col in rep_cells:
        if ck not in all_cells:
            continue
        tl = np.array(all_cells[ck][0].get('trace_length', []), dtype=float)
        ax.hist(tl, bins=25, alpha=0.55, label=ck.split('/')[-1], color=col, density=True)
        ax.axvline(np.median(tl), color=col, linestyle='--', lw=1.5)
    ax.set_xlabel('trace_length (tokens)')
    ax.set_ylabel('Density')
    ax.set_title('Trace-length distributions\n(dashed = median; note right-censoring at cap)')
    ax.legend(fontsize=7)

    # Panel 2: Fraction +1 per cell — should be ~50%
    ax = axes[1]
    cell_names = list(tl_stats.keys())
    frac_vals  = [tl_stats[k]['frac_pos'] for k in cell_names]
    short_names = [k.split('/')[-1][:22] for k in cell_names]
    bar_cols   = [DOMAIN_COLORS.get(k.split('/')[0], 'gray') for k in cell_names]
    ax.bar(range(len(frac_vals)), frac_vals, color=bar_cols, alpha=0.85)
    ax.axhline(0.50, color='crimson', linestyle='--', lw=1.5, label='Ideal 50%')
    ax.set_xticks(range(len(frac_vals)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Fraction +1 after median binarization')
    ax.set_title('trace_length: class balance\n(<< 50% = degenerate from integer ties at cap)')
    ax.set_ylim(0, 0.75)
    for dom, col in DOMAIN_COLORS.items():
        ax.bar([], [], color=col, alpha=0.85, label=dom)
    ax.legend(fontsize=7)

    # Panel 3: AUROC vs quantile from lsml_opt_quantiles.pkl
    ax = axes[2]
    if quant_curves and 'trace_length' in quant_curves:
        QUANTILE_GRID = np.arange(0.35, 0.66, 0.05)
        ax.plot(QUANTILE_GRID, quant_curves['trace_length'], 'o-',
                color='steelblue', ms=6, lw=2, label='trace_length')
        if 'epr' in quant_curves:
            ax.plot(QUANTILE_GRID, quant_curves['epr'], 's--',
                    color='darkorange', ms=5, lw=1.5, alpha=0.8,
                    label='epr (in GOOD_FEATURES, for comparison)')
        ax.axvline(0.50, color='gray', linestyle=':', lw=1.5, label='median q=0.50')
        ax.axvline(opt_q, color='steelblue', linestyle='--', lw=1.5,
                   label=f'trace_length q*={opt_q:.2f}')
        ax.axhline(MIN_IND_AUC_THRESHOLD, color='crimson', linestyle=':', lw=1.2,
                   label=f'Keep threshold={MIN_IND_AUC_THRESHOLD:.2f}')
        ax.set_xlabel('Binarization quantile')
        ax.set_ylabel('Mean individual AUROC (across all cells)')
        ax.set_title('trace_length: AUROC vs quantile\n(does a lower threshold rescue it?)')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'lsml_opt_quantiles.pkl not found\n(run Colab Cell 5 first)',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title('trace_length: AUROC vs quantile')

    plt.suptitle('trace_length — Median Binarization Failure Diagnosis',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'trace_length_investigation.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved -> {out_path}')


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='./local_cache',
                        help='Directory containing the cached feature pkls (default: ./local_cache)')
    parser.add_argument('--out-dir', default='./local_cache',
                        help='Where to save output PNGs (default: ./local_cache)')
    parser.add_argument('--analysis', choices=['clusters', 'trace_length', 'both'],
                        default='both', help='Which analysis to run (default: both)')
    parser.add_argument('--features', nargs='+', metavar='FEAT',
                        help=('Feature subset for cluster analysis. '
                              'Use "all" for all 16, "good" for GOOD_FEATURES (default), '
                              'or list names explicitly: --features epr sw_var_peak cusum_max'))
    args = parser.parse_args()

    # Resolve feature list
    if args.features is None or args.features == ['good']:
        features = GOOD_FEATURES
    elif args.features == ['all']:
        features = FEAT_NAMES
    else:
        unknown = [f for f in args.features if f not in FEAT_NAMES]
        if unknown:
            parser.error(f'Unknown feature(s): {unknown}. Valid names: {FEAT_NAMES}')
        if len(args.features) < 3:
            parser.error('L-SML requires at least 3 features.')
        features = args.features

    os.makedirs(args.out_dir, exist_ok=True)
    all_cells = load_all_cells(args.data_dir)

    if args.analysis in ('clusters', 'both'):
        analyze_clusters(all_cells, args.out_dir, features=features)

    if args.analysis in ('trace_length', 'both'):
        analyze_trace_length(all_cells, args.data_dir, args.out_dir)


if __name__ == '__main__':
    main()
