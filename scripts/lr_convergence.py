"""
LR convergence vs feature count — how does the supervised oracle behave as you
add features, and does adding features actually help?

Answers Bracha's "5 features performs best / surprisingly close to unsupervised"
question directly. The three named sets (5/9/16) are NON-nested (the 9-feat set
drops spectral_entropy that the 5-feat set has), so a clean convergence curve
needs a proper nested sequence. We build ONE global feature ranking over the 16
features (by mean in-sample univariate AUROC across cells) and add features
most-informative-first, k = 3..16.

For each k, per LR-valid common cell, we compute the corrected balanced 5-fold CV
AUROC (`bal_cv`) and the balanced in-sample ceiling (`bal_in`) using the exact
same helpers as the main oracle table (no reimplementation of the CV protocol),
then macro-average across cells. The story: CV plateaus after ~5-6 features while
the in-sample ceiling keeps climbing -> the extra features are fit-able but not
generalizable, which is why 5 is nominally best and 5 ~ 16.

Usage:
    python scripts/lr_convergence.py [--data-dir ./local_cache]

Writes:
    results/lr_convergence.pkl   (ranking + per-k macro/per-domain curves)
    results/lr_convergence.png   (panel a: macro CV vs ceiling; panel b: per-domain)
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPTS_DIR)
for _p in (SCRIPTS_DIR, REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the corrected oracle helpers — single source of truth for the CV protocol.
from logistic_oracle import (
    iter_cells, build_X, lr_oracle_auc_variants, safe_auc_raw,
    ALL_H16, GOOD_5, STABLE_H9, is_saturated,
)

RES_PKL = os.path.join(REPO_DIR, 'results', 'lr_convergence.pkl')
RES_PNG = os.path.join(REPO_DIR, 'results', 'lr_convergence.png')
ORACLE_PKL = os.path.join(REPO_DIR, 'results', 'logistic_oracle.pkl')

MIN_PER_CLASS = 5   # matches run_cell's guard in logistic_oracle.py
MIN_FEATS = 3       # build_X requires >= 3 available features

# Domain grouping used in the advisor email.
DOMAIN_GROUPS = {
    'Reasoning': ('math500', 'gsm8k'),
    'GPQA':      ('gpqa',),
    'RAG+QA':    ('rag', 'qa'),
}


def domain_of(cell):
    return cell.split('/', 1)[0]


def load_valid_cells(data_dir):
    """Collect (cell_name, fd, labels) for cells usable by the LR oracle."""
    cells = []
    for cell_name, fd, lbl in iter_cells(data_dir, verbose=False):
        lbl = np.asarray(lbl, dtype=int)
        n_pos = int(lbl.sum())
        n_neg = len(lbl) - n_pos
        if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
            continue
        cells.append((cell_name, fd, lbl))
    return cells


def rank_features(cells):
    """Global feature ranking by mean in-sample univariate AUROC across cells.

    Uses safe_auc_raw (= max(auc, 1-auc)) so orientation is irrelevant. Features
    that are unavailable/saturated in a cell simply do not contribute there.
    """
    scores = {f: [] for f in ALL_H16}
    for _, fd, lbl in cells:
        n = len(lbl)
        for f in ALL_H16:
            if f in fd and fd[f] is not None and len(fd[f]) == n and not is_saturated(fd[f]):
                arr = np.asarray(fd[f], dtype=float)
                arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
                scores[f].append(safe_auc_raw(lbl, arr))
    mean_auc = {f: (float(np.mean(v)) if v else 0.5) for f, v in scores.items()}
    ranked = sorted(ALL_H16, key=lambda f: mean_auc[f], reverse=True)
    return ranked, mean_auc


def sweep(cells, ranked):
    """For k = MIN_FEATS..16, collect per-cell bal_cv + bal_in on ranked[:k]."""
    ks = list(range(MIN_FEATS, len(ranked) + 1))
    per_k = {k: {'cv': [], 'in': [], 'cells': [],
                 'cv_by_dom': {g: [] for g in DOMAIN_GROUPS},
                 'in_by_dom': {g: [] for g in DOMAIN_GROUPS}} for k in ks}

    def group_of(cell):
        d = domain_of(cell)
        for g, doms in DOMAIN_GROUPS.items():
            if d in doms:
                return g
        return None

    for cell_name, fd, lbl in cells:
        n = len(lbl)
        g = group_of(cell_name)
        for k in ks:
            X, avail = build_X(fd, ranked[:k], n)
            if X is None:            # < MIN_FEATS available at this k for this cell
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    v = lr_oracle_auc_variants(X, lbl, n_boot=0, compute_legacy=False)
            except Exception:
                continue
            cv = v['bal_cv'][0]
            ceil = v['bal_in']
            per_k[k]['cv'].append(cv)
            per_k[k]['in'].append(ceil)
            per_k[k]['cells'].append(cell_name)
            if g is not None:
                per_k[k]['cv_by_dom'][g].append(cv)
                per_k[k]['in_by_dom'][g].append(ceil)
    return ks, per_k


def named_set_overlay():
    """Read the named 5/9/16-set common-cell macros from logistic_oracle.pkl.

    Returns {'cv': {k: macro}, 'in': {k: macro}} on the same common-cell basis as
    Part A, so the markers line up with the reported table. Missing pkl -> None.
    """
    if not os.path.exists(ORACLE_PKL):
        return None
    with open(ORACLE_PKL, 'rb') as f:
        rows = [r for r in pickle.load(f) if not r.get('skipped')]
    out = {'cv': {}, 'in': {}}
    for k in ('5', '9', '16'):
        cc = [r for r in rows
              if r.get(f'cont_{k}') is not None and r.get(f'lr_{k}') is not None]
        cv = [r[f'lr_{k}'] for r in cc]
        ceil = [r[f'lr_bal_in_{k}'] for r in cc if r.get(f'lr_bal_in_{k}') is not None]
        out['cv'][int(k)] = float(np.mean(cv)) if cv else None
        out['in'][int(k)] = float(np.mean(ceil)) if ceil else None
    return out


def macro_band(vals):
    a = np.asarray(vals, dtype=float)
    return (float(np.mean(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)),
            len(a))


def make_plot(ks, per_k, overlay, ranked, mean_auc, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('[plot] matplotlib not available, skipping')
        return

    cv_mean = [macro_band(per_k[k]['cv'])[0] for k in ks]
    cv_lo   = [macro_band(per_k[k]['cv'])[1] for k in ks]
    cv_hi   = [macro_band(per_k[k]['cv'])[2] for k in ks]
    in_mean = [macro_band(per_k[k]['in'])[0] for k in ks]
    in_lo   = [macro_band(per_k[k]['in'])[1] for k in ks]
    in_hi   = [macro_band(per_k[k]['in'])[2] for k in ks]

    ks_arr = np.array(ks)
    fig = plt.figure(figsize=(19, 6.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.66], wspace=0.26, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    axt = fig.add_subplot(gs[0, 2])
    axt.axis('off')

    # ── Panel (a): macro CV vs in-sample ceiling ─────────────────────────────
    ax0.fill_between(ks_arr, np.array(in_lo) * 100, np.array(in_hi) * 100,
                     color='#E91E63', alpha=0.12)
    ax0.plot(ks_arr, np.array(in_mean) * 100, '-o', color='#E91E63', lw=2,
             label='In-sample ceiling (bal_in)', markersize=4)
    ax0.fill_between(ks_arr, np.array(cv_lo) * 100, np.array(cv_hi) * 100,
                     color='#2196F3', alpha=0.15)
    ax0.plot(ks_arr, np.array(cv_mean) * 100, '-o', color='#2196F3', lw=2,
             label='5-fold CV (bal_cv)', markersize=4)

    for xk in (5, 9, 16):
        ax0.axvline(xk, color='gray', ls='--', lw=0.8, alpha=0.6)
    if overlay is not None:
        xs_cv = [k for k in (5, 9, 16) if overlay['cv'].get(k) is not None]
        ys_cv = [overlay['cv'][k] * 100 for k in xs_cv]
        ax0.scatter(xs_cv, ys_cv, marker='s', s=90, facecolors='none',
                    edgecolors='#0D47A1', linewidths=1.8, zorder=5,
                    label='named-set CV (5/9/16)')
        xs_in = [k for k in (5, 9, 16) if overlay['in'].get(k) is not None]
        ys_in = [overlay['in'][k] * 100 for k in xs_in]
        ax0.scatter(xs_in, ys_in, marker='s', s=90, facecolors='none',
                    edgecolors='#880E4F', linewidths=1.8, zorder=5,
                    label='named-set ceiling (5/9/16)')

    ax0.set_xlabel('Number of features (nested top-k, ranks 1..k — see table at right)')
    ax0.set_ylabel('Macro AUROC (%)')
    ax0.set_title('LR convergence: CV plateaus while the in-sample ceiling keeps rising\n'
                  '(gap = overfitting: extra features are fit-able but not generalizable)',
                  fontsize=10)
    ax0.set_xticks(ks_arr)
    ax0.legend(fontsize=8, loc='lower right')
    ax0.grid(alpha=0.25)

    # ── Panel (b): per-domain CV convergence ─────────────────────────────────
    dom_colors = {'Reasoning': '#4CAF50', 'GPQA': '#9C27B0', 'RAG+QA': '#00838F'}
    for g, col in dom_colors.items():
        ys = []
        for k in ks:
            v = per_k[k]['cv_by_dom'][g]
            ys.append(np.mean(v) * 100 if v else np.nan)
        ax1.plot(ks_arr, ys, '-o', color=col, lw=2, markersize=4, label=g)
    for xk in (5, 9, 16):
        ax1.axvline(xk, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax1.axhline(50, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax1.set_xlabel('Number of features (nested top-k, ranks 1..k — see table at right)')
    ax1.set_ylabel('LR CV macro AUROC (%)')
    ax1.set_title('Per-domain CV convergence\n'
                  'reasoning saturates high; GPQA/RAG plateau low = features are the bottleneck',
                  fontsize=10)
    ax1.set_xticks(ks_arr)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(alpha=0.25)

    # ── Right column: ranked feature table + named-set membership ─────────────
    # Makes "k features" unambiguous (k = ranks 1..k) and shows how the named
    # CONT/LR sets differ from the nested top-k: GOOD_5 swaps rank 5 for rank 6,
    # STABLE_H9 omits spectral_entropy (rank 3). ● = in set, · = not in set.
    g5, h9 = set(GOOD_5), set(STABLE_H9)

    def _row(i, feat, auc, a, b):
        return f'{i:>2}  {feat:<22}{auc:>6}   {a:^3}{b:^3}'

    tbl = [_row('k', 'feature', 'AUROC', 'G5', 'H9'), ' ' + '-' * 40]
    for i, f in enumerate(ranked, 1):
        tbl.append(_row(i, f, f'{100 * mean_auc[f]:.1f}%',
                        '●' if f in g5 else '·',
                        '●' if f in h9 else '·'))
    axt.text(0.0, 1.0, '\n'.join(tbl), family='monospace', fontsize=8.4,
             va='top', ha='left', transform=axt.transAxes)
    axt.set_title('Feature entry order (ranked by univariate AUROC)\n'
                  'G5 = GOOD_5 (5-feat) · H9 = STABLE_H9 (9-feat) · ● in set, · not',
                  fontsize=9, loc='left')

    fig.suptitle('LR oracle convergence vs feature count (nested, ranked by univariate AUROC)',
                 fontsize=12, fontweight='bold', y=0.985)
    # Manual margins (not tight_layout — it can't handle the text axis and warns).
    fig.subplots_adjust(left=0.05, right=0.995, top=0.86, bottom=0.185, wspace=0.24)
    caption = (
        'Curve at feature-count k = the NESTED top-k (ranks 1..k in the table).   '
        'Open squares = the named sets from the CONT / LR oracle table.\n'
        'GOOD_5 = ranks 1-4 + 6 (low_band_power instead of the rank-5 stft_spectral_entropy).    '
        'STABLE_H9 omits spectral_entropy (rank 3), reaching down to ranks 10-14\n'
        'so the k=9 named square sits below the nested curve (it drops a top-3 feature), '
        'while the k=5 square is slightly above it.'
    )
    fig.text(0.34, 0.075, caption, ha='center', va='top', fontsize=8, color='#333333')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./local_cache')
    args = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    cells = load_valid_cells(data_dir)
    print(f'LR-valid cells: {len(cells)}')
    if not cells:
        print('No usable cells found.')
        sys.exit(1)

    ranked, mean_auc = rank_features(cells)
    print('\nGlobal feature ranking (mean in-sample univariate AUROC across cells):')
    for i, f in enumerate(ranked, 1):
        print(f'  {i:>2}. {f:<24} {100 * mean_auc[f]:.1f}%')

    ks, per_k = sweep(cells, ranked)
    overlay = named_set_overlay()

    print('\nMacro AUROC vs #features (common LR-valid cells):')
    print(f"  {'k':>3} | {'CV':>7} | {'ceiling':>7} | {'gap':>6} | {'#cells':>6}")
    print('  ' + '-' * 44)
    for k in ks:
        cv_m, _, _, n_cv = macro_band(per_k[k]['cv'])
        in_m, _, _, _ = macro_band(per_k[k]['in'])
        gap = in_m - cv_m
        star = '  <--' if k in (3, 5, 9, 16) else ''
        print(f"  {k:>3} | {100*cv_m:>6.1f}% | {100*in_m:>6.1f}% | "
              f"{100*gap:>+5.1f} | {n_cv:>6}{star}")

    out = {
        'ranked': ranked, 'mean_auc': mean_auc, 'ks': ks,
        'per_k': {k: {'cv': per_k[k]['cv'], 'in': per_k[k]['in'],
                      'cells': per_k[k]['cells'],
                      'cv_by_dom': per_k[k]['cv_by_dom'],
                      'in_by_dom': per_k[k]['in_by_dom']} for k in ks},
        'overlay': overlay, 'domain_groups': DOMAIN_GROUPS,
    }
    os.makedirs(os.path.dirname(RES_PKL), exist_ok=True)
    with open(RES_PKL, 'wb') as f:
        pickle.dump(out, f)
    print(f'\nSaved results -> {RES_PKL}')

    make_plot(ks, per_k, overlay, ranked, mean_auc, RES_PNG)


if __name__ == '__main__':
    main()
