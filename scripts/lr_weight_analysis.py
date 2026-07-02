"""
LR weights vs L-SML weights — do the supervised and unsupervised methods agree on
which features matter? (Bracha Q4.)

For each eval cell and feature set (lead with 16-feat, which has the most groups
and least degenerate weights) we compute two per-feature weight vectors:

  * LR weights: balanced LR (StandardScaler -> LogisticRegression(class_weight=
    'balanced')) fit on the full cell; coef_ is on standardized features, so the
    entries are directly comparable in scale.
  * L-SML composite weights: from lsml_continuous_pipeline's meta_dict, the
    EFFECTIVE per-feature coefficient in the fused score,
        composite[i] = cross_weights[group(i)] * within_group_weight[i].
    This is exactly the linear weight each oriented + z-scored feature receives.

Agreement per cell: Spearman rho of |LR coef| vs |L-SML composite| (headline,
scale-free rank agreement on importance), sign-aligned Pearson, top-3 overlap,
plus a free cross-check against U-PCR's rho-hat (estimated Cov(feature, Y)) read
from upcr_comparison.pkl.

This script also CAPTURES AND SAVES the L-SML internals that were never persisted
before (K, cluster assignment c, group_weights, cross_weights) and prints a
verification table, so the "5-feat collapses to K=2 with ~0.5/0.5 cross-weights"
claim can be checked rather than assumed.

Usage:
    python scripts/lr_weight_analysis.py [--data-dir ./local_cache]

Writes:
    results/lr_weight_analysis.pkl    (per-cell weight vectors + correlations + L-SML meta)
    results/lr_weight_agreement.png
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

from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from logistic_oracle import iter_cells, build_X, FEATURE_SETS
from run_upcr_comparison import FEATURE_SIGNS
from spectral_utils import lsml_continuous_pipeline

RES_PKL = os.path.join(REPO_DIR, 'results', 'lr_weight_analysis.pkl')
RES_PNG = os.path.join(REPO_DIR, 'results', 'lr_weight_agreement.png')
CONT_PKL = os.path.join(REPO_DIR, 'results', 'upcr_comparison.pkl')

MIN_PER_CLASS = 5
FEAT_SETS = ['5', '9', '16']

DOMAIN_GROUPS = {
    'Reasoning': ('math500', 'gsm8k'),
    'GPQA':      ('gpqa',),
    'RAG+QA':    ('rag', 'qa'),
}


def domain_of(cell):
    return cell.split('/', 1)[0]


def group_of(cell):
    d = domain_of(cell)
    for g, doms in DOMAIN_GROUPS.items():
        if d in doms:
            return g
    return None


def load_upcr_lookup():
    if not os.path.exists(CONT_PKL):
        return {}
    with open(CONT_PKL, 'rb') as f:
        rows = pickle.load(f)
    return {r['cell']: r for r in rows if not r.get('skipped')}


def fit_lr_coef(X, y):
    """Balanced LR (same config as the oracle); return coef_ on standardized features."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs'),
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipe.fit(X, y)
    return pipe.named_steps['logisticregression'].coef_[0]


def reconstruct_composite(meta, m):
    """Effective per-feature weight in the L-SML fused score, aligned to columns 0..m-1.

    composite[col] = cross_weights[group_order] * within_group_weight[col].
    group_weights and cross_weights are both ordered by np.unique(c).
    """
    c = np.asarray(meta['c'])
    groups = list(np.unique(c))
    gw = meta['group_weights']
    cw = np.asarray(meta['cross_weights'], dtype=float)
    composite = np.zeros(m, dtype=float)
    for go, _g in enumerate(groups):
        idx, w = gw[go]
        idx = np.asarray(idx)
        w = np.asarray(w, dtype=float)
        for local, col in enumerate(idx):
            composite[int(col)] = cw[go] * w[local]
    return composite


def group_sizes(meta):
    c = np.asarray(meta['c'])
    return [int(np.sum(c == g)) for g in np.unique(c)]


def sign_agreement(lr_oriented, comp):
    """Fraction of features where LR (oriented) and L-SML composite share sign,
    after resolving L-SML's arbitrary global sign to maximise agreement."""
    lr_s = np.sign(lr_oriented)
    cp_s = np.sign(comp)
    agree = np.sum(lr_s == cp_s)
    flip = np.sum(lr_s == -cp_s)
    return float(max(agree, flip)) / len(lr_oriented)


def top_overlap(a, b, k=3):
    ta = set(np.argsort(-np.abs(a))[:k].tolist())
    tb = set(np.argsort(-np.abs(b))[:k].tolist())
    return len(ta & tb) / float(k)


def safe_corr(fn, x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = fn(x, y)[0]
    return float(r) if np.isfinite(r) else None


def analyze_cell(cell_name, fd, lbl, upcr_row):
    lbl = np.asarray(lbl, dtype=int)
    n = len(lbl)
    out = {'cell': cell_name, 'n': n, 'domain': domain_of(cell_name),
           'group': group_of(cell_name)}

    for k, feat_list in FEATURE_SETS.items():
        X, available = build_X(fd, feat_list, n)
        if X is None:
            out[k] = None
            continue
        m = len(available)
        feat_signs = {f: FEATURE_SIGNS.get(f, -1) for f in available}

        # LR coefficients (standardized-feature scale).
        lr_coef = fit_lr_coef(X, lbl)

        # L-SML composite weights + meta.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _score, meta = lsml_continuous_pipeline(fd, available, feat_signs)
        composite = reconstruct_composite(meta, m)

        # LR coef brought into the oriented space L-SML lives in.
        signs = np.array([feat_signs[f] for f in available], dtype=float)
        lr_oriented = lr_coef * signs

        # U-PCR rho-hat aligned to `available` (if present for this cell/set).
        rho = None
        if upcr_row is not None and upcr_row.get(f'upcrauto_rho_{k}') is not None:
            rmap = dict(zip(upcr_row.get(f'upcrauto_feats_{k}', []),
                            upcr_row.get(f'upcrauto_rho_{k}', [])))
            if all(f in rmap for f in available):
                rho = np.array([rmap[f] for f in available], dtype=float)

        rec = {
            'feats': available, 'm': m,
            'lr_coef': lr_coef.tolist(),
            'lr_oriented': lr_oriented.tolist(),
            'composite': composite.tolist(),
            'rho': rho.tolist() if rho is not None else None,
            # correlations
            'spearman_abs': safe_corr(spearmanr, np.abs(lr_coef), np.abs(composite)),
            'pearson_signed': safe_corr(pearsonr, lr_oriented, composite),
            'top3_overlap': top_overlap(lr_coef, composite, k=3),
            'sign_agree': sign_agreement(lr_oriented, composite),
            'spearman_lr_rho': (safe_corr(spearmanr, np.abs(lr_coef), np.abs(rho))
                                if rho is not None else None),
            # L-SML internals (never persisted before)
            'K': int(meta['K']),
            'n_groups': int(len(np.unique(meta['c']))),
            'cluster': np.asarray(meta['c']).tolist(),
            'group_sizes': group_sizes(meta),
            'cross_weights': np.asarray(meta['cross_weights'], dtype=float).tolist(),
        }
        out[k] = rec
    return out


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_meta_verification(results):
    print('\n' + '=' * 96)
    print('L-SML internals verification — K, #groups, group sizes, cross-weights (per cell)')
    print('=' * 96)
    for k in FEAT_SETS:
        print(f'\n--- feature set {k} ---')
        print(f"  {'cell':<44} | {'K':>2} {'#grp':>4} | {'group sizes':<16} | cross-weights")
        print('  ' + '-' * 92)
        Ks = []
        cw_two = []
        for r in sorted(results, key=lambda x: x['cell']):
            rec = r.get(k)
            if rec is None:
                continue
            Ks.append(rec['n_groups'])
            gs = ','.join(str(s) for s in rec['group_sizes'])
            cw = ', '.join(f'{w:+.3f}' for w in rec['cross_weights'])
            if rec['n_groups'] == 2:
                cw_two.append(np.abs(rec['cross_weights']))
            print(f"  {r['cell']:<44} | {rec['K']:>2} {rec['n_groups']:>4} | {gs:<16} | {cw}")
        if Ks:
            from collections import Counter
            dist = dict(Counter(Ks))
            print(f"  -> #groups distribution: {dist}")
            if cw_two:
                mean_cw = np.mean(np.vstack(cw_two), axis=0)
                print(f"  -> among 2-group cells: mean |cross-weight| = "
                      f"[{mean_cw[0]:.3f}, {mean_cw[1]:.3f}]  "
                      f"({'~0.5/0.5 CONFIRMED' if np.allclose(mean_cw, 0.5, atol=0.05) else 'NOT 0.5/0.5'})")


def print_corr_summary(results):
    print('\n' + '=' * 80)
    print('LR-vs-L-SML weight agreement — by domain (16-feat unless noted)')
    print('=' * 80)
    for k in FEAT_SETS:
        print(f'\n--- feature set {k} ---')
        print(f"  {'group':<12} | {'#cells':>6} | {'Spearman|w|':>11} | "
              f"{'sign-agree':>10} | {'top3':>5} | {'LR~rho':>7}")
        print('  ' + '-' * 66)
        for g in list(DOMAIN_GROUPS) + ['ALL']:
            recs = [r[k] for r in results if r.get(k) is not None
                    and (g == 'ALL' or r['group'] == g)]
            if not recs:
                continue
            sp = [x['spearman_abs'] for x in recs if x['spearman_abs'] is not None]
            sa = [x['sign_agree'] for x in recs if x['sign_agree'] is not None]
            t3 = [x['top3_overlap'] for x in recs if x['top3_overlap'] is not None]
            lr_rho = [x['spearman_lr_rho'] for x in recs if x['spearman_lr_rho'] is not None]
            def m(v): return f'{np.mean(v):.3f}' if v else '  N/A'
            print(f"  {g:<12} | {len(recs):>6} | {m(sp):>11} | {m(sa):>10} | "
                  f"{m(t3):>5} | {m(lr_rho):>7}")


# ── Plot ────────────────────────────────────────────────────────────────────────

def make_plot(results, out_path, k='16'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[plot] matplotlib not available, skipping')
        return

    recs = [(r['cell'], r['group'], r[k]) for r in results if r.get(k) is not None]
    if not recs:
        print('[plot] no records to plot')
        return

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(19, 6))

    # ── (a) scatter |LR coef| vs |composite|, colored by feature ──────────────
    all_feats = sorted({f for _, _, rec in recs for f in rec['feats']})
    cmap = plt.get_cmap('tab20')
    fcolor = {f: cmap(i % 20) for i, f in enumerate(all_feats)}
    for _cell, _g, rec in recs:
        lr = np.abs(np.array(rec['lr_coef']))
        cp = np.abs(np.array(rec['composite']))
        cols = [fcolor[f] for f in rec['feats']]
        ax0.scatter(cp, lr, c=cols, s=28, alpha=0.75, edgecolors='none')
    ax0.set_xlabel('|L-SML composite weight|')
    ax0.set_ylabel('|LR coefficient|')
    ax0.set_title(f'Per-(cell, feature) weight magnitudes ({k}-feat)\n'
                  'one point per feature per cell, colored by feature', fontsize=10)
    ax0.grid(alpha=0.25)
    handles = [plt.Line2D([0], [0], marker='o', ls='', color=fcolor[f], label=f)
               for f in all_feats]
    ax0.legend(handles=handles, fontsize=6, ncol=2, loc='upper right', framealpha=0.85)

    # ── (b) mean relative |weight| per feature (LR vs L-SML) ──────────────────
    def mean_rel(weight_key):
        acc = {f: [] for f in all_feats}
        for _cell, _g, rec in recs:
            w = np.abs(np.array(rec[weight_key], dtype=float))
            s = w.sum()
            if s <= 0:
                continue
            w = w / s
            for f, wi in zip(rec['feats'], w):
                acc[f].append(wi)
        return np.array([np.mean(acc[f]) if acc[f] else 0.0 for f in all_feats])

    lr_rel = mean_rel('lr_coef')
    cp_rel = mean_rel('composite')
    order = np.argsort(-(lr_rel + cp_rel))
    feats_o = [all_feats[i] for i in order]
    x = np.arange(len(feats_o))
    w = 0.4
    ax1.bar(x - w / 2, lr_rel[order] * 100, w, label='LR (|coef|, norm.)', color='#E91E63', alpha=0.85)
    ax1.bar(x + w / 2, cp_rel[order] * 100, w, label='L-SML (|composite|, norm.)', color='#2196F3', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(feats_o, rotation=55, ha='right', fontsize=7)
    ax1.set_ylabel('mean relative weight (%)')
    ax1.set_title(f'Which features each method leans on ({k}-feat)\n'
                  'per-cell normalized to unit sum, then averaged', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25, axis='y')

    # ── (c) per-cell Spearman(|LR|,|composite|) histogram, by domain ──────────
    dom_colors = {'Reasoning': '#4CAF50', 'GPQA': '#9C27B0', 'RAG+QA': '#00838F'}
    for g, col in dom_colors.items():
        vals = [rec['spearman_abs'] for _cell, gg, rec in recs
                if gg == g and rec['spearman_abs'] is not None]
        if vals:
            ax2.hist(vals, bins=np.linspace(-1, 1, 13), alpha=0.55, color=col,
                     label=f'{g} (n={len(vals)}, mean={np.mean(vals):+.2f})')
    ax2.axvline(0, color='black', ls='--', lw=1)
    ax2.set_xlabel('per-cell Spearman rho ( |LR coef| vs |L-SML composite| )')
    ax2.set_ylabel('cell count')
    ax2.set_title(f'Rank agreement on feature importance ({k}-feat)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis='y')

    fig.suptitle('LR weights vs L-SML composite weights — do supervised and unsupervised agree?',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./local_cache')
    args = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    upcr_lookup = load_upcr_lookup()
    results = []
    for cell_name, fd, lbl in iter_cells(data_dir, verbose=True):
        lbl = np.asarray(lbl, dtype=int)
        n_pos = int(lbl.sum())
        if n_pos < MIN_PER_CLASS or (len(lbl) - n_pos) < MIN_PER_CLASS:
            continue
        r = analyze_cell(cell_name, fd, lbl, upcr_lookup.get(cell_name))
        results.append(r)

    print_meta_verification(results)
    print_corr_summary(results)

    os.makedirs(os.path.dirname(RES_PKL), exist_ok=True)
    with open(RES_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved results -> {RES_PKL}')

    make_plot(results, RES_PNG, k='16')


if __name__ == '__main__':
    main()
