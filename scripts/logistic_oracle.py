"""
Logistic Regression Oracle — Item 2 from advisor meeting (Jun 17, 2026).

Fits supervised logistic regression (5-fold stratified OOF CV) on spectral
features to upper-bound what is achievable with these features when labels
are available. Compares against pre-computed CONT (L-SML) results.

Usage:
    python scripts/logistic_oracle.py [--data-dir ./local_cache] [--smoke-test]

Reads:  local_cache/{math500_res,gsm8k_res,gpqa_res,qa_res,rag_feats_all}.pkl
        results/upcr_comparison.pkl  (pre-computed CONT AUROCs — not recomputed here)
Writes: results/logistic_oracle.pkl
        results/logistic_oracle.png
"""

import os
import sys
import pickle
import warnings
import argparse
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spectral_utils import boot_auc, FEAT_NAMES

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ── Feature sets ──────────────────────────────────────────────────────────────

GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

STABLE_H9 = [
    'epr', 'low_band_power', 'high_band_power', 'hl_ratio',
    'spectral_centroid', 'sw_var_peak', 'rpdi', 'pe_mean', 'cusum_max',
]

ALL_H16 = FEAT_NAMES[:16]

FEATURE_SETS = {'5': GOOD_5, '9': STABLE_H9, '16': ALL_H16}

PKL_NAMES = {
    'math500': 'math500_res.pkl',
    'gsm8k':   'gsm8k_res.pkl',
    'gpqa':    'gpqa_res.pkl',
    'rag':     'rag_feats_all.pkl',
    'qa':      'qa_res.pkl',
}

CONT_PKL = os.path.join(REPO_DIR, 'results', 'upcr_comparison.pkl')
OUT_PKL  = os.path.join(REPO_DIR, 'results', 'logistic_oracle.pkl')
OUT_PNG  = os.path.join(REPO_DIR, 'results', 'logistic_oracle.png')

# ── Helpers ───────────────────────────────────────────────────────────────────

from sklearn.metrics import roc_auc_score

def safe_auc_raw(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(set(y_true.tolist())) < 2 or np.all(y_prob == y_prob[0]):
        return 0.5
    try:
        p = roc_auc_score(y_true, y_prob)
        return max(p, 1.0 - p)
    except:
        return 0.5

def cv_avg_auc_with_ci(pipe, X, y, skf, n_boot=1000):
    fold_targets = []
    fold_probs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        fold_targets.append(y_test)
        fold_probs.append(probs)
        
    fold_aucs = []
    for target, prob in zip(fold_targets, fold_probs):
        fold_aucs.append(safe_auc_raw(target, prob))
    base_auc = np.mean(fold_aucs)
    
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        boot_aucs = []
        for target, prob in zip(fold_targets, fold_probs):
            if len(target) < 2 or len(np.unique(target)) < 2:
                continue
            idx = rng.integers(0, len(target), len(target))
            boot_aucs.append(safe_auc_raw(target[idx], prob[idx]))
        if boot_aucs:
            boot_means.append(np.mean(boot_aucs))
            
    if not boot_means:
        return base_auc, base_auc, base_auc
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return base_auc, lo, hi

def safe_auc(lbl, scores):
    lbl    = np.asarray(lbl,    dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(set(lbl.tolist())) < 2 or np.all(scores == scores[0]):
        return 0.5, 0.5, 0.5
    p, pl, ph = boot_auc(lbl,  scores)
    n, nl, nh = boot_auc(lbl, -scores)
    return (p, pl, ph) if p >= n else (n, nl, nh)


def is_saturated(arr, threshold=0.40):
    a = np.asarray(arr, dtype=float)
    return float(np.mean(a == np.median(a))) > threshold


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cached_feats(pkl_path):
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'feats' in obj:
        return obj['feats']
    return obj


def load_cont_results():
    """Load pre-computed CONT AUROCs from upcr_comparison.pkl, keyed by cell name."""
    if not os.path.exists(CONT_PKL):
        print(f'[WARN] {CONT_PKL} not found — CONT column will be empty in table/plots')
        return {}
    with open(CONT_PKL, 'rb') as f:
        rows = pickle.load(f)
    return {r['cell']: r for r in rows if not r.get('skipped')}


def iter_cells(data_dir, verbose=True):
    """
    Yield (cell_name, feat_dict, labels) for every valid eval cell under data_dir.

    Single source of truth for the RAG/QA cache-nesting + schema detection so the
    oracle / convergence / weight-analysis scripts all iterate cells identically.
    Skips unknown-schema and single-class cells (they can't be evaluated).
    """
    for domain, pkl_name in PKL_NAMES.items():
        path = os.path.join(data_dir, pkl_name)
        feats = load_cached_feats(path)
        if feats is None:
            if verbose:
                print(f'[MISSING] {pkl_name}')
            continue
        if verbose:
            print(f'\n--- {domain.upper()} ({len(feats)} cells) ---')
        for cell_key, payload in feats.items():
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                fd, lbl = payload
            elif isinstance(payload, dict) and 'feats' in payload:
                fd, lbl = payload['feats'], payload['labels']
            else:
                if verbose:
                    print(f'  [{cell_key}] unknown schema — skip')
                continue
            lbl_arr = np.asarray(lbl, dtype=int)
            if len(set(lbl_arr.tolist())) < 2:
                if verbose:
                    print(f'  [{cell_key}] single class — skip')
                continue
            yield f'{domain}/{cell_key}', fd, lbl_arr


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_X(fd, feat_list, n_samples):
    """
    Stack available features into X of shape (n_samples, n_features).
    Returns (None, []) if fewer than 3 features are available.
    No sign orientation — LR learns the direction from labels.
    """
    available = [
        f for f in feat_list
        if f in fd and fd[f] is not None
        and len(fd[f]) == n_samples
        and not is_saturated(fd[f])
    ]
    if len(available) < 3:
        return None, available

    X = np.column_stack([np.asarray(fd[f], dtype=float) for f in available])
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = col_medians[j]
    return X, available


# ── LR oracle ─────────────────────────────────────────────────────────────────

def lr_oracle_auc_variants(X, y, n_boot=1000, compute_legacy=True):
    """
    Compute multiple supervised Logistic Regression AUROC variants.

    n_boot=0 skips the per-fold bootstrap CI (lo/hi collapse to the point
    estimate) — used by the feature-count convergence sweep, where the
    across-cell percentile band is the uncertainty that gets shown, so
    per-point CIs are wasted compute. compute_legacy=False skips the
    known-buggy concatenated-OOF reference variant for the same reason.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Standard CV (Averaged Fold CV)
    pipe_std = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'))
    auc_std, std_lo, std_hi = cv_avg_auc_with_ci(pipe_std, X, y, skf, n_boot=n_boot)

    # 2. Balanced CV (Averaged Fold CV)
    pipe_bal = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs'))
    auc_bal, bal_lo, bal_hi = cv_avg_auc_with_ci(pipe_bal, X, y, skf, n_boot=n_boot)

    # 3. Concatenated OOF (Legacy Standard) — buggy reference, optional
    if compute_legacy:
        pipe_legacy = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            oof_prob = cross_val_predict(pipe_legacy, X, y, cv=skf, method='predict_proba')[:, 1]
        legacy_auc, legacy_lo, legacy_hi = safe_auc(y, oof_prob)
    else:
        legacy_auc = legacy_lo = legacy_hi = None

    # 4. Standard In-Sample
    pipe_std.fit(X, y)
    preds_in_std = pipe_std.predict_proba(X)[:, 1]
    in_std_auc = safe_auc(y, preds_in_std)[0]
    
    # 5. Balanced In-Sample
    pipe_bal.fit(X, y)
    preds_in_bal = pipe_bal.predict_proba(X)[:, 1]
    in_bal_auc = safe_auc(y, preds_in_bal)[0]
    
    return {
        'std_cv': (auc_std, std_lo, std_hi),
        'bal_cv': (auc_bal, bal_lo, bal_hi),
        'legacy_cv': (legacy_auc, legacy_lo, legacy_hi),
        'std_in': in_std_auc,
        'bal_in': in_bal_auc
    }



# ── Per-cell runner ───────────────────────────────────────────────────────────

def run_cell(cell_name, fd, labels, cont_row):
    """Run LR oracle for all three feature sets. Pull CONT from pre-loaded cont_row."""
    labels = np.asarray(labels, dtype=int)
    n      = len(labels)
    n_pos  = int(labels.sum())
    n_neg  = n - n_pos

    cell_res = {
        'cell':       cell_name,
        'n':          n,
        'prevalence': float(n_pos / n) if n > 0 else 0.0,
    }

    for fs_name, feat_list in FEATURE_SETS.items():
        cont_auc = cont_row.get(f'cont_{fs_name}') if cont_row else None
        cell_res[f'cont_{fs_name}'] = cont_auc

        if n_pos < 5 or n_neg < 5:
            for key in (f'lr_{fs_name}', f'lr_ci_lo_{fs_name}', f'lr_ci_hi_{fs_name}',
                        f'delta_{fs_name}', f'n_avail_{fs_name}',
                        f'lr_std_cv_{fs_name}', f'lr_bal_cv_{fs_name}',
                        f'lr_legacy_cv_{fs_name}', f'lr_std_in_{fs_name}',
                        f'lr_bal_in_{fs_name}'):
                cell_res[key] = None
            continue

        X, available = build_X(fd, feat_list, n)
        cell_res[f'n_avail_{fs_name}'] = len(available)

        if X is None:
            for key in (f'lr_{fs_name}', f'lr_ci_lo_{fs_name}',
                        f'lr_ci_hi_{fs_name}', f'delta_{fs_name}',
                        f'lr_std_cv_{fs_name}', f'lr_bal_cv_{fs_name}',
                        f'lr_legacy_cv_{fs_name}', f'lr_std_in_{fs_name}',
                        f'lr_bal_in_{fs_name}'):
                cell_res[key] = None
            continue

        try:
            vars_dict = lr_oracle_auc_variants(X, labels)
            lr_auc, lr_lo, lr_hi = vars_dict['bal_cv']  # Use Balanced CV as the primary lr oracle score
        except Exception:
            vars_dict = None
            lr_auc = lr_lo = lr_hi = None

        cell_res[f'lr_{fs_name}']       = lr_auc
        cell_res[f'lr_ci_lo_{fs_name}'] = lr_lo
        cell_res[f'lr_ci_hi_{fs_name}'] = lr_hi
        cell_res[f'delta_{fs_name}']    = (
            (lr_auc - cont_auc)
            if (lr_auc is not None and cont_auc is not None)
            else None
        )
        
        # Store other variants
        if vars_dict:
            cell_res[f'lr_std_cv_{fs_name}'] = vars_dict['std_cv'][0]
            cell_res[f'lr_bal_cv_{fs_name}'] = vars_dict['bal_cv'][0]
            cell_res[f'lr_legacy_cv_{fs_name}'] = vars_dict['legacy_cv'][0]
            cell_res[f'lr_std_in_{fs_name}'] = vars_dict['std_in']
            cell_res[f'lr_bal_in_{fs_name}'] = vars_dict['bal_in']
        else:
            cell_res[f'lr_std_cv_{fs_name}'] = None
            cell_res[f'lr_bal_cv_{fs_name}'] = None
            cell_res[f'lr_legacy_cv_{fs_name}'] = None
            cell_res[f'lr_std_in_{fs_name}'] = None
            cell_res[f'lr_bal_in_{fs_name}'] = None

    return cell_res


# ── Table printer ─────────────────────────────────────────────────────────────

def print_table(results):
    cols = ['5', '9', '16']
    hdr = (f"{'Cell':<36} | "
           + ' | '.join(f"{'CONT-'+k:>8} {'LR-'+k:>8} {'D-'+k:>6}" for k in cols))
    sep = '-' * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    accum_cont = {k: [] for k in cols}
    accum_lr   = {k: [] for k in cols}

    for r in results:
        if r.get('skipped'):
            continue
        parts = []
        for k in cols:
            c   = r.get(f'cont_{k}')
            lr  = r.get(f'lr_{k}')
            d   = r.get(f'delta_{k}')
            cs  = f'{100*c:.1f}%'  if c  is not None else '   N/A'
            lrs = f'{100*lr:.1f}%' if lr is not None else '   N/A'
            ds  = f'{100*d:+.1f}'  if d  is not None else '  N/A'
            parts.append(f'{cs:>8} {lrs:>8} {ds:>6}')
            # Common-cell basis: only count a cell in the macro when BOTH CONT and
            # LR exist for this feature set. Otherwise CONT-only cells (e.g. the
            # trivia_qa_traces cell where LR=N/A) inflate the CONT macro and
            # understate the supervised gap by ~1pp.
            if c is not None and lr is not None:
                accum_cont[k].append(c)
                accum_lr[k].append(lr)
        print(f"  {r['cell']:<34} | " + ' | '.join(parts))

    print(sep)
    macro_parts = []
    for k in cols:
        mc  = np.mean(accum_cont[k]) if accum_cont[k] else None
        mlr = np.mean(accum_lr[k])   if accum_lr[k]   else None
        md  = (mlr - mc) if (mc is not None and mlr is not None) else None
        cs  = f'{100*mc:.1f}%'  if mc  is not None else '   N/A'
        lrs = f'{100*mlr:.1f}%' if mlr is not None else '   N/A'
        ds  = f'{100*md:+.1f}'  if md  is not None else '  N/A'
        macro_parts.append(f'{cs:>8} {lrs:>8} {ds:>6}')
    print(f"  {'MACRO':<34} | " + ' | '.join(macro_parts))
    print(sep)


# ── Visualization ─────────────────────────────────────────────────────────────

DOMAIN_COLORS = {
    'math500': '#4CAF50',
    'gsm8k':   '#FF9800',
    'gpqa':    '#9C27B0',
    'rag':     '#00BCD4',
    'qa':      '#795548',
}
COLOR_CONT = '#2196F3'
COLOR_LR   = '#E91E63'


def make_plots(results, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('[plot] matplotlib not available, skipping')
        return

    cols       = ['5', '9', '16']
    labels_map = {'5': '5-feat (GOOD_5)', '9': '9-feat (STABLE_H9)', '16': '16-feat (ALL_H16)'}
    valid      = [r for r in results if not r.get('skipped')]

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ── Row 0: Macro AUROC bar chart ─────────────────────────────────────────
    for ci, fs in enumerate(cols):
        ax = fig.add_subplot(gs[0, ci])
        # Common-cell basis: average CONT and LR over the SAME cells (those where
        # both scores exist). Averaging CONT over cells LR can't score (single-
        # class → no CV) inflates the CONT macro and understates the gap ~1pp.
        # This keeps the bars consistent with the per-cell headroom histogram
        # below and with oracle_report.py / oracle_feature_count.png.
        common    = [r for r in valid
                     if r.get(f'cont_{fs}') is not None and r.get(f'lr_{fs}') is not None]
        cont_vals = [r[f'cont_{fs}'] for r in common]
        lr_vals   = [r[f'lr_{fs}']   for r in common]
        macro_c   = np.mean(cont_vals) * 100 if cont_vals else 0.0
        macro_lr  = np.mean(lr_vals)   * 100 if lr_vals   else 0.0

        bars = ax.bar([0, 1], [macro_c, macro_lr],
                      color=[COLOR_CONT, COLOR_LR], width=0.5, alpha=0.85)
        ax.bar_label(bars, fmt='%.1f%%', padding=2, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['CONT\n(L-SML, unsup.)', 'LR Oracle\n(supervised)'], fontsize=9)
        ymin = max(0,   min(macro_c, macro_lr) - 4)
        ymax = min(100, max(macro_c, macro_lr) + 6)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('Macro AUROC (%)')
        ax.set_title(labels_map[fs], fontsize=10, fontweight='bold')
        ax.axhline(50, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        delta = macro_lr - macro_c
        color = 'darkred' if delta > 3 else ('darkorange' if delta > 1 else 'gray')
        ax.text(0.5, 0.03, f'Headroom: {delta:+.1f}pp',
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=9, color=color, fontweight='bold')

    # ── Row 1: Per-cell scatter — LR oracle vs CONT ───────────────────────────
    for ci, fs in enumerate(cols):
        ax = fig.add_subplot(gs[1, ci])
        all_x, all_y = [], []
        for domain, dcolor in DOMAIN_COLORS.items():
            xs = [r[f'cont_{fs}'] * 100 for r in valid
                  if r['cell'].split('/')[0] == domain
                  and r.get(f'cont_{fs}') is not None
                  and r.get(f'lr_{fs}')   is not None]
            ys = [r[f'lr_{fs}']   * 100 for r in valid
                  if r['cell'].split('/')[0] == domain
                  and r.get(f'cont_{fs}') is not None
                  and r.get(f'lr_{fs}')   is not None]
            if xs:
                ax.scatter(xs, ys, c=dcolor, s=50, alpha=0.85,
                           edgecolors='k', linewidths=0.4, label=domain)
                all_x.extend(xs)
                all_y.extend(ys)

        if all_x:
            lo = max(0,   min(all_x + all_y) - 3)
            hi = min(100, max(all_x + all_y) + 3)
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.9, alpha=0.6, label='y = x')

        ax.set_xlabel('CONT (L-SML) AUROC (%)', fontsize=9)
        ax.set_ylabel('LR Oracle AUROC (%)',     fontsize=9)
        ax.set_title(
            f'{labels_map[fs]}\nabove diagonal → LR beats CONT',
            fontsize=9
        )
        ax.legend(fontsize=7, loc='lower right')

    # ── Row 2: Headroom histogram ─────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[2, :])
    hist_colors = {'5': COLOR_CONT, '9': '#FF9800', '16': COLOR_LR}
    for fs in cols:
        deltas = [r[f'delta_{fs}'] * 100 for r in valid if r.get(f'delta_{fs}') is not None]
        if deltas:
            mean_d = np.mean(deltas)
            ax_h.hist(deltas, bins=15, alpha=0.50, color=hist_colors[fs],
                      label=f'{labels_map[fs]}  (mean Δ = {mean_d:+.1f}pp)')
    ax_h.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    ax_h.set_xlabel('Δ AUROC = LR Oracle − CONT  (pp)', fontsize=10)
    ax_h.set_ylabel('Cell count', fontsize=10)
    ax_h.set_title(
        'Supervised headroom per cell — how much does label access add?\n'
        'Right of 0 = LR beats L-SML; left of 0 = L-SML beats LR oracle',
        fontsize=10
    )
    ax_h.legend(fontsize=9)

    fig.suptitle(
        'Logistic Regression Oracle vs L-SML Continuous (unsupervised)\n'
        '5-fold stratified CV (per-fold AUROC averaged) · common-cell macro · '
        'Δ = supervised headroom',
        fontsize=11, fontweight='bold', y=0.99
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot -> {out_path}')


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_smoke_test():
    rng = np.random.default_rng(42)
    n   = 80
    fd  = {f: rng.standard_normal(n).tolist() for f in ALL_H16}
    lbl = (rng.random(n) > 0.45).astype(int)
    print('=== SMOKE TEST ===')
    r = run_cell('smoke_test', fd, lbl, cont_row=None)
    for k in ['5', '9', '16']:
        lr  = r.get(f'lr_{k}')
        lrs = f'{lr:.3f}' if lr is not None else 'N/A'
        print(f'  feat{k}: lr_auc={lrs}  (expect 0.40–0.65 on random data)')
    print('Smoke test OK.')
    sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./local_cache',
                        help='Directory containing domain .pkl files')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick sanity check on synthetic data then exit')
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()

    data_dir    = os.path.abspath(args.data_dir)
    cont_lookup = load_cont_results()
    print(f'data_dir : {data_dir}')
    print(f'CONT src : {CONT_PKL} ({len(cont_lookup)} cells loaded)\n')

    all_results = []
    for cell_name, fd, lbl_arr in iter_cells(data_dir):
        cont_row = cont_lookup.get(cell_name)
        r        = run_cell(cell_name, fd, lbl_arr, cont_row)
        all_results.append(r)

        cell_key = cell_name.split('/', 1)[-1]
        lr5 = r.get('lr_5')
        c5  = r.get('cont_5')
        d5  = r.get('delta_5')
        if lr5 is not None and c5 is not None and d5 is not None:
            status = f'CONT={100*c5:.1f}%  LR={100*lr5:.1f}%  D={100*d5:+.1f}pp'
        else:
            status = 'N/A (small cell or insufficient features)'
        print(f'  [{cell_key}] {status}')

    if not all_results:
        print('\nNo results — download pkl files to', data_dir)
        sys.exit(1)

    print()
    print_table(all_results)

    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved results -> {OUT_PKL}')

    make_plots(all_results, OUT_PNG)


if __name__ == '__main__':
    main()
