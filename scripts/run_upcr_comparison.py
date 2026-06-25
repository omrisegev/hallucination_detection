"""
Compare CONT (continuous L-SML) vs U-PCR-1 (1-component) vs U-PCR-auto
(auto λ₂ threshold, 2 components when λ₂ > 0.1·Trace).

Usage:
    python scripts/run_upcr_comparison.py [--data-dir ./local_cache] [--smoke-test]

Reads:  local_cache/{math500_res,gsm8k_res,gpqa_res,qa_res,rag_feats_all}.pkl
        Schema per pkl: {cell_key: (feat_dict, labels)}
Writes: results/upcr_comparison.pkl
        results/upcr_comparison.png
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

from spectral_utils import (
    lsml_continuous_pipeline,
    upcr_pipeline,
    boot_auc,
    FEAT_NAMES,
)


def safe_auc(lbl, scores):
    lbl = np.asarray(lbl, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(set(lbl.tolist())) < 2 or np.all(scores == scores[0]):
        return 0.5, 0.5, 0.5
    p, pl, ph = boot_auc(lbl, scores)
    n, nl, nh = boot_auc(lbl, -scores)
    return (p, pl, ph) if p >= n else (n, nl, nh)

# ── Feature sets ──────────────────────────────────────────────────────────────

GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

STABLE_H9 = [
    'epr', 'low_band_power', 'high_band_power', 'hl_ratio',
    'spectral_centroid', 'sw_var_peak', 'rpdi', 'pe_mean', 'cusum_max',
]

ALL_H16 = FEAT_NAMES[:16]

FEATURE_SETS = {'5': GOOD_5, '9': STABLE_H9, '16': ALL_H16}

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
}

PKL_NAMES = {
    'math500': 'math500_res.pkl',
    'gsm8k':   'gsm8k_res.pkl',
    'gpqa':    'gpqa_res.pkl',
    'rag':     'rag_feats_all.pkl',
    'qa':      'qa_res.pkl',
}

OUT_PKL  = os.path.join(REPO_DIR, 'results', 'upcr_comparison.pkl')
OUT_PNG  = os.path.join(REPO_DIR, 'results', 'upcr_comparison.png')

# ── Data loading ──────────────────────────────────────────────────────────────

def load_cached_feats(pkl_path):
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and 'feats' in obj:
        return obj['feats']
    return obj


def is_saturated(arr, threshold=0.40):
    a = np.asarray(arr, dtype=float)
    return float(np.mean(a == np.median(a))) > threshold

# ── Per-cell runner ───────────────────────────────────────────────────────────

def run_cell(cell_name, fd, labels):
    """Run CONT, U-PCR-1, and U-PCR-auto for all three feature sets on one cell."""
    labels = np.array(labels, dtype=float)
    p = float(np.mean(labels))
    var_y = max(p * (1.0 - p), 0.05)

    cell_res = {'cell': cell_name, 'n': len(labels), 'prevalence': p, 'var_y': var_y}

    for fs_name, feat_list in FEATURE_SETS.items():
        available = [
            f for f in feat_list
            if f in fd and fd[f] is not None
            and len(fd[f]) == len(labels)
            and not is_saturated(fd[f])
        ]
        cell_res[f'avail_{fs_name}'] = available
        n_avail = len(available)

        if n_avail < 3:
            for method in ('cont', 'upcr1', 'upcrauto'):
                cell_res[f'{method}_{fs_name}'] = None
            cell_res[f'lambda2_frac_{fs_name}'] = None
            cell_res[f'n_comp_{fs_name}'] = None
            continue

        feat_signs = {f: FEATURE_SIGNS.get(f, -1) for f in available}

        # CONT
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                cont_score, _ = lsml_continuous_pipeline(fd, available, feat_signs)
                cont_auc, _, _ = safe_auc(labels, cont_score)
            except Exception:
                cont_auc = None

        # U-PCR-1 (1 component, no auto — matches old behaviour / Step 140)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                r1 = upcr_pipeline(
                    fd, available, feat_signs, var_y=var_y,
                    auto_components=False, n_components=1,
                    return_diagnostics=True,
                )
                upcr1_score, w1, rho1, g2_1, diag1 = r1
                upcr1_auc, _, _ = safe_auc(labels, upcr1_score)
            except Exception:
                upcr1_auc = None
                w1 = rho1 = g2_1 = diag1 = None

        # U-PCR-auto (auto lam2 threshold -- corrected algorithm)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                rauto = upcr_pipeline(
                    fd, available, feat_signs, var_y=var_y,
                    auto_components=True, lambda2_threshold=0.1,
                    return_diagnostics=True,
                )
                upcrauto_score, wauto, rhoauto, g2auto, diagauto = rauto
                upcrauto_auc, _, _ = safe_auc(labels, upcrauto_score)
            except Exception:
                upcrauto_auc = None
                wauto = rhoauto = g2auto = diagauto = None

        cell_res[f'cont_{fs_name}'] = cont_auc
        cell_res[f'upcr1_{fs_name}'] = upcr1_auc
        cell_res[f'upcrauto_{fs_name}'] = upcrauto_auc
        cell_res[f'delta1_{fs_name}'] = (
            (upcr1_auc - cont_auc) if (upcr1_auc is not None and cont_auc is not None) else None
        )
        cell_res[f'deltaauto_{fs_name}'] = (
            (upcrauto_auc - cont_auc) if (upcrauto_auc is not None and cont_auc is not None) else None
        )
        cell_res[f'delta_auto_vs_1_{fs_name}'] = (
            (upcrauto_auc - upcr1_auc) if (upcrauto_auc is not None and upcr1_auc is not None) else None
        )

        if diagauto is not None:
            cell_res[f'lambda2_frac_{fs_name}'] = diagauto['lambda2_frac']
            cell_res[f'n_comp_{fs_name}'] = diagauto['n_components_used']
        else:
            cell_res[f'lambda2_frac_{fs_name}'] = None
            cell_res[f'n_comp_{fs_name}'] = None

        if wauto is not None:
            cell_res[f'upcrauto_weights_{fs_name}'] = wauto.tolist()
            cell_res[f'upcrauto_rho_{fs_name}'] = rhoauto.tolist()
            cell_res[f'upcrauto_feats_{fs_name}'] = available

    return cell_res

# ── Table printer ─────────────────────────────────────────────────────────────

def print_table(results):
    cols = ['5', '9', '16']
    hdr = (f"{'Cell':<36} | "
           + ' | '.join(
               f"{'CONT'+k:>8} {'U1-'+k:>7} {'UA-'+k:>7} {'dU1-'+k:>6} {'dUA-'+k:>6} {'lam2%':>6} {'k':>2}"
               for k in cols))
    sep = '-' * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    accum = {}
    for m in ('cont', 'upcr1', 'upcrauto'):
        for k in cols:
            accum[f'{m}_{k}'] = []

    for r in results:
        if r.get('skipped'):
            continue
        parts = []
        for k in cols:
            c   = r.get(f'cont_{k}')
            u1  = r.get(f'upcr1_{k}')
            ua  = r.get(f'upcrauto_{k}')
            d1  = r.get(f'delta1_{k}')
            da  = r.get(f'deltaauto_{k}')
            l2  = r.get(f'lambda2_frac_{k}')
            nc  = r.get(f'n_comp_{k}')
            cs  = f'{100*c:.1f}%' if c  is not None else '   N/A'
            u1s = f'{100*u1:.1f}%' if u1 is not None else '   N/A'
            uas = f'{100*ua:.1f}%' if ua is not None else '   N/A'
            d1s = f'{100*d1:+.1f}' if d1 is not None else '  N/A'
            das = f'{100*da:+.1f}' if da is not None else '  N/A'
            l2s = f'{100*l2:.0f}%' if l2 is not None else ' N/A'
            ncs = str(nc) if nc is not None else ' -'
            parts.append(f'{cs:>8} {u1s:>7} {uas:>7} {d1s:>5} {das:>5} {l2s:>5} {ncs:>2}')
            if c  is not None: accum[f'cont_{k}'].append(c)
            if u1 is not None: accum[f'upcr1_{k}'].append(u1)
            if ua is not None: accum[f'upcrauto_{k}'].append(ua)
        print(f"  {r['cell']:<34} | " + ' | '.join(parts))

    print(sep)
    macro_parts = []
    for k in cols:
        mc  = np.mean(accum[f'cont_{k}'])  if accum[f'cont_{k}']  else None
        mu1 = np.mean(accum[f'upcr1_{k}']) if accum[f'upcr1_{k}'] else None
        mua = np.mean(accum[f'upcrauto_{k}']) if accum[f'upcrauto_{k}'] else None
        md1  = (mu1 - mc)  if (mc is not None and mu1 is not None) else None
        mda  = (mua - mc)  if (mc is not None and mua is not None) else None
        cs   = f'{100*mc:.1f}%'  if mc  is not None else '   N/A'
        u1s  = f'{100*mu1:.1f}%' if mu1 is not None else '   N/A'
        uas  = f'{100*mua:.1f}%' if mua is not None else '   N/A'
        d1s  = f'{100*md1:+.1f}' if md1 is not None else '  N/A'
        das  = f'{100*mda:+.1f}' if mda is not None else '  N/A'
        macro_parts.append(f'{cs:>8} {u1s:>7} {uas:>7} {d1s:>5} {das:>5} {"":>5} {"":>2}')
    print(f"  {'MACRO':<34} | " + ' | '.join(macro_parts))
    print(sep)

# ── Visualization ─────────────────────────────────────────────────────────────

def make_plots(results, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('[plot] matplotlib not available, skipping')
        return

    cols = ['5', '9', '16']
    labels_map = {'5': '5-feat (GOOD_5)', '9': '9-feat (STABLE_H9)', '16': '16-feat (ALL_H16)'}
    methods = [('cont', 'CONT (L-SML)', '#2196F3'),
               ('upcr1', 'U-PCR-1 (1-comp)', '#FF9800'),
               ('upcrauto', 'U-PCR-auto (λ₂ thresh)', '#4CAF50')]

    valid = [r for r in results if not r.get('skipped')]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── Row 0: Macro AUROC bar chart per feature set ──────────────────────────
    for ci, fs in enumerate(cols):
        ax = fig.add_subplot(gs[0, ci])
        macro = {}
        for mkey, mlabel, mc in methods:
            vals = [r.get(f'{mkey}_{fs}') for r in valid if r.get(f'{mkey}_{fs}') is not None]
            macro[mkey] = np.mean(vals) * 100 if vals else 0.0
        xs = np.arange(len(methods))
        bars = ax.bar(xs, [macro[mk] for mk, _, _ in methods],
                      color=[mc for _, _, mc in methods], width=0.6, alpha=0.85)
        ax.bar_label(bars, fmt='%.1f%%', padding=2, fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([ml for _, ml, _ in methods], rotation=18, ha='right', fontsize=8)
        ax.set_ylim(max(0, min(macro.values()) - 3), min(100, max(macro.values()) + 5))
        ax.set_ylabel('Macro AUROC (%)')
        ax.set_title(labels_map[fs], fontsize=10, fontweight='bold')
        ax.axhline(50, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

    # ── Row 1: Per-cell delta scatter — U-PCR-auto vs CONT ────────────────────
    for ci, fs in enumerate(cols):
        ax = fig.add_subplot(gs[1, ci])
        deltas = [r.get(f'deltaauto_{fs}') for r in valid
                  if r.get(f'deltaauto_{fs}') is not None]
        l2_fracs = [r.get(f'lambda2_frac_{fs}') for r in valid
                    if r.get(f'deltaauto_{fs}') is not None
                    and r.get(f'lambda2_frac_{fs}') is not None]
        n_comp = [r.get(f'n_comp_{fs}') for r in valid
                  if r.get(f'deltaauto_{fs}') is not None]

        if deltas and l2_fracs and len(deltas) == len(l2_fracs):
            colors = ['#4CAF50' if nc == 2 else '#FF9800' for nc in n_comp]
            ax.scatter(np.array(l2_fracs) * 100, np.array(deltas) * 100,
                       c=colors, s=50, alpha=0.8, edgecolors='k', linewidths=0.4)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(10, color='red', linestyle=':', linewidth=1.0, label='λ₂ = 10% thresh')
            ax.set_xlabel('λ₂ / Trace(C)  (%)')
            ax.set_ylabel('Δ AUROC (U-PCR-auto − CONT)  pp')
            ax.set_title(f'{labels_map[fs]}\ngreen=2-comp, orange=1-comp', fontsize=9)
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(labels_map[fs])

    # ── Row 2: λ₂/Trace distribution + per-feature weight comparison (5-feat) ─
    ax_l2 = fig.add_subplot(gs[2, 0])
    all_l2_5  = [r.get('lambda2_frac_5')  for r in valid if r.get('lambda2_frac_5')  is not None]
    all_l2_16 = [r.get('lambda2_frac_16') for r in valid if r.get('lambda2_frac_16') is not None]
    bins = np.linspace(0, 0.5, 20)
    if all_l2_5:
        ax_l2.hist(np.array(all_l2_5)  * 100, bins=bins * 100, alpha=0.6, label='5-feat',  color='#2196F3')
    if all_l2_16:
        ax_l2.hist(np.array(all_l2_16) * 100, bins=bins * 100, alpha=0.6, label='16-feat', color='#FF5722')
    ax_l2.axvline(10, color='red', linestyle=':', linewidth=1.2, label='λ₂=10% threshold')
    ax_l2.set_xlabel('λ₂ / Trace(C)  (%)')
    ax_l2.set_ylabel('Cell count')
    ax_l2.set_title('λ₂ fraction distribution\n(triggers 2-comp above red line)', fontsize=9)
    ax_l2.legend(fontsize=8)

    # Per-feature weight comparison for 5-feat: one representative cell with 2-comp
    ax_wt = fig.add_subplot(gs[2, 1:])
    two_comp_cells = [
        r for r in valid
        if r.get('n_comp_5') == 2
        and r.get('upcrauto_weights_5') is not None
        and r.get('upcrauto_feats_5') is not None
    ]
    one_comp_cells = [
        r for r in valid
        if r.get('n_comp_5') == 1
        and r.get('upcrauto_weights_5') is not None
        and r.get('upcrauto_feats_5') is not None
    ]

    if two_comp_cells or one_comp_cells:
        # Show up to 3 representative cells: first 2-comp (if any), then 1-comp
        rep_cells = (two_comp_cells[:2] + one_comp_cells[:1]) if two_comp_cells else one_comp_cells[:3]
        n_rep = len(rep_cells)
        x = np.arange(len(GOOD_5))
        width = 0.25
        for ri, rc in enumerate(rep_cells):
            feats_in_cell = rc.get('upcrauto_feats_5', [])
            weights_in_cell = rc.get('upcrauto_weights_5', [])
            nc = rc.get('n_comp_5', 1)
            # Map weights to canonical 5-feat order
            w_map = dict(zip(feats_in_cell, weights_in_cell))
            w_arr = np.array([w_map.get(f, 0.0) for f in GOOD_5])
            label = f"{rc['cell'][-20:]} (k={nc})"
            color = '#4CAF50' if nc == 2 else '#FF9800'
            ax_wt.bar(x + ri * width, w_arr, width, label=label, color=color, alpha=0.75)
        ax_wt.set_xticks(x + width)
        ax_wt.set_xticklabels(GOOD_5, rotation=20, ha='right', fontsize=9)
        ax_wt.set_ylabel('U-PCR-auto weight')
        ax_wt.axhline(0, color='gray', linewidth=0.7)
        ax_wt.set_title(
            'Per-feature weights — U-PCR-auto (5-feat)\n'
            'green bars = 2-component (v₁+v₂), orange = 1-component (v₁ only)\n'
            '2-comp "clusters" features by adding a v₂ correction to each weight',
            fontsize=9
        )
        ax_wt.legend(fontsize=7, loc='upper right')
    else:
        ax_wt.text(0.5, 0.5, 'No cells with 2-comp weight data', ha='center', va='center',
                   transform=ax_wt.transAxes)

    fig.suptitle(
        'U-PCR Algorithm Comparison: CONT vs U-PCR-1 vs U-PCR-auto (λ₂ threshold)\n'
        'U-PCR-auto uses 2 eigenvectors when λ₂ > 10%·Trace(C) — analogous to soft clustering',
        fontsize=11, fontweight='bold', y=0.98
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot -> {out_path}')


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_smoke_test():
    rng = np.random.default_rng(42)
    n = 80
    fd = {f: rng.standard_normal(n).tolist() for f in ALL_H16}
    lbl = (rng.random(n) > 0.45).astype(int)
    print('=== SMOKE TEST ===')
    r = run_cell('smoke_test', fd, lbl)
    for k in ['5', '9', '16']:
        c  = r.get(f'cont_{k}')
        u1 = r.get(f'upcr1_{k}')
        ua = r.get(f'upcrauto_{k}')
        l2 = r.get(f'lambda2_frac_{k}')
        nc = r.get(f'n_comp_{k}')
        cs  = f'{c:.3f}'  if c  is not None else 'N/A'
        u1s = f'{u1:.3f}' if u1 is not None else 'N/A'
        uas = f'{ua:.3f}' if ua is not None else 'N/A'
        if l2 is not None:
            print(f'  feat{k}: cont={cs} upcr1={u1s} upcrauto={uas}  lam2={l2:.2%} k={nc}')
        else:
            print(f'  feat{k}: cont={cs} upcr1={u1s} upcrauto={uas}')
    print('Smoke test OK.')
    sys.exit(0)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./local_cache')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()

    data_dir = os.path.abspath(args.data_dir)
    print(f'data_dir: {data_dir}\n')

    all_results = []
    for domain, pkl_name in PKL_NAMES.items():
        path = os.path.join(data_dir, pkl_name)
        feats = load_cached_feats(path)
        if feats is None:
            print(f'[MISSING] {pkl_name}')
            continue
        print(f'\n--- {domain.upper()} ({len(feats)} cells) ---')
        for cell_key, payload in feats.items():
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                fd, lbl = payload
            elif isinstance(payload, dict) and 'feats' in payload:
                fd, lbl = payload['feats'], payload['labels']
            else:
                print(f'  [{cell_key}] unknown schema — skip')
                all_results.append({'cell': f'{domain}/{cell_key}', 'skipped': True})
                continue
            lbl_arr = np.asarray(lbl, dtype=int)
            if len(set(lbl_arr.tolist())) < 2:
                print(f'  [{cell_key}] single class — skip')
                continue
            r = run_cell(f'{domain}/{cell_key}', fd, lbl_arr)
            all_results.append(r)
            c5  = r.get('cont_5')
            u1_5 = r.get('upcr1_5')
            ua5 = r.get('upcrauto_5')
            l2  = r.get('lambda2_frac_5')
            nc  = r.get('n_comp_5')
            status = (
                f"CONT={100*c5:.1f}% U1={100*u1_5:.1f}% UA={100*ua5:.1f}%"
                f"  lam2={100*l2:.0f}% k={nc}"
                if (c5 is not None and u1_5 is not None and ua5 is not None and l2 is not None)
                else 'N/A'
            )
            print(f'  [{cell_key}] {status}')

    if not all_results:
        print('\nNo results. Download pkl files to', data_dir)
        sys.exit(1)

    print()
    print_table(all_results)

    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved to {OUT_PKL}')

    make_plots(all_results, OUT_PNG)


if __name__ == '__main__':
    main()
