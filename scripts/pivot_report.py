"""
Pivot-alternatives pilot report (Step 151) — merges the Track A/B result
pkls, prints memo-ready markdown tables, and writes figures.

Reads:  results/pivot_trackA.pkl, results/pivot_trackB.pkl,
        results/upcr_comparison.pkl, results/method_comparison_table1.csv
Writes: results/figs/pivot_trackA_macro.png
        results/figs/pivot_trackA_orientation.png
        results/figs/pivot_trackB_bars.png

Usage:
    python scripts/pivot_report.py
"""

import os
import sys
import pickle
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

A_PKL = os.path.join(REPO_DIR, 'results', 'pivot_trackA.pkl')
B_PKL = os.path.join(REPO_DIR, 'results', 'pivot_trackB.pkl')
FIG_DIR = os.path.join(REPO_DIR, 'results', 'figs')

METHOD_ORDER = ['maha', 'gmm2', 'kde', 'iforest', 'ae', 'prae']
TIERS = ['auc_raw', 'auc_anchored', 'auc_oracle']


def macro(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')


def load_comparators():
    sys.path.insert(0, os.path.join(REPO_DIR, 'scripts'))
    from pivot_trackA import load_comparators as _lc
    return _lc()


def collect_a(res, comp, fs):
    """Per-method dict: tier macros on cont-common cells + per-regime."""
    out = {}
    for m in METHOD_ORDER:
        rows = []
        for cell, ce in res.items():
            me = ce.get('fs', {}).get(fs, {}).get('methods', {}).get(m)
            cv = comp.get(cell, {}).get(f'cont_{fs}')
            if me is None or cv is None or not np.isfinite(cv):
                continue
            rows.append({
                'regime': ce['regime'], 'cont': cv,
                'raw': me['auc_raw'][0], 'anch': me['auc_anchored'][0],
                'oracle': me['auc_oracle'],
                'flip': me['anchor_flip'],
            })
        if rows:
            out[m] = rows
    return out


def report_track_a(res, comp):
    print('\n## Track A — anomaly scorers vs L-SML continuous (markdown)\n')
    for fs in ('16', '5'):
        data = collect_a(res, comp, fs)
        if not data:
            continue
        any_rows = next(iter(data.values()))
        mc = macro([r['cont'] for r in any_rows])
        print(f'### feature set {fs} — L-SML continuous macro = {mc:.3f} '
              f'({len(any_rows)} common cells)\n')
        print('| method | raw | **anchored** | oracle | delta vs L-SML (anch) | '
              'reasoning | gpqa | rag | flips |')
        print('|---|---|---|---|---|---|---|---|---|')
        for m, rows in data.items():
            ma = macro([r['anch'] for r in rows])
            mr = macro([r['raw'] for r in rows])
            mo = macro([r['oracle'] for r in rows])
            regs = {}
            for reg in ('reasoning', 'gpqa', 'rag'):
                regs[reg] = macro([r['anch'] for r in rows
                                   if r['regime'] == reg])
            nflip = sum(r['flip'] for r in rows)
            print(f'| {m} | {mr:.3f} | **{ma:.3f}** | {mo:.3f} | '
                  f'{(ma - mc) * 100:+.1f}pp | {regs["reasoning"]:.3f} | '
                  f'{regs["gpqa"]:.3f} | {regs["rag"]:.3f} | '
                  f'{nflip}/{len(rows)} |')
        print()


def report_track_b(res):
    print('\n## Track B — temporal candidates (markdown)\n')
    for name, entry in res.items():
        tag = 'PRIMARY' if entry['primary'] else 'SECONDARY / NON-CANONICAL'
        dc = {t: v['auc'][0] for t, v in entry['baselines'].items()
              if t.startswith('deepconf')}
        best_dc = max(dc, key=dc.get)
        print(f'### {name} ({tag}, n={entry["n"]}, '
              f'frac_correct={entry["frac_correct"]:.2f})\n')
        print(f'Baselines: lsml5={entry["baselines"]["lsml5"]["auc"][0]:.3f} '
              f'lsml16={entry["baselines"]["lsml16"]["auc"][0]:.3f} '
              f'epr={entry["baselines"]["epr"]["auc"][0]:.3f} '
              f'best DeepConf = {best_dc} {dc[best_dc]:.3f}\n')
        print('| candidate | raw | **anchored** | oracle |')
        print('|---|---|---|---|')
        for t, ce in entry['candidates'].items():
            print(f'| {t} | {ce["auc_raw"][0]:.3f} | '
                  f'**{ce["auc_anchored"][0]:.3f}** | {ce["auc_oracle"]:.3f} |')
        print()


def figures(res_a, comp, res_b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)

    # A: macro anchored per method vs L-SML line, fs=16 and fs=5
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, fs in zip(axes, ('16', '5')):
        data = collect_a(res_a, comp, fs)
        if not data:
            continue
        ms = [m for m in METHOD_ORDER if m in data]
        vals = [macro([r['anch'] for r in data[m]]) for m in ms]
        mc = macro([r['cont'] for r in next(iter(data.values()))])
        ax.bar(ms, vals, color='steelblue')
        ax.axhline(mc, color='crimson', ls='--',
                   label=f'L-SML cont ({mc:.3f})')
        ax.axhline(0.5, color='gray', lw=0.5)
        ax.set_title(f'feature set {fs} (anchored tier)')
        ax.set_ylim(0.4, max(max(vals), mc) + 0.05)
        ax.legend(fontsize=8)
    fig.suptitle('Track A: anomaly scorers vs L-SML continuous (macro AUROC, common cells)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'pivot_trackA_macro.png'), dpi=150)
    plt.close(fig)

    # A: orientation tiers per method (fs=16) — inversion diagnostic
    data = collect_a(res_a, comp, '16')
    if data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ms = [m for m in METHOD_ORDER if m in data]
        x = np.arange(len(ms))
        for i, (tier, lab) in enumerate((('raw', 'raw'),
                                         ('anch', 'anchored (primary)'),
                                         ('oracle', 'oracle (diagnostic)'))):
            vals = [macro([r[tier] for r in data[m]]) for m in ms]
            ax.bar(x + (i - 1) * 0.27, vals, width=0.25, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(ms)
        ax.axhline(0.5, color='gray', lw=0.5)
        ax.set_ylim(0.4, 0.85)
        ax.set_title('Track A fs=16: orientation tiers '
                     '(raw<anchored gap = inversion on hallucination-majority cells)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'pivot_trackA_orientation.png'), dpi=150)
        plt.close(fig)

    # B: candidate vs baseline bars per cell
    if res_b:
        fig, axes = plt.subplots(1, len(res_b), figsize=(6 * len(res_b), 4.5),
                                 squeeze=False)
        for ax, (name, entry) in zip(axes[0], res_b.items()):
            tags, vals, colors = [], [], []
            for t in ('lsml5', 'epr', 'deepconf_w32', 'deepconf_w64'):
                tags.append(t)
                vals.append(entry['baselines'][t]['auc'][0])
                colors.append('gray')
            for t, ce in entry['candidates'].items():
                if t.endswith('_l50') or t.endswith('_l200'):
                    continue
                tags.append(t)
                vals.append(ce['auc_anchored'][0])
                colors.append('steelblue')
            ax.barh(tags[::-1], vals[::-1],
                    color=colors[::-1])
            ax.axvline(0.5, color='gray', lw=0.5)
            ax.set_xlim(0.35, 0.85)
            ax.set_title(f'{name}\n(gray=baselines, blue=candidates, anchored)',
                         fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'pivot_trackB_bars.png'), dpi=150)
        plt.close(fig)
    print(f'\nfigures written to {FIG_DIR}')


def main():
    res_a, res_b = {}, {}
    if os.path.exists(A_PKL):
        with open(A_PKL, 'rb') as f:
            res_a = pickle.load(f)
    else:
        print(f'[WARN] {A_PKL} missing — run scripts/pivot_trackA.py')
    if os.path.exists(B_PKL):
        with open(B_PKL, 'rb') as f:
            res_b = pickle.load(f)
    else:
        print(f'[WARN] {B_PKL} missing — run scripts/pivot_trackB.py')

    comp = load_comparators() if res_a else {}
    if res_a:
        report_track_a(res_a, comp)
    if res_b:
        report_track_b(res_b)
    if res_a or res_b:
        figures(res_a, comp, res_b)


if __name__ == '__main__':
    main()
