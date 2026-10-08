"""Figures for the presentation panels, rendered only from the built CSV tables (light + dark, PNG + SVG).

Palette: dataviz reference instance; 3 categorical slots validated all-pairs in both modes
(ours = slot 1 blue, supervised PRM = slot 2 orange, Mind the Gap = slot 3 aqua); baselines in muted ink.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = Path(__file__).resolve().parent; FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True)
HL = 'B16'; METHODS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
THEME = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  ours='#2a78d6', prm='#eb6834', mtg='#1baf7a', mid='#f0efec', neg='#e34948', pos='#2a78d6'),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 ours='#3987e5', prm='#d95926', mtg='#199e70', mid='#383835', neg='#e66767', pos='#3987e5')}
PRM_L = 'Qwen2.5-Math-PRM-7B (supervised)'; MTG_L = 'Mind the Gap (local repro.)'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def style(ax, t):
    ax.set_facecolor(t['surface'])
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(t['axis'])
    ax.tick_params(colors=t['ink2'], labelsize=9); ax.grid(axis='x', color=t['grid'], linewidth=.8); ax.set_axisbelow(True)


def save(fig, name, t, mode):
    fig.patch.set_facecolor(t['surface'])
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}_{mode}.{ext}', dpi=180, facecolor=t['surface'], bbox_inches='tight')
    plt.close(fig)


def ours_range(df, col):
    v = df[(df.bank == HL) & df.scorer.isin(METHODS)][col]
    return float(v.min()), float(v.mean()), float(v.max())


def fig_prmb_headline(t, mode):
    B = pd.read_csv(OUT / 'PANEL_B_PRMBENCH.csv')
    metrics = [('within_auc', 'Step ranking inside the answer\n(within-answer AUC)'),
               ('top1_on_error', 'Top-ranked step is an error'),
               ('prmscore_top1', 'PRMScore, one flag per answer\n(equal flag budget)'),
               ('prmscore_own', "PRMScore, each method's own rule\n(different calibration access)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    rows = ['Ours, B16 (4 methods)', PRM_L, MTG_L, 'Step index', 'Step length']
    for ax, (col, title) in zip(axes, metrics):
        style(ax, t); y = np.arange(len(rows))[::-1]
        lo, mean, hi = ours_range(B, col)
        ax.plot([lo, hi], [y[0]] * 2, color=t['ours'], linewidth=6, solid_capstyle='round', alpha=.45)
        ax.plot(mean, y[0], 'o', color=t['ours'], markersize=9, markeredgecolor=t['surface'], markeredgewidth=2)
        vals = [mean]
        for k, (lab, color, mk) in enumerate([(PRM_L, t['prm'], 'o'), (MTG_L, t['mtg'], 'o'), ('Step index (later = riskier)', t['muted'], 'D'), ('Step token length', t['muted'], 'D')], start=1):
            v = B[B.scorer == lab][col]
            v = float(v.iloc[0]) if len(v) and np.isfinite(v.iloc[0]) else np.nan
            vals.append(v)
            if np.isfinite(v):
                ax.plot(v, y[k], mk, color=color, markersize=9 if mk == 'o' else 7, markeredgecolor=t['surface'] if mk == 'o' else color,
                        markeredgewidth=2 if mk == 'o' else 1.5, markerfacecolor=color if mk == 'o' else 'none')
            else:
                ax.text(.5, y[k], 'n/a', color=t['muted'], fontsize=8, va='center', transform=ax.get_yaxis_transform())
        for k, v in enumerate(vals):
            if np.isfinite(v): ax.annotate(f'{v:.3f}', (v, y[k]), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8, color=t['ink2'])
        ax.set_title(title, color=t['ink'], fontsize=10); ax.set_xlim(.45, .9 if col == 'within_auc' else (.75 if 'prmscore' in col else .7))
        ax.set_yticks(y); ax.set_yticklabels(rows, color=t['ink'])
        ax.set_ylim(-.6, len(rows) - .4)
    fig.suptitle('PRMBench (6,969 answers): label-free fusion vs the supervised PRM and Mind the Gap', color=t['ink'], fontsize=12, x=.01, ha='left')
    fig.text(.01, -.06, 'Blue bar = range over the 4 registered methods on bank B16 (chosen post hoc on development data); dot = their mean. '
             'Development data, already exposed; no superiority claim.', color=t['ink2'], fontsize=8.5)
    fig.tight_layout(); save(fig, 'B1_prmbench_headline', t, mode)


def fig_types(t, mode):
    T = pd.read_csv(OUT / 'PANEL_B_PRMBENCH_TYPES.csv')
    T = T[T.group.str.startswith('cls:')]
    labels = {'cls:redundency': 'redundancy', 'cls:circular': 'circular logic', 'cls:domain_inconsistency': 'domain inconsistency',
              'cls:step_contradiction': 'step contradiction', 'cls:missing_condition': 'missing condition', 'cls:deception': 'deception',
              'cls:counterfactual': 'counterfactual', 'cls:confidence': 'confidence (stated confidently)'}
    rows = []
    for g, lab in labels.items():
        o = T[(T.group == g) & (T.bank == HL) & T.scorer.isin(METHODS)].within_auc
        p = float(T[(T.group == g) & (T.scorer == PRM_L)].within_auc.iloc[0]); m = float(T[(T.group == g) & (T.scorer == MTG_L)].within_auc.iloc[0])
        rows.append((lab, o.min(), o.mean(), o.max(), p, m))
    rows.sort(key=lambda r: r[2] - r[4], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5.2)); style(ax, t)
    y = np.arange(len(rows))[::-1]
    for yy, (lab, lo, mean, hi, p, m) in zip(y, rows):
        ax.plot([min(mean, p), max(mean, p)], [yy, yy], color=t['axis'], linewidth=2, zorder=1)
        ax.plot([lo, hi], [yy, yy], color=t['ours'], linewidth=6, alpha=.45, solid_capstyle='round', zorder=2)
        ax.plot(mean, yy, 'o', color=t['ours'], markersize=9, markeredgecolor=t['surface'], markeredgewidth=2, zorder=3)
        ax.plot(p, yy, 'o', color=t['prm'], markersize=9, markeredgecolor=t['surface'], markeredgewidth=2, zorder=3)
        ax.plot(m, yy, 's', color=t['mtg'], markersize=7, markeredgecolor=t['surface'], markeredgewidth=1.5, zorder=3)
        ax.annotate(f'{mean - p:+.3f}', (max(mean, p, m), yy), xytext=(8, 0), textcoords='offset points', va='center', fontsize=8.5,
                    color=t['ink'] if mean > p else t['ink2'])
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=t['ink']); ax.set_xlim(.58, .97)
    ax.set_xlabel('Within-answer AUC (ranking error steps above correct steps)', color=t['ink2'])
    handles = [plt.Line2D([], [], marker='o', ls='', color=t['ours'], markersize=8, label='Ours, B16 (mean; bar = 4-method range)'),
               plt.Line2D([], [], marker='o', ls='', color=t['prm'], markersize=8, label='Supervised PRM (Qwen2.5-Math-PRM-7B)'),
               plt.Line2D([], [], marker='s', ls='', color=t['mtg'], markersize=7, label='Mind the Gap (local repro.)')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.32), ncol=2, frameon=False, labelcolor=t['ink'], fontsize=8.5)
    ax.set_title('PRMBench by error type: numbers = ours minus the supervised PRM', color=t['ink'], fontsize=11, loc='left')
    fig.tight_layout(); save(fig, 'B2_error_types', t, mode)


def fig_pb(t, mode):
    P = pd.read_csv(OUT / 'PANEL_B_PROCESSBENCH.csv')
    strata = [('ALL', 'All erroneous answers'), ('content-only (not step 0, not longest)', 'Error not at step 0\nand not the longest step'),
              ('error at step 0', 'Error at step 0'), ('error in longest step', 'Error in the longest step'),
              ('subset:gsm8k', 'GSM8K'), ('subset:math', 'MATH'), ('subset:olympiadbench', 'OlympiadBench'), ('subset:omnimath', 'Omni-MATH')]
    fig, ax = plt.subplots(figsize=(10, 5.4)); style(ax, t)
    y = np.arange(len(strata))[::-1]
    for yy, (s, lab) in zip(y, strata):
        o = P[(P.stratum == s) & (P.bank == HL) & P.scorer.isin(METHODS)].argmax_hit
        mt = float(P[(P.stratum == s) & (P.scorer == MTG_L)].argmax_hit.iloc[0]); pr = float(P[(P.stratum == s) & (P.scorer == 'Qwen2.5-Math-PRM-7B argmax')].argmax_hit.iloc[0])
        s0 = float(P[(P.stratum == s) & (P.scorer == 'Always step 0')].argmax_hit.iloc[0]); lg = float(P[(P.stratum == s) & (P.scorer == 'Longest step')].argmax_hit.iloc[0])
        n = int(P[P.stratum == s].n_error_rows.iloc[0])
        ax.plot([o.min(), o.max()], [yy, yy], color=t['ours'], linewidth=6, alpha=.45, solid_capstyle='round')
        ax.plot(o.mean(), yy, 'o', color=t['ours'], markersize=9, markeredgecolor=t['surface'], markeredgewidth=2, zorder=3)
        ax.plot(pr, yy, 'o', color=t['prm'], markersize=9, markeredgecolor=t['surface'], markeredgewidth=2, zorder=3)
        ax.plot(mt, yy, 's', color=t['mtg'], markersize=7, markeredgecolor=t['surface'], markeredgewidth=1.5, zorder=3)
        if s not in ('error at step 0', 'error in longest step', 'content-only (not step 0, not longest)'):
            ax.plot(s0, yy, 'D', color=t['muted'], markersize=6, markerfacecolor='none', markeredgewidth=1.5)
            ax.plot(lg, yy, '^', color=t['muted'], markersize=7, markerfacecolor='none', markeredgewidth=1.5)
        ax.text(1.005, yy, f'n={n}', transform=ax.get_yaxis_transform(), va='center', fontsize=8, color=t['muted'])
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in strata], color=t['ink']); ax.set_xlim(0, .8)
    ax.set_xlabel('Top-ranked step is the first error (erroneous answers; no gate)', color=t['ink2'])
    handles = [plt.Line2D([], [], marker='o', ls='', color=t['ours'], markersize=8, label='Ours, B16 (mean; bar = 4-method range)'),
               plt.Line2D([], [], marker='o', ls='', color=t['prm'], markersize=8, label='Supervised PRM, argmax'),
               plt.Line2D([], [], marker='s', ls='', color=t['mtg'], markersize=7, label='Mind the Gap (local repro.)'),
               plt.Line2D([], [], marker='D', ls='', color=t['muted'], markerfacecolor='none', markersize=6, label='Always step 0'),
               plt.Line2D([], [], marker='^', ls='', color=t['muted'], markerfacecolor='none', markersize=7, label='Longest step')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.45, -.36), ncol=3, frameon=False, labelcolor=t['ink'], fontsize=8.5)
    ax.set_title('ProcessBench, gate-free localization (8 cells = 4 subsets x 2 telemetry models, cell-equal mean)', color=t['ink'], fontsize=11, loc='left')
    fig.tight_layout(); save(fig, 'B3_processbench_localization', t, mode)


def fig_panel_a(t, mode):
    C = pd.read_csv(OUT / 'PANEL_A_CONTRASTS.csv'); C = C[C.reference == 'DS_FILTER_EQ']
    order = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'HEM_MOM_ga', 'TCI_ONLY_ga']; banks = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
    cmap = LinearSegmentedColormap.from_list('div', [t['neg'], t['mid'], t['pos']])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (metric, title, lim) in zip(axes, [('prmb_within_auc', 'PRMBench within-answer AUC', .025), ('pb_argmax_hit', 'ProcessBench top-step first-error hit', .04)]):
        sub = C[C.metric == metric]; M = sub.pivot(index='bank', columns='method', values='delta').loc[banks, order]
        lo = sub.pivot(index='bank', columns='method', values='lo95').loc[banks, order]; hi = sub.pivot(index='bank', columns='method', values='hi95').loc[banks, order]
        ax.imshow(M.values, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
        for i in range(len(banks)):
            for j in range(len(order)):
                v = M.values[i, j]; sig = lo.values[i, j] > 0 or hi.values[i, j] < 0
                ax.text(j, i, f'{v:+.3f}' + ('*' if sig else ''), ha='center', va='center', fontsize=8.5, color=t['ink'], fontweight='bold' if sig else 'normal')
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, color=t['ink'], fontsize=8.5)
        ax.set_yticks(range(len(banks))); ax.set_yticklabels([f'{b} ({int(b[1:])} ch{" +digits" if b in ("B16","B23","B35","B54") else ""})' for b in banks], color=t['ink'], fontsize=8.5)
        ax.set_title(title, color=t['ink'], fontsize=10.5)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_facecolor(t['surface']); ax.tick_params(length=0)
        ax.add_patch(plt.Rectangle((-.5, banks.index('B16') - .5), len(order), 1, fill=False, edgecolor=t['ink'], linewidth=1.5))
    fig.suptitle('Does the fusion algorithm add over the simple average of the same filtered channels (DS_FILTER_EQ)?', color=t['ink'], fontsize=12, x=.01, ha='left')
    fig.text(.01, -.03, 'Cell = method minus DS_FILTER_EQ on the same bank; blue = method better, red = worse; * = paired source-group 95% interval excludes 0 '
             '(descriptive, no multiplicity correction). Outlined row = highlighted bank B16.', color=t['ink2'], fontsize=8.5)
    fig.patch.set_facecolor(t['surface']); fig.tight_layout(); save(fig, 'A1_algorithm_vs_simple_average', t, mode)


def main():
    for mode, t in THEME.items():
        fig_prmb_headline(t, mode); fig_types(t, mode); fig_pb(t, mode); fig_panel_a(t, mode)
    man = {'source_tables': {p.name: sha(p) for p in sorted(OUT.glob('PANEL_*.csv'))}, 'renderer_sha256': sha(Path(__file__)),
           'figures': {p.name: sha(p) for p in sorted(FIG.glob('*'))}}
    (OUT / 'FIGURES_MANIFEST.json').write_text(json.dumps(man, indent=1), encoding='utf8'); print(len(man['figures']), 'figures')


if __name__ == '__main__':
    main()
