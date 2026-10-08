"""Figures for monday_types_singles_v1: T1 error types across all 80 label-free configurations; S1 single channels vs fusion."""
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent; (OUT / 'figures').mkdir(exist_ok=True)
T = {'bg': '#fcfcfb', 'ink': '#1b2430', 'ink2': '#5b6575', 'grid': '#e4e2dc', 'ours': '#2a78d6', 'ref': '#eb6834', 'mtg': '#1baf7a', 'muted': '#898781'}
LAB = {'redundency': 'redundancy', 'circular': 'circular\nlogic', 'domain_inconsistency': 'domain\ninconsistency', 'step_contradiction': 'step\ncontradiction',
       'missing_condition': 'missing\ncondition', 'confidence': 'confidence\n(stated firmly)', 'counterfactual': 'counter-\nfactual', 'deception': 'deception'}


def style(ax):
    ax.set_facecolor(T['bg'])
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(T['muted'])
    ax.grid(axis='y', color=T['grid'], linewidth=.8); ax.set_axisbelow(True); ax.tick_params(colors=T['ink2'])


def fig_types():
    A = pd.read_csv(OUT / 'TYPES_ALL_CONFIGS.csv'); S = pd.read_csv(OUT / 'TYPES_SUMMARY.csv')
    S['med'] = S.median_minus_PRM; S = S.sort_values('med', ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5.4), facecolor=T['bg']); style(ax)
    for x, r in enumerate(S.itertuples()):
        L = A[(A.type == r.type) & (A.family != 'reference')]
        ax.vlines(x - .15, L.auc.min(), L.auc.max(), color=T['ours'], alpha=.35, linewidth=9)
        ax.plot(x - .15, L.auc.median(), 'o', color=T['ours'], markersize=8)
        ax.errorbar(x + .12, r.PRM_auc, yerr=[[r.PRM_auc - r.PRM_lo], [r.PRM_hi - r.PRM_auc]], fmt='o', color=T['ref'], markersize=8, capsize=3)
        ax.plot(x + .32, r.step_index_auc, 'D', mfc='none', mec=T['muted'], mew=1.6, markersize=7)
        top = max(L.auc.max(), r.PRM_hi, r.step_index_auc)
        txt = f'{r.clearly_above_PRM}/{r.n_configs} above' if r.clearly_above_PRM >= r.clearly_below_PRM else f'{r.clearly_below_PRM}/{r.n_configs} below'
        ax.annotate(txt, (x, top), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9, color=T['ink'] if r.clearly_above_PRM >= r.clearly_below_PRM else T['ink2'],
                    fontweight='bold' if r.clearly_above_PRM >= r.clearly_below_PRM else 'normal')
    ax.set_xticks(range(len(S))); ax.set_xticklabels([f'{LAB.get(t, t)}\nn={n}' for t, n in zip(S.type, S.n_answers)], color=T['ink'], fontsize=9.5)
    ax.set_ylabel('Within-answer AUC (PRMBench)', color=T['ink2'])
    hs = [plt.Line2D([], [], marker='o', ls='', color=T['ours'], markersize=8, label='80 label-free configurations (dot = median, bar = range)'),
          plt.Line2D([], [], marker='o', ls='', color=T['ref'], markersize=8, label='Supervised PRM-7B (95% interval)'),
          plt.Line2D([], [], marker='D', ls='', mfc='none', mec=T['muted'], markersize=7, label='Step index alone')]
    ax.legend(handles=hs, loc='lower center', bbox_to_anchor=(.5, -.3), ncol=3, frameon=False, labelcolor=T['ink'], fontsize=9.5)
    ax.set_title('PRMBench by error type: label-free fusion vs the supervised PRM (text = configurations clearly above or below the PRM)', color=T['ink'], fontsize=11, loc='left')
    fig.text(.01, -.1, '80 configurations = 4 pipeline methods (where complete), plain average, oriented plain average, continuous L-SML and IU-PCR, on 10 nested banks. '
             '"Clearly" = paired source-group 95% interval excludes 0.', color=T['ink2'], fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / 'figures/T1_types_all_configs_light.png', dpi=170, bbox_inches='tight', facecolor=T['bg'])
    fig.savefig(OUT / 'figures/T1_types_all_configs_light.svg', bbox_inches='tight', facecolor=T['bg']); plt.close(fig)




def fig_singles():
    S = pd.read_csv(OUT / 'SINGLE_CHANNELS.csv').set_index('channel')
    V2 = OUT.parents[0] / 'monday_nested_banks_v2'; CB = OUT.parents[0] / 'monday_cont_baselines_v1'
    A = pd.read_csv(V2 / 'PANEL_A_ABSOLUTE.csv'); C = pd.read_csv(CB / 'ABSOLUTE.csv')
    M4 = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
    best_b = S.pb_hit.idxmax()
    singles = [('realized_drv', 'Surprise rise\n(best on\nPRMBench)'), (best_b, 'Digit spread\n(best on\nProcessBench)'), ('ct7_ve0', 'Varentropy\n(best of 4)'), ('q15_H1', 'Entropy')]
    fus = [('Plain\naverage', A[(A.bank == 'N16') & (A.scorer == 'ALL_EQ')]), ('Oriented\naverage', C[(C.bank == 'N16') & (C.scorer == 'ALL_EQ_ORIENTED')]),
           ('IU-PCR', C[(C.bank == 'N16') & (C.scorer == 'IU_PCR')]), ('Continuous\nL-SML', C[(C.bank == 'N16') & (C.scorer == 'LSML_CONT')]),
           ('Our 4\nmethods', A[(A.bank == 'N16') & A.scorer.isin(M4)])]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.4), facecolor=T['bg'])
    for ax, (col, lo, hi, scol, title, yl) in zip(axes, [('prmb_auc', 'prmb_lo', 'prmb_hi', 'prmb_within_auc', 'PRMBench: within-answer AUC', 'Within-answer AUC'),
                                                        ('pb_hit', 'pb_lo', 'pb_hi', 'pb_argmax_hit', 'ProcessBench: top-ranked step is the first error', 'Top-step hit (8-cell mean)')]):
        style(ax); x = 0; ticks = []
        for ch, lab in singles:
            r = S.loc[ch]; ax.errorbar(x, r[col], yerr=[[r[col] - r[lo]], [r[hi] - r[col]]], fmt='s', color=T['muted'], markersize=8, capsize=3)
            ax.annotate(f'{r[col]:.3f}', (x, r[hi]), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8.5, color=T['ink2']); ticks.append(lab); x += 1
        ax.axvline(x - .5, color=T['grid'], linewidth=1.2)
        for lab, d in fus:
            v = d[scol]
            if len(v) > 1: ax.vlines(x, v.min(), v.max(), color=T['ours'], alpha=.35, linewidth=10)
            ax.plot(x, v.mean(), 'o', color=T['ours'] if lab.startswith('Our') else T['ink'], markersize=8)
            ax.annotate(f'{v.max():.3f}' if len(v) == 1 else f'{v.min():.3f}-{v.max():.3f}', (x, v.max()), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8.5, color=T['ink2'])
            ticks.append(lab); x += 1
        ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, color=T['ink'], fontsize=9)
        ax.text(1.5, 1.02, 'single channels (grey)', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=9.5)
        ax.text(6, 1.02, 'label-free fusion of N16 (16 channels)', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=9.5)
        ax.set_title(title, color=T['ink'], fontsize=11.5, loc='left', pad=22); ax.set_ylabel(yl, color=T['ink2'])
    fig.text(.01, -.06, 'Of the 56 single channels, the median scores 0.63 AUC and 0.24 hit. Single channels: label-free sign (correlation with the level mean), 95% source-group intervals. "Best" channels are picked with labels, per benchmark. '
             'Fusion: N16 bank, fitted label-free per cell.', color=T['ink2'], fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / 'figures/S1_single_vs_fusion_light.png', dpi=170, bbox_inches='tight', facecolor=T['bg'])
    fig.savefig(OUT / 'figures/S1_single_vs_fusion_light.svg', bbox_inches='tight', facecolor=T['bg']); plt.close(fig)


if __name__ == '__main__':
    fig_types(); fig_singles(); print('ok')
