"""Figure C1: the two historical label-free baselines (continuous L-SML, IU-PCR) on the nested banks,
next to the 4 Monday methods and the two plain averages. Names on X, scores on Y."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent; V2 = OUT.parents[0] / 'monday_nested_banks_v2'
M4 = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
FAM = [['N11', 'N13', 'N22', 'N40', 'N53'], ['N14', 'N16', 'N25', 'N43', 'N56']]
T = {'bg': '#fcfcfb', 'ink': '#1b2430', 'ink2': '#5b6575', 'grid': '#e4e2dc', 'ours': '#2a78d6', 'ref': '#eb6834', 'muted': '#898781'}

B = pd.read_csv(OUT / 'ABSOLUTE.csv'); A = pd.read_csv(V2 / 'PANEL_A_ABSOLUTE.csv'); PBK = pd.read_csv(V2 / 'PER_BENCHMARK_COMPLETENESS.csv')
LRS = pd.read_csv(OUT.parents[0] / 'monday_lr_reference_v1/LR_SCORES.csv'); LRS = LRS[LRS.scorer == 'LR']
nch = {k: len(v) for k, v in json.loads((V2 / 'BANKS.json').read_text()).items()}


def val(scorer, bank, metric):
    if scorer in ('LSML_CONT', 'IU_PCR', 'ALL_EQ_ORIENTED'):
        return float(B[(B.scorer == scorer) & (B.bank == bank)][metric].iloc[0])
    if metric == 'prmb_within_auc':
        r = PBK[(PBK.scorer == scorer) & (PBK.bank == bank)]
        return float(r.prmb_within_auc.iloc[0]) if len(r) and r.prmbench_complete.iloc[0] else np.nan
    return float(A[(A.scorer == scorer) & (A.bank == bank)][metric].iloc[0])


fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), facecolor=T['bg'])
for ax, (metric, title, ylab) in zip(axes, [('prmb_within_auc', 'PRMBench: within-answer AUC', 'Within-answer AUC'),
                                            ('pb_argmax_hit', 'ProcessBench: top-ranked step is the first error', 'Top-step hit (8-cell mean)')]):
    ax.set_facecolor(T['bg'])
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(T['muted'])
    ax.grid(axis='y', color=T['grid'], linewidth=.8); ax.set_axisbelow(True)
    for k, fam in enumerate(FAM):
        xs = np.arange(5) + (0 if k == 0 else 6)
        ours = np.array([[val(m, b, metric) for m in M4] for b in fam])
        lo, hi, mean = np.nanmin(ours, 1), np.nanmax(ours, 1), np.nanmean(ours, 1)
        ax.vlines(xs - .2, lo, hi, color=T['ours'], alpha=.35, linewidth=7)
        ax.plot(xs - .2, mean, 'o', color=T['ours'], markersize=7, label='Our 4 methods (dot = mean, bar = range)' if k == 0 else None)
        ax.plot(xs - .07, [val('ALL_EQ', b, metric) for b in fam], 'D', mfc='none', mec=T['muted'], mew=1.6, markersize=7, label='Plain average' if k == 0 else None)
        ax.plot(xs + .06, [val('ALL_EQ_ORIENTED', b, metric) for b in fam], 's', color=T['muted'], markersize=7, label='Oriented plain average' if k == 0 else None)
        ax.plot(xs + .19, [val('LSML_CONT', b, metric) for b in fam], '^', color=T['ref'], markersize=9, label='Continuous L-SML' if k == 0 else None)
        ax.plot(xs + .32, [val('IU_PCR', b, metric) for b in fam], 'v', mfc='none', mec=T['ref'], mew=1.8, markersize=9, label='IU-PCR' if k == 0 else None)
        lr = LRS.set_index('bank')
        ax.plot(xs + .45, [lr.loc[b, metric] for b in fam], '*', color='#1b2430', markersize=11, label='Supervised logistic regression (uses labels)' if k == 0 else None)
    ax.axvline(5, color=T['grid'], linewidth=1)
    ax.set_xticks(list(range(5)) + list(range(6, 11)))
    ax.set_xticklabels([f'{b}\n({nch[b]} ch)' for b in FAM[0] + FAM[1]], color=T['ink'], fontsize=9)
    ax.text(2, 1.02, 'no digit channels', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=9.5)
    ax.text(8, 1.02, '+ 3 digit channels', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=9.5)
    ax.set_title(title, color=T['ink'], fontsize=11.5, pad=22, loc='left'); ax.set_ylabel(ylab, color=T['ink2'])
    ax.tick_params(colors=T['ink2'])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, labelcolor=T['ink'], fontsize=9.5, bbox_to_anchor=(.5, -.02))
fig.text(.01, -.07, 'All scorers except the logistic regression are fitted label-free on each cell, on the same z-scored step channels. Continuous L-SML and IU-PCR: no tail votes, no filter, no merge. '
         'N11 group methods are not reported on ProcessBench (grouping failed in one cell).', color=T['ink2'], fontsize=8.5)
fig.tight_layout(rect=(0, .07, 1, 1))
(OUT / 'figures').mkdir(exist_ok=True)
fig.savefig(OUT / 'figures/C1_baselines_light.png', dpi=170, bbox_inches='tight', facecolor=T['bg'])
fig.savefig(OUT / 'figures/C1_baselines_light.svg', bbox_inches='tight', facecolor=T['bg'])
print('ok')
