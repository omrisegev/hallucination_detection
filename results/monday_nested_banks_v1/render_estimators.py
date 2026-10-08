"""Estimator accuracy vs truth (PRMBench, original Monday banks, answer quota, MEAN within) from
results/estimator_input_v1/run_20261004_resume/ESTIMATOR_ACCURACY.csv. Vertical layout, light + dark."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent; FIG = OUT / 'figures'
SRC = OUT.parents[0] / 'estimator_input_v1/run_20261004_resume/ESTIMATOR_ACCURACY.csv'
E = pd.read_csv(SRC); E = E[(E.quota == 'answer') & (E.within == 'MEAN')]
THEME = {'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', grid='#e1e0d9', axis='#c3c2b7', c1='#2a78d6', c2='#eb6834', ref='#52514e'),
         'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', grid='#2c2c2a', axis='#383835', c1='#3987e5', c2='#d95926', ref='#c3c2b7')}
panels = [('estimated_prevalence', 'Estimated error prevalence\n(dashed = true 0.140)'), ('sensitivity_MAE', 'Sensitivity (ψ) mean abs. error'),
          ('specificity_MAE', 'Specificity (η) mean abs. error'), ('reliability_spearman', 'Spearman: estimated vs true\nchannel reliability')]
for mode, t in THEME.items():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, (col, title) in zip(axes, panels):
        ax.set_facecolor(t['surface'])
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(t['axis'])
        ax.tick_params(colors=t['ink2'], labelsize=10); ax.grid(axis='y', color=t['grid']); ax.set_axisbelow(True)
        for k, (vt, color, lab) in enumerate((('channels', t['c1'], 'channel votes'), ('groups', t['c2'], 'group votes'))):
            for x, est in enumerate(['DS', 'MOM', 'SML']):
                v = E[(E.estimator == est) & (E.vote_type == vt)][col]
                if v.isna().all():
                    if k == 0: ax.text(x, .5, 'ranking\nonly', ha='center', va='center', color=t['ink2'], fontsize=9, transform=ax.get_xaxis_transform())
                    continue
                xx = x + (-.14 if k == 0 else .14)
                ax.plot([xx, xx], [v.min(), v.max()], color=color, linewidth=6, alpha=.35, solid_capstyle='round')
                ax.plot(xx, v.mean(), 'o', color=color, markersize=9, markeredgecolor=t['surface'], markeredgewidth=2, label=lab if x == 0 else None)
                ax.annotate(f'{v.mean():.3f}', (xx, v.max()), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8.5, color=t['ink2'])
        if col == 'estimated_prevalence': ax.axhline(.1396, color=t['ref'], linestyle='--', linewidth=1.4)
        ax.set_xticks(range(3)); ax.set_xticklabels(['Dawid-Skene EM', 'Tensor MoM', 'SML'], color=t['ink'], fontsize=10)
        ax.set_title(title, color=t['ink'], fontsize=10.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, labelcolor=t['ink'], bbox_to_anchor=(.5, -.05), fontsize=10)
    fig.suptitle('How close are the label-free estimates to the truth? (PRMBench, 94,203 steps; dot = mean of 8 banks, bar = range)', color=t['ink'], fontsize=12, x=.01, ha='left')
    fig.patch.set_facecolor(t['surface']); fig.tight_layout(rect=(0, .05, 1, 1))
    for ext in ('png', 'svg'): fig.savefig(FIG / f'C1_estimator_accuracy_{mode}.{ext}', dpi=180, facecolor=t['surface'], bbox_inches='tight')
    plt.close(fig)
print('done')
