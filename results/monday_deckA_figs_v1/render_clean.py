"""Deck A figure K1: the clean-answer decision on ProcessBench (official F1 and its two parts), from existing results only."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent; V2 = OUT.parents[0] / 'monday_nested_banks_v2'
T = {'bg': '#fcfcfb', 'ink': '#1b2430', 'ink2': '#5b6575', 'grid': '#e4e2dc', 'ours': '#2a78d6', 'ref': '#eb6834', 'muted': '#898781'}
F = pd.read_csv(V2 / 'PANEL_B_PROCESSBENCH_F1.csv'); M4 = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
o = F[(F.bank == 'N16') & F.scorer.isin(M4)]
rows = [('Ours N16\ntop-20% rule', o[o.rule == 'ANSWER_Z_Q80'], 'ours'), ('Ours N16\ncount rule', o[o.rule == 'R2_COUNT'], 'ours'),
        ('Supervised\nPRM-7B (0.5)', F[F.scorer == 'Qwen2.5-Math-PRM-7B native (0.5)'], 'ref'), ('72B LLM judge\n(prompted)', F[F.scorer == 'Qwen2.5-72B critic native'], 'ref')]
series = [('error_hit', 'First error found (wrong answers)', 'o'), ('clean_correct', 'Left clean (correct answers)', 's'), ('official_f1', 'Official F1 (harmonic mean)', 'D')]
fig, ax = plt.subplots(figsize=(13, 5), facecolor=T['bg']); ax.set_facecolor(T['bg'])
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'): ax.spines[sp].set_color(T['muted'])
ax.grid(axis='y', color=T['grid'], linewidth=.8); ax.set_axisbelow(True); ax.tick_params(colors=T['ink2'])
for x, (lab, d, kind) in enumerate(rows):
    for k, (col, slab, mk) in enumerate(series):
        v = d[col]; xx = x + (k - 1) * .22; c = T['ours'] if kind == 'ours' else T['ref']
        if len(v) > 1: ax.vlines(xx, v.min(), v.max(), color=c, alpha=.3, linewidth=8)
        ax.plot(xx, v.mean(), mk, color=c, mfc=c if col != 'official_f1' else 'none', mew=1.8, markersize=9)
        ax.annotate(f'{v.mean():.2f}', (xx, v.max()), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8.5, color=T['ink2'])
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], color=T['ink'], fontsize=10)
ax.set_ylim(0, 1.0); ax.set_ylabel('Share (8-cell mean)', color=T['ink2'])
hs = [plt.Line2D([], [], marker=m, ls='', color=T['ink2'], mfc=T['ink2'] if c != 'official_f1' else 'none', markersize=8, label=l) for c, l, m in series]
ax.legend(handles=hs, loc='lower center', bbox_to_anchor=(.5, -.32), ncol=3, frameon=False, labelcolor=T['ink'], fontsize=9.5)
ax.set_title('ProcessBench with the clean-answer decision: finding the error vs leaving correct answers alone', color=T['ink'], fontsize=11.5, loc='left')
fig.text(.01, -.1, 'Blue = ours (N16, 4 methods: dot = mean, bar = range); orange = references reproduced on our cells. Top-20% rule: flag steps in the top 20% of the cell. '
         'Count rule: flag as many steps as the answer has in the cell\'s top 20% of the raw channel average; none = clean. Source: PANEL_B_PROCESSBENCH_F1.csv.', color=T['ink2'], fontsize=8.3)
fig.tight_layout(); fig.savefig(OUT / 'K1_clean_decision_light.png', dpi=170, bbox_inches='tight', facecolor=T['bg']); print('ok')
