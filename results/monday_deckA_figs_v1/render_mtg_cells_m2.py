"""Deck A figure M2 (= M1 plus the N16 plain average): ProcessBench SLA per cell, ours vs Mind the Gap.

Cells = 4 ProcessBench subsets x 2 scoring models (Qwen3-4B, Qwen3-8B); wrong answers only.
- Ours: bank N16, the 4 registered pipeline variants (HEM_MOM_cc, MEAN_MOM_cc, MEAN_MOM_ga, TCI_ONLY_ga);
  bar = mean of the 4, whisker = min..max. Top-ranked step (argmax) must equal the first error; every wrong
  answer counted. Source: monday_nested_banks_v2/PER_ANSWER.npz (8-cell means replay PANEL_B_PROCESSBENCH.csv).
- Mind the Gap as we run it: same file, scorer MTG (EVIDENCE_DROP step_m5: their per-token signal, our
  step rule = mean of the 5 largest drops in the step, argmax, every wrong answer counted).
- Mind the Gap paper: Table 3, Shannon Drop column (Chen et al., ICML 2026), context only; its step rule
  (first step past an undefined threshold) and its handling of answers where nothing fires are not given.
CPU only, no new inference.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
V2 = OUT.parents[0] / 'monday_nested_banks_v2'
T = {'bg': '#fcfcfb', 'ink': '#1b2430', 'ink2': '#5b6575', 'grid': '#e4e2dc', 'ours': '#2a78d6', 'mtg': '#eb6834', 'eq': '#24508a', 'muted': '#898781'}
M4 = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
CELLS = [('gsm8k', 'GSM8K'), ('math', 'MATH'), ('olympiadbench', 'Olympiad'), ('omnimath', 'Omni-MATH')]
MODELS = [('q4', '4B'), ('q8', '8B')]
# Mind the Gap, Table 3, Shannon Drop (%), papers/extracted/mind-the-gap-...md lines 887-937
PAPER = {('gsm8k', 'q4'): 43.42, ('math', 'q4'): 32.03, ('olympiadbench', 'q4'): 43.06, ('omnimath', 'q4'): 38.04,
         ('gsm8k', 'q8'): 46.11, ('math', 'q8'): 32.90, ('olympiadbench', 'q8'): 41.52, ('omnimath', 'q8'): 37.04}

d = np.load(V2 / 'PER_ANSWER.npz', allow_pickle=True)
cells = d['cells'].astype(str); tgt = d['target']
arg = {m: d[f'N16__{m}__argmax'] for m in M4}; mtg = d['MTG__argmax']; eq = d['N16__ALL_EQ__argmax']

rows = []
for s, slab in CELLS:
    for q, qlab in MODELS:
        msk = (cells == f'pb_{s}_{q}') & (tgt >= 0)
        o = np.array([(arg[m][msk] == tgt[msk]).mean() for m in M4])
        rows.append({'cell': f'pb_{s}_{q}', 'label': f'{slab}\n{qlab}', 'n_wrong': int(msk.sum()),
                     'ours_mean4': float(o.mean()), 'ours_min': float(o.min()), 'ours_max': float(o.max()),
                     'ours_by_method': dict(zip(M4, map(float, o))),
                     'mtg_ours_run': float((mtg[msk] == tgt[msk]).mean()), 'plain_avg': float((eq[msk] == tgt[msk]).mean()), 'mtg_paper': PAPER[(s, q)] / 100})
mean = {k: float(np.mean([r[k] for r in rows])) for k in ('ours_mean4', 'mtg_ours_run', 'mtg_paper', 'plain_avg')}
mean_by_method = {m: float(np.mean([r['ours_by_method'][m] for r in rows])) for m in M4}
mean['ours_min'] = min(mean_by_method.values()); mean['ours_max'] = max(mean_by_method.values())
# replay check against the frozen panel (8-cell equal mean)
import csv
panel = {(r['scorer'], r['bank']): float(r['argmax_hit']) for r in csv.DictReader(open(V2 / 'PANEL_B_PROCESSBENCH.csv')) if r['stratum'] == 'ALL'}
for m in M4: assert abs(panel[(m, 'N16')] - mean_by_method[m]) < 1e-9, m
assert abs(panel[('Mind the Gap (local repro.)', '-')] - mean['mtg_ours_run']) < 1e-9
assert abs(mean['mtg_paper'] - 0.39265) < 1e-9
assert abs(panel[('ALL_EQ', 'N16')] - mean['plain_avg']) < 1e-9
json.dump({'rows': rows, 'mean_8cell': mean, 'ours_mean_by_method': mean_by_method,
           'ours_beats_mtg_run_cells_min_of_4': sum(r['ours_min'] > r['mtg_ours_run'] for r in rows),
           'ours_mean_above_paper_cells': sum(r['ours_mean4'] > r['mtg_paper'] for r in rows),
           'plain_avg_above_mtg_run_cells': sum(r['plain_avg'] > r['mtg_ours_run'] for r in rows),
           'plain_avg_above_paper_cells': sum(r['plain_avg'] > r['mtg_paper'] for r in rows)},
          open(OUT / 'M2_mtg_per_cell.json', 'w'), indent=1)

labels = [r['label'] for r in rows] + ['8-cell\nmean']
O = [r['ours_mean4'] for r in rows] + [mean['ours_mean4']]
LO = [r['ours_min'] for r in rows] + [mean['ours_min']]; HI = [r['ours_max'] for r in rows] + [mean['ours_max']]
R = [r['mtg_ours_run'] for r in rows] + [mean['mtg_ours_run']]
P = [r['mtg_paper'] for r in rows] + [mean['mtg_paper']]
Q = [r['plain_avg'] for r in rows] + [mean['plain_avg']]

plt.rcParams['hatch.linewidth'] = 1.6
fig, ax = plt.subplots(figsize=(16.6, 4.7), facecolor=T['bg']); ax.set_facecolor(T['bg'])
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'): ax.spines[sp].set_color(T['muted'])
ax.grid(axis='y', color=T['grid'], linewidth=.9); ax.set_axisbelow(True)
x = np.arange(len(labels)) * 1.5; x[-1] += .4; w = .27
W = np.full(len(labels), w); W[-1] = w * 1.45  # wider spacing in the mean group so the value labels do not collide
ax.bar(x - 1.5 * W, O, w * .92, color=T['ours'], edgecolor=T['bg'], linewidth=2)
ax.vlines(x - 1.5 * W, LO, HI, color=T['ink'], linewidth=1.8)
ax.bar(x - .5 * W, Q, w * .92, color=T['eq'], edgecolor=T['bg'], linewidth=2)
ax.bar(x + .5 * W, R, w * .92, color=T['mtg'], edgecolor=T['bg'], linewidth=2)
ax.bar(x + 1.5 * W, P, w * .92, color='white', edgecolor=T['mtg'], hatch='//', linewidth=1.6)
ax.axvline((x[-2] + x[-1]) / 2, color=T['muted'], linewidth=1, linestyle=(0, (3, 3)))
LAB = [(x[-1] - 1.5 * W[-1], max(O[-1], HI[-1]), O[-1], 4), (x[-1] - .5 * W[-1], Q[-1], Q[-1], 24),
       (x[-1] + .5 * W[-1], R[-1], R[-1], 4), (x[-1] + 1.5 * W[-1], P[-1], P[-1], 4)]
for xx, ytop, v, dy in LAB:
    ax.annotate(f'{v:.3f}'.lstrip('0'), (xx, ytop), xytext=(0, dy), textcoords='offset points',
                ha="center", fontsize=14, color=T["ink"], fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=16, color=T['ink'])
ax.set_xlabel('ProcessBench subset, scoring model (Qwen3-4B / Qwen3-8B)', fontsize=16, color=T['ink2'])
ax.set_ylim(0, .62); ax.set_yticks([0, .1, .2, .3, .4, .5, .6]); ax.tick_params(axis='y', labelsize=16, colors=T['ink2'])
ax.set_ylabel('SLA: exact first error,\nwrong answers only', fontsize=17, color=T['ink2'])
hs = [Patch(facecolor=T['ours'], label='Ours, N16 pipeline (mean of 4 variants; line = range)'),
      Patch(facecolor=T['eq'], label='Ours, N16 plain average'),
      Patch(facecolor=T['mtg'], label='Mind the Gap, our run'),
      Patch(facecolor='white', edgecolor=T['mtg'], hatch='//', label='Mind the Gap, paper (Table 3)')]
ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(.5, 1.19), ncol=4, frameon=False, fontsize=14, labelcolor=T['ink'],
          handlelength=1.6, columnspacing=1.6)
fig.tight_layout()
fig.savefig(OUT / 'M2_mtg_per_cell_light.png', dpi=110, bbox_inches='tight', facecolor=T['bg'])
print(json.dumps(mean), json.dumps(mean_by_method))
for r in rows: print(r['cell'], r['n_wrong'], round(r['ours_mean4'], 3), round(r['ours_min'], 3), round(r['ours_max'], 3), round(r['mtg_ours_run'], 3), r['mtg_paper'], round(r['plain_avg'], 3))
