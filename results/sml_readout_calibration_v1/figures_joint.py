"""Charts for joint.py: (3) change vs A0 with 95% CI for A2, B1, B2; (4) conditional |corr| of the votes given the label."""
import csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
P = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']; PL = ['7B', '72B', 'PURE', 'RLHFlow']
B = ['prmbench', 'socratic', 'hard2verify', 'processbench']
TITLE = {'prmbench': 'PRMBench (PRMScore)', 'socratic': 'Socratic (PRMScore)', 'hard2verify': 'Hard2Verify (balanced F1)', 'processbench': 'ProcessBench (macro SLA)'}
ARMS = [('A2 min rank-1 misfit', 'A2 readout only (earlier run)', '#2a78d6', ''), ('B1 joint readout+grouping', 'B1 readout + grouping', '#eb6834', '__joint'),
        ('B2 joint readout+vote rule+grouping', 'B2 readout + vote rule + grouping', '#1baf7a', '__joint')]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))
for ax, b in zip(axs, B):
    for i, (arm, lab, col, suf) in enumerate(ARMS):
        R = {(r['prm'], r['arm'].split(' MINUS')[0]): r for r in csv.DictReader(open(HERE / f'BOOT{suf}_{b}.csv')) if r['kind'] == 'diff'}
        v = np.array([float(R[(p, arm)]['value']) for p in P]); lo = np.array([float(R[(p, arm)]['lo']) for p in P]); hi = np.array([float(R[(p, arm)]['hi']) for p in P])
        ax.errorbar(np.arange(4) + (i - 1) * .22, v, yerr=[v - lo, hi - v], fmt='o', ms=6, color=col, lw=2, label=lab, mec='white', mew=1)
    ax.axhline(0, color='#888', lw=1); ax.set_xticks(range(4)); ax.set_xticklabels(PL); ax.set_title(TITLE[b], fontsize=11)
    ax.grid(axis='y', color='#e6e6e6', lw=.8); ax.set_axisbelow(True)
axs[2].set_ylim(-30, 10); axs[0].set_ylabel('change vs top-10 readout (points, 95% CI)')
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(HERE / 'fig3_joint_change_vs_top10.png', dpi=160)

V = {}
for b in B:
    for r in csv.DictReader(open(HERE / f'RESULTS_joint_{b}.csv')):
        if r['metric'] == 'cond |corr| votes y=0': V[(b, r['prm'], r['arm'])] = float(r['value'])
fig, ax = plt.subplots(figsize=(10, 4.2))
cols = [('A0 top10', '#9a9a9a', 'A0 top-10 (current)'), ('B1 joint readout+grouping', '#eb6834', 'B1'), ('B2 joint readout+vote rule+grouping', '#1baf7a', 'B2')]
x = 0; ticks, tl = [], []
for b in B:
    for p, pl in zip(P, PL):
        for i, (arm, col, lab) in enumerate(cols):
            ax.bar(x + (i - 1) * .27, V[(b, p, arm)], width=.25, color=col, edgecolor='white', lw=1.5, label=lab if x == 0 else None)
        ticks.append(x); tl.append(pl); x += 1
    x += .6
for k, b in enumerate(B):
    ax.text(k * 4.6 + 1.5, .232, TITLE[b].split(' (')[0], ha='center', fontsize=10, color='#333')
ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=8); ax.set_ylim(0, .24)
ax.set_ylabel('mean |corr| between votes, correct steps')
ax.grid(axis='y', color='#e6e6e6', lw=.8); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.14)); fig.tight_layout(); fig.savefig(HERE / 'fig4_votes_conditional_corr.png', dpi=160)
