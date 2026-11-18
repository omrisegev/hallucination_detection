"""Charts: (1) paired change vs A0 with 95% CI per benchmark; (2) ProcessBench macro SLA vs the equal-mean reference."""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
P = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']; PL = ['7B', '72B', 'PURE', 'RLHFlow']
ARMS = [('A2', 'A2 min rank-1 misfit', '#2a78d6'), ('A3', 'A3 residualize + min misfit', '#eb6834'), ('A4', 'A4 residualize, top10', '#1baf7a')]
TITLE = {'prmbench': 'PRMBench (PRMScore)', 'socratic': 'Socratic (PRMScore)', 'hard2verify': 'Hard2Verify (balanced F1)', 'processbench': 'ProcessBench (macro SLA)'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), sharey=False)
for ax, b in zip(axs, ['prmbench', 'socratic', 'hard2verify', 'processbench']):
    R = {(r['prm'], r['arm'].split(' MINUS')[0]): r for r in csv.DictReader(open(HERE / f'BOOT_{b}.csv')) if r['kind'] == 'diff'}
    for i, (short, arm, col) in enumerate(ARMS):
        x = np.arange(4) + (i - 1) * .22
        v = np.array([float(R[(p, arm)]['value']) for p in P]); lo = np.array([float(R[(p, arm)]['lo']) for p in P]); hi = np.array([float(R[(p, arm)]['hi']) for p in P])
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt='o', ms=6, color=col, lw=2, capsize=0, label=arm, mec='white', mew=1)
    ax.axhline(0, color='#888', lw=1); ax.set_xticks(range(4)); ax.set_xticklabels(PL); ax.set_title(TITLE[b], fontsize=11)
    ax.grid(axis='y', color='#e6e6e6', lw=.8); ax.set_axisbelow(True)
axs[0].set_ylabel('change vs top-10 readout (points, 95% CI)')
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(HERE / 'fig1_change_vs_top10.png', dpi=160)

# fig 2: PB macro SLA absolute
R = {(r['prm'], r['arm']): float(r['value']) for r in csv.DictReader(open(HERE / 'RESULTS_processbench.csv')) if r['metric'] == 'macro SLA'}
Q = {r['arm']: float(r['value']) for r in csv.DictReader(open(HERE / 'RESULTS_qwen3_8b_pb.csv')) if r['metric'] == 'macro SLA'}
MEAN = [38.4, 38.3, 33.6, 35.3, 36.9]   # mean+GAP, gsp_ablation_baseline_v1 (memory project-pb-inverted-channels-2026-11-12)
fig, ax = plt.subplots(figsize=(9, 4.2))
cols = [('A0 top10', '#9a9a9a', 'A0 top-10 (current)'), ('A2 min rank-1 misfit', '#2a78d6', 'A2 min rank-1 misfit'), ('A4 residualize, top10', '#1baf7a', 'A4 residualize, top-10')]
for i, (arm, col, lab) in enumerate(cols):
    v = [R[(p, arm)] for p in P] + ([Q['A0 top10 | GAP']] if arm.startswith('A0') else [np.nan] if arm.startswith('A2') else [Q['A4 residualize (position only) | GAP']])
    ax.bar(np.arange(5) + (i - 1) * .26, v, width=.24, color=col, label=lab, edgecolor='white', lw=2)
for j, m in enumerate(MEAN):
    ax.plot([j - .42, j + .42], [m, m], color='#333', lw=2, ls='--', label='equal mean + GAP' if j == 0 else None)
ax.set_xticks(range(5)); ax.set_xticklabels(PL + ['Qwen3-8B\n(A2 n/a)']); ax.set_ylabel('ProcessBench macro SLA')
ax.set_ylim(15, 42); ax.grid(axis='y', color='#e6e6e6', lw=.8); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.13)); fig.tight_layout(); fig.savefig(HERE / 'fig2_pb_sla_vs_mean.png', dpi=160)
