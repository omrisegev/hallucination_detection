"""domfreq_variants_v1 plots (drawn off-cluster from the small SCATTER.npz / HIST_TOP.npz / TOP_TABLE.json payloads).

    python plots.py <dir with SCATTER.npz HIST_TOP.npz TOP_TABLE.json>
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(sys.argv[1])
PRMS = ('qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm')
LAB = {'qwen_prm_7b': 'Qwen-7B', 'qwen_prm_72b': 'Qwen-72B', 'pure_prm': 'PURE', 'rlhflow_prm': 'RLHFlow'}
S = np.load(D / 'SCATTER.npz'); names = S['names'].astype(str)
top = json.load(open(D / 'TOP_TABLE.json'))['top_by_min_within'][:6]
ref = np.array([n.startswith('entropyREF') for n in names])
sel = np.isin(names, [t['name'] for t in top])

fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
for r, bench in enumerate(('prmbench', 'socratic')):
    for c, p in enumerate(PRMS):
        ax = axes[r, c]; A = S[f'{bench}__{p}']
        ent = A[:, 2] > .5
        ax.scatter(A[~ref & ~ent, 1], A[~ref & ~ent, 0], s=5, c='#3b6fb6', alpha=.45, label='non-entropy source, max|rho_ent|<=.5')
        ax.scatter(A[~ref & ent, 1], A[~ref & ent, 0], s=5, c='#e08a2c', alpha=.45, label='non-entropy source, max|rho_ent|>.5')
        ax.scatter(A[ref, 1], A[ref, 0], s=5, c='#9a9a9a', alpha=.45, label='entropy (reference source)')
        ax.scatter(A[sel, 1], A[sel, 0], s=40, facecolors='none', edgecolors='k', lw=1.2, label='top 6 (table)')
        ax.axhline(.5, c='k', lw=.6, ls=':')
        ax.set_title(f'{"PRMBench" if bench == "prmbench" else "Socratic"} / {LAB[p]}', fontsize=10)
        if r == 1: ax.set_xlabel('R^2 on the 16 existing channels (normal scores)')
        if c == 0: ax.set_ylabel('within-answer AUC (PRMBench sign)')
axes[0, 0].legend(fontsize=7, loc='lower right')
fig.suptitle('dominant_freq16 variants: signal vs redundancy, every variant, every cell (development data)')
fig.tight_layout(); fig.savefig(D / 'auc_vs_redundancy.png', dpi=130)

H = np.load(D / 'HIST_TOP.npz')
fig, axes = plt.subplots(2, 6, figsize=(20, 6.5))
for r, (bench, p) in enumerate((('prmbench', 'qwen_prm_7b'), ('socratic', 'qwen_prm_7b'))):
    y = H[f'{bench}/{p}/__y'].astype(bool)
    for c, t in enumerate(top):
        ax = axes[r, c]; v = H[f'{bench}/{p}/{t["name"]}'].astype(float)
        lo, hi = np.percentile(v, [.5, 99.5]); bins = np.linspace(lo, hi, 60)
        ax.hist(v[~y], bins, density=True, alpha=.55, color='#3b6fb6', label='valid step')
        ax.hist(v[y], bins, density=True, alpha=.55, color='#c0392b', label='error step')
        ax.set_title(t['name'].replace('|', ' | '), fontsize=7.5)
        if c == 0: ax.set_ylabel(f'{"PRMBench" if bench == "prmbench" else "Socratic"} / Qwen-7B')
axes[0, 0].legend(fontsize=7)
fig.suptitle('Top 6 variants: step value distribution, error vs valid')
fig.tight_layout(); fig.savefig(D / 'top6_histograms.png', dpi=130)
print('wrote', D / 'auc_vs_redundancy.png', D / 'top6_histograms.png')
