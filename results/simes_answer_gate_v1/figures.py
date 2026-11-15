"""Figures for the Simes answer gate run. python -B figures.py"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
LBL = ['Qwen2.5-Math\n7B-Instruct', 'Qwen2.5-Math\n72B-Instruct', 'Qwen2.5-Math\n7B (base)', 'Llama-3.1\n8B-Instruct']
C1, C2, INK, MUTED, GRID, BG = '#2a78d6', '#eb6834', '#1f1f1e', '#6b6b68', '#e6e6e3', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': BG, 'axes.facecolor': BG})
B = list(csv.DictReader(open(HERE / 'BOOT.csv')))
R = list(csv.DictReader(open(HERE / 'RESULTS.csv')))
S = list(csv.DictReader(open(HERE / 'SWEEP.csv')))


def boot(bench, prm, gated):
    for r in B:
        if r['bench'] == bench and r['prm'] == prm and r['kind'] == 'abs' and (('Simes' in r['arm']) == gated):
            return float(r['value']), float(r['lo']), float(r['hi'])


def res(bench, prm, arm, metric):
    for r in R:
        if r['bench'] == bench and r['prm'] == prm and r['arm'] == arm and r['metric'] == metric:
            return float(r['value'])


# fig1: main result
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(4); w = .36
for ax, bench, title, yl in ((axs[0], 'prmbench', 'PRMBench: official PRMScore', 'PRMScore'),
                             (axs[1], 'processbench', 'ProcessBench: official average F1', 'average F1')):
    for k, (gated, col, name) in enumerate(((False, C1, 'no gate'), (True, C2, 'Simes gate, alpha = .05'))):
        v = np.array([boot(bench, p, gated) for p in PRMS])
        ax.bar(x + (k - .5) * w, v[:, 0], w - .03, color=col, label=name, zorder=2)
        ax.errorbar(x + (k - .5) * w, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt='none', ecolor=INK, lw=1, capsize=2, zorder=3)
        for xi, vi, hi in zip(x + (k - .5) * w, v[:, 0], v[:, 2]):
            ax.text(xi, hi + .5, f'{vi:.1f}', ha='center', va='bottom', fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(LBL, fontsize=8); ax.set_ylabel(yl); ax.set_title(title, color=INK, loc='left')
    ax.grid(axis='y', color=GRID, zorder=0)
    ax.set_ylim(0, 90 if bench == 'prmbench' else 30)
axs[0].legend(frameon=False, loc='upper left', ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(HERE / 'fig1_main.png', dpi=160); plt.close(fig)

# fig2: what the gate closes vs true clean share
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, bench, title in ((axs[0], 'prmbench', 'PRMBench'), (axs[1], 'processbench', 'ProcessBench')):
    cl = [res(bench, p, 'gate', 'closed_among_clean') for p in PRMS]
    wr = [res(bench, p, 'gate', 'closed_among_wrong') for p in PRMS]
    tc = res(bench, PRMS[0], 'gate', 'true_clean_share'); gc = [res(bench, p, 'gate', 'gate_closed_share') for p in PRMS]
    ax.bar(x - w / 2, cl, w - .03, color=C1, label='closed among clean answers', zorder=2)
    ax.bar(x + w / 2, wr, w - .03, color=C2, label='closed among wrong answers', zorder=2)
    ax.axhline(tc, color=INK, ls='--', lw=1); ax.text(3.6, tc + 2, f'true clean share {tc:.1f}%', ha='right', fontsize=8, color=INK)
    for xi, a, b in zip(x, cl, wr):
        ax.text(xi - w / 2, a + 1, f'{a:.0f}', ha='center', fontsize=8); ax.text(xi + w / 2, b + 1, f'{b:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(LBL, fontsize=8); ax.set_ylim(0, 115); ax.set_ylabel('% of answers called clean')
    ax.set_title(f'{title}: gate closes {np.mean(gc):.0f}% of all answers', color=INK, loc='left'); ax.grid(axis='y', color=GRID, zorder=0)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc='lower center', ncol=2, fontsize=8)
fig.tight_layout(); fig.subplots_adjust(bottom=.25); fig.savefig(HERE / 'fig2_gate_closed.png', dpi=160); plt.close(fig)

# fig3: diagnostic sweep
cols = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, bench, title, yl in ((axs[0], 'prmbench', 'PRMBench (diagnostic)', 'PRMScore'), (axs[1], 'processbench', 'ProcessBench (diagnostic)', 'average F1')):
    for p, l, c in zip(PRMS, LBL, cols):
        rr = [r for r in S if r['bench'] == bench and r['prm'] == p]
        cl = [float(r['closed']) for r in rr]; v = [float(r['metric']) for r in rr]
        ax.plot(cl, v, '-o', color=c, lw=2, ms=4, label=l)
    ax.axvline(res(bench, PRMS[0], 'gate', 'true_clean_share'), color=INK, ls='--', lw=1)
    ax.text(res(bench, PRMS[0], 'gate', 'true_clean_share') + 1, ax.get_ylim()[0] + 1, 'true clean share', fontsize=8)
    ax.axvline(94.4, color=MUTED, ls=':', lw=1); ax.text(93, ax.get_ylim()[0] + 3, 'alpha = .05', fontsize=8, ha='right', color=MUTED)
    ax.set_xlabel('% of answers the gate closes (alpha from 1.0 down to .001)'); ax.set_ylabel(yl)
    ax.set_title(title, color=INK, loc='left'); ax.grid(color=GRID)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc='lower center', ncol=4)
fig.tight_layout(); fig.subplots_adjust(bottom=.27); fig.savefig(HERE / 'fig3_sweep_diagnostic.png', dpi=160); plt.close(fig)
print('ok')
