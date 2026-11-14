"""Figures for the cross-model answer gate check (reads AUROC.csv, COND_CORR.csv, DS_PI.csv). python figures.py"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
NAME = {'processbench': 'ProcessBench', 'prmbench': 'PRMBench', 'socratic': 'Socratic', 'hard2verify': 'Hard2Verify',
        'crv_arith': 'CRV-Arith', 'crv_gsm8k': 'CRV-GSM8K'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.facecolor': BG, 'figure.facecolor': BG, 'axes.spines.top': False,
                     'axes.spines.right': False})


def rd(n):
    return list(csv.DictReader(open(HERE / n)))


def style(ax):
    ax.grid(axis='y', color=GRID, lw=.8); ax.set_axisbelow(True)


def fig_auc():
    R = [r for r in rd('AUROC.csv') if r['stat'] == 'simes']
    B = list(dict.fromkeys(r['bench'] for r in R)); M = list(dict.fromkeys(r['model'] for r in R))
    L = {r['bench']: float(r['auc_pooled']) for r in rd('AUROC.csv') if r['stat'] == 'n_steps'}
    fig, ax = plt.subplots(figsize=(9, 4.2)); w = .8 / len(M)
    for j, m in enumerate(M):
        v = [next((float(r['auc_len_strat']) for r in R if r['bench'] == b and r['model'] == m), np.nan) for b in B]
        ax.bar(np.arange(len(B)) + (j - (len(M) - 1) / 2) * w, v, w * .9, color=C[j], label=m)
    for i, b in enumerate(B):
        ax.plot([i - .42, i + .42], [L[b]] * 2, color=INK, lw=1.5, ls=':', label='length only (n_steps)' if i == 0 else None)
    ax.axhline(.5, color=INK2, lw=.8)
    ax.set_xticks(range(len(B)), [NAME[b] for b in B]); ax.set_ylim(.4, 1); ax.set_ylabel('answer-level AUROC (within length bins)')
    ax.set_title('(1) Simes answer statistic per base model: answer-error AUROC, length-stratified', loc='left', color=INK)
    style(ax); ax.legend(frameon=False, ncol=5, fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig(HERE / 'fig1_auroc.png', dpi=200); plt.close(fig)


def fig_corr():
    R = rd('COND_CORR.csv'); B = list(dict.fromkeys(r['bench'] for r in R))
    groups = [('cross-model, main triplet pairs', lambda r: r['kind'] == 'cross-model' and 'base' not in r['pair'], C[0], 'o'),
              ('cross-model, 7B-Inst vs 7B-base', lambda r: r['pair'] == '7B-Inst vs 7B-base', C[1], 's'),
              ('within-model A_mean vs Simes', lambda r: r['kind'] == 'within-model A-S', C[2], 'D')]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for j, (lab, f, col, mk) in enumerate(groups):
        for i, b in enumerate(B):
            v = [float(r['cond_abs_corr']) for r in R if r['bench'] == b and f(r)]
            ax.scatter(np.full(len(v), i + (j - 1) * .22), v, s=40, color=col, marker=mk, edgecolor=BG, lw=1.5,
                       label=lab if i == 0 else None, zorder=3)
    ax.axhline(.26, color=INK, lw=1, ls='--'); ax.text(len(B) - .5, .265, 'earlier within-model ref .26', ha='right', fontsize=8, color=INK)
    ax.axhline(.20, color=INK2, lw=1, ls=':'); ax.text(len(B) - .5, .205, 'go bar .20', ha='right', fontsize=8, color=INK2)
    ax.set_xticks(range(len(B)), [NAME[b] for b in B]); ax.set_ylim(0, max(.6, ax.get_ylim()[1]))
    ax.set_ylabel('|Spearman| given the answer label'); style(ax)
    ax.set_title('(2) Conditional dependence of the answer statistics', loc='left', color=INK)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout(); fig.savefig(HERE / 'fig2_cond_corr.png', dpi=200); plt.close(fig)


def fig_pi():
    R = [r for r in rd('DS_PI.csv') if r['triplet'].startswith('main')]
    B = list(dict.fromkeys(r['bench'] for r in R))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for j, (q, col) in enumerate(((.3, C[3]), (.5, C[0]), (.7, C[2]))):
        for i, b in enumerate(B):
            r = next((r for r in R if r['bench'] == b and abs(float(r['q']) - q) < 1e-9), None)
            if r is None: continue
            x = i + (j - 1) * .2; p = float(r['pi_hat'])
            ax.errorbar(x, p, yerr=[[p - float(r['lo'])], [float(r['hi']) - p]], fmt='o', color=col, ms=6 if q == .5 else 4,
                        capsize=3, lw=1.5, label=f'DS pi-hat, vote cut q={q}' if i == 0 else None, zorder=3)
    for i, b in enumerate(B):
        t = float(next(r['pi_true'] for r in R if r['bench'] == b))
        ax.fill_between([i - .4, i + .4], t - .1, t + .1, color=GRID, zorder=1, label='true prevalence +- .10' if i == 0 else None)
        ax.plot([i - .4, i + .4], [t, t], color=INK, lw=1.5, zorder=2)
    ax.set_xticks(range(len(B)), [NAME[b] for b in B]); ax.set_ylim(0, 1); ax.set_ylabel('share of answers with an error')
    ax.set_title('(3) Dawid-Skene prevalence from 3 base models vs true prevalence (90% bootstrap)', loc='left', color=INK)
    style(ax); ax.legend(frameon=False, fontsize=8, loc='lower left', ncol=2)
    fig.tight_layout(); fig.savefig(HERE / 'fig3_ds_pi.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    fig_auc(); fig_corr(); fig_pi()
