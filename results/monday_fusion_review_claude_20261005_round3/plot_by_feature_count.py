"""Candidate panel by bank (feature count), plotted from Codex's verified CSVs. No recomputation."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection'); P = ROOT / 'results/monday_candidate_panel_v1/run_20261004'; OUT = Path(__file__).resolve().parent
c = pd.read_csv(P / 'CANDIDATE_PANEL.csv', low_memory=False); l = pd.read_csv(P / 'LR_PANEL.csv')
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']; NF = [int(b[1:]) for b in BANKS]
ROWS = [('ALL_EQ', 'groups_answer', 'All channels, plain mean', '#9e9e9e', 'o', '-'), ('DS_FILTER_EQ', 'groups_answer', 'DS filter, plain mean', '#333333', 's', '-'),
        ('MEAN_MOM', 'groups_answer', 'MEAN+MoM, group/answer', '#7fb3d5', 'v', '--'), ('MEAN_MOM', 'channels_cell', 'MEAN+MoM, channel/cell', '#1f77b4', '^', '-'),
        ('HEM_MOM', 'groups_answer', 'HEM+MoM, group/answer', '#f5b041', 'v', '--'), ('HEM_MOM', 'channels_cell', 'HEM+MoM, channel/cell', '#d35400', '^', '-')]
def g(bench, rec, var, rule, col):
    q = c[(c.benchmark == bench) & (c.recipe == rec) & (c.variant == var) & (c.decision_rule == rule)].set_index('bank')
    return [float(q.loc[b, col]) for b in BANKS]
def glr(bench, pos, col):
    q = l[(l.benchmark == bench) & (l.recipe == 'LR_full') & (l.position_variant == pos) & (l.score_scope == 'OOF') & (l.decision_rule == 'ANSWER_Z_Q80')].set_index('bank')
    return [float(q.loc[b, col]) for b in BANKS]
PANELS = [('PRMBench', 'within_answer_auc', 'ANSWER_Z_Q80', 'PRMBench: within-answer AUC', True),
          ('ProcessBench', 'pb_first_error_accuracy', 'ANSWER_Z_Q80', 'ProcessBench: argmax hits first error', True),
          ('ProcessBench', 'pb_official_f1', 'R2_COUNT', 'ProcessBench official F1, R2 count rule', False),
          ('ProcessBench', 'pb_official_f1', 'ENTROPY_GATE_Q80', 'ProcessBench official F1, entropy gate rule', False)]
fig, ax = plt.subplots(2, 2, figsize=(15, 10)); fig.patch.set_facecolor('white'); table = []
for a, (bench, col, rule, title, with_lr) in zip(ax.ravel(), PANELS):
    for rec, var, name, colr, mk, ls in ROWS:
        y = g(bench, rec, var, rule, col); a.plot(NF, y, marker=mk, ls=ls, color=colr, label=name, lw=1.6)
        table.append({'panel': title, 'method': name, **dict(zip(BANKS, y)), 'mean_of_8': np.mean(y)})
    if with_lr:
        y = glr(bench, 'CONTENT', col); a.plot(NF, y, ls=':', color='crimson', marker='x', label='Supervised LR, out of fold (reference)')
        table.append({'panel': title, 'method': 'Supervised LR OOF, no position', **dict(zip(BANKS, y)), 'mean_of_8': np.mean(y)})
    a.set_xticks(NF); a.set_xticklabels([f'{b}\n{n} feat.' for b, n in zip(BANKS, NF)], fontsize=8)
    a.set_title(title, fontsize=11); a.grid(alpha=.3); a.set_xlabel('Feature bank (number of features before the DS filter)')
h, lb = ax[0, 0].get_legend_handles_labels(); fig.legend(h, lb, loc='lower center', ncol=4, fontsize=9, frameon=False)
fig.suptitle('Candidate panel by feature bank. Each point is one bank; ProcessBench points average 8 cells. Development data, before selection.\n'
             'Source: results/monday_candidate_panel_v1/run_20261004/CANDIDATE_PANEL.csv, LR_PANEL.csv (PRMBench 6,030 answers with errors; ProcessBench 4,442 erroneous / 6,800 total)', fontsize=10)
fig.tight_layout(rect=(0, .06, 1, .94))
for ext in ['png', 'svg']: fig.savefig(OUT / f'BY_FEATURE_COUNT.{ext}', dpi=150, facecolor='white')
pd.DataFrame(table).to_csv(OUT / 'BY_FEATURE_COUNT.csv', index=False)
pd.set_option('display.width', 250); print(pd.DataFrame(table).round(4).to_string(index=False))
