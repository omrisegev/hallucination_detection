"""Raw (absolute) results of the candidate panel, plotted from Codex's verified CSVs. No recomputation."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection'); P = ROOT / 'results/monday_candidate_panel_v1/run_20261004'; OUT = Path(__file__).resolve().parent
c = pd.read_csv(P / 'CANDIDATE_PANEL.csv', low_memory=False); l = pd.read_csv(P / 'LR_PANEL.csv')
ROWS = [('ALL_EQ', 'groups_answer', 'All channels, plain mean'), ('DS_FILTER_EQ', 'groups_answer', 'DS filter, plain mean'),
        ('MEAN_MOM', 'groups_answer', 'MEAN+MoM, group/answer'), ('MEAN_MOM', 'channels_cell', 'MEAN+MoM, channel/cell'),
        ('HEM_MOM', 'groups_answer', 'HEM+MoM, group/answer'), ('HEM_MOM', 'channels_cell', 'HEM+MoM, channel/cell')]
COL = ['#9e9e9e', '#555555', '#7fb3d5', '#1f77b4', '#f5b041', '#d35400']
RULES = [('ANSWER_Z_Q80', 'q80'), ('R2_COUNT', 'R2 count'), ('ENTROPY_GATE_Q80', 'Entropy gate')]
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']
def get(bench, rec, var, rule, col, bank):
    q = c[(c.benchmark == bench) & (c.recipe == rec) & (c.variant == var) & (c.decision_rule == rule) & (c.bank == bank)]
    assert len(q) == 1, (bench, rec, var, rule, bank); return float(q[col].iloc[0])
def lr(bench, rec, pos, col, bank='EQUAL_8_BANK_MEAN'):
    q = l[(l.benchmark == bench) & (l.recipe == rec) & (l.position_variant == pos) & (l.score_scope == 'OOF') & (l.decision_rule == 'ANSWER_Z_Q80') & (l.bank == bank)]
    assert len(q) == 1; return float(q[col].iloc[0])
fig, ax = plt.subplots(2, 2, figsize=(16, 11)); fig.patch.set_facecolor('white')
# (a),(b): localization, independent of the decision rule; dots = 8 banks, bar = equal-bank mean
for a, bench, col, title, lab in [(ax[0, 0], 'PRMBench', 'within_answer_auc', 'PRMBench: within-answer AUC (6,030 answers with errors)', 'Within-answer AUC'),
                                  (ax[0, 1], 'ProcessBench', 'pb_first_error_accuracy', 'ProcessBench: first-error localization by argmax (4,442 erroneous answers)', 'Argmax hits first error')]:
    for k, (rec, var, name) in enumerate(ROWS):
        vals = [get(bench, rec, var, 'ANSWER_Z_Q80', col, b) for b in BANKS]; m = get(bench, rec, var, 'ANSWER_Z_Q80', col, 'EQUAL_8_BANK_MEAN')
        a.barh(k, m, color=COL[k], alpha=.85); a.scatter(vals, np.full(8, k) + np.linspace(-.25, .25, 8), s=12, color='k', zorder=3)
        a.text(1.01, k, f'{m:.4f}', va='center', ha='left', fontsize=9.5, fontweight='bold', transform=a.get_yaxis_transform())
    for rec, pos, ls in [('LR_full', 'CONTENT', '--'), ('LR_full', 'POS', ':')]:
        v = lr(bench, rec, pos, col); a.axvline(v, color='crimson', ls=ls, lw=1.5, label=f'Supervised LR, out of fold, {"with" if pos == "POS" else "no"} position: {v:.3f}')
    a.set_yticks(range(len(ROWS))); a.set_yticklabels([r[2] for r in ROWS]); a.set_xlabel(lab); a.set_title(title, fontsize=10.5)
    lo = min(get(bench, r, v, 'ANSWER_Z_Q80', col, b) for r, v, _ in ROWS for b in BANKS); a.set_xlim(lo - .01, max(a.get_xlim()[1], lr(bench, 'LR_full', 'POS', col) + .01))
    a.legend(loc='upper center', bbox_to_anchor=(.5, -.13), fontsize=8, ncol=1, frameon=False); a.grid(axis='x', alpha=.3)
# (c),(d): ProcessBench decisions by rule
w = .13
for a, col, title, lab in [(ax[1, 0], 'pb_official_f1', 'ProcessBench official F1 by decision rule (6,800 answers)', 'Official F1 (higher is better)'),
                           (ax[1, 1], 'clean_flag_rate', 'ProcessBench: share of CLEAN answers flagged (2,358 clean answers)', 'Clean answers flagged (lower is better)')]:
    for k, (rec, var, name) in enumerate(ROWS):
        ys = [get('ProcessBench', rec, var, r, col, 'EQUAL_8_BANK_MEAN') for r, _ in RULES]
        bars = a.bar(np.arange(3) + (k - 2.5) * w, ys, w, color=COL[k], label=name)
        for b_, y in zip(bars, ys): a.text(b_.get_x() + w / 2, y + .004, f'{y:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    a.set_xticks(range(3)); a.set_xticklabels([n for _, n in RULES]); a.set_ylabel(lab); a.set_title(title, fontsize=10.5); a.grid(axis='y', alpha=.3)
    a.set_ylim(0, 1.08 if col == 'clean_flag_rate' else .40)
h, lb = ax[1, 0].get_legend_handles_labels(); fig.legend(h, lb, loc='lower center', ncol=6, fontsize=9, frameon=False)
fig.suptitle('Monday candidate panel, raw values (equal mean of 8 banks; ProcessBench also of 8 cells). Pre-selection development data, not untouched confirmation.\n'
             'Sources: results/monday_candidate_panel_v1/run_20261004/CANDIDATE_PANEL.csv, LR_PANEL.csv', fontsize=10)
fig.tight_layout(rect=(0, .04, 1, .95), h_pad=3)
for ext in ['png', 'svg']: fig.savefig(OUT / f'RAW_RESULTS.{ext}', dpi=150, facecolor='white')
print('saved')
