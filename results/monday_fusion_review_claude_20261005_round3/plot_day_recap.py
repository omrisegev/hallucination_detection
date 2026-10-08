"""Recap figure for 2026-10-05: every attempt of the day, per feature bank, against plain mean and supervised LR.
Values are read from the saved result files (no recomputation); each row names its source. PRMBench within-answer AUC
(6,030 answers) unless stated."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(__file__).resolve().parent
D = ROOT / 'results/mom_filter_within_diagnostic_20261004'


def csv(p): return pd.read_csv(p, low_memory=False)


# ---------- top row: Monday candidates on all 8 banks, as share of the plain->LR gap ----------
bf = csv(OUT / 'BY_FEATURE_COUNT.csv'); BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']


def row(panel, method): return bf[(bf.panel == panel) & (bf.method == method)][BANKS].iloc[0].to_numpy(float)


# ---------- detail rows: B13 and B32 ----------
cs = csv(D / 'CONTRAST_GLS_RESULTS.csv'); st = csv(D / 'CONTRAST_STRUCT_RESULTS.csv'); tu = csv(D / 'TENSOR_UNMIX_V2_RESULTS.csv')
t3 = csv(D / 'TCI_STAGE3.csv').set_index('bank'); fr = csv(D / 'FRS_TABLE.csv').set_index('score'); cd = csv(D / 'CONTRAST_DIRECTION_DIAGNOSTIC.csv')
cx1 = csv(ROOT / 'results/contrast_span_oracle_20261005/run_v3/SUMMARY.csv'); cx2 = csv(ROOT / 'results/group_contrast_oracle_20261005/run_v1/SUMMARY.csv')
cx3 = csv(ROOT / 'results/orthogonal_family_fusion_20261005/run_v1/SUMMARY.csv')


def g(df, bk, col, key, val):
    q = df[(df.bank == bk) & (df[col] == key)]; return float(q[val].iloc[0]) if len(q) else np.nan


def cx(df, bk, m): return g(df[df.panel == 'PRMB'], bk, 'method', m, 'value')


ATTEMPTS = [  # (label, kind, function bank -> value)
    ('Monday pipeline: MEAN+MoM, group/answer (sealed)', 'monday', lambda b: cx(cx2, b, 'MEAN_MOM')),
    ('Monday pipeline: MEAN+MoM, channel/cell', 'monday', lambda b: float(bf[(bf.panel.str.startswith('PRMBench')) & (bf.method == 'MEAN+MoM, channel/cell')][b].iloc[0])),
    ('Monday pipeline: HEM+MoM, channel/cell', 'monday', lambda b: float(bf[(bf.panel.str.startswith('PRMBench')) & (bf.method == 'HEM+MoM, channel/cell')][b].iloc[0])),
    ('Claude: common factor PC1 only', 'claude', lambda b: g(cs, b, 'variant', 'PC1', 'PRMB_within_auc')),
    ('Claude: surprise from fusion residual (S + 0.5 one-sided)', 'claude', lambda b: float(fr.loc['A_residual__onesided__S+0.5', 'PRMB_' + b])),
    ('Claude: FUSE tail calibration, MoM between groups', 'claude', lambda b: float(t3.loc[b, 'PRMB_GRP_MOM_cal'])),
    ('Claude: whitening Sigma^-1 v (rank-one direction)', 'claude', lambda b: g(cs, b, 'variant', 'GLS_R1_a0.05', 'PRMB_within_auc')),
    ('Claude: whitening with MoM direction', 'claude', lambda b: g(cs, b, 'variant', 'MOM_GLS_a0.05', 'PRMB_within_auc')),
    ('Claude: contrasts by structure, skew sign, MoM filter', 'claude', lambda b: g(st, b, 'variant', 'A_EQ', 'PRMB_within_auc')),
    ('Claude: second factor by skew pursuit', 'claude', lambda b: g(st, b, 'variant', 'B_1', 'PRMB_within_auc')),
    ('Claude: tensor rank 2, largest component (GLS readout)', 'claude', lambda b: g(tu, b, 'score', 'joint rank2 GLS comp1 (largest sigma)', 'PRMB_within_auc')),
    ('Codex: MoM within + MoM between (raw features)', 'codex', lambda b: cx(cx3, b, 'RAW_MOM_MOM')),
    ('Codex: orthogonal common+contrast basis, HEM + MoM', 'codex', lambda b: cx(cx3, b, 'ORTH_HEM_MOM')),
    ('Codex: orthogonal basis, plain mean', 'codex', lambda b: cx(cx3, b, 'ORTH_EQ_EQ')),
    ('LABELS: LR on the group scores only (3-8 dims)', 'labels', lambda b: cx(cx2, b, 'LR_GROUPS')),
    ('LABELS: LR on PC1 + 3 tensor components (4 dims)', 'labels', lambda b: g(tu, b, 'score', 'joint rank3 SPAN: LR_OOF on PC1 + 3 GLS comps', 'PRMB_within_auc')),
    ('LABELS: LR on PC1 + single contrast drv - surprisal (2 dims)', 'labels', lambda b: cx(cx1, b, 'LR_PC1_single_linear')),
    ('LABELS: LR on PC1 + all within-group contrasts', 'labels', lambda b: cx(cx1, b, 'LR_PC1_within_group_linear')),
    ('LABELS: LR on group scores + within-group contrasts', 'labels', lambda b: cx(cx2, b, 'LR_GROUPS_PLUS_WITHIN')),
]
COL = {'monday': '#555555', 'claude': '#d35400', 'codex': '#1f77b4', 'labels': '#2e8b57'}
NAME = {'monday': 'Monday pipeline (label-free)', 'claude': 'Claude, today (label-free)', 'codex': 'Codex, today (label-free)', 'labels': 'Diagnostic WITH labels (not a method)'}

fig = plt.figure(figsize=(17, 21)); fig.patch.set_facecolor('white')
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.6], hspace=.28, wspace=.75)
for k, (panel, title) in enumerate([('PRMBench: within-answer AUC', 'PRMBench'), ('ProcessBench: argmax hits first error', 'ProcessBench (first-error localization)')]):
    a = fig.add_subplot(gs[0, k]); plain = row(panel, 'DS filter, plain mean'); lr = row(panel, 'Supervised LR OOF, no position'); x = np.arange(8); w = .38
    for j, (m, c) in enumerate([('MEAN+MoM, channel/cell', '#1f77b4'), ('HEM+MoM, channel/cell', '#d35400')]):
        share = 100 * (row(panel, m) - plain) / (lr - plain); bars = a.bar(x + (j - .5) * w, share, w, color=c, label=m)
        for b_, s in zip(bars, share): a.text(b_.get_x() + w / 2, s + (1 if s >= 0 else -1), f'{s:.0f}', ha='center', va='bottom' if s >= 0 else 'top', fontsize=7.5)
    a.axhline(0, color='k', lw=.8); a.axhline(100, color='#2e8b57', ls='--', lw=1.2, label='100% = supervised LR (out of fold)')
    a.set_xticks(x); a.set_xticklabels([f'{b}\nplain {p:.3f}\nLR {l:.3f}' for b, p, l in zip(BANKS, plain, lr)], fontsize=7.5)
    a.set_ylabel('% of the gap plain mean -> LR closed'); a.set_title(f'{title}: Monday candidates on all 8 banks', fontsize=11); a.grid(axis='y', alpha=.3)
    a.set_ylim(-60, 110); a.legend(fontsize=8, loc='upper right')
for k, bk in enumerate(['B13', 'B32']):
    a = fig.add_subplot(gs[1, k]); plain = cx(cx2, bk, 'plain'); lr = cx(cx1, bk, 'LR_all_X')
    labs, vals, cols = [], [], []
    for lab, kind, f in ATTEMPTS:
        v = f(bk)
        if np.isfinite(v): labs.append(lab); vals.append(v - plain); cols.append(COL[kind])
    y = np.arange(len(vals))[::-1]
    a.barh(y, vals, color=cols, alpha=.9)
    for yy, v in zip(y, vals):
        if v >= -.1: a.text(v + (.0015 if v >= 0 else -.0015), yy, f'{v + plain:.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
    a.axvline(0, color='k', lw=1); a.axvline(lr - plain, color='#2e8b57', ls='--', lw=1.5)
    a.text(lr - plain, len(vals) - .3, f' LR {lr:.3f}', color='#2e8b57', fontsize=9, va='bottom')
    a.set_yticks(y); a.set_yticklabels(labs, fontsize=8.5); a.set_xlim(-.1, lr - plain + .014)
    a.set_xlabel(f'AUC minus plain mean of the same bank ({plain:.3f}); clipped at -0.10'); a.set_title(f'{bk}: every attempt today, PRMBench (6,030 answers)', fontsize=11)
    a.grid(axis='x', alpha=.3)
    for yy, v in zip(y, vals):
        if v < -.1: a.text(-.099, yy, f'{v + plain:.3f} (off scale)', va='center', ha='left', fontsize=7.5, color='white')
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()] + [plt.Line2D([0], [0], color='#2e8b57', ls='--')]
fig.legend(h, list(NAME.values()) + ['Supervised LR on the same features, out of fold'], loc='lower center', ncol=5, fontsize=9.5, frameon=False)
fig.suptitle('2026-10-05 recap: where each attempt stands between the plain mean (0) and supervised LR (dashed). Development data, before selection.\n'
             'Top: all 8 banks, Monday candidates (results/monday_candidate_panel_v1). Bottom: B13 and B32, all experiments run today (sources in plot_day_recap.py).',
             fontsize=11)
for ext in ['png', 'svg']: fig.savefig(OUT / f'DAY_RECAP.{ext}', dpi=130, facecolor='white', bbox_inches='tight')
print('saved')
