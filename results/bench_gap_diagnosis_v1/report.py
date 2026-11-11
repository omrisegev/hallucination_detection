"""Summary tables, attribution and figures for the benchmark-gap diagnosis (reads T1-T4 + cells/).

    python report.py   -> SUMMARY.md, T5_attribution.csv, T6_channel_ci.csv, T7_cells.csv, fig_*.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import analyze as A

HERE = Path(__file__).resolve().parent
B = A.BENCHES
NAME = dict(prmbench='PRMBench', processbench='ProcessBench', socratic='Socratic', hard2verify='Hard2Verify',
            crv_arith='CRV-Arith', crv_gsm8k='CRV-GSM8K', prm800k='PRM800K', deltabench_ext='DeltaBench')
PN = dict(qwen_prm_7b='Qwen-7B', qwen_prm_72b='Qwen-72B', pure_prm='PURE', rlhflow_prm='RLHFlow')
COL = dict(PRM='#4C6EF5', TELE16='#F08C00', C4U='#2B8A3E', position='#868E96')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})

t1 = pd.read_csv(HERE / 'T1_label_structure.csv'); t2 = pd.read_csv(HERE / 'T2_arm_quality.csv')
t3 = pd.read_csv(HERE / 'T3_channels.csv'); t4 = pd.read_csv(HERE / 'T4_lr_headroom.csv')


# ---------------------------------------------------------------- class-conditional dependence on the answer-z scale
def channel_ci():
    rows = []
    for b in B:
        for p in A.PRMS:
            d = A.load(b, p); X = d['X'].astype(float); k = d['keep']; e = d['err']; zp = X[:, 0]
            for j, ch in enumerate(A.N16):
                x = X[:, 2 + j]
                if np.std(x[k]) == 0: continue
                rows.append(dict(bench=b, prm=p, channel=ch, rz_correct=spearmanr(x[k & ~e], zp[k & ~e])[0],
                                 rz_error=spearmanr(x[k & e], zp[k & e])[0]))
    return pd.DataFrame(rows)


ci = channel_ci(); ci.to_csv(HERE / 'T6_channel_ci.csv', index=False)

# ---------------------------------------------------------------- one row per cell
q = t2.pivot_table(index=['bench', 'prm'], columns='arm', values=['pooled', 'within', 'fe_strat', 'answer'])
q.columns = [f'{a}_{m}' for m, a in q.columns]
cell = t1.set_index(['bench', 'prm']).join(q).join(t4.set_index(['bench', 'prm']))
chs = t3.groupby(['bench', 'prm']).agg(ch_fe_abs=('fe_strat', lambda x: np.mean(np.abs(x - .5))),
                                       ch_pooled_abs=('pooled', lambda x: np.mean(np.abs(x - .5))))
cis = ci.groupby(['bench', 'prm']).agg(ci_correct=('rz_correct', lambda x: np.mean(np.abs(x))),
                                       ci_error=('rz_error', lambda x: np.mean(np.abs(x))))
cell = cell.join(chs).join(cis)
cell['norm_penalty'] = cell['PRM_pooled'] - cell['PRM_answer_z_pooled']      # PRM: raw vs per-answer z
cell['lr_headroom'] = cell['LR_PRM16'] - cell['LR_PRM']
cell['tele_ceiling_gap'] = cell['LR_16'] - cell['TELE16_pooled']            # supervised telemetry vs label-free
for m in ('pooled', 'within', 'fe_strat'):
    cell[f'gapT_{m}'] = cell[f'TELE16_{m}'] - cell[f'PRM_{m}']
    cell[f'gapC_{m}'] = cell[f'C4U_{m}'] - cell[f'PRM_{m}']
cell['c4u_flipped'] = cell['C4U_pooled'] < .5
cell.to_csv(HERE / 'T7_cells.csv')

FACTORS = ['clean_share', 'between_answer_label_share', 'step_error_rate', 'steps_per_answer', 'cascade_share',
           'first_error_is_last_kept', 'position_within_auc', 'PRM_pooled', 'norm_penalty', 'LR_16', 'ch_fe_abs',
           'ci_correct', 'ci_error', 'lr_headroom', 'c4u_prev_ratio']
att = []
for g in ('gapT_pooled', 'gapC_pooled', 'gapT_fe_strat', 'gapC_fe_strat', 'gapT_within', 'gapC_within'):
    for f in FACTORS:
        r, pv = spearmanr(cell[f], cell[g], nan_policy='omit')
        att.append(dict(gap=g, factor=f, spearman=r, p=pv, n=int(cell[[f, g]].dropna().shape[0])))
att = pd.DataFrame(att); att.to_csv(HERE / 'T5_attribution.csv', index=False)

# leave-one-benchmark-out check for the strongest factors (does one benchmark drive the correlation?)
loo = []
for g in ('gapT_pooled', 'gapC_pooled'):
    for f in ('between_answer_label_share', 'clean_share', 'norm_penalty', 'PRM_pooled', 'c4u_prev_ratio'):
        rs = [spearmanr(cell.drop(index=b, level=0)[f], cell.drop(index=b, level=0)[g])[0] for b in B]
        loo.append(dict(gap=g, factor=f, min_r=min(rs), max_r=max(rs)))
loo = pd.DataFrame(loo); loo.to_csv(HERE / 'T5b_attribution_loo.csv', index=False)

bm = cell.groupby(level=0, sort=False).mean(numeric_only=True).reindex(B)


# ---------------------------------------------------------------- figures
def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / name, dpi=150); plt.close(fig)


x = np.arange(len(B)); labels = [NAME[b] for b in B]

# F1 label structure
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].bar(x - .2, bm['clean_share'], .4, label='clean answers', color='#74C0FC')
ax[0].bar(x + .2, bm['step_error_rate'], .4, label='error steps', color='#FA5252')
ax[0].set_title('Share of clean answers / error steps'); ax[0].legend(frameon=False)
ax[1].bar(x - .2, bm['cascade_share'], .4, label='all steps after 1st error labelled wrong', color='#BE4BDB')
ax[1].bar(x + .2, bm['first_error_is_last_kept'], .4, label='1st error = last labelled step', color='#495057')
ax[1].set_title('Labelling protocol (error answers)'); ax[1].legend(frameon=False, fontsize=8)
ax[2].bar(x - .2, bm['between_answer_label_share'], .4, label='label variance between answers', color='#F59F00')
ax[2].bar(x + .2, bm['position_within_auc'] - .5, .4, bottom=.5, label='position-only within-AUC', color='#868E96')
ax[2].axhline(.5, color='k', lw=.5); ax[2].set_title('Answer-level signal and position'); ax[2].legend(frameon=False, fontsize=8)
for a in ax:
    a.set_xticks(x); a.set_xticklabels(labels, rotation=35, ha='right')
save(fig, 'fig1_label_structure.png')

# F2 arm quality per target
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for i, (m, t) in enumerate((('pooled', 'Pooled step AUROC'), ('within', 'Within-answer AUC'),
                            ('fe_strat', 'First-error AUC, position-stratified'))):
    for k, a in enumerate(('PRM', 'TELE16', 'C4U', 'position')):
        ax[i].bar(x + (k - 1.5) * .2, bm[f'{a}_{m}'], .2, label=a, color=COL[a])
    ax[i].axhline(.5, color='k', lw=.6); ax[i].set_ylim(.2, 1.02); ax[i].set_title(t)
    ax[i].set_xticks(x); ax[i].set_xticklabels(labels, rotation=35, ha='right')
ax[0].legend(frameon=False, ncol=2, fontsize=8); ax[0].set_ylabel('mean over 4 PRMs')
save(fig, 'fig2_arm_quality.png')

# F3 gap vs between-answer label share and vs normalization penalty
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
mk = dict(zip(B, 'osD^v<>P'))
for b in B:
    s = cell.loc[b]
    ax[0].scatter(s['between_answer_label_share'], s['gapT_pooled'], marker=mk[b], color=COL['TELE16'], label=NAME[b])
    ax[0].scatter(s['between_answer_label_share'], s['gapC_pooled'], marker=mk[b], color=COL['C4U'])
    ax[1].scatter(s['norm_penalty'], s['gapT_pooled'], marker=mk[b], color=COL['TELE16'])
    ax[1].scatter(s['norm_penalty'], s['gapC_pooled'], marker=mk[b], color=COL['C4U'])
for a in ax:
    a.axhline(0, color='k', lw=.6); a.set_ylabel('method - PRM, pooled AUROC')
ax[0].set_xlabel('share of step-label variance between answers'); ax[1].set_xlabel('PRM loss from per-answer z-scoring (raw - answer-z AUROC)')
r1 = spearmanr(cell['between_answer_label_share'], cell['gapT_pooled'])[0]; r2 = spearmanr(cell['norm_penalty'], cell['gapT_pooled'])[0]
ax[0].set_title(f'orange TELE16, green C4U; Spearman (TELE16) {r1:.2f}'); ax[1].set_title(f'Spearman (TELE16) {r2:.2f}')
ax[0].legend(frameon=False, fontsize=7, ncol=2)
save(fig, 'fig3_gap_vs_answer_level_signal.png')

# F4 channel effects across benchmarks (first-error, position-stratified)
M = t3.groupby(['channel', 'bench']).fe_strat.mean().unstack()[B].reindex(A.N16)
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(M.values - .5, cmap='RdBu_r', vmin=-.3, vmax=.3, aspect='auto')
ax.set_xticks(range(len(B))); ax.set_xticklabels(labels, rotation=35, ha='right'); ax.set_yticks(range(16)); ax.set_yticklabels(A.N16)
for i in range(16):
    for j in range(len(B)):
        ax.text(j, i, f'{M.values[i, j]:.2f}', ha='center', va='center', fontsize=7)
fig.colorbar(im, ax=ax, label='first-error AUC - 0.5 (red = higher on the error step)')
ax.set_title('Per-channel first-error AUC (position-stratified), mean over 4 PRMs')
save(fig, 'fig4_channel_effects.png')

# F5 conditional dependence vs supervised headroom
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for b in B:
    s = cell.loc[b]
    ax[0].scatter(s['ci_correct'], s['lr_headroom'], marker=mk[b], color='#5F3DC4', label=NAME[b])
    ax[1].scatter(s['PRM_pooled'], s['lr_headroom'], marker=mk[b], color='#5F3DC4')
ax[0].set_xlabel('mean |Spearman(channel, PRM)| within correct steps'); ax[1].set_xlabel('PRM pooled AUROC')
for a in ax: a.set_ylabel('LR(PRM+16) - LR(PRM), CV AUROC'); a.axhline(0, color='k', lw=.6)
ax[0].legend(frameon=False, fontsize=7, ncol=2); ax[0].set_title('Conditional dependence vs added value')
ax[1].set_title('PRM strength vs added value')
save(fig, 'fig5_ci_and_headroom.png')

# F6 C4U prevalence vs truth
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for b in B:
    s = cell.loc[b]
    ax.scatter(s['step_error_rate'], s['c4u_prevalence'], marker=mk[b], label=NAME[b], color='#2B8A3E')
lim = [0, .5]; ax.plot(lim, lim, 'k--', lw=.7); ax.set_xlabel('true step error rate'); ax.set_ylabel('C4U estimated prevalence')
ax.legend(frameon=False, fontsize=7); ax.set_title('C4U flags a fixed ~17-40% of steps regardless of the truth')
save(fig, 'fig6_c4u_prevalence.png')


# ---------------------------------------------------------------- markdown summary
def md(df, fmt='{:.3f}'):
    cols = list(df.columns); out = ['| | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append(f'| {i} | ' + ' | '.join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + ' |')
    return '\n'.join(out)


L = ['# Benchmark-gap diagnosis tables (generated by report.py)', '']
s1 = bm[['answers', 'steps_per_answer', 'step_error_rate', 'clean_share', 'errors_per_error_answer', 'cascade_share',
         'first_error_is_last_kept', 'between_answer_label_share', 'position_within_auc', 'position_first_error_auc']].rename(index=NAME)
L += ['## T1 label structure (mean over PRMs; identical labels per PRM)', md(s1), '']
s2 = bm[[f'{a}_{m}' for m in ('pooled', 'within', 'fe_strat') for a in ('PRM', 'TELE16', 'C4U', 'position')]].rename(index=NAME)
L += ['## T2 quality per target (mean over 4 PRMs)', md(s2), '']
s3 = bm[['PRM_pooled', 'PRM_answer_z_pooled', 'norm_penalty', 'LR_PRM', 'LR_16', 'LR_PRM16', 'lr_headroom', 'TELE16_pooled',
         'tele_ceiling_gap', 'ci_correct', 'ci_error', 'c4u_prevalence', 'c4u_prev_ratio']].rename(index=NAME)
L += ['## T3 PRM normalization penalty, LR headroom, conditional dependence, C4U prevalence', md(s3), '']
L += ['## T5 attribution: Spearman across 32 cells', md(att.pivot(index='factor', columns='gap', values='spearman').reindex(FACTORS)), '']
L += ['## T5b leave-one-benchmark-out range of Spearman', md(loo.set_index(['gap', 'factor'])), '']
fl = cell[cell['c4u_flipped']][['C4U_pooled', 'C4U_within', 'C4U_fe_strat', 'K']]
L += ['## C4U cells with pooled AUROC < .5 (orientation failure)', md(fl), '']
(HERE / 'SUMMARY.md').write_text('\n'.join(L), encoding='utf8')
print('\n'.join(L))


# ---------------------------------------------------------------- additions: gap by target, error-answers-only, partial/regression
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for i, (arm, g) in enumerate((('TELE16', 'gapT'), ('C4U', 'gapC'))):
    for k, (m, lab, c) in enumerate((('pooled', 'pooled step AUROC', '#ADB5BD'), ('within', 'within-answer AUC', '#495057'),
                                     ('fe_strat', 'first-error, position-stratified', COL[arm]))):
        ax[i].bar(x + (k - 1) * .27, bm[f'{g}_{m}'], .27, label=lab, color=c)
    ax[i].axhline(0, color='k', lw=.6); ax[i].set_title(f'{arm} minus PRM, by target'); ax[i].set_xticks(x)
    ax[i].set_xticklabels(labels, rotation=35, ha='right')
ax[0].set_ylabel('AUC difference (mean over 4 PRMs)'); ax[0].legend(frameon=False, fontsize=8)
save(fig, 'fig7_gap_by_target.png')

t8 = pd.read_csv(HERE / 'T8_error_answers_only.csv').groupby('bench', sort=False).mean(numeric_only=True).reindex(B).rename(index=NAME)
from numpy.linalg import lstsq
Z = cell[['between_answer_label_share', 'PRM_pooled', 'cascade_share', 'steps_per_answer']].astype(float)
Zs = (Z - Z.mean()) / Z.std()
reg = []
for g in ('gapT_pooled', 'gapC_pooled', 'gapT_fe_strat', 'gapC_fe_strat'):
    y = cell[g].astype(float).values; D = np.column_stack([np.ones(len(y)), Zs.values])
    beta, *_ = lstsq(D, y, rcond=None); res = y - D @ beta; r2 = 1 - res.var() / y.var()
    reg.append(dict(gap=g, R2=r2, **{f'b_{c}': v for c, v in zip(Z.columns, beta[1:])}))
reg = pd.DataFrame(reg).set_index('gap'); reg.to_csv(HERE / 'T9_regression.csv')
extra = ['## T8 pooled AUROC on error answers only (clean answers removed)', md(t8), '',
         '## T9 standardized regression of the gap on 4 factors (32 cells; descriptive, factors are correlated)', md(reg), '']
with open(HERE / 'SUMMARY.md', 'a', encoding='utf8') as f:
    f.write('\n' + '\n'.join(extra))
print('\n'.join(extra))
