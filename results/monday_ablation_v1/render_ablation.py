"""Ablation figures (vertical layout: categories on X, values on Y), light + dark, PNG + SVG; from the CSVs only."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent; FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True)
M = pd.read_csv(OUT / 'METRICS.csv'); C = pd.read_csv(OUT / 'CONTRASTS.csv'); G = pd.read_csv(OUT / 'GROUPS.csv'); E = pd.read_csv(OUT / 'ESTIMATOR_VS_TRUTH.csv')
BANKS = ['N13', 'N16', 'N25', 'N43', 'N56']
TH = {'light': dict(bg='#fcfcfb', ink='#0b0b0b', ink2='#52514e', grid='#e1e0d9', axis='#c3c2b7', s=['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'], muted='#898781'),
      'dark': dict(bg='#1a1a19', ink='#ffffff', ink2='#c3c2b7', grid='#2c2c2a', axis='#383835', s=['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'], muted='#898781')}


def style(ax, t):
    ax.set_facecolor(t['bg'])
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(t['axis'])
    ax.tick_params(colors=t['ink2'], labelsize=10); ax.grid(axis='y', color=t['grid']); ax.set_axisbelow(True)


def save(fig, name, t, mode):
    fig.patch.set_facecolor(t['bg'])
    for ext in ('png', 'svg'): fig.savefig(FIG / f'{name}_{mode}.{ext}', dpi=180, facecolor=t['bg'], bbox_inches='tight')
    plt.close(fig)


def delta_panel(ax, t, variants, labels, scorer, metric, title):
    style(ax, t); ax.axhline(0, color=t['ink2'], linewidth=1.2)
    n = len(variants); w = .8 / n
    for k, (v, lab) in enumerate(zip(variants, labels)):
        c = C[(C.variant == v) & (C.scorer == scorer) & (C.metric == metric)].set_index('bank').reindex(BANKS)
        x = np.arange(len(BANKS)) - .4 + w * (k + .5)
        ax.vlines(x, c.lo95, c.hi95, color=t['s'][k], linewidth=2.5, alpha=.6)
        ax.plot(x, c.delta, 'o', color=t['s'][k], markersize=8, markeredgecolor=t['bg'], markeredgewidth=1.5, label=lab)
    ax.set_xticks(range(len(BANKS))); ax.set_xticklabels(BANKS, color=t['ink'], fontsize=11); ax.set_title(title, color=t['ink'], fontsize=11)


def fig_filter(t, mode):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    vs = ['filter_MOM', 'filter_SML', 'filter_NONE', 'filter_ORIENT']; labs = ['MoM filter', 'SML filter', 'no filter', 'keep all, flip sign (label-free)']
    delta_panel(axes[0], t, vs, labs, 'MEAN_MOM', 'prmb_auc', 'PRMBench within-answer AUC, change vs DS filter')
    delta_panel(axes[1], t, vs, labs, 'MEAN_MOM', 'pb_hit', 'ProcessBench top-step hit, change vs DS filter')
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, frameon=False, labelcolor=t['ink'], bbox_to_anchor=(.5, -.07), fontsize=10)
    fig.suptitle('Channel filter: replace DS by another rule, everything else fixed (MEAN_MOM; bar = paired 95% interval)', color=t['ink'], fontsize=12, x=.01, ha='left')
    fig.tight_layout(rect=(0, .05, 1, 1)); save(fig, 'X1_filter', t, mode)


def fig_merge(t, mode):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={'width_ratios': [1.25, 1, 1]})
    z = np.load(OUT / 'MERGE_EXAMPLE_default.npz'); R = z['R']; gb = z['gb']; g = z['groups']; names = [str(x) for x in z['names']]
    order = np.lexsort((gb, g)); Rm = R[np.ix_(order, order)]
    ax = axes[0]; im = ax.imshow(Rm, cmap='Blues' if mode == 'light' else 'magma', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks(range(len(order))); ax.set_yticklabels([names[i] for i in order], fontsize=7, color=t['ink2'])
    def boxes(lab, color, lw, ls):
        s = lab[order]; start = 0
        for j in range(1, len(s) + 1):
            if j == len(s) or s[j] != s[start]:
                ax.add_patch(plt.Rectangle((start - .5, start - .5), j - start, j - start, fill=False, edgecolor=color, linewidth=lw, linestyle=ls)); start = j
    boxes(gb, t['s'][1], 1.6, '--'); boxes(g, t['ink'], 2.4, '-')
    ax.set_title('N16 / PRMBench: channel correlation (tail votes)\ndashed = groups before merge, solid = after', color=t['ink'], fontsize=10)
    ax2 = axes[1]; style(ax2, t)
    p = G[(G.cell == 'prmbench_qwen3_8b') & (G.variant == 'default')].set_index('bank').reindex(BANKS)
    x = np.arange(len(BANKS))
    ax2.bar(x - .2, p.dep_err_before, .38, color=t['s'][1], label='before merge'); ax2.bar(x + .2, p.dep_err_after, .38, color=t['s'][0], label='after merge')
    for xi, (b, a, m) in enumerate(zip(p.dep_err_before, p.dep_err_after, p.merges)): ax2.text(xi, max(b, a) + .01, f'{m} merge' + ('s' if m != 1 else ''), ha='center', fontsize=9, color=t['ink2'])
    ax2.set_xticks(x); ax2.set_xticklabels(BANKS, color=t['ink']); ax2.set_title('Dependence between groups on error steps\n(mean |corr|; lower = closer to the model)', color=t['ink'], fontsize=10)
    ax2.legend(frameon=False, labelcolor=t['ink'], fontsize=9)
    delta_panel(axes[2], t, ['merge_OFF'], ['no merge'], 'MEAN_MOM', 'prmb_auc', 'PRMBench AUC: no merge minus merge')
    fig.tight_layout(); save(fig, 'X2_merge', t, mode)


def fig_tails(t, mode):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    q = E.groupby('quota').mean(numeric_only=True); qs = [int(v * 100) for v in q.index]
    ax = axes[0]; style(ax, t)
    ax.plot(qs, q.marked_share, '--', color=t['muted'], label='share of steps marked')
    ax.plot(qs, q.MOM_prevalence_est, '-o', color=t['s'][0], label='MoM estimate'); ax.plot(qs, q.DS_prevalence_est, '-o', color=t['s'][1], label='DS estimate')
    ax.axhline(.1396, color=t['s'][2], linewidth=2, label='true error rate 0.14')
    ax.set_xticks(qs); ax.set_xticklabels([f'{v}%' for v in qs], color=t['ink']); ax.set_title('Estimated error rate follows the quota', color=t['ink'], fontsize=11)
    ax.legend(frameon=False, labelcolor=t['ink'], fontsize=9)
    ax = axes[1]; style(ax, t)
    ax.plot(qs, q.MOM_spearman, '-o', color=t['s'][0], label='MoM / SML'); ax.plot(qs, q.DS_spearman, '-o', color=t['s'][1], label='DS')
    ax.set_xticks(qs); ax.set_xticklabels([f'{v}%' for v in qs], color=t['ink']); ax.set_title('Channel ranking vs truth (Spearman)', color=t['ink'], fontsize=11)
    ax.legend(frameon=False, labelcolor=t['ink'], fontsize=9)
    ax = axes[2]; style(ax, t)
    for k, b in enumerate(BANKS):
        vals = [M[(M.bank == b) & (M.variant == v) & (M.scorer == 'MEAN_MOM')].prmb_auc.iloc[0] for v in ('quota_10', 'default', 'quota_30', 'quota_40')]
        ax.plot(qs, vals, '-o', color=t['s'][k], label=b)
    ax.set_xticks(qs); ax.set_xticklabels([f'{v}%' for v in qs], color=t['ink']); ax.set_title('Final PRMBench AUC (MEAN_MOM) by quota', color=t['ink'], fontsize=11)
    ax.legend(frameon=False, labelcolor=t['ink'], fontsize=9, ncol=2)
    fig.suptitle('Tail quota: the share of steps that vote "error" in each answer (PRMBench, 5 nested banks)', color=t['ink'], fontsize=12, x=.01, ha='left')
    fig.tight_layout(); save(fig, 'X3_tails', t, mode)


def fig_between(t, mode):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    sc = ['FILTER_EQ', 'MEAN_EQ', 'MEAN_SML', 'MEAN_DS', 'MEAN_MOM']; lab = ['filter only\n(no groups)', 'equal\nweights', 'SML', 'DS', 'MoM']
    for ax, metric, title in zip(axes[:2], ('prmb_auc', 'pb_hit'), ('PRMBench within-answer AUC', 'ProcessBench top-step hit')):
        style(ax, t)
        for k, b in enumerate(BANKS):
            vals = [M[(M.bank == b) & (M.variant == 'default') & (M.scorer == s)][metric].iloc[0] for s in sc]
            ax.plot(range(len(sc)), vals, '-o', color=t['s'][k], label=b, linewidth=1.5)
        ax.set_xticks(range(len(sc))); ax.set_xticklabels(lab, color=t['ink'], fontsize=10); ax.set_title(title, color=t['ink'], fontsize=11)
    axes[0].legend(frameon=False, labelcolor=t['ink'], fontsize=9, ncol=2)
    ax = axes[2]; style(ax, t)
    d = G[G.variant == 'default']
    for k, b in enumerate(BANKS):
        dd = d[d.bank == b]; ax.scatter(dd.K_after + (k - 2) * .08, dd.MOM_eff, color=t['s'][k], s=40, label=b)
    ks = np.arange(3, 9); ax.plot(ks, ks, '--', color=t['muted'], label='equal weights')
    ax.set_xlabel('number of groups', color=t['ink2']); ax.set_ylabel('effective number of groups (MoM)', color=t['ink2'])
    ax.set_title('How far MoM weights are from equal\n(on the line = equal weights; 9 cells x 5 banks)', color=t['ink'], fontsize=10)
    fig.suptitle('Between-group weighting: estimator changed, everything else fixed (default filter, merge, 20% tails)', color=t['ink'], fontsize=12, x=.01, ha='left')
    fig.tight_layout(); save(fig, 'X4_between', t, mode)


for mode, t in TH.items():
    fig_filter(t, mode); fig_merge(t, mode); fig_tails(t, mode); fig_between(t, mode)
print('ok')
