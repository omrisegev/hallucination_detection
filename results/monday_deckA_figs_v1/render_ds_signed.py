"""Deck figures with the DS-signed average (Dawid-Skene sign per channel, then plain average) replacing the
level-anchor "oriented average" control. Data: results/monday_ds_signed_v1 (+ the frozen runs it was compared with).
Light theme, names on X, scores on Y, fonts >= 15 pt. ProcessBench metric is labelled SLA (top-ranked step = first error,
erroneous answers only, 8-cell mean). New file names; the old PNGs are kept."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = Path(__file__).resolve().parent; R = OUT.parents[0]
DS = R / 'monday_ds_signed_v1'; V2 = R / 'monday_nested_banks_v2'; CB = R / 'monday_cont_baselines_v1'; LRD = R / 'monday_lr_reference_v1'; TS = R / 'monday_types_singles_v1'
M4 = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
FAM = [['N11', 'N13', 'N22', 'N40', 'N53'], ['N14', 'N16', 'N25', 'N43', 'N56']]
T = {'bg': '#fcfcfb', 'ink': '#1b2430', 'ink2': '#5b6575', 'grid': '#e4e2dc', 'ours': '#2a78d6', 'ref': '#eb6834', 'muted': '#898781', 'ds': '#1baf7a', 'neg': '#e34948', 'mid': '#f0efec'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 16})
DSL = 'DS-signed average'
NCH = {k: len(v) for k, v in json.loads((V2 / 'BANKS.json').read_text()).items()}
ABS = pd.read_csv(DS / 'ABSOLUTE.csv'); CON = pd.read_csv(DS / 'CONTRASTS.csv')
A = pd.read_csv(V2 / 'PANEL_A_ABSOLUTE.csv'); PBK = pd.read_csv(V2 / 'PER_BENCHMARK_COMPLETENESS.csv'); CBA = pd.read_csv(CB / 'ABSOLUTE.csv')
LRS = pd.read_csv(LRD / 'LR_SCORES.csv'); LRS = LRS[LRS.scorer == 'LR'].set_index('bank')


def style(ax):
    ax.set_facecolor(T['bg'])
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(T['muted'])
    ax.grid(axis='y', color=T['grid'], linewidth=.8); ax.set_axisbelow(True); ax.tick_params(colors=T['ink2'])


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=150, bbox_inches='tight', facecolor=T['bg']); plt.close(fig); print('wrote', name)


def val(scorer, bank, metric):
    """metric in {'prmb_within_auc', 'pb_sla'}"""
    if scorer == 'DS_SIGNED':
        return float(ABS[(ABS.scorer == 'DS_SIGNED') & (ABS.bank == bank)][metric].iloc[0])
    m = 'pb_argmax_hit' if metric == 'pb_sla' else metric
    if scorer in ('LSML_CONT', 'IU_PCR'):
        return float(CBA[(CBA.scorer == scorer) & (CBA.bank == bank)][m].iloc[0])
    if scorer == 'LR': return float(LRS.loc[bank, m])
    if m == 'prmb_within_auc':
        r = PBK[(PBK.scorer == scorer) & (PBK.bank == bank)]
        return float(r.prmb_within_auc.iloc[0]) if len(r) and r.prmbench_complete.iloc[0] else np.nan
    return float(A[(A.scorer == scorer) & (A.bank == bank)][m].iloc[0])


def bank_axes(ax):
    ax.axvline(5, color=T['grid'], linewidth=1.2)
    ax.set_xticks(list(range(5)) + list(range(6, 11)))
    ax.set_xticklabels([f'{b}\n({NCH[b]} ch)' for b in FAM[0] + FAM[1]], color=T['ink'])
    ax.text(2, 1.02, 'no digit channels', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=15)
    ax.text(8, 1.02, '+ 3 digit channels', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=15)


METRICS = [('prmb_within_auc', 'PRMBench: within-answer AUC', 'Within-answer AUC'), ('pb_sla', 'ProcessBench: SLA', 'SLA (8-cell mean)')]


# ---------------------------------------------------------------- gainA: gain heatmap (PRMBench AUC)
def fig_gain():
    P = pd.read_csv(V2 / 'PANEL_A_CONTRASTS.csv'); banks = FAM[0] + FAM[1]
    rows_m = ['DS_FILTER_EQ'] + M4
    a = P[(P.reference == 'ALL_EQ') & (P.metric == 'prmb_within_auc')][['bank', 'method', 'delta', 'lo95', 'hi95']]
    d = CON[(CON.method == 'ALL_EQ') & (CON.metric == 'prmb_within_auc')].copy()          # ALL_EQ minus DS -> negate
    d = d.assign(method='DS_SIGNED', delta=-d.delta, lo95=-d.hi95, hi95=-d.lo95)[['bank', 'method', 'delta', 'lo95', 'hi95']]
    left = pd.concat([d, a]); right = CON[(CON.metric == 'prmb_within_auc') & CON.method.isin(rows_m) & ~((CON.bank == 'N11') & CON.method.isin(M4))][['bank', 'method', 'delta', 'lo95', 'hi95']]  # N11 group methods n/a, as in the left panel
    cmap = LinearSegmentedColormap.from_list('div', [T['neg'], T['mid'], T['ours']]); lim = .03
    lab = {'DS_SIGNED': DSL, 'DS_FILTER_EQ': 'DS filter + average', 'HEM_MOM_cc': 'pipeline HEM_MOM_cc', 'MEAN_MOM_cc': 'pipeline MEAN_MOM_cc',
           'MEAN_MOM_ga': 'pipeline MEAN_MOM_ga', 'TCI_ONLY_ga': 'pipeline TCI_ONLY_ga'}
    fig, axes = plt.subplots(1, 2, figsize=(26, 8.4), facecolor=T['bg'], gridspec_kw={'width_ratios': [1, 1]})
    for ax, (tab, rows, title) in zip(axes, ((left, ['DS_SIGNED'] + rows_m, 'Gain over the plain average (no sign change)'),
                                             (right, rows_m, f'Gain over the {DSL}'))):
        M = tab.pivot(index='method', columns='bank', values='delta').reindex(index=rows, columns=banks)
        lo = tab.pivot(index='method', columns='bank', values='lo95').reindex(index=rows, columns=banks)
        hi = tab.pivot(index='method', columns='bank', values='hi95').reindex(index=rows, columns=banks)
        ax.imshow(np.nan_to_num(M.values), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
        for i in range(len(rows)):
            for j in range(len(banks)):
                v = M.values[i, j]
                if not np.isfinite(v): ax.text(j, i, 'n/a', ha='center', va='center', fontsize=14, color=T['muted']); continue
                sig = lo.values[i, j] > 0 or hi.values[i, j] < 0
                ax.text(j, i, f'{v:+.3f}' + ('*' if sig else ''), ha='center', va='center', fontsize=14, color=T['ink'], fontweight='bold' if sig else 'normal')
        ax.set_xticks(range(len(banks))); ax.set_xticklabels([f'{b}\n{"+digits" if b in FAM[1] else ""}' for b in banks], color=T['ink'], fontsize=15)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab[r] for r in rows], color=T['ink'], fontsize=15)
        ax.axvline(4.5, color=T['ink'], linewidth=1); ax.set_title(title, color=T['ink'], fontsize=19, loc='left')
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0); ax.set_facecolor(T['bg'])
    fig.text(.01, -.03, 'PRMBench within-answer AUC. Blue = better, red = worse; * = paired source-group 95% interval excludes 0 (descriptive). '
             'n/a = N11 group methods not reported (grouping failed in one ProcessBench cell).', color=T['ink2'], fontsize=15)
    fig.tight_layout(); save(fig, 'G1_gain_prmbench_dssigned_light')


# ---------------------------------------------------------------- baselines / lrref: per-bank dot plot
def fig_banks(name, show_lr, show_old):
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), facecolor=T['bg'])
    for ax, (metric, title, ylab) in zip(axes, METRICS):
        style(ax)
        for k, fam in enumerate(FAM):
            xs = np.arange(5) + (0 if k == 0 else 6)
            ours = np.array([[val(m, b, metric) for m in M4] for b in fam])
            ok = np.isfinite(ours).all(1)
            lo, hi, mean = np.where(ok, np.nanmin(ours, 1), np.nan), np.where(ok, np.nanmax(ours, 1), np.nan), np.where(ok, np.nanmean(ours, 1), np.nan)
            ax.vlines(xs - .24, lo, hi, color=T['ours'], alpha=.35, linewidth=9)
            ax.plot(xs - .24, mean, 'o', color=T['ours'], markersize=10, label='Our pipeline, 4 variants (dot = mean, bar = range)' if k == 0 else None)
            ax.plot(xs - .08, [val('ALL_EQ', b, metric) for b in fam], 'D', mfc='none', mec=T['muted'], mew=2, markersize=10, label='Plain average' if k == 0 else None)
            ax.plot(xs + .08, [val('DS_SIGNED', b, metric) for b in fam], 's', color=T['ds'], markersize=10, label=DSL if k == 0 else None)
            if show_old:
                ax.plot(xs + .24, [val('LSML_CONT', b, metric) for b in fam], '^', color=T['ref'], markersize=12, label='Continuous L-SML' if k == 0 else None)
                ax.plot(xs + .38, [val('IU_PCR', b, metric) for b in fam], 'v', mfc='none', mec=T['ref'], mew=2.2, markersize=12, label='IU-PCR' if k == 0 else None)
            if show_lr:
                ax.plot(xs + (.5 if show_old else .26), [val('LR', b, metric) for b in fam], '*', color='#1b2430', markersize=17, label='Supervised logistic regression (uses labels)' if k == 0 else None)
        bank_axes(ax); ax.set_title(title, color=T['ink'], loc='left', pad=30); ax.set_ylabel(ylab, color=T['ink2'])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, labelcolor=T['ink'], bbox_to_anchor=(.5, -.07))
    fig.tight_layout(rect=(0, .06, 1, 1)); save(fig, name)


# ---------------------------------------------------------------- nesting
def fig_nesting():
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), facecolor=T['bg'])
    for ax, (metric, title, ylab) in zip(axes, METRICS):
        style(ax)
        for k, fam in enumerate(FAM):
            xs = np.arange(5) + (0 if k == 0 else 6)
            ours = np.array([[val(m, b, metric) for m in M4] for b in fam])
            ok = np.isfinite(ours).all(1)
            ax.fill_between(xs, np.where(ok, np.nanmin(ours, 1), np.nan), np.where(ok, np.nanmax(ours, 1), np.nan), color=T['ours'], alpha=.18, linewidth=0)
            ax.plot(xs, np.where(ok, np.nanmean(ours, 1), np.nan), '-o', color=T['ours'], linewidth=2.5, markersize=9, label='Our pipeline, 4 variants (mean; band = range)' if k == 0 else None)
            ax.plot(xs, [val('ALL_EQ', b, metric) for b in fam], '--D', color=T['muted'], linewidth=2, markersize=9, markerfacecolor='none', label='Plain average' if k == 0 else None)
            ax.plot(xs, [val('DS_SIGNED', b, metric) for b in fam], ':s', color=T['ds'], linewidth=2.2, markersize=9, label=DSL if k == 0 else None)
        bank_axes(ax); ax.set_title(title, color=T['ink'], loc='left', pad=30); ax.set_ylabel(ylab, color=T['ink2'])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, labelcolor=T['ink'], bbox_to_anchor=(.5, -.05))
    fig.tight_layout(rect=(0, .05, 1, 1)); save(fig, 'N2_nested_banks_dssigned_light')


# ---------------------------------------------------------------- external official metric, N16
def fig_external():
    Rm = pd.read_csv(V2 / 'METRICS.csv'); Pp = json.loads((R / 'monday_external_panel_v1/PROTOCOL.json').read_text(encoding='utf8'))['comparators_published_only']
    CBX = pd.read_csv(CB / 'EXTERNAL_METRICS.csv'); EX = pd.read_csv(DS / 'EXTERNAL_DS_SIGNED.csv'); HL = 'N16'
    cells = {'hard2verify_qwen3_8b': ('Hard2Verify · Qwen3-8B\n(Balanced F1)', Pp['hard2verify_balanced_f1'], 'Qwen3-8B critic'),
             'socratic_qwen3_8b': ('Socratic · Qwen3-8B\n(PRMScore)', Pp['socratic_prmscore'], 'QwQ-32B critic'),
             'socratic_qwq32b': ('Socratic · QwQ-32B\n(PRMScore)', Pp['socratic_prmscore'], 'QwQ-32B critic')}
    names = [('Ours', 'ours'), ('Plain\navg', 'ctl'), ('DS-\nsigned\navg', 'ds'), ('IU-\nPCR', 'ctl'), ('Cont.\nL-SML', 'ctl'), ('Step\nindex', 'base'), ('PRM-7B\n(paper)', 'prm'), ('LLM\njudge\n(paper)', 'prm')]
    fig, axes = plt.subplots(1, 3, figsize=(26, 7.6), sharey=True, facecolor=T['bg'])
    for ax, (cell, (title, pub, critic)) in zip(axes, cells.items()):
        style(ax); g = Rm[Rm.cell == cell]
        o = g[(g.bank == HL) & g.scorer.isin(M4) & (g.rule == 'ANSWER_Z_Q80')].official_metric
        vals = [(o.min(), o.mean(), o.max()), g[(g.bank == HL) & (g.scorer == 'ALL_EQ') & (g.rule == 'ANSWER_Z_Q80')].official_metric.iloc[0],
                EX[(EX.cell == cell) & (EX.bank == HL)].official_metric.iloc[0],
                CBX[(CBX.cell == cell) & (CBX.bank == HL) & (CBX.scorer == 'IU_PCR')].official_metric.iloc[0],
                CBX[(CBX.cell == cell) & (CBX.bank == HL) & (CBX.scorer == 'LSML_CONT')].official_metric.iloc[0],
                g[(g.scorer == 'STEP_INDEX') & (g.rule == 'ANSWER_Z_Q80')].official_metric.iloc[0], pub['Qwen2.5-Math-PRM-7B'] / 100, pub[critic] / 100]
        for x, ((lab, kind), v) in enumerate(zip(names, vals)):
            if kind == 'ours':
                ax.vlines(x, v[0], v[2], color=T['ours'], alpha=.4, linewidth=9); y = v[1]; ax.plot(x, y, 'o', color=T['ours'], markersize=11)
            else:
                y = v
                st = {'ctl': dict(marker='o', color=T['muted']), 'ds': dict(marker='s', color=T['ds']), 'base': dict(marker='D', color=T['muted'], mfc='none', mew=2),
                      'prm': dict(marker='o', color=T['ref'], mfc='none', mew=2.4)}[kind]
                ax.plot(x, y, ls='', markersize=11, **st)
            ax.annotate(f'{y:.3f}', (x, y), xytext=(0, 14), textcoords='offset points', ha='center', fontsize=14, color=T['ink2'])
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n for n, _ in names], color=T['ink'], fontsize=14); ax.set_title(title, color=T['ink'], loc='left')
    axes[0].set_ylabel('Official metric', color=T['ink2'])
    fig.text(.01, -.04, 'Bank N16. Blue = our pipeline (dot = mean of 4 variants, bar = range). All label-free scores flag the top 20% of the cell. '
             'Hollow orange = paper-reported, not reproduced, different threshold access. Exploratory.', color=T['ink2'], fontsize=15)
    fig.tight_layout(); save(fig, 'X2_external_official_dssigned_light')


# ---------------------------------------------------------------- error types, 80 configurations
def fig_types():
    LAB = {'redundency': 'redundancy', 'circular': 'circular\nlogic', 'domain_inconsistency': 'domain\ninconsist.', 'step_contradiction': 'step\ncontradict.',
           'missing_condition': 'missing\ncondition', 'confidence': 'confidence', 'counterfactual': 'counter-\nfactual', 'deception': 'deception'}
    Aa = pd.read_csv(DS / 'TYPES_ALL_CONFIGS_DS.csv'); S = pd.read_csv(DS / 'TYPES_SUMMARY_DS.csv').sort_values('median_minus_PRM', ascending=False)
    fig, ax = plt.subplots(figsize=(24, 8.4), facecolor=T['bg']); style(ax)
    for x, r in enumerate(S.itertuples()):
        L = Aa[(Aa.type == r.type) & (Aa.family != 'reference')]
        ax.vlines(x - .15, L.auc.min(), L.auc.max(), color=T['ours'], alpha=.35, linewidth=12)
        ax.plot(x - .15, L.auc.median(), 'o', color=T['ours'], markersize=11)
        ax.errorbar(x + .12, r.PRM_auc, yerr=[[r.PRM_auc - r.PRM_lo], [r.PRM_hi - r.PRM_auc]], fmt='o', color=T['ref'], markersize=11, capsize=4)
        ax.plot(x + .34, r.step_index_auc, 'D', mfc='none', mec=T['muted'], mew=2, markersize=10)
        top = max(L.auc.max(), r.PRM_hi, r.step_index_auc); up = r.clearly_above_PRM >= r.clearly_below_PRM
        ax.annotate(f'{r.clearly_above_PRM}/{r.n_configs} above' if up else f'{r.clearly_below_PRM}/{r.n_configs} below', (x, top), xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=15, color=T['ink'] if up else T['ink2'], fontweight='bold' if up else 'normal')
    ax.set_xticks(range(len(S))); ax.set_xticklabels([LAB.get(t, t) for t in S.type], color=T['ink'])
    ax.set_ylabel('Within-answer AUC (PRMBench)', color=T['ink2'])
    hs = [plt.Line2D([], [], marker='o', ls='', color=T['ours'], markersize=10, label='80 label-free configurations (median, range)'),
          plt.Line2D([], [], marker='o', ls='', color=T['ref'], markersize=10, label='Supervised PRM-7B (95% interval)'),
          plt.Line2D([], [], marker='D', ls='', mfc='none', mec=T['muted'], markersize=9, label='Step index alone')]
    ax.legend(handles=hs, loc='lower center', bbox_to_anchor=(.5, -.28), ncol=3, frameon=False, labelcolor=T['ink'])
    fig.text(.01, -.1, f'80 = 8 label-free scorers (4 pipeline variants, plain average, {DSL}, continuous L-SML, IU-PCR) x 10 banks. '
             'Text = configurations whose paired 95% interval vs the PRM excludes 0.', color=T['ink2'], fontsize=15)
    fig.tight_layout(); save(fig, 'T2_types_all_configs_dssigned_light')


# ---------------------------------------------------------------- single channels vs fusion (N16)
def fig_singles():
    S = pd.read_csv(TS / 'SINGLE_CHANNELS.csv').set_index('channel'); best_b = S.pb_hit.idxmax()
    singles = [('realized_drv', 'Surprise\nrise'), (best_b, 'Digit\nspread'), ('ct7_ve0', 'Var-\nentropy'), ('q15_H1', 'Entropy')]
    fus = [('Plain\naverage', ['ALL_EQ']), ('DS-\nsigned\naverage', ['DS_SIGNED']), ('IU-PCR', ['IU_PCR']), ('Cont.\nL-SML', ['LSML_CONT']), ('Our\npipeline\n(4)', M4)]
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), facecolor=T['bg'])
    for ax, (col, lo, hi, metric, title, yl) in zip(axes, [('prmb_auc', 'prmb_lo', 'prmb_hi', 'prmb_within_auc', 'PRMBench: within-answer AUC', 'Within-answer AUC'),
                                                          ('pb_hit', 'pb_lo', 'pb_hi', 'pb_sla', 'ProcessBench: SLA', 'SLA (8-cell mean)')]):
        style(ax); x = 0; ticks = []
        for ch, lab in singles:
            r = S.loc[ch]; ax.errorbar(x, r[col], yerr=[[r[col] - r[lo]], [r[hi] - r[col]]], fmt='s', color=T['muted'], markersize=10, capsize=4)
            ax.annotate(f'{r[col]:.3f}', (x, r[hi]), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=14, color=T['ink2']); ticks.append(lab); x += 1
        ax.axvline(x - .5, color=T['grid'], linewidth=1.4)
        for lab, sc in fus:
            v = np.array([val(s, 'N16', metric) for s in sc])
            if len(v) > 1: ax.vlines(x, v.min(), v.max(), color=T['ours'], alpha=.35, linewidth=12)
            ax.plot(x, v.mean(), 's' if 'signed' in lab else 'o', color=T['ours'] if lab.startswith('Our') else (T['ds'] if 'signed' in lab else T['ink']), markersize=10)
            ax.annotate(f'{v.max():.3f}' if len(v) == 1 else f'{v.min():.3f}-{v.max():.3f}', (x, v.max()), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=14, color=T['ink2'])
            ticks.append(lab); x += 1
        ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, color=T['ink'], fontsize=14)
        ax.text(1.5, 1.02, 'single channels (grey)', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=15)
        ax.text(6, 1.02, 'label-free fusion, N16 bank', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=15)
        ax.set_title(title, color=T['ink'], loc='left', pad=30); ax.set_ylabel(yl, color=T['ink2'])
    fig.text(.01, -.04, 'Single channels: sign from correlation with the level mean, 95% source-group intervals; the best channel per benchmark is picked with labels. '
             'Fusion: N16, fitted label-free per cell.', color=T['ink2'], fontsize=15)
    fig.tight_layout(); save(fig, 'S2_single_vs_fusion_dssigned_light')


# ---------------------------------------------------------------- single channels with DS signs vs fusion (N16)
NICE = {'realized_drv': 'Surprise\nrise', 'digit_spread': 'Digit\nspread', 'ct7_ve0': 'Var-\nentropy', 'q15_H1': 'Entropy'}
def fig_singles_ds():
    S = pd.read_csv(DS / 'singles/SINGLE_CHANNELS_DS.csv').set_index('channel'); best_p = S.prmb_auc.idxmax(); best_b = S.pb_sla.idxmax()
    order = list(dict.fromkeys([best_p, best_b, 'ct7_ve0', 'q15_H1']))
    singles = [(c, NICE.get(c, c.replace('_', '\n'))) for c in order]
    fus = [('Plain\naverage', ['ALL_EQ']), ('DS-\nsigned\naverage', ['DS_SIGNED']), ('IU-PCR', ['IU_PCR']), ('Cont.\nL-SML', ['LSML_CONT']), ('Our\npipeline\n(4)', M4)]
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), facecolor=T['bg'])
    for ax, (col, lo, hi, metric, title, yl) in zip(axes, [('prmb_auc', 'prmb_lo', 'prmb_hi', 'prmb_within_auc', 'PRMBench: within-answer AUC', 'Within-answer AUC'),
                                                          ('pb_sla', 'pb_lo', 'pb_hi', 'pb_sla', 'ProcessBench: SLA', 'SLA (8-cell mean)')]):
        style(ax); x = 0; ticks = []
        for ch, lab in singles:
            r = S.loc[ch]; ax.errorbar(x, r[col], yerr=[[r[col] - r[lo]], [r[hi] - r[col]]], fmt='s', color=T['muted'], markersize=10, capsize=4)
            ax.annotate(f'{r[col]:.3f}', (x, r[hi]), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=14, color=T['ink2']); ticks.append(lab); x += 1
        ns = x; ax.axvline(x - .5, color=T['grid'], linewidth=1.4)
        for lab, sc in fus:
            v = np.array([val(s_, 'N16', metric) for s_ in sc])
            if len(v) > 1: ax.vlines(x, v.min(), v.max(), color=T['ours'], alpha=.35, linewidth=12)
            ax.plot(x, v.mean(), 's' if 'signed' in lab else 'o', color=T['ours'] if lab.startswith('Our') else (T['ds'] if 'signed' in lab else T['ink']), markersize=10)
            ax.annotate(f'{v.max():.3f}' if len(v) == 1 else f'{v.min():.3f}-{v.max():.3f}', (x, v.max()), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=14, color=T['ink2'])
            ticks.append(lab); x += 1
        ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, color=T['ink'], fontsize=14)
        ax.text((ns - 1) / 2, 1.02, 'single channels (grey)', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=15)
        ax.text(ns + 2, 1.02, 'label-free fusion, N16 bank', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=15)
        ax.set_title(title, color=T['ink'], loc='left', pad=30); ax.set_ylabel(yl, color=T['ink2'])
    fig.text(.01, -.04, 'Single channels: sign per cell from Dawid-Skene (label-free), 95% source-group intervals; the best channel per benchmark is picked with labels. '
             'Fusion: N16, fitted label-free per cell.', color=T['ink2'], fontsize=15)
    fig.tight_layout(); save(fig, 'S3_single_vs_fusion_dssigned_light')


if __name__ == '__main__':
    fig_gain(); fig_banks('C2_baselines_dssigned_light', show_lr=False, show_old=True); fig_banks('L4_lr_vs_labelfree_dssigned_light', show_lr=True, show_old=False)
    fig_nesting(); fig_external(); fig_types(); fig_singles()
    if (DS / 'singles/SINGLE_CHANNELS_DS.csv').exists(): fig_singles_ds()
