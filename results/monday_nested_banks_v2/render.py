"""Figures for the nested-bank run (vertical layout: names on X, scores on Y), light + dark, PNG + SVG.
Palette: dataviz reference instance; 3 categorical slots validated all-pairs both modes; baselines muted ink."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = Path(__file__).resolve().parent; FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True)
HL = json.loads((OUT / 'HIGHLIGHT.json').read_text(encoding='utf8'))['highlight']
M4 = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
FAM = (['N11', 'N13', 'N22', 'N40', 'N53'], ['N14', 'N16', 'N25', 'N43', 'N56'])
PRM_L = 'Qwen2.5-Math-PRM-7B (supervised)'; MTG_L = 'Mind the Gap (local repro.)'
THEME = {'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7', ours='#2a78d6', prm='#eb6834', mtg='#1baf7a', mid='#f0efec', neg='#e34948', pos='#2a78d6'),
         'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835', ours='#3987e5', prm='#d95926', mtg='#199e70', mid='#383835', neg='#e66767', pos='#3987e5')}


def style(ax, t, grid='y'):
    ax.set_facecolor(t['surface'])
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(t['axis'])
    ax.tick_params(colors=t['ink2'], labelsize=9)
    if grid: ax.grid(axis=grid, color=t['grid'], linewidth=.8)
    ax.set_axisbelow(True)


def save(fig, name, t, mode):
    fig.patch.set_facecolor(t['surface'])
    for ext in ('png', 'svg'): fig.savefig(FIG / f'{name}_{mode}.{ext}', dpi=180, facecolor=t['surface'], bbox_inches='tight')
    plt.close(fig)


def vdot(ax, x, v, t, kind, label=True):
    """kind: ours (tuple lo,mean,hi), prm, mtg, base, ctl"""
    if kind == 'ours':
        lo, mean, hi = v
        ax.plot([x, x], [lo, hi], color=t['ours'], linewidth=7, alpha=.4, solid_capstyle='round')
        ax.plot(x, mean, 'o', color=t['ours'], markersize=9, markeredgecolor=t['surface'], markeredgewidth=2, zorder=3); v = mean
    elif kind == 'prm': ax.plot(x, v, 'o', color=t['prm'], markersize=9, markeredgecolor=t['surface'], markeredgewidth=2, zorder=3)
    elif kind == 'mtg': ax.plot(x, v, 's', color=t['mtg'], markersize=8, markeredgecolor=t['surface'], markeredgewidth=1.5, zorder=3)
    elif kind == 'ctl': ax.plot(x, v, 'o', color=t['muted'], markersize=8, zorder=3)
    else: ax.plot(x, v, 'D', color=t['muted'], markersize=7, markerfacecolor='none', markeredgewidth=1.5)
    if label and np.isfinite(v): ax.annotate(f'{v:.3f}', (x, v), xytext=(13, 0), textcoords='offset points', va='center', fontsize=8, color=t['ink2'])


def fig_A(t, mode):
    """Panel A (Amendment A2): gain over ALL_EQ (Omri's reference) and over ALL_EQ_ORIENTED, PRMBench AUC and ProcessBench hit."""
    C = pd.read_csv(OUT / 'PANEL_A_CONTRASTS.csv'); O = pd.read_csv(OUT / 'ORIENTED_CONTRASTS.csv')
    rows_m = ['DS_FILTER_EQ', 'HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']; banks = list(FAM[0]) + list(FAM[1])
    cmap = LinearSegmentedColormap.from_list('div', [t['neg'], t['mid'], t['pos']])
    for metric, mtitle, lim, stem in (('prmb_within_auc', 'PRMBench within-answer AUC', .03, 'A1_gain_prmbench'), ('pb_argmax_hit', 'ProcessBench top-step first-error hit', .05, 'A1_gain_processbench')):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        a = C[(C.reference == 'ALL_EQ') & (C.metric == metric)][['bank', 'method', 'delta', 'lo95', 'hi95']]
        oa = O[(O.method == 'ALL_EQ') & (O.metric == metric)].copy()          # ALL_EQ minus ORIENTED -> negate to ORIENTED minus ALL_EQ
        oa = oa.assign(method='ALL_EQ_ORIENTED', delta=-oa.delta, lo95=-oa.hi95, hi95=-oa.lo95)[['bank', 'method', 'delta', 'lo95', 'hi95']]
        left = pd.concat([oa, a]); right = O[(O.metric == metric) & O.method.isin(rows_m)][['bank', 'method', 'delta', 'lo95', 'hi95']]
        for ax, (tab, rows, title) in zip(axes, ((left, ['ALL_EQ_ORIENTED'] + rows_m, 'minus plain average (channels as they are)'),
                                                 (right, rows_m, 'minus oriented plain average (label-free sign per channel)'))):
            M = tab.pivot(index='method', columns='bank', values='delta').reindex(index=rows, columns=banks)
            lo = tab.pivot(index='method', columns='bank', values='lo95').reindex(index=rows, columns=banks); hi = tab.pivot(index='method', columns='bank', values='hi95').reindex(index=rows, columns=banks)
            ax.imshow(np.nan_to_num(M.values), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
            for i in range(len(rows)):
                for j in range(len(banks)):
                    v = M.values[i, j]
                    if not np.isfinite(v): ax.text(j, i, 'n/a', ha='center', va='center', fontsize=7.5, color=t['muted']); continue
                    sig = lo.values[i, j] > 0 or hi.values[i, j] < 0
                    ax.text(j, i, f'{v:+.3f}' + ('*' if sig else ''), ha='center', va='center', fontsize=7.8, color=t['ink'], fontweight='bold' if sig else 'normal')
            ax.set_xticks(range(len(banks))); ax.set_xticklabels([f'{b}\n{"+digits" if b in FAM[1] else ""}' for b in banks], color=t['ink'], fontsize=8.5)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(['oriented plain avg' if r == 'ALL_EQ_ORIENTED' else 'DS filter + avg' if r == 'DS_FILTER_EQ' else r for r in rows], color=t['ink'], fontsize=8.5)
            ax.axvline(4.5, color=t['ink'], linewidth=1); ax.set_title(title, color=t['ink'], fontsize=10)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(length=0); ax.set_facecolor(t['surface'])
            ax.add_patch(plt.Rectangle((banks.index(HL) - .5, -.5), 1, len(rows), fill=False, edgecolor=t['ink'], linewidth=1.5))
        fig.suptitle(f'{mtitle}: gain of each scorer over two plain averages of the same nested bank', color=t['ink'], fontsize=12, x=.01, ha='left')
        fig.text(.01, -.04, 'Blue = better, red = worse; * = paired source-group 95% interval excludes 0 (descriptive). Nested: N11 < N13 < N22 < N40 < N53 (left block), '
                 'the same + 3 digit channels (right). n/a = N11 group methods not reported (grouping failed in one ProcessBench cell). Outlined = highlight.', color=t['ink2'], fontsize=8)
        fig.tight_layout(); save(fig, stem, t, mode)


def fig_nesting(t, mode):
    A = pd.read_csv(OUT / 'PANEL_A_ABSOLUTE.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (metric, title) in zip(axes, [('prmb_within_auc', 'PRMBench within-answer AUC'), ('pb_argmax_hit', 'ProcessBench top-step first-error hit')]):
        style(ax, t)
        for k, fam in enumerate(FAM):
            xs = np.arange(5) + (0 if k == 0 else 6)
            src = A.copy()
            if metric == 'prmb_within_auc':
                pbk = pd.read_csv(OUT / 'PER_BENCHMARK_COMPLETENESS.csv'); fill_ = pbk[pbk.prmbench_complete][['bank', 'scorer', 'prmb_within_auc']]
                src = src.drop(columns='prmb_within_auc').merge(fill_, on=['bank', 'scorer'], how='left')
            m = src[src.scorer.isin(M4) & src.bank.isin(fam)].groupby('bank')[metric].agg(['min', 'mean', 'max']).reindex(fam)
            ax.fill_between(xs, m['min'], m['max'], color=t['ours'], alpha=.18, linewidth=0)
            ax.plot(xs, m['mean'], '-o', color=t['ours'], linewidth=2, markersize=6, label='4 methods (mean; band = range)' if k == 0 else None)
            a = A[(A.scorer == 'ALL_EQ') & A.bank.isin(fam)].set_index('bank').loc[fam, metric]
            ax.plot(xs, a, '--D', color=t['muted'], linewidth=1.6, markersize=6, markerfacecolor='none', label='Plain average' if k == 0 else None)
            oc = pd.read_csv(OUT / 'ORIENTED_CONTROL.csv').set_index('bank').loc[fam, metric]
            ax.plot(xs, oc, ':s', color=t['ink2'], linewidth=1.6, markersize=6, label='oriented plain average' if k == 0 else None)
        ax.set_xticks(list(range(5)) + list(range(6, 11)))
        ax.set_xticklabels([f'{b}\n({len(json.loads((OUT / "BANKS.json").read_text())[b])} ch)' for b in FAM[0] + FAM[1]], color=t['ink'], fontsize=8.5)
        ax.text(2, 1.02, 'no digits', transform=ax.get_xaxis_transform(), ha='center', color=t['ink2'], fontsize=9)
        ax.text(8, 1.02, '+ 3 digit channels', transform=ax.get_xaxis_transform(), ha='center', color=t['ink2'], fontsize=9)
        ax.set_title(title, color=t['ink'], fontsize=10.5, pad=18); ax.set_ylabel('Within-answer AUC' if metric == 'prmb_within_auc' else 'Top-step hit (8-cell mean)', color=t['ink2'])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, labelcolor=t['ink'], bbox_to_anchor=(.5, -.06), fontsize=9)
    fig.suptitle('Nested banks: what each added channel family does', color=t['ink'], fontsize=12, x=.01, ha='left')
    fig.tight_layout(rect=(0, .05, 1, 1)); save(fig, 'A2_nested_banks', t, mode)


def fig_B1(t, mode):
    B = pd.read_csv(OUT / 'PANEL_B_PRMBENCH.csv')
    cols = [('within_auc', 'Within-answer AUC'), ('top1_on_error', 'Top-ranked step is an error'), ('prmscore_top1', 'PRMScore, one flag per answer'), ('prmscore_own', "PRMScore, each method's own rule")]
    names = [(f'Ours {HL}\n(4 methods)', 'ours'), (f'Plain avg {HL}', 'ctl'), ('Supervised\nPRM-7B', 'prm'), ('Mind the\nGap', 'mtg'), ('Step\nindex', 'base'), ('Step\nlength', 'base')]
    names = [(f'Ours\n{HL}', 'ours'), (f'Plain avg\n{HL}', 'ctl'), ('IU-PCR', 'ctl'), ('Contin.\nL-SML', 'ctl'), ('PRM-7B\n(sup.)', 'prm'), ('Mind the\nGap', 'mtg'), ('Step\nindex', 'base'), ('Step\nlength', 'base')]
    CBA = pd.read_csv(OUT.parent / 'monday_cont_baselines_v1/ABSOLUTE.csv'); cmap = {'within_auc': 'prmb_within_auc', 'top1_on_error': 'prmb_top1_on_error', 'prmscore_top1': 'prmb_prmscore_top1'}
    cb = lambda sc, col: float(CBA[(CBA.bank == HL) & (CBA.scorer == sc)][cmap[col]].iloc[0]) if col in cmap else np.nan
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    for ax, (col, title) in zip(axes, cols):
        style(ax, t)
        o = B[(B.bank == HL) & B.scorer.isin(M4)][col]
        vals = [(o.min(), o.mean(), o.max()), B[(B.bank == HL) & (B.scorer == 'ALL_EQ')][col].iloc[0], cb('IU_PCR', col), cb('LSML_CONT', col), B[B.scorer == PRM_L][col].iloc[0], B[B.scorer == MTG_L][col].iloc[0],
                B[B.scorer == 'Step index (later = riskier)'][col].iloc[0], B[B.scorer == 'Step token length'][col].iloc[0]]
        for x, ((lab, kind), v) in enumerate(zip(names, vals)):
            if kind != 'ours' and not np.isfinite(v): ax.text(x, .5, 'n/a', ha='center', color=t['muted'], fontsize=8, transform=ax.get_xaxis_transform()); continue
            vdot(ax, x, v, t, kind)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n for n, _ in names], color=t['ink'], fontsize=8.5)
        ax.set_xlim(-.6, len(names) - .2); ax.set_title(title, color=t['ink'], fontsize=10)
        if col.startswith('prmscore'): ax.set_ylim(.48, .70)
    fig.suptitle(f'PRMBench (6,969 answers): bank {HL} (post-hoc highlight) vs competitors', color=t['ink'], fontsize=12, x=.01, ha='left')
    fig.text(.01, -.05, 'Blue = mean of the 4 methods, bar = their range. Plain average, IU-PCR and continuous L-SML on the same bank (grey). Own-rule column (n/a for rules not run): ours ANSWER_Z_Q80 (20% of steps) vs PRM native 0.5 (8.7%) - different calibration access. Development data.', color=t['ink2'], fontsize=8.5)
    fig.tight_layout(); save(fig, 'B1_prmbench_headline', t, mode)


def fig_B2(t, mode):
    T = pd.read_csv(OUT / 'PANEL_B_PRMBENCH_TYPES.csv')
    types = sorted(set(T.group))
    rows = []
    for g in types:
        o = T[(T.group == g) & (T.bank == HL) & T.scorer.isin(M4)].within_auc
        rows.append((g, (o.min(), o.mean(), o.max()), T[(T.group == g) & (T.scorer == PRM_L)].within_auc.iloc[0], T[(T.group == g) & (T.scorer == MTG_L)].within_auc.iloc[0], T[(T.group == g) & (T.scorer == 'Step index (later = riskier)')].within_auc.iloc[0]))
    rows.sort(key=lambda r: r[1][1] - r[2], reverse=True)
    fig, ax = plt.subplots(figsize=(12, 4.8)); style(ax, t)
    for x, (g, o, p, m, si) in enumerate(rows):
        ax.plot([x, x], [min(o[1], p), max(o[1], p)], color=t['axis'], linewidth=2, zorder=1)
        vdot(ax, x - .12, o, t, 'ours', label=False); vdot(ax, x + .12, p, t, 'prm', label=False); vdot(ax, x, m, t, 'mtg', label=False)
        ax.scatter([x + .3], [si], marker='D', s=40, facecolors='none', edgecolors=t['muted'], linewidths=1.6, zorder=3)
        ax.annotate(f'{o[1] - p:+.3f}', (x, max(o[2], p, m, si)), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8.5, color=t['ink'] if o[1] > p else t['ink2'])
    lab = {'redundency': 'redundancy', 'circular': 'circular\nlogic', 'domain_inconsistency': 'domain\ninconsistency', 'step_contradiction': 'step\ncontradiction',
           'missing_condition': 'missing\ncondition', 'confidence': 'confidence\n(stated firmly)', 'counterfactual': 'counter-\nfactual', 'deception': 'deception'}
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([lab.get(r[0], r[0]) for r in rows], color=t['ink'], fontsize=9)
    ax.set_ylabel('Within-answer AUC', color=t['ink2'])
    hs = [plt.Line2D([], [], marker='o', ls='', color=t['ours'], markersize=8, label=f'Ours, {HL} (bar = 4-method range)'), plt.Line2D([], [], marker='o', ls='', color=t['prm'], markersize=8, label='Supervised PRM-7B'),
          plt.Line2D([], [], marker='s', ls='', color=t['mtg'], markersize=7, label='Mind the Gap (our reproduction)'), plt.Line2D([], [], marker='D', ls='', color=t['muted'], markerfacecolor='none', markersize=7, label='Step index alone')]
    ax.legend(handles=hs, loc='lower center', bbox_to_anchor=(.5, -.32), ncol=4, frameon=False, labelcolor=t['ink'], fontsize=9)
    ax.set_title('PRMBench by error type (numbers = ours minus supervised PRM)', color=t['ink'], fontsize=11, loc='left')
    fig.tight_layout(); save(fig, 'B2_error_types', t, mode)


def fig_B3(t, mode):
    P = pd.read_csv(OUT / 'PANEL_B_PROCESSBENCH.csv')
    strata = [('ALL', 'all erroneous'), ('content-only (not step 0, not longest)', 'not step 0,\nnot longest'), ('error at step 0', 'error at\nstep 0'),
              ('error in longest step', 'error in\nlongest step'), ('subset:gsm8k', 'GSM8K'), ('subset:math', 'MATH'), ('subset:olympiadbench', 'Olympiad-\nBench'), ('subset:omnimath', 'Omni-\nMATH')]
    PS = pd.read_csv(OUT.parent / 'monday_cont_baselines_v1/PB_STRATA.csv')
    fig, ax = plt.subplots(figsize=(15, 5)); style(ax, t)
    for x, (s, _) in enumerate(strata):
        g = P[P.stratum == s]; o = g[(g.bank == HL) & g.scorer.isin(M4)].argmax_hit
        vdot(ax, x - .15, (o.min(), o.mean(), o.max()), t, 'ours', label=False)
        vdot(ax, x + .15, g[g.scorer == 'Qwen2.5-Math-PRM-7B argmax'].argmax_hit.iloc[0], t, 'prm', label=False)
        vdot(ax, x, g[g.scorer == MTG_L].argmax_hit.iloc[0], t, 'mtg', label=False)
        q = PS[(PS.stratum == s) & (PS.bank == HL)].set_index('scorer').argmax_hit
        ax.plot(x - .42, q['IU_PCR'], 'v', color=t['ink2'], markersize=7); ax.plot(x - .3, q['LSML_CONT'], '^', color=t['ink2'], markersize=7)
        if s.startswith('subset') or s == 'ALL':
            ax.plot(x + .3, g[g.scorer == 'Longest step'].argmax_hit.iloc[0], '^', color=t['muted'], markerfacecolor='none', markersize=7, markeredgewidth=1.5)
        ax.text(x, -.07, f'n={int(g.n_error_rows.iloc[0])}', transform=ax.get_xaxis_transform(), ha='center', color=t['muted'], fontsize=8)
    ax.set_xticks(range(len(strata))); ax.set_xticklabels([l for _, l in strata], color=t['ink'], fontsize=9); ax.tick_params(axis='x', pad=16)
    ax.set_ylim(0, .85); ax.set_ylabel('Top-ranked step = first error', color=t['ink2'])
    hs = [plt.Line2D([], [], marker='o', ls='', color=t['ours'], markersize=8, label=f'Ours, {HL}'), plt.Line2D([], [], marker='o', ls='', color=t['prm'], markersize=8, label='Supervised PRM-7B, argmax'),
          plt.Line2D([], [], marker='s', ls='', color=t['mtg'], markersize=7, label='Mind the Gap (our reproduction)'), plt.Line2D([], [], marker='^', ls='', color=t['muted'], markerfacecolor='none', markersize=7, label='Longest step (baseline)'),
          plt.Line2D([], [], marker='v', ls='', color=t['ink2'], markersize=7, label=f'IU-PCR, {HL}'), plt.Line2D([], [], marker='^', ls='', color=t['ink2'], markersize=7, label=f'Continuous L-SML, {HL}')]
    ax.legend(handles=hs, loc='lower center', bbox_to_anchor=(.5, -.4), ncol=3, frameon=False, labelcolor=t['ink'], fontsize=9)
    ax.set_title('ProcessBench, gate-free localization (cell-equal mean over 8 cells)', color=t['ink'], fontsize=11, loc='left')
    fig.tight_layout(); save(fig, 'B3_processbench', t, mode)


def fig_E2(t, mode):
    if not (OUT / 'METRICS.csv').exists(): return
    R = pd.read_csv(OUT / 'METRICS.csv'); Pp = json.loads((OUT.parent / 'monday_external_panel_v1/PROTOCOL.json').read_text(encoding='utf8'))['comparators_published_only']
    cells = {'hard2verify_qwen3_8b': ('Hard2Verify · Qwen3-8B (Balanced F1)', Pp['hard2verify_balanced_f1'], 'Qwen3-8B critic'),
             'socratic_qwen3_8b': ('Socratic · Qwen3-8B (PRMScore)', Pp['socratic_prmscore'], 'QwQ-32B critic'), 'socratic_qwq32b': ('Socratic · QwQ-32B (PRMScore)', Pp['socratic_prmscore'], 'QwQ-32B critic')}
    names = [(f'Ours\n{HL}', 'ours'), (f'Plain avg\n{HL}', 'ctl'), (f'Oriented\navg {HL}', 'ctl'), ('IU-PCR', 'ctl'), ('Contin.\nL-SML', 'ctl'), ('Step\nindex', 'base'), ('PRM-7B\n(paper)', 'prm'), ('LLM judge\n(paper)', 'prm')]
    CBX = OUT.parent / 'monday_cont_baselines_v1/EXTERNAL_METRICS.csv'; CB = pd.read_csv(CBX) if CBX.exists() else None
    OR = pd.read_csv(OUT / 'EXTERNAL_ORIENTED_CONTROL.csv') if (OUT / 'EXTERNAL_ORIENTED_CONTROL.csv').exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), sharey=True)
    for ax, (cell, (title, pub, critic)) in zip(axes, cells.items()):
        style(ax, t); g = R[R.cell == cell]
        o = g[(g.bank == HL) & g.scorer.isin(M4) & (g.rule == 'ANSWER_Z_Q80')].official_metric
        vals = [(o.min(), o.mean(), o.max()), g[(g.bank == HL) & (g.scorer == 'ALL_EQ') & (g.rule == 'ANSWER_Z_Q80')].official_metric.iloc[0],
                (OR[(OR.cell == cell) & (OR.bank == HL)].official_metric.iloc[0] if OR is not None else np.nan),
                (CB[(CB.cell == cell) & (CB.bank == HL) & (CB.scorer == 'IU_PCR')].official_metric.iloc[0] if CB is not None else np.nan),
                (CB[(CB.cell == cell) & (CB.bank == HL) & (CB.scorer == 'LSML_CONT')].official_metric.iloc[0] if CB is not None else np.nan),
                g[(g.scorer == 'STEP_INDEX') & (g.rule == 'ANSWER_Z_Q80')].official_metric.iloc[0], pub['Qwen2.5-Math-PRM-7B'] / 100, pub[critic] / 100]
        for x, ((lab, kind), v) in enumerate(zip(names, vals)):
            if kind == 'prm': ax.plot(x, v, 'o', color=t['prm'], markersize=9, markerfacecolor='none', markeredgewidth=1.8); ax.annotate(f'{v:.3f}', (x, v), xytext=(13, 0), textcoords='offset points', va='center', fontsize=8, color=t['ink2'])
            else: vdot(ax, x, v, t, kind)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n for n, _ in names], color=t['ink'], fontsize=8.5); ax.set_title(title, color=t['ink'], fontsize=10)
    fig.text(.01, -.05, 'Ours, both plain averages, IU-PCR and continuous L-SML (sealed run, 7 Oct): label-free refit on each external cell, flag the top 20% of the cell (q80). Hollow orange = paper-reported, not reproduced; different threshold access, context only. Exploratory: external labels were opened in earlier rounds.', color=t['ink2'], fontsize=8.5)
    fig.tight_layout(); save(fig, 'E2_external_official', t, mode)


def fig_E1(t, mode):
    if not (OUT / 'TRANSFER.csv').exists(): return
    T = pd.read_csv(OUT / 'TRANSFER.csv'); R = pd.read_csv(OUT / 'METRICS.csv')
    titles = {'hard2verify_qwen3_8b': 'Hard2Verify · Qwen3-8B', 'socratic_qwen3_8b': 'Socratic · Qwen3-8B', 'socratic_qwq32b': 'Socratic · QwQ-32B'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (cell, title) in zip(axes, titles.items()):
        style(ax, t, grid='both'); d = pd.read_csv(OUT / cell / 'DEV_VS_EXTERNAL.csv')
        hl = d[(d.bank == HL) & d.scorer.isin(M4)]; ctl = d[d.scorer == 'ALL_EQ']; oth = d[~d.index.isin(hl.index) & ~d.index.isin(ctl.index)]
        ax.scatter(oth.prmb_within_auc, oth.within_auc, s=26, color=t['muted'], alpha=.7, label='other bank × scorer')
        ax.scatter(ctl.prmb_within_auc, ctl.within_auc, s=40, facecolors='none', edgecolors=t['ink2'], label='Plain average per bank')
        ax.scatter(hl.prmb_within_auc, hl.within_auc, s=70, color=t['ours'], edgecolors=t['surface'], linewidths=1.5, zorder=3, label=f'{HL} methods')
        si = R[(R.cell == cell) & (R.scorer == 'STEP_INDEX') & (R.rule == 'TOP_1')].within_auc.iloc[0]
        ax.axhline(si, color=t['prm'], linestyle='--', linewidth=1.5); ax.text(d.prmb_within_auc.min(), si, ' step index alone', color=t['prm'], fontsize=8.5, va='bottom')
        r = T[T.cell == cell].iloc[0]
        ax.set_title(f'{title}\nSpearman dev→external {r.spearman_dev_auc_vs_ext_auc:.2f} ({int(r.configs)} configs)', color=t['ink'], fontsize=10)
        ax.set_xlabel('Development PRMBench AUC', color=t['ink2']); ax.set_ylabel('External AUC', color=t['ink2'])
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, labelcolor=t['ink'], bbox_to_anchor=(.5, -.06), fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1)); save(fig, 'E1_dev_vs_external', t, mode)


if __name__ == '__main__':
    for mode, t in THEME.items():
        for f in (fig_A, fig_nesting, fig_B1, fig_B2, fig_B3, fig_E1, fig_E2): f(t, mode)
    (OUT / 'FIGURES_MANIFEST.json').write_text(json.dumps({'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figures': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FIG.glob('*'))}}, indent=1), encoding='utf8')
    print('figures', len(list(FIG.glob('*'))))
