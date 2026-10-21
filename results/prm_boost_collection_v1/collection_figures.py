"""Figures for plan sections 2.1-2.3: one PNG per figure in figures/, the data behind it in figures/data/<name>.csv.
Every caption (figures/CAPTIONS.md) names the benchmark, N, the count/threshold rule and the access of each series.

    python collection_figures.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate_collection as E

FIG = E.HERE / 'figures'; FD = FIG / 'data'; T = E.TABLES
# Fixed entity -> colour (validated default categorical palette, slots 1-5, light mode); never by rank.
COL = {'PRM': '#2a78d6', 'CHAIN_N16': '#eb6834', 'LSML_CONT': '#1baf7a', 'CHAIN_N13': '#eda100', 'TELE_N16': '#e87ba4',
       'paper': '#5f5e5a', 'LR': '#6250d6', 'other': '#8a8983'}
MRK = {'PRM': 'o', 'CHAIN_N16': 's', 'LSML_CONT': '^', 'CHAIN_N13': 'D', 'TELE_N16': 'v', 'paper': '*', 'LR': 'P', 'other': 'x'}
LS = {'PRM': '-', 'CHAIN_N16': '--', 'LSML_CONT': '-.', 'CHAIN_N13': ':', 'TELE_N16': (0, (5, 1))}
INK, MUTED, GRID = '#1f1e1c', '#6b6a64', '#e6e5e0'
CAPS = []


def style(ax, title=None):
    ax.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title: ax.set_title(title, fontsize=9, color=INK, loc='left')


def ent(name):
    for k in ('CHAIN_N16', 'CHAIN_N13', 'LSML_CONT', 'TELE_N16'):
        if k in name: return k
    if 'LR' in name: return 'LR'
    if 'PRM' in name or name.startswith('Direct'): return 'PRM'
    return 'other'


def rd(p): return E.read_csv(p)


def save(fig, name, caption, rows=None):
    fig.savefig(FIG / f'{name}.png', dpi=160, bbox_inches='tight', facecolor='white'); plt.close(fig)
    if rows is not None: E.write_csv(FD / f'{name}.csv', rows)
    CAPS.append(f'### {name}.png\n{caption}\nData: `figures/data/{name}.csv`.\n')


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan


# ---------------------------------------------------------------------------------------------- PRISM Fig 5b
def prism_5b():
    R = rd(FD / 'prism_fig5b_fpr_fnr.csv')
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.6), sharex=True, sharey=True)
    for ax, prm in zip(axs, E.PRMS):
        for arm in ('PRM_native', 'CHAIN_N16', 'CHAIN_N13', 'LSML_CONT'):
            pts = sorted([(f(r['FPR']), f(r['FNR'])) for r in R if r['prm'] == prm and r['arm'] == arm and r['kind'] == 'global threshold sweep'])
            e = ent(arm); ax.plot(*zip(*pts), color=COL[e], ls=LS[e], lw=2, label=f"{'PRM (global threshold)' if e == 'PRM' else arm + ' (global answer-z threshold)'}")
        for r in R:
            if r['prm'] != prm or not r['kind'].startswith('operating') or r['arm'] == 'TELE_N16': continue
            e = ent(r['arm']); rule = r['kind'].split()[-1]
            if r['arm'] in ('PRM_Q80', 'PRM_Q80_PROB'): e = 'PRM'
            ax.scatter(f(r['FPR']), f(r['FNR']), s=46, color=COL[e], marker={'native': 'o', 'Q80': 's', 'GAP': 'D'}[rule],
                       edgecolor='white', lw=1.2, zorder=4)
        if prm == 'qwen_prm_7b':
            ax.scatter(47.13, 12.69, marker='*', s=180, color=COL['paper'], edgecolor='white', zorder=5, label='PRISM (paper, re-trained PRM)')
            ax.annotate('PRISM', (47.13, 12.69), (52, 18), fontsize=8, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED))
        style(ax, E.PRM_LABEL[prm]); ax.set_xlabel('FPR (% error steps passed as valid)', fontsize=8)
        ax.set_xlim(20, 100); ax.set_ylim(0, 45)
    axs[0].set_ylabel('FNR (% valid steps flagged)', fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker=m, ls='', color=MUTED, label=lab) for m, lab in (('o', 'native rule'), ('s', 'Q80 point'), ('D', 'GAP point'))]
    fig.legend(handles=h, loc='lower center', ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(.5, -.08))
    save(fig, 'prism_fig5b_fpr_fnr',
         'PRMBench, 6,211 non-control answers (83,371 steps per PRM). FPR-FNR trade-off of one global threshold swept over the PRM risk '
         '(PRM trained by its authors) and over the answer-z score of CHAIN_N16 / CHAIN_N13 (label-free fusion) and LSML_CONT (label-selected '
         'channels). Points: native rule (PRM), the label-free Q80 and GAP rules. Star: PRISM (re-trained PRM, PRM800K labels), Qwen-7B only. '
         'Positive = valid step, as in PRISM. GAP is a per-answer count, not a global threshold, so its point need not lie on the curve.',
         R)


def prism_7():
    R = rd(FD / 'prism_fig7_histograms.csv')
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, arm in zip(axs, ('PRM reward', 'CHAIN_N16', 'LSML_CONT')):
        for lab, c in (('error step', '#e34948'), ('correct step', COL['PRM'])):
            rr = [r for r in R if r['prm'] == 'qwen_prm_7b' and r['arm'] == arm and r['steps'] == lab]
            x = [(f(r['bin_lo']) + f(r['bin_hi'])) / 2 for r in rr]; y = [f(r['density']) for r in rr]
            ax.plot(x, y, color=c, lw=2, label=f"{lab} (n={rr[0]['n']})", ls='-' if lab.startswith('error') else '--')
        style(ax, f'{arm} ({rr[0]["scale"]})'); ax.set_ylabel('density', fontsize=8)
        ax.legend(fontsize=7, frameon=False)
    save(fig, 'prism_fig7_score_histograms',
         'PRMBench, Qwen2.5-Math-PRM-7B, 6,211 non-control answers. Score densities on error vs correct steps: PRM reward (trained PRM), '
         'CHAIN_N16 answer-z (label-free fusion), LSML_CONT answer-z (label-selected channels). Data for all 5 PRMs in the CSV.',
         [r for r in R])


def prism_4():
    R = rd(FD / 'prism_fig4_pairwise_by_margin.csv')
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.2), sharey=True)
    for ax, prm in zip(axs, E.PRMS):
        for arm in ('PRM_Q80', 'CHAIN_N16', 'CHAIN_N13', 'LSML_CONT'):
            rr = [r for r in R if r['prm'] == prm and r['arm'] == arm and r['margin_lo'] != 'all']
            e = ent(arm) if arm != 'PRM_Q80' else 'PRM'
            ax.plot(range(1, len(rr) + 1), [f(r['pairwise_acc']) for r in rr], color=COL[e], ls=LS[e], marker=MRK[e], ms=5, lw=2,
                    label='PRM (answer-z)' if e == 'PRM' else arm)
        style(ax, E.PRM_LABEL[prm]); ax.set_xlabel('margin decile (|z_err - z_valid|)', fontsize=8); ax.axhline(.5, color=MUTED, lw=1, ls=':')
    axs[0].set_ylabel('pairwise accuracy (error above valid)', fontsize=8); axs[0].legend(fontsize=7, frameon=False)
    save(fig, 'prism_fig4_pairwise_by_margin',
         'PRMBench, within-answer pairs (one error step, one valid step of the same answer). Accuracy = share of pairs where the error step '
         'scores higher, by decile of the answer-z score margin. No decision rule. Access: PRM trained by its authors; CHAIN label-free; '
         'LSML_CONT label-selected.', R)


# ---------------------------------------------------------------------------------------------- CROP figures
def crop_figs():
    A = rd(FD / 'crop_fig3_kept_vs_alpha.csv')
    show = ['Direct PRM (1-reward), quantile grid', 'Direct PRM (1-reward)', 'CHAIN_N16', 'CHAIN_N13', 'LSML_CONT', 'Telemetry13+PRM LR (supervised)']
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
    for ax, prm in zip(axs, E.PRMS):
        for p in show:
            rr = sorted([r for r in A if r['prm'] == prm and r['proxy'] == p], key=lambda r: f(r['alpha']))
            seen = {}
            for r in rr: seen[f(r['alpha'])] = f(r['kept_mean'])
            e = ent(p); ls = ':' if p == 'Direct PRM (1-reward)' else LS.get(e, '-')
            ax.plot(list(seen), list(seen.values()), color=COL[e], ls=ls, marker=MRK[e], ms=4, lw=2, label=p)
        style(ax, E.PRM_LABEL[prm]); ax.set_xlabel('alpha (target prefix risk)', fontsize=8)
    axs[0].set_ylabel('prefix kept (%)', fontsize=8)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(.5, -.1))
    save(fig, 'crop_fig3_kept_vs_alpha',
         'PRMBench, CROP protocol: first 5,000 non-control traces (Hub order), 10 seeds, 60/20/20 splits; lambda calibrated with labels on '
         'the 20% split for EVERY proxy (CROP access). Prefix kept vs alpha. Direct PRM: trained PRM, fixed 101-point grid (dotted) and '
         'train-quantile grid; CHAIN label-free fusion; LSML_CONT label-selected; Telemetry13+PRM LR supervised. Qwen-7B is CROP\'s own '
         'PRM; the other 4 PRMs are our extension.', A)
    R = rd(T / 'COMP_CROP_T6.csv')
    fig, ax = plt.subplots(figsize=(7.5, 5))
    pm = {p: m for p, m in zip(E.PRMS, ('o', 's', '^', 'D', 'v'))}
    for r in R:
        e = ent(r['proxy']);
        if r['proxy'] == 'Random': e = 'other'
        ax.scatter(f(r['auroc_mean']), f(r['kept_mean']), color=COL[e], marker=pm[r['prm']], s=50, edgecolor='white', lw=1)
    for k, (au, kept, _) in {'Trace+PRM (paper)': (.8336, 21.8, 0), 'Direct PRM (paper)': (.8010, 4.8, 0), 'Token/format (paper)': (.7751, 16.2, 0),
                             'Trace features (paper)': (.7812, 15.9, 0)}.items():
        ax.scatter(au, kept, marker='*', s=160, color=COL['paper'], edgecolor='white'); ax.annotate(k, (au, kept), (4, -11) if k.startswith('Trace features') else (4, 4), textcoords='offset points', fontsize=7, color=INK)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', color=COL[e], label=lab) for e, lab in (('PRM', 'Direct PRM (trained)'), ('CHAIN_N16', 'CHAIN_N16'),
         ('CHAIN_N13', 'CHAIN_N13'), ('LSML_CONT', 'LSML_CONT (label-selected)'), ('TELE_N16', 'TELE_N16'), ('LR', 'supervised LR'), ('other', 'Random'))]
    h += [Line2D([], [], marker=m, ls='', color=MUTED, label=E.PRM_LABEL[p]) for p, m in pm.items()]
    ax.legend(handles=h, fontsize=7, frameon=False, loc='upper left'); style(ax, 'CROP Fig 4 analog: pooled step AUROC vs prefix kept (alpha .05)')
    ax.set_xlabel('pooled step AUROC (test split)', fontsize=8); ax.set_ylabel('prefix kept (%) at risk <= 5%', fontsize=8)
    save(fig, 'crop_fig4_auroc_vs_kept',
         'PRMBench CROP protocol (5,000 traces, 10 seeds, alpha .05). One point per proxy x PRM (marker = PRM, colour = proxy). Stars: CROP '
         'Table 6 (Qwen-7B). Thresholds are label-calibrated for every proxy; LR proxies are supervised; our fused scores are label-free.', R)
    L = rd(FD / 'crop_fig5_by_length.csv'); P8 = rd(FD / 'crop_fig8_by_first_error.csv')
    for name, D, key, cats, ylab, cap in (
            ('crop_fig5_by_length', L, 'steps_bin', ['1-8', '9-12', '13-16', '17-max'], 'retention (%)',
             'PRMBench CROP protocol, alpha .05, test traces pooled over 10 seeds; retention (mean kept fraction) by trace length (steps). Risk per bin in the CSV.'),
            ('crop_fig8_by_first_error', P8, 'first_error', ['early', 'mid', 'late'], 'contamination (%)',
             'PRMBench CROP protocol, alpha .05, test traces with an error; share whose kept prefix contains an error, by relative position of the first error (thirds).')):
        fig, axs = plt.subplots(1, 5, figsize=(17, 3.2), sharey=True)
        for ax, prm in zip(axs, E.PRMS):
            for j, p in enumerate(['Direct PRM (1-reward), quantile grid', 'CHAIN_N16', 'LSML_CONT', 'Telemetry13+PRM LR (supervised)']):
                vals = [f(next((r for r in D if r['prm'] == prm and r['proxy'] == p and r[key] == cc), {}).get('retention' if 'fig5' in name else 'contamination')) for cc in cats]
                ax.bar(np.arange(len(cats)) + (j - 1.5) * .2, vals, .18, color=COL[ent(p)], label=p, edgecolor='white', lw=1)
            ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=8); style(ax, E.PRM_LABEL[prm])
        axs[0].set_ylabel(ylab, fontsize=8)
        fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(.5, -.1))
        save(fig, name, cap + ' Direct PRM trained; CHAIN label-free; LSML_CONT label-selected; LR supervised.', D)


# ---------------------------------------------------------------------------------------------- status figures (2.1)
def status_figs():
    M = rd(T / 'STATUS_MAIN.csv')
    bn = {'prmbench': 'PRMBench PRMScore', 'socratic': 'Socratic PRMScore (x100)', 'hard2verify': 'Hard2Verify step Balanced F1 (x100, 177 answers)'}
    rows_out = []
    fig, axs = plt.subplots(1, 3, figsize=(17, 4))
    arms = [('PRM native', 'native', 'PRM'), ('PRM same rule (better scale under GAP)', 'GAP', 'PRM'), ('TELE_N16', 'GAP', 'TELE_N16'),
            ('CHAIN_N16', 'GAP', 'CHAIN_N16'), ('CHAIN_N13', 'GAP', 'CHAIN_N13'), ('LSML_CONT', 'GAP', 'LSML_CONT')]
    for ax, b in zip(axs, E.BENCHES):
        for j, (row, rule, e) in enumerate(arms):
            vals = []
            for prm in E.PRMS:
                r = next((r for r in M if r['benchmark'] == b and r['prm'] == prm and r['row'] == row), None)
                v = f(r['native'] if rule == 'native' else r['GAP']) if r else np.nan; vals.append(v)
                rows_out.append(dict(benchmark=b, prm=prm, row=row, rule=rule, value=v, access=r['access'] if r else ''))
            ax.bar(np.arange(5) + (j - 2.5) * .13, vals, .12, color=COL[e] if row != 'PRM native' else '#a9c9f0',
                   edgecolor='white', lw=1, label=f'{row} [{rule}]', hatch='//' if row == 'PRM native' else None)
        ax.set_xticks(range(5)); ax.set_xticklabels([E.PRM_LABEL[p].replace('-Mistral', '').replace('Qwen2.5-Math-', 'Qwen-') for p in E.PRMS], fontsize=7, rotation=15)
        style(ax, bn[b]); lo = np.nanmin([x['value'] for x in rows_out if x['benchmark'] == b]); ax.set_ylim(max(0, lo - 8), None)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(.5, -.12))
    save(fig, 'status_official_by_prm_gap',
         'Official metric per benchmark and PRM under the label-free GAP count rule (PRM native rule hatched). PRMBench: 6,211 non-control answers; '
         'Socratic: 2,995 answers; Hard2Verify: 177 answers every PRM can read. PRM same rule = the PRM answer-z score with the same GAP rule, '
         'on the better of logit(1-reward) / (1-reward) (chosen after scoring; favours the PRM). Access: PRM trained by its authors; TELE/CHAIN '
         'label-free fusion; LSML_CONT label-selected channels. Development data, already inspected.', rows_out)
    C = rd(T / 'STATUS_CONTRASTS.csv')
    want = [('CHAIN_N16[GAP] - ', 'PRM ref', 'CHAIN_N16 - PRM (GAP)'), ('EQ_N16[GAP] - ', 'PRM ref', 'EQ_N16 - PRM (GAP)'),
            ('CHAIN_N16[GAP] - EQ_N16[GAP]', None, 'CHAIN - EQ (GAP)'), ('TELE_N16[GAP] - MEAN_N16[GAP]', None, 'TELE - MEAN (GAP)'),
            ('CHAIN_N16[GAP] - CHAIN_N16[Q80]', None, 'GAP - Q80 (CHAIN_N16)')]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4)); out = []
    for ax, b in zip(axs, E.BENCHES):
        for j, (pref, kind, lab) in enumerate(want):
            for i, prm in enumerate(E.PRMS):
                rr = [r for r in C if r['benchmark'] == b and r['prm'] == prm and r['metric'] == 'official' and
                      (r['contrast'] == pref if kind is None else (r['contrast'].startswith(pref) and 'PRM' in r['contrast'].split(' - ')[1] and '[GAP]' in r['contrast'].split(' - ')[1]))]
                if not rr: continue
                r = rr[0]; x = i + (j - 2) * .16; d, lo, hi = f(r['delta']), f(r['ci_low']), f(r['ci_high'])
                ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=['o', 's', 'D', '^', 'v'][j], color=['#eb6834', '#e87ba4', '#1baf7a', '#6250d6', '#eda100'][j],
                            ms=5, lw=1.5, capsize=2, label=lab if i == 0 else None)
                out.append(dict(benchmark=b, prm=prm, contrast=r['contrast'], label=lab, delta=d, ci_low=lo, ci_high=hi))
        ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(range(5))
        ax.set_xticklabels([E.PRM_LABEL[p].replace('-Mistral', '').replace('Qwen2.5-Math-', 'Qwen-') for p in E.PRMS], fontsize=7, rotation=15)
        style(ax, bn[b] + ': paired contrasts, 95% CI')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(.5, -.12))
    save(fig, 'status_contrasts_gap',
         'Paired bootstrap contrasts (2,000 draws; PRMBench by source group, external sets by answer), official metric points. PRM reference = '
         'PRM answer-z with GAP on the better scale (post hoc). All rules label-free.', out)


def main():
    FIG.mkdir(exist_ok=True); FD.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.labelcolor': INK})
    for fn in (prism_5b, prism_7, prism_4, crop_figs, status_figs):
        try:
            fn(); print('ok', fn.__name__)
        except FileNotFoundError as e:
            print('skip', fn.__name__, e)
    (FIG / 'CAPTIONS.md').write_text('# Figure captions\n\n' + '\n'.join(CAPS), encoding='utf-8')


if __name__ == '__main__':
    main()
