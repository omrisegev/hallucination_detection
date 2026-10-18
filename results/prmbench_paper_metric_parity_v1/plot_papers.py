"""One PNG per paper: their published points/bars (blue circles / blue bars) with ours overlaid (orange,
distinct markers). Missing arms are grey hatched with 'GAP: <reason>'. Values come from the *_RESULTS.json files."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, ARMS
from prism_metrics import PRISM_T2, PRISM_T7, T7_COLS, PRISM_T6

THEIRS, OURS, GAPC = '#2a78d6', '#eb6834', '#b9b8b2'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
MK = {'PRM_native': 'o', 'LSML_CONT': '*', 'CHAIN_N16': 'D', 'CHAIN_N13': 's'}
LBL = {'PRM_native': 'PRM native (ours, recomputed)', 'LSML_CONT': 'LSML_CONT (label-chosen 3 ch.)',
       'CHAIN_N16': 'CHAIN_N16 (label-free, digit ch.)', 'CHAIN_N13': 'CHAIN_N13 (label-free, digit-free)'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False})
R = lambda n: json.load(open(HERE / n))


def gapbar(ax, x, h, w, text, rot=90):
    ax.bar(x, h, w, color='none', edgecolor=GAPC, hatch='///', linewidth=1)
    ax.text(x, h / 2, text, rotation=rot, ha='center', va='center', fontsize=6.5, color=INK2,
            bbox=dict(facecolor=SURF, edgecolor='none', pad=1.5))


def prism():
    P = R('PRISM_RESULTS.json')
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={'width_ratios': [1.05, 1.6, 1.25]})
    ax = axs[0]
    for a in ARMS:
        c = np.array(P['curves'][a]); ax.plot(c[:, 0], 100 - c[:, 1], color=OURS if a != 'PRM_native' else INK2,
                                            lw=1.2, alpha=.55, ls='-' if a == 'LSML_CONT' else ':')
    for nm, d in PRISM_T2.items():
        ax.scatter(d['FPR'], d['TPR'], s=46, color=THEIRS, edgecolor=SURF, lw=1.2, zorder=3)
        ax.annotate(nm.replace('Baseline (Qwen-PRM-7B)', 'Baseline').replace(' Loss', ''), (d['FPR'], d['TPR']),
                    textcoords='offset points', xytext=(4, 4), fontsize=6.5, color=THEIRS)
    for a in ARMS:
        r = P['rates'][a]
        ax.scatter(r['FPR'], r['TPR'], marker=MK[a], s=110 if a == 'LSML_CONT' else 55, color=OURS, edgecolor=INK, lw=.6, zorder=4, label=LBL[a])
        m = P['matched_tpr_87_30'][a]
        ax.scatter(m['FPR'], m['TPR'], marker=MK[a], s=50, facecolor='none', edgecolor=OURS, lw=1.2, zorder=4)
    ax.axhline(87.30, color=THEIRS, lw=.6, ls='--')
    ax.text(76, 87.6, "PRISM's TPR 87.30", fontsize=6.5, color=THEIRS)
    ax.set(xlim=(30, 80), ylim=(80, 100), xlabel='FPR % (error steps passed as valid)  - lower is better',
           ylabel='TPR % (valid steps kept)', title='Table 2: PRMBench step rates (filled = deployed rule;\nhollow = moved to TPR 87.30; lines = our threshold sweeps)')
    ax.legend(fontsize=6.5, loc='lower right', frameon=False)
    ax = axs[1]
    cols = T7_COLS; x = np.arange(len(cols)); w = .14
    series = [('PRISM baseline (published)', PRISM_T7['Baseline'], THEIRS, 1.0), ('PRISM 132k C4 (published)', PRISM_T7['132k Curriculum-4 (best FPR)'], THEIRS, .55)]
    for j, (nm, v, col, al) in enumerate(series):
        ax.bar(x + (j - 2.5) * w, v, w, color=col, alpha=al, label=nm, edgecolor=SURF, lw=.8)
    for j, a in enumerate(ARMS):
        cat = P['categories'][a]; v = [P['rates'][a]['PRMScore']] + [cat[c] for c in cols[1:]]
        ax.bar(x + (j - .5) * w, v, w, color=OURS, alpha=[.35, 1, .75, .55][j], label=LBL[a] + ' (ours)', edgecolor=SURF, lw=.8,
               hatch=['', '', '..', '\\\\'][j])
    ax.set_xticks(x); ax.set_xticklabels(cols, fontsize=8, rotation=35, ha='right'); ax.set_ylim(40, 102)
    ax.set(ylabel='official PRMScore', title='Table 7: PRMBench Overall and categories (same official definition)')
    ax.legend(fontsize=6.5, ncol=2, loc='upper left', frameon=False)
    ax = axs[2]
    subs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']; x = np.arange(len(subs)); w = .16
    ax.bar(x - 1.5 * w, [PRISM_T6[s][2] for s in subs], w, color=THEIRS, label='PRISM baseline F1 (published)')
    ax.bar(x - .5 * w, [P['processbench_native'][s]['F1'] for s in subs], w, color=OURS, alpha=.35, label='PRM native F1 (ours, recomputed)')
    for k, s in enumerate(subs):
        gapbar(ax, x[k] + 1.0 * w, 70, 2 * w, 'GAP: fused arms\nno backbone telemetry\non ProcessBench')
    ax.set_xticks(x); ax.set_xticklabels(subs); ax.set_ylim(0, 100)
    ax.set(ylabel='F1 (harmonic mean of error/correct accuracy)', title='Table 6: ProcessBench (fused arms missing)')
    ax.text(.02, .97, 'hatched = LSML_CONT / CHAIN arms: not computable', transform=ax.transAxes, fontsize=6.5, color=INK2, va='top')
    ax.text(.02, .91, "their olympiad/omnimath cells appear transposed (see README)", transform=ax.transAxes, fontsize=6.5, color=INK2, va='top')
    ax.legend(fontsize=6.5, loc='lower left', frameon=True, facecolor=SURF, edgecolor=GRID)
    fig.suptitle('PRISM (arXiv 2606.09078) vs our PRM-boosting arms on PRMBench, N = 6,211 non-control answers. Blue = published, orange = ours.', color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / 'PRISM_PARITY.png', dpi=150); plt.close(fig)


def crop():
    C = R('CROP_RESULTS.json')['arms']
    T6 = {'Random': (0.4994, 0.0050, 8.7, 0.5), 'Token/format': (0.7751, 0.0034, 16.2, 1.0), 'Trace features': (0.7812, 0.0034, 15.9, 0.7),
          'Direct PRM': (0.8010, 0.0049, 4.8, 0.2), 'Trace+PRM': (0.8336, 0.0027, 21.8, 0.9)}
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for nm, (a, ae, k, ke) in T6.items():
        ax.errorbar(a, k, xerr=ae, yerr=ke, fmt='o', ms=7, color=THEIRS, mec=SURF, capsize=2, zorder=3)
        off = {'Token/format': (-95, 6), 'Trace features': (6, -12)}.get(nm, (5, -10))
        ax.annotate(nm + ' (published)', (a, k), textcoords='offset points', xytext=off, fontsize=7, color=THEIRS)
    ours = {'Random': ('x', 'Random (ours)'), 'Direct PRM (1-reward)': ('o', 'Direct PRM, fixed grid (ours)'),
            'Direct PRM (1-reward), quantile grid': ('o', 'Direct PRM, quantile grid (ours, bridge)'),
            'Position/length LR (supervised)': ('v', 'Position/length LR, supervised (ours)'),
            'TELE13+PRM LR (supervised)': ('^', 'Telemetry13+PRM LR, supervised (ours)'),
            'LSML_CONT': ('*', 'LSML_CONT, label-free (ours)'), 'CHAIN_N16': ('D', 'CHAIN_N16, label-free (ours)'),
            'CHAIN_N13': ('s', 'CHAIN_N13, label-free (ours)')}
    for nm, (mk, lb) in ours.items():
        a = C[nm]
        hollow = 'supervised' in nm or 'quantile' in nm
        ax.errorbar(a['auroc'], a['kept'], xerr=a['auroc_ci95'], yerr=a['kept_ci95'], fmt=mk, ms=12 if mk == '*' else 7,
                    mfc='none' if hollow else OURS, mec=OURS if hollow else INK, mew=1.2, ecolor=OURS, capsize=2, zorder=4, label=lb + f"  risk {a['risk']:.1f}%")
    ax.set(xlabel='step AUROC, pooled over all test steps', ylabel='prefix kept % at alpha = 0.05 (mean M/T over test traces)',
           xlim=(0.47, 0.86), ylim=(0, 30),
           title='CROP (arXiv 2605.30085) Table 6 / Fig. 4, PRMBench: published (blue) vs ours (orange)')
    ax.text(0.475, 2.6, 'ours: first 5,000 non-control rows (67,465 vs their 67,445 steps), 10 re-implemented 60/20/20 splits.\n'
            'The CROP threshold uses calibration labels for every arm. Our Direct PRM (fixed grid) lands on the published point.', fontsize=6.5, color=INK2)
    ax.legend(fontsize=7, loc='upper left', frameon=False, title='ours (filled = label-free fusion; hollow = supervised or bridge)', title_fontsize=7)
    ax.text(0.475, 0.5, 'GAP: their Token/format & Trace-features rows need step text / Llama-3.1-8B features\n(not local); our position/length LR is a subset of Token/format.', fontsize=6.5, color=INK2)
    fig.tight_layout(); fig.savefig(HERE / 'CROP_PARITY.png', dpi=150); plt.close(fig)


def slider():
    S = R('SLIDER_RESULTS.json')
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    pub = [('Step-RRI', 61.91), ('Cosine', 51.60), ('InfoGain', 48.38)]
    xs = 0
    for nm, v in pub:
        ax.bar(xs, v, .7, color=THEIRS); ax.text(xs, v + 1, f'{v:.2f}', ha='center', fontsize=7, color=INK)
        ax.text(xs, 2, nm + '\n(published)', ha='center', fontsize=7, color=SURF); xs += 1
    xs += .6; ticks = []
    for a in ARMS:
        r = S['arms'][a]
        for k, (key, al, h) in enumerate((('all_steps__LR_balanced__acc', 1, ''), ('all_steps__LR_balanced__bal_acc', .6, '..'),
                                          ('all_steps__label_free_Q80__bal_acc', .35, '\\\\'))):
            v = r[key]; ax.bar(xs + k * .27, v, .25, color=OURS, alpha=al, hatch=h, edgecolor=SURF,
                               label=['accuracy, balanced-LR threshold', 'balanced accuracy, balanced-LR threshold', 'balanced accuracy, label-free Q80 (no threshold labels)'][k] if a == 'PRM_native' else None)
            ax.text(xs + k * .27, v + 1, f'{v:.1f}', ha='center', fontsize=6.5, color=INK)
        ticks.append((xs + .27, a)); xs += 1.2
    maj = S['arms']['PRM_native']['all_steps__majority_acc']
    ax.axhline(maj, color=INK2, lw=.8, ls='--'); ax.text(-.4, maj + 1, f"always 'not redundant' = {maj:.1f}% plain accuracy (base rate {100*S['redundant_rate']:.1f}% redundant)", fontsize=7, color=INK2)
    ax.set_xticks([0, 1, 2] + [t for t, _ in ticks]); ax.set_xticklabels(['', '', ''] + [a for _, a in ticks], fontsize=8)
    ax.set(ylim=(0, 115), ylabel='% on test steps', title='SLIDER (arXiv 2610.00571) Fig. 7: PRMBench redundancy class, step-level redundancy detection')
    ax.text(-.4, 112, f"ours: {S['n_problems']} problems / {S['n_steps']} steps, 10 random 50/50 trace splits. APPROXIMATE: their split, step scope\n"
            "and accuracy type are not stated. Their Step-RRI uses 50 GPT-4o-mini variants per problem.", fontsize=7, color=INK2, va='top')
    ax.legend(fontsize=7, loc='upper right', frameon=False, title='ours (orange)', title_fontsize=7)
    fig.tight_layout(); fig.savefig(HERE / 'SLIDER_PARITY.png', dpi=150); plt.close(fig)


def optagg():
    G = R('AGGREGATION_PROXY.json')
    OA = [('Optimal\n(oracle)', 66.5), ('Logit WV', 63.3), ('Linear WV', 62.9), ('BoN', 61.8), ('Vanilla WV', 60.8), ('KDE WV', 60.2), ('MV', 57.1)]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for k, (nm, v) in enumerate(OA):
        ax.bar(k, v, .7, color=THEIRS); ax.text(k, v + 1, f'{v:.1f}', ha='center', fontsize=7)
    for j, a in enumerate(ARMS):
        gapbar(ax, len(OA) + .3 + j, 60, .7, f'GAP: {a}\nneeds N sampled responses\nwith graded final answers\n(PRMBench has 1 trace\nper perturbation)')
    ax.set_xticks(list(range(len(OA))) + [len(OA) + .3 + j for j in range(4)])
    ax.set_xticklabels([n for n, _ in OA] + ARMS, fontsize=7.5)
    ax.set(ylim=(0, 80), ylabel='MATH final-answer accuracy %, N = 32',
           title='Optimal Aggregation (arXiv 2510.13918) Table 1, Qwen2.5-PRM-7B row (mean of 5 LLMs): published (blue);\n'
                 'our arms cannot be placed: different task (Best-of-N answer selection), no compatible data')
    ax.text(.01, .97, 'Proxy (NOT their metric): pick the clean trace inside a PRMBench source group: ' +
            ', '.join(f"{a} {G['arms'][a]:.1f}%" for a in ARMS) + f"; random {G['random_pick_hit_rate']:.1f}%",
            transform=ax.transAxes, fontsize=6.5, color=INK2, va='top')
    fig.tight_layout(); fig.savefig(HERE / 'OPTAGG_PARITY.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    prism(); crop(); slider(); optagg(); print('ok')
