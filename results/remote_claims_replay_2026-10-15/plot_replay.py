"""Render exploratory graphs from the independently recomputed CSVs.

This script only reads the audit results and writes figures alongside them.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'plots'
OUT.mkdir(exist_ok=True)
with (HERE / 'PRM7B_METRICS.csv').open() as f:
    DATA = {r['arm']: r for r in csv.DictReader(f)}
with (HERE / 'PRM7B_CONTRASTS.csv').open() as f:
    CONTRASTS = {(r['contrast'], r['metric']): r for r in csv.DictReader(f)}

BLUE = '#1765a7'
ORANGE = '#b56a15'
GRAY = '#687482'
LIGHTGRAY = '#99a2ab'
INK = '#192b3c'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 11,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK,
                     'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def number(arm, metric):
    return float(DATA[arm][metric])


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e7ebef', lw=.8)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#bdc6ce')
    ax.tick_params(axis='y', length=0)


def save(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=190, facecolor='white')
    plt.close(fig)


def mechanism():
    arms = ['PRM_Q80', 'S3_HEAD_FLAT_MEAN', 'S3_HEAD_CLUSTER_EQUAL_L2',
            'S3_HEAD_CLUSTER_MEANS', 'S3_HEAD_LSML']
    labels = ['PRM head alone', 'Flat mean of all 4 inputs',
              'Same clusters + equal L2 weights',
              'Same clusters + mean within / across', 'Full L-SML (head + 3 signals)']
    colors = [LIGHTGRAY, GRAY, ORANGE, ORANGE, BLUE]
    y = np.arange(len(arms))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.7), sharey=True)
    fig.subplots_adjust(left=.33, right=.95, top=.76, bottom=.31, wspace=.30)
    fig.suptitle('What adds value on PRM-7B?', x=.035, y=.965, ha='left',
                 fontsize=20, fontweight='bold')
    fig.text(.035, .875, 'The four fusion inputs are PRM risk + energy + entropy CUSUM + realized_drv.\n'
             'All come from the same PRM checkpoint. Every row uses ANSWER_Z_Q80.', fontsize=11, va='top')
    for ax, metric, title, lim in zip(axes, ['prmscore', 'within_auc'],
                                     ['Official PRMScore', 'Within-answer AUC'],
                                     [(65, 70.35), (.795, .861)]):
        values = [number(a, metric) for a in arms]
        ax.scatter(values, y, c=colors, s=100, edgecolors='white', linewidths=.8, zorder=3)
        for v, yy, c in zip(values, y, colors):
            ax.annotate(f'{v:.2f}' if metric == 'prmscore' else f'{v:.4f}',
                        (v, yy), xytext=(7, 0), textcoords='offset points',
                        va='center', fontsize=11, color=c, fontweight='bold')
        ax.set_title(title + '  |  higher is better', loc='left', pad=16)
        ax.set_xlim(*lim)
        ax.set_ylim(-.6, 4.6)
        ax.set_xlabel('PRMScore (points, 0–100)' if metric == 'prmscore' else 'AUC (0–1)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f' if metric == 'prmscore' else '%.2f'))
        style(ax)
    axes[0].set_yticks(y, labels)
    axes[0].set_ylabel('Method / control', labelpad=18)
    axes[1].tick_params(axis='y', labelleft=False)
    d = CONTRASTS['S3_HEAD_LSML - S3_HEAD_CLUSTER_EQUAL_L2', 'prmscore']
    a = CONTRASTS['S3_HEAD_LSML - S3_HEAD_CLUSTER_EQUAL_L2', 'within_auc']
    fig.text(.035, .185, 'Full L-SML minus the same-cluster, equal-L2 control:', fontsize=12, fontweight='bold')
    fig.text(.035, .137,
             f"PRMScore {float(d['delta']):+.3f} points  (95% CI {float(d['ci_low']):+.3f} to {float(d['ci_high']):+.3f})"
             f"     |     AUC {float(a['delta']):+.5f}  (95% CI {float(a['ci_low']):+.5f} to {float(a['ci_high']):+.5f})", fontsize=10)
    fig.text(.035, .062, 'Source: independent Drive step-data replay, 15 Oct 2026. PRMBench: 6,969 answers; official score on 6,211 noncontrols; AUC on 6,030 answers.\n'
             'Point estimates above; zoomed axes. Paired pointwise CIs: 707 source groups, 2,000 draws, fixed fits. Label-selected features; pooled label-free fitting.',
             fontsize=8.6, color=GRAY)
    save(fig, '01_prm7b_mechanism')


def expansion(metric='prmscore'):
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 7.2))
    fig.subplots_adjust(left=.075, right=.965, top=.73, bottom=.30, wspace=.23)
    title = 'Adding backbone features: does spectral weighting help?'
    fig.suptitle(title, x=.035, y=.96, ha='left', fontsize=19, fontweight='bold')
    fig.text(.035, .865, ('Official PRMScore under the same ANSWER_Z_Q80 rule' if metric == 'prmscore' else 'Within-answer AUC, independent of the decision threshold')+
             '\nThe x-axis counts backbone signals; the PRM head is one additional input in the right panel.', fontsize=11, va='top')
    styles = [('FLAT_MEAN', 'Flat mean', GRAY, 's', (0, (5, 3))),
              ('CLUSTER_EQUAL_L2', 'Same clusters + equal L2 weights', ORANGE, 'D', (0, (2, 2))),
              ('LSML', 'Full L-SML', BLUE, 'o', '-')]
    xs = np.array([3,5,6])
    for ax, part, title in zip(axes, ['ONLY', 'HEAD'], ['Backbone signals only', 'PRM head + backbone signals']):
        for suffix, label, color, marker, line in styles:
            values = [number(f'S{k}_{part}_{suffix}', metric) for k in [3,5,6]]
            ax.plot(xs, values, linestyle=line, color=color, marker=marker, ms=8,
                    lw=2.1 if suffix == 'LSML' else 1.7, label=label, zorder=3 if suffix=='LSML' else 2)
        # Keep full L-SML labels visually distinct from the two controls.
        for k, xx in zip([3,5,6], xs):
            v = number(f'S{k}_{part}_LSML', metric)
            ax.annotate(f'{v:.2f}' if metric == 'prmscore' else f'{v:.4f}', (xx,v),
                        xytext=(0,-19), textcoords='offset points', ha='center', color=BLUE, fontsize=10, fontweight='bold')
        if part == 'HEAD':
            base = number('PRM_Q80', metric)
            ax.axhline(base, color=LIGHTGRAY, lw=1.1, linestyle=':')
            ax.text(.03, base+.06 if metric=='prmscore' else base+.0007,
                    f'PRM head alone: {base:.2f}' if metric=='prmscore' else f'PRM head alone: {base:.4f}',
                    transform=ax.get_yaxis_transform(), color=GRAY, fontsize=9)
        if part == 'ONLY':
            y0 = number('S3_ONLY_LSML', metric)
            ax.annotate('Identical predictions here:\nL-SML = cluster/equal-L2 control',
                        xy=(3,y0), xytext=(.23,.14), textcoords='axes fraction',
                        fontsize=9, color=INK,
                        arrowprops=dict(arrowstyle='-',color=GRAY,lw=.9))
        ax.set_title(title, loc='left', pad=13, fontweight='bold')
        ax.set_xticks(xs, ['3','5','6'])
        ax.set_xlim(2.7,6.3)
        ax.set_ylim((65.4,70.05) if metric=='prmscore' else (.785,.862))
        ax.set_xlabel('Number of backbone signals')
        ax.set_ylabel('PRMScore (points, 0–100)' if metric=='prmscore' else 'Within-answer AUC (0–1)')
        ax.grid(axis='y',color='#e7ebef',lw=.8)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.225),ncol=3,frameon=False,fontsize=10)
    fig.text(.035,.105,'3 signals: energy_level, hist_entropy_cusum_abs_series, realized_drv.\n'
             '5 signals: add ct7_H0lim_prefix_innovation and digit_spread.  6 signals: also add hist_topk_varentropy_series.', fontsize=9.5)
    fig.text(.035,.035,'Source: independent Drive replay, 15 Oct 2026; same PRM-7B checkpoint and populations as the first graph. Zoomed axes; points are estimates, not CIs.\n'
             'Nested development candidates, chosen from previously inspected features. No untouched confirmation. All six L-SML fits selected two groups.',fontsize=8.7,color=GRAY)
    save(fig,'02_feature_expansion' if metric=='prmscore' else '03_feature_expansion_auc')


if __name__=='__main__':
    mechanism()
    expansion()
    expansion('within_auc')
    print(OUT)
