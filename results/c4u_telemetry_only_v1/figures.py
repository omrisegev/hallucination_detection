"""RESULTS.csv -> fig_prmbench.png, fig_processbench.png (grouped bars, 4 PRMs x 2 telemetry sources)."""
import csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R = list(csv.DictReader(open('RESULTS.csv')))
P = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']; PL = ['Qwen-PRM-7B', 'Qwen-PRM-72B', 'PURE', 'RLHFlow']
ARMS = [('PRM native', 'PRM alone', '#9aa0a6'), ('C4U (PRM + telemetry)', 'C4U full (PRM + telemetry)', '#2a78d6'),
        ('C4U-T (telemetry only)', 'C4U telemetry only', '#eb6834')]
g = lambda m, p, b, a, met: next(float(r['value']) for r in R if r['mode'] == m and r['prm'] == p and r['benchmark'] == b and r['arm'] == a and r['metric'] == met)


def fig(bench, metrics, out, title):
    fig, ax = plt.subplots(len(metrics), 2, figsize=(12, 3.6 * len(metrics)), squeeze=False)
    for i, (met, ylab, ylim) in enumerate(metrics):
        for j, (m, mt) in enumerate((('sa', 'Telemetry from the standalone base model'), ('prm', 'Telemetry from the PRM backbone'))):
            a = ax[i, j]; x = np.arange(4); w = .26
            for k, (arm, lab, col) in enumerate(ARMS):
                v = [g(m, p, bench, arm, met) for p in P]
                b = a.bar(x + (k - 1) * w, v, w, label=lab, color=col)
                a.bar_label(b, fmt='%.3f' if met == 'within-AUC' else '%.1f', fontsize=7, padding=1)
            a.set_xticks(x, PL); a.set_ylabel(ylab); a.set_ylim(*ylim); a.set_title(mt, fontsize=10)
            a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', alpha=.25)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(.5, .955))
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(out, dpi=200); plt.close(fig)


fig('prmbench', [('PRMScore', 'PRMScore (own flags)', (50, 76)), ('within-AUC', 'within-answer AUC', (.6, .88))],
    'fig_prmbench.png', 'PRMBench: C4U without any PRM input vs full C4U and the PRM alone')
fig('processbench', [('macro SLA', 'macro SLA (first-error exact match, %)', (0, 80)), ('average F1', 'PRISM T6 average F1', (0, 85))],
    'fig_processbench.png', 'ProcessBench: C4U without any PRM input vs full C4U and the PRM alone')
