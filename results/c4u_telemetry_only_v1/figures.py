"""CI.csv -> fig_prmscore.png, fig_processbench_sla.png, fig_processbench_f1.png, fig_gap.png.
Bars = point, whiskers = 95% paired answer-bootstrap CI (boot.py). Dashed lines = published competitor points."""
import csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R = list(csv.DictReader(open('CI.csv')))
P = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']; PL = ['Qwen-PRM-7B', 'Qwen-PRM-72B', 'PURE', 'RLHFlow']
ARMS = [('PRM native', 'PRM alone', '#9aa0a6'), ('C4U (PRM + telemetry)', 'C4U full (PRM + telemetry)', '#2a78d6'),
        ('C4U-T (telemetry only)', 'C4U telemetry only (no PRM)', '#eb6834')]
MODES = (('sa', 'Telemetry from the standalone base model'), ('prm', 'Telemetry from the PRM backbone (reward not used)'))


def get(m, p, b, met, a):
    r = next(r for r in R if r['mode'] == m and r['prm'] == p and r['bench'] == b and r['metric'] == met and r['arm'] == a)
    return float(r['point']), float(r['lo']), float(r['hi'])


def bars(bench, met, ylab, ylim, refs, out, title, fmt='%.1f'):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a, (m, mt) in zip(ax, MODES):
        x = np.arange(4); w = .27
        for k, (arm, lab, col) in enumerate(ARMS):
            v = np.array([get(m, p, bench, met, arm) for p in P])
            b = a.bar(x + (k - 1) * w, v[:, 0], w, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=2.5,
                      color=col, label=lab, error_kw=dict(lw=.8, ecolor='#333'))
            for xi, (pt, lo, hi) in zip(x + (k - 1) * w, v):
                a.text(xi, hi + (ylim[1] - ylim[0]) * .012, fmt % pt, ha='center', va='bottom', fontsize=7.5)
        for y, lab, ls in refs:
            a.axhline(y, color='#444', ls=ls, lw=1, label=lab)
        a.set_xticks(x, PL); a.set_ylim(*ylim); a.set_title(mt, fontsize=10)
        a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', alpha=.25)
    ax[0].set_ylabel(ylab)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(.5, .93))
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=(0, 0, 1, .88)); fig.savefig(out, dpi=200); plt.close(fig)


bars('prmbench', 'PRMScore', 'PRMScore (own flags)', (50, 77), [(68.00, 'PRISM (retrained 7B) 68.0', '--')],
     'fig_prmscore.png', 'PRMBench PRMScore, 95% CI (2,000 paired bootstrap draws over source questions)')
bars('processbench', 'macro SLA', 'macro SLA (%): first flagged step = first error', (0, 80),
     [(39.39, 'Mind the Gap (Qwen3-8B, no PRM) 39.4', '--')],
     'fig_processbench_sla.png', 'ProcessBench first-error exact match (macro SLA), 95% CI')
bars('processbench', 'average F1', 'PRISM Table 6 average F1', (0, 88), [(77.0, 'PRISM CL1 (published) 77.0', '--')],
     'fig_processbench_f1.png', 'ProcessBench F1 (error-answer and correct-answer accuracy), 95% CI')

# ---- gap of telemetry-only vs full C4U, with CI (0 inside the CI = statistical tie)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6)); TXT = []
for a, (b, met, t) in zip(ax, (('prmbench', 'PRMScore', 'PRMBench PRMScore'), ('processbench', 'macro SLA', 'ProcessBench macro SLA'),
                               ('processbench', 'average F1', 'ProcessBench average F1'))):
    y = np.arange(8); lab = []; i = 0
    for m, mt in MODES:
        for p, pl in zip(P, PL):
            pt, lo, hi = get(m, p, b, met, 'C4U-T (telemetry only) minus C4U (PRM + telemetry)')
            col = '#1baf7a' if lo > 0 else ('#d03b3b' if hi < 0 else '#9aa0a6')
            a.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt='o', color=col, capsize=3)
            TXT.append((a, i, f'{pt:+.1f} [{lo:+.1f}, {hi:+.1f}]'))
            lab.append(f"{pl} ({'standalone' if m == 'sa' else 'backbone'})"); i += 1
    a.axvline(0, color='#333', lw=.8); a.set_yticks(y, lab, fontsize=8); a.invert_yaxis(); a.set_title(t, fontsize=10)
    a.spines[['top', 'right']].set_visible(False); a.set_xlabel('telemetry-only C4U minus full C4U (points)')
    xl = a.get_xlim(); a.set_xlim(xl[0], xl[1] + (xl[1] - xl[0]) * .7)
    for aa, ii, tt in TXT:
        if aa is a: a.text(xl[1] + (xl[1] - xl[0]) * .05, ii, tt, va='center', fontsize=7.5)
fig.suptitle('How much does removing the PRM cost C4U? red = loss outside the CI, green = gain, grey = tie', fontsize=11)
fig.tight_layout(); fig.savefig('fig_gap.png', dpi=200); plt.close(fig)
