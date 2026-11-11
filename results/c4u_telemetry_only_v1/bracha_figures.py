"""BRACHA_PANEL.csv + chart values -> fig_bracha_bars.png, fig_bracha_tpr_fpr.png."""
import csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bracha_panel_values import CHART
R = list(csv.DictReader(open('BRACHA_PANEL.csv')))
P = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']; PL = ['Qwen-PRM-7B', 'Qwen-PRM-72B', 'PURE-PRM-7B', 'RLHFlow-PRM-8B']
NEW = 'C4U-T standalone (no PRM)'
ARMS = [('PRM', 'PRM alone', '#9aa0a6'), ('OCT22', 'Oct 22 fusion (PRM + telemetry)', '#a9c6ea'),
        ('LR', 'LR trained on PRM800K labels', '#7b5ea7'), ('GSP', 'GSP (PRM + backbone telemetry)', '#2a78d6'),
        ('NEW', 'GSP without PRM (standalone telemetry)', '#eb6834'), ('TELE', 'TELE16 without PRM (standalone, GAP)', '#f2b134')]
TELE = 'TELE16 standalone (no PRM)'
val = lambda b, p, arm=NEW: float(next(r['value'] for r in R if r['benchmark'] == b and r['prm'] == p and r['arm'] == arm))

fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
for a, (b, t, yl) in zip(ax, (('prmbench', 'PRMBench: PRMScore', (40, 80)), ('socratic', 'Socratic-PRMBench: PRMScore', (20, 80)),
                               ('hard2verify', 'Hard2Verify: balanced F1', (0, 60)))):
    x = np.arange(4); w = .14
    for k, (key, lab, col) in enumerate(ARMS):
        v = [val(b, p) for p in P] if key == 'NEW' else [val(b, p, TELE) for p in P] if key == 'TELE' else CHART[b][key]
        bb = a.bar(x + (k - 2.5) * w, v, w, color=col, label=lab, edgecolor='#222' if key in ('NEW', 'TELE') else 'none', lw=.8)
        a.bar_label(bb, fmt='%.1f', fontsize=6.5, padding=1, rotation=90)
    a.set_xticks(x, PL, fontsize=8); a.set_ylim(*yl); a.set_title(t, fontsize=10)
    a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', alpha=.25)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(.5, .97))
fig.suptitle('Same per-answer label-free rule for every arm except the LR (trained on PRM800K labels)', fontsize=11, y=.995)
fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig('fig_bracha_bars.png', dpi=200); plt.close(fig)

# ---- PRMBench TPR vs FPR (positive = correct step; non-control rows)
fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True, sharey=True)
sty = {'PRM': ('o', '#9aa0a6', 'PRM alone'), 'OCT22': ('s', '#a9c6ea', 'Oct 22 fusion'), 'GSP': ('D', '#2a78d6', 'GSP (PRM + backbone)'),
       NEW: ('*', '#eb6834', 'GSP without PRM (standalone)'), TELE: ('^', '#f2b134', 'TELE16 without PRM (standalone, GAP)')}
for a, p, pl in zip(ax, P, PL):
    for r in R:
        if r['prm'] != p or r['benchmark'] != 'prmbench': continue
        m, col, lab = sty[r['arm']]
        a.scatter(float(r['FPR']), float(r['TPR']), marker=m, s=160 if m == '*' else 60, color=col, edgecolor='#222', lw=.6, label=lab, zorder=3)
    if p == 'qwen_prm_7b':
        a.scatter(47.2, 87.3, marker='P', s=80, color='#1baf7a', edgecolor='#222', lw=.6, label='PRISM (published)', zorder=3)
    a.set_title(pl, fontsize=10); a.set_xlabel('FPR % (erroneous steps called correct)'); a.grid(alpha=.25)
    a.spines[['top', 'right']].set_visible(False)
ax[0].set_ylabel('TPR % (correct steps called correct)')
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(.5, 1.0))
fig.suptitle('PRMBench operating points (up and left is better). The LR points are not on this machine.', fontsize=10, y=.9)
fig.tight_layout(rect=(0, 0, 1, .86)); fig.savefig('fig_bracha_tpr_fpr.png', dpi=200); plt.close(fig)
