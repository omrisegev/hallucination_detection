"""Figures F1-F4 of the one SSL experiment (ONE_EXPERIMENT.md section 4). Reads RESULTS.json + SCORES.npz.

    python figures.py <OUT folder of run_analysis.py>
F1_<bench>.png  histograms (test pool): fused step score and answer score; real clean / real wrong / synthetic wrong
F2_<bench>.png  error-detection plane with iso-metric lines (PRMBench: step TPR/FPR, iso-PRMScore; ProcessBench:
                answer-level error accuracy vs 100 - clean accuracy, iso-F1); points = marking arms (test)
F3_pi.png       pi-hat vs true pi (dev and test, both positives)
F4_cond_corr.png conditional mean |corr| before / after stages S and A (dev, test)
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(sys.argv[1]); R = json.load(open(OUT / 'RESULTS.json')); SC = np.load(OUT / 'SCORES.npz')
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3, S4, S5 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e87ba4'
NEUTRAL = '#8a8984'
NAME = {'processbench': 'ProcessBench', 'prmbench': 'PRMBench'}
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'text.color': INK,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
                     'axes.titleweight': 'bold', 'legend.frameon': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': .6})


def hist_lines(ax, series, bins=50):
    allv = np.concatenate([v for _, v, _ in series if len(v)])
    lo, hi = np.percentile(allv, [.5, 99.5]); e = np.linspace(lo, hi, bins + 1)
    for name, v, col in series:
        if not len(v): continue
        h, _ = np.histogram(np.clip(v, lo, hi), e, density=True)
        ax.step(e[:-1], h, where='post', color=col, lw=2, label=f'{name} (n={len(v):,})')
    ax.set_ylabel('density'); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=1, fontsize=8.5)


def f1(bench):
    k = f'{bench}_test'
    if f'{k}__s' not in SC.files: return
    s, y, sA, ya = SC[f'{k}__s'], SC[f'{k}__y_step'], SC[f'{k}__sA'], SC[f'{k}__y_ans'].astype(bool)
    pos = R[bench].get('gate_positive') or 'synthetic'
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    hist_lines(ax[0], [('real clean steps', s[y == -1], S1), ('real wrong steps', s[y == 1], S2),
                       ('synthetic wrong step (V2 step k)', SC[f'{k}__s_pos_k'] if f'{k}__s_pos_k' in SC.files else np.zeros(0), S3)])
    ax[0].set_xlabel('fused step score s (L-SML on R0; higher = more risk)'); ax[0].set_title('Step score, test pool')
    hist_lines(ax[1], [('real clean answers', sA[~ya], S1), ('real wrong answers', sA[ya], S2),
                       (f'synthetic wrong answers ({pos})', SC[f'{k}__sA_pos'] if f'{k}__sA_pos' in SC.files else np.zeros(0), S3)])
    ax[1].set_xlabel('answer score (answer L-SML; higher = more risk)'); ax[1].set_title('Answer score, test pool')
    fig.suptitle(f'F1 {NAME[bench]}: score histograms (labels used for the colors only)', x=.01, ha='left', fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f'F1_{bench}.png', dpi=160); plt.close(fig)


ARMS = [('method of record (top10, no gate)', 'method of record', S1, 'o'),
        ('final + gate (pi-hat)', 'final + gate (pi-hat)', S2, 's'),
        ('final + change point (prior 1 - P)', 'final + change point', S4, 'D'),
        ('final + Simes gate alpha .05', 'Simes gate .05', NEUTRAL, 'v'),
        ('final + label-perfect gate (oracle)', 'label-perfect gate (oracle)', S3, '*')]


def f2(bench):
    M = R[bench].get('test', {}).get('marking')
    if not M: return
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    g = np.linspace(0, 100, 401); X, Y = np.meshgrid(g, g)
    if bench == 'prmbench':
        pe = M['method of record (top10, no gate)']['pi_step']
        E, V = pe, 1 - pe; tn = Y / 100 * E; fp = E - tn; fn = X / 100 * V; tp = V - fn
        Z = 50 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))
        xk, yk, lab = 'FPR_err', 'TPR_err', 'PRMScore'
        ax.set_xlabel('valid steps flagged (%)'); ax.set_ylabel('error steps flagged (%)')
        levels = np.arange(40, 100, 5)
    else:
        Z = np.where(X + Y > 0, 0, 0.); ca = 100 - X; ea = Y
        Z = np.where(ea + ca > 0, 2 * ea * ca / np.maximum(ea + ca, 1e-9), 0)
        xk, yk, lab = 'clean_acc', 'err_acc', 'F1'
        ax.set_xlabel('clean answers with a flag (%) = 100 - clean accuracy'); ax.set_ylabel('error answers: exact first error (%)')
        levels = np.arange(10, 100, 10)
    cs = ax.contour(X, Y, Z, levels=levels, colors=GRID, linewidths=1)
    ax.clabel(cs, fmt=lambda v: f'{lab} {v:.0f}', fontsize=7.5, colors=INK2)
    for key, name, col, mk in ARMS:
        if key not in M: continue
        p = M[key]; x = p[xk] if bench == 'prmbench' else 100 - p['clean_acc']; y = p[yk]
        ax.scatter([x], [y], s=70 if mk != '*' else 160, color=col, marker=mk, edgecolor=SURF, linewidth=1.5, zorder=3,
                   label=f"{name}: {lab} {p['PRMScore' if bench == 'prmbench' else 'F1']:.1f}")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.legend(loc='lower right', fontsize=8.5)
    ax.set_title(f'F2 {NAME[bench]}: error detection plane, test pool', loc='left')
    if bench == 'processbench':
        ax.text(1, 98, 'iso lines: harmonic mean of the two accuracies\n(reported avg F1 = mean over 4 subsets)', fontsize=7.5, color=INK2, va='top')
    fig.tight_layout(); fig.savefig(OUT / f'F2_{bench}.png', dpi=160); plt.close(fig)


def f3():
    rows = []
    for b in ('processbench', 'prmbench'):
        if b not in R: continue
        Gd, T = R[b].get('stage_G_dev', {}), R[b].get('test', {})
        if not Gd.get('run'): continue
        pos = R[b]['gate_positive']; oth = Gd['pi_hat_other_positive']['positive']
        rows.append((f'{NAME[b]} dev', Gd['pi_true'], [(f'BBE, {pos} positive (used)', Gd['pi_hat'], S2), (f'BBE, {oth} positive (check)', Gd['pi_hat_other_positive']['pi_hat'], S4)]))
        if 'pi_hat' in T:
            rows.append((f'{NAME[b]} test', T['pi_true'], [(f'BBE, {pos} positive (used)', T['pi_hat'], S2), (f'BBE, {oth} positive (check)', T['pi_hat_other_positive']['pi_hat'], S4)]))
    if not rows: return
    fig, ax = plt.subplots(figsize=(7.5, .8 + .7 * len(rows)))
    seen = set()
    for i, (name, true, est) in enumerate(rows):
        yy = len(rows) - 1 - i
        ax.plot([true], [yy], marker='|', ms=22, mew=2.5, color=INK, label='true pi (labels)' if 'true' not in seen else None); seen.add('true')
        for j, (en, v, col) in enumerate(est):
            lab = en.split(',')[1].strip() if en not in seen else None
            ax.scatter([v], [yy + (j - .5) * .22], s=60, color=col, edgecolor=SURF, zorder=3, label=en if en not in seen else None); seen.add(en)
        ax.axvspan(true - .10, true + .10, ymin=(yy + .05) / len(rows), ymax=(yy + .95) / len(rows), color=GRID, alpha=.6, lw=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1]); ax.set_xlim(0, 1.02)
    ax.set_xlabel('share of wrong answers in the pool (gray band: true pi +- .10)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.45 if len(rows) < 4 else -.25), ncol=3, fontsize=8.5)
    ax.set_title('F3: pi-hat vs true pi', loc='left'); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(OUT / 'F3_pi.png', dpi=160); plt.close(fig)


def f4():
    items = []
    for b in ('processbench', 'prmbench'):
        if b not in R: continue
        r = R[b]
        if 'dev_top10' not in r: continue      # P0-only run (no dev stages)
        items.append((f'{NAME[b]} step, dev', r['dev_top10']['cond'], r['dev_final_readout']['cond'], r['dev_top10'].get('cond_ci'), r['dev_final_readout'].get('cond_ci')))
        t = r.get('test', {}).get('cond_step')
        if t: items.append((f'{NAME[b]} step, test', t['top10']['point'], t['final']['point'], t['top10']['ci'], t['final']['ci']))
        items.append((f'{NAME[b]} answer, dev', r['dev_answer_ref']['cond'], r['dev_answer_final']['cond'], None, None))
        ta = r.get('test', {}).get('cond_answer')
        if ta: items.append((f'{NAME[b]} answer, test', ta['ref'], ta['final'], None, None))
    if not items: return
    fig, ax = plt.subplots(figsize=(7.5, .9 + .5 * len(items)))
    for i, (name, a, b_, ca, cb) in enumerate(items):
        y = len(items) - 1 - i
        ax.plot([a, b_], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([a], [y], s=60, color=S1, zorder=3, edgecolor=SURF, label='before (top10 readout / M = 1)' if i == 0 else None)
        ax.scatter([b_], [y], s=60, color=S2, zorder=3, edgecolor=SURF, label='after (stages kept on dev)' if i == 0 else None)
        for v, c, col in ((a, ca, S1), (b_, cb, S2)):
            if c: ax.plot(c, [y, y], color=col, lw=1.2, alpha=.7, zorder=2)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items][::-1])
    ax.set_xlabel('conditional mean |Spearman| between channels (lower = more independent)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14 - .2 / max(len(items), 1) * 4), ncol=2, fontsize=8.5)
    ax.set_title('F4: conditional dependence before / after calibration', loc='left'); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(OUT / 'F4_cond_corr.png', dpi=160); plt.close(fig)


for b in ('processbench', 'prmbench'):
    if b in R: f1(b); f2(b)
f3(); f4()
print('figures written to', OUT)
