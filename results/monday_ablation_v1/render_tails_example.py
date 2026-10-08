"""Illustration: binary tail votes on one PRMBench answer (raw step values -> top 20% steps vote 'error')."""
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True)
SE = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
F = np.load(MAIN / '.worktrees/ssl-pseudolabel-residual-v1/results/external_banks_v4/source/FEATURES.npz')
names = [str(x) for x in F['names']]; X = F['values']; off = F['offsets']
A = pd.read_csv(SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
with np.load(SE / 'OOF_STEP_SCORES.npz') as z: LAB = z['labels'].astype(bool)
fr = json.loads((SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
meta = {m['idx']: m for m in pickle.load(open(fr['prm_metadata']['path'], 'rb')).values()}
chans = [('q15_H1', 'Token entropy (top-15)'), ('chosen_surprisal', 'Surprise of the written token'), ('realized_drv', 'Rise in written-token surprise')]
pick = None
for i in np.flatnonzero(A.cell.to_numpy() == 'prmbench_qwen3_8b'):
    y = LAB[off[i]:off[i + 1]]; n = len(y)
    if n == 10 and y.sum() == 1 and 3 <= np.flatnonzero(y)[0] <= 7 and meta[A.id.iloc[i]]['classification'] == 'step_contradiction':
        hits = 0
        for c, _ in chans:
            v = X[off[i]:off[i + 1], names.index(c)]; top = np.argsort(-v, kind='stable')[:2]
            hits += int(np.flatnonzero(y)[0] in top)
        if hits >= 2: pick = i; break
i = pick; y = LAB[off[i]:off[i + 1]]; n = len(y); k = int(np.ceil(.2 * n)); e = int(np.flatnonzero(y)[0])
for mode, t in {'light': dict(bg='#fcfcfb', ink='#0b0b0b', ink2='#52514e', grid='#e1e0d9', bar='#c3c2b7', vote='#2a78d6', err='#eb6834'),
                'dark': dict(bg='#1a1a19', ink='#ffffff', ink2='#c3c2b7', grid='#2c2c2a', bar='#4a4a46', vote='#3987e5', err='#d95926')}.items():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (c, lab) in zip(axes, chans):
        v = X[off[i]:off[i + 1], names.index(c)]; top = np.argsort(-v, kind='stable')[:k]
        cols = [t['vote'] if j in top else t['bar'] for j in range(n)]
        ax.axvspan(e - .45, e + .45, color=t['err'], alpha=.18, linewidth=0)
        ax.bar(range(n), v, color=cols, width=.7)
        for j in top: ax.text(j, v[j], '+1', ha='center', va='bottom', fontsize=10, color=t['vote'], fontweight='bold')
        ax.set_xticks(range(n)); ax.set_xticklabels([str(j + 1) for j in range(n)], color=t['ink2'])
        ax.set_title(lab, color=t['ink'], fontsize=11); ax.set_facecolor(t['bg']); ax.tick_params(colors=t['ink2'])
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(t['bar'])
        ax.set_xlabel('step', color=t['ink2'])
    fig.suptitle(f'One PRMBench answer ({n} steps, error at step {e + 1}, shaded): each channel votes "+1 = error" on its top {k} steps (20%)', color=t['ink'], fontsize=12, x=.01, ha='left')
    fig.patch.set_facecolor(t['bg']); fig.tight_layout()
    for ext in ('png', 'svg'): fig.savefig(FIG / f'T0_tail_votes_example_{mode}.{ext}', dpi=180, facecolor=t['bg'], bbox_inches='tight')
    plt.close(fig)
print('answer', A.id.iloc[i], 'error step', e + 1)
