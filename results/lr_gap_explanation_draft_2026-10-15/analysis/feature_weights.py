"""Per-channel weights: L-SML vs LR vs non-negative LR (NNLR), one bank (default N56). 2026-10-17.

LR and L-SML weights: mean over the 5 outer folds of l1_normalized_common_weight from
advisor_completion_2026-10-11/lr/WEIGHTS.csv (same standardized scale, |w| sums to 1).
NNLR: per outer fold, channels standardized and sign-oriented by training-fold mu, logistic loss
with w >= 0 (as in handoff_checks.py); weights L1-normalized and mapped back to V signs.
Single-channel AUC: pooled within-answer AUC of each V column (label-free signs), descriptive.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import rankdata

ROOT = Path('/Users/osegev/Desktop/hallucination_detection/results/advisor_completion_2026-10-11')
OUT = Path(__file__).resolve().parent
BANK = sys.argv[1] if len(sys.argv) > 1 else 'N56'

def within(score, off, y):
    a = []
    for b, e in zip(off[:-1], off[1:]):
        ss, yy = score[b:e], y[b:e]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 and n0: a.append((rankdata(ss)[yy.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(a))

def nnlr(Z, y, l2=1e-4):
    n, m = Z.shape
    def f(p):
        w, b = p[:m], p[m]; s = Z @ w + b
        g = 1 / (1 + np.exp(-s)) - y
        return (np.logaddexp(0, s).sum() - y @ s) / n + l2 * w @ w, np.r_[Z.T @ g / n + 2 * l2 * w, g.mean()]
    return minimize(f, np.zeros(m + 1), jac=True, method='L-BFGS-B',
                    bounds=[(0, None)] * m + [(None, None)], options={'maxiter': 2000}).x[:m]

rows = [r for r in csv.DictReader((ROOT / 'lr/WEIGHTS.csv').open()) if r['bank'] == BANK]
names = [r['channel'] for r in rows if r['method'] == 'LR' and r['outer_fold'] == '0']
fam = {r['channel']: r['family'] for r in rows}
W = {m: np.array([[float(r['l1_normalized_common_weight']) for r in rows if r['method'] == m and r['outer_fold'] == str(o)]
                  for o in range(5)]) for m in ('LR', 'LSML')}
for m in W:
    for o in range(5):
        assert [r['channel'] for r in rows if r['method'] == m and r['outer_fold'] == str(o)] == names

p = np.load(ROOT / 'inputs/population.npz'); off = p['offsets'].astype(int); y = p['y'].astype(int)
sf = np.repeat(p['fold'].astype(int), np.diff(off))
z = np.load(ROOT / f'inputs/bank_V_{BANK}.npz'); assert z['columns'].astype(str).tolist() == names
X = z['V'].astype(float)
NN = np.zeros((5, len(names)))
for o in range(5):
    tr = sf != o; mu0, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
    Z = (X[tr] - mu0) / sd
    sg = np.where(Z[y[tr] == 1].mean(0) - Z[y[tr] == 0].mean(0) >= 0, 1., -1.)
    w = nnlr(Z * sg, y[tr]) * sg
    NN[o] = w / np.abs(w).sum()
auc = np.array([within(X[:, j], off, y) for j in range(len(names))])

lr, ls, nn = W['LR'].mean(0), W['LSML'].mean(0), NN.mean(0)
kept = (np.abs(NN) > 1e-4).sum(0)          # folds in which NNLR gives the channel nonzero weight
table = [{'channel': n, 'family': fam[n], 'lsml_w': ls[j], 'lr_w': lr[j], 'diff_lr_minus_lsml': lr[j] - ls[j],
          'nnlr_w': nn[j], 'nnlr_kept_folds': int(kept[j]), 'lr_sign_flips_folds': int((np.sign(W['LR'][:, j]) != np.sign(lr[j])).sum()),
          'single_auc': auc[j]} for j, n in enumerate(names)]
with (OUT / f'feature_weights_{BANK}.csv').open('w', newline='') as fh:
    wr = csv.DictWriter(fh, list(table[0]), lineterminator='\n'); wr.writeheader()
    wr.writerows([{k: (round(v, 5) if isinstance(v, float) else v) for k, v in t.items()} for t in table])

# Figure: L-SML vs non-negative LR (the selection view), sorted by |NNLR| - |L-SML|.
order = np.argsort(np.abs(nn) - np.abs(ls))
famcol = {'F1': '#1766a1', 'F2': '#d4543c', 'F3': '#3b8c88', 'F4': '#8d99a5', 'F5': '#6f58a5', 'F6': '#c77b21', 'F7': '#203044'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#cbd5df', 'xtick.color': '#526173', 'ytick.color': '#526173', 'text.color': '#203044'})
n = len(names); yy = np.arange(n)
fig, axs = plt.subplots(1, 3, figsize=(15, 0.2 * n + 2.4), sharey=True, gridspec_kw={'width_ratios': [1.3, 1, .7]})
ax = axs[0]
ax.barh(yy + .2, ls[order], .4, color='#1766a1', label='L-SML (label-free)')
ax.barh(yy - .2, nn[order], .4, color='#c77b21', label='LR, weights ≥ 0 (supervised)')
ax.axvline(0, color='#526173', lw=.8); ax.set_title('Weight per channel (|w| sums to 1 per method)', loc='left', fontsize=10, pad=22)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, fontsize=8, frameon=False)
labels = [f'{fam[names[j]]}  {names[j]}' + ('  ✕' if kept[j] == 0 else '') for j in order]
ax.set_yticks(yy, labels, fontsize=7.2)
for t, j in zip(ax.get_yticklabels(), order): t.set_color(famcol[fam[names[j]]])
d = (np.abs(nn) - np.abs(ls))[order]
axs[1].barh(yy, d, .7, color=np.where(d >= 0, '#c77b21', '#1766a1'))
axs[1].axvline(0, color='#526173', lw=.8)
axs[1].set_title('|LR(≥0)| − |L-SML|  (orange: LR relies on it more)', loc='left', fontsize=10)
axs[2].barh(yy, auc[order] - .5, .7, left=.5, color='#8d99a5')
axs[2].axvline(.5, color='#526173', lw=.8); axs[2].set_xlim(.35, .8)
axs[2].set_title('Single-channel AUC (label-free sign)', loc='left', fontsize=10)
for a_ in axs: a_.set_ylim(-.8, n - .2); a_.grid(axis='x', alpha=.2)
fig.suptitle(f'{BANK}: which channels supervised LR keeps.  ✕ = weight 0 in all 5 folds (dropped)', x=.01, ha='left', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / f'feature_weights_{BANK}.png', dpi=130, bbox_inches='tight')

summary = {'bank': BANK, 'n_channels': n,
           'nnlr_kept_ge3': [names[j] for j in range(n) if kept[j] >= 3],
           'lr_negative': [names[j] for j in range(n) if lr[j] < -0.005],
           'lr_share_negative_mass': float(np.abs(lr[lr < 0]).sum()),
           'lsml_share_negative_mass': float(np.abs(ls[ls < 0]).sum()),
           'top_lr_minus_lsml': [(names[j], round(float((lr - ls)[j]), 4)) for j in np.argsort(-(lr - ls))[:8]],
           'top_lsml_minus_lr': [(names[j], round(float((ls - lr)[j]), 4)) for j in np.argsort(-(ls - lr))[:8]]}
(OUT / f'feature_weights_{BANK}.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
