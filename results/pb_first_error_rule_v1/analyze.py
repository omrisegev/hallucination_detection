"""Paired answer bootstrap (1000, within subset) of key differences + tables + PNG charts from PREDS.npz/RESULTS.csv."""
import csv, sys
import numpy as np
sys.path.insert(0, '.')
import pbeval as V
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']; LBL = dict(zip(PRMS, ['Qwen-7B', 'Qwen-72B', 'PURE', 'RLHFlow']))
Z = np.load('PREDS.npz'); ids = np.load('cells/PB_qwen_prm_7b.npz')['ids']
lab = V.labels(ids); sub = np.array([V.GOLD[u]['subset'] for u in ids]); rng = np.random.default_rng(0)
MTG = 39.39


def f1_sla(pred, idx):
    f, s = [], []
    for ss in V.SUBSETS:
        m = idx[sub[idx] == ss]; e = m[lab[m] >= 0]; c = m[lab[m] < 0]
        ea = 100 * np.mean(pred[e] == lab[e]); ca = 100 * np.mean(pred[c] == -1)
        f.append(2 * ea * ca / (ea + ca) if ea + ca else 0.); s.append(ea)
    return np.mean(f), np.mean(s)


def boot(pa, pb):
    base = np.arange(len(ids)); groups = [base[sub == s] for s in V.SUBSETS]
    pt = np.subtract(f1_sla(pa, base), f1_sla(pb, base)); D = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        D.append(np.subtract(f1_sla(pa, idx), f1_sla(pb, idx)))
    D = np.array(D); return pt, np.percentile(D, 2.5, 0), np.percentile(D, 97.5, 0)


def ci(pa, metric):
    base = np.arange(len(ids)); groups = [base[sub == s] for s in V.SUBSETS]; D = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups]); D.append(f1_sla(pa, idx)[metric])
    return np.percentile(D, [2.5, 97.5])


NEW = 'CP PRM + C4U rank (anchored)'
COMP = [(NEW, 'REF C4U NC (PRM gate+count)', ''), (NEW, 'REF PRM native (p>.5)', ''),
        (NEW, 'REF C4U NC (PRM gate+count)', 'LOC|'), (NEW, 'REF PRM native (p>.5)', 'LOC|'),
        ('CP PRM + TELE16 (anchored)', 'REF PRM native (p>.5)', ''), ('CP PRM + drop (anchored)', 'REF PRM native (p>.5)', '')]
out = []
for prm in PRMS:
    for a, b, loc in COMP:
        pt, lo, hi = boot(Z[f'{loc}{prm}|{a}'], Z[f'{loc}{prm}|{b}'])
        k = 1 if loc else 0      # LOC: only SLA is meaningful (F1 of forced predictions is not)
        out.append(dict(prm=prm, a=a, b=b, metric='macro SLA (forced)' if loc else 'avg F1', diff=pt[k], lo=lo[k], hi=hi[k]))
        if not loc:
            out.append(dict(prm=prm, a=a, b=b, metric='macro SLA', diff=pt[1], lo=lo[1], hi=hi[1]))
    for arm in ('FREE CP TELE16 standalone (EM)', 'FREE CP TELE16 backbone (EM, pi=1/2)'):
        p = Z[f'LOC|{prm}|{arm}']; v = f1_sla(p, np.arange(len(ids)))[1]; l, h = ci(p, 1)
        out.append(dict(prm=prm, a=arm, b='Mind the Gap Qwen3-8B (39.39, published)', metric='macro SLA (forced)', diff=v - MTG, lo=l - MTG, hi=h - MTG))
with open('BOOTSTRAP.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

R = {(r['prm'], r['arm']): r for r in csv.DictReader(open('RESULTS.csv'))}
g = lambda prm, arm, k: float(R[(prm, arm)][k])

# ---- charts
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
C = {'PRM': '#7f7f7f', 'NC': '#d62728', 'CP': '#1f77b4', 'TELE': '#2ca02c', 'MTG': '#ff7f0e'}
x = np.arange(4); W = .26
fig, ax = plt.subplots(figsize=(8, 4))
for i, (arm, c, lb) in enumerate([('REF PRM native (p>.5)', C['PRM'], 'PRM alone (p>0.5)'),
                                  ('REF C4U NC (PRM gate+count)', C['NC'], 'C4U + current rule (NC)'),
                                  (NEW, C['CP'], 'C4U + change-point rule (new)')]):
    v = [g(p, arm, 'avg_F1') for p in PRMS]; b = ax.bar(x + (i - 1) * W, v, W, color=c, label=lb)
    ax.bar_label(b, fmt='%.1f', fontsize=8)
ax.set_xticks(x, [LBL[p] for p in PRMS]); ax.set_ylabel('ProcessBench average F1'); ax.set_ylim(0, 90)
ax.set_title('ProcessBench F1: change-point first-error rule vs current C4U rule'); ax.legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig('charts/01_pb_f1.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
for i, (arm, c, lb) in enumerate([('REF PRM native (p>.5)', C['PRM'], 'PRM alone (forced)'),
                                  ('REF C4U NC (PRM gate+count)', C['NC'], 'C4U + NC (forced)'),
                                  (NEW, C['CP'], 'C4U + change-point (argmax)')]):
    v = [g(p, arm, 'macro_SLA_loc') for p in PRMS]; b = ax.bar(x + (i - 1) * W, v, W, color=c, label=lb)
    ax.bar_label(b, fmt='%.1f', fontsize=8)
ax.axhline(MTG, color=C['MTG'], ls='--', lw=1.2, label='Mind the Gap Qwen3-8B (39.39)')
ax.set_xticks(x, [LBL[p] for p in PRMS]); ax.set_ylabel('macro SLA (error answers)'); ax.set_ylim(0, 85)
ax.set_title('Step localization accuracy, Mind the Gap protocol'); ax.legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig('charts/02_pb_sla_backbone.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
prev = {'qwen_prm_7b': 38.47, 'qwen_prm_72b': 38.10, 'pure_prm': 31.29, 'rlhflow_prm': 34.29}   # 2026-11-02 standalone TELE16 + GAP rule
for i, (v, c, lb) in enumerate([([prev[p] for p in PRMS], '#9467bd', 'standalone TELE16 + GAP rule (2026-11-02)'),
                                ([g(p, 'FREE CP TELE16 standalone (EM)', 'macro_SLA_loc') for p in PRMS], C['TELE'], 'standalone TELE16 + change-point (new)')]):
    b = ax.bar(x + (i - .5) * W, v, W, color=c, label=lb); ax.bar_label(b, fmt='%.1f', fontsize=8)
ax.axhline(MTG, color=C['MTG'], ls='--', lw=1.2, label='Mind the Gap Qwen3-8B (39.39)')
ax.set_xticks(x, [LBL[p] + '\n(its base model)' for p in PRMS]); ax.set_ylabel('macro SLA (error answers)'); ax.set_ylim(0, 55)
ax.set_title('No PRM at all: base-model telemetry only'); ax.legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig('charts/03_pb_sla_standalone.png', dpi=150); plt.close(fig)

# error type breakdown for the new rule vs NC (error answers): exact / early / late / miss, averaged over subsets
fig, axs = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
for ax, prm in zip(axs, PRMS):
    for j, (arm, lb) in enumerate([('REF C4U NC (PRM gate+count)', 'C4U+NC'), (NEW, 'C4U+CP')]):
        r = R[(prm, arm)]; vals = [np.mean([float(r[f'{s}_{k}']) for s in V.SUBSETS]) for k in ('TPR', 'early', 'late', 'miss')]
        left = 0
        for v, c, k in zip(vals, ['#2ca02c', '#ff7f0e', '#9467bd', '#bbbbbb'], ['exact', 'too early', 'too late', 'said no error']):
            ax.barh(j, v, left=left, color=c, label=k if (prm == PRMS[0] and j == 0) else None); left += v
    ax.set_yticks([0, 1], ['C4U+NC', 'C4U+CP']); ax.set_title(LBL[prm]); ax.set_xlim(0, 100); ax.set_xlabel('% of error answers')
fig.legend(loc='lower center', ncol=4, frameon=False, fontsize=8); fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig('charts/04_error_types.png', dpi=150); plt.close(fig)
for r in out: print(r['prm'], r['a'][:28], 'vs', r['b'][:24], r['metric'], '%.2f [%.2f, %.2f]' % (r['diff'], r['lo'], r['hi']))
