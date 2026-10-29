"""Independent T-LSML and T-CHAIN (own code; frozen package L-SML fit on an own-built answer-z N16 bank).
    .venv/bin/python -I results/answer_mixture_gate_v1/verification/verify_t2.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
REPO = Path('/Users/osegev/Desktop/hallucination_detection'); sys.path.insert(0, str(REPO))
from spectral_utils.fusion_utils import lsml_continuous_guarded as L
OUT = REPO / 'results/answer_mixture_gate_v1/verification'
B = Path('/private/tmp/claude-501/-Users-osegev-Desktop-hallucination-detection/98d49308-5ca3-48f7-a452-63da0bb7d2a1/scratchpad/tt/bank_rebuilt')
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']; SEED = 20261029
F, R = np.load(B / 'FEATURES.npz'), np.load(B / 'REWARDS.npz')
names16 = json.loads((REPO / 'results/monday_nested_banks_v2/BANKS.json').read_text())['N16']
nm = [str(x) for x in F['names']]; X = F['values'][:, [nm.index(c) for c in names16]].astype(float)
off = F['offsets'].astype(np.int64); uids = F['uids'].astype(str); nst = np.diff(off)
act = np.ones(X.shape, bool); da = F['digit_active'].astype(bool)
for j, c in enumerate(names16):
    if c in DIG: act[:, j] = da[:, DIG.index(c)]
def az(x):
    out = np.zeros_like(x)
    for a, b in zip(off[:-1], off[1:]):
        for j in range(x.shape[1]):
            m = act[a:b, j] if x.shape[1] == 16 else np.ones(b - a, bool); v = x[a:b, j][m]
            if len(v) and v.std() > 1e-12: blk = out[a:b, j]; blk[m] = (v - v.mean()) / v.std(); out[a:b, j] = blk
    return out
gz = lambda v: (v - v.mean()) / v.std()
r = np.clip(1 - R['rewards'].astype(float), 1e-6, 1 - 1e-6); prm = np.log(r / (1 - r))
V = az(X)
s, d = L(*V.T, groups=None); s = np.asarray(s); ft = np.corrcoef(s, V.mean(1))[0, 1] < 0
vg = np.asarray(d['virtual_classifiers'], float)
VC = np.column_stack([az(prm[:, None])[:, 0]] + [az(vg[:, [g]])[:, 0] for g in range(vg.shape[1])])
c, dc = L(*VC.T, groups=np.arange(VC.shape[1])); c = np.asarray(c); fc = np.corrcoef(c, VC[:, 0])[0, 1] < 0
Xg = np.zeros_like(X)
for j in range(16): m = act[:, j]; Xg[m, j] = gz(X[m, j])
cw = np.asarray(d['cross_weights'], float); w = np.zeros(16)
for g, (idx, wg) in enumerate(d['group_weights']): w[np.asarray(idx, int)] = cw[g] * np.asarray(wg, float)
assert np.max(np.abs((-1 if ft else 1) * (V @ w) - (-s if ft else s))) < 1e-8      # TELE is linear in V with these weights
steps = {'T-LSML': (-1 if ft else 1) * (Xg @ w)}
grp = [Xg[:, np.asarray(idx, int)] @ np.asarray(wg, float) for idx, wg in d['group_weights']]
VCg = np.column_stack([gz(prm)] + [gz(g) for g in grp])
cwc = np.asarray(dc['cross_weights'], float)
assert np.max(np.abs((-1 if fc else 1) * (VC @ cwc) - (-c if fc else c))) < 1e-8
steps['T-CHAIN'] = (-1 if fc else 1) * (VCg @ cwc)
T = np.load(REPO / 'results/answer_mixture_gate_v1/trackT/PB_QWEN_ANSWER_TOKENS.npz'); tk = dict(zip(T['uids'].astype(str), T['answer_tokens'].astype(float)))
A = np.column_stack([np.ones(len(nst)), np.log(nst), np.log([tk[u] for u in uids])])
G = {g['uid']: g for g in json.loads((REPO / 'results/prm_boost_processbench_v1/local/processbench_gold.json').read_text())}
ref = {r['score']: r for r in csv.DictReader(open(REPO / 'results/answer_mixture_gate_v1/trackT/TRACK_T.csv')) if r['primary'] == 'True'}
rep = {'tele_flip': bool(ft), 'chain_flip': bool(fc), 'chain_cross_weights': cwc.tolist()}
for arm, st in steps.items():
    s_a = np.array([st[a:b].max() for a, b in zip(off[:-1], off[1:])]); res = s_a - A @ np.linalg.solve(A.T @ A, A.T @ s_a)
    x = res[:, None]; g2 = GaussianMixture(2, covariance_type='full', n_init=10, random_state=SEED).fit(x)
    g1 = GaussianMixture(1, covariance_type='full', n_init=10, random_state=SEED).fit(x)
    k = int(np.argmax(g2.means_.ravel())); clean = g2.predict_proba(x)[:, k] < .5
    y = np.array([int(G[u]['label']) != -1 for u in uids])
    o = {'pi_hat': float(g2.weights_[k]), 'dbic': float(g2.bic(x) - g1.bic(x)), 'clean_closed': float(clean[~y].mean()),
         'err_kept': float((~clean[y]).mean()), 'auroc': float(roc_auc_score(y, s_a))}
    rf = ref[arm]; o['ref'] = {'pi_hat': float(rf['pi_hat']), 'dbic': float(rf['bic2_minus_bic1']), 'clean_closed': float(rf['clean_closed']),
                               'err_kept': float(rf['err_kept']), 'auroc': float(rf['auroc_s_a'])}
    o['abs_diff'] = {k2: abs(o[k2] - o['ref'][k2]) for k2 in o['ref']}; rep[arm] = o
    print(arm, json.dumps(o))
(OUT / 'verify_t2_summary.json').write_text(json.dumps(rep, indent=1))
