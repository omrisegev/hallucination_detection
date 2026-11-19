"""Self-consistent parent weighting: children of all parents, weighted by P(clean | parent score) from the current pi-hat;
weighted BBE; iterate. Same caricature as sim_bbe.py."""
import numpy as np, sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from sim_bbe import make, score, W, rng, auroc
from sklearn.isotonic import IsotonicRegression

def inject_pairs(ans, mu, lvl, reps=1):
    par, ch = [], []
    for r in range(reps):
        for i, (s, w) in enumerate(ans):
            s = s.copy(); k = rng.integers(0, len(s)); s[k] = rng.normal(mu, 1); s[k+1:] += lvl * (not w)
            par.append(i); ch.append((s, True))
    return np.array(par), ch

def wbbe(sp, wp, su, delta=.1, gamma=.01):
    o = np.argsort(-sp); sp, wp = sp[o], wp[o]; cum = np.cumsum(wp) / wp.sum()
    neff = wp.sum() ** 2 / (wp ** 2).sum(); nU = len(su); su_s = np.sort(su)
    best = (np.inf, 1.)
    for j in range(len(sp)):
        c = sp[j]; qP = cum[j]; qU = 1 - np.searchsorted(su_s, c, 'left') / nU
        eps = np.sqrt(np.log(4 / delta) / (2 * nU)) + np.sqrt(np.log(4 / delta) / (2 * neff))
        v = qU / qP + (1 + gamma) * eps / qP
        if v < best[0]: best = (v, min(1., qU / qP))
    return best[1]

def posterior(sp, wp, su, pi):
    x = np.r_[sp, su]; y = np.r_[np.ones(len(sp)), np.zeros(len(su))]
    w = np.r_[.5 * wp / wp.sum(), np.full(len(su), .5 / len(su))]
    g = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds='clip').fit(x, y, sample_weight=w).predict(su)
    return np.minimum(1., pi * g / (1 - g))

pi, mu, lvl, nA = .646, 2.2, .4, 850
out = {k: [] for k in ('unweighted', 'iter1', 'iter3', 'iter10', 'oracle weights')}
for rep in range(20):
    U = make(nA, pi, mu, lvl); ref = np.concatenate([s for s, _ in U]); sU = score(U, ref)
    y = np.array([w for _, w in U])
    par, ch = inject_pairs(U, mu, lvl, reps=3); sP = score(ch, ref)
    wp = np.ones(len(sP)); p = wbbe(sP, wp, sU); out['unweighted'].append(p)
    for it in range(1, 11):
        r = posterior(sP, wp, sU, p); wp = 1 - r[par] + 1e-6; p = wbbe(sP, wp, sU)
        if it in (1, 3, 10): out[f'iter{it}'].append(p)
    out['oracle weights'].append(wbbe(sP, (~y[par]).astype(float) + 1e-9, sU))
for k, v in out.items():
    v = np.array(v); print(f'{k:16s} mean {v.mean():.3f} sd {v.std():.3f}')
