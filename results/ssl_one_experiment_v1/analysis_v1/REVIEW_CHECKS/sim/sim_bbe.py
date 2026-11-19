"""Caricature of the answer gate: one step channel, answer score = soft top-M (weights .477/.289/.140/.058/.022) of pool ECDF.
Checks: (1) BBE with perfect positives vs n_pos; (2) realistic injection into all parents vs clean parents only;
(3) more positives (several injections per parent)."""
import numpy as np, sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from ssl_lib import bbe
W = np.array([.477, .289, .140, .058, .022])
rng = np.random.default_rng(1)

def make(nA, pi, mu, lvl):
    ans = []
    for i in range(nA):
        n = rng.integers(4, 17); s = rng.normal(0, 1, n); wrong = rng.random() < pi
        if wrong:
            t = rng.integers(0, n); s[t] = rng.normal(mu, 1); s[t+1:] += lvl
        ans.append((s, wrong))
    return ans

def inject(ans, mu, lvl, only=None):
    out = []
    for s, w in ans:
        if only is not None and w != only: continue
        s = s.copy(); n = len(s); k = rng.integers(0, n)
        s[k] = rng.normal(mu, 1); s[k+1:] += lvl * (not w)   # level shift only if this is the first error
        out.append((s, True))
    return out

def score(ans, ref):
    ref = np.sort(ref); out = []
    for s, _ in ans:
        e = np.sort(np.searchsorted(ref, s) / len(ref))[::-1]
        e = np.r_[e, np.zeros(5)][:5]; out.append(e @ W)
    return np.array(out)

def auroc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

if __name__ == '__main__':
    pi, mu, lvl, nA = .646, 2.2, .4, 850
    res = {}
    for rep in range(30):
        U = make(nA, pi, mu, lvl); ref = np.concatenate([s for s, _ in U])
        sU = score(U, ref); y = np.array([w for _, w in U])
        res.setdefault('auroc', []).append(auroc(y, sU))
        # perfect positives: fresh real wrong answers, n_pos small / large
        for npos in (75, 200, 2000):
            Wp = [a for a in make(int(npos / pi * 1.3), pi, mu, lvl) if a[1]][:npos]
            res.setdefault(f'perfect n_pos={npos}', []).append(bbe(score(Wp, ref), sU)[0])
        # realistic injection into all parents (1 child each, and 5 children each)
        res.setdefault('inject all parents x1', []).append(bbe(score(inject(U, mu, lvl), ref), sU)[0])
        res.setdefault('inject all parents x5', []).append(bbe(score(sum([inject(U, mu, lvl) for _ in range(5)], []), ref), sU)[0])
        res.setdefault('inject clean parents x1', []).append(bbe(score(inject(U, mu, lvl, only=False), ref), sU)[0])
        res.setdefault('inject clean parents x5', []).append(bbe(score(sum([inject(U, mu, lvl, only=False) for _ in range(5)], []), ref), sU)[0])
    for k, v in res.items():
        v = np.array(v); print(f'{k:28s} mean {v.mean():.3f}  sd {v.std():.3f}')
