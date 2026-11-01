import sys, time, numpy as np
sys.path.insert(0, '/Users/osegev/Desktop/hallucination_detection')
from spectral_utils import markov_ds_count as M
sys.path.insert(0, '/Users/osegev/Desktop/hallucination_detection/results/prm_boost_collection_v1')

def answer_z(s, off):
    out = np.zeros(len(s))
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; out[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return out

def gap_counts(z, off):
    k = []
    for a, b in zip(off[:-1], off[1:]):
        m = b - a
        if m == 1: k.append(1); continue
        zz = np.sort(z[a:b])[::-1]; km = min(int(np.ceil(m / 2)), m - 1)
        k.append(int(np.argmax(zz[:km] - zz[1:km + 1]) + 1))
    return np.array(k)

def gen(N, delta, e, rho, seed, nlo=4, nhi=14, trend=0.):
    rng = np.random.default_rng(seed); ys, xs, ns = [], [], []
    while len(ns) < N:
        n = rng.integers(nlo, nhi + 1); y = np.zeros(n, bool); y[0] = rng.random() < e
        for t in range(1, n): y[t] = rng.random() < (rho if y[t-1] else e)
        if not y.any(): continue
        a = rng.normal(0, 3, len(delta)); b = np.exp(rng.normal(0, .7, len(delta)))
        pos = np.linspace(0, 1, n)
        x = a + b * (np.outer(y, delta) + trend * pos[:, None] + rng.normal(size=(n, len(delta))))
        ys.append(y); xs.append(x); ns.append(n)
    off = np.r_[0, np.cumsum(ns)]; return np.vstack(xs), np.concatenate(ys), off

if __name__ == '__main__':
    delta = np.array([1.5, 1., .8, .6]); X, y, off = gen(3000, delta, .15, .5, 1)
    Z = np.column_stack([answer_z(X[:, g], off) for g in range(4)]); f = Z.mean(1)
    t0 = time.time(); B = M.Batch(Z, f, off); print('batch', time.time() - t0, len(B.ans))
    th = np.r_[np.log(delta), np.log(.15/.85), 0.]
    v, g = M.objective(th, B); h = 1e-6; fd = []
    for j in range(len(th)):
        tp = th.copy(); tp[j] += h; tm = th.copy(); tm[j] -= h
        fd.append((M.objective(tp, B)[0] - M.objective(tm, B)[0]) / 2 / h)
    print('grad', np.round(g, 4), '\nfd  ', np.round(fd, 4))
    t0 = time.time(); F = M.fit(Z, f, off, B=B); print('fit', time.time() - t0, 'delta', np.round(F['delta'], 3), 'e', round(F['e'], 3), 'rho', round(F['rho'], 3), 'share', round(F['implied_share'], 3), F['valid'], F['reason'])
    for s in F['starts']: print('  ', s['start'], round(s['nll'], 3), s['success'], s['nit'])
    FI = M.fit(Z, f, off, markov=False, B=B); print('IND delta', np.round(FI['delta'], 3), 'e', round(FI['e'], 3), FI['valid'])
    true = np.add.reduceat(y.astype(int), off[:-1]); fz = answer_z(f, off)
    kg = gap_counts(fz, off); km = M.counts(F['theta'], B, off); ki = M.counts(FI['theta'], B, off, markov=False)
    ks = M.share_counts(F['implied_share'], off); n = np.diff(off); cap = np.ceil(n / 2)
    tc = np.minimum(true, cap)
    print('true share', y.mean(), 'MAE MDS', np.abs(km - true).mean(), 'IND', np.abs(ki - true).mean(), 'GAP', np.abs(kg - true).mean(), 'SHARE', np.abs(ks - true).mean(), 'oracle-cap', np.abs(tc-true).mean())
