import numpy as np, itertools, time
from syn import *
from scipy.optimize import minimize
from scipy.special import expit

class ExactBatch:
    """Every configuration with >= 1 error, for answers with 3 <= n <= 9 (enumeration)."""
    def __init__(self, Z, off):
        self.K = Z.shape[1]; R = {k: [] for k in ('ans','n','k','S','y1','n00','n01','n10','n11')}
        cfg = {}
        for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
            n = b - a
            if n not in cfg:
                Y = np.array(list(itertools.product([0, 1], repeat=n)), bool)[1:]; cfg[n] = Y
            Y = cfg[n]; U = np.column_stack([M._unit_centered(Z[a:b, g])[0] for g in range(self.K)])
            p, q = Y[:, :-1], Y[:, 1:]
            R['ans'].append(np.full(len(Y), i)); R['n'].append(np.full(len(Y), n)); R['k'].append(Y.sum(1))
            R['S'].append(Y.astype(float) @ U); R['y1'].append(Y[:, 0])
            R['n00'].append((~p & ~q).sum(1)); R['n01'].append((~p & q).sum(1)); R['n10'].append((p & ~q).sum(1)); R['n11'].append((p & q).sum(1))
        c = {k: np.concatenate(v) for k, v in R.items()}
        self.ans = c['ans']; self.n = c['n'].astype(float); self.k = c['k'].astype(float); self.S = c['S']
        self.mask = np.ones_like(self.S); self.y1 = c['y1'].astype(float)
        self.n00, self.n01, self.n10, self.n11 = (c[x].astype(float) for x in ('n00','n01','n10','n11'))
        self.kk = self.k * (self.n - self.k) / (2 * self.n)
        _, self.seg = np.unique(self.ans, return_index=True)
        self.seg_id = np.repeat(np.arange(len(self.seg)), np.diff(np.r_[self.seg, len(self.ans)]))

for (e, rho, delta) in [(.15, .5, [1.5, 1., .8, .6]), (.25, .6, [1.2, 1., .7, .5])]:
    delta = np.array(delta); X, y, off = gen(800, delta, e, rho, 3, nlo=4, nhi=9)
    Z = np.column_stack([answer_z(X[:, g], off) for g in range(len(delta))]); f = Z.mean(1)
    true = np.add.reduceat(y.astype(int), off[:-1]); n = np.diff(off); cap = np.ceil(n / 2)
    t0 = time.time(); EB = ExactBatch(Z, off)
    th0 = np.r_[np.zeros(len(delta)), np.log(.2 / .8), np.log(.4 / .6)]
    r = minimize(M.objective, th0, args=(EB, True), jac=True, method='L-BFGS-B', options=dict(maxiter=300))
    d = np.exp(r.x[:len(delta)]); ef = expit(r.x[len(delta)]); rf = expit(r.x[len(delta) + 1])
    def count_med(th):
        L, _ = M._entries(th, EB, True); w, _ = M._post(L, EB)
        med = []
        for j, s in enumerate(EB.seg):
            e_ = EB.seg[j + 1] if j + 1 < len(EB.seg) else len(L)
            ks = EB.k[s:e_]; pk = np.bincount(ks.astype(int), w[s:e_], minlength=12)
            med.append(min(int(np.argmax(np.cumsum(pk) >= .5)), cap[j]))
        return np.array(med)
    km = count_med(r.x); thT = np.r_[np.log(delta), np.log(e / (1 - e)), np.log(rho / (1 - rho))]; kt = count_med(thT)
    kg = gap_counts(answer_z(f, off), off); share = y.mean()
    ks = M.share_counts(share, off)
    print(f'TRUE e={e} rho={rho} delta={delta} | EXACT fit delta {np.round(d,2)} e {ef:.3f} rho {rf:.3f} ok={r.success} {time.time()-t0:.0f}s')
    print(f'   MAE exact-fit {np.abs(km-true).mean():.3f} exact-true-theta {np.abs(kt-true).mean():.3f} GAP {np.abs(kg-true).mean():.3f} SHARE(true share) {np.abs(ks-true).mean():.3f}', flush=True)
