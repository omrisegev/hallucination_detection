"""How accurate is the module's 20-node Gauss-Hermite integral over the answer effect b, on data drawn FROM THE MODEL?
Reference: my route-R2 likelihood (2-state log-space HMM minus the all-clean path), vectorised over a dense b grid
(8001 points over +-10 tau, trapezoid in log space). Reports per-answer |d log-lik|, |d P(A=1|x)|, max |d P(y=1|x)|,
and the total NLL error per answer (what the optimizer sees), by f and answer length."""
import importlib.util, pathlib, numpy as np, pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
sp = importlib.util.spec_from_file_location('hh', pathlib.Path(__file__).resolve().parents[3] / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(sp); sp.loader.exec_module(H)


def dense(x, P, mu0, s0, G=8001):
    n = len(x); tau = np.sqrt(P['f']) * s0; sig0 = np.sqrt(1 - P['f']) * s0
    b = np.linspace(-10 * tau, 10 * tau, G); lpb = norm.logpdf(b, 0, tau) + np.log(b[1] - b[0])
    l0 = norm.logpdf(x[None, :], mu0 + b[:, None], sig0); l1 = norm.logpdf(x[None, :], mu0 + P['delta'] + b[:, None], P['sig1'])
    le = np.stack([l0, l1], -1)                                      # (G, n, 2)
    lT = np.log(np.array([[1 - P['h'], P['h']], [1 - P['rho'], P['rho']]]))
    la = np.zeros((G, n, 2)); la[:, 0] = np.log([1 - P['e1'], P['e1']]) + le[:, 0]
    for i in range(1, n):
        la[:, i] = logsumexp(la[:, i - 1, :, None] + lT[None], 1) + le[:, i]
    lb = np.zeros((G, n, 2))
    for i in range(n - 2, -1, -1):
        lb[:, i] = logsumexp(lT[None] + (le[:, i + 1] + lb[:, i + 1])[:, None, :], 2)
    lall = logsumexp(la[:, -1], -1); lzero = np.log(1 - P['e1']) + (n - 1) * np.log(1 - P['h']) + l0.sum(1)
    lnz = lall + np.log1p(-np.exp(np.minimum(lzero - lall, -1e-300)))
    lpany = np.log1p(-(1 - P['e1']) * (1 - P['h']) ** (n - 1))
    lA1 = np.log(P['pi']) - lpany + lnz; lA0 = np.log(1 - P['pi']) + l0.sum(1)
    lx = np.logaddexp(lA0, lA1) + lpb
    # trapezoid weights in log space: end points get half weight
    w = np.zeros(G); w[[0, -1]] = np.log(.5)
    LL = logsumexp(lx + w)
    pA = np.exp(logsumexp(lA1 + lpb + w) - LL)
    lyi = np.log(P['pi']) - lpany + la[:, :, 1] + lb[:, :, 1] + (lpb + w)[:, None]
    py = np.exp(logsumexp(lyi, 0) - LL)
    return LL, pA, py


rows = []
P0 = dict(pi=.6, e1=.1, h=.08, rho=.4, delta=1.0, sig1=1.0)
for f in (0.1, 0.3, 0.5, 0.7, 0.85):
    P = dict(P0, f=f)
    for nlen in (4, 8, 15, 30, 60, 107):
        x, off, y, A = H.simulate(np.full(12, nlen), **P, seed=int(f * 100) + nlen)
        X, M, _ = H.pad(x, off)
        ll, pA, py = H._forward(X, M, P, 0.0, 1.0, need_backward=True)
        for a in range(12):
            xa = x[off[a]:off[a + 1]]; L, qA, qy = dense(xa, P, 0.0, 1.0)
            rows.append(dict(f=f, n=nlen, d_ll=ll[a] - L, d_pA=abs(pA[a] - qA), d_py=np.abs(py[a, :nlen] - qy).max()))
d = pd.DataFrame(rows); d.to_csv(pathlib.Path(__file__).with_name('QUADRATURE_CHECK.csv'), index=False)
pd.set_option('display.width', 200)
t = d.assign(abs_d_ll=d.d_ll.abs()).groupby(['f', 'n']).agg(max_abs_d_ll=('abs_d_ll', 'max'), mean_d_ll=('d_ll', 'mean'),
                                                             max_d_pA=('d_pA', 'max'), max_d_py=('d_py', 'max'))
print(t.round(4).to_string())
