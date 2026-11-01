import numpy as np
from scipy.special import logsumexp, expit
from syn import *

def count_logprior(nmax, e, rho):
    """log P(count=k | n, >=1 error) for n=1..nmax, k=0..n under the Markov chain (DP over (state, count))."""
    out = {}
    p0 = np.zeros(nmax + 2); p1 = np.zeros(nmax + 2)        # P(count=c, y_t=0/1)
    p0[0] = 1 - e; p1[1] = e
    for n in range(1, nmax + 1):
        if n > 1:
            n0 = p0 * (1 - e) + p1 * (1 - rho); n1 = np.r_[0, (p0 * e + p1 * rho)[:-1]]
            p0, p1 = n0, n1
        tot = p0 + p1; z = 1 - tot[0]
        out[n] = np.log(np.maximum(tot[:n + 1], 1e-300)) - np.log(z)
    return out

orig = M._entries
def ent(theta, B, markov=True):
    L, G = orig(theta, B, markov)
    K = B.K; e = expit(theta[K]); rho = expit(theta[K+1]) if markov else e
    def prior(e, rho):
        T = count_logprior(int(B.n.max()), e, rho)
        return np.array([T[int(n)][int(k)] for n, k in zip(B.n, B.k)])  # slow but fine
    # remove configuration prior, add count prior
    Lc = L - B.cfg_prior(theta, markov) + prior(e, rho)
    h = 1e-5; G = G.copy()
    th = np.array(theta, float)
    for j in ([K, K+1] if markov else [K]):
        tp = th.copy(); tp[j] += h; tm = th.copy(); tm[j] -= h
        ep, rp = expit(tp[K]), (expit(tp[K+1]) if markov else expit(tp[K])); em_, rm = expit(tm[K]), (expit(tm[K+1]) if markov else expit(tm[K]))
        G[:, j] = (prior(ep, rp) - prior(em_, rm)) / 2 / h
    return Lc, G

def cfg_prior_factory(B):
    def f(theta, markov):
        K = B.K; e = expit(theta[K]); rho = expit(theta[K+1]) if markov else e
        le, l1e, lr, l1r = np.log(e), np.log1p(-e), np.log(rho), np.log1p(-rho); q = (1-e)**B.n
        return B.y1*le + (1-B.y1)*l1e + B.n01*le + B.n00*l1e + B.n11*lr + B.n10*l1r - np.log1p(-q)
    return f
M._entries = ent
for (e,rho,delta) in [(.15,.5,[1.5,1.,.8,.6]), (.1,.3,[1.,.8,.6]), (.25,.6,[1.2,1.,.7,.5,.4])]:
    delta=np.array(delta); X, y, off = gen(1500, delta, e, rho, 2)
    Z = np.column_stack([answer_z(X[:, g], off) for g in range(len(delta))]); f = Z.mean(1)
    true = np.add.reduceat(y.astype(int), off[:-1]); kg = gap_counts(answer_z(f,off), off)
    B = M.Batch(Z, f, off); B.cfg_prior = cfg_prior_factory(B)
    th = np.r_[np.log(delta), np.log(e/(1-e)), np.log(rho/(1-rho))]
    F = M.fit(Z, f, off, B=B); km = M.counts(F['theta'], B, off); kt = M.counts(th, B, off); ks = M.share_counts(F['implied_share'], off)
    print(f'e={e} rho={rho} share={y.mean():.3f} GAP {np.abs(kg-true).mean():.3f} | fit delta', np.round(F['delta'],2), 'e %.3f rho %.3f share %.3f'%(F['e'],F['rho'],F['implied_share']),
          'MAE true-th %.3f MDS %.3f SHARE %.3f'%(np.abs(kt-true).mean(), np.abs(km-true).mean(), np.abs(ks-true).mean()), flush=True)
