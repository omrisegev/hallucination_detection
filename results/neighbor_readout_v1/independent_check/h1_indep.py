"""Independent re-implementation of H1 (HMM-POS) and H1-NOPOS from the prereg text only."""
import sys, json, time
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, '/Users/osegev/Desktop/hallucination_detection')
sys.path.insert(0, '/Users/osegev/Desktop/hallucination_detection/results/prm_boost_collection_v1')
import evaluate_collection as E

OUT = '/private/tmp/claude-501/-Users-osegev-Desktop-hallucination-detection/06ffb662-6235-4776-9351-8647e8b98b70/scratchpad/indep/'
TOL, MAXIT, NEWTON, SFLOOR = 1e-7, 500, 10, 1e-3

def logsig(z): return -np.logaddexp(0.0, -z)

class Data:
    def __init__(self, s, off):
        n = np.diff(off); A, T = len(n), int(n.max())
        self.n, self.A, self.T = n, A, T
        self.M = np.arange(T)[None, :] < n[:, None]
        idx = np.where(self.M, off[:-1, None] + np.arange(T)[None, :], 0)
        self.S = np.where(self.M, s[idx], 0.0)
        t1 = np.arange(1, T + 1)[None, :]
        self.R = np.where(self.M, (t1 - .5) / n[:, None], 0.0)
        self.idx = idx

def estep(D, mu, sd, rho, a, b):
    M, S, R, A, T = D.M, D.S, D.R, D.A, D.T
    le = (-0.5 * np.log(2 * np.pi) - np.log(sd)[None, None, :]
          - (S[..., None] - mu[None, None, :]) ** 2 / (2 * sd[None, None, :] ** 2))
    le = np.where(M[..., None], le, 0.0)
    z = a + b * R
    le_in, le_out = logsig(z), logsig(-z)          # log e_t, log(1-e_t)
    # logA[:, t, i, j]
    logA = np.empty((A, T, 2, 2))
    logA[:, :, 0, 0] = le_out; logA[:, :, 0, 1] = le_in
    logA[:, :, 1, 0] = np.log1p(-rho); logA[:, :, 1, 1] = np.log(rho)
    la = np.empty((A, T, 2)); lb = np.zeros((A, T, 2))
    la[:, 0, 0] = le_out[:, 0] + le[:, 0, 0]; la[:, 0, 1] = le_in[:, 0] + le[:, 0, 1]
    for t in range(1, T):
        new = logsumexp(la[:, t - 1, :, None] + logA[:, t], axis=1) + le[:, t]
        la[:, t] = np.where(M[:, t, None], new, la[:, t - 1])
    for t in range(T - 2, -1, -1):
        new = logsumexp(logA[:, t + 1] + (le[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
        lb[:, t] = np.where(M[:, t + 1, None], new, 0.0)
    llA = logsumexp(la[np.arange(A), D.n - 1], axis=1)
    lp = la + lb                                       # unnormalized log posterior
    gam = np.where(M[..., None], np.exp(lp - llA[:, None, None]), 0.0)
    xi = np.exp(la[:, :-1, :, None] + logA[:, 1:] + (le[:, 1:] + lb[:, 1:])[:, :, None, :]
                - llA[:, None, None, None])
    xi = np.where(M[:, 1:, None, None], xi, 0.0)
    return float(llA.sum()), gam, xi, lp

def mstep(D, gam, xi, a, b, use_pos):
    S, M, R = D.S, D.M, D.R
    w = gam.reshape(-1, 2); x = S.reshape(-1)
    W = w.sum(0)
    mu = (w * x[:, None]).sum(0) / W
    sd = np.sqrt((w * (x[:, None] - mu[None]) ** 2).sum(0) / W)
    sd = np.maximum(sd, SFLOOR)
    rho = xi[:, :, 1, 1].sum() / (xi[:, :, 1, 0].sum() + xi[:, :, 1, 1].sum())
    # logistic regression on expected transitions out of state 0, incl. first step (from implicit state 0)
    pos = np.concatenate([gam[:, 0, 1], xi[:, :, 0, 1].reshape(-1)])
    neg = np.concatenate([gam[:, 0, 0], xi[:, :, 0, 0].reshape(-1)])
    r = np.concatenate([R[:, 0], R[:, 1:].reshape(-1)])
    keep = np.concatenate([M[:, 0], M[:, 1:].reshape(-1)])
    pos, neg, r = pos[keep], neg[keep], r[keep]; wt = pos + neg
    X = np.stack([np.ones_like(r), r], 1) if use_pos else np.ones((len(r), 1))
    th = np.array([a, b]) if use_pos else np.array([a])
    for _ in range(NEWTON):
        p = 1 / (1 + np.exp(-(X @ th)))
        g = X.T @ (pos - wt * p)
        H = (X * (wt * p * (1 - p))[:, None]).T @ X
        th = th + np.linalg.solve(H, g)
    a, b = (th[0], th[1]) if use_pos else (th[0], 0.0)
    return mu, sd, float(rho), float(a), float(b)

def fit_one(D, start, use_pos):
    mu, sd, rho, a, b = start
    ll_prev = None; n = 0
    for n in range(MAXIT + 1):
        ll, gam, xi, lp = estep(D, mu, sd, rho, a, b)
        if ll_prev is not None and abs(ll - ll_prev) / abs(ll_prev) < TOL:
            conv = True; break
        conv = False
        if n == MAXIT: break
        mu, sd, rho, a, b = mstep(D, gam, xi, a, b, use_pos)
        ll_prev = ll
    return dict(mu=mu, sd=sd, rho=rho, a=a, b=b, ll=ll, n_iter=n, converged=conv, lp=lp)

def fit(D, s, use_pos):
    qs = [(.5, .90), (.5, .95), (.6, .90), (.4, .85), (.5, .80)]
    psd = float(np.std(s)); a0 = float(np.log(.1 / .9))
    fits = []
    for q0, q1 in qs:
        st = (np.array([np.quantile(s, q0), np.quantile(s, q1)]), np.array([psd, psd]), .4, a0, 0.0)
        fits.append(fit_one(D, st, use_pos))
    best = int(np.argmax([f['ll'] for f in fits]))
    f = fits[best]
    e = int(np.argmax(f['mu'])); o = 1 - e
    logit_pad = f['lp'][:, :, e] - f['lp'][:, :, o]
    logit = logit_pad[D.M]                              # row-major == answer order
    # implied share: q_1 = e_1, q_t = q_{t-1} rho + (1-q_{t-1}) e_t  (prob of state 1)
    et = 1 / (1 + np.exp(-(f['a'] + f['b'] * D.R)))
    q = np.empty_like(et); q[:, 0] = et[:, 0]
    for t in range(1, D.T): q[:, t] = q[:, t - 1] * f['rho'] + (1 - q[:, t - 1]) * et[:, t]
    q1 = float(q[D.M].mean()); share = q1 if e == 1 else 1 - q1
    ratio = float(f['sd'][e] / f['sd'][o]); gap = float(f['mu'][e] - f['mu'][o])
    reasons = []
    if share < .01 or share > .60: reasons.append(f'share {share:.4f} outside [.01,.60]')
    if ratio < .1 or ratio > 10: reasons.append(f'sd ratio {ratio:.4f} outside [.1,10]')
    if gap < .1: reasons.append(f'mu gap {gap:.4f} < .1')
    par = dict(mu=f['mu'].tolist(), sd=f['sd'].tolist(), rho=f['rho'], a=f['a'], b=f['b'],
               loglik=f['ll'], n_iter=f['n_iter'], converged=f['converged'], start_won=best,
               start_quantiles=qs[best], err_state=e, implied_share=share, sd_ratio_err_over_other=ratio,
               mu_gap=gap, valid=not reasons, reasons=reasons,
               all_starts=[dict(ll=g['ll'], n_iter=g['n_iter'], converged=g['converged'],
                                mu=g['mu'].tolist(), sd=g['sd'].tolist(), rho=g['rho'], a=g['a'], b=g['b'])
                           for g in fits])
    return par, logit

if __name__ == '__main__':
    t0 = time.time()
    c = E.load('prmbench', 'qwen_prm_7b')
    s = E.answer_z(c.arms['CHAIN_N16'], c.off); off = c.off
    k_gap = E.gap_counts(s, off)
    D = Data(s, off)
    assert np.array_equal(D.S[D.M], s)
    out, res = {}, {}
    for name, use_pos in [('h1', True), ('h0', False)]:
        t1 = time.time()
        par, logit = fit(D, s, use_pos)
        par['runtime_s'] = time.time() - t1
        res[name] = par
        out[f'{name}_logit'] = logit
        out[f'{name}a'] = E.topk_flags(logit, off, k_gap)
        out[f'{name}b'] = logit > 0
        gapf = E.topk_flags(s, off, k_gap)
        out[f'{name}a_final'] = out[f'{name}a'] if par['valid'] else gapf
        out[f'{name}b_final'] = out[f'{name}b'] if par['valid'] else gapf
        print(name, json.dumps({k: v for k, v in par.items() if k != 'all_starts'}))
    res['meta'] = dict(N_answers=int(len(off) - 1), N_steps=int(off[-1]), total_runtime_s=time.time() - t0,
                       h1b_count=int(out['h1b'].sum()), h0b_count=int(out['h0b'].sum()),
                       gap_count=int(k_gap.sum()))
    np.savez(OUT + 'h1_indep.npz', **out)
    json.dump(res, open(OUT + 'h1_indep.json', 'w'), indent=1)
    print(res['meta'])
