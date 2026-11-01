"""Multi-voter Dawid-Skene error count with a Markov label chain on answer-normalized voters.

Pre-registration: docs/experiments/MARKOV_DS_COUNT_PREREG_2026-11-01.md (section 3 = model; amendment A = exact sum).

One answer has n steps and K voters. Voter g has raw scores x_gt = a + b * (delta_g * y_t + eps_gt) with an unknown
per-answer location a and scale b > 0, so only the direction u_g = z_g / ||z_g|| of its answer-z vector is informative.
That direction has the projected-normal density on the sphere of the centered subspace (dimension d = n - 1):

    l_g(y) = -delta_g^2 * k(n-k) / (2n) + log J_{n-1}(delta_g * sum_{t: y_t=1} u_gt)   (+ const(n))
    J_d(tau) = int_0^inf r^(d-1) exp(-r^2/2 + r*tau) dr

The labels follow a two-state Markov chain (P(y_1=1)=e, P(1|0)=e, P(1|1)=rho), conditioned on >= 1 error.

EXACT (primary, amendment A): the answer likelihood sums over all 2^n - 1 configurations with >= 1 error
(enumeration, answers with 3 <= n <= nmax). The count is the median of the exact posterior of the number of errors,
clipped to [1, ceil(n/2)].
TOPK (secondary, the section 3.3 approximation as first written): one configuration per k, the top k steps of the
arm's fused ranking. It fails the synthetic recovery test (prelabel_synthetic/PRELABEL_STOP.md) and is kept as a
declared secondary arm only. Labels never enter this module.
"""
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

_GX, _GW = np.polynomial.legendre.leggauss(48)
_SPAN, _SPAN_RIGHT = 15.0, 40.0      # panel widths in local SDs left and right of the mode (right: heavier tail)
_CHUNK = 40000

START_PAIRS = [(.1, .4), (.2, .2), (.15, .6), (.3, .5), (.05, .3)]
START_MULTS = [.5, 1., 2., .5, 1.]             # start j: delta_g = START_MULTS[j] * gap_g, (e, rho) = START_PAIRS[j]
DELTA_BOUNDS, PROB_BOUNDS, SHARE_BOUNDS = (.05, 10.), (.001, .999), (.01, .60)


# ----------------------------------------------------------------------------------------------- log J_d
def log_j(d, tau, with_psi=False, with_var=False):
    """log J_d(tau) for integer d >= 2 (vectorized), by two-panel Gauss-Legendre quadrature around the mode of the
    log-concave integrand. with_psi also returns psi = d/dtau log J_d = E[r]; with_var also returns
    d psi/dtau = Var[r] (r under the tilted density)."""
    d = np.asarray(d, float); tau = np.asarray(tau, float)
    d, tau = np.broadcast_arrays(d, tau); shp = d.shape
    d = d.ravel(); tau = tau.ravel()
    out = np.empty(d.size); psi = np.empty(d.size); var = np.empty(d.size)
    for s in range(0, d.size, _CHUNK):
        dd, tt = d[s:s + _CHUNK, None], tau[s:s + _CHUNK, None]
        rm = (tt + np.sqrt(tt * tt + 4 * (dd - 1))) / 2
        sd = 1 / np.sqrt(1 + (dd - 1) / np.maximum(rm, 1e-300) ** 2)
        lo = np.maximum(0., rm - _SPAN * sd); hi = rm + _SPAN_RIGHT * sd
        h1, h2 = (rm - lo) / 2, (hi - rm) / 2
        r = np.concatenate([lo + h1 * (_GX + 1), rm + h2 * (_GX + 1)], 1)
        lw = np.concatenate([np.log(h1 * _GW + 1e-300), np.log(h2 * _GW)], 1)
        lf = (dd - 1) * np.log(np.maximum(r, 1e-300)) - r * r / 2 + r * tt + lw
        lz = logsumexp(lf, axis=1)
        out[s:s + _CHUNK] = lz
        if with_psi or with_var:
            p = np.exp(lf - lz[:, None]); m1 = (p * r).sum(1)
            psi[s:s + _CHUNK] = m1; var[s:s + _CHUNK] = (p * (r - m1[:, None]) ** 2).sum(1)
    res = [out.reshape(shp)]
    if with_psi: res.append(psi.reshape(shp))
    if with_var: res.append(var.reshape(shp))
    return res[0] if len(res) == 1 else tuple(res)


def log_j_recursion(d, tau):
    """Reference J_d(tau) by the forward recursion (exact, stable for tau >= 0; used by the tests only)."""
    from scipy.stats import norm
    j1 = np.sqrt(2 * np.pi) * np.exp(tau * tau / 2) * norm.cdf(tau)
    if d == 1: return np.log(j1)
    jm, j = j1, 1 + tau * j1
    for dd in range(2, d):
        jm, j = j, tau * j + (dd - 1) * jm
    return np.log(j)


class JTable:
    """Cubic Hermite tables of log J_d (slope psi) and psi (slope Var r) on tau in [-TMAX, TMAX], step H, for
    d = 2..dmax. Values outside the grid are computed by direct quadrature."""
    TMAX, H = 60., .01

    def __init__(self, dmax):
        self.grid = np.arange(-self.TMAX, self.TMAX + self.H / 2, self.H); self.dmax = dmax
        D = np.repeat(np.arange(2, dmax + 1)[:, None], len(self.grid), 1)
        self.lj, self.psi, self.var = log_j(D, np.broadcast_to(self.grid, D.shape), with_psi=True, with_var=True)

    def __call__(self, d, tau):
        """d: int array (broadcast with tau), tau: array. Returns (log J_d(tau), psi)."""
        d = np.broadcast_to(np.asarray(d, int), tau.shape)
        x = (tau + self.TMAX) / self.H; i = np.clip(np.floor(x).astype(int), 0, len(self.grid) - 2); t = x - i
        r = d - 2
        f0, f1 = self.lj[r, i], self.lj[r, i + 1]; p0, p1 = self.psi[r, i], self.psi[r, i + 1]
        v0, v1 = self.var[r, i], self.var[r, i + 1]
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1; h10 = t ** 3 - 2 * t ** 2 + t; h01 = -2 * t ** 3 + 3 * t ** 2; h11 = t ** 3 - t ** 2
        lj = h00 * f0 + h10 * self.H * p0 + h01 * f1 + h11 * self.H * p1
        ps = h00 * p0 + h10 * self.H * v0 + h01 * p1 + h11 * self.H * v1
        out = np.abs(tau) > self.TMAX
        if out.any():
            lj = lj.copy(); ps = ps.copy()
            lj[out], ps[out] = log_j(d[out], tau[out], with_psi=True)
        return lj, ps


def _unit_centered(v):
    v = v - v.mean(); nr = np.sqrt((v * v).sum())
    return (v / nr, True) if nr > 1e-10 else (np.zeros_like(v), False)


def _as2d(v):
    v = np.asarray(v, float)
    return v[:, None] if v.ndim == 1 else v


def start_gaps(voters, off, answers):
    """Each voter's pooled top-quartile-vs-rest standardized gap of its answer-z values over the given answers."""
    voters = _as2d(voters); gaps = []
    for g in range(voters.shape[1]):
        z = np.concatenate([_unit_centered(voters[off[i]:off[i + 1], g])[0] * np.sqrt(off[i + 1] - off[i])
                            for i in answers])
        q = np.quantile(z, .75); hi = z > q
        gaps.append(max((z[hi].mean() - z[~hi].mean()) / max(z.std(), 1e-12), .1) if hi.any() and (~hi).any() else 1.)
    return np.array(gaps)


def _prior_terms(theta, K, markov):
    e = expit(theta[K]); rho = expit(theta[K + 1]) if markov else e
    return e, rho, np.log(e), np.log1p(-e), np.log(rho), np.log1p(-rho)


def _fit_loop(obj, B, K, gaps, markov):
    starts = []
    for j, ((e0, r0), m) in enumerate(zip(START_PAIRS, START_MULTS)):
        th0 = np.r_[np.log(np.clip(m * gaps, .06, 9.)), np.log(e0 / (1 - e0))]
        if markov: th0 = np.r_[th0, np.log(r0 / (1 - r0))]
        res = minimize(obj, th0, args=(B, markov), jac=True, method='L-BFGS-B',
                       options=dict(maxiter=500, maxfun=1500, ftol=1e-10, gtol=1e-6))
        starts.append(dict(start=j, nll=float(res.fun), success=bool(res.success), nit=int(res.nit),
                           message=str(res.message), theta=res.x.tolist()))
    return starts


def _summarize(starts, K, markov, share_fn):
    best = min(starts, key=lambda s: s['nll'])
    th = np.asarray(best['theta']); delta = np.exp(th[:K]); e = float(expit(th[K]))
    rho = float(expit(th[K + 1])) if markov else e
    share = float(share_fn(th))
    reasons = []
    if not best['success']: reasons.append('not converged: ' + best['message'])
    if not np.all((delta >= DELTA_BOUNDS[0]) & (delta <= DELTA_BOUNDS[1])): reasons.append('delta out of bounds')
    if not (PROB_BOUNDS[0] <= e <= PROB_BOUNDS[1] and PROB_BOUNDS[0] <= rho <= PROB_BOUNDS[1]): reasons.append('e/rho out of bounds')
    if not (SHARE_BOUNDS[0] <= share <= SHARE_BOUNDS[1]): reasons.append('implied share out of bounds')
    nlls = sorted(s['nll'] for s in starts)
    return dict(theta=th.tolist(), delta=delta.tolist(), e=e, rho=rho, markov=markov, K=K, implied_share=share,
                nll=best['nll'], best_start=best['start'], n_starts_at_best=int(sum(x - nlls[0] < 1e-3 for x in nlls)),
                starts=starts, valid=not reasons, reason='; '.join(reasons))


# ======================================================================================= EXACT (primary)
class Exact:
    """Answers with 3 <= n <= nmax, grouped by n. For each n: all configurations with >= 1 error (Y), their error
    count, Markov transition counts, and the answers' unit-centered voter directions U (A_n, K, n)."""

    def __init__(self, voters, off, nmax, table=None):
        voters = _as2d(voters); self.K = voters.shape[1]; self.off = np.asarray(off, np.int64); self.nmax = nmax
        self.n_all = np.diff(self.off)
        self.fit_answers = np.flatnonzero((self.n_all >= 3) & (self.n_all <= nmax))
        self.table = table or JTable(nmax - 1)
        self.groups = {}
        for n in np.unique(self.n_all[self.fit_answers]):
            n = int(n); A = self.fit_answers[self.n_all[self.fit_answers] == n]
            Y = np.array(list(itertools.product([0, 1], repeat=n)), bool)[1:]
            p, q = Y[:, :-1], Y[:, 1:]
            U = np.zeros((len(A), self.K, n)); M = np.zeros((len(A), self.K))
            for j, i in enumerate(A):
                for g in range(self.K):
                    U[j, g], M[j, g] = _unit_centered(voters[self.off[i]:self.off[i + 1], g])
            k = Y.sum(1).astype(float)
            self.groups[n] = dict(A=A, Y=Y, Yf=Y.astype(float).T.copy(), k=k, kk=k * (n - k) / (2 * n), U=U, M=M,
                                  y1=Y[:, 0].astype(float), n00=(~p & ~q).sum(1).astype(float),
                                  n01=(~p & q).sum(1).astype(float), n10=(p & ~q).sum(1).astype(float),
                                  n11=(p & q).sum(1).astype(float))
        self.chunk = 3_000_000

    def _group_pass(self, theta, markov, n, G, want=None):
        """Yields per answer chunk: (rows, L (a, c), dL/d log delta (a, c, K) or None)."""
        K = self.K; dl = np.exp(theta[:K])
        e, rho, le, l1e, lr, l1r = _prior_terms(theta, K, markov)
        prior = (G['y1'] * le + (1 - G['y1']) * l1e + G['n01'] * le + G['n00'] * l1e + G['n11'] * lr + G['n10'] * l1r
                 - np.log1p(-(1 - e) ** n))
        C = len(G['k']); step = max(1, self.chunk // (C * K))
        lj0 = log_j(n - 1, 0.)
        for s in range(0, len(G['A']), step):
            U = G['U'][s:s + step]; M = G['M'][s:s + step]
            S = U @ G['Yf']                                                   # (a, K, C)
            tau = S * dl[None, :, None]
            lj, ps = self.table(n - 1, tau)
            em = (-dl[None, :, None] ** 2 * G['kk'][None, None] + lj - lj0) * M[:, :, None]
            L = prior[None] + em.sum(1)
            dD = None
            if want == 'grad':
                dD = (-2 * dl[None, :, None] ** 2 * G['kk'][None, None] + ps * tau) * M[:, :, None]
            yield slice(s, s + step), L, dD

    def objective(self, theta, _B=None, markov=True):
        theta = np.asarray(theta, float); K = self.K; nll = 0.; grad = np.zeros(len(theta))
        e, rho, *_ = _prior_terms(theta, K, markov)
        for n, G in self.groups.items():
            de_core = (G['y1'] + G['n01']) / e - (1 - G['y1'] + G['n00']) / (1 - e) - n * (1 - e) ** (n - 1) / (1 - (1 - e) ** n)
            dr_core = G['n11'] / rho - G['n10'] / (1 - rho)
            for _, L, dD in self._group_pass(theta, markov, n, G, want='grad'):
                lse = logsumexp(L, axis=1); w = np.exp(L - lse[:, None]); nll -= lse.sum()
                grad[:K] -= np.einsum('ac,agc->g', w, dD)
                wc = w.sum(0)
                if markov:
                    grad[K] -= (wc @ de_core) * e * (1 - e); grad[K + 1] -= (wc @ dr_core) * rho * (1 - rho)
                else:
                    grad[K] -= (wc @ (de_core + dr_core)) * e * (1 - e)
        return nll, grad

    def posterior(self, theta, markov=True):
        """Per fit answer: count posterior P(count = k), k = 0..n (index), and per-step error marginals."""
        theta = np.asarray(theta, float); out = {}
        for n, G in self.groups.items():
            onehot = np.zeros((len(G['k']), n + 1)); onehot[np.arange(len(G['k'])), G['k'].astype(int)] = 1
            for sl, L, _ in self._group_pass(theta, markov, n, G):
                w = np.exp(L - logsumexp(L, axis=1)[:, None])
                pk = w @ onehot; marg = w @ G['Y'].astype(float)
                for j, i in enumerate(G['A'][sl]):
                    out[int(i)] = (pk[j], marg[j])
        return out


def fit_exact(voters, off, nmax, markov=True, X=None):
    """Label-free pooled exact fit over answers with 3 <= n <= nmax; five deterministic starts, best kept."""
    X = X or Exact(voters, off, nmax)
    gaps = start_gaps(voters, X.off, X.fit_answers)
    starts = _fit_loop(lambda th, B, mk: X.objective(th, None, mk), None, X.K, gaps, markov)

    def share(th):
        P = X.posterior(th, markov)
        return sum(float(pk @ np.arange(len(pk))) for pk, _ in P.values()) / X.n_all[X.fit_answers].sum()
    return _summarize(starts, X.K, markov, share)


def counts_exact(theta, X, markov=True):
    """k-hat per answer: median of the exact count posterior clipped to [1, ceil(n/2)] for 3 <= n <= nmax; 1 for
    n <= 2; -1 (caller falls back to GAP) for n > nmax. Also returns per-step error marginals (NaN outside the fit)."""
    k = np.full(len(X.off) - 1, -1); k[X.n_all <= 2] = 1
    marg = np.full(X.off[-1], np.nan)
    for i, (pk, mg) in X.posterior(theta, markov).items():
        n = X.n_all[i]; cdf = np.cumsum(pk)
        k[i] = int(np.clip(np.argmax(cdf >= .5 - 1e-12), 1, int(np.ceil(n / 2))))
        marg[X.off[i]:X.off[i + 1]] = mg
    return k, marg


def share_counts(share, off):
    """SHARE control: clip(round(share * n), 1, cap) with round half up."""
    n = np.diff(off); cap = np.maximum(np.ceil(n / 2), 1)
    return np.clip(np.floor(share * n + .5), 1, cap).astype(int)


# ======================================================================================= TOPK (secondary, as first written)
class TopK:
    """All (answer, k) entries of the answers with n >= 3: one configuration per k, the top k steps by f."""

    def __init__(self, voters, f, off):
        voters = _as2d(voters)
        self.K = voters.shape[1]; self.off = np.asarray(off, np.int64)
        n_all = np.diff(self.off); self.n_all = n_all
        self.fit_answers = np.flatnonzero(n_all >= 3)
        rows = dict(ans=[], n=[], k=[], S=[], mask=[], y1=[], n00=[], n01=[], n10=[], n11=[])
        for i in self.fit_answers:
            a, b = self.off[i], self.off[i + 1]; n = b - a; cap = int(np.ceil(n / 2))
            o = np.argsort(-f[a:b], kind='stable')
            U, M = np.zeros((n, self.K)), np.zeros(self.K, bool)
            for g in range(self.K):
                U[:, g], M[g] = _unit_centered(voters[a:b, g])
            S = np.cumsum(U[o], 0)[:cap]
            Y = np.zeros((cap, n), bool)
            for k in range(1, cap + 1): Y[k - 1, o[:k]] = True
            p, q = Y[:, :-1], Y[:, 1:]
            rows['ans'].append(np.full(cap, i)); rows['n'].append(np.full(cap, n)); rows['k'].append(np.arange(1, cap + 1))
            rows['S'].append(S); rows['mask'].append(np.repeat(M[None], cap, 0)); rows['y1'].append(Y[:, 0])
            rows['n00'].append((~p & ~q).sum(1)); rows['n01'].append((~p & q).sum(1))
            rows['n10'].append((p & ~q).sum(1)); rows['n11'].append((p & q).sum(1))
        cat = {k: np.concatenate(v) for k, v in rows.items()}
        self.ans, self.n, self.k = cat['ans'], cat['n'].astype(float), cat['k'].astype(float)
        self.S, self.mask = cat['S'], cat['mask'].astype(float)
        self.y1 = cat['y1'].astype(float)
        self.n00, self.n01, self.n10, self.n11 = (cat[x].astype(float) for x in ('n00', 'n01', 'n10', 'n11'))
        self.kk = self.k * (self.n - self.k) / (2 * self.n)
        _, self.seg = np.unique(self.ans, return_index=True)
        self.seg_id = np.repeat(np.arange(len(self.seg)), np.diff(np.r_[self.seg, len(self.ans)]))
        self.n_ans = n_all[self.fit_answers].astype(float)

    def entries(self, theta, markov=True):
        K = self.K; dl = np.exp(theta[:K])
        e, rho, le, l1e, lr, l1r = _prior_terms(theta, K, markov)
        q = np.power(1 - e, self.n)
        prior = (self.y1 * le + (1 - self.y1) * l1e + self.n01 * le + self.n00 * l1e + self.n11 * lr + self.n10 * l1r
                 - np.log1p(-q))
        tau = self.S * dl[None]
        lj, psi = log_j((self.n - 1)[:, None] * np.ones((1, K)), tau, with_psi=True)
        lj0 = log_j(self.n - 1, 0.)
        L = prior + ((-dl[None] ** 2 * self.kk[:, None] + lj - lj0[:, None]) * self.mask).sum(1)
        G = np.zeros((len(L), len(theta)))
        G[:, :K] = (-2 * dl[None] ** 2 * self.kk[:, None] + psi * tau) * self.mask
        dq = self.n * np.power(1 - e, self.n - 1) / (1 - q)
        de_core = (self.y1 + self.n01) / e - (1 - self.y1 + self.n00) / (1 - e) - dq
        dr_core = self.n11 / rho - self.n10 / (1 - rho)
        if markov:
            G[:, K] = de_core * e * (1 - e); G[:, K + 1] = dr_core * rho * (1 - rho)
        else:
            G[:, K] = (de_core + dr_core) * e * (1 - e)
        return L, G

    def post(self, L):
        lse = np.maximum.reduceat(L, self.seg)
        ex = np.exp(L - lse[self.seg_id]); Z = np.add.reduceat(ex, self.seg)
        return ex / Z[self.seg_id], np.log(Z) + lse

    def objective(self, theta, _B=None, markov=True):
        L, G = self.entries(np.asarray(theta, float), markov)
        w, ll = self.post(L)
        return -ll.sum(), -(w[:, None] * G).sum(0)

    def counts(self, theta, markov=True):
        """Posterior median over k = 1..cap for n >= 3; 1 for n <= 2."""
        L, _ = self.entries(np.asarray(theta, float), markov); w, _ = self.post(L)
        k = np.ones(len(self.off) - 1, int); within = np.cumsum(w)
        base = np.r_[0, within[self.seg[1:] - 1]][self.seg_id]; within = within - base
        for j, (s, e_) in enumerate(zip(self.seg, np.r_[self.seg[1:], len(L)])):
            k[self.fit_answers[j]] = int(self.k[s + np.argmax(within[s:e_] >= .5 - 1e-12)])
        return k


def fit_topk(voters, f, off, markov=True, T=None):
    T = T or TopK(voters, f, off)
    gaps = start_gaps(voters, T.off, T.fit_answers)
    starts = _fit_loop(lambda th, B, mk: T.objective(th, None, mk), None, T.K, gaps, markov)

    def share(th):
        L, _ = T.entries(th, markov); w, _ = T.post(L)
        return np.add.reduceat(w * T.k, T.seg).sum() / T.n_ans.sum()
    return _summarize(starts, T.K, markov, share)
