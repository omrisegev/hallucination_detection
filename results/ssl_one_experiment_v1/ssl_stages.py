"""Stages S, S2, A, G of the one SSL experiment (ANSWERS_TO_MAC_2026-11-18.md, answers 22-55). CPU, numpy/scipy.

Optimizer (answer 27): Adam, lr .05, 500 full-batch steps on theta, parameter = lo + (hi - lo) * sigmoid(theta)
(stage S: N_c in [1, 40] tokens; the step tables keep the 48 largest tokens, so 40 stays inside the stored values;
stage A: M_c in [.5, 15] steps). Gradient: central finite difference per parameter (h = 1e-3). This is exact up to
O(h^2) here because one parameter changes one column only, and only the terms of that column depend on it.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import expit, logit
from scipy.stats import rankdata

import ssl_data as D
import ssl_lib as L

def zstd(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v)


def answer_z_vec(r, off):
    """Vectorized answer z (answer_z_masked with all-true mask: sd <= 1e-12 -> 0). For the objective only."""
    n = np.diff(off).astype(float); a = off[:-1]
    mu = np.add.reduceat(r, a) / n
    d = r - np.repeat(mu, np.diff(off))
    sd = np.sqrt(np.add.reduceat(d * d, a) / n)
    sd_r = np.repeat(sd, np.diff(off))
    return np.where(sd_r > 1e-12, d / np.where(sd_r > 1e-12, sd_r, 1.), 0.)


class Topk:
    """One token channel's sorted token values with NaN -> 0 and a finite mask, for fast soft top-N."""

    def __init__(self, top_c):
        ok = np.isfinite(top_c); self.v = np.where(ok, top_c, 0.).astype(float); self.m = ok.astype(float)

    def __call__(self, N, tau=1.):
        w = L.soft_topn_weights(N, self.v.shape[1], tau)
        return (self.v @ w) / (self.m @ w)


class CorrObjective:
    """L(theta) = sum over sample sets of sum_{pairs with >= 1 free column} corr^2 - lam * sum_c (AUC_c - .5).
    evalc(c, value) -> ([z of column c in each sample set], soft paired AUC of column c or None)."""

    def __init__(self, n_par, fixed, evalc, lam, lo, hi):
        self.n_par, self.evalc, self.lam, self.lo, self.hi = n_par, evalc, lam, lo, hi
        self.fixed = [np.column_stack([zstd(f[:, j]) for j in range(f.shape[1])]) if f is not None and f.shape[1] else None
                      for f in fixed]

    def val(self, th):
        return self.lo + (self.hi - self.lo) * expit(th)

    def _cols(self, vals):
        out = [self.evalc(c, v) for c, v in enumerate(vals)]
        Z = [np.column_stack([zstd(o[0][s]) for o in out]) for s in range(len(self.fixed))]
        auc = np.array([np.nan if o[1] is None else o[1] for o in out])
        return Z, auc

    def full(self, vals):
        Z, auc = self._cols(vals); J = 0.; parts = []
        for s, Zs in enumerate(Z):
            A = Zs if self.fixed[s] is None else np.column_stack([Zs, self.fixed[s]])
            C = (A.T @ A) / len(A); np.fill_diagonal(C, 0.)
            jj = (C[:self.n_par] ** 2).sum() - (C[:self.n_par, :self.n_par] ** 2).sum() / 2   # each free pair once
            J += jj; parts.append(float(jj))
        a = np.nansum(auc - .5)
        return dict(L=J - self.lam * a, J=J, J_sets=parts, auc=auc, Z=Z)

    def _partial(self, c, v, Z):
        zc, auc = self.evalc(c, v); f = 0.
        for s, Zs in enumerate(Z):
            z = zstd(zc[s]); oth = np.delete(Zs, c, 1)
            if self.fixed[s] is not None: oth = np.column_stack([oth, self.fixed[s]])
            f += ((z @ oth) / len(z)) ** 2 @ np.ones(oth.shape[1])
        return f - (0. if auc is None else self.lam * (auc - .5))

    def fit(self, init_val, steps=500, lr=.05, h=1e-3, b1=.9, b2=.999):
        th = logit((np.broadcast_to(np.asarray(init_val, float), (self.n_par,)) - self.lo) / (self.hi - self.lo))
        m = np.zeros(self.n_par); v2 = np.zeros(self.n_par); trace = []
        for t in range(1, steps + 1):
            vals = self.val(th); Z, _ = self._cols(vals); g = np.zeros(self.n_par)
            for c in range(self.n_par):
                hp = min(h, self.hi - vals[c]); hm = min(h, vals[c] - self.lo)
                dv = (self._partial(c, vals[c] + hp, Z) - self._partial(c, vals[c] - hm, Z)) / max(hp + hm, 1e-12)
                s = expit(th[c]); g[c] = dv * (self.hi - self.lo) * s * (1 - s)
            m = b1 * m + (1 - b1) * g; v2 = b2 * v2 + (1 - b2) * g * g
            th = th - lr * (m / (1 - b1 ** t)) / (np.sqrt(v2 / (1 - b2 ** t)) + 1e-8)
            if t % 50 == 0 or t == 1: trace.append((t, float(self.full(self.val(th))['L'])))
        vals = self.val(th); r = self.full(vals)
        return dict(vals=vals, L=float(r['L']), J=float(r['J']), J_sets=r['J_sets'], auc=r['auc'].tolist(), trace=trace)


# ============================================================== stage S (token readout)
class StageSData:
    """Precomputed arrays of stage S on one fit set: V2 answers (subset) and the C answers of the same groups."""

    def __init__(self, V2, Cg, proxy_rows=None):
        cpos = {u: j for j, u in enumerate(Cg.uids)}
        pj = np.array([cpos[p] for p in V2.parent]); assert (np.diff(Cg.off)[pj] == np.diff(V2.off)).all()
        self.kidx = V2.off[:-1] + V2.k                     # V2 row of the foreign step
        self.crow = Cg.off[pj] + V2.k                      # parent C row at the same position
        self.offv, self.offc = V2.off, Cg.off
        self.tv = [Topk(V2.top[:, c, :]) for c in range(11)]
        self.tc = [Topk(Cg.top[:, c, :]) for c in range(11)]
        s5v = L.r0_cols(np.column_stack([np.zeros((V2.off[-1], 11)), V2.step5]), V2.act, V2.off)[:, 11:]
        self.fix_v = s5v[self.kidx]
        self.proxy = proxy_rows
        if proxy_rows is not None:
            s5c = L.r0_cols(np.column_stack([np.zeros((Cg.off[-1], 11)), Cg.step5]), Cg.act, Cg.off)[:, 11:]
            self.fix_p = s5c[proxy_rows]
        self.n_v = len(self.kidx)

    def evalc(self, c, N):
        rv = self.tv[c](N); rc = self.tc[c](N)
        zv = answer_z_vec(rv, self.offv)[self.kidx]
        sc = .1 * rc.std()
        auc = float(np.mean(expit((rv[self.kidx] - rc[self.crow]) / max(sc, 1e-12))))
        sets = [zv]
        if self.proxy is not None:
            sets.append(answer_z_vec(rc, self.offc)[self.proxy])
        return sets, auc

    def objective(self, lam):
        fixed = [self.fix_v] + ([self.fix_p] if self.proxy is not None else [])
        return CorrObjective(11, fixed, self.evalc, lam, 1., 40.)


def fit_stage_s(data, lam, inits=(3., 10., 30.), steps=500):
    obj = data.objective(lam); best = None; runs = []
    for v0 in inits:
        r = obj.fit(v0, steps=steps); r['init'] = v0; runs.append({k: r[k] for k in ('init', 'L', 'J')})
        if best is None or r['L'] < best['L']: best = r
    best['inits'] = runs
    return best


def objective_at(data, lam, N):
    """L of a fixed N (e.g. N = 10 everywhere) on a fit set (for the half-B check)."""
    r = data.objective(lam).full(np.asarray(N, float))
    return dict(L=float(r['L']), J=float(r['J']), auc=r['auc'].tolist())


# ============================================================== dev evaluation of a step readout
def step_eval(C, raw, boot=None, gi=None):
    """Method-of-record pipeline on a C pool with given raw 16 channels: R0, L-SML (fit on this pool), GAP.
    Returns metrics + conditional |corr| (Spearman, answer 29) + per-channel AUROC (wrong vs clean)."""
    X = L.r0_cols(raw, C.act, C.off); s, W, meta = L.lsml_linear(X); f = L.gap_flags(s, C.off)
    out = dict(lsml=meta, W=W.tolist())
    if C.bench == 'processbench':
        m = L.pb_metrics(L.pb_predict(f, C.off), C.lab, C.sub)
        out.update(SLA=m['macro SLA'], F1=m['average F1'], main=m['macro SLA'])
    else:
        out.update(PRMScore=L.prmscore(C.err, f, C.keep, C.off)); out['main'] = out['PRMScore']
    w, c = C.wrong_step, C.clean_step
    out['auc_channel'] = [L.auroc(np.r_[np.ones(w.sum()), np.zeros(c.sum())].astype(bool), np.r_[X[w, j], X[c, j]]) for j in range(16)]
    if boot is not None:
        Wd = boot.W; gidx = gi[np.repeat(np.arange(C.n), np.diff(C.off))]
        cc = L.cond_abs_corr(L.ranks(X[w]), gidx[w], L.ranks(X[c]), gidx[c], Wd)
        out['cond'] = cc
    return out, s, f, X


class Boot:
    def __init__(self, groups, draws=2000, seed=20261118):
        self.u, self.gi = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(seed); G = len(self.u)
        self.W = np.vstack([np.ones(G)] + [np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(draws)]).astype(float)


# ============================================================== ILP over hard readout candidates (answer 30)
CANDS = ['top1', 'top2', 'top5', 'top10', 'top20', 'mean', 'median', 'p90']


def ilp_readout(V2, Cg, lam):
    """Exact Spearman of R0 at V2 step k and exact paired AUROC; one candidate per token channel; MILP."""
    cpos = {u: j for j, u in enumerate(Cg.uids)}; pj = np.array([cpos[p] for p in V2.parent])
    kidx = V2.off[:-1] + V2.k; crow = Cg.off[pj] + V2.k
    R, AUC = {}, {}
    for c in range(11):
        for q, name in enumerate(CANDS):
            rv = D.candidate(V2, c, name); rc = D.candidate(Cg, c, name)
            zv = L.answer_z_masked(rv, V2.off, np.ones(len(rv), bool))[kidx]
            R[c, q] = zstd(rankdata(zv))
            dv = rv[kidx] - rc[crow]; AUC[c, q] = float(np.mean((dv > 0) + .5 * (dv == 0)))
    s5 = L.r0_cols(np.column_stack([np.zeros((V2.off[-1], 11)), V2.step5]), V2.act, V2.off)[kidx, 11:]
    F = np.column_stack([zstd(rankdata(s5[:, j])) for j in range(5)]); n = len(kidx); Q = len(CANDS)
    nx = 11 * Q; xi = lambda c, q: c * Q + q
    lin = np.zeros(nx)
    for c in range(11):
        for q in range(Q):
            lin[xi(c, q)] = ((R[c, q] @ F / n) ** 2).sum() - lam * (AUC[c, q] - .5)
    pairs = [(c, d, q, r) for c in range(11) for d in range(c + 1, 11) for q in range(Q) for r in range(Q)]
    ny = len(pairs); cost = np.r_[lin, [(R[c, q] @ R[d, r] / n) ** 2 for c, d, q, r in pairs]]
    A = np.zeros((11 + ny, nx + ny)); lb = np.r_[np.ones(11), np.full(ny, -np.inf)]; ub = np.r_[np.ones(11), np.full(ny, 1.)]
    for c in range(11): A[c, [xi(c, q) for q in range(Q)]] = 1
    for i, (c, d, q, r) in enumerate(pairs):          # y >= x_cq + x_dr - 1  <=>  x_cq + x_dr - y <= 1
        A[11 + i, xi(c, q)] = 1; A[11 + i, xi(d, r)] = 1; A[11 + i, nx + i] = -1
    res = milp(cost, constraints=LinearConstraint(A, lb, ub), integrality=np.r_[np.ones(nx), np.zeros(ny)],
               bounds=Bounds(np.zeros(nx + ny), np.ones(nx + ny)))
    x = np.round(res.x[:nx]).reshape(11, Q)
    choice = [CANDS[int(np.argmax(x[c]))] for c in range(11)]
    top10 = float(sum(lin[xi(c, 3)] for c in range(11)) + sum((R[c, 3] @ R[d, 3] / n) ** 2 for c in range(11) for d in range(c + 1, 11)))
    return dict(choice=choice, objective=float(res.fun), objective_top10=top10, status=int(res.status))


def ilp_raw(C, choice):
    tok = np.column_stack([D.candidate(C, c, choice[c]) for c in range(11)])
    return np.column_stack([tok, C.step5])


# ============================================================== stage A (answer features)
class AnswerModel:
    """Label-free answer features fitted on a C pool (raw 16 step channels of the final readout):
    orientation sign_c = sign corr(R0_c, s) on the pool (risk-high), ECDF per channel on the pool steps
    (digit channels: active steps only, inactive -> 0), soft top-M over the sorted step ECDF values of the answer,
    Simes of s_g (L-SML on G16 fitted on the pool; step p-values against the pool steps), answer standardization and
    answer L-SML (17 views) fitted on the pool answers."""

    def __init__(self, C, rawC, sC):
        X0 = L.r0_cols(rawC, C.act, C.off)
        self.sign = np.array([1. if np.corrcoef(X0[:, j], sC)[0, 1] >= 0 else -1. for j in range(16)])
        self.ecdf = L.ECDF(rawC * self.sign, C.act)
        self.rg = L.RankGauss(rawC, C.act)
        G = self.rg(rawC, C.act); self.sg_C, self.Wg, self.gmeta = L.lsml_linear(G)
        self.C = C

    def sorted_ecdf(self, P, raw):
        e = self.ecdf(raw * self.sign, P.act); n = np.diff(P.off); Lm = int(n.max())
        S = np.full((P.n, 16, Lm), np.nan)
        for j, (a, b) in enumerate(zip(P.off[:-1], P.off[1:])):
            S[j, :, :b - a] = -np.sort(-e[a:b].T, 1)
        return S

    def simes(self, P, raw):
        sg = self.rg(raw, P.act) @ self.Wg
        return -np.log10(np.maximum(L.simes(sg, P.off, ref=self.sg_C), 1e-300)), sg

    def agg(self, S, M):
        return L.soft_topn(S, M)

    def fit_fusion(self, featC):
        self.mu, self.sd = featC.mean(0), featC.std(0); self.sd[self.sd == 0] = 1
        Z = (featC - self.mu) / self.sd
        self.sA_C, self.Wa, self.ameta = L.lsml_linear(Z)

    def score(self, feat):
        return ((feat - self.mu) / self.sd) @ self.Wa


class StageAData:
    def __init__(self, Sv2, Sc, simes_v2, cpos_idx):
        """Sv2: (n_v2, 16, L) sorted ECDF of the fit V2 answers; Sc: same for the pool C answers;
        simes_v2: (n_v2,) fixed Simes feature; cpos_idx: index of each V2 answer's parent in Sc."""
        self.tv = [Topk(Sv2[:, c, :]) for c in range(16)]; self.tc = [Topk(Sc[:, c, :]) for c in range(16)]
        self.fix = simes_v2[:, None]; self.pj = cpos_idx

    def evalc(self, c, M):
        av = self.tv[c](M); ac = self.tc[c](M); sc = .1 * ac.std()
        auc = float(np.mean(expit((av - ac[self.pj]) / max(sc, 1e-12))))
        return [av], auc

    def objective(self, lam):
        return CorrObjective(16, [self.fix], self.evalc, lam, .5, 15.)


def fit_stage_a(data, lam, inits=(1., 3., 10.), steps=500):
    obj = data.objective(lam); best = None; runs = []
    for v0 in inits:
        r = obj.fit(v0, steps=steps); r['init'] = v0; runs.append({k: r[k] for k in ('init', 'L', 'J')})
        if best is None or r['L'] < best['L']: best = r
    best['inits'] = runs
    return best
