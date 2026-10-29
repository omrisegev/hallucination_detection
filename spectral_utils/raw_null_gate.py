"""Label-free clean-answer gate from RAW step scores: empirical-null local fdr, SML/L-SML combination, answer-level DS.

Pre-registration: docs/experiments/RAW_NULL_GATE_PREREG_2026-10-30.md. Literature for every idea used here:
docs/research_notes/RAW_NULL_GATE_LITERATURE_2026-10-30.md.

Why this module exists. Earlier label-free gates estimated the share of erroneous answers and failed: GAP votes and
per-answer z-scores remove each answer's own level (ds_gate_v1/v1b/e1b), and an answer-level two-component mixture
invents a second group on heavy-tailed data (answer_mixture_gate_v1, Step 484). Here the gate never needs that share.
It only needs what a CORRECT step looks like, learned from the centre of the raw step-score distribution, where almost
all steps are correct (Efron's "empirical null"). A step's probability of being wrong is 1 - lfdr.

Three gates, all label-free (labels only score, in the runner):
    G1   fused score: P(err) of the answer's worst step (the step GAP flags first) < 0.5  ->  answer clean (k = 0)
    G1b  the same worst step, conformal p-value against the GAP-unflagged steps, Sidak over the answer's steps,
         answer p > 0.05  ->  clean
    G2   SML / L-SML combination: under the L-SML model the telemetry groups are conditionally independent given the
         step label (Jaffe et al. 2016), so the step's posterior odds = prior odds x product of each group's likelihood
         ratio; each group's ratio comes from its own two-groups fit, on the group's defined steps, one-sided (LR = 1 at
         or below its null mean). Worst step = max combined P(err); < 0.5 -> clean
    G3   answer-level Dawid-Skene: each group votes "this answer has an abnormal step" (its own worst step has
         P(err) >= 0.5). Two-coin DS EM over the K group votes, the same estimator as the shipped between-group stage,
         gives each group's answer-level sensitivity/specificity, the share of erroneous answers and P(answer has an
         error); < 0.5 -> clean. Needs K >= 3 groups.
Flags of non-clean answers are plain GAP on the arm's usual (answer-z) fused score; clean answers get no flag.

All functions take arrays; none reads files or labels.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .fusion_utils import lsml_continuous_guarded
from .prm_anchored_lsml import ds_em_votes

P_ERR_CUT = .5
ALPHA_CONFORMAL = .05
PI0_CLIP = (.5, .995)
LR_CLIP = (1e-3, 1e3)


# ----------------------------------------------------------------------------------------------- the arm's L-SML
def linear_lsml(V, anchor=None):
    """Fit L-SML on the columns of V exactly as stage8_auto_forced_k.fit (lsml_continuous_guarded, sign by the
    correlation with `anchor`, default column 0) and return it as a LINEAR map, so the same weights can be read out
    on other (raw) values of the same channels. Asserts the linear map reproduces the fitted score to 1e-6."""
    V = np.asarray(V, float)
    s, d = lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])])
    s = np.asarray(s, float)
    ref = V[:, 0] if anchor is None else np.asarray(anchor, float)
    sign = -1. if np.corrcoef(s, ref)[0, 1] < 0 else 1.
    gw = [(np.asarray(i, int), np.asarray(w, float)) for i, w in d['group_weights']]
    cw = np.asarray(d['cross_weights'], float)
    p = dict(gw=gw, cw=cw, sign=sign, K=int(d['K']), c=np.asarray(d['c'], int).tolist())
    rebuilt = fused_linear(V, p)
    err = float(np.max(np.abs(rebuilt - sign * s)))
    if err > 1e-6:
        raise AssertionError(f'linear L-SML readout does not reproduce the fit: max diff {err:.3g}')
    p['parity'] = err
    return sign * s, p


def group_scores(X, p):
    """Per-group linear scores X[:, idx] @ w (singleton groups: the column itself), one column per group."""
    X = np.asarray(X, float)
    return np.column_stack([X[:, i[0]] if len(i) == 1 else X[:, i] @ w for i, w in p['gw']])


def group_defined(finite, p):
    """(steps x groups) True where at least one of the group's channels is defined (finite before NaN -> 0)."""
    finite = np.asarray(finite, bool)
    return np.column_stack([finite[:, i].any(1) for i, _ in p['gw']])


def fused_linear(X, p):
    G = group_scores(X, p)
    s = G[:, 0] if G.shape[1] == 1 else G @ p['cw']
    return p['sign'] * s


def group_orientation(p):
    """+1/-1 per group: the direction in which the group raises the fused score (0 if its cross weight is 0)."""
    if len(p['gw']) == 1:
        return np.array([p['sign']])
    return p['sign'] * np.sign(p['cw'])


def pooled_z(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


# ----------------------------------------------------------------------------------------------- empirical null
def _legendre_basis(x, lo, hi, deg):
    t = np.clip((np.asarray(x, float) - lo) / (hi - lo) * 2 - 1, -1, 1)
    return np.polynomial.legendre.legvander(t, deg)


def _poisson_glm(B, y, iters=100, tol=1e-10):
    """Unpenalized Poisson regression log mu = B beta by IRLS (Lindsey's method on histogram counts)."""
    """Returns (beta, converged)."""
    beta = np.zeros(B.shape[1]); beta[0] = np.log(max(y.mean(), 1e-12))
    for _ in range(iters):
        eta = np.clip(B @ beta, -50, 50); mu = np.exp(eta)
        z = eta + (y - mu) / np.maximum(mu, 1e-12)
        W = mu
        new = np.linalg.lstsq(B * np.sqrt(W)[:, None], z * np.sqrt(W), rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            return new, True
        beta = new
    return beta, False


def empirical_null(x, nbins: int = 120, deg: int = 7, pct0: float = .25):
    """Efron's two-groups fit of one score: f = pi0 f0 + (1 - pi0) f1, with f0 = N(mu0, sigma0^2).

    f is Lindsey's estimate (Poisson regression of histogram counts on a degree-`deg` polynomial, as locfdr's
    natural-spline default with df = 7). f0 and pi0 come from CENTRAL MATCHING (locfdr nulltype 2): a quadratic fitted
    to log f over the central bins, between the pct0 and 1 - pct0 quantiles of x, read as log(pi0 f0).

    Returns dict(ok, pi0 (clipped to PI0_CLIP), pi0_raw, mu0, sigma0, reason, and an `lfdr` callable)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = np.quantile(x, [.0005, .9995])
    if not hi > lo:
        return dict(ok=False, reason='degenerate score', pi0=np.nan, pi0_raw=np.nan, mu0=np.nan, sigma0=np.nan)
    edges = np.linspace(lo, hi, nbins + 1)
    y, _ = np.histogram(np.clip(x, lo, hi), bins=edges)
    mid = (edges[:-1] + edges[1:]) / 2; width = edges[1] - edges[0]
    B = _legendre_basis(mid, lo, hi, deg)
    beta, converged = _poisson_glm(B, y.astype(float))
    if not converged:
        return dict(ok=False, reason='IRLS did not converge', pi0=np.nan, pi0_raw=np.nan, mu0=np.nan, sigma0=np.nan)
    n = len(x)
    logf = lambda z: B_at(z) @ beta - np.log(n * width)
    B_at = lambda z: _legendre_basis(z, lo, hi, deg)
    q1, q2 = np.quantile(x, [pct0, 1 - pct0])
    cen = (mid >= q1) & (mid <= q2)
    if cen.sum() < 5:
        return dict(ok=False, reason='too few central bins', pi0=np.nan, pi0_raw=np.nan, mu0=np.nan, sigma0=np.nan)
    b2, b1, b0 = np.polyfit(mid[cen], logf(mid[cen]), 2)
    if b2 >= 0:
        return dict(ok=False, reason='log f not concave at the centre', pi0=np.nan, pi0_raw=np.nan, mu0=np.nan,
                    sigma0=np.nan)
    s0 = float(np.sqrt(-1 / (2 * b2))); m0 = float(b1 * s0 ** 2)
    pi0_raw = float(np.exp(b0 + m0 ** 2 / (2 * s0 ** 2)) * s0 * np.sqrt(2 * np.pi))
    pi0 = float(np.clip(pi0_raw, *PI0_CLIP))

    def lfdr(z):
        z = np.asarray(z, float)
        f = np.exp(logf(z))
        f0 = norm.pdf(z, m0, s0)
        return np.clip(pi0 * f0 / np.maximum(f, 1e-300), 0., 1.)

    return dict(ok=True, reason='', pi0=pi0, pi0_raw=pi0_raw, mu0=m0, sigma0=s0, lfdr=lfdr)


def p_error(x, fit):
    """One-sided P(step is wrong | score): 1 - lfdr above the null mean, 0 at or below it (low risk is not an error)."""
    x = np.asarray(x, float)
    if not fit['ok']:
        return np.full(len(x), np.nan)
    return np.where(x > fit['mu0'], 1. - fit['lfdr'](x), 0.)


# ----------------------------------------------------------------------------------------------- answers
def worst_step(score_answer_z, off):
    """Index of the highest step of each answer (ties: first), i.e. the step GAP flags first."""
    return np.array([a + int(np.argmax(score_answer_z[a:b])) for a, b in zip(off[:-1], off[1:])])


def answer_max(v, off):
    return np.array([float(np.max(v[a:b])) for a, b in zip(off[:-1], off[1:])])


def gate_g1(raw_fused, z_fused, off):
    fit = empirical_null(raw_fused)
    pe = p_error(raw_fused, fit)
    w = worst_step(z_fused, off)
    pw = pe[w]
    return dict(clean=(pw < P_ERR_CUT) if fit['ok'] else np.zeros(len(w), bool), p_err_worst=pw, p_err_step=pe,
                fit={k: v for k, v in fit.items() if k != 'lfdr'})


def gate_g1b(raw_fused, z_fused, off, gap_flags):
    """Conformal p-value of the worst step against the GAP-unflagged steps (all answers), Sidak over n steps."""
    ref = np.sort(raw_fused[~gap_flags])
    w = worst_step(z_fused, off)
    n_ge = len(ref) - np.searchsorted(ref, raw_fused[w], side='left')
    p = (1 + n_ge) / (1 + len(ref))
    n = np.diff(off)
    p_ans = 1 - (1 - p) ** n
    return dict(clean=p_ans > ALPHA_CONFORMAL, p_answer=p_ans, n_ref=int(len(ref)))


def _group_null(raw_g, o_g, defined_g):
    """u_g = o_g * pooled z over the group's DEFINED steps (undefined: 0), and its two-groups fit on defined steps only.
    Undefined steps (e.g. digit-inactive) would otherwise form a spike of exact zeros that central matching reads as a
    narrow null (review finding, 2026-10-30)."""
    u = np.zeros(len(raw_g))
    if defined_g.sum() < 50:
        return u, dict(ok=False, reason=f'{int(defined_g.sum())} defined steps')
    u[defined_g] = o_g * pooled_z(raw_g[defined_g])
    return u, empirical_null(u[defined_g])


def _defined(raw_groups, defined):
    return np.ones(raw_groups.shape, bool) if defined is None else np.asarray(defined, bool)


def group_fit_summary(fits):
    return [dict(group=int(g), ok=bool(f['ok']), reason=f.get('reason', ''), pi0=f.get('pi0'), pi0_raw=f.get('pi0_raw'),
                 mu0=f.get('mu0'), sigma0=f.get('sigma0'), share_defined=float(d)) for g, f, d in fits]


def gate_g2(raw_groups, orient, off, defined=None):
    """Product of group likelihood ratios (L-SML between-group conditional independence) with a shared pi0.
    One-sided: LR = 1 at or below the group's null mean; LR = 1 on steps where the group is undefined."""
    D = _defined(raw_groups, defined)
    use = np.flatnonzero(orient != 0)
    fits, summ = [], []
    for g in use:
        u, f = _group_null(raw_groups[:, g], orient[g], D[:, g])
        summ.append((g, f, D[:, g].mean()))
        if f['ok']:
            fits.append((g, f, u))
    if not fits:
        return dict(ok=False, reason='no group null identified', clean=np.zeros(len(off) - 1, bool),
                    groups=group_fit_summary(summ))
    pi0s = np.array([f['pi0'] for _, f, _ in fits])
    pi0 = float(np.clip(np.median(pi0s), *PI0_CLIP))
    log_odds = np.full(raw_groups.shape[0], np.log((1 - pi0) / pi0))
    for g, f, u in fits:
        l = np.maximum(f['lfdr'](u), 1e-12)
        lr = np.clip((pi0 / (1 - pi0)) * (1 / l - 1), *LR_CLIP)
        lr = np.where((u > f['mu0']) & D[:, g], lr, 1.)
        log_odds += np.log(lr)
    pe = 1 / (1 + np.exp(-log_odds))
    pmax = answer_max(pe, off)
    return dict(ok=True, reason='', clean=pmax < P_ERR_CUT, p_err_step=pe, p_err_worst=pmax, pi0=pi0,
                pi0_groups=pi0s.tolist(), groups_used=[int(g) for g, _, _ in fits], groups=group_fit_summary(summ))


def ds_answer_posterior(Bv, fit):
    """P(answer has an error | its K group votes) under the fitted two-coin DS model (psi = P(vote 1 | error),
    eta = P(vote 0 | clean))."""
    V = np.asarray(Bv) > 0
    psi, eta, p = np.asarray(fit['psi']), np.asarray(fit['eta']), fit['prevalence']
    l1 = np.where(V, np.log(psi), np.log(1 - psi)).sum(1) + np.log(p)
    l0 = np.where(V, np.log(1 - eta), np.log(eta)).sum(1) + np.log(1 - p)
    return 1 / (1 + np.exp(l0 - l1))


def gate_g3(raw_groups, orient, off, anchor_group=None, defined=None):
    D = _defined(raw_groups, defined)
    use = np.flatnonzero(orient != 0)
    votes, used, summ = [], [], []
    for g in use:
        u, f = _group_null(raw_groups[:, g], orient[g], D[:, g])
        summ.append((g, f, D[:, g].mean()))
        if not f['ok']:
            continue
        pe = np.where(D[:, g], p_error(u, f), 0.)
        votes.append(np.where(answer_max(pe, off) >= P_ERR_CUT, 1., -1.)); used.append(int(g))
    if len(used) < 3:
        return dict(ok=False, reason=f'{len(used)} group votes (< 3)', clean=np.zeros(len(off) - 1, bool),
                    groups=group_fit_summary(summ))
    Bv = np.column_stack(votes)
    anc = used.index(anchor_group) if (anchor_group is not None and anchor_group in used) else None
    fit = ds_em_votes(Bv, anchor=anc, init='majority')
    post = ds_answer_posterior(Bv, fit)
    M = np.cov(Bv.T); M = M - np.diag(np.diag(M)); v = np.linalg.eigh(M)[1][:, -1]; v = v if v.sum() >= 0 else -v
    return dict(ok=True, reason='', clean=post < P_ERR_CUT, p_answer=post, prevalence=float(fit['prevalence']),
                psi=np.asarray(fit['psi']).tolist(), eta=np.asarray(fit['eta']).tolist(), groups_used=used,
                anchor_used=anc is not None, vote_rate=((Bv > 0).mean(0)).tolist(), sml_ranking=v.tolist(),
                groups=group_fit_summary(summ))


def gap_flags(z_fused, off, gap_counts, topk_flags):
    """GAP flags of the answer-z fused score with the collection's own gap_counts / topk_flags (passed in)."""
    return topk_flags(z_fused, off, gap_counts(z_fused, off))
