"""Independent verifier for the pre-registered raw-null gate (docs/experiments/RAW_NULL_GATE_PREREG_2026-10-30.md).

Written from Sections 2-5 of the pre-registration ONLY. The author never opened spectral_utils/raw_null_gate.py or
run_raw_null_gate.py. Shared estimators named by the spec are used as-is: lsml_continuous_guarded (fusion_utils) and
ds_em_votes (prm_anchored_lsml). Everything else (linear L-SML readout, raw readout, Lindsey/central-matching
empirical null, gates G1/G1b/G2/G3, GAP flags in synthetic mode, endpoints) is re-implemented here.

    python verify_raw_null_gate.py synthetic   # no data; compares this file with the module (black box) value by value
    python verify_raw_null_gate.py real        # other machine, after `run_raw_null_gate.py run`; writes VERIFY.json

Exit codes: 0 PASS, 1 FAIL (a disagreement), 2 refused (missing inputs / unreadable schema / crash before comparing).

Follows the 2026-10-30 amendment (Section 6a): defined steps per group (3.2a), one-sided G2, groups with < 50 defined
steps dropped, clipped Legendre variable and IRLS predictor/working-response floors (3.3.2), finite raw values.

Real-mode output schemas are NOT specified by the pre-registration, so they are read flexibly and any unreadable part
is a refusal or a FAIL, never a skip:
  RESULTS.csv  columns arm, mode (backbone|standalone|prm|sa; optional for AUTO), benchmark|bench, prm, and either a
               `gate` column (long format) or `<gate>_<field>` columns (wide). Fields: closed_clean, kept_error,
               share_called_error (exact), metric, metric_gap_only, ece_step, true_share_error (1e-6), valid.
  FITS.json    list of per-cell dicts (arm/mode/benchmark/prm) or a dict keyed by cell strings. Every dict holding
               pi0/mu0/sigma0 is a null fit: G1 (or no gate in its path) = fused fit; under G2/G3 = a group fit,
               identified by its 'group' field or its list index. Also checked when present: orientation(s),
               cross_weights, groups_used, anchor_used, share_defined, G2 pi0/pi0_groups, G3 prevalence/psi/eta/
               sml_ranking/vote_rate.
  CHECK.json   per-cell status ('ok' or other); all 80 cells must be listed and only the 12 PRMBench backbone cells
               may be not ok (Section 6).
  labelfree/   one <cell>.npz per ok cell; arrays named like fused, raw_fused, gap, <G>_clean, <G>_p_err_step,
               <G>_p_answer, <G>_p_err_worst are compared, other keys are listed in VERIFY.json as not checked.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import csv, json, math, re, sys, time, traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
try:
    from lever_imports import ensure_spectral_package
    ensure_spectral_package()
except Exception:                                    # pragma: no cover - fall back to a plain path insert
    sys.path.insert(0, str(ROOT))
from spectral_utils.fusion_utils import lsml_continuous_guarded  # noqa: E402
from spectral_utils.prm_anchored_lsml import ds_em_votes  # noqa: E402

# ------------------------------------------------------------------ constants fixed by the spec (Sections 3-5)
Q_RANGE = (.0005, .9995)
NBINS, DEG = 120, 7
IRLS_ITERS, IRLS_TOL = 100, 1e-10
Q_CENTRAL = (.25, .75)
MIN_CENTRAL_BINS = 5
PI0_CLIP = (.5, .995)
P_CUT = .5
ALPHA = .05
LR_CLIP = (1e-3, 1e3)
LFDR_FLOOR = 1e-12
SD_FLOOR_POOLED = 1e-12
SD_FLOOR_ANSWER = 1e-8          # E.answer_z / evaluate_collection.answer_z
FLOAT_TOL = 1e-6
MIN_DEFINED = 50                 # 3.4 G2 (amended): groups with fewer defined steps are dropped (G3: "as in G2")
ETA_CLIP, MU_FLOOR = 50., 1e-12   # amendment (c): IRLS linear predictor clip, working-response floor
# History: the first synthetic comparison (before the 2026-10-30 amendment) found that Section 3.3 did not say
# whether z is clipped when f is evaluated outside [lo, hi]; the module clipped. Amendment (c) now fixes
# t = clip(..., -1, 1) at every evaluation point, which is what _basis implements.
GATES = ('G1', 'G1b', 'G2', 'G3')


# ================================================================== independent implementation
def pooled_z(x):
    """(x - mean) / sd over all steps of the cell (population sd); sd <= 1e-12 gives 0."""
    x = np.asarray(x, float); sd = x.std()
    return np.zeros_like(x) if sd <= SD_FLOOR_POOLED else (x - x.mean()) / sd


def answer_z(s, off):
    out = np.zeros(len(s))
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; out[a:b] = (v - v.mean()) / max(v.std(), SD_FLOOR_ANSWER)
    return out


def lsml_linear(V, anchor=None):
    """3.1: lsml_continuous_guarded on the columns of V; sign by correlation with column 0 (anchor=None) or anchor.
    Returns (fused, readout) where readout holds the linear map: groups (idx, w), cross weights c, sign sigma."""
    V = np.asarray(V, float)
    s, d = lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])])
    s = np.asarray(s, float)
    ref = V[:, 0] if anchor is None else np.asarray(anchor, float)
    sigma = -1. if np.corrcoef(s, ref)[0, 1] < 0 else 1.
    gw = [(np.asarray(i, int), np.asarray(w, float)) for i, w in d['group_weights']]
    cw = np.asarray(d['cross_weights'], float).ravel()
    if len(cw) != len(gw):
        raise AssertionError(f'cross weights ({len(cw)}) do not match groups ({len(gw)})')
    fused = sigma * s
    rebuilt = linear_map(V, gw, cw, sigma)
    ro = dict(gw=gw, cw=cw, sigma=sigma, K=int(d['K']), c=np.asarray(d['c'], int),
              parity=float(np.max(np.abs(rebuilt - fused))))
    if ro['parity'] > FLOAT_TOL:
        raise AssertionError(f'linear readout does not rebuild the L-SML fit: max diff {ro["parity"]:.3g}')
    return fused, ro


def group_raw(Z, idx, w):
    return Z[:, idx] @ w if len(idx) > 1 else Z[:, idx[0]] * float(w[0])


def linear_map(Z, gw, cw, sigma):
    return sigma * sum(cg * group_raw(Z, i, w) for (i, w), cg in zip(gw, cw))


def orientation(ro):
    """o_g = sigma * sign(c_g); with a single group o = sigma."""
    if len(ro['gw']) == 1:
        return np.array([ro['sigma']])
    return ro['sigma'] * np.sign(ro['cw'])


def raw_readout(Z, ro):
    """3.2 with the 2026-10-30 amendment (a). Z may hold NaN (undefined entries, e.g. digit channels on
    digit-inactive steps). Fused: NaN -> 0, then raw_fused = pooled-z(sigma sum_g c_g Z_g w_g) over all steps.
    Groups: defined_g = at least one of the group's channels defined; u_g = o_g pooled-z(Z0_g w_g) with the pooled
    mean/sd over defined steps only (Z0 = Z with NaN -> 0); u_g = 0 on undefined steps (never used there)."""
    o = orientation(ro)
    fin = np.isfinite(Z); Z0 = np.where(fin, Z, 0.)
    lin = linear_map(Z0, ro['gw'], ro['cw'], ro['sigma'])
    GL = np.column_stack([group_raw(Z0, i, w) for i, w in ro['gw']])
    D = np.column_stack([fin[:, i].any(1) for i, _ in ro['gw']])
    G0 = np.zeros_like(GL)
    for g in range(GL.shape[1]):
        G0[D[:, g], g] = pooled_z(GL[D[:, g], g])
    return dict(lin=lin, raw_fused=pooled_z(lin), u=G0 * o[None, :], g_unoriented=G0, g_lin=GL, defined=D,
                orient=o)


def _basis(x, lo, hi):
    """Amendment (c): t = clip(2(z - lo)/(hi - lo) - 1, -1, 1) at every evaluation point."""
    t = np.clip(2. * (np.asarray(x, float) - lo) / (hi - lo) - 1., -1., 1.)
    return np.polynomial.legendre.legvander(t, DEG)


def empirical_null(x):
    """3.3 (Efron two-groups, Lindsey density + central matching). Returns a dict with ok/reason and, if ok,
    pi0, pi0_raw, mu0, sigma0 and the callables logf / lfdr / p_err."""
    x = np.asarray(x, float); N = len(x)
    out = dict(ok=False, reason='', pi0=np.nan, pi0_raw=np.nan, mu0=np.nan, sigma0=np.nan)
    if N == 0 or not np.all(np.isfinite(x)):
        out['reason'] = 'empty or non-finite'; return out
    lo, hi = np.quantile(x, Q_RANGE)
    if not hi > lo:
        out['reason'] = 'degenerate range'; return out
    cnt, edges = np.histogram(np.clip(x, lo, hi), bins=NBINS, range=(lo, hi))
    mids = .5 * (edges[:-1] + edges[1:]); width = (hi - lo) / NBINS
    B = _basis(mids, lo, hi); y = cnt.astype(float)
    beta = np.zeros(B.shape[1]); beta[0] = np.log(y.mean())
    converged = False
    for _ in range(IRLS_ITERS):
        # Amendment (c): linear predictor clipped to [-50, 50]; mu floored at 1e-12 in the working response.
        eta = np.clip(B @ beta, -ETA_CLIP, ETA_CLIP); mu = np.exp(eta)
        zw = eta + (y - mu) / np.maximum(mu, MU_FLOOR)
        try:
            with np.errstate(all='ignore'):
                new = np.linalg.solve((B * mu[:, None]).T @ B, (B * mu[:, None]).T @ zw)
        except np.linalg.LinAlgError:
            out['reason'] = 'IRLS singular'; return out
        step = new - beta; beta = new
        if not np.all(np.isfinite(beta)):
            out['reason'] = 'IRLS non-finite'; return out
        if np.max(np.abs(step)) < IRLS_TOL:
            converged = True; break
    out['irls_converged'] = converged
    if not converged:                                    # 3.3.2 (amended): non-convergence means 3.3 fails
        out['reason'] = 'IRLS did not converge'; return out
    lognorm = np.log(N * width)
    logf = lambda z: _basis(z, lo, hi) @ beta - lognorm
    q1, q3 = np.quantile(x, Q_CENTRAL)
    sel = (mids >= q1) & (mids <= q3)
    if sel.sum() < MIN_CENTRAL_BINS:
        out['reason'] = f'only {int(sel.sum())} central bins'; return out
    A = np.column_stack([np.ones(sel.sum()), mids[sel], mids[sel] ** 2])
    b0, b1, b2 = np.linalg.lstsq(A, logf(mids[sel]), rcond=None)[0]
    if not b2 < 0:
        out['reason'] = 'b2 >= 0 (null not identified)'; return out
    s0 = math.sqrt(-1. / (2. * b2)); m0 = b1 * s0 ** 2
    p0raw = math.exp(b0 + m0 ** 2 / (2 * s0 ** 2)) * s0 * math.sqrt(2 * math.pi)
    p0 = float(np.clip(p0raw, *PI0_CLIP))

    def lfdr(z):
        z = np.asarray(z, float)
        phi = np.exp(-.5 * ((z - m0) / s0) ** 2) / (s0 * math.sqrt(2 * math.pi))
        return np.clip(p0 * phi / np.exp(logf(z)), 0., 1.)

    def p_err(z):
        z = np.asarray(z, float)
        return np.where(z > m0, 1. - lfdr(z), 0.)

    out.update(ok=True, pi0=p0, pi0_raw=float(p0raw), mu0=float(m0), sigma0=float(s0), b=(b0, b1, b2),
               lo=float(lo), hi=float(hi), logf=logf, lfdr=lfdr, p_err=p_err)
    return out


def fit_public(f):
    return {k: f[k] for k in ('ok', 'reason', 'pi0', 'pi0_raw', 'mu0', 'sigma0')}


def worst_step(z_fused, off):
    """Index of the argmax (first on ties) of the fused score within each answer."""
    return np.array([a + int(np.argmax(z_fused[a:b])) for a, b in zip(off[:-1], off[1:])])


def answer_max(v, off):
    return np.array([float(np.max(v[a:b])) for a, b in zip(off[:-1], off[1:])])


def gate_g1(raw_fused, z_fused, off):
    f = empirical_null(raw_fused); n_ans = len(off) - 1
    if not f['ok']:
        return dict(ok=False, clean=np.zeros(n_ans, bool), p_err_step=np.full(len(raw_fused), np.nan), fit=f)
    pe = f['p_err'](raw_fused); w = worst_step(z_fused, off)
    return dict(ok=True, clean=pe[w] < P_CUT, p_err_step=pe, p_err_worst=pe[w], fit=f)


def gate_g1b(raw_fused, z_fused, off, gap):
    ref = np.sort(raw_fused[~np.asarray(gap, bool)]); w = worst_step(z_fused, off); x = raw_fused[w]
    n_ge = len(ref) - np.searchsorted(ref, x, side='left')
    p = (1. + n_ge) / (1. + len(ref)); n_a = np.diff(off)
    p_ans = 1. - (1. - p) ** n_a
    return dict(ok=True, clean=p_ans > ALPHA, p_answer=p_ans, n_ref=int(len(ref)))


def group_fits(u, orient, D):
    """3.3 on u_g over the group's defined steps only (3.2a), for each group with o_g != 0; a group with fewer than
    MIN_DEFINED defined steps is dropped (3.4 G2; applied to G3 too, which reuses G2's fits 'as in G2')."""
    out = {}
    for g in range(u.shape[1]):
        if orient[g] == 0:
            continue
        if D[:, g].sum() < MIN_DEFINED:
            out[g] = dict(ok=False, reason=f'{int(D[:, g].sum())} defined steps < {MIN_DEFINED}', pi0=np.nan,
                          pi0_raw=np.nan, mu0=np.nan, sigma0=np.nan)
        else:
            out[g] = empirical_null(u[D[:, g], g])
    return out


def group_lfdr(f, u_g, d_g):
    out = np.ones(len(u_g)); out[d_g] = f['lfdr'](u_g[d_g]); return out


def group_perr(f, u_g, d_g):
    out = np.zeros(len(u_g)); out[d_g] = f['p_err'](u_g[d_g]); return out


def gate_g2(u, orient, off, D, fits=None):
    fits = group_fits(u, orient, D) if fits is None else fits
    used = [g for g, f in fits.items() if f['ok']]; n_ans = len(off) - 1
    if not used:
        return dict(ok=False, clean=np.zeros(n_ans, bool), p_err_step=np.full(len(u), np.nan), used=[], fits=fits)
    p0g = [fits[g]['pi0'] for g in used]
    p0 = float(np.clip(np.median(p0g), *PI0_CLIP))
    lo = np.log((1 - p0) / p0) * np.ones(len(u))
    for g in used:
        lf = np.maximum(group_lfdr(fits[g], u[:, g], D[:, g]), LFDR_FLOOR)
        lr = np.clip((p0 / (1 - p0)) * (1. / lf - 1.), *LR_CLIP)
        lr = np.where(D[:, g] & (u[:, g] > fits[g]['mu0']), lr, 1.)    # amendment (a) undefined, (b) one-sided
        lo = lo + np.log(lr)
    pe = 1. / (1. + np.exp(-lo))
    mx = answer_max(pe, off)
    return dict(ok=True, clean=mx < P_CUT, p_err_step=pe, p_err_worst=mx, pi0=p0, pi0_groups=p0g, used=used,
                fits=fits)


def ds_posterior(Bv, fit):
    """P(answer has error | votes) by Bayes' rule with the fitted two-coin parameters (vote +1 = 'error')."""
    P = np.asarray(Bv) > 0
    psi, eta, pr = np.asarray(fit['psi']), np.asarray(fit['eta']), float(fit['prevalence'])
    l1 = np.log(pr) + np.where(P, np.log(psi), np.log(1 - psi)).sum(1)
    l0 = np.log(1 - pr) + np.where(P, np.log(1 - eta), np.log(eta)).sum(1)
    return 1. / (1. + np.exp(l0 - l1))


def sml_ranking(Bv):
    C = np.cov(np.asarray(Bv, float).T); C = C - np.diag(np.diag(C))
    v = np.linalg.eigh(C)[1][:, -1]
    return -v if v.sum() < 0 else v


def gate_g3(u, orient, off, D, anchor_group=None, fits=None):
    fits = group_fits(u, orient, D) if fits is None else fits
    used = [g for g, f in fits.items() if f['ok']]; n_ans = len(off) - 1
    if len(used) < 3:
        return dict(ok=False, clean=np.zeros(n_ans, bool), p_answer=np.full(n_ans, np.nan), used=used,
                    reason=f'{len(used)} votes per answer')
    Bv = votes(u, D, fits, used, off)
    anchor = used.index(anchor_group) if anchor_group is not None and anchor_group in used else None
    fit = ds_em_votes(Bv, anchor=anchor, init='majority')
    pa = ds_posterior(Bv, fit)
    return dict(ok=True, clean=pa < P_CUT, p_answer=pa, prevalence=float(fit['prevalence']),
                psi=np.asarray(fit['psi']).tolist(), eta=np.asarray(fit['eta']).tolist(), used=used,
                anchor_used=anchor, anchor_requested=anchor_group, vote_rate=(Bv > 0).mean(0).tolist(),
                sml_ranking=sml_ranking(Bv).tolist())


def votes(u, D, fits, used, off):
    return np.column_stack([np.where(answer_max(group_perr(fits[g], u[:, g], D[:, g]), off) >= P_CUT, 1, -1)
                            for g in used])


def gap_counts(z, off):
    k = []
    for a, b in zip(off[:-1], off[1:]):
        m = b - a
        if m == 1: k.append(1); continue
        zz = np.sort(z[a:b])[::-1]; km = min(int(np.ceil(m / 2)), m - 1)
        k.append(int(np.argmax(zz[:km] - zz[1:km + 1]) + 1))
    return np.array(k)


def gap_flags_own(score, off):
    z = answer_z(score, off); k = gap_counts(z, off); f = np.zeros(off[-1], bool)
    for j, (a, b) in enumerate(zip(off[:-1], off[1:])):
        f[a + np.argsort(-z[a:b], kind='stable')[:min(k[j], b - a)]] = True
    return f


def prm_group(ro):
    """Position (in the readout's group list) of the group that contains column 0 (the PRM in AUTO)."""
    for g, (i, _) in enumerate(ro['gw']):
        if 0 in set(i.tolist()):
            return g
    return None


def label_free_stage(V, Z, off, arm, gap_fn=None):
    """Everything label-free for one cell and arm. V = answer-z inputs, Z = cleaned raw inputs (same columns).
    gap_fn(fused) -> GAP flags (3.5); default: this file's own GAP implementation."""
    anchor = None if arm == 'AUTO' else V.mean(1)
    fused, ro = lsml_linear(V, anchor)
    rr = raw_readout(Z, ro)
    gap = np.asarray((gap_fn or (lambda s: gap_flags_own(s, off)))(fused), bool)
    fits = group_fits(rr['u'], rr['orient'], rr['defined'])
    out = dict(arm=arm, fused=fused, ro=ro, rr=rr, gap=gap,
               G1=gate_g1(rr['raw_fused'], fused, off),
               G1b=gate_g1b(rr['raw_fused'], fused, off, gap),
               G2=gate_g2(rr['u'], rr['orient'], off, rr['defined'], fits),
               G3=gate_g3(rr['u'], rr['orient'], off, rr['defined'], prm_group(ro) if arm == 'AUTO' else None,
                          fits))
    return out


# ================================================================== comparison helpers
class Ledger:
    def __init__(self):
        self.rows = []

    def num(self, ctx, name, mine, theirs, tol=FLOAT_TOL):
        a = np.asarray(mine, float).ravel(); b = np.asarray(theirs, float).ravel()
        if a.shape != b.shape:
            self.rows.append(dict(ctx=ctx, name=name, ok=False, max_diff=None, tol=tol,
                                  note=f'shape {a.shape} vs {b.shape}')); return False
        both_nan = np.isnan(a) & np.isnan(b)
        bad_nan = np.isnan(a) ^ np.isnan(b)
        d = np.where(both_nan | bad_nan, 0., np.abs(a - b))
        md = float(d.max()) if len(d) else 0.
        ok = bool(md <= tol and not bad_nan.any())
        self.rows.append(dict(ctx=ctx, name=name, ok=ok, max_diff=md, tol=tol, n=int(len(a)),
                              nan_mismatch=int(bad_nan.sum())))
        return ok

    def exact(self, ctx, name, mine, theirs):
        a = np.asarray(mine); b = np.asarray(theirs)
        ok = a.shape == b.shape and bool(np.array_equal(a.astype(bool), b.astype(bool)))
        nd = int((a.astype(bool) != b.astype(bool)).sum()) if a.shape == b.shape else None
        self.rows.append(dict(ctx=ctx, name=name, ok=ok, max_diff=nd, tol=0, n=int(a.size), note='exact'))
        return ok

    def note(self, ctx, name, ok, note):
        self.rows.append(dict(ctx=ctx, name=name, ok=bool(ok), max_diff=None, tol=None, note=note))

    @property
    def failed(self):
        return [r for r in self.rows if not r['ok']]


# ================================================================== synthetic mode
def synth_cell(seed, n_ans=1500, p_clean=.4, groups=(3, 3, 2, 2), d=(3.0, 2.8, 2.5, 2.3), answer_sd=.3,
               nuisance=.4, with_prm=False, constant_col=False, invert_group=None, bimodal_group=None,
               bimodal_all=False, undef_group=None, undef_channels=None, undef_frac=.3):
    """Answers of 6-12 steps; p_clean clean answers; the others 1-2 contiguous error steps. Each channel =
    d_g y + group noise + shared nuisance + channel noise + answer effect. Options exercise edge paths: a constant
    channel (dropped view), an inverted group (o_g = -1), a bimodal group noise (that group's null fails), a bimodal
    shared nuisance (every null fails: G1, G2, G3 invalid), undefined (NaN) entries on ~undef_frac of the steps
    for the channels `undef_channels` (default: all) of group `undef_group`, sharing one mask like the digit
    channels' digit_active."""
    rng = np.random.default_rng(seed)
    lens = rng.integers(6, 13, n_ans); off = np.r_[0, np.cumsum(lens)].astype(np.int64); n = int(off[-1])
    y = np.zeros(n, bool); has = rng.random(n_ans) > p_clean
    for j in np.flatnonzero(has):
        a, b = off[j], off[j + 1]; k = int(rng.integers(1, 3)); s = int(rng.integers(a, b - k + 1)); y[s:s + k] = True
    a_eff = np.repeat(rng.normal(0, answer_sd, n_ans), lens)
    eta = rng.standard_normal(n) + (3. * rng.choice([-1., 1.], n) if bimodal_all else 0.)
    cols = []
    undef = rng.random(n) < undef_frac
    nan_cols = []
    for g, (gsize, dg) in enumerate(zip(groups, d)):
        if g == undef_group:
            nan_cols += [len(cols) + k for k in range(gsize) if undef_channels is None or k in undef_channels]
        e = rng.standard_normal(n) + (4. * rng.choice([-1., 1.], n) if bimodal_group == g else 0.)
        sgn = -1. if invert_group == g else 1.
        for _ in range(gsize):
            cols.append(sgn * (dg * y + .8 * e + nuisance * eta + .5 * rng.standard_normal(n) + a_eff))
    if constant_col:
        cols.append(np.full(n, 3.7))
    X = np.column_stack(cols)
    X[np.ix_(undef, nan_cols)] = np.nan
    if with_prm:
        prm = 2.6 * y + rng.standard_normal(n) + .3 * eta + a_eff
        X = np.column_stack([prm, X])
    return X, off, y, has


def answer_z_cols(X, off):
    """Stored-channel construction (votes_e1b.raw_matrix's parity check): answer-z over the defined entries of each
    answer (population sd; sd <= 1e-12 gives 0), 0 on undefined entries."""
    Zs = np.zeros(X.shape)
    for j in range(X.shape[1]):
        for a, b in zip(off[:-1], off[1:]):
            m = np.isfinite(X[a:b, j]); v = X[a:b, j][m]
            if len(v) and v.std() > 1e-12:
                Zs[a:b, j][m] = (v - v.mean()) / v.std()
    return Zs


def pooled_z_defined(x):
    """Synthetic 'cleaned raw': pooled z over defined entries, NaN elsewhere (votes_e1b.clean without the cubic)."""
    out = np.full(len(x), np.nan); m = np.isfinite(x)
    if m.any(): out[m] = pooled_z(x[m])
    return out


def compare_synthetic(L, name, X, off, arm, Gm):
    """Run this file's pipeline and the module's on the same synthetic cell; record every comparison in L."""
    V = answer_z_cols(X, off)
    Z = np.column_stack([pooled_z_defined(X[:, j]) for j in range(X.shape[1])])   # synthetic cleaned raw (NaN kept)
    mine = label_free_stage(V, Z, off, arm)
    fin = np.isfinite(Z); Z0 = np.where(fin, Z, 0.)          # module inputs: NaN -> 0 plus the finite mask
    gap = mine['gap']
    # ---------- module, end to end on its own readout
    s_m, p_m = Gm.linear_lsml(V, anchor=None if arm == 'AUTO' else V.mean(1))
    L.num(name, 'fused (linear_lsml)', mine['fused'], s_m)
    L.num(name, 'fused_linear(raw)', mine['rr']['lin'], Gm.fused_linear(Z0, p_m))
    gs_m = np.asarray(Gm.group_scores(Z0, p_m), float)
    D_m = np.asarray(Gm.group_defined(fin, p_m), bool)
    L.exact(name, 'group_defined', mine['rr']['defined'], D_m)
    o_m = np.asarray(Gm.group_orientation(p_m), float)
    L.num(name, 'group_orientation', mine['rr']['orient'], o_m)
    # The spec defines u_g = o_g * pooled-z(Z_g w_g). The module's group_scores() returns Z_g w_g itself (neither
    # pooled-z'd nor oriented); its gates take (group_scores, orientation). Compare it with Z_g w_g, and compare
    # u_g through the gates' outputs (null fits per group, P(err), votes), which the module computes internally.
    L.num(name, 'group_scores (= Z_g w_g, un-normalized)', mine['rr']['g_lin'], gs_m)
    u_from_m = np.zeros_like(gs_m)
    for g in range(gs_m.shape[1]):
        u_from_m[D_m[:, g], g] = o_m[g] * Gm.pooled_z(gs_m[D_m[:, g], g])
    L.num(name, 'u_g (defined steps) from module group_scores', mine['rr']['u'][mine['rr']['defined']],
          u_from_m[mine['rr']['defined']] if D_m.shape == mine['rr']['defined'].shape else np.nan)
    raw_m = Gm.pooled_z(Gm.fused_linear(Z0, p_m))
    L.num(name, 'raw_fused', mine['rr']['raw_fused'], raw_m)
    # ---------- empirical null on raw_fused and on each u_g (module fed the same vector as mine)
    f_mine = mine['G1']['fit']; f_mod = Gm.empirical_null(mine['rr']['raw_fused'])
    L.exact(name, 'null(raw_fused).ok', [f_mine['ok']], [f_mod['ok']])
    for k in ('pi0_raw', 'pi0', 'mu0', 'sigma0'):
        L.num(name, f'null(raw_fused).{k}', f_mine[k], f_mod[k])
    for g, fg in mine['G2']['fits'].items():
        ug = mine['rr']['u'][mine['rr']['defined'][:, g], g]           # defined steps only (3.2a)
        if len(ug) < MIN_DEFINED:
            continue                     # dropped by the gate rule, not by 3.3; checked through the gate summaries
        fm = Gm.empirical_null(ug)
        L.exact(name, f'null(u_{g}).ok', [fg['ok']], [fm['ok']])
        for k in ('pi0_raw', 'pi0', 'mu0', 'sigma0'):
            L.num(name, f'null(u_{g}).{k}', fg[k], fm[k])
        if fg['ok'] and fm['ok']:
            L.num(name, f'p_error(u_{g})', fg['p_err'](ug), Gm.p_error(ug, fm))
    # ---------- gates: module on its own readout (raw_m, gs_m, o_m); mine on mine
    m1 = Gm.gate_g1(raw_m, s_m, off)
    L.exact(name, 'G1.clean', mine['G1']['clean'], m1['clean'])
    L.num(name, 'G1.p_err_step', mine['G1']['p_err_step'], m1['p_err_step'])
    L.exact(name, 'G1.worst_step', worst_step(mine['fused'], off), Gm.worst_step(s_m, off))
    m1b = Gm.gate_g1b(raw_m, s_m, off, gap)
    L.exact(name, 'G1b.clean', mine['G1b']['clean'], m1b['clean'])
    L.num(name, 'G1b.p_answer', mine['G1b']['p_answer'], m1b['p_answer'])
    L.exact(name, 'G1b.n_ref', [mine['G1b']['n_ref']], [m1b['n_ref']])
    m2 = Gm.gate_g2(gs_m, o_m, off, defined=D_m)
    L.exact(name, 'G2.ok', [mine['G2']['ok']], [m2['ok']])
    L.exact(name, 'G2.clean', mine['G2']['clean'], m2['clean'])
    L.num(name, 'G2.p_err_step', mine['G2']['p_err_step'], m2['p_err_step'])
    # per-group summaries reported by the module's gates (the FITS.json diagnostics): ok / null / share defined
    for gate_name, mg in (('G2', m2),):
        summ = {int(d['group']): d for d in mg.get('groups', [])}
        L.exact(name, f'{gate_name}.groups summarized', sorted(mine['G2']['fits']), sorted(summ))
        for g, fg in mine['G2']['fits'].items():
            if g not in summ:
                continue
            L.exact(name, f'{gate_name}.group.ok', [fg['ok']], [bool(summ[g]['ok'])])
            for k in ('pi0_raw', 'pi0', 'mu0', 'sigma0'):
                L.num(name, f'{gate_name}.group.{k}', fg[k], parse_float(summ[g][k]))
            if 'share_defined' in summ[g]:
                L.num(name, f'{gate_name}.group.share_defined', mine['rr']['defined'][:, g].mean(),
                      summ[g]['share_defined'])
    if mine['G2']['ok'] and m2['ok']:
        L.num(name, 'G2.pi0', mine['G2']['pi0'], m2['pi0'])
        L.num(name, 'G2.pi0_groups', mine['G2']['pi0_groups'], m2['pi0_groups'])
        L.exact(name, 'G2.groups_used', mine['G2']['used'], m2['groups_used'])
    ag = prm_group(mine['ro']) if arm == 'AUTO' else None
    m3 = Gm.gate_g3(gs_m, o_m, off, anchor_group=ag, defined=D_m)
    L.exact(name, 'G3.ok', [mine['G3']['ok']], [m3['ok']])
    L.exact(name, 'G3.clean', mine['G3']['clean'], m3['clean'])
    if mine['G3']['ok'] and m3['ok']:
        L.num(name, 'G3.p_answer', mine['G3']['p_answer'], m3['p_answer'])
        for k in ('prevalence', 'psi', 'eta', 'vote_rate', 'sml_ranking'):
            L.num(name, f'G3.{k}', mine['G3'][k], m3[k])
        Bv = votes(mine['rr']['u'], mine['rr']['defined'], mine['G2']['fits'], mine['G3']['used'], off)
        L.num(name, 'G3.ds_answer_posterior', ds_posterior(Bv, mine['G3']), Gm.ds_answer_posterior(Bv, mine['G3']))
    return mine


def run_synthetic():
    from spectral_utils import raw_null_gate as Gm          # black box: called, never read
    L = Ledger(); summary = []
    cells = [
        ('TELE16_strong', dict(seed=11), 'TELE16'),
        ('AUTO_strong', dict(seed=12, with_prm=True), 'AUTO'),
        ('TELE16_constant_channel', dict(seed=13, constant_col=True), 'TELE16'),
        ('TELE16_inverted_group', dict(seed=14, invert_group=2), 'TELE16'),
        ('AUTO_inverted_group', dict(seed=15, with_prm=True, invert_group=1), 'AUTO'),
        ('TELE16_weak', dict(seed=16, d=(1.2, 1.0, .9, .8)), 'TELE16'),
        ('AUTO_small', dict(seed=17, n_ans=120, with_prm=True), 'AUTO'),
        ('TELE16_bimodal_group', dict(seed=18, bimodal_group=3), 'TELE16'),
        ('TELE16_two_groups', dict(seed=19, groups=(5, 5), d=(2.8, 2.4)), 'TELE16'),
        ('TELE16_bimodal_all', dict(seed=20, bimodal_all=True), 'TELE16'),
        ('AUTO_bimodal_all', dict(seed=21, with_prm=True, bimodal_all=True), 'AUTO'),
        ('TELE16_undefined_group', dict(seed=22, undef_group=3), 'TELE16'),
        ('AUTO_undefined_group', dict(seed=23, with_prm=True, undef_group=2), 'AUTO'),
        ('TELE16_undefined_one_channel', dict(seed=24, undef_group=0, undef_channels=[1]), 'TELE16'),
        ('TELE16_undefined_weak', dict(seed=25, undef_group=3, d=(1.2, 1.0, .9, .8)), 'TELE16'),
        ('TELE16_group_under_50_defined', dict(seed=26, undef_group=3, undef_frac=.997), 'TELE16'),
        ('AUTO_group_under_50_defined', dict(seed=27, with_prm=True, undef_group=2, undef_frac=.997), 'AUTO'),
    ]

    for name, kw, arm in cells:
        t = time.time()
        X, off, y, has = synth_cell(**kw)
        mine = compare_synthetic(L, name, X, off, arm, Gm)
        rates = {g: (round(float(mine[g]['clean'][~has].mean()), 3), round(float((~mine[g]['clean'][has]).mean()), 3))
                 for g in GATES}
        summary.append(dict(cell=name, arm=arm, n_ans=len(off) - 1, K=mine['ro']['K'],
                            orient=mine['rr']['orient'].tolist(), seconds=round(time.time() - t, 1),
                            valid={G: bool(mine[G]['ok']) for G in GATES}, G2_groups_used=mine['G2']['used'],
                            G3_groups_used=mine['G3']['used'],
                            undefined_steps_per_group=(~mine['rr']['defined']).sum(0).tolist(),
                            closed_clean_kept_error=rates, G1_fit=fit_public(mine['G1']['fit'])))
    # direct empirical-null checks on crafted vectors, including failure paths
    rng = np.random.default_rng(99)
    crafted = {
        'two_groups_N(0,1)+N(3,1)': np.r_[rng.normal(0, 1, 42_500), rng.normal(3, 1, 7_500)],
        'bimodal_dip(+-2.5)': np.r_[rng.normal(-2.5, .6, 20_000), rng.normal(2.5, .6, 20_000)],
        'heavy_tail_t3': rng.standard_t(3, 30_000),
        'few_values': np.r_[np.zeros(500), np.ones(500), 2 * np.ones(10)],
        'tiny_N': rng.normal(0, 1, 40),
    }
    spec_gaps = []     # none since the amendment 'IRLS not converged in 100 iterations -> 3.3 fails'
    for nm, x in crafted.items():
        fm = empirical_null(x); fo = Gm.empirical_null(x)
        L.exact('crafted:' + nm, 'ok', [fm['ok']], [fo['ok']])
        L.exact('crafted:' + nm, 'reason is IRLS non-convergence', [fm['reason'] == 'IRLS did not converge'],
                [fo.get('reason') == 'IRLS did not converge'])
        if fm['ok'] and fo['ok']:
            for k in ('pi0_raw', 'pi0', 'mu0', 'sigma0'):
                L.num('crafted:' + nm, k, fm[k], fo[k])
            L.num('crafted:' + nm, 'p_error', fm['p_err'](x), Gm.p_error(x, fo))
        summary.append(dict(cell='crafted:' + nm, mine=fit_public(fm),
                            module={k: fo.get(k) for k in ('ok', 'reason', 'pi0_raw', 'mu0', 'sigma0')}))
    # report
    for s in summary:
        print(json.dumps(s, default=float))
    print()
    by_name = {}
    for r in L.rows:
        k = r['name'].split('[')[0]
        cur = by_name.setdefault(k, dict(max_diff=0., ok=True, n=0, kind=r.get('note', 'float')))
        cur['n'] += 1; cur['ok'] &= r['ok']
        if r['max_diff'] is not None:
            cur['max_diff'] = max(cur['max_diff'], float(r['max_diff']))
    print(f"{'quantity':45s} {'checks':>6s} {'max diff':>12s}  result")
    for k, v in by_name.items():
        md = f"{v['max_diff']:.3g}" if v['kind'] != 'exact' else f"{int(v['max_diff'])} diff"
        print(f"{k:45s} {v['n']:6d} {md:>12s}  {'PASS' if v['ok'] else 'FAIL'}")
    fails = L.failed
    print()
    for g in spec_gaps:
        print('SPEC GAP (not counted as FAIL):', json.dumps(g, default=float))
    for r in fails:
        print('FAIL', json.dumps(r, default=float))
    print('SYNTHETIC', 'PASS' if not fails else 'FAIL', f'({len(L.rows)} comparisons, {len(fails)} failed, '
          f'{len(spec_gaps)} spec gap(s) reported above)')
    return 0 if not fails else 1


# ================================================================== real mode
ARM_MODES = [('AUTO', 'backbone'), ('TELE16', 'backbone'), ('TELE16', 'standalone'), ('SUB5', 'backbone'),
             ('SUB5', 'standalone')]
BENCHES = ['prmbench', 'processbench', 'crv_arith', 'crv_gsm8k']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
MODE_ALIASES = {'backbone': 'backbone', 'prm': 'backbone', 'bb': 'backbone',
                'standalone': 'standalone', 'sa': 'standalone'}
LOADER_MODE = {'backbone': 'prm', 'standalone': 'sa'}
ENDPOINT_ALIASES = {
    'closed_clean': ['closed_clean'],
    'kept_error': ['kept_error'],
    'share_called_error': ['share_called_error'],
    'true_share_error': ['true_share_error'],
    'metric': ['metric', 'metric_gated', 'gated_metric'],
    'metric_gap_only': ['metric_gap_only', 'gap_only', 'metric_gap', 'gap_only_metric', 'metric_GAP_ONLY'],
    'ece_step': ['ece_step', 'ece'],
    'valid': ['valid', 'gate_valid', 'ok'],
}
RATE_FIELDS = ('closed_clean', 'kept_error', 'share_called_error')
FLOAT_FIELDS = ('metric', 'metric_gap_only', 'ece_step', 'true_share_error')


class Refuse(Exception):
    pass


def die(msg):
    raise Refuse(msg)


def norm_mode(m):
    m = str(m).strip().lower()
    if m not in MODE_ALIASES:
        die(f'unknown mode value {m!r}; expected one of {sorted(MODE_ALIASES)}')
    return MODE_ALIASES[m]


RESULT_GATES = GATES + ('GAP_ONLY',)   # the runner also writes a plain-GAP reference row per cell


def norm_gate(g):
    g = str(g).strip().lower()
    for G in RESULT_GATES:
        if g == G.lower():
            return G
    die(f'unknown gate value {g!r}; expected one of {RESULT_GATES}')


def cell_from_text(s):
    """Find (arm, mode, bench, prm) in a free-form key such as 'TELE16|sa|processbench|pure_prm'. None if absent."""
    t = '|' + re.sub(r'[^A-Za-z0-9_]+', '|', str(s)) + '|'
    low = t.lower()
    arm = next((a for a in ('TELE16', 'SUB5', 'AUTO') if re.search(rf'(?<![A-Za-z0-9]){a}(?![A-Za-z0-9])', t,
                                                                   re.I)), None)
    bench = next((b for b in sorted(BENCHES, key=len, reverse=True) if re.search(rf'(?<![a-z0-9]){b}(?![a-z0-9])',
                                                                                  low)), None)
    prm = next((p for p in sorted(PRMS, key=len, reverse=True) if re.search(rf'(?<![a-z0-9]){p}(?![a-z0-9])', low)),
               None)
    mode = None
    if arm:
        mm = re.search(rf'(?<![a-z0-9]){arm.lower()}[|_](backbone|standalone|prm|sa|bb)(?![a-z0-9])', low)
        mode = MODE_ALIASES[mm.group(1)] if mm else None
    if mode is None:
        mode = next((MODE_ALIASES[m] for m in ('backbone', 'standalone', 'sa', 'bb')
                     if re.search(rf'[|_]{m}[|_]', low)), None)
    if mode is None and arm == 'AUTO':
        mode = 'backbone'
    if arm and bench and prm and mode:
        return (arm, mode, bench, prm)
    return None


def cell_from_dict(d):
    if not isinstance(d, dict):
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    arm = low.get('arm'); bench = low.get('benchmark', low.get('bench')); prm = low.get('prm')
    mode = low.get('mode', 'backbone' if str(arm).upper() == 'AUTO' else None)
    if arm is None or bench is None or prm is None or mode is None:
        return None
    return (str(arm).upper(), norm_mode(mode), str(bench), str(prm))


def ok_cells_from_check(chk):
    """CHECK.json: accept {key: status}, {key: {status|ok}}, [{arm, mode, bench, prm, status|ok}], or a dict holding
    one such container under 'cells'. 'ok' = status string 'ok' or ok == True."""
    if isinstance(chk, dict) and 'cells' in chk:
        chk = chk['cells']
    status = {}

    def st(v):
        if isinstance(v, str):
            return v.strip().lower()
        if isinstance(v, bool):
            return 'ok' if v else 'not_ok'
        if isinstance(v, dict):
            low = {str(k).lower(): x for k, x in v.items()}
            if 'status' in low: return str(low['status']).strip().lower()
            if 'ok' in low: return 'ok' if low['ok'] is True else 'not_ok'
        return None
    if isinstance(chk, dict):
        for k, v in chk.items():
            cell = cell_from_dict(v) if isinstance(v, dict) else None
            cell = cell or cell_from_text(k)
            if cell is None:
                die(f'CHECK.json: cannot identify the cell of key {k!r}')
            s = st(v)
            if s is None:
                die(f'CHECK.json: cannot read the status of {k!r}: {str(v)[:200]}')
            status[cell] = s
    elif isinstance(chk, list):
        for v in chk:
            cell = cell_from_dict(v)
            if cell is None:
                die(f'CHECK.json: list entry without arm/mode/benchmark/prm: {str(v)[:200]}')
            s = st(v)
            if s is None:
                die(f'CHECK.json: entry without status/ok: {str(v)[:200]}')
            status[cell] = s
    else:
        die(f'CHECK.json: unsupported top-level type {type(chk).__name__}')
    if not status:
        die('CHECK.json lists no cells')
    return status


def find_col(cols, field, gate=None):
    """Column of `field` (long format) or `<gate>_<field>` / `<field>_<gate>` (wide format); None if absent."""
    low = {c.lower(): c for c in cols}
    for a in ENDPOINT_ALIASES[field]:
        cands = [a] if gate is None else [f'{gate}_{a}', f'{a}_{gate}', f'{gate}.{a}', f'{a}.{gate}']
        for cnd in cands:
            if cnd.lower() in low:
                return low[cnd.lower()]
    return None


def parse_float(s):
    if s is None: return np.nan
    s = str(s).strip()
    if s == '' or s.lower() in ('nan', 'none', 'null'): return np.nan
    if s.lower() in ('true', 'false'): return float(s.lower() == 'true')
    return float(s)


def sig_digits(s):
    s = str(s).strip().lower().lstrip('-+')
    mant = s.split('e')[0].replace('.', '').lstrip('0')
    return len(mant)


def read_results(path):
    with open(path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        die(f'{path} has no rows')
    cols = list(rows[0].keys())
    low = {c.lower(): c for c in cols}
    for need in ('arm', 'prm'):
        if need not in low:
            die(f'{path}: no {need!r} column; columns are {cols}')
    bcol = low.get('benchmark', low.get('bench'))
    if bcol is None:
        die(f'{path}: no benchmark/bench column; columns are {cols}')
    mcol = low.get('mode')
    gcol = low.get('gate')
    out = {}                    # (arm, mode, bench, prm, gate) -> {field: (raw string, column)}
    for r in rows:
        arm = r[low['arm']].strip().upper()
        mode = norm_mode(r[mcol]) if mcol else ('backbone' if arm == 'AUTO' else die(
            f'{path}: no mode column and arm {arm} is not AUTO'))
        key = (arm, mode, r[bcol].strip(), r[low['prm']].strip())
        gates = [norm_gate(r[gcol])] if gcol else list(GATES)
        for G in gates:
            rec = {}
            for f in ENDPOINT_ALIASES:
                c = find_col(cols, f) if gcol else find_col(cols, f, G)
                if c is not None:
                    rec[f] = (r[c], c)
            rec['_status'] = r.get(low.get('status', ''), '') if 'status' in low else ''
            k = key + (G,)
            if k in out:
                die(f'{path}: duplicate row for {k}')
            out[k] = rec
    return out, cols


def fits_by_cell(fits):
    """FITS.json -> {cell: sub-object}. Accepts a list of dicts with cell fields, or a dict keyed by cell strings."""
    out = {}
    if isinstance(fits, dict) and 'cells' in fits:
        fits = fits['cells']
    items = fits.items() if isinstance(fits, dict) else enumerate(fits) if isinstance(fits, list) else None
    if items is None:
        die(f'FITS.json: unsupported top-level type {type(fits).__name__}')
    for k, v in items:
        cell = cell_from_dict(v) if isinstance(v, dict) else None
        if cell is None and isinstance(k, str):
            cell = cell_from_text(k)
        if cell is None:
            die(f'FITS.json: cannot identify the cell of entry {k!r}: {str(v)[:200]}')
        if cell in out:
            die(f'FITS.json: two entries for cell {cell}')
        out[cell] = v
    return out


NULL_FIELDS = ('pi0', 'pi0_raw', 'mu0', 'sigma0')
GATE_SCALARS = {'G2': ('pi0',), 'G3': ('prevalence',)}
GATE_VECTORS = {'G2': ('pi0_groups',), 'G3': ('psi', 'eta', 'sml_ranking', 'vote_rate')}


def gate_of(path):
    """Gate named by the path (G1b before G1); None if no gate token. 'fused' / 'raw_fused' count as G1's fit."""
    toks = []
    for q in path:
        if isinstance(q, str):
            toks += [t for t in re.split(r'[^a-z0-9]+', q.lower()) if t]
    for want, names in (('G1b', ('g1b',)), ('G1', ('g1', 'fused')), ('G2', ('g2',)), ('G3', ('g3',))):
        if any(t in names for t in toks):
            return want
    return None


def _num_close(L, ctx, name, mine, theirs, note=''):
    L.num(ctx, name, mine, theirs)
    L.rows[-1].update(mine=np.asarray(mine, float).tolist(), theirs=np.asarray(theirs, float).tolist(), note=note)


def compare_fits(L, ctx, entry, lf):
    """Walk the cell's FITS.json entry. Every dict holding pi0/mu0/sigma0 is a null fit: it is matched to the G1 fit
    (gate G1 / fused, or no gate in the path) or to group g's fit (G2/G3; g from a 'group' field, else from the list
    index, tried both as group id and as position among used groups). Also checks the G2 pooled pi0 / pi0_groups and
    the G3 prevalence / psi / eta / SML ranking / vote rates when present. Unmappable null fits are FAILs."""
    n = [0]
    fits = lf['G2']['fits']; used = lf['G2']['used']

    def check_fit(path, d, mine_fit, label):
        for k in NULL_FIELDS:
            if k in d:
                _num_close(L, ctx, f'FITS:{"/".join(map(str, path + (k,)))}', mine_fit[k], parse_float(d[k]), label)
                n[0] += 1
        if 'ok' in d:
            L.exact(ctx, f'FITS:{"/".join(map(str, path + ("ok",)))}', [mine_fit['ok']], [bool(d['ok'])]); n[0] += 1

    def walk(obj, path):
        if isinstance(obj, dict):
            keys = {str(k): k for k in obj}
            gate = gate_of(path)
            if any(k in keys for k in ('mu0', 'sigma0')) or ('pi0' in keys and gate in (None, 'G1')):
                if gate in (None, 'G1'):
                    check_fit(path, obj, lf['G1']['fit'], 'G1 fused fit')
                elif gate == 'G1b':
                    L.note(ctx, f'FITS:{"/".join(map(str, path))}', False, 'G1b has no empirical-null fit in the spec')
                else:
                    gid = obj.get('group')
                    cands = []
                    if gid is not None:
                        cands = [(f'group id {int(gid)}', int(gid))]
                    else:
                        ints = [q for q in path if isinstance(q, int)] + \
                               [int(q) for q in path if isinstance(q, str) and re.fullmatch(r'\d+', q)]
                        if ints:
                            i = ints[-1]
                            cands = [(f'group id {i}', i)] + ([(f'used position {i} = group {used[i]}', used[i])]
                                                              if i < len(used) else [])
                    cands = [(lab, g) for lab, g in cands if g in fits]
                    if not cands:
                        L.note(ctx, f'FITS:{"/".join(map(str, path))}', False,
                               f'null fit under {gate} that cannot be mapped to a group of this verifier')
                    else:
                        def dist(g):
                            return max((abs(fits[g][k] - parse_float(obj[k])) if np.isfinite(fits[g][k])
                                        else 0. if np.isnan(parse_float(obj[k])) else np.inf)
                                       for k in NULL_FIELDS if k in obj) if any(k in obj for k in NULL_FIELDS) else 0.
                        lab, g = min(cands, key=lambda c: dist(c[1]))
                        check_fit(path, obj, fits[g], lab)
                        if 'share_defined' in obj:
                            _num_close(L, ctx, f'FITS:{"/".join(map(str, path + ("share_defined",)))}',
                                       float(lf['rr']['defined'][:, g].mean()), parse_float(obj['share_defined']), lab)
                return
            gate = gate_of(path)
            for k in ('orientation', 'orientations', 'orient', 'group_orientation'):
                if k in keys and isinstance(obj[keys[k]], list):
                    _num_close(L, ctx, f'FITS:{"/".join(map(str, path + (k,)))}', lf['rr']['orient'],
                               [parse_float(v) for v in obj[keys[k]]]); n[0] += 1
            for k in ('cross_weights', 'cw'):
                if k in keys and isinstance(obj[keys[k]], list):
                    _num_close(L, ctx, f'FITS:{"/".join(map(str, path + (k,)))}', lf['ro']['cw'],
                               [parse_float(v) for v in obj[keys[k]]]); n[0] += 1
            if gate in ('G2', 'G3') and 'groups_used' in keys and isinstance(obj[keys['groups_used']], list):
                mine = lf[gate]['used']; theirs = [int(v) for v in obj[keys['groups_used']]]
                L.rows.append(dict(ctx=ctx, name=f'FITS:{"/".join(map(str, path + ("groups_used",)))}',
                                   ok=list(mine) == theirs, mine=list(mine), theirs=theirs, max_diff=None, tol=0,
                                   note='exact')); n[0] += 1
            if gate == 'G3' and 'anchor_used' in keys and not isinstance(obj[keys['anchor_used']], (dict, list)):
                mine = lf['G3'].get('anchor_used') is not None; theirs = obj[keys['anchor_used']]
                theirs = bool(theirs) if isinstance(theirs, bool) else theirs is not None and theirs is not False
                L.exact(ctx, f'FITS:{"/".join(map(str, path + ("anchor_used",)))}', [mine], [theirs]); n[0] += 1
            if gate in GATE_SCALARS:
                gk = lf[gate]
                for k in GATE_SCALARS[gate]:
                    if k in keys and not isinstance(obj[keys[k]], (dict, list)):
                        mine = gk.get(k, np.nan) if gk['ok'] else np.nan
                        _num_close(L, ctx, f'FITS:{"/".join(map(str, path + (k,)))}', mine, parse_float(obj[keys[k]]))
                        n[0] += 1
                for k in GATE_VECTORS[gate]:
                    if k in keys and isinstance(obj[keys[k]], list):
                        mine = gk.get(k) if gk['ok'] else None
                        theirs = [parse_float(v) for v in obj[keys[k]]]
                        if mine is None:
                            L.note(ctx, f'FITS:{"/".join(map(str, path + (k,)))}', len(theirs) == 0,
                                   'this verifier has no value (gate invalid)')
                        else:
                            _num_close(L, ctx, f'FITS:{"/".join(map(str, path + (k,)))}', mine, theirs)
                        n[0] += 1
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    walk(v, path + (str(k),))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    walk(v, path + (i,))

    # flat per-gate keys written by the runner (e.g. g2_pi0, g2_pi0_groups, g3_prevalence, g3_anchor_used): regroup
    # them as {'G2': {'pi0': ...}} so the walker sees the gate in the path
    if isinstance(entry, dict):
        regrouped, rest = {}, {}
        for k, v in entry.items():
            m = re.fullmatch(r'(g1b|g1|g2|g3)_(.+)', str(k), re.I)
            if m and m.group(2).lower() != 'fit':
                regrouped.setdefault(m.group(1).upper().replace('B', 'b'), {})[m.group(2).lower()] = v
            else:
                rest[k] = v
        for G, d in regrouped.items():
            if 'groups' in d and isinstance(d['groups'], list):
                d['groups_fits'] = d.pop('groups')      # list of per-group fit dicts
            rest[G] = d
        entry = rest
        # cell-level readout diagnostics
        if 'K' in entry:
            L.exact(ctx, 'FITS:K', [lf['ro']['K']], [int(entry['K'])]); n[0] += 1
        if 'sign' in entry:
            _num_close(L, ctx, 'FITS:sign', lf['ro']['sigma'], parse_float(entry['sign'])); n[0] += 1
        if 'groups' in entry and isinstance(entry['groups'], list):
            L.rows.append(dict(ctx=ctx, name='FITS:groups (L-SML partition c)', ok=lf['ro']['c'].tolist() ==
                               [int(v) for v in entry['groups']], mine=lf['ro']['c'].tolist(),
                               theirs=entry['groups'], max_diff=None, tol=0, note='exact')); n[0] += 1
        if 'share_defined' in entry and isinstance(entry['share_defined'], list):
            _num_close(L, ctx, 'FITS:share_defined', lf['rr']['defined'].mean(0),
                       [parse_float(v) for v in entry['share_defined']]); n[0] += 1
        if 'anchor_group' in entry:
            mine = prm_group(lf['ro']) if lf.get('arm') == 'AUTO' else None
            theirs = entry['anchor_group']
            L.rows.append(dict(ctx=ctx, name='FITS:anchor_group', ok=mine == (None if theirs is None else int(theirs)),
                               mine=mine, theirs=theirs, max_diff=None, tol=0, note='exact')); n[0] += 1
        if 'parity_linear' in entry:
            L.rows.append(dict(ctx=ctx, name='FITS:parity_linear <= 1e-6', ok=parse_float(entry['parity_linear'])
                               <= FLOAT_TOL, theirs=entry['parity_linear'], max_diff=None, tol=FLOAT_TOL)); n[0] += 1
    walk(entry, ())
    if n[0] == 0:
        L.note(ctx, 'FITS:entry', False, "no pi0/mu0/sigma0 found in this cell's FITS.json entry")
    return n[0]


def ece_equal_count(p, y, nb=10):
    p = np.asarray(p, float); y = np.asarray(y, float)
    if len(p) == 0 or not np.all(np.isfinite(p)):
        return np.nan
    idx = np.empty(len(p), int); idx[np.argsort(p, kind='stable')] = np.arange(len(p)) * nb // len(p)
    return float(sum(abs(p[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean() for k in range(nb)
                     if (idx == k).any()))


def real_imports():
    for sub in ('bracha_lr_transfer_v1', 'ds_gate_e1b_fuse', 'ds_gate_v1b_fixed', 'prm_boost_math_rigor_v1'):
        p = ROOT / 'results' / sub
        if not p.is_dir():
            die(f'loader directory missing: {p}')
        sys.path.insert(0, str(p))
    import lr_transfer as Lt
    import votes_e1b as Vb
    import run_e1 as R1
    import stage7_hygiene as S7
    return Lt, Vb, R1, S7


def real_cell(arm, mode, bench, prm, mods):
    Lt, Vb, R1, S7 = mods
    E, S8 = Lt.E, Lt.S8
    lm = LOADER_MODE[mode]
    c = Lt.load(bench, prm, lm)
    X = np.asarray(E.channels(c, 'N16'), float); R1.check_names(c)
    off = np.asarray(c.off, np.int64); n = np.diff(off); pool = np.repeat(n, n) > 1
    Xr, act = Vb.raw_matrix(lm, bench, prm, c)
    if arm == 'SUB5':
        idx = [R1.N16N.index(nm) for nm in R1.SUB5]
        X, Xr, act = X[:, idx], Xr[:, idx], act[:, idx]
    # amendment (d): every defined raw entry must be finite
    bad = ~np.isfinite(Xr) & act
    if bad.any():
        raise AssertionError(f'{int(bad.sum())} defined raw entries are not finite (columns '
                             f'{sorted(set(np.flatnonzero(bad.any(0)).tolist()))})')
    Zr = Vb.clean(Xr, act, off, pool)          # NaN = undefined; kept: amendment (a) needs the defined mask
    if np.isinf(Zr).any():
        raise AssertionError('votes_e1b.clean returned infinite values')
    par = {}
    if arm == 'AUTO':
        risk = np.asarray(c.arms['PRM_Q80' if bench == 'prmbench' else 'PRM'], float)
        zp = E.answer_z(risk, off)
        V = np.column_stack([zp, X])
        if not np.all(np.isfinite(risk)):
            raise AssertionError(f'{int((~np.isfinite(risk)).sum())} PRM risk values are not finite')
        zc = Vb.clean(risk[:, None], np.ones((len(risk), 1), bool), off, pool)[:, 0]
        Z = np.column_stack([zc, Zr])
        ref, _ = S8.auto(X, zp)
    else:
        V, Z = X, Zr
        ref, _ = S8.fit(X, anchor=X.mean(1))
    lf = label_free_stage(V, Z, off, arm, gap_fn=lambda s: S7.gap_flags(c, s))   # 3.5: stage7_hygiene.gap_flags
    par['stage8'] = float(np.max(np.abs(lf['fused'] - np.asarray(ref, float))))
    if (arm == 'AUTO' and mode == 'backbone') or (arm in ('TELE16', 'SUB5') and mode == 'standalone'):
        _, _, _, s_e1 = R1.arm_inputs(arm, bench, prm)
        par['run_e1.arm_inputs'] = float(np.max(np.abs(lf['fused'] - np.asarray(s_e1, float))))
    gap = lf['gap']
    # ---------------- labels from here on (definitions of run_e1.run_cell)
    keep_gold = c.keep_gold.astype(bool) if bench == 'processbench' else None
    keep_eval = keep_gold if bench == 'processbench' else np.asarray(c.keep, bool)
    err = np.asarray(c.err, bool)
    has_err = np.array([err[a:b][keep_eval[a:b]].any() for a, b in zip(off[:-1], off[1:])])
    ev = (np.asarray(c.cls) != 'correct') if bench == 'prmbench' else np.ones(len(n), bool)
    he = has_err[ev]
    stepmask = keep_eval & np.repeat(ev, n)
    gap_metric = R1.metric(c, bench, gap, keep_gold)
    ends = {}
    for G in GATES:
        g = lf[G]; cl = np.asarray(g['clean'], bool)[ev]
        e = dict(valid=bool(g['ok']),
                 closed_clean=float(cl[~he].mean()) if (~he).any() else np.nan,
                 kept_error=float((~cl[he]).mean()) if he.any() else np.nan,
                 share_called_error=float((~cl).mean()),
                 true_share_error=float(he.mean()),
                 metric=float(R1.metric(c, bench, gap & ~np.repeat(np.asarray(g['clean'], bool), n), keep_gold)),
                 metric_gap_only=float(gap_metric),
                 n_eval=int(ev.sum()), n_clean_eval=int((~he).sum()))
        if G in ('G1', 'G2'):
            e['ece_step'] = ece_equal_count(g['p_err_step'][stepmask], err[stepmask]) if g['ok'] else np.nan
        else:
            e['ece_step'] = np.nan
        ends[G] = e
    ends['GAP_ONLY'] = dict(valid=True, closed_clean=0. if (~he).any() else np.nan,
                            kept_error=1. if he.any() else np.nan, share_called_error=1., true_share_error=float(he.mean()),
                            metric=float(gap_metric), metric_gap_only=float(gap_metric), ece_step=np.nan)
    return lf, ends, par


def compare_npz(L, ctx, path, lf):
    """Compare arrays of labelfree/<cell>.npz whose names identify a quantity of this verifier; list the rest."""
    z = np.load(path, allow_pickle=False)
    checked, skipped = [], []
    mine_arrays = {'fused': lf['fused'], 'raw_fused': lf['rr']['raw_fused'], 'gap': lf['gap']}
    for G in GATES:
        for q in ('clean', 'p_err_step', 'p_answer', 'p_err_worst'):
            if q in lf[G] and lf[G][q] is not None:
                mine_arrays[f'{G}_{q}'] = lf[G][q]
    for G in GATES:                                          # runner-style names: clean_G1, p_err_step_G2, ...
        for q in ('clean', 'p_err_step', 'p_answer', 'p_err_worst'):
            if f'{G}_{q}' in mine_arrays:
                mine_arrays[f'{q}_{G}'] = mine_arrays[f'{G}_{q}']
    mine_arrays['z_fused'] = mine_arrays['fused']; mine_arrays['f_gap'] = mine_arrays['gap']
    for k in z.files:
        kl = k.lower(); hit = None
        for mk in mine_arrays:
            ml = mk.lower()
            if kl == ml or kl.replace('.', '_') == ml or kl == ml.replace('_', ''):
                hit = mk; break
        if hit is None:
            skipped.append(k); continue
        a = np.asarray(z[k])
        if 'clean' in hit or hit in ('gap', 'f_gap'):
            L.exact(ctx, f'npz:{k}', mine_arrays[hit], a)
        else:
            L.num(ctx, f'npz:{k}', mine_arrays[hit], a)
        checked.append(k)
    return checked, skipped


def run_real(here=HERE, mods=None):
    """here: directory holding RESULTS.csv / FITS.json / CHECK.json / labelfree/; mods: loader modules (tests only)."""
    here = Path(here)
    res_p, fits_p, chk_p = here / 'RESULTS.csv', here / 'FITS.json', here / 'CHECK.json'
    for p in (res_p, fits_p, chk_p):
        if not p.exists():
            die(f'{p} is missing. Run `run_raw_null_gate.py run` first; this verifier refuses to run without it.')
    results, res_cols = read_results(res_p)
    fits = fits_by_cell(json.loads(fits_p.read_text(encoding='utf-8')))
    status = ok_cells_from_check(json.loads(chk_p.read_text(encoding='utf-8')))
    ok_cells = sorted(k for k, s in status.items() if s == 'ok')
    not_ok = {('|'.join(k)): s for k, s in status.items() if s != 'ok'}
    if not ok_cells:
        die('CHECK.json lists no cell with status ok')
    L0 = []          # grid-coverage findings (Section 6: missing data must be exactly the 12 PRMBench backbone cells)
    grid = {(a, m, b, p) for a, m in ARM_MODES for b in BENCHES for p in PRMS}
    expected_missing = {(a, 'backbone', 'prmbench', p) for a in ('AUTO', 'TELE16', 'SUB5') for p in PRMS}
    absent = sorted(grid - set(status))
    if absent:
        L0.append(dict(ctx='CHECK.json', name='grid coverage', ok=False, max_diff=None, tol=None,
                       note=f'{len(absent)} of {len(grid)} pre-registered cells absent: {["|".join(c) for c in absent]}'))
    for cell, st_ in sorted(status.items()):
        if st_ != 'ok' and cell not in expected_missing:
            L0.append(dict(ctx='|'.join(cell), name='CHECK status', ok=False, max_diff=None, tol=None,
                           note=f'status {st_!r}: only the 12 PRMBench backbone cells may be not ok'))
        elif st_ != 'ok' and st_ != 'missing':
            L0.append(dict(ctx='|'.join(cell), name='CHECK status', ok=False, max_diff=None, tol=None,
                           note=f'status {st_!r} (expected "missing" for an expected-missing cell)'))
    unknown = [k for k in ok_cells if (k[0], k[1]) not in ARM_MODES or k[2] not in BENCHES or k[3] not in PRMS]
    if unknown:
        die(f'CHECK.json has ok cells outside the pre-registered grid: {unknown}')
    mods = mods or real_imports()
    L = Ledger(); L.rows.extend(L0); per_cell = []; warnings = []; t0 = time.time()
    # every RESULTS row of an ok cell must be verified; rows of other cells are listed, never silently ignored
    rows_not_ok = sorted('|'.join(k) for k in results if k[:4] not in set(ok_cells))
    for cell in ok_cells:
        ctx = '|'.join(cell); t = time.time()
        print('cell', ctx, flush=True)
        entry = dict(cell=ctx)
        try:
            lf, ends, par = real_cell(*cell, mods)
        except Exception as ex:
            L.note(ctx, 'recompute', False, f'{type(ex).__name__}: {ex}')
            entry.update(error=traceback.format_exc()); per_cell.append(entry); continue
        for k, v in par.items():
            L.num(ctx, f'parity fused vs {k}', 0., v)
        L.num(ctx, 'linear readout rebuilds fit', 0., lf['ro']['parity'])
        for G in RESULT_GATES:
            key = cell + (G,)
            if G == 'GAP_ONLY' and key not in results:
                continue                 # optional reference row; the four gate rows are mandatory
            if key not in results:
                L.note(ctx, f'{G}: RESULTS row', False, 'cell is ok in CHECK.json but RESULTS.csv has no row')
                continue
            rec = results[key]
            for f in RATE_FIELDS:
                if f not in rec:
                    L.note(ctx, f'{G}.{f}', False, f'column for {f} not found in RESULTS.csv ({res_cols})'); continue
                s, col = rec[f]; theirs = parse_float(s); mine = ends[G][f]
                ok = (np.isnan(mine) and np.isnan(theirs)) or mine == theirs
                note = 'exact'
                if not ok and np.isfinite(theirs) and sig_digits(s) >= 6 and \
                        float(f'{mine:.{sig_digits(s)}g}') == float(f'{theirs:.{sig_digits(s)}g}'):
                    ok, note = True, f'equal at the CSV precision ({sig_digits(s)} significant digits)'
                L.rows.append(dict(ctx=ctx, name=f'{G}.{f}', ok=bool(ok), mine=mine, theirs=theirs,
                                   max_diff=float(abs(mine - theirs)) if np.isfinite(mine - theirs) else None,
                                   tol=0, note=note))
            for f in FLOAT_FIELDS:
                if f not in rec:
                    if f in ('metric',) or (f == 'ece_step' and G in ('G1', 'G2')):
                        L.note(ctx, f'{G}.{f}', False, f'column for {f} not found in RESULTS.csv ({res_cols})')
                    continue
                theirs = parse_float(rec[f][0]); mine = ends[G][f]
                if G == 'GAP_ONLY' and f == 'true_share_error' and not np.isfinite(theirs):
                    continue             # the reference row leaves it blank; it is checked on every gate row
                if f == 'ece_step' and G not in ('G1', 'G2'):
                    if np.isfinite(theirs):
                        L.note(ctx, f'{G}.ece_step', False, f'ece_step {theirs} reported for a gate without P(err)')
                    continue
                L.num(ctx, f'{G}.{f}', mine, theirs)
                L.rows[-1].update(mine=mine, theirs=theirs)
            if 'valid' in rec and str(rec['valid'][0]).strip() != '':
                L.exact(ctx, f'{G}.valid', [ends[G]['valid']], [bool(parse_float(rec['valid'][0]))])
            elif G != 'GAP_ONLY':
                L.note(ctx, f'{G}.valid', False, 'no gate-validity value (valid/ok column) in RESULTS.csv')
        if cell in fits:
            compare_fits(L, ctx, fits[cell], lf)
        else:
            L.note(ctx, 'FITS entry', False, 'cell is ok in CHECK.json but has no FITS.json entry')
        npzs = [p for p in (here / 'labelfree').glob('*.npz') if cell_from_text(p.stem) == cell] \
            if (here / 'labelfree').is_dir() else []
        if len(npzs) == 1:
            chk, skp = compare_npz(L, ctx, npzs[0], lf)
            entry.update(npz=str(npzs[0].name), npz_checked=chk, npz_not_checked=skp)
        else:
            L.note(ctx, 'labelfree npz', False, f'expected one labelfree/*.npz for this cell, found {len(npzs)}')
        nonconv = ([] if lf['G1']['fit'].get('irls_converged', True) else ['G1 fused']) + \
            [f'group {g}' for g, f in lf['G2']['fits'].items() if not f.get('irls_converged', True)]
        if nonconv:
            warnings.append(dict(cell=ctx, warning='IRLS did not converge in 100 iterations, so 3.3 failed for these '
                                                   'fits (amended 3.3.2)', fits=nonconv))
        entry.update(endpoints=ends, parity=par, K=lf['ro']['K'], orient=lf['rr']['orient'].tolist(),
                     share_defined_per_group=lf['rr']['defined'].mean(0).tolist(),
                     irls_not_converged=nonconv,
                     G1_fit=fit_public(lf['G1']['fit']),
                     G2=dict(ok=lf['G2']['ok'], pi0=lf['G2'].get('pi0'), pi0_groups=lf['G2'].get('pi0_groups'),
                             used=lf['G2']['used']),
                     G3={k: lf['G3'].get(k) for k in ('ok', 'prevalence', 'psi', 'eta', 'sml_ranking', 'used',
                                                       'anchor_used')},
                     seconds=round(time.time() - t, 1))
        per_cell.append(entry)
    fails = L.failed
    out = dict(mode='real', verdict='PASS' if not fails else 'FAIL', n_ok_cells=len(ok_cells),
               n_comparisons=len(L.rows), n_failed=len(fails), failures=fails,
               warnings=warnings, cells_not_ok_in_CHECK=not_ok, results_rows_of_cells_not_ok=rows_not_ok,
               results_columns=res_cols, comparisons=L.rows, cells=per_cell, seconds=round(time.time() - t0, 1),
               tolerances=dict(rates='exact (or equal at >= 6 significant CSV digits)', floats=FLOAT_TOL))
    (here / 'VERIFY.json').write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist')
                                                 else float(o) if isinstance(o, (np.floating, np.integer)) else str(o)),
                                      encoding='utf-8')
    for r in fails[:50]:
        print('FAIL', json.dumps(r, default=str))
    print('REAL', out['verdict'], f'({len(L.rows)} comparisons over {len(ok_cells)} cells, {len(fails)} failed)',
          '->', here / 'VERIFY.json')
    return 0 if not fails else 1


def main(argv):
    if len(argv) != 2 or argv[1] not in ('synthetic', 'real'):
        print(__doc__); return 2
    try:
        return run_synthetic() if argv[1] == 'synthetic' else run_real()
    except Refuse as ex:
        print('REFUSED:', ex, file=sys.stderr); return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
