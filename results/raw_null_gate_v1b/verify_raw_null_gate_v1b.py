"""Independent verifier for the v1b bug-fix release of the raw-null gate.

Specs: docs/experiments/RAW_NULL_GATE_PREREG_2026-10-30.md (v1, Sections 2-6a) and
docs/experiments/RAW_NULL_GATE_V1B_BUGFIX_2026-10-31.md (fixes F1-F7).

WRITTEN FROM THE SPECS ONLY. It starts from a copy of the independent v1 verifier
(results/raw_null_gate_v1/verify_raw_null_gate.py, itself written without reading the module). The author of this file
did NOT open spectral_utils/raw_null_gate.py, spectral_utils/raw_null_gate_v1b.py, any run_raw_null_gate*.py or
tests/test_raw_null_gate*.py. Shared estimators named by the spec are used as-is: lsml_continuous_guarded
(fusion_utils) and ds_em_votes (prm_anchored_lsml), plus the same read-only data loaders as v1.

Fixes implemented here (v1b spec table):
  F1  G2: LR_g = (pi0_g/(1-pi0_g))(1/lfdr_g - 1) with the group's OWN (clipped) pi0_g; the prior-odds term
      log((1-pi0)/pi0) keeps the shared median pi0.
  F2  REMOVED from v1b by the coordinator's spec change of 2026-10-31 (on no-error N(0,1) data central matching
      gives pi0_raw ~1.02 and sigma0/sd ~1.03, so the strict rule rejected every no-error fit). The empirical-null
      fit is exactly v1's; the population sd of the fitted values is only a diagnostic ('sd', compared to 1e-6 when
      the module reports it). Accepted fits with pi0_raw > 1 or sigma0 >= sd are counted as diagnostics only.
  F3  variant fix_mono only: above mu0, P(err) = running max of 1 - lfdr over the scored points sorted by z (ties get
      the same value); scored set = all steps of the cell (G1) or the group's defined steps (G2/G3). G2 uses
      lfdr_mono = 1 - P_mono inside LR_g. At or below mu0: P(err) = 0, LR = 1 (as v1).
  F4  G3 drops voters whose answer votes are constant (all +1 or all -1) before DS-EM; < 3 non-constant voters ->
      G3 invalid.
  F5  VERDICT.json: PRMBench no-harm counts only cells where the gate is valid.
  F6  ECE ranks np.round(P(err), 9) with a stable sort.
  F7  compares labelfree npz p_err_step_G1/G2, p_err_worst_G1/G2, p_answer_G1b/G3 (1e-6, NaN positions must match)
      and clean_<G> exactly; recomputes VERDICT.json and compares it.

    python verify_raw_null_gate_v1b.py synthetic <fix|fix_mono>  # no data; black-box comparison with
                                                                 # spectral_utils.raw_null_gate_v1b (called, never read)
    python verify_raw_null_gate_v1b.py real <fix|fix_mono>       # after the v1b run; reads
                                                                 # results/raw_null_gate_v1b/<variant>/ and writes
                                                                 # VERIFY.json there

Exit codes: 0 PASS, 1 FAIL (a disagreement), 2 refused (missing inputs / unreadable schema / crash before comparing).
The text below is the v1 verifier's description of the real-mode schemas, still valid for v1b.

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
    # F2 was REMOVED from v1b (coordinator spec change, 2026-10-31): the fit is accepted exactly as in v1. The
    # population sd of the fitted values is kept only as a diagnostic ('sd', compared when the module reports it).
    sd_x = float(np.std(x))
    out.update(sd_x=sd_x)
    p0 = float(np.clip(p0raw, *PI0_CLIP))

    def lfdr(z):
        z = np.asarray(z, float)
        phi = np.exp(-.5 * ((z - m0) / s0) ** 2) / (s0 * math.sqrt(2 * math.pi))
        return np.clip(p0 * phi / np.exp(logf(z)), 0., 1.)

    def p_err(z, mono=False):
        """3.3.5 (one-sided). mono (F3): running max of 1 - lfdr over the points of z above mu0, sorted by z, ties
        equal. z is the whole scored set (the caller passes all scored points at once)."""
        z = np.asarray(z, float)
        base = np.where(z > m0, 1. - lfdr(z), 0.)
        return monotone_above(z, base, m0) if mono else base

    out.update(ok=True, pi0=p0, pi0_raw=float(p0raw), mu0=float(m0), sigma0=float(s0), sd=sd_x, b=(b0, b1, b2),
               lo=float(lo), hi=float(hi), logf=logf, lfdr=lfdr, p_err=p_err)
    return out


def monotone_above(z, base, m0):
    """F3: for the points with z > m0, replace base by its running maximum over the points sorted by z; tied z values
    get the same value (the running max at the end of their tie block). Points at or below m0 keep base (= 0)."""
    z = np.asarray(z, float); out = np.array(base, float, copy=True)
    idx = np.flatnonzero(z > m0)
    if len(idx) == 0:
        return out
    o = np.argsort(z[idx], kind='stable')
    zs = z[idx][o]
    run = np.maximum.accumulate(out[idx][o])
    last = np.searchsorted(zs, zs, side='right') - 1          # last position of each value's tie block
    out[idx[o]] = run[last]
    return out


def non_monotone_pairs(z, p, m0):
    """Label-free audit: number of adjacent (in z order, distinct z) pairs above m0 where P(err) decreases."""
    z = np.asarray(z, float); p = np.asarray(p, float)
    idx = np.flatnonzero(z > m0)
    if len(idx) < 2:
        return 0
    o = np.argsort(z[idx], kind='stable'); zs, ps = z[idx][o], p[idx][o]
    return int(((np.diff(ps) < -1e-15) & (np.diff(zs) > 0)).sum())


def fit_public(f):
    return {k: f[k] for k in ('ok', 'reason', 'pi0', 'pi0_raw', 'mu0', 'sigma0')}


def worst_step(z_fused, off):
    """Index of the argmax (first on ties) of the fused score within each answer."""
    return np.array([a + int(np.argmax(z_fused[a:b])) for a, b in zip(off[:-1], off[1:])])


def answer_max(v, off):
    return np.array([float(np.max(v[a:b])) for a, b in zip(off[:-1], off[1:])])


def gate_g1(raw_fused, z_fused, off, mono=False):
    f = empirical_null(raw_fused); n_ans = len(off) - 1
    if not f['ok']:
        return dict(ok=False, clean=np.zeros(n_ans, bool), p_err_step=np.full(len(raw_fused), np.nan),
                    p_err_worst=np.full(n_ans, np.nan), fit=f)
    pe = f['p_err'](raw_fused, mono); w = worst_step(z_fused, off)   # F3 scored set: all steps of the cell
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


def group_lfdr(f, u_g, d_g, mono=False):
    """lfdr on the group's defined steps (1 elsewhere). mono (F3): lfdr_mono = 1 - P_mono above mu0; at or below
    mu0 the plain lfdr is returned (G2 sets LR = 1 there anyway)."""
    out = np.ones(len(u_g)); x = u_g[d_g]
    lf = f['lfdr'](x)
    if mono:
        lf = np.where(x > f['mu0'], 1. - f['p_err'](x, True), lf)
    out[d_g] = lf; return out


def group_perr(f, u_g, d_g, mono=False):
    """P(err) on the group's defined steps (0 elsewhere); F3 scored set = the group's defined steps."""
    out = np.zeros(len(u_g)); out[d_g] = f['p_err'](u_g[d_g], mono); return out


def gate_g2(u, orient, off, D, fits=None, mono=False):
    fits = group_fits(u, orient, D) if fits is None else fits
    used = [g for g, f in fits.items() if f['ok']]; n_ans = len(off) - 1
    if not used:
        return dict(ok=False, clean=np.zeros(n_ans, bool), p_err_step=np.full(len(u), np.nan),
                    p_err_worst=np.full(n_ans, np.nan), used=[], fits=fits)
    p0g = [fits[g]['pi0'] for g in used]
    p0 = float(np.clip(np.median(p0g), *PI0_CLIP))
    lo = np.log((1 - p0) / p0) * np.ones(len(u))              # prior odds: the SHARED median pi0 (unchanged)
    for g in used:
        p0_g = fits[g]['pi0']                                  # F1: the group's own (clipped) pi0_g
        lf = np.maximum(group_lfdr(fits[g], u[:, g], D[:, g], mono), LFDR_FLOOR)
        lr = np.clip((p0_g / (1 - p0_g)) * (1. / lf - 1.), *LR_CLIP)
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


def gate_g3(u, orient, off, D, anchor_group=None, fits=None, mono=False):
    fits = group_fits(u, orient, D) if fits is None else fits
    fitted = [g for g, f in fits.items() if f['ok']]; n_ans = len(off) - 1
    # F4: drop voters whose answer votes are constant (all +1 or all -1) before DS-EM
    Ball = votes(u, D, fits, fitted, off, mono) if fitted else np.zeros((n_ans, 0), int)
    varying = [j for j in range(len(fitted)) if (Ball[:, j] > 0).any() and (Ball[:, j] < 0).any()]
    used = [fitted[j] for j in varying]
    dropped = [g for g in fitted if g not in used]
    base = dict(used=used, fitted=fitted, constant_dropped=dropped)
    if len(used) < 3:
        return dict(ok=False, clean=np.zeros(n_ans, bool), p_answer=np.full(n_ans, np.nan),
                    reason=f'{len(used)} non-constant votes per answer ({len(fitted)} fitted)', **base)
    Bv = Ball[:, varying]
    anchor = used.index(anchor_group) if anchor_group is not None and anchor_group in used else None
    fit = ds_em_votes(Bv, anchor=anchor, init='majority')
    pa = ds_posterior(Bv, fit)
    return dict(ok=True, clean=pa < P_CUT, p_answer=pa, prevalence=float(fit['prevalence']),
                psi=np.asarray(fit['psi']).tolist(), eta=np.asarray(fit['eta']).tolist(),
                anchor_used=anchor, anchor_requested=anchor_group, vote_rate=(Bv > 0).mean(0).tolist(),
                vote_rate_fitted=(Ball > 0).mean(0).tolist(), sml_ranking=sml_ranking(Bv).tolist(), Bv=Bv, **base)


def votes(u, D, fits, used, off, mono=False):
    return np.column_stack([np.where(answer_max(group_perr(fits[g], u[:, g], D[:, g], mono), off) >= P_CUT, 1, -1)
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


def label_free_stage(V, Z, off, arm, gap_fn=None, mono=False):
    """Everything label-free for one cell and arm. V = answer-z inputs, Z = cleaned raw inputs (same columns).
    gap_fn(fused) -> GAP flags (3.5); default: this file's own GAP implementation. mono = F3 (variant fix_mono)."""
    anchor = None if arm == 'AUTO' else V.mean(1)
    fused, ro = lsml_linear(V, anchor)
    rr = raw_readout(Z, ro)
    gap = np.asarray((gap_fn or (lambda s: gap_flags_own(s, off)))(fused), bool)
    fits = group_fits(rr['u'], rr['orient'], rr['defined'])
    out = dict(arm=arm, fused=fused, ro=ro, rr=rr, gap=gap, mono=mono,
               G1=gate_g1(rr['raw_fused'], fused, off, mono),
               G1b=gate_g1b(rr['raw_fused'], fused, off, gap),
               G2=gate_g2(rr['u'], rr['orient'], off, rr['defined'], fits, mono),
               G3=gate_g3(rr['u'], rr['orient'], off, rr['defined'], prm_group(ro) if arm == 'AUTO' else None,
                          fits, mono))
    return out


def fix_audit(lf):
    """Label-free audit of this verifier's own recomputation: accepted fits with pi0_raw > 1 or sigma0 >= sd (a
    DIAGNOSTIC count only; F2 was removed from v1b, so these fits are accepted), non-monotone P(err) pairs above mu0
    (0 required for fix_mono), G3 constant voters used (0 by construction) and dropped."""
    rr = lf['rr']; fits = [('G1 fused', lf['G1']['fit'])] + [(f'group {g}', f) for g, f in lf['G2']['fits'].items()]
    acc_pi0_gt1 = [n for n, f in fits if f['ok'] and f['pi0_raw'] > 1]
    acc_wide = [n for n, f in fits if f['ok'] and f['sigma0'] >= f.get('sd_x', np.inf)]
    nm = 0
    if lf['G1']['ok']:
        nm += non_monotone_pairs(rr['raw_fused'], lf['G1']['p_err_step'], lf['G1']['fit']['mu0'])
    for g, f in lf['G2']['fits'].items():
        if f['ok']:
            d = rr['defined'][:, g]; x = rr['u'][d, g]
            nm += non_monotone_pairs(x, f['p_err'](x, lf['mono']), f['mu0'])
    const_used = 0
    if lf['G3']['ok']:
        Bv = lf['G3']['Bv']; const_used = int(sum(not ((Bv[:, j] > 0).any() and (Bv[:, j] < 0).any())
                                                 for j in range(Bv.shape[1])))
    return dict(diag_accepted_fits_pi0_raw_gt_1=acc_pi0_gt1, diag_accepted_fits_sigma0_ge_sd=acc_wide,
                non_monotone_pairs_above_mu0=nm,
                g3_constant_voters_used=const_used, g3_constant_voters_dropped=lf['G3'].get('constant_dropped', []))


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


class ApiFail(Exception):
    pass


def api(L, ctx, Gm, fname, *a, **k):
    """Call Gm.<fname>(*a, **k) as a black box. A missing function, a signature mismatch (TypeError, e.g. no
    `monotone` keyword) or any exception is recorded as a FAIL row naming the call; None is returned and the dependent
    comparisons are skipped (never guessed)."""
    fn = getattr(Gm, fname, None)
    if fn is None:
        L.note(ctx, f'API {fname}', False, f'API mismatch: module has no public function {fname!r}')
        return None
    try:
        return fn(*a, **k)
    except TypeError as ex:
        L.note(ctx, f'API {fname}', False, f'API mismatch: {fname}(<{len(a)} positional>, keywords={sorted(k)}) '
                                           f'raised TypeError: {ex}. Not guessing the module internals.')
    except Exception as ex:                              # noqa: BLE001 - reported, not swallowed
        L.note(ctx, f'API {fname}', False, f'{fname} raised {type(ex).__name__}: {ex}')
    return None


def key_of(L, ctx, d, k, required=True):
    """d[k] from a module result dict; a missing required key is a FAIL row (API mismatch), an optional one a note."""
    if isinstance(d, dict) and k in d:
        return d[k]
    L.note(ctx, f'API result key {k!r}', not required,
           ('API mismatch: ' if required else 'not returned by the module (optional): ') +
           f'result has keys {sorted(d) if isinstance(d, dict) else type(d).__name__}')
    return None


def _cat(f):
    # Reason texts are not fixed by the spec; as in v1 only the amended 3.3.2 IRLS non-convergence is compared.
    return [str((f or {}).get('reason', '')) == 'IRLS did not converge']


def compare_null(L, ctx, label, mine, theirs):
    """Empirical-null fit comparison (unchanged from v1; F2 removed): ok exactly; 'reason is IRLS non-convergence'
    exactly (other reason texts are not fixed by the spec); pi0_raw/pi0/mu0/sigma0 to 1e-6 when both fits are ok; the
    diagnostic 'sd' (population sd of the fitted values) to 1e-6 only when the module reports it."""
    if theirs is None:
        return
    L.exact(ctx, f'{label}.ok', [mine['ok']], [bool(key_of(L, ctx, theirs, 'ok'))])
    if not mine['ok'] or not theirs.get('ok'):
        L.exact(ctx, f'{label}.reason is IRLS non-convergence', _cat(mine), _cat(theirs))
        return
    for k in ('pi0_raw', 'pi0', 'mu0', 'sigma0'):
        v = key_of(L, ctx, theirs, k)
        if v is not None:
            L.num(ctx, f'{label}.{k}', mine[k], v)
    if 'sd' in theirs:
        L.num(ctx, f'{label}.sd (diagnostic)', mine['sd_x'], theirs['sd'])


def compare_synthetic(L, name, X, off, arm, Gm, mono):
    """Run this file's pipeline and the v1b module's on the same synthetic cell; record every comparison in L.
    mono=True is variant fix_mono (F3): passed to the module as the keyword `monotone`."""
    V = answer_z_cols(X, off)
    Z = np.column_stack([pooled_z_defined(X[:, j]) for j in range(X.shape[1])])   # synthetic cleaned raw (NaN kept)
    mine = label_free_stage(V, Z, off, arm, mono=mono)
    fin = np.isfinite(Z); Z0 = np.where(fin, Z, 0.)          # module inputs: NaN -> 0 plus the finite mask
    gap = mine['gap']
    # ---------- module, end to end on its own readout (functions unaffected by `monotone`: no keyword passed)
    r = api(L, name, Gm, 'linear_lsml', V, anchor=None if arm == 'AUTO' else V.mean(1))
    if r is None:
        return mine
    s_m, p_m = r
    L.num(name, 'fused (linear_lsml)', mine['fused'], s_m)
    fl = api(L, name, Gm, 'fused_linear', Z0, p_m)
    if fl is not None:
        L.num(name, 'fused_linear(raw)', mine['rr']['lin'], fl)
    gs_m = api(L, name, Gm, 'group_scores', Z0, p_m)
    D_m = api(L, name, Gm, 'group_defined', fin, p_m)
    o_m = api(L, name, Gm, 'group_orientation', p_m)
    if gs_m is None or D_m is None or o_m is None or fl is None:
        return mine
    gs_m = np.asarray(gs_m, float); D_m = np.asarray(D_m, bool); o_m = np.asarray(o_m, float)
    L.exact(name, 'group_defined', mine['rr']['defined'], D_m)
    L.num(name, 'group_orientation', mine['rr']['orient'], o_m)
    # As in v1: group_scores() is Z_g w_g (un-normalized, un-oriented); u_g is checked through the gates.
    L.num(name, 'group_scores (= Z_g w_g, un-normalized)', mine['rr']['g_lin'], gs_m)
    if D_m.shape == mine['rr']['defined'].shape:
        u_from_m = np.zeros_like(gs_m)
        for g in range(gs_m.shape[1]):
            pz = api(L, name, Gm, 'pooled_z', gs_m[D_m[:, g], g])
            if pz is not None:
                u_from_m[D_m[:, g], g] = o_m[g] * np.asarray(pz, float)
        L.num(name, 'u_g (defined steps) from module group_scores', mine['rr']['u'][mine['rr']['defined']],
              u_from_m[mine['rr']['defined']])
    pz = api(L, name, Gm, 'pooled_z', fl)
    if pz is None:
        return mine
    raw_m = np.asarray(pz, float)
    L.num(name, 'raw_fused', mine['rr']['raw_fused'], raw_m)
    # ---------- empirical null (unchanged by `monotone`) on raw_fused and each u_g; module fed the same vector
    f_mine = mine['G1']['fit']
    f_mod = api(L, name, Gm, 'empirical_null', mine['rr']['raw_fused'])
    compare_null(L, name, 'null(raw_fused)', f_mine, f_mod)
    if f_mine['ok'] and f_mod is not None and f_mod.get('ok'):
        pm = api(L, name, Gm, 'p_error', mine['rr']['raw_fused'], f_mod, monotone=mono)
        if pm is not None:
            L.num(name, 'p_error(raw_fused)', f_mine['p_err'](mine['rr']['raw_fused'], mono), pm)
    for g, fg in mine['G2']['fits'].items():
        ug = mine['rr']['u'][mine['rr']['defined'][:, g], g]           # defined steps only (3.2a)
        if len(ug) < MIN_DEFINED:
            continue                     # dropped by the gate rule, not by 3.3; checked through the gate summaries
        fm = api(L, name, Gm, 'empirical_null', ug)
        compare_null(L, name, f'null(u_{g})', fg, fm)
        if fg['ok'] and fm is not None and fm.get('ok'):
            pm = api(L, name, Gm, 'p_error', ug, fm, monotone=mono)
            if pm is not None:
                L.num(name, f'p_error(u_{g})', fg['p_err'](ug, mono), pm)
    # ---------- gates: module on its own readout (raw_m, gs_m, o_m); mine on mine
    m1 = api(L, name, Gm, 'gate_g1', raw_m, s_m, off, monotone=mono)
    if m1 is not None:
        L.exact(name, 'G1.ok', [mine['G1']['ok']], [bool(key_of(L, name, m1, 'ok'))])
        L.exact(name, 'G1.clean', mine['G1']['clean'], key_of(L, name, m1, 'clean'))
        L.num(name, 'G1.p_err_step', mine['G1']['p_err_step'], key_of(L, name, m1, 'p_err_step'))
        pw = key_of(L, name, m1, 'p_err_worst', required=False)
        if pw is not None:
            L.num(name, 'G1.p_err_worst', mine['G1']['p_err_worst'], pw)
    ws = api(L, name, Gm, 'worst_step', s_m, off)
    if ws is not None:
        L.exact(name, 'G1.worst_step', worst_step(mine['fused'], off), ws)
    m1b = api(L, name, Gm, 'gate_g1b', raw_m, s_m, off, gap)
    if m1b is not None:
        L.exact(name, 'G1b.clean', mine['G1b']['clean'], key_of(L, name, m1b, 'clean'))
        L.num(name, 'G1b.p_answer', mine['G1b']['p_answer'], key_of(L, name, m1b, 'p_answer'))
        L.exact(name, 'G1b.n_ref', [mine['G1b']['n_ref']], [key_of(L, name, m1b, 'n_ref')])
    m2 = api(L, name, Gm, 'gate_g2', gs_m, o_m, off, defined=D_m, monotone=mono)
    if m2 is not None:
        L.exact(name, 'G2.ok', [mine['G2']['ok']], [bool(key_of(L, name, m2, 'ok'))])
        L.exact(name, 'G2.clean', mine['G2']['clean'], key_of(L, name, m2, 'clean'))
        L.num(name, 'G2.p_err_step', mine['G2']['p_err_step'], key_of(L, name, m2, 'p_err_step'))
        pw = key_of(L, name, m2, 'p_err_worst', required=False)
        if pw is not None:
            L.num(name, 'G2.p_err_worst', mine['G2']['p_err_worst'], pw)
        summ = {int(d['group']): d for d in m2.get('groups', [])}
        L.exact(name, 'G2.groups summarized', sorted(mine['G2']['fits']), sorted(summ))
        for g, fg in mine['G2']['fits'].items():
            if g not in summ:
                continue
            L.exact(name, 'G2.group.ok', [fg['ok']], [bool(summ[g]['ok'])])
            if fg['ok'] and bool(summ[g]['ok']):
                for k in ('pi0_raw', 'pi0', 'mu0', 'sigma0'):
                    L.num(name, f'G2.group.{k}', fg[k], parse_float(summ[g].get(k)))
            if 'share_defined' in summ[g]:
                L.num(name, 'G2.group.share_defined', mine['rr']['defined'][:, g].mean(), summ[g]['share_defined'])
        if mine['G2']['ok'] and m2.get('ok'):
            L.num(name, 'G2.pi0', mine['G2']['pi0'], key_of(L, name, m2, 'pi0'))
            L.num(name, 'G2.pi0_groups', mine['G2']['pi0_groups'], key_of(L, name, m2, 'pi0_groups'))
            L.exact(name, 'G2.groups_used', mine['G2']['used'], key_of(L, name, m2, 'groups_used'))
    ag = prm_group(mine['ro']) if arm == 'AUTO' else None
    m3 = api(L, name, Gm, 'gate_g3', gs_m, o_m, off, anchor_group=ag, defined=D_m, monotone=mono)
    if m3 is not None:
        L.exact(name, 'G3.ok', [mine['G3']['ok']], [bool(key_of(L, name, m3, 'ok'))])
        L.exact(name, 'G3.clean', mine['G3']['clean'], key_of(L, name, m3, 'clean'))
        pa = key_of(L, name, m3, 'p_answer', required=False)
        if pa is not None:
            L.num(name, 'G3.p_answer', mine['G3']['p_answer'], pa)
        gu = key_of(L, name, m3, 'groups_used', required=False)
        if gu is not None:
            compare_g3_used(L, name, 'G3.groups_used', mine['G3'], gu)
        if mine['G3']['ok'] and m3.get('ok'):
            for k in ('prevalence', 'psi', 'eta', 'sml_ranking'):
                v = key_of(L, name, m3, k)
                if v is not None:
                    L.num(name, f'G3.{k}', mine['G3'][k], v)
            v = key_of(L, name, m3, 'vote_rate', required=False)
            if v is not None:
                compare_vote_rate(L, name, 'G3.vote_rate', mine['G3'], v)
            if hasattr(Gm, 'ds_answer_posterior'):
                dp = api(L, name, Gm, 'ds_answer_posterior', mine['G3']['Bv'], mine['G3'])
                if dp is not None:
                    L.num(name, 'G3.ds_answer_posterior', ds_posterior(mine['G3']['Bv'], mine['G3']), dp)
    return mine


def compare_g3_used(L, ctx, name, g3, theirs):
    """F4 makes 'groups used by G3' ambiguous: the identified-null groups, or the non-constant voters passed to DS-EM.
    Exact match with either is accepted; the row says which."""
    theirs = [int(v) for v in theirs]
    if theirs == list(g3['used']):
        L.rows.append(dict(ctx=ctx, name=name, ok=True, max_diff=None, tol=0, note='exact: non-constant voters'))
    elif theirs == list(g3['fitted']):
        L.rows.append(dict(ctx=ctx, name=name, ok=True, max_diff=None, tol=0,
                           note='exact: identified-null groups (before the F4 constant-voter drop)'))
    else:
        L.rows.append(dict(ctx=ctx, name=name, ok=False, max_diff=None, tol=0, mine=dict(
            non_constant=list(g3['used']), fitted=list(g3['fitted'])), theirs=theirs, note='exact'))


def compare_vote_rate(L, ctx, name, g3, theirs):
    """vote_rate over the DS-EM voters, or over all identified-null groups (length decides; ambiguous after F4)."""
    theirs = np.asarray([parse_float(v) for v in theirs], float)
    if len(theirs) == len(g3['vote_rate']):
        L.num(ctx, name, g3['vote_rate'], theirs)
    else:
        L.num(ctx, name + ' (all identified-null groups)', g3['vote_rate_fitted'], theirs)


def self_checks(L, name, mine):
    """Properties of this verifier's own v1b implementation (spec 'proof' items, applied to mine)."""
    rr = mine['rr']; mono = mine['mono']
    # F1: with exactly one G2 group, G2 P(err) = that group's P(err) (P_mono under F3) on defined steps above mu0
    # where LR is not clipped.
    G2 = mine['G2']
    if G2['ok'] and len(G2['used']) == 1:
        g = G2['used'][0]; f = G2['fits'][g]; d = rr['defined'][:, g]; u = rr['u'][:, g]
        P = group_perr(f, u, d, mono)
        lf = np.maximum(group_lfdr(f, u, d, mono), LFDR_FLOOR)
        lr = (f['pi0'] / (1 - f['pi0'])) * (1. / lf - 1.)
        m = d & (u > f['mu0']) & (lr > LR_CLIP[0]) & (lr < LR_CLIP[1])
        L.num(name, 'selfcheck F1: one-group G2 P(err) = group P(err)', G2['p_err_step'][m], P[m], tol=1e-9)
    a = fix_audit(mine)
    if mono:
        L.note(name, 'selfcheck F3: P(err) non-decreasing above mu0', a['non_monotone_pairs_above_mu0'] == 0,
               f"{a['non_monotone_pairs_above_mu0']} decreasing pairs")
    L.note(name, 'selfcheck F4: no constant voter in G3', a['g3_constant_voters_used'] == 0,
           f"dropped {a['g3_constant_voters_dropped']}")
    return a


def run_synthetic(variant):
    mono = variant == 'fix_mono'
    try:
        from spectral_utils import raw_null_gate_v1b as Gm  # black box: called, never read
    except Exception as ex:                                  # noqa: BLE001
        die(f'cannot import spectral_utils.raw_null_gate_v1b ({type(ex).__name__}: {ex}); synthetic mode needs it')
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
        # v1b additions: one group (F1), every / almost every answer erroneous (F4 constant voters, PRMBench-like)
        ('TELE16_one_group', dict(seed=28, groups=(6,), d=(2.8,)), 'TELE16'),
        ('TELE16_all_error', dict(seed=29, p_clean=0.), 'TELE16'),
        ('AUTO_mostly_error', dict(seed=30, with_prm=True, p_clean=.03), 'AUTO'),
        ('TELE16_strong_all_clean_votes', dict(seed=31, d=(6., 6., 6., 6.), p_clean=0.), 'TELE16'),
    ]

    for name, kw, arm in cells:
        t = time.time()
        X, off, y, has = synth_cell(**kw)
        mine = compare_synthetic(L, name, X, off, arm, Gm, mono)
        audit = self_checks(L, name, mine)
        rates = {g: (round(float(mine[g]['clean'][~has].mean()), 3) if (~has).any() else None,
                     round(float((~mine[g]['clean'][has]).mean()), 3)) for g in GATES}
        summary.append(dict(cell=name, arm=arm, variant=variant, n_ans=len(off) - 1, K=mine['ro']['K'],
                            orient=mine['rr']['orient'].tolist(), seconds=round(time.time() - t, 1),
                            valid={G: bool(mine[G]['ok']) for G in GATES}, G2_groups_used=mine['G2']['used'],
                            G3_groups_used=mine['G3']['used'], G3_constant_dropped=mine['G3']['constant_dropped'],
                            undefined_steps_per_group=(~mine['rr']['defined']).sum(0).tolist(),
                            closed_clean_kept_error=rates, G1_fit=fit_public(mine['G1']['fit']),
                            diag_pi0_raw_gt_1=audit['diag_accepted_fits_pi0_raw_gt_1'],
                            diag_sigma0_ge_sd=audit['diag_accepted_fits_sigma0_ge_sd']))
    # direct empirical-null checks on crafted vectors, including failure paths
    rng = np.random.default_rng(99)
    crafted = {
        'two_groups_N(0,1)+N(3,1)': np.r_[rng.normal(0, 1, 42_500), rng.normal(3, 1, 7_500)],
        'bimodal_dip(+-2.5)': np.r_[rng.normal(-2.5, .6, 20_000), rng.normal(2.5, .6, 20_000)],
        'heavy_tail_t3': rng.standard_t(3, 30_000),
        'few_values': np.r_[np.zeros(500), np.ones(500), 2 * np.ones(10)],
        'tiny_N': rng.normal(0, 1, 40),
        'contaminated_centre_N(0,1)+N(.8,1)': np.r_[rng.normal(0, 1, 20_000), rng.normal(.8, 1, 20_000)],
        'uniform(-1,1)': rng.uniform(-1, 1, 30_000),
        'lognormal': rng.lognormal(0, 1, 30_000),
        'well_separated_N(0,1)+N(5,.5)': np.r_[rng.normal(0, 1, 45_000), rng.normal(5, .5, 5_000)],
    }
    for nm, x in crafted.items():
        fm = empirical_null(x); fo = api(L, 'crafted:' + nm, Gm, 'empirical_null', x)
        if fo is None:
            continue
        compare_null(L, 'crafted:' + nm, 'null', fm, fo)
        if fm['ok'] and fo.get('ok'):
            pm = api(L, 'crafted:' + nm, Gm, 'p_error', x, fo, monotone=mono)
            if pm is not None:
                L.num('crafted:' + nm, 'p_error', fm['p_err'](x, mono), pm)
            if mono:
                L.note('crafted:' + nm, 'selfcheck F3: P(err) non-decreasing above mu0',
                       non_monotone_pairs(x, fm['p_err'](x, True), fm['mu0']) == 0, '')
        summary.append(dict(cell='crafted:' + nm, mine={**fit_public(fm), 'sd_x': fm.get('sd_x')},
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
    print(f"{'quantity':55s} {'checks':>6s} {'max diff':>12s}  result")
    for k, v in by_name.items():
        md = f"{v['max_diff']:.3g}" if v['kind'] != 'exact' else f"{int(v['max_diff'])} diff"
        print(f"{k:55s} {v['n']:6d} {md:>12s}  {'PASS' if v['ok'] else 'FAIL'}")
    fails = L.failed
    print()
    api_fails = [r for r in fails if str(r.get('note', '')).startswith('API mismatch')]
    for r in fails:
        print('FAIL', json.dumps(r, default=float))
    if api_fails:
        print(f'NOTE: {len(api_fails)} failure(s) are API mismatches with spectral_utils.raw_null_gate_v1b '
              f'(see "API ..." rows); the dependent comparisons were skipped, not guessed.')
    print('SYNTHETIC', variant, 'PASS' if not fails else 'FAIL', f'({len(L.rows)} comparisons, {len(fails)} failed, '
          f'{len(api_fails)} API mismatch(es))')
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
            if gate == 'G3' and 'groups_used' in keys and isinstance(obj[keys['groups_used']], list):
                compare_g3_used(L, ctx, f'FITS:{"/".join(map(str, path + ("groups_used",)))}', lf['G3'],
                                obj[keys['groups_used']]); n[0] += 1
            elif gate == 'G2' and 'groups_used' in keys and isinstance(obj[keys['groups_used']], list):
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
                        elif gate == 'G3' and k == 'vote_rate':
                            compare_vote_rate(L, ctx, f'FITS:{"/".join(map(str, path + (k,)))}', gk, theirs)
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
    # F6: rank np.round(P(err), 9) with a stable sort (ties by index); bin means use P(err) itself (the rounding
    # changes the mean by < 5e-10, far below the 1e-6 tolerance)
    idx = np.empty(len(p), int); idx[np.argsort(np.round(p, 9), kind='stable')] = np.arange(len(p)) * nb // len(p)
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


def real_cell(arm, mode, bench, prm, mods, mono=False):
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
    lf = label_free_stage(V, Z, off, arm, gap_fn=lambda s: S7.gap_flags(c, s), mono=mono)   # 3.5: S7.gap_flags
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


NPZ_REQUIRED = ('z_fused', 'raw_fused', 'f_gap', 'clean_G1', 'clean_G1b', 'clean_G2', 'clean_G3',
                'p_err_step_G1', 'p_err_step_G2', 'p_err_worst_G1', 'p_err_worst_G2', 'p_answer_G1b', 'p_answer_G3')


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
    # F7 / v1b output contract: these keys MUST be present (P(err) arrays all NaN when the gate is invalid)
    missing = [k for k in NPZ_REQUIRED if k not in z.files]
    if missing:
        L.note(ctx, 'npz required keys', False, f'{path.name} lacks {missing} (v1b output contract)')
    for k in NPZ_REQUIRED:                  # NaN exactly where this verifier's array is NaN (gate invalid)
        if k in z.files and k.startswith('p_'):
            a = np.asarray(z[k], float); m = np.asarray(mine_arrays[k], float)
            if a.shape == m.shape:
                L.exact(ctx, f'npz:{k} NaN positions', np.isnan(m), np.isnan(a))
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


PASS_BENCHES = ('processbench', 'crv_arith', 'crv_gsm8k')
CC_MIN, KE_MIN, NO_HARM_MARGIN, N_NEEDED, N_PER_BENCH = .30, .85, .5, 3, 4


def compute_verdict(ends_by_cell):
    """Section 5 pass rule with F5, from this verifier's own recomputed endpoints (ok cells only).
    Returns {(arm, mode, gate): entry}. Entry fields mirror v1's VERDICT.json: status, per_benchmark {n, n_good},
    prmbench_no_harm 'k/n', passes; plus the alternatives the spec leaves open (see compare_verdict)."""
    out = {}
    for arm, mode in ARM_MODES:
        for G in GATES:
            cells = {c: e[G] for c, e in ends_by_cell.items() if c[0] == arm and c[1] == mode}
            per = {}
            for b in PASS_BENCHES:
                es = [e for c, e in cells.items() if c[2] == b]
                good = [e for e in es if e['valid'] and e['closed_clean'] >= CC_MIN and e['kept_error'] >= KE_MIN]
                per[b] = dict(n=len(es), n_good=len(good))
            pb = [e for c, e in cells.items() if c[2] == 'prmbench']
            k = sum(1 for e in pb if e['valid'] and e['metric'] >= e['metric_gap_only'] - NO_HARM_MARGIN)   # F5
            n_valid = sum(1 for e in pb if e['valid'])
            status = 'complete' if all(per[b]['n'] >= N_PER_BENCH for b in PASS_BENCHES) else 'incomplete'
            core = status == 'complete' and all(per[b]['n_good'] >= N_NEEDED for b in PASS_BENCHES)
            out[(arm, mode, G)] = dict(status=status, per_benchmark=per, no_harm_k=k, no_harm_n=len(pb),
                                       no_harm_n_valid=n_valid, prmbench_no_harm=f'{k}/{len(pb)}',
                                       passes=bool(core and (k >= N_NEEDED if pb else True)),
                                       passes_if_no_prmbench_fails=bool(core and k >= N_NEEDED))
    return out


def compare_verdict(L, mine, theirs):
    """Compare VERDICT.json with the recomputation. Exact on status, per-benchmark n / n_good, the no-harm numerator
    and passes. Spec-ambiguous points, accepted either way and recorded in the row note:
      * the no-harm denominator after F5: all PRMBench cells present, or only the valid ones;
      * passes when an entry has no PRMBench cell (AUTO, backbone-only arms): the PRMBench clause read as vacuous
        (True) or as unmet;
      * passes of an 'incomplete' entry: False or null."""
    if not isinstance(theirs, dict):
        L.note('VERDICT.json', 'top level', False, f'expected a dict, got {type(theirs).__name__}'); return
    parsed = {}
    for key, v in theirs.items():
        parts = str(key).split('|')
        if len(parts) != 3:
            L.note('VERDICT.json', f'key {key!r}', False, 'expected ARM|mode|GATE'); continue
        try:
            parsed[(parts[0].upper(), norm_mode(parts[1]), norm_gate(parts[2]))] = v
        except Refuse as ex:
            L.note('VERDICT.json', f'key {key!r}', False, str(ex))
    L.rows.append(dict(ctx='VERDICT.json', name='entries (arm, mode, gate)', ok=sorted(parsed) == sorted(mine),
                       mine=len(mine), theirs=len(parsed), max_diff=None, tol=0,
                       note=f'missing {sorted(set(mine) - set(parsed))}, extra {sorted(set(parsed) - set(mine))}'))
    for key, m in sorted(mine.items()):
        ctx = 'VERDICT:' + '|'.join(key)
        t = parsed.get(key)
        if not isinstance(t, dict):
            continue
        L.rows.append(dict(ctx=ctx, name='VERDICT.status', ok=t.get('status') == m['status'], mine=m['status'],
                           theirs=t.get('status'), max_diff=None, tol=0, note='exact'))
        tp = t.get('per_benchmark', {}) or {}
        for b in PASS_BENCHES:
            tb = tp.get(b, {}) or {}
            for f in ('n', 'n_good'):
                L.rows.append(dict(ctx=ctx, name=f'VERDICT.{b}.{f}', ok=tb.get(f) == m['per_benchmark'][b][f],
                                   mine=m['per_benchmark'][b][f], theirs=tb.get(f), max_diff=None, tol=0,
                                   note='exact'))
        nh = str(t.get('prmbench_no_harm', ''))
        mm = re.fullmatch(r'\s*(\d+)\s*/\s*(\d+)\s*', nh)
        if not mm:
            L.note(ctx, 'VERDICT.prmbench_no_harm', False, f'unreadable {nh!r}')
        else:
            tk, tn = int(mm.group(1)), int(mm.group(2))
            L.rows.append(dict(ctx=ctx, name='VERDICT.prmbench_no_harm numerator (F5: valid cells only)',
                               ok=tk == m['no_harm_k'], mine=m['no_harm_k'], theirs=tk, max_diff=None, tol=0,
                               note='exact'))
            dn_ok = tn in (m['no_harm_n'], m['no_harm_n_valid'])
            L.rows.append(dict(ctx=ctx, name='VERDICT.prmbench_no_harm denominator', ok=dn_ok,
                               mine=dict(all_cells=m['no_harm_n'], valid_cells=m['no_harm_n_valid']), theirs=tn,
                               max_diff=None, tol=0,
                               note=('all PRMBench cells' if tn == m['no_harm_n'] else 'valid PRMBench cells')
                               if dn_ok else 'neither reading'))
        tpass = t.get('passes')
        accept = {m['passes']}
        if m['no_harm_n'] == 0:
            accept.add(m['passes_if_no_prmbench_fails'])
        if m['status'] == 'incomplete':
            accept |= {False, None}
        L.rows.append(dict(ctx=ctx, name='VERDICT.passes', ok=tpass in accept, mine=m['passes'], theirs=tpass,
                           max_diff=None, tol=0, note=f'accepted {sorted(accept, key=str)}'))


def run_real(here=HERE, mods=None, mono=False):
    """here: directory holding RESULTS.csv / FITS.json / CHECK.json / labelfree/; mods: loader modules (tests only)."""
    here = Path(here)
    res_p, fits_p, chk_p, ver_p = here / 'RESULTS.csv', here / 'FITS.json', here / 'CHECK.json', here / 'VERDICT.json'
    for p in (res_p, fits_p, chk_p, ver_p):
        if not p.exists():
            die(f'{p} is missing. Run the v1b runner (`run`) for this variant first; this verifier refuses to run '
                f'without it.')
    if not (here / 'labelfree').is_dir():
        die(f'{here / "labelfree"} is missing (F7 compares its arrays); refusing.')
    try:
        theirs_verdict = json.loads(ver_p.read_text(encoding='utf-8'))
    except Exception as ex:                                  # noqa: BLE001
        die(f'VERDICT.json unreadable: {ex}')
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
    L = Ledger(); L.rows.extend(L0); per_cell = []; warnings = []; t0 = time.time(); ends_by_cell = {}
    # every RESULTS row of an ok cell must be verified; rows of other cells are listed, never silently ignored
    rows_not_ok = sorted('|'.join(k) for k in results if k[:4] not in set(ok_cells))
    for cell in ok_cells:
        ctx = '|'.join(cell); t = time.time()
        print('cell', ctx, flush=True)
        entry = dict(cell=ctx)
        try:
            lf, ends, par = real_cell(*cell, mods, mono=mono)
        except Exception as ex:
            L.note(ctx, 'recompute', False, f'{type(ex).__name__}: {ex}')
            entry.update(error=traceback.format_exc()); per_cell.append(entry); continue
        ends_by_cell[cell] = ends
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
                     irls_not_converged=nonconv, fix_audit=fix_audit(lf),
                     G1_fit=fit_public(lf['G1']['fit']),
                     G2=dict(ok=lf['G2']['ok'], pi0=lf['G2'].get('pi0'), pi0_groups=lf['G2'].get('pi0_groups'),
                             used=lf['G2']['used']),
                     G3={k: lf['G3'].get(k) for k in ('ok', 'prevalence', 'psi', 'eta', 'sml_ranking', 'used',
                                                       'fitted', 'constant_dropped', 'anchor_used')},
                     seconds=round(time.time() - t, 1))
        per_cell.append(entry)
    # F7: recompute VERDICT.json (Section 5 with F5) from this verifier's endpoints. A cell whose recomputation
    # crashed is already a FAIL above; it is absent from the recomputation.
    my_verdict = compute_verdict(ends_by_cell)
    compare_verdict(L, my_verdict, theirs_verdict)
    audit_tot = dict(
        accepted_fits_pi0_raw_gt_1=sum(len(c.get('fix_audit', {}).get('diag_accepted_fits_pi0_raw_gt_1', []))
                                       for c in per_cell),
        accepted_fits_sigma0_ge_sd=sum(len(c.get('fix_audit', {}).get('diag_accepted_fits_sigma0_ge_sd', []))
                                       for c in per_cell),
        non_monotone_pairs_above_mu0=sum(c.get('fix_audit', {}).get('non_monotone_pairs_above_mu0', 0)
                                         for c in per_cell),
        g3_cells_using_constant_voter=sum(1 for c in per_cell if c.get('fix_audit', {}).get('g3_constant_voters_used')),
        g3_constant_voters_dropped=sum(len(c.get('fix_audit', {}).get('g3_constant_voters_dropped', []))
                                       for c in per_cell))
    if mono and audit_tot['non_monotone_pairs_above_mu0']:
        L.note('fix_audit', 'F3 monotone P(err) (this verifier)', False, str(audit_tot))
    fails = L.failed
    out = dict(mode='real', variant='fix_mono' if mono else 'fix', verdict='PASS' if not fails else 'FAIL',
               n_ok_cells=len(ok_cells), fix_audit_totals=audit_tot,
               verdict_recomputed={'|'.join((a, LOADER_MODE[m], g)): v for (a, m, g), v in my_verdict.items()},
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


VARIANTS = ('fix', 'fix_mono')


def main(argv):
    if len(argv) != 3 or argv[1] not in ('synthetic', 'real') or argv[2] not in VARIANTS:
        print(__doc__); return 2
    variant = argv[2]; mono = variant == 'fix_mono'
    try:
        if argv[1] == 'synthetic':
            return run_synthetic(variant)
        return run_real(HERE / variant, mono=mono)
    except Refuse as ex:
        print('REFUSED:', ex, file=sys.stderr)
        if argv[1] == 'real':
            print('REAL FAIL (refused, nothing compared):', ex)
        return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
