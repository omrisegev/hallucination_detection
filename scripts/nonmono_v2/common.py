#!/usr/bin/env python
"""
nonmono_v2/common.py — shared loading, arm scoring and shape statistics.

WHY THIS EXISTS
---------------
Step 217 answered "do these three symmetric transforms help" and got a null. Reading
the artifacts afterwards showed the null was not informative, for four reasons that
are all fixed here rather than in the old scripts (which stay on disk as the record):

1. THE SURVIVOR LIST IS TWO POPULATIONS. Splitting `shape_test.csv` by cell size:
   n >= 2000 -> 22/188 pairs survive (11.7%, a real excess over the 5% per-pair null);
   n < 2000 -> 10/494 (2.0%, BELOW the null rate). 22 of the 32 sit on five large QA
   cells. Every small-cell survivor -- including the headline
   `spilled_triviaqa_llama8b/rpdi +0.2943` on a cell with n_pos = 6 -- is consistent
   with noise. Detection has to be power-aware or it is just re-reporting variance.

2. THE NULL WAS THE WRONG ONE. Label permutation tests "no relationship". The
   hypothesis we need to reject is "the relationship is MONOTONE". Under permutation
   the truth is flat, so isotonic and a 10-bin map are both fitting noise; under a
   real monotone signal isotonic exploits it at low effective df while the bin map
   pays K parameters, which pushes `shape_gain` NEGATIVE relative to its permutation
   reference. That is exactly the under-representation seen in the small stratum.
   `isoboot_labels` below is the correct null: resample y under the isotonic (i.e.
   monotone MLE) fit, holding x, the marginal and the prevalence fixed.

3. THE ISOTONIC DIRECTION WAS PICKED BY PEARSON `corrcoef` (nonmono_shape_test.py:109).
   On a U-shaped x that correlation is ~0 and its SIGN IS COIN-FLIP NOISE, so the
   "best monotone function" baseline gets fitted backwards, scores below 0.5, and the
   gain is inflated -- a false-POSITIVE source aimed squarely at the shapes we hunt.
   `iso_fit` picks by train Bernoulli log-likelihood over the union of both monotone
   cones, which is the actual constrained MLE.

4. THE SCREEN WAS n-INDEPENDENT (`--screen 0.02`) while the null scales ~1/sqrt(n_min).
   At n=8460 the observed null p95 is 0.0158, so a true gain of 0.018 was never
   tested. Nothing here screens; `besag_clifford` makes testing everything affordable.

Everything that touches the deployed pipeline (cell loading, z-scoring, the anchor,
the two fusion arms) is imported, never re-implemented -- `inscope_bench_common`
exists because three earlier scripts each rolled their own and each got it wrong.
"""
import hashlib
import os
import pickle
import sys

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, genpareto
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inscope_bench_common import load_cells, assert_good6, score_cols   # noqa: E402,F401
from inscope_cells import INSCOPE, QA_CELLS, MATH_CELLS                 # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous                 # noqa: E402
from spectral_utils.streaming_utils import anchor_orient                # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS, PKL_NAMES            # noqa: E402
from spectral_utils.upcr import upcr_fit                                # noqa: E402

OUT = os.path.join(REPO, "results", "nonmono_v2")
CACHE = os.path.join(REPO, "local_cache", "nonmono_v2_cells.pkl")

# exp06's fitted U-PCR configuration, verbatim (nonmono_transform_bench.py:152).
FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

DUFS_BENCH = os.path.join(REPO, "results", "selector_bench", "a2_groupfs__c46.csv")

# Detection gates, written down before any run.
MIN_NMIN = 30          # min(n_pos, n_neg); replaces nonmono_shape_test.py's MIN_N=150 on n
KS = (5, 10, 20)       # K-adaptive unconstrained side
N_SEEDS = 3
N_FOLDS = 5
BH_Q = 0.10

GROUP = {**{c: "QA" for c in QA_CELLS}, **{c: "math" for c in MATH_CELLS}}


# ══════════════════════════════════════════════════════════════════════════════
# Cell loading (cached on the source pkl mtimes, gated by GOOD_6 regardless)
# ══════════════════════════════════════════════════════════════════════════════
def _source_fingerprint():
    """(name, size, mtime) of every pkl `load_all_inscope_cells` reads, hashed.

    Caching the prepared cells saves ~2 min per script invocation, but a stale cache
    would silently invalidate everything downstream. Keying on the source files makes
    staleness impossible rather than unlikely; `assert_good6` still runs after every
    load as the second line of defence."""
    d = os.path.join(REPO, "local_cache")
    names = list(PKL_NAMES) + ["derived_views.pkl", "trace_cells.pkl"]
    h = hashlib.sha256()
    for n in sorted(names):
        p = os.path.join(d, n)
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{n}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def load_cells_cached(use_cache=True, verbose=True):
    """`load_cells()` with an mtime-keyed disk cache. Always validity-gated."""
    fp = _source_fingerprint()
    if use_cache and os.path.exists(CACHE):
        try:
            with open(CACHE, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("fingerprint") == fp:
                cells = blob["cells"]
                if verbose:
                    print(f"loaded {len(cells)} cells from cache")
                _gate(cells, verbose)
                return cells
        except Exception as e:                      # noqa: BLE001 - cache is advisory
            print(f"  (cache unusable: {e}; reloading)")
    cells = load_cells()
    _gate(cells, verbose)
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "wb") as fh:
            pickle.dump({"fingerprint": fp, "cells": cells}, fh, protocol=4)
    except Exception as e:                          # noqa: BLE001
        print(f"  (could not write cache: {e})")
    return cells


def _gate(cells, verbose=True):
    ok, macro = assert_good6(cells, verbose=verbose)
    if not ok:
        raise SystemExit(f"validity gate failed: GOOD_6 macro {macro:.4f}")
    missing = [c for c in INSCOPE if c not in cells]
    if missing:
        raise SystemExit(f"cells missing from load: {missing}")
    return macro


def n_min(y):
    """min(n_pos, n_neg) — the quantity that actually governs power here.

    `nonmono_shape_test.py` gated on len(y) >= 150, which let
    `spilled_triviaqa_llama8b` (n=256 but n_pos=6) through and produce the largest
    `shape_gain` in the whole sweep."""
    y = np.asarray(y, int)
    return int(min(y.sum(), len(y) - y.sum()))


# ══════════════════════════════════════════════════════════════════════════════
# The two deployed arms — score-returning twins of nonmono_transform_bench.py:398/407
# ══════════════════════════════════════════════════════════════════════════════
def lsml_score(V, anchor, cols):
    """Arm A fused score (oriented). Mirrors `lsml_arm` / `score_cols` exactly."""
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return None
    fused, _ = lsml_continuous(*[V[:, c] for c in cols])
    s, _ = anchor_orient(np.asarray(fused, dtype=float), anchor)
    return np.asarray(s, dtype=float)


def upcr_score(V, pool, anchor):
    """Arm B fused score (oriented), plus the fit objects the mechanism table needs.

    Two-pass, exactly as `upcr_arm`: the first fit's `sign(rho_hat_full)` re-orients
    every view, the second fit on the re-oriented matrix supplies the weights.

    Returns (score, res2, rho_full_pass1). `res2.keep` is the object that explains why
    Step 217's arm B was bit-identical on 4 of the 5 high-detection cells: `upcr.py:289`
    keeps a view only if `rho_hat >= min_frac*var_y`, and rho_hat is estimated from
    LINEAR covariance -- so a U-shaped view has rho_hat ~ 0 and is dropped, after which
    its column contents cannot affect the fused score at all."""
    hand = np.array([ALL_SIGNS.get(f, +1) for f in pool], dtype=float)
    V_un = V * hand
    r1 = upcr_fit(V_un.T, **FIT)
    d = np.sign(r1.rho_hat_full)
    d[d == 0] = 1.0
    F = (V_un * d).T
    res = upcr_fit(F, **FIT)
    s, _ = anchor_orient(res.w @ F, anchor)
    return np.asarray(s, dtype=float), res, np.asarray(r1.rho_hat_full, dtype=float)


def dufs_selection(path=DUFS_BENCH):
    """{cell: [feature names]} chosen by `a2.dufs_pf` in the selector bench."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {r["cell"]: [f for f in r["chosen"].split("|") if f]
            for r in rows if r["variant"] == "a2.dufs_pf"}


# ══════════════════════════════════════════════════════════════════════════════
# Shape statistics — the monotone baseline, the unconstrained side, the nulls
# ══════════════════════════════════════════════════════════════════════════════
def pct(x):
    """Within-cell percentile rank in (0,1). Every transform family works here, which
    makes them scale-free and marginal-free, and makes shapes comparable ACROSS cells."""
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    return (rankdata(x) - 0.5) / len(x)


def _bern_loglik(y, p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, int)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def iso_fit(xtr, ytr):
    """Best MONOTONE fit over the UNION of both monotone cones.

    THE FIX for `nonmono_shape_test.py:109`, which chose the direction with
    `np.corrcoef(xtr, ytr)[0,1] >= 0`. Pearson correlation of a binary y against a
    heavy-tailed x is ~0 for a U-shape and its sign is then noise; a backwards
    isotonic fit scores below 0.5 and inflates the gain. Isotonic IS the monotone MLE
    under a Bernoulli likelihood, so choosing the cone by train log-likelihood is the
    constrained MLE over the union and is the honest baseline."""
    best, best_ll = None, -np.inf
    for inc in (True, False):
        ir = IsotonicRegression(increasing=inc, out_of_bounds="clip")
        try:
            ir.fit(xtr, ytr)
        except Exception:                            # noqa: BLE001
            continue
        ll = _bern_loglik(ytr, ir.predict(xtr))
        if ll > best_ll:
            best, best_ll = ir, ll
    return best


def bin_fit_apply(xtr, ytr, xte, k, discrete=False):
    """Unconstrained positive-rate map, fitted on train, applied to test.

    `discrete=True` maps distinct values instead of quantile bins. Needed because
    `np.unique` on quantile edges silently collapses ties, so a nominal 10-bin map is
    a 3-bin map on views like `cusum_shift_idx` (an index) -- the old code had no way
    to notice this had happened."""
    xtr = np.asarray(xtr, float)
    xte = np.asarray(xte, float)
    base = float(np.mean(ytr))
    if discrete:
        vals = np.unique(xtr)
        if len(vals) < 2:
            return None, 0
        rates = {v: float(ytr[xtr == v].mean()) for v in vals}
        idx = np.searchsorted(vals, xte)
        idx = np.clip(idx, 0, len(vals) - 1)
        return np.array([rates[vals[i]] for i in idx]), len(vals)
    edges = np.unique(np.quantile(xtr, np.linspace(0, 1, k + 1)))
    if len(edges) < 3:
        return None, 0
    tb = np.digitize(xtr, edges[1:-1])
    nb = len(edges) - 1
    rates = {b: (float(ytr[tb == b].mean()) if np.any(tb == b) else base)
             for b in range(nb)}
    return np.array([rates.get(b, base) for b in np.digitize(xte, edges[1:-1])]), nb


def _auc(y, s):
    if len(np.unique(y)) < 2 or np.std(s) < 1e-12:
        return 0.5
    return float(roc_auc_score(y, s))


def d1_shape_gain(x, y, seeds=tuple(range(N_SEEDS)), ks=KS, discrete=False,
                  return_per_k=False):
    """D1 = max_K [ AUROC(unconstrained K-bin map) - AUROC(isotonic) ], cross-fitted.

    Both sides are fitted on the same training folds and scored on the same held-out
    folds, so estimation noise is charged symmetrically. Taking the max over K is a
    legitimate statistic PROVIDED the resampling null is computed for the same max
    statistic -- which it is, since the null re-runs this exact function. K=5 buys
    power on sharp thresholds and small cells, K=20 on W shapes at large n.

    No `max(a, 1-a)` anywhere: the bin map carries its own direction by construction
    and `iso_fit` resolves the monotone direction on TRAIN.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    nan = (float("nan"), float("nan"), -1, {}) if return_per_k else (float("nan"), float("nan"), -1)
    if n_min(y) < MIN_NMIN or np.std(x) < 1e-12:
        return nan
    per_k = {k: [] for k in ks}
    iso_per_seed = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + s)
        gi, gk = [], {k: [] for k in ks}
        for tr, te in skf.split(x.reshape(-1, 1), y):
            if len(np.unique(y[te])) < 2:
                continue
            ir = iso_fit(x[tr], y[tr])
            if ir is None:
                continue
            gi.append(_auc(y[te], ir.predict(x[te])))
            for k in ks:
                u, _ = bin_fit_apply(x[tr], y[tr], x[te], k, discrete=discrete)
                if u is not None:
                    gk[k].append(_auc(y[te], u))
        if not gi:
            continue
        iso_per_seed.append(float(np.mean(gi)))
        for k in ks:
            if gk[k]:
                per_k[k].append(float(np.mean(gk[k])))
    if not iso_per_seed:
        return nan
    iso_m = float(np.mean(iso_per_seed))
    gains = {k: float(np.mean(v)) - iso_m for k, v in per_k.items() if v}
    if not gains:
        return nan
    argmax_k = max(gains, key=gains.get)
    # across-seed sd of the winning K, so the reported sd matches the reported gain
    sd = float(np.std([v - i for v, i in zip(per_k[argmax_k], iso_per_seed)])) \
        if len(per_k[argmax_k]) == len(iso_per_seed) else float("nan")
    if return_per_k:
        return gains[argmax_k], sd, argmax_k, gains
    return gains[argmax_k], sd, argmax_k


def d2_residual_chi2(x, y, k=10, seeds=(0,), discrete=False):
    """D2 = score test on the ISOTONIC RESIDUAL.

    Under H0 (the curve is monotone), E[y - iso(x) | x] = 0 everywhere. Cross-fit iso
    out-of-fold, bin x, and form X2 = sum_b n_b * rbar_b^2 / (pbar(1-pbar)).

    Statistically orthogonal to D1 in the way that matters: D1 is a RANKING statistic,
    so a bend in a low-density region barely moves it. D2 is a mean-residual statistic
    and sees exactly that. Two agreeing detectors with different failure modes is much
    stronger evidence than one."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    if n_min(y) < MIN_NMIN or np.std(x) < 1e-12:
        return float("nan"), 0
    stats = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + s)
        resid = np.full(len(y), np.nan)
        for tr, te in skf.split(x.reshape(-1, 1), y):
            ir = iso_fit(x[tr], y[tr])
            if ir is None:
                continue
            resid[te] = y[te] - ir.predict(x[te])
        m = np.isfinite(resid)
        if m.sum() < MIN_NMIN * 2:
            continue
        pbar = float(y[m].mean())
        denom = max(pbar * (1 - pbar), 1e-9)
        if discrete:
            groups = [np.where(m & (x == v))[0] for v in np.unique(x)]
        else:
            edges = np.unique(np.quantile(x[m], np.linspace(0, 1, k + 1)))
            if len(edges) < 3:
                continue
            b = np.digitize(x, edges[1:-1])
            groups = [np.where(m & (b == i))[0] for i in range(len(edges) - 1)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        stats.append(float(sum(len(g) * np.mean(resid[g]) ** 2 for g in groups) / denom))
    if not stats:
        return float("nan"), 0
    return float(np.mean(stats)), len(groups) - 1


def d3_slope_reversal(x, y, n_win=10, w_min_frac=0.05):
    """D3 = sup-t test for a LOCAL SLOPE SIGN REVERSAL (Chetverikov-2019 style).

    Over `n_win` quantile windows, for every ordered pair (A left of B):
        t_inc = (pA - pB)/SE   -- a violation of "increasing"
        t_dec = (pB - pA)/SE   -- a violation of "decreasing"
    T_inc = max t_inc, T_dec = max t_dec.

    H0 here is "monotone in SOME direction", which is a UNION of two one-sided nulls,
    so this is an intersection-union test: reject only if BOTH T_inc and T_dec are
    extreme. An IUT at level alpha per component is exactly level alpha for the union
    null, so no multiplicity correction is needed between the two. The p-value is
    therefore max(p_inc, p_dec) -- computed by the caller against a permutation null,
    the least-favourable (flat) configuration for this statistic.

    Also returns the violating windows, which is the characterisation output."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    n = len(y)
    w_min = max(20, int(w_min_frac * n))
    if n_min(y) < MIN_NMIN or np.std(x) < 1e-12 or n < 2 * w_min:
        return float("nan"), float("nan"), []
    q = np.quantile(x, np.linspace(0, 1, n_win + 1))
    q = np.unique(q)
    if len(q) < 3:
        return float("nan"), float("nan"), []
    b = np.digitize(x, q[1:-1])
    cells_ = []
    for i in range(len(q) - 1):
        idx = np.where(b == i)[0]
        if len(idx) >= w_min:
            cells_.append((i, float(y[idx].mean()), len(idx)))
    if len(cells_) < 2:
        return float("nan"), float("nan"), []
    t_inc, t_dec, viol = -np.inf, -np.inf, []
    for ai in range(len(cells_)):
        ia, pa, na = cells_[ai]
        for bi in range(ai + 1, len(cells_)):
            ib, pb, nb = cells_[bi]
            se = np.sqrt(max(pa * (1 - pa) / na + pb * (1 - pb) / nb, 1e-12))
            ti, td = (pa - pb) / se, (pb - pa) / se
            if ti > t_inc:
                t_inc = ti
            if td > t_dec:
                t_dec = td
            if max(ti, td) > 2.0:
                viol.append((ia, ib, round(float(max(ti, td)), 2)))
    return float(t_inc), float(t_dec), viol


# ══════════════════════════════════════════════════════════════════════════════
# Nulls
# ══════════════════════════════════════════════════════════════════════════════
def perm_labels(y, rng):
    """H0: no relationship at all. Kept for continuity with Step 217."""
    return rng.permutation(y)


def isoboot_labels(x, y, rng, shrink=1.0):
    """H0: THE RELATIONSHIP IS MONOTONE — the null that matches the hypothesis.

    Fit isotonic on the full data (the monotone MLE), then resample
    y* ~ Bernoulli(iso(x)) with x held fixed. This preserves the x-marginal, the
    prevalence AND a monotone signal, none of which label permutation does.

    `shrink` < 1 pulls the fitted curve toward the base rate; the plan runs shrink=0.9
    as a sensitivity check, because the in-sample isotonic plug-in bootstrap is not
    uniformly valid at the boundary of the monotone cone (Sen-Banerjee-Woodroofe 2010).
    That concern is mitigated here because the statistic is a CROSS-FITTED AUROC
    difference rather than a local functional of the isotonic estimator."""
    ir = iso_fit(x, y)
    if ir is None:
        return rng.permutation(y)
    p = np.clip(ir.predict(x), 1e-6, 1 - 1e-6)
    if shrink < 1.0:
        p = shrink * p + (1 - shrink) * float(np.mean(y))
    return (rng.random(len(p)) < p).astype(int)


def besag_clifford(stat_fn, resample_fn, observed, rng, n_target=10, b_max=999):
    """Sequential Monte Carlo p-value (Besag & Clifford 1991).

    Draw until `n_target` exceedances or `b_max` draws; p_hat = (h+1)/(b+1). A pair
    with a large p stops after ~20 draws while only genuinely extreme pairs pay the
    full budget, which is what makes testing all 682 pairs affordable after the
    n-independent `--screen 0.02` is removed.

    Returns (p_hat, h, b, null_values)."""
    vals, h, b = [], 0, 0
    while b < b_max and h < n_target:
        v = stat_fn(resample_fn(rng))
        b += 1
        if np.isfinite(v):
            vals.append(v)
            if v >= observed:
                h += 1
    return float((h + 1) / (b + 1)), h, b, np.asarray(vals, float)


def gpd_tail_p(null_vals, observed, tail_frac=0.10, min_tail=25):
    """Generalised-Pareto tail p-value (Knijnenburg et al. 2009, Bioinformatics 25:i161).

    BH at q=0.10 over 682 tests needs p <= 1.5e-4 at the top rank, which B=999 draws
    cannot resolve -- every truly strong pair would pile up at the same 1/(B+1) floor
    and BH would lose all resolution among them. Fitting a GPD to the upper tail of
    the null distribution extends the p-value below 1/B. Only used where the direct
    count is < 5, and always reported alongside `p_mc`."""
    v = np.asarray(null_vals, float)
    v = v[np.isfinite(v)]
    if len(v) < min_tail * 2 or not np.isfinite(observed):
        return float("nan")
    n_t = max(min_tail, int(tail_frac * len(v)))
    thr = np.sort(v)[-n_t]
    exc = v[v > thr] - thr
    if len(exc) < min_tail or observed <= thr:
        return float("nan")
    try:
        c, loc, scale = genpareto.fit(exc, floc=0.0)
        tail = float(genpareto.sf(observed - thr, c, loc=loc, scale=scale))
    except Exception:                                # noqa: BLE001
        return float("nan")
    if not np.isfinite(tail):
        return float("nan")
    return float(np.clip((n_t / len(v)) * tail, 1e-16, 1.0))


def simes(pvals):
    """Simes combination — valid under positive dependence (PRDS), which the three
    detectors are (same data, same folds, correlated statistics)."""
    p = np.sort(np.asarray([q for q in pvals if np.isfinite(q)], float))
    if len(p) == 0:
        return float("nan")
    m = len(p)
    return float(np.min(p * m / np.arange(1, m + 1)))


def bh_fdr(pvals, q=BH_Q):
    """Benjamini-Hochberg. Returns (qvalues, rejected) aligned to the input order.

    The reason this matters here: 682 pairs each tested at a per-pair 95th percentile
    yields ~34 expected false positives under the global null, and Step 217 found 32.
    At the family level that list is indistinguishable from noise; only after
    stratifying by cell size does a real signal appear. Family-level control is not
    optional for this question."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out, np.zeros(len(p), bool)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        val = p[order[rank]] * m / (rank + 1)
        prev = min(prev, val)
        out[order[rank]] = prev
    return out, np.nan_to_num(out, nan=1.0) <= q


def storey_pi0(pvals, lam=0.5):
    """Estimated fraction of true nulls — reported so the reader sees how much of the
    family is null before reading any survivor count."""
    p = np.asarray([q for q in pvals if np.isfinite(q)], float)
    if len(p) == 0:
        return float("nan")
    return float(min(1.0, np.mean(p > lam) / (1 - lam)))


# ══════════════════════════════════════════════════════════════════════════════
# Label-free marginal descriptors
# ══════════════════════════════════════════════════════════════════════════════
def kde_modes(x, grid=512, min_prominence=0.05):
    """(n_modes, 2nd-peak relative prominence, MODE PERCENTILE).

    `nonmono_shape_test.py:189` already computed the peak grid indices and then threw
    the LOCATION away at line 194, returning only the count. The count is useless here
    (Spearman with shape_gain = +0.014, p=0.72), but the location has never been
    tested and is exactly the label-free centre `r0` that transform T1 needs. That is
    the whole fix: keep `g[pk[argmax(prominences)]]` instead of discarding it."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 50 or np.std(x) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    try:
        kde = gaussian_kde(x)
    except Exception:                                # noqa: BLE001
        return float("nan"), float("nan"), float("nan")
    g = np.linspace(x.min(), x.max(), grid)
    d = kde(g)
    if not np.isfinite(d).all() or d.max() <= 0:
        return float("nan"), float("nan"), float("nan")
    pk, props = find_peaks(d, prominence=min_prominence * d.max())
    if len(pk) == 0:
        return 0.0, 0.0, float(np.mean(x < g[int(np.argmax(d))]))
    order = np.argsort(-props["prominences"])
    mode_x = float(g[pk[order[0]]])
    second = float(props["prominences"][order[1]] / d.max()) if len(pk) > 1 else 0.0
    return float(len(pk)), second, float(np.mean(x < mode_x))


def spike_frac(x):
    """Largest share of mass on one exact value — a discreteness flag, not a mode."""
    _, c = np.unique(np.asarray(x, float), return_counts=True)
    return float(c.max() / len(x))


def is_discrete(x, max_distinct=20):
    return int(len(np.unique(np.asarray(x, float))) <= max_distinct)
