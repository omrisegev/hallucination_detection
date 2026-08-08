"""
selective_metrics.py — selective prediction, risk-coverage / AURC, and the Neyman-Pearson
calibration rule from "Mind the Gap" (ICML 2026).

**Nothing in this repo implemented any of these before** — grep for `aurc`, `risk_coverage`,
`selective` across `spectral_utils/`, `scripts/` and `cluster/` returns nothing as of Step 218.
Our entire evaluation history is AUROC + bootstrap CI. This module is the new machinery.

CONVENTIONS (fixed here so every caller agrees)
----------------------------------------------
* `risk`   — per-sample score, **higher = more likely hallucinated** (the paper's phi(x, y)).
* `labels` — 1 = the model's answer was CORRECT, 0 = incorrect. This matches our pkl `label`
  key, and is the opposite polarity from "risk", so read the sign of every comparison carefully.
* A sample is **accepted** iff `risk <= tau` (their Eq. 45).

THE QUANTILE DIRECTION — READ THIS BEFORE CHANGING `calibrate_tau`
------------------------------------------------------------------
The paper contradicts itself. Sec. 4 and Appendix C.2 both say tau-hat is the
**(1-alpha)-quantile** of the calibration risk distribution. But Eq. 43 states the guarantee as
`P(phi <= tau | H0) <= alpha` where H0 = "this is a hallucination", and the decision rule is
`Accept if phi <= tau-hat`. Accepting at most an alpha fraction of hallucinations requires the
**alpha-quantile** of the incorrect samples' risk distribution.

Table 1 settles it empirically: selective accuracy *decreases* monotonically as alpha grows
(Shannon Drop, GSM8K/Qwen3-4B: 100 -> 100 -> 90.51). Under the (1-alpha)-quantile the threshold
would *loosen* as alpha shrinks and accuracy would move the other way. We implement the
alpha-quantile. See the digest's caveat 1.
"""
import numpy as np


# ── risk-coverage and AURC ───────────────────────────────────────────────────

def risk_coverage_curve(risk, labels):
    """(coverage, risk_at_coverage) for every prefix of the risk-ascending ordering.

    Coverage k/n = accepting the k lowest-risk samples; the curve's y value is the ERROR RATE
    within that accepted set. A perfect ranker puts every correct answer first, so its curve
    stays at 0 until coverage reaches the base accuracy.

    Ties are broken by the given order (stable sort), which matters on cells where a risk score
    saturates — e.g. `evidence_drop_risk` returns exactly 0.0 for every trace with no drop.
    """
    r = np.asarray(risk, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    if r.shape != y.shape:
        raise ValueError(f"risk {r.shape} and labels {y.shape} must align")
    n = r.size
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(r, kind="stable")
    errors = 1 - y[order]
    coverage = np.arange(1, n + 1) / n
    return coverage, np.cumsum(errors) / np.arange(1, n + 1)


def aurc(risk, labels, scale: float = 1000.0) -> float:
    """Area under the risk-coverage curve. **Lower is better.**

    The paper reports "AURC (x1000)" (Table 2 caption), so `scale=1000` is the default and the
    returned value is directly comparable to their 45.4 / 288.8 / etc.

    Computed as the mean of the per-coverage risks (the standard Geifman & El-Yaniv finite-sample
    estimator), not a trapezoid — with n equally spaced coverage points the two differ only in
    the endpoint weighting, and the mean is what the selective-prediction literature reports.
    """
    _, rc = risk_coverage_curve(risk, labels)
    return float(rc.mean() * scale) if rc.size else float("nan")


def selective_accuracy(risk, labels, tau: float) -> float:
    """Accuracy over the accepted set {risk <= tau} (their Eq. 13).

    Returns NaN when nothing is accepted — an empty accepted set has no accuracy, and reporting
    0.0 there would silently look like a catastrophic score instead of an undefined one. The
    paper's LogTokU Avg "0" entries at alpha=0.05 are the real thing (a populated but wholly
    incorrect accepted set), not this case.
    """
    r = np.asarray(risk, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    acc = r <= tau
    return float(y[acc].mean()) if acc.any() else float("nan")


def coverage(risk, tau: float) -> float:
    """Fraction of samples accepted at threshold tau. Always report this beside selective accuracy —
    a threshold that accepts 3 samples can hit 100% and mean nothing."""
    r = np.asarray(risk, dtype=np.float64)
    return float((r <= tau).mean()) if r.size else float("nan")


# ── Neyman-Pearson calibration (Appendix C.2) ────────────────────────────────

def calibrate_tau(cal_risk_incorrect, alpha: float, delta: float = None) -> float:
    """tau-hat from the risk scores of the INCORRECT calibration samples.

    `cal_risk_incorrect` must already be filtered to `D_cal = {(x,y) | y is incorrect}` — the
    paper is explicit that the threshold is computed on failures only.

    With `delta=None` (default) this is Eq. 43: the plain empirical alpha-quantile.

    With `delta` set, it additionally applies Eq. 44's Binomial-tail finite-sample correction,
    which the paper states but never supplies a delta for.

    THE CORRECTION, DERIVED (the paper only gives the inequality, and its prose "we seek the
    smallest tau" is the wrong direction — any sufficiently small tau satisfies the constraint,
    so the useful threshold is the LARGEST one that still does):

      Setting tau = s_(k), the k-th smallest calibration risk, means k of n known hallucinations
      fall in the accept region. We need a (1-delta)-confidence guarantee that the TRUE
      acceptance probability p = P(phi <= tau | H0) is at most alpha. The one-sided
      Clopper-Pearson upper bound for k successes in n is at most alpha exactly when
      `P(Binom(n, alpha) <= k) <= delta`. That cdf increases in k, so the admissible k form a
      prefix and we take the largest.

      This is strictly more conservative than the plain quantile — a lower tau, fewer
      acceptances — and the gap is widest precisely where our calibration sets are smallest,
      which is the regime the paper's own protocol lands in at high model accuracy.

    Whichever is used MUST be reported — "we used Appendix C.2" is not a specification.
    """
    s = np.sort(np.asarray(cal_risk_incorrect, dtype=np.float64))
    n = s.size
    if n == 0:
        raise ValueError(
            "empty calibration set: no incorrect samples in D_cal. The paper's rule is undefined "
            "here — this is the degenerate case that appears when a cell's accuracy is so high "
            "that the negative class vanishes."
        )
    if delta is None:
        # Eq. 43. `method='lower'` picks an actual observed risk value rather than interpolating
        # between two, which keeps tau on the empirical support.
        return float(np.quantile(s, alpha, method="lower"))

    from scipy.stats import binom
    k = np.arange(1, n + 1)
    admissible = binom.cdf(k, n, alpha) <= delta
    if not admissible.any():
        # Even accepting a single hallucination cannot be certified at this (n, alpha, delta):
        # accept nothing, i.e. a threshold strictly below the smallest observed risk.
        return float(s[0] - 1.0)
    return float(s[int(k[admissible].max()) - 1])


# ── the repeated-split estimator ─────────────────────────────────────────────

def repeated_split_eval(risk, labels, alpha, n_splits: int = 100, seed: int = 0,
                        delta: float = None) -> dict:
    """Selective accuracy at alpha, averaged over `n_splits` random 50/50 cal/eval partitions.

    WHY REPEATED SPLITS. The paper reports `+/- 0.31`, `+/- 0.26`, `+/- 0.21` on its accuracy
    column, but its protocol is **greedy, tau = 0** — decoding contributes no variance at all.
    The only stochastic element in the whole pipeline is the 50/50 `D_cal` / `D_eval` partition,
    so those error bars can only come from repeating it. A single split is not their estimator
    and will not reproduce their spread.

    Also returns `n_cal_incorrect`, which must be reported alongside every number: the threshold
    is a quantile of the incorrect calibration samples only, and at high accuracy that set gets
    small enough that the alpha-quantile degenerates into the minimum order statistic.
    """
    r = np.asarray(risk, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    n = r.size
    rng = np.random.default_rng(seed)

    accs, covs, taus, n_incorrect = [], [], [], []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, ev = perm[: n // 2], perm[n // 2:]
        cal_wrong = r[cal][y[cal] == 0]
        n_incorrect.append(cal_wrong.size)
        if cal_wrong.size == 0:
            continue                      # undefined split; counted in n_cal_incorrect
        tau = calibrate_tau(cal_wrong, alpha, delta)
        taus.append(tau)
        accs.append(selective_accuracy(r[ev], y[ev], tau))
        covs.append(coverage(r[ev], tau))

    accs = np.asarray(accs, dtype=np.float64)
    valid = accs[~np.isnan(accs)]
    return {
        "alpha": alpha,
        "selective_accuracy": float(valid.mean()) if valid.size else float("nan"),
        "selective_accuracy_sd": float(valid.std(ddof=1)) if valid.size > 1 else float("nan"),
        "coverage": float(np.nanmean(covs)) if covs else float("nan"),
        "tau_mean": float(np.mean(taus)) if taus else float("nan"),
        "n_splits_used": int(valid.size),
        "n_splits_requested": int(n_splits),
        "n_cal_incorrect_mean": float(np.mean(n_incorrect)) if n_incorrect else 0.0,
        "n_cal_incorrect_min": int(np.min(n_incorrect)) if n_incorrect else 0,
        "delta": delta,
    }


# ── known-answer tests ───────────────────────────────────────────────────────

def smoke() -> None:
    """Known-answer tests. Every expected value here is derivable by hand."""
    # 1. PERFECT ranker: all correct answers have lower risk than all incorrect ones.
    #    Its risk-coverage curve is 0 until coverage passes the base accuracy, so AURC is small.
    y = np.array([1, 1, 1, 1, 0, 0])            # base accuracy 4/6
    perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 1.0])
    cov, rc = risk_coverage_curve(perfect, y)
    assert np.allclose(cov, np.arange(1, 7) / 6)
    assert np.allclose(rc[:4], 0.0), rc                      # no error while only correct accepted
    assert np.isclose(rc[-1], 2 / 6)                         # full coverage -> base error rate
    # hand-computed: mean of [0,0,0,0,1/5,2/6] = 0.0733... -> x1000
    assert np.isclose(aurc(perfect, y), (0 + 0 + 0 + 0 + 1 / 5 + 2 / 6) / 6 * 1000), aurc(perfect, y)

    # 2. WORST ranker (perfectly inverted) must score strictly higher AURC than the perfect one,
    #    and full coverage must agree between them (same set, same base error).
    worst = -perfect
    assert aurc(worst, y) > aurc(perfect, y)
    assert np.isclose(risk_coverage_curve(worst, y)[1][-1], risk_coverage_curve(perfect, y)[1][-1])

    # 3. A ranker on an all-correct set has zero risk everywhere -> AURC exactly 0.
    assert aurc(np.array([0.5, 0.1, 0.9]), np.array([1, 1, 1])) == 0.0

    # 4. Selective accuracy at a threshold above every risk == base accuracy at coverage 1.0.
    assert np.isclose(selective_accuracy(perfect, y, tau=99.0), 4 / 6)
    assert np.isclose(coverage(perfect, tau=99.0), 1.0)
    # ...and a threshold below every risk accepts nothing -> NaN, not 0.0.
    assert np.isnan(selective_accuracy(perfect, y, tau=-1.0))
    assert coverage(perfect, tau=-1.0) == 0.0

    # 5. calibrate_tau takes the ALPHA-quantile of the incorrect risks (not 1-alpha).
    #    With risks 1..10 and alpha=0.1, method='lower' gives the 10th percentile = 1.0.
    wrong = np.arange(1.0, 11.0)
    assert calibrate_tau(wrong, 0.1) == 1.0, calibrate_tau(wrong, 0.1)
    #    A LARGER alpha must give a LARGER (more permissive) threshold — the direction that
    #    produces Table 1's monotone decrease in selective accuracy as alpha grows.
    assert calibrate_tau(wrong, 0.5) > calibrate_tau(wrong, 0.1)
    assert calibrate_tau(wrong, 0.9) > calibrate_tau(wrong, 0.5)

    # 6. Eq. 44's correction is strictly more conservative than the plain quantile, in the
    #    direction that matters: a LOWER tau, accepting fewer hallucinations.
    #    Hand-check, n=10, alpha=0.5, delta=0.05: cdf(1;10,.5)=0.0107 <= 0.05 but
    #    cdf(2;10,.5)=0.0547 > 0.05, so k_max=1 and tau = s_(1) = 1.0, against a plain
    #    alpha-quantile of 5.0. (The earlier upper-tail form returned 9.0 — MORE permissive
    #    than the uncorrected threshold, which is the opposite of a safety correction.)
    t_plain = calibrate_tau(wrong, 0.5)
    t_corr = calibrate_tau(wrong, 0.5, delta=0.05)
    assert t_plain == 5.0, t_plain
    assert t_corr == 1.0, t_corr
    assert t_corr <= t_plain, (t_corr, t_plain)
    #    More calibration data must relax the correction toward the plain quantile.
    big = np.arange(1.0, 1001.0)
    gap_small = calibrate_tau(wrong, 0.5) - calibrate_tau(wrong, 0.5, delta=0.05)
    gap_big = (calibrate_tau(big, 0.5) - calibrate_tau(big, 0.5, delta=0.05)) / 100.0
    assert gap_big < gap_small, (gap_big, gap_small)
    #    And with too little data to certify anything, it accepts nothing.
    assert calibrate_tau(np.array([3.0, 4.0]), 0.01, delta=0.05) < 3.0

    # 7. An empty calibration set raises rather than silently returning a threshold.
    try:
        calibrate_tau([], 0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("empty D_cal must raise")

    # 8. repeated_split_eval on a PERFECTLY separable problem hits the analytic ceiling, which
    #    is NOT 100%. The alpha-quantile rule deliberately admits an alpha fraction of the
    #    hallucinations, so even a flawless risk score accepts every correct answer plus
    #    alpha*(1-p) of the incorrect ones:
    #        selective accuracy -> p / (p + alpha*(1-p))
    #    At p ~ 0.5, alpha = 0.05 that is ~0.952, not ~1.0. Asserting this exact value is what
    #    makes the test a check on the alpha SEMANTICS rather than on "is the number big".
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 400)
    rr = np.where(yy == 1, rng.normal(0, 0.1, 400), rng.normal(5, 0.1, 400))
    out = repeated_split_eval(rr, yy, alpha=0.05, n_splits=50, seed=1)
    p = float(yy.mean())
    expected = p / (p + 0.05 * (1 - p))
    assert abs(out["selective_accuracy"] - expected) < 0.02, (out["selective_accuracy"], expected)
    assert out["n_splits_used"] == 50, out
    assert out["n_cal_incorrect_min"] > 50, out       # ~100 incorrect in each cal half

    # 8b. The same separable data at a LOOSER alpha must give LOWER selective accuracy and
    #     HIGHER coverage — the monotonicity that Table 1 exhibits across alpha in
    #     {0.05, 0.10, 0.50}, and the empirical signature of the alpha- (not 1-alpha-) quantile.
    prev_acc, prev_cov = 1.1, -0.1
    for a in (0.05, 0.10, 0.50):
        o = repeated_split_eval(rr, yy, alpha=a, n_splits=30, seed=1)
        assert o["selective_accuracy"] < prev_acc, (a, o["selective_accuracy"], prev_acc)
        assert o["coverage"] > prev_cov, (a, o["coverage"], prev_cov)
        prev_acc, prev_cov = o["selective_accuracy"], o["coverage"]

    # 9. Sanity on the direction of the whole pipeline: a GOOD risk score must yield higher
    #    selective accuracy than a random one on the same data.
    rand = rng.normal(0, 1, 400)
    out_rand = repeated_split_eval(rand, yy, alpha=0.05, n_splits=50, seed=1)
    assert out["selective_accuracy"] > out_rand["selective_accuracy"], (out, out_rand)

    print("selective_metrics.smoke: PASS (9 checks)")


if __name__ == "__main__":
    smoke()
