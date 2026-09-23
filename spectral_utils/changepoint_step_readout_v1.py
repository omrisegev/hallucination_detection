"""Change-point readouts of a risk series: which step is the FIRST error.

Every readout in this module answers one question -- given a risk-oriented series for
one answer (higher = more suspect), which index is the first error -- and they differ
only in what "first" means:

  argmax          where the series is largest.  The project's incumbent readout.
  first_crossing  the first index reaching a within-answer quantile of its own series.
  page_cusum      the first index at which accumulated excess evidence crosses h,
                  and the start of the excursion that produced it.
  bocpd           the index at which a Bayesian online change-point filter puts its
                  segment boundary.

The motivation is a measured asymmetry, not an aesthetic one: on this population our
argmax misses land AFTER the true first error far more often than before it (+9.1 pp on
GSM8K up to +31.6 on Omni-MATH).  An argmax has no ordering prior at all -- it has to
beat every competing step -- while a sequential rule commits to the earliest index that
clears a bar and never has to win that competition.

Two conventions are fixed here because they are worth more than a percentage point on
this population and are silent if left implicit:

* **Index 0 must be reachable.**  12.25% of erroneous ProcessBench answers have their
  first error at step 0 (17.9% on GSM8K).  A rule that structurally cannot return 0
  forfeits that mass.  Every function here can return 0; where a convention resists it
  (BOCPD's reset mass at t=0 is the hazard by construction) the docstring says so.
* **The no-alarm fallback is part of the rule, not an implementation detail.**  When
  CUSUM never crosses h, `fallback="statistic"` predicts the argmax of the CUSUM
  statistic itself, `fallback="series"` the argmax of the raw series.  These are
  different rules and are reported separately; a fallback must never silently carry a
  result.

BOCPD here is the reset-BEFORE-observation Gaussian product-partition model that
`docs/reviews/bocpd_boundary_audit_2026-09-07.md` verified against brute-force partition
enumeration.  The older `temporal_models.bocpd_gaussian` is deliberately NOT used: that
audit found it scores the reset branch with the prior predictive but leaves the fresh
segment's sufficient statistics unupdated, which mixes the two conventions.

Borrowed ideas, deliberately not transplants (project rule: tailor, never transplant).
Page's CUSUM and Adams-MacKay BOCPD are cited as mechanisms; nothing here is labelled
with an author's name or claims to be their method.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "zscore_within",
    "argmax_readout",
    "first_crossing_readout",
    "page_cusum",
    "page_cusum_readout",
    "bocpd_filter",
    "bocpd_readout",
    "token_index_to_step",
]


def _series(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("expected a non-empty finite scalar series")
    return x


def zscore_within(values: np.ndarray) -> np.ndarray:
    """Standardize a series against its own mean and spread.

    A constant series becomes zeros rather than NaN, so a degenerate answer falls
    through to index 0 instead of poisoning the readout.
    """
    x = _series(values)
    std = x.std()
    return (x - x.mean()) / std if std > 1e-12 else np.zeros_like(x)


# --------------------------------------------------------------------------- level
def argmax_readout(values: np.ndarray) -> int:
    """The incumbent: the largest value, earliest index on a tie."""
    return int(np.argmax(_series(values)))


def first_crossing_readout(values: np.ndarray, q: float = 0.90) -> int:
    """First index reaching the within-answer quantile ``q`` of this answer's series.

    Label-free: the bar is computed from the answer alone.  As ``q -> 1`` the rule
    degenerates into ``argmax_readout`` by construction, which is exactly why a
    first-crossing family tested on a level series is dominated by its own endpoint --
    the crossing has to be paired with a statistic whose *ordering* carries the signal.
    """
    x = _series(values)
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    hit = np.flatnonzero(x >= np.quantile(x, q))
    return int(hit[0]) if len(hit) else int(np.argmax(x))


# --------------------------------------------------------------------------- CUSUM
def page_cusum(values: np.ndarray, k: float = 0.5, standardize: bool = True) -> dict:
    """One-sided upward Page CUSUM, in closed form.

    ``S_t = max(0, S_{t-1} + z_t - k)`` is a non-linear recursion, but it has an exact
    vectorized form: with ``C`` the prefix sums of ``z - k`` (``C[0] = 0``),
    ``S_t = C[t+1] - min_{i <= t+1} C[i]``.  That matters here only because it also hands
    back the change-point *estimate* for free -- the running argmin IS the start of the
    excursion, which is the quantity Page's procedure estimates and the one whose
    semantics are "where the drift began" rather than "when I became sure".

    Returns ``statistic`` (S_t), ``onset`` (start-of-excursion index per t) and
    ``crossed`` conveniences.
    """
    z = zscore_within(values) if standardize else _series(values)
    increments = z - float(k)
    prefix = np.concatenate([[0.0], np.cumsum(increments)])
    running_min = np.minimum.accumulate(prefix)
    statistic = prefix[1:] - running_min[1:]
    # Last index attaining the running minimum: `<=` keeps the latest tie, which is the
    # start of the CURRENT excursion rather than of an older, already-closed one.
    is_min = prefix <= running_min
    onset = np.maximum.accumulate(np.where(is_min, np.arange(len(prefix)), -1))[1:]
    return {"statistic": statistic, "onset": onset.astype(int), "prefix": prefix}


def page_cusum_readout(values: np.ndarray, k: float = 0.5, h: float = 5.0,
                       estimator: str = "alarm", fallback: str = "statistic") -> int:
    """Index chosen by the CUSUM rule.

    ``estimator="alarm"`` returns the first index whose statistic reaches ``h``;
    ``estimator="onset"`` returns the start of the excursion that raised that alarm.
    When no alarm fires, ``fallback="statistic"`` returns the argmax of S (the point of
    greatest accumulated evidence, which is the statistic's own answer) and
    ``fallback="series"`` returns the argmax of the raw series.
    """
    if estimator not in ("alarm", "onset"):
        raise ValueError("estimator must be 'alarm' or 'onset'")
    if fallback not in ("statistic", "series"):
        raise ValueError("fallback must be 'statistic' or 'series'")
    run = page_cusum(values, k=k)
    hit = np.flatnonzero(run["statistic"] >= float(h))
    if len(hit):
        alarm = int(hit[0])
        return alarm if estimator == "alarm" else int(run["onset"][alarm])
    if fallback == "series":
        return argmax_readout(values)
    best = int(np.argmax(run["statistic"]))
    return best if estimator == "alarm" else int(run["onset"][best])


# --------------------------------------------------------------------------- BOCPD
def _normal_logpdf(x: float, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2 * np.pi * variance) + (x - mean) ** 2 / variance)


def bocpd_filter(values: np.ndarray, hazard: float = 1.0 / 32.0,
                 observation_variance: float = 1.0, prior_mean: float = 0.0,
                 prior_variance: float = 1.0, r_max: int | None = None,
                 standardize: bool = True) -> dict:
    """Gaussian product-partition BOCPD, reset BEFORE the current datum.

    Same recursion as `spectral_utils.fused_trajectory_readouts.bocpd_filter`, which was
    verified to 1e-12 against brute-force enumeration of every partition of every prefix
    of a five-observation series.  Reproduced here rather than imported because that
    module is not on this branch; `tests/test_changepoint_step_readout_v1.py` re-runs the
    same enumeration against this copy so the two cannot drift apart silently.

    ``r_max`` truncates the run-length posterior, folding the overflow mass into the
    longest kept run.  ``None`` is exact and is what the test uses; token-granularity
    callers pass a finite cap because the untruncated filter is O(T^2).

    Returns, all length T:
      ``reset_probability``  P(run length resets at t).  At t=0 there is no preceding
                             run, so this is the hazard by construction -- a constant,
                             which is why `reset` alone is a poor onset curve at the
                             first index and `rise` is the primary one.
      ``rise``               reset probability weighted by how surprising x_t was under
                             the continuation predictive, clipped at zero.  Defined and
                             informative at t=0, so index 0 stays reachable.
      ``level``              posterior mean of the segment level.
      ``log_predictive``     one-step predictive log evidence.
    """
    x = zscore_within(values) if standardize else _series(values)
    if not 0.0 < hazard < 1.0 or min(observation_variance, prior_variance) <= 0:
        raise ValueError("invalid BOCPD parameters")
    posterior = np.ones(1)
    means = np.array([float(prior_mean)])
    variances = np.array([float(prior_variance)])
    levels, resets, rises, logs = [], [], [], []
    for t, value in enumerate(x):
        continuation_mean = float(posterior @ means)
        continuation_var = float(
            posterior @ (variances + (means - continuation_mean) ** 2)) + observation_variance
        if t == 0:
            weights, candidate_mean, candidate_var = np.ones(1), means, variances
        else:
            weights = np.r_[hazard, (1 - hazard) * posterior]
            candidate_mean = np.r_[prior_mean, means]
            candidate_var = np.r_[prior_variance, variances]
        with np.errstate(divide="ignore"):
            log_mass = np.log(weights) + _normal_logpdf(
                value, candidate_mean, candidate_var + observation_variance)
        log_evidence = float(logsumexp(log_mass))
        posterior = np.exp(log_mass - log_evidence)
        gain = candidate_var / (candidate_var + observation_variance)
        means = candidate_mean + gain * (value - candidate_mean)
        variances = (1 - gain) * candidate_var
        if r_max is not None and len(posterior) > r_max:
            # Fold the tail into the oldest kept run: its mass still competes, it just
            # stops carrying its own sufficient statistics.
            tail = slice(r_max - 1, None)
            weight = posterior[tail].sum()
            if weight > 0:
                means[r_max - 1] = float(posterior[tail] @ means[tail]) / weight
                variances[r_max - 1] = float(posterior[tail] @ variances[tail]) / weight
            posterior = np.r_[posterior[:r_max - 1], weight]
            means, variances = means[:r_max], variances[:r_max]
        reset = float(posterior[0]) if t else float(hazard)
        levels.append(float(posterior @ means))
        resets.append(reset)
        rises.append(reset * max(
            float((value - continuation_mean) / np.sqrt(continuation_var)), 0.0))
        logs.append(log_evidence)
    return {"level": np.asarray(levels), "reset_probability": np.asarray(resets),
            "rise": np.asarray(rises), "log_predictive": np.asarray(logs)}


def bocpd_readout(values: np.ndarray, curve: str = "rise", hazard: float = 1.0 / 32.0,
                  r_max: int | None = None, threshold: float | None = None) -> int:
    """Index chosen by the BOCPD rule.

    ``curve="rise"`` (default) or ``"reset"``.  ``threshold=None`` takes the argmax of
    the curve -- still an argmax, but of a change-point posterior rather than of the
    level, so it responds to a shift against the preceding run instead of to raw size.
    A numeric ``threshold`` makes it a first-crossing rule on that curve instead.
    """
    if curve not in ("rise", "reset"):
        raise ValueError("curve must be 'rise' or 'reset'")
    out = bocpd_filter(values, hazard=hazard, r_max=r_max)
    series = out["rise"] if curve == "rise" else out["reset_probability"]
    if threshold is None:
        return int(np.argmax(series))
    hit = np.flatnonzero(series >= threshold)
    return int(hit[0]) if len(hit) else int(np.argmax(series))


# --------------------------------------------------------------------- granularity
def token_index_to_step(token_index: int, spans: np.ndarray) -> int:
    """Map a token index to the step whose span contains it.

    ``spans`` is the ``[steps, 2]`` half-open token span array, already 0-based WITHIN
    the answer (do not subtract the answer's token offset -- that bug is recorded in the
    handoff).  A token past the last span is clamped to the last step.
    """
    spans = np.asarray(spans, dtype=int)
    if spans.ndim != 2 or spans.shape[1] != 2 or not len(spans):
        raise ValueError("expected a [steps, 2] span array")
    return int(np.clip(np.searchsorted(spans[:, 0], token_index, side="right") - 1,
                       0, len(spans) - 1))
