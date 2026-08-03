"""
localization_metrics.py — step-level error localization: the paper's SLA and ProcessBench's
own official F1.

WHY BOTH METRICS
----------------
"Mind the Gap" reports only its own **SLA** (Step-level Localization Accuracy), which appears in
no other paper, so an SLA number is comparable to their Table 3 and to nothing else.
ProcessBench's **official F1** is what the whole process-reward-model literature reports. Running
both costs one extra function and is the difference between "we reproduced their table" and "we
can be placed on the benchmark".

=============================================================================================
PRE-REGISTERED CHOICES — fixed here BEFORE any number was computed
=============================================================================================
The paper defines `Delta_j` at the **token** level (Sec. 3.3) but SLA at the **step** level
("the first step t where Delta_t exceeds a step-wise threshold", Sec. 4), and never says how one
becomes the other. It also never says what to do with rows that contain no error, nor whether
"matches" allows any tolerance, nor how the step-wise threshold is set. Those are the dominant
free parameters for Table 3. Ours:

1. TOKEN -> STEP.  `Delta_t := min over tokens in step t of the token-level flux`, reported as a
   positive risk `-min(...)`. Rationale: the method is explicitly a *worst-case* Evidence Drop
   detector (`phi` averages the M most negative fluxes), so the worst drop inside a step is the
   reading consistent with the sequence-level statistic. `AGGREGATIONS['mean']` is provided and
   reported as a documented sensitivity, never as the headline.

2. FLUX ATTRIBUTION.  Flux `j = E[j+1] - E[j]` is attributed to the step containing token
   **j+1**, i.e. the step being entered or continued. A drop that happens as the model moves
   into an erroneous step belongs to that step, not to the sound one before it.

3. SMOOTHING IS GLOBAL.  The EMA runs over the whole chain once, then fluxes are partitioned by
   step. Smoothing each step independently would reset the filter at every boundary and
   manufacture a drop at each one.

4. ROWS WITH `label == -1` (no erroneous step) are EXCLUDED from SLA — there is no correct step
   to hit, so including them would only measure the abstention rate. They ARE used by
   ProcessBench F1, which scores exactly that abstention. This is the main reason to report both.

5. "MATCHES" is the exact step index. A +/-1-step tolerance is reported as a secondary column,
   because step boundaries in a model-written chain are somewhat arbitrary.

6. THE STEP-WISE THRESHOLD is the `alpha`-quantile of the per-step risk scores observed on
   erroneous steps in a calibration split — the same Neyman-Pearson construction the paper uses
   at sequence level (`selective_metrics.calibrate_tau`), applied to steps. The paper gives no
   rule at all here.
"""
import numpy as np

from evidence_drop import ema, _flux_tol
from selective_metrics import calibrate_tau

NO_ERROR = -1

AGGREGATIONS = {
    # name -> reducer over the fluxes attributed to one step; all return a POSITIVE risk.
    "min": lambda f: float(-f.min()) if f.size else 0.0,     # pre-registered headline
    "mean": lambda f: float(-f.mean()) if f.size else 0.0,   # documented sensitivity
}


def step_drop_scores(evidence, step_spans, ema_span: int = 5, aggregation: str = "min"):
    """Per-step risk scores from a per-token evidence series. Higher = more suspect.

    `evidence` is oriented higher = more evidence (see `evidence_drop.DROP_SIGN_CONVENTION`).
    `step_spans[i]` is the half-open token range of step i, or None if that step mapped to no
    token (in which case its score is NaN rather than 0.0 — an unmeasurable step must not look
    like a confidently clean one).
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; known: {sorted(AGGREGATIONS)}")
    reduce = AGGREGATIONS[aggregation]

    e = np.asarray(evidence, dtype=np.float64)
    if e.size < 2:
        return np.full(len(step_spans), np.nan)
    flux = np.diff(ema(e, ema_span))               # flux[j] = Etilde[j+1] - Etilde[j]
    tol = _flux_tol(e)

    out = np.full(len(step_spans), np.nan)
    for i, span in enumerate(step_spans):
        if span is None:
            continue
        lo, hi = span
        # Attribution rule 2: flux j belongs to the step containing token j+1, so the fluxes
        # for tokens [lo, hi) are flux indices [lo-1, hi-1), clipped to the valid range.
        a, b = max(lo - 1, 0), min(hi - 1, flux.size)
        seg = flux[a:b]
        neg = seg[seg < -tol]                      # only genuine drops, not float jitter
        out[i] = reduce(neg) if neg.size else 0.0
    return out


def predict_first_error(step_scores, threshold: float) -> int:
    """First step whose risk exceeds `threshold`, or NO_ERROR if none does.

    NaN steps (unmapped) are skipped rather than treated as clean, so a tokenization gap cannot
    be mistaken for evidence of correctness.
    """
    s = np.asarray(step_scores, dtype=np.float64)
    hits = np.flatnonzero(np.isfinite(s) & (s > threshold))
    return int(hits[0]) if hits.size else NO_ERROR


# ── the two metrics ──────────────────────────────────────────────────────────

def sla(preds, labels, tolerance: int = 0) -> float:
    """Step-level Localization Accuracy over rows that HAVE an erroneous step (rule 4).

    A prediction of NO_ERROR on such a row counts as a miss (the method failed to fire), which
    is the only reading that keeps SLA a localization metric rather than a detection one.
    """
    p = np.asarray(preds, dtype=int)
    y = np.asarray(labels, dtype=int)
    m = y != NO_ERROR
    if not m.any():
        return float("nan")
    hit = (p[m] != NO_ERROR) & (np.abs(p[m] - y[m]) <= tolerance)
    return float(hit.mean())


def processbench_f1(preds, labels) -> dict:
    """ProcessBench's official metric.

    Two accuracies, then their harmonic mean:
      * `acc_erroneous`  — on rows WITH an error, did we name the exact first erroneous step?
      * `acc_correct`    — on rows with NO error, did we correctly abstain (predict -1)?
      * `f1`             — harmonic mean of the two.

    The harmonic mean is what makes the benchmark hard: a detector that flags everything scores
    well on the first and zero on the second, and F1 collapses. Reporting either accuracy alone
    is the standard way to look good on ProcessBench without being useful.
    """
    p = np.asarray(preds, dtype=int)
    y = np.asarray(labels, dtype=int)
    err, cor = y != NO_ERROR, y == NO_ERROR

    acc_err = float((p[err] == y[err]).mean()) if err.any() else float("nan")
    acc_cor = float((p[cor] == NO_ERROR).mean()) if cor.any() else float("nan")
    if not np.isfinite(acc_err) or not np.isfinite(acc_cor) or (acc_err + acc_cor) == 0:
        f1 = 0.0 if (np.isfinite(acc_err) and np.isfinite(acc_cor)) else float("nan")
    else:
        f1 = 2 * acc_err * acc_cor / (acc_err + acc_cor)
    return {
        "acc_erroneous": acc_err,
        "acc_correct": acc_cor,
        "f1": f1,
        "n_erroneous": int(err.sum()),
        "n_correct": int(cor.sum()),
    }


def calibrate_step_threshold(step_scores_by_row, labels, alpha: float, delta: float = None):
    """Step-wise threshold from the risk scores of the ANNOTATED erroneous steps (rule 6).

    Mirrors the sequence-level construction: gather the scores the method assigns to the steps
    that are *known* to be wrong, and take the alpha-quantile so that at most an alpha fraction
    of true errors fall below the firing threshold.
    """
    vals = []
    for scores, lab in zip(step_scores_by_row, labels):
        if lab == NO_ERROR:
            continue
        s = np.asarray(scores, dtype=np.float64)
        if 0 <= lab < s.size and np.isfinite(s[lab]):
            vals.append(s[lab])
    if not vals:
        raise ValueError("no annotated erroneous step had a finite score — cannot calibrate")
    # `calibrate_tau` returns the alpha-quantile of a risk distribution; here a LOW score on a
    # known-bad step is the failure mode, so the same quantile is what we want to fire above.
    return calibrate_tau(np.asarray(vals), alpha, delta)


def evaluate(step_scores_by_row, labels, alpha: float = 0.1, delta: float = None,
             n_splits: int = 100, seed: int = 0) -> dict:
    """Full SLA + F1 evaluation with repeated calibration splits (same estimator as A3).

    The threshold is fitted on the calibration half and applied to the evaluation half, so no
    row contributes to both. Reported with a spread, for the same reason as at sequence level:
    with greedy decoding the split is the only source of variance.
    """
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    rng = np.random.default_rng(seed)
    out = {"sla": [], "sla_tol1": [], "f1": [], "acc_erroneous": [], "acc_correct": [], "tau": []}

    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, ev = perm[: n // 2], perm[n // 2:]
        try:
            tau = calibrate_step_threshold([step_scores_by_row[i] for i in cal],
                                           labels[cal], alpha, delta)
        except ValueError:
            continue
        preds = [predict_first_error(step_scores_by_row[i], tau) for i in ev]
        f = processbench_f1(preds, labels[ev])
        out["tau"].append(tau)
        out["sla"].append(sla(preds, labels[ev], 0))
        out["sla_tol1"].append(sla(preds, labels[ev], 1))
        out["f1"].append(f["f1"])
        out["acc_erroneous"].append(f["acc_erroneous"])
        out["acc_correct"].append(f["acc_correct"])

    res = {"alpha": alpha, "delta": delta, "n_splits_used": len(out["sla"]), "n_rows": n}
    for k, v in out.items():
        a = np.asarray(v, dtype=np.float64)
        a = a[np.isfinite(a)]
        res[k] = float(a.mean()) if a.size else float("nan")
        res[f"{k}_sd"] = float(a.std(ddof=1)) if a.size > 1 else float("nan")
    return res


# ── known-answer tests ───────────────────────────────────────────────────────

def smoke() -> None:
    # 1. A planted drop inside step 2 must produce the highest risk on step 2 and ~0 elsewhere.
    #    Evidence is flat, then falls sharply within the third step's token range.
    spans = [(0, 20), (20, 40), (40, 60), (60, 80)]
    ev = np.zeros(80)
    ev[45:] = -2.0                       # the cliff lands inside step 2
    s = step_drop_scores(ev, spans, ema_span=5)
    assert np.argmax(s) == 2, s
    assert s[0] == 0.0 and s[1] == 0.0, s          # nothing before the cliff
    assert s[2] > 0.1, s

    # 2. `predict_first_error` fires on the FIRST crossing, and abstains when none crosses.
    assert predict_first_error([0.0, 0.0, 0.9, 0.95], 0.5) == 2
    assert predict_first_error([0.0, 0.6, 0.0, 0.9], 0.5) == 1     # first, not largest
    assert predict_first_error([0.0, 0.1, 0.2], 0.5) == NO_ERROR
    # NaN (unmapped) steps are skipped, not treated as clean OR as a hit.
    assert predict_first_error([np.nan, 0.9], 0.5) == 1
    assert predict_first_error([np.nan, 0.1], 0.5) == NO_ERROR

    # 3. SLA: only rows with an error count; abstaining on such a row is a miss.
    preds = [2, 1, NO_ERROR, 0]
    labs = [2, 3, 1, NO_ERROR]
    #  row0 hit, row1 miss (1 vs 3), row2 miss (abstained), row3 excluded  -> 1/3
    assert np.isclose(sla(preds, labs), 1 / 3), sla(preds, labs)
    #  with +/-1 tolerance row1 is still a miss (|1-3| = 2)                 -> 1/3
    assert np.isclose(sla(preds, labs, tolerance=1), 1 / 3)
    #  but a prediction one step off IS a hit under tolerance 1
    assert sla([2], [3], tolerance=1) == 1.0 and sla([2], [3], tolerance=0) == 0.0
    #  all-correct rows only -> undefined, not 0
    assert np.isnan(sla([NO_ERROR], [NO_ERROR]))

    # 4. ProcessBench F1: the harmonic mean punishes a flag-everything detector.
    #    Two erroneous rows (both located), two clean rows (both wrongly flagged).
    f = processbench_f1([0, 1, 0, 0], [0, 1, NO_ERROR, NO_ERROR])
    assert f["acc_erroneous"] == 1.0 and f["acc_correct"] == 0.0, f
    assert f["f1"] == 0.0, f                       # perfect recall, zero abstention -> 0
    #    ...and a perfect detector scores 1.0.
    f = processbench_f1([0, 1, NO_ERROR, NO_ERROR], [0, 1, NO_ERROR, NO_ERROR])
    assert f["f1"] == 1.0, f
    #    hand-computed intermediate: acc_err = 1/2, acc_cor = 1 -> F1 = 2*.5*1/1.5 = 2/3
    f = processbench_f1([0, 9, NO_ERROR, NO_ERROR], [0, 1, NO_ERROR, NO_ERROR])
    assert np.isclose(f["f1"], 2 / 3), f

    # 5. The threshold calibrates off the ANNOTATED bad steps, and a larger alpha is more
    #    permissive (fires more readily), matching the sequence-level direction.
    scores = [np.array([0.0, 0.0, float(i)]) for i in range(1, 11)]
    labs10 = [2] * 10
    t_lo = calibrate_step_threshold(scores, labs10, 0.1)
    t_hi = calibrate_step_threshold(scores, labs10, 0.9)
    assert t_lo < t_hi, (t_lo, t_hi)

    # 6. End-to-end on separable synthetic data: the erroneous step always has the top score,
    #    clean rows have none, so both SLA and F1 should be high.
    rng = np.random.default_rng(0)
    ss, ll = [], []
    for _ in range(200):
        k = rng.integers(0, 5)
        row = rng.uniform(0.0, 0.1, 6)
        if rng.random() < 0.5:
            row[k] = rng.uniform(1.0, 2.0)
            ll.append(int(k))
        else:
            ll.append(NO_ERROR)
        ss.append(row)
    res = evaluate(ss, ll, alpha=0.1, n_splits=40, seed=1)
    assert res["sla"] > 0.85, res
    assert res["f1"] > 0.85, res
    assert res["n_splits_used"] == 40, res

    # 7. A detector with NO signal (uniform noise) must score far worse on the same harness.
    ss_rand = [rng.uniform(0, 1, 6) for _ in ll]
    res_rand = evaluate(ss_rand, ll, alpha=0.1, n_splits=40, seed=1)
    assert res_rand["f1"] < res["f1"] - 0.2, (res_rand, res)

    # 8. An unmapped step yields NaN, not a confident zero.
    s = step_drop_scores(np.zeros(40), [(0, 20), None, (20, 40)], ema_span=5)
    assert np.isnan(s[1]) and np.isfinite(s[0]) and np.isfinite(s[2]), s

    print("localization_metrics.smoke: PASS (8 checks)")


if __name__ == "__main__":
    smoke()
