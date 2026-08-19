"""
The single frozen evaluator library for the paper-exact cycle.

Handoff §P0.5: "Freeze a single evaluator library for exact answer parsing, ProcessBench
F1, SLA, AUROC/AUPRC, pass@1, token count, and grouped bootstrap." One library, one
revision string, recorded in every manifest — so that when two rows disagree it is because
the methods disagree, not because two scorers rounded differently.

`EVALUATOR_REVISION` must be bumped on any change that can move a number, and any run
manifest carrying an older revision must be re-scored rather than compared across the
boundary.

Grouping
--------
Every interval in this module is a **question-level** bootstrap. K traces of one question
are not K independent examples (phase-1 checkpoint §7.10); trace-level intervals on a
K-sampled pool are roughly sqrt(K) too narrow and would manufacture significance.

AUROC95
-------
`auroc95_target` is `0.5 + 0.95 * (AUROC_full - 0.5)`, i.e. 95% of the *above-chance*
discrimination, not `0.95 * AUROC_full` (handoff §P1). At AUROC_full = 0.61 the two differ
by 0.03 AUROC — enough to move the reported "earliest budget reaching 95% of signal" by
several hundred tokens.
"""
import math
import re

import numpy as np

EVALUATOR_REVISION = "paper_exact_evaluator_v1.0.0"

NO_ERROR = -1  # ProcessBench's "trace is clean" label


# ── answer parsing ───────────────────────────────────────────────────────────────
# Delegates to the project's existing balanced-brace boxed extractor (the one that fixed
# the Phase-13 grading bug) and adds an explicit parse status, because parser coverage is
# a promotion gate and a silent fallback would make coverage look perfect.

def extract_boxed(text: str):
    from spectral_utils.data_loaders import _extract_boxed
    return _extract_boxed(text or "")


def parse_math_answer(text: str) -> dict:
    """Parse a final math answer, reporting *how* it was obtained.

    Returns {'answer': str|None, 'status': 'boxed'|'fallback_number'|'none'}.
    A `fallback_number` parse is a real answer for grading but does not count towards
    parser coverage: the papers we reproduce all instruct the model to box its answer, so
    an unboxed trace is a protocol failure that must stay visible in the gate.
    """
    from spectral_utils.data_loaders import _normalize_math_answer
    boxed = extract_boxed(text)
    if boxed is not None:
        val = _normalize_math_answer(boxed)
        if val:
            return {"answer": val, "status": "boxed"}
    nums = re.findall(r"-?\d+(?:\.\d+)?", (text or "").replace(",", ""))
    if nums:
        return {"answer": nums[-1], "status": "fallback_number"}
    return {"answer": None, "status": "none"}


def math_equal(pred: str, gold: str) -> bool:
    """Numeric-then-string equality on two already-normalised answers."""
    if not pred or not gold:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):
        return pred.strip() == gold.strip()


def grade_math(gen_text: str, gold_answer: str) -> dict:
    """Grade one generation against a gold answer string."""
    from spectral_utils.data_loaders import _normalize_math_answer
    p = parse_math_answer(gen_text)
    g = _normalize_math_answer(str(gold_answer))
    return {
        "correct": bool(math_equal(p["answer"], g)),
        "pred_answer": p["answer"],
        "gold_answer": g,
        "parse_status": p["status"],
    }


def parser_coverage(parse_statuses) -> float:
    """Fraction of traces whose answer was recovered from an actual \\boxed{} block."""
    st = list(parse_statuses)
    return float(np.mean([s == "boxed" for s in st])) if st else 0.0


# ── ProcessBench ────────────────────────────────────────────────────────────────

def processbench_f1(predictions, labels) -> dict:
    """Official ProcessBench metric (Zheng et al., arXiv:2412.06559).

    `predictions` / `labels` are earliest-erroneous-step indices, or `NO_ERROR` (-1) for a
    clean trace. The reported F1 is the **harmonic mean of two accuracies** — accuracy on
    erroneous traces and accuracy on correct traces — not the usual precision/recall F1.
    Getting that wrong inflates every localization row, so it is computed explicitly here
    rather than borrowed from sklearn.
    """
    preds = [int(p) if p is not None else None for p in predictions]
    labs = [int(l) for l in labels]
    if len(preds) != len(labs):
        raise ValueError(f"length mismatch: {len(preds)} predictions vs {len(labs)} labels")

    err_i = [i for i, l in enumerate(labs) if l != NO_ERROR]
    cor_i = [i for i, l in enumerate(labs) if l == NO_ERROR]
    # An unparseable prediction counts as wrong, never as abstention — dropping it would
    # let a method raise its score by refusing on the rows it finds hard.
    err_hits = sum(1 for i in err_i if preds[i] == labs[i])
    cor_hits = sum(1 for i in cor_i if preds[i] == NO_ERROR)

    err_acc = err_hits / len(err_i) if err_i else float("nan")
    cor_acc = cor_hits / len(cor_i) if cor_i else float("nan")
    if err_i and cor_i and (err_acc + cor_acc) > 0:
        f1 = 2 * err_acc * cor_acc / (err_acc + cor_acc)
    else:
        f1 = float("nan")
    return {
        "error_acc": err_acc,
        "correct_acc": cor_acc,
        "f1": f1,
        "n_error": len(err_i),
        "n_correct": len(cor_i),
        "n_total": len(labs),
        "n_unparsed": sum(1 for p in preds if p is None),
    }


def macro_f1(per_subset: dict) -> float:
    """Unweighted mean of the four ProcessBench subset F1 values."""
    vals = [v["f1"] for v in per_subset.values() if not math.isnan(v.get("f1", float("nan")))]
    return float(np.mean(vals)) if vals else float("nan")


def mind_the_gap_sla(predictions, labels, tolerance: int = 0) -> dict:
    """Mind-the-Gap's native Step-Level Accuracy.

    SLA is computed over **erroneous traces only** — clean traces have no first error to
    localize. That is why the handoff insists SLA lives in its own subtable: an SLA number
    and a ProcessBench F1 number are not on the same population and averaging them, or
    putting them in adjacent columns of one ranked table, is a category error.
    """
    err = [(p, l) for p, l in zip(predictions, labels) if int(l) != NO_ERROR]
    if not err:
        return {"sla": float("nan"), "n": 0, "tolerance": tolerance}
    hits = sum(1 for p, l in err
               if p is not None and abs(int(p) - int(l)) <= tolerance)
    return {"sla": hits / len(err), "n": len(err), "tolerance": tolerance}


# ── detection metrics ───────────────────────────────────────────────────────────

def auroc(y, scores) -> float:
    """AUROC by the rank formula (ties handled by average ranks). NaN rows dropped."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0 or np.allclose(s, s[0]):
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y, scores) -> float:
    """Average precision (step-interpolated, the sklearn `average_precision_score` rule)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0 or len(y) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def normalized_ap(y, scores) -> float:
    """Average precision rescaled by prevalence, so budgets with different class balance
    are comparable: (AP - prevalence) / (1 - prevalence)."""
    y = np.asarray(y, dtype=float)
    prev = float(np.mean(y)) if len(y) else float("nan")
    ap = auprc(y, scores)
    if not np.isfinite(ap) or not np.isfinite(prev) or prev >= 1.0:
        return float("nan")
    return float((ap - prev) / (1.0 - prev))


def auroc95_target(auroc_full: float) -> float:
    """95% of the ABOVE-CHANCE discrimination: 0.5 + 0.95*(AUROC_full - 0.5).

    Not 0.95 * AUROC_full — see the module docstring and handoff §P1.
    """
    if not np.isfinite(auroc_full):
        return float("nan")
    return 0.5 + 0.95 * (auroc_full - 0.5)


def recovered_signal(auroc_t: float, auroc_full: float) -> float:
    """(AUROC_t - 0.5) / (AUROC_full - 0.5) — the fraction of eventual above-chance
    discrimination already available at budget t."""
    denom = auroc_full - 0.5
    if not np.isfinite(auroc_t) or not np.isfinite(auroc_full) or abs(denom) < 1e-9:
        return float("nan")
    return float((auroc_t - 0.5) / denom)


def earliest_budget_reaching(budgets, aurocs, auroc_full: float, frac: float = 0.95):
    """Smallest budget whose AUROC reaches `frac` of the above-chance full-trace signal."""
    target = 0.5 + frac * (auroc_full - 0.5)
    for b, a in sorted(zip(budgets, aurocs)):
        if np.isfinite(a) and a >= target:
            return b
    return None


# ── stopping-lane metrics ───────────────────────────────────────────────────────

def pass_at_1(correct_flags) -> float:
    c = [bool(x) for x in correct_flags]
    return float(np.mean(c)) if c else float("nan")


def token_accounting(records) -> dict:
    """Total generated tokens for a policy.

    Counts reasoning + forced-closure tokens. Handoff §5: "A policy saves tokens only
    after the closure is actually generated and graded; truncation-only estimates are not
    realized savings." So `closure_tokens` is summed from real generations; a record that
    stopped early but never generated its closure is counted as incomplete, not as free.
    """
    reasoning = sum(int(r.get("n_reasoning_tokens", 0)) for r in records)
    closure = sum(int(r.get("n_closure_tokens", 0)) for r in records)
    missing = sum(1 for r in records
                  if r.get("stopped_early") and not r.get("closure_generated"))
    return {
        "reasoning_tokens": reasoning,
        "closure_tokens": closure,
        "total_tokens": reasoning + closure,
        "n_traces": len(records),
        "mean_tokens_per_trace": (reasoning + closure) / len(records) if records else float("nan"),
        "n_stopped_without_closure": missing,
        "realized_savings_valid": missing == 0,
    }


# ── grouped bootstrap ───────────────────────────────────────────────────────────

def grouped_bootstrap(groups, fn, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Bootstrap a statistic by resampling **question groups** with replacement.

    `groups` maps question_id -> payload; `fn(list_of_payloads) -> float`.
    Returns {'point', 'lo', 'hi', 'n_groups', 'n_valid'}.
    """
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    point = fn([groups[k] for k in keys])
    stats = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(keys), size=len(keys))
        val = fn([groups[keys[i]] for i in pick])
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_groups": len(keys), "n_valid": 0}
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "n_groups": len(keys), "n_valid": len(stats)}


def paired_grouped_bootstrap(groups_a, groups_b, fn, n_boot: int = 2000,
                             seed: int = 42, alpha: float = 0.05):
    """Paired delta (A - B) bootstrapped over the shared question groups.

    Paired on question id, so the same resampled questions feed both methods on every
    replicate. Comparing two independent bootstraps instead would ignore the (large)
    positive correlation between methods on the same questions and widen the interval to
    the point of never resolving anything.
    """
    keys = sorted(set(groups_a) & set(groups_b))
    if not keys:
        raise ValueError("no shared question ids between the two methods")
    rng = np.random.default_rng(seed)
    point = fn([groups_a[k] for k in keys]) - fn([groups_b[k] for k in keys])
    stats = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(keys), size=len(keys))
        sel = [keys[i] for i in pick]
        d = fn([groups_a[k] for k in sel]) - fn([groups_b[k] for k in sel])
        if np.isfinite(d):
            stats.append(d)
    lo, hi = (np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if stats else (float("nan"), float("nan")))
    return {"delta": float(point), "lo": float(lo), "hi": float(hi),
            "n_groups": len(keys), "n_valid": len(stats),
            "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))}


# ── repeated-monitor alarm calibration ──────────────────────────────────────────

def calibrate_ever_alarm_threshold(score_paths, target_fpr: float = 0.05) -> dict:
    """Freeze an alarm threshold at a target **trace-level ever-alarm** FPR.

    Handoff §P1 / phase-1 checkpoint §7.8: a monitor inspected at every budget is a
    repeated test, so a fixed-time 5% threshold does not give a 5% probability of ever
    alarming. The threshold is therefore calibrated on `max_t score(t)` over the whole
    registered horizon of the *negative* (correct) traces.

    `score_paths` is a list of per-trace score sequences from correct traces only.
    """
    maxes = np.array([np.nanmax(p) for p in score_paths if len(p)], dtype=float)
    maxes = maxes[np.isfinite(maxes)]
    if len(maxes) == 0:
        return {"threshold": float("nan"), "n_calib": 0, "target_fpr": target_fpr}
    thr = float(np.quantile(maxes, 1.0 - target_fpr))
    realized = float(np.mean(maxes >= thr))
    return {
        "threshold": thr,
        "n_calib": int(len(maxes)),
        "target_fpr": target_fpr,
        "realized_fpr": realized,
        # With ~60 MATH calibration problems a 5% target is coarse; report the count so a
        # reader can see the granularity rather than trusting the nominal rate.
        "fpr_granularity": 1.0 / len(maxes),
    }


def summary_dict() -> dict:
    """Evaluator identity for the run manifest."""
    return {"evaluator_revision": EVALUATOR_REVISION}
