"""Frozen CPU evaluator for Fair Paper-Exact Comparisons v1.

This module derives its ProcessBench and token-accounting definitions from
``paper_exact_evaluator_v1.0.0`` and tightens the interfaces needed for direct,
row-aligned comparison tables:

* label 1 always means error/risk;
* calibration is performed only on registered training folds;
* prefix false-positive control uses the maximum over the complete six-budget path;
* prefix rows are eligible iff ``final_length > budget``;
* separately refit continuous scores are evaluated foldwise, never pooled; and
* uncertainty resamples source questions within family while carrying every method,
  budget, arm, and scorer copy in the group payload.

Missing scores are errors in this evaluator.  A headline join must reach 100% before it
gets here; silently dropping NaNs would change the population and defeat the comparison
contract.  Unparsed *discrete* ProcessBench predictions remain rows and count wrong.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from spectral_utils.paper_exact import evaluator as paper_evaluator

from .folds import canonical_sha256


EVALUATOR_REVISION = "fair_paper_exact_evaluator_v1.0.0"
DERIVED_FROM_EVALUATOR = paper_evaluator.EVALUATOR_REVISION
POSITIVE_CLASS = "error/risk"
NO_ERROR = -1
CANONICAL_PREFIX_BUDGETS = (16, 32, 64, 128, 256, 512)
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260818


def _binary_labels(values: Sequence[Any]) -> np.ndarray:
    labels = np.asarray(list(values))
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.size == 0:
        return labels.astype(np.int8)
    try:
        numeric = labels.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be binary with 1=error/risk") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(np.isin(numeric, (0.0, 1.0))):
        raise ValueError("labels must be binary with 1=error/risk")
    return numeric.astype(np.int8)


def _finite_scores(values: Sequence[Any], *, name: str = "scores") -> np.ndarray:
    try:
        scores = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(scores)):
        bad = np.flatnonzero(~np.isfinite(scores))[:5].tolist()
        raise ValueError(f"{name} contain non-finite values at positions {bad}")
    return scores


def _labels_and_scores(labels: Sequence[Any], scores: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    y = _binary_labels(labels)
    s = _finite_scores(scores)
    if len(y) != len(s):
        raise ValueError(f"length mismatch: {len(y)} labels versus {len(s)} scores")
    return y, s


# ── error-positive detection metrics ──────────────────────────────────────────

def auroc(labels: Sequence[Any], risk_scores: Sequence[Any]) -> float:
    """Error-positive AUROC with exact average ranks for tied scores.

    All-equal scores correctly return 0.5 when both classes are present.  This differs
    from the older acquisition helper, which returned NaN for a constant score vector.
    """

    y, scores = _labels_and_scores(labels, risk_scores)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        # Ranks are one-indexed; every tied member receives the average rank.
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    rank_sum = float(np.sum(ranks[y == 1]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: Sequence[Any], risk_scores: Sequence[Any]) -> float:
    """Error AUPRC / average precision, grouping tied thresholds exactly.

    AP is a threshold metric, so examples with equal scores enter together.  Processing
    tied examples one by one makes AP depend on row order and can manufacture a gain.
    """

    y, scores = _labels_and_scores(labels, risk_scores)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    ordered_y = y[order]
    ordered_scores = scores[order]
    true_positives = 0
    seen = 0
    previous_recall = 0.0
    ap = 0.0
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        true_positives += int(np.sum(ordered_y[start:stop] == 1))
        seen += stop - start
        recall = true_positives / n_pos
        precision = true_positives / seen
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(ap)


def prevalence_normalized_ap(labels: Sequence[Any], risk_scores: Sequence[Any]) -> float:
    """Return ``(error_AP - prevalence) / (1 - prevalence)``."""

    y, scores = _labels_and_scores(labels, risk_scores)
    if len(y) == 0:
        return float("nan")
    prevalence = float(np.mean(y))
    ap = average_precision(y, scores)
    if not np.isfinite(ap) or prevalence >= 1.0:
        return float("nan")
    return float((ap - prevalence) / (1.0 - prevalence))


def detection_metrics(labels: Sequence[Any], risk_scores: Sequence[Any]) -> dict[str, Any]:
    """Core error-positive metrics on one homogeneous, identically joined population."""

    y, scores = _labels_and_scores(labels, risk_scores)
    return {
        "positive_class": POSITIVE_CLASS,
        "n": int(len(y)),
        "n_error": int(np.sum(y == 1)),
        "n_correct": int(np.sum(y == 0)),
        "error_prevalence": float(np.mean(y)) if len(y) else float("nan"),
        "auroc": auroc(y, scores),
        "error_auprc": average_precision(y, scores),
        "prevalence_normalized_ap": prevalence_normalized_ap(y, scores),
    }


def recovered_above_chance_signal(auroc_at_budget: float, final_auroc: float) -> float:
    """Fraction of final above-chance AUROC recovered at a causal budget."""

    return paper_evaluator.recovered_signal(auroc_at_budget, final_auroc)


def earliest_budget_reaching_signal(
    budgets: Sequence[int],
    aurocs: Sequence[float],
    final_auroc: float,
    *,
    fraction: float = 0.95,
) -> int | None:
    """Earliest budget reaching a fraction of final *above-chance* AUROC."""

    return paper_evaluator.earliest_budget_reaching(
        budgets, aurocs, final_auroc, frac=fraction
    )


# ── fixed-FPR calibration ─────────────────────────────────────────────────────

def calibrate_correct_only_threshold(
    labels: Sequence[Any],
    risk_scores: Sequence[Any],
    *,
    target_fpr: float,
) -> dict[str, Any]:
    """Choose the least-stringent threshold with empirical correct-row FPR <= target.

    Predictions use ``score >= threshold``.  Ties are never split.  If the highest tied
    score block is already larger than the false-positive allowance, the threshold is
    moved just above it and the realized FPR is zero.  This conservative rule makes the
    nominal 5%/10% claim true even on small calibration populations.
    """

    if not 0.0 <= float(target_fpr) < 1.0:
        raise ValueError("target_fpr must satisfy 0 <= target_fpr < 1")
    y, scores = _labels_and_scores(labels, risk_scores)
    correct_scores = scores[y == 0]
    if len(correct_scores) == 0:
        raise ValueError("fixed-FPR calibration requires at least one correct row")

    allowed = float(target_fpr) * len(correct_scores)
    unique = np.unique(correct_scores)
    candidates: list[float] = []
    for value in unique:
        candidates.append(float(value))
        candidates.append(float(np.nextafter(value, np.inf)))
    # ``nextafter(max,+inf)`` always permits zero warnings for finite scores.
    feasible = [
        threshold
        for threshold in candidates
        if int(np.sum(correct_scores >= threshold)) <= allowed + 1e-12
    ]
    if not feasible:  # defensive; finite input makes this unreachable
        raise RuntimeError("could not construct a feasible fixed-FPR threshold")
    threshold = min(feasible)
    false_positives = int(np.sum(correct_scores >= threshold))
    return {
        "threshold": float(threshold),
        "target_fpr": float(target_fpr),
        "observed_calibration_fpr": false_positives / len(correct_scores),
        "n_correct_calibration": int(len(correct_scores)),
        "n_false_positive_calibration": false_positives,
        "fpr_granularity": 1.0 / len(correct_scores),
        "tie_policy": "score_gte_threshold_no_tie_splitting",
        "calibration_population": "correct_rows_only",
    }


def calibrate_global_thresholds(
    labels: Sequence[Any],
    risk_scores: Sequence[Any],
    *,
    targets: Sequence[float] = (0.05, 0.10),
) -> dict[str, dict[str, Any]]:
    """Calibrate Global operating points using correct calibration rows only."""

    return {
        f"fpr_{int(round(100 * float(target))):02d}": calibrate_correct_only_threshold(
            labels, risk_scores, target_fpr=float(target)
        )
        for target in targets
    }


def operating_point(
    labels: Sequence[Any],
    risk_scores: Sequence[Any],
    threshold: float,
) -> dict[str, Any]:
    """Observed error TPR/precision and correct-row FPR at a frozen threshold."""

    y, scores = _labels_and_scores(labels, risk_scores)
    predicted = scores >= float(threshold)
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "error_tpr": tp / (tp + fn) if (tp + fn) else float("nan"),
        "error_precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "observed_fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
    }


# ── strict causal-budget and repeated-warning evaluation ──────────────────────

def eligible_at_budget(final_length: int, budget: int) -> bool:
    """A causal prefix exists only when the trace is unfinished: length > budget."""

    if int(final_length) != final_length or int(budget) != budget:
        raise ValueError("final_length and budget must be integers")
    if int(final_length) < 0 or int(budget) < 0:
        raise ValueError("final_length and budget must be non-negative")
    return int(final_length) > int(budget)


def unfinished_rows(
    rows: Iterable[Mapping[str, Any]],
    budget: int,
    *,
    length_key: str = "final_length",
) -> list[Mapping[str, Any]]:
    """Filter a population using the strict ``final_length > budget`` rule."""

    return [row for row in rows if eligible_at_budget(row[length_key], budget)]


def _complete_prefix_path(
    path: Sequence[Any] | Mapping[Any, Any],
    budgets: Sequence[int],
) -> np.ndarray:
    expected = tuple(int(budget) for budget in budgets)
    if len(expected) != len(set(expected)) or tuple(sorted(expected)) != expected:
        raise ValueError("budgets must be unique and strictly increasing")
    if isinstance(path, Mapping):
        normalized: dict[int, Any] = {}
        for key, value in path.items():
            try:
                budget = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid prefix budget key {key!r}") from exc
            if budget in normalized:
                raise ValueError(f"duplicate normalized prefix budget {budget}")
            normalized[budget] = value
        if set(normalized) != set(expected):
            missing = sorted(set(expected) - set(normalized))
            extra = sorted(set(normalized) - set(expected))
            raise ValueError(f"incomplete prefix horizon: missing={missing}, extra={extra}")
        values = [normalized[budget] for budget in expected]
    else:
        values = list(path)
        if len(values) != len(expected):
            raise ValueError(
                f"prefix path has {len(values)} values; complete horizon requires {len(expected)}"
            )
    return _finite_scores(values, name="prefix path")


def calibrate_prefix_ever_warning_thresholds(
    labels: Sequence[Any],
    score_paths: Sequence[Sequence[Any] | Mapping[Any, Any]],
    *,
    budgets: Sequence[int] = CANONICAL_PREFIX_BUDGETS,
    targets: Sequence[float] = (0.05, 0.10),
) -> dict[str, dict[str, Any]]:
    """Calibrate trace-level ever-warning FPR on full six-budget correct paths."""

    y = _binary_labels(labels)
    paths = list(score_paths)
    if len(y) != len(paths):
        raise ValueError(f"length mismatch: {len(y)} labels versus {len(paths)} paths")
    correct_maxima: list[float] = []
    for label, path in zip(y, paths):
        if label == 0:
            complete = _complete_prefix_path(path, budgets)
            correct_maxima.append(float(np.max(complete)))
    if not correct_maxima:
        raise ValueError("prefix calibration requires at least one complete correct trace")
    maxima_labels = [0] * len(correct_maxima)
    result = calibrate_global_thresholds(maxima_labels, correct_maxima, targets=targets)
    for ledger in result.values():
        ledger["calibration_population"] = "correct_trace_six_budget_maximum"
        ledger["budgets"] = [int(budget) for budget in budgets]
    return result


def prefix_warning_metrics(
    labels: Sequence[Any],
    score_paths: Sequence[Sequence[Any] | Mapping[Any, Any]],
    *,
    threshold: float,
    budgets: Sequence[int] = CANONICAL_PREFIX_BUDGETS,
) -> dict[str, Any]:
    """Evaluate repeated warnings on the same complete registered budget horizon."""

    y = _binary_labels(labels)
    paths = list(score_paths)
    if len(y) != len(paths):
        raise ValueError(f"length mismatch: {len(y)} labels versus {len(paths)} paths")
    first_warnings: list[int | None] = []
    ever: list[bool] = []
    canonical_budgets = [int(budget) for budget in budgets]
    for path in paths:
        scores = _complete_prefix_path(path, canonical_budgets)
        indices = np.flatnonzero(scores >= float(threshold))
        warning = int(canonical_budgets[int(indices[0])]) if len(indices) else None
        first_warnings.append(warning)
        ever.append(warning is not None)
    ever_arr = np.asarray(ever, dtype=bool)
    wrong = y == 1
    correct = y == 0
    warned_budgets = [value for value in first_warnings if value is not None]
    return {
        "threshold": float(threshold),
        "wrong_trace_warning_coverage": float(np.mean(ever_arr[wrong])) if np.any(wrong) else float("nan"),
        "correct_trace_ever_warning_fpr": float(np.mean(ever_arr[correct])) if np.any(correct) else float("nan"),
        "median_first_warning_budget": float(np.median(warned_budgets)) if warned_budgets else float("nan"),
        "first_warning_budgets": first_warnings,
        "budgets": canonical_budgets,
        "n": int(len(y)),
    }


# ── ProcessBench first-error localization ─────────────────────────────────────

def processbench_f1(predictions: Sequence[Any], labels: Sequence[Any]) -> dict[str, Any]:
    """Official harmonic F1; ``None`` predictions remain rows and count wrong."""

    return paper_evaluator.processbench_f1(predictions, labels)


def mind_the_gap_sla(
    predictions: Sequence[Any], labels: Sequence[Any], *, tolerance: int = 0
) -> dict[str, Any]:
    """Erroneous-traces-only SLA, kept separate from ProcessBench F1."""

    return paper_evaluator.mind_the_gap_sla(predictions, labels, tolerance=tolerance)


def parser_coverage(parse_statuses: Sequence[str]) -> float:
    """Coverage of actual boxed parses; fallback-number parses do not count."""

    return paper_evaluator.parser_coverage(parse_statuses)


def token_accounting(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count realized reasoning plus generated closure tokens.

    A stopped trace without an actual generated closure invalidates a realized-savings
    claim.  This is the frozen acquisition evaluator's definition, surfaced here so all
    stopping tables use the same accounting boundary.
    """

    return paper_evaluator.token_accounting(records)


def _step_pairs(
    row: Mapping[str, Any],
    *,
    scores_key: str,
    indices_key: str,
) -> list[tuple[int, float]]:
    raw_scores = row[scores_key]
    if isinstance(raw_scores, Mapping):
        pairs = [(int(index), float(score)) for index, score in raw_scores.items()]
    else:
        scores = list(raw_scores)
        if indices_key in row and row[indices_key] is not None:
            indices = [int(index) for index in row[indices_key]]
        else:
            indices = list(range(len(scores)))
        if len(indices) != len(scores):
            raise ValueError("localization step_indices and step_scores length mismatch")
        pairs = [(index, float(score)) for index, score in zip(indices, scores)]
    if len({index for index, _ in pairs}) != len(pairs):
        raise ValueError("duplicate localization step index")
    if any(index < 0 for index, _ in pairs):
        raise ValueError("localization step indices must be non-negative")
    if any(not np.isfinite(score) for _, score in pairs):
        raise ValueError("localization step scores must be finite")
    return sorted(pairs)


def localization_prediction(
    row: Mapping[str, Any],
    threshold: float,
    *,
    scores_key: str = "step_scores",
    indices_key: str = "step_indices",
) -> int:
    """Earliest registered step whose error-risk score reaches the threshold."""

    hits = [
        index
        for index, score in _step_pairs(row, scores_key=scores_key, indices_key=indices_key)
        if score >= float(threshold)
    ]
    return min(hits) if hits else NO_ERROR


def localization_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Any],
    *,
    subset_key: str = "subset",
    label_key: str = "first_error",
    expected_subsets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Official per-subset F1, equal-subset macro-F1, and within-one error accuracy."""

    rows = list(rows)
    predictions = list(predictions)
    if len(rows) != len(predictions):
        raise ValueError("localization rows and predictions length mismatch")
    observed = {str(row[subset_key]) for row in rows}
    subsets = [str(value) for value in expected_subsets] if expected_subsets else sorted(observed)
    if expected_subsets is not None and observed != set(subsets):
        raise ValueError(
            f"localization subset mismatch: expected={sorted(set(subsets))}, observed={sorted(observed)}"
        )

    per_subset: dict[str, dict[str, Any]] = {}
    for subset in subsets:
        indices = [i for i, row in enumerate(rows) if str(row[subset_key]) == subset]
        if not indices:
            raise ValueError(f"localization subset {subset!r} has no rows")
        per_subset[subset] = processbench_f1(
            [predictions[i] for i in indices],
            [rows[i][label_key] for i in indices],
        )
    f1_values = [item["f1"] for item in per_subset.values()]
    error_values = [item["error_acc"] for item in per_subset.values()]
    clean_values = [item["correct_acc"] for item in per_subset.values()]
    macro_f1 = (
        float(np.mean(f1_values)) if f1_values and np.all(np.isfinite(f1_values)) else float("nan")
    )
    macro_clean = (
        float(np.mean(clean_values))
        if clean_values and np.all(np.isfinite(clean_values))
        else float("nan")
    )
    macro_error = (
        float(np.mean(error_values))
        if error_values and np.all(np.isfinite(error_values))
        else float("nan")
    )

    erroneous = [i for i, row in enumerate(rows) if int(row[label_key]) != NO_ERROR]
    within_one_hits = sum(
        1
        for i in erroneous
        if predictions[i] is not None
        and abs(int(predictions[i]) - int(rows[i][label_key])) <= 1
    )
    return {
        "per_subset": per_subset,
        "equal_subset_macro_f1": macro_f1,
        "equal_subset_error_accuracy": macro_error,
        "equal_subset_clean_accuracy": macro_clean,
        "within_one_error_accuracy": (
            within_one_hits / len(erroneous) if erroneous else float("nan")
        ),
        "n": len(rows),
        "n_error": len(erroneous),
        "n_unparsed": sum(prediction is None for prediction in predictions),
    }


def fit_localization_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    subset_key: str = "subset",
    label_key: str = "first_error",
    scores_key: str = "step_scores",
    indices_key: str = "step_indices",
    expected_subsets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit the registered threshold and deterministic tie-breaks on calibration rows.

    The primary objective is equal-subset ProcessBench macro-F1.  Exact ties prefer
    higher equal-subset clean accuracy, then the numerically higher threshold.
    """

    rows = list(rows)
    if not rows:
        raise ValueError("cannot fit localization threshold on zero rows")
    pairs_by_row = [
        _step_pairs(row, scores_key=scores_key, indices_key=indices_key)
        for row in rows
    ]
    scores = [score for pairs in pairs_by_row for _, score in pairs]
    if not scores:
        raise ValueError("cannot fit localization threshold without step scores")

    # Every v1 fair-comparison score method has one frozen detector score and one
    # frozen locator per trace.  In that common case a threshold only toggles the
    # prediction between NO_ERROR and the fixed locator, so the exact objective can
    # be swept in O(n log n), blockwise over tied scores.  The generic multi-step
    # implementation remains below for methods whose predicted step itself changes
    # as the threshold moves.
    if all(len(pairs) == 1 for pairs in pairs_by_row):
        subsets = (
            [str(value) for value in expected_subsets]
            if expected_subsets is not None
            else sorted({str(row[subset_key]) for row in rows})
        )
        observed = {str(row[subset_key]) for row in rows}
        if observed != set(subsets):
            raise ValueError(
                f"localization subset mismatch: expected={sorted(set(subsets))}, "
                f"observed={sorted(observed)}"
            )
        subset_index = {subset: index for index, subset in enumerate(subsets)}
        n_error = np.zeros(len(subsets), dtype=np.int64)
        n_clean = np.zeros(len(subsets), dtype=np.int64)
        error_hits = np.zeros(len(subsets), dtype=np.int64)
        clean_hits = np.zeros(len(subsets), dtype=np.int64)
        packed = []
        for row, pairs in zip(rows, pairs_by_row):
            subset = subset_index[str(row[subset_key])]
            label = int(row[label_key])
            locator, score = pairs[0]
            if label == NO_ERROR:
                n_clean[subset] += 1
                clean_hits[subset] += 1  # all-inactive threshold predicts NO_ERROR
            else:
                n_error[subset] += 1
            packed.append((float(score), subset, label, int(locator)))
        if np.any(n_error == 0) or np.any(n_clean == 0):
            raise ValueError("localization calibration requires clean and error rows per subset")

        def objective_at(threshold: float) -> tuple[tuple[float, float, float], dict[str, Any]]:
            error_accuracy = error_hits / n_error
            clean_accuracy = clean_hits / n_clean
            denominator = error_accuracy + clean_accuracy
            f1 = np.where(
                denominator > 0.0,
                2.0 * error_accuracy * clean_accuracy / denominator,
                np.nan,
            )
            macro_f1 = float(np.mean(f1))
            macro_clean = float(np.mean(clean_accuracy))
            value = {
                "threshold": float(threshold),
                "equal_subset_macro_f1": macro_f1,
                "equal_subset_clean_accuracy": macro_clean,
                "n_calibration_rows": len(rows),
                "n_threshold_candidates": len(set(scores)) + 1,
                "tie_break": ["macro_f1", "clean_accuracy", "higher_threshold"],
                "threshold_sweep": "single_frozen_locator_blockwise_exact",
            }
            return (macro_f1, macro_clean, float(threshold)), value

        maximum = max(scores)
        best = objective_at(float(np.nextafter(maximum, np.inf)))
        packed.sort(key=lambda item: item[0], reverse=True)
        start = 0
        while start < len(packed):
            stop = start + 1
            while stop < len(packed) and packed[stop][0] == packed[start][0]:
                stop += 1
            for _, subset, label, locator in packed[start:stop]:
                if label == NO_ERROR:
                    if locator != NO_ERROR:
                        clean_hits[subset] -= 1
                elif locator == label:
                    error_hits[subset] += 1
            candidate = objective_at(packed[start][0])
            if candidate[0] > best[0]:
                best = candidate
            start = stop
        if not np.isfinite(best[1]["equal_subset_macro_f1"]):
            raise ValueError("localization calibration has no valid subset F1 objective")
        return best[1]

    maximum = max(scores)
    candidates = sorted(set(scores) | {float(np.nextafter(maximum, np.inf))})

    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    for threshold in candidates:
        predictions = [
            localization_prediction(
                row, threshold, scores_key=scores_key, indices_key=indices_key
            )
            for row in rows
        ]
        metrics = localization_metrics(
            rows,
            predictions,
            subset_key=subset_key,
            label_key=label_key,
            expected_subsets=expected_subsets,
        )
        macro_f1 = metrics["equal_subset_macro_f1"]
        macro_clean = metrics["equal_subset_clean_accuracy"]
        objective = (
            float(macro_f1) if np.isfinite(macro_f1) else -math.inf,
            float(macro_clean) if np.isfinite(macro_clean) else -math.inf,
            float(threshold),
        )
        candidate = {
            "threshold": float(threshold),
            "equal_subset_macro_f1": macro_f1,
            "equal_subset_clean_accuracy": macro_clean,
            "n_calibration_rows": len(rows),
            "n_threshold_candidates": len(candidates),
            "tie_break": ["macro_f1", "clean_accuracy", "higher_threshold"],
        }
        if best is None or objective > best[0]:
            best = (objective, candidate)
    assert best is not None
    if not np.isfinite(best[1]["equal_subset_macro_f1"]):
        raise ValueError("localization calibration has no valid subset F1 objective")
    return best[1]


def crossfit_localization_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_key: str = "fold",
    n_folds: int = 5,
    subset_key: str = "subset",
    label_key: str = "first_error",
    scores_key: str = "step_scores",
    indices_key: str = "step_indices",
    expected_subsets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit on four folds, apply once to the held-out fold, and concatenate decisions.

    Concatenation is valid here because the outputs are discrete first-error decisions.
    Continuously refit scores must use :func:`foldwise_detection_metrics` instead.
    """

    rows = list(rows)
    observed_folds = {int(row[fold_key]) for row in rows}
    required_folds = set(range(int(n_folds)))
    if observed_folds != required_folds:
        raise ValueError(
            f"cross-fitting requires folds {sorted(required_folds)}, got {sorted(observed_folds)}"
        )
    predictions: list[int | None] = [None] * len(rows)
    ledgers: list[dict[str, Any]] = []
    for held_out in range(int(n_folds)):
        train = [row for row in rows if int(row[fold_key]) != held_out]
        test_indices = [i for i, row in enumerate(rows) if int(row[fold_key]) == held_out]
        fit = fit_localization_threshold(
            train,
            subset_key=subset_key,
            label_key=label_key,
            scores_key=scores_key,
            indices_key=indices_key,
            expected_subsets=expected_subsets,
        )
        threshold = fit["threshold"]
        for index in test_indices:
            predictions[index] = localization_prediction(
                rows[index], threshold, scores_key=scores_key, indices_key=indices_key
            )
        ledger = dict(fit)
        ledger.update(
            {
                "held_out_fold": held_out,
                "train_folds": [fold for fold in range(int(n_folds)) if fold != held_out],
                "n_held_out_rows": len(test_indices),
            }
        )
        ledger["calibration_hash"] = canonical_sha256(ledger)
        ledgers.append(ledger)
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("a held-out localization row did not receive a prediction")
    official = localization_metrics(
        rows,
        predictions,
        subset_key=subset_key,
        label_key=label_key,
        expected_subsets=expected_subsets,
    )
    return {
        "predictions": [int(prediction) for prediction in predictions],
        "calibration_ledgers": ledgers,
        "official_oof_metrics": official,
        "aggregation": "concatenated_discrete_out_of_fold_predictions",
    }


# ── foldwise guards for continuously refit scores ─────────────────────────────

def foldwise_detection_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_key: str = "label",
    score_key: str = "score",
    fold_key: str = "fold",
    cell_key: str | None = "cell_id",
) -> dict[str, Any]:
    """Evaluate continuous refit scores within fold, then average folds equally.

    One call may cover only one homogeneous cell.  Heterogeneous cells must be scored
    separately and combined by the lane's registered equal-cell/equal-family rule.
    """

    rows = list(rows)
    if not rows:
        raise ValueError("cannot evaluate zero rows")
    if cell_key is not None:
        cells = {str(row[cell_key]) for row in rows}
        if len(cells) != 1:
            raise ValueError(
                "pooled AUROC across heterogeneous cells is forbidden; evaluate each cell separately"
            )
    folds = sorted({int(row[fold_key]) for row in rows})
    per_fold: dict[str, dict[str, Any]] = {}
    for fold in folds:
        selected = [row for row in rows if int(row[fold_key]) == fold]
        per_fold[str(fold)] = detection_metrics(
            [row[label_key] for row in selected],
            [row[score_key] for row in selected],
        )
    metric_names = ("auroc", "error_auprc", "prevalence_normalized_ap")
    mean_metrics = {
        name: (
            float(np.mean([item[name] for item in per_fold.values()]))
            if np.all(np.isfinite([item[name] for item in per_fold.values()]))
            else float("nan")
        )
        for name in metric_names
    }
    return {
        "per_fold": per_fold,
        "equal_fold_mean": mean_metrics,
        "n_folds": len(folds),
        "n": len(rows),
        "aggregation": "evaluate_within_fold_then_equal_fold_mean",
    }


def pooled_detection_metrics(
    labels: Sequence[Any],
    risk_scores: Sequence[Any],
    *,
    scores_refit_within_fold: bool = False,
    cell_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Pool only globally frozen scores from one homogeneous cell.

    This explicit guard exists so a caller cannot accidentally concatenate continuous
    scores that were independently refit/calibrated in five folds.
    """

    if scores_refit_within_fold:
        raise ValueError(
            "pooled metrics are forbidden for continuously refit fold scores; use foldwise_detection_metrics"
        )
    if cell_ids is not None and len({str(value) for value in cell_ids}) > 1:
        raise ValueError(
            "pooled metrics are forbidden across heterogeneous cells; evaluate each cell separately"
        )
    return detection_metrics(labels, risk_scores)


# ── paired grouped bootstrap ──────────────────────────────────────────────────

def paired_grouped_bootstrap(
    groups: Mapping[str, Any],
    statistic: Callable[[list[Any], Any], float | Mapping[str, float]],
    *,
    strata: Mapping[str, Any],
    recompute: Callable[[list[Any]], Any] | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired source-question bootstrap with optional per-replicate recalibration.

    ``groups[group_id]`` is an opaque payload that must contain every method, budget,
    scorer/model copy, or stopping arm for that source question.  The callback receives
    a *list* of payloads, so repeated bootstrap draws remain repeated rather than being
    collapsed by ID.  ``strata`` maps group IDs to family; each family is resampled
    independently at its observed size.  ``recompute`` is called for the point estimate
    and every replicate, allowing registered decision thresholds to be refit inside the
    bootstrap while leaving frozen score/model parameters untouched.
    """

    if not groups:
        raise ValueError("grouped bootstrap requires at least one source question")
    if int(n_boot) != n_boot or int(n_boot) < 1:
        raise ValueError("n_boot must be a positive integer")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")

    group_ids = sorted((str(group_id) for group_id in groups))
    if len(set(group_ids)) != len(groups):
        raise ValueError("group IDs collide after string normalization")
    normalized_groups = {str(group_id): payload for group_id, payload in groups.items()}
    normalized_strata = {str(group_id): str(value) for group_id, value in strata.items()}
    if set(normalized_strata) != set(group_ids):
        missing = sorted(set(group_ids) - set(normalized_strata))
        extra = sorted(set(normalized_strata) - set(group_ids))
        raise ValueError(f"bootstrap strata mismatch: missing={missing}, extra={extra}")

    ids_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group_id in group_ids:
        ids_by_stratum[normalized_strata[group_id]].append(group_id)
    for stratum in ids_by_stratum:
        ids_by_stratum[stratum].sort()

    point_sample = [normalized_groups[group_id] for group_id in group_ids]
    point_fit = recompute(point_sample) if recompute is not None else None
    point_raw = statistic(point_sample, point_fit)
    is_mapping = isinstance(point_raw, Mapping)
    if is_mapping:
        point_values = {str(key): float(value) for key, value in point_raw.items()}
        if not point_values:
            raise ValueError("statistic mapping must not be empty")
    else:
        point_values = {"statistic": float(point_raw)}

    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {key: [] for key in point_values}
    for _ in range(int(n_boot)):
        sampled_ids: list[str] = []
        for stratum in sorted(ids_by_stratum):
            source = ids_by_stratum[stratum]
            picks = rng.integers(0, len(source), size=len(source))
            sampled_ids.extend(source[int(index)] for index in picks)
        sampled_payloads = [normalized_groups[group_id] for group_id in sampled_ids]
        fit = recompute(sampled_payloads) if recompute is not None else None
        raw = statistic(sampled_payloads, fit)
        if isinstance(raw, Mapping) != is_mapping:
            raise ValueError("statistic changed scalar/mapping shape during bootstrap")
        values = {str(key): float(value) for key, value in raw.items()} if is_mapping else {"statistic": float(raw)}
        if set(values) != set(point_values):
            raise ValueError("statistic mapping keys changed during bootstrap")
        for key, value in values.items():
            if np.isfinite(value):
                samples[key].append(value)

    intervals: dict[str, dict[str, Any]] = {}
    for key, point in point_values.items():
        valid = samples[key]
        if valid:
            low, high = np.percentile(
                valid, [100.0 * float(alpha) / 2.0, 100.0 * (1.0 - float(alpha) / 2.0)]
            )
        else:
            low, high = float("nan"), float("nan")
        intervals[key] = {
            "point": point,
            "ci_low": float(low),
            "ci_high": float(high),
            "n_valid": len(valid),
        }

    common = {
        "n_groups": len(group_ids),
        "n_groups_by_stratum": {
            stratum: len(ids) for stratum, ids in sorted(ids_by_stratum.items())
        },
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
        "resampling_unit": "source_question_within_family",
        "paired_payload": "all_methods_budgets_arms_and_copies",
        "recomputed_each_replicate": recompute is not None,
    }
    if is_mapping:
        return {**common, "statistics": intervals}
    scalar = intervals["statistic"]
    return {**common, **scalar}


def summary_dict() -> dict[str, Any]:
    """Machine-readable evaluator identity for method and result registries."""

    return {
        "evaluator_revision": EVALUATOR_REVISION,
        "derived_from": DERIVED_FROM_EVALUATOR,
        "positive_class": POSITIVE_CLASS,
        "processbench_clean_label": NO_ERROR,
        "prefix_budgets": list(CANONICAL_PREFIX_BUDGETS),
        "bootstrap_replicates": DEFAULT_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
    }
