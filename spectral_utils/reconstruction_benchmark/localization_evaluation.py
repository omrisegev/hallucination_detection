"""Post-freeze evaluation contracts for the strict localization lane.

This module is deliberately downstream of the localization A/B score gate.  It
contains no fitting code for response or token risks.  ProcessBench opens labels
only to cross-fit one operating threshold around already frozen per-step scores;
PRMBench keeps the frozen step scores continuous.  Integer first-error decisions
are written to their own table so they cannot be coerced into the repository's
boolean response-prediction schema.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import math
from typing import Any

import numpy as np

from .io import canonical_json_bytes, sha256_bytes
from .localization_contract import NO_ERROR, payload_sha256


EVALUATOR_SCHEMA_VERSION = "reconstruction-localization-evaluator-v2"
PROCESSBENCH_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
PRMBENCH_ERROR_FAMILIES = (
    "circular", "confidence", "counterfactual", "deception",
    "domain_inconsistency", "missing_condition", "multi_solutions",
    "redundency", "step_contradiction",
)
DEFAULT_BOOTSTRAP_DRAWS = 20_000
PROCESSBENCH_BOOTSTRAP_SEED = 2026082403
PRMBENCH_BOOTSTRAP_SEED = 2026082404
UNDEFINED_SINGLE_CLASS = "METRIC_UNDEFINED_SINGLE_CLASS"

LOCALIZATION_DECISION_FIELDS = (
    "task_id", "dataset_id", "population_id", "cell_id", "slice_id",
    "model_id", "system_id", "row_id", "cohort_id", "group_id", "fold",
    "prediction_step", "true_first_error", "status", "access_level",
    "fidelity", "comparison_group_id", "run_hash",
)
METRIC_FIELDS = (
    "task_id", "dataset_id", "population_id", "cell_id", "slice_id",
    "model_id", "system_id", "metric_id", "value", "ci_low", "ci_high",
    "n_examples", "n_positive", "n_negative", "status", "access_level",
    "fidelity", "comparison_group_id", "bootstrap_unit", "bootstrap_draws",
    "cohort_id", "run_hash",
)


def evaluator_contract() -> dict[str, Any]:
    value = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "processbench": {
            "decision": "argmax_step_if_max_score_strictly_above_threshold_else_minus_one",
            "threshold_stage": "post_score_freeze",
            "folds": 5,
            "fold_unit": "source_question",
            "fit_scope": "one_scorer_model_across_four_subsets",
            "objective": "equal_subset_mean_official_processbench_f1",
            "tie_break": "largest_numeric_threshold",
            "metrics": [
                "official_macro_f1", "first_error_exact", "first_error_within_one",
                "clean_abstention_accuracy", "overall_decision_accuracy",
            ],
        },
        "prmbench": {
            "score_type": "continuous_step_risk",
            "positive_class": "annotated_error_step",
            "metrics": ["auroc", "auprc"],
            "single_class_status": UNDEFINED_SINGLE_CLASS,
            "single_class_alternatives": ["mean_risk", "risk_q90", "coverage"],
        },
        "bootstrap": {
            "draws": DEFAULT_BOOTSTRAP_DRAWS,
            "processbench_unit": "source_question",
            "prmbench_unit": "source_idx",
            "paired": True,
        },
        "decision_table": "localization_decisions.csv",
        "historical_075_025_blend_allowed": False,
    }
    value["contract_sha256"] = payload_sha256(value)
    return value


def _integer_step(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer step, not boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer step") from exc
    if result != value or result < NO_ERROR:
        raise ValueError(f"{name} must be -1 or a non-negative integer")
    return result


def _finite_step_scores(row: Mapping[str, Any]) -> np.ndarray:
    scores = np.asarray(row["step_scores"], dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("each ProcessBench row needs a nonempty finite step-score vector")
    return scores


def processbench_prediction(step_scores: Sequence[float], threshold: float) -> int:
    """Return the earliest argmax step iff its maximum is strictly above threshold."""

    scores = np.asarray(step_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("ProcessBench step scores must be nonempty and finite")
    if not np.isfinite(float(threshold)):
        raise ValueError("ProcessBench threshold must be finite")
    return int(np.argmax(scores)) if float(np.max(scores)) > float(threshold) else NO_ERROR


def processbench_trace_metrics(
    labels: Sequence[Any], predictions: Sequence[Any]
) -> dict[str, Any]:
    labels_i = [_integer_step(value, name="true_first_error") for value in labels]
    predictions_i = [
        None if value is None else _integer_step(value, name="prediction_step")
        for value in predictions
    ]
    if len(labels_i) != len(predictions_i) or not labels_i:
        raise ValueError("ProcessBench labels/predictions must be nonempty and aligned")
    errors = [index for index, label in enumerate(labels_i) if label != NO_ERROR]
    clean = [index for index, label in enumerate(labels_i) if label == NO_ERROR]
    exact = (
        sum(predictions_i[index] == labels_i[index] for index in errors) / len(errors)
        if errors else float("nan")
    )
    within_one = (
        sum(
            predictions_i[index] is not None
            and predictions_i[index] != NO_ERROR
            and abs(predictions_i[index] - labels_i[index]) <= 1
            for index in errors
        )
        / len(errors) if errors else float("nan")
    )
    abstention = (
        sum(predictions_i[index] == NO_ERROR for index in clean) / len(clean)
        if clean else float("nan")
    )
    official = (
        2.0 * exact * abstention / (exact + abstention)
        if errors and clean and exact + abstention > 0.0
        else (0.0 if errors and clean else float("nan"))
    )
    return {
        "status": "OK" if errors and clean else UNDEFINED_SINGLE_CLASS,
        "official_macro_f1": float(official),
        "first_error_exact": float(exact),
        "first_error_within_one": float(within_one),
        "clean_abstention_accuracy": float(abstention),
        "overall_decision_accuracy": float(np.mean([
            prediction is not None and prediction == label
            for prediction, label in zip(predictions_i, labels_i)
        ])),
        "n_examples": len(labels_i),
        "n_error": len(errors),
        "n_clean": len(clean),
    }


def processbench_panel_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Any],
    *,
    expected_subsets: Sequence[str] = PROCESSBENCH_SUBSETS,
) -> dict[str, Any]:
    rows = list(rows)
    predictions = list(predictions)
    if len(rows) != len(predictions) or not rows:
        raise ValueError("ProcessBench panel rows/predictions must be nonempty and aligned")
    subsets = tuple(map(str, expected_subsets))
    observed = {str(row["slice_id"]) for row in rows}
    if len(subsets) != len(set(subsets)) or observed != set(subsets):
        raise ValueError("ProcessBench panel does not contain the exact registered subsets")
    per_subset: dict[str, dict[str, Any]] = {}
    for subset in subsets:
        indices = [index for index, row in enumerate(rows) if str(row["slice_id"]) == subset]
        per_subset[subset] = processbench_trace_metrics(
            [rows[index]["first_error"] for index in indices],
            [predictions[index] for index in indices],
        )
    metric_ids = (
        "official_macro_f1", "first_error_exact", "first_error_within_one",
        "clean_abstention_accuracy", "overall_decision_accuracy",
    )
    aggregate = {
        metric_id: (
            float(np.mean([per_subset[subset][metric_id] for subset in subsets]))
            if np.isfinite([per_subset[subset][metric_id] for subset in subsets]).all()
            else float("nan")
        )
        for metric_id in metric_ids
    }
    return {
        "status": "OK" if all(row["status"] == "OK" for row in per_subset.values())
        else UNDEFINED_SINGLE_CLASS,
        "aggregation": "equal_subset_mean",
        "per_subset": per_subset,
        "aggregate": aggregate,
        "n_examples": len(rows),
    }


def assign_processbench_folds(
    rows: Sequence[Mapping[str, Any]], *, n_folds: int = 5
) -> dict[str, int]:
    """Assign deterministic, label-stratified source-group folds.

    The hash input deliberately excludes scorer model and score values.  Reusing a
    namespaced source ``group_id`` therefore gives the same fold in all three scorer
    models without exposing group linkage to the fit capsule.
    """

    if int(n_folds) != n_folds or int(n_folds) < 2:
        raise ValueError("ProcessBench n_folds must be an integer >= 2")
    groups: dict[str, tuple[str, int]] = {}
    for row in rows:
        group_id = str(row["group_id"])
        if not group_id:
            raise ValueError("ProcessBench group_id must be nonempty")
        label = _integer_step(row["first_error"], name="true_first_error")
        stratum = (str(row["slice_id"]), int(label != NO_ERROR))
        previous = groups.setdefault(group_id, stratum)
        if previous != stratum:
            raise ValueError("a ProcessBench source group has conflicting subset/label metadata")
    by_stratum: dict[tuple[str, int], list[str]] = defaultdict(list)
    for group_id, stratum in groups.items():
        by_stratum[stratum].append(group_id)
    assignments: dict[str, int] = {}
    for stratum in sorted(by_stratum):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda group_id: (
                hashlib.sha256(group_id.encode("utf-8")).hexdigest(), group_id
            ),
        )
        for index, group_id in enumerate(ordered):
            assignments[group_id] = index % int(n_folds)
    return assignments


def fit_processbench_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_subsets: Sequence[str] = PROCESSBENCH_SUBSETS,
) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot fit a ProcessBench threshold on zero rows")
    maxima = [float(np.max(_finite_step_scores(row))) for row in rows]
    below_minimum = float(np.nextafter(min(maxima), -np.inf))
    if not np.isfinite(below_minimum):
        raise ValueError("ProcessBench score range cannot form a finite threshold sweep")
    candidates = sorted(set(maxima) | {below_minimum})
    best: tuple[tuple[float, float], dict[str, Any]] | None = None
    for threshold in candidates:
        predictions = [processbench_prediction(row["step_scores"], threshold) for row in rows]
        metrics = processbench_panel_metrics(
            rows, predictions, expected_subsets=expected_subsets
        )
        objective = metrics["aggregate"]["official_macro_f1"]
        candidate = {
            "threshold": float(threshold),
            "objective_equal_subset_official_macro_f1": float(objective),
            "n_calibration_rows": len(rows),
            "n_threshold_candidates": len(candidates),
            "decision_rule": "argmax_if_max_strictly_greater_than_threshold",
            "tie_break": "largest_numeric_threshold",
        }
        order = (float(objective) if np.isfinite(objective) else -math.inf, float(threshold))
        if best is None or order > best[0]:
            best = (order, candidate)
    assert best is not None
    if not np.isfinite(best[1]["objective_equal_subset_official_macro_f1"]):
        raise ValueError("ProcessBench calibration lacks clean/error support in every subset")
    best[1]["calibration_sha256"] = payload_sha256(best[1])
    return best[1]


def crossfit_processbench_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_subsets: Sequence[str] = PROCESSBENCH_SUBSETS,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Cross-fit only a decision threshold around immutable per-step scores."""

    rows = list(rows)
    if not rows:
        raise ValueError("cannot cross-fit ProcessBench on zero rows")
    if len({str(row["row_id"]) for row in rows}) != len(rows):
        raise ValueError("ProcessBench cross-fit rows must have unique row_id values")
    assignments = assign_processbench_folds(rows, n_folds=n_folds)
    observed_folds = set(assignments.values())
    if observed_folds != set(range(int(n_folds))):
        raise ValueError("ProcessBench source groups do not populate all five folds")
    predictions = [NO_ERROR] * len(rows)
    ledgers: list[dict[str, Any]] = []
    for held_out in range(int(n_folds)):
        train = [row for row in rows if assignments[str(row["group_id"])] != held_out]
        test_indices = [
            index for index, row in enumerate(rows)
            if assignments[str(row["group_id"])] == held_out
        ]
        fit = fit_processbench_threshold(train, expected_subsets=expected_subsets)
        for index in test_indices:
            predictions[index] = processbench_prediction(
                rows[index]["step_scores"], fit["threshold"]
            )
        ledger = {
            **fit,
            "held_out_fold": held_out,
            "train_folds": [fold for fold in range(int(n_folds)) if fold != held_out],
            "n_held_out_rows": len(test_indices),
            "held_out_group_sha256": payload_sha256(sorted({
                str(rows[index]["group_id"]) for index in test_indices
            })),
        }
        ledger["fold_ledger_sha256"] = payload_sha256(ledger)
        ledgers.append(ledger)
    metrics = processbench_panel_metrics(
        rows, predictions, expected_subsets=expected_subsets
    )
    decisions = [
        {
            "row_id": str(row["row_id"]),
            "group_id": str(row["group_id"]),
            "slice_id": str(row["slice_id"]),
            "fold": assignments[str(row["group_id"])],
            "prediction_step": int(prediction),
            "true_first_error": _integer_step(row["first_error"], name="true_first_error"),
        }
        for row, prediction in zip(rows, predictions)
    ]
    return {
        "predictions": [int(value) for value in predictions],
        "decisions": decisions,
        "calibration_ledgers": ledgers,
        "metrics": metrics,
        "fold_assignment_sha256": payload_sha256(sorted(assignments.items())),
        "threshold_fit_stage": "post_score_freeze",
        "score_parameters_refit": False,
    }


def _binary_metrics(labels: Sequence[Any], scores: Sequence[Any]) -> tuple[float, float]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not len(y):
        raise ValueError("PRMBench labels/scores must be nonempty aligned vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise ValueError("PRMBench requires finite scores and binary step labels")
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    if not n_positive or not n_negative:
        return float("nan"), float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and s[order[stop]] == s[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    rank_sum = float(np.sum(ranks[y == 1]))
    auroc = (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)

    descending = np.argsort(-s, kind="mergesort")
    y_sorted = y[descending]
    s_sorted = s[descending]
    seen = true_positive = 0
    previous_recall = average_precision = 0.0
    start = 0
    while start < len(s):
        stop = start + 1
        while stop < len(s) and s_sorted[stop] == s_sorted[start]:
            stop += 1
        true_positive += int(np.sum(y_sorted[start:stop]))
        seen += stop - start
        recall = true_positive / n_positive
        precision = true_positive / seen
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(auroc), float(average_precision)


def prmbench_step_metrics(labels: Sequence[Any], scores: Sequence[Any]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not len(y):
        raise ValueError("PRMBench labels/scores must be nonempty aligned vectors")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("PRMBench labels must be binary with 1=annotated error step")
    finite = np.isfinite(s)
    coverage = float(np.mean(finite))
    if not finite.all():
        raise ValueError("PRMBench non-finite scores are failures, not droppable rows")
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    auroc, auprc = _binary_metrics(y, s)
    single_class = n_positive == 0 or n_negative == 0
    return {
        "status": UNDEFINED_SINGLE_CLASS if single_class else "OK",
        "positive_class": "annotated_error_step",
        "auroc": None if single_class else auroc,
        "auprc": None if single_class else auprc,
        "mean_risk": float(np.mean(s)),
        "risk_q90": float(np.quantile(s, 0.90)),
        "coverage": coverage,
        "n_examples": len(y),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def prmbench_panel_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_families: Sequence[str] = PRMBENCH_ERROR_FAMILIES,
) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot evaluate zero PRMBench step rows")
    families = tuple(map(str, expected_families))
    observed = {str(row["error_family"]) for row in rows}
    if len(families) != 9 or observed != set(families):
        raise ValueError("PRMBench evaluation must expose all nine registered families")
    per_family = {
        family: prmbench_step_metrics(
            [row["step_label"] for row in rows if str(row["error_family"]) == family],
            [row["step_score"] for row in rows if str(row["error_family"]) == family],
        )
        for family in families
    }
    overall = prmbench_step_metrics(
        [row["step_label"] for row in rows], [row["step_score"] for row in rows]
    )
    return {
        "overall": overall,
        "per_family": per_family,
        "family_roster": list(families),
        "all_nine_families_visible": len(per_family) == 9,
    }


def grouped_bootstrap_metric_map(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[list[Mapping[str, Any]]], Mapping[str, float | None]],
    *,
    group_key: str = "group_id",
    stratum_key: str | None = None,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int,
    alpha: float = 0.05,
    bootstrap_unit: str,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Deterministic grouped bootstrap; repeated sampled groups remain repeated."""

    rows = list(rows)
    if not rows or int(draws) != draws or int(draws) < 1:
        raise ValueError("grouped bootstrap needs rows and a positive integer draw count")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("bootstrap alpha must lie in (0,1)")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    strata: dict[str, str] = {}
    for row in rows:
        group_id = str(row[group_key])
        grouped[group_id].append(row)
        stratum = str(row[stratum_key]) if stratum_key is not None else "all"
        previous = strata.setdefault(group_id, stratum)
        if previous != stratum:
            raise ValueError("a bootstrap group crosses registered strata")
    ids_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group_id, stratum in strata.items():
        ids_by_stratum[stratum].append(group_id)
    for value in ids_by_stratum.values():
        value.sort()

    def expand(ids: Sequence[str]) -> list[Mapping[str, Any]]:
        return [row for group_id in ids for row in grouped[group_id]]

    canonical_ids = sorted(grouped)
    point_raw = {str(key): value for key, value in statistic(expand(canonical_ids)).items()}
    if not point_raw:
        raise ValueError("bootstrap statistic must return at least one metric")
    samples: dict[str, list[float | None]] = {key: [] for key in point_raw}
    rng = np.random.default_rng(int(seed))
    draw_hasher = hashlib.sha256()
    draws_executed = 0
    for draw_index in range(int(draws)):
        sampled: list[str] = []
        for stratum in sorted(ids_by_stratum):
            source = ids_by_stratum[stratum]
            picks = rng.integers(0, len(source), size=len(source))
            draw_hasher.update(stratum.encode("utf-8") + b"\0")
            draw_hasher.update(np.asarray(picks, dtype="<i8").tobytes(order="C"))
            sampled.extend(source[int(index)] for index in picks)
        raw = {str(key): value for key, value in statistic(expand(sampled)).items()}
        if set(raw) != set(point_raw):
            raise ValueError("bootstrap statistic changed its metric roster")
        for key, value in raw.items():
            if value is not None and np.isfinite(float(value)):
                samples[key].append(float(value))
            else:
                samples[key].append(None)
        draws_executed = draw_index + 1
    intervals: dict[str, dict[str, Any]] = {}
    for key, point in point_raw.items():
        valid = [value for value in samples[key] if value is not None]
        low, high = (
            np.percentile(valid, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
            if valid else (float("nan"), float("nan"))
        )
        intervals[key] = {
            "point": None if point is None or not np.isfinite(float(point)) else float(point),
            "ci_low": None if not valid else float(low),
            "ci_high": None if not valid else float(high),
            "n_valid": len(valid),
            "status": "OK" if valid else UNDEFINED_SINGLE_CLASS,
        }
    result = {
        "statistics": intervals,
        "draws": int(draws),
        "draws_executed": draws_executed,
        "seed": int(seed),
        "alpha": float(alpha),
        "n_groups": len(grouped),
        "n_groups_by_stratum": {
            key: len(value) for key, value in sorted(ids_by_stratum.items())
        },
        "bootstrap_unit": str(bootstrap_unit),
        "paired_payload": True,
        "draw_stream_sha256": draw_hasher.hexdigest(),
        "sample_stream_sha256": sha256_bytes(canonical_json_bytes(samples)),
    }
    if include_samples:
        result["samples"] = samples
    return result


def bootstrap_processbench_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    expected_subsets: Sequence[str] = PROCESSBENCH_SUBSETS,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = PROCESSBENCH_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    subsets = tuple(map(str, expected_subsets))

    def statistic(sample: list[Mapping[str, Any]]) -> Mapping[str, float]:
        panel = processbench_panel_metrics(
            sample,
            [row["prediction_step"] for row in sample],
            expected_subsets=subsets,
        )
        return panel["aggregate"]

    return grouped_bootstrap_metric_map(
        decisions, statistic, group_key="group_id", stratum_key="slice_id",
        draws=draws, seed=seed, bootstrap_unit="source_question",
    )


def bootstrap_prmbench_steps(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = PRMBENCH_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    def statistic(sample: list[Mapping[str, Any]]) -> Mapping[str, float | None]:
        metrics = prmbench_step_metrics(
            [row["step_label"] for row in sample],
            [row["step_score"] for row in sample],
        )
        return {"auroc": metrics["auroc"], "auprc": metrics["auprc"]}

    return grouped_bootstrap_metric_map(
        rows, statistic, group_key="group_id", draws=draws, seed=seed,
        bootstrap_unit="source_idx",
    )


__all__ = [
    "DEFAULT_BOOTSTRAP_DRAWS", "EVALUATOR_SCHEMA_VERSION",
    "LOCALIZATION_DECISION_FIELDS", "METRIC_FIELDS", "PRMBENCH_ERROR_FAMILIES",
    "PROCESSBENCH_SUBSETS", "UNDEFINED_SINGLE_CLASS",
    "assign_processbench_folds", "bootstrap_prmbench_steps",
    "bootstrap_processbench_decisions", "crossfit_processbench_threshold",
    "evaluator_contract", "fit_processbench_threshold", "grouped_bootstrap_metric_map",
    "prmbench_panel_metrics", "prmbench_step_metrics", "processbench_panel_metrics",
    "processbench_prediction", "processbench_trace_metrics",
]
