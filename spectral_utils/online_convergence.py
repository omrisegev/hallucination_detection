"""Causal prefix-score convergence and early-declaration utilities.

This module implements the frozen protocol in
``docs/experiments/EARLY_ONLINE_EXISTING_DATA_V1.md``.  It is deliberately a
CPU-only retrospective adapter over saved token telemetry.  A prefix is always
rebuilt from truncated telemetry; a whole-trace feature matrix is never sliced.

All public scores are risk-oriented: larger values mean more likely final-
answer error.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from .fixed_application_pipelines import (
    SHARED_GLOBAL_FEATURES,
    raw_token_feature_matrix,
)
from .repeated_measurement_reliability import FixedMixedV2Transformer
from .streaming_utils import deepconf_lowest_group_conf
from .upcr import upcr_fit


DEFAULT_BUDGETS = (16, 32, 64, 128, 256, 512)
IU_FIT = {
    "loss": "l2",
    "exclusion": False,
    "difficulty_gate": False,
    "simple_avg_fallback": False,
    "recompute_after_exclusion": False,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "n_components": 2,
    "auto_components": False,
}
EPS = 1e-12


def _flat_records(cache_obj: Any) -> list[dict[str, Any]]:
    """Normalize the flat per-trace cache schemas used by the current pilot.

    K-trace caches with ``traces``/``corrects`` are expanded and retain their
    question index as ``_group``.  Rich per-candidate dictionaries should be
    adapted by their dataset-specific loader rather than guessed here.
    """

    if isinstance(cache_obj, dict):
        keys = list(cache_obj)
        if keys and all(isinstance(key, (int, np.integer)) for key in keys):
            source = [cache_obj[key] for key in sorted(keys)]
        else:
            source = None
            for key in ("results", "samples", "data"):
                if isinstance(cache_obj.get(key), list):
                    source = cache_obj[key]
                    break
            if source is None:
                raise ValueError(f"unrecognized cache dictionary keys: {keys[:8]}")
    elif isinstance(cache_obj, list):
        source = cache_obj
    else:
        raise ValueError(f"unsupported cache type: {type(cache_obj).__name__}")

    output: list[dict[str, Any]] = []
    for group, item in enumerate(source):
        if not isinstance(item, dict):
            continue
        entropy = item.get("token_entropies")
        if entropy is None:
            entropy = item.get("main_entropies")
        # ProcessBench's ``label`` is the step-error annotation, not final
        # answer correctness.  Prefer the explicit answer-level field whenever
        # it exists; the flat Phase-15 caches then fall back to ``label``.
        label = item.get("final_answer_correct")
        if label is None:
            label = item.get("label")
        if label is None:
            label = item.get("correct")
        if entropy is not None and label is not None:
            row = dict(item)
            row["label"] = int(bool(label))
            row["_group"] = item.get(
                "question_id", item.get("id", item.get("group", group))
            )
            row["_trace_id"] = item.get("trace_id", item.get("id", f"{group}:0"))
            output.append(row)
            continue
        traces, corrects = item.get("traces"), item.get("corrects")
        if traces is None or corrects is None:
            continue
        for candidate, (trace, correct) in enumerate(zip(traces, corrects)):
            output.append({
                "token_entropies": trace,
                "label": int(bool(correct)),
                "_group": item.get("question_id", item.get("group", group)),
                "_trace_id": f"{group}:{candidate}",
            })
    return output


def normalize_cache_records(cache_obj: Any, min_tokens: int = 16) -> list[dict[str, Any]]:
    """Return usable aligned records with stable trace/group metadata."""

    output = []
    for row in _flat_records(cache_obj):
        entropy = np.asarray(row.get("token_entropies", []), dtype=float)
        if len(entropy) < int(min_tokens) or not np.isfinite(entropy).all():
            continue
        label = row.get("label", row.get("correct"))
        if label is None:
            continue
        clean = dict(row)
        clean["label"] = int(bool(label))
        clean["_length"] = int(len(entropy))
        output.append(clean)
    return output


def truncate_telemetry(row: Mapping[str, Any], budget: int | None) -> dict[str, Any]:
    """Truncate every aligned token channel to one observed prefix."""

    n_total = len(row["token_entropies"])
    n = n_total if budget is None else min(n_total, int(budget))
    output: dict[str, Any] = {}
    for name in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
        values = row.get(name)
        if values is not None:
            output[name] = np.asarray(values)[:n]
    top_k = row.get("top_k_logprobs")
    if isinstance(top_k, dict):
        output["top_k_logprobs"] = {
            name: np.asarray(values)[:n] for name, values in top_k.items()
        }
    return output


def causal_raw_prefix_matrix(
    row: Mapping[str, Any],
    budget: int | None,
    *,
    include_elapsed_length: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the 28-stream primary or 29-stream elapsed-length adapter."""

    full = raw_token_feature_matrix(truncate_telemetry(row, budget))
    if include_elapsed_length:
        return full, tuple(SHARED_GLOBAL_FEATURES)
    return full[:, 1:], tuple(SHARED_GLOBAL_FEATURES[1:])


def _equal_trace_rows(matrices: Sequence[np.ndarray], rows_per_trace: int) -> np.ndarray:
    """Sample the same deterministic number of token rows from every trace."""

    sampled = []
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or not len(matrix):
            continue
        indexes = np.linspace(0, len(matrix) - 1, int(rows_per_trace), dtype=int)
        sampled.append(matrix[indexes])
    if not sampled:
        raise ValueError("no non-empty trace matrices were supplied")
    return np.vstack(sampled)


@dataclass
class FrozenPrefixIUModel:
    """Population-fitted mixed-v2/IU-PCR scorer, frozen across all budgets."""

    feature_names: tuple[str, ...]
    raw_keep: np.ndarray
    transformer: FixedMixedV2Transformer
    transformed_keep: np.ndarray
    transformed_mean: np.ndarray
    transformed_std: np.ndarray
    weights: np.ndarray
    diagnostics: dict[str, Any]

    def risk(self, raw_matrix: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_matrix, dtype=float)[:, self.raw_keep]
        transformed = self.transformer.transform(raw)
        selected = transformed[:, self.transformed_keep]
        clean = np.where(
            np.isfinite(selected), selected, self.transformed_mean[None, :]
        )
        standardized = (
            clean - self.transformed_mean[None, :]
        ) / self.transformed_std[None, :]
        return -(standardized @ self.weights)


def fit_frozen_prefix_iu(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_elapsed_length: bool,
    rows_per_trace: int = 32,
) -> FrozenPrefixIUModel:
    """Fit one label-free model on complete calibration traces.

    Each trace contributes exactly ``rows_per_trace`` rows.  Labels are neither
    accepted nor inspected by this routine.
    """

    matrices, names = [], None
    for row in rows:
        matrix, current_names = causal_raw_prefix_matrix(
            row, None, include_elapsed_length=include_elapsed_length
        )
        matrices.append(matrix)
        names = current_names
    if names is None:
        raise ValueError("at least one calibration trace is required")
    raw_fit = _equal_trace_rows(matrices, rows_per_trace)

    finite = np.isfinite(raw_fit)
    finite_any = finite.any(axis=0)
    medians = np.full(raw_fit.shape[1], np.nan)
    medians[finite_any] = np.nanmedian(raw_fit[:, finite_any], axis=0)
    filled = np.where(finite, raw_fit, medians[None, :])
    raw_scale = np.nanstd(filled, axis=0)
    raw_keep = finite_any & np.isfinite(raw_scale) & (raw_scale > 1e-8)
    if int(raw_keep.sum()) < 3:
        raise ValueError("fewer than three non-degenerate raw streams remain")
    kept_names = tuple(name for name, keep in zip(names, raw_keep) if keep)
    transformer = FixedMixedV2Transformer.fit(raw_fit[:, raw_keep], kept_names)
    transformed = np.asarray(transformer.training_output, dtype=float)
    transformed_median = np.nanmedian(transformed, axis=0)
    transformed_clean = np.where(
        np.isfinite(transformed), transformed, transformed_median[None, :]
    )
    scale = transformed_clean.std(axis=0)
    transformed_keep = (
        np.isfinite(transformed_median) & np.isfinite(scale) & (scale > 1e-8)
    )
    if int(transformed_keep.sum()) < 3:
        raise ValueError("fewer than three mixed-v2 streams remain")
    mean = transformed_clean[:, transformed_keep].mean(axis=0)
    std = transformed_clean[:, transformed_keep].std(axis=0)
    standardized = (
        transformed_clean[:, transformed_keep] - mean[None, :]
    ) / std[None, :]
    fitted = upcr_fit(standardized.T, **IU_FIT)
    weights = np.asarray(fitted.w, dtype=float)
    anchor = standardized.mean(axis=1)
    raw_score = standardized @ weights
    corr = (
        float(np.corrcoef(raw_score, anchor)[0, 1])
        if raw_score.std() > EPS and anchor.std() > EPS else float("nan")
    )
    flipped = bool(np.isfinite(corr) and corr < 0)
    if flipped:
        weights = -weights
    diagnostics = {
        "include_elapsed_length": bool(include_elapsed_length),
        "n_fit_traces": int(len(rows)),
        "rows_per_trace": int(rows_per_trace),
        "n_fit_rows": int(len(raw_fit)),
        "input_streams": int(len(names)),
        "raw_kept_streams": int(raw_keep.sum()),
        "kept_feature_names": list(kept_names),
        "dropped_feature_names": [
            name for name, keep in zip(names, raw_keep) if not keep
        ],
        "mixed_kept_streams": int(transformed_keep.sum()),
        "orientation_correlation": corr,
        "orientation_flipped": flipped,
        "labels_seen_during_fit": False,
        "iu_fit": dict(IU_FIT),
        "g2_hat": float(fitted.g2_hat),
        "projection_residual": float(fitted.proj_residual),
    }
    return FrozenPrefixIUModel(
        kept_names,
        raw_keep,
        transformer,
        transformed_keep,
        mean,
        std,
        weights,
        diagnostics,
    )


def _stable_hash(seed: int, value: Any) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def grouped_calibration_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    calibration_fraction: float = 0.5,
    seed: int = 1729,
) -> tuple[list[int], list[int], int]:
    """Deterministic group-disjoint split with a two-class validity retry."""

    groups = sorted({str(row["_group"]) for row in rows})
    if len(groups) < 4:
        raise ValueError("at least four question groups are required")
    n_cal = min(len(groups) - 1, max(1, int(round(len(groups) * calibration_fraction))))
    for offset in range(100):
        used_seed = int(seed + offset)
        ordered = sorted(groups, key=lambda value: _stable_hash(used_seed, value))
        calibration_groups = set(ordered[:n_cal])
        calibration = [
            index for index, row in enumerate(rows)
            if str(row["_group"]) in calibration_groups
        ]
        evaluation = [index for index in range(len(rows)) if index not in set(calibration)]
        y_cal = {int(not bool(rows[index]["label"])) for index in calibration}
        y_eval = {int(not bool(rows[index]["label"])) for index in evaluation}
        if len(y_cal) == 2 and len(y_eval) == 2:
            return calibration, evaluation, used_seed
    raise ValueError("could not create a group-disjoint two-class split")


def prefix_method_scores(
    row: Mapping[str, Any],
    budget: int | None,
    models: Mapping[str, FrozenPrefixIUModel],
) -> dict[str, float]:
    """Score one observed prefix with frozen IU models and access-matched controls."""

    prefix = truncate_telemetry(row, budget)
    entropy = np.asarray(prefix["token_entropies"], dtype=float)
    raw29 = raw_token_feature_matrix(prefix)
    scores = {
        "mean_entropy": float(entropy.mean()),
        "max_entropy": float(entropy.max()),
        "deepconf_entropy_w32": float(-deepconf_lowest_group_conf(entropy, 32)),
        "deepconf_entropy_w64": float(-deepconf_lowest_group_conf(entropy, 64)),
    }
    for name, model in models.items():
        raw = raw29 if model.diagnostics["include_elapsed_length"] else raw29[:, 1:]
        scores[name] = float(np.max(model.risk(raw)))
    return scores


def build_score_rows(
    rows: Sequence[Mapping[str, Any]],
    indexes: Iterable[int],
    models: Mapping[str, FrozenPrefixIUModel],
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> list[dict[str, Any]]:
    """Return long-form score trajectories for one split."""

    output: list[dict[str, Any]] = []
    for index in indexes:
        row = rows[index]
        length = int(row["_length"])
        common = {
            "unit_index": int(index),
            "trace_id": str(row["_trace_id"]),
            "group": str(row["_group"]),
            "label_error": int(not bool(row["label"])),
            "trace_length": length,
        }
        for budget in budgets:
            if length <= int(budget):
                continue
            scores = prefix_method_scores(row, int(budget), models)
            output.extend({
                **common,
                "budget": int(budget),
                "is_final": False,
                "method": method,
                "score": float(score),
            } for method, score in scores.items())
        scores = prefix_method_scores(row, None, models)
        output.extend({
            **common,
            "budget": length,
            "is_final": True,
            "method": method,
            "score": float(score),
        } for method, score in scores.items())
    return output


def threshold_youden(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float(np.median(scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return float(np.median(scores))
    quality = np.where(finite, tpr - fpr, -np.inf)
    return float(thresholds[int(np.argmax(quality))])


def final_calibration(
    score_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Freeze each method's final decision threshold, scale, and tolerance."""

    methods = sorted({str(row["method"]) for row in score_rows})
    output = {}
    for method in methods:
        selected = [
            row for row in score_rows
            if row["method"] == method and bool(row["is_final"])
        ]
        labels = np.asarray([row["label_error"] for row in selected], dtype=int)
        scores = np.asarray([row["score"] for row in selected], dtype=float)
        scale = float(scores.std())
        output[method] = {
            "threshold": threshold_youden(labels, scores),
            "final_score_std": max(scale, EPS),
            "tolerance": max(0.25 * scale, EPS),
        }
    return output


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def convergence_table(
    score_rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Mapping[str, float]],
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> list[dict[str, Any]]:
    """Compute fixed-budget convergence on each budget's at-risk cohort."""

    methods = sorted({str(row["method"]) for row in score_rows})
    finals = {
        (str(row["method"]), int(row["unit_index"])): row
        for row in score_rows if bool(row["is_final"])
    }
    lookup = {
        (str(row["method"]), int(row["unit_index"]), int(row["budget"])): row
        for row in score_rows if not bool(row["is_final"])
    }
    output = []
    for method in methods:
        previous_decision: dict[int, int] = {}
        for budget in budgets:
            pairs = []
            for (m, unit, b), prefix in lookup.items():
                if m == method and b == int(budget) and (method, unit) in finals:
                    pairs.append((prefix, finals[method, unit]))
            if not pairs:
                continue
            labels = np.asarray([p[0]["label_error"] for p in pairs], dtype=int)
            prefix_scores = np.asarray([p[0]["score"] for p in pairs], dtype=float)
            final_scores = np.asarray([p[1]["score"] for p in pairs], dtype=float)
            threshold = float(calibration[method]["threshold"])
            prefix_decision = (prefix_scores >= threshold).astype(int)
            final_decision = (final_scores >= threshold).astype(int)
            flips = []
            for (prefix, _), decision in zip(pairs, prefix_decision):
                unit = int(prefix["unit_index"])
                if unit in previous_decision:
                    flips.append(int(previous_decision[unit] != int(decision)))
                previous_decision[unit] = int(decision)
            auc = safe_auc(labels, prefix_scores)
            final_auc_same = safe_auc(labels, final_scores)
            recovery = (
                (auc - 0.5) / (final_auc_same - 0.5)
                if np.isfinite(auc) and np.isfinite(final_auc_same)
                and abs(final_auc_same - 0.5) > EPS else float("nan")
            )
            if (
                len(prefix_scores) >= 3
                and prefix_scores.std() > EPS
                and final_scores.std() > EPS
            ):
                spearman = spearmanr(prefix_scores, final_scores).statistic
                kendall = kendalltau(prefix_scores, final_scores).statistic
            else:
                spearman = kendall = float("nan")
            output.append({
                "method": method,
                "budget": int(budget),
                "n_at_risk": int(len(pairs)),
                "n_error": int(labels.sum()),
                "auroc": auc,
                "auprc": (
                    float(average_precision_score(labels, prefix_scores))
                    if len(np.unique(labels)) == 2 else float("nan")
                ),
                "final_auroc_same_cohort": final_auc_same,
                "above_chance_auc_recovery": float(recovery),
                "spearman_vs_final": float(spearman),
                "kendall_vs_final": float(kendall),
                "normalized_mae_vs_final": float(
                    np.mean(np.abs(prefix_scores - final_scores))
                    / calibration[method]["final_score_std"]
                ),
                "final_decision_agreement": float(np.mean(prefix_decision == final_decision)),
                "flip_rate_vs_previous_budget": (
                    float(np.mean(flips)) if flips else float("nan")
                ),
            })
    return output


def _method_trajectories(
    score_rows: Sequence[Mapping[str, Any]], method: str
) -> dict[int, list[Mapping[str, Any]]]:
    trajectories: dict[int, list[Mapping[str, Any]]] = {}
    for row in score_rows:
        if row["method"] != method or bool(row["is_final"]):
            continue
        trajectories.setdefault(int(row["unit_index"]), []).append(row)
    for values in trajectories.values():
        values.sort(key=lambda row: int(row["budget"]))
    return trajectories


def apply_declaration_policy(
    score_rows: Sequence[Mapping[str, Any]],
    method: str,
    *,
    low: float,
    high: float,
    stable_observations: int = 2,
) -> list[dict[str, Any]]:
    """Apply the first stable two-sided declaration to every trace."""

    if not low < high:
        raise ValueError("low must be smaller than high")
    trajectories = _method_trajectories(score_rows, method)
    finals = {
        int(row["unit_index"]): row for row in score_rows
        if row["method"] == method and bool(row["is_final"])
    }
    output = []
    for unit, final in finals.items():
        history: list[int | None] = []
        declaration = None
        declaration_budget = None
        declaration_score = None
        for row in trajectories.get(unit, []):
            score = float(row["score"])
            side = 1 if score >= high else (0 if score <= low else None)
            history.append(side)
            recent = history[-int(stable_observations):]
            if (
                len(recent) == int(stable_observations)
                and recent[0] is not None
                and all(value == recent[0] for value in recent)
            ):
                declaration = int(recent[0])
                declaration_budget = int(row["budget"])
                declaration_score = score
                break
        truth = int(final["label_error"])
        output.append({
            "unit_index": int(unit),
            "group": str(final["group"]),
            "truth_error": truth,
            "declared": declaration is not None,
            "declaration": declaration,
            "correct_declaration": (
                bool(declaration == truth) if declaration is not None else None
            ),
            "declaration_budget": declaration_budget,
            "declaration_score": declaration_score,
            "trace_length": int(final["trace_length"]),
            "potential_tokens_remaining": (
                int(final["trace_length"] - declaration_budget)
                if declaration_budget is not None else 0
            ),
        })
    return output


def declaration_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    n = len(rows)
    declared = [row for row in rows if bool(row["declared"])]
    wrong = [row for row in declared if not bool(row["correct_declaration"])]
    false_alarm = [
        row for row in declared
        if int(row["declaration"]) == 1 and int(row["truth_error"]) == 0
    ]
    false_clearance = [
        row for row in declared
        if int(row["declaration"]) == 0 and int(row["truth_error"]) == 1
    ]
    return {
        "n": int(n),
        "n_declared": int(len(declared)),
        "coverage": float(len(declared) / n) if n else float("nan"),
        "ever_wrong_rate_all": float(len(wrong) / n) if n else float("nan"),
        "selective_error_rate": (
            float(len(wrong) / len(declared)) if declared else float("nan")
        ),
        "false_alarm_rate_all": float(len(false_alarm) / n) if n else float("nan"),
        "false_clearance_rate_all": (
            float(len(false_clearance) / n) if n else float("nan")
        ),
        "mean_decision_budget": (
            float(np.mean([row["declaration_budget"] for row in declared]))
            if declared else float("nan")
        ),
        "mean_potential_tokens_remaining": (
            float(np.mean([row["potential_tokens_remaining"] for row in declared]))
            if declared else 0.0
        ),
    }


def calibrate_declaration_policy(
    score_rows: Sequence[Mapping[str, Any]],
    method: str,
    *,
    max_ever_wrong: float = 0.10,
    stable_observations: int = 2,
) -> dict[str, Any]:
    """Choose the highest-coverage calibration-feasible two-sided thresholds."""

    observed = np.asarray([
        row["score"] for row in score_rows
        if row["method"] == method and not bool(row["is_final"])
    ], dtype=float)
    if not len(observed):
        raise ValueError(f"no prefix observations for {method}")
    quantiles = np.linspace(0.02, 0.98, 25)
    candidates = np.unique(np.quantile(observed, quantiles))
    best = None
    for low_index, low in enumerate(candidates[:-1]):
        for high in candidates[low_index + 1:]:
            declarations = apply_declaration_policy(
                score_rows, method, low=float(low), high=float(high),
                stable_observations=stable_observations,
            )
            summary = declaration_summary(declarations)
            if summary["ever_wrong_rate_all"] > float(max_ever_wrong) + EPS:
                continue
            decision_budget = summary["mean_decision_budget"]
            if not np.isfinite(decision_budget):
                decision_budget = float("inf")
            objective = (
                float(summary["coverage"]),
                -float(summary["ever_wrong_rate_all"]),
                -float(decision_budget),
                float(high - low),
            )
            if best is None or objective > best[0]:
                best = (objective, float(low), float(high), summary)
    if best is None:
        raise RuntimeError("no declaration threshold pair was calibration-feasible")
    return {
        "method": method,
        "low": best[1],
        "high": best[2],
        "stable_observations": int(stable_observations),
        "max_ever_wrong": float(max_ever_wrong),
        "calibration_summary": best[3],
    }


def per_trace_convergence(
    score_rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Retain oracle convergence diagnostics without using them as a policy."""

    methods = sorted({str(row["method"]) for row in score_rows})
    finals = {
        (str(row["method"]), int(row["unit_index"])): row
        for row in score_rows if bool(row["is_final"])
    }
    output = []
    for method in methods:
        threshold = float(calibration[method]["threshold"])
        tolerance = float(calibration[method]["tolerance"])
        for unit, trajectory in _method_trajectories(score_rows, method).items():
            final = finals[method, unit]
            final_score = float(final["score"])
            decisions = [int(float(row["score"]) >= threshold) for row in trajectory]
            crossings = sum(a != b for a, b in zip(decisions[:-1], decisions[1:]))
            last_flip_budget = None
            for index in range(1, len(decisions)):
                if decisions[index] != decisions[index - 1]:
                    last_flip_budget = int(trajectory[index]["budget"])
            first_within = None
            for row in trajectory:
                if abs(float(row["score"]) - final_score) <= tolerance:
                    first_within = int(row["budget"])
                    break
            output.append({
                "method": method,
                "unit_index": int(unit),
                "group": str(final["group"]),
                "label_error": int(final["label_error"]),
                "trace_length": int(final["trace_length"]),
                "final_score": final_score,
                "threshold_crossings": int(crossings),
                "oracle_last_flip_budget": last_flip_budget,
                "first_budget_within_final_tolerance": first_within,
            })
    return output


__all__ = [
    "DEFAULT_BUDGETS",
    "FrozenPrefixIUModel",
    "apply_declaration_policy",
    "build_score_rows",
    "calibrate_declaration_policy",
    "causal_raw_prefix_matrix",
    "convergence_table",
    "declaration_summary",
    "final_calibration",
    "fit_frozen_prefix_iu",
    "grouped_calibration_split",
    "normalize_cache_records",
    "per_trace_convergence",
    "prefix_method_scores",
    "truncate_telemetry",
]
