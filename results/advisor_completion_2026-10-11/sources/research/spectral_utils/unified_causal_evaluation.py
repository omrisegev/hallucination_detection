"""Supervised development and evaluation helpers for Unified Causal IU-PCR.

This module owns every label-aware operation.  Keeping it separate from
``unified_causal_iu`` makes leakage boundaries inspectable: the deployed stateful
object cannot import correctness or first-error fields by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .multitask_trajectory import equal_positions
from .unified_causal_iu import (
    AccumulatorSpec,
    BaseReference,
    UnifiedCausalIU,
    all_feature_names,
    causal_feature_matrix,
)


DEFAULT_BUDGETS = (16, 32, 64, 128, 256, 512)
PRIMARY_EARLY_BUDGETS = (64, 128)
EPS = 1e-12


def _row_identity(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("_unit", row.get("id", index)))


def _row_family(row: Mapping[str, Any]) -> str:
    return str(row.get("family", row.get("dataset_family", row.get("dataset", "unknown"))))


def _row_model(row: Mapping[str, Any]) -> str:
    return str(row.get("model", row.get("scorer_model", "unknown")))


def source_group(row: Mapping[str, Any], index: int = 0) -> str:
    """Dataset-qualified source question shared by scorer/model copies."""

    return f"{_row_family(row)}::{_row_identity(row, index)}"


def final_wrong(row: Mapping[str, Any]) -> int:
    if row.get("final_answer_correct") is not None:
        return int(not bool(row["final_answer_correct"]))
    if row.get("is_correct") is not None:
        return int(not bool(row["is_correct"]))
    if row.get("correct") is not None:
        return int(not bool(row["correct"]))
    raise KeyError("row has no final-answer correctness field")


def token_to_step(token: int, row: Mapping[str, Any]) -> int:
    spans = row.get("step_token_spans") or ()
    for index, span in enumerate(spans):
        if span is not None and int(span[0]) <= int(token) < int(span[1]):
            return int(index)
    return int(len(spans) - 1) if spans else -1


def first_error_mask(row: Mapping[str, Any], length: int) -> np.ndarray:
    output = np.zeros(int(length), dtype=int)
    label = int(row.get("label", -1))
    spans = row.get("step_token_spans") or ()
    if label < 0 or label >= len(spans) or spans[label] is None:
        return output
    start, stop = map(int, spans[label])
    output[max(0, start):min(len(output), stop)] = 1
    return output


def _balanced_weights(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    weight = np.zeros(len(labels), dtype=float)
    unique, counts = np.unique(groups, return_counts=True)
    group_count = dict(zip(unique, counts))
    for index, group in enumerate(groups):
        weight[index] = 1.0 / group_count[group]
    for target in np.unique(labels):
        mask = labels == target
        mass = float(np.sum(weight[mask]))
        if mass > 0:
            weight[mask] *= 0.5 / mass
    return weight / (np.mean(weight) + EPS)


def _categorical_context(
    budgets: Sequence[str], families: Sequence[str], models: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns, names = [], []
    for prefix, values in (("budget", budgets), ("family", families), ("model", models)):
        values = np.asarray(values, dtype=str)
        for value in sorted(set(values)):
            columns.append((values == value).astype(float))
            names.append(f"context::{prefix}={value}")
    matrix = np.column_stack(columns) if columns else np.empty((len(budgets), 0))
    return matrix, tuple(names)


@dataclass(frozen=True)
class AtlasSamples:
    target: str
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    families: np.ndarray
    models: np.ndarray
    budgets: np.ndarray
    context: np.ndarray
    context_names: tuple[str, ...]
    baseline: np.ndarray
    baseline_names: tuple[str, ...]

    def subset(self, mask: np.ndarray) -> "AtlasSamples":
        mask = np.asarray(mask, dtype=bool)
        return AtlasSamples(
            target=self.target,
            feature_names=self.feature_names,
            X=self.X[mask],
            y=self.y[mask],
            groups=self.groups[mask],
            families=self.families[mask],
            models=self.models[mask],
            budgets=self.budgets[mask],
            context=self.context[mask],
            context_names=self.context_names,
            baseline=self.baseline[mask],
            baseline_names=self.baseline_names,
        )


def build_atlas_samples(
    rows: Sequence[Mapping[str, Any]],
    reference: BaseReference,
    *,
    target: str,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    positions_per_trace: int = 32,
    iu28_curves: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None = None,
    feature_matrices: Sequence[np.ndarray] | None = None,
) -> AtlasSamples:
    """Construct equal-trace grouped samples for one information target."""

    feature_names = all_feature_names(reference.names)
    values, labels, groups, families, models, budget_labels = [], [], [], [], [], []
    baselines = []
    entropy_name = "raw::entropy::level"
    sw_name = "broad::entropy_sw_var_series::level"
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    baseline_features = tuple(
        (label, name)
        for label, name in (("entropy", entropy_name), ("sw_var", sw_name))
        if name in name_to_index
    )
    if feature_matrices is not None and len(feature_matrices) != len(rows):
        raise ValueError("feature matrices do not match atlas rows")
    for row_index, row in enumerate(rows):
        identity = source_group(row, row_index)
        family, model = _row_family(row), _row_model(row)
        matrix = (
            causal_feature_matrix(row, reference)
            if feature_matrices is None
            else np.asarray(feature_matrices[row_index], dtype=float)
        )
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
            raise ValueError("precomputed atlas matrix violates frozen bank schema")
        n = len(matrix)
        if target in {"global", "early"}:
            positions = [min(int(budget), n) - 1 for budget in budgets]
            labels_here = [final_wrong(row)] * len(positions)
            labels_at = [str(int(budget)) for budget in budgets]
            if target == "global":
                positions.append(n - 1)
                labels_here.append(final_wrong(row))
                labels_at.append("final")
        elif target == "localization":
            positions = list(equal_positions(n, positions_per_trace))
            mask = first_error_mask(row, n)
            if np.any(mask):
                centre = int(np.flatnonzero(mask)[len(np.flatnonzero(mask)) // 2])
                closest = int(np.argmin(np.abs(np.asarray(positions) - centre)))
                positions[closest] = centre
            labels_here = mask[np.asarray(positions)].tolist()
            labels_at = ["token"] * len(positions)
        else:
            raise KeyError(target)
        for position, label, budget_name in zip(positions, labels_here, labels_at):
            values.append(matrix[position])
            labels.append(label)
            groups.append(identity)
            families.append(family)
            models.append(model)
            budget_labels.append(budget_name)
            iu28 = float("nan")
            if iu28_curves is not None:
                if isinstance(iu28_curves, Mapping):
                    # The model-qualified key prevents scorer copies of one source
                    # question from overwriting each other.
                    curve_values = iu28_curves.get(f"{identity}::{model}")
                else:
                    curve_values = iu28_curves[row_index]
                if curve_values is not None:
                    curve = np.asarray(curve_values, dtype=float)
                    if len(curve):
                        iu28 = float(curve[min(position, len(curve) - 1)])
            baselines.append([
                *[
                    matrix[position, name_to_index[name]]
                    for _, name in baseline_features
                ],
                iu28,
            ])
    budgets_array = np.asarray(budget_labels, dtype=str)
    families_array = np.asarray(families, dtype=str)
    models_array = np.asarray(models, dtype=str)
    context, context_names = _categorical_context(
        budgets_array, families_array, models_array
    )
    baseline = np.asarray(baselines, dtype=float)
    keep_baseline = np.isfinite(baseline).any(axis=0)
    baseline_names = tuple(label for label, _ in baseline_features) + ("IU28",)
    baseline = baseline[:, keep_baseline]
    median = np.nanmedian(baseline, axis=0)
    baseline = np.where(np.isfinite(baseline), baseline, median)
    return AtlasSamples(
        target=target,
        feature_names=feature_names,
        X=np.asarray(values, dtype=float),
        y=np.asarray(labels, dtype=int),
        groups=np.asarray(groups, dtype=str),
        families=families_array,
        models=models_array,
        budgets=budgets_array,
        context=context,
        context_names=context_names,
        baseline=baseline,
        baseline_names=tuple(name for name, use in zip(baseline_names, keep_baseline) if use),
    )


def grouped_mutual_information(samples: AtlasSamples, seed: int = 20260817) -> np.ndarray:
    """Descriptive MI; equal rows per source trace prevent length weighting."""

    X = np.asarray(samples.X, dtype=float)
    median = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, median)
    return np.asarray(mutual_info_classif(X, samples.y, random_state=seed), dtype=float)


def _splitter(labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    unique_groups = np.unique(groups)
    n_splits = min(int(n_splits), len(unique_groups))
    if n_splits < 2:
        raise ValueError("at least two source-question groups are required")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(labels)), labels, groups
    )


def _nonlinear_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=60,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def heldout_logloss_gain(
    samples: AtlasSamples,
    candidate_indices: Sequence[int],
    *,
    n_splits: int = 5,
    seed: int = 20260817,
    labels: np.ndarray | None = None,
) -> tuple[float, tuple[float, ...]]:
    """Mean foldwise nonlinear log-loss improvement beyond context+baselines."""

    y = np.asarray(samples.y if labels is None else labels, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan"), ()
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    base = np.column_stack([samples.context, samples.baseline])
    candidate = np.asarray(samples.X[:, candidate_indices], dtype=float)
    median = np.nanmedian(candidate, axis=0)
    candidate = np.where(np.isfinite(candidate), candidate, median)
    augmented = np.column_stack([base, candidate])
    fold_gains = []
    try:
        splits = list(_splitter(y, samples.groups, n_splits, seed))
    except ValueError:
        return float("nan"), ()
    for fold, (train, test) in enumerate(splits):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        train_weight = _balanced_weights(y[train], samples.groups[train])
        base_model = _nonlinear_model(seed + 2 * fold)
        full_model = _nonlinear_model(seed + 2 * fold + 1)
        base_model.fit(base[train], y[train], sample_weight=train_weight)
        full_model.fit(augmented[train], y[train], sample_weight=train_weight)
        base_probability = np.clip(base_model.predict_proba(base[test])[:, 1], 1e-6, 1 - 1e-6)
        full_probability = np.clip(full_model.predict_proba(augmented[test])[:, 1], 1e-6, 1 - 1e-6)
        test_weight = _balanced_weights(y[test], samples.groups[test])
        base_loss = log_loss(y[test], base_probability, sample_weight=test_weight, labels=[0, 1])
        full_loss = log_loss(y[test], full_probability, sample_weight=test_weight, labels=[0, 1])
        fold_gains.append(float(base_loss - full_loss))
    return (
        float(np.mean(fold_gains)) if fold_gains else float("nan"),
        tuple(fold_gains),
    )


def grouped_block_permutation(
    labels: Sequence[int], groups: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    """Permute complete equal-sized source-question label blocks."""

    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique]
    sizes = {len(index) for index in indices}
    if len(sizes) != 1:
        raise ValueError("grouped block permutation requires equal samples per question")
    source = rng.permutation(len(unique))
    output = np.empty_like(labels)
    for target_index, source_index in enumerate(source):
        output[indices[target_index]] = labels[indices[source_index]]
    return output


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return q
    order = finite[np.argsort(p[finite], kind="mergesort")]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def redundancy_prune(
    X: np.ndarray,
    names: Sequence[str],
    conditional_gains: Mapping[str, float],
    *,
    threshold: float = 0.98,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Fold-local exact/|rho|>=threshold pruning, strongest conditional gain wins."""

    X = np.asarray(X, dtype=float)
    names = tuple(names)
    median = np.nanmedian(X, axis=0)
    clean = np.where(np.isfinite(X), X, median)
    priority = sorted(
        range(len(names)),
        key=lambda index: (-float(conditional_gains.get(names[index], float("-inf"))), names[index]),
    )
    kept: list[int] = []
    audit: list[dict[str, Any]] = []
    for index in priority:
        rejected = None
        reason = None
        for existing in kept:
            if np.array_equal(clean[:, index], clean[:, existing]):
                rejected, reason = existing, "exact_duplicate"
                break
            left, right = clean[:, index], clean[:, existing]
            if np.std(left) <= EPS or np.std(right) <= EPS:
                continue
            rho = float(np.corrcoef(left, right)[0, 1])
            if np.isfinite(rho) and abs(rho) >= threshold:
                rejected, reason = existing, f"abs_rho={abs(rho):.6f}"
                break
        if rejected is None:
            kept.append(index)
        else:
            audit.append({
                "dropped": names[index],
                "kept": names[rejected],
                "reason": reason,
                "dropped_gain": float(conditional_gains.get(names[index], float("nan"))),
                "kept_gain": float(conditional_gains.get(names[rejected], float("nan"))),
            })
    kept.sort()
    return tuple(names[index] for index in kept), audit


def derive_supervised_signs(
    sample_sets: Sequence[AtlasSamples], roster: Sequence[str]
) -> dict[str, float]:
    """Freeze one sign per feature from an equal-target/equal-family effect vote."""

    roster = tuple(roster)
    votes = {name: [] for name in roster}
    for samples in sample_sets:
        index = {name: position for position, name in enumerate(samples.feature_names)}
        for family in sorted(set(samples.families)):
            mask = samples.families == family
            y = samples.y[mask]
            if len(np.unique(y)) < 2:
                continue
            for name in roster:
                values = np.asarray(samples.X[mask, index[name]], dtype=float)
                positive = values[y == 1]
                negative = values[y == 0]
                effect = float(np.nanmedian(positive) - np.nanmedian(negative))
                scale = float(np.nanmedian(np.abs(values - np.nanmedian(values)))) + EPS
                votes[name].append(np.clip(effect / scale, -5.0, 5.0))
    return {
        name: (-1.0 if np.nanmean(votes[name]) < 0.0 else 1.0)
        if votes[name] else 1.0
        for name in roster
    }


def build_information_atlas(
    sample_sets: Sequence[AtlasSamples],
    *,
    permutation_repeats: int = 20,
    n_splits: int = 5,
    seed: int = 20260817,
    candidate_names: Sequence[str] | None = None,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Compute MI, conditional gain, family consistency, and grouped FDR.

    ``permutation_repeats`` is deliberately explicit because the full 1,036-feature
    atlas is expensive.  Publication runs should use at least 200; unit/smoke runs may
    use fewer and are stamped with the actual count.
    """

    records: list[dict[str, Any]] = []
    for samples in sample_sets:
        feature_index = {name: index for index, name in enumerate(samples.feature_names)}
        names = tuple(candidate_names) if candidate_names is not None else samples.feature_names
        if not set(names) <= set(feature_index):
            raise ValueError("atlas candidate is absent from sample matrix")
        mi = grouped_mutual_information(samples, seed=seed)
        rng = np.random.default_rng(seed + sum(ord(char) for char in samples.target))
        null_mi = []
        clean = np.asarray(samples.X, dtype=float)
        median = np.nanmedian(clean, axis=0)
        clean = np.where(np.isfinite(clean), clean, median)
        for repeat in range(int(permutation_repeats)):
            permuted = grouped_block_permutation(samples.y, samples.groups, rng)
            null_mi.append(mutual_info_classif(
                clean,
                permuted,
                random_state=seed + 1000 + repeat,
            ))
        null_mi = (
            np.asarray(null_mi, dtype=float)
            if null_mi else np.empty((0, len(samples.feature_names)), dtype=float)
        )

        def score_feature(name: str) -> dict[str, Any]:
            index = feature_index[name]
            gain, folds = heldout_logloss_gain(
                samples, [index], n_splits=n_splits, seed=seed
            )
            finite_null = null_mi[:, index]
            p_value = float(
                (1 + np.sum(finite_null >= mi[index])) / (1 + len(finite_null))
            ) if np.isfinite(mi[index]) else float("nan")
            family_gains = {}
            for family in sorted(set(samples.families)):
                subset = samples.subset(samples.families == family)
                family_gain, _ = heldout_logloss_gain(
                    subset,
                    [index],
                    n_splits=min(n_splits, 3),
                    seed=seed,
                )
                family_gains[family] = family_gain
            return {
                "target": samples.target,
                "feature": name,
                "mutual_information": float(mi[index]),
                "conditional_logloss_gain": gain,
                "fold_gains": list(folds),
                "positive_families": int(sum(value > 0.0 for value in family_gains.values() if np.isfinite(value))),
                "family_gains": family_gains,
                "permutation_p": p_value,
                "permutation_statistic": "grouped_block_mutual_information",
                "permutation_repeats": int(permutation_repeats),
                "n_samples": len(samples.y),
                "n_groups": len(np.unique(samples.groups)),
                "baseline_names": list(samples.baseline_names),
                "conditioned_on": list(samples.context_names),
            }
        scored = Parallel(n_jobs=int(n_jobs), prefer="processes")(
            delayed(score_feature)(name) for name in names
        ) if int(n_jobs) != 1 else [score_feature(name) for name in names]
        records.extend(scored)
    q = benjamini_hochberg([row["permutation_p"] for row in records])
    for row, value in zip(records, q):
        row["fdr_q"] = float(value)
        row["roster_pass"] = bool(
            row["conditional_logloss_gain"] > 0.0
            and row["positive_families"] >= 3
            and np.isfinite(value)
            and value <= 0.10
        )
    return records


def _transform_family(name: str) -> str:
    transform = str(name).split("::")[-1]
    if transform == "level":
        return "level"
    if transform.startswith("ewma") or transform.startswith("fastminus"):
        return "multiscale_ewma"
    if transform.startswith(("mean", "var", "mad")):
        return "moving_statistics"
    if transform.startswith("innovation"):
        return "innovation"
    if transform in {"positive_area", "persistence"}:
        return "area_persistence"
    if transform.startswith(("cusum", "page_hinkley")):
        return "sequential_change"
    if transform.startswith("bocpd"):
        return "bocpd"
    return "other"


def feature_groups(feature_names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Return transform-family and source-stream groups for synergy measurement."""

    output: dict[str, list[str]] = {}
    for name in feature_names:
        parts = str(name).split("::")
        base = "::".join(parts[:-1])
        output.setdefault(f"transform::{_transform_family(name)}", []).append(str(name))
        output.setdefault(f"source::{base}", []).append(str(name))
    return {key: tuple(value) for key, value in sorted(output.items()) if len(value) > 1}


def build_group_synergy_atlas(
    sample_sets: Sequence[AtlasSamples],
    individual_atlas: Sequence[Mapping[str, Any]],
    *,
    permutation_repeats: int = 20,
    n_splits: int = 5,
    seed: int = 20260817,
) -> list[dict[str, Any]]:
    """Measure joint family gain and gain beyond its strongest single member."""

    individual = {
        (str(row["target"]), str(row["feature"])): float(row["conditional_logloss_gain"])
        for row in individual_atlas
    }
    rng = np.random.default_rng(seed + 771)
    records = []
    for samples in sample_sets:
        lookup = {name: index for index, name in enumerate(samples.feature_names)}
        clean = np.asarray(samples.X, dtype=float)
        median = np.nanmedian(clean, axis=0)
        clean = np.where(np.isfinite(clean), clean, median)
        for group, members in feature_groups(samples.feature_names).items():
            indices = [lookup[name] for name in members]
            gain, fold_gains = heldout_logloss_gain(
                samples, indices, n_splits=n_splits, seed=seed
            )
            best_member = max(
                (individual.get((samples.target, name), float("-inf")) for name in members),
                default=float("nan"),
            )
            observed_mi = float(np.sum(mutual_info_classif(
                clean[:, indices], samples.y, random_state=seed
            )))
            null = []
            for repeat in range(int(permutation_repeats)):
                permuted = grouped_block_permutation(samples.y, samples.groups, rng)
                null.append(float(np.sum(mutual_info_classif(
                    clean[:, indices], permuted, random_state=seed + 3000 + repeat
                ))))
            finite = np.asarray([value for value in null if np.isfinite(value)])
            p_value = float((1 + np.sum(finite >= observed_mi)) / (1 + len(finite)))
            family_gains = {}
            for family in sorted(set(samples.families)):
                subset = samples.subset(samples.families == family)
                family_gains[family], _ = heldout_logloss_gain(
                    subset, indices, n_splits=min(n_splits, 3), seed=seed
                )
            records.append({
                "target": samples.target,
                "group": group,
                "n_members": len(members),
                "members": list(members),
                "conditional_logloss_gain": gain,
                "best_member_gain": best_member,
                "synergy_beyond_best_member": float(gain - best_member),
                "fold_gains": list(fold_gains),
                "positive_families": int(sum(value > 0.0 for value in family_gains.values() if np.isfinite(value))),
                "family_gains": family_gains,
                "permutation_p": p_value,
                "permutation_statistic": "sum_grouped_block_mutual_information",
                "permutation_repeats": int(permutation_repeats),
            })
    q = benjamini_hochberg([row["permutation_p"] for row in records])
    for row, value in zip(records, q):
        row["fdr_q"] = float(value)
        row["family_pass"] = bool(
            row["conditional_logloss_gain"] > 0.0
            and row["positive_families"] >= 3
            and np.isfinite(value)
            and value <= 0.10
        )
    return records


def select_atlas_roster(
    atlas: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    X_development: np.ndarray,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Union target-specific passes, then apply the fold-local redundancy rule."""

    selected = sorted({str(row["feature"]) for row in atlas if bool(row.get("roster_pass"))})
    fallback_used = len(selected) < 3
    if len(selected) < 3:
        # This is a transparent diagnostic fallback, not a hidden selection rule: keep
        # the three highest conditional-gain coordinates so IU-PCR remains identifiable.
        ranked = sorted(
            atlas,
            key=lambda row: (-float(row.get("conditional_logloss_gain", float("-inf"))), str(row["feature"])),
        )
        selected = list(dict.fromkeys(str(row["feature"]) for row in ranked))[:3]
    gain = {}
    for row in atlas:
        name = str(row["feature"])
        gain[name] = max(gain.get(name, float("-inf")), float(row["conditional_logloss_gain"]))
    index = {name: position for position, name in enumerate(feature_names)}
    matrix = np.column_stack([X_development[:, index[name]] for name in selected])
    roster, audit = redundancy_prune(matrix, selected, gain)
    if fallback_used:
        audit.insert(0, {
            "dropped": None,
            "kept": list(selected),
            "reason": "top_three_identifiability_fallback",
            "dropped_gain": float("nan"),
            "kept_gain": float("nan"),
        })
    if len(roster) < 3:
        raise RuntimeError("redundancy pruning left fewer than three IU coordinates")
    return roster, audit


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_ap(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def processbench_metrics(prediction: Sequence[int], target: Sequence[int]) -> dict[str, float]:
    prediction, target = np.asarray(prediction, dtype=int), np.asarray(target, dtype=int)
    error, clean = target != -1, target == -1
    exact = float(np.mean(prediction[error] == target[error])) if error.any() else float("nan")
    abstention = float(np.mean(prediction[clean] == -1)) if clean.any() else float("nan")
    f1 = (
        2.0 * exact * abstention / (exact + abstention)
        if np.isfinite(exact) and np.isfinite(abstention) and exact + abstention > 0
        else 0.0
    )
    within = float(np.mean(
        (prediction[error] != -1) & (np.abs(prediction[error] - target[error]) <= 1)
    )) if error.any() else float("nan")
    return {"f1": f1, "exact": exact, "within_one": within, "clean_abstention": abstention}


def best_localization_threshold(
    terminal_risk: Sequence[float], localization_steps: Sequence[int], labels: Sequence[int]
) -> tuple[float, float]:
    risk = np.asarray(terminal_risk, dtype=float)
    locator = np.asarray(localization_steps, dtype=int)
    labels = np.asarray(labels, dtype=int)
    candidates = np.r_[float("inf"), np.unique(risk), float("-inf")]
    best = (float("inf"), -1.0)
    for threshold in candidates:
        prediction = np.where(risk > threshold, locator, -1)
        f1 = processbench_metrics(prediction, labels)["f1"]
        if f1 > best[1]:
            best = (float(threshold), float(f1))
    return best


def evaluate_unified_model(
    model: UnifiedCausalIU,
    rows: Sequence[Mapping[str, Any]],
    *,
    localization_threshold: float | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all three tasks from the exact same risk trajectories."""

    finals = [model.score_row(row) for row in rows]
    wrong = np.asarray([final_wrong(row) for row in rows], dtype=int)
    terminal = np.asarray([item.global_score for item in finals], dtype=float)
    target_step = np.asarray([int(row.get("label", -1)) for row in rows], dtype=int)
    localized_step = np.asarray([
        token_to_step(item.localization_token, row) for item, row in zip(finals, rows)
    ])
    if localization_threshold is None:
        localization_threshold, _ = best_localization_threshold(terminal, localized_step, target_step)
    prediction = np.where(terminal > localization_threshold, localized_step, -1)
    localization = processbench_metrics(prediction, target_step)
    first_crossing_token = np.asarray([
        item.first_alarm_token_10pct
        if item.first_alarm_token_10pct is not None else item.localization_token
        for item in finals
    ])
    persistent_token = []
    for item in finals:
        flags = np.asarray([step.warning_10pct for step in item.trajectory], dtype=bool)
        located = None
        for index in range(2, len(flags)):
            if bool(flags[index - 2:index + 1].all()):
                located = index - 2
                break
        persistent_token.append(item.localization_token if located is None else located)
    first_crossing_step = np.asarray([
        token_to_step(token, row) for token, row in zip(first_crossing_token, rows)
    ])
    persistent_step = np.asarray([
        token_to_step(token, row) for token, row in zip(persistent_token, rows)
    ])
    localization_ablations = {
        "max_positive_contribution": localization,
        "first_crossing": processbench_metrics(
            np.where(terminal > localization_threshold, first_crossing_step, -1), target_step
        ),
        "persistent_crossing_3": processbench_metrics(
            np.where(terminal > localization_threshold, persistent_step, -1), target_step
        ),
    }

    early = {}
    for budget in budgets:
        scores = np.asarray([
            item.trajectory[min(int(budget), len(item.trajectory)) - 1].risk for item in finals
        ])
        early[str(int(budget))] = {
            "auroc": safe_auc(wrong, scores),
            "auprc": safe_ap(wrong, scores),
        }
    alarm_5 = np.asarray([item.first_alarm_token_5pct is not None for item in finals])
    alarm_10 = np.asarray([item.first_alarm_token_10pct is not None for item in finals])
    clean = wrong == 0
    warning = {
        "ever_warning_fpr_5pct": float(np.mean(alarm_5[clean])) if clean.any() else float("nan"),
        "ever_warning_fpr_10pct": float(np.mean(alarm_10[clean])) if clean.any() else float("nan"),
        "mean_first_alarm_5pct": float(np.mean([
            item.first_alarm_token_5pct for item, label in zip(finals, wrong)
            if label and item.first_alarm_token_5pct is not None
        ])) if any(label and item.first_alarm_token_5pct is not None for item, label in zip(finals, wrong)) else float("nan"),
    }
    summary = {
        "global": {"auroc": safe_auc(wrong, terminal), "auprc": safe_ap(wrong, terminal)},
        "localization": localization,
        "localization_ablations": localization_ablations,
        "early": early,
        "warning": warning,
        "localization_threshold": float(localization_threshold),
        "n": len(rows),
    }
    per_question = []
    for index, (row, item) in enumerate(zip(rows, finals)):
        per_question.append({
            "unit": _row_identity(row, index),
            "family": _row_family(row),
            "model": _row_model(row),
            "wrong": int(wrong[index]),
            "target_step": int(target_step[index]),
            "global_score": float(item.global_score),
            "localization_token": int(item.localization_token),
            "localization_step": int(localized_step[index]),
            "prediction": int(prediction[index]),
            "first_alarm_5pct": item.first_alarm_token_5pct,
            "first_alarm_10pct": item.first_alarm_token_10pct,
            **{
                f"risk_at_{int(budget)}": float(
                    item.trajectory[min(int(budget), len(item.trajectory)) - 1].risk
                ) for budget in budgets
            },
        })
    return summary, per_question


def choose_accumulator(
    candidate_metrics: Mapping[str, Mapping[str, float]],
    incumbent: Mapping[str, float],
    *,
    global_margin: float = 0.010,
    localization_margin: float = 0.010,
    early_margin: float = 0.015,
) -> tuple[str, list[dict[str, Any]]]:
    """Frozen no-regression filter followed by maximin and simplicity tie-break."""

    complexity = {"identity": 0, "cumulative_hazard": 2}
    ledger = []
    for name, values in candidate_metrics.items():
        deltas = {
            "global": float(values["global"] - incumbent["global"]),
            "localization": float(values["localization"] - incumbent["localization"]),
            "early": float(values["early"] - incumbent["early"]),
        }
        survives = bool(
            deltas["global"] >= -global_margin
            and deltas["localization"] >= -localization_margin
            and deltas["early"] >= -early_margin
        )
        ledger.append({
            "candidate": name,
            **{f"delta_{key}": value for key, value in deltas.items()},
            "worst_delta": min(deltas.values()),
            "survives": survives,
            "complexity": complexity.get(name, 1),
        })
    survivors = [row for row in ledger if row["survives"]]
    if not survivors:
        raise RuntimeError("no accumulator survives the preregistered regression margins")
    survivors.sort(key=lambda row: (-row["worst_delta"], row["complexity"], row["candidate"]))
    return str(survivors[0]["candidate"]), ledger


def finalist_gate(
    candidate: Mapping[str, float], incumbent: Mapping[str, float]
) -> dict[str, Any]:
    delta = {key: float(candidate[key] - incumbent[key]) for key in ("global", "localization", "early")}
    no_regression = delta["global"] >= -0.010 and delta["localization"] >= -0.010 and delta["early"] >= -0.015
    improves = max(delta.values()) >= 0.010
    return {"pass": bool(no_regression and improves), "no_regression": no_regression, "improves_one_by_0p010": improves, "deltas": delta}


def assert_group_split_isolation(splits: Sequence[tuple[Sequence[int], Sequence[int]]], groups: Sequence[str]) -> None:
    groups = np.asarray(groups, dtype=str)
    for fold, (train, test) in enumerate(splits):
        overlap = set(groups[np.asarray(train, dtype=int)]) & set(groups[np.asarray(test, dtype=int)])
        if overlap:
            raise AssertionError(f"fold {fold} leaks source questions: {sorted(overlap)[:3]}")


__all__ = [
    "AtlasSamples",
    "DEFAULT_BUDGETS",
    "PRIMARY_EARLY_BUDGETS",
    "assert_group_split_isolation",
    "benjamini_hochberg",
    "best_localization_threshold",
    "build_atlas_samples",
    "build_information_atlas",
    "build_group_synergy_atlas",
    "choose_accumulator",
    "derive_supervised_signs",
    "evaluate_unified_model",
    "final_wrong",
    "finalist_gate",
    "feature_groups",
    "first_error_mask",
    "grouped_block_permutation",
    "grouped_mutual_information",
    "heldout_logloss_gain",
    "processbench_metrics",
    "redundancy_prune",
    "safe_ap",
    "safe_auc",
    "select_atlas_roster",
    "source_group",
    "token_to_step",
]
