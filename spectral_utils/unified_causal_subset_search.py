"""Structured subset search helpers for the Unified Causal IU-PCR bank.

The 1,036 causal coordinates are 37 base streams crossed with 28 DSP
transforms.  Treating them as 1,036 unrelated switches is both statistically
fragile and computationally wasteful.  This module defines reproducible,
interpretable roster families and the small amount of supervised development
logic used by the local subset-search runner.

Nothing in this module is a confirmation procedure.  Labels are deliberately
used to derive signs, relevance priors, and the maximin development ranking.
Every such quantity must therefore be learned inside a development split and
frozen before a robustness panel is scored.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from .multitask_trajectory import equal_positions
from .unified_causal_evaluation import AtlasSamples
from .unified_causal_iu import (
    BASE_NAMES,
    TRANSFORM_NAMES,
    UnifiedCausalIU,
    _robust_location_scale,
    all_feature_names,
    feature_name,
)


EPS = 1e-12
TASKS = ("global", "localization", "early")
REGRESSION_MARGINS = {
    "global": 0.010,
    "localization": 0.010,
    "early": 0.015,
}


RAW9_BASES = tuple(name for name in BASE_NAMES if name.startswith("raw::"))
BROAD28_BASES = tuple(name for name in BASE_NAMES if name.startswith("broad::"))
CORE5_BASES = (
    "broad::entropy_series",
    "broad::entropy_sw_var_series",
    "broad::entropy_cusum_abs_series",
    "broad::spilled_sw_var_series",
    "broad::spilled_cusum_abs_series",
)
ENTROPY_DYNAMIC_BASES = (
    "broad::entropy_sw_var_series",
    "broad::entropy_cusum_abs_series",
    "broad::entropy_rolling_tail_ratio",
)
SPILLED_DYNAMIC_BASES = (
    "broad::spilled_sw_var_series",
    "broad::spilled_cusum_abs_series",
    "broad::spilled_rolling_min",
)
PARTITION_DYNAMIC_BASES = (
    "broad::energy_rolling_min",
    "broad::energy_sw_var_series",
    "broad::energy_cusum_abs_series",
)
JOINT15_BASES = tuple(dict.fromkeys(
    RAW9_BASES + ENTROPY_DYNAMIC_BASES + SPILLED_DYNAMIC_BASES
))
JOINT18_BASES = tuple(dict.fromkeys(JOINT15_BASES + PARTITION_DYNAMIC_BASES))


TRANSFORM_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "level": ("level",),
    "fastslow": ("ewma8", "ewma32", "fastminus8_32"),
    "sustained": ("ewma16", "positive_area", "persistence"),
    "multiscale_sustained": (
        "level",
        "ewma4",
        "ewma8",
        "ewma16",
        "ewma32",
        "ewma64",
        "fastminus4_16",
        "fastminus8_32",
        "fastminus16_64",
        "positive_area",
        "persistence",
    ),
    "window_moments": tuple(
        f"{operator}{window}"
        for window in (8, 16, 32, 64)
        for operator in ("mean", "var", "mad")
    ),
    "change": (
        "innovation64",
        "cusum_pos_k025",
        "page_hinkley_pos_d005",
        "bocpd_h50",
        "bocpd_h100",
    ),
    "no_bocpd": tuple(name for name in TRANSFORM_NAMES if not name.startswith("bocpd_")),
    "full": tuple(TRANSFORM_NAMES),
}


def cross_roster(
    bases: Sequence[str], transforms: Sequence[str]
) -> tuple[str, ...]:
    """Return one canonical base-major slice of the causal feature bank."""

    bases = tuple(bases)
    transforms = tuple(transforms)
    unknown_bases = set(bases) - set(BASE_NAMES)
    unknown_transforms = set(transforms) - set(TRANSFORM_NAMES)
    if unknown_bases or unknown_transforms:
        raise ValueError(
            f"unknown roster coordinates: bases={sorted(unknown_bases)}, "
            f"transforms={sorted(unknown_transforms)}"
        )
    wanted = {
        feature_name(base, transform)
        for base in bases
        for transform in transforms
    }
    return tuple(name for name in all_feature_names() if name in wanted)


def structured_rosters(stage: str = "a") -> dict[str, tuple[str, ...]]:
    """Return the evidence-grounded structured discovery rosters.

    ``family6`` is intentionally absent: the historical arm averages streams
    before IU-PCR and is not a literal slice of this 37x28 bank.
    """

    stage = str(stage).lower()
    if stage == "a":
        return {
            "core5_level": cross_roster(CORE5_BASES, TRANSFORM_FAMILIES["level"]),
            "raw9_level": cross_roster(RAW9_BASES, TRANSFORM_FAMILIES["level"]),
            "raw9_fastslow": cross_roster(RAW9_BASES, TRANSFORM_FAMILIES["fastslow"]),
            "raw9_sustained": cross_roster(RAW9_BASES, TRANSFORM_FAMILIES["sustained"]),
            "all37_level": cross_roster(BASE_NAMES, TRANSFORM_FAMILIES["level"]),
            "all37_multiscale_sustained": cross_roster(
                BASE_NAMES, TRANSFORM_FAMILIES["multiscale_sustained"]
            ),
            "all37_window_moments": cross_roster(
                BASE_NAMES, TRANSFORM_FAMILIES["window_moments"]
            ),
            "all37_change": cross_roster(BASE_NAMES, TRANSFORM_FAMILIES["change"]),
            "all37_no_bocpd": cross_roster(
                BASE_NAMES, TRANSFORM_FAMILIES["no_bocpd"]
            ),
            "all37_full": cross_roster(BASE_NAMES, TRANSFORM_FAMILIES["full"]),
        }
    if stage == "b":
        # Stage B freezes the winning transform family in the runner and uses
        # these memberships only.  Level is the deterministic standalone form.
        return {
            "core5_level": cross_roster(CORE5_BASES, ("level",)),
            "raw9_level": cross_roster(RAW9_BASES, ("level",)),
            "joint15_level": cross_roster(JOINT15_BASES, ("level",)),
            "joint18_level": cross_roster(JOINT18_BASES, ("level",)),
            "broad28_level": cross_roster(BROAD28_BASES, ("level",)),
            "all37_level": cross_roster(BASE_NAMES, ("level",)),
        }
    if stage == "c":
        level_raw9 = cross_roster(RAW9_BASES, ("level",))
        level_joint18 = cross_roster(JOINT18_BASES, ("level",))
        level_broad28 = cross_roster(BROAD28_BASES, ("level",))
        fastslow_raw9 = cross_roster(RAW9_BASES, TRANSFORM_FAMILIES["fastslow"])
        sustained_raw9 = cross_roster(RAW9_BASES, TRANSFORM_FAMILIES["sustained"])

        def union(*parts: Sequence[str]) -> tuple[str, ...]:
            wanted = set().union(*map(set, parts))
            return tuple(name for name in all_feature_names() if name in wanted)

        return {
            "raw9_level": level_raw9,
            "joint18_level": level_joint18,
            "broad28_level": level_broad28,
            "raw9_fastslow": fastslow_raw9,
            "raw9_level_fastslow": union(level_raw9, fastslow_raw9),
            "joint18_level_plus_raw9_fastslow": union(level_joint18, fastslow_raw9),
            "broad28_level_plus_raw9_fastslow": union(level_broad28, fastslow_raw9),
            "raw9_level_sustained": union(level_raw9, sustained_raw9),
            "joint18_level_plus_raw9_sustained": union(level_joint18, sustained_raw9),
            "broad28_level_plus_raw9_sustained": union(level_broad28, sustained_raw9),
        }
    if stage == "d":
        transforms = ("level", "ewma16", "positive_area", "persistence")
        output = {
            "raw9_level": cross_roster(RAW9_BASES, ("level",)),
            "raw9_level_ewma16": cross_roster(RAW9_BASES, ("level", "ewma16")),
            "raw9_level_area": cross_roster(RAW9_BASES, ("level", "positive_area")),
            "raw9_level_persistence": cross_roster(RAW9_BASES, ("level", "persistence")),
            "raw9_level_ewma16_area": cross_roster(
                RAW9_BASES, ("level", "ewma16", "positive_area")
            ),
            "raw9_level_ewma16_persistence": cross_roster(
                RAW9_BASES, ("level", "ewma16", "persistence")
            ),
            "raw9_level_area_persistence": cross_roster(
                RAW9_BASES, ("level", "positive_area", "persistence")
            ),
            "raw9_sustained": cross_roster(
                RAW9_BASES, ("ewma16", "positive_area", "persistence")
            ),
            "raw9_level_sustained": cross_roster(RAW9_BASES, transforms),
        }
        for dropped in RAW9_BASES:
            suffix = dropped.removeprefix("raw::")
            output[f"winner_drop_{suffix}"] = cross_roster(
                tuple(base for base in RAW9_BASES if base != dropped), transforms
            )
        return output
    if stage == "e":
        all_transforms = ("level", "ewma16", "positive_area", "persistence")
        no_area = ("level", "ewma16", "persistence")
        without = lambda *removed: tuple(
            base for base in RAW9_BASES if base not in set(removed)
        )
        spilled = "raw::spilled"
        margin = "raw::neg_margin"
        entropy = "raw::entropy"
        logsumexp = "raw::neg_logsumexp"
        top1 = "raw::neg_top1"
        base7 = without(spilled, margin)
        return {
            "raw9_full36": cross_roster(RAW9_BASES, all_transforms),
            "drop_margin_full32": cross_roster(without(margin), all_transforms),
            "drop_spilled_full32": cross_roster(without(spilled), all_transforms),
            "base7_full28": cross_roster(base7, all_transforms),
            "raw9_no_area27": cross_roster(RAW9_BASES, no_area),
            "drop_margin_no_area24": cross_roster(without(margin), no_area),
            "drop_spilled_no_area24": cross_roster(without(spilled), no_area),
            "base7_no_area21": cross_roster(base7, no_area),
            "base6_no_entropy18": cross_roster(without(spilled, margin, entropy), no_area),
            "base6_no_logsumexp18": cross_roster(
                without(spilled, margin, logsumexp), no_area
            ),
            "base6_no_top1_18": cross_roster(without(spilled, margin, top1), no_area),
            "base5_no_entropy_top1_15": cross_roster(
                without(spilled, margin, entropy, top1), no_area
            ),
        }
    raise ValueError("stage must be 'a', 'b', 'c', 'd', or 'e'")


def base_mask_rosters(transforms: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Cross one frozen transform family with the Stage-B base masks."""

    return {
        "core5": cross_roster(CORE5_BASES, transforms),
        "raw9": cross_roster(RAW9_BASES, transforms),
        "joint15": cross_roster(JOINT15_BASES, transforms),
        "joint18": cross_roster(JOINT18_BASES, transforms),
        "broad28": cross_roster(BROAD28_BASES, transforms),
        "all37": cross_roster(BASE_NAMES, transforms),
    }


def supervised_relevance(
    sample_sets: Sequence[AtlasSamples], roster: Sequence[str]
) -> np.ndarray:
    """Return target- and family-balanced nonlinear-agnostic effect magnitudes.

    The score is the median class shift divided by MAD, clipped per
    target/family and then averaged with equal target and family weight.  It is
    deliberately simple: this is a coordinate prior for a sensitivity arm,
    not a second supervised classifier.
    """

    roster = tuple(roster)
    if not roster:
        raise ValueError("relevance requires a non-empty roster")
    per_target = []
    for samples in sample_sets:
        lookup = {name: index for index, name in enumerate(samples.feature_names)}
        if not set(roster) <= set(lookup):
            raise ValueError("relevance roster is absent from Atlas samples")
        columns = np.asarray([lookup[name] for name in roster], dtype=int)
        per_family = []
        for family in sorted(set(samples.families)):
            mask = np.asarray(samples.families == family)
            y = np.asarray(samples.y[mask], dtype=int)
            if len(np.unique(y)) < 2:
                continue
            X = np.asarray(samples.X[mask][:, columns], dtype=float)
            medians = np.nanmedian(X, axis=0)
            mad = np.nanmedian(np.abs(X - medians[None, :]), axis=0)
            shift = np.abs(
                np.nanmedian(X[y == 1], axis=0)
                - np.nanmedian(X[y == 0], axis=0)
            )
            effect = np.divide(
                shift,
                mad + EPS,
                out=np.zeros_like(shift),
                where=np.isfinite(shift) & np.isfinite(mad),
            )
            per_family.append(np.clip(effect, 0.0, 5.0))
        if per_family:
            per_target.append(np.nanmean(np.vstack(per_family), axis=0))
    if not per_target:
        return np.ones(len(roster), dtype=float)
    relevance = np.nanmean(np.vstack(per_target), axis=0)
    relevance = np.where(np.isfinite(relevance), relevance, 0.0)
    positive = relevance[relevance > EPS]
    normalizer = float(np.median(positive)) if len(positive) else 1.0
    return np.clip(relevance / max(normalizer, EPS), 0.25, 4.0)


def blended_multipliers(relevance: Sequence[float], alpha: float) -> np.ndarray:
    """Blend the ordinary all-ones IU weights with a supervised prior."""

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    relevance = np.asarray(relevance, dtype=float).reshape(-1)
    if not len(relevance) or not np.isfinite(relevance).all() or np.any(relevance <= 0.0):
        raise ValueError("relevance must be finite and strictly positive")
    return (1.0 - alpha) + alpha * relevance


def reweight_model(
    model: UnifiedCausalIU,
    multipliers: Sequence[float],
    rows: Sequence[Mapping[str, Any]],
    feature_matrices: Sequence[np.ndarray],
    *,
    positions_per_trace: int = 32,
    alpha: float | None = None,
) -> UnifiedCausalIU:
    """Apply a coordinate prior to IU weights and recalibrate evidence on fit rows."""

    multipliers = np.asarray(multipliers, dtype=float).reshape(-1)
    if multipliers.shape != model.weights.shape:
        raise ValueError("weight multipliers do not match retained IU coordinates")
    if len(rows) != len(feature_matrices):
        raise ValueError("fit matrices do not match rows")
    if not np.isfinite(multipliers).all() or np.any(multipliers <= 0.0):
        raise ValueError("weight multipliers must be finite and positive")
    weights = np.asarray(model.weights, dtype=float) * multipliers
    samples = []
    for matrix in feature_matrices:
        matrix = np.asarray(matrix, dtype=float)
        positions = equal_positions(len(matrix), positions_per_trace)
        selected = matrix[positions][:, model.feature_indices]
        clean = np.where(np.isfinite(selected), selected, model.feature_medians)
        oriented = (
            (clean - model.feature_centres) / model.feature_scales
        ) * model.feature_signs
        samples.append(oriented @ weights)
    evidence = np.concatenate(samples)
    centre, scale = _robust_location_scale(evidence)
    diagnostics = {
        **dict(model.diagnostics),
        "claim_label": "supervised-weighted, IU-PCR-fused, causal streaming",
        "supervised_coordinate_reweighting": True,
        "weight_prior_alpha": None if alpha is None else float(alpha),
        "weight_multiplier_min": float(np.min(multipliers)),
        "weight_multiplier_median": float(np.median(multipliers)),
        "weight_multiplier_max": float(np.max(multipliers)),
    }
    return replace(
        model,
        weights=weights,
        evidence_centre=centre,
        evidence_scale=scale,
        diagnostics=diagnostics,
    )


def pareto_front(records: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return candidates nondominated in G/L/E and feature count."""

    records = list(records)
    front: set[str] = set()
    for candidate in records:
        dominated = False
        for other in records:
            if other is candidate:
                continue
            no_worse = all(
                float(other[task]) >= float(candidate[task])
                for task in TASKS
            ) and int(other["n_features"]) <= int(candidate["n_features"])
            strictly_better = any(
                float(other[task]) > float(candidate[task])
                for task in TASKS
            ) or int(other["n_features"]) < int(candidate["n_features"])
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(str(candidate["candidate"]))
    return front


def rank_against_control(
    records: Sequence[Mapping[str, Any]], control: str
) -> list[dict[str, Any]]:
    """Rank aggregate candidates by normalized maximin noninferiority."""

    records = [dict(record) for record in records]
    lookup = {str(record["candidate"]): record for record in records}
    if control not in lookup:
        raise KeyError(f"control candidate {control!r} is absent")
    reference = lookup[control]
    front = pareto_front(records)
    oracle = {
        task: max(float(record[task]) for record in records)
        for task in TASKS
    }
    output = []
    for record in records:
        deltas = {
            task: float(record[task]) - float(reference[task])
            for task in TASKS
        }
        normalized = {
            task: deltas[task] / REGRESSION_MARGINS[task]
            for task in TASKS
        }
        regret = {
            task: (oracle[task] - float(record[task])) / REGRESSION_MARGINS[task]
            for task in TASKS
        }
        survives = all(
            deltas[task] >= -REGRESSION_MARGINS[task]
            for task in TASKS
        )
        output.append({
            **record,
            **{f"delta_{task}": deltas[task] for task in TASKS},
            "survives_noninferiority": bool(survives),
            "maximin_normalized": float(min(normalized.values())),
            "mean_normalized": float(np.mean(list(normalized.values()))),
            "max_oracle_regret": float(max(regret.values())),
            "mean_oracle_regret": float(np.mean(list(regret.values()))),
            "pareto": str(record["candidate"]) in front,
        })
    output.sort(key=lambda row: (
        not bool(row["survives_noninferiority"]),
        not bool(row["pareto"]),
        -float(row["maximin_normalized"]),
        -float(row["mean_normalized"]),
        int(row["n_features"]),
        str(row["candidate"]),
    ))
    return output


__all__ = [
    "BROAD28_BASES",
    "CORE5_BASES",
    "JOINT15_BASES",
    "JOINT18_BASES",
    "RAW9_BASES",
    "REGRESSION_MARGINS",
    "TASKS",
    "TRANSFORM_FAMILIES",
    "base_mask_rosters",
    "blended_multipliers",
    "cross_roster",
    "pareto_front",
    "rank_against_control",
    "reweight_model",
    "structured_rosters",
    "supervised_relevance",
]
