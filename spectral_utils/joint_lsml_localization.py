"""Frozen active-23 adapters for retrospective Joint L-SML localization.

This module has no target/label API. It consumes raw token telemetry already
joined to opaque response IDs and produces label-free token-risk curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .fusion_utils import lsml_continuous
from .joint_lsml import (
    continuous_lsml_weight_vector,
    covariance_matrix,
    discover_loao_consensus_groups,
    fit_joint_lsml,
    hard_lsml_misfit,
    hierarchical_joint_weights,
    weight_maps,
)
from .reconstruction_benchmark.localization_contract import FIT_TOKEN_CAP, payload_sha256
from .specrage_views import FEATURE_TO_VIEW, VIEW_ORDER
from .token_local_fusion import IU_CONFIG
from .upcr import upcr_fit


JOINT_METHOD = "joint_lsml23_hierarchical_v1_1"
IU_METHOD = "iu_pcr_active23"
EQUAL_FAMILY_METHOD = "equal_family_active23"
FIXED_FAMILY_METHOD = "fixed_family_continuous_lsml_active23"
METHODS = (JOINT_METHOD, IU_METHOD, EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD)
K_RANGE = (3, 4, 6, 8)
PAIRWISE_DIAGNOSTIC_CAP = 32768
MINIMUM_HELD_ADMISSIBLE_FRACTION = 0.95
MINIMUM_WEIGHT_MAP_SCORE_SPEARMAN = 0.50
APPLICATION_CHUNK = 100_000


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class Active23Preparation:
    raw: np.ndarray
    token_offsets: np.ndarray
    row_ids: tuple[str, ...]
    retained_indices: np.ndarray
    signs: np.ndarray
    feature_names: tuple[str, ...]
    family_names: tuple[str, ...]
    fit_indices: np.ndarray
    fit_row_indices: np.ndarray
    medians: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    standardized_fit: np.ndarray
    diagnostics: Mapping[str, Any]

    def standardized_slice(self, lo: int, hi: int) -> np.ndarray:
        values = np.asarray(self.raw[int(lo):int(hi), self.retained_indices], dtype=np.float64)
        values = values * self.signs[None, :]
        clean = np.where(np.isfinite(values), values, self.mean[None, :])
        return (clean - self.mean[None, :]) / self.std[None, :]

    def token_risk(self, weight: Sequence[float]) -> np.ndarray:
        current = np.asarray(weight, dtype=np.float64)
        if current.shape != (len(self.feature_names),) or not np.isfinite(current).all():
            raise ValueError("active-23 weight is malformed")
        output = np.empty(len(self.raw), dtype=np.float64)
        for lo in range(0, len(output), APPLICATION_CHUNK):
            hi = min(lo + APPLICATION_CHUNK, len(output))
            output[lo:hi] = -(self.standardized_slice(lo, hi) @ current)
        if not np.isfinite(output).all():
            raise RuntimeError("fusion produced non-finite token risk")
        return output


def prepare_active23(
    raw: np.ndarray,
    token_offsets: Sequence[int],
    row_ids: Sequence[str],
    *,
    retained_indices: Sequence[int],
    confidence_signs_29: Sequence[int],
    stream_names_29: Sequence[str],
    raw_feature_names_29: Sequence[str],
    fit_token_cap: int = FIT_TOKEN_CAP,
) -> Active23Preparation:
    values = np.asarray(raw, dtype=np.float64)
    offsets = np.asarray(token_offsets, dtype=np.int64)
    rows = tuple(map(str, row_ids))
    retained = np.asarray(retained_indices, dtype=np.int64)
    all_signs = np.asarray(confidence_signs_29, dtype=np.int64)
    stream_names = tuple(map(str, stream_names_29))
    raw_names = tuple(map(str, raw_feature_names_29))
    if values.ndim != 2 or values.shape[1] != 29 or len(values) == 0:
        raise ValueError("raw token telemetry must be nonempty tokens-by-29")
    if offsets.shape != (len(rows) + 1,) or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("token offsets do not bind row IDs")
    if np.any(np.diff(offsets) <= 0) or len(set(rows)) != len(rows):
        raise ValueError("rows must be unique and nonempty")
    if retained.shape != (23,) or len(set(retained.tolist())) != 23:
        raise ValueError("the frozen roster must contain exactly 23 distinct indices")
    if np.any(retained < 0) or np.any(retained >= 29):
        raise ValueError("retained index outside 29-stream schema")
    if all_signs.shape != (29,) or not np.isin(all_signs, (-1, 1)).all():
        raise ValueError("absolute orientation must contain 29 +/-1 signs")
    if len(stream_names) != 29 or len(raw_names) != 29:
        raise ValueError("feature schema must contain 29 ordered names")
    selected = values[:, retained] * all_signs[retained][None, :]
    if len(values) > int(fit_token_cap):
        fit_indices = np.linspace(0, len(values) - 1, int(fit_token_cap), dtype=np.int64)
    else:
        fit_indices = np.arange(len(values), dtype=np.int64)
    fit = selected[fit_indices]
    medians = np.nanmedian(fit, axis=0)
    clean = np.where(np.isfinite(fit), fit, medians[None, :])
    scale = clean.std(axis=0)
    if not np.isfinite(medians).all() or np.any(~np.isfinite(scale)) or np.any(scale <= 1e-8):
        raise RuntimeError("a frozen active-23 stream is degenerate on this cell")
    mean = clean.mean(axis=0)
    std = clean.std(axis=0)
    standardized = (clean - mean[None, :]) / std[None, :]
    owners = np.searchsorted(offsets[1:], fit_indices, side="right").astype(np.int64)
    active_raw_names = tuple(raw_names[index] for index in retained)
    families = tuple(FEATURE_TO_VIEW[name] for name in active_raw_names)
    active_streams = tuple(stream_names[index] for index in retained)
    family_counts = {family: int(families.count(family)) for family in VIEW_ORDER if family in families}
    diagnostics = {
        "schema": "joint-lsml-active23-preparation-v1",
        "n_rows": len(rows),
        "n_tokens": int(len(values)),
        "n_fit_tokens": int(len(fit_indices)),
        "fit_token_cap": int(fit_token_cap),
        "fit_indices_sha256": payload_sha256(fit_indices.tolist()),
        "retained_indices": retained.tolist(),
        "feature_names": list(active_streams),
        "family_names": list(families),
        "family_counts": family_counts,
        "orientation": "absolute raw-domain signs before population z-score",
        "labels_accessed": False,
    }
    diagnostics["payload_sha256"] = payload_sha256(diagnostics)
    return Active23Preparation(
        raw=_readonly(values), token_offsets=_readonly(offsets), row_ids=rows,
        retained_indices=_readonly(retained), signs=_readonly(all_signs[retained]),
        feature_names=active_streams, family_names=families,
        fit_indices=_readonly(fit_indices), fit_row_indices=_readonly(owners),
        medians=_readonly(medians), mean=_readonly(mean), std=_readonly(std),
        standardized_fit=_readonly(standardized),
        diagnostics=MappingProxyType(diagnostics),
    )


def _orient_weight(values: np.ndarray, weight: Sequence[float]) -> tuple[np.ndarray, float, bool]:
    matrix = np.asarray(values, dtype=np.float64)
    output = np.asarray(weight, dtype=np.float64).copy()
    anchor = matrix.mean(axis=1)
    score = matrix @ output
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    flipped = bool(np.isfinite(correlation) and correlation < 0.0)
    if flipped:
        output *= -1.0
    return output, correlation, flipped


def _fixed_family_labels(families: Sequence[str]) -> np.ndarray:
    order = [name for name in VIEW_ORDER if name in families]
    mapping = {name: index for index, name in enumerate(order)}
    return np.asarray([mapping[str(name)] for name in families], dtype=np.int64)


def fit_active23_arms(preparation: Active23Preparation, *, seed: int) -> dict[str, Any]:
    values = np.asarray(preparation.standardized_fit, dtype=np.float64)
    anchor_index = preparation.feature_names.index("entropy_series")
    grouping = discover_loao_consensus_groups(
        values, preparation.fit_row_indices, k_range=K_RANGE, seed=int(seed),
        minimum_group_size=3, pairwise_diagnostic_cap=PAIRWISE_DIAGNOSTIC_CAP,
        minimum_held_admissible_fraction=MINIMUM_HELD_ADMISSIBLE_FRACTION,
        use_minimum_ari_tiebreak=True,
    )
    if grouping["status"] != "SELECTED":
        return {
            "status": "BLOCKED_NO_ADMISSIBLE_PARTITION",
            "grouping": grouping,
            "preparation": dict(preparation.diagnostics),
            "preprocessing_parameters": {
                "medians": preparation.medians,
                "mean": preparation.mean,
                "std": preparation.std,
            },
            "labels_accessed": False,
        }
    labels = np.asarray(grouping["labels"], dtype=np.int64)
    covariance = covariance_matrix(values)
    joint = fit_joint_lsml(covariance, labels, anchor_index=anchor_index, seed=int(seed) + 10_000)
    maps = weight_maps(values, covariance, labels, joint, anchor_index=anchor_index, target_condition=1e3)
    joint_weight = np.asarray(maps["weights"]["hierarchical_joint"], dtype=np.float64)
    joint_meta = maps["diagnostics"]["hierarchical_joint"]
    minimum_map_agreement = float(min(maps["pairwise_score_spearman"].values()))
    iu_fit = upcr_fit(values.T, **dict(IU_CONFIG))
    iu_weight, iu_corr, iu_flip = _orient_weight(values, iu_fit.w)
    present = [name for name in VIEW_ORDER if name in preparation.family_names]
    equal_weight = np.zeros(values.shape[1], dtype=np.float64)
    for family in present:
        indices = np.flatnonzero(np.asarray(preparation.family_names) == family)
        equal_weight[indices] = 1.0 / (len(present) * len(indices))
    equal_weight, equal_corr, equal_flip = _orient_weight(values, equal_weight)
    fixed_labels = _fixed_family_labels(preparation.family_names)
    fixed_score, fixed_meta = lsml_continuous(
        *[values[:, index] for index in range(values.shape[1])],
        groups=fixed_labels, compute_score_matrix=False,
    )
    fixed_weight = continuous_lsml_weight_vector(fixed_meta, values.shape[1])
    fixed_weight, fixed_corr, fixed_flip = _orient_weight(values, fixed_weight)
    expected_fixed = values @ fixed_weight
    signed_reference = np.asarray(fixed_score) * (-1.0 if fixed_flip else 1.0)
    if not np.allclose(expected_fixed, signed_reference, atol=1e-10, rtol=1e-10):
        raise RuntimeError("fixed-family continuous L-SML reconstruction drift")
    weights = {
        JOINT_METHOD: np.asarray(joint_weight, dtype=np.float64),
        IU_METHOD: iu_weight,
        EQUAL_FAMILY_METHOD: equal_weight,
        FIXED_FAMILY_METHOD: fixed_weight,
    }
    finite = bool(all(np.isfinite(weight).all() for weight in weights.values()))
    hard = hard_lsml_misfit(covariance, labels)
    structural_pass = bool(
        joint.converged and joint.multistart_audit["status"] == "PASS"
        and joint.jacobian_audit["full_global_rank"] and finite
        and len(joint.starts) == 5
        and all(
            not row.failed_monotonicity
            and np.isfinite(row.objective_trace).all()
            and np.isfinite(row.model_change_trace).all()
            and np.all(np.diff(row.objective_trace) <= 1e-12)
            for row in joint.starts
        )
        and minimum_map_agreement >= MINIMUM_WEIGHT_MAP_SCORE_SPEARMAN
    )
    return {
        "status": "FIT_COMPLETE" if structural_pass else "BLOCKED_STRUCTURAL_FIT",
        "structural_fit_pass": structural_pass,
        "preparation": dict(preparation.diagnostics),
        "preprocessing_parameters": {
            "medians": preparation.medians,
            "mean": preparation.mean,
            "std": preparation.std,
        },
        "grouping": grouping,
        "covariance": covariance,
        "joint_fit": joint,
        "hard_lsml_relative_offdiag_misfit": float(hard["relative_offdiag_misfit"]),
        "joint_lower_misfit": bool(joint.relative_offdiag_misfit < hard["relative_offdiag_misfit"]),
        "weight_map_agreement": {
            "pairwise_score_spearman": maps["pairwise_score_spearman"],
            "minimum": minimum_map_agreement,
            "minimum_required": MINIMUM_WEIGHT_MAP_SCORE_SPEARMAN,
            "status": "PASS" if minimum_map_agreement >= MINIMUM_WEIGHT_MAP_SCORE_SPEARMAN else "BLOCKED",
            "maps_are_diagnostics_only": True,
        },
        "weights": weights,
        "diagnostics": {
            JOINT_METHOD: joint_meta,
            IU_METHOD: {
                "g2_hat": float(iu_fit.g2_hat),
                "projection_residual": float(iu_fit.proj_residual),
                "anchor_correlation": iu_corr,
                "orientation_flipped": iu_flip,
            },
            EQUAL_FAMILY_METHOD: {
                "present_families": present,
                "anchor_correlation": equal_corr,
                "orientation_flipped": equal_flip,
            },
            FIXED_FAMILY_METHOD: {
                "K": int(fixed_meta["K"]),
                "anchor_correlation": fixed_corr,
                "orientation_flipped": fixed_flip,
            },
        },
        "labels_accessed": False,
    }


def score_active23_arms(preparation: Active23Preparation, fitted: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if fitted.get("status") != "FIT_COMPLETE" or not fitted.get("structural_fit_pass"):
        raise RuntimeError("cannot score a structurally blocked cell")
    return {
        method: preparation.token_risk(weight)
        for method, weight in fitted["weights"].items()
    }


__all__ = [
    "Active23Preparation", "EQUAL_FAMILY_METHOD", "FIXED_FAMILY_METHOD",
    "IU_METHOD", "JOINT_METHOD", "K_RANGE", "METHODS", "MINIMUM_HELD_ADMISSIBLE_FRACTION",
    "MINIMUM_WEIGHT_MAP_SCORE_SPEARMAN", "PAIRWISE_DIAGNOSTIC_CAP", "fit_active23_arms",
    "prepare_active23", "score_active23_arms",
]
