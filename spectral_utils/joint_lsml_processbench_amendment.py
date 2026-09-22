"""Target-free structural fallback for the ProcessBench Joint L-SML amendment."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .fusion_utils import lsml_continuous
from .joint_lsml import continuous_lsml_weight_vector, dispatch_alias
from .joint_lsml_localization import (
    EQUAL_FAMILY_METHOD,
    FIXED_FAMILY_METHOD,
    IU_CONFIG,
    IU_METHOD,
    JOINT_METHOD,
    Active23Preparation,
)
from .upcr import upcr_fit
from .specrage_views import VIEW_ORDER


COVERAGE_METHOD = "joint_lsml23_hierarchical_v1_1__flat_sml_structural_fallback"
COVERAGE_METHODS = (
    COVERAGE_METHOD,
    IU_METHOD,
    EQUAL_FAMILY_METHOD,
    FIXED_FAMILY_METHOD,
)


def _orient_to_mean_confidence(values: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float64)
    output = np.asarray(weight, dtype=np.float64).copy()
    score = matrix @ output
    anchor = matrix.mean(axis=1)
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    flipped = bool(np.isfinite(correlation) and correlation < 0.0)
    if flipped:
        output *= -1.0
    return output, {"anchor_correlation": correlation, "orientation_flipped": flipped}


def _fixed_family_labels(families: tuple[str, ...]) -> np.ndarray:
    order = [name for name in VIEW_ORDER if name in families]
    mapping = {name: index for index, name in enumerate(order)}
    return np.asarray([mapping[name] for name in families], dtype=np.int64)


def fit_flat_fallback_and_controls(preparation: Active23Preparation) -> dict[str, Any]:
    """Fit the blocked-cell fallback and the three registered controls without targets."""

    values = np.asarray(preparation.standardized_fit, dtype=np.float64)
    _, flat_weight, flat_meta = dispatch_alias(
        values, np.zeros(values.shape[1], dtype=np.int64), mode="flat_sml"
    )
    if flat_weight is None or not bool(flat_meta.get("bit_exact_alias")):
        raise RuntimeError("flat-SML alias did not dispatch bit-exactly")
    flat_weight, flat_orientation = _orient_to_mean_confidence(values, flat_weight)

    iu = upcr_fit(values.T, **dict(IU_CONFIG))
    iu_weight, iu_orientation = _orient_to_mean_confidence(values, np.asarray(iu.w))

    present = [name for name in VIEW_ORDER if name in preparation.family_names]
    equal_weight = np.zeros(values.shape[1], dtype=np.float64)
    for family in present:
        indices = np.flatnonzero(np.asarray(preparation.family_names) == family)
        equal_weight[indices] = 1.0 / (len(present) * len(indices))
    equal_weight, equal_orientation = _orient_to_mean_confidence(values, equal_weight)

    labels = _fixed_family_labels(preparation.family_names)
    fixed_score, fixed_meta = lsml_continuous(
        *[values[:, index] for index in range(values.shape[1])],
        groups=labels,
        compute_score_matrix=False,
    )
    fixed_weight = continuous_lsml_weight_vector(fixed_meta, values.shape[1])
    fixed_weight, fixed_orientation = _orient_to_mean_confidence(values, fixed_weight)
    signed_reference = np.asarray(fixed_score) * (
        -1.0 if fixed_orientation["orientation_flipped"] else 1.0
    )
    if not np.allclose(values @ fixed_weight, signed_reference, atol=1e-10, rtol=1e-10):
        raise RuntimeError("fixed-family L-SML reconstruction drift")

    weights = {
        JOINT_METHOD: flat_weight,
        IU_METHOD: iu_weight,
        EQUAL_FAMILY_METHOD: equal_weight,
        FIXED_FAMILY_METHOD: fixed_weight,
    }
    if not all(np.isfinite(weight).all() for weight in weights.values()):
        raise RuntimeError("fallback policy produced a non-finite weight")
    return {
        "weights": weights,
        "candidate_component": "G_empty_exact_flat_sml_alias",
        "diagnostics": {
            "flat_alias": {**flat_meta, **flat_orientation},
            "iu_pcr": {
                "g2_hat": float(iu.g2_hat),
                "projection_residual": float(iu.proj_residual),
                **iu_orientation,
            },
            "equal_family": {"present_families": present, **equal_orientation},
            "fixed_family": {"K": int(fixed_meta["K"]), **fixed_orientation},
        },
    }


def rename_candidate_method(frozen: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {key: np.asarray(value) for key, value in frozen.items()}
    methods = list(output["method_ids"].astype(str))
    if not methods or methods[0] != JOINT_METHOD:
        raise RuntimeError("parent candidate column is not first")
    methods[0] = COVERAGE_METHOD
    output["method_ids"] = np.asarray(methods, dtype="<U80")
    if tuple(methods) != COVERAGE_METHODS:
        raise RuntimeError("coverage method roster drift")
    return output


__all__ = [
    "COVERAGE_METHOD",
    "COVERAGE_METHODS",
    "fit_flat_fallback_and_controls",
    "rename_candidate_method",
]
