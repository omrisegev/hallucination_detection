"""Strict 30-atom feature admission and target-local coordinates for A6.

The mixed-v2 transformer is safe only after explicit presence and degeneracy
checks: its generic implementation intentionally median-imputes and would turn
an all-missing column into non-finite state.  This module is the sole A6 entry
point and fails before that can happen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .group_free_research import canonical_feature_names
from .dufs_liu_feature_contract import FEATURE_TRANSFORMS
from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from .laplacian_upcr import IU_FIT_DEFAULTS
from .repeated_measurement_reliability import FixedMixedV2Transformer
from .upcr import UPCRResult, upcr_fit


A6_FEATURE_ROSTER = canonical_feature_names()
if len(A6_FEATURE_ROSTER) != 30 or "min_spilled" not in A6_FEATURE_ROSTER:
    raise RuntimeError("A6 requires the exact canonical 30-feature roster")

PRESENCE_THRESHOLD = 0.99
RAW_VARIANCE_EPS = 1e-8
TRANSFORMED_VARIANCE_EPS = 1e-8
MIN_CANDIDATE_FEATURES = 17


@dataclass(frozen=True)
class A6MixedV2Transformer:
    """A6 wrapper fixing fit/deploy ECDF identity without changing old A4 code."""

    base: FixedMixedV2Transformer

    @property
    def names(self):
        return self.base.names

    @property
    def training_output(self):
        return self.base.training_output

    @property
    def oriented_mean(self):
        return self.base.oriented_mean

    def transform(self, raw_matrix):
        raw = np.asarray(raw_matrix, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(self.base.names):
            raise ValueError("raw_matrix has the wrong A6 transform shape")
        filled = np.where(np.isfinite(raw), raw, self.base.raw_median[None, :])
        signs = np.asarray(
            [CONFIDENCE_FEATURE_SIGNS_V1[name] for name in self.base.names], dtype=float
        )
        oriented = (
            filled * signs[None, :] - self.base.oriented_mean[None, :]
        ) / self.base.oriented_std[None, :]
        transformed = oriented.copy()
        for index, name in enumerate(self.base.names):
            operation = FEATURE_TRANSFORMS.get(name, "raw")
            if operation == "squared":
                transformed[:, index] = -(oriented[:, index] ** 2)
            elif operation == "mode":
                reference = self.base.sorted_oriented[index]
                left = np.searchsorted(reference, oriented[:, index], side="left")
                right = np.searchsorted(reference, oriented[:, index], side="right")
                # rankdata(values)-0.5 equals 0.5*(left+right), including ties.
                percentile = 0.5 * (left + right) / len(reference)
                transformed[:, index] = -np.abs(
                    percentile - self.base.mode_centres[index]
                )
            elif operation != "raw":
                raise RuntimeError(f"unsupported mixed-v2 operation: {operation}")
        return (
            transformed - self.base.output_mean[None, :]
        ) / self.base.output_std[None, :]


def _require_exact_roster(names) -> tuple[str, ...]:
    names = tuple(str(name) for name in names)
    if names != A6_FEATURE_ROSTER:
        raise ValueError("A6 input names must equal the canonical 30-feature roster in order")
    return names


def validate_complete_quartet_tensor(raw_tensor, feature_names=A6_FEATURE_ROSTER) -> np.ndarray:
    """Validate one scorer's complete `(group,2,2,4,30)` source tensor."""
    _require_exact_roster(feature_names)
    raw = np.asarray(raw_tensor, dtype=float)
    if raw.ndim != 5 or raw.shape[1:4] != (2, 2, 4) or raw.shape[-1] != 30:
        raise ValueError("source tensor must have shape (groups,2,2,4,30)")
    if raw.shape[0] == 0:
        raise ValueError("source tensor has no reciprocal groups")
    if not np.isfinite(raw).all():
        raise ValueError("source tensor must be complete and finite; group-level drop comes first")
    return raw


@dataclass(frozen=True)
class A6NaturalCoordinateSystem:
    """One target's label-free mixed-v2 transform and ordinary IU fit."""

    names: tuple[str, ...]
    source_indices: tuple[int, ...]
    presence_rates: np.ndarray
    raw_medians: np.ndarray
    transformer: A6MixedV2Transformer
    iu: UPCRResult
    candidate_eligible: bool
    excluded: tuple[tuple[str, str], ...]

    def transform_natural_or_evaluation(self, raw_matrix, feature_names=A6_FEATURE_ROSTER):
        """Apply the frozen target-local median and transformer to natural rows."""
        _require_exact_roster(feature_names)
        raw = np.asarray(raw_matrix, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 30:
            raise ValueError("natural/evaluation matrix must have shape (rows,30)")
        selected = raw[:, self.source_indices]
        filled = np.where(np.isfinite(selected), selected, self.raw_medians[None, :])
        if not np.isfinite(filled).all():
            raise ValueError("target-local imputation left non-finite values")
        transformed = self.transformer.transform(filled)
        if not np.isfinite(transformed).all():
            raise ValueError("mixed-v2 transform produced non-finite values")
        return transformed

    def transform_complete_source(self, raw_tensor, feature_names=A6_FEATURE_ROSTER):
        """Apply unchanged target-local coordinates to a complete source tensor."""
        raw = validate_complete_quartet_tensor(raw_tensor, feature_names)
        flat = raw.reshape(-1, 30)
        selected = flat[:, self.source_indices]
        if not np.isfinite(selected).all():
            raise ValueError("source intervention rows may not be imputed")
        transformed = self.transformer.transform(selected)
        if not np.isfinite(transformed).all():
            raise ValueError("mixed-v2 transform produced non-finite source values")
        return transformed.reshape(*raw.shape[:-1], len(self.names))

    def iu_scores(self, transformed_matrix):
        transformed = np.asarray(transformed_matrix, dtype=float)
        if transformed.shape[-1] != len(self.names) or not np.isfinite(transformed).all():
            raise ValueError("transformed matrix and target-local roster disagree")
        return transformed @ np.asarray(self.iu.w, dtype=float)


def fit_natural_coordinate_system(
    raw_matrix,
    feature_names=A6_FEATURE_ROSTER,
) -> A6NaturalCoordinateSystem:
    """Fit A6's label-free target-local transform/IU with fail-closed admission."""
    names = _require_exact_roster(feature_names)
    raw = np.asarray(raw_matrix, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 30 or raw.shape[0] < 3:
        raise ValueError("natural calibration must have shape (at least 3 rows,30)")
    presence = np.mean(np.isfinite(raw), axis=0)
    present_indices = [index for index, rate in enumerate(presence)
                       if rate >= PRESENCE_THRESHOLD]
    excluded = [
        (name, "presence_below_0.99")
        for index, name in enumerate(names) if index not in present_indices
    ]
    if not present_indices:
        raise ValueError("no feature meets the frozen 99% presence rule")
    selected = raw[:, present_indices]
    medians = np.nanmedian(selected, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("an admitted feature has no finite target-local median")
    filled = np.where(np.isfinite(selected), selected, medians[None, :])
    raw_std = np.std(filled, axis=0)
    keep = raw_std > RAW_VARIANCE_EPS
    for local_index, retained in enumerate(keep):
        if not retained:
            excluded.append((names[present_indices[local_index]], "raw_degenerate"))
    present_indices = [index for index, retained in zip(present_indices, keep) if retained]
    if len(present_indices) < 3:
        raise ValueError("fewer than three present nondegenerate features cannot fit IU")

    # A transformed nonlinear coordinate can still degenerate even when raw
    # values vary.  Detect it once, remove it by immutable name, and refit.
    selected = raw[:, present_indices]
    medians = np.nanmedian(selected, axis=0)
    filled = np.where(np.isfinite(selected), selected, medians[None, :])
    kept_names = tuple(names[index] for index in present_indices)
    probe_base = FixedMixedV2Transformer.fit(filled, kept_names)
    probe = A6MixedV2Transformer(probe_base)
    if not np.allclose(probe.transform(filled), probe.training_output, atol=1e-12, rtol=0):
        raise RuntimeError("A6 mixed-v2 fit/deploy coordinate identity failed")
    transformed_std = np.std(probe.training_output, axis=0)
    transformed_keep = transformed_std > TRANSFORMED_VARIANCE_EPS
    if not transformed_keep.all():
        for name, retained in zip(kept_names, transformed_keep):
            if not retained:
                excluded.append((name, "transformed_degenerate"))
        present_indices = [
            index for index, retained in zip(present_indices, transformed_keep) if retained
        ]
        if len(present_indices) < 3:
            raise ValueError("fewer than three transformed nondegenerate features cannot fit IU")
        selected = raw[:, present_indices]
        medians = np.nanmedian(selected, axis=0)
        filled = np.where(np.isfinite(selected), selected, medians[None, :])
        kept_names = tuple(names[index] for index in present_indices)
        transformer = A6MixedV2Transformer(
            FixedMixedV2Transformer.fit(filled, kept_names)
        )
        if not np.allclose(
            transformer.transform(filled), transformer.training_output,
            atol=1e-12, rtol=0,
        ):
            raise RuntimeError("A6 mixed-v2 refit/deploy coordinate identity failed")
    else:
        transformer = probe
    if not np.isfinite(transformer.training_output).all():
        raise ValueError("mixed-v2 training coordinates are non-finite")
    iu = upcr_fit(transformer.training_output.T, **IU_FIT_DEFAULTS)
    if not np.isfinite(iu.w).all():
        raise ValueError("IU fit returned non-finite weights")
    return A6NaturalCoordinateSystem(
        kept_names, tuple(present_indices), presence,
        np.asarray(medians, dtype=float), transformer, iu,
        len(kept_names) >= MIN_CANDIDATE_FEATURES,
        tuple(sorted(excluded)),
    )


__all__ = [
    "A6_FEATURE_ROSTER", "A6MixedV2Transformer", "A6NaturalCoordinateSystem",
    "MIN_CANDIDATE_FEATURES",
    "PRESENCE_THRESHOLD", "fit_natural_coordinate_system",
    "validate_complete_quartet_tensor",
]
