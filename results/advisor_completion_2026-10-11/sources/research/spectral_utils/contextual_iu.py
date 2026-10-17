"""Question-balanced DSP-contextual IU-PCR.

The module is deliberately label blind.  It treats causal DSP states as a
neighbourhood coordinate, never as a correctness target or a verified
nuisance.  Local covariance estimates are shrunk to the ordinary global IU
covariance and every unsafe query falls back exactly to the global IU score.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors

from .upcr import UPCRResult, upcr_fit_covariance


EPS = 1e-12
DSP_CONTEXT_OPERATORS = (
    "innovation",
    "shortlong",
    "positive_mean",
    "persistence",
    "recovery",
)
DEFAULT_IU_FIT = {
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


def _as_2d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty 2-D array")
    return array


def _robust_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    standard = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, standard)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return centre, scale


def _weighted_location_scale(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / (np.sum(weights) + EPS)
    mean = weights @ values
    variance = weights @ ((values - mean) ** 2)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return np.round(mean, 12), np.round(scale, 12)


def _question_balancing_weights(group_codes: np.ndarray) -> np.ndarray:
    counts = np.bincount(group_codes)
    weights = 1.0 / counts[group_codes]
    return weights / np.sum(weights)


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.round(np.asarray(reference, dtype=float), 12))
    query = np.round(np.asarray(values, dtype=float), 12)
    return np.searchsorted(ordered, query, side="right") / max(len(ordered), 1)


@dataclass(frozen=True)
class ContextSpec:
    """Frozen robust scaling and block weighting for a causal context."""

    mode: str
    centres: np.ndarray
    scales: np.ndarray
    block_slices: tuple[tuple[int, int], ...]
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        raw_context: np.ndarray,
        *,
        mode: str,
        block_slices: Sequence[tuple[int, int]],
        feature_names: Sequence[str],
    ) -> "ContextSpec":
        raw = _as_2d(raw_context, name="raw_context")
        blocks = tuple((int(left), int(right)) for left, right in block_slices)
        if mode not in {"core", "dsp"}:
            raise ValueError("mode must be 'core' or 'dsp'")
        if len(tuple(feature_names)) != raw.shape[1]:
            raise ValueError("feature_names and raw_context disagree")
        if not blocks or blocks[0][0] != 0 or blocks[-1][1] != raw.shape[1]:
            raise ValueError("context blocks must cover every column")
        if any(left >= right for left, right in blocks):
            raise ValueError("context blocks must be non-empty")
        centres, scales = _robust_columns(raw)
        centres = np.round(centres, 12)
        scales = np.round(scales, 12)
        return cls(
            mode=mode,
            centres=centres,
            scales=scales,
            block_slices=blocks,
            feature_names=tuple(map(str, feature_names)),
        )

    def transform(self, raw_context: np.ndarray) -> np.ndarray:
        raw = _as_2d(raw_context, name="raw_context")
        if raw.shape[1] != len(self.centres):
            raise ValueError("raw_context width disagrees with ContextSpec")
        output = np.clip((raw - self.centres) / self.scales, -8.0, 8.0)
        for left, right in self.block_slices:
            output[:, left:right] /= math.sqrt(right - left)
        return np.round(output, 12)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "centres": self.centres.tolist(),
            "scales": self.scales.tolist(),
            "block_slices": [list(value) for value in self.block_slices],
            "feature_names": list(self.feature_names),
        }


@dataclass(frozen=True)
class ContextualScore:
    score: np.ndarray
    baseline_score: np.ndarray
    weights: np.ndarray
    family_mass: np.ndarray
    n_eff: np.ndarray
    alpha: np.ndarray
    fallback: np.ndarray
    fallback_reason: tuple[str, ...]
    neighbour_groups: np.ndarray


@dataclass
class ContextualIUModel:
    """A frozen question-balanced local-covariance IU router."""

    context_spec: ContextSpec
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    global_covariance: np.ndarray
    global_result: UPCRResult
    reference_features: np.ndarray
    reference_context: np.ndarray
    reference_group_codes: np.ndarray
    group_names: tuple[str, ...]
    family_indices: tuple[np.ndarray, ...]
    score_reference: np.ndarray
    iu_fit: Mapping[str, object]
    min_effective_questions: int
    k_questions: int
    neighbour_candidate_count: int
    _neighbours: NearestNeighbors | None = None

    @classmethod
    def fit(
        cls,
        features,
        positions,
        group_ids,
        family_indices: Sequence[Sequence[int]],
        *,
        dsp_context=None,
        mode: str = "dsp",
        iu_fit: Mapping[str, object] | None = None,
    ) -> "ContextualIUModel":
        """Fit from target-free reference landmarks.

        ``features`` is sample-by-feature. ``group_ids`` identifies source
        questions and is used both for global moment balancing and local
        effective sample size. ``dsp_context`` must contain the flattened five
        by six registered causal state blocks when ``mode='dsp'``.
        """
        X = _as_2d(features, name="features")
        positions = np.asarray(positions, dtype=float)
        groups = np.asarray(group_ids, dtype=str)
        if positions.shape != (len(X),) or groups.shape != (len(X),):
            raise ValueError("positions/group_ids must have one value per sample")
        if np.any(positions < 0) or not np.isfinite(positions).all():
            raise ValueError("positions must be finite and nonnegative")
        group_names, group_codes = np.unique(groups, return_inverse=True)
        balance = _question_balancing_weights(group_codes)
        feature_mean, feature_scale = _weighted_location_scale(X, balance)
        Z = (X - feature_mean) / feature_scale
        covariance = (Z * balance[:, None]).T @ Z
        covariance = np.round(0.5 * (covariance + covariance.T), 12)
        fit_kwargs = dict(DEFAULT_IU_FIT)
        if iu_fit:
            fit_kwargs.update(iu_fit)
        global_result = upcr_fit_covariance(covariance, **fit_kwargs)
        score_reference = Z @ global_result.w
        families = tuple(np.asarray(value, dtype=int) for value in family_indices)
        covered = sorted(index for values in families for index in values.tolist())
        if covered != list(range(X.shape[1])):
            raise ValueError("family_indices must partition every feature exactly once")
        raw, names, blocks = cls._raw_context_static(
            Z,
            positions,
            dsp_context,
            global_result.w,
            families,
            score_reference,
            mode,
        )
        spec = ContextSpec.fit(
            raw, mode=mode, block_slices=blocks, feature_names=names
        )
        transformed = spec.transform(raw)
        q = len(group_names)
        min_effective_questions = max(32, 4 * X.shape[1])
        # With a non-uniform Gaussian kernel, exactly ``n_eff`` neighbours can
        # never attain effective size ``n_eff``.  Eight deterministic extra
        # questions keep the registered fallback gate reachable.
        k_questions = min(
            q,
            max(min_effective_questions + 8, int(math.ceil(math.sqrt(q)))),
        )
        max_landmarks_per_question = int(np.max(np.bincount(group_codes)))
        # This count guarantees that the candidate list contains at least the
        # k nearest distinct questions even when all landmarks of k-1 questions
        # precede the next question.  It also makes exact row duplication inert.
        candidate_count = min(
            len(X),
            max(64, (k_questions - 1) * max_landmarks_per_question + 1),
        )
        model = cls(
            context_spec=spec,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            global_covariance=covariance,
            global_result=global_result,
            reference_features=Z,
            reference_context=transformed,
            reference_group_codes=group_codes,
            group_names=tuple(map(str, group_names)),
            family_indices=families,
            score_reference=np.sort(np.round(score_reference, 12)),
            iu_fit=fit_kwargs,
            min_effective_questions=min_effective_questions,
            k_questions=k_questions,
            neighbour_candidate_count=candidate_count,
        )
        model._neighbours = NearestNeighbors(metric="euclidean", algorithm="auto")
        model._neighbours.fit(transformed)
        return model

    @staticmethod
    def _raw_context_static(
        standardized_features: np.ndarray,
        positions: np.ndarray,
        dsp_context,
        weights: np.ndarray,
        family_indices: tuple[np.ndarray, ...],
        score_reference: np.ndarray,
        mode: str,
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[int, int], ...]]:
        score = standardized_features @ weights
        contributions = np.column_stack([
            standardized_features[:, indices] @ weights[indices]
            for indices in family_indices
        ])
        family_mad = np.median(
            np.abs(contributions - np.median(contributions, axis=1)[:, None]),
            axis=1,
        )
        core = np.column_stack([
            _ecdf(score_reference, score),
            np.log1p(positions),
            family_mad,
        ])
        names = ("iu_score_ecdf", "log1p_position", "family_contribution_mad")
        blocks: list[tuple[int, int]] = [(0, 3)]
        if mode == "core":
            return core, names, tuple(blocks)
        dsp = _as_2d(dsp_context, name="dsp_context")
        expected = len(family_indices) * len(DSP_CONTEXT_OPERATORS)
        if dsp.shape != (len(core), expected):
            raise ValueError(
                f"dsp_context must have shape {(len(core), expected)}, got {dsp.shape}"
            )
        dsp_names = []
        offset = 3
        for operator in DSP_CONTEXT_OPERATORS:
            width = len(family_indices)
            blocks.append((offset, offset + width))
            dsp_names.extend(
                f"{operator}__family_{index}" for index in range(width)
            )
            offset += width
        return np.column_stack([core, dsp]), names + tuple(dsp_names), tuple(blocks)

    def _query_context(self, Z: np.ndarray, positions, dsp_context) -> np.ndarray:
        raw, _, _ = self._raw_context_static(
            Z,
            np.asarray(positions, dtype=float),
            dsp_context,
            self.global_result.w,
            self.family_indices,
            self.score_reference,
            self.context_spec.mode,
        )
        return self.context_spec.transform(raw)

    def _unique_group_neighbours(
        self, query: np.ndarray, distances: np.ndarray, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chosen: dict[int, tuple[float, int]] = {}
        for distance, index in zip(distances, indices):
            group = int(self.reference_group_codes[index])
            if group not in chosen:
                chosen[group] = (float(distance), int(index))
            if len(chosen) >= self.k_questions:
                break
        if len(chosen) < self.k_questions:
            all_distances = np.linalg.norm(self.reference_context - query, axis=1)
            for index in np.argsort(all_distances, kind="mergesort"):
                group = int(self.reference_group_codes[index])
                if group not in chosen:
                    chosen[group] = (float(all_distances[index]), int(index))
                if len(chosen) >= self.k_questions:
                    break
        ordered = sorted(
            ((distance, index, group) for group, (distance, index) in chosen.items()),
            key=lambda value: (value[0], value[2], value[1]),
        )[: self.k_questions]
        return (
            np.asarray([value[0] for value in ordered], dtype=float),
            np.asarray([value[1] for value in ordered], dtype=int),
            np.asarray([value[2] for value in ordered], dtype=int),
        )

    def score(
        self,
        features,
        positions,
        *,
        dsp_context=None,
        context_override=None,
    ) -> ContextualScore:
        X = _as_2d(features, name="features")
        positions = np.asarray(positions, dtype=float)
        if X.shape[1] != len(self.feature_mean) or positions.shape != (len(X),):
            raise ValueError("query features/positions disagree with the fitted model")
        Z = (X - self.feature_mean) / self.feature_scale
        baseline = Z @ self.global_result.w
        context = (
            _as_2d(context_override, name="context_override")
            if context_override is not None
            else self._query_context(Z, positions, dsp_context)
        )
        if context.shape != (len(X), self.reference_context.shape[1]):
            raise ValueError("query context has the wrong shape")
        if self._neighbours is None:
            self._neighbours = NearestNeighbors(metric="euclidean", algorithm="auto")
            self._neighbours.fit(self.reference_context)
        m = X.shape[1]
        weights_out = np.tile(self.global_result.w, (len(X), 1))
        score = baseline.copy()
        n_eff_out = np.zeros(len(X), dtype=float)
        alpha_out = np.zeros(len(X), dtype=float)
        fallback = np.ones(len(X), dtype=bool)
        reasons = ["insufficient_effective_questions"] * len(X)
        neighbour_groups = np.full((len(X), self.k_questions), -1, dtype=int)
        global_l1 = float(np.sum(np.abs(self.global_result.w)))

        for batch_start in range(0, len(X), 256):
            batch_stop = min(len(X), batch_start + 256)
            candidate_distances, candidate_indices = self._neighbours.kneighbors(
                context[batch_start:batch_stop],
                n_neighbors=self.neighbour_candidate_count,
                return_distance=True,
            )
            for local_index in range(batch_stop - batch_start):
                query_index = batch_start + local_index
                distances, indices, groups = self._unique_group_neighbours(
                    context[query_index],
                    candidate_distances[local_index],
                    candidate_indices[local_index],
                )
                neighbour_groups[query_index, : len(groups)] = groups
                positive = distances[distances > EPS]
                bandwidth = float(distances[-1]) if len(distances) else 0.0
                if bandwidth <= EPS and len(positive):
                    bandwidth = float(positive[-1])
                if bandwidth <= EPS:
                    reasons[query_index] = "zero_context_bandwidth"
                    continue
                kernel = np.exp(-0.5 * (distances / bandwidth) ** 2)
                n_eff = float(np.sum(kernel) ** 2 / (np.sum(kernel ** 2) + EPS))
                n_eff_out[query_index] = n_eff
                if n_eff < self.min_effective_questions:
                    continue
                local = self.reference_features[indices]
                norm = kernel / (np.sum(kernel) + EPS)
                local_mean = norm @ local
                centred = local - local_mean
                local_covariance = (centred * norm[:, None]).T @ centred
                alpha = float(n_eff / (n_eff + 4.0 * m))
                covariance = (
                    (1.0 - alpha) * self.global_covariance
                    + alpha * local_covariance
                )
                # Canonicalize sub-floating-point accumulation differences.
                # Without this, an almost-degenerate local top-two space can
                # rotate materially when identical rows are merely duplicated.
                covariance = np.round(0.5 * (covariance + covariance.T), 12)
                eigenvalues = eigh(
                    covariance, subset_by_index=[m - min(2, m), m - 1],
                    eigvals_only=True,
                )[::-1]
                if (
                    not np.isfinite(eigenvalues).all()
                    or eigenvalues[-1] <= EPS
                    or eigenvalues[0] / eigenvalues[-1] > 1e8
                ):
                    reasons[query_index] = "ill_conditioned_projected_covariance"
                    continue
                try:
                    result = upcr_fit_covariance(covariance, **self.iu_fit)
                except (ValueError, np.linalg.LinAlgError):
                    reasons[query_index] = "iu_fit_failure"
                    continue
                local_weights = np.asarray(result.w, dtype=float)
                if not np.isfinite(local_weights).all():
                    reasons[query_index] = "nonfinite_weights"
                    continue
                if float(local_weights @ self.global_result.w) < 0.0:
                    local_weights = -local_weights
                local_l1 = float(np.sum(np.abs(local_weights)))
                if local_l1 <= EPS or global_l1 <= EPS:
                    reasons[query_index] = "zero_weight_leverage"
                    continue
                local_weights *= global_l1 / local_l1
                weights_out[query_index] = local_weights
                score[query_index] = float(local_weights @ Z[query_index])
                alpha_out[query_index] = alpha
                fallback[query_index] = False
                reasons[query_index] = ""

        family_mass = np.column_stack([
            np.sum(np.abs(weights_out[:, indices]), axis=1)
            for indices in self.family_indices
        ])
        family_mass /= np.sum(family_mass, axis=1, keepdims=True) + EPS
        return ContextualScore(
            score=score,
            baseline_score=baseline,
            weights=weights_out,
            family_mass=family_mass,
            n_eff=n_eff_out,
            alpha=alpha_out,
            fallback=fallback,
            fallback_reason=tuple(reasons),
            neighbour_groups=neighbour_groups,
        )

    def as_dict(self) -> dict:
        return {
            "context": self.context_spec.as_dict(),
            "n_reference_landmarks": int(len(self.reference_features)),
            "n_reference_questions": int(len(self.group_names)),
            "n_features": int(len(self.feature_mean)),
            "n_families": int(len(self.family_indices)),
            "min_effective_questions": int(self.min_effective_questions),
            "k_questions": int(self.k_questions),
            "global_weights": self.global_result.w.tolist(),
            "global_covariance_eigenvalues": np.linalg.eigvalsh(
                self.global_covariance
            ).tolist(),
            "iu_fit": dict(self.iu_fit),
        }


def flatten_dsp_context(states: Mapping[str, np.ndarray]) -> np.ndarray:
    """Flatten the five registered family-resolved DSP state blocks."""
    missing = [name for name in DSP_CONTEXT_OPERATORS if name not in states]
    if missing:
        raise KeyError("missing DSP states: " + ", ".join(missing))
    matrices = [_as_2d(states[name], name=name) for name in DSP_CONTEXT_OPERATORS]
    if len({matrix.shape for matrix in matrices}) != 1:
        raise ValueError("all DSP state matrices must have the same shape")
    return np.column_stack(matrices)


def family_partition(n_families: int, operators: int = 1) -> tuple[tuple[int, ...], ...]:
    """Return family indices for operator-major feature concatenation."""
    n_families = int(n_families)
    operators = int(operators)
    if n_families < 1 or operators < 1:
        raise ValueError("n_families and operators must be positive")
    return tuple(
        tuple(operator * n_families + family for operator in range(operators))
        for family in range(n_families)
    )


def tangent_dimension(local_covariance: np.ndarray, variance_fraction: float = 0.90) -> int:
    """Label-blind LPCA dimension, clipped to the registered [2, 8] range."""
    covariance = np.asarray(local_covariance, dtype=float)
    values = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
    if not len(values) or np.sum(values) <= EPS:
        return min(2, len(values))
    dimension = int(np.searchsorted(np.cumsum(values) / np.sum(values), variance_fraction) + 1)
    return int(np.clip(dimension, min(2, len(values)), min(8, len(values))))
