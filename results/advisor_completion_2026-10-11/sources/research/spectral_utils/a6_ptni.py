"""Numeric-only factorial PTNI-IU core for A6.

This module cannot receive prompts, responses, benchmark labels, or diagnostic
truth.  Reciprocal polarity is structural: prompt/response axes are always
`A,B`, render axis is `canonical,paraphrase,layout,notation`, and the target
effect is the frozen off-diagonal-minus-diagonal crossover contrast.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .a6_features import (
    A6_FEATURE_ROSTER,
    A6NaturalCoordinateSystem,
    MIN_CANDIDATE_FEATURES,
)
from .a6_interventions import DOMAINS, MUTATIONS, RENDERINGS, RESPONSE_GRAMMARS
from .laplacian_upcr import IU_FIT_DEFAULTS
from .upcr import upcr_fit


LAMBDA_GRID = (0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0)
ALPHA_GRID = (0.0, 0.0625, 0.125, 0.25, 0.50, 1.0)
QWEN_SOURCE_SCORERS = ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B")


@dataclass(frozen=True)
class NamedCoordinateMatrix:
    names: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class NamedCoordinateVector:
    names: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class SourceQuartetBatch:
    """Observed-only complete reciprocal tensors from one or more scorers."""

    values: np.ndarray  # (group, scorer, prompt, response, render, feature)
    feature_names: tuple[str, ...]
    scorer_names: tuple[str, ...]
    group_ids: tuple[str, ...]
    domains: tuple[str, ...]
    mutations: tuple[str, ...]
    grammars: tuple[str, ...]
    expected_domains: tuple[str, ...] = DOMAINS
    expected_mutations: tuple[str, ...] = MUTATIONS
    expected_grammars: tuple[str, ...] = RESPONSE_GRAMMARS
    coordinate_kind: str = "canonical"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 6 or values.shape[2:5] != (2, 2, 4):
            raise ValueError("values must have shape (groups,scorers,2,2,4,features)")
        if not np.isfinite(values).all():
            raise ValueError("PTNI source values must be complete and finite")
        n_groups, n_scorers, _, _, _, n_features = values.shape
        if n_features != len(self.feature_names) or len(set(self.feature_names)) != n_features:
            raise ValueError("feature_names must uniquely match the source tensor")
        if self.coordinate_kind == "canonical":
            roster_positions = [A6_FEATURE_ROSTER.index(name) for name in self.feature_names
                                if name in A6_FEATURE_ROSTER]
            if len(roster_positions) != n_features or roster_positions != sorted(roster_positions):
                raise ValueError("source features must be a canonical-order A6 roster subset")
        elif self.coordinate_kind == "exact_duplicate_quotient":
            flattened = []
            for name in self.feature_names:
                members = name.removeprefix("mean::").split("|")
                if any(member not in A6_FEATURE_ROSTER for member in members):
                    raise ValueError("quotient coordinate contains an unknown feature identity")
                flattened.extend(members)
            if len(flattened) != len(set(flattened)):
                raise ValueError("quotient coordinate components overlap")
        else:
            raise ValueError("unknown A6 coordinate kind")
        if self.coordinate_kind == "canonical" and n_features < MIN_CANDIDATE_FEATURES:
            raise ValueError("PTNI source requires at least 17 coordinates")
        if n_features < 1:
            raise ValueError("PTNI source has no coordinates")
        if self.scorer_names != QWEN_SOURCE_SCORERS or n_scorers != 2:
            raise ValueError("source scorer axis must be the exact two registered Qwen views")
        fields = (self.group_ids, self.domains, self.mutations, self.grammars)
        if any(len(field) != n_groups for field in fields):
            raise ValueError("group metadata must match the group axis")
        if len(set(self.group_ids)) != n_groups:
            raise ValueError("reciprocal group IDs must be unique")
        if any(value not in DOMAINS for value in self.domains):
            raise ValueError("unregistered semantic domain")
        if any(value not in MUTATIONS for value in self.mutations):
            raise ValueError("unregistered target mutation")
        if any(value not in RESPONSE_GRAMMARS for value in self.grammars):
            raise ValueError("unregistered response grammar")
        if (
            not self.expected_domains or not self.expected_mutations or not self.expected_grammars
            or not set(self.expected_domains).issubset(DOMAINS)
            or not set(self.expected_mutations).issubset(MUTATIONS)
            or not set(self.expected_grammars).issubset(RESPONSE_GRAMMARS)
        ):
            raise ValueError("expected family sets are invalid")
        observed = set(zip(self.domains, self.mutations, self.grammars))
        expected = {
            (domain, mutation, grammar)
            for domain in self.expected_domains
            for mutation in self.expected_mutations
            for grammar in self.expected_grammars
        }
        if observed != expected:
            raise ValueError("source batch does not contain the exact expected family cartesian set")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ExactDuplicateQuotient:
    """Immutable-name arithmetic-mean quotient for bit-identical coordinates."""

    original_names: tuple[str, ...]
    components: tuple[tuple[str, ...], ...]
    coordinate_names: tuple[str, ...]
    transform: np.ndarray  # original p by quotient q; rows map X @ transform

    def reduce(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape[-1] != len(self.original_names) or not np.isfinite(matrix).all():
            raise ValueError("quotient input disagrees with original source coordinates")
        return matrix @ self.transform

    def expand_correction(self, quotient_weight):
        weight = np.asarray(quotient_weight, dtype=float)
        if weight.shape != (len(self.coordinate_names),) or not np.isfinite(weight).all():
            raise ValueError("quotient correction has the wrong shape")
        return self.transform @ weight

    def target_equality_pass(self, target: A6NaturalCoordinateSystem) -> bool:
        name_to_index = {name: index for index, name in enumerate(target.names)}
        z = np.asarray(target.transformer.training_output, dtype=float)
        for component in self.components:
            if len(component) <= 1:
                continue
            if any(name not in name_to_index for name in component):
                return False
            reference = z[:, name_to_index[component[0]]]
            for name in component[1:]:
                if np.max(np.abs(reference - z[:, name_to_index[name]])) > 1e-10:
                    return False
        return True


def discover_exact_duplicate_quotient(
    batch: SourceQuartetBatch,
    natural_transformed_by_scorer: dict[str, NamedCoordinateMatrix],
) -> ExactDuplicateQuotient:
    """Discover connected bit-identical components using training rows only."""
    if tuple(sorted(natural_transformed_by_scorer)) != tuple(sorted(QWEN_SOURCE_SCORERS)):
        raise ValueError("duplicate discovery needs the exact two named Qwen natural matrices")
    p = len(batch.feature_names)
    natural = []
    for scorer in QWEN_SOURCE_SCORERS:
        record = natural_transformed_by_scorer[scorer]
        if record.names != batch.feature_names:
            raise ValueError("natural duplicate names do not match source coordinates")
        matrix = np.asarray(record.values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != p or not np.isfinite(matrix).all():
            raise ValueError("natural duplicate matrix has the wrong finite shape")
        natural.append(matrix)
    source = batch.values.reshape(-1, p)
    def bit_identical(left, right) -> bool:
        left = np.ascontiguousarray(left, dtype=np.float64).view(np.uint64)
        right = np.ascontiguousarray(right, dtype=np.float64).view(np.uint64)
        return np.array_equal(left, right)
    parent = list(range(p))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(p):
        for right in range(left + 1, p):
            if (
                bit_identical(source[:, left], source[:, right])
                and all(bit_identical(matrix[:, left], matrix[:, right]) for matrix in natural)
            ):
                union(left, right)
    grouped: dict[int, list[str]] = {}
    for index, name in enumerate(batch.feature_names):
        grouped.setdefault(find(index), []).append(name)
    components = tuple(
        tuple(sorted(members, key=A6_FEATURE_ROSTER.index))
        for _, members in sorted(
            grouped.items(), key=lambda item: min(A6_FEATURE_ROSTER.index(v) for v in item[1])
        )
    )
    transform = np.zeros((p, len(components)), dtype=float)
    coordinate_names = []
    name_index = {name: index for index, name in enumerate(batch.feature_names)}
    for component_index, component in enumerate(components):
        for name in component:
            transform[name_index[name], component_index] = 1.0 / len(component)
        coordinate_names.append(
            component[0] if len(component) == 1 else "mean::" + "|".join(component)
        )
    return ExactDuplicateQuotient(
        batch.feature_names, components, tuple(coordinate_names), transform
    )


def quotient_batch(
    batch: SourceQuartetBatch,
    quotient: ExactDuplicateQuotient,
) -> SourceQuartetBatch:
    if quotient.original_names != batch.feature_names:
        raise ValueError("quotient and source batch names disagree")
    return SourceQuartetBatch(
        quotient.reduce(batch.values), quotient.coordinate_names, batch.scorer_names,
        batch.group_ids, batch.domains, batch.mutations, batch.grammars,
        batch.expected_domains, batch.expected_mutations, batch.expected_grammars,
        "exact_duplicate_quotient",
    )


@dataclass(frozen=True)
class FactorialEffects:
    tau: np.ndarray  # (g,s,4,p), invalid minus valid
    nuisance: np.ndarray  # (g,s,4 prompt-response cells,3 noncanonical,p)
    interaction: np.ndarray  # (g,s,3,p)


@dataclass(frozen=True)
class FactorialMoments:
    feature_names: tuple[str, ...]
    mu_target: np.ndarray
    target_covariance: np.ndarray
    nuisance_second_moment: np.ndarray
    interaction_second_moment: np.ndarray
    total: np.ndarray
    intervention_energy: np.ndarray
    target_weights: np.ndarray
    nuisance_weights: np.ndarray
    interaction_weights: np.ndarray
    fitted_renderings: tuple[str, ...]


@dataclass(frozen=True)
class SourceRiskDirection:
    feature_names: tuple[str, ...]
    ridge: float
    weight: np.ndarray
    active: np.ndarray
    trace_scale: float
    zero_evidence_reason: str | None
    duplicate_components: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class AnchoredAffineScore:
    feature_names: tuple[str, ...]
    alpha: float
    iu_weight: np.ndarray
    unit_correction: np.ndarray
    weight: np.ndarray
    intercept: float
    zero_evidence_reason: str | None
    ordinary_iu_weight: np.ndarray
    duplicate_components: tuple[tuple[str, ...], ...] = ()
    evaluation_bound: bool = True

    def bind_evaluation(self, transformed_matrix):
        matrix = np.asarray(transformed_matrix, dtype=float)
        if matrix.shape[-1] != len(self.feature_names) or not np.isfinite(matrix).all():
            raise ValueError("score matrix and target-local transformed roster disagree")
        if self.evaluation_bound:
            return self
        name_to_index = {name: index for index, name in enumerate(self.feature_names)}
        for component in self.duplicate_components:
            if len(component) <= 1:
                continue
            if any(name not in name_to_index for name in component):
                return AnchoredAffineScore(
                    feature_names=self.feature_names, alpha=self.alpha,
                    iu_weight=self.ordinary_iu_weight.copy(),
                    unit_correction=np.zeros_like(self.ordinary_iu_weight),
                    weight=self.ordinary_iu_weight.copy(), intercept=self.intercept,
                    zero_evidence_reason="target_duplicate_evaluation_failed",
                    ordinary_iu_weight=self.ordinary_iu_weight.copy(),
                    duplicate_components=(), evaluation_bound=True,
                )
            reference = matrix[..., name_to_index[component[0]]]
            if any(
                np.max(np.abs(reference - matrix[..., name_to_index[name]])) > 1e-10
                for name in component[1:]
            ):
                return AnchoredAffineScore(
                    feature_names=self.feature_names, alpha=self.alpha,
                    iu_weight=self.ordinary_iu_weight.copy(),
                    unit_correction=np.zeros_like(self.ordinary_iu_weight),
                    weight=self.ordinary_iu_weight.copy(), intercept=self.intercept,
                    zero_evidence_reason="target_duplicate_evaluation_failed",
                    ordinary_iu_weight=self.ordinary_iu_weight.copy(),
                    duplicate_components=(), evaluation_bound=True,
                )
        return replace(self, evaluation_bound=True)

    def score(self, transformed_matrix):
        matrix = np.asarray(transformed_matrix, dtype=float)
        if not self.evaluation_bound:
            raise ValueError("duplicate-aware score must bind the full frozen target first")
        if matrix.shape[-1] != len(self.feature_names) or not np.isfinite(matrix).all():
            raise ValueError("score matrix and target-local transformed roster disagree")
        return matrix @ self.weight + self.intercept


@dataclass(frozen=True)
class StructuralMetrics:
    target_margin: float
    nuisance_ratios: tuple[tuple[str, float], ...]
    interaction_ratios: tuple[tuple[str, float], ...]
    nuisance_rms: tuple[tuple[str, float], ...]
    interaction_rms: tuple[tuple[str, float], ...]


def factorial_effects(batch: SourceQuartetBatch) -> FactorialEffects:
    z = batch.values
    tau = 0.5 * (
        (z[:, :, 1, 0, :, :] - z[:, :, 0, 0, :, :])
        + (z[:, :, 0, 1, :, :] - z[:, :, 1, 1, :, :])
    )
    canonical = z[..., 0, :]
    noncanonical = z[..., 1:, :]
    nuisance = noncanonical - canonical[..., None, :]
    nuisance = nuisance.reshape(z.shape[0], z.shape[1], 4, 3, z.shape[-1])
    interaction = tau[:, :, 1:, :] - tau[:, :, :1, :]
    return FactorialEffects(tau, nuisance, interaction)


def _normalized_cell_weights(keys: Sequence[tuple]) -> np.ndarray:
    counts: dict[tuple, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise ValueError("cannot weight an empty factorial sample")
    n_cells = len(counts)
    weights = np.asarray([1.0 / (n_cells * counts[key]) for key in keys], dtype=float)
    if abs(float(weights.sum()) - 1.0) > 1e-12:
        raise RuntimeError("factorial weights do not sum to one")
    return weights


def _weighted_second(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("n,ni,nj->ij", weights, samples, samples)


def factorial_moments(
    batch: SourceQuartetBatch,
    *,
    fitted_renderings: tuple[str, ...] = RENDERINGS,
) -> FactorialMoments:
    fitted_renderings = tuple(fitted_renderings)
    registered_order = tuple(
        rendering for rendering in RENDERINGS if rendering in fitted_renderings
    )
    if (
        fitted_renderings != registered_order
        or fitted_renderings[0:1] != ("canonical",)
        or len(fitted_renderings) not in (3, 4)
        or len(set(fitted_renderings)) != len(fitted_renderings)
    ):
        raise ValueError(
            "fitted renderings must be all registered renderings or canonical plus "
            "exactly two registered nuisance renderings in frozen order"
        )
    render_indices = tuple(RENDERINGS.index(rendering) for rendering in fitted_renderings)
    nuisance_indices = tuple(index - 1 for index in render_indices if index > 0)
    effects = factorial_effects(batch)
    g, s, _, p = effects.tau.shape
    target_samples, target_keys = [], []
    nuisance_samples, nuisance_keys = [], []
    interaction_samples, interaction_keys = [], []
    for group in range(g):
        base = (batch.domains[group], batch.mutations[group], batch.grammars[group])
        for scorer in range(s):
            for rendering in render_indices:
                target_samples.append(effects.tau[group, scorer, rendering])
                target_keys.append((scorer, *base, RENDERINGS[rendering]))
            for rendering in nuisance_indices:
                for cell in range(4):
                    nuisance_samples.append(effects.nuisance[group, scorer, cell, rendering])
                    nuisance_keys.append((scorer, *base, RENDERINGS[rendering + 1], cell))
                interaction_samples.append(effects.interaction[group, scorer, rendering])
                interaction_keys.append((scorer, *base, RENDERINGS[rendering + 1]))
    target_samples = np.asarray(target_samples, dtype=float).reshape(-1, p)
    nuisance_samples = np.asarray(nuisance_samples, dtype=float).reshape(-1, p)
    interaction_samples = np.asarray(interaction_samples, dtype=float).reshape(-1, p)
    target_weights = _normalized_cell_weights(target_keys)
    nuisance_weights = _normalized_cell_weights(nuisance_keys)
    interaction_weights = _normalized_cell_weights(interaction_keys)
    mu = target_weights @ target_samples
    centered = target_samples - mu[None, :]
    target_covariance = _weighted_second(centered, target_weights)
    nuisance_second = _weighted_second(nuisance_samples, nuisance_weights)
    interaction_second = _weighted_second(interaction_samples, interaction_weights)
    total = target_covariance + nuisance_second + interaction_second
    energy = (
        target_weights @ (target_samples ** 2)
        + nuisance_weights @ (nuisance_samples ** 2)
        + interaction_weights @ (interaction_samples ** 2)
    )
    for matrix in (target_covariance, nuisance_second, interaction_second, total):
        if not np.isfinite(matrix).all() or np.max(np.abs(matrix - matrix.T)) > 1e-10:
            raise ValueError("factorial moment is non-finite or asymmetric")
    return FactorialMoments(
        batch.feature_names, mu, target_covariance, nuisance_second,
        interaction_second, total, energy,
        target_weights, nuisance_weights, interaction_weights,
        fitted_renderings,
    )


def fit_source_risk_direction(
    moments: FactorialMoments,
    natural_variances_by_scorer: dict[str, NamedCoordinateVector],
    ridge: float,
    *,
    quotient: ExactDuplicateQuotient | None = None,
) -> SourceRiskDirection:
    names = moments.feature_names
    p = len(names)
    if ridge not in LAMBDA_GRID:
        raise ValueError("ridge is outside the frozen A6 grid")
    if tuple(sorted(natural_variances_by_scorer)) != tuple(sorted(QWEN_SOURCE_SCORERS)):
        raise ValueError("natural variances must be bound to the exact Qwen scorer names")
    if any(
        natural_variances_by_scorer[scorer].names != names
        for scorer in QWEN_SOURCE_SCORERS
    ):
        raise ValueError("natural variance names do not match moment coordinates")
    variances = np.asarray([
        natural_variances_by_scorer[scorer].values for scorer in QWEN_SOURCE_SCORERS
    ], dtype=float)
    if variances.shape != (2, p):
        raise ValueError("natural variances must have shape (2,p)")
    if not np.isfinite(variances).all():
        raise ValueError("natural variances are non-finite")
    active = np.all(variances > 1e-8, axis=0) & (moments.intervention_energy > 1e-10)
    quotient_weight = np.zeros(p, dtype=float)
    nominal_weight = np.zeros(len(A6_FEATURE_ROSTER), dtype=float)
    if quotient is None:
        if any(name not in A6_FEATURE_ROSTER for name in names):
            raise ValueError("noncanonical moment names require an exact duplicate quotient")
        quotient = ExactDuplicateQuotient(
            names, tuple((name,) for name in names), names, np.eye(p)
        )
    if quotient.coordinate_names != names:
        raise ValueError("moments and duplicate quotient coordinates disagree")

    def finish(reason, trace_scale=0.0):
        expanded_active = quotient.expand_correction(active.astype(float)) != 0
        nominal_active = np.zeros(len(A6_FEATURE_ROSTER), dtype=bool)
        for index, name in enumerate(quotient.original_names):
            nominal_active[A6_FEATURE_ROSTER.index(name)] = expanded_active[index]
        return SourceRiskDirection(
            A6_FEATURE_ROSTER, ridge, nominal_weight.copy(), nominal_active,
            float(trace_scale), reason, quotient.components,
        )

    if int(active.sum()) < MIN_CANDIDATE_FEATURES:
        return finish("fewer_than_17_active")
    subspace = moments.total[np.ix_(active, active)]
    trace_scale = float(np.trace(subspace) / int(active.sum()))
    if not np.isfinite(trace_scale) or trace_scale <= 1e-12:
        return finish("invalid_trace_scale", trace_scale)
    scaled = subspace / trace_scale
    solution = np.linalg.solve(
        scaled + float(ridge) * np.eye(int(active.sum())), moments.mu_target[active]
    )
    if not np.isfinite(solution).all():
        return finish("nonfinite_direction", trace_scale)
    quotient_weight[active] = solution
    expanded = quotient.expand_correction(quotient_weight)
    nominal_active = np.zeros(len(A6_FEATURE_ROSTER), dtype=bool)
    for index, name in enumerate(quotient.original_names):
        nominal_index = A6_FEATURE_ROSTER.index(name)
        nominal_weight[nominal_index] = expanded[index]
        nominal_active[nominal_index] = bool(expanded[index] != 0 or any(
            active[component_index] and name in component
            for component_index, component in enumerate(quotient.components)
        ))
    return SourceRiskDirection(
        A6_FEATURE_ROSTER, ridge, nominal_weight, nominal_active,
        trace_scale, None, quotient.components,
    )


def anchor_source_direction(
    source: SourceRiskDirection,
    target: A6NaturalCoordinateSystem,
    alpha: float,
) -> AnchoredAffineScore:
    if alpha not in ALPHA_GRID:
        raise ValueError("alpha is outside the frozen A6 grid")
    ordinary_iu = np.asarray(target.iu.w, dtype=float).copy()
    target_index = {name: index for index, name in enumerate(target.names)}
    target_z = np.asarray(target.transformer.training_output, dtype=float)
    active_duplicate_components = tuple(
        component for component in source.duplicate_components if len(component) > 1
    )
    for component in source.duplicate_components:
        if len(component) <= 1:
            continue
        if any(name not in target_index for name in component):
            return AnchoredAffineScore(
                feature_names=target.names,
                alpha=alpha,
                iu_weight=ordinary_iu.copy(),
                unit_correction=np.zeros_like(ordinary_iu),
                weight=ordinary_iu.copy(),
                intercept=0.0,
                zero_evidence_reason="target_duplicate_component_missing",
                ordinary_iu_weight=ordinary_iu.copy(),
                duplicate_components=(),
                evaluation_bound=True,
            )
        reference = target_z[:, target_index[component[0]]]
        if any(
            np.max(np.abs(reference - target_z[:, target_index[name]])) > 1e-10
            for name in component[1:]
        ):
            return AnchoredAffineScore(
                feature_names=target.names,
                alpha=alpha,
                iu_weight=ordinary_iu.copy(),
                unit_correction=np.zeros_like(ordinary_iu),
                weight=ordinary_iu.copy(),
                intercept=0.0,
                zero_evidence_reason="target_duplicate_equality_failed",
                ordinary_iu_weight=ordinary_iu.copy(),
                duplicate_components=(),
                evaluation_bound=True,
            )
    iu_weight = ordinary_iu
    if active_duplicate_components:
        assigned = set()
        target_components = []
        for component in source.duplicate_components:
            present = tuple(name for name in component if name in target_index)
            if present:
                target_components.append(present)
                assigned.update(present)
        target_components.extend(
            (name,) for name in target.names if name not in assigned
        )
        target_components.sort(key=lambda component: min(target_index[name] for name in component))
        transform = np.zeros((len(target.names), len(target_components)), dtype=float)
        for component_index, component in enumerate(target_components):
            for name in component:
                transform[target_index[name], component_index] = 1.0 / len(component)
        quotient_z = target_z @ transform
        quotient_iu = upcr_fit(quotient_z.T, **IU_FIT_DEFAULTS)
        iu_weight = transform @ quotient_iu.w

    fallback = lambda reason: AnchoredAffineScore(  # noqa: E731
        feature_names=target.names,
        alpha=alpha,
        iu_weight=iu_weight.copy(),
        unit_correction=np.zeros_like(iu_weight),
        weight=iu_weight.copy(),
        intercept=0.0,
        zero_evidence_reason=reason,
        ordinary_iu_weight=ordinary_iu.copy(),
        duplicate_components=active_duplicate_components,
        evaluation_bound=not active_duplicate_components,
    )
    if alpha == 0.0:
        return fallback("alpha_zero_exact_iu")
    if source.zero_evidence_reason is not None:
        return fallback(source.zero_evidence_reason)
    if not target.candidate_eligible:
        return fallback("target_fewer_than_17")
    source_index = {name: index for index, name in enumerate(source.feature_names)}
    risk = np.asarray([source.weight[source_index[name]] for name in target.names])
    z = target_z
    covariance = (z.T @ z) / len(z)
    iu_norm2 = float(iu_weight @ covariance @ iu_weight)
    risk_norm2 = float(risk @ covariance @ risk)
    if iu_norm2 <= 1e-10:
        return fallback("degenerate_iu")
    if risk_norm2 <= 1e-10:
        return fallback("degenerate_source_direction")
    correction = risk - iu_weight * float(iu_weight @ covariance @ risk) / iu_norm2
    correction_norm2 = float(correction @ covariance @ correction)
    if correction_norm2 <= 0:
        return fallback("degenerate_orthogonal_correction")
    if np.sqrt(correction_norm2 / risk_norm2) < 0.25:
        return fallback("insufficient_retained_correction")
    correction = correction * np.sqrt(iu_norm2 / correction_norm2)
    if abs(float(iu_weight @ covariance @ correction)) > 1e-10:
        raise RuntimeError("A6 correction is not IU covariance-orthogonal")
    weight = iu_weight - float(alpha) * correction
    return AnchoredAffineScore(
        feature_names=target.names,
        alpha=alpha,
        iu_weight=iu_weight,
        unit_correction=correction,
        weight=weight,
        intercept=0.0,
        zero_evidence_reason=None,
        ordinary_iu_weight=ordinary_iu.copy(),
        duplicate_components=active_duplicate_components,
        evaluation_bound=not active_duplicate_components,
    )


def quartet_delta(confidence_scores) -> np.ndarray:
    scores = np.asarray(confidence_scores, dtype=float)
    if scores.ndim != 5 or scores.shape[2:5] != (2, 2, 4):
        raise ValueError("confidence scores must have shape (groups,scorers,2,2,4)")
    if not np.isfinite(scores).all():
        raise ValueError("confidence scores must be finite")
    return 0.5 * (
        scores[:, :, 0, 0, :] - scores[:, :, 1, 0, :]
        + scores[:, :, 1, 1, :] - scores[:, :, 0, 1, :]
    )


def _macro_mean(values: np.ndarray, batch: SourceQuartetBatch, include_render: bool) -> float:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("macro inputs must be finite")
    expected = (len(batch.group_ids), len(batch.scorer_names))
    if include_render:
        expected = (*expected, values.shape[-1])
    if values.shape[:2] != expected[:2]:
        raise ValueError("macro values and batch axes disagree")
    samples, keys = [], []
    for group in range(len(batch.group_ids)):
        base = (batch.domains[group], batch.mutations[group], batch.grammars[group])
        for scorer in range(len(batch.scorer_names)):
            if include_render:
                for rendering in range(values.shape[2]):
                    samples.append(values[group, scorer, rendering])
                    keys.append((scorer, *base, rendering))
            else:
                samples.append(values[group, scorer])
                keys.append((scorer, *base))
    weights = _normalized_cell_weights(keys)
    return float(weights @ np.asarray(samples, dtype=float))


def ordering_objective(confidence_scores, batch: SourceQuartetBatch) -> float:
    if not np.isfinite(np.asarray(confidence_scores, dtype=float)).all():
        raise ValueError("ordering scores must be finite")
    delta = quartet_delta(confidence_scores)
    half_credit = (delta > 0).astype(float) + 0.5 * (delta == 0)
    return _macro_mean(half_credit, batch, include_render=True)


def structural_metrics(
    unit_correction_scores,
    batch: SourceQuartetBatch,
) -> StructuralMetrics:
    scores = np.asarray(unit_correction_scores, dtype=float)
    if scores.shape != batch.values.shape[:-1]:
        raise ValueError("unit correction scores and source batch disagree")
    if not np.isfinite(scores).all():
        raise ValueError("unit correction scores must be finite")
    delta = quartet_delta(scores)
    target_margin = _macro_mean(delta, batch, include_render=True)
    nuisance_ratios, interaction_ratios = [], []
    nuisance_rms_values, interaction_rms_values = [], []
    canonical = scores[..., 0]
    for render_index, rendering in enumerate(RENDERINGS[1:], 1):
        nuisance = scores[..., render_index] - canonical
        nuisance_sq = nuisance.reshape(
            nuisance.shape[0], nuisance.shape[1], -1
        ) ** 2
        nuisance_cell_means = np.mean(nuisance_sq, axis=2)
        nuisance_rms = np.sqrt(_macro_mean(nuisance_cell_means, batch, False))
        interaction = delta[:, :, render_index] - delta[:, :, 0]
        interaction_rms = np.sqrt(_macro_mean(interaction ** 2, batch, False))
        family_margin = 0.5 * _macro_mean(
            delta[:, :, render_index] + delta[:, :, 0], batch, False
        )
        denominator = abs(family_margin)
        nuisance_ratio = np.inf if denominator <= 1e-12 else nuisance_rms / denominator
        interaction_ratio = np.inf if denominator <= 1e-12 else interaction_rms / denominator
        nuisance_ratios.append((rendering, float(nuisance_ratio)))
        interaction_ratios.append((rendering, float(interaction_ratio)))
        nuisance_rms_values.append((rendering, float(nuisance_rms)))
        interaction_rms_values.append((rendering, float(interaction_rms)))
    return StructuralMetrics(
        target_margin, tuple(nuisance_ratios), tuple(interaction_ratios),
        tuple(nuisance_rms_values), tuple(interaction_rms_values),
    )


__all__ = [
    "ALPHA_GRID", "LAMBDA_GRID", "QWEN_SOURCE_SCORERS",
    "AnchoredAffineScore", "ExactDuplicateQuotient", "FactorialEffects",
    "FactorialMoments", "NamedCoordinateMatrix", "NamedCoordinateVector",
    "SourceQuartetBatch", "SourceRiskDirection", "StructuralMetrics",
    "anchor_source_direction", "discover_exact_duplicate_quotient",
    "factorial_effects", "factorial_moments", "fit_source_risk_direction",
    "ordering_objective", "quartet_delta", "quotient_batch",
    "structural_metrics",
]
