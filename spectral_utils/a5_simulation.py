"""Target-firewalled development/sealed simulator for group-free IU Phase A5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from .anchored_sparse_latent_mixture import (
    SparseMixtureFit,
    GRAPH_EDGE_THRESHOLD,
    Standardization,
    anchored_direction,
    diagonal_support,
    fit_constrained_direction_mixture,
    fit_fixed_support_precision,
    fit_sparse_equal_covariance_mixture,
    fit_standardization,
    held_mean_log_likelihood,
    gaussian_log_density,
    mixture_log_density,
    support_from_residual_correlations,
    within_component_correlation,
)
from .a5_target_free_data import CORE_FEATURES
from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1


PENALTIES = (0.01, 0.02, 0.05, 0.10, 0.20)
ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0)
DEVELOPMENT_SEED_MIN, DEVELOPMENT_SEED_MAX = 510000, 519999
SEALED_SEED_MIN, SEALED_SEED_MAX = 520000, 529999
NUISANCE_WORLD_INDEX = 8
SEALED_REPETITIONS = 100


def sealed_world_seed(world_index: int, repetition: int) -> int:
    if int(world_index) not in range(1, 12):
        raise ValueError("world_index must be in 1..11")
    if int(repetition) not in range(SEALED_REPETITIONS):
        raise ValueError("repetition must be in 0..99")
    return 520000 + 200 * int(world_index) + int(repetition)


@dataclass(frozen=True)
class ObservedEnvironment:
    """The only environment type accepted by fitting/model-selection APIs."""
    environment_id: str
    graph_population: np.ndarray
    adaptation: np.ndarray
    evaluation: np.ndarray
    iu_weight: np.ndarray
    feature_names: tuple[str, ...]
    confidence_signs: np.ndarray


@dataclass(frozen=True)
class DiagnosticTruth:
    """Inaccessible planted truth joined only after all model selection."""
    adaptation_y: np.ndarray
    evaluation_y: np.ndarray
    adaptation_z: np.ndarray
    evaluation_z: np.ndarray
    covariance: np.ndarray
    target_weight: np.ndarray
    nuisance_weight: np.ndarray
    centre: np.ndarray


@dataclass(frozen=True)
class StructuralSamplingAudit:
    """Label-free evidence that prompt-level deterministic reduction occurred."""
    prompt_count: int
    candidates_per_prompt: int
    selected_ordinal_counts: tuple[int, ...]
    selected_ordinals_sha256: str


@dataclass(frozen=True)
class SyntheticWorld:
    world: int
    seed: int
    observed: tuple[ObservedEnvironment, ...]
    diagnostics: tuple[DiagnosticTruth, ...]
    sampling_audits: tuple[StructuralSamplingAudit, ...]
    true_support: np.ndarray
    description: str
    semantic_swap: bool = False


@dataclass(frozen=True)
class GraphFit:
    support: np.ndarray
    quotient: "DuplicateQuotient"
    penalty: float
    initial_edge_count: int
    final_edge_count: int
    environment_fits: tuple[SparseMixtureFit, ...]
    diagnostics: dict


@dataclass(frozen=True)
class HeldDirectionFit:
    standardization: Standardization
    mixture: SparseMixtureFit
    alpha_weights: dict[float, np.ndarray]
    alpha_corrections: dict[float, np.ndarray]
    alpha_likelihoods: dict[float, float]
    alpha_scores: dict[float, np.ndarray]
    iu_scores: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class HeldControlFit:
    sparse_alpha_zero_log_likelihood: float
    one_gaussian_log_likelihood: float
    unanchored_mixture_log_likelihood: float
    diagnostics: dict


@dataclass(frozen=True)
class DuplicateQuotient:
    """Mean-plus-contrast redundancy transform with an exact affine inverse."""
    groups: tuple[tuple[int, ...], ...]
    transform: np.ndarray
    reducer: np.ndarray
    correction_mask: np.ndarray
    contrast_coordinates: tuple[int, ...]
    original_feature_names: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    training_average_covariance: np.ndarray
    threshold: float

    @property
    def original_dimension(self) -> int:
        return int(self.transform.shape[0])

    @property
    def quotient_dimension(self) -> int:
        return int(self.transform.shape[1])

    def reduce_matrix(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) @ self.transform

    def reduce_weight(self, weight: np.ndarray) -> np.ndarray:
        return self.reducer @ np.asarray(weight, dtype=float)

    def expand_correction(self, weight: np.ndarray) -> np.ndarray:
        return self.transform @ np.asarray(weight, dtype=float)


def anchored_redundancy_direction(
    mixture: SparseMixtureFit, iu_weight: np.ndarray,
    quotient: DuplicateQuotient, alpha: float,
):
    """IU-orthogonal correction constrained away from redundancy contrasts."""
    if quotient.correction_mask.all():
        return anchored_direction(mixture, iu_weight, alpha)
    covariance = mixture.covariance
    iu_weight = np.asarray(iu_weight, dtype=float)
    candidate = mixture.discriminant.copy()
    candidate[~quotient.correction_mask] = 0.0
    iu_norm = float(iu_weight @ covariance @ iu_weight)
    candidate_norm = float(candidate @ covariance @ candidate)
    orientation = float(candidate @ covariance @ iu_weight)
    tolerance = 1e-10 * np.sqrt(max(iu_norm * candidate_norm, 0.0))
    if iu_norm <= 1e-12 or candidate_norm <= 1e-12 or abs(orientation) <= tolerance:
        return iu_weight.copy(), np.zeros_like(iu_weight), {
            "orientation_inner_product": orientation,
            "degenerate_mixture_direction": True,
            "degeneracy_reason": "zero_or_unorientable_redundancy_safe_evidence",
            "iu_correction_covariance": 0.0,
            "contrast_correction_max_abs": 0.0,
        }
    if orientation < 0:
        candidate = -candidate
    candidate *= np.sqrt(iu_norm / candidate_norm)
    anchor_basis = iu_weight.copy()
    anchor_basis[~quotient.correction_mask] = 0.0
    denominator = float(iu_weight @ covariance @ anchor_basis)
    if abs(denominator) <= tolerance:
        return iu_weight.copy(), np.zeros_like(iu_weight), {
            "orientation_inner_product": orientation,
            "degenerate_mixture_direction": True,
            "degeneracy_reason": "no_allowed_iu_orthogonal_projection",
            "iu_correction_covariance": 0.0,
            "contrast_correction_max_abs": 0.0,
        }
    correction = candidate - anchor_basis * float(
        iu_weight @ covariance @ candidate
    ) / denominator
    correction[~quotient.correction_mask] = 0.0
    return iu_weight + float(alpha) * correction, correction, {
        "orientation_inner_product": orientation,
        "degenerate_mixture_direction": False,
        "iu_correction_covariance": float(iu_weight @ covariance @ correction),
        "contrast_correction_max_abs": float(np.max(
            np.abs(correction[~quotient.correction_mask])
        )),
        "alpha": float(alpha),
    }


def _build_duplicate_quotient(
    groups: Sequence[Sequence[int]], *, feature_names: Sequence[str],
    training_average_covariance: np.ndarray, threshold: float,
) -> DuplicateQuotient:
    """Build a full-rank canonical mean/contrast coordinate system."""
    names = tuple(str(value) for value in feature_names)
    p = len(names)
    covariance = np.asarray(training_average_covariance, dtype=float)
    if covariance.shape != (p, p):
        raise ValueError("training covariance and feature names disagree")
    canonical_groups = tuple(sorted(
        (tuple(sorted((int(index) for index in group), key=lambda index: names[index]))
         for group in groups),
        key=lambda group: tuple(names[index] for index in group),
    ))
    transform_columns, reducer_rows, mask, contrasts, coordinate_names = [], [], [], [], []
    for group in canonical_groups:
        members = np.asarray(group, dtype=int)
        member_names = tuple(names[index] for index in group)
        mean = np.zeros(p); mean[members] = 1.0 / len(group)
        dual_mean = np.zeros(p); dual_mean[members] = 1.0
        transform_columns.append(mean); reducer_rows.append(dual_mean); mask.append(True)
        coordinate_names.append("mean(" + ",".join(member_names) + ")")
        for level in range(1, len(group)):
            base = np.zeros(p)
            base[members[:level]] = 1.0 / level
            base[members[level]] = -1.0
            base /= np.linalg.norm(base)
            variance = float(base @ covariance @ base)
            # Retain even a graph-train-exact contrast.  A unit scale for a
            # numerically zero variance keeps alpha=0 deployable if a held
            # environment later departs from the empirical equality.
            scale = 1.0 if variance <= 1e-12 else 1.0 / np.sqrt(variance)
            transform_columns.append(base * scale)
            reducer_rows.append(base / scale)
            contrasts.append(len(mask)); mask.append(False)
            coordinate_names.append(
                "contrast(" + ",".join(member_names[:level]) + ";"
                + member_names[level] + ")"
            )
    transform = np.column_stack(transform_columns)
    reducer = np.vstack(reducer_rows)
    if transform.shape != (p, p) or np.linalg.matrix_rank(transform) != p:
        raise RuntimeError("redundancy transform is not a full-rank coordinate system")
    if np.max(np.abs(transform @ reducer - np.eye(p))) > 1e-10:
        raise RuntimeError("redundancy transform failed affine inverse check")
    return DuplicateQuotient(
        canonical_groups, transform, reducer, np.asarray(mask, dtype=bool),
        tuple(contrasts), names, tuple(coordinate_names), covariance.copy(),
        float(threshold),
    )


def discover_duplicate_quotient(
    standardized_environments: Sequence[np.ndarray], *, feature_names: Sequence[str],
    threshold: float = 0.998,
) -> DuplicateQuotient:
    """Join coordinates correlated above threshold in every training environment."""
    matrices = [np.asarray(value, dtype=float) for value in standardized_environments]
    if not matrices:
        raise ValueError("duplicate quotient requires at least one environment")
    p = matrices[0].shape[1]
    if any(value.ndim != 2 or value.shape[1] != p for value in matrices):
        raise ValueError("duplicate quotient environments have inconsistent dimensions")
    names = tuple(str(value) for value in feature_names)
    if len(names) != p or len(set(names)) != p:
        raise ValueError("immutable feature names must be unique and match matrices")
    correlations = [np.corrcoef(value, rowvar=False) for value in matrices]
    parent = list(range(p))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for left in range(p):
        for right in range(left + 1, p):
            if min(value[left, right] for value in correlations) >= float(threshold):
                union(left, right)
    components = {}
    for index in range(p):
        components.setdefault(find(index), []).append(index)
    groups = tuple(tuple(value) for value in components.values())
    average_covariance = np.mean(
        [np.cov(value, rowvar=False, ddof=0) for value in matrices], axis=0
    )
    return _build_duplicate_quotient(
        groups, feature_names=names,
        training_average_covariance=average_covariance, threshold=threshold,
    )


def _balanced_bit(rng: np.random.Generator, n: int) -> np.ndarray:
    values = np.tile(np.asarray([-1.0, 1.0]), (n + 1) // 2)[:n]
    return values[rng.permutation(n)]


def _unit_metric(weight: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    norm = float(weight @ covariance @ weight)
    if norm <= 1e-14:
        raise ValueError("cannot normalize a zero metric vector")
    return weight / np.sqrt(norm)


def _orthogonal_noise(rng, target, covariance):
    value = rng.normal(size=len(target))
    value -= target * float(target @ covariance @ value) / float(target @ covariance @ target)
    return _unit_metric(value, covariance)


def true_precision(p: int = 17, *, independent: bool = False) -> np.ndarray:
    if independent:
        return np.eye(p)
    adjacency = np.zeros((p, p), dtype=float)
    for j in range(p - 1):
        adjacency[j, j + 1] = adjacency[j + 1, j] = 1.0
    for j in (0, 3, 6, 9, 12):
        if j + 3 < p:
            adjacency[j, j + 3] = adjacency[j + 3, j] = 1.0
    return np.eye(p) + 0.18 * (np.diag(adjacency.sum(axis=1)) - adjacency)


def _base_vectors(p: int, *, anti_majority: bool = False):
    if anti_majority:
        target = np.asarray([-1 / np.sqrt(p)] * min(10, p)
                            + [1 / np.sqrt(p)] * max(0, p - 10))
    else:
        target = np.zeros(p)
        for index, value in zip((0, 4, 8, 12), (1.0, 0.8, -0.7, 0.6)):
            if index < p:
                target[index] = value
    nuisance = np.zeros(p)
    for index, value in zip((2, 9, 15), (1.0, -0.9, 0.7)):
        if index < p:
            nuisance[index] = value
    return target, nuisance


def _world_sizes(world: int) -> list[int]:
    if world == 3:
        return [500] * 11 + [300] * 4 + [500, 500, 198] + [500] * 5
    return [800] * 12


def _split_indices(world: int, seed: int, environment: int, n: int):
    prefix = "A5-small-n" if world == 3 else "A5-synthetic"
    hashes = [
        hashlib.sha256(f"{prefix}\0{seed}\0{environment}\0{i}".encode()).digest()
        for i in range(n)
    ]
    order = np.asarray(sorted(range(n), key=lambda i: (hashes[i], i)), dtype=int)
    return np.sort(order[: n // 2]), np.sort(order[n // 2 :])


def _augment_duplicate(X, covariance, variant, rng):
    if variant == "exact":
        duplicate = X[:, 0].copy()
        rho = 1.0
    elif variant == "near":
        eta = rng.normal(size=len(X))
        eta = (eta - eta.mean()) / eta.std()
        source = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        duplicate = X[:, 0].mean() + X[:, 0].std() * (
            0.999 * source + np.sqrt(1 - 0.999**2) * eta
        )
        rho = 0.999
    else:
        raise ValueError("world 6 requires duplicate_variant exact or near")
    augmented = np.empty((len(covariance) + 1, len(covariance) + 1))
    augmented[:-1, :-1] = covariance
    augmented[-1, :-1] = rho * covariance[0, :]
    augmented[:-1, -1] = rho * covariance[:, 0]
    augmented[-1, -1] = covariance[0, 0]
    return np.column_stack([X, duplicate]), augmented


def _select_primary_response_ordinals(environment_id: str, prompt_count: int, k: int):
    ordinals = []
    for prompt in range(int(prompt_count)):
        content_group = hashlib.sha256(
            f"A5-synthetic-content\0{prompt}".encode()
        ).hexdigest()
        ranked = []
        for ordinal in range(int(k)):
            payload = (
                "A5-primary-response\0" + environment_id + "\0" + content_group
                + "\0" + str(prompt) + "\0" + str(ordinal)
            )
            ranked.append((hashlib.sha256(payload.encode()).hexdigest(), ordinal))
        ordinals.append(min(ranked)[1])
    return np.asarray(ordinals, dtype=int)


def simulate_synthetic_world(world: int, seed: int, *, duplicate_variant=None,
                             semantic_swap: bool = False) -> SyntheticWorld:
    if int(world) not in range(1, 12):
        raise ValueError("world must be in 1..11")
    if world == 6 and duplicate_variant not in {"exact", "near"}:
        raise ValueError("world 6 requires exact or near duplicate_variant")
    if world != 6 and duplicate_variant is not None:
        raise ValueError("duplicate_variant is legal only for world 6")
    if semantic_swap and world != 11:
        raise ValueError("semantic_swap is legal only for world 11")
    p = 17
    rng = np.random.default_rng(int(seed))
    omega0 = true_precision(p, independent=(world == 2))
    true_support = np.abs(omega0) > 0
    b_target, b_nuisance = _base_vectors(p, anti_majority=(world == 7))
    d_target, d_nuisance = (1.0, 1.8) if world in {8, 11} else (1.4, 0.0)
    if world == 9:
        d_target = 1.0
    if world == 10:
        d_target = d_nuisance = 0.0
    observed, diagnostics, sampling_audits = [], [], []
    for environment, n in enumerate(_world_sizes(world)):
        erng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        scales = np.exp(0.10 * erng.normal(size=p))
        covariance = np.diag(scales) @ np.linalg.inv(omega0) @ np.diag(scales)
        precision = np.linalg.inv(covariance)
        centre = 0.3 * erng.normal(size=p)
        w_target = _unit_metric(precision @ (np.diag(scales) @ b_target), covariance)
        w_nuisance = _unit_metric(precision @ (np.diag(scales) @ b_nuisance), covariance)
        if world == 2:
            iu_weight = w_target.copy()
        else:
            anchor_base = w_nuisance if world == 11 else w_target
            iu_weight = _unit_metric(
                anchor_base + 0.8 * _orthogonal_noise(erng, anchor_base, covariance),
                covariance,
            )
        local_dz = ((-1.8, 0.0, 1.8)[environment % 3] if world == 9 else d_nuisance)
        y, z = _balanced_bit(erng, n), _balanced_bit(erng, n)
        noise = erng.multivariate_normal(np.zeros(p), covariance, size=n)
        if world == 4:
            noise *= np.sqrt(5 / erng.chisquare(5, size=n))[:, None]
        elif world == 5:
            noise *= np.sqrt(np.where(y < 0, 0.65, 1.35))[:, None]
        X = (centre + y[:, None] * (covariance @ w_target) * d_target / 2
             + z[:, None] * (covariance @ w_nuisance) * local_dz / 2 + noise)
        # World 8 explicitly constructs half K=1 and half K=10 observed response
        # pools.  The deterministic primary rule selects response zero per prompt;
        # all ten responses of a K=10 prompt share Z.  Structural X therefore has
        # exactly 800 independent prompt rows in both halves.
        environment_id = f"synthetic_{environment:02d}"
        candidate_count = 10 if world == 8 and environment >= 6 else 1
        selected_ordinals = _select_primary_response_ordinals(
            environment_id, n, candidate_count
        )
        if candidate_count == 10:
            response_noise = erng.multivariate_normal(np.zeros(p), covariance, size=(n, 10))
            response_pool = (centre + y[:, None, None] * (covariance @ w_target)[None, None, :]
                             * d_target / 2
                             + z[:, None, None] * (covariance @ w_nuisance)[None, None, :]
                             * d_nuisance / 2 + response_noise)
            X = response_pool[np.arange(n), selected_ordinals, :]
        if world == 6:
            X, covariance = _augment_duplicate(X, covariance, duplicate_variant, erng)
            centre = np.r_[centre, centre[0]]
            w_target = np.r_[w_target, 0.0]
            w_nuisance = np.r_[w_nuisance, 0.0]
            iu_weight = np.r_[iu_weight, 0.0]
        adaptation_indices, evaluation_indices = _split_indices(world, seed, environment, n)
        if semantic_swap:
            diagnostic_y, diagnostic_z = z, y
            diagnostic_target, diagnostic_nuisance = w_nuisance, w_target
        else:
            diagnostic_y, diagnostic_z = y, z
            diagnostic_target, diagnostic_nuisance = w_target, w_nuisance
        feature_names = tuple(CORE_FEATURES)
        confidence_signs = np.asarray(
            [CONFIDENCE_FEATURE_SIGNS_V1[name] for name in feature_names], dtype=float
        )
        if world == 6:
            feature_names = feature_names + ("duplicate(" + feature_names[0] + ")",)
            confidence_signs = np.r_[confidence_signs, confidence_signs[0]]
        observed.append(ObservedEnvironment(
            environment_id=environment_id, graph_population=X,
            adaptation=X[adaptation_indices], evaluation=X[evaluation_indices],
            iu_weight=iu_weight,
            feature_names=feature_names, confidence_signs=confidence_signs,
        ))
        diagnostics.append(DiagnosticTruth(
            adaptation_y=diagnostic_y[adaptation_indices],
            evaluation_y=diagnostic_y[evaluation_indices],
            adaptation_z=diagnostic_z[adaptation_indices],
            evaluation_z=diagnostic_z[evaluation_indices],
            covariance=covariance, target_weight=diagnostic_target,
            nuisance_weight=diagnostic_nuisance, centre=centre,
        ))
        counts = np.bincount(selected_ordinals, minlength=candidate_count)
        sampling_audits.append(StructuralSamplingAudit(
            prompt_count=int(n), candidates_per_prompt=int(candidate_count),
            selected_ordinal_counts=tuple(int(value) for value in counts),
            selected_ordinals_sha256=hashlib.sha256(
                np.ascontiguousarray(selected_ordinals).view(np.uint8)
            ).hexdigest(),
        ))
    descriptions = {1: "favorable_sparse_gaussian", 2: "independent_exact_iu",
                    3: "a0_small_n", 4: "student_t5", 5: "heteroscedastic",
                    6: f"duplicates_{duplicate_variant}", 7: "anti_majority",
                    8: "nuisance_dominant_prompt_shared", 9: "environment_specific_nuisance",
                    10: "one_gaussian_no_latent", 11: "anchor_points_to_nuisance"}
    return SyntheticWorld(int(world), int(seed), tuple(observed), tuple(diagnostics),
                          tuple(sampling_audits), true_support,
                          descriptions[int(world)], bool(semantic_swap))


def _anchor_in_standard_coordinates(raw_weight, standardization):
    return standardization.scale * standardization.signs * raw_weight


def _require_mixture(fit, context):
    if not fit.converged:
        raise RuntimeError(f"CLOSE_NUMERICAL_NONCONVERGENCE: {context}")
    return fit


def _environment_contract(environment: ObservedEnvironment, p: int | None = None):
    if not isinstance(environment, ObservedEnvironment):
        raise TypeError("fitting accepts only ObservedEnvironment values")
    dimension = environment.graph_population.shape[1]
    if p is not None and dimension != int(p):
        raise ValueError("environment feature dimensions disagree")
    if (len(environment.feature_names) != dimension
            or len(set(environment.feature_names)) != dimension
            or np.asarray(environment.confidence_signs).shape != (dimension,)
            or not np.isin(environment.confidence_signs, (-1.0, 1.0)).all()):
        raise ValueError("environment feature-name/sign contract is invalid")
    return dimension


def fit_graph_pipeline(environments: Sequence[ObservedEnvironment], penalty: float) -> GraphFit:
    if not environments or any(not isinstance(value, ObservedEnvironment)
                               for value in environments):
        raise TypeError("graph fitting accepts only ObservedEnvironment values")
    p = _environment_contract(environments[0])
    names = environments[0].feature_names
    signs = np.asarray(environments[0].confidence_signs, dtype=float)
    standardized, raw_anchors = [], []
    for environment in environments:
        _environment_contract(environment, p)
        if environment.feature_names != names or not np.array_equal(
            environment.confidence_signs, signs
        ):
            raise ValueError("graph environments do not share the frozen feature contract")
        standardization, X = fit_standardization(environment.graph_population, signs)
        anchor = _anchor_in_standard_coordinates(environment.iu_weight, standardization)
        standardized.append(X); raw_anchors.append(anchor)
    quotient = discover_duplicate_quotient(standardized, feature_names=names)
    transformed = [quotient.reduce_matrix(value) for value in standardized]
    anchors = [quotient.reduce_weight(value) for value in raw_anchors]
    q = quotient.quotient_dimension
    initial = []
    for X, anchor, environment in zip(transformed, anchors, environments):
        initial.append(_require_mixture(
            fit_sparse_equal_covariance_mixture(X, diagonal_support(q), anchor),
            f"diagonal initialization/{environment.environment_id}",
        ))
    first = support_from_residual_correlations(
        [within_component_correlation(fit, X) for fit, X in zip(initial, transformed)], penalty
    )
    if not first.converged:
        raise RuntimeError(f"CLOSE_NUMERICAL_NONCONVERGENCE: graphical lasso {first}")
    first_sparse = [
        _require_mixture(fit_sparse_equal_covariance_mixture(X, first.support, anchor),
                         f"first sparse/{environment.environment_id}")
        for X, anchor, environment in zip(transformed, anchors, environments)
    ]
    final = support_from_residual_correlations(
        [within_component_correlation(fit, X) for fit, X in zip(first_sparse, transformed)], penalty
    )
    if not final.converged:
        raise RuntimeError(f"CLOSE_NUMERICAL_NONCONVERGENCE: graphical lasso {final}")
    fits = tuple(
        _require_mixture(fit_sparse_equal_covariance_mixture(X, final.support, anchor),
                         f"final sparse/{environment.environment_id}")
        for X, anchor, environment in zip(transformed, anchors, environments)
    )
    return GraphFit(final.support, quotient, float(penalty),
                    int((first.support.sum()-q)//2),
                    int((final.support.sum()-q)//2), fits,
                    {"first_graphical_lasso": first, "final_graphical_lasso": final,
                     "duplicate_groups": quotient.groups,
                     "original_dimension": p, "quotient_dimension": q})


def fit_fixed_graph_pipeline(environments: Sequence[ObservedEnvironment], support: np.ndarray,
                             *, quotient: DuplicateQuotient, name: str) -> GraphFit:
    """Refit all graph-training mixtures under one already frozen support."""
    if any(not isinstance(value, ObservedEnvironment) for value in environments):
        raise TypeError("fixed graph fitting accepts only ObservedEnvironment values")
    support = np.asarray(support, dtype=bool)
    fits = []
    for environment in environments:
        _environment_contract(environment, quotient.original_dimension)
        if environment.feature_names != quotient.original_feature_names:
            raise ValueError("fixed graph environment feature names disagree")
        standardization, X = fit_standardization(
            environment.graph_population, environment.confidence_signs
        )
        X = quotient.reduce_matrix(X)
        anchor = quotient.reduce_weight(
            _anchor_in_standard_coordinates(environment.iu_weight, standardization)
        )
        fits.append(_require_mixture(
            fit_sparse_equal_covariance_mixture(X, support, anchor),
            f"{name}/{environment.environment_id}",
        ))
    q = quotient.quotient_dimension
    return GraphFit(
        support, quotient, float("nan"), int((support.sum()-q)//2),
        int((support.sum()-q)//2), tuple(fits), {"control_name": str(name)},
    )


def degree_matched_random_supports(
    support: np.ndarray, *, split_seed: int, penalty: float, count: int = 32,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, ...]:
    """Generate the preregistered unique degree-preserving double-edge swaps."""
    support = np.asarray(support, dtype=bool)
    p = support.shape[0]
    if support.shape != (p, p) or not np.array_equal(support, support.T):
        raise ValueError("support must be square and symmetric")
    names = tuple(str(value) for value in feature_names)
    if len(names) != p or len(set(names)) != p:
        raise ValueError("feature_names must be unique and match support")
    canonical_order = np.asarray(sorted(range(p), key=lambda index: names[index]), dtype=int)
    inverse_order = np.argsort(canonical_order)
    canonical = support[np.ix_(canonical_order, canonical_order)]
    base_edges = tuple((i, j) for i in range(p) for j in range(i+1, p) if canonical[i, j])
    edge_count = len(base_edges)
    if edge_count < 2:
        raise RuntimeError("CLOSE_INADEQUATE_RANDOM_GRAPH_CONTROL: fewer than two edges")
    outputs, seen = [], {np.packbits(support).tobytes()}
    accepted_target = max(100, 20 * edge_count)
    attempt_limit = 1000 * max(edge_count, 1)
    for arm in range(int(count)):
        penalty_bytes = float(penalty).hex()
        seed_payload = (
            f"A5-random-support\0{int(split_seed)}\0{penalty_bytes}\0{arm}"
        )
        seed = int.from_bytes(hashlib.sha256(seed_payload.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        edges = set(base_edges)
        accepted = attempts = 0
        while accepted < accepted_target and attempts < attempt_limit:
            attempts += 1
            ordered = sorted(edges)
            chosen = rng.choice(len(ordered), size=2, replace=False)
            (a, b), (c, d) = ordered[int(chosen[0])], ordered[int(chosen[1])]
            if len({a, b, c, d}) != 4:
                continue
            proposed = ((min(a, c), max(a, c)), (min(b, d), max(b, d))) if rng.integers(2) == 0 else (
                (min(a, d), max(a, d)), (min(b, c), max(b, c))
            )
            if proposed[0] == proposed[1] or any(edge in edges for edge in proposed):
                continue
            edges.remove((a, b)); edges.remove((c, d))
            edges.update(proposed); accepted += 1
        if accepted != accepted_target:
            raise RuntimeError("CLOSE_INADEQUATE_RANDOM_GRAPH_CONTROL: swap attempts exhausted")
        candidate = np.eye(p, dtype=bool)
        for left, right in edges:
            candidate[left, right] = candidate[right, left] = True
        candidate = candidate[np.ix_(inverse_order, inverse_order)]
        key = np.packbits(candidate).tobytes()
        if key in seen:
            raise RuntimeError("CLOSE_INADEQUATE_RANDOM_GRAPH_CONTROL: duplicate support")
        seen.add(key); outputs.append(candidate)
    return tuple(outputs)


def _held_control_fit(environment: ObservedEnvironment, graph: GraphFit) -> HeldControlFit:
    """Fit capacity-identical alpha-zero, one-Gaussian, and unanchored controls."""
    p = environment.adaptation.shape[1]
    _environment_contract(environment, graph.quotient.original_dimension)
    standardization, adaptation_original = fit_standardization(
        environment.adaptation, environment.confidence_signs
    )
    evaluation_original = standardization.transform(environment.evaluation)
    adaptation = graph.quotient.reduce_matrix(adaptation_original)
    evaluation = graph.quotient.reduce_matrix(evaluation_original)
    q = graph.quotient.quotient_dimension
    anchor = graph.quotient.reduce_weight(
        _anchor_in_standard_coordinates(environment.iu_weight, standardization)
    )
    anchored = _require_mixture(
        fit_sparse_equal_covariance_mixture(adaptation, graph.support, anchor),
        f"control anchored/{environment.environment_id}",
    )
    weight0, _, _ = anchored_direction(anchored, anchor, 0.0)
    constrained0 = fit_constrained_direction_mixture(
        adaptation, anchored.covariance, anchored.precision, weight0
    )
    if not constrained0.converged:
        raise RuntimeError("CLOSE_NUMERICAL_NONCONVERGENCE: alpha-zero control")

    centre = adaptation.mean(axis=0)
    residual = adaptation - centre
    empirical = residual.T @ residual / len(adaptation)
    empirical[np.diag_indices(q)] += 1e-8
    gaussian_precision = fit_fixed_support_precision(empirical, graph.support)
    if not gaussian_precision.converged:
        raise RuntimeError("CLOSE_NUMERICAL_NONCONVERGENCE: one-Gaussian control")
    one_gaussian = float(np.mean(gaussian_log_density(
        evaluation, centre, gaussian_precision.precision
    )) / q)

    covariance = np.cov(adaptation, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    pca_anchor = eigenvectors[:, int(np.argmax(eigenvalues))]
    pivot = int(np.argmax(np.abs(pca_anchor)))
    if pca_anchor[pivot] < 0:
        pca_anchor = -pca_anchor
    unanchored = _require_mixture(
        fit_sparse_equal_covariance_mixture(adaptation, graph.support, pca_anchor),
        f"control unanchored/{environment.environment_id}",
    )
    unanchored_ll = float(np.mean(mixture_log_density(
        evaluation, unanchored.centre, unanchored.delta,
        unanchored.precision, unanchored.prior,
    )) / q)
    return HeldControlFit(
        sparse_alpha_zero_log_likelihood=(
            held_mean_log_likelihood(evaluation, constrained0, anchored.precision) / q
        ),
        one_gaussian_log_likelihood=one_gaussian,
        unanchored_mixture_log_likelihood=unanchored_ll,
        diagnostics={"unanchored_initialization": "leading_pca_loading",
                     "quotient_dimension": q},
    )


def select_fixed_support_alpha(
    graph_environments: Sequence[ObservedEnvironment],
    validation_environments: Sequence[ObservedEnvironment],
    support: np.ndarray, *, quotient: DuplicateQuotient, alphas=ALPHAS,
    name: str = "fixed_support",
):
    graph = fit_fixed_graph_pipeline(
        graph_environments, support, quotient=quotient, name=name
    )
    held = [fit_held_directions(environment, graph, alphas=alphas)
            for environment in validation_environments]
    records = []
    for alpha in alphas:
        per_environment = tuple(
            float(fit.alpha_likelihoods[float(alpha)]) for fit in held
        )
        records.append({
            "alpha": float(alpha),
            "mean_log_likelihood": float(np.mean(per_environment)),
            "per_environment_log_likelihood": per_environment,
        })
    empirical_best = max(
        records, key=lambda value: (value["mean_log_likelihood"], -value["alpha"])
    )
    best_values = np.asarray(
        empirical_best["per_environment_log_likelihood"], dtype=float
    )
    admissible = []
    for value in records:
        differences = best_values - np.asarray(
            value["per_environment_log_likelihood"], dtype=float
        )
        standard_error = (float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
                          if len(differences) > 1 else 0.0)
        value["paired_standard_error_from_empirical_best"] = standard_error
        if float(np.mean(differences)) <= standard_error + 1e-12:
            admissible.append(value)
    best = min(admissible, key=lambda value: (
        value["alpha"], -value["mean_log_likelihood"]
    ))
    return graph, float(best["alpha"]), records


def select_diagonal_control(
    graph_environments: Sequence[ObservedEnvironment],
    validation_environments: Sequence[ObservedEnvironment], *, alphas=ALPHAS,
):
    names = graph_environments[0].feature_names
    standardized = [fit_standardization(
        value.graph_population, value.confidence_signs
    )[1] for value in graph_environments]
    quotient = discover_duplicate_quotient(standardized, feature_names=names)
    return select_fixed_support_alpha(
        graph_environments, validation_environments,
        diagonal_support(quotient.quotient_dimension), quotient=quotient,
        alphas=alphas, name="diagonal_control",
    )


def select_random_graph_control(
    graph_environments: Sequence[ObservedEnvironment],
    validation_environments: Sequence[ObservedEnvironment], candidate_graph: GraphFit,
    *, split_seed: int, alphas=ALPHAS, arm_count: int = 32,
):
    supports = degree_matched_random_supports(
        candidate_graph.support, split_seed=split_seed,
        penalty=candidate_graph.penalty, count=arm_count,
        feature_names=candidate_graph.quotient.coordinate_names,
    )
    arms = []
    for index, support in enumerate(supports):
        try:
            graph, alpha, records = select_fixed_support_alpha(
                graph_environments, validation_environments, support,
                quotient=candidate_graph.quotient, alphas=alphas,
                name=f"random_control_{index:02d}",
            )
            selected_record = next(
                record for record in records if record["alpha"] == alpha
            )
            value = selected_record["mean_log_likelihood"]
            arms.append({"arm": index, "graph": graph, "alpha": alpha,
                         "mean_log_likelihood": value,
                         "empirical_best_mean_log_likelihood": max(
                             record["mean_log_likelihood"] for record in records
                         ),
                         "records": records,
                         "usable": True})
        except RuntimeError as error:
            arms.append({"arm": index, "graph": None, "alpha": None,
                         "mean_log_likelihood": None, "records": [],
                         "usable": False, "failure": str(error)})
    usable = [value for value in arms if value["usable"]]
    if len(usable) != int(arm_count):
        error = RuntimeError("CLOSE_INADEQUATE_RANDOM_GRAPH_CONTROL")
        error.arm_records = arms
        raise error
    best_value = max(value["mean_log_likelihood"] for value in usable)
    tied = [value for value in usable
            if best_value - value["mean_log_likelihood"] <= 1e-8]
    best = min(tied, key=lambda value: (value["alpha"], value["arm"]))
    return best, arms


def held_mechanism_controls(
    environment: ObservedEnvironment, candidate_graph: GraphFit,
    diagonal_graph: GraphFit, random_graph: GraphFit,
    *, candidate_alpha: float, diagonal_alpha: float, random_alpha: float,
) -> dict:
    """Evaluate every density-mechanism control on one identical held split."""
    candidate = fit_held_directions(
        environment, candidate_graph, alphas=(0.0, candidate_alpha)
    )
    diagonal = fit_held_directions(environment, diagonal_graph, alphas=(diagonal_alpha,))
    random = fit_held_directions(environment, random_graph, alphas=(random_alpha,))
    diagnostics = _held_control_fit(environment, candidate_graph)
    return {
        "candidate_log_likelihood": candidate.alpha_likelihoods[candidate_alpha],
        "capacity_identical_alpha_zero_log_likelihood": candidate.alpha_likelihoods[0.0],
        "diagonal_log_likelihood": diagonal.alpha_likelihoods[diagonal_alpha],
        "random_graph_log_likelihood": random.alpha_likelihoods[random_alpha],
        "one_gaussian_log_likelihood": diagnostics.one_gaussian_log_likelihood,
        "unanchored_mixture_log_likelihood": diagnostics.unanchored_mixture_log_likelihood,
        "candidate_scores": candidate.alpha_scores[candidate_alpha],
        "diagonal_scores": diagonal.alpha_scores[diagonal_alpha],
        "random_scores": random.alpha_scores[random_alpha],
        "iu_scores": candidate.iu_scores,
    }


def fit_held_directions(environment: ObservedEnvironment, graph: GraphFit,
                        *, alphas: Iterable[float] = ALPHAS) -> HeldDirectionFit:
    if not isinstance(environment, ObservedEnvironment):
        raise TypeError("held fitting accepts only ObservedEnvironment")
    if not isinstance(graph, GraphFit):
        raise TypeError("held fitting requires a fitted GraphFit boundary")
    p = _environment_contract(environment)
    if graph.quotient.original_dimension != p:
        raise ValueError("held environment and graph quotient dimensions disagree")
    if environment.feature_names != graph.quotient.original_feature_names:
        raise ValueError("held feature names disagree with graph boundary")
    signs = np.asarray(environment.confidence_signs, dtype=float)
    standardization, adaptation_original = fit_standardization(environment.adaptation, signs)
    evaluation_original = standardization.transform(environment.evaluation)
    anchor_original = _anchor_in_standard_coordinates(environment.iu_weight, standardization)
    adaptation = graph.quotient.reduce_matrix(adaptation_original)
    evaluation = graph.quotient.reduce_matrix(evaluation_original)
    anchor = graph.quotient.reduce_weight(anchor_original)
    mixture = _require_mixture(
        fit_sparse_equal_covariance_mixture(adaptation, graph.support, anchor),
        f"held mixture/{environment.environment_id}",
    )
    weights, corrections, likelihoods, scores, diagnostics = {}, {}, {}, {}, {}
    for alpha in tuple(float(value) for value in alphas):
        weight, correction, diag = anchored_redundancy_direction(
            mixture, anchor, graph.quotient, alpha
        )
        constrained = fit_constrained_direction_mixture(
            adaptation, mixture.covariance, mixture.precision, weight
        )
        if not constrained.converged:
            raise RuntimeError("CLOSE_NUMERICAL_NONCONVERGENCE: constrained alpha")
        expanded_correction = graph.quotient.expand_correction(correction)
        expanded_weight = anchor_original + alpha * expanded_correction
        weights[alpha], corrections[alpha] = expanded_weight, expanded_correction
        likelihoods[alpha] = held_mean_log_likelihood(
            evaluation, constrained, mixture.precision
        ) / graph.quotient.quotient_dimension
        scores[alpha] = evaluation_original @ expanded_weight
        diagnostics[alpha] = {**diag, "constrained_beta": constrained.beta}
    return HeldDirectionFit(standardization, mixture, weights, corrections, likelihoods,
                            scores, evaluation_original @ anchor_original,
                            {"alphas": diagnostics,
                             "duplicate_groups": graph.quotient.groups})


def feature_deletion_indices(seed: int, environment_id: str, p: int, count: int):
    if int(count) not in {0, 1, 2, 3} or int(count) >= int(p):
        raise ValueError("feature deletion count must be 0, 1, 2, or 3 and below p")
    ranked = sorted(
        range(int(p)),
        key=lambda index: (
            hashlib.sha256(
                f"A5-feature-deletion\0{seed}\0{environment_id}\0{index}".encode()
            ).digest(), index,
        ),
    )
    return tuple(sorted(ranked[: int(count)]))


def induce_held_boundary(environment: ObservedEnvironment, graph: GraphFit,
                         *, seed: int, count: int):
    """Project one held environment and induce the already learned graph."""
    p = environment.adaptation.shape[1]
    drop = set(feature_deletion_indices(seed, environment.environment_id, p, count))
    keep = np.asarray([index for index in range(p) if index not in drop], dtype=int)
    if len(keep) == p:
        return environment, graph, keep
    retained_names = tuple(environment.feature_names[index] for index in keep)
    retained_groups_by_name = []
    for group in graph.quotient.groups:
        member_names = tuple(
            graph.quotient.original_feature_names[index]
            for index in group if index in set(keep)
        )
        members = tuple(retained_names.index(name) for name in member_names)
        if members:
            retained_groups_by_name.append(members)
    quotient = _build_duplicate_quotient(
        retained_groups_by_name, feature_names=retained_names,
        training_average_covariance=graph.quotient.training_average_covariance[
            np.ix_(keep, keep)
        ], threshold=graph.quotient.threshold,
    )
    # Pull the learned structural support through the deterministic basis
    # change. For singleton groups this is exactly the principal subgraph; for
    # grouped coordinates it conservatively retains every edge with a nonzero
    # algebraic path into a retained mean/contrast coordinate.
    basis_map = graph.quotient.reducer[:, keep] @ quotient.transform
    support = (
        np.abs(basis_map).T @ graph.support.astype(float) @ np.abs(basis_map)
    ) > 1e-12
    support = np.asarray(support | support.T, dtype=bool)
    np.fill_diagonal(support, True)
    q = quotient.quotient_dimension
    induced_graph = GraphFit(
        support, quotient, graph.penalty, int((support.sum() - q) // 2),
        int((support.sum() - q) // 2), tuple(),
        {**graph.diagnostics, "held_deletion_indices": tuple(sorted(drop)),
         "induced_from_full_graph": True},
    )
    projected = ObservedEnvironment(
        environment.environment_id, environment.graph_population[:, keep],
        environment.adaptation[:, keep], environment.evaluation[:, keep],
        environment.iu_weight[keep],
        retained_names, environment.confidence_signs[keep],
    )
    return projected, induced_graph, keep


def project_diagnostic_truth(truth: DiagnosticTruth, keep: np.ndarray):
    """Recompute the correct marginal Bayes directions after deletion."""
    keep = np.asarray(keep, dtype=int)
    covariance = truth.covariance[np.ix_(keep, keep)]
    target = np.linalg.solve(covariance, (truth.covariance @ truth.target_weight)[keep])
    nuisance = np.linalg.solve(covariance, (truth.covariance @ truth.nuisance_weight)[keep])
    return DiagnosticTruth(
        truth.adaptation_y, truth.evaluation_y, truth.adaptation_z, truth.evaluation_z,
        covariance, target, nuisance, truth.centre[keep],
    )


def select_graph_and_alpha(graph_environments: Sequence[ObservedEnvironment],
                           validation_environments: Sequence[ObservedEnvironment],
                           *, penalties=PENALTIES, alphas=ALPHAS,
                           deletion_seed: int | None = None, deletion_count: int = 0):
    records, graphs = [], {}
    for penalty in tuple(float(value) for value in penalties):
        try:
            graph = fit_graph_pipeline(graph_environments, penalty)
            projected = [
                induce_held_boundary(environment, graph, seed=int(deletion_seed),
                                     count=deletion_count)[:2]
                if deletion_count else (environment, graph)
                for environment in validation_environments
            ]
            held = [fit_held_directions(environment, induced, alphas=alphas)
                    for environment, induced in projected]
        except RuntimeError as error:
            records.append({"penalty": penalty, "alpha": None,
                            "mean_log_likelihood": None,
                            "usable": False, "failure": str(error)})
            continue
        graphs[penalty] = graph
        for alpha in tuple(float(value) for value in alphas):
            per_environment = tuple(float(fit.alpha_likelihoods[alpha]) for fit in held)
            records.append({"penalty": penalty, "alpha": alpha,
                            "mean_log_likelihood": float(np.mean(per_environment)),
                            "per_environment_log_likelihood": per_environment,
                            "usable": True})
    usable = [value for value in records if value.get("usable")]
    if not usable:
        raise RuntimeError("CLOSE_NUMERICAL_NONCONVERGENCE: no usable penalty arm")
    empirical_best = max(
        usable, key=lambda value: (value["mean_log_likelihood"],
                                    -value["alpha"], value["penalty"])
    )
    best_per_environment = np.asarray(
        empirical_best["per_environment_log_likelihood"], dtype=float
    )
    # Pre-seal conservative one-standard-error rule.  A correction is trusted
    # only when its paired likelihood advantage is consistent across validation
    # environments; otherwise the simplest exact-IU alpha is selected.
    admissible = []
    for value in usable:
        differences = best_per_environment - np.asarray(
            value["per_environment_log_likelihood"], dtype=float
        )
        standard_error = (float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
                          if len(differences) > 1 else 0.0)
        value["paired_standard_error_from_empirical_best"] = standard_error
        if float(np.mean(differences)) <= standard_error + 1e-12:
            admissible.append(value)
    best = min(admissible, key=lambda value: (
        value["alpha"], -value["penalty"], -value["mean_log_likelihood"]
    ))
    return graphs[best["penalty"]], float(best["alpha"]), records


def _cos2_metric(left, right, covariance):
    numerator = float(left @ covariance @ right) ** 2
    denominator = float(left @ covariance @ left) * float(right @ covariance @ right)
    return float(numerator / denominator) if denominator > 1e-14 else float("nan")


def _raw_weight(weight, standardization):
    return standardization.signs * weight / standardization.scale


def _support_f1(estimated, truth):
    if estimated.shape != truth.shape:
        # Support recovery is not an estimand in augmented/deleted-coordinate
        # stress worlds. Never pretend a top-left slice is group-aware truth.
        return None
    mask = np.triu(np.ones_like(truth, dtype=bool), 1)
    a, b = estimated[mask], truth[mask]
    tp, fp, fn = np.sum(a & b), np.sum(a & ~b), np.sum(~a & b)
    return float(2*tp / max(2*tp+fp+fn, 1))


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else None


def run_synthetic_repetition(
    world: SyntheticWorld, *, penalties=PENALTIES, alphas=ALPHAS,
    deletion_count: int = 0,
) -> dict:
    rng = np.random.default_rng(world.seed + 17001)
    permutation = rng.permutation(len(world.observed))
    graph_count, validation_count = ((17, 3) if len(permutation) == 23 else (8, 2))
    graph_indices = permutation[:graph_count]
    validation_indices = permutation[graph_count:graph_count+validation_count]
    test_indices = permutation[graph_count+validation_count:]
    graph, alpha, selection = select_graph_and_alpha(
        [world.observed[i] for i in graph_indices],
        [world.observed[i] for i in validation_indices],
        penalties=penalties, alphas=alphas,
        deletion_seed=world.seed, deletion_count=deletion_count,
    )
    projected = [
        induce_held_boundary(world.observed[i], graph, seed=world.seed,
                             count=deletion_count)
        if deletion_count else (
            world.observed[i], graph,
            np.arange(world.observed[i].adaptation.shape[1]),
        )
        for i in test_indices
    ]
    held = [fit_held_directions(environment, induced, alphas=alphas)
            for environment, induced, _ in projected]
    candidate_auc=[]; iu_auc=[]; oracle_auc=[]; final_target=[]; final_nuisance=[]
    correction_target=[]; correction_nuisance=[]; fallback=[]
    for index, fitted, (environment, _, keep) in zip(test_indices, held, projected):
        truth = project_diagnostic_truth(world.diagnostics[index], keep)
        candidate, iu = fitted.alpha_scores[alpha], fitted.iu_scores
        labels = (truth.evaluation_y > 0).astype(int)
        candidate_auc.append(roc_auc_score(labels, candidate)); iu_auc.append(roc_auc_score(labels, iu))
        oracle_auc.append(roc_auc_score(labels, environment.evaluation @ truth.target_weight))
        raw_final = _raw_weight(fitted.alpha_weights[alpha], fitted.standardization)
        raw_correction = _raw_weight(fitted.alpha_corrections[alpha], fitted.standardization)
        covariance = truth.covariance
        target_residual = truth.target_weight - environment.iu_weight * (
            environment.iu_weight @ covariance @ truth.target_weight
        ) / (environment.iu_weight @ covariance @ environment.iu_weight)
        nuisance_residual = truth.nuisance_weight - environment.iu_weight * (
            environment.iu_weight @ covariance @ truth.nuisance_weight
        ) / (environment.iu_weight @ covariance @ environment.iu_weight)
        final_target.append(_cos2_metric(raw_final, truth.target_weight, covariance))
        final_nuisance.append(_cos2_metric(raw_final, truth.nuisance_weight, covariance))
        correction_target.append(_cos2_metric(raw_correction, target_residual, covariance))
        correction_nuisance.append(_cos2_metric(raw_correction, nuisance_residual, covariance))
        fallback.append(float(np.max(np.abs(candidate-iu))) if alpha == 0 else np.nan)
    return {"world": world.world, "description": world.description, "seed": world.seed,
            "deletion_count": int(deletion_count),
            "selected_penalty": graph.penalty, "selected_alpha": alpha,
            "support_f1": _support_f1(graph.support, world.true_support),
            "support_edges": graph.final_edge_count,
            "candidate_auroc": float(np.mean(candidate_auc)), "iu_auroc": float(np.mean(iu_auc)),
            "oracle_auroc": float(np.mean(oracle_auc)),
            "candidate_minus_iu": float(np.mean(candidate_auc)-np.mean(iu_auc)),
            "oracle_minus_iu": float(np.mean(oracle_auc)-np.mean(iu_auc)),
            "final_cos2_target": float(np.nanmean(final_target)),
            "final_cos2_nuisance": float(np.nanmean(final_nuisance)),
            "correction_cos2_target": _finite_mean(correction_target),
            "correction_cos2_nuisance": _finite_mean(correction_nuisance),
            "target_preferred_final": bool(np.nanmean(final_target)>np.nanmean(final_nuisance)),
            "target_preferred_correction": (
                None if _finite_mean(correction_target) is None
                or _finite_mean(correction_nuisance) is None
                else bool(_finite_mean(correction_target)>_finite_mean(correction_nuisance))
            ),
            "fallback_error": float(np.nanmax(fallback)) if np.isfinite(fallback).any() else None,
            "selection_records": selection,
            "test_environment_ids": [world.observed[i].environment_id for i in test_indices]}


def duplicate_pair_diagnostics(
    baseline_world: SyntheticWorld,
    augmented_world: SyntheticWorld,
    *, penalties=PENALTIES,
    alphas=ALPHAS,
) -> dict:
    """Paired world-6 coefficient-mass and score-rank diagnostic."""
    if augmented_world.world != 6 or baseline_world.seed != augmented_world.seed:
        raise ValueError("duplicate diagnostic requires paired seed and world 6 augmentation")
    rng = np.random.default_rng(baseline_world.seed + 17001)
    permutation = rng.permutation(len(baseline_world.observed))
    graph_indices, validation_indices, test_indices = permutation[:8], permutation[8:10], permutation[10:]
    baseline_graph, baseline_alpha, _ = select_graph_and_alpha(
        [baseline_world.observed[i] for i in graph_indices],
        [baseline_world.observed[i] for i in validation_indices],
        penalties=penalties, alphas=alphas,
    )
    augmented_graph, augmented_alpha, _ = select_graph_and_alpha(
        [augmented_world.observed[i] for i in graph_indices],
        [augmented_world.observed[i] for i in validation_indices],
        penalties=penalties, alphas=alphas,
    )
    masses, correlations, same_alpha_correlations = [], [], []
    for index in test_indices:
        base = fit_held_directions(
            baseline_world.observed[index], baseline_graph, alphas=(baseline_alpha,)
        )
        aug = fit_held_directions(
            augmented_world.observed[index], augmented_graph,
            alphas=tuple(sorted({baseline_alpha, augmented_alpha})),
        )
        base_weight = base.alpha_corrections[baseline_alpha]
        aug_weight = aug.alpha_corrections[augmented_alpha]
        masses.append(float(
            (abs(aug_weight[0]) + abs(aug_weight[-1]))
            / max(abs(base_weight[0]), 1e-12)
        ))
        paired_base_scores = (
            base.standardization.transform(baseline_world.observed[index].evaluation)
            @ base.alpha_weights[baseline_alpha]
        )
        paired_aug_scores = aug.alpha_scores[augmented_alpha]
        correlations.append(float(spearmanr(paired_base_scores, paired_aug_scores).statistic))
        same_alpha_correlations.append(float(spearmanr(
            paired_base_scores, aug.alpha_scores[baseline_alpha]
        ).statistic))
    return {
        "baseline_alpha": baseline_alpha,
        "augmented_alpha": augmented_alpha,
        "correction_combined_mass_ratios": masses,
        "score_spearman": correlations,
        "same_alpha_score_spearman": same_alpha_correlations,
        "median_combined_mass_ratio": float(np.median(masses)),
        "median_score_spearman": float(np.median(correlations)),
        "median_same_alpha_score_spearman": float(np.median(same_alpha_correlations)),
        "selected_alpha_absolute_difference": abs(baseline_alpha - augmented_alpha),
    }


__all__ = ["ALPHAS", "DEVELOPMENT_SEED_MAX", "DEVELOPMENT_SEED_MIN", "DiagnosticTruth",
           "DuplicateQuotient", "GraphFit", "HeldControlFit", "HeldDirectionFit",
           "NUISANCE_WORLD_INDEX", "ObservedEnvironment", "PENALTIES",
           "SEALED_REPETITIONS", "SEALED_SEED_MAX", "SEALED_SEED_MIN",
           "StructuralSamplingAudit", "SyntheticWorld",
           "duplicate_pair_diagnostics",
           "degree_matched_random_supports", "discover_duplicate_quotient",
           "feature_deletion_indices", "fit_fixed_graph_pipeline", "fit_graph_pipeline",
           "fit_held_directions", "induce_held_boundary", "project_diagnostic_truth",
           "held_mechanism_controls",
           "run_synthetic_repetition",
           "sealed_world_seed",
           "select_diagonal_control", "select_fixed_support_alpha",
           "select_graph_and_alpha", "select_random_graph_control",
           "simulate_synthetic_world", "true_precision"]
