"""Exploratory continuous latent-factor fusion from second and third moments.

This is *not* an algorithm from Ibrahim et al. (2025), and it is not a literal
extension of Tenzer et al.'s U-PCR model.  The categorical Dawid--Skene tensor
identifiability results reviewed by Ibrahim et al. do not transfer directly to
continuous regressors.  This module implements an auditable premise test:

1. covariance defines a low-dimensional feature subspace;
2. a symmetric CP model separates that subspace using its third central moment;
3. the component closest to IU-PCR's label-free reliability vector is treated
   as the target candidate;
4. the other components are subtracted and ordinary IU-PCR is fitted again.

Every model-selection quantity is label-free.  Ambiguous or unstable fits are
expected and must fall back to the unchanged IU-PCR input matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Callable

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares, linear_sum_assignment


EPS = 1e-12


def zscore_columns(values: np.ndarray) -> np.ndarray:
    """Column-standardize a finite sample-by-feature matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 3:
        raise ValueError("values must have shape (samples>=3, features>=3)")
    if not np.isfinite(values).all():
        raise ValueError("values contain non-finite entries")
    centred = values - values.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale < EPS] = 1.0
    return centred / scale


def third_central_moment(values: np.ndarray) -> np.ndarray:
    """Return E[(X-EX) outer (X-EX) outer (X-EX)]."""
    values = np.asarray(values, dtype=float)
    centred = values - values.mean(axis=0, keepdims=True)
    return np.einsum(
        "ni,nj,nk->ijk", centred, centred, centred, optimize=True
    ) / max(len(centred), 1)


def all_distinct_third_moments(values: np.ndarray) -> np.ndarray:
    """Third moments whose three feature indices are different.

    Diagonal tensor entries contain feature-specific marginal skew.  The
    all-distinct entries are the closest continuous analogue of cross-worker
    moments and provide a stricter split-half stability diagnostic.
    """
    values = np.asarray(values, dtype=float)
    centred = values - values.mean(axis=0, keepdims=True)
    triples = list(itertools.combinations(range(values.shape[1]), 3))
    if not triples:
        return np.empty(0, dtype=float)
    return np.asarray([
        np.mean(centred[:, i] * centred[:, j] * centred[:, k])
        for i, j, k in triples
    ], dtype=float)


def distinct_triples(n_features: int) -> np.ndarray:
    """Return the canonical i<j<k index array used by masked CP."""
    return np.asarray(
        list(itertools.combinations(range(int(n_features)), 3)), dtype=int
    )


def masked_projected_third_tensor(
    values: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Project a third-moment tensor after zeroing every repeated-index entry."""
    tensor = third_central_moment(values)
    m = tensor.shape[0]
    index = np.arange(m)
    distinct = (
        (index[:, None, None] != index[None, :, None])
        & (index[:, None, None] != index[None, None, :])
        & (index[None, :, None] != index[None, None, :])
    )
    tensor = np.where(distinct, tensor, 0.0)
    return np.einsum(
        "ia,jb,kc,ijk->abc",
        basis,
        basis,
        basis,
        tensor,
        optimize=True,
    )


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    if left.shape != right.shape or left.size < 2:
        return 0.0
    if np.std(left) < EPS or np.std(right) < EPS:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def covariance_basis(
    values: np.ndarray,
    dimension: int = 6,
    *,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Leading covariance basis, returned in descending eigenvalue order.

    ``standardize=False`` preserves a previously frozen feature scale while
    still centering the supplied sample. This is required for held-out moment
    validation: the covariance basis and third moments must use one coordinate
    system.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 3:
        raise ValueError("values must have shape (samples>=3, features>=3)")
    if not np.isfinite(values).all():
        raise ValueError("values contain non-finite entries")
    values = (
        zscore_columns(values)
        if standardize
        else values - values.mean(axis=0, keepdims=True)
    )
    m = values.shape[1]
    dimension = min(max(1, int(dimension)), m)
    covariance = values.T @ values / len(values)
    eigenvalues, basis = eigh(
        covariance, subset_by_index=[m - dimension, m - 1]
    )
    order = np.argsort(eigenvalues)[::-1]
    return basis[:, order], eigenvalues[order]


def _tensor_from_components(directions: np.ndarray, strengths: np.ndarray) -> np.ndarray:
    return np.einsum(
        "r,ir,jr,kr->ijk",
        strengths,
        directions,
        directions,
        directions,
        optimize=True,
    )


def _normalise_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / np.maximum(np.linalg.norm(values, axis=0, keepdims=True), EPS)


@dataclass
class CPSeedFit:
    seed: int
    directions: np.ndarray
    strengths: np.ndarray
    relative_error: float
    converged: bool
    n_evaluations: int


@dataclass
class SymmetricCPFit:
    rank: int
    directions: np.ndarray
    strengths: np.ndarray
    relative_error: float
    converged: bool
    best_seed: int
    seed_agreement: float
    seed_fits: list[CPSeedFit] = field(default_factory=list)


def _masked_component_atoms(
    directions: np.ndarray,
    basis: np.ndarray,
    triples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original-feature loadings and their all-distinct cubic atoms."""
    loadings = basis @ directions
    atoms = (
        loadings[triples[:, 0], :]
        * loadings[triples[:, 1], :]
        * loadings[triples[:, 2], :]
    )
    return loadings, atoms


def _fit_one_masked_cp_seed(
    projected_tensor: np.ndarray,
    observed_distinct: np.ndarray,
    basis: np.ndarray,
    triples: np.ndarray,
    rank: int,
    seed: int,
    *,
    max_nfev: int,
) -> CPSeedFit:
    """Fit CP only to i<j<k moments, excluding marginal feature skew."""
    dimension = projected_tensor.shape[0]
    initial = _initial_directions(projected_tensor, rank, seed)
    _, atoms = _masked_component_atoms(initial, basis, triples)
    strengths = np.linalg.lstsq(atoms, observed_distinct, rcond=None)[0]
    # For an odd-order symmetric tensor, the signed cube root of each strength
    # can be absorbed into its component. This removes the direction/strength
    # scale degeneracy that otherwise makes finite-difference least squares
    # stop at its evaluation limit for ranks above two.
    components0 = initial * np.cbrt(strengths)[None, :]
    theta0 = components0.ravel()

    def component_loadings(theta):
        components = theta.reshape(dimension, rank)
        return components, basis @ components

    def residual(theta):
        _, loadings = component_loadings(theta)
        predicted = np.sum(
            loadings[triples[:, 0], :]
            * loadings[triples[:, 1], :]
            * loadings[triples[:, 2], :],
            axis=1,
        )
        return predicted - observed_distinct

    def jacobian(theta):
        _, loadings = component_loadings(theta)
        left = triples[:, 0]
        middle = triples[:, 1]
        right = triples[:, 2]
        jac = np.empty((len(triples), dimension * rank), dtype=float)
        for component in range(rank):
            li = loadings[left, component]
            lj = loadings[middle, component]
            lk = loadings[right, component]
            block = (
                basis[left, :] * (lj * lk)[:, None]
                + basis[middle, :] * (li * lk)[:, None]
                + basis[right, :] * (li * lj)[:, None]
            )
            jac[:, component::rank] = block
        return jac

    fit = least_squares(
        residual,
        theta0,
        jac=jacobian,
        max_nfev=max(20, int(max_nfev)),
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    components, _ = component_loadings(fit.x)
    magnitudes = np.linalg.norm(components, axis=0)
    directions = components / np.maximum(magnitudes[None, :], EPS)
    strengths = magnitudes ** 3
    return CPSeedFit(
        seed=int(seed),
        directions=directions,
        strengths=np.asarray(strengths, dtype=float),
        relative_error=float(
            np.linalg.norm(residual(fit.x))
            / (np.linalg.norm(observed_distinct) + EPS)
        ),
        converged=bool(fit.success),
        n_evaluations=int(fit.nfev),
    )


def _initial_directions(tensor: np.ndarray, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    direction = rng.normal(size=tensor.shape[0])
    contracted = np.einsum("ijk,k->ij", tensor, direction, optimize=True)
    contracted = 0.5 * (contracted + contracted.T)
    eigenvalues, vectors = eigh(contracted)
    # Odd-order components can produce either positive or negative contracted
    # eigenvalues. Ordering by value would silently discard strong negative
    # components, so Jennrich-style initialization must use magnitude.
    chosen = np.argsort(np.abs(eigenvalues))[-rank:]
    initial = vectors[:, chosen]
    return initial[:, rng.permutation(rank)]


def _fit_one_cp_seed(
    tensor: np.ndarray,
    rank: int,
    seed: int,
    *,
    max_nfev: int,
) -> CPSeedFit:
    dimension = tensor.shape[0]
    initial = _initial_directions(tensor, rank, seed)
    atoms = np.stack([
        np.einsum(
            "i,j,k->ijk",
            initial[:, index], initial[:, index], initial[:, index],
            optimize=True,
        ).ravel()
        for index in range(rank)
    ], axis=1)
    strengths = np.linalg.lstsq(atoms, tensor.ravel(), rcond=None)[0]
    theta0 = np.concatenate([initial.ravel(), strengths])

    def unpack(theta):
        directions = theta[: dimension * rank].reshape(dimension, rank)
        directions = _normalise_columns(directions)
        return directions, theta[dimension * rank:]

    def residual(theta):
        directions, weights = unpack(theta)
        return (_tensor_from_components(directions, weights) - tensor).ravel()

    fit = least_squares(
        residual,
        theta0,
        max_nfev=max(20, int(max_nfev)),
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    directions, strengths = unpack(fit.x)
    relative_error = float(
        np.linalg.norm(residual(fit.x)) / (np.linalg.norm(tensor) + EPS)
    )
    return CPSeedFit(
        seed=int(seed),
        directions=directions,
        strengths=np.asarray(strengths, dtype=float),
        relative_error=relative_error,
        converged=bool(fit.success),
        n_evaluations=int(fit.nfev),
    )


def component_agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Hungarian-matched absolute cosine between two component sets."""
    left = _normalise_columns(left)
    right = _normalise_columns(right)
    if left.shape != right.shape:
        return 0.0
    similarity = np.abs(left.T @ right)
    rows, columns = linear_sum_assignment(-similarity)
    return float(np.mean(similarity[rows, columns]))


def fit_symmetric_cp(
    tensor: np.ndarray,
    rank: int,
    *,
    seeds: tuple[int, ...] = (11, 23, 37),
    max_nfev: int = 240,
) -> SymmetricCPFit:
    """Fit a symmetric CP model with deterministic multi-start refinement."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.ndim != 3 or len(set(tensor.shape)) != 1:
        raise ValueError("tensor must be cubic")
    if not np.isfinite(tensor).all():
        raise ValueError("tensor contains non-finite entries")
    dimension = tensor.shape[0]
    rank = int(rank)
    if rank < 1 or rank > dimension:
        raise ValueError("rank must be between one and tensor dimension")
    fits = [
        _fit_one_cp_seed(tensor, rank, int(seed), max_nfev=max_nfev)
        for seed in seeds
    ]
    converged_fits = [item for item in fits if item.converged]
    best = min(converged_fits or fits, key=lambda item: item.relative_error)
    agreements = [
        component_agreement(best.directions, item.directions)
        for item in fits if item.seed != best.seed
    ]
    return SymmetricCPFit(
        rank=rank,
        directions=best.directions,
        strengths=best.strengths,
        relative_error=best.relative_error,
        converged=best.converged,
        best_seed=best.seed,
        seed_agreement=float(np.median(agreements)) if agreements else 1.0,
        seed_fits=fits,
    )


def fit_masked_symmetric_cp(
    values: np.ndarray,
    basis: np.ndarray,
    rank: int,
    *,
    seeds: tuple[int, ...] = (11, 23, 37),
    max_nfev: int = 240,
) -> SymmetricCPFit:
    """Fit CP to original-feature moments with three distinct indices only."""
    values = np.asarray(values, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if values.ndim != 2 or basis.ndim != 2 or values.shape[1] != basis.shape[0]:
        raise ValueError("values and basis dimensions disagree")
    rank = int(rank)
    if rank < 1 or rank > basis.shape[1]:
        raise ValueError("rank exceeds the covariance subspace")
    triples = distinct_triples(values.shape[1])
    observed = all_distinct_third_moments(values)
    projected_tensor = masked_projected_third_tensor(values, basis)
    fits = [
        _fit_one_masked_cp_seed(
            projected_tensor,
            observed,
            basis,
            triples,
            rank,
            int(seed),
            max_nfev=max_nfev,
        )
        for seed in seeds
    ]
    converged_fits = [item for item in fits if item.converged]
    best = min(converged_fits or fits, key=lambda item: item.relative_error)
    agreements = [
        component_agreement(best.directions, item.directions)
        for item in fits if item.seed != best.seed
    ]
    return SymmetricCPFit(
        rank=rank,
        directions=best.directions,
        strengths=best.strengths,
        relative_error=best.relative_error,
        converged=best.converged,
        best_seed=best.seed,
        seed_agreement=float(np.median(agreements)) if agreements else 1.0,
        seed_fits=fits,
    )


def target_component(
    loadings: np.ndarray,
    reliability: np.ndarray,
) -> tuple[int, np.ndarray, float, float, np.ndarray]:
    """Select and orient the loading closest to IU-PCR reliability."""
    loadings = _normalise_columns(loadings)
    reliability = np.asarray(reliability, dtype=float).ravel()
    if loadings.shape[0] != reliability.size:
        raise ValueError("loading and reliability dimensions disagree")
    reliability = reliability / (np.linalg.norm(reliability) + EPS)
    signed = loadings.T @ reliability
    similarities = np.abs(signed)
    order = np.argsort(similarities)[::-1]
    chosen = int(order[0])
    oriented = loadings.copy()
    if signed[chosen] < 0:
        oriented[:, chosen] *= -1.0
    margin = float(
        similarities[order[0]] - similarities[order[1]]
        if len(order) > 1 else similarities[order[0]]
    )
    return (
        chosen,
        oriented,
        float(similarities[chosen]),
        margin,
        similarities,
    )


@dataclass
class MomentFactorFit:
    rank: int
    basis: np.ndarray
    covariance_eigenvalues: np.ndarray
    cp: SymmetricCPFit
    loadings: np.ndarray
    factor_scores: np.ndarray
    target_index: int
    target_alignment: float
    target_margin: float
    target_score: np.ndarray
    deflated_values: np.ndarray
    loading_condition: float


def fit_moment_factors(
    values: np.ndarray,
    rank: int,
    reliability: np.ndarray,
    *,
    standardize: bool = True,
    ambient_dimension: int = 6,
    seeds: tuple[int, ...] = (11, 23, 37),
    max_nfev: int = 240,
) -> MomentFactorFit:
    """Fit the covariance-subspace/third-moment model and remove nuisances."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 3:
        raise ValueError("values must have shape (samples>=3, features>=3)")
    if not np.isfinite(values).all():
        raise ValueError("values contain non-finite entries")
    if standardize:
        values = zscore_columns(values)
    basis, eigenvalues = covariance_basis(
        values, ambient_dimension, standardize=False
    )
    projected = values @ basis
    cp = fit_masked_symmetric_cp(
        values, basis, rank, seeds=seeds, max_nfev=max_nfev
    )
    loadings = basis @ cp.directions
    chosen, loadings, alignment, margin, _ = target_component(loadings, reliability)
    gram = loadings.T @ loadings
    factor_scores = values @ loadings @ np.linalg.pinv(gram, rcond=1e-10)
    target_score = factor_scores[:, chosen]
    if safe_correlation(target_score, values @ reliability) < 0:
        target_score = -target_score
        factor_scores[:, chosen] *= -1.0
        loadings[:, chosen] *= -1.0
    nuisance = [index for index in range(rank) if index != chosen]
    if nuisance:
        nuisance_reconstruction = (
            factor_scores[:, nuisance] @ loadings[:, nuisance].T
        )
        deflated = zscore_columns(values - nuisance_reconstruction)
    else:
        # With standardize=False, this is an exact identity.  The experiment
        # uses that mode because mixed-v2 is already standardized.
        deflated = values.copy()
    return MomentFactorFit(
        rank=int(rank),
        basis=basis,
        covariance_eigenvalues=eigenvalues,
        cp=cp,
        loadings=loadings,
        factor_scores=factor_scores,
        target_index=chosen,
        target_alignment=alignment,
        target_margin=margin,
        target_score=np.asarray(target_score, dtype=float),
        deflated_values=deflated,
        loading_condition=float(np.linalg.cond(gram)),
    )


def grouped_half_splits(
    n_samples: int,
    *,
    groups: np.ndarray | None = None,
    seeds: tuple[int, ...] = (101, 211, 307, 401),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic half splits, keeping every supplied group intact."""
    if groups is None:
        groups = np.arange(n_samples)
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError("groups must have one value per sample")
    unique = np.unique(groups)
    output = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        shuffled = unique[rng.permutation(len(unique))]
        train_groups = set(shuffled[: len(shuffled) // 2].tolist())
        train = np.asarray([value in train_groups for value in groups], dtype=bool)
        if train.sum() < 3 or (~train).sum() < 3:
            raise ValueError("split has fewer than three samples in one half")
        output.append((np.where(train)[0], np.where(~train)[0]))
    return output


@dataclass
class RankSelection:
    selected_rank: int
    proposed_rank: int
    fallback_reasons: list[str]
    third_moment_stability: float
    validation_errors: dict[int, list[float]]
    validation_means: dict[int, float]
    validation_standard_errors: dict[int, float]
    split_target_loadings: dict[int, list[np.ndarray]]
    target_stability: dict[int, float]
    target_margin: dict[int, float]
    near_best_frequency: dict[int, float]
    convergence_frequency: dict[int, float]
    rank_rejection_reasons: dict[int, list[str]]


def _median_pairwise_loading_cosine(loadings: list[np.ndarray]) -> float:
    if len(loadings) < 2:
        return 1.0
    values = []
    for left, right in itertools.combinations(loadings, 2):
        values.append(float(abs(np.dot(left, right)) /
                            (np.linalg.norm(left) * np.linalg.norm(right) + EPS)))
    return float(np.median(values))


def select_rank_label_free(
    values: np.ndarray,
    reliability: np.ndarray,
    *,
    max_rank: int = 5,
    ambient_dimension: int = 6,
    groups: np.ndarray | None = None,
    split_seeds: tuple[int, ...] = (101, 211, 307, 401),
    cp_seeds: tuple[int, ...] = (11, 23),
    max_nfev: int = 140,
    minimum_third_stability: float = 0.75,
    minimum_target_stability: float = 0.75,
    minimum_target_margin: float = 0.05,
    minimum_near_best_frequency: float = 0.70,
    reliability_estimator: Callable[[np.ndarray], np.ndarray] | None = None,
) -> RankSelection:
    """Select total factor rank without labels, with an exact rank-one fallback."""
    values = zscore_columns(values)
    reliability = np.asarray(reliability, dtype=float)
    max_rank = min(int(max_rank), int(ambient_dimension), values.shape[1])
    splits = grouped_half_splits(len(values), groups=groups, seeds=split_seeds)
    errors = {rank: [] for rank in range(1, max_rank + 1)}
    target_loadings = {rank: [] for rank in range(1, max_rank + 1)}
    margins = {rank: [] for rank in range(1, max_rank + 1)}
    converged = {rank: [] for rank in range(1, max_rank + 1)}
    third_stability = []
    split_rank_errors = []
    for train_index, validation_index in splits:
        train = values[train_index]
        validation = values[validation_index]
        raw_train = all_distinct_third_moments(train)
        raw_validation = all_distinct_third_moments(validation)
        third_stability.append(safe_correlation(raw_train, raw_validation))

        basis, _ = covariance_basis(
            train, ambient_dimension, standardize=False
        )
        train_reliability = (
            np.asarray(reliability_estimator(train), dtype=float)
            if reliability_estimator is not None else reliability
        )
        validation_distinct = all_distinct_third_moments(validation)
        triples = distinct_triples(values.shape[1])
        split_errors = []
        for rank in range(1, max_rank + 1):
            cp = fit_masked_symmetric_cp(
                train, basis, rank, seeds=cp_seeds, max_nfev=max_nfev
            )
            loadings, atoms = _masked_component_atoms(
                cp.directions, basis, triples
            )
            error = (
                float(np.linalg.norm(validation_distinct - atoms @ cp.strengths)
                      / (np.linalg.norm(validation_distinct) + EPS))
                if cp.converged else 1e6
            )
            errors[rank].append(error)
            converged[rank].append(bool(cp.converged))
            split_errors.append(error)
            chosen, oriented, _, margin, _ = target_component(
                loadings, train_reliability
            )
            target_loadings[rank].append(oriented[:, chosen])
            margins[rank].append(margin)
        split_rank_errors.append(split_errors)

    means = {rank: float(np.mean(errors[rank])) for rank in errors}
    standard_errors = {
        rank: float(np.std(errors[rank], ddof=1) / np.sqrt(len(splits)))
        if len(splits) > 1 else 0.0
        for rank in errors
    }
    stability = {
        rank: _median_pairwise_loading_cosine(target_loadings[rank])
        for rank in errors
    }
    margin_summary = {rank: float(np.median(margins[rank])) for rank in errors}
    split_rank_errors = np.asarray(split_rank_errors, dtype=float)
    near_best = {}
    for rank in errors:
        column = split_rank_errors[:, rank - 1]
        minimum = split_rank_errors.min(axis=1)
        near_best[rank] = float(np.mean(column <= minimum * 1.05 + 1e-12))

    convergence_frequency = {
        rank: float(np.mean(converged[rank])) for rank in errors
    }

    median_third = float(np.median(third_stability))
    rejection_reasons = {rank: [] for rank in errors}
    for rank in range(2, max_rank + 1):
        if median_third < minimum_third_stability:
            rejection_reasons[rank].append("third_moment_unstable")
        if stability[rank] < minimum_target_stability:
            rejection_reasons[rank].append("target_loading_unstable")
        if margin_summary[rank] < minimum_target_margin:
            rejection_reasons[rank].append("target_component_ambiguous")
        if near_best[rank] < minimum_near_best_frequency:
            rejection_reasons[rank].append("rank_not_consistently_near_best")
        if convergence_frequency[rank] < 1.0:
            rejection_reasons[rank].append("split_cp_nonconvergence")

    raw_best = min(errors, key=lambda rank: means[rank])
    threshold = means[raw_best] + standard_errors[raw_best]
    raw_within_one_se = [rank for rank in errors if means[rank] <= threshold]
    proposed = min(raw_within_one_se)
    eligible_within = [
        rank for rank in raw_within_one_se
        if rank == 1 or not rejection_reasons[rank]
    ]
    selected = min(eligible_within) if eligible_within else 1
    reasons = []
    if selected == 1 and proposed > 1:
        reasons = list(rejection_reasons[proposed]) or [
            "no_stable_higher_rank_within_one_se"
        ]
    return RankSelection(
        selected_rank=int(selected),
        proposed_rank=int(proposed),
        fallback_reasons=reasons,
        third_moment_stability=median_third,
        validation_errors=errors,
        validation_means=means,
        validation_standard_errors=standard_errors,
        split_target_loadings=target_loadings,
        target_stability=stability,
        target_margin=margin_summary,
        near_best_frequency=near_best,
        convergence_frequency=convergence_frequency,
        rank_rejection_reasons=rejection_reasons,
    )


def permuted_cross_moment_values(values: np.ndarray, seed: int) -> np.ndarray:
    """Destroy cross-feature moments while preserving every marginal column."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    output = np.empty_like(values)
    for column in range(values.shape[1]):
        output[:, column] = values[rng.permutation(len(values)), column]
    return output


def pca_deflation(values: np.ndarray, rank: int, reliability: np.ndarray) -> np.ndarray:
    """Second-order-only same-rank nuisance-deflation control."""
    values = zscore_columns(values)
    rank = min(max(1, int(rank)), values.shape[1])
    covariance = values.T @ values / len(values)
    _, loadings = eigh(
        covariance,
        subset_by_index=[values.shape[1] - rank, values.shape[1] - 1],
    )
    loadings = loadings[:, ::-1]
    chosen, loadings, _, _, _ = target_component(loadings, reliability)
    scores = values @ loadings
    nuisance = [index for index in range(rank) if index != chosen]
    if not nuisance:
        return values.copy()
    return zscore_columns(values - scores[:, nuisance] @ loadings[:, nuisance].T)


__all__ = [
    "CPSeedFit",
    "MomentFactorFit",
    "RankSelection",
    "SymmetricCPFit",
    "all_distinct_third_moments",
    "component_agreement",
    "covariance_basis",
    "fit_moment_factors",
    "fit_masked_symmetric_cp",
    "fit_symmetric_cp",
    "grouped_half_splits",
    "masked_projected_third_tensor",
    "pca_deflation",
    "permuted_cross_moment_values",
    "safe_correlation",
    "select_rank_label_free",
    "target_component",
    "distinct_triples",
    "third_central_moment",
    "zscore_columns",
]
