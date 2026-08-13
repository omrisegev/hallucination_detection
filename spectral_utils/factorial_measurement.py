"""Label-free factorial measurement models for group-free IU research.

The module reconstructs environment-specific residual correlation matrices
from mechanically registered feature axes and/or anonymized covariance
loadings.  It never accepts correctness labels.  Its primary validation task
is masked feature-by-environment covariance reconstruction, not detector
performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class FactorialConfiguration:
    """One completely structural reconstruction configuration."""

    basis_kind: str
    rank: int
    interaction: bool
    ridge: float
    alpha: float = 0.5
    random_seed: int = 0

    def payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FactorialFit:
    """A fitted shared feature basis and its global covariance parameters."""

    feature_names: tuple[str, ...]
    basis: np.ndarray
    global_parameters: np.ndarray
    configuration: FactorialConfiguration
    duplicate_classes: tuple[tuple[int, ...], ...]
    diagnostics: dict


def pairwise_environment_mean(covariances: Sequence[np.ndarray]) -> np.ndarray:
    """Equal-environment pairwise mean with no sample-count weighting."""

    values = np.asarray(covariances, dtype=float)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("covariances must have shape environment x feature x feature")
    present = np.isfinite(values)
    counts = present.sum(axis=0)
    if np.any(counts == 0):
        missing = np.argwhere(counts == 0)
        raise ValueError(f"uncovered covariance pair(s): {missing[:5].tolist()}")
    total = np.where(present, values, 0.0).sum(axis=0)
    mean = total / counts
    return 0.5 * (mean + mean.T)


def covariance_from_residuals(
    residuals: np.ndarray,
    local_feature_names: Sequence[str],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Embed one standardized residual covariance in a NaN-padded roster."""

    residuals = np.asarray(residuals, dtype=float)
    local = tuple(str(value) for value in local_feature_names)
    roster = tuple(str(value) for value in feature_names)
    if residuals.ndim != 2 or residuals.shape[1] != len(local):
        raise ValueError("residual matrix and local feature names disagree")
    if not np.isfinite(residuals).all():
        raise ValueError("residuals must be finite")
    if len(set(local)) != len(local) or len(set(roster)) != len(roster):
        raise ValueError("feature names must be unique")
    lookup = {name: index for index, name in enumerate(roster)}
    if any(name not in lookup for name in local):
        raise ValueError("local feature is absent from the global roster")
    centred = residuals - residuals.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    if np.any(scale < EPS):
        raise ValueError("inactive residual feature")
    standardized = centred / scale
    local_covariance = standardized.T @ standardized / len(standardized)
    output = np.full((len(roster), len(roster)), np.nan, dtype=float)
    columns = np.asarray([lookup[name] for name in local], dtype=int)
    output[np.ix_(columns, columns)] = local_covariance
    return output


def mechanical_design(
    feature_names: Sequence[str],
    feature_dag: Sequence[Mapping],
    *,
    axes: str = "factorial",
    random_seed: int | None = None,
) -> np.ndarray:
    """Build mechanically derived channel/operator incidence coordinates."""

    if axes not in {"channel", "operator", "factorial"}:
        raise ValueError("axes must be channel, operator, or factorial")
    records = {str(row["feature_name"]): row for row in feature_dag}
    names = tuple(str(value) for value in feature_names)
    if any(name not in records for name in names):
        raise ValueError("feature DAG does not cover the requested roster")
    channels = [str(records[name]["source_stream"]) for name in names]
    operators = [str(records[name]["operator"]) for name in names]
    if random_seed is not None:
        rng = np.random.default_rng(int(random_seed))
        channels = list(np.asarray(channels, dtype=object)[rng.permutation(len(names))])
        operators = list(np.asarray(operators, dtype=object)[rng.permutation(len(names))])

    blocks = [np.ones((len(names), 1), dtype=float)]
    if axes in {"channel", "factorial"}:
        levels = sorted(set(channels))
        blocks.append(np.asarray([[value == level for level in levels]
                                  for value in channels], dtype=float))
    if axes in {"operator", "factorial"}:
        levels = sorted(set(operators))
        blocks.append(np.asarray([[value == level for level in levels]
                                  for value in operators], dtype=float))
    return np.column_stack(blocks)


def exact_duplicate_classes(
    covariances: Sequence[np.ndarray],
    *,
    tolerance: float = 1e-10,
) -> tuple[tuple[int, ...], ...]:
    """Find data-identical measurements using only shared covariance profiles."""

    values = np.asarray(covariances, dtype=float)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("covariances must have shape environment x feature x feature")
    p = values.shape[1]
    parent = list(range(p))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for left in range(p):
        for right in range(left + 1, p):
            compared = 0
            identical = True
            for covariance in values:
                if not (np.isfinite(covariance[left, left])
                        and np.isfinite(covariance[right, right])
                        and np.isfinite(covariance[left, right])):
                    continue
                common = np.isfinite(covariance[left]) & np.isfinite(covariance[right])
                if not np.any(common):
                    continue
                compared += 1
                if (abs(float(covariance[left, right]) - 1.0) > tolerance
                        or np.max(np.abs(
                            covariance[left, common] - covariance[right, common]
                        )) > tolerance):
                    identical = False
                    break
            if identical and compared:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(p):
        groups.setdefault(find(index), []).append(index)
    return tuple(tuple(group) for group in groups.values())


def _orthogonal_columns(matrix: np.ndarray, rank: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("basis source must be a two-dimensional feature matrix")
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    available = int(np.sum(singular > 1e-10))
    keep = min(max(int(rank), 1), available)
    if keep < 1:
        raise ValueError("basis source has zero rank")
    return left[:, :keep]


def _learned_basis(covariances: Sequence[np.ndarray], rank: int) -> np.ndarray:
    mean = pairwise_environment_mean(covariances)
    values, vectors = np.linalg.eigh(mean)
    order = np.argsort(np.abs(values))[::-1]
    keep = min(max(int(rank), 1), len(order))
    return vectors[:, order[:keep]]


def _uncollapsed_basis(
    covariances: Sequence[np.ndarray],
    feature_names: Sequence[str],
    feature_dag: Sequence[Mapping],
    configuration: FactorialConfiguration,
) -> np.ndarray:
    kind = configuration.basis_kind
    if kind == "pca":
        return _learned_basis(covariances, configuration.rank)
    if kind in {"channel", "operator", "factorial", "random"}:
        axes = "factorial" if kind == "random" else kind
        seed = configuration.random_seed if kind == "random" else None
        design = mechanical_design(
            feature_names, feature_dag, axes=axes, random_seed=seed
        )
        return _orthogonal_columns(design, configuration.rank)
    if kind == "hybrid":
        hard = _orthogonal_columns(
            mechanical_design(feature_names, feature_dag, axes="factorial"),
            min(len(feature_names), max(configuration.rank, 2)),
        )
        learned = _learned_basis(covariances, configuration.rank)
        kernel = (
            float(configuration.alpha) * (hard @ hard.T)
            + (1.0 - float(configuration.alpha)) * (learned @ learned.T)
        )
        values, vectors = np.linalg.eigh(0.5 * (kernel + kernel.T))
        order = np.argsort(values)[::-1]
        keep = min(max(int(configuration.rank), 1), len(order))
        return vectors[:, order[:keep]]
    raise ValueError(f"unsupported basis kind: {kind}")


def _parameter_design(basis: np.ndarray, pairs: np.ndarray, interaction: bool) -> np.ndarray:
    basis = np.asarray(basis, dtype=float)
    pairs = np.asarray(pairs, dtype=int)
    left = basis[pairs[:, 0]]
    right = basis[pairs[:, 1]]
    rank = basis.shape[1]
    if not interaction:
        return left * right
    columns = []
    for first in range(rank):
        columns.append(left[:, first] * right[:, first])
        for second in range(first + 1, rank):
            columns.append(
                left[:, first] * right[:, second]
                + left[:, second] * right[:, first]
            )
    return np.column_stack(columns)


def _observed_pairs(covariance: np.ndarray, indices: np.ndarray) -> np.ndarray:
    pairs = []
    for offset, left in enumerate(indices):
        for right in indices[offset + 1:]:
            if np.isfinite(covariance[left, right]):
                pairs.append((int(left), int(right)))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def _fit_parameters(
    covariance: np.ndarray,
    basis: np.ndarray,
    indices: np.ndarray,
    *,
    interaction: bool,
    ridge: float,
    centre: np.ndarray | None = None,
) -> np.ndarray:
    pairs = _observed_pairs(covariance, indices)
    if not len(pairs):
        raise ValueError("no observed off-diagonal covariance pairs")
    design = _parameter_design(basis, pairs, interaction)
    target = covariance[pairs[:, 0], pairs[:, 1]]
    if centre is None:
        centre = np.zeros(design.shape[1], dtype=float)
    penalty = float(ridge)
    gram = design.T @ design + penalty * np.eye(design.shape[1])
    rhs = design.T @ target + penalty * np.asarray(centre, dtype=float)
    return np.linalg.solve(gram, rhs)


def fit_factorial_measurement(
    covariances: Sequence[np.ndarray],
    feature_names: Sequence[str],
    feature_dag: Sequence[Mapping],
    configuration: FactorialConfiguration,
) -> FactorialFit:
    """Fit one shared basis after automatically collapsing exact duplicates."""

    values = np.asarray(covariances, dtype=float)
    names = tuple(str(value) for value in feature_names)
    if values.ndim != 3 or values.shape[1:] != (len(names), len(names)):
        raise ValueError("covariance dimensions and feature roster disagree")
    classes = exact_duplicate_classes(values)
    representatives = np.asarray([group[0] for group in classes], dtype=int)
    collapsed = values[:, representatives][:, :, representatives]
    collapsed_names = tuple(names[index] for index in representatives)
    basis_unique = _uncollapsed_basis(
        collapsed, collapsed_names, feature_dag, configuration
    )
    basis = np.empty((len(names), basis_unique.shape[1]), dtype=float)
    for unique_index, group in enumerate(classes):
        basis[np.asarray(group, dtype=int)] = basis_unique[unique_index]
    mean = pairwise_environment_mean(values)
    all_indices = np.arange(len(names), dtype=int)
    global_parameters = _fit_parameters(
        mean,
        basis,
        all_indices,
        interaction=configuration.interaction,
        ridge=max(float(configuration.ridge), 1e-8),
    )
    return FactorialFit(
        feature_names=names,
        basis=basis,
        global_parameters=global_parameters,
        configuration=configuration,
        duplicate_classes=classes,
        diagnostics={
            "n_features": len(names),
            "n_unique_measurements": len(classes),
            "basis_rank": int(basis.shape[1]),
            "parameter_count": int(len(global_parameters)),
            "uses_labels": False,
        },
    )


def masked_feature_reconstruction_rows(
    fit: FactorialFit,
    covariance: np.ndarray,
    environment_id: str,
) -> list[dict]:
    """Predict each available feature row from the remaining covariance block."""

    covariance = np.asarray(covariance, dtype=float)
    p = len(fit.feature_names)
    available = np.flatnonzero(np.isfinite(np.diag(covariance)))
    rows = []
    for held_out in available:
        observed = available[available != held_out]
        local = _fit_parameters(
            covariance,
            fit.basis,
            observed,
            interaction=fit.configuration.interaction,
            ridge=fit.configuration.ridge,
            centre=fit.global_parameters,
        )
        partners = observed[np.isfinite(covariance[held_out, observed])]
        pairs = np.column_stack((
            np.full(len(partners), held_out, dtype=int), partners
        ))
        prediction = _parameter_design(
            fit.basis, pairs, fit.configuration.interaction
        ) @ local
        actual = covariance[held_out, partners]
        for partner, observed_value, predicted_value in zip(
            partners, actual, prediction
        ):
            rows.append({
                "environment": str(environment_id),
                "held_out_feature": fit.feature_names[held_out],
                "partner_feature": fit.feature_names[int(partner)],
                "actual": float(observed_value),
                "prediction": float(predicted_value),
                "squared_error": float((observed_value - predicted_value) ** 2),
            })
    return rows


def pooled_mean_reconstruction_rows(
    train_covariances: Sequence[np.ndarray],
    covariance: np.ndarray,
    feature_names: Sequence[str],
    environment_id: str,
) -> list[dict]:
    """Non-adaptive pooled covariance baseline for the same masked entries."""

    mean = pairwise_environment_mean(train_covariances)
    names = tuple(str(value) for value in feature_names)
    available = np.flatnonzero(np.isfinite(np.diag(covariance)))
    rows = []
    for held_out in available:
        for partner in available:
            if held_out == partner:
                continue
            actual = float(covariance[held_out, partner])
            predicted = float(mean[held_out, partner])
            rows.append({
                "environment": str(environment_id),
                "held_out_feature": names[held_out],
                "partner_feature": names[partner],
                "actual": actual,
                "prediction": predicted,
                "squared_error": float((actual - predicted) ** 2),
            })
    return rows


def reconstruction_mse(rows: Iterable[Mapping]) -> float:
    errors = np.asarray([float(row["squared_error"]) for row in rows], dtype=float)
    if not len(errors):
        raise ValueError("no reconstruction rows")
    return float(np.mean(errors))


def cross_validated_mse(
    covariances: Sequence[np.ndarray],
    environment_ids: Sequence[str],
    feature_names: Sequence[str],
    feature_dag: Sequence[Mapping],
    configuration: FactorialConfiguration,
) -> float:
    """Leave-one-environment-out structural selection score."""

    values = np.asarray(covariances, dtype=float)
    if len(values) != len(environment_ids):
        raise ValueError("environment IDs and covariances disagree")
    errors = []
    for held_out in range(len(values)):
        train = np.delete(values, held_out, axis=0)
        fit = fit_factorial_measurement(
            train, feature_names, feature_dag, configuration
        )
        rows = masked_feature_reconstruction_rows(
            fit, values[held_out], str(environment_ids[held_out])
        )
        errors.extend(float(row["squared_error"]) for row in rows)
    return float(np.mean(errors))


def select_configuration(
    covariances: Sequence[np.ndarray],
    environment_ids: Sequence[str],
    feature_names: Sequence[str],
    feature_dag: Sequence[Mapping],
    configurations: Sequence[FactorialConfiguration],
) -> tuple[FactorialConfiguration, list[dict]]:
    """Select only by label-free leave-one-environment reconstruction."""

    rows = []
    for configuration in configurations:
        mse = cross_validated_mse(
            covariances,
            environment_ids,
            feature_names,
            feature_dag,
            configuration,
        )
        rows.append({**configuration.payload(), "cv_mse": mse})
    rows.sort(key=lambda row: (
        row["cv_mse"],
        row["rank"],
        row["interaction"],
        row["ridge"],
        row["alpha"],
    ))
    return FactorialConfiguration(**{
        key: rows[0][key]
        for key in FactorialConfiguration.__dataclass_fields__
    }), rows


def subspace_stability(
    covariances: Sequence[np.ndarray],
    feature_names: Sequence[str],
    feature_dag: Sequence[Mapping],
    configuration: FactorialConfiguration,
) -> list[float]:
    """Normalized projector overlap after each environment deletion."""

    values = np.asarray(covariances, dtype=float)
    reference = fit_factorial_measurement(
        values, feature_names, feature_dag, configuration
    ).basis
    reference, _ = np.linalg.qr(reference)
    overlaps = []
    for held_out in range(len(values)):
        candidate = fit_factorial_measurement(
            np.delete(values, held_out, axis=0),
            feature_names,
            feature_dag,
            configuration,
        ).basis
        candidate, _ = np.linalg.qr(candidate)
        rank = min(reference.shape[1], candidate.shape[1])
        overlaps.append(float(
            np.linalg.norm(reference[:, :rank].T @ candidate[:, :rank], ord="fro") ** 2
            / rank
        ))
    return overlaps


def soft_quotient_weights(fit: FactorialFit) -> tuple[np.ndarray, dict]:
    """Convert soft feature loadings into duplicate-balanced effective masses."""

    representatives = np.asarray([group[0] for group in fit.duplicate_classes], dtype=int)
    unique = np.asarray(fit.basis[representatives], dtype=float)
    norms = np.linalg.norm(unique, axis=1, keepdims=True)
    norms[norms < EPS] = 1.0
    normalized = unique / norms
    similarity = np.abs(normalized @ normalized.T)
    np.fill_diagonal(similarity, 1.0)
    class_mass = 1.0 / np.maximum(similarity.sum(axis=1), EPS)
    class_mass /= class_mass.sum()
    weights = np.zeros(len(fit.feature_names), dtype=float)
    for mass, group in zip(class_mass, fit.duplicate_classes):
        weights[np.asarray(group, dtype=int)] = mass / len(group)
    return weights, {
        "class_mass": class_mass,
        "similarity": similarity,
        "duplicate_classes": fit.duplicate_classes,
        "weight_sum": float(weights.sum()),
        "uses_labels": False,
    }


def augment_correlated_duplicate(
    covariances: Sequence[np.ndarray],
    feature_index: int,
    *,
    correlation: float = 1.0,
) -> np.ndarray:
    """Append an exact or near duplicate while preserving covariance validity."""

    if not 0 <= float(correlation) <= 1:
        raise ValueError("correlation must lie in [0, 1]")
    values = np.asarray(covariances, dtype=float)
    p = values.shape[1]
    output = np.full((len(values), p + 1, p + 1), np.nan, dtype=float)
    output[:, :p, :p] = values
    for environment, covariance in enumerate(values):
        if not np.isfinite(covariance[feature_index, feature_index]):
            continue
        available = np.isfinite(np.diag(covariance))
        output[environment, p, np.flatnonzero(available)] = (
            float(correlation) * covariance[feature_index, available]
        )
        output[environment, np.flatnonzero(available), p] = output[
            environment, p, np.flatnonzero(available)
        ]
        output[environment, p, p] = 1.0
    return output


__all__ = [
    "FactorialConfiguration",
    "FactorialFit",
    "augment_correlated_duplicate",
    "covariance_from_residuals",
    "cross_validated_mse",
    "exact_duplicate_classes",
    "fit_factorial_measurement",
    "masked_feature_reconstruction_rows",
    "mechanical_design",
    "pairwise_environment_mean",
    "pooled_mean_reconstruction_rows",
    "reconstruction_mse",
    "select_configuration",
    "soft_quotient_weights",
    "subspace_stability",
]
