"""Multi-environment joint (block) diagonalization for Phase A2.

The implementation follows the structural premise used in non-stationary
source separation: covariance matrices share a mixing basis while latent
variance profiles change across environments.  A randomized linear
combination supplies a deterministic RJD initializer, optional Jacobi sweeps
minimize aggregate off-diagonal energy, and a residual coupling graph merges
components into jointly diagonal blocks.  The API accepts no correctness
labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


EPS = 1e-10


@dataclass(frozen=True)
class JBDConfiguration:
    method: str
    ridge: float
    block_quantile: float = 1.0
    random_draws: int = 64
    random_seed: int = 20260813
    jacobi_sweeps: int = 64
    jacobi_tolerance: float = 1e-10


@dataclass(frozen=True)
class JBDModel:
    feature_names: tuple[str, ...]
    mean_covariance: np.ndarray
    mixing: np.ndarray
    blocks: tuple[tuple[int, ...], ...]
    atoms: np.ndarray
    atom_labels: tuple[str, ...]
    transformed_covariances: np.ndarray
    configuration: JBDConfiguration
    diagnostics: dict


def _validate_covariances(covariances: Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(covariances, dtype=float)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("covariances must have shape environment x p x p")
    if len(values) < 2 or values.shape[1] < 2:
        raise ValueError("joint diagonalization needs multiple matrices and features")
    if not np.isfinite(values).all():
        raise ValueError("joint diagonalization requires a complete finite roster")
    asymmetry = np.max(np.abs(values - values.transpose(0, 2, 1)))
    if asymmetry > 1e-8:
        raise ValueError("covariance matrices must be symmetric")
    return 0.5 * (values + values.transpose(0, 2, 1))


def matrix_sqrt_and_inverse(matrix: np.ndarray, floor: float = 1e-6):
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, float(floor) * scale)
    root = (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T
    inverse = (eigenvectors * (1.0 / np.sqrt(clipped))) @ eigenvectors.T
    return root, inverse, clipped


def whiten_covariances(covariances: Sequence[np.ndarray]):
    values = _validate_covariances(covariances)
    mean = np.mean(values, axis=0)
    root, inverse, eigenvalues = matrix_sqrt_and_inverse(mean)
    whitened = np.asarray([
        inverse @ covariance @ inverse for covariance in values
    ])
    whitened = 0.5 * (whitened + whitened.transpose(0, 2, 1))
    return mean, root, inverse, whitened, eigenvalues


def off_diagonal_energy(matrices: Sequence[np.ndarray]) -> float:
    values = np.asarray(matrices, dtype=float)
    diagonal = np.zeros_like(values)
    index = np.arange(values.shape[1])
    diagonal[:, index, index] = values[:, index, index]
    numerator = float(np.sum((values - diagonal) ** 2))
    denominator = float(np.sum(values ** 2)) + EPS
    return numerator / denominator


def randomized_joint_basis(
    whitened: Sequence[np.ndarray],
    *,
    draws: int = 64,
    seed: int = 20260813,
) -> tuple[np.ndarray, dict]:
    """Diagonalize the best seeded random linear combination (RJD)."""

    values = _validate_covariances(whitened)
    if int(draws) < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(int(seed))
    centered = values - np.mean(values, axis=0, keepdims=True)
    best = None
    for draw in range(int(draws)):
        weights = rng.normal(size=len(centered))
        weights -= weights.mean()
        norm = np.linalg.norm(weights)
        if norm < EPS:
            continue
        weights /= norm
        combination = np.tensordot(weights, centered, axes=(0, 0))
        eigenvalues, basis = np.linalg.eigh(0.5 * (combination + combination.T))
        order = np.argsort(np.abs(eigenvalues))[::-1]
        basis = basis[:, order]
        transformed = np.asarray([basis.T @ matrix @ basis for matrix in values])
        cost = off_diagonal_energy(transformed)
        candidate = (cost, draw, basis, weights, eigenvalues[order])
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("RJD produced no non-degenerate random combination")
    return best[2], {
        "selected_draw": int(best[1]),
        "rjd_off_diagonal_energy": float(best[0]),
        "random_weights": best[3],
        "combination_eigenvalues": best[4],
        "random_draws": int(draws),
        "random_seed": int(seed),
    }


def jacobi_refine(
    matrices: Sequence[np.ndarray],
    initial_basis: np.ndarray,
    *,
    max_sweeps: int = 64,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """Orthogonal Jacobi AJD minimizing aggregate off-diagonal Frobenius energy."""

    values = _validate_covariances(matrices)
    basis = np.asarray(initial_basis, dtype=float).copy()
    if basis.shape != values.shape[1:]:
        raise ValueError("initial basis has the wrong shape")
    transformed = np.asarray([basis.T @ matrix @ basis for matrix in values])
    previous = off_diagonal_energy(transformed)
    history = [previous]
    p = values.shape[1]
    for _ in range(int(max_sweeps)):
        for left in range(p - 1):
            for right in range(left + 1, p):
                x = transformed[:, left, left] - transformed[:, right, right]
                y = 2.0 * transformed[:, left, right]
                gram = np.asarray([
                    [float(x @ x), float(x @ y)],
                    [float(x @ y), float(y @ y)],
                ])
                eigenvalues, eigenvectors = np.linalg.eigh(gram)
                vector = eigenvectors[:, int(np.argmax(eigenvalues))]
                angle = 0.5 * np.arctan2(vector[1], vector[0])
                # Equivalent optima differ by a component swap.  Choose the
                # smallest-magnitude representative for stable iteration.
                alternatives = np.asarray([
                    angle, angle + np.pi / 2.0, angle - np.pi / 2.0
                ])
                angle = float(alternatives[np.argmin(np.abs(alternatives))])
                cosine, sine = np.cos(angle), np.sin(angle)
                rotation = np.asarray([[cosine, -sine], [sine, cosine]])
                columns = np.asarray([left, right])
                basis[:, columns] = basis[:, columns] @ rotation
                # Apply G.T @ A @ G only to the affected rows/columns.  This is
                # algebraically identical to two dense p-by-p multiplies but
                # reduces every Jacobi pair update from O(E p^3) to O(E p).
                transformed[:, :, columns] = np.einsum(
                    "epr,rs->eps", transformed[:, :, columns].copy(), rotation
                )
                transformed[:, columns, :] = np.einsum(
                    "ab,ebp->eap", rotation.T,
                    transformed[:, columns, :].copy(),
                )
        current = off_diagonal_energy(transformed)
        history.append(current)
        if previous - current <= float(tolerance) * max(previous, 1.0):
            break
        previous = current
    return basis, {
        "jacobi_sweeps": len(history) - 1,
        "initial_off_diagonal_energy": float(history[0]),
        "final_off_diagonal_energy": float(history[-1]),
        "history": tuple(float(value) for value in history),
    }


def residual_coupling(transformed: Sequence[np.ndarray]) -> np.ndarray:
    """Scale-free RMS off-diagonal coupling between recovered components."""

    values = _validate_covariances(transformed)
    diagonal_second = np.mean(
        np.diagonal(values, axis1=1, axis2=2) ** 2, axis=0
    )
    numerator = np.sqrt(np.mean(values ** 2, axis=0))
    denominator = np.sqrt(np.sqrt(
        diagonal_second[:, None] * diagonal_second[None, :]
    ))
    coupling = numerator / np.maximum(denominator, EPS)
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 1.0)
    return coupling


def blocks_from_coupling(
    coupling: np.ndarray,
    *,
    quantile: float,
) -> tuple[tuple[int, ...], ...]:
    """Connected components after a data-derived coupling quantile cut."""

    coupling = np.asarray(coupling, dtype=float)
    p = coupling.shape[0]
    if coupling.shape != (p, p) or not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("invalid coupling matrix or quantile")
    if float(quantile) >= 1.0:
        return tuple((index,) for index in range(p))
    upper = coupling[np.triu_indices(p, 1)]
    threshold = float(np.quantile(upper, float(quantile)))
    adjacency = (coupling >= threshold) & (coupling > EPS)
    np.fill_diagonal(adjacency, True)
    seen = np.zeros(p, dtype=bool)
    blocks = []
    for start in range(p):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        block = []
        while stack:
            current = stack.pop()
            block.append(current)
            neighbors = np.flatnonzero(adjacency[current] & ~seen)
            for neighbor in neighbors:
                seen[neighbor] = True
                stack.append(int(neighbor))
        blocks.append(tuple(sorted(block)))
    return tuple(sorted(blocks, key=lambda block: block[0]))


def covariance_atoms(
    mixing: np.ndarray,
    blocks: Sequence[Sequence[int]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return symmetric covariance mechanisms allowed within each block."""

    mixing = np.asarray(mixing, dtype=float)
    p, rank = mixing.shape
    normalized_blocks = tuple(tuple(int(value) for value in block) for block in blocks)
    if (
        p == rank
        and len(normalized_blocks) == 1
        and sorted(normalized_blocks[0]) == list(range(rank))
    ):
        # A full-rank full block is the complete symmetric covariance space,
        # regardless of the chosen mixing basis.  Emit its canonical
        # Frobenius-orthonormal basis directly.  Besides exposing that
        # non-identifiability, this avoids an unnecessary SVD of 465 dense
        # 30-by-30 atoms in every cross-validation fit.
        atoms = []
        for left in range(p):
            atom = np.zeros((p, p), dtype=float)
            atom[left, left] = 1.0
            atoms.append(atom)
            for right in range(left + 1, p):
                atom = np.zeros((p, p), dtype=float)
                atom[left, right] = atom[right, left] = 1.0 / np.sqrt(2.0)
                atoms.append(atom)
        return np.asarray(atoms), tuple(
            f"canonical_symmetric_{index}" for index in range(len(atoms))
        )

    atoms = []
    labels = []
    used = []
    for block_index, block in enumerate(normalized_blocks):
        used.extend(block)
        for offset, left in enumerate(block):
            for right in block[offset:]:
                if left == right:
                    atom = np.outer(mixing[:, left], mixing[:, left])
                else:
                    atom = (
                        np.outer(mixing[:, left], mixing[:, right])
                        + np.outer(mixing[:, right], mixing[:, left])
                    )
                atoms.append(atom)
                labels.append(f"block{block_index}:{left},{right}")
    if sorted(used) != list(range(rank)):
        raise ValueError("blocks must partition every mixing coordinate")
    raw = np.asarray(atoms)
    # Ridge acts on mechanism coefficients.  Orthonormalizing the mechanism
    # span in the Frobenius geometry makes that penalty invariant to the
    # arbitrary diagonalizing basis.  In particular, any two full-block bases
    # now induce the same symmetric-covariance predictor.
    flattened = raw.reshape(len(raw), -1).T
    left, singular, _ = np.linalg.svd(flattened, full_matrices=False)
    keep = singular > 1e-10 * max(float(singular[0]), 1.0)
    orthonormal = left[:, keep]
    for column in range(orthonormal.shape[1]):
        pivot = int(np.argmax(np.abs(orthonormal[:, column])))
        if orthonormal[pivot, column] < 0:
            orthonormal[:, column] *= -1.0
    atoms = orthonormal.T.reshape(-1, p, p)
    labels = tuple(f"frobenius_mechanism_{index}" for index in range(len(atoms)))
    return atoms, labels


def _fit_environment_parameters(
    model: JBDModel,
    covariance: np.ndarray,
    observed_features: np.ndarray,
) -> np.ndarray:
    pairs = []
    for offset, left in enumerate(observed_features):
        for right in observed_features[offset + 1:]:
            pairs.append((int(left), int(right)))
    pairs = np.asarray(pairs, dtype=int)
    if not len(pairs):
        raise ValueError("too few observed features for covariance reconstruction")
    design = model.atoms[:, pairs[:, 0], pairs[:, 1]].T
    target = (
        covariance[pairs[:, 0], pairs[:, 1]]
        - model.mean_covariance[pairs[:, 0], pairs[:, 1]]
    )
    ridge = float(model.configuration.ridge)
    if design.shape[1] <= design.shape[0]:
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        return np.linalg.solve(gram, design.T @ target)
    # The missing-aware 30-atom full-block control can have 465 mechanisms
    # but fewer observed covariance entries.  The exact ridge-dual solution
    # avoids solving a much larger coefficient-space system for every held-out
    # feature and is algebraically identical for ridge > 0.
    dual = design @ design.T + ridge * np.eye(design.shape[0])
    return design.T @ np.linalg.solve(dual, target)


def masked_reconstruction_rows(
    model: JBDModel,
    covariance: np.ndarray,
    environment_id: str,
) -> list[dict]:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != model.mean_covariance.shape:
        raise ValueError("held-out covariance has the wrong shape")
    available = np.flatnonzero(np.isfinite(np.diag(covariance)))
    if len(available) < 3:
        raise ValueError("held-out covariance has fewer than three observed features")
    observed_submatrix = covariance[np.ix_(available, available)]
    if not np.isfinite(observed_submatrix).all():
        raise ValueError("observed held-out covariance submatrix is incomplete")
    rows = []
    full_symmetric_rank = len(model.feature_names) * (len(model.feature_names) + 1) // 2
    for held_out in available:
        observed = available[available != held_out]
        if len(model.atoms) == full_symmetric_rank:
            # With the complete Frobenius-orthogonal symmetric space, parameters
            # fitted on the observed principal submatrix contain no information
            # about a held-out row. Its ridge prediction is exactly the mean.
            prediction_matrix = model.mean_covariance
        else:
            parameters = _fit_environment_parameters(model, covariance, observed)
            prediction_matrix = (
                model.mean_covariance
                + np.tensordot(parameters, model.atoms, axes=(0, 0))
            )
        for partner in observed:
            actual = float(covariance[held_out, partner])
            prediction = float(prediction_matrix[held_out, partner])
            rows.append({
                "environment": str(environment_id),
                "held_out_feature": model.feature_names[held_out],
                "partner_feature": model.feature_names[int(partner)],
                "actual": actual,
                "prediction": prediction,
                "squared_error": float((actual - prediction) ** 2),
            })
    return rows


def project_to_correlation(matrix: np.ndarray, *, floor: float = 1e-8) -> np.ndarray:
    """Return a deterministic PSD correlation approximation."""

    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    value = np.where(np.isfinite(value), value, 0.0)
    value = 0.5 * (value + value.T)
    np.fill_diagonal(value, 1.0)
    for _ in range(3):
        eigenvalues, eigenvectors = np.linalg.eigh(value)
        value = (eigenvectors * np.maximum(eigenvalues, float(floor))) @ eigenvectors.T
        scale = np.sqrt(np.maximum(np.diag(value), float(floor)))
        value = value / scale[:, None] / scale[None, :]
        value = 0.5 * (value + value.T)
        np.fill_diagonal(value, 1.0)
    return value


def pairwise_psd_mean(covariances: Sequence[np.ndarray]) -> np.ndarray:
    """Pairwise environment mean followed by an explicit PSD-correlation projection."""

    values = np.asarray(covariances, dtype=float)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("covariances must have shape environment x p x p")
    count = np.sum(np.isfinite(values), axis=0)
    total = np.nansum(values, axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    np.fill_diagonal(mean, 1.0)
    return project_to_correlation(mean)


def complete_missing_covariances(
    covariances: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    """Complete environment-wise missing feature rows without altering observations.

    Missing coordinates receive the train-only pairwise PSD mean.  Cross terms
    between observed and missing coordinates are shrunk only as far as needed
    for a valid PSD completion.  Because missingness is by entire feature row
    and column, the observed principal correlation block is retained exactly.
    """

    values = np.asarray(covariances, dtype=float)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("covariances must have shape environment x p x p")
    fallback = pairwise_psd_mean(values)
    output = []
    diagnostics = []
    for environment, covariance in enumerate(values):
        observed = np.flatnonzero(np.isfinite(np.diag(covariance)))
        missing = np.flatnonzero(~np.isfinite(np.diag(covariance)))
        if len(observed) < 3:
            raise ValueError("each environment needs at least three observed features")
        actual = covariance[np.ix_(observed, observed)]
        if not np.isfinite(actual).all():
            raise ValueError("observed feature block is not complete")
        completed = fallback.copy()
        completed[np.ix_(observed, observed)] = actual
        if len(missing):
            cross = fallback[np.ix_(observed, missing)].copy()

            def candidate(shrink: float) -> np.ndarray:
                value = completed.copy()
                value[np.ix_(observed, missing)] = float(shrink) * cross
                value[np.ix_(missing, observed)] = float(shrink) * cross.T
                return 0.5 * (value + value.T)

            if float(np.min(np.linalg.eigvalsh(candidate(1.0)))) >= -1e-9:
                shrink = 1.0
            else:
                lower, upper = 0.0, 1.0
                for _ in range(60):
                    middle = 0.5 * (lower + upper)
                    if float(np.min(np.linalg.eigvalsh(candidate(middle)))) >= -1e-9:
                        lower = middle
                    else:
                        upper = middle
                shrink = lower
            completed = candidate(shrink)
        else:
            shrink = 1.0
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(completed)))
        if minimum_eigenvalue < -2e-9 or not np.isfinite(completed).all():
            raise RuntimeError("failed to produce a finite PSD covariance completion")
        output.append(completed)
        diagnostics.append({
            "environment_index": int(environment),
            "observed_feature_count": int(len(observed)),
            "missing_feature_count": int(len(missing)),
            "cross_covariance_shrink": float(shrink),
            "minimum_eigenvalue": minimum_eigenvalue,
            "maximum_observed_entry_error": float(np.max(np.abs(
                completed[np.ix_(observed, observed)] - actual
            ))),
        })
    return np.asarray(output), diagnostics


def _fit_full_model(
    covariances: np.ndarray,
    feature_names: tuple[str, ...],
    configuration: JBDConfiguration,
) -> JBDModel:
    mean, root, _, whitened, pooled_eigenvalues = whiten_covariances(covariances)
    if configuration.method in {"pca", "pca_full"}:
        eigenvalues, eigenvectors = np.linalg.eigh(mean)
        order = np.argsort(eigenvalues)[::-1]
        mixing = eigenvectors[:, order] * np.sqrt(
            np.maximum(eigenvalues[order], 1e-6)
        )
        transformed = np.asarray([
            np.linalg.pinv(mixing) @ covariance @ np.linalg.pinv(mixing).T
            for covariance in covariances
        ])
        blocks = (
            (tuple(range(mixing.shape[1])),)
            if configuration.method == "pca_full"
            else tuple((index,) for index in range(mixing.shape[1]))
        )
        method_diagnostics = {
            "initializer": "pooled_covariance_eigenvectors",
            "capacity_matched_full_block": configuration.method == "pca_full",
        }
    else:
        basis, rjd = randomized_joint_basis(
            whitened,
            draws=configuration.random_draws,
            seed=configuration.random_seed,
        )
        if configuration.method in {"ajd", "jbd"}:
            basis, jacobi = jacobi_refine(
                whitened,
                basis,
                max_sweeps=configuration.jacobi_sweeps,
                tolerance=configuration.jacobi_tolerance,
            )
        else:
            jacobi = {"jacobi_sweeps": 0}
        mixing = root @ basis
        transformed = np.asarray([basis.T @ matrix @ basis for matrix in whitened])
        blocks = (
            blocks_from_coupling(
                residual_coupling(transformed),
                quantile=configuration.block_quantile,
            )
            if configuration.method == "jbd"
            else tuple((index,) for index in range(mixing.shape[1]))
        )
        method_diagnostics = {**rjd, **jacobi}
    atoms, labels = covariance_atoms(mixing, blocks)
    return JBDModel(
        feature_names=feature_names,
        mean_covariance=mean,
        mixing=mixing,
        blocks=blocks,
        atoms=atoms,
        atom_labels=labels,
        transformed_covariances=transformed,
        configuration=configuration,
        diagnostics={
            **method_diagnostics,
            "pooled_eigenvalues": pooled_eigenvalues,
            "block_sizes": tuple(len(block) for block in blocks),
            "n_blocks": len(blocks),
            "n_covariance_atoms": len(atoms),
            "off_diagonal_energy": off_diagonal_energy(transformed),
            "uses_labels": False,
        },
    )


def _fit_factorial_model(
    covariances: np.ndarray,
    feature_names: tuple[str, ...],
    configuration: JBDConfiguration,
    factor_basis: np.ndarray,
) -> JBDModel:
    factor_basis = np.asarray(factor_basis, dtype=float)
    if factor_basis.ndim != 2 or factor_basis.shape[0] != len(feature_names):
        raise ValueError("factor basis and feature roster disagree")
    factor_basis, _ = np.linalg.qr(factor_basis)
    projected = np.asarray([
        factor_basis.T @ covariance @ factor_basis for covariance in covariances
    ])
    mean_sub, root_sub, _, whitened, pooled_eigenvalues = whiten_covariances(projected)
    basis, rjd = randomized_joint_basis(
        whitened,
        draws=configuration.random_draws,
        seed=configuration.random_seed,
    )
    basis, jacobi = jacobi_refine(
        whitened,
        basis,
        max_sweeps=configuration.jacobi_sweeps,
        tolerance=configuration.jacobi_tolerance,
    )
    mixing = factor_basis @ root_sub @ basis
    transformed = np.asarray([basis.T @ matrix @ basis for matrix in whitened])
    blocks = blocks_from_coupling(
        residual_coupling(transformed),
        quantile=configuration.block_quantile,
    )
    atoms, labels = covariance_atoms(mixing, blocks)
    return JBDModel(
        feature_names=feature_names,
        mean_covariance=np.mean(covariances, axis=0),
        mixing=mixing,
        blocks=blocks,
        atoms=atoms,
        atom_labels=labels,
        transformed_covariances=transformed,
        configuration=configuration,
        diagnostics={
            **rjd,
            **jacobi,
            "factorial_coordinate_rank": int(factor_basis.shape[1]),
            "projected_mean_covariance": mean_sub,
            "pooled_eigenvalues": pooled_eigenvalues,
            "block_sizes": tuple(len(block) for block in blocks),
            "n_blocks": len(blocks),
            "n_covariance_atoms": len(atoms),
            "off_diagonal_energy": off_diagonal_energy(transformed),
            "uses_labels": False,
        },
    )


def fit_jbd_model(
    covariances: Sequence[np.ndarray],
    feature_names: Sequence[str],
    configuration: JBDConfiguration,
    *,
    factor_basis: np.ndarray | None = None,
) -> JBDModel:
    values = _validate_covariances(covariances)
    names = tuple(str(value) for value in feature_names)
    if values.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("feature roster and covariance dimensions disagree")
    methods = {"pca", "pca_full", "rjd", "ajd", "jbd", "factorial_jbd"}
    if configuration.method not in methods:
        raise ValueError(f"unsupported JBD method: {configuration.method}")
    # Canonicalize by feature identity so Jacobi sweep order and seeded RJD are
    # invariant to the caller's column order.  Restore the requested order at
    # the public boundary.
    order = np.asarray(sorted(range(len(names)), key=lambda index: names[index]))
    inverse = np.argsort(order)
    sorted_names = tuple(names[index] for index in order)
    sorted_values = values[:, order][:, :, order]
    if configuration.method == "factorial_jbd":
        if factor_basis is None:
            raise ValueError("factorial_jbd requires a factor basis")
        sorted_basis = np.asarray(factor_basis, dtype=float)[order]
        model = _fit_factorial_model(
            sorted_values, sorted_names, configuration, sorted_basis
        )
    else:
        if factor_basis is not None:
            raise ValueError("factor_basis is legal only for factorial_jbd")
        model = _fit_full_model(sorted_values, sorted_names, configuration)
    return replace(
        model,
        feature_names=names,
        mean_covariance=model.mean_covariance[np.ix_(inverse, inverse)],
        mixing=model.mixing[inverse],
        atoms=model.atoms[:, inverse][:, :, inverse],
    )


def fit_pca_block_model(
    covariances: Sequence[np.ndarray],
    feature_names: Sequence[str],
    *,
    ridge: float,
    block_sizes: Sequence[int],
) -> JBDModel:
    """Pooled-PCA control with a prescribed covariance-mechanism capacity.

    Blocks are contiguous in descending pooled-PCA order.  Passing the block
    sizes recovered by a train-only JBD fit yields the same number of covariance
    atoms and the same ridge, isolating the value of the JBD orientation from
    the generic value of a block-sparse covariance span.
    """

    names = tuple(str(value) for value in feature_names)
    # The control uses a fixed strongest-first allocation in descending pooled-
    # PCA order; it does not inherit JBD component identities.  Sorting makes
    # the comparator invariant to the arbitrary ordering of recovered blocks
    # and gives the largest allowed interaction block to the leading PCs.
    sizes = tuple(sorted((int(value) for value in block_sizes), reverse=True))
    if any(size < 1 for size in sizes) or sum(sizes) != len(names):
        raise ValueError("block sizes must be positive and partition the roster")
    configuration = JBDConfiguration(method="pca_full", ridge=float(ridge))
    model = fit_jbd_model(covariances, names, configuration)
    blocks = []
    start = 0
    for size in sizes:
        blocks.append(tuple(range(start, start + size)))
        start += size
    atoms, labels = covariance_atoms(model.mixing, blocks)
    return replace(
        model,
        blocks=tuple(blocks),
        atoms=atoms,
        atom_labels=labels,
        configuration=replace(configuration, method="pca_matched"),
        diagnostics={
            **model.diagnostics,
            "capacity_matched_full_block": False,
            "capacity_matched_to_jbd_block_sizes": sizes,
            "block_sizes": sizes,
            "n_blocks": len(sizes),
            "n_covariance_atoms": len(atoms),
        },
    )


def cross_validated_mse(
    covariances: Sequence[np.ndarray],
    environment_ids: Sequence[str],
    feature_names: Sequence[str],
    configuration: JBDConfiguration,
    *,
    factor_basis: np.ndarray | None = None,
) -> float:
    values = _validate_covariances(covariances)
    errors = []
    for held_out in range(len(values)):
        model = fit_jbd_model(
            np.delete(values, held_out, axis=0),
            feature_names,
            configuration,
            factor_basis=factor_basis,
        )
        rows = masked_reconstruction_rows(
            model, values[held_out], str(environment_ids[held_out])
        )
        errors.extend(row["squared_error"] for row in rows)
    return float(np.mean(errors))


def align_mixing(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Match scale-free mixing columns and report absolute cosine recovery."""

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    reference = reference / np.maximum(np.linalg.norm(reference, axis=0), EPS)
    candidate = candidate / np.maximum(np.linalg.norm(candidate, axis=0), EPS)
    similarity = np.abs(reference.T @ candidate)
    rows, columns = linear_sum_assignment(-similarity)
    matched = similarity[rows, columns]
    return {
        "mean_absolute_cosine": float(np.mean(matched)),
        "minimum_absolute_cosine": float(np.min(matched)),
        "matched_absolute_cosines": matched,
    }


def block_membership(blocks: Sequence[Sequence[int]], rank: int) -> np.ndarray:
    membership = np.zeros((rank, rank), dtype=float)
    for block in blocks:
        indices = np.asarray(block, dtype=int)
        membership[np.ix_(indices, indices)] = 1.0
    return membership


def mechanism_subspace_overlap(reference: JBDModel, candidate: JBDModel) -> dict:
    """Compare covariance-mechanism spans, invariant to within-block rotation."""

    p = len(reference.feature_names)
    if candidate.feature_names != reference.feature_names:
        raise ValueError("mechanism models use different feature rosters")
    upper = np.triu_indices(p)

    def orthogonal_span(model):
        matrix = model.atoms[:, upper[0], upper[1]].T
        left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        keep = singular > 1e-9 * max(float(singular[0]), 1.0)
        return left[:, keep]

    left = orthogonal_span(reference)
    right = orthogonal_span(candidate)
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    common_rank = min(left.shape[1], right.shape[1])
    overlap = float(np.sum(singular[:common_rank] ** 2) / max(common_rank, 1))
    return {
        "projector_overlap_on_smaller_span": overlap,
        "reference_mechanism_rank": int(left.shape[1]),
        "candidate_mechanism_rank": int(right.shape[1]),
        "rank_ratio": float(
            min(left.shape[1], right.shape[1]) / max(left.shape[1], right.shape[1])
        ),
        "minimum_principal_cosine": float(np.min(singular[:common_rank])),
    }


def shuffled_environment_row_null(
    residual_matrices: Sequence[np.ndarray],
    *,
    seed: int,
) -> np.ndarray:
    """Erase environments by reassigning pooled rows, then recompute PSD correlations.

    The null preserves every environment's sample count and the pooled empirical
    residual distribution.  Unlike independently shuffling covariance entries,
    every returned matrix is a realizable positive-semidefinite correlation
    matrix.
    """

    matrices = tuple(np.asarray(matrix, dtype=float) for matrix in residual_matrices)
    if len(matrices) < 2 or any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("residual_matrices must contain multiple two-dimensional arrays")
    p = matrices[0].shape[1]
    if p < 2 or any(matrix.shape[1] != p for matrix in matrices):
        raise ValueError("all residual matrices must share a nontrivial feature roster")
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("residual matrices must be finite")
    sizes = [len(matrix) for matrix in matrices]
    if any(size < p + 1 for size in sizes):
        raise ValueError("each null environment needs more rows than features")
    pooled = np.concatenate(matrices, axis=0)
    rng = np.random.default_rng(int(seed))
    pooled = pooled[rng.permutation(len(pooled))]
    output = []
    start = 0
    for size in sizes:
        local = pooled[start:start + size]
        start += size
        local = local - local.mean(axis=0, keepdims=True)
        scale = local.std(axis=0, keepdims=True)
        scale[scale < EPS] = 1.0
        local = local / scale
        covariance = local.T @ local / len(local)
        output.append(0.5 * (covariance + covariance.T))
    return np.asarray(output)


def balanced_environment_row_null(
    residual_matrices: Sequence[np.ndarray],
    *,
    seed: int,
) -> np.ndarray:
    """Build PSD null environments with equal contribution from every source cell."""

    matrices = tuple(np.asarray(matrix, dtype=float) for matrix in residual_matrices)
    if len(matrices) < 2 or any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("residual_matrices must contain multiple two-dimensional arrays")
    p = matrices[0].shape[1]
    if any(matrix.shape[1] != p or not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("residual matrices must share a finite feature roster")
    rng = np.random.default_rng(int(seed))
    output = []
    source_count = len(matrices)
    for target_size in map(len, matrices):
        base, remainder = divmod(target_size, source_count)
        allocations = np.full(source_count, base, dtype=int)
        if remainder:
            allocations[rng.permutation(source_count)[:remainder]] += 1
        rows = []
        for matrix, count in zip(matrices, allocations):
            if count:
                rows.append(matrix[rng.choice(len(matrix), size=count, replace=True)])
        local = np.concatenate(rows, axis=0)
        local = local[rng.permutation(len(local))]
        local -= local.mean(axis=0, keepdims=True)
        scale = local.std(axis=0, keepdims=True)
        scale[scale < EPS] = 1.0
        local /= scale
        covariance = local.T @ local / len(local)
        output.append(0.5 * (covariance + covariance.T))
    return np.asarray(output)


def missingness_preserving_stationary_null(
    covariances: Sequence[np.ndarray],
    sample_counts: Sequence[int],
    *,
    seed: int,
    reference_covariances: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """PSD null with one shared covariance and the original missingness pattern.

    Each cell receives an independent Gaussian sample from the train-pool
    pairwise PSD correlation, restricted to that cell's observed coordinates.
    Thus sample size and feature availability are retained while coherent
    environment-specific covariance trajectories are removed.
    """

    values = np.asarray(covariances, dtype=float)
    counts = np.asarray(sample_counts, dtype=int)
    if values.ndim != 3 or len(values) != len(counts):
        raise ValueError("covariances and sample counts disagree")
    if np.any(counts < 4):
        raise ValueError("stationary null needs at least four rows per environment")
    reference = values if reference_covariances is None else np.asarray(
        reference_covariances, dtype=float
    )
    fallback = pairwise_psd_mean(reference)
    rng = np.random.default_rng(int(seed))
    output = np.full_like(values, np.nan)
    for index, (covariance, count) in enumerate(zip(values, counts)):
        observed = np.flatnonzero(np.isfinite(np.diag(covariance)))
        local_covariance = fallback[np.ix_(observed, observed)]
        sample = rng.multivariate_normal(
            np.zeros(len(observed)), local_covariance, size=int(count)
        )
        sample -= sample.mean(axis=0, keepdims=True)
        scale = sample.std(axis=0, keepdims=True)
        scale[scale < EPS] = 1.0
        sample /= scale
        empirical = sample.T @ sample / len(sample)
        output[index][np.ix_(observed, observed)] = 0.5 * (
            empirical + empirical.T
        )
    return output


__all__ = [
    "JBDConfiguration",
    "JBDModel",
    "align_mixing",
    "balanced_environment_row_null",
    "block_membership",
    "blocks_from_coupling",
    "complete_missing_covariances",
    "covariance_atoms",
    "cross_validated_mse",
    "fit_jbd_model",
    "fit_pca_block_model",
    "jacobi_refine",
    "masked_reconstruction_rows",
    "matrix_sqrt_and_inverse",
    "mechanism_subspace_overlap",
    "missingness_preserving_stationary_null",
    "off_diagonal_energy",
    "pairwise_psd_mean",
    "project_to_correlation",
    "randomized_joint_basis",
    "residual_coupling",
    "shuffled_environment_row_null",
    "whiten_covariances",
]
