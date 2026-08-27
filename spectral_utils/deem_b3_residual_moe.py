"""Residual-family mixture extensions for a frozen five-seed B3 ensemble.

The historical neutral-residual method used IU-PCR as its baseline.  This
module makes the baseline generic and, for the new experiment, uses the exact
mean-of-five B3 posterior instead.  No target labels are accepted.

For a cell, let ``b`` be the logit of the frozen B3 ensemble posterior and let
``s_g`` be the mean aligned contribution of provenance family ``g``.  Each
family expert is converted to an orthogonal residual coordinate

    r_g = standardize(s_g) - loading_g * standardize(b).

The residual columns are standardized again.  A calibration over donor cells
selects the residual-covariance eigenmode closest to the unit-variance null,
following the mechanism that previously worked for IU-PCR.  Optional gates
act on the *residual expert terms*, not on raw agreement:

    gamma_ig = (1-rho) + rho * G * softmax(a_ig),
    correction_i = sum_g gamma_ig r_ig v_g.

Thus ``trust == 0`` is an exact array alias of the frozen B3 score, while
``gate_strength == 0`` is the ungated neutral-residual correction.  The gate
uses only donor-derived family survival, per-sample residual novelty, and
five-seed stability.  Multipliers are positive, bounded, and mean one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.special import expit, logit, ndtr
from sklearn.linear_model import Ridge

from .adapted_dufs import adapted_dufs_soft_gates


EPS = 1e-12
FAMILY_ORDER = (
    "entropy_level",
    "entropy_dynamics",
    "sampled_token_energy",
    "partition_energy",
    "topk_distribution",
    "structural",
)


@dataclass(frozen=True)
class FrozenB3Ensemble:
    """Exact aligned outputs of the historical five-seed B3 ensemble."""

    cell_id: str
    seeds: tuple[int, ...]
    score: np.ndarray
    seed_scores: np.ndarray
    seed_logits: np.ndarray
    seed_biases: np.ndarray
    seed_family_contributions: np.ndarray
    present_mask: np.ndarray
    family_order: tuple[str, ...] = FAMILY_ORDER
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ResidualCell:
    """One cell expressed in B3-orthogonal family residual coordinates."""

    cell_id: str
    baseline_score: np.ndarray
    baseline_logit: np.ndarray
    baseline_mean: float
    baseline_scale: float
    baseline_z: np.ndarray
    family_mean: np.ndarray
    contribution_mean: np.ndarray
    contribution_scale: np.ndarray
    baseline_loadings: np.ndarray
    residual_mean: np.ndarray
    residual_scale: np.ndarray
    residuals: np.ndarray
    seed_instability: np.ndarray
    loo_residuals: np.ndarray
    loo_seed_instability: np.ndarray
    loo_predictability: np.ndarray
    present_mask: np.ndarray
    family_order: tuple[str, ...] = FAMILY_ORDER
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ResidualCalibration:
    """Donor-only neutral direction and optional family survival prior."""

    direction: np.ndarray
    residual_covariance: np.ndarray
    eigenvalues: np.ndarray
    selected_index: int
    family_survival: np.ndarray
    family_order: tuple[str, ...] = FAMILY_ORDER
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ResidualMoEScore:
    """B3 plus a bounded correction from gated residual family experts."""

    score: np.ndarray
    logit: np.ndarray
    correction_z: np.ndarray
    raw_correction: np.ndarray
    expert_terms: np.ndarray
    gate_probabilities: np.ndarray
    gates: np.ndarray
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphRoughnessMoment:
    """Within-cell graph cross-roughness moments for PGRD-style directions."""

    cell_id: str
    residual_source: str
    a0: np.ndarray
    c0: np.ndarray
    present_mask: np.ndarray
    trace_a0: float
    diagnostics: dict = field(default_factory=dict)


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    return logit(np.clip(values, 1e-12, 1.0 - 1e-12))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_b3_ensemble(
    baseline_dir: str | Path,
    cell_id: str,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    family_order: Sequence[str] = FAMILY_ORDER,
    expected_bundle_sha256: str | None = None,
    expected_ordered_row_id_sha256: str | None = None,
) -> FrozenB3Ensemble:
    """Load and mechanically verify aligned frozen B3 seed artifacts."""

    root = Path(baseline_dir) / "fits" / str(cell_id)
    order = tuple(str(value) for value in family_order)
    seed_scores = []
    seed_logits = []
    seed_biases = []
    seed_family = []
    present_mask = None
    reconstruction_errors = []
    row_count = None
    for seed in seeds:
        path = root / f"B3__seed{int(seed)}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        unhashed = dict(metadata)
        expected_content = unhashed.pop("content_sha256", None)
        if (
            metadata.get("status") != "complete"
            or metadata.get("arm_id") != "B3"
            or metadata.get("cell_id") != str(cell_id)
            or metadata.get("stem") != f"B3__seed{int(seed)}"
            or int(metadata.get("seed", -1)) != int(seed)
            or _canonical_sha256(unhashed) != expected_content
            or _sha256_file(path) != metadata.get("array_sha256")
            or (
                expected_bundle_sha256 is not None
                and metadata.get("bundle_sha256") != expected_bundle_sha256
            )
            or (
                expected_ordered_row_id_sha256 is not None
                and metadata.get("ordered_row_id_sha256")
                != expected_ordered_row_id_sha256
            )
        ):
            raise ValueError(f"frozen B3 hash mismatch for {cell_id}/seed{seed}")
        with np.load(path, allow_pickle=False) as arrays:
            score = np.asarray(arrays["score"], dtype=float)
            ell = np.asarray(arrays["logit"], dtype=float)
            bias = float(metadata["aligned_bias"])
            current_present = np.asarray(
                [f"family_contribution__{name}" in arrays.files for name in order],
                dtype=bool,
            )
            if present_mask is None:
                present_mask = current_present
            elif not np.array_equal(present_mask, current_present):
                raise ValueError(f"B3 family roster changes across seeds for {cell_id}")
            family = np.column_stack(
                [
                    (
                        np.asarray(arrays[f"family_contribution__{name}"], dtype=float)
                        if current_present[column]
                        else np.zeros_like(score)
                    )
                    for column, name in enumerate(order)
                ]
            )
        if row_count is None:
            row_count = len(score)
        if score.shape != (row_count,) or ell.shape != (row_count,):
            raise ValueError(f"B3 seed row mismatch for {cell_id}/seed{seed}")
        if family.shape != (row_count, len(order)):
            raise ValueError(f"B3 family shape mismatch for {cell_id}/seed{seed}")
        reconstruction = bias + family.sum(axis=1)
        error = float(np.max(np.abs(reconstruction - ell)))
        if error > 1e-8:
            raise ValueError(
                f"B3 family reconstruction failed for {cell_id}/seed{seed}: {error:.3e}"
            )
        if not np.isfinite(score).all() or not np.isfinite(family).all():
            raise ValueError(f"non-finite B3 artifact for {cell_id}/seed{seed}")
        posterior_error = float(np.max(np.abs(expit(ell) - score)))
        if posterior_error > 1e-10:
            raise ValueError(
                f"B3 posterior/logit mismatch for {cell_id}/seed{seed}: "
                f"{posterior_error:.3e}"
            )
        seed_scores.append(score)
        seed_logits.append(ell)
        seed_biases.append(bias)
        seed_family.append(family)
        reconstruction_errors.append(error)
    score_matrix = np.asarray(seed_scores, dtype=float)
    score = score_matrix.mean(axis=0)
    return FrozenB3Ensemble(
        cell_id=str(cell_id),
        seeds=tuple(int(seed) for seed in seeds),
        score=score,
        seed_scores=score_matrix,
        seed_logits=np.asarray(seed_logits, dtype=float),
        seed_biases=np.asarray(seed_biases, dtype=float),
        seed_family_contributions=np.asarray(seed_family, dtype=float),
        present_mask=np.asarray(present_mask, dtype=bool),
        family_order=order,
        diagnostics={
            "n_rows": int(row_count or 0),
            "n_seeds": int(len(seed_scores)),
            "n_present_families": int(np.sum(present_mask)),
            "max_family_reconstruction_error": float(max(reconstruction_errors)),
            "mean_seed_score_sd": float(np.mean(np.std(score_matrix, axis=0))),
        },
    )


def build_residual_cell(
    ensemble: FrozenB3Ensemble,
    *,
    baseline_score: np.ndarray | None = None,
    folds: np.ndarray | None = None,
) -> ResidualCell:
    """Residualize mean B3 family contributions against a B3 teacher score.

    ``baseline_score`` allows iterative refinement.  The initial call omits it
    and therefore uses the exact frozen mean-of-five posterior.  A common
    transform is fitted to the ensemble-mean family contributions; seed
    instability is measured in those same standardized coordinates.
    """

    score = ensemble.score if baseline_score is None else np.asarray(baseline_score, dtype=float)
    if score.shape != ensemble.score.shape or not np.isfinite(score).all():
        raise ValueError("baseline_score must be a finite vector aligned to B3")
    baseline_logit = _safe_logit(score)
    baseline_mean = float(np.mean(baseline_logit))
    baseline_scale = float(np.std(baseline_logit))
    if baseline_scale <= EPS:
        raise ValueError("B3 ensemble score is constant")
    baseline_z = (baseline_logit - baseline_mean) / baseline_scale

    seed_family = np.asarray(ensemble.seed_family_contributions, dtype=float)
    family_mean = seed_family.mean(axis=0)
    present = np.asarray(ensemble.present_mask, dtype=bool)
    if int(np.sum(present)) < 3:
        raise ValueError("B3 residual MoE requires at least three present families")
    contribution_mean = np.zeros(len(ensemble.family_order), dtype=float)
    contribution_scale = np.ones(len(ensemble.family_order), dtype=float)
    contribution_mean[present] = family_mean[:, present].mean(axis=0)
    contribution_scale[present] = family_mean[:, present].std(axis=0)
    contribution_scale = np.where(contribution_scale > EPS, contribution_scale, 1.0)
    standardized = np.zeros_like(family_mean)
    standardized[:, present] = (
        family_mean[:, present] - contribution_mean[None, present]
    ) / contribution_scale[None, present]
    denominator = float(np.dot(baseline_z, baseline_z))
    loadings = np.zeros(len(ensemble.family_order), dtype=float)
    loadings[present] = baseline_z @ standardized[:, present] / max(denominator, EPS)
    raw_residuals = standardized - baseline_z[:, None] * loadings[None, :]
    residual_mean = np.zeros(len(ensemble.family_order), dtype=float)
    residual_scale = np.ones(len(ensemble.family_order), dtype=float)
    residual_mean[present] = raw_residuals[:, present].mean(axis=0)
    residual_scale[present] = raw_residuals[:, present].std(axis=0)
    residual_scale = np.where(residual_scale > EPS, residual_scale, 1.0)
    residuals = np.zeros_like(raw_residuals)
    residuals[:, present] = (
        raw_residuals[:, present] - residual_mean[None, present]
    ) / residual_scale[None, present]

    standardized_seeds = (
        seed_family - contribution_mean[None, None, :]
    ) / contribution_scale[None, None, :]
    seed_raw_residuals = (
        standardized_seeds
        - baseline_z[None, :, None] * loadings[None, None, :]
    )
    seed_residuals = (
        seed_raw_residuals - residual_mean[None, None, :]
    ) / residual_scale[None, None, :]
    seed_instability = np.std(seed_residuals, axis=0)
    seed_instability[:, ~present] = 0.0

    # A second residual has a different purpose.  It is the genuine
    # leave-one-family-out novelty signal used by the router: predict family g
    # only from the other present family contributions on grouped donor rows.
    # Unlike the neutral-mode residual above, it is divided by donor target SD
    # rather than residual SD, so a nearly deterministic family stays close to
    # zero instead of being inflated back to unit variance.
    n_rows = len(family_mean)
    if folds is None:
        fold_values = np.arange(n_rows, dtype=int) % 5
    else:
        fold_values = np.asarray(folds, dtype=int)
        if fold_values.shape != (n_rows,) or len(np.unique(fold_values)) < 2:
            raise ValueError("folds must align to rows and contain at least two folds")
    loo_residuals = np.zeros_like(family_mean)
    loo_seed_residuals = np.zeros_like(seed_family)
    loo_predictability = np.zeros(len(ensemble.family_order), dtype=float)
    for family_index in np.flatnonzero(present):
        predictors = np.asarray(
            [index for index in np.flatnonzero(present) if index != family_index],
            dtype=int,
        )
        standardized_target = np.zeros(n_rows, dtype=float)
        for fold in sorted(np.unique(fold_values)):
            held = np.flatnonzero(fold_values == fold)
            donor = np.flatnonzero(fold_values != fold)
            if not len(held) or len(donor) <= len(predictors):
                raise ValueError("invalid grouped fold for family residualization")
            donor_x = family_mean[np.ix_(donor, predictors)]
            held_x = family_mean[np.ix_(held, predictors)]
            x_mean = donor_x.mean(axis=0)
            x_scale = donor_x.std(axis=0)
            x_scale = np.where(x_scale > EPS, x_scale, 1.0)
            donor_x = (donor_x - x_mean[None, :]) / x_scale[None, :]
            held_x = (held_x - x_mean[None, :]) / x_scale[None, :]
            donor_y = family_mean[donor, family_index]
            y_mean = float(np.mean(donor_y))
            y_scale = float(np.std(donor_y))
            y_scale = y_scale if y_scale > EPS else 1.0
            donor_y = (donor_y - y_mean) / y_scale
            held_y = (family_mean[held, family_index] - y_mean) / y_scale
            estimator = Ridge(alpha=1.0, fit_intercept=True)
            estimator.fit(donor_x, donor_y)
            prediction = estimator.predict(held_x)
            loo_residuals[held, family_index] = held_y - prediction
            standardized_target[held] = held_y
            for seed_index in range(len(seed_family)):
                seed_x = (
                    seed_family[seed_index][np.ix_(held, predictors)]
                    - x_mean[None, :]
                ) / x_scale[None, :]
                seed_y = (
                    seed_family[seed_index, held, family_index] - y_mean
                ) / y_scale
                loo_seed_residuals[seed_index, held, family_index] = (
                    seed_y - estimator.predict(seed_x)
                )
        denominator = float(np.sum(standardized_target ** 2))
        loo_predictability[family_index] = 1.0 - float(
            np.sum(loo_residuals[:, family_index] ** 2) / max(denominator, EPS)
        )
    loo_seed_instability = np.std(loo_seed_residuals, axis=0)
    loo_seed_instability[:, ~present] = 0.0

    covariance = residuals.T @ residuals / len(residuals)
    orthogonality = baseline_z @ residuals / len(residuals)
    return ResidualCell(
        cell_id=ensemble.cell_id,
        baseline_score=np.asarray(score, dtype=float).copy(),
        baseline_logit=baseline_logit,
        baseline_mean=baseline_mean,
        baseline_scale=baseline_scale,
        baseline_z=baseline_z,
        family_mean=family_mean,
        contribution_mean=contribution_mean,
        contribution_scale=contribution_scale,
        baseline_loadings=np.asarray(loadings, dtype=float),
        residual_mean=residual_mean,
        residual_scale=residual_scale,
        residuals=residuals,
        seed_instability=seed_instability,
        loo_residuals=loo_residuals,
        loo_seed_instability=loo_seed_instability,
        loo_predictability=loo_predictability,
        present_mask=present,
        family_order=ensemble.family_order,
        diagnostics={
            "residual_covariance_diagonal_error": float(
                np.max(np.abs(np.diag(covariance)[present] - 1.0))
            ),
            "baseline_residual_covariance_max_abs": float(np.max(np.abs(orthogonality))),
            "mean_seed_instability": float(np.mean(seed_instability)),
            "mean_loo_seed_instability": float(
                np.mean(loo_seed_instability[:, present])
            ),
            "loo_predictability": loo_predictability.tolist(),
            "n_present_families": int(np.sum(present)),
        },
    )


def _neutral_direction(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    selected = int(np.argmin(np.abs(eigenvalues - 1.0)))
    direction = np.asarray(eigenvectors[:, selected], dtype=float)
    anchor = float(np.sum(direction))
    if abs(anchor) <= EPS:
        pivot = int(np.argmax(np.abs(direction)))
        sign = 1.0 if direction[pivot] >= 0.0 else -1.0
    else:
        sign = 1.0 if anchor > 0.0 else -1.0
    return direction * sign, eigenvalues, selected


def fit_residual_calibration(
    cells: Iterable[ResidualCell],
    *,
    survival: str = "uniform",
    dufs_seeds: Sequence[int] = (0, 1, 2),
    dufs_epochs: int = 120,
) -> ResidualCalibration:
    """Fit a neutral B3 residual mode and a label-free family gate prior.

    ``survival='jackknife'`` treats direction stability across donor cells as
    a stochastic-gate survival probability.  ``'dufs'`` uses the raw STG
    survival returned by adapted-DUFS on stacked residual coordinates.
    ``'hybrid'`` takes their geometric mean.  The neutral direction itself is
    always computed before this optional prior and remains the primary signal.
    """

    records = tuple(cells)
    if len(records) < 3:
        raise ValueError("at least three donor cells are required")
    if any(tuple(record.family_order) != FAMILY_ORDER for record in records):
        raise ValueError("residual cells do not share the frozen family order")
    def pooled_covariance(selected: Sequence[ResidualCell]) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros((len(FAMILY_ORDER), len(FAMILY_ORDER)), dtype=float)
        counts = np.zeros_like(total, dtype=int)
        for record in selected:
            present = np.flatnonzero(record.present_mask)
            local = record.residuals[:, present].T @ record.residuals[:, present]
            local /= len(record.residuals)
            total[np.ix_(present, present)] += local
            counts[np.ix_(present, present)] += 1
        if np.any(counts == 0):
            raise ValueError("donor cells do not jointly cover every family pair")
        covariance = total / counts
        return 0.5 * (covariance + covariance.T), counts

    covariance, pair_counts = pooled_covariance(records)
    direction, eigenvalues, selected = _neutral_direction(covariance)

    jackknife = []
    for held in range(len(records)):
        donor = [record for index, record in enumerate(records) if index != held]
        child_covariance, _ = pooled_covariance(donor)
        child, _, _ = _neutral_direction(child_covariance)
        if float(np.dot(child, direction)) < 0.0:
            child = -child
        jackknife.append(child)
    jackknife = np.asarray(jackknife, dtype=float)
    component_mean = jackknife.mean(axis=0)
    component_sd = jackknife.std(axis=0)
    # A small denominator floor prevents a numerically constant tiny component
    # from receiving artificial certainty.
    z = np.abs(component_mean) / np.maximum(component_sd, 0.05)
    jackknife_survival = ndtr(z)

    dufs_survival = np.ones(len(FAMILY_ORDER), dtype=float)
    dufs_diagnostics = {"mean_seed_std": 0.0}
    if survival in {"dufs", "hybrid"}:
        fully_present = [record for record in records if bool(np.all(record.present_mask))]
        if len(fully_present) < 3:
            raise ValueError("DUFS survival requires three fully covered donor cells")
        stacked = np.row_stack([record.residuals for record in fully_present])
        _, dufs_diagnostics = adapted_dufs_soft_gates(
            stacked.T,
            seeds=tuple(int(seed) for seed in dufs_seeds),
            epochs=int(dufs_epochs),
        )
        dufs_survival = np.asarray(
            dufs_diagnostics["raw_probabilities"], dtype=float
        )
    if survival == "uniform":
        family_survival = np.ones(len(FAMILY_ORDER), dtype=float)
    elif survival == "jackknife":
        family_survival = jackknife_survival
    elif survival == "dufs":
        family_survival = dufs_survival
    elif survival == "hybrid":
        family_survival = np.sqrt(jackknife_survival * dufs_survival)
    else:
        raise ValueError("survival must be uniform, jackknife, dufs, or hybrid")
    family_survival = np.clip(family_survival, 1e-6, 1.0)
    return ResidualCalibration(
        direction=direction,
        residual_covariance=covariance,
        eigenvalues=eigenvalues,
        selected_index=selected,
        family_survival=family_survival,
        diagnostics={
            "n_donor_cells": int(len(records)),
            "n_donor_rows": int(sum(len(record.residuals) for record in records)),
            "pair_counts": pair_counts.tolist(),
            "selected_eigenvalue": float(eigenvalues[selected]),
            "distance_from_unit": float(abs(eigenvalues[selected] - 1.0)),
            "direction_anchor_dot": float(np.sum(direction)),
            "jackknife_min_abs_cosine": float(
                np.min(np.abs(jackknife @ direction))
            ),
            "jackknife_survival": jackknife_survival.tolist(),
            "dufs_survival": dufs_survival.tolist(),
            "survival_mode": str(survival),
            "dufs_mean_seed_std": float(dufs_diagnostics["mean_seed_std"]),
            "uses_labels": False,
        },
    )


def score_residual_moe(
    cell: ResidualCell,
    calibration: ResidualCalibration,
    *,
    trust: float | None = None,
    gate_strength: float = 0.0,
    novelty_strength: float = 1.0,
    stability_strength: float = 1.0,
    temperature: float = 1.0,
) -> ResidualMoEScore:
    """Apply a bounded residual-expert gate on top of exact ensemble B3.

    ``trust`` is measured in standard deviations of the B3 logit coordinate;
    the historical neutral residual default is ``1 / G``.  Gate strength is a
    convex interpolation between using every residual expert equally and a
    sample-dependent mean-one softmax gate.
    """

    if tuple(cell.family_order) != tuple(calibration.family_order):
        raise ValueError("cell and calibration family orders disagree")
    present = np.asarray(cell.present_mask, dtype=bool)
    count = int(np.sum(present))
    trust = 1.0 / count if trust is None else float(trust)
    gate_strength = float(gate_strength)
    if trust < 0.0 or not 0.0 <= gate_strength <= 1.0:
        raise ValueError("trust must be nonnegative and gate_strength in [0,1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if trust == 0.0:
        uniform = np.zeros((len(cell.baseline_score), len(cell.family_order)), dtype=float)
        uniform[:, present] = 1.0 / count
        identity_gates = np.zeros_like(uniform)
        identity_gates[:, present] = 1.0
        return ResidualMoEScore(
            score=cell.baseline_score,
            logit=cell.baseline_logit,
            correction_z=np.zeros(len(cell.baseline_score)),
            raw_correction=np.zeros(len(cell.baseline_score)),
            expert_terms=np.zeros_like(cell.residuals),
            gate_probabilities=uniform,
            gates=identity_gates,
            diagnostics={"exact_b3_alias": True, "trust": 0.0},
        )

    expert_terms = cell.residuals * calibration.direction[None, :]
    row_instability_scale = np.median(
        cell.loo_seed_instability[:, present], axis=1, keepdims=True
    )
    row_instability_scale = np.maximum(row_instability_scale, EPS)
    stability = 1.0 / (
        1.0 + cell.loo_seed_instability / row_instability_scale
    )
    novelty = np.abs(cell.loo_residuals)
    novelty = novelty / np.maximum(
        np.median(novelty[:, present], axis=1, keepdims=True), EPS
    )
    logits = np.full_like(expert_terms, -np.inf)
    logits[:, present] = (
        np.log(calibration.family_survival[None, :])
        + float(novelty_strength) * np.log1p(novelty)
        + float(stability_strength) * np.log(np.clip(stability, 1e-12, None))
    )[:, present] / float(temperature)
    logits = logits - np.max(logits[:, present], axis=1, keepdims=True)
    probabilities = np.zeros_like(logits)
    probabilities[:, present] = np.exp(logits[:, present])
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    gates = np.zeros_like(probabilities)
    gates[:, present] = (
        (1.0 - gate_strength) + gate_strength * count * probabilities[:, present]
    )
    raw_correction = np.sum(gates * expert_terms, axis=1)
    raw_scale = float(np.std(raw_correction))
    if raw_scale <= EPS:
        correction_z = np.zeros(len(raw_correction), dtype=float)
    else:
        correction_z = trust * raw_correction / raw_scale
    updated_logit = cell.baseline_logit + cell.baseline_scale * correction_z
    score = expit(updated_logit)
    reconstruction = float(
        np.max(
            np.abs(
                updated_logit
                - (cell.baseline_logit + cell.baseline_scale * correction_z)
            )
        )
    )
    return ResidualMoEScore(
        score=score,
        logit=updated_logit,
        correction_z=correction_z,
        raw_correction=raw_correction,
        expert_terms=expert_terms,
        gate_probabilities=probabilities,
        gates=gates,
        diagnostics={
            "exact_b3_alias": False,
            "trust": trust,
            "gate_strength": gate_strength,
            "n_present_families": count,
            "gate_sum_max_abs_error": float(np.max(np.abs(gates.sum(axis=1) - count))),
            "gate_min": float(np.min(gates)),
            "gate_max": float(np.max(gates)),
            "gate_mean_abs_deviation_from_one": float(np.mean(np.abs(gates - 1.0))),
            "raw_correction_scale": raw_scale,
            "correction_scale": float(np.std(correction_z)),
            "logit_reconstruction_max_abs": reconstruction,
            "uses_labels": False,
        },
    )


def _row_tie_keys(row_ids: Sequence[str]) -> np.ndarray:
    values = np.asarray([str(value) for value in row_ids], dtype=str)
    if len(values) != len(set(values.tolist())):
        raise ValueError("row IDs must be unique for duplicate-safe graph ties")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def graph_roughness_moment(
    cell: ResidualCell,
    row_ids: Sequence[str],
    *,
    residual_source: str = "baseline",
    k: int = 7,
) -> GraphRoughnessMoment:
    """Compute the PGRD cross-gradient moment without labels.

    ``baseline`` reproduces the standardized B3-orthogonal residual geometry.
    ``loo`` instead uses genuine leave-one-family-out novelty coordinates.  In
    both cases the graph and cross-gradient are built inside the cell.
    """

    from .graph_topology import self_safe_knn_graph
    from .laplacian_upcr import symmetric_normalized_laplacian

    present = np.asarray(cell.present_mask, dtype=bool)
    if residual_source == "baseline":
        residuals = np.asarray(cell.residuals[:, present], dtype=float)
    elif residual_source == "loo":
        residuals = np.asarray(cell.loo_residuals[:, present], dtype=float)
    else:
        raise ValueError("residual_source must be baseline or loo")
    if residuals.shape[1] < 3 or residuals.shape[0] != len(row_ids):
        raise ValueError("PGRD requires aligned rows and three residual families")
    graph = self_safe_knn_graph(
        residuals, k=int(k), tie_keys=_row_tie_keys(row_ids)
    )
    laplacian = symmetric_normalized_laplacian(graph)
    n = len(residuals)
    local_a0 = np.asarray(residuals.T @ (laplacian @ residuals) / n, dtype=float)
    local_a0 = 0.5 * (local_a0 + local_a0.T)
    local_c0 = np.asarray(
        residuals.T @ (laplacian @ cell.baseline_z) / n, dtype=float
    )
    trace_a0 = float(np.trace(local_a0))
    if not np.isfinite(trace_a0) or trace_a0 <= EPS:
        raise ValueError("PGRD residual graph has nonpositive roughness trace")
    # PGRD's transferable unit is one trace-normalized cell moment.  Scaling
    # before pooling prevents high-roughness cells from receiving more weight
    # merely because their graph coordinate scale is larger.
    trace_scale = float(np.sum(present) / trace_a0)
    a0 = np.zeros((len(FAMILY_ORDER), len(FAMILY_ORDER)), dtype=float)
    c0 = np.zeros(len(FAMILY_ORDER), dtype=float)
    indices = np.flatnonzero(present)
    a0[np.ix_(indices, indices)] = trace_scale * local_a0
    c0[indices] = trace_scale * local_c0
    return GraphRoughnessMoment(
        cell_id=cell.cell_id,
        residual_source=residual_source,
        a0=a0,
        c0=c0,
        present_mask=present,
        trace_a0=trace_a0,
        diagnostics={
            "n_rows": n,
            "n_present_families": int(np.sum(present)),
            "graph_k": int(min(k, n - 1)),
            "graph_nnz": int(graph.nnz),
            "trace_a0": trace_a0,
            "trace_scale": trace_scale,
            "scaled_trace_a0": float(np.trace(a0)),
            "cross_gradient_norm": float(np.linalg.norm(local_c0)),
            "scaled_cross_gradient_norm": float(np.linalg.norm(c0)),
            "uses_labels": False,
        },
    )


def pooled_graph_roughness_direction(
    moments: Sequence[GraphRoughnessMoment],
    dataset_families: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Pool PGRD moments equally across donor dataset families.

    Passing no dataset-family vector gives an equal-cell pool.  Missing family
    coordinates are embedded as zeros before averaging, exactly as in the
    historical pooled-PGRD mechanism; there is no pairwise-availability
    reweighting.
    """

    records = tuple(moments)
    if not records:
        raise ValueError("at least one graph moment is required")
    if len({record.residual_source for record in records}) != 1:
        raise ValueError("cannot pool different residual sources")
    if dataset_families is None:
        group_labels = ["__all__"] * len(records)
    else:
        labels = [str(value) for value in dataset_families]
        if len(labels) != len(records):
            raise ValueError("dataset families do not align to moments")
        group_labels = labels
    unique_groups = sorted(set(group_labels))
    group_a = []
    group_c = []
    group_presence = []
    for group in unique_groups:
        selected = [
            record
            for record, label in zip(records, group_labels)
            if label == group
        ]
        group_a.append(np.mean([record.a0 for record in selected], axis=0))
        group_c.append(np.mean([record.c0 for record in selected], axis=0))
        group_presence.append(
            np.sum([record.present_mask for record in selected], axis=0)
        )
    pooled_a = np.mean(group_a, axis=0)
    pooled_c = np.mean(group_c, axis=0)
    pooled_a = 0.5 * (pooled_a + pooled_a.T)
    trace = float(np.trace(pooled_a))
    if not np.isfinite(trace) or trace <= EPS:
        raise ValueError("pooled PGRD moment has nonpositive trace")
    direction = -pooled_c
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("pooled PGRD cross-gradient is negligible")
    return direction, {
        "n_moments": len(records),
        "n_pool_groups": len(unique_groups),
        "residual_source": records[0].residual_source,
        "trace_a0": trace,
        "cross_gradient_norm": float(np.linalg.norm(pooled_c)),
        "direction_norm": norm,
        "family_coverage_by_pool_group": [
            np.asarray(value, dtype=int).tolist() for value in group_presence
        ],
        "uses_labels": False,
    }


def score_graph_roughness_direction(
    cell: ResidualCell,
    direction: np.ndarray,
    *,
    residual_source: str = "baseline",
    trust: float | None = None,
    trust_factor: float | None = None,
    gate_strength: float = 0.0,
    temperature: float = 1.0,
) -> ResidualMoEScore:
    """Apply a PGRD direction as gated residual family experts over B3."""

    present = np.asarray(cell.present_mask, dtype=bool)
    count = int(np.sum(present))
    if trust is not None and trust_factor is not None:
        raise ValueError("specify trust or trust_factor, not both")
    if trust_factor is not None:
        trust = float(trust_factor) / count
    else:
        trust = 1.0 / count if trust is None else float(trust)
    if not 0.0 <= float(gate_strength) <= 1.0 or trust < 0.0:
        raise ValueError("invalid PGRD trust/gate strength")
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (len(FAMILY_ORDER),) or not np.isfinite(direction).all():
        raise ValueError("PGRD direction has the wrong shape")
    source = cell.residuals if residual_source == "baseline" else cell.loo_residuals
    if residual_source not in {"baseline", "loo"}:
        raise ValueError("residual_source must be baseline or loo")
    expert_terms = source * direction[None, :]
    probabilities = np.zeros_like(expert_terms)
    gates = np.zeros_like(expert_terms)
    if trust == 0.0:
        probabilities[:, present] = 1.0 / count
        gates[:, present] = 1.0
        return ResidualMoEScore(
            score=cell.baseline_score,
            logit=cell.baseline_logit,
            correction_z=np.zeros(len(cell.baseline_score)),
            raw_correction=np.zeros(len(cell.baseline_score)),
            expert_terms=expert_terms,
            gate_probabilities=probabilities,
            gates=gates,
            diagnostics={"exact_b3_alias": True, "trust": 0.0},
        )
    novelty = np.abs(cell.loo_residuals[:, present])
    novelty /= np.maximum(np.median(novelty, axis=1, keepdims=True), EPS)
    stability_scale = np.maximum(
        np.median(cell.loo_seed_instability[:, present], axis=1, keepdims=True), EPS
    )
    stability = 1.0 / (
        1.0 + cell.loo_seed_instability[:, present] / stability_scale
    )
    logits = (np.log1p(novelty) + np.log(np.clip(stability, EPS, None))) / float(
        temperature
    )
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities[:, present] = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    gates[:, present] = (
        (1.0 - gate_strength)
        + float(gate_strength) * count * probabilities[:, present]
    )
    raw = np.sum(gates * expert_terms, axis=1)
    raw_scale = float(np.std(raw))
    correction_z = (
        np.zeros(len(raw), dtype=float)
        if raw_scale <= EPS
        else trust * raw / raw_scale
    )
    updated_logit = cell.baseline_logit + cell.baseline_scale * correction_z
    return ResidualMoEScore(
        score=expit(updated_logit),
        logit=updated_logit,
        correction_z=correction_z,
        raw_correction=raw,
        expert_terms=expert_terms,
        gate_probabilities=probabilities,
        gates=gates,
        diagnostics={
            "exact_b3_alias": False,
            "residual_source": residual_source,
            "trust": trust,
            "trust_factor": (
                None if trust_factor is None else float(trust_factor)
            ),
            "gate_strength": float(gate_strength),
            "n_present_families": count,
            "gate_sum_max_abs_error": float(np.max(np.abs(gates.sum(axis=1) - count))),
            "gate_min_present": float(np.min(gates[:, present])),
            "gate_max_present": float(np.max(gates[:, present])),
            "raw_correction_scale": raw_scale,
            "correction_scale": float(np.std(correction_z)),
            "uses_labels": False,
        },
    )


__all__ = [
    "EPS",
    "FAMILY_ORDER",
    "FrozenB3Ensemble",
    "GraphRoughnessMoment",
    "ResidualCalibration",
    "ResidualCell",
    "ResidualMoEScore",
    "build_residual_cell",
    "fit_residual_calibration",
    "graph_roughness_moment",
    "load_frozen_b3_ensemble",
    "pooled_graph_roughness_direction",
    "score_graph_roughness_direction",
    "score_residual_moe",
]
