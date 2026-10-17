"""Frozen target-free token-local fusion ladder.

Every method in this module consumes one :class:`TokenFusionPreparation`.
That object reproduces the incumbent localization token cap, imputation,
nondegenerate-coordinate mask, and population standardization exactly.  The
module accepts no correctness target, first-error position, comparator score,
or error-family field.

The registered Phase-1 roster is deliberately small:

``LOCAL_EQUAL29``
    Uniform mean of all kept standardized confidence coordinates.
``LOCAL_EQUAL_FAMILY``
    Equal mass across the five non-structural provenance families.  Constant
    response trace length remains in the shared preparation but has zero local
    expert weight; the fixed response head already carries it as context.
``LOCAL_IU29``
    Exact numerical alias of the incumbent two-component token IU-PCR.
``LOCAL_SU29``
    Existing fixed SU-PCR reproduction.
``LOCAL_STG_SU29``
    Group-held-out stochastic gates select a stable sparse covariance-error
    support, followed by the unchanged SU-PCR head on that fixed support.
``LOCAL_DUFS_LIU29``
    Parameter-free DUFS soft gates define a sample graph, followed by the
    historical local lambda=0.3 Laplacian-IU solver.

All score orientation is label-free and uses the equal-mean confidence anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import ndtr

from .adapted_dufs import adapted_dufs_soft_gates
from .dependency_fusion import projected_sparse_decomposition, sparse_upcr_fit
from .fixed_application_pipelines import SHARED_GLOBAL_FEATURES, SHARED_TOKEN_VIEWS
from .laplacian_upcr import (
    build_graph_from_features,
    laplacian_iu_fit,
    permute_graph,
)
from .reconstruction_benchmark.localization_contract import FIT_TOKEN_CAP, payload_sha256
from .specrage_views import FEATURE_TO_VIEW, VIEW_ORDER
from .upcr import upcr_fit


LOCAL_EQUAL29 = "LOCAL_EQUAL29"
LOCAL_EQUAL_FAMILY = "LOCAL_EQUAL_FAMILY"
LOCAL_IU29 = "LOCAL_IU29"
LOCAL_SU29 = "LOCAL_SU29"
LOCAL_STG_SU29 = "LOCAL_STG_SU29"
LOCAL_DUFS_LIU29 = "LOCAL_DUFS_LIU29"

PRIMARY_METHOD_IDS = (
    LOCAL_EQUAL29,
    LOCAL_EQUAL_FAMILY,
    LOCAL_IU29,
    LOCAL_SU29,
    LOCAL_STG_SU29,
    LOCAL_DUFS_LIU29,
)
CONTROL_METHOD_IDS = (
    "CONTROL_STG_SU29_PERMUTED_SUPPORT",
    "CONTROL_DUFS_LIU29_PERMUTED_GRAPH",
)

IU_CONFIG = MappingProxyType({
    "loss": "l2",
    "exclusion": False,
    "difficulty_gate": False,
    "simple_avg_fallback": False,
    "recompute_after_exclusion": False,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "n_components": 2,
    "auto_components": False,
})
SU_CONFIG = MappingProxyType({
    "scale_ratio": 0.25,
    "rank": 2,
    "n_components": 2,
    "g2_projection_components": 1,
    "g2_grid": 300,
    "threshold_multiplier": 1.0,
    "max_iter": 100,
    "inner_completion_iter": 40,
    "decomposition_tol": 1e-8,
    "max_sparse_fraction": None,
    "target_condition": 100.0,
})

DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_GRAPH_K = 7
DUFS_LAMBDA = 0.3

STG_FOLDS = 5
STG_SEEDS = (11, 23, 37)
STG_EPOCHS = 120
STG_SIGMA = 0.5
STG_LR = 0.05
STG_PENALTIES = (0.10, 1.0, 3.0, 4.0, 5.0)
STG_PROBABILITY_THRESHOLD = 0.75
STG_MIN_FOLD_FRACTION = 0.60
STG_PERMUTATION_SEED = 2026082801
DUFS_GRAPH_PERMUTATION_SEED = 2026082802
APPLICATION_CHUNK = 100_000
_EPS = 1e-12


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value)
    output.setflags(write=False)
    return output


def _row_folds(row_ids: Sequence[str], n_folds: int = STG_FOLDS) -> np.ndarray:
    if int(n_folds) != n_folds or int(n_folds) < 2:
        raise ValueError("n_folds must be an integer >= 2")
    return np.asarray([
        int(hashlib.sha256(str(row_id).encode("utf-8")).hexdigest()[:16], 16)
        % int(n_folds)
        for row_id in row_ids
    ], dtype=np.int64)


def _family_names() -> tuple[str, ...]:
    names = []
    for global_name in SHARED_GLOBAL_FEATURES:
        if global_name not in FEATURE_TO_VIEW:
            raise RuntimeError(f"unregistered token provenance feature: {global_name}")
        names.append(FEATURE_TO_VIEW[global_name])
    return tuple(names)


TOKEN_FAMILY_NAMES = _family_names()
NONSTRUCTURAL_FAMILIES = tuple(name for name in VIEW_ORDER if name != "structural")


@dataclass(frozen=True)
class TokenFusionPreparation:
    """One exact, immutable preprocessing result shared by every local arm."""

    values: np.ndarray
    token_offsets: np.ndarray
    row_ids: tuple[str, ...]
    fit_indices: np.ndarray
    fit_row_indices: np.ndarray
    row_folds: np.ndarray
    medians: np.ndarray
    keep: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    standardized_fit: np.ndarray
    stream_names: tuple[str, ...]
    kept_stream_names: tuple[str, ...]
    kept_family_names: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        offsets = np.asarray(self.token_offsets, dtype=np.int64)
        fit_indices = np.asarray(self.fit_indices, dtype=np.int64)
        fit_rows = np.asarray(self.fit_row_indices, dtype=np.int64)
        folds = np.asarray(self.row_folds, dtype=np.int64)
        keep = np.asarray(self.keep, dtype=bool)
        standardized = np.asarray(self.standardized_fit, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.stream_names):
            raise ValueError("token values and stream roster disagree")
        if offsets.shape != (len(self.row_ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(values):
            raise ValueError("token offsets and row roster disagree")
        if fit_indices.ndim != 1 or fit_rows.shape != fit_indices.shape:
            raise ValueError("fit token indices/owners are malformed")
        if folds.shape != (len(self.row_ids),):
            raise ValueError("row folds and row roster disagree")
        if keep.shape != (values.shape[1],) or int(keep.sum()) < 3:
            raise ValueError("fewer than three kept token streams")
        if standardized.shape != (len(fit_indices), int(keep.sum())):
            raise ValueError("standardized fit matrix is malformed")
        if not np.isfinite(standardized).all():
            raise ValueError("standardized fit matrix contains non-finite values")
        for array in (
            values, offsets, fit_indices, fit_rows, folds, self.medians, keep,
            self.mean, self.std, standardized,
        ):
            np.asarray(array).setflags(write=False)

    @property
    def n_features(self) -> int:
        return int(self.keep.sum())

    @property
    def F(self) -> np.ndarray:
        """Return the frozen feature-by-fit-token matrix used by all solvers."""

        return self.standardized_fit.T

    def standardized_slice(self, lo: int, hi: int) -> np.ndarray:
        selected = np.asarray(self.values[int(lo):int(hi), self.keep], dtype=np.float64)
        # This intentionally mirrors the incumbent application-time fallback.
        clean = np.where(np.isfinite(selected), selected, self.mean[None, :])
        return (clean - self.mean[None, :]) / self.std[None, :]

    def token_risk(self, weights: Sequence[float]) -> np.ndarray:
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != (self.n_features,) or not np.isfinite(weights_array).all():
            raise ValueError("token fusion weight vector is malformed")
        output = np.empty(len(self.values), dtype=np.float64)
        for lo in range(0, len(output), APPLICATION_CHUNK):
            hi = min(lo + APPLICATION_CHUNK, len(output))
            output[lo:hi] = -(self.standardized_slice(lo, hi) @ weights_array)
        if not np.isfinite(output).all():
            raise RuntimeError("token fusion produced non-finite risk")
        return output


@dataclass(frozen=True)
class TokenFusionResult:
    method_id: str
    weights: np.ndarray
    token_risk: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        risk = np.asarray(self.token_risk, dtype=np.float64)
        if weights.ndim != 1 or risk.ndim != 1:
            raise ValueError("token fusion result arrays must be one-dimensional")
        if not np.isfinite(weights).all() or not np.isfinite(risk).all():
            raise ValueError("token fusion result contains non-finite values")
        weights.setflags(write=False)
        risk.setflags(write=False)


@dataclass(frozen=True)
class STGSupportResult:
    support: np.ndarray
    pair_probability: np.ndarray
    fold_selection_fraction: np.ndarray
    selected_penalty: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=bool)
        probability = np.asarray(self.pair_probability, dtype=np.float64)
        stability = np.asarray(self.fold_selection_fraction, dtype=np.float64)
        if support.ndim != 2 or support.shape[0] != support.shape[1]:
            raise ValueError("STG support must be square")
        if not np.array_equal(support, support.T) or np.any(np.diag(support)):
            raise ValueError("STG support must be symmetric with zero diagonal")
        if probability.shape != support.shape or stability.shape != support.shape:
            raise ValueError("STG support diagnostics have the wrong shape")
        support.setflags(write=False)
        probability.setflags(write=False)
        stability.setflags(write=False)


def prepare_token_fusion(
    token_confidence: np.ndarray,
    token_offsets: Sequence[int],
    row_ids: Sequence[str],
    *,
    fit_token_cap: int = FIT_TOKEN_CAP,
) -> TokenFusionPreparation:
    """Reproduce the incumbent token-IU preparation once for every arm."""

    values = np.asarray(token_confidence, dtype=np.float64)
    offsets = np.asarray(token_offsets, dtype=np.int64)
    rows = tuple(map(str, row_ids))
    if values.ndim != 2 or values.shape[1] != len(SHARED_TOKEN_VIEWS) or not len(values):
        raise ValueError("token confidence must be a nonempty tokens-by-29 matrix")
    if offsets.shape != (len(rows) + 1,) or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("token offsets do not bind the supplied row IDs")
    if np.any(np.diff(offsets) <= 0) or len(set(rows)) != len(rows):
        raise ValueError("token rows must be nonempty and uniquely identified")
    if int(fit_token_cap) != fit_token_cap or int(fit_token_cap) < 3:
        raise ValueError("fit_token_cap must be an integer >= 3")
    if len(values) > int(fit_token_cap):
        fit_indices = np.linspace(
            0, len(values) - 1, int(fit_token_cap), dtype=np.int64
        )
    else:
        fit_indices = np.arange(len(values), dtype=np.int64)
    fit = values[fit_indices]
    medians = np.median(fit, axis=0)
    clean_fit = np.where(np.isfinite(fit), fit, medians[None, :])
    scale = clean_fit.std(axis=0)
    keep = np.isfinite(medians) & np.isfinite(scale) & (scale > 1e-8)
    if int(keep.sum()) < 3:
        raise RuntimeError("token fusion has fewer than three nondegenerate streams")
    mean = clean_fit[:, keep].mean(axis=0)
    std = clean_fit[:, keep].std(axis=0)
    standardized = (clean_fit[:, keep] - mean[None, :]) / std[None, :]
    owners = np.searchsorted(offsets[1:], fit_indices, side="right").astype(np.int64)
    folds = _row_folds(rows)
    diagnostics = {
        "schema_version": "token-local-fusion-preparation-v1",
        "n_tokens": int(len(values)),
        "n_rows": int(len(rows)),
        "n_fit_tokens": int(len(fit_indices)),
        "fit_token_cap": int(fit_token_cap),
        "fit_index_sha256": payload_sha256(fit_indices.tolist()),
        "n_input_streams": int(values.shape[1]),
        "n_kept_streams": int(keep.sum()),
        "kept_stream_mask": keep.astype(int).tolist(),
        "stream_names": list(SHARED_TOKEN_VIEWS),
        "kept_stream_names": [name for name, use in zip(SHARED_TOKEN_VIEWS, keep) if use],
        "kept_family_names": [name for name, use in zip(TOKEN_FAMILY_NAMES, keep) if use],
        "row_fold_sha256": payload_sha256(folds.tolist()),
        "fit_row_owner_sha256": payload_sha256(owners.tolist()),
        "imputation": "incumbent exact: fit median; application mean fallback",
        "standardization": "population mean/std on sampled fit tokens",
        "labels_seen_during_fit": False,
    }
    diagnostics["preparation_sha256"] = payload_sha256(diagnostics)
    return TokenFusionPreparation(
        values=_readonly(values),
        token_offsets=_readonly(offsets),
        row_ids=rows,
        fit_indices=_readonly(fit_indices),
        fit_row_indices=_readonly(owners),
        row_folds=_readonly(folds),
        medians=_readonly(medians),
        keep=_readonly(keep),
        mean=_readonly(mean),
        std=_readonly(std),
        standardized_fit=_readonly(standardized),
        stream_names=tuple(SHARED_TOKEN_VIEWS),
        kept_stream_names=tuple(name for name, use in zip(SHARED_TOKEN_VIEWS, keep) if use),
        kept_family_names=tuple(name for name, use in zip(TOKEN_FAMILY_NAMES, keep) if use),
        diagnostics=MappingProxyType(diagnostics),
    )


def prepare_localization_cell(cell: Any) -> TokenFusionPreparation:
    """Narrow adapter from the fit-safe localization cell contract."""

    return prepare_token_fusion(
        cell.token_confidence, cell.token_offsets, cell.row_ids,
        fit_token_cap=FIT_TOKEN_CAP,
    )


def _oriented_result(
    preparation: TokenFusionPreparation,
    method_id: str,
    weights: Sequence[float],
    diagnostics: Mapping[str, Any],
) -> TokenFusionResult:
    weights_array = np.asarray(weights, dtype=np.float64).copy()
    anchor = preparation.standardized_fit.mean(axis=1)
    raw_fit_score = preparation.standardized_fit @ weights_array
    correlation = float(np.corrcoef(raw_fit_score, anchor)[0, 1])
    flipped = bool(np.isfinite(correlation) and correlation < 0.0)
    if flipped:
        weights_array *= -1.0
    payload = {
        "schema_version": "token-local-fusion-method-v1",
        "method_id": method_id,
        "preparation_sha256": preparation.diagnostics["preparation_sha256"],
        "confidence_anchor_correlation": correlation,
        "orientation_flipped": flipped,
        "labels_seen_during_fit": False,
        **dict(diagnostics),
    }
    payload["fit_sha256"] = payload_sha256(payload)
    return TokenFusionResult(
        method_id=method_id,
        weights=_readonly(weights_array),
        token_risk=_readonly(preparation.token_risk(weights_array)),
        diagnostics=MappingProxyType(payload),
    )


def fit_local_equal29(preparation: TokenFusionPreparation) -> TokenFusionResult:
    weights = np.full(preparation.n_features, 1.0 / preparation.n_features)
    return _oriented_result(
        preparation, LOCAL_EQUAL29, weights,
        {"fusion": "uniform kept-coordinate mean"},
    )


def fit_local_equal_family(preparation: TokenFusionPreparation) -> TokenFusionResult:
    present = tuple(
        family for family in NONSTRUCTURAL_FAMILIES
        if family in preparation.kept_family_names
    )
    if len(present) < 2:
        raise RuntimeError("fewer than two non-structural token families remain")
    weights = np.zeros(preparation.n_features, dtype=np.float64)
    for family in present:
        indices = np.asarray([
            index for index, value in enumerate(preparation.kept_family_names)
            if value == family
        ], dtype=np.int64)
        weights[indices] = 1.0 / (len(present) * len(indices))
    structural = [
        name for name, family in zip(
            preparation.kept_stream_names, preparation.kept_family_names
        ) if family == "structural"
    ]
    return _oriented_result(
        preparation, LOCAL_EQUAL_FAMILY, weights,
        {
            "fusion": "equal non-structural provenance-family mean",
            "present_families": list(present),
            "context_streams_zero_local_weight": structural,
            "family_weight_sum": float(weights.sum()),
        },
    )


def fit_local_iu29(preparation: TokenFusionPreparation) -> TokenFusionResult:
    fitted = upcr_fit(preparation.F, **dict(IU_CONFIG))
    return _oriented_result(
        preparation, LOCAL_IU29, fitted.w,
        {
            "fusion": "incumbent two-component iu_pcr",
            "components": 2,
            "scale_ratio": 0.25,
            "g2_hat": float(fitted.g2_hat),
            "projection_residual": float(fitted.proj_residual),
        },
    )


def _su_diagnostics(fitted: Any, *, support_source: str) -> dict[str, Any]:
    decomposition = fitted.decomposition
    return {
        "fusion": "su_pcr_reproduction",
        "support_source": support_source,
        "g2_hat": float(fitted.g2_hat),
        "projection_residual": float(fitted.projection_residual),
        "decomposition_converged": bool(decomposition.converged),
        "decomposition_iterations": int(decomposition.n_iter),
        "sparse_fraction": float(decomposition.sparse_fraction),
        "sparse_pairs": int(decomposition.meta["nnz_pairs"]),
        "theorem_support_ok": bool(decomposition.theorem_support_ok),
        "decomposition_relative_residual": float(decomposition.relative_residual),
    }


def fit_local_su29(preparation: TokenFusionPreparation) -> TokenFusionResult:
    fitted = sparse_upcr_fit(preparation.F, **dict(SU_CONFIG))
    return _oriented_result(
        preparation, LOCAL_SU29, fitted.w_pcr,
        _su_diagnostics(fitted, support_source="fixed Tenzer threshold reproduction"),
    )


def _covariance(samples_by_feature: np.ndarray) -> np.ndarray:
    F = np.asarray(samples_by_feature, dtype=np.float64)
    return 0.5 * ((F @ F.T) / F.shape[1] + ((F @ F.T) / F.shape[1]).T)


def _stg_probabilities(
    residual: np.ndarray,
    low_rank: np.ndarray,
    held_covariance: np.ndarray,
    *,
    penalty: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Fit pairwise stochastic gates against one held response-group fold."""

    import torch

    torch.set_num_threads(1)
    iu = np.triu_indices(residual.shape[0], 1)
    residual_v = torch.tensor(residual[iu], dtype=torch.float64)
    low_v = torch.tensor(low_rank[iu], dtype=torch.float64)
    held_v = torch.tensor(held_covariance[iu], dtype=torch.float64)
    normalizer = torch.mean(torch.square(held_v)).clamp_min(_EPS)
    means = torch.zeros(len(residual_v), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([means], lr=STG_LR)
    generator = torch.Generator().manual_seed(int(seed))
    sqrt_two = np.sqrt(2.0)
    for _ in range(STG_EPOCHS):
        noise = torch.randn(len(means), dtype=torch.float64, generator=generator)
        gates = torch.clamp(means + STG_SIGMA * noise, 0.0, 1.0)
        prediction = low_v + gates * residual_v
        survival = 0.5 * (
            1.0 + torch.erf(means / (STG_SIGMA * sqrt_two))
        )
        loss = torch.mean(torch.square(held_v - prediction)) / normalizer
        loss = loss + float(penalty) * torch.mean(survival)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mu = means.detach().numpy()
    probability = ndtr(mu / STG_SIGMA)
    deterministic_prediction = (
        low_rank[iu] + probability * residual[iu]
    )
    validation_error = float(
        np.mean(np.square(held_covariance[iu] - deterministic_prediction))
        / max(float(np.mean(np.square(held_covariance[iu]))), _EPS)
    )
    return probability, validation_error


def learn_stg_sparse_support(
    preparation: TokenFusionPreparation,
    *,
    probability_threshold: float = STG_PROBABILITY_THRESHOLD,
    minimum_fold_fraction: float = STG_MIN_FOLD_FRACTION,
) -> STGSupportResult:
    """Select a stable sparse pair support using only held token covariance."""

    probability_threshold = float(probability_threshold)
    minimum_fold_fraction = float(minimum_fold_fraction)
    if not 0.0 < probability_threshold <= 1.0:
        raise ValueError("probability_threshold must lie in (0, 1]")
    if not 0.0 < minimum_fold_fraction <= 1.0:
        raise ValueError("minimum_fold_fraction must lie in (0, 1]")

    F = preparation.F
    token_folds = preparation.row_folds[preparation.fit_row_indices]
    if set(token_folds.tolist()) != set(range(STG_FOLDS)):
        raise RuntimeError("sampled fit tokens do not populate all STG folds")
    m = F.shape[0]
    iu = np.triu_indices(m, 1)
    by_penalty: dict[float, list[tuple[int, int, np.ndarray, float]]] = {
        penalty: [] for penalty in STG_PENALTIES
    }
    for held_fold in range(STG_FOLDS):
        donor = token_folds != held_fold
        held = ~donor
        if int(donor.sum()) < 3 or int(held.sum()) < 3:
            raise RuntimeError("an STG covariance fold has fewer than three tokens")
        donor_covariance = _covariance(F[:, donor])
        held_covariance = _covariance(F[:, held])
        base = projected_sparse_decomposition(donor_covariance, rank=2)
        residual = donor_covariance - base.low_rank
        np.fill_diagonal(residual, 0.0)
        for penalty in STG_PENALTIES:
            for seed in STG_SEEDS:
                probability, error = _stg_probabilities(
                    residual, base.low_rank, held_covariance,
                    penalty=penalty,
                    seed=seed + 1009 * held_fold,
                )
                by_penalty[penalty].append((held_fold, seed, probability, error))

    max_pairs = max(0, int(np.ceil((m - 1) / 2.0) - 1))
    penalty_rows = []
    penalty_consensus: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for penalty in STG_PENALTIES:
        current = by_penalty[penalty]
        errors = np.asarray([row[3] for row in current], dtype=float)
        stacked = np.vstack([row[2] for row in current])
        pair_probability = stacked.mean(axis=0)
        per_fold = np.vstack([
            np.vstack([row[2] for row in current if row[0] == fold]).mean(axis=0)
            for fold in range(STG_FOLDS)
        ])
        fold_fraction = np.mean(
            per_fold >= probability_threshold, axis=0
        )
        selected = (
            (pair_probability >= probability_threshold)
            & (fold_fraction >= minimum_fold_fraction)
        )
        penalty_consensus[penalty] = (
            pair_probability, fold_fraction, selected
        )
        penalty_rows.append({
            "penalty": float(penalty),
            "mean_held_covariance_error": float(errors.mean()),
            "standard_error": float(errors.std(ddof=1) / np.sqrt(len(errors))),
            "uncapped_consensus_pairs": int(selected.sum()),
            "sparse_theorem_feasible": bool(int(selected.sum()) <= max_pairs),
        })
    eligible = [row for row in penalty_rows if row["sparse_theorem_feasible"]]
    if not eligible:
        raise RuntimeError(
            "no frozen STG penalty yields an uncapped theorem-sparse support"
        )
    chosen = min(
        eligible,
        key=lambda row: (row["mean_held_covariance_error"], -row["penalty"]),
    )
    selected_penalty = float(chosen["penalty"])
    selected = by_penalty[selected_penalty]
    stacked = np.vstack([row[2] for row in selected])
    pair_probability_v, fold_fraction_v, selected_v = (
        value.copy() for value in penalty_consensus[selected_penalty]
    )
    if int(selected_v.sum()) > max_pairs:
        raise AssertionError("selected STG support escaped its feasibility gate")

    support = np.zeros((m, m), dtype=bool)
    support[iu] = selected_v
    support |= support.T
    probability = np.zeros((m, m), dtype=np.float64)
    probability[iu] = pair_probability_v
    probability += probability.T
    fold_fraction = np.zeros((m, m), dtype=np.float64)
    fold_fraction[iu] = fold_fraction_v
    fold_fraction += fold_fraction.T
    selected_pair_indices = np.flatnonzero(selected_v)
    diagnostics = {
        "schema_version": "token-local-stg-support-v1",
        "folds": STG_FOLDS,
        "seeds": list(STG_SEEDS),
        "epochs": STG_EPOCHS,
        "sigma": STG_SIGMA,
        "learning_rate": STG_LR,
        "penalty_selection": (
            "minimum held covariance error among penalties whose uncapped "
            "consensus satisfies the SU sparse-support theorem"
        ),
        "penalty_roster": penalty_rows,
        "selected_penalty": selected_penalty,
        "probability_threshold": probability_threshold,
        "minimum_fold_fraction": minimum_fold_fraction,
        "maximum_support_pairs": max_pairs,
        "support_cap_applied": False,
        "selected_support_pairs": int(len(selected_pair_indices)),
        "selected_pair_flat_indices": selected_pair_indices.astype(int).tolist(),
        "mean_pair_probability": float(pair_probability_v.mean()),
        "mean_seed_fold_probability_sd": float(stacked.std(axis=0).mean()),
        "labels_seen_during_fit": False,
    }
    diagnostics["support_sha256"] = payload_sha256(diagnostics)
    return STGSupportResult(
        support=_readonly(support),
        pair_probability=_readonly(probability),
        fold_selection_fraction=_readonly(fold_fraction),
        selected_penalty=selected_penalty,
        diagnostics=MappingProxyType(diagnostics),
    )


def fit_local_stg_su29(
    preparation: TokenFusionPreparation,
    *,
    support_result: STGSupportResult | None = None,
    method_id: str = LOCAL_STG_SU29,
) -> TokenFusionResult:
    support_result = support_result or learn_stg_sparse_support(preparation)
    fitted = sparse_upcr_fit(
        preparation.F, **dict(SU_CONFIG), fixed_support=support_result.support
    )
    return _oriented_result(
        preparation, method_id, fitted.w_pcr,
        {
            **_su_diagnostics(fitted, support_source="group-held-out STG consensus"),
            "stg_support": dict(support_result.diagnostics),
        },
    )


def fit_local_dufs_liu29(
    preparation: TokenFusionPreparation,
    *,
    graph: Any | None = None,
    gates: np.ndarray | None = None,
    gate_diagnostics: Mapping[str, Any] | None = None,
    method_id: str = LOCAL_DUFS_LIU29,
) -> TokenFusionResult:
    if graph is None:
        if gates is None:
            gates, learned_diagnostics = adapted_dufs_soft_gates(
                preparation.F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )
            gate_diagnostics = learned_diagnostics
        graph = build_graph_from_features(
            preparation.F, gates=gates, k=DUFS_GRAPH_K
        )
    fitted = laplacian_iu_fit(
        preparation.F, lambda_=DUFS_LAMBDA, graph=graph
    )
    return _oriented_result(
        preparation, method_id, fitted.w,
        {
            "fusion": "dufs_soft_gate_laplacian_iu",
            "dufs_seeds": list(DUFS_SEEDS),
            "dufs_epochs": DUFS_EPOCHS,
            "graph_k": DUFS_GRAPH_K,
            "lambda": DUFS_LAMBDA,
            "gate_diagnostics": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in dict(gate_diagnostics or {}).items()
            },
            "laplacian_diagnostics": dict(fitted.diagnostics),
        },
    )


def fit_phase1_ladder(
    preparation: TokenFusionPreparation,
    *,
    include_controls: bool = True,
) -> dict[str, TokenFusionResult]:
    """Fit the frozen Phase-1 ladder and preregistered negative controls."""

    output: dict[str, TokenFusionResult] = {}
    output[LOCAL_EQUAL29] = fit_local_equal29(preparation)
    output[LOCAL_EQUAL_FAMILY] = fit_local_equal_family(preparation)
    output[LOCAL_IU29] = fit_local_iu29(preparation)
    output[LOCAL_SU29] = fit_local_su29(preparation)
    support = learn_stg_sparse_support(preparation)
    output[LOCAL_STG_SU29] = fit_local_stg_su29(
        preparation, support_result=support
    )

    gates, gate_diagnostics = adapted_dufs_soft_gates(
        preparation.F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    graph = build_graph_from_features(
        preparation.F, gates=gates, k=DUFS_GRAPH_K
    )
    output[LOCAL_DUFS_LIU29] = fit_local_dufs_liu29(
        preparation, graph=graph, gates=gates,
        gate_diagnostics=gate_diagnostics,
    )
    if include_controls:
        feature_permutation = np.random.default_rng(
            STG_PERMUTATION_SEED
        ).permutation(preparation.n_features)
        permuted_support = support.support[np.ix_(feature_permutation, feature_permutation)]
        permuted_support_result = STGSupportResult(
            support=_readonly(permuted_support),
            pair_probability=support.pair_probability[
                np.ix_(feature_permutation, feature_permutation)
            ],
            fold_selection_fraction=support.fold_selection_fraction[
                np.ix_(feature_permutation, feature_permutation)
            ],
            selected_penalty=support.selected_penalty,
            diagnostics=MappingProxyType({
                **dict(support.diagnostics),
                "control": "deterministic feature-label permutation of frozen support",
                "permutation_seed": STG_PERMUTATION_SEED,
            }),
        )
        output[CONTROL_METHOD_IDS[0]] = fit_local_stg_su29(
            preparation,
            support_result=permuted_support_result,
            method_id=CONTROL_METHOD_IDS[0],
        )
        node_permutation = np.random.default_rng(
            DUFS_GRAPH_PERMUTATION_SEED
        ).permutation(preparation.F.shape[1])
        output[CONTROL_METHOD_IDS[1]] = fit_local_dufs_liu29(
            preparation,
            graph=permute_graph(graph, node_permutation),
            gates=gates,
            gate_diagnostics={
                **gate_diagnostics,
                "control": "deterministic graph-node permutation",
                "permutation_seed": DUFS_GRAPH_PERMUTATION_SEED,
            },
            method_id=CONTROL_METHOD_IDS[1],
        )
    if tuple(output) != PRIMARY_METHOD_IDS + (CONTROL_METHOD_IDS if include_controls else ()):
        raise AssertionError("token-local Phase-1 method roster drifted")
    return output


def step_maxima(
    token_risk: Sequence[float],
    segment_starts: Sequence[int],
    segment_ends: Sequence[int],
) -> np.ndarray:
    risk = np.asarray(token_risk, dtype=np.float64)
    starts = np.asarray(segment_starts, dtype=np.int64)
    ends = np.asarray(segment_ends, dtype=np.int64)
    if starts.shape != ends.shape or np.any(starts < 0) or np.any(ends <= starts) or np.any(ends > len(risk)):
        raise ValueError("token-to-step spans are malformed")
    output = np.asarray([
        float(np.max(risk[int(lo):int(hi)])) for lo, hi in zip(starts, ends)
    ], dtype=np.float64)
    if not np.isfinite(output).all():
        raise RuntimeError("token-to-step maximum produced non-finite scores")
    return output


__all__ = [
    "CONTROL_METHOD_IDS", "DUFS_EPOCHS", "DUFS_GRAPH_K", "DUFS_LAMBDA",
    "DUFS_SEEDS", "LOCAL_DUFS_LIU29", "LOCAL_EQUAL29",
    "LOCAL_EQUAL_FAMILY", "LOCAL_IU29", "LOCAL_STG_SU29", "LOCAL_SU29",
    "NONSTRUCTURAL_FAMILIES", "PRIMARY_METHOD_IDS", "STGSupportResult",
    "TokenFusionPreparation", "TokenFusionResult", "fit_local_dufs_liu29",
    "fit_local_equal29", "fit_local_equal_family", "fit_local_iu29",
    "fit_local_stg_su29", "fit_local_su29", "fit_phase1_ladder",
    "learn_stg_sparse_support", "prepare_localization_cell",
    "prepare_token_fusion", "step_maxima",
]
