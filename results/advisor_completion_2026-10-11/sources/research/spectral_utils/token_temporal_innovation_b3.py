"""Target-free token-local B3 with causal temporal innovations.

This module implements the score-fitting side of
``TOKEN_LOCAL_TEMPORAL_INNOVATION_B3_V1``.  It deliberately has no argument
for correctness, first-error position, response score, or evaluation target.

The predictive graph is a *nuisance-support* graph.  Every optional edge is a
lagged predictor; contemporaneous and future values are never part of a
design matrix.  Sparsity therefore does not imply Granger causality or causal
sufficiency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import ndtr

from .feature_contract import confidence_sign_vector
from .fixed_application_pipelines import (
    MAX_FIT_TOKENS,
    SHARED_GLOBAL_FEATURES,
    SHARED_TOKEN_VIEWS,
)
from .residual_graph_deem import (
    ContinuousDeemConfig,
    ContinuousDeemResult,
    EPS,
    _FamilyAdditiveEnergy,
    equal_family_risk_anchor,
    fit_continuous_deem,
    persistent_mala,
    predict_continuous_deem,
    set_determinism,
)
from .specrage_views import FEATURE_TO_VIEW, VIEW_ORDER
from .token_local_fusion import TokenFusionPreparation


LOCAL_TOKEN_B3 = "LOCAL_TOKEN_B3"
LOCAL_TOKEN_B3_SELF_INNOV = "LOCAL_TOKEN_B3_SELF_INNOV"
LOCAL_TOKEN_B3_ROOK_ALL_INNOV = "LOCAL_TOKEN_B3_ROOK_ALL_INNOV"
LOCAL_TOKEN_B3_ROOK_PSTG_INNOV = "LOCAL_TOKEN_B3_ROOK_PSTG_INNOV"
LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL = "LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL"

METHOD_IDS = (
    LOCAL_TOKEN_B3,
    LOCAL_TOKEN_B3_SELF_INNOV,
    LOCAL_TOKEN_B3_ROOK_ALL_INNOV,
    LOCAL_TOKEN_B3_ROOK_PSTG_INNOV,
    LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL,
)

CORE_TOKEN_VIEWS = (
    "entropy_series",
    "entropy_sw_var_series",
    "entropy_cusum_abs_series",
    "spilled_series",
    "spilled_sw_var_series",
    "spilled_cusum_abs_series",
    "energy_series",
    "energy_sw_var_series",
    "energy_cusum_abs_series",
)
CORE_SOURCES = ("entropy", "spilled", "energy")
CORE_OPERATORS = ("level", "sliding_variance", "absolute_cusum")
INNOVATION_GROUPS = MappingProxyType({
    operator: tuple(source_index * 3 + operator_index for source_index in range(3))
    for operator_index, operator in enumerate(CORE_OPERATORS)
})

RIDGE = 1.0
STG_SIGMA = 0.5
STG_INITIAL_MU = 0.5
STG_EPOCHS = 120
STG_LEARNING_RATE = 0.10
STG_PENALTIES = (0.01, 0.03, 0.10, 0.30, 1.00)
STG_PROBABILITY_THRESHOLD = 0.75
STG_BOOTSTRAP_REPLICATES = 20
STG_BOOTSTRAP_SEEDS = tuple(range(2026082800, 2026082820))
STG_BOOTSTRAP_MIN_COUNT = 15
OUTER_FOLDS = 5


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value)
    output.setflags(write=False)
    return output


def _payload_sha(value: Any) -> str:
    from .reconstruction_benchmark.localization_contract import payload_sha256

    return payload_sha256(value)


def _hash_fold(value: str, *, namespace: str, n_folds: int = 5) -> int:
    digest = hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(n_folds)


def _core_indices() -> tuple[int, ...]:
    lookup = {name: index for index, name in enumerate(SHARED_TOKEN_VIEWS)}
    missing = [name for name in CORE_TOKEN_VIEWS if name not in lookup]
    if missing:
        raise RuntimeError(f"core temporal stream roster drifted: {missing}")
    return tuple(lookup[name] for name in CORE_TOKEN_VIEWS)


CORE_INDICES = _core_indices()


def rook_peers(target: int) -> tuple[int, ...]:
    """Return the four same-source or same-operator peers in stable order."""

    target = int(target)
    source, operator = divmod(target, 3)
    peers = [index for index in range(9) if index != target and (
        index // 3 == source or index % 3 == operator
    )]
    if len(peers) != 4:
        raise AssertionError("3x3 rook graph must have four peers per node")
    return tuple(peers)


def nonrook_peers(target: int) -> tuple[int, ...]:
    rook = set(rook_peers(target))
    peers = tuple(index for index in range(9) if index != int(target) and index not in rook)
    if len(peers) != 4:
        raise AssertionError("3x3 non-rook control must have four peers per node")
    return peers


ROOK_PEERS = tuple(rook_peers(index) for index in range(9))
NONROOK_PEERS = tuple(nonrook_peers(index) for index in range(9))


@dataclass(frozen=True)
class FoldTokenData:
    risk: np.ndarray
    donor_fit_indices: np.ndarray
    donor_rows: np.ndarray
    held_rows: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        risk = np.asarray(self.risk, dtype=np.float64)
        fit = np.asarray(self.donor_fit_indices, dtype=np.int64)
        if risk.ndim != 2 or risk.shape[1] != 29 or not np.isfinite(risk).all():
            raise ValueError("fold risk matrix must be finite tokens-by-29")
        if fit.ndim != 1 or not len(fit):
            raise ValueError("fold donor cap is empty")
        for value in (risk, fit, self.donor_rows, self.held_rows, self.median, self.mean, self.scale):
            np.asarray(value).setflags(write=False)


@dataclass(frozen=True)
class ProjectedSTGResult:
    support: np.ndarray
    mean_survival_probability: np.ndarray
    bootstrap_pass_count: np.ndarray
    selected_penalty: float
    cv_records: tuple[Mapping[str, Any], ...]
    exact_subset_records: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=bool)
        probability = np.asarray(self.mean_survival_probability, dtype=np.float64)
        count = np.asarray(self.bootstrap_pass_count, dtype=np.int64)
        if support.shape != (9, 9) or probability.shape != support.shape or count.shape != support.shape:
            raise ValueError("Projected-STG support arrays must be 9x9")
        if np.any(np.diag(support)):
            raise ValueError("self-lag is mandatory and may not appear as an optional edge")
        for target in range(9):
            legal = set(ROOK_PEERS[target])
            if any(support[target, source] and source not in legal for source in range(9)):
                raise ValueError("Projected-STG selected a non-rook predictor")
        for value in (support, probability, count):
            np.asarray(value).setflags(write=False)


@dataclass(frozen=True)
class InnovationMap:
    method_id: str
    support: np.ndarray
    intercept: np.ndarray
    time_coefficient: np.ndarray
    self_coefficient: np.ndarray
    cross_coefficients: np.ndarray
    residual_scale: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=bool)
        cross = np.asarray(self.cross_coefficients, dtype=np.float64)
        if support.shape != (9, 9) or cross.shape != support.shape:
            raise ValueError("innovation support and coefficients must be 9x9")
        if np.any(np.diag(support)) or np.any(cross[~support] != 0.0):
            raise ValueError("innovation support/coefficients disagree")
        for value in (
            support, self.intercept, self.time_coefficient, self.self_coefficient,
            cross, self.residual_scale,
        ):
            array = np.asarray(value)
            if not np.isfinite(array).all():
                raise ValueError("innovation map contains non-finite values")
            array.setflags(write=False)


@dataclass(frozen=True)
class TokenB3Result:
    """Nonlinear result record; intentionally has no linear weight vector."""

    method_id: str
    token_risk: np.ndarray
    per_seed_model_records: tuple[Mapping[str, Any], ...]
    innovation_maps: tuple[InnovationMap, ...]
    fold_diagnostics: tuple[Mapping[str, Any], ...]
    health: Mapping[str, Any]

    def __post_init__(self) -> None:
        risk = np.asarray(self.token_risk, dtype=np.float64)
        if self.method_id not in METHOD_IDS or risk.ndim != 1 or not np.isfinite(risk).all():
            raise ValueError("malformed nonlinear token-B3 result")
        risk.setflags(write=False)


@dataclass
class InnovationB3Fit:
    score: np.ndarray
    posterior: np.ndarray
    logit: np.ndarray
    original_contributions: np.ndarray
    innovation_contributions: np.ndarray
    original_family_contributions: dict[str, np.ndarray]
    innovation_family_contributions: dict[str, np.ndarray]
    aligned_bias: float
    orientation: int
    state: dict[str, np.ndarray]
    objective_history: list[dict[str, float]]
    health: dict[str, Any]
    config: dict[str, Any]
    seed: int
    innovation_gain: float


def _row_token_indices(offsets: np.ndarray, rows: Sequence[int]) -> np.ndarray:
    pieces = [np.arange(int(offsets[row]), int(offsets[row + 1]), dtype=np.int64) for row in rows]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def deterministic_donor_cap(
    offsets: Sequence[int], row_ids: Sequence[str], donor_rows: Sequence[int],
    *, max_tokens: int = MAX_FIT_TOKENS,
) -> np.ndarray:
    """Select up to 60k donor tokens, invariant to input row ordering."""

    offsets_array = np.asarray(offsets, dtype=np.int64)
    rows = sorted((int(row) for row in donor_rows), key=lambda row: str(row_ids[row]))
    indices = _row_token_indices(offsets_array, rows)
    if len(indices) > int(max_tokens):
        positions = np.linspace(0, len(indices) - 1, int(max_tokens), dtype=np.int64)
        indices = indices[positions]
    return indices


def prepare_fold_data(preparation: TokenFusionPreparation, fold: int) -> FoldTokenData:
    fold = int(fold)
    held_rows = np.flatnonzero(np.asarray(preparation.row_folds) == fold)
    donor_rows = np.flatnonzero(np.asarray(preparation.row_folds) != fold)
    if not len(held_rows) or not len(donor_rows):
        raise RuntimeError(f"outer fold {fold} has an empty donor or held population")
    fit_indices = deterministic_donor_cap(
        preparation.token_offsets, preparation.row_ids, donor_rows,
    )
    raw = np.asarray(preparation.values, dtype=np.float64)
    fit_raw = raw[fit_indices]
    median = np.nanmedian(fit_raw, axis=0)
    if not np.isfinite(median).all():
        raise RuntimeError("donor cap contains an all-nonfinite coordinate")
    clean_fit = np.where(np.isfinite(fit_raw), fit_raw, median[None, :])
    mean = clean_fit.mean(axis=0)
    scale = clean_fit.std(axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-8):
        raise RuntimeError("all 29 original streams must be nondegenerate in every donor fold")
    clean = np.where(np.isfinite(raw), raw, median[None, :])
    signs = confidence_sign_vector(SHARED_GLOBAL_FEATURES).astype(np.float64)
    risk = -((clean - mean[None, :]) / scale[None, :]) * signs[None, :]
    diagnostics = {
        "fold": fold,
        "donor_rows": int(len(donor_rows)),
        "held_rows": int(len(held_rows)),
        "donor_fit_tokens": int(len(fit_indices)),
        "donor_fit_index_sha256": _payload_sha(fit_indices.tolist()),
        "preprocessing": "donor-only median, population mean/std, registered risk signs",
        "labels_seen_during_fit": False,
    }
    return FoldTokenData(
        risk=_readonly(risk), donor_fit_indices=_readonly(fit_indices),
        donor_rows=_readonly(donor_rows), held_rows=_readonly(held_rows),
        median=_readonly(median), mean=_readonly(mean), scale=_readonly(scale),
        diagnostics=MappingProxyType(diagnostics),
    )


def _positions_and_owners(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.empty(int(offsets[-1]), dtype=np.int64)
    owners = np.empty_like(positions)
    predecessor = np.full_like(positions, -1)
    for row, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        lo_i, hi_i = int(lo), int(hi)
        positions[lo_i:hi_i] = np.arange(hi_i - lo_i, dtype=np.int64)
        owners[lo_i:hi_i] = row
        if hi_i - lo_i > 1:
            predecessor[lo_i + 1:hi_i] = np.arange(lo_i, hi_i - 1, dtype=np.int64)
    return positions, owners, predecessor


def _weighted_ridge(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    X = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    gram = X.T @ (w[:, None] * X)
    penalty = np.eye(X.shape[1], dtype=np.float64) * RIDGE
    penalty[0, 0] = 0.0
    rhs = X.T @ (w * y)
    try:
        return np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram + penalty, rcond=1e-12) @ rhs


def _question_weights(owners: np.ndarray) -> np.ndarray:
    owners = np.asarray(owners, dtype=np.int64)
    unique, counts = np.unique(owners, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    return np.asarray([1.0 / (len(unique) * count_map[int(owner)]) for owner in owners], dtype=np.float64)


def _row_mse(residual: np.ndarray, owners: np.ndarray) -> dict[int, float]:
    return {
        int(owner): float(np.mean(np.square(residual[owners == owner])))
        for owner in np.unique(owners)
    }


def _design_for_target(
    core: np.ndarray, positions: np.ndarray, predecessor: np.ndarray,
    indices: np.ndarray, target: int, peers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    prior = predecessor[indices]
    if np.any(prior < 0):
        raise ValueError("lag design includes a first token")
    base = np.column_stack([
        np.ones(len(indices), dtype=np.float64),
        np.log1p(positions[indices].astype(np.float64)),
        core[prior, int(target)],
    ])
    optional = core[prior][:, list(map(int, peers))] if peers else np.empty((len(indices), 0))
    return base, optional


def _fit_projected_stg(
    base: np.ndarray, optional: np.ndarray, target: np.ndarray, owners: np.ndarray,
    *, penalty: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projected Gaussian STG for a four-edge linear nuisance model."""

    if optional.shape[1] != 4:
        raise ValueError("Projected-STG v1 is frozen to four optional predictors")
    rng = np.random.default_rng(int(seed))
    weights = _question_weights(owners)
    # The gate updates run for 120 epochs and are repeated for every
    # target/fold/penalty.  Cache weighted sufficient statistics so each epoch
    # solves only a 7x7 system instead of rescanning all donor tokens.
    weighted_base = base * weights[:, None]
    weighted_optional = optional * weights[:, None]
    gram_bb = base.T @ weighted_base
    gram_bo = base.T @ weighted_optional
    gram_oo = optional.T @ weighted_optional
    rhs_b = base.T @ (weights * target)
    rhs_o = optional.T @ (weights * target)

    def gated_ridge(gate: np.ndarray) -> np.ndarray:
        design_gram = np.zeros((7, 7), dtype=np.float64)
        design_gram[:3, :3] = gram_bb
        design_gram[:3, 3:] = gram_bo * gate[None, :]
        design_gram[3:, :3] = design_gram[:3, 3:].T
        design_gram[3:, 3:] = gram_oo * gate[:, None] * gate[None, :]
        penalty_matrix = np.eye(7, dtype=np.float64) * RIDGE
        penalty_matrix[0, 0] = 0.0
        rhs = np.concatenate((rhs_b, rhs_o * gate))
        try:
            return np.linalg.solve(design_gram + penalty_matrix, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(design_gram + penalty_matrix, rcond=1e-12) @ rhs

    mu = np.full(4, STG_INITIAL_MU, dtype=np.float64)
    beta = np.zeros(base.shape[1] + 4, dtype=np.float64)
    for _epoch in range(STG_EPOCHS):
        epsilon = rng.normal(size=4)
        raw_gate = mu + STG_SIGMA * epsilon
        gate = np.clip(raw_gate, 0.0, 1.0)
        beta = gated_ridge(gate)
        weighted_optional_residual = (
            rhs_o - gram_bo.T @ beta[:3] - gram_oo @ (gate * beta[3:])
        )
        active_derivative = ((raw_gate > 0.0) & (raw_gate < 1.0)).astype(np.float64)
        data_gradient = np.asarray([
            -2.0 * weighted_optional_residual[index] * beta[3 + index]
            for index in range(4)
        ]) * active_derivative
        density = np.exp(-0.5 * np.square(mu / STG_SIGMA)) / math.sqrt(2.0 * math.pi)
        # The data term is a question-normalized mean while the L0 surrogate
        # is the expected *number* of open edges.  This is the normalized STG
        # convention used by the registered lambda grid; dividing again by
        # four makes lambda a no-op at this scale and leaves the null dense.
        sparse_gradient = float(penalty) * density / STG_SIGMA
        mu = np.clip(mu - STG_LEARNING_RATE * (data_gradient + sparse_gradient), -4.0, 4.0)
    probability = ndtr(mu / STG_SIGMA)
    # Deployment and cross-validation use the same registered survival
    # threshold; a latent gate above one half is not yet a stable edge.
    support = probability >= STG_PROBABILITY_THRESHOLD
    hard_design = np.column_stack([base, optional[:, support]])
    hard_beta = _weighted_ridge(hard_design, target, weights)
    return probability, support, hard_beta


def _fit_subset(
    base: np.ndarray, optional: np.ndarray, target: np.ndarray, owners: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    return _weighted_ridge(
        np.column_stack([base, optional[:, mask]]), target, _question_weights(owners)
    )


def _cv_split_records(row_ids: Sequence[str], donor_rows: np.ndarray) -> np.ndarray:
    folds = np.full(len(row_ids), -1, dtype=np.int64)
    for row in donor_rows:
        folds[int(row)] = _hash_fold(str(row_ids[int(row)]), namespace="temporal-pstg-inner")
    present = set(folds[donor_rows].tolist())
    if present != set(range(5)):
        raise RuntimeError(f"Projected-STG donor population lacks an inner fold: {sorted(present)}")
    return folds


def select_projected_stg_support(
    core: np.ndarray, offsets: Sequence[int], row_ids: Sequence[str],
    donor_rows: Sequence[int], donor_fit_indices: Sequence[int],
) -> ProjectedSTGResult:
    """Select and stability-audit the 36 directed lagged rook edges."""

    core = np.asarray(core, dtype=np.float64)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    donor_rows_array = np.asarray(donor_rows, dtype=np.int64)
    fit_indices = np.asarray(donor_fit_indices, dtype=np.int64)
    positions, owners_all, predecessor = _positions_and_owners(offsets_array)
    donor_set = set(donor_rows_array.tolist())
    indices = np.asarray([
        int(index) for index in fit_indices
        if predecessor[int(index)] >= 0 and int(owners_all[int(index)]) in donor_set
    ], dtype=np.int64)
    if len(indices) < 100:
        raise RuntimeError("too few donor lag pairs for Projected-STG")
    owners = owners_all[indices]
    inner_folds = _cv_split_records(row_ids, donor_rows_array)
    cv_records: list[dict[str, Any]] = []
    lambda_errors: dict[float, list[float]] = {value: [] for value in STG_PENALTIES}
    lambda_edges: dict[float, list[int]] = {value: [] for value in STG_PENALTIES}
    for fold in range(5):
        train_mask = inner_folds[owners] != fold
        held_mask = inner_folds[owners] == fold
        if not np.any(train_mask) or not np.any(held_mask):
            raise RuntimeError("empty inner Projected-STG token fold")
        for penalty in STG_PENALTIES:
            held_row_errors: dict[int, list[float]] = {}
            edge_count = 0
            target_records = []
            for target_index in range(9):
                base, optional = _design_for_target(
                    core, positions, predecessor, indices, target_index, ROOK_PEERS[target_index]
                )
                probability, support, beta = _fit_projected_stg(
                    base[train_mask], optional[train_mask], core[indices[train_mask], target_index],
                    owners[train_mask], penalty=penalty,
                    seed=2026082800 + 1000 * fold + 100 * target_index + int(round(100 * penalty)),
                )
                held_design = np.column_stack([base[held_mask], optional[held_mask][:, support]])
                residual = core[indices[held_mask], target_index] - held_design @ beta
                for row, value in _row_mse(residual, owners[held_mask]).items():
                    held_row_errors.setdefault(row, []).append(value)
                edge_count += int(support.sum())
                target_records.append({
                    "target": target_index,
                    "survival_probability": probability.tolist(),
                    "support_mask": support.astype(int).tolist(),
                })
            question_errors = [float(np.mean(values)) for values in held_row_errors.values()]
            lambda_errors[penalty].extend(question_errors)
            lambda_edges[penalty].append(edge_count)
            cv_records.append({
                "fold": fold,
                "penalty": penalty,
                "held_question_mse": float(np.mean(question_errors)),
                "held_question_se": float(np.std(question_errors, ddof=1) / math.sqrt(len(question_errors))) if len(question_errors) > 1 else 0.0,
                "selected_edges": edge_count,
                "targets": target_records,
            })
    summary = {}
    for penalty in STG_PENALTIES:
        errors = np.asarray(lambda_errors[penalty], dtype=np.float64)
        summary[penalty] = {
            "mse": float(errors.mean()),
            "se": float(errors.std(ddof=1) / math.sqrt(len(errors))) if len(errors) > 1 else 0.0,
            "mean_edges": float(np.mean(lambda_edges[penalty])),
        }
    minimum_penalty = min(STG_PENALTIES, key=lambda value: (summary[value]["mse"], value))
    threshold = summary[minimum_penalty]["mse"] + summary[minimum_penalty]["se"]
    eligible = [value for value in STG_PENALTIES if summary[value]["mse"] <= threshold]
    selected_penalty = min(
        eligible,
        key=lambda value: (summary[value]["mean_edges"], -value),
    )

    base_optional_by_target = [
        _design_for_target(core, positions, predecessor, indices, target, ROOK_PEERS[target])
        for target in range(9)
    ]
    full_probability = np.zeros((9, 4), dtype=np.float64)
    for target_index, (base, optional) in enumerate(base_optional_by_target):
        probability, _support, _beta = _fit_projected_stg(
            base, optional, core[indices, target_index], owners,
            penalty=selected_penalty, seed=2026082900 + target_index,
        )
        full_probability[target_index] = probability

    bootstrap_probability = np.zeros((STG_BOOTSTRAP_REPLICATES, 9, 4), dtype=np.float64)
    unique_rows = np.asarray(sorted(set(owners.tolist())), dtype=np.int64)
    for replicate, seed in enumerate(STG_BOOTSTRAP_SEEDS):
        rng = np.random.default_rng(seed)
        sampled_rows = rng.choice(unique_rows, size=len(unique_rows), replace=True)
        sampled_positions = np.concatenate([np.flatnonzero(owners == row) for row in sampled_rows])
        sampled_owners = np.concatenate([
            np.full(np.sum(owners == row), position, dtype=np.int64)
            for position, row in enumerate(sampled_rows)
        ])
        for target_index, (base, optional) in enumerate(base_optional_by_target):
            probability, _support, _beta = _fit_projected_stg(
                base[sampled_positions], optional[sampled_positions],
                core[indices[sampled_positions], target_index], sampled_owners,
                penalty=selected_penalty, seed=seed + 97 * target_index,
            )
            bootstrap_probability[replicate, target_index] = probability
    mean_probability_small = bootstrap_probability.mean(axis=0)
    pass_count_small = np.sum(
        bootstrap_probability >= STG_PROBABILITY_THRESHOLD, axis=0
    ).astype(np.int64)
    stable_small = (
        (mean_probability_small >= STG_PROBABILITY_THRESHOLD)
        & (pass_count_small >= STG_BOOTSTRAP_MIN_COUNT)
    )
    support = np.zeros((9, 9), dtype=bool)
    mean_probability = np.zeros((9, 9), dtype=np.float64)
    pass_count = np.zeros((9, 9), dtype=np.int64)
    for target in range(9):
        for peer_position, source in enumerate(ROOK_PEERS[target]):
            support[target, source] = stable_small[target, peer_position]
            mean_probability[target, source] = mean_probability_small[target, peer_position]
            pass_count[target, source] = pass_count_small[target, peer_position]

    # Exact four-peer audit: all 16 subsets for every target, on the same
    # held-question folds and ridge definition.
    exact_records: list[dict[str, Any]] = []
    exact_passes = []
    for target_index, (base, optional) in enumerate(base_optional_by_target):
        subset_question_errors: dict[int, list[float]] = {mask: [] for mask in range(16)}
        for fold in range(5):
            train_mask = inner_folds[owners] != fold
            held_mask = inner_folds[owners] == fold
            for mask_value in range(16):
                mask = np.asarray([(mask_value >> bit) & 1 for bit in range(4)], dtype=bool)
                beta = _fit_subset(
                    base[train_mask], optional[train_mask], core[indices[train_mask], target_index],
                    owners[train_mask], mask,
                )
                held_design = np.column_stack([base[held_mask], optional[held_mask][:, mask]])
                residual = core[indices[held_mask], target_index] - held_design @ beta
                subset_question_errors[mask_value].extend(_row_mse(residual, owners[held_mask]).values())
        subset_stats = {}
        for mask_value, values in subset_question_errors.items():
            array = np.asarray(values, dtype=np.float64)
            subset_stats[mask_value] = {
                "mse": float(array.mean()),
                "se": float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
                "edges": int(mask_value.bit_count()),
            }
        optimum = min(range(16), key=lambda value: (subset_stats[value]["mse"], subset_stats[value]["edges"], value))
        selected_mask = np.asarray([support[target_index, source] for source in ROOK_PEERS[target_index]], dtype=bool)
        selected_value = sum((1 << bit) for bit, enabled in enumerate(selected_mask) if enabled)
        within = subset_stats[selected_value]["mse"] <= (
            subset_stats[optimum]["mse"] + subset_stats[optimum]["se"]
        )
        capacity = subset_stats[selected_value]["edges"] <= subset_stats[optimum]["edges"] + 1
        passed = bool(within and capacity)
        exact_passes.append(passed)
        exact_records.append({
            "target": target_index,
            "optimal_subset": optimum,
            "optimal_mse": subset_stats[optimum]["mse"],
            "optimal_se": subset_stats[optimum]["se"],
            "optimal_edges": subset_stats[optimum]["edges"],
            "pstg_subset": selected_value,
            "pstg_mse": subset_stats[selected_value]["mse"],
            "pstg_edges": subset_stats[selected_value]["edges"],
            "within_one_se": bool(within),
            "at_most_one_extra_edge": bool(capacity),
            "passed": passed,
            "all_subsets": {str(key): value for key, value in subset_stats.items()},
        })
    diagnostics = {
        "schema_version": "projected-stg-temporal-support-v1",
        "gate_count": 36,
        "sigma": STG_SIGMA,
        "initial_mu": STG_INITIAL_MU,
        "epochs": STG_EPOCHS,
        "penalty_grid": list(STG_PENALTIES),
        "selected_penalty": selected_penalty,
        "one_se_minimum_penalty": minimum_penalty,
        "one_se_threshold": threshold,
        "cv_summary": {str(key): value for key, value in summary.items()},
        "bootstrap_seeds": list(STG_BOOTSTRAP_SEEDS),
        "bootstrap_unit": "whole_question",
        "survival_probability_threshold": STG_PROBABILITY_THRESHOLD,
        "bootstrap_min_count": STG_BOOTSTRAP_MIN_COUNT,
        "selected_edge_count": int(support.sum()),
        "exact_subset_audit_passed": bool(all(exact_passes)),
        "full_fit_probabilities": full_probability.tolist(),
        "labels_seen_during_fit": False,
    }
    if not all(exact_passes):
        failed_targets = [
            int(record["target"])
            for record in exact_records if not bool(record["passed"])
        ]
        raise RuntimeError(
            "Projected-STG failed the registered 16-subset one-SE/capacity audit "
            f"for targets {failed_targets}"
        )
    diagnostics["support_sha256"] = _payload_sha({
        "support": support.astype(int).tolist(),
        "mean_probability": mean_probability.tolist(),
        "pass_count": pass_count.tolist(),
        "selected_penalty": selected_penalty,
    })
    return ProjectedSTGResult(
        support=_readonly(support),
        mean_survival_probability=_readonly(mean_probability),
        bootstrap_pass_count=_readonly(pass_count),
        selected_penalty=float(selected_penalty),
        cv_records=tuple(MappingProxyType(record) for record in cv_records),
        exact_subset_records=tuple(MappingProxyType(record) for record in exact_records),
        diagnostics=MappingProxyType(diagnostics),
    )


def fixed_support(method_id: str) -> np.ndarray:
    support = np.zeros((9, 9), dtype=bool)
    if method_id == LOCAL_TOKEN_B3_SELF_INNOV:
        return support
    peers = ROOK_PEERS if method_id == LOCAL_TOKEN_B3_ROOK_ALL_INNOV else NONROOK_PEERS
    if method_id not in {
        LOCAL_TOKEN_B3_ROOK_ALL_INNOV,
        LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL,
    }:
        raise KeyError(method_id)
    for target in range(9):
        support[target, list(peers[target])] = True
    return support


def fit_innovation_map(
    method_id: str, core: np.ndarray, offsets: Sequence[int], donor_rows: Sequence[int],
    donor_fit_indices: Sequence[int], support: np.ndarray,
    *, support_diagnostics: Mapping[str, Any] | None = None,
) -> InnovationMap:
    core = np.asarray(core, dtype=np.float64)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    support_array = np.asarray(support, dtype=bool)
    if core.ndim != 2 or core.shape[1] != 9 or support_array.shape != (9, 9):
        raise ValueError("innovation-map inputs are malformed")
    if np.any(np.diag(support_array)):
        raise ValueError("self-lag is mandatory and is not an optional cross edge")
    if not np.isfinite(core).all():
        raise ValueError("innovation-map core must be finite")
    positions, owners, predecessor = _positions_and_owners(offsets_array)
    donor_set = set(map(int, donor_rows))
    fit_indices = np.asarray([
        int(index) for index in donor_fit_indices
        if predecessor[int(index)] >= 0 and int(owners[int(index)]) in donor_set
    ], dtype=np.int64)
    if not len(fit_indices):
        raise RuntimeError("innovation-map donor cap contains no valid lag pairs")
    weights = _question_weights(owners[fit_indices])
    intercept = np.zeros(9, dtype=np.float64)
    time_coefficient = np.zeros(9, dtype=np.float64)
    self_coefficient = np.zeros(9, dtype=np.float64)
    cross = np.zeros((9, 9), dtype=np.float64)
    residual_scale = np.zeros(9, dtype=np.float64)
    donor_mse = np.zeros(9, dtype=np.float64)
    for target in range(9):
        peers = tuple(np.flatnonzero(support_array[target]).tolist())
        base, optional = _design_for_target(
            core, positions, predecessor, fit_indices, target, peers
        )
        design = np.column_stack([base, optional])
        beta = _weighted_ridge(design, core[fit_indices, target], weights)
        prediction = design @ beta
        residual = core[fit_indices, target] - prediction
        scale = float(np.sqrt(np.mean(np.square(residual))))
        if not np.isfinite(scale) or scale <= 1e-8:
            raise RuntimeError(f"degenerate donor innovation residual scale for target {target}")
        intercept[target], time_coefficient[target], self_coefficient[target] = beta[:3]
        if peers:
            cross[target, list(peers)] = beta[3:]
        residual_scale[target] = scale
        donor_mse[target] = float(np.mean(np.square(residual)))
    diagnostics = {
        "schema_version": "token-temporal-innovation-map-v1",
        "method_id": method_id,
        "ridge": RIDGE,
        "time_predictor": "log1p(token_position)",
        "self_lag_mandatory": True,
        "contemporaneous_predictors": False,
        "future_predictors": False,
        "donor_lag_pairs": int(len(fit_indices)),
        "donor_question_count": int(len(donor_set)),
        "donor_residual_mse": donor_mse.tolist(),
        "selected_cross_edges": int(support_array.sum()),
        "optional_edge_roster": "four lagged peers per target",
        "labels_seen_during_fit": False,
        **dict(support_diagnostics or {}),
    }
    diagnostics["innovation_map_sha256"] = _payload_sha({
        "support": support_array.astype(int).tolist(),
        "intercept": intercept.tolist(),
        "time_coefficient": time_coefficient.tolist(),
        "self_coefficient": self_coefficient.tolist(),
        "cross_coefficients": cross.tolist(),
        "residual_scale": residual_scale.tolist(),
    })
    return InnovationMap(
        method_id=method_id, support=_readonly(support_array.copy()),
        intercept=_readonly(intercept), time_coefficient=_readonly(time_coefficient),
        self_coefficient=_readonly(self_coefficient), cross_coefficients=_readonly(cross),
        residual_scale=_readonly(residual_scale), diagnostics=MappingProxyType(diagnostics),
    )


def apply_innovation_map(core: np.ndarray, offsets: Sequence[int], fitted: InnovationMap) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fitted map causally.  First-token innovations are exactly zero."""

    values = np.asarray(core, dtype=np.float64)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    positions, _owners, predecessor = _positions_and_owners(offsets_array)
    output = np.zeros_like(values)
    mask = predecessor >= 0
    valid = np.flatnonzero(mask)
    prior = predecessor[valid]
    for target in range(9):
        prediction = (
            fitted.intercept[target]
            + fitted.time_coefficient[target] * np.log1p(positions[valid])
            + fitted.self_coefficient[target] * values[prior, target]
            + values[prior] @ fitted.cross_coefficients[target]
        )
        output[valid, target] = (values[valid, target] - prediction) / fitted.residual_scale[target]
    if not np.isfinite(output).all() or np.any(output[~mask] != 0.0):
        raise RuntimeError("causal innovation application failed")
    return output, mask.astype(np.float64)


class _InnovationAdditiveEnergy:
    """Original B3 plus three operator-grouped innovation subnetworks."""

    def __init__(self, config: ContinuousDeemConfig, seed: int, gain: float):
        import torch

        self.torch = torch
        self.config = config
        self.seed = int(seed)
        self.gain = float(gain)
        self.base = _FamilyAdditiveEnergy(SHARED_GLOBAL_FEATURES, config, seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + 7_000_001)
        dtype = torch.float64
        self.w = torch.nn.ParameterDict()
        self.W = torch.nn.ParameterDict()
        self.d = torch.nn.ParameterDict()
        self.V = torch.nn.ParameterDict()
        self.e = torch.nn.ParameterDict()
        for operator, indices in INNOVATION_GROUPS.items():
            size = len(indices)
            self.w[operator] = torch.nn.Parameter(torch.full((size,), 2.0 / (3 * size), dtype=dtype))
            self.W[operator] = torch.nn.Parameter(torch.randn(config.family_width, size, dtype=dtype, generator=generator) * config.init_sd)
            self.d[operator] = torch.nn.Parameter(torch.randn(config.family_width, dtype=dtype, generator=generator) * config.init_sd)
            self.V[operator] = torch.nn.Parameter(torch.randn(size, config.family_width, dtype=dtype, generator=generator) * config.init_sd)
            self.e[operator] = torch.nn.Parameter(torch.randn(size, dtype=dtype, generator=generator) * config.init_sd)

    def parameters(self):
        output = list(self.base.parameters())
        for collection in (self.w, self.W, self.d, self.V, self.e):
            output.extend(collection.values())
        return output

    def innovation_contributions(self, innovation, mask):
        torch = self.torch
        atomic = torch.zeros_like(innovation)
        families = {}
        for operator, indices in INNOVATION_GROUPS.items():
            xg = innovation[:, list(indices)]
            hidden = torch.tanh(xg @ self.W[operator].T + self.d[operator])
            contribution = self.w[operator] * xg + (2.0 / len(indices)) * torch.tanh(
                hidden @ self.V[operator].T + self.e[operator]
            )
            contribution = contribution * mask[:, None]
            atomic[:, list(indices)] = contribution
            families[operator] = contribution.sum(dim=1)
        return atomic, families

    def logit(self, original, innovation, mask):
        base_logit, base_atomic, base_family = self.base.logit(original)
        innovation_atomic, innovation_family = self.innovation_contributions(innovation, mask)
        total = base_logit + self.gain * innovation_atomic.sum(dim=1)
        return total, base_atomic, innovation_atomic, base_family, innovation_family

    def free_energy(self, packed):
        original = packed[:, :29]
        innovation = packed[:, 29:38]
        mask = packed[:, 38]
        total, _ba, _ia, _bf, _if = self.logit(original, innovation, mask)
        return 0.5 * ((original - self.base.a) ** 2).sum(dim=1) - self.torch.nn.functional.softplus(total)

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        output = {f"base::{key}": value for key, value in self.base.state_dict_numpy().items()}
        for label, collection in (("w", self.w), ("W", self.W), ("d", self.d), ("V", self.V), ("e", self.e)):
            for operator, parameter in collection.items():
                output[f"innovation::{label}::{operator}"] = parameter.detach().cpu().numpy().copy()
        return output

    def load_state_numpy(self, state: Mapping[str, np.ndarray]) -> None:
        import torch

        self.base.load_state_numpy({key.removeprefix("base::"): value for key, value in state.items() if key.startswith("base::")})
        with torch.no_grad():
            for label, collection in (("w", self.w), ("W", self.W), ("d", self.d), ("V", self.V), ("e", self.e)):
                for operator, parameter in collection.items():
                    parameter.copy_(torch.as_tensor(state[f"innovation::{label}::{operator}"], dtype=torch.float64))


def fit_innovation_b3(
    original_risk: np.ndarray, innovations: np.ndarray, innovation_mask: np.ndarray,
    *, seed: int, config: ContinuousDeemConfig | None = None, gain: float = 1.0,
) -> InnovationB3Fit | ContinuousDeemResult:
    """Fit B3.  ``gain=0`` delegates directly to frozen B3 for exact identity."""

    if float(gain) == 0.0:
        return fit_continuous_deem(
            np.asarray(original_risk, dtype=np.float64), SHARED_GLOBAL_FEATURES,
            seed=int(seed), config=config or ContinuousDeemConfig(),
        )
    import torch

    started = time.perf_counter()
    config = config or ContinuousDeemConfig()
    if config.dtype != "float64" or config.device != "cpu":
        raise ValueError("token B3 v1 is frozen to float64 CPU")
    X = np.asarray(original_risk, dtype=np.float64)
    U = np.asarray(innovations, dtype=np.float64)
    M = np.asarray(innovation_mask, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 29 or U.shape != (len(X), 9) or M.shape != (len(X),):
        raise ValueError("held token innovation B3 matrices disagree")
    if not np.isfinite(X).all() or not np.isfinite(U).all() or not np.isin(M, (0.0, 1.0)).all():
        raise ValueError("held token innovation B3 input is not finite/binary")
    set_determinism(seed)
    model = _InnovationAdditiveEnergy(config, int(seed), float(gain))
    packed = torch.as_tensor(np.column_stack([X, U, M]), dtype=torch.float64)
    buffer = packed.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1_000_003)
    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=config.learning_rate, momentum=config.momentum)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        refresh = torch.rand(len(packed), generator=generator) < config.replay_refresh
        if bool(refresh.any()):
            replacements = torch.randint(len(packed), (int(refresh.sum()),), generator=generator)
            buffer[refresh] = packed[replacements]
        buffer, acceptance = persistent_mala(
            model, buffer, delta=config.mala_delta, steps=config.mala_steps, generator=generator,
        )
        loss = model.free_energy(packed).mean() - model.free_energy(buffer).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite token B3 loss at epoch {epoch}")
        optimizer.zero_grad()
        loss.backward()
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
            raise FloatingPointError(f"non-finite token B3 gradient at epoch {epoch}")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise FloatingPointError(f"non-finite token B3 parameter at epoch {epoch}")
        history.append({"epoch": float(epoch), "loss": float(loss.detach()), "mala_acceptance": acceptance})
    with torch.no_grad():
        total_t, base_t, innov_t, base_family_t, innov_family_t = model.logit(
            packed[:, :29], packed[:, 29:38], packed[:, 38]
        )
    total = total_t.cpu().numpy()
    base_atomic = base_t.cpu().numpy()
    innov_atomic = innov_t.cpu().numpy()
    base_family = {name: value.cpu().numpy() for name, value in base_family_t.items()}
    innov_family = {name: value.cpu().numpy() for name, value in innov_family_t.items()}
    posterior = 1.0 / (1.0 + np.exp(-np.clip(total, -700.0, 700.0)))
    anchor = equal_family_risk_anchor(X, SHARED_GLOBAL_FEATURES)
    high = float(np.sum(posterior * anchor) / max(np.sum(posterior), EPS))
    low = float(np.sum((1.0 - posterior) * anchor) / max(np.sum(1.0 - posterior), EPS))
    difference = high - low
    if abs(difference) <= config.anchor_tolerance:
        raise RuntimeError("token B3 orientation anchor is ambiguous")
    orientation = 1 if difference > 0 else -1
    if orientation < 0:
        posterior = 1.0 - posterior
        total = -total
        base_atomic = -base_atomic
        innov_atomic = -innov_atomic
        base_family = {name: -value for name, value in base_family.items()}
        innov_family = {name: -value for name, value in innov_family.items()}
    aligned_bias = orientation * float(model.base.b.detach())
    reconstruction = float(np.max(np.abs(
        aligned_bias + base_atomic.sum(axis=1) + float(gain) * innov_atomic.sum(axis=1) - total
    )))
    masked_max = float(np.max(np.abs(innov_atomic[M == 0.0]))) if np.any(M == 0.0) else 0.0
    health = {
        "healthy": bool(
            np.std(posterior) >= config.posterior_sd_min
            and reconstruction <= 1e-8 and masked_max == 0.0
        ),
        "posterior_sd": float(np.std(posterior)),
        "additive_logit_reconstruction_max_abs": reconstruction,
        "masked_first_token_innovation_contribution_max_abs": masked_max,
        "mala_acceptance_mean": float(np.mean([row["mala_acceptance"] for row in history])),
        "epochs_completed": len(history),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return InnovationB3Fit(
        score=posterior.copy(), posterior=np.column_stack([1.0 - posterior, posterior]),
        logit=total, original_contributions=base_atomic,
        innovation_contributions=innov_atomic,
        original_family_contributions=base_family,
        innovation_family_contributions=innov_family,
        aligned_bias=aligned_bias, orientation=orientation,
        state=model.state_dict_numpy(), objective_history=history, health=health,
        config={"continuous": asdict(config)}, seed=int(seed), innovation_gain=float(gain),
    )


def predict_innovation_b3(
    fitted: InnovationB3Fit | ContinuousDeemResult, original_risk: np.ndarray,
    innovations: np.ndarray, innovation_mask: np.ndarray,
) -> dict[str, Any]:
    if isinstance(fitted, ContinuousDeemResult):
        return predict_continuous_deem(fitted, np.asarray(original_risk, dtype=np.float64))
    import torch

    X = np.asarray(original_risk, dtype=np.float64)
    U = np.asarray(innovations, dtype=np.float64)
    M = np.asarray(innovation_mask, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 29 or U.shape != (len(X), 9) or M.shape != (len(X),):
        raise ValueError("held token innovation B3 matrices disagree")
    if not np.isfinite(X).all() or not np.isfinite(U).all() or not np.isin(M, (0.0, 1.0)).all():
        raise ValueError("held token innovation B3 input is not finite/binary")
    config = ContinuousDeemConfig(**fitted.config["continuous"])
    model = _InnovationAdditiveEnergy(config, fitted.seed, fitted.innovation_gain)
    model.load_state_numpy(fitted.state)
    with torch.no_grad():
        total_t, base_t, innov_t, base_family_t, innov_family_t = model.logit(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(U, dtype=torch.float64),
            torch.as_tensor(M, dtype=torch.float64),
        )
    orientation = fitted.orientation
    total = orientation * total_t.cpu().numpy()
    base = orientation * base_t.cpu().numpy()
    innov = orientation * innov_t.cpu().numpy()
    score = 1.0 / (1.0 + np.exp(-np.clip(total, -700.0, 700.0)))
    reconstruction = float(np.max(np.abs(
        fitted.aligned_bias + base.sum(axis=1)
        + fitted.innovation_gain * innov.sum(axis=1) - total
    )))
    if reconstruction > 1e-8:
        raise RuntimeError("held token B3 additive-logit reconstruction failed")
    return {
        "score": score,
        "posterior": np.column_stack([1.0 - score, score]),
        "logit": total,
        "original_contributions": base,
        "innovation_contributions": innov,
        "original_family_contributions": {
            name: orientation * value.cpu().numpy() for name, value in base_family_t.items()
        },
        "innovation_family_contributions": {
            name: orientation * value.cpu().numpy() for name, value in innov_family_t.items()
        },
        "reconstruction_max_abs": reconstruction,
    }


def _state_record(fitted: InnovationB3Fit | ContinuousDeemResult, *, fold: int, method_id: str) -> dict[str, Any]:
    if isinstance(fitted, ContinuousDeemResult):
        return {
            "fold": int(fold), "method_id": method_id, "seed": int(fitted.seed),
            "kind": "continuous_b3", "orientation": int(fitted.orientation),
            "aligned_bias": float(fitted.aligned_bias), "health": dict(fitted.health),
            "config": dict(fitted.config), "state": fitted.state,
        }
    return {
        "fold": int(fold), "method_id": method_id, "seed": int(fitted.seed),
        "kind": "continuous_b3_plus_innovation", "orientation": int(fitted.orientation),
        "aligned_bias": float(fitted.aligned_bias), "innovation_gain": fitted.innovation_gain,
        "health": dict(fitted.health), "config": dict(fitted.config), "state": fitted.state,
    }


def fit_token_b3_ladder(
    preparation: TokenFusionPreparation,
    *, seeds: Sequence[int] = (0, 1, 2, 3, 4),
    config: ContinuousDeemConfig | None = None,
    innovation_gain: float = 1.0,
    execution_workers: int = 1,
) -> dict[str, TokenB3Result]:
    """Five-fold question-held target-free fit of all five frozen Phase-2 arms."""

    seeds = tuple(map(int, seeds))
    if seeds != (0, 1, 2, 3, 4):
        raise ValueError("Phase-2 B3 seed roster is frozen to (0,1,2,3,4)")
    config = config or ContinuousDeemConfig()
    execution_workers = int(execution_workers)
    if execution_workers < 1:
        raise ValueError("execution_workers must be positive")
    output_scores = {method: np.empty(len(preparation.values), dtype=np.float64) for method in METHOD_IDS}
    model_records: dict[str, list[Mapping[str, Any]]] = {method: [] for method in METHOD_IDS}
    maps: dict[str, list[InnovationMap]] = {method: [] for method in METHOD_IDS}
    fold_records: dict[str, list[Mapping[str, Any]]] = {method: [] for method in METHOD_IDS}
    for fold in range(OUTER_FOLDS):
        fold_data = prepare_fold_data(preparation, fold)
        core = np.asarray(fold_data.risk[:, CORE_INDICES], dtype=np.float64)
        pstg = select_projected_stg_support(
            core, preparation.token_offsets, preparation.row_ids,
            fold_data.donor_rows, fold_data.donor_fit_indices,
        )
        supports = {
            LOCAL_TOKEN_B3_SELF_INNOV: fixed_support(LOCAL_TOKEN_B3_SELF_INNOV),
            LOCAL_TOKEN_B3_ROOK_ALL_INNOV: fixed_support(LOCAL_TOKEN_B3_ROOK_ALL_INNOV),
            LOCAL_TOKEN_B3_ROOK_PSTG_INNOV: np.asarray(pstg.support, dtype=bool),
            LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL: fixed_support(LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL),
        }
        innovation_by_method = {}
        mask_by_method = {}
        for method_id, support in supports.items():
            fitted_map = fit_innovation_map(
                method_id, core, preparation.token_offsets, fold_data.donor_rows,
                fold_data.donor_fit_indices, support,
                support_diagnostics=(pstg.diagnostics if method_id == LOCAL_TOKEN_B3_ROOK_PSTG_INNOV else None),
            )
            innovation, mask = apply_innovation_map(core, preparation.token_offsets, fitted_map)
            maps[method_id].append(fitted_map)
            innovation_by_method[method_id] = innovation
            mask_by_method[method_id] = mask
        held_indices = _row_token_indices(preparation.token_offsets, fold_data.held_rows)
        fit_indices = np.asarray(fold_data.donor_fit_indices, dtype=np.int64)
        def fit_seed(task: tuple[str, int]):
            method_id, seed = task
            if method_id == LOCAL_TOKEN_B3:
                fitted = fit_continuous_deem(
                    fold_data.risk[fit_indices], SHARED_GLOBAL_FEATURES,
                    seed=seed, config=config,
                )
                prediction = predict_continuous_deem(fitted, fold_data.risk[held_indices])
            else:
                fitted = fit_innovation_b3(
                    fold_data.risk[fit_indices], innovation_by_method[method_id][fit_indices],
                    mask_by_method[method_id][fit_indices], seed=seed, config=config,
                    gain=innovation_gain,
                )
                prediction = predict_innovation_b3(
                    fitted, fold_data.risk[held_indices], innovation_by_method[method_id][held_indices],
                    mask_by_method[method_id][held_indices],
                )
            if not fitted.health.get("healthy", False):
                raise RuntimeError(
                    f"unhealthy token B3 fit: fold={fold} method={method_id} seed={seed}"
                )
            return method_id, seed, fitted, np.asarray(prediction["score"], dtype=np.float64)

        tasks = [(method_id, seed) for method_id in METHOD_IDS for seed in seeds]
        if execution_workers == 1:
            fitted_tasks = list(map(fit_seed, tasks))
        else:
            # PyTorch is frozen to one intra-op thread by set_determinism.  A
            # thread pool therefore schedules independent method/seed fits
            # without copying the 60k-token donor matrix.  map() preserves the
            # registered method/seed order, so serialization is schedule-free.
            with ThreadPoolExecutor(max_workers=execution_workers) as executor:
                fitted_tasks = list(executor.map(fit_seed, tasks))

        by_method: dict[str, list[tuple[int, Any, np.ndarray]]] = {
            method: [] for method in METHOD_IDS
        }
        for method_id, seed, fitted, score in fitted_tasks:
            by_method[method_id].append((seed, fitted, score))

        for method_id in METHOD_IDS:
            seed_scores = []
            seed_health = []
            for seed, fitted, score in by_method[method_id]:
                seed_scores.append(score)
                seed_health.append(dict(fitted.health))
                model_records[method_id].append(MappingProxyType(_state_record(fitted, fold=fold, method_id=method_id)))
            posterior_mean = np.mean(np.vstack(seed_scores), axis=0)
            output_scores[method_id][held_indices] = posterior_mean
            fold_record = {
                "fold": fold,
                "method_id": method_id,
                "donor_row_count": int(len(fold_data.donor_rows)),
                "held_row_count": int(len(fold_data.held_rows)),
                "donor_fit_token_count": int(len(fit_indices)),
                "held_token_count": int(len(held_indices)),
                "execution_workers": execution_workers,
                "seed_spearman_median": _median_pairwise_spearman(seed_scores),
                "all_seed_health": seed_health,
                "preprocessing": dict(fold_data.diagnostics),
                "labels_seen_during_fit": False,
            }
            if method_id == LOCAL_TOKEN_B3_ROOK_PSTG_INNOV:
                fold_record["projected_stg"] = dict(pstg.diagnostics)
                fold_record["exact_subset_records"] = list(pstg.exact_subset_records)
            fold_records[method_id].append(MappingProxyType(fold_record))
    output = {}
    for method_id in METHOD_IDS:
        scores = output_scores[method_id]
        if not np.isfinite(scores).all():
            raise RuntimeError(f"Phase-2 arm has incomplete token scores: {method_id}")
        seed_spearman = [float(record["seed_spearman_median"]) for record in fold_records[method_id]]
        health = {
            "healthy": True,
            "fold_count": OUTER_FOLDS,
            "seed_count": len(seeds),
            "median_seed_spearman": float(np.median(seed_spearman)),
            "minimum_fold_seed_spearman": float(np.min(seed_spearman)),
            "all_scores_finite": True,
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
            "execution_workers": execution_workers,
        }
        output[method_id] = TokenB3Result(
            method_id=method_id, token_risk=_readonly(scores),
            per_seed_model_records=tuple(model_records[method_id]),
            innovation_maps=tuple(maps[method_id]),
            fold_diagnostics=tuple(fold_records[method_id]),
            health=MappingProxyType(health),
        )
    if tuple(output) != METHOD_IDS:
        raise AssertionError("Phase-2 method roster drifted")
    return output


def _median_pairwise_spearman(scores: Sequence[np.ndarray]) -> float:
    from scipy.stats import spearmanr

    values = []
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            correlation = float(spearmanr(scores[left], scores[right]).statistic)
            if np.isfinite(correlation):
                values.append(correlation)
    return float(np.median(values)) if values else float("nan")


def innovation_map_record(value: InnovationMap) -> dict[str, Any]:
    return {
        "method_id": value.method_id,
        "support": value.support.astype(int).tolist(),
        "intercept": value.intercept.tolist(),
        "time_coefficient": value.time_coefficient.tolist(),
        "self_coefficient": value.self_coefficient.tolist(),
        "cross_coefficients": value.cross_coefficients.tolist(),
        "residual_scale": value.residual_scale.tolist(),
        "diagnostics": dict(value.diagnostics),
    }


def restore_innovation_map(value: Mapping[str, Any]) -> InnovationMap:
    return InnovationMap(
        method_id=str(value["method_id"]),
        support=_readonly(np.asarray(value["support"], dtype=bool)),
        intercept=_readonly(np.asarray(value["intercept"], dtype=np.float64)),
        time_coefficient=_readonly(np.asarray(value["time_coefficient"], dtype=np.float64)),
        self_coefficient=_readonly(np.asarray(value["self_coefficient"], dtype=np.float64)),
        cross_coefficients=_readonly(np.asarray(value["cross_coefficients"], dtype=np.float64)),
        residual_scale=_readonly(np.asarray(value["residual_scale"], dtype=np.float64)),
        diagnostics=MappingProxyType(dict(value["diagnostics"])),
    )


__all__ = [
    "CORE_INDICES", "CORE_OPERATORS", "CORE_SOURCES", "CORE_TOKEN_VIEWS",
    "LOCAL_TOKEN_B3", "LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL",
    "LOCAL_TOKEN_B3_ROOK_ALL_INNOV", "LOCAL_TOKEN_B3_ROOK_PSTG_INNOV",
    "LOCAL_TOKEN_B3_SELF_INNOV", "METHOD_IDS", "InnovationB3Fit", "InnovationMap",
    "ProjectedSTGResult", "TokenB3Result", "apply_innovation_map", "deterministic_donor_cap",
    "fit_innovation_b3", "fit_innovation_map", "fit_token_b3_ladder", "fixed_support",
    "innovation_map_record", "nonrook_peers", "predict_innovation_b3", "prepare_fold_data",
    "restore_innovation_map", "rook_peers", "select_projected_stg_support",
]
