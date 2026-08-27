"""Target-free crossed source-by-operator residual extension of DEEM B3.

The historical continuous B3 model is loaded from a frozen same-cell,
same-seed state and is never updated.  The only new trainable path is a
bounded correction over the physical 3x3 telemetry core::

    source   = H15 entropy | sampled-token surprisal | raw log-partition
    operator = mean        | sliding variance        | CUSUM

The nine frozen B3 atomic contributions are donor-centered/scaled and squashed
to ``z``.  Each permitted edge contributes one quadratic product ``z_i*z_j``.
The row, column and union arms contain respectively 9, 9 and 18 independent
edge weights; an exact 18-edge non-rook complement is the matched topology
control.  Consequently the module is a structured "rook-move" mixer rather
than a dense layer.

The correction is nested exactly around B3:

    logit_crossed(x) = logit_B3(x) + sum_ij delta_ij(x)

where ``sum_ij delta_ij`` is bounded by ``strength * correction_cap``.  Zero
strength, or the all-zero initialization, is an exact B3 logit identity.  Fit
uses only the same persistent-MALA contrastive free-energy objective as B3;
this module accepts no labels or targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .residual_graph_deem import (
    ContinuousDeemConfig,
    ResidualGraphDeemError,
    _FamilyAdditiveEnergy,
    equal_family_risk_anchor,
    jsonable,
    persistent_mala,
    set_determinism,
    validate_inventory,
)


EPS = 1e-12

SOURCES = ("entropy_h15", "sampled_surprisal", "raw_log_partition")
OPERATORS = ("mean", "sliding_variance", "cusum")
CORE_GRID = {
    ("entropy_h15", "mean"): "epr",
    ("entropy_h15", "sliding_variance"): "sw_var_peak",
    ("entropy_h15", "cusum"): "cusum_max",
    ("sampled_surprisal", "mean"): "epr_spilled",
    ("sampled_surprisal", "sliding_variance"): "sw_var_peak_spilled",
    ("sampled_surprisal", "cusum"): "cusum_max_spilled",
    ("raw_log_partition", "mean"): "epr_energy",
    ("raw_log_partition", "sliding_variance"): "sw_var_peak_energy",
    ("raw_log_partition", "cusum"): "cusum_max_energy",
}
CORE_FEATURES = tuple(
    CORE_GRID[(source, operator)] for source in SOURCES for operator in OPERATORS
)
MODES = ("alias", "row_only", "column_only", "crossed", "nonrook_18")


@dataclass(frozen=True)
class CrossedRookConfig:
    """Configuration for a frozen-B3 crossed residual fit."""

    mode: str = "crossed"
    strength: float = 1.0
    correction_cap: float = 0.50
    epochs: int = 100
    learning_rate: float = 2e-3
    optimizer: str = "adam"
    gradient_clip: float = 5.0
    mala_delta: float = 0.10
    mala_steps: int = 5
    replay_refresh: float = 0.05
    trust_weight: float = 0.0
    l2_weight: float = 1e-4
    family_width: int = 8
    base_init_sd: float = 0.005
    anchor_tolerance: float = 1e-6
    posterior_sd_min: float = 1e-3
    dtype: str = "float64"
    device: str = "cpu"
    deterministic: bool = True


@dataclass
class CrossedRookResult:
    score: np.ndarray
    posterior: np.ndarray
    logit: np.ndarray
    base_logit: np.ndarray
    contributions: np.ndarray
    base_contributions: np.ndarray
    correction: np.ndarray
    cell_delta: np.ndarray
    edge_raw_contribution: np.ndarray
    edge_values: np.ndarray
    edge_pairs: tuple[tuple[int, int], ...]
    edge_kinds: tuple[str, ...]
    edge_weights: np.ndarray
    core_features: tuple[str, ...]
    core_indices: tuple[int, ...]
    aligned_bias: float
    orientation: int
    risk_anchor_difference: float
    feature_names: tuple[str, ...]
    state: dict[str, np.ndarray]
    objective_history: list[dict[str, float]]
    health: dict[str, Any]
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    seed: int


def _validate_config(config: CrossedRookConfig) -> None:
    if config.mode not in MODES:
        raise ValueError(f"unknown crossed-rook mode {config.mode!r}; expected {MODES}")
    if not 0.0 <= float(config.strength) <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if float(config.correction_cap) < 0.0:
        raise ValueError("correction_cap must be nonnegative")
    if int(config.epochs) < 0 or int(config.mala_steps) < 1:
        raise ValueError("epochs must be nonnegative and mala_steps positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.optimizer != "adam" or float(config.gradient_clip) <= 0.0:
        raise ValueError("v1 optimizer is Adam with a positive gradient clip")
    if not 0.0 <= float(config.replay_refresh) <= 1.0:
        raise ValueError("replay_refresh must be in [0, 1]")
    if float(config.trust_weight) < 0.0 or float(config.l2_weight) < 0.0:
        raise ValueError("penalty weights must be nonnegative")
    if config.dtype != "float64" or config.device != "cpu":
        raise ValueError("crossed-rook v1 is frozen to float64 CPU")
    if config.mode == "alias" and (
        float(config.strength) != 0.0 or int(config.epochs) != 0
    ):
        raise ValueError("alias mode requires strength=0 and epochs=0")


def core_index_grid(feature_names: Sequence[str]) -> tuple[int, ...]:
    """Return the row-major physical 3x3 core indices.

    V1 is intentionally strict: a cell that lacks any of the nine universal
    physical measurements is ineligible rather than silently imputed.
    """

    names = tuple(str(name) for name in feature_names)
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in CORE_FEATURES if name not in lookup]
    if missing:
        raise ResidualGraphDeemError(
            "crossed-rook physical core is incomplete: " + ", ".join(missing)
        )
    return tuple(lookup[name] for name in CORE_FEATURES)


class _CrossedRookEnergy:
    """Frozen B3 plus a bounded quadratic rook-move correction.

    The coordinate vector is the frozen B3 atomic contribution on the nine
    physical core features, centered/scaled on the donor rows and squashed by
    tanh.  Every active edge has one independent coefficient.  Row, column,
    union, and a cardinality-matched permuted-union mask therefore differ only
    in topology, not capacity.
    """

    def __init__(self, feature_names: Sequence[str], config: CrossedRookConfig, seed: int):
        import torch

        _validate_config(config)
        self.torch = torch
        self.names = tuple(str(name) for name in feature_names)
        self.config = config
        self.seed = int(seed)
        self.core_indices = core_index_grid(self.names)
        base_config = ContinuousDeemConfig(
            family_width=int(config.family_width),
            init_sd=float(config.base_init_sd),
        )
        self.base = _FamilyAdditiveEnergy(self.names, base_config, self.seed)
        dtype = torch.float64

        self.row_edges = tuple(
            (3 * row + left, 3 * row + right)
            for row in range(3)
            for left in range(3)
            for right in range(left + 1, 3)
        )
        self.column_edges = tuple(
            (3 * upper + column, 3 * lower + column)
            for column in range(3)
            for upper in range(3)
            for lower in range(upper + 1, 3)
        )
        self.rook_edges = self.row_edges + self.column_edges
        # Over F_3, (r,c)->(r+c,r-c) maps each rook edge to a non-rook
        # edge.  Thus this is a genuine vertex-permuted, 18-edge,
        # cardinality-matched topology control rather than a random-capacity
        # change.
        permutation = tuple(
            3 * ((row + column) % 3) + ((row - column) % 3)
            for row in range(3)
            for column in range(3)
        )
        self.permuted_edges = tuple(
            tuple(sorted((permutation[left], permutation[right])))
            for left, right in self.rook_edges
        )
        if len(set(self.rook_edges)) != 18 or len(set(self.permuted_edges)) != 18:
            raise AssertionError("crossed-rook edge construction is not one-to-one")
        if set(self.rook_edges) & set(self.permuted_edges):
            raise AssertionError("permuted 18-edge control must be the rook complement")
        if config.mode == "row_only":
            self.edge_pairs = self.row_edges
            self.edge_kinds = ("row",) * len(self.row_edges)
        elif config.mode == "column_only":
            self.edge_pairs = self.column_edges
            self.edge_kinds = ("column",) * len(self.column_edges)
        elif config.mode == "nonrook_18":
            self.edge_pairs = self.permuted_edges
            self.edge_kinds = ("nonrook",) * len(self.permuted_edges)
        else:
            self.edge_pairs = self.rook_edges
            self.edge_kinds = ("row",) * len(self.row_edges) + (
                "column",
            ) * len(self.column_edges)
        self.edge_weight = torch.nn.Parameter(
            torch.zeros((len(self.edge_pairs),), dtype=dtype)
        )
        # Fitted once from donor rows before any negative-phase sampling, then
        # held fixed and applied to both positive and proposal coordinates.
        self.core_mean = torch.zeros((9,), dtype=dtype)
        self.core_scale = torch.ones((9,), dtype=dtype)
        self.edge_mean = torch.zeros((len(self.edge_pairs),), dtype=dtype)

    def load_baseline_state(self, state: Mapping[str, np.ndarray]) -> None:
        self.base.load_state_numpy(state)

    def fit_coordinate_transform(self, X) -> None:
        with self.torch.no_grad():
            base_atomic, _ = self.base.contributions(X)
            core = base_atomic[:, list(self.core_indices)]
            mean = core.mean(dim=0)
            scale = core.std(dim=0, unbiased=False).clamp_min(1e-8)
            self.core_mean.copy_(mean)
            self.core_scale.copy_(scale)
            z = self.torch.tanh((core - mean[None, :]) / scale[None, :])
            edge_values = self._edge_values_from_z(z)
            self.edge_mean.copy_(edge_values.mean(dim=0))

    def _edge_values_from_z(self, z):
        return self.torch.stack(
            [z[:, left] * z[:, right] for left, right in self.edge_pairs], dim=1
        )

    def coordinate_values(self, base_atomic):
        core = base_atomic[:, list(self.core_indices)]
        z = self.torch.tanh(
            (core - self.core_mean[None, :]) / self.core_scale[None, :]
        )
        return self._edge_values_from_z(z) - self.edge_mean[None, :]

    def parameters(self):
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.edge_weight.requires_grad_(self.config.mode != "alias")
        return [self.edge_weight] if self.config.mode != "alias" else []

    def correction_components(self, base_atomic):
        edge_values = self.coordinate_values(base_atomic)
        edge_raw_contribution = (
            edge_values * self.edge_weight[None, :] / math.sqrt(len(self.edge_pairs))
        )
        raw = edge_raw_contribution.sum(dim=1)
        correction = (
            float(self.config.strength)
            * float(self.config.correction_cap)
            * self.torch.tanh(raw)
        )
        # Preserve an exact atomic decomposition without pretending the outer
        # tanh has a unique per-edge attribution.  Each physical core cell
        # receives one ninth; edge_raw_contribution is stored separately as the
        # pre-saturation mechanism trace.
        cell_delta = correction[:, None, None].expand((-1, 3, 3)) / 9.0
        return {
            "cell_delta": cell_delta,
            "correction": correction,
            "edge_values": edge_values,
            "edge_raw_contribution": edge_raw_contribution,
        }

    def components(self, X):
        base_atomic, base_family = self.base.contributions(X)
        values = self.correction_components(base_atomic)
        contributions = base_atomic.clone()
        flat_delta = values["cell_delta"].reshape((-1, 9))
        contributions[:, list(self.core_indices)] = (
            contributions[:, list(self.core_indices)] + flat_delta
        )
        return {
            **values,
            "base_atomic": base_atomic,
            "base_family": base_family,
            "contributions": contributions,
        }

    def logit(self, X):
        values = self.components(X)
        ell = self.base.b + values["contributions"].sum(dim=1)
        return ell, values

    def free_energy(self, X):
        ell, _ = self.logit(X)
        return 0.5 * ((X - self.base.a) ** 2).sum(dim=1) - self.torch.nn.functional.softplus(
            ell
        )

    def penalties(self, X):
        base_atomic, _ = self.base.contributions(X)
        values = self.correction_components(base_atomic)
        trust = values["correction"].square().mean()
        l2 = self.edge_weight.square().mean()
        return trust, l2

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        output = {
            f"base::{name}": value for name, value in self.base.state_dict_numpy().items()
        }
        output.update(
            {
                "rook::edge_weight": self.edge_weight.detach().cpu().numpy().copy(),
                "rook::core_mean": self.core_mean.detach().cpu().numpy().copy(),
                "rook::core_scale": self.core_scale.detach().cpu().numpy().copy(),
                "rook::edge_mean": self.edge_mean.detach().cpu().numpy().copy(),
            }
        )
        return output

    def load_state_numpy(self, state: Mapping[str, np.ndarray]) -> None:
        base_state = {
            name.removeprefix("base::"): value
            for name, value in state.items()
            if name.startswith("base::")
        }
        self.base.load_state_numpy(base_state)
        torch = self.torch
        with torch.no_grad():
            self.edge_weight.copy_(
                torch.as_tensor(state["rook::edge_weight"], dtype=torch.float64)
            )
            self.core_mean.copy_(
                torch.as_tensor(state["rook::core_mean"], dtype=torch.float64)
            )
            self.core_scale.copy_(
                torch.as_tensor(state["rook::core_scale"], dtype=torch.float64)
            )
            self.edge_mean.copy_(
                torch.as_tensor(state["rook::edge_mean"], dtype=torch.float64)
            )


def fit_deem_b3_crossed_rook(
    X_risk: np.ndarray,
    feature_names: Sequence[str],
    baseline_state: Mapping[str, np.ndarray],
    *,
    baseline_orientation: int,
    baseline_score: np.ndarray | None = None,
    seed: int = 0,
    config: CrossedRookConfig | None = None,
) -> CrossedRookResult:
    """Fit one target-free frozen-B3 crossed-rook residual model."""

    import torch

    started = time.perf_counter()
    config = config or CrossedRookConfig()
    _validate_config(config)
    X = validate_inventory(X_risk, feature_names)
    names = tuple(str(name) for name in feature_names)
    set_determinism(seed)
    model = _CrossedRookEnergy(names, config, seed)
    model.load_baseline_state(baseline_state)
    tensor = torch.as_tensor(X, dtype=torch.float64)
    model.fit_coordinate_transform(tensor)
    parameters = list(model.parameters())
    if int(config.epochs) > 0 and not parameters:
        raise ValueError("active crossed-rook fit has no trainable parameters")

    buffer = tensor.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 8_310_017)
    optimizer = None
    if parameters:
        optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate))
    history: list[dict[str, float]] = []
    last_finite_state = model.state_dict_numpy()
    try:
        for epoch in range(int(config.epochs)):
            refresh = torch.rand(len(X), generator=generator) < float(config.replay_refresh)
            if bool(refresh.any()):
                replacements = torch.randint(
                    len(X), (int(refresh.sum()),), generator=generator
                )
                buffer[refresh] = tensor[replacements]
            buffer, acceptance = persistent_mala(
                model,
                buffer,
                delta=float(config.mala_delta),
                steps=int(config.mala_steps),
                generator=generator,
            )
            positive = model.free_energy(tensor).mean()
            negative = model.free_energy(buffer).mean()
            deem_loss = positive - negative
            trust, l2 = model.penalties(tensor)
            loss = (
                deem_loss
                + float(config.trust_weight) * trust
                + float(config.l2_weight) * l2
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite crossed-rook objective at epoch {epoch}"
                )
            optimizer.zero_grad()
            loss.backward()
            for index, parameter in enumerate(parameters):
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(
                        f"non-finite crossed-rook gradient for parameter {index} at epoch {epoch}"
                    )
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters, max_norm=float(config.gradient_clip)
                ).detach()
            )
            if not np.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite crossed-rook gradient norm at epoch {epoch}"
                )
            optimizer.step()
            for index, parameter in enumerate(parameters):
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(
                        f"non-finite crossed-rook parameter {index} after epoch {epoch}"
                    )
            last_finite_state = model.state_dict_numpy()
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach()),
                    "deem_loss": float(deem_loss.detach()),
                    "trust_penalty_raw": float(trust.detach()),
                    "l2_penalty_raw": float(l2.detach()),
                    "gradient_norm_preclip": gradient_norm,
                    "mala_acceptance": float(acceptance),
                }
            )
    except Exception as exc:
        setattr(exc, "objective_history", history)
        setattr(exc, "last_finite_state", last_finite_state)
        raise

    state = model.state_dict_numpy()
    with torch.no_grad():
        ell_t, values = model.logit(tensor)
        base_logit_t, base_atomic_t, _ = model.base.logit(tensor)
        ell = ell_t.cpu().numpy()
        base_logit = base_logit_t.cpu().numpy()
        contributions = values["contributions"].cpu().numpy()
        base_contributions = base_atomic_t.cpu().numpy()
        correction = values["correction"].cpu().numpy()
        cell_delta = values["cell_delta"].cpu().numpy()
        edge_values = values["edge_values"].cpu().numpy()
        edge_raw_contribution = values["edge_raw_contribution"].cpu().numpy()

    raw_reconstruction = float(
        np.max(np.abs(model.base.b.detach().item() + contributions.sum(axis=1) - ell))
    )
    residual_identity = float(np.max(np.abs(ell - base_logit - correction)))
    if raw_reconstruction > 1e-8 or residual_identity > 1e-8:
        raise ResidualGraphDeemError(
            "crossed-rook decomposition failed: "
            f"atomic={raw_reconstruction:.3e}, residual={residual_identity:.3e}"
        )

    orientation = int(baseline_orientation)
    if orientation not in {-1, 1}:
        raise ValueError("baseline_orientation must be -1 or +1")
    q = 1.0 / (1.0 + np.exp(-np.clip(ell, -700.0, 700.0)))
    if orientation < 0:
        q = 1.0 - q
        ell = -ell
        base_logit = -base_logit
        contributions = -contributions
        base_contributions = -base_contributions
        correction = -correction
        cell_delta = -cell_delta
        edge_raw_contribution = -edge_raw_contribution
    if config.mode == "alias":
        if baseline_score is None:
            raise ValueError("alias mode requires the saved baseline_score")
        saved = np.asarray(baseline_score, dtype=np.float64)
        if saved.shape != q.shape:
            raise ValueError("saved baseline score shape mismatch")
        alias_error = float(np.max(np.abs(saved - q)))
        if alias_error > 1e-12:
            raise ResidualGraphDeemError(
                f"saved B3 alias reconstruction failed: {alias_error:.3e}"
            )
        # The alias arm writes the frozen B3 bytes, not a recomputed sigmoid.
        q = saved.copy()
    else:
        alias_error = None

    anchor = equal_family_risk_anchor(X, names)
    high = float(np.sum(q * anchor) / max(np.sum(q), EPS))
    low = float(np.sum((1.0 - q) * anchor) / max(np.sum(1.0 - q), EPS))
    difference = high - low

    aligned_bias = float(orientation * model.base.b.detach().item())
    aligned_reconstruction = float(
        np.max(np.abs(aligned_bias + contributions.sum(axis=1) - ell))
    )
    aligned_residual_identity = float(np.max(np.abs(ell - base_logit - correction)))
    posterior_sd = float(np.std(q))
    finite = bool(
        np.isfinite(q).all()
        and np.isfinite(correction).all()
        and all(np.isfinite(row["loss"]) for row in history)
    )
    acceptance = [row["mala_acceptance"] for row in history]
    acceptance_healthy = bool(
        not acceptance
        or (
            all(0.0 <= float(value) <= 1.0 for value in acceptance)
            and float(np.mean(acceptance)) > 0.0
        )
    )
    diagnostics = {
        "correction_mean": float(np.mean(correction)),
        "correction_sd": float(np.std(correction)),
        "correction_max_abs": float(np.max(np.abs(correction))),
        "correction_bound": float(config.strength) * float(config.correction_cap),
        "correction_saturation_fraction_abs_gt_0p45": float(
            np.mean(np.abs(correction) > 0.45)
        ),
        "edge_raw_contribution_sd": float(np.std(edge_raw_contribution)),
        "edge_value_mean_max_abs": float(np.max(np.abs(edge_values.mean(axis=0)))),
        "edge_weight_l2": float(np.sqrt(np.sum(state["rook::edge_weight"] ** 2))),
        "edge_weight_nonzero": int(np.count_nonzero(state["rook::edge_weight"])),
        "gradient_norm_initial": float(history[0]["gradient_norm_preclip"]) if history else None,
        "gradient_norm_final": float(history[-1]["gradient_norm_preclip"]) if history else None,
        "n_edges": len(model.edge_pairs),
        "coordinate_source": "frozen_b3_atomic_core_centered_scaled_tanh",
        "length_residualization": "none_v1_unconditional_ebm",
        "orientation_policy": "fixed_same_seed_frozen_b3",
        "saved_alias_max_abs": alias_error,
        "base_logit_pearson": float(np.corrcoef(ell, base_logit)[0, 1]),
        "residual_identity_max_abs": aligned_residual_identity,
    }
    healthy = bool(
        finite
        and acceptance_healthy
        and posterior_sd >= float(config.posterior_sd_min)
        and aligned_reconstruction <= 1e-8
        and aligned_residual_identity <= 1e-8
        and difference > float(config.anchor_tolerance)
        and diagnostics["correction_max_abs"]
        <= diagnostics["correction_bound"] + 1e-10
    )
    return CrossedRookResult(
        score=q.copy(),
        posterior=np.column_stack([1.0 - q, q]),
        logit=ell,
        base_logit=base_logit,
        contributions=contributions,
        base_contributions=base_contributions,
        correction=correction,
        cell_delta=cell_delta,
        edge_raw_contribution=edge_raw_contribution,
        edge_values=edge_values,
        edge_pairs=model.edge_pairs,
        edge_kinds=model.edge_kinds,
        edge_weights=state["rook::edge_weight"].copy(),
        core_features=CORE_FEATURES,
        core_indices=model.core_indices,
        aligned_bias=aligned_bias,
        orientation=orientation,
        risk_anchor_difference=float(difference),
        feature_names=names,
        state=state,
        objective_history=history,
        health={
            "healthy": healthy,
            "finite": finite,
            "posterior_sd": posterior_sd,
            "contribution_reconstruction_max_abs": aligned_reconstruction,
            "residual_identity_max_abs": aligned_residual_identity,
            "epochs_completed": len(history),
            "mala_acceptance_mean": (
                float(np.mean(acceptance)) if acceptance else None
            ),
            "runtime_seconds": float(time.perf_counter() - started),
        },
        diagnostics=diagnostics,
        config=jsonable(asdict(config)),
        seed=int(seed),
    )


def predict_deem_b3_crossed_rook(
    result: CrossedRookResult,
    X_risk: np.ndarray,
) -> dict[str, Any]:
    """Replay a fitted crossed-rook result on a matrix with the same inventory."""

    import torch

    X = validate_inventory(X_risk, result.feature_names)
    config = CrossedRookConfig(**result.config)
    model = _CrossedRookEnergy(result.feature_names, config, result.seed)
    model.load_state_numpy(result.state)
    with torch.no_grad():
        ell_t, values = model.logit(torch.as_tensor(X, dtype=torch.float64))
        base_logit_t, base_atomic_t, _ = model.base.logit(
            torch.as_tensor(X, dtype=torch.float64)
        )
    orientation = int(result.orientation)
    ell = orientation * ell_t.cpu().numpy()
    base_logit = orientation * base_logit_t.cpu().numpy()
    contribution = orientation * values["contributions"].cpu().numpy()
    base_contribution = orientation * base_atomic_t.cpu().numpy()
    correction = orientation * values["correction"].cpu().numpy()
    cell_delta = orientation * values["cell_delta"].cpu().numpy()
    edge_raw_contribution = (
        orientation * values["edge_raw_contribution"].cpu().numpy()
    )
    score = 1.0 / (1.0 + np.exp(-np.clip(ell, -700.0, 700.0)))
    if config.mode == "alias":
        # ``result.score`` is the verbatim saved B3 vector bound at fit time.
        # Replay must reconstruct it; return the bound vector on exact-row
        # replay to preserve the explicit alias contract.
        if len(score) == len(result.score):
            alias_error = float(np.max(np.abs(score - result.score)))
            if alias_error <= 1e-12:
                score = result.score.copy()
    reconstruction = float(
        np.max(np.abs(result.aligned_bias + contribution.sum(axis=1) - ell))
    )
    residual_identity = float(np.max(np.abs(ell - base_logit - correction)))
    if reconstruction > 1e-8 or residual_identity > 1e-8:
        raise ResidualGraphDeemError("held crossed-rook decomposition failed")
    return {
        "score": score,
        "posterior": np.column_stack([1.0 - score, score]),
        "logit": ell,
        "base_logit": base_logit,
        "contributions": contribution,
        "base_contributions": base_contribution,
        "correction": correction,
        "cell_delta": cell_delta,
        "edge_values": values["edge_values"].cpu().numpy(),
        "edge_raw_contribution": edge_raw_contribution,
        "edge_pairs": np.asarray(model.edge_pairs, dtype=np.int64),
        "edge_kinds": np.asarray(model.edge_kinds, dtype=str),
        "edge_weights": model.edge_weight.detach().cpu().numpy().copy(),
        "reconstruction_max_abs": reconstruction,
        "residual_identity_max_abs": residual_identity,
    }


__all__ = [
    "CORE_FEATURES",
    "CORE_GRID",
    "MODES",
    "OPERATORS",
    "SOURCES",
    "CrossedRookConfig",
    "CrossedRookResult",
    "_CrossedRookEnergy",
    "core_index_grid",
    "fit_deem_b3_crossed_rook",
    "predict_deem_b3_crossed_rook",
]
