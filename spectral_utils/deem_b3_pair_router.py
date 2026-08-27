"""Pairwise antisymmetric residual router on a completely frozen B3 energy.

This is a new, label-free experiment rather than a modification of historical
B3.  For every unordered pair of provenance families ``(g, h)``, a transfer is
predicted only from the remaining families.  The family multipliers are

    gamma_g = 1 + rho / (G - 1) * sum_h t_gh,   t_hg = -t_gh.

Consequently their row sum is exactly ``G``, every multiplier lies in
``[1-rho, 1+rho]``, and ``rho=0`` is an exact B3 alias for arbitrary router
parameters.  Each pair context is a fixed conditional residual: a family in
the context is predicted from the other context families, never from either
member of the routed pair.  This makes ``d gamma_g / d s_g = 0`` structurally.

The router can be trained by the frozen B3 contrastive energy objective and,
optionally, by a fixed graph-roughness prior.  Gates are deterministic inside
the energy, so MALA continues to use a valid fixed Metropolis-Hastings target.
The sigmoid pair-survival head borrows the "close to identity" idea from STG;
it is not claimed to reproduce DUFS or GroupFS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from .residual_graph_deem import (
    ContinuousDeemConfig,
    EPS,
    ResidualGraphDeemError,
    _FamilyAdditiveEnergy,
    _sparse_quadratic,
    equal_family_risk_anchor,
    jsonable,
    persistent_mala,
    set_determinism,
    validate_inventory,
)


@dataclass(frozen=True)
class PairRouterConfig:
    rho: float = 0.25
    epochs: int = 100
    learning_rate: float = 2e-3
    deem_weight: float = 1.0
    graph_weight: float = 0.0
    trust_weight: float = 0.10
    open_weight: float = 0.01
    l2_weight: float = 1e-4
    open_warmup_epochs: int = 10
    grad_clip: float = 5.0
    mala_delta: float = 0.10
    mala_steps: int = 5
    replay_refresh: float = 0.05
    family_width: int = 8
    base_init_sd: float = 0.005
    posterior_sd_min: float = 1e-3
    dtype: str = "float64"
    device: str = "cpu"
    deterministic: bool = True


@dataclass
class PairRouterResult:
    score: np.ndarray
    logit: np.ndarray
    contributions: np.ndarray
    base_family_contributions: np.ndarray
    routed_family_contributions: np.ndarray
    gates: np.ndarray
    family_probabilities: np.ndarray
    pair_transfers: np.ndarray
    pair_open_probabilities: np.ndarray
    pair_context_residuals: np.ndarray
    feature_names: tuple[str, ...]
    family_order: tuple[str, ...]
    pair_order: tuple[tuple[str, str], ...]
    aligned_bias: float
    orientation: int
    state: dict[str, np.ndarray]
    objective_history: list[dict[str, float]]
    health: dict[str, Any]
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    seed: int


def _validate_config(config: PairRouterConfig) -> None:
    if not 0.0 <= float(config.rho) <= 1.0:
        raise ValueError("rho must be in [0,1]")
    if int(config.epochs) < 0 or int(config.mala_steps) < 1:
        raise ValueError("epochs must be nonnegative and MALA steps positive")
    if float(config.learning_rate) <= 0.0 or float(config.grad_clip) <= 0.0:
        raise ValueError("learning rate and gradient clip must be positive")
    for name in (
        "deem_weight",
        "graph_weight",
        "trust_weight",
        "open_weight",
        "l2_weight",
    ):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if not 0.0 <= float(config.replay_refresh) <= 1.0:
        raise ValueError("replay_refresh must be in [0,1]")
    if int(config.open_warmup_epochs) < 0:
        raise ValueError("open_warmup_epochs must be nonnegative")
    if config.dtype != "float64" or config.device != "cpu":
        raise ValueError("pair router v1 is frozen to float64 CPU")


class _PairResidualRouterEnergy:
    """Frozen B3 experts with pairwise, self-free family transfers."""

    def __init__(
        self,
        feature_names: Sequence[str],
        config: PairRouterConfig,
        seed: int,
    ) -> None:
        import torch

        _validate_config(config)
        self.torch = torch
        self.names = tuple(str(value) for value in feature_names)
        self.config = config
        self.seed = int(seed)
        self.base = _FamilyAdditiveEnergy(
            self.names,
            ContinuousDeemConfig(
                family_width=int(config.family_width),
                init_sd=float(config.base_init_sd),
            ),
            seed,
        )
        self.groups = self.base.groups
        self.family_order = tuple(self.groups)
        self.family_count = len(self.family_order)
        if self.family_count < 4:
            raise ValueError("pairwise LOO2 routing requires at least four families")
        self.pair_indices = tuple(itertools.combinations(range(self.family_count), 2))
        self.pair_order = tuple(
            (self.family_order[left], self.family_order[right])
            for left, right in self.pair_indices
        )
        pair_count = len(self.pair_indices)
        dtype = torch.float64
        context_mask = torch.ones((pair_count, self.family_count), dtype=dtype)
        incidence = torch.zeros((pair_count, self.family_count), dtype=dtype)
        for pair_index, (left, right) in enumerate(self.pair_indices):
            context_mask[pair_index, left] = 0.0
            context_mask[pair_index, right] = 0.0
            incidence[pair_index, left] = 1.0
            incidence[pair_index, right] = -1.0
        self.context_mask = context_mask.detach()
        self.incidence = incidence.detach()
        self.pair_weight = torch.nn.Parameter(
            torch.zeros((pair_count, self.family_count), dtype=dtype)
        )
        self.pair_open_logit = torch.nn.Parameter(torch.zeros(pair_count, dtype=dtype))
        self.context_center = torch.zeros(self.family_count, dtype=dtype)
        self.context_scale = torch.ones(self.family_count, dtype=dtype)
        self.context_coefficient = torch.zeros(
            (pair_count, self.family_count, self.family_count), dtype=dtype
        )
        self.context_intercept = torch.zeros(
            (pair_count, self.family_count), dtype=dtype
        )
        self.context_calibrated = False
        self.open_override: float | None = None

    def load_baseline_state(self, state: Mapping[str, np.ndarray]) -> None:
        self.base.load_state_numpy(state)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def parameters(self):
        return [self.pair_weight, self.pair_open_logit]

    def _family_matrix(self, family: Mapping[str, Any]):
        return self.torch.stack(
            [family[name] for name in self.family_order], dim=1
        )

    def calibrate_context(self, X) -> dict[str, float]:
        """Fit fixed pair-specific conditional residualizers without labels."""

        with self.torch.no_grad():
            _, family = self.base.contributions(X)
            values = self._family_matrix(family).cpu().numpy()
        center = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale > EPS, scale, 1.0)
        standardized = (values - center[None, :]) / scale[None, :]
        coefficient = np.zeros(
            (len(self.pair_indices), self.family_count, self.family_count),
            dtype=float,
        )
        intercept = np.zeros((len(self.pair_indices), self.family_count), dtype=float)
        residual = np.zeros(
            (len(values), len(self.pair_indices), self.family_count), dtype=float
        )
        for pair_index, (left, right) in enumerate(self.pair_indices):
            context = [
                index
                for index in range(self.family_count)
                if index not in {left, right}
            ]
            for target in context:
                predictors = [index for index in context if index != target]
                estimator = Ridge(alpha=1.0, fit_intercept=True)
                estimator.fit(standardized[:, predictors], standardized[:, target])
                coefficient[pair_index, target, predictors] = estimator.coef_
                intercept[pair_index, target] = float(estimator.intercept_)
                residual[:, pair_index, target] = (
                    standardized[:, target]
                    - estimator.predict(standardized[:, predictors])
                )
        self.context_center = self.torch.as_tensor(center, dtype=self.torch.float64)
        self.context_scale = self.torch.as_tensor(scale, dtype=self.torch.float64)
        self.context_coefficient = self.torch.as_tensor(
            coefficient, dtype=self.torch.float64
        )
        self.context_intercept = self.torch.as_tensor(
            intercept, dtype=self.torch.float64
        )
        self.context_calibrated = True
        active = self.context_mask.cpu().numpy().astype(bool)
        return {
            "context_residual_sd": float(np.std(residual[:, active])),
            "context_residual_mean_abs": float(np.mean(np.abs(residual[:, active]))),
            "n_pair_context_coordinates": int(np.sum(active)),
        }

    def _context_residual(self, family_values):
        if not self.context_calibrated:
            raise RuntimeError("pair context has not been calibrated")
        standardized = (
            family_values - self.context_center[None, :]
        ) / self.context_scale[None, :]
        prediction = self.torch.einsum(
            "ng,ekg->nek", standardized, self.context_coefficient
        ) + self.context_intercept[None, :, :]
        residual = (
            standardized[:, None, :] - prediction
        ) * self.context_mask[None, :, :]
        return residual

    def routing_from_family_values(self, family_values):
        """Return residual contexts, antisymmetric transfers, and gates."""

        context_residual = self._context_residual(family_values)
        masked_weight = self.pair_weight * self.context_mask
        raw_pair = (
            (self.torch.tanh(context_residual) * masked_weight[None, :, :]).sum(dim=2)
            / math.sqrt(self.family_count - 2)
        )
        if self.open_override is None:
            open_probability = self.torch.sigmoid(self.pair_open_logit)
        else:
            open_probability = self.torch.full_like(
                self.pair_open_logit, float(self.open_override)
            )
        pair_transfer = open_probability[None, :] * self.torch.tanh(raw_pair)
        family_transfer = pair_transfer @ self.incidence
        gates = 1.0 + float(self.config.rho) * family_transfer / (
            self.family_count - 1
        )
        family_probabilities = gates / self.family_count
        return {
            "context_residual": context_residual,
            "pair_transfer": pair_transfer,
            "pair_open_probability": open_probability,
            "family_transfer": family_transfer,
            "gates": gates,
            "family_probabilities": family_probabilities,
        }

    def components(self, X):
        base_atomic, base_family = self.base.contributions(X)
        family_values = self._family_matrix(base_family)
        routing = self.routing_from_family_values(family_values)
        gates = routing["gates"]
        routed_atomic = base_atomic.clone()
        routed_family = []
        for column, family_name in enumerate(self.family_order):
            indices = self.groups[family_name]
            routed_atomic[:, list(indices)] = (
                routed_atomic[:, list(indices)] * gates[:, column : column + 1]
            )
            routed_family.append(family_values[:, column] * gates[:, column])
        return {
            "base_atomic": base_atomic,
            "base_family": family_values,
            "routed_atomic": routed_atomic,
            "routed_family": self.torch.stack(routed_family, dim=1),
            **routing,
        }

    def logit(self, X):
        values = self.components(X)
        return self.base.b + values["routed_atomic"].sum(dim=1), values

    def free_energy(self, X):
        ell, _ = self.logit(X)
        return 0.5 * ((X - self.base.a) ** 2).sum(dim=1) - self.torch.nn.functional.softplus(
            ell
        )

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        output = {
            f"base::{name}": value
            for name, value in self.base.state_dict_numpy().items()
        }
        output.update(
            {
                "router::pair_weight": self.pair_weight.detach().cpu().numpy().copy(),
                "router::pair_open_logit": self.pair_open_logit.detach().cpu().numpy().copy(),
                "router::context_center": self.context_center.detach().cpu().numpy().copy(),
                "router::context_scale": self.context_scale.detach().cpu().numpy().copy(),
                "router::context_coefficient": self.context_coefficient.detach().cpu().numpy().copy(),
                "router::context_intercept": self.context_intercept.detach().cpu().numpy().copy(),
            }
        )
        return output

    def load_state_numpy(self, state: Mapping[str, np.ndarray]) -> None:
        base_state = {
            name.removeprefix("base::"): value
            for name, value in state.items()
            if name.startswith("base::")
        }
        self.load_baseline_state(base_state)
        with self.torch.no_grad():
            self.pair_weight.copy_(
                self.torch.as_tensor(state["router::pair_weight"], dtype=self.torch.float64)
            )
            self.pair_open_logit.copy_(
                self.torch.as_tensor(
                    state["router::pair_open_logit"], dtype=self.torch.float64
                )
            )
        self.context_center = self.torch.as_tensor(
            state["router::context_center"], dtype=self.torch.float64
        )
        self.context_scale = self.torch.as_tensor(
            state["router::context_scale"], dtype=self.torch.float64
        )
        self.context_coefficient = self.torch.as_tensor(
            state["router::context_coefficient"], dtype=self.torch.float64
        )
        self.context_intercept = self.torch.as_tensor(
            state["router::context_intercept"], dtype=self.torch.float64
        )
        self.context_calibrated = True


def fit_pair_residual_router(
    X_risk: np.ndarray,
    feature_names: Sequence[str],
    baseline_state: Mapping[str, np.ndarray],
    *,
    baseline_score: np.ndarray,
    baseline_orientation: int,
    seed: int = 0,
    config: PairRouterConfig | None = None,
    laplacian=None,
) -> PairRouterResult:
    """Fit a deterministic label-free pair router over frozen B3 experts."""

    import torch

    started = time.perf_counter()
    config = config or PairRouterConfig()
    _validate_config(config)
    if int(baseline_orientation) not in {-1, 1}:
        raise ValueError("baseline_orientation must be +/-1")
    X = validate_inventory(X_risk, feature_names)
    names = tuple(str(value) for value in feature_names)
    frozen_baseline_score = np.asarray(baseline_score, dtype=np.float64)
    if (
        frozen_baseline_score.shape != (len(X),)
        or not np.isfinite(frozen_baseline_score).all()
        or np.any((frozen_baseline_score < 0.0) | (frozen_baseline_score > 1.0))
    ):
        raise ValueError("baseline_score must be an aligned finite probability vector")
    if float(config.graph_weight) > 0.0:
        if laplacian is None or tuple(laplacian.shape) != (len(X), len(X)):
            raise ValueError("graph_weight requires an aligned fixed Laplacian")
        try:
            from scipy import sparse

            if sparse.issparse(laplacian):
                if not np.isfinite(laplacian.data).all():
                    raise ValueError("Laplacian must be finite")
                difference = laplacian - laplacian.T
                asymmetry = (
                    float(np.max(np.abs(difference.data)))
                    if difference.nnz
                    else 0.0
                )
            else:
                dense_laplacian = np.asarray(laplacian, dtype=np.float64)
                if not np.isfinite(dense_laplacian).all():
                    raise ValueError("Laplacian must be finite")
                asymmetry = float(
                    np.max(np.abs(dense_laplacian - dense_laplacian.T))
                )
        except ImportError as exc:  # pragma: no cover - scipy is a pinned dependency
            raise RuntimeError("scipy is required for graph-weighted pair routing") from exc
        if asymmetry > 1e-10:
            raise ValueError(f"Laplacian must be symmetric; error={asymmetry:.3e}")
    set_determinism(seed)
    model = _PairResidualRouterEnergy(names, config, seed)
    model.load_baseline_state(baseline_state)
    tensor = torch.as_tensor(X, dtype=torch.float64)
    calibration_diagnostics = model.calibrate_context(tensor)
    with torch.no_grad():
        base_ell, _, _ = model.base.logit(tensor)
        base_ell = base_ell.detach()
    orientation = int(baseline_orientation)
    reconstructed_baseline_score = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                orientation * base_ell.cpu().numpy(),
                -700.0,
                700.0,
            )
        )
    )
    baseline_state_identity_error = float(
        np.max(np.abs(reconstructed_baseline_score - frozen_baseline_score))
    )
    if baseline_state_identity_error > 1e-12:
        raise ResidualGraphDeemError(
            "frozen B3 score/state mismatch: "
            f"{baseline_state_identity_error:.3e}"
        )
    base_state_before = {
        name: np.asarray(value).copy() for name, value in model.base.state_dict_numpy().items()
    }
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate))
    buffer = tensor.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 8_700_013)
    history: list[dict[str, float]] = []
    last_finite_state = model.state_dict_numpy()
    try:
        for epoch in range(int(config.epochs)):
            model.open_override = (
                0.5 if epoch < int(config.open_warmup_epochs) else None
            )
            if float(config.deem_weight) > 0.0:
                refresh = torch.rand(len(X), generator=generator) < float(
                    config.replay_refresh
                )
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
                deem_loss = model.free_energy(tensor).mean() - model.free_energy(
                    buffer
                ).mean()
            else:
                acceptance = 0.0
                deem_loss = torch.zeros((), dtype=torch.float64)
            ell, values = model.logit(tensor)
            centered = ell - ell.mean()
            graph_raw = torch.zeros((), dtype=torch.float64)
            if float(config.graph_weight) > 0.0:
                graph_raw = _sparse_quadratic(centered, laplacian) / (
                    centered.square().sum() + EPS
                )
            delta = ell - base_ell
            trust_raw = delta.square().mean() / (base_ell.var(unbiased=False) + EPS)
            warmup = epoch < int(config.open_warmup_epochs)
            open_raw = (
                torch.zeros((), dtype=torch.float64)
                if warmup
                else torch.sigmoid(model.pair_open_logit).mean()
            )
            l2_raw = (model.pair_weight * model.context_mask).square().mean()
            if not warmup:
                l2_raw = l2_raw + model.pair_open_logit.square().mean()
            loss = (
                float(config.deem_weight) * deem_loss
                + float(config.graph_weight) * graph_raw
                + float(config.trust_weight) * trust_raw
                + float(config.open_weight) * open_raw
                + float(config.l2_weight) * l2_raw
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite pair-router loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            for index, parameter in enumerate(parameters):
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError(
                        f"non-finite pair-router gradient {index} at epoch {epoch}"
                    )
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip))
            optimizer.step()
            for index, parameter in enumerate(parameters):
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(
                        f"non-finite pair-router parameter {index} at epoch {epoch}"
                    )
            last_finite_state = model.state_dict_numpy()
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach()),
                    "deem_loss": float(deem_loss.detach()),
                    "graph_raw": float(graph_raw.detach()),
                    "trust_raw": float(trust_raw.detach()),
                    "open_raw": float(open_raw.detach()),
                    "l2_raw": float(l2_raw.detach()),
                    "grad_norm_before_clip": grad_norm,
                    "mala_acceptance": float(acceptance),
                }
            )
    except Exception as exc:
        setattr(exc, "objective_history", history)
        setattr(exc, "last_finite_state", last_finite_state)
        raise
    finally:
        model.open_override = None

    state = model.state_dict_numpy()
    with torch.no_grad():
        raw_ell, values = model.logit(tensor)
        raw_ell = raw_ell.cpu().numpy()
        raw_atomic = values["routed_atomic"].cpu().numpy()
        base_family = values["base_family"].cpu().numpy()
        routed_family = values["routed_family"].cpu().numpy()
        gates = values["gates"].cpu().numpy()
        probabilities = values["family_probabilities"].cpu().numpy()
        transfers = values["pair_transfer"].cpu().numpy()
        pair_open = values["pair_open_probability"].cpu().numpy()
        context_residual = values["context_residual"].cpu().numpy()

    ell = orientation * raw_ell
    contributions = orientation * raw_atomic
    base_family = orientation * base_family
    routed_family = orientation * routed_family
    computed_score = 1.0 / (1.0 + np.exp(-np.clip(ell, -700.0, 700.0)))
    # The identity arm is an artifact-level control, not merely a numerically
    # equivalent reconstruction.  Preserve the frozen bytes verbatim.
    score = (
        frozen_baseline_score.copy()
        if float(config.rho) == 0.0
        else computed_score
    )
    aligned_bias = float(orientation * model.base.b.detach().item())
    reconstruction = float(
        np.max(np.abs(aligned_bias + contributions.sum(axis=1) - ell))
    )
    gate_sum_error = float(
        np.max(np.abs(gates.sum(axis=1) - model.family_count))
    )
    lower, upper = 1.0 - float(config.rho), 1.0 + float(config.rho)
    gate_bound_error = float(
        max(np.max(lower - gates), np.max(gates - upper), 0.0)
    )
    base_frozen = max(
        float(np.max(np.abs(model.base.state_dict_numpy()[name] - value)))
        for name, value in base_state_before.items()
    )
    with torch.no_grad():
        base_raw, _, _ = model.base.logit(tensor)
    alias_error = float(np.max(np.abs(score - frozen_baseline_score)))
    anchor = equal_family_risk_anchor(X, names)
    high = float(np.sum(score * anchor) / max(np.sum(score), EPS))
    low = float(
        np.sum((1.0 - score) * anchor) / max(np.sum(1.0 - score), EPS)
    )
    posterior_sd = float(np.std(score))
    healthy = bool(
        np.isfinite(score).all()
        and np.isfinite(gates).all()
        and posterior_sd >= float(config.posterior_sd_min)
        and reconstruction <= 1e-8
        and gate_sum_error <= 1e-10
        and gate_bound_error <= 1e-10
        and base_frozen == 0.0
        and baseline_state_identity_error <= 1e-12
        and (float(config.rho) != 0.0 or alias_error <= 1e-12)
        and all(np.isfinite(row["loss"]) for row in history)
    )
    diagnostics = {
        **calibration_diagnostics,
        "gate_sum_max_abs_error": gate_sum_error,
        "gate_bound_max_violation": gate_bound_error,
        "gate_min": float(np.min(gates)),
        "gate_max": float(np.max(gates)),
        "gate_mean_abs_deviation_from_one": float(np.mean(np.abs(gates - 1.0))),
        "gate_family_mean": {
            name: float(np.mean(gates[:, column]))
            for column, name in enumerate(model.family_order)
        },
        "gate_family_sd": {
            name: float(np.std(gates[:, column]))
            for column, name in enumerate(model.family_order)
        },
        "pair_transfer_mean_abs": float(np.mean(np.abs(transfers))),
        "pair_transfer_max_abs": float(np.max(np.abs(transfers))),
        "pair_open_mean": float(np.mean(pair_open)),
        "router_delta_logit_sd": float(np.std(ell - orientation * base_raw.cpu().numpy())),
        "baseline_alias_max_abs": alias_error,
        "baseline_state_identity_max_abs": baseline_state_identity_error,
        "fixed_orientation_anchor_difference": high - low,
        "base_state_max_abs_change": base_frozen,
        "contribution_reconstruction_max_abs": reconstruction,
        "uses_labels": False,
    }
    return PairRouterResult(
        score=score,
        logit=ell,
        contributions=contributions,
        base_family_contributions=base_family,
        routed_family_contributions=routed_family,
        gates=gates,
        family_probabilities=probabilities,
        pair_transfers=transfers,
        pair_open_probabilities=pair_open,
        pair_context_residuals=context_residual,
        feature_names=names,
        family_order=model.family_order,
        pair_order=model.pair_order,
        aligned_bias=aligned_bias,
        orientation=orientation,
        state=state,
        objective_history=history,
        health={
            "healthy": healthy,
            "finite": bool(np.isfinite(score).all()),
            "posterior_sd": posterior_sd,
            "epochs_completed": len(history),
            "mala_acceptance_mean": float(
                np.mean([row["mala_acceptance"] for row in history])
            ) if history else 0.0,
            "runtime_seconds": float(time.perf_counter() - started),
            "contribution_reconstruction_max_abs": reconstruction,
        },
        diagnostics=diagnostics,
        config=jsonable(asdict(config)),
        seed=int(seed),
    )


def predict_pair_residual_router(
    result: PairRouterResult,
    X_risk: np.ndarray,
    *,
    baseline_score: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Replay a frozen router on aligned rows without refitting its context."""

    import torch

    X = validate_inventory(X_risk, result.feature_names)
    config = PairRouterConfig(**result.config)
    model = _PairResidualRouterEnergy(result.feature_names, config, result.seed)
    model.load_state_numpy(result.state)
    tensor = torch.as_tensor(X, dtype=torch.float64)
    with torch.no_grad():
        raw_ell, values = model.logit(tensor)
    orientation = int(result.orientation)
    ell = orientation * raw_ell.cpu().numpy()
    contributions = orientation * values["routed_atomic"].cpu().numpy()
    base_family = orientation * values["base_family"].cpu().numpy()
    routed_family = orientation * values["routed_family"].cpu().numpy()
    score = 1.0 / (1.0 + np.exp(-np.clip(ell, -700.0, 700.0)))
    if baseline_score is not None:
        frozen = np.asarray(baseline_score, dtype=np.float64)
        if frozen.shape != (len(X),) or not np.isfinite(frozen).all():
            raise ValueError("baseline_score must align with replay rows")
        with torch.no_grad():
            base_raw, _, _ = model.base.logit(tensor)
        reconstructed = 1.0 / (
            1.0
            + np.exp(
                -np.clip(orientation * base_raw.cpu().numpy(), -700.0, 700.0)
            )
        )
        error = float(np.max(np.abs(reconstructed - frozen)))
        if error > 1e-12:
            raise ResidualGraphDeemError(
                f"held frozen B3 score/state mismatch: {error:.3e}"
            )
        if float(config.rho) == 0.0:
            score = frozen.copy()
    reconstruction = float(
        np.max(np.abs(result.aligned_bias + contributions.sum(axis=1) - ell))
    )
    if reconstruction > 1e-8:
        raise ResidualGraphDeemError(
            f"held pair-router contribution reconstruction failed: {reconstruction:.3e}"
        )
    return {
        "score": score,
        "logit": ell,
        "contributions": contributions,
        "base_family_contributions": base_family,
        "routed_family_contributions": routed_family,
        "gates": values["gates"].cpu().numpy(),
        "family_probabilities": values["family_probabilities"].cpu().numpy(),
        "pair_transfers": values["pair_transfer"].cpu().numpy(),
        "pair_open_probabilities": values["pair_open_probability"].cpu().numpy(),
        "pair_context_residuals": values["context_residual"].cpu().numpy(),
        "reconstruction_max_abs": reconstruction,
    }


__all__ = [
    "PairRouterConfig",
    "PairRouterResult",
    "_PairResidualRouterEnergy",
    "fit_pair_residual_router",
    "predict_pair_residual_router",
]
