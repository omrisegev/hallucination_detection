"""Label-free mixture-of-experts extensions of the frozen continuous DEEM B3.

This module is deliberately additive: it does not change the historical B3
implementation or artifact contract.  A fitted B3 family block is treated as
an expert and a small deterministic router learns bounded, sample-dependent
family multipliers under the same contrastive free-energy objective.

The key identity is

    gamma_g(x) = (1 - alpha) + alpha * G * pi_g(x),

where ``pi`` is a simplex-valued router.  Consequently ``alpha == 0`` is
exactly B3 and ``sum_g gamma_g == G`` for every row.  The bounded multipliers
preserve the Gaussian-tail normalizability of the original energy.
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
ROUTERS = (
    "uniform",
    "static",
    "loo_scalar",
    "loo_shared",
    "self_scalar",
    "dense_loo",
    "dense_full",
    "global_residual",
    "pairwise_residual",
    "multinomial_diagonal",
    "multinomial_offdiag",
    "multinomial_full",
)
NORMALIZERS = ("softmax", "sparsemax")


@dataclass(frozen=True)
class DeemMoEConfig:
    """Configuration for a warm-started B3 family router.

    ``train_experts=False`` is the primary isolation arm: the historical B3
    parameters (including its Gaussian location and bias) stay frozen and only
    the router is optimized.  ``train_experts=True`` is an explicit joint
    fine-tuning ablation.
    """

    router: str = "loo_scalar"
    normalizer: str = "softmax"
    router_strength: float = 0.50
    temperature: float = 1.0
    epochs: int = 40
    learning_rate: float = 5e-3
    momentum: float = 0.0
    mala_delta: float = 0.10
    mala_steps: int = 5
    replay_refresh: float = 0.05
    balance_penalty: float = 0.0
    router_l2: float = 1e-4
    router_init_sd: float = 0.0
    train_experts: bool = False
    train_router_bias: bool = True
    expert_learning_rate: float | None = None
    family_width: int = 8
    base_init_sd: float = 0.005
    anchor_tolerance: float = 1e-6
    posterior_sd_min: float = 1e-3
    dtype: str = "float64"
    device: str = "cpu"
    deterministic: bool = True


@dataclass
class DeemMoEResult:
    score: np.ndarray
    posterior: np.ndarray
    logit: np.ndarray
    contributions: np.ndarray
    family_contributions: dict[str, np.ndarray]
    base_family_contributions: dict[str, np.ndarray]
    gates: np.ndarray
    router_probabilities: np.ndarray
    router_logits: np.ndarray
    interaction: np.ndarray
    family_state_input: np.ndarray
    family_state_output: np.ndarray
    family_state_delta: np.ndarray
    family_order: tuple[str, ...]
    aligned_bias: float
    orientation: int
    risk_anchor_difference: float
    feature_names: tuple[str, ...]
    family_indices: dict[str, tuple[int, ...]]
    state: dict[str, np.ndarray]
    objective_history: list[dict[str, float]]
    health: dict[str, Any]
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    seed: int


def _validate_config(config: DeemMoEConfig) -> None:
    if config.router not in ROUTERS:
        raise ValueError(f"unknown router {config.router!r}; expected one of {ROUTERS}")
    if config.normalizer not in NORMALIZERS:
        raise ValueError(
            f"unknown router normalizer {config.normalizer!r}; expected one of {NORMALIZERS}"
        )
    if not 0.0 <= float(config.router_strength) <= 1.0:
        raise ValueError("router_strength must be in [0, 1]")
    if float(config.temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if int(config.epochs) < 0 or int(config.mala_steps) < 1:
        raise ValueError("epochs must be nonnegative and mala_steps positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.expert_learning_rate is not None and float(config.expert_learning_rate) <= 0.0:
        raise ValueError("expert_learning_rate must be positive when provided")
    if not 0.0 <= float(config.replay_refresh) <= 1.0:
        raise ValueError("replay_refresh must be in [0, 1]")
    if (
        float(config.balance_penalty) < 0.0
        or float(config.router_l2) < 0.0
        or float(config.router_init_sd) < 0.0
    ):
        raise ValueError("router penalties must be nonnegative")
    if config.dtype != "float64" or config.device != "cpu":
        raise ValueError("DEEM-MoE v1 is frozen to float64 CPU")


def sparsemax(logits, *, dim: int = -1):
    """Sparsemax projection onto the probability simplex.

    The implementation is deterministic and differentiable almost everywhere.
    It intentionally has no external ``entmax`` dependency so the new arm can
    run in the same local environment as continuous B3.
    """

    torch = __import__("torch")
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    ordered = torch.sort(shifted, dim=dim, descending=True).values
    cumulative = ordered.cumsum(dim) - 1.0
    size = ordered.shape[dim]
    shape = [1] * ordered.ndim
    shape[dim] = size
    ranks = torch.arange(
        1, size + 1, dtype=ordered.dtype, device=ordered.device
    ).reshape(shape)
    support = ranks * ordered > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold_index = (support_size - 1).to(torch.long)
    threshold = cumulative.gather(dim, threshold_index) / support_size.to(ordered.dtype)
    return torch.clamp(shifted - threshold, min=0.0)


class _FamilyRouterEnergy:
    """B3 family experts plus a bounded, identity-nested router."""

    def __init__(self, feature_names: Sequence[str], config: DeemMoEConfig, seed: int):
        import torch

        _validate_config(config)
        self.torch = torch
        self.names = tuple(str(name) for name in feature_names)
        self.config = config
        self.seed = int(seed)
        base_config = ContinuousDeemConfig(
            family_width=int(config.family_width),
            init_sd=float(config.base_init_sd),
        )
        self.base = _FamilyAdditiveEnergy(self.names, base_config, self.seed)
        self.groups = self.base.groups
        self.family_order = tuple(self.groups)
        count = len(self.family_order)
        dtype = torch.float64

        # Zero initialization makes the initial router exactly uniform.  Unlike
        # a two-layer all-zero network, every implemented parameter has a
        # nonzero first-order path from the objective at this initialization.
        self.router_bias = torch.nn.Parameter(torch.zeros(count, dtype=dtype))
        self.router_weight = torch.nn.Parameter(torch.zeros(count, dtype=dtype))
        self.router_scale = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.router_matrix = torch.nn.Parameter(torch.zeros((count, count), dtype=dtype))
        self.multinomial_matrix = torch.nn.Parameter(
            torch.zeros((count, 2, count, 2), dtype=dtype)
        )
        self.multinomial_bias = torch.nn.Parameter(torch.zeros((count, 2), dtype=dtype))
        self._off_diagonal = (1.0 - torch.eye(count, dtype=dtype)).detach()
        if float(config.router_init_sd) > 0.0:
            generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7_700_031)
            with torch.no_grad():
                self.multinomial_matrix.copy_(
                    float(config.router_init_sd)
                    * torch.randn(
                        self.multinomial_matrix.shape,
                        dtype=dtype,
                        generator=generator,
                    )
                )
                if config.train_router_bias:
                    self.multinomial_bias.copy_(
                        float(config.router_init_sd)
                        * torch.randn(
                            self.multinomial_bias.shape,
                            dtype=dtype,
                            generator=generator,
                        )
                    )

    def load_baseline_state(self, state: Mapping[str, np.ndarray]) -> None:
        self.base.load_state_numpy(state)

    def router_parameters(self):
        values = []
        if self.config.router == "static":
            if not self.config.train_router_bias:
                raise ValueError("static router requires train_router_bias=True")
            values.append(self.router_bias)
        elif self.config.router in {"loo_scalar", "self_scalar"}:
            if self.config.train_router_bias:
                values.append(self.router_bias)
            values.append(self.router_weight)
        elif self.config.router == "loo_shared":
            if self.config.train_router_bias:
                values.append(self.router_bias)
            values.append(self.router_scale)
        elif self.config.router in {"dense_loo", "dense_full"}:
            if self.config.train_router_bias:
                values.append(self.router_bias)
            values.append(self.router_matrix)
        elif self.config.router == "global_residual":
            values.append(self.router_weight)
        elif self.config.router == "pairwise_residual":
            values.append(self.router_matrix)
        elif self.config.router in {
            "multinomial_diagonal",
            "multinomial_offdiag",
            "multinomial_full",
        }:
            values.append(self.multinomial_matrix)
            if self.config.train_router_bias:
                values.append(self.multinomial_bias)
        elif self.config.router != "uniform":
            raise AssertionError(f"unhandled router {self.config.router}")
        return values

    def parameters(self, *, train_experts: bool | None = None):
        train_base = self.config.train_experts if train_experts is None else bool(train_experts)
        values = []
        if train_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(True)
                values.append(parameter)
        else:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

        values.extend(self.router_parameters())
        return values

    def _family_matrix(self, family: Mapping[str, Any]):
        return self.torch.stack([family[name] for name in self.family_order], dim=1)

    def router_logits_from_family(self, family_values):
        torch = self.torch
        count = family_values.shape[1]
        if self.config.router == "uniform":
            raw = torch.zeros_like(family_values)
        elif self.config.router == "static":
            raw = self.router_bias[None, :].expand_as(family_values)
        elif self.config.router == "loo_scalar":
            scale = math.sqrt(max(count - 1, 1))
            other = (family_values.sum(dim=1, keepdim=True) - family_values) / scale
            raw = self.router_bias[None, :] + self.router_weight[None, :] * torch.tanh(other)
        elif self.config.router == "loo_shared":
            scale = math.sqrt(max(count - 1, 1))
            other = (family_values.sum(dim=1, keepdim=True) - family_values) / scale
            raw = self.router_scale * torch.tanh(other)
            if self.config.train_router_bias:
                raw = raw + self.router_bias[None, :]
        elif self.config.router == "self_scalar":
            raw = self.router_bias[None, :] + self.router_weight[None, :] * torch.tanh(
                family_values
            )
        elif self.config.router in {"dense_loo", "dense_full"}:
            matrix = self.router_matrix
            if self.config.router == "dense_loo":
                matrix = matrix * self._off_diagonal.to(matrix.device)
            raw = torch.tanh(family_values) @ matrix.T + self.router_bias[None, :]
        elif self.config.router in {
            "global_residual",
            "pairwise_residual",
            "multinomial_diagonal",
            "multinomial_offdiag",
            "multinomial_full",
        }:
            raw = torch.zeros_like(family_values)
        else:
            raise AssertionError(f"unhandled router {self.config.router}")
        # Softmax/sparsemax is shift-invariant.  Explicit centering removes a
        # redundant degree of freedom and makes diagnostics comparable.
        return raw - raw.mean(dim=1, keepdim=True)

    def gate_from_logits(self, logits):
        scaled = logits / float(self.config.temperature)
        if self.config.normalizer == "softmax":
            probabilities = self.torch.softmax(scaled, dim=1)
        elif self.config.normalizer == "sparsemax":
            probabilities = sparsemax(scaled, dim=1)
        else:
            raise AssertionError(f"unhandled normalizer {self.config.normalizer}")
        count = probabilities.shape[1]
        strength = float(self.config.router_strength)
        gates = (1.0 - strength) + strength * count * probabilities
        return probabilities, gates

    def components(self, X):
        torch = self.torch
        base_atomic, base_family = self.base.contributions(X)
        family_values = self._family_matrix(base_family)
        logits = self.router_logits_from_family(family_values)
        probabilities, gates = self.gate_from_logits(logits)
        interaction = torch.zeros(len(X), dtype=X.dtype, device=X.device)
        family_state_input = torch.stack(
            [torch.sigmoid(-family_values), torch.sigmoid(family_values)], dim=2
        )
        family_state_output = family_state_input
        family_state_delta = torch.zeros_like(family_values)
        if self.config.router == "global_residual":
            z = torch.tanh(family_values)
            raw_interaction = (z @ self.router_weight) / math.sqrt(max(len(self.family_order), 1))
            interaction = 2.0 * float(self.config.router_strength) * torch.tanh(
                raw_interaction
            )
            gates = torch.ones_like(gates)
            probabilities = torch.full_like(probabilities, 1.0 / probabilities.shape[1])
        elif self.config.router == "pairwise_residual":
            z = torch.tanh(family_values)
            matrix = 0.5 * (self.router_matrix + self.router_matrix.T)
            matrix = matrix * self._off_diagonal.to(matrix.device)
            pair_count = max(len(self.family_order) * (len(self.family_order) - 1) / 2.0, 1.0)
            raw_interaction = 0.5 * ((z @ matrix) * z).sum(dim=1) / math.sqrt(pair_count)
            interaction = 2.0 * float(self.config.router_strength) * torch.tanh(
                raw_interaction
            )
            gates = torch.ones_like(gates)
            probabilities = torch.full_like(probabilities, 1.0 / probabilities.shape[1])
        elif self.config.router in {
            "multinomial_diagonal",
            "multinomial_offdiag",
            "multinomial_full",
        }:
            count = len(self.family_order)
            block_mask = torch.ones((count, count), dtype=X.dtype, device=X.device)
            if self.config.router == "multinomial_diagonal":
                block_mask = torch.eye(count, dtype=X.dtype, device=X.device)
            elif self.config.router == "multinomial_offdiag":
                block_mask = self._off_diagonal.to(device=X.device, dtype=X.dtype)
            effective = self.multinomial_matrix * block_mask[:, None, :, None]
            residual_logits = torch.einsum(
                "ngl,jmgl->njm", family_state_input, effective
            )
            if self.config.train_router_bias:
                residual_logits = residual_logits + self.multinomial_bias[None, :, :]
            # Remove the per-output-family translation gauge. Sparsemax is
            # applied over the two latent states, exactly as the categorical
            # DEEM layer applies it over classes rather than over experts.
            residual_logits = residual_logits - residual_logits.mean(dim=2, keepdim=True)
            if self.config.normalizer == "sparsemax":
                base_state_logits = family_state_input
                baseline_state_output = sparsemax(base_state_logits, dim=2)
                family_state_output = sparsemax(
                    base_state_logits + residual_logits, dim=2
                )
            else:
                base_state_logits = torch.log(family_state_input.clamp_min(EPS))
                baseline_state_output = torch.softmax(base_state_logits, dim=2)
                family_state_output = torch.softmax(
                    base_state_logits + residual_logits, dim=2
                )
            family_state_delta = 2.0 * float(self.config.router_strength) * (
                family_state_output[:, :, 1] - baseline_state_output[:, :, 1]
            )
            interaction = family_state_delta.sum(dim=1)
            gates = torch.ones_like(gates)
            probabilities = torch.full_like(probabilities, 1.0 / probabilities.shape[1])
        routed_atomic = base_atomic.clone()
        routed_family = {}
        for column, family_name in enumerate(self.family_order):
            indices = self.groups[family_name]
            routed_atomic[:, list(indices)] = (
                routed_atomic[:, list(indices)] * gates[:, column : column + 1]
            )
            routed_family[family_name] = family_values[:, column] * gates[:, column]
            if self.config.router in {"global_residual", "pairwise_residual"}:
                share = interaction / len(self.family_order)
                routed_atomic[:, list(indices)] = (
                    routed_atomic[:, list(indices)]
                    + share[:, None] / len(indices)
                )
                routed_family[family_name] = routed_family[family_name] + share
            elif self.config.router in {
                "multinomial_diagonal",
                "multinomial_offdiag",
                "multinomial_full",
            }:
                delta = family_state_delta[:, column]
                routed_atomic[:, list(indices)] = (
                    routed_atomic[:, list(indices)] + delta[:, None] / len(indices)
                )
                routed_family[family_name] = routed_family[family_name] + delta
        return {
            "base_atomic": base_atomic,
            "base_family": base_family,
            "family_values": family_values,
            "router_logits": logits,
            "probabilities": probabilities,
            "gates": gates,
            "routed_atomic": routed_atomic,
            "routed_family": routed_family,
            "interaction": interaction,
            "family_state_input": family_state_input,
            "family_state_output": family_state_output,
            "family_state_delta": family_state_delta,
        }

    def logit(self, X):
        values = self.components(X)
        ell = self.base.b + values["routed_atomic"].sum(dim=1)
        return ell, values

    def free_energy(self, X):
        ell, _ = self.logit(X)
        return 0.5 * ((X - self.base.a) ** 2).sum(dim=1) - self.torch.nn.functional.softplus(
            ell
        )

    def router_penalties(self, X):
        _, values = self.logit(X)
        probabilities = values["probabilities"]
        logits = values["router_logits"]
        expected = 1.0 / probabilities.shape[1]
        balance = ((probabilities.mean(dim=0) - expected) ** 2).sum()
        magnitude = (
            logits.square().mean()
            + values["interaction"].square().mean()
            + values["family_state_delta"].square().mean()
        )
        return balance, magnitude

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        output = {
            f"base::{name}": value for name, value in self.base.state_dict_numpy().items()
        }
        output.update(
            {
                "router::bias": self.router_bias.detach().cpu().numpy().copy(),
                "router::weight": self.router_weight.detach().cpu().numpy().copy(),
                "router::scale": self.router_scale.detach().cpu().numpy().copy(),
                "router::matrix": self.router_matrix.detach().cpu().numpy().copy(),
                "router::multinomial_matrix": self.multinomial_matrix.detach().cpu().numpy().copy(),
                "router::multinomial_bias": self.multinomial_bias.detach().cpu().numpy().copy(),
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
            self.router_bias.copy_(
                torch.as_tensor(state["router::bias"], dtype=torch.float64)
            )
            self.router_weight.copy_(
                torch.as_tensor(state["router::weight"], dtype=torch.float64)
            )
            self.router_scale.copy_(
                torch.as_tensor(state["router::scale"], dtype=torch.float64)
            )
            self.router_matrix.copy_(
                torch.as_tensor(state["router::matrix"], dtype=torch.float64)
            )
            self.multinomial_matrix.copy_(
                torch.as_tensor(state["router::multinomial_matrix"], dtype=torch.float64)
            )
            self.multinomial_bias.copy_(
                torch.as_tensor(state["router::multinomial_bias"], dtype=torch.float64)
            )


def _router_diagnostics(probabilities: np.ndarray, gates: np.ndarray, family_order) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    gates = np.asarray(gates, dtype=float)
    count = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, EPS, None)), axis=1)
    normalized_entropy = entropy / math.log(count) if count > 1 else np.ones(len(entropy))
    top = np.argmax(probabilities, axis=1)
    return {
        "gate_sum_max_abs_error": float(np.max(np.abs(gates.sum(axis=1) - count))),
        "gate_mean_abs_deviation_from_one": float(np.mean(np.abs(gates - 1.0))),
        "gate_sd": float(np.std(gates)),
        "gate_min": float(np.min(gates)),
        "gate_max": float(np.max(gates)),
        "normalized_entropy_mean": float(np.mean(normalized_entropy)),
        "effective_experts_mean": float(np.mean(np.exp(entropy))),
        "family_gate_mean": {
            name: float(np.mean(gates[:, column]))
            for column, name in enumerate(family_order)
        },
        "family_top_route_fraction": {
            name: float(np.mean(top == column))
            for column, name in enumerate(family_order)
        },
    }


def fit_deem_b3_moe(
    X_risk: np.ndarray,
    feature_names: Sequence[str],
    baseline_state: Mapping[str, np.ndarray],
    *,
    seed: int = 0,
    config: DeemMoEConfig | None = None,
) -> DeemMoEResult:
    """Warm-start B3 and fit a label-free family router.

    Target labels are neither accepted nor imported.  ``baseline_state`` must
    be the raw (pre-orientation) state of the matching B3 seed.
    """

    import torch

    started = time.perf_counter()
    config = config or DeemMoEConfig()
    _validate_config(config)
    X = validate_inventory(X_risk, feature_names)
    names = tuple(str(name) for name in feature_names)
    set_determinism(seed)
    model = _FamilyRouterEnergy(names, config, seed)
    model.load_baseline_state(baseline_state)
    router_parameters = list(model.router_parameters())
    base_parameters = list(model.base.parameters()) if config.train_experts else []
    for parameter in model.base.parameters():
        parameter.requires_grad_(bool(config.train_experts))
    parameters = base_parameters + router_parameters
    if not parameters and int(config.epochs) > 0:
        raise ValueError("uniform frozen-expert identity has no trainable parameters")

    tensor = torch.as_tensor(X, dtype=torch.float64)
    buffer = tensor.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 2_000_033)
    optimizer = None
    if parameters:
        groups = []
        if base_parameters:
            groups.append(
                {
                    "params": base_parameters,
                    "lr": float(
                        config.expert_learning_rate
                        if config.expert_learning_rate is not None
                        else config.learning_rate
                    ),
                }
            )
        if router_parameters:
            groups.append({"params": router_parameters, "lr": float(config.learning_rate)})
        optimizer = torch.optim.SGD(
            groups,
            lr=float(config.learning_rate),
            momentum=float(config.momentum),
        )
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
            balance, magnitude = model.router_penalties(tensor)
            loss = (
                deem_loss
                + float(config.balance_penalty) * balance
                + float(config.router_l2) * magnitude
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite DEEM-MoE objective at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            for index, parameter in enumerate(parameters):
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(
                        f"non-finite DEEM-MoE gradient for parameter {index} at epoch {epoch}"
                    )
            optimizer.step()
            for index, parameter in enumerate(parameters):
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(
                        f"non-finite DEEM-MoE parameter {index} after epoch {epoch}"
                    )
            last_finite_state = model.state_dict_numpy()
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach()),
                    "deem_loss": float(deem_loss.detach()),
                    "balance_penalty_raw": float(balance.detach()),
                    "router_l2_raw": float(magnitude.detach()),
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
        ell = ell_t.cpu().numpy()
        routed_atomic = values["routed_atomic"].cpu().numpy()
        routed_family = {
            name: values["routed_family"][name].cpu().numpy()
            for name in model.family_order
        }
        base_family = {
            name: values["base_family"][name].cpu().numpy()
            for name in model.family_order
        }
        probabilities = values["probabilities"].cpu().numpy()
        gates = values["gates"].cpu().numpy()
        router_logits = values["router_logits"].cpu().numpy()
        interaction = values["interaction"].cpu().numpy()
        family_state_input = values["family_state_input"].cpu().numpy()
        family_state_output = values["family_state_output"].cpu().numpy()
        family_state_delta = values["family_state_delta"].cpu().numpy()

    raw_reconstruction = float(
        np.max(np.abs(model.base.b.detach().item() + routed_atomic.sum(axis=1) - ell))
    )
    if raw_reconstruction > 1e-8:
        raise ResidualGraphDeemError(
            f"DEEM-MoE contribution reconstruction failed: {raw_reconstruction:.3e}"
        )
    q = 1.0 / (1.0 + np.exp(-np.clip(ell, -700.0, 700.0)))
    anchor = equal_family_risk_anchor(X, names)
    high = float(np.sum(q * anchor) / max(np.sum(q), EPS))
    low = float(np.sum((1.0 - q) * anchor) / max(np.sum(1.0 - q), EPS))
    difference = high - low
    if abs(difference) <= float(config.anchor_tolerance):
        raise ResidualGraphDeemError(
            f"DEEM-MoE risk-anchor alignment ambiguous: {abs(difference):.3e}"
        )
    orientation = 1 if difference > 0 else -1
    if orientation < 0:
        q = 1.0 - q
        ell = -ell
        routed_atomic = -routed_atomic
        routed_family = {name: -value for name, value in routed_family.items()}
        base_family = {name: -value for name, value in base_family.items()}
        interaction = -interaction
        family_state_input = family_state_input[:, :, ::-1].copy()
        family_state_output = family_state_output[:, :, ::-1].copy()
        family_state_delta = -family_state_delta
        difference = -difference

    aligned_bias = float(orientation * model.base.b.detach().item())
    aligned_reconstruction = float(
        np.max(np.abs(aligned_bias + routed_atomic.sum(axis=1) - ell))
    )
    diagnostics = _router_diagnostics(probabilities, gates, model.family_order)
    diagnostics.update(
        {
            "interaction_sd": float(np.std(interaction)),
            "interaction_max_abs": float(np.max(np.abs(interaction))),
            "family_state_delta_mean_abs": float(np.mean(np.abs(family_state_delta))),
            "family_state_delta_max_abs": float(np.max(np.abs(family_state_delta))),
            "family_state_output_zero_fraction": float(
                np.mean(family_state_output <= EPS)
            ),
        }
    )
    posterior_sd = float(np.std(q))
    acceptance_values = [row["mala_acceptance"] for row in history]
    healthy = bool(
        posterior_sd >= float(config.posterior_sd_min)
        and np.isfinite(q).all()
        and np.isfinite(gates).all()
        and aligned_reconstruction <= 1e-8
        and diagnostics["gate_sum_max_abs_error"] <= 1e-10
        and all(np.isfinite(row["loss"]) for row in history)
    )
    return DeemMoEResult(
        score=q.copy(),
        posterior=np.column_stack([1.0 - q, q]),
        logit=ell,
        contributions=routed_atomic,
        family_contributions=routed_family,
        base_family_contributions=base_family,
        gates=gates,
        router_probabilities=probabilities,
        router_logits=router_logits,
        interaction=interaction,
        family_state_input=family_state_input,
        family_state_output=family_state_output,
        family_state_delta=family_state_delta,
        family_order=model.family_order,
        aligned_bias=aligned_bias,
        orientation=orientation,
        risk_anchor_difference=float(difference),
        feature_names=names,
        family_indices=model.groups,
        state=state,
        objective_history=history,
        health={
            "healthy": healthy,
            "posterior_sd": posterior_sd,
            "finite": bool(np.isfinite(q).all() and np.isfinite(gates).all()),
            "contribution_reconstruction_max_abs": aligned_reconstruction,
            "epochs_completed": len(history),
            "mala_acceptance_mean": (
                float(np.mean(acceptance_values)) if acceptance_values else float("nan")
            ),
            "runtime_seconds": float(time.perf_counter() - started),
        },
        diagnostics=diagnostics,
        config=jsonable(asdict(config)),
        seed=int(seed),
    )


def predict_deem_b3_moe(result: DeemMoEResult, X_risk: np.ndarray) -> dict[str, Any]:
    """Score a batch with a fitted DEEM-MoE result."""

    import torch

    X = validate_inventory(X_risk, result.feature_names)
    config = DeemMoEConfig(**result.config)
    model = _FamilyRouterEnergy(result.feature_names, config, result.seed)
    model.load_state_numpy(result.state)
    with torch.no_grad():
        ell_t, values = model.logit(torch.as_tensor(X, dtype=torch.float64))
    orientation = int(result.orientation)
    ell = orientation * ell_t.cpu().numpy()
    contribution = orientation * values["routed_atomic"].cpu().numpy()
    family = {
        name: orientation * values["routed_family"][name].cpu().numpy()
        for name in model.family_order
    }
    probabilities = values["probabilities"].cpu().numpy()
    gates = values["gates"].cpu().numpy()
    interaction = values["interaction"].cpu().numpy()
    family_state_input = values["family_state_input"].cpu().numpy()
    family_state_output = values["family_state_output"].cpu().numpy()
    family_state_delta = values["family_state_delta"].cpu().numpy()
    if orientation < 0:
        interaction = -interaction
        family_state_input = family_state_input[:, :, ::-1].copy()
        family_state_output = family_state_output[:, :, ::-1].copy()
        family_state_delta = -family_state_delta
    score = 1.0 / (1.0 + np.exp(-np.clip(ell, -700.0, 700.0)))
    reconstruction = float(
        np.max(np.abs(result.aligned_bias + contribution.sum(axis=1) - ell))
    )
    if reconstruction > 1e-8:
        raise ResidualGraphDeemError("held DEEM-MoE contribution reconstruction failed")
    return {
        "score": score,
        "posterior": np.column_stack([1.0 - score, score]),
        "logit": ell,
        "contributions": contribution,
        "family_contributions": family,
        "gates": gates,
        "router_probabilities": probabilities,
        "router_logits": values["router_logits"].cpu().numpy(),
        "interaction": interaction,
        "family_state_input": family_state_input,
        "family_state_output": family_state_output,
        "family_state_delta": family_state_delta,
        "router_diagnostics": _router_diagnostics(
            probabilities, gates, model.family_order
        ),
        "reconstruction_max_abs": reconstruction,
    }


__all__ = [
    "DeemMoEConfig",
    "DeemMoEResult",
    "NORMALIZERS",
    "ROUTERS",
    "_FamilyRouterEnergy",
    "fit_deem_b3_moe",
    "predict_deem_b3_moe",
    "sparsemax",
]
