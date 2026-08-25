"""B3 on Feature Contract V2, with no additional input layer.

The energy, optimizer, MALA sampler, width, and epoch count are the frozen B3
recipe.  Only the target-free input contract and its block map differ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .feature_contract import confidence_sign_vector
from .residual_graph_deem import (
    ContinuousDeemConfig,
    EPS,
    ResidualGraphDeemError,
    persistent_mala,
    set_determinism,
)


SCHEMA = "deem_b3_feature_contract_v2_core_2026_08_25"
BLOCK_ORDER = (
    "predictive_distribution",
    "entropy_dynamics",
    "realized_token_surprisal",
    "raw_logit_partition",
)
PREDICTIVE = frozenset({
    "entropy_common", "entropy_support_delta", "mean_top1_logprob",
    "logprob_margin", "varentropy", "renyi_entropy_2", "topk_tail_mass",
})
DYNAMICS = frozenset({
    "spectral_entropy", "low_band_power", "high_band_power", "dominant_freq",
    "spectral_centroid", "stft_max_high_power", "stft_spectral_entropy", "rpdi",
    "sw_var_peak", "pe_mean", "hurst_exponent", "cusum_max", "cusum_shift_idx",
})
REALIZED = frozenset({
    "epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled", "min_spilled",
})
PARTITION = frozenset({
    "epr_energy", "min_energy", "sw_var_peak_energy", "cusum_max_energy",
})


@dataclass(frozen=True)
class V2Transform:
    mean: np.ndarray
    scale: np.ndarray
    constant_mask: np.ndarray
    delta_scale_source: str


@dataclass
class V2B3Result:
    score: np.ndarray
    posterior: np.ndarray
    logit: np.ndarray
    contributions: np.ndarray
    family_contributions: dict[str, np.ndarray]
    aligned_bias: float
    orientation: int
    risk_anchor_difference: float
    feature_names: tuple[str, ...]
    family_indices: dict[str, tuple[int, ...]]
    state: dict[str, np.ndarray]
    objective_history: list[dict[str, float]]
    health: dict[str, Any]
    config: dict[str, Any]
    seed: int


def block_for_feature(name: str) -> str:
    if name in PREDICTIVE:
        return "predictive_distribution"
    if name in DYNAMICS:
        return "entropy_dynamics"
    if name in REALIZED:
        return "realized_token_surprisal"
    if name in PARTITION:
        return "raw_logit_partition"
    raise KeyError(f"Feature Contract V2 feature has no block: {name}")


def block_index_map(feature_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    names = tuple(map(str, feature_names))
    if len(names) != len(set(names)):
        raise ValueError("feature names are not unique")
    assigned = {name: block_for_feature(name) for name in names}
    output = {
        block: tuple(i for i, name in enumerate(names) if assigned[name] == block)
        for block in BLOCK_ORDER
    }
    output = {block: indices for block, indices in output.items() if indices}
    if len(output) < 3:
        raise ValueError("Feature Contract V2 requires at least three present blocks")
    return output


def prepare_v2_risk(
    X_contract_raw: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, V2Transform]:
    X = np.asarray(X_contract_raw, dtype=np.float64)
    names = tuple(map(str, feature_names))
    if X.ndim != 2 or X.shape != (len(X), len(names)) or len(X) < 3:
        raise ValueError("invalid Feature Contract V2 matrix")
    if not np.isfinite(X).all():
        raise ValueError("non-finite Feature Contract V2 matrix")
    block_index_map(names)
    lookup = {name: i for i, name in enumerate(names)}
    if not {"entropy_common", "entropy_support_delta"} <= set(names):
        raise ValueError("Feature Contract V2 entropy coordinates missing")
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    constant = scale < EPS
    scale = scale.copy()
    scale[constant] = 1.0
    common_scale = float(scale[lookup["entropy_common"]])
    scale[lookup["entropy_support_delta"]] = common_scale
    Z = (X - mean[None, :]) / scale[None, :]
    risk = np.empty_like(Z)
    for index, name in enumerate(names):
        if name in {"entropy_common", "entropy_support_delta"}:
            risk[:, index] = Z[:, index]
        else:
            sign = float(confidence_sign_vector((name,))[0])
            risk[:, index] = -Z[:, index] * sign
    if not np.isfinite(risk).all():
        raise ValueError("non-finite prepared V2 risk matrix")
    return risk, V2Transform(
        mean=mean,
        scale=scale,
        constant_mask=constant,
        delta_scale_source="entropy_common_sd",
    )


def equal_block_anchor(X_risk: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    X = np.asarray(X_risk, dtype=np.float64)
    names = tuple(map(str, feature_names))
    groups = block_index_map(names)
    block_values = []
    for block, indices in groups.items():
        usable = [i for i in indices if names[i] != "entropy_support_delta"]
        if not usable:
            continue
        block_values.append(X[:, usable].mean(axis=1))
    if len(block_values) < 3:
        raise ValueError("orientation anchor has fewer than three blocks")
    return np.mean(block_values, axis=0)


class _V2FamilyAdditiveEnergy:
    def __init__(self, feature_names: Sequence[str], config: ContinuousDeemConfig, seed: int):
        import torch

        self.torch = torch
        self.names = tuple(map(str, feature_names))
        self.groups = block_index_map(self.names)
        self.config = config
        self.seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        dtype = torch.float64
        p = len(self.names)
        self.a = torch.nn.Parameter(torch.zeros(p, dtype=dtype))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.w = torch.nn.ParameterDict()
        self.W = torch.nn.ParameterDict()
        self.d = torch.nn.ParameterDict()
        self.V = torch.nn.ParameterDict()
        self.e = torch.nn.ParameterDict()
        for family, indices in self.groups.items():
            size = len(indices)
            self.w[family] = torch.nn.Parameter(
                torch.full((size,), 2.0 / (len(self.groups) * size), dtype=dtype)
            )
            self.W[family] = torch.nn.Parameter(
                torch.randn(config.family_width, size, dtype=dtype, generator=generator) * config.init_sd
            )
            self.d[family] = torch.nn.Parameter(
                torch.randn(config.family_width, dtype=dtype, generator=generator) * config.init_sd
            )
            self.V[family] = torch.nn.Parameter(
                torch.randn(size, config.family_width, dtype=dtype, generator=generator) * config.init_sd
            )
            self.e[family] = torch.nn.Parameter(
                torch.randn(size, dtype=dtype, generator=generator) * config.init_sd
            )

    def parameters(self):
        values = [self.a, self.b]
        for collection in (self.w, self.W, self.d, self.V, self.e):
            values.extend(collection.values())
        return values

    def contributions(self, X):
        torch = self.torch
        atomic = torch.zeros_like(X)
        families = {}
        for family, indices in self.groups.items():
            index = torch.tensor(indices, dtype=torch.long, device=X.device)
            xg = X.index_select(1, index)
            hidden = torch.tanh(xg @ self.W[family].T + self.d[family])
            nonlinear = (2.0 / len(indices)) * torch.tanh(
                hidden @ self.V[family].T + self.e[family]
            )
            contribution = self.w[family] * xg + nonlinear
            atomic[:, list(indices)] = contribution
            families[family] = contribution.sum(dim=1)
        return atomic, families

    def logit(self, X):
        contribution, family = self.contributions(X)
        return self.b + contribution.sum(dim=1), contribution, family

    def free_energy(self, X):
        ell, _, _ = self.logit(X)
        return 0.5 * ((X - self.a) ** 2).sum(dim=1) - self.torch.nn.functional.softplus(ell)

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        output = {"a": self.a.detach().cpu().numpy().copy(), "b": self.b.detach().cpu().numpy().copy()}
        for name, collection in (("w", self.w), ("W", self.W), ("d", self.d), ("V", self.V), ("e", self.e)):
            for family, value in collection.items():
                output[f"{name}::{family}"] = value.detach().cpu().numpy().copy()
        return output


def fit_v2_b3(
    X_risk: np.ndarray,
    feature_names: Sequence[str],
    *,
    seed: int,
    config: ContinuousDeemConfig | None = None,
) -> V2B3Result:
    import torch

    started = time.perf_counter()
    X = np.asarray(X_risk, dtype=np.float64)
    names = tuple(map(str, feature_names))
    if X.ndim != 2 or X.shape[1] != len(names) or not np.isfinite(X).all():
        raise ValueError("invalid V2 risk matrix")
    config = config or ContinuousDeemConfig()
    if config.dtype != "float64" or config.device != "cpu":
        raise ValueError("V2 baseline preserves frozen float64 CPU B3")
    set_determinism(seed)
    model = _V2FamilyAdditiveEnergy(names, config, seed)
    tensor = torch.as_tensor(X, dtype=torch.float64)
    buffer = tensor.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1_000_003)
    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=config.learning_rate, momentum=config.momentum)
    history: list[dict[str, float]] = []
    for epoch in range(int(config.epochs)):
        refresh = torch.rand(len(X), generator=generator) < float(config.replay_refresh)
        if bool(refresh.any()):
            replacements = torch.randint(len(X), (int(refresh.sum()),), generator=generator)
            buffer[refresh] = tensor[replacements]
        buffer, acceptance = persistent_mala(
            model, buffer, delta=config.mala_delta, steps=config.mala_steps, generator=generator
        )
        loss = model.free_energy(tensor).mean() - model.free_energy(buffer).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite objective at epoch {epoch}")
        optimizer.zero_grad()
        loss.backward()
        for index, parameter in enumerate(parameters):
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"non-finite gradient {index} at epoch {epoch}")
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise FloatingPointError(f"non-finite parameter at epoch {epoch}")
        history.append({
            "epoch": float(epoch),
            "loss": float(loss.detach()),
            "mala_acceptance": float(acceptance),
        })

    with torch.no_grad():
        ell_t, atomic_t, family_t = model.logit(tensor)
    ell = ell_t.cpu().numpy()
    atomic = atomic_t.cpu().numpy()
    family = {name: value.cpu().numpy() for name, value in family_t.items()}
    reconstruction = float(np.max(np.abs(float(model.b.detach()) + atomic.sum(axis=1) - ell)))
    if reconstruction > 1e-8:
        raise ResidualGraphDeemError("V2 contribution reconstruction failed")
    q = 1.0 / (1.0 + np.exp(-np.clip(ell, -700.0, 700.0)))
    anchor = equal_block_anchor(X, names)
    high = float(np.sum(q * anchor) / max(np.sum(q), EPS))
    low = float(np.sum((1.0 - q) * anchor) / max(np.sum(1.0 - q), EPS))
    difference = high - low
    if abs(difference) <= config.anchor_tolerance:
        raise ResidualGraphDeemError("V2 risk-anchor alignment ambiguous")
    orientation = 1 if difference > 0 else -1
    if orientation < 0:
        q = 1.0 - q
        ell = -ell
        atomic = -atomic
        family = {name: -value for name, value in family.items()}
        difference = -difference
    posterior_sd = float(np.std(q))
    health = {
        "healthy": bool(posterior_sd >= config.posterior_sd_min and np.isfinite(q).all()),
        "finite": bool(np.isfinite(q).all()),
        "posterior_sd": posterior_sd,
        "epochs_completed": len(history),
        "contribution_reconstruction_max_abs": reconstruction,
        "mala_acceptance_mean": float(np.mean([row["mala_acceptance"] for row in history])),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return V2B3Result(
        score=q.copy(),
        posterior=np.column_stack((1.0 - q, q)),
        logit=ell,
        contributions=atomic,
        family_contributions=family,
        aligned_bias=float(orientation * model.b.detach()),
        orientation=orientation,
        risk_anchor_difference=float(difference),
        feature_names=names,
        family_indices=model.groups,
        state=model.state_dict_numpy(),
        objective_history=history,
        health=health,
        config=asdict(config),
        seed=int(seed),
    )


__all__ = [
    "BLOCK_ORDER", "SCHEMA", "V2B3Result", "V2Transform", "block_index_map",
    "equal_block_anchor", "fit_v2_b3", "prepare_v2_risk",
]
