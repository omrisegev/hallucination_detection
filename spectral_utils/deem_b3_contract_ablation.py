"""Target-free contract/block ablations around frozen continuous B3."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np

from .feature_contract import confidence_sign_vector
from .residual_graph_deem import (
    ContinuousDeemConfig,
    EPS,
    persistent_mala,
    set_determinism,
)
from .specrage_views import FEATURE_TO_VIEW, VIEW_ORDER


ARMS = ("D1_TRANSFORM_ONLY", "D2_DROP_ONLY", "D3_REGROUP_ONLY")


@dataclass
class PreparedArm:
    X_risk: np.ndarray
    feature_names: tuple[str, ...]
    groups: dict[str, tuple[int, ...]]
    anchor_exclusions: frozenset[str]
    transform_mean: np.ndarray
    transform_scale: np.ndarray


@dataclass
class GenericResult:
    score: np.ndarray
    logit: np.ndarray
    contributions: np.ndarray
    family_contributions: dict[str, np.ndarray]
    aligned_bias: float
    orientation: int
    health: dict[str, Any]
    state: dict[str, np.ndarray]


def legacy_groups(names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    return {
        family: tuple(i for i, name in enumerate(names) if FEATURE_TO_VIEW[name] == family)
        for family in VIEW_ORDER
        if any(FEATURE_TO_VIEW[name] == family for name in names)
    }


def standardize_legacy(X: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    signs = confidence_sign_vector(names)
    return -(X - mean) / scale * signs[None, :], mean, scale


def prepare_arm(X_raw: np.ndarray, feature_names: Sequence[str], arm: str) -> PreparedArm:
    X = np.asarray(X_raw, dtype=np.float64)
    names = tuple(map(str, feature_names))
    lookup = {name: i for i, name in enumerate(names)}
    if arm == "D1_TRANSFORM_ONLY":
        h15 = X[:, lookup["epr"]]
        hsaved = X[:, lookup["mean_logprob_entropy"]]
        common, delta = 0.5 * (h15 + hsaved), hsaved - h15
        new_names, columns = [], []
        for name in names:
            if name == "epr":
                new_names.append("entropy_common")
                columns.append(common)
            elif name == "mean_logprob_entropy":
                new_names.append("entropy_support_delta")
                columns.append(delta)
            else:
                new_names.append(name)
                columns.append(X[:, lookup[name]])
        Y = np.column_stack(columns)
        mean, scale = Y.mean(axis=0), Y.std(axis=0)
        scale = np.where(scale > EPS, scale, 1.0)
        common_i, delta_i = new_names.index("entropy_common"), new_names.index("entropy_support_delta")
        scale[delta_i] = scale[common_i]
        risk = np.empty_like(Y)
        for i, name in enumerate(new_names):
            z = (Y[:, i] - mean[i]) / scale[i]
            if name in {"entropy_common", "entropy_support_delta"}:
                risk[:, i] = z
            else:
                risk[:, i] = -z * float(confidence_sign_vector((name,))[0])
        groups = legacy_groups(["epr" if n == "entropy_common" else "mean_logprob_entropy" if n == "entropy_support_delta" else n for n in new_names])
        return PreparedArm(risk, tuple(new_names), groups, frozenset({"entropy_support_delta"}), mean, scale)
    if arm == "D2_DROP_ONLY":
        keep = [i for i, name in enumerate(names) if name not in {"trace_length", "hl_ratio"}]
        new_names = tuple(names[i] for i in keep)
        risk, mean, scale = standardize_legacy(X[:, keep], new_names)
        return PreparedArm(risk, new_names, legacy_groups(new_names), frozenset(), mean, scale)
    if arm == "D3_REGROUP_ONLY":
        risk, mean, scale = standardize_legacy(X, names)
        groups: dict[str, tuple[int, ...]] = {}
        groups["predictive_distribution"] = tuple(
            i for i, name in enumerate(names)
            if FEATURE_TO_VIEW[name] in {"entropy_level", "topk_distribution"}
        )
        for old, new in (
            ("entropy_dynamics", "entropy_dynamics"),
            ("sampled_token_energy", "realized_token_surprisal"),
            ("partition_energy", "raw_logit_partition"),
            ("structural", "structural_context"),
        ):
            indices = tuple(i for i, name in enumerate(names) if FEATURE_TO_VIEW[name] == old)
            if indices:
                groups[new] = indices
        return PreparedArm(risk, names, groups, frozenset(), mean, scale)
    raise KeyError(arm)


class GenericEnergy:
    def __init__(self, names: Sequence[str], groups: dict[str, tuple[int, ...]], config: ContinuousDeemConfig, seed: int):
        import torch

        self.torch = torch
        self.names, self.groups = tuple(names), groups
        generator = torch.Generator(device="cpu").manual_seed(seed)
        dtype = torch.float64
        self.a = torch.nn.Parameter(torch.zeros(len(names), dtype=dtype))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.w, self.W, self.d, self.V, self.e = (torch.nn.ParameterDict() for _ in range(5))
        for family, indices in groups.items():
            size = len(indices)
            self.w[family] = torch.nn.Parameter(torch.full((size,), 2.0 / (len(groups) * size), dtype=dtype))
            self.W[family] = torch.nn.Parameter(torch.randn(config.family_width, size, dtype=dtype, generator=generator) * config.init_sd)
            self.d[family] = torch.nn.Parameter(torch.randn(config.family_width, dtype=dtype, generator=generator) * config.init_sd)
            self.V[family] = torch.nn.Parameter(torch.randn(size, config.family_width, dtype=dtype, generator=generator) * config.init_sd)
            self.e[family] = torch.nn.Parameter(torch.randn(size, dtype=dtype, generator=generator) * config.init_sd)

    def parameters(self):
        output = [self.a, self.b]
        for values in (self.w, self.W, self.d, self.V, self.e):
            output.extend(values.values())
        return output

    def contributions(self, X):
        torch = self.torch
        atomic, families = torch.zeros_like(X), {}
        for family, indices in self.groups.items():
            xg = X[:, list(indices)]
            hidden = torch.tanh(xg @ self.W[family].T + self.d[family])
            contribution = self.w[family] * xg + (2.0 / len(indices)) * torch.tanh(hidden @ self.V[family].T + self.e[family])
            atomic[:, list(indices)] = contribution
            families[family] = contribution.sum(dim=1)
        return atomic, families

    def logit(self, X):
        atomic, families = self.contributions(X)
        return self.b + atomic.sum(dim=1), atomic, families

    def free_energy(self, X):
        ell, _, _ = self.logit(X)
        return 0.5 * ((X - self.a) ** 2).sum(dim=1) - self.torch.nn.functional.softplus(ell)

    def state(self):
        output = {"a": self.a.detach().numpy().copy(), "b": self.b.detach().numpy().copy()}
        for prefix, values in (("w", self.w), ("W", self.W), ("d", self.d), ("V", self.V), ("e", self.e)):
            for family, value in values.items():
                output[f"{prefix}::{family}"] = value.detach().numpy().copy()
        return output


def fit_generic(prepared: PreparedArm, *, seed: int = 0, config: ContinuousDeemConfig | None = None) -> GenericResult:
    import torch

    config = config or ContinuousDeemConfig()
    started = time.perf_counter()
    X, names, groups = prepared.X_risk, prepared.feature_names, prepared.groups
    set_determinism(seed)
    model = GenericEnergy(names, groups, config, seed)
    tensor = torch.as_tensor(X, dtype=torch.float64)
    buffer = tensor.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=config.learning_rate, momentum=config.momentum)
    acceptances = []
    for _epoch in range(config.epochs):
        refresh = torch.rand(len(X), generator=generator) < config.replay_refresh
        if bool(refresh.any()):
            buffer[refresh] = tensor[torch.randint(len(X), (int(refresh.sum()),), generator=generator)]
        buffer, acceptance = persistent_mala(model, buffer, delta=config.mala_delta, steps=config.mala_steps, generator=generator)
        loss = model.free_energy(tensor).mean() - model.free_energy(buffer).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(p).all()) for p in parameters):
            raise FloatingPointError("non-finite generic B3 fit")
        acceptances.append(float(acceptance))
    with torch.no_grad():
        ell_t, atomic_t, family_t = model.logit(tensor)
    ell, atomic = ell_t.numpy(), atomic_t.numpy()
    families = {name: value.numpy() for name, value in family_t.items()}
    q = 1.0 / (1.0 + np.exp(-np.clip(ell, -700, 700)))
    block_values = []
    for family, indices in groups.items():
        usable = [i for i in indices if names[i] not in prepared.anchor_exclusions]
        if usable:
            block_values.append(X[:, usable].mean(axis=1))
    anchor = np.mean(block_values, axis=0)
    high = float(np.sum(q * anchor) / max(np.sum(q), EPS))
    low = float(np.sum((1 - q) * anchor) / max(np.sum(1 - q), EPS))
    orientation = 1 if high - low > 0 else -1
    if orientation < 0:
        q, ell, atomic = 1 - q, -ell, -atomic
        families = {name: -value for name, value in families.items()}
    reconstruction = float(np.max(np.abs(orientation * float(model.b.detach()) + atomic.sum(axis=1) - ell)))
    health = {
        "healthy": bool(np.std(q) >= config.posterior_sd_min and reconstruction <= 1e-8),
        "posterior_sd": float(np.std(q)),
        "reconstruction": reconstruction,
        "mala_acceptance": float(np.mean(acceptances)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return GenericResult(q, ell, atomic, families, orientation * float(model.b.detach()), orientation, health, model.state())


__all__ = ["ARMS", "GenericResult", "PreparedArm", "fit_generic", "prepare_arm", "standardize_legacy"]
