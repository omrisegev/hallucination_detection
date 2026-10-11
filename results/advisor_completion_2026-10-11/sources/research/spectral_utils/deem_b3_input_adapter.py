"""Identity-initialized structured input adapter for continuous B3.

The adapter matches the observed covariance geometry with two tiny components:
block-local PCA directions and global cross-block PCA directions.  It preserves
the old B3 family boundaries and uses the entropy common/difference coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .deem_b3_contract_ablation import GenericEnergy, GenericResult, PreparedArm
from .residual_graph_deem import ContinuousDeemConfig, EPS, persistent_mala, set_determinism


ARMS = ("I1_WITHIN_R2", "I2_CROSS_R2", "I3_BLOCK_PLUS_CROSS_R2")
ADAPTER_SCALE = 0.5


@dataclass(frozen=True)
class AdapterBases:
    within: np.ndarray
    cross: np.ndarray
    diagnostics: dict[str, Any]


def _stable_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = sorted(range(len(values)), key=lambda i: (-abs(float(values[i])), i))
    values, vectors = values[order], vectors[:, order]
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    return values, vectors


def build_adapter_bases(prepared: PreparedArm, rank: int = 2) -> AdapterBases:
    X = np.asarray(prepared.X_risk, dtype=np.float64)
    covariance = X.T @ X / max(len(X) - 1, 1)
    p = X.shape[1]
    within_columns = []
    block_covariance = np.zeros_like(covariance)
    for indices in prepared.groups.values():
        block = covariance[np.ix_(indices, indices)]
        block_covariance[np.ix_(indices, indices)] = block
        if len(indices) < 2:
            continue
        _, vectors = _stable_eigh(block)
        for column in range(min(rank, len(indices))):
            embedded = np.zeros(p, dtype=np.float64)
            embedded[list(indices)] = vectors[:, column]
            within_columns.append(embedded)
    cross_covariance = covariance - block_covariance
    cross_values, cross_vectors = _stable_eigh(cross_covariance)
    within = np.column_stack(within_columns) if within_columns else np.zeros((p, 0))
    cross = cross_vectors[:, : min(rank, p)]
    return AdapterBases(
        within=within,
        cross=cross,
        diagnostics={
            "within_rank": int(within.shape[1]),
            "cross_rank": int(cross.shape[1]),
            "cross_eigenvalues": cross_values[: min(rank, p)].tolist(),
            "offblock_covariance_fraction": float(
                np.linalg.norm(cross_covariance, ord="fro") / max(np.linalg.norm(covariance, ord="fro"), EPS)
            ),
        },
    )


class AdapterEnergy(GenericEnergy):
    def __init__(self, prepared: PreparedArm, config: ContinuousDeemConfig, seed: int, bases: AdapterBases, arm: str):
        super().__init__(prepared.feature_names, prepared.groups, config, seed)
        torch = self.torch
        self.arm = arm
        self.within_basis = torch.as_tensor(bases.within, dtype=torch.float64)
        self.cross_basis = torch.as_tensor(bases.cross, dtype=torch.float64)
        self.theta_within = torch.nn.Parameter(torch.zeros(self.within_basis.shape[1], dtype=torch.float64))
        self.theta_cross = torch.nn.Parameter(torch.zeros(self.cross_basis.shape[1], dtype=torch.float64))

    def parameters(self):
        values = super().parameters()
        if self.arm in {"I1_WITHIN_R2", "I3_BLOCK_PLUS_CROSS_R2"}:
            values.append(self.theta_within)
        if self.arm in {"I2_CROSS_R2", "I3_BLOCK_PLUS_CROSS_R2"}:
            values.append(self.theta_cross)
        return values

    def input_transform(self, X):
        torch = self.torch
        delta = torch.zeros_like(X)
        if self.arm in {"I1_WITHIN_R2", "I3_BLOCK_PLUS_CROSS_R2"} and self.within_basis.shape[1]:
            delta = delta + ((X @ self.within_basis) * self.theta_within) @ self.within_basis.T
        if self.arm in {"I2_CROSS_R2", "I3_BLOCK_PLUS_CROSS_R2"} and self.cross_basis.shape[1]:
            delta = delta + ((X @ self.cross_basis) * self.theta_cross) @ self.cross_basis.T
        return X + ADAPTER_SCALE * torch.tanh(delta)

    def contributions(self, X):
        return super().contributions(self.input_transform(X))

    def state(self):
        output = super().state()
        output["adapter::theta_within"] = self.theta_within.detach().numpy().copy()
        output["adapter::theta_cross"] = self.theta_cross.detach().numpy().copy()
        output["adapter::within_basis"] = self.within_basis.detach().numpy().copy()
        output["adapter::cross_basis"] = self.cross_basis.detach().numpy().copy()
        return output


def fit_adapter(
    prepared: PreparedArm,
    arm: str,
    *,
    seed: int = 0,
    config: ContinuousDeemConfig | None = None,
) -> tuple[GenericResult, AdapterBases, dict[str, float]]:
    import torch

    if arm not in ARMS:
        raise KeyError(arm)
    config = config or ContinuousDeemConfig()
    bases = build_adapter_bases(prepared)
    started = time.perf_counter()
    X = prepared.X_risk
    set_determinism(seed)
    model = AdapterEnergy(prepared, config, seed, bases, arm)
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
            raise FloatingPointError("non-finite input-adapter fit")
        acceptances.append(float(acceptance))
    with torch.no_grad():
        ell_t, atomic_t, family_t = model.logit(tensor)
        transformed = model.input_transform(tensor).numpy()
    ell, atomic = ell_t.numpy(), atomic_t.numpy()
    families = {name: value.numpy() for name, value in family_t.items()}
    q = 1.0 / (1.0 + np.exp(-np.clip(ell, -700, 700)))
    block_values = []
    for indices in prepared.groups.values():
        usable = [i for i in indices if prepared.feature_names[i] not in prepared.anchor_exclusions]
        if usable:
            block_values.append(X[:, usable].mean(axis=1))
    anchor = np.mean(block_values, axis=0)
    high = float(np.sum(q * anchor) / max(np.sum(q), EPS)); low = float(np.sum((1 - q) * anchor) / max(np.sum(1 - q), EPS))
    orientation = 1 if high - low > 0 else -1
    if orientation < 0:
        q, ell, atomic = 1 - q, -ell, -atomic
        families = {name: -value for name, value in families.items()}
    reconstruction = float(np.max(np.abs(orientation * float(model.b.detach()) + atomic.sum(axis=1) - ell)))
    adapter_diag = {
        "theta_within_norm": float(np.linalg.norm(model.theta_within.detach().numpy())),
        "theta_cross_norm": float(np.linalg.norm(model.theta_cross.detach().numpy())),
        "input_correction_sd": float(np.std(transformed - X)),
        "input_correction_max_abs": float(np.max(np.abs(transformed - X))),
    }
    health = {
        "healthy": bool(np.std(q) >= config.posterior_sd_min and reconstruction <= 1e-8),
        "posterior_sd": float(np.std(q)), "reconstruction": reconstruction,
        "mala_acceptance": float(np.mean(acceptances)), "runtime_seconds": float(time.perf_counter() - started),
        **adapter_diag,
    }
    result = GenericResult(q, ell, atomic, families, orientation * float(model.b.detach()), orientation, health, model.state())
    return result, bases, adapter_diag


__all__ = ["ADAPTER_SCALE", "ARMS", "AdapterBases", "build_adapter_bases", "fit_adapter"]
