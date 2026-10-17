"""Fixed crossed source×operator innovation layer before continuous B3.

The layer is estimated without labels.  For each coordinate in the universal
3×3 core it predicts that coordinate from either its rook peers (same source
or same operator) or a cardinality-matched non-rook control, and passes a
standardized innovation to the unchanged B3 family network.  The map is
linear, fixed during EBM fitting, and the visible quadratic remains on the
original input, so the energy keeps Gaussian tails.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .deem_b3_contract_ablation import GenericEnergy, GenericResult, PreparedArm
from .residual_graph_deem import ContinuousDeemConfig, EPS, persistent_mala, set_determinism


ARMS = ("K1_ROOK_INNOVATION", "K2_NONROOK_INNOVATION")
RIDGE = 1.0
CORE_GRID = (
    ("entropy_common", "sw_var_peak", "cusum_max"),
    ("epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled"),
    ("epr_energy", "sw_var_peak_energy", "cusum_max_energy"),
)


@dataclass(frozen=True)
class InnovationMap:
    matrix: np.ndarray
    output_scale: np.ndarray
    coefficients: np.ndarray
    diagnostics: dict[str, float]


def build_innovation_map(prepared: PreparedArm, arm: str) -> InnovationMap:
    if arm not in ARMS:
        raise KeyError(arm)
    X = np.asarray(prepared.X_risk, dtype=np.float64)
    lookup = {name: i for i, name in enumerate(prepared.feature_names)}
    if any(name not in lookup for row in CORE_GRID for name in row):
        raise ValueError("universal crossed core is incomplete")
    core = [(r, c, lookup[name]) for r, row in enumerate(CORE_GRID) for c, name in enumerate(row)]
    matrix = np.eye(X.shape[1], dtype=np.float64)
    coefficients = np.zeros((9, 9), dtype=np.float64)
    target_r2 = []
    for target_position, (target_row, target_col, target_index) in enumerate(core):
        if arm == "K1_ROOK_INNOVATION":
            peers = [(position, index) for position, (row, col, index) in enumerate(core)
                     if position != target_position and (row == target_row or col == target_col)]
        else:
            peers = [(position, index) for position, (row, col, index) in enumerate(core)
                     if row != target_row and col != target_col]
        if len(peers) != 4:
            raise AssertionError("each core coordinate must have four peers")
        peer_positions = [position for position, _ in peers]
        peer_indices = [index for _, index in peers]
        P, y = X[:, peer_indices], X[:, target_index]
        gram = P.T @ P / len(X) + RIDGE * np.eye(len(peer_indices))
        beta = np.linalg.solve(gram, P.T @ y / len(X))
        matrix[target_index, peer_indices] -= beta
        coefficients[target_position, peer_positions] = beta
        prediction = P @ beta
        target_r2.append(1.0 - float(np.mean((y - prediction) ** 2)) / max(float(np.var(y)), EPS))
    transformed = X @ matrix.T
    output_scale = np.ones(X.shape[1], dtype=np.float64)
    for _, _, index in core:
        output_scale[index] = max(float(np.std(transformed[:, index])), EPS)
    normalized = transformed / output_scale[None, :]
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e6:
        raise ValueError(f"ill-conditioned innovation map: {condition}")
    return InnovationMap(
        matrix=matrix,
        output_scale=output_scale,
        coefficients=coefficients,
        diagnostics={
            "mean_target_r2": float(np.mean(target_r2)),
            "min_target_r2": float(np.min(target_r2)),
            "max_target_r2": float(np.max(target_r2)),
            "matrix_condition": condition,
            "mean_absolute_input_change": float(np.mean(np.abs(normalized - X))),
        },
    )


class InnovationEnergy(GenericEnergy):
    def __init__(self, prepared: PreparedArm, config: ContinuousDeemConfig, seed: int, innovation: InnovationMap):
        super().__init__(prepared.feature_names, prepared.groups, config, seed)
        self.input_matrix = self.torch.as_tensor(innovation.matrix, dtype=self.torch.float64)
        self.output_scale = self.torch.as_tensor(innovation.output_scale, dtype=self.torch.float64)

    def input_transform(self, X):
        return (X @ self.input_matrix.T) / self.output_scale

    def contributions(self, X):
        return super().contributions(self.input_transform(X))

    def state(self):
        output = super().state()
        output["input::matrix"] = self.input_matrix.detach().numpy().copy()
        output["input::output_scale"] = self.output_scale.detach().numpy().copy()
        return output


def fit_crossed_innovation(prepared: PreparedArm, arm: str, *, seed: int = 0,
                           config: ContinuousDeemConfig | None = None):
    import torch

    config = config or ContinuousDeemConfig()
    innovation = build_innovation_map(prepared, arm)
    X = prepared.X_risk
    started = time.perf_counter()
    set_determinism(seed)
    model = InnovationEnergy(prepared, config, seed, innovation)
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
        buffer, acceptance = persistent_mala(model, buffer, delta=config.mala_delta,
                                             steps=config.mala_steps, generator=generator)
        loss = model.free_energy(tensor).mean() - model.free_energy(buffer).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(p).all()) for p in parameters):
            raise FloatingPointError("non-finite crossed-innovation fit")
        acceptances.append(float(acceptance))
    with torch.no_grad():
        ell_t, atomic_t, family_t = model.logit(tensor)
    ell, atomic = ell_t.numpy(), atomic_t.numpy()
    families = {name: value.numpy() for name, value in family_t.items()}
    q = 1.0 / (1.0 + np.exp(-np.clip(ell, -700, 700)))
    block_values = []
    for indices in prepared.groups.values():
        usable = [i for i in indices if prepared.feature_names[i] not in prepared.anchor_exclusions]
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
        **innovation.diagnostics,
    }
    result = GenericResult(q, ell, atomic, families, orientation * float(model.b.detach()),
                           orientation, health, model.state())
    return result, innovation


__all__ = ["ARMS", "CORE_GRID", "RIDGE", "build_innovation_map", "fit_crossed_innovation"]
