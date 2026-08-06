"""Isolated parameter-free adapted-DUFS gate learner.

The repository's historical implementation lives inside the selector registry,
whose package initializer imports many unrelated selector families.  This small
module extracts only the exact DUFS Eq.-7 adaptation used by the Laplacian
experiments so the executed local dependency surface can be frozen completely.

This remains an adaptation of Lindenbaum et al. (2021): self-tuning k=7 kernel,
Adam with a CPU-affordable 120-epoch default, and seed averaging.  It is not
claimed to be the paper's original optimizer or bandwidth configuration.
"""

import numpy as np
import torch
from scipy.special import ndtr


K_NN = 7
DIFFUSION_T = 2
STG_SIGMA = 0.5
GATE_LR = 2e-2
MU_INIT = 0.5
BATCH = 256
EPOCHS_STAB = 120
PF_DELTA = 1e-8
_SQRT2 = 1.4142135623730951
_EPS = 1e-8


def _self_tuning_affinity(points, k):
    distances_squared = torch.cdist(points, points) ** 2
    k = int(max(1, min(k, points.shape[0] - 1)))
    kth_squared = torch.topk(
        distances_squared, k + 1, largest=False
    ).values[:, -1]
    scale = torch.sqrt(kth_squared.clamp_min(_EPS))
    affinity = torch.exp(
        -distances_squared / (scale[:, None] * scale[None, :] + _EPS)
    )
    return affinity - torch.diag(torch.diagonal(affinity))


def _random_walk_power(affinity, steps):
    walk = affinity / affinity.sum(1, keepdim=True).clamp_min(_EPS)
    output = walk
    for _ in range(steps - 1):
        output = output @ walk
    return output


def _train_parameter_free(features, epochs, batch, seed):
    torch.manual_seed(int(seed))
    generator = torch.Generator().manual_seed(int(seed))
    n, d = features.shape
    batch = int(min(batch, n))
    graph_k = int(min(K_NN, batch - 1))
    means = torch.full(
        (d,), MU_INIT, dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam([means], lr=GATE_LR)
    for _ in range(int(epochs)):
        indices = torch.randperm(n, generator=generator)[:batch]
        batch_features = features[indices]
        stochastic_gates = torch.clamp(
            means + torch.randn(d, generator=generator) * STG_SIGMA,
            0.0,
            1.0,
        )
        gated = batch_features * stochastic_gates[None, :]
        walk = _random_walk_power(
            _self_tuning_affinity(gated, graph_k), DIFFUSION_T
        )
        trace = -(gated * (walk @ gated)).sum() / batch
        survival = 0.5 * (
            1.0 + torch.erf(means / (STG_SIGMA * _SQRT2))
        )
        loss = trace / (survival.sum() + PF_DELTA)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return means.detach().numpy()


def adapted_dufs_soft_gates(F, *, seeds=(0, 1, 2), epochs=None):
    """Return continuous survival probabilities and seed diagnostics."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] < 3 or F.shape[1] < 3:
        raise ValueError("F must have shape (features>=3, samples>=3)")
    if not np.isfinite(F).all():
        raise ValueError("F contains non-finite values")
    epochs = EPOCHS_STAB if epochs is None else int(epochs)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(1)
    features = torch.tensor(F.T, dtype=torch.float32)
    per_seed = np.asarray([
        ndtr(
            np.asarray(
                _train_parameter_free(features, epochs, BATCH, int(seed)),
                dtype=float,
            ) / STG_SIGMA
        )
        for seed in seeds
    ], dtype=float)
    raw = per_seed.mean(axis=0)
    rms = float(np.sqrt(np.mean(raw ** 2)))
    gates = raw / (rms if rms > 1e-12 else 1.0)
    diagnostics = {
        "raw_probabilities": raw,
        "per_seed_probabilities": per_seed,
        "mean_probability": float(raw.mean()),
        "near_zero_fraction": float(np.mean(raw < 0.05)),
        "near_one_fraction": float(np.mean(raw > 0.95)),
        "effective_feature_count": float(
            (raw.sum() ** 2) / (np.sum(raw ** 2) + 1e-12)
        ),
        "mean_seed_std": float(np.mean(per_seed.std(axis=0))),
    }
    return gates, diagnostics


__all__ = [
    "BATCH",
    "EPOCHS_STAB",
    "STG_SIGMA",
    "adapted_dufs_soft_gates",
]
