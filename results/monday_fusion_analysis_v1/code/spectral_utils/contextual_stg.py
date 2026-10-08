"""Small CPU implementation of an oriented conditional stochastic-gate router.

The implementation follows the c-STG Gaussian relaxation: a hypernetwork maps
context variables to gate locations, Gaussian noise is added during training,
and the result is clipped to ``[0, 1]``.  This project-specific diagnostic is
deliberately more restrictive than weighted c-STG: explanatory-feature signs
are frozen by constraining the prediction weights to be non-negative.  The
context can therefore change reliability/leverage, but cannot reverse a risk
feature's declared direction or enter the prediction through a direct path.

This is a supervised diagnostic, not a deployable label-free fusion method.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


EPS = 1e-12


def _finite_2d(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty 2-D array")
    return array


def _weighted_location_scale(
    values: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + EPS)
    mean = weights @ values
    variance = weights @ ((values - mean) ** 2)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return mean, scale


def balanced_group_weights(labels, groups) -> np.ndarray:
    """Give every source question and both target classes equal total mass."""

    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=str)
    if labels.ndim != 1 or groups.shape != labels.shape:
        raise ValueError("labels and groups must be aligned vectors")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary labels must contain both classes")
    _, group_codes = np.unique(groups, return_inverse=True)
    counts = np.bincount(group_codes)
    weights = 1.0 / counts[group_codes]
    for target in (0, 1):
        mask = labels == target
        weights[mask] *= 0.5 / (weights[mask].sum() + EPS)
    return weights / (weights.sum() + EPS)


@dataclass(frozen=True)
class ContextualSTGConfig:
    hidden_dim: int = 16
    sigma: float = 0.5
    sparsity: float = 0.01
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    epochs: int = 600
    minimum_epochs: int = 200
    patience: int = 100


@dataclass(frozen=True)
class ContextualSTGPrediction:
    score: np.ndarray
    family_gates: np.ndarray
    feature_gates: np.ndarray


class ContextualSTGModel:
    """Context-conditioned sparse gates with a positive linear risk head."""

    def __init__(self, config: ContextualSTGConfig | None = None):
        self.config = config or ContextualSTGConfig()
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.z_mean_: np.ndarray | None = None
        self.z_scale_: np.ndarray | None = None
        self.feature_group_ids_: np.ndarray | None = None
        self.state_dict_: dict[str, np.ndarray] | None = None
        self.diagnostics_: dict[str, object] | None = None

    def fit(
        self,
        explanatory,
        context,
        labels,
        groups,
        *,
        feature_group_ids: Sequence[int] | None = None,
        seed: int = 0,
    ) -> "ContextualSTGModel":
        try:
            import torch
            from torch import nn
            from torch.nn import functional as F
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("ContextualSTGModel requires PyTorch") from exc

        X = _finite_2d(explanatory, name="explanatory")
        Z = _finite_2d(context, name="context")
        y = np.asarray(labels, dtype=np.int64)
        groups = np.asarray(groups, dtype=str)
        if len(Z) != len(X) or y.shape != (len(X),) or groups.shape != y.shape:
            raise ValueError("explanatory/context/labels/groups disagree")
        weights = balanced_group_weights(y, groups)
        if feature_group_ids is None:
            feature_group_ids = np.arange(X.shape[1], dtype=np.int64)
        feature_group_ids = np.asarray(feature_group_ids, dtype=np.int64)
        if feature_group_ids.shape != (X.shape[1],) or np.min(feature_group_ids) < 0:
            raise ValueError("feature_group_ids must identify every explanatory column")
        unique_groups = np.unique(feature_group_ids)
        if not np.array_equal(unique_groups, np.arange(len(unique_groups))):
            raise ValueError("feature_group_ids must be contiguous from zero")

        self.x_mean_, self.x_scale_ = _weighted_location_scale(X, weights)
        self.z_mean_, self.z_scale_ = _weighted_location_scale(Z, weights)
        Xs = np.clip((X - self.x_mean_) / self.x_scale_, -8.0, 8.0)
        Zs = np.clip((Z - self.z_mean_) / self.z_scale_, -8.0, 8.0)
        self.feature_group_ids_ = feature_group_ids.copy()

        torch.manual_seed(int(seed))
        torch.set_num_threads(1)
        x_tensor = torch.as_tensor(Xs, dtype=torch.float32)
        z_tensor = torch.as_tensor(Zs, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
        group_tensor = torch.as_tensor(feature_group_ids, dtype=torch.long)
        n_families = len(unique_groups)
        config = self.config

        class GateNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.hyper = nn.Sequential(
                    nn.Linear(Z.shape[1], config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, n_families),
                )
                nn.init.xavier_uniform_(self.hyper[0].weight)
                nn.init.zeros_(self.hyper[0].bias)
                nn.init.zeros_(self.hyper[2].weight)
                nn.init.constant_(self.hyper[2].bias, 0.5)
                self.raw_feature_weight = nn.Parameter(torch.zeros(X.shape[1]))
                self.bias = nn.Parameter(torch.zeros(()))

            def forward(self, x, z, *, noisy: bool):
                mu = self.hyper(z)
                noise = torch.randn_like(mu) * config.sigma if noisy else 0.0
                family_gates = torch.clamp(mu + noise, 0.0, 1.0)
                feature_gates = family_gates[:, group_tensor]
                positive = F.softplus(self.raw_feature_weight)
                positive = positive * (len(positive) / (positive.sum() + 1e-8))
                logits = (
                    (x * feature_gates * positive[None, :]).sum(dim=1)
                    / math.sqrt(x.shape[1])
                    + self.bias
                )
                return logits, mu, family_gates, feature_gates

        network = GateNetwork()
        optimizer = torch.optim.Adam(
            network.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best_loss = float("inf")
        best_state = None
        stale = 0
        epochs_run = 0
        for epoch in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits, mu, _, _ = network(x_tensor, z_tensor, noisy=True)
            predictive = (
                F.binary_cross_entropy_with_logits(logits, y_tensor, reduction="none")
                * weight_tensor
            ).sum()
            open_probability = 0.5 * (
                1.0 + torch.erf(mu / (config.sigma * math.sqrt(2.0)))
            )
            loss = predictive + config.sparsity * open_probability.mean()
            loss.backward()
            optimizer.step()
            epochs_run = epoch + 1
            current = float(loss.detach())
            if current < best_loss - 1e-7:
                best_loss = current
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if epoch + 1 >= config.minimum_epochs and stale >= config.patience:
                break
        if best_state is None:
            raise RuntimeError("c-STG optimization produced no finite state")
        network.load_state_dict(best_state)
        self.state_dict_ = {
            name: value.numpy().copy() for name, value in best_state.items()
        }
        network.eval()
        with torch.no_grad():
            logits, mu, family_gates, _ = network(x_tensor, z_tensor, noisy=False)
            fitted_loss = float((
                F.binary_cross_entropy_with_logits(logits, y_tensor, reduction="none")
                * weight_tensor
            ).sum())
            positive = F.softplus(network.raw_feature_weight)
            positive = positive * (len(positive) / (positive.sum() + 1e-8))
        self.diagnostics_ = {
            "supervised_diagnostic": True,
            "labels_seen_during_fit": True,
            "context_direct_prediction_path": False,
            "feature_weights_constrained_nonnegative": True,
            "seed": int(seed),
            "epochs_run": int(epochs_run),
            "best_objective": float(best_loss),
            "balanced_predictive_loss": fitted_loss,
            "mean_family_gate": family_gates.mean(dim=0).numpy().tolist(),
            "std_family_gate": family_gates.std(dim=0).numpy().tolist(),
            "feature_weights": positive.numpy().tolist(),
            "config": vars(config),
        }
        return self

    def predict(self, explanatory, context) -> ContextualSTGPrediction:
        if self.state_dict_ is None or self.feature_group_ids_ is None:
            raise RuntimeError("fit must be called before predict")
        import torch
        from torch import nn
        from torch.nn import functional as F

        X = _finite_2d(explanatory, name="explanatory")
        Z = _finite_2d(context, name="context")
        if len(X) != len(Z) or X.shape[1] != len(self.feature_group_ids_):
            raise ValueError("prediction matrices disagree with fitted dimensions")
        Xs = np.clip((X - self.x_mean_) / self.x_scale_, -8.0, 8.0)
        Zs = np.clip((Z - self.z_mean_) / self.z_scale_, -8.0, 8.0)
        n_families = int(np.max(self.feature_group_ids_)) + 1
        config = self.config

        class PredictionNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.hyper = nn.Sequential(
                    nn.Linear(Z.shape[1], config.hidden_dim), nn.ReLU(),
                    nn.Linear(config.hidden_dim, n_families),
                )
                self.raw_feature_weight = nn.Parameter(torch.zeros(X.shape[1]))
                self.bias = nn.Parameter(torch.zeros(()))

        network = PredictionNetwork()
        network.load_state_dict({
            name: torch.as_tensor(value) for name, value in self.state_dict_.items()
        })
        network.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(Xs, dtype=torch.float32)
            z_tensor = torch.as_tensor(Zs, dtype=torch.float32)
            mu = network.hyper(z_tensor)
            family_gates = torch.clamp(mu, 0.0, 1.0)
            group_tensor = torch.as_tensor(self.feature_group_ids_, dtype=torch.long)
            feature_gates = family_gates[:, group_tensor]
            positive = F.softplus(network.raw_feature_weight)
            positive = positive * (len(positive) / (positive.sum() + 1e-8))
            logits = (
                (x_tensor * feature_gates * positive[None, :]).sum(dim=1)
                / math.sqrt(X.shape[1])
                + network.bias
            )
        return ContextualSTGPrediction(
            score=logits.numpy().astype(np.float64),
            family_gates=family_gates.numpy().astype(np.float64),
            feature_gates=feature_gates.numpy().astype(np.float64),
        )


__all__ = [
    "ContextualSTGConfig",
    "ContextualSTGModel",
    "ContextualSTGPrediction",
    "balanced_group_weights",
]
