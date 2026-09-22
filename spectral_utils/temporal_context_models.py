"""Label-free telemetry predictors and a Diverging Flows adaptation.

DiFlo loss and generated-endpoint DOT: Tsakonas, Ivaldi, Mouret,
arXiv:2602.13061v2, equations 4--8 and Algorithm 2. Conditioning on reasoning
telemetry and interpreting DOT as error evidence are project hypotheses.
No model in this module receives correctness annotations.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def history_windows(values, indices, history=16, include_current=False):
    """Build within-answer contexts; padding has an explicit observed mask."""
    values = np.asarray(values)
    indices = np.asarray(indices, dtype=int)
    if values.ndim != 2 or np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("invalid sequence/indices")
    width = history + int(include_current)
    stop = indices + int(include_current)
    positions = stop[:, None] - np.arange(width, 0, -1)[None, :]
    mask = positions >= 0
    windows = values[np.maximum(positions, 0)].copy()
    windows[~mask] = 0
    return windows, mask


def flow_condition(windows, mask, relative_position):
    """Features, unchanged history mask, and separately declared offline clock."""
    return torch.cat((windows.flatten(1), mask.to(windows.dtype), relative_position.reshape(-1, 1)), dim=1)


class ConditionalFlow(nn.Module):
    def __init__(self, condition_dim, dimensions=4, width=128, depth=3):
        super().__init__()
        layers = []
        incoming = condition_dim + dimensions + 1
        for _ in range(depth):
            layers.extend((nn.Linear(incoming, width), nn.SiLU()))
            incoming = width
        layers.append(nn.Linear(incoming, dimensions))
        self.net = nn.Sequential(*layers)
        self.dimensions = dimensions

    def forward(self, state, tau, condition):
        tau = torch.as_tensor(tau, dtype=state.dtype, device=state.device)
        if tau.numel() == 1:
            tau = tau.expand(len(state)).reshape(-1, 1)
        else:
            tau = tau.reshape(-1, 1)
        return self.net(torch.cat((state, tau, condition), dim=1))


def flow_objective(model, state, tau, velocity_target, condition, negative_condition=None,
                   repel_weight=.1, curve_weight=.1, repel_margin=1., curve_margin=.9):
    positive = model(state, tau, condition)
    fm = (positive - velocity_target).square().sum(dim=-1).mean()
    repel = curve = fm.new_zeros(())
    if negative_condition is not None and (repel_weight or curve_weight):
        negative = model(state, tau, negative_condition)
        positive_error = torch.linalg.vector_norm(velocity_target-positive, dim=-1)
        negative_error = torch.linalg.vector_norm(velocity_target-negative, dim=-1)
        repel = F.relu(positive_error-negative_error+repel_margin).mean()
        positive_distance = 1-F.cosine_similarity(velocity_target, positive, dim=-1, eps=1e-8)
        negative_distance = 1-F.cosine_similarity(velocity_target, negative, dim=-1, eps=1e-8)
        curve = F.relu(positive_distance-negative_distance+curve_margin).mean()
    total = fm + repel_weight*repel + curve_weight*curve
    return total, {"fm": fm.detach(), "repel": repel.detach(), "curve": curve.detach()}


@torch.no_grad()
def flow_dot(model, condition, initial_noise, steps=50, solver="euler"):
    """Integrate and score a generated path, with no observed future argument.

    DOT is the paper's sum over N steps of mean absolute spatial deviation.
    Divide DOT by N when comparing discretization stability across step counts.
    """
    if steps < 2 or solver not in ("euler", "heun"):
        raise ValueError("need >=2 steps and an explicit supported integrator")
    initial = initial_noise.clone()
    state = initial.clone()
    trajectory = []
    for i in range(steps):
        velocity = model(state, i/steps, condition)
        if solver == "euler":
            state = state + velocity/steps
        else:
            trial = state + velocity/steps
            state = state + (velocity + model(trial, (i+1)/steps, condition))/(2*steps)
        if not torch.isfinite(state).all():
            raise FloatingPointError("nonfinite flow path; no mean-score fallback")
        trajectory.append(state.clone())
    dot = state.new_zeros(len(state))
    for i, point in enumerate(trajectory, 1):
        reference = initial + (i/steps)*(state-initial)
        dot += (point-reference).abs().mean(dim=-1)
    return state, dot


def probability_features(logits, signs, epsilon=1e-12):
    """Differentiable q15 features, retaining source-probability constraints.

    The answer's frozen feature-orientation signs are held fixed in PGD;
    a perturbation does not relabel its own coordinates by changing orientation.
    """
    q = torch.softmax(logits, dim=-1)
    surprisal = -(q+epsilon).log()
    columns = [-surprisal.mean(dim=-1)]
    for alpha in (0., .75, 1.):
        weights = torch.ones_like(q)/q.shape[-1] if alpha == 0 else q.pow(alpha)/q.pow(alpha).sum(dim=-1, keepdim=True)
        mean = (weights*surprisal).sum(dim=-1)
        columns.append((weights*surprisal.square()).sum(dim=-1)-mean.square())
    return torch.stack(columns, dim=-1) * signs


def probability_pgd(model, state, tau, velocity_target, source_logits, make_condition,
                    epsilon=.1, iterations=3):
    """Mine negatives in logit coordinates, recomputing valid q-derived features.

    ``make_condition`` must preserve position/mask and apply frozen training
    normalization. This is our constrained adaptation of paper Eq.7.
    """
    if epsilon < 0 or iterations < 1:
        raise ValueError("invalid PGD budget")
    origin = source_logits.detach()
    adversary = origin.clone()
    for _ in range(iterations):
        adversary.requires_grad_(True)
        condition = make_condition(adversary)
        error = (model(state.detach(), tau, condition)-velocity_target.detach()).square().sum(dim=-1).mean()
        gradient, = torch.autograd.grad(error, adversary)
        adversary = adversary.detach() + (epsilon/iterations)*gradient.sign()
        adversary = torch.maximum(torch.minimum(adversary, origin+epsilon), origin-epsilon)
    return make_condition(adversary.detach()).detach(), adversary.detach()


class CausalBlock(nn.Module):
    def __init__(self, incoming, width, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(incoming, width, kernel_size=3, dilation=dilation)
        self.residual = nn.Identity() if incoming == width else nn.Conv1d(incoming, width, 1)

    def forward(self, x):
        return F.silu(self.conv(F.pad(x, (2*self.dilation, 0))) + self.residual(x))


class TelemetryTCN(nn.Module):
    """Predict current features from past-only windows, with Gaussian variance."""
    def __init__(self, dimensions=4, width=32):
        super().__init__()
        self.blocks = nn.Sequential(CausalBlock(dimensions+1, width, 1),
                                    CausalBlock(width, width, 2), CausalBlock(width, width, 4))
        self.head = nn.Linear(width+1, 2*dimensions)
        self.dimensions = dimensions

    def forward(self, history, mask, position):
        x = torch.cat((history, mask.to(history.dtype).unsqueeze(-1)), dim=-1).transpose(1, 2)
        last = self.blocks(x)[:, :, -1]
        mean, raw_variance = self.head(torch.cat((last, position.reshape(-1,1)), dim=1)).chunk(2, dim=-1)
        variance = F.softplus(raw_variance) + 1e-4
        return mean, variance


def gaussian_prediction_loss(mean, variance, target):
    return .5 * ((target-mean).square()/variance + variance.log()).sum(dim=-1).mean()


def residual_step_score(base, auxiliary, gamma):
    """Retain original score units; constant auxiliary contributes exactly zero."""
    base, auxiliary = np.asarray(base, float), np.asarray(auxiliary, float)
    if base.shape != auxiliary.shape or not np.isfinite(base).all() or not np.isfinite(auxiliary).all():
        raise ValueError("invalid residual scores")
    if gamma == 0 or np.std(auxiliary) <= 1e-12:
        return base.copy()
    return base + float(gamma)*np.std(base)*(auxiliary-auxiliary.mean())/auxiliary.std()


@dataclass
class RidgePredictor:
    coefficient: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, contexts, observations, ridge=1.):
        contexts, observations = np.asarray(contexts, float), np.asarray(observations, float)
        if len(contexts) < 2 or ridge <= 0 or not np.isfinite(contexts).all() or not np.isfinite(observations).all():
            raise ValueError("invalid predictor training data")
        mean = contexts.mean(axis=0); scale = np.maximum(contexts.std(axis=0), 1e-8)
        design = np.column_stack(((contexts-mean)/scale, np.ones(len(contexts))))
        penalty = np.eye(design.shape[1])*ridge; penalty[-1,-1] = 0
        coefficient = np.linalg.solve(design.T@design + penalty, design.T@observations)
        return cls(coefficient, mean, scale)

    def predict(self, contexts):
        design = np.column_stack(((np.asarray(contexts)-self.mean)/self.scale, np.ones(len(contexts))))
        return design@self.coefficient
