"""Adapted SpecRaGE graph learning for Laplacian-regularized IU-PCR.

This module independently implements the equations in Yacobi, Lindenbaum &
Shaham (2025), *Generalizable and Robust Spectral Method for Multi-view
Representation Learning*.  It deliberately exposes only label-free inputs.

For view ``v`` and sample ``i``, a view encoder produces ``y_i^(v)`` and a
fusion network produces simplex weights ``alpha_i^(v)``.  The fused embedding
and reliability-weighted affinities are

    y_i = sum_v alpha_i^(v) y_i^(v)
    W_tilde_ij^(v) = W_ij^(v) alpha_i^(v) alpha_j^(v).

The encoders and weights minimize the SpecRaGE Rayleigh loss under an audited
orthogonality layer.  The original IU-PCR adaptation used the aggregate
weighted graph

    W_SR = (1 / V) sum_v W_tilde^(v)

as the graph supplied to :mod:`spectral_utils.laplacian_upcr`. V2 also exposes
a seed-averaged rotation-invariant k-NN graph on SpecRaGE's actual fused
embedding ``Y``. The two interfaces are kept separate and must not be reported
as the same method. That downstream module constructs a symmetric normalized
Laplacian and changes only IU-PCR's final projected weight equation. The
ordinary covariance, rho estimate and two-dimensional IU-PCR head remain
untouched.

V2 optionally adds a label-free cross-view diffusion-agreement target and an
edge-mass constraint. Those terms are a SpecRaGE-derived extension, not part of
the unchanged paper method. They are disabled when their registered strengths
are zero.

The paper optionally learns affinities with Siamese networks.  The first
registered version here uses the paper's Gaussian k-NN affinity (Eq. 5) on
standardized, provenance-defined low-dimensional views.  This avoids adding a
second learnable metric before the sample-specific fusion mechanism itself has
been falsified.  It is therefore an adapted SpecRaGE core, not a byte-for-byte
port of the authors' implementation.

No function in this module accepts correctness labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from .laplacian_upcr import graph_diagnostics, symmetric_normalized_laplacian


EPS = 1e-12


@dataclass(frozen=True)
class SpecRaGEConfig:
    """Registered numerical and scientific settings for one graph fit."""

    output_dim: int = 2
    n_neighbors: int = 7
    temperature: float = 10.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 45
    min_epochs: int = 15
    patience: int = 10
    lr_patience: int = 5
    min_delta: float = 1e-6
    encoder_hidden: tuple[int, ...] = (32, 16)
    fusion_hidden: tuple[int, ...] = (32, 16)
    gradient_clip: float = 5.0
    checkpoint_mode: str = "final"
    orthogonalization: str = "qr"
    orthogonal_floor: float = 1e-4
    agreement_strength: float = 0.0
    agreement_temperature: float = 0.08
    edge_mass_strength: float = 0.0
    fusion_mode: str = "sample"
    view_mass_normalization: bool = False
    fit_sample_cap: int | None = None

    def validate(self) -> "SpecRaGEConfig":
        if self.output_dim < 1:
            raise ValueError("output_dim must be positive")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.batch_size < 4:
            raise ValueError("batch_size must be at least four")
        if self.max_epochs < 1 or self.min_epochs < 1:
            raise ValueError("epoch counts must be positive")
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs cannot exceed max_epochs")
        if self.patience < 1 or self.lr_patience < 1:
            raise ValueError("patience values must be positive")
        if any(int(width) < 1 for width in (*self.encoder_hidden, *self.fusion_hidden)):
            raise ValueError("hidden widths must be positive")
        if self.checkpoint_mode not in ("final", "best_unlabeled"):
            raise ValueError("checkpoint_mode must be 'final' or 'best_unlabeled'")
        if self.orthogonalization not in ("qr", "svd_floor"):
            raise ValueError("orthogonalization must be 'qr' or 'svd_floor'")
        if not np.isfinite(self.orthogonal_floor) or self.orthogonal_floor <= 0:
            raise ValueError("orthogonal_floor must be finite and positive")
        if not np.isfinite(self.agreement_strength) or self.agreement_strength < 0:
            raise ValueError("agreement_strength must be finite and nonnegative")
        if not np.isfinite(self.agreement_temperature) or self.agreement_temperature <= 0:
            raise ValueError("agreement_temperature must be finite and positive")
        if not np.isfinite(self.edge_mass_strength) or self.edge_mass_strength < 0:
            raise ValueError("edge_mass_strength must be finite and nonnegative")
        if self.fusion_mode not in ("sample", "global", "uniform"):
            raise ValueError("fusion_mode must be 'sample', 'global', or 'uniform'")
        if self.fusion_mode != "sample" and self.agreement_strength > 0:
            raise ValueError("agreement targets require sample-specific fusion")
        if not isinstance(self.view_mass_normalization, bool):
            raise ValueError("view_mass_normalization must be boolean")
        if self.fit_sample_cap is not None and int(self.fit_sample_cap) < 32:
            raise ValueError("fit_sample_cap must be at least 32 or None")
        return self

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SpecRaGESeedResult:
    """Output from one independently initialized SpecRaGE fit."""

    seed: int
    alpha: np.ndarray
    embedding: np.ndarray
    graph: csr_matrix
    embedding_graph: csr_matrix
    history: list[dict]
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SpecRaGEGraphResult:
    """Seed-ensembled graph and the diagnostics required for auditing it."""

    graph: csr_matrix
    embedding_graph: csr_matrix
    alpha: np.ndarray
    view_names: tuple[str, ...]
    base_graphs: tuple[csr_matrix, ...]
    seed_results: tuple[SpecRaGESeedResult, ...]
    config: SpecRaGEConfig
    view_prior: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale < EPS] = 1.0
    return centered / scale


def prepare_views(views: Mapping[str, np.ndarray]):
    """Validate and standardize a named sample-by-coordinate view mapping."""
    if not isinstance(views, Mapping) or len(views) < 2:
        raise ValueError("at least two named views are required")
    names = tuple(str(name) for name in views)
    if len(set(names)) != len(names):
        raise ValueError("view names must be unique")
    matrices = []
    n_samples = None
    for name, raw in views.items():
        matrix = np.asarray(raw, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[1] < 1:
            raise ValueError(
                f"view {name!r} must have shape (samples>=4, coordinates>=1)"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"view {name!r} contains non-finite values")
        if n_samples is None:
            n_samples = matrix.shape[0]
        elif matrix.shape[0] != n_samples:
            raise ValueError("all views must contain the same samples in the same order")
        matrices.append(_standardize_columns(matrix))
    return names, tuple(matrices)


def gaussian_knn_affinity(
    samples: np.ndarray,
    *,
    n_neighbors: int = 7,
    tie_keys: np.ndarray | None = None,
):
    """Paper Eq. 5 with a global median k-NN scale and max symmetrization."""
    from .graph_topology import _knn_table

    samples = _standardize_columns(np.asarray(samples, dtype=float))
    n = samples.shape[0]
    k = int(max(1, min(int(n_neighbors), n - 1)))
    neighbour_distances, indices, tie_diagnostics = _knn_table(
        samples, k, tie_keys=tie_keys
    )
    positive = neighbour_distances[neighbour_distances > EPS]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-8)
    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)
    values = np.exp(
        -(neighbour_distances.reshape(-1) ** 2) / (2.0 * sigma * sigma)
    )
    directed = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    graph = directed.maximum(directed.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph, {
        "n_neighbors": k,
        "sigma": sigma,
        **tie_diagnostics,
    }


def embedding_knn_affinity(
    embedding: np.ndarray,
    *,
    n_neighbors: int = 7,
    tie_keys: np.ndarray | None = None,
):
    """Build a rotation-invariant graph from SpecRaGE's fused embedding.

    SpecRaGE's scientific output is a joint spectral representation.  Its
    columns may rotate or change sign between seeds, so the seed-safe object is
    a graph built from Euclidean distances in each seed's embedding, followed
    by graph averaging.  Unlike :func:`gaussian_knn_affinity`, this helper uses
    one scalar scale rather than column-wise standardization, preserving those
    Euclidean distances under orthogonal rotations.
    """
    from .graph_topology import _knn_table

    values = np.asarray(embedding, dtype=float)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 1:
        raise ValueError("embedding must have shape (samples>=4, coordinates>=1)")
    if not np.isfinite(values).all():
        raise ValueError("embedding contains non-finite values")
    values = values - values.mean(axis=0, keepdims=True)
    scalar = float(np.sqrt(np.mean(values ** 2)))
    if scalar > EPS:
        values = values / scalar
    n = values.shape[0]
    k = int(max(1, min(int(n_neighbors), n - 1)))
    neighbour_distances, indices, tie_diagnostics = _knn_table(
        values, k, tie_keys=tie_keys
    )
    positive = neighbour_distances[neighbour_distances > EPS]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-8)
    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)
    weights = np.exp(-(neighbour_distances.reshape(-1) ** 2) / (sigma * sigma))
    directed = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    graph = ((directed + directed.T) * 0.5).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph, {
        "n_neighbors": k,
        "sigma": sigma,
        **tie_diagnostics,
    }


def base_view_graphs(
    views: Sequence[np.ndarray],
    *,
    n_neighbors: int,
    tie_keys: np.ndarray | None = None,
):
    graphs, metadata = [], []
    for view in views:
        graph, details = gaussian_knn_affinity(
            view, n_neighbors=n_neighbors, tie_keys=tie_keys
        )
        graphs.append(graph)
        metadata.append(details)
    return tuple(graphs), tuple(metadata)


def cross_view_agreement_targets(
    base_graphs: Sequence[csr_matrix], *, temperature: float = 0.08,
    view_prior: np.ndarray | None = None,
):
    """Return label-free per-sample reliability targets from view consensus.

    Each affinity is converted to a one-step/two-step diffusion profile.  A
    view is reliable for a sample when that profile agrees with the profiles
    from the other views.  With three or more views this implements the
    majority-agreement assumption needed to identify a conditionally corrupted
    view.  With exactly two views the scores are necessarily symmetric and the
    function correctly returns uniform targets rather than pretending that the
    clean side is identifiable.
    """
    graphs = tuple(csr_matrix(graph, dtype=float) for graph in base_graphs)
    if len(graphs) < 2:
        raise ValueError("at least two graphs are required")
    n = graphs[0].shape[0]
    if any(graph.shape != (n, n) for graph in graphs):
        raise ValueError("base graphs must be shape-matched")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    profiles = []
    for graph in graphs:
        row_sum = np.asarray(graph.sum(axis=1)).ravel()
        transition = diags(1.0 / np.maximum(row_sum, EPS)) @ graph
        profile = 0.5 * transition + 0.5 * (transition @ transition)
        norm = np.sqrt(np.asarray(profile.multiply(profile).sum(axis=1)).ravel())
        profiles.append((diags(1.0 / np.maximum(norm, EPS)) @ profile).tocsr())
    n_views = len(profiles)
    if view_prior is None:
        prior = np.full(n_views, 1.0 / n_views)
    else:
        prior = np.asarray(view_prior, dtype=float)
        if prior.shape != (n_views,) or not np.isfinite(prior).all() \
                or np.any(prior <= 0):
            raise ValueError("view_prior must contain one positive finite value per view")
        prior = prior / prior.sum()
    agreement = np.zeros((n, n_views), dtype=float)
    for view in range(n_views):
        similarities, weights = [], []
        for other in range(n_views):
            if other != view:
                similarities.append(np.asarray(
                    profiles[view].multiply(profiles[other]).sum(axis=1)
                ).ravel())
                weights.append(prior[other])
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        agreement[:, view] = np.average(similarities, axis=0, weights=weights)
    if n_views == 2:
        targets = np.repeat(prior[None, :], n, axis=0)
    else:
        logits = agreement / temperature + np.log(prior[None, :])
        logits -= logits.max(axis=1, keepdims=True)
        targets = np.exp(logits)
        targets /= targets.sum(axis=1, keepdims=True)
    return targets, agreement


def weighted_multiview_graph(
    base_graphs: Sequence[csr_matrix],
    alpha: np.ndarray,
    *,
    view_prior: np.ndarray | None = None,
    mass_normalize: bool = False,
):
    """Apply SpecRaGE's edge weights and average the view affinities."""
    graphs = tuple(csr_matrix(graph, dtype=float) for graph in base_graphs)
    if len(graphs) < 2:
        raise ValueError("at least two base graphs are required")
    n = graphs[0].shape[0]
    if any(graph.shape != (n, n) for graph in graphs):
        raise ValueError("base graphs must be square and shape-matched")
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (n, len(graphs)):
        raise ValueError("alpha must have shape (samples, views)")
    if not np.isfinite(alpha).all() or np.any(alpha < -1e-10):
        raise ValueError("alpha must be finite and nonnegative")
    row_sum = alpha.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("every alpha row must lie on the probability simplex")

    if view_prior is None:
        prior = np.full(len(graphs), 1.0 / len(graphs))
    else:
        prior = np.asarray(view_prior, dtype=float)
        if prior.shape != (len(graphs),) or not np.isfinite(prior).all() \
                or np.any(prior <= 0):
            raise ValueError("view_prior must contain one positive finite value per view")
        prior = prior / prior.sum()

    fused = csr_matrix((n, n), dtype=float)
    for view_index, graph in enumerate(graphs):
        if mass_normalize:
            reliability = np.maximum(alpha[:, view_index], 0.0) / prior[view_index]
            weights = diags(reliability)
            fused = fused + prior[view_index] * (weights @ graph @ weights)
        else:
            weights = diags(np.maximum(alpha[:, view_index], 0.0))
            fused = fused + weights @ graph @ weights
    if not mass_normalize:
        fused = fused / len(graphs)
    fused = fused.tocsr()
    fused = fused.maximum(fused.T).tocsr()
    fused.setdiag(0.0)
    fused.eliminate_zeros()
    return fused


def control_alpha(
    alpha: np.ndarray, mode: str, *, seed: int = 0,
    view_prior: np.ndarray | None = None,
):
    """Construct registered controls while preserving the alpha simplex."""
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim != 2:
        raise ValueError("alpha must be two-dimensional")
    n, n_views = alpha.shape
    prior = (
        np.full(n_views, 1.0 / n_views) if view_prior is None
        else np.asarray(view_prior, dtype=float)
    )
    if prior.shape != (n_views,) or not np.isfinite(prior).all() \
            or np.any(prior <= 0):
        raise ValueError("view_prior must contain one positive finite value per view")
    prior = prior / prior.sum()
    if mode == "sample_specific":
        return alpha.copy()
    if mode == "uniform":
        return np.repeat(prior[None, :], n, axis=0)
    if mode == "global":
        global_weight = alpha.mean(axis=0)
        global_weight /= global_weight.sum()
        return np.repeat(global_weight[None, :], n, axis=0)
    if mode == "permuted":
        permutation = np.random.default_rng(int(seed)).permutation(n)
        return alpha[permutation].copy()
    raise ValueError(f"unknown alpha control: {mode}")


def _torch_affinities(batch_views, n_neighbors):
    """Dense Gaussian k-NN affinities used inside the differentiable loss."""
    import torch

    output = []
    for view in batch_views:
        batch = view.shape[0]
        k = int(max(1, min(int(n_neighbors), batch - 1)))
        distances = torch.cdist(view, view, p=2)
        distances = distances + torch.eye(
            batch, dtype=view.dtype, device=view.device
        ) * 1e12
        nearest_values, nearest_indices = torch.topk(
            distances, k=k, largest=False, dim=1
        )
        positive = nearest_values[nearest_values > EPS]
        sigma = torch.median(positive) if positive.numel() else view.new_tensor(1.0)
        sigma = torch.clamp(sigma.detach(), min=1e-8)
        values = torch.exp(-(nearest_values ** 2) / (2.0 * sigma * sigma))
        affinity = torch.zeros_like(distances)
        affinity.scatter_(1, nearest_indices, values)
        affinity = torch.maximum(affinity, affinity.T)
        affinity.fill_diagonal_(0.0)
        output.append(affinity)
    return output


def _make_mlp(torch, in_features, hidden, out_features, *, final_tanh=False):
    layers = []
    width = int(in_features)
    for next_width in hidden:
        layers.extend([
            torch.nn.Linear(width, int(next_width)),
            torch.nn.LeakyReLU(),
        ])
        width = int(next_width)
    layers.append(torch.nn.Linear(width, int(out_features)))
    if final_tanh:
        layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def _build_model(matrices, config, view_prior):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = torch.nn.ModuleList([
                _make_mlp(
                    torch,
                    matrix.shape[1],
                    config.encoder_hidden,
                    config.output_dim,
                    final_tanh=True,
                )
                for matrix in matrices
            ])
            total_width = int(sum(matrix.shape[1] for matrix in matrices))
            self.fusion = _make_mlp(
                torch, total_width, config.fusion_hidden, len(matrices)
            )
            self.global_logits = torch.nn.Parameter(torch.zeros(len(matrices)))
            self.register_buffer(
                "log_view_prior",
                torch.log(torch.as_tensor(view_prior, dtype=torch.float32)),
            )

        def forward(self, batch_views):
            per_view = torch.stack([
                encoder(view) for encoder, view in zip(self.encoders, batch_views)
            ], dim=1)
            if config.fusion_mode == "uniform":
                weights = torch.exp(self.log_view_prior)
                alpha = weights[None, :].expand(per_view.shape[0], -1)
            elif config.fusion_mode == "global":
                weights = torch.softmax(
                    self.global_logits / config.temperature + self.log_view_prior,
                    dim=0,
                )
                alpha = weights[None, :].expand(per_view.shape[0], -1)
            else:
                logits = self.fusion(torch.cat(batch_views, dim=1))
                alpha = torch.softmax(
                    logits / config.temperature + self.log_view_prior[None, :],
                    dim=1,
                )
            fused = torch.sum(alpha[:, :, None] * per_view, dim=1)
            return fused, alpha, per_view

    return Model()


def _orthogonal_transform(raw_embedding, config, *, return_diagnostics=False):
    """Detached sqrt(batch) R^-1 transform used by released SpecRaGE.

    The released implementation enforces ``Y.T @ Y / batch = I``.  The paper
    writes the equivalent unscaled constraint, but the batch normalization is
    material when the Rayleigh loss is divided by ``batch**2``.
    """
    import torch

    with torch.no_grad():
        singular = torch.linalg.svdvals(raw_embedding)
        largest = torch.max(singular)
        smallest = torch.min(singular)
        condition = largest / torch.clamp(smallest, min=1e-12)
        if config.orthogonalization == "qr":
            _, upper = torch.linalg.qr(raw_embedding, mode="reduced")
            diagonal = torch.abs(torch.diagonal(upper))
            if not torch.isfinite(upper).all() or torch.min(diagonal) < 1e-8:
                raise FloatingPointError("SpecRaGE QR layer became rank deficient")
            transform = math.sqrt(raw_embedding.shape[0]) * torch.linalg.inv(upper)
            clipped = False
        else:
            # A small-sample stabilization of the same whitening operation.
            # It only clips directions below a registered fraction of the
            # leading singular value and records when clipping was necessary.
            _, singular_full, right_t = torch.linalg.svd(
                raw_embedding, full_matrices=False
            )
            floor = config.orthogonal_floor * torch.max(singular_full)
            clipped_singular = torch.clamp(singular_full, min=floor)
            right = right_t.T
            transform = (
                right
                @ torch.diag(math.sqrt(raw_embedding.shape[0]) / clipped_singular)
                @ right.T
            )
            clipped = bool(torch.any(singular_full < floor).cpu())
        diagnostics = {
            "orthogonal_condition": float(condition.cpu()),
            "orthogonal_smallest_singular": float(smallest.cpu()),
            "orthogonal_transform_norm": float(torch.linalg.norm(transform).cpu()),
            "orthogonal_clipped": clipped,
        }
        transform = transform.detach()
        return (transform, diagnostics) if return_diagnostics else transform


def _spectral_rayleigh_terms(
    embedding, affinities, alpha, *, view_prior=None,
    view_mass_normalization=False, edge_mass_enabled=False,
):
    """Return the paper-scaled spectral energy and the mass safeguard.

    For a symmetric affinity ``W``, ``sum_ij W_ij ||y_i-y_j||^2`` equals
    ``2 Tr(Y.T L Y)``.  Therefore division by ``B**2 * V`` reproduces the
    ``2/(B**2 V)`` Rayleigh term in SpecRaGE.  The prior-normalized project
    extension already averages views through ``q_v`` and uses ``B**2`` only.
    """
    import torch

    pair_distance = torch.sum(
        (embedding[:, None, :] - embedding[None, :, :]) ** 2,
        dim=2,
    )
    rayleigh = embedding.new_tensor(0.0)
    edge_mass = embedding.new_tensor(0.0)
    if view_prior is None:
        prior = embedding.new_full((len(affinities),), 1.0 / len(affinities))
    else:
        prior = view_prior.to(dtype=embedding.dtype, device=embedding.device)
    for view_index, affinity in enumerate(affinities):
        reliability = alpha[:, view_index]
        if view_mass_normalization:
            reliability = reliability / prior[view_index]
        weighted = affinity * reliability[:, None] * reliability[None, :]
        multiplier = prior[view_index] if view_mass_normalization else 1.0
        rayleigh = rayleigh + multiplier * torch.sum(weighted * pair_distance)
        if edge_mass_enabled:
            expected = (
                torch.mean(reliability) ** 2 * torch.sum(affinity)
            ).clamp_min(1e-12)
            edge_mass = edge_mass + multiplier * (
                (torch.sum(weighted) / expected) - 1.0
            ) ** 2
    denominator = float(embedding.shape[0] ** 2)
    if not view_mass_normalization:
        denominator *= len(affinities)
        edge_mass = edge_mass / len(affinities)
    return rayleigh / denominator, edge_mass


def _rayleigh_loss(
    model, gradient_views, orthogonal_views, config, agreement_target=None,
    view_prior=None,
):
    import torch

    orthogonal_raw, _, _ = model(orthogonal_views)
    transform, orthogonal_diagnostics = _orthogonal_transform(
        orthogonal_raw, config, return_diagnostics=True
    )
    raw, alpha, _ = model(gradient_views)
    embedding = raw @ transform
    affinities = _torch_affinities(gradient_views, config.n_neighbors)
    rayleigh, edge_mass = _spectral_rayleigh_terms(
        embedding,
        affinities,
        alpha,
        view_prior=view_prior,
        view_mass_normalization=config.view_mass_normalization,
        edge_mass_enabled=config.edge_mass_strength > 0,
    )
    agreement_loss = embedding.new_tensor(0.0)
    if agreement_target is not None and config.agreement_strength > 0:
        target = torch.clamp(agreement_target, min=1e-12)
        agreement_loss = torch.mean(torch.sum(
            target * (torch.log(target) - torch.log(torch.clamp(alpha, min=1e-12))),
            dim=1,
        ))
    loss = (
        rayleigh
        + config.agreement_strength * agreement_loss
        + config.edge_mass_strength * edge_mass
    )
    gram = (embedding.T @ embedding) / embedding.shape[0]
    identity = torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )
    orthogonality_error = torch.linalg.norm(gram - identity)
    entropy = -torch.sum(alpha * torch.log(torch.clamp(alpha, min=1e-12)), dim=1)
    components = {
        "rayleigh_loss": float(rayleigh.detach().cpu()),
        "agreement_loss": float(agreement_loss.detach().cpu()),
        "edge_mass_loss": float(edge_mass.detach().cpu()),
        **orthogonal_diagnostics,
    }
    return loss, orthogonality_error, torch.mean(entropy), components


def _paired_training_batches(rng, indices, batch_size, minimum_batch):
    """Yield every full pair of independently shuffled training batches.

    This follows the released implementation's zipped gradient and
    orthogonalization DataLoaders.  The earlier adaptation sampled only one
    pair per epoch, reducing the smoke fit to 6--12 updates total.
    """
    indices = np.asarray(indices, dtype=int)
    size = int(min(max(minimum_batch, batch_size), len(indices)))
    if size < minimum_batch:
        raise ValueError("not enough training samples for SpecRaGE batches")
    gradient_order = rng.permutation(indices)
    orthogonal_order = rng.permutation(indices)
    stops = list(range(0, len(indices) - size + 1, size))
    if not stops:
        stops = [0]
    for start in stops:
        yield (
            orthogonal_order[start:start + size],
            gradient_order[start:start + size],
        )


def _fit_one_seed(
    names, matrices, base_graph_tuple, config, seed, view_prior=None,
    tie_keys=None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "SpecRaGE graph learning requires PyTorch; install it in the experiment runtime"
        ) from exc

    config.validate()
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    device = torch.device("cpu")
    tensors = tuple(
        torch.as_tensor(matrix, dtype=torch.float32, device=device)
        for matrix in matrices
    )
    n = matrices[0].shape[0]
    minimum_batch = max(config.output_dim + 1, config.n_neighbors + 1, 4)
    split_rng = np.random.default_rng(int(seed) + 4019)
    fit_count = n if config.fit_sample_cap is None else min(n, int(config.fit_sample_cap))
    fit_pool = np.sort(split_rng.permutation(n)[:fit_count])
    validation_count = max(minimum_batch, int(math.ceil(0.10 * fit_count)))
    if fit_count - validation_count < 2 * minimum_batch:
        raise ValueError(
            "sample count is too small for disjoint training batches and validation"
        )
    split = split_rng.permutation(fit_pool)
    validation_index = np.sort(split[:validation_count])
    training_index = np.sort(split[validation_count:])

    # The agreement target is part of fitting, so it must obey the same
    # registered sample cap as the neural learner.  It is constructed only on
    # the seed-specific unlabeled fit pool; labels are never accepted here.
    agreement_targets = None
    agreement_scores = None
    agreement_tensor = None
    if config.agreement_strength > 0:
        pool_graphs, _ = base_view_graphs(
            tuple(matrix[fit_pool] for matrix in matrices),
            n_neighbors=config.n_neighbors,
            tie_keys=None if tie_keys is None else np.asarray(tie_keys)[fit_pool],
        )
        agreement_targets, agreement_scores = cross_view_agreement_targets(
            pool_graphs,
            temperature=config.agreement_temperature,
            view_prior=view_prior,
        )
        full_targets = np.zeros((n, len(names)), dtype=float)
        full_targets[fit_pool] = agreement_targets
        agreement_tensor = torch.as_tensor(
            full_targets, dtype=torch.float32, device=device
        )

    view_prior_tensor = torch.as_tensor(view_prior, dtype=torch.float32, device=device)
    model = _build_model(matrices, config, view_prior).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=config.lr_patience,
        min_lr=1e-7,
    )
    rng = np.random.default_rng(int(seed) + 9187)
    best_loss = math.inf
    best_state = None
    stale = 0
    history = []
    optimizer_updates = 0
    for epoch in range(config.max_epochs):
        model.train()
        epoch_rows = []
        for orthogonal_index, gradient_index in _paired_training_batches(
            rng, training_index, config.batch_size, minimum_batch
        ):
            gradient_views = tuple(view[gradient_index] for view in tensors)
            orthogonal_views = tuple(view[orthogonal_index] for view in tensors)
            batch_target = None if agreement_tensor is None \
                else agreement_tensor[gradient_index]
            optimizer.zero_grad(set_to_none=True)
            loss, orthogonality_error, entropy, components = _rayleigh_loss(
                model, gradient_views, orthogonal_views, config, batch_target,
                view_prior_tensor,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("SpecRaGE loss became non-finite")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            ))
            optimizer.step()
            optimizer_updates += 1
            epoch_rows.append({
                "loss": float(loss.detach().cpu()),
                "orthogonality_error": float(orthogonality_error.detach().cpu()),
                "alpha_entropy": float(entropy.detach().cpu()),
                "gradient_norm": gradient_norm,
                **components,
            })
        training_loss = float(np.mean([row["loss"] for row in epoch_rows]))
        model.eval()
        with torch.no_grad():
            validation_views = tuple(view[validation_index] for view in tensors)
            validation_target = None if agreement_tensor is None \
                else agreement_tensor[validation_index]
            validation_loss_tensor, validation_orthogonality, validation_entropy, \
                validation_components = _rayleigh_loss(
                    model, validation_views, validation_views, config,
                    validation_target, view_prior_tensor,
                )
        validation_loss = float(validation_loss_tensor.detach().cpu())
        model.train()
        scheduler.step(validation_loss)
        history.append({
            "epoch": int(epoch + 1),
            "optimizer_updates": int(optimizer_updates),
            "updates_this_epoch": int(len(epoch_rows)),
            "loss": validation_loss,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "orthogonality_error": float(np.mean([
                row["orthogonality_error"] for row in epoch_rows
            ])),
            "alpha_entropy": float(np.mean([
                row["alpha_entropy"] for row in epoch_rows
            ])),
            "validation_orthogonality_error": float(
                validation_orthogonality.detach().cpu()
            ),
            "validation_alpha_entropy": float(validation_entropy.detach().cpu()),
            "gradient_norm": float(np.mean([
                row["gradient_norm"] for row in epoch_rows
            ])),
            "rayleigh_loss": float(np.mean([
                row["rayleigh_loss"] for row in epoch_rows
            ])),
            "agreement_loss": float(np.mean([
                row["agreement_loss"] for row in epoch_rows
            ])),
            "edge_mass_loss": float(np.mean([
                row["edge_mass_loss"] for row in epoch_rows
            ])),
            "orthogonal_condition": float(np.mean([
                row["orthogonal_condition"] for row in epoch_rows
            ])),
            "orthogonal_condition_max": float(np.max([
                row["orthogonal_condition"] for row in epoch_rows
            ])),
            "orthogonal_transform_norm": float(np.mean([
                row["orthogonal_transform_norm"] for row in epoch_rows
            ])),
            "orthogonal_clipped_fraction": float(np.mean([
                row["orthogonal_clipped"] for row in epoch_rows
            ])),
            "validation_rayleigh_loss": validation_components["rayleigh_loss"],
            "validation_agreement_loss": validation_components["agreement_loss"],
            "validation_edge_mass_loss": validation_components["edge_mass_loss"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        improved = best_loss - validation_loss > \
            config.min_delta * max(1.0, abs(best_loss))
        if improved or best_state is None:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if (
            config.checkpoint_mode == "best_unlabeled"
            and epoch + 1 >= config.min_epochs
            and stale >= config.patience
        ):
            break
    if best_state is None:
        raise RuntimeError("SpecRaGE training did not produce a valid state")
    if config.checkpoint_mode == "best_unlabeled":
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw, alpha_tensor, _ = model(tensors)
        transform = _orthogonal_transform(raw, config)
        embedding_tensor = raw @ transform
    alpha = alpha_tensor.cpu().numpy().astype(float, copy=False)
    embedding = embedding_tensor.cpu().numpy().astype(float, copy=False)
    graph = weighted_multiview_graph(
        base_graph_tuple, alpha, view_prior=view_prior,
        mass_normalize=config.view_mass_normalization,
    )
    embedding_graph, embedding_graph_metadata = embedding_knn_affinity(
        embedding, n_neighbors=config.n_neighbors, tie_keys=tie_keys
    )
    diagnostics = _seed_diagnostics(
        names, base_graph_tuple, alpha, embedding, graph, history
    )
    diagnostics["embedding_graph"] = {
        **embedding_graph_metadata,
        **graph_diagnostics(embedding_graph),
    }
    diagnostics["optimizer_updates"] = int(optimizer_updates)
    diagnostics["fit_sample_count"] = int(fit_count)
    diagnostics["inference_sample_count"] = int(n)
    if agreement_targets is not None:
        target_entropy = -np.sum(
            agreement_targets * np.log(np.maximum(agreement_targets, EPS)), axis=1
        )
        diagnostics["agreement_target"] = {
            "temperature": float(config.agreement_temperature),
            "fit_sample_count": int(fit_count),
            "fit_sample_fraction": float(fit_count / n),
            "entropy_normalized": float(
                np.mean(target_entropy) / np.log(agreement_targets.shape[1])
            ),
            "alpha_target_mad": float(
                np.mean(np.abs(alpha[fit_pool] - agreement_targets))
            ),
            "score_std": float(np.std(agreement_scores)),
            "score_median": float(np.median(agreement_scores)),
            "score_per_view_median": {
                name: float(np.median(agreement_scores[:, index]))
                for index, name in enumerate(names)
            },
        }
    return SpecRaGESeedResult(
        seed=int(seed),
        alpha=alpha,
        embedding=embedding,
        graph=graph,
        embedding_graph=embedding_graph,
        history=history,
        diagnostics=diagnostics,
    )


def _frobenius_sparse(matrix):
    matrix = csr_matrix(matrix, dtype=float)
    return float(np.sqrt(matrix.multiply(matrix).sum()))


def _off_diagonal_ratio(matrix):
    matrix = np.asarray(matrix, dtype=float)
    off = matrix - np.diag(np.diag(matrix))
    return float(np.linalg.norm(off) / (np.linalg.norm(matrix) + EPS))


def _seed_diagnostics(names, base_graph_tuple, alpha, embedding, graph, history):
    normalized = tuple(
        symmetric_normalized_laplacian(base) for base in base_graph_tuple
    )
    joint = []
    for name, laplacian in zip(names, normalized):
        projected = embedding.T @ (laplacian @ embedding)
        joint.append({
            "view": name,
            "off_diagonal_ratio": _off_diagonal_ratio(projected),
            "rayleigh_trace": float(np.trace(projected)),
        })
    commutators = []
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            commutator = normalized[left] @ normalized[right] \
                - normalized[right] @ normalized[left]
            denominator = (
                _frobenius_sparse(normalized[left])
                * _frobenius_sparse(normalized[right])
                + EPS
            )
            commutators.append(
                _frobenius_sparse(commutator) / denominator
            )
    entropy = -np.sum(alpha * np.log(np.maximum(alpha, EPS)), axis=1)
    gram = (embedding.T @ embedding) / embedding.shape[0]
    return {
        "epochs": len(history),
        "best_loss": float(min(item["loss"] for item in history)),
        "final_loss": float(history[-1]["loss"]),
        "alpha_entropy_mean": float(np.mean(entropy)),
        "alpha_entropy_normalized": float(np.mean(entropy) / np.log(alpha.shape[1])),
        "alpha_max_mean": float(np.mean(np.max(alpha, axis=1))),
        "dominant_view_fraction": {
            str(name): float(np.mean(np.argmax(alpha, axis=1) == index))
            for index, name in enumerate(names)
        },
        "mean_view_weight": {
            str(name): float(np.mean(alpha[:, index]))
            for index, name in enumerate(names)
        },
        "orthogonality_error": float(
            np.linalg.norm(gram - np.eye(embedding.shape[1]))
        ),
        "joint_diagonalization": joint,
        "mean_laplacian_commutator": float(np.mean(commutators))
            if commutators else 0.0,
        **graph_diagnostics(graph),
    }


def _mean_sparse(graphs):
    graphs = tuple(csr_matrix(graph, dtype=float) for graph in graphs)
    output = csr_matrix(graphs[0].shape, dtype=float)
    for graph in graphs:
        output = output + graph
    output = (output / len(graphs)).tocsr()
    output = output.maximum(output.T).tocsr()
    output.setdiag(0.0)
    output.eliminate_zeros()
    return output


def _edge_support_jaccard(left, right):
    left = csr_matrix(left, dtype=float).copy()
    right = csr_matrix(right, dtype=float).copy()
    left.data[:] = 1.0
    right.data[:] = 1.0
    intersection = float(left.multiply(right).sum())
    union = float((left + right).astype(bool).sum())
    return intersection / (union + EPS)


def _graph_collapse_diagnostics(graph, uniform_graph):
    graph = csr_matrix(graph, dtype=float)
    uniform_graph = csr_matrix(uniform_graph, dtype=float)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    positive = graph.data[graph.data > 0]
    effective_edges = float(
        (positive.sum() ** 2) / (np.sum(positive ** 2) + EPS)
    ) if positive.size else 0.0
    mean_degree = float(np.mean(degree))
    return {
        "effective_edge_fraction": float(effective_edges / max(graph.nnz, 1)),
        "degree_p05_over_mean": float(
            np.quantile(degree, 0.05) / (mean_degree + EPS)
        ),
        "near_isolated_fraction": float(np.mean(degree < 1e-3 * (mean_degree + EPS))),
        "total_affinity_vs_uniform": float(
            graph.sum() / (uniform_graph.sum() + EPS)
        ),
    }


def fit_specrage_graph(
    views: Mapping[str, np.ndarray],
    *,
    config: SpecRaGEConfig | None = None,
    seeds: Sequence[int] = (11, 23, 37),
    view_prior: Mapping[str, float] | Sequence[float] | None = None,
    tie_keys: np.ndarray | None = None,
):
    """Fit and seed-ensemble a label-free SpecRaGE reliability graph."""
    config = SpecRaGEConfig() if config is None else config
    config.validate()
    names, matrices = prepare_views(views)
    if view_prior is None:
        prior = np.full(len(names), 1.0 / len(names))
    elif isinstance(view_prior, Mapping):
        missing = [name for name in names if name not in view_prior]
        extra = [name for name in view_prior if name not in names]
        if missing or extra:
            raise ValueError(f"view_prior names disagree: missing={missing}, extra={extra}")
        prior = np.asarray([view_prior[name] for name in names], dtype=float)
    else:
        prior = np.asarray(view_prior, dtype=float)
    if prior.shape != (len(names),) or not np.isfinite(prior).all() \
            or np.any(prior <= 0):
        raise ValueError("view_prior must contain one positive finite value per view")
    prior = prior / prior.sum()
    if matrices[0].shape[0] <= config.output_dim:
        raise ValueError("sample count must exceed SpecRaGE output_dim")
    if tie_keys is not None:
        tie_keys = np.asarray(tie_keys, dtype=float)
        if tie_keys.shape != (matrices[0].shape[0],) or not np.isfinite(tie_keys).all():
            raise ValueError("tie_keys must be one finite value per sample")
        if len(np.unique(tie_keys)) != len(tie_keys):
            raise ValueError("tie_keys must be unique")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("SpecRaGE seeds must be unique")
    base_graph_tuple, affinity_metadata = base_view_graphs(
        matrices, n_neighbors=config.n_neighbors, tie_keys=tie_keys
    )
    seed_results = tuple(
        _fit_one_seed(
            names, matrices, base_graph_tuple, config, seed,
            view_prior=prior, tie_keys=tie_keys,
        )
        for seed in seeds
    )
    alpha_stack = np.stack([result.alpha for result in seed_results], axis=0)
    alpha = alpha_stack.mean(axis=0)
    alpha /= alpha.sum(axis=1, keepdims=True)
    # Every reliance control is rebuilt from this same seed-mean alpha.  This
    # avoids comparing a nonlinear mean-of-graphs headline to controls formed
    # from a graph-of-mean-weights.
    graph = weighted_multiview_graph(
        base_graph_tuple, alpha, view_prior=prior,
        mass_normalize=config.view_mass_normalization,
    )
    embedding_graph = _mean_sparse(
        result.embedding_graph for result in seed_results
    )
    uniform_graph = weighted_multiview_graph(
        base_graph_tuple, np.repeat(prior[None, :], alpha.shape[0], axis=0),
        view_prior=prior, mass_normalize=config.view_mass_normalization,
    )
    pairwise_seed_difference = []
    for left in range(len(seed_results)):
        for right in range(left + 1, len(seed_results)):
            pairwise_seed_difference.append(float(np.mean(np.abs(
                seed_results[left].alpha - seed_results[right].alpha
            ))))
    graph_seed_distances = []
    for left in range(len(seed_results)):
        for right in range(left + 1, len(seed_results)):
            difference = seed_results[left].graph - seed_results[right].graph
            graph_seed_distances.append(
                _frobenius_sparse(difference)
                / (_frobenius_sparse(seed_results[left].graph) + EPS)
            )
    edge_jaccards = []
    for left in range(len(base_graph_tuple)):
        for right in range(left + 1, len(base_graph_tuple)):
            edge_jaccards.append(
                _edge_support_jaccard(base_graph_tuple[left], base_graph_tuple[right])
            )
    diagnostics = {
        **graph_diagnostics(graph),
        **_graph_collapse_diagnostics(graph, uniform_graph),
        "config_fingerprint": config.fingerprint,
        "seeds": list(seeds),
        "affinity_metadata": {
            name: details for name, details in zip(names, affinity_metadata)
        },
        "alpha_seed_mad": float(np.mean(pairwise_seed_difference))
            if pairwise_seed_difference else 0.0,
        "alpha_seed_std_mean": float(np.mean(np.std(alpha_stack, axis=0))),
        "graph_seed_relative_distance_mean": float(np.mean(graph_seed_distances))
            if graph_seed_distances else 0.0,
        "base_edge_jaccard_mean": float(np.mean(edge_jaccards))
            if edge_jaccards else 1.0,
        "base_edge_jaccard_median": float(np.median(edge_jaccards))
            if edge_jaccards else 1.0,
        "view_effective_rank": {
            name: float(
                np.sum(np.linalg.svd(matrix, compute_uv=False) ** 2) ** 2
                / (
                    np.sum(np.linalg.svd(matrix, compute_uv=False) ** 4)
                    + EPS
                )
            )
            for name, matrix in zip(names, matrices)
        },
        "alpha_entropy_normalized": float(np.mean(
            -np.sum(alpha * np.log(np.maximum(alpha, EPS)), axis=1)
        ) / np.log(alpha.shape[1])),
        "alpha_effective_views_mean": float(np.mean(
            1.0 / (np.sum(alpha ** 2, axis=1) + EPS)
        )),
        "alpha_kl_from_prior_mean": float(np.mean(np.sum(
            alpha * (
                np.log(np.maximum(alpha, EPS))
                - np.log(np.maximum(prior[None, :], EPS))
            ),
            axis=1,
        ))),
        "mean_view_weight": {
            name: float(np.mean(alpha[:, index]))
            for index, name in enumerate(names)
        },
        "dominant_view_fraction": {
            name: float(np.mean(np.argmax(alpha, axis=1) == index))
            for index, name in enumerate(names)
        },
        "primary_interface": "fused_embedding_knn_graph",
        "view_mass_normalization": bool(config.view_mass_normalization),
        "view_prior": {
            name: float(prior[index]) for index, name in enumerate(names)
        },
    }
    seed_agreement = [
        result.diagnostics["agreement_target"]
        for result in seed_results
        if "agreement_target" in result.diagnostics
    ]
    if seed_agreement:
        diagnostics["agreement_target"] = {
            "temperature": float(config.agreement_temperature),
            "target_scope": "seed_specific_unlabeled_fit_pool",
            "fit_sample_count_by_seed": [
                int(row["fit_sample_count"]) for row in seed_agreement
            ],
            "fit_sample_fraction_mean": float(np.mean([
                row["fit_sample_fraction"] for row in seed_agreement
            ])),
            "entropy_normalized": float(np.mean([
                row["entropy_normalized"] for row in seed_agreement
            ])),
            "alpha_target_mad": float(np.mean([
                row["alpha_target_mad"] for row in seed_agreement
            ])),
            "score_std": float(np.mean([
                row["score_std"] for row in seed_agreement
            ])),
            "score_median": float(np.mean([
                row["score_median"] for row in seed_agreement
            ])),
            "score_per_view_median": {
                name: float(np.mean([
                    row["score_per_view_median"][name]
                    for row in seed_agreement
                ]))
                for name in names
            },
        }
    diagnostics["embedding_graph"] = graph_diagnostics(embedding_graph)
    return SpecRaGEGraphResult(
        graph=graph,
        embedding_graph=embedding_graph,
        alpha=alpha,
        view_names=names,
        base_graphs=base_graph_tuple,
        seed_results=seed_results,
        config=config,
        view_prior=prior,
        diagnostics=diagnostics,
    )


def graph_for_control(result: SpecRaGEGraphResult, mode: str, *, seed: int = 0):
    """Rebuild a graph from learned weights for a registered reliance control."""
    alpha = control_alpha(
        result.alpha, mode, seed=seed, view_prior=result.view_prior
    )
    return weighted_multiview_graph(
        result.base_graphs, alpha, view_prior=result.view_prior,
        mass_normalize=result.config.view_mass_normalization,
    )


__all__ = [
    "SpecRaGEConfig",
    "SpecRaGEGraphResult",
    "SpecRaGESeedResult",
    "base_view_graphs",
    "control_alpha",
    "cross_view_agreement_targets",
    "embedding_knn_affinity",
    "fit_specrage_graph",
    "gaussian_knn_affinity",
    "graph_for_control",
    "prepare_views",
    "weighted_multiview_graph",
]
