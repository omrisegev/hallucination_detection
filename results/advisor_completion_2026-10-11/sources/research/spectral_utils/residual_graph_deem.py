"""Label-free primitives for the frozen Residual-Graph DEEM 24-cell audit.

The public functions in this module never accept natural labels.  Evaluation is
implemented in a separate CLI.  Continuous-visible DEEM is an explicitly named
adaptation: it is not an implementation of the categorical DEEM theorem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import ndtr
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from .feature_contract import confidence_sign_vector
from .graph_topology import self_safe_knn_graph
from .laplacian_upcr import symmetric_normalized_laplacian
from .specrage_views import FEATURE_TO_VIEW, VIEW_ORDER, view_members


EPS = 1e-12
SCHEMA_VERSION = "residual_graph_deem_core_v1"
SEEDS = (0, 1, 2, 3, 4)
LAMBDA_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)


class ResidualGraphDeemError(RuntimeError):
    """A frozen mechanical, provenance, fit-health, or serialization gate failed."""


@dataclass(frozen=True)
class ContinuousDeemConfig:
    family_width: int = 8
    epochs: int = 100
    learning_rate: float = 1e-3
    momentum: float = 0.0
    mala_delta: float = 0.10
    mala_steps: int = 5
    replay_refresh: float = 0.05
    dtype: str = "float64"
    device: str = "cpu"
    anchor_tolerance: float = 1e-6
    posterior_sd_min: float = 1e-3
    init_sd: float = 0.005
    deterministic: bool = True


@dataclass(frozen=True)
class GraphDeemConfig:
    k: int = 7
    lambda_: float = 0.0
    mechanism: str = "target"
    nuisance_width: int = 8
    nuisance_dim: int = 3
    whitening_ridge: float = 1e-6
    orthogonality_gamma: float = 1.0
    largest_component_min: float = 0.90
    isolated_fraction_max: float = 0.05


@dataclass(frozen=True)
class DufsConfig:
    k: int = 7
    gate_sigma: float = 0.5
    mu0: float = 0.5
    learning_rate: float = 0.02
    epochs: int = 120
    seeds: tuple[int, ...] = SEEDS
    median_cosine_min: float = 0.80


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    name: str
    graph: bool
    primary: bool = False
    lambda_zero_alias: str | None = None


@dataclass(frozen=True)
class Standardization:
    mean: np.ndarray
    scale: np.ndarray
    constant_mask: np.ndarray


@dataclass(frozen=True)
class FoldManifest:
    fold: int
    donor_indices: tuple[int, ...]
    held_indices: tuple[int, ...]
    donor_group_sha256: str
    held_group_sha256: str
    standardization_sha256: str = ""
    residualizer_sha256: str = ""


@dataclass
class ContinuousDeemResult:
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
    alias_of: str | None = None


@dataclass
class CrossFitResult:
    logit: np.ndarray
    posterior: np.ndarray
    contributions: np.ndarray
    residuals: np.ndarray
    folds: np.ndarray
    fold_manifests: tuple[FoldManifest, ...]
    fit_results: tuple[ContinuousDeemResult, ...]
    residualizer_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FrozenFitManifest:
    schema: str
    cell_id: str
    arm_id: str
    seed: int
    source_sha256: str
    bundle_sha256: str
    code_sha256: str
    config_sha256: str
    artifact_sha256: str
    status: str
    debug: bool = False


ARM_SPECS = (
    ArmSpec("B0", "iu_pcr_inventory", False),
    ArmSpec("B1", "deem_inventory_hard_adapter020", False),
    ArmSpec("B2", "deem_inventory_soft_rank_adapter020_repaired", False),
    ArmSpec("B3", "deem_inventory_continuous_additive", False, True),
    ArmSpec("G0", "deem_inventory_raw_graph_uniform_target", True, False, "B3"),
    ArmSpec("G1", "deem_inventory_raw_graph_dufs_target", True, False, "B3"),
    ArmSpec("G2", "deem_inventory_residual_graph_uniform_target", True, False, "B3"),
    ArmSpec("G3", "deem_inventory_residual_graph_dufs_target", True, True, "B3"),
    ArmSpec("G4", "deem_inventory_residual_graph_dufs_nuisance", True, True, "B3"),
    ArmSpec("G5", "deem_inventory_present_family_laplacian", True, False, "B3"),
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_fingerprint() -> dict[str, Any]:
    """Return the compact, hashable runtime contract stored with every fit."""

    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "deem"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    value = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }
    value["environment_sha256"] = canonical_sha256(value)
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_write_json(path: str | Path, value: Any, *, immutable: bool = False) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if immutable and target.exists():
        raise FileExistsError(f"immutable artifact already exists: {target}")
    payload = json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        if immutable:
            target.chmod(0o444)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_file(target)


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        value = np.asarray(array)
        if value.dtype.hasobject:
            raise TypeError(f"object dtype forbidden in allow_pickle=False artifact: {name}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npz", dir=target.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_file(target)


def set_determinism(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
    except ImportError:
        pass


def validate_inventory(X_raw: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    X = np.asarray(X_raw, dtype=float)
    names = tuple(str(name) for name in feature_names)
    if X.ndim != 2 or X.shape[1] != len(names) or X.shape[0] < 3:
        raise ValueError("X_raw must have shape (n>=3, len(feature_names))")
    if len(names) != len(set(names)):
        raise ValueError("feature names must be unique")
    if not np.isfinite(X).all():
        raise ValueError("inventory contains non-finite values")
    unknown = sorted(set(names) - set(FEATURE_TO_VIEW))
    if unknown:
        raise KeyError("unregistered inventory feature(s): " + ", ".join(unknown))
    return X


def fit_standardization(X_raw: np.ndarray) -> Standardization:
    X = np.asarray(X_raw, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    constant = scale < EPS
    scale = scale.copy()
    scale[constant] = 1.0
    return Standardization(mean=mean, scale=scale, constant_mask=constant)


def apply_standardization(X_raw: np.ndarray, transform: Standardization) -> np.ndarray:
    X = np.asarray(X_raw, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(transform.mean):
        raise ValueError("matrix and donor transform disagree")
    return (X - transform.mean[None, :]) / transform.scale[None, :]


def donor_risk_matrix(
    donor_raw: np.ndarray,
    held_raw: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, Standardization]:
    donor = validate_inventory(donor_raw, feature_names)
    held = validate_inventory(held_raw, feature_names)
    transform = fit_standardization(donor)
    signs = confidence_sign_vector(feature_names)
    return (
        -apply_standardization(donor, transform) * signs[None, :],
        -apply_standardization(held, transform) * signs[None, :],
        transform,
    )


def family_index_map(feature_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    names = tuple(str(name) for name in feature_names)
    output = {}
    for family in VIEW_ORDER:
        indices = tuple(index for index, name in enumerate(names) if FEATURE_TO_VIEW[name] == family)
        if indices:
            output[family] = indices
    if not output:
        raise ValueError("inventory has no present family")
    return output


def equal_family_risk_anchor(X_risk: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    X = validate_inventory(X_risk, feature_names)
    groups = family_index_map(feature_names)
    return np.mean([X[:, indices].mean(axis=1) for indices in groups.values()], axis=0)


def metric_weights(feature_names: Sequence[str]) -> np.ndarray:
    groups = family_index_map(feature_names)
    output = np.zeros(len(tuple(feature_names)), dtype=float)
    for indices in groups.values():
        output[list(indices)] = 1.0 / (len(groups) * len(indices))
    if not np.isclose(output.sum(), 1.0, atol=1e-12):
        raise AssertionError("equal-family metric mass does not sum to one")
    return output


class _FamilyAdditiveEnergy:
    def __init__(self, feature_names: Sequence[str], config: ContinuousDeemConfig, seed: int):
        import torch
        self.torch = torch
        self.names = tuple(str(name) for name in feature_names)
        self.groups = family_index_map(self.names)
        self.config = config
        self.seed = int(seed)
        dtype = torch.float64
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
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
                torch.randn(config.family_width, size, dtype=dtype, generator=generator)
                * config.init_sd
            )
            self.d[family] = torch.nn.Parameter(
                torch.randn(config.family_width, dtype=dtype, generator=generator)
                * config.init_sd
            )
            self.V[family] = torch.nn.Parameter(
                torch.randn(size, config.family_width, dtype=dtype, generator=generator)
                * config.init_sd
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
        pieces = []
        family_output = {}
        for family, indices in self.groups.items():
            index = torch.tensor(indices, dtype=torch.long, device=X.device)
            xg = X.index_select(1, index)
            u = torch.tanh(xg @ self.W[family].T + self.d[family])
            nonlinear = (2.0 / len(indices)) * torch.tanh(
                u @ self.V[family].T + self.e[family]
            )
            contribution = self.w[family] * xg + nonlinear
            pieces.append((indices, contribution))
            family_output[family] = contribution.sum(dim=1)
        atomic = torch.zeros_like(X)
        for indices, contribution in pieces:
            atomic[:, list(indices)] = contribution
        return atomic, family_output

    def logit(self, X):
        contribution, family = self.contributions(X)
        return self.b + contribution.sum(dim=1), contribution, family

    def free_energy(self, X):
        torch = self.torch
        ell, _, _ = self.logit(X)
        return 0.5 * ((X - self.a) ** 2).sum(dim=1) - torch.nn.functional.softplus(ell)

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        # .copy() is required, not cosmetic.  On CPU ``Tensor.numpy()`` shares
        # storage with the tensor, and the optimizer mutates parameters in
        # place -- so without it this "snapshot" aliases the live parameters
        # and a state captured while finite silently becomes NaN at a later
        # step, which is exactly when the failure artifact needs to be read.
        output = {
            "a": self.a.detach().cpu().numpy().copy(),
            "b": self.b.detach().cpu().numpy().copy(),
        }
        for label, collection in (("w", self.w), ("W", self.W), ("d", self.d), ("V", self.V), ("e", self.e)):
            for family, parameter in collection.items():
                output[f"{label}::{family}"] = parameter.detach().cpu().numpy().copy()
        return output

    def load_state_numpy(self, state: Mapping[str, np.ndarray]) -> None:
        torch = self.torch
        with torch.no_grad():
            self.a.copy_(torch.as_tensor(state["a"], dtype=torch.float64))
            self.b.copy_(torch.as_tensor(state["b"], dtype=torch.float64))
            for label, collection in (("w", self.w), ("W", self.W), ("d", self.d), ("V", self.V), ("e", self.e)):
                for family, parameter in collection.items():
                    parameter.copy_(torch.as_tensor(state[f"{label}::{family}"], dtype=torch.float64))


def _mala_step(model: _FamilyAdditiveEnergy, current, *, delta: float, generator):
    torch = model.torch

    def energy_gradient(value):
        candidate = value.detach().clone().requires_grad_(True)
        total = model.free_energy(candidate).sum()
        gradient = torch.autograd.grad(total, candidate, create_graph=False)[0]
        return model.free_energy(candidate).detach(), gradient.detach()

    current_energy, current_gradient = energy_gradient(current)
    noise = torch.randn(current.shape, dtype=current.dtype, device=current.device, generator=generator)
    forward_mean = current - 0.5 * delta * delta * current_gradient
    proposal = forward_mean + delta * noise
    proposal_energy, proposal_gradient = energy_gradient(proposal)
    reverse_mean = proposal - 0.5 * delta * delta * proposal_gradient
    log_q_reverse = -((current - reverse_mean) ** 2).sum(dim=1) / (2.0 * delta * delta)
    log_q_forward = -((proposal - forward_mean) ** 2).sum(dim=1) / (2.0 * delta * delta)
    log_accept = -proposal_energy + current_energy + log_q_reverse - log_q_forward
    uniform = torch.rand(len(current), dtype=current.dtype, device=current.device, generator=generator)
    accept = torch.log(uniform.clamp_min(torch.finfo(current.dtype).tiny)) < torch.minimum(
        log_accept, torch.zeros_like(log_accept)
    )
    updated = torch.where(accept[:, None], proposal.detach(), current.detach())
    return updated, float(accept.to(torch.float64).mean())


def persistent_mala(
    model: _FamilyAdditiveEnergy,
    current,
    *,
    delta: float,
    steps: int,
    generator,
):
    acceptance = []
    value = current
    for _ in range(int(steps)):
        value, rate = _mala_step(model, value, delta=float(delta), generator=generator)
        acceptance.append(rate)
    return value.detach(), float(np.mean(acceptance))


def _sparse_quadratic(values, laplacian: csr_matrix):
    """Differentiable ``sum_d x_d^T L x_d`` without dense materialization."""
    import torch
    coo = sparse.coo_matrix(laplacian)
    indices = torch.as_tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long, device=values.device
    )
    weights = torch.as_tensor(coo.data, dtype=values.dtype, device=values.device)
    operator = torch.sparse_coo_tensor(indices, weights, coo.shape, dtype=values.dtype).coalesce()
    product = torch.sparse.mm(operator, values if values.ndim == 2 else values[:, None])
    matrix = values if values.ndim == 2 else values[:, None]
    return (matrix * product).sum()


class _NuisanceEncoder:
    def __init__(self, p: int, config: GraphDeemConfig, seed: int):
        import torch
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 9173)
        dtype = torch.float64
        self.W1 = torch.nn.Parameter(
            torch.randn(config.nuisance_width, p, generator=generator, dtype=dtype) * 0.005
        )
        self.d1 = torch.nn.Parameter(torch.zeros(config.nuisance_width, dtype=dtype))
        self.W2 = torch.nn.Parameter(
            torch.randn(config.nuisance_dim, config.nuisance_width, generator=generator, dtype=dtype)
            * 0.005
        )
        self.d2 = torch.nn.Parameter(torch.zeros(config.nuisance_dim, dtype=dtype))
        self.ridge = float(config.whitening_ridge)

    def parameters(self):
        return [self.W1, self.d1, self.W2, self.d2]

    def __call__(self, X):
        import torch
        raw = torch.tanh(X @ self.W1.T + self.d1) @ self.W2.T + self.d2
        centered = raw - raw.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(len(X) - 1, 1)
        # Ridge-Cholesky whitening via triangular solve.  This is the same full
        # whitening as the symmetric (ZCA) inverse square root up to an
        # orthogonal rotation of the whitened coordinates, and every penalty
        # built on U is invariant to that rotation -- ||U||_F^2, trace(U^T L U)
        # and ||U^T e||^2 all are -- so the objective and its gradients are
        # unchanged (verified identical to 10 decimal places).
        #
        # The eigendecomposition form is not usable here: its backward carries
        # 1/(lambda_i - lambda_j) terms, and adding ridge*I shifts every
        # eigenvalue equally without separating them.  A collapsing nuisance
        # head drives those gaps to zero, the gradient becomes NaN, and the
        # next forward pass then hands an all-NaN covariance to eigh.  That is
        # the AIRCC job 217597 failure.  Cholesky's backward is a triangular
        # solve and stays finite on the same input.
        gram = covariance + self.ridge * torch.eye(covariance.shape[0], dtype=X.dtype)
        factor = torch.linalg.cholesky(gram)
        return torch.linalg.solve_triangular(factor, centered.T, upper=False).T


def _family_parameter_penalty(model: _FamilyAdditiveEnergy, family_laplacian: csr_matrix):
    import torch
    order = tuple(model.groups)
    vectors = torch.stack([model.V[family].mean(dim=0) for family in order], dim=0)
    return _sparse_quadratic(vectors, family_laplacian)


def fit_continuous_deem(
    X_risk: np.ndarray,
    feature_names: Sequence[str],
    *,
    seed: int = 0,
    config: ContinuousDeemConfig | None = None,
    graph_config: GraphDeemConfig | None = None,
    laplacian: csr_matrix | None = None,
    family_laplacian: csr_matrix | None = None,
    baseline_result: ContinuousDeemResult | None = None,
) -> ContinuousDeemResult:
    """Fit B3 or one frozen graph arm without accepting target labels.

    A graph configuration with ``lambda_=0`` is an exact direct alias to the
    caller-supplied B3 result.  No graph validation or encoder construction occurs.
    """
    import torch

    started = time.perf_counter()

    X = validate_inventory(X_risk, feature_names)
    names = tuple(str(name) for name in feature_names)
    config = config or ContinuousDeemConfig()
    graph_config = graph_config or GraphDeemConfig(lambda_=0.0)
    if config.dtype != "float64" or config.device != "cpu":
        raise ValueError("v1 continuous DEEM is frozen to float64 CPU")
    if graph_config.lambda_ == 0.0 and baseline_result is not None:
        if tuple(baseline_result.feature_names) != names:
            raise ValueError("lambda-zero baseline inventory mismatch")
        aliased = replace(baseline_result)
        aliased.alias_of = "B3"
        aliased.health = {**baseline_result.health, "runtime_seconds": 0.0}
        return aliased
    if graph_config.lambda_ != 0.0 and laplacian is None and family_laplacian is None:
        raise ValueError("nonzero graph fit requires a frozen Laplacian")
    if laplacian is not None and laplacian.shape != (len(X), len(X)):
        raise ValueError("sample Laplacian shape mismatch")

    set_determinism(seed)
    model = _FamilyAdditiveEnergy(names, config, seed)
    tensor = torch.as_tensor(X, dtype=torch.float64)
    buffer = tensor.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1_000_003)
    nuisance = None
    parameters = list(model.parameters())
    if graph_config.mechanism == "nuisance" and graph_config.lambda_ != 0.0:
        nuisance = _NuisanceEncoder(X.shape[1], graph_config, seed)
        parameters.extend(nuisance.parameters())
    optimizer = torch.optim.SGD(
        parameters, lr=float(config.learning_rate), momentum=float(config.momentum)
    )
    history: list[dict[str, float]] = []
    last_finite_state = None
    try:
        for epoch in range(int(config.epochs)):
            refresh = torch.rand(len(X), generator=generator) < float(config.replay_refresh)
            if bool(refresh.any()):
                replacements = torch.randint(len(X), (int(refresh.sum()),), generator=generator)
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
            base_loss = positive - negative
            penalty = torch.zeros((), dtype=torch.float64)
            target_penalty = torch.zeros((), dtype=torch.float64)
            nuisance_smooth = torch.zeros((), dtype=torch.float64)
            nuisance_orth = torch.zeros((), dtype=torch.float64)
            family_penalty = torch.zeros((), dtype=torch.float64)
            ell, _, _ = model.logit(tensor)
            ell_centered = ell - ell.mean()
            if graph_config.lambda_ != 0.0 and graph_config.mechanism == "target":
                numerator = _sparse_quadratic(ell_centered, laplacian)
                target_penalty = numerator / (ell_centered.square().sum() + EPS)
                penalty = target_penalty
            elif graph_config.lambda_ != 0.0 and graph_config.mechanism == "nuisance":
                U = nuisance(tensor)
                nuisance_smooth = _sparse_quadratic(U, laplacian) / (U.square().sum() + EPS)
                cross = U.T @ ell_centered[:, None]
                nuisance_orth = cross.square().sum() / (
                    (U.square().sum() + EPS) * (ell_centered.square().sum() + EPS)
                )
                penalty = nuisance_smooth + float(graph_config.orthogonality_gamma) * nuisance_orth
            elif graph_config.lambda_ != 0.0 and graph_config.mechanism == "family":
                family_penalty = _family_parameter_penalty(model, family_laplacian)
                penalty = family_penalty
            loss = base_loss + float(graph_config.lambda_) * penalty
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite objective at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            # Catch non-finite gradients before they are written into the
            # parameters.  The objective is checked above, but a NaN can enter
            # through a backward pass while the forward value is still finite,
            # which is precisely how the nuisance head used to die.
            for index, parameter in enumerate(parameters):
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(
                        f"non-finite gradient for parameter {index} at epoch {epoch}"
                    )
            optimizer.step()
            for index, parameter in enumerate(parameters):
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(
                        f"non-finite parameter {index} after step at epoch {epoch}"
                    )
            last_finite_state = model.state_dict_numpy()
            if nuisance is not None:
                last_finite_state.update({
                    "nuisance::W1": nuisance.W1.detach().cpu().numpy().copy(),
                    "nuisance::d1": nuisance.d1.detach().cpu().numpy().copy(),
                    "nuisance::W2": nuisance.W2.detach().cpu().numpy().copy(),
                    "nuisance::d2": nuisance.d2.detach().cpu().numpy().copy(),
                })
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach()),
                    "deem_loss": float(base_loss.detach()),
                    "target_penalty": float(target_penalty.detach()),
                    "nuisance_smoothness": float(nuisance_smooth.detach()),
                    "nuisance_orthogonality": float(nuisance_orth.detach()),
                    "family_penalty": float(family_penalty.detach()),
                    "mala_acceptance": acceptance,
                }
            )
    except Exception as exc:
        setattr(exc, "objective_history", history)
        setattr(exc, "last_finite_state", last_finite_state)
        raise

    nuisance_diagnostics = {}
    state = model.state_dict_numpy()
    with torch.no_grad():
        ell_t, contributions_t, family_t = model.logit(tensor)
        ell = ell_t.cpu().numpy()
        contributions = contributions_t.cpu().numpy()
        family = {name: values.cpu().numpy() for name, values in family_t.items()}
        if nuisance is not None:
            U = nuisance(tensor).cpu().numpy()
            covariance = U.T @ U / max(len(U) - 1, 1)
            centered_logit = ell - float(np.mean(ell))
            cross = U.T @ centered_logit
            dependence = float(
                np.sum(cross * cross)
                / ((np.sum(U * U) + EPS) * (np.sum(centered_logit * centered_logit) + EPS))
            )
            nuisance_diagnostics = {
                "nuisance_variance_min": float(np.min(np.var(U, axis=0))),
                "nuisance_whitening_max_abs": float(
                    np.max(np.abs(covariance - np.eye(covariance.shape[0])))
                ),
                "logit_nuisance_dependence": dependence,
            }
            state.update({
                "nuisance::W1": nuisance.W1.detach().cpu().numpy().copy(),
                "nuisance::d1": nuisance.d1.detach().cpu().numpy().copy(),
                "nuisance::W2": nuisance.W2.detach().cpu().numpy().copy(),
                "nuisance::d2": nuisance.d2.detach().cpu().numpy().copy(),
                "nuisance::U": U,
            })
    reconstruction = float(np.max(np.abs(model.b.detach().item() + contributions.sum(axis=1) - ell)))
    if reconstruction > 1e-8:
        raise ResidualGraphDeemError(f"contribution reconstruction failed: {reconstruction:.3e}")
    q = 1.0 / (1.0 + np.exp(-np.clip(ell, -700.0, 700.0)))
    anchor = equal_family_risk_anchor(X, names)
    high = float(np.sum(q * anchor) / max(np.sum(q), EPS))
    low = float(np.sum((1.0 - q) * anchor) / max(np.sum(1.0 - q), EPS))
    difference = high - low
    if abs(difference) <= float(config.anchor_tolerance):
        raise ResidualGraphDeemError(
            f"risk-anchor alignment ambiguous: abs difference {abs(difference):.3e}"
        )
    orientation = 1 if difference > 0 else -1
    if orientation < 0:
        q = 1.0 - q
        ell = -ell
        contributions = -contributions
        family = {name: -values for name, values in family.items()}
        difference = -difference
    posterior = np.column_stack([1.0 - q, q])
    posterior_sd = float(np.std(q))
    healthy = bool(
        posterior_sd >= float(config.posterior_sd_min)
        and np.isfinite(q).all()
        and all(np.isfinite(row["loss"]) for row in history)
    )
    return ContinuousDeemResult(
        score=q.copy(),
        posterior=posterior,
        logit=ell,
        contributions=contributions,
        family_contributions=family,
        aligned_bias=float(orientation * model.b.detach().item()),
        orientation=orientation,
        risk_anchor_difference=float(difference),
        feature_names=names,
        family_indices=model.groups,
        state=state,
        objective_history=history,
        health={
            "healthy": healthy,
            "posterior_sd": posterior_sd,
            "finite": bool(np.isfinite(q).all()),
            "contribution_reconstruction_max_abs": reconstruction,
            "epochs_completed": len(history),
            "mala_acceptance_mean": float(np.mean([row["mala_acceptance"] for row in history])),
            "runtime_seconds": float(time.perf_counter() - started),
            **nuisance_diagnostics,
        },
        config={"continuous": asdict(config), "graph": asdict(graph_config)},
        seed=int(seed),
    )


def predict_continuous_deem(result: ContinuousDeemResult, X_risk: np.ndarray) -> dict[str, Any]:
    import torch
    X = validate_inventory(X_risk, result.feature_names)
    config = ContinuousDeemConfig(**result.config["continuous"])
    model = _FamilyAdditiveEnergy(result.feature_names, config, result.seed)
    model.load_state_numpy(result.state)
    with torch.no_grad():
        ell_t, contribution_t, family_t = model.logit(torch.as_tensor(X, dtype=torch.float64))
    orientation = int(result.orientation)
    ell = orientation * ell_t.cpu().numpy()
    contribution = orientation * contribution_t.cpu().numpy()
    family = {name: orientation * values.cpu().numpy() for name, values in family_t.items()}
    score = 1.0 / (1.0 + np.exp(-np.clip(ell, -700.0, 700.0)))
    reconstruction = np.max(np.abs(result.aligned_bias + contribution.sum(axis=1) - ell))
    if reconstruction > 1e-8:
        raise ResidualGraphDeemError("held contribution reconstruction failed")
    return {
        "score": score,
        "posterior": np.column_stack([1.0 - score, score]),
        "logit": ell,
        "contributions": contribution,
        "family_contributions": family,
        "reconstruction_max_abs": float(reconstruction),
    }


def assign_grouped_length_folds(
    group_ids: Sequence[str],
    raw_lengths: Sequence[float],
    *,
    n_folds: int = 5,
    n_bins: int = 10,
) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=str)
    lengths = np.asarray(raw_lengths, dtype=float)
    if groups.shape != lengths.shape or groups.ndim != 1 or not np.isfinite(lengths).all():
        raise ValueError("group_ids and raw_lengths must be aligned finite vectors")
    unique = sorted(set(groups.tolist()))
    medians = {group: float(np.median(lengths[groups == group])) for group in unique}
    ordered = sorted(unique, key=lambda group: (medians[group], group))
    bins = np.array_split(np.asarray(ordered, dtype=str), min(int(n_bins), len(ordered)))
    assignment: dict[str, int] = {}
    for bin_index, values in enumerate(bins):
        for position, group in enumerate(values.tolist()):
            assignment[group] = int((position + bin_index) % int(n_folds))
    folds = np.asarray([assignment[group] for group in groups], dtype=np.int64)
    for group in unique:
        if len(np.unique(folds[groups == group])) != 1:
            raise AssertionError("sibling candidates were split across folds")
    return folds


def _id_hash(values: Sequence[str]) -> str:
    return canonical_sha256([str(value) for value in values])


def residualize_oof_contributions(
    contributions: np.ndarray,
    logit: np.ndarray,
    raw_lengths: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    values = np.asarray(contributions, dtype=float)
    ell = np.asarray(logit, dtype=float)
    lengths = np.asarray(raw_lengths, dtype=float)
    fold_values = np.asarray(folds, dtype=int)
    if values.ndim != 2 or ell.shape != (len(values),) or lengths.shape != (len(values),):
        raise ValueError("OOF residualizer inputs disagree")
    output = np.empty_like(values)
    records = []
    predictors = np.column_stack([ell, np.log1p(np.maximum(lengths, 0.0))])
    for fold in sorted(np.unique(fold_values)):
        held = np.flatnonzero(fold_values == fold)
        donor = np.flatnonzero(fold_values != fold)
        pred_transform = fit_standardization(predictors[donor])
        donor_predictors = apply_standardization(predictors[donor], pred_transform)
        held_predictors = apply_standardization(predictors[held], pred_transform)
        polynomial = PolynomialFeatures(degree=3, include_bias=False)
        donor_design = polynomial.fit_transform(donor_predictors)
        held_design = polynomial.transform(held_predictors)
        estimator = Ridge(alpha=1.0, fit_intercept=True)
        estimator.fit(donor_design, values[donor])
        donor_residual = values[donor] - estimator.predict(donor_design)
        held_residual = values[held] - estimator.predict(held_design)
        residual_transform = fit_standardization(donor_residual)
        output[held] = apply_standardization(held_residual, residual_transform)
        records.append(
            {
                "fold": int(fold),
                "donor_indices": donor.tolist(),
                "held_indices": held.tolist(),
                "predictor_mean": pred_transform.mean,
                "predictor_scale": pred_transform.scale,
                "predictor_constant_mask": pred_transform.constant_mask,
                "residual_mean": residual_transform.mean,
                "residual_scale": residual_transform.scale,
                "residual_constant_mask": residual_transform.constant_mask,
                "polynomial_powers": polynomial.powers_,
                "ridge_coef": estimator.coef_,
                "ridge_intercept": estimator.intercept_,
            }
        )
    if not np.isfinite(output).all():
        raise ResidualGraphDeemError("non-finite OOF residual")
    return output, tuple(records)


def donor_residualize_contributions(
    donor_contributions: np.ndarray,
    held_contributions: np.ndarray,
    donor_logit: np.ndarray,
    held_logit: np.ndarray,
    donor_lengths: np.ndarray,
    held_lengths: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and apply the frozen residualizer without cross-fold model leakage."""

    donor_values = np.asarray(donor_contributions, dtype=float)
    held_values = np.asarray(held_contributions, dtype=float)
    donor_predictors_raw = np.column_stack([
        np.asarray(donor_logit, dtype=float),
        np.log1p(np.maximum(np.asarray(donor_lengths, dtype=float), 0.0)),
    ])
    held_predictors_raw = np.column_stack([
        np.asarray(held_logit, dtype=float),
        np.log1p(np.maximum(np.asarray(held_lengths, dtype=float), 0.0)),
    ])
    if (
        donor_values.ndim != 2 or held_values.ndim != 2
        or donor_values.shape[1] != held_values.shape[1]
        or len(donor_values) != len(donor_predictors_raw)
        or len(held_values) != len(held_predictors_raw)
    ):
        raise ValueError("donor/held residualizer arrays disagree")
    predictor_transform = fit_standardization(donor_predictors_raw)
    donor_predictors = apply_standardization(donor_predictors_raw, predictor_transform)
    held_predictors = apply_standardization(held_predictors_raw, predictor_transform)
    polynomial = PolynomialFeatures(degree=3, include_bias=False)
    donor_design = polynomial.fit_transform(donor_predictors)
    held_design = polynomial.transform(held_predictors)
    estimator = Ridge(alpha=1.0, fit_intercept=True)
    estimator.fit(donor_design, donor_values)
    donor_residual = donor_values - estimator.predict(donor_design)
    held_residual = held_values - estimator.predict(held_design)
    residual_transform = fit_standardization(donor_residual)
    output = apply_standardization(held_residual, residual_transform)
    if not np.isfinite(output).all():
        raise ResidualGraphDeemError("non-finite donor-only held residual")
    return output, {
        "predictor_mean": predictor_transform.mean,
        "predictor_scale": predictor_transform.scale,
        "predictor_constant_mask": predictor_transform.constant_mask,
        "residual_mean": residual_transform.mean,
        "residual_scale": residual_transform.scale,
        "residual_constant_mask": residual_transform.constant_mask,
        "polynomial_powers": polynomial.powers_,
        "ridge_coef": estimator.coef_,
        "ridge_intercept": estimator.intercept_,
        "donor_model_shared_with_held_transform": True,
    }


def crossfit_continuous_deem(
    X_raw: np.ndarray,
    feature_names: Sequence[str],
    confidence_signs: Sequence[int],
    group_ids: Sequence[str],
    raw_lengths: Sequence[float],
    *,
    seed: int,
    config: ContinuousDeemConfig | None = None,
) -> CrossFitResult:
    X = validate_inventory(X_raw, feature_names)
    registered_signs = confidence_sign_vector(feature_names).astype(int)
    if not np.array_equal(np.asarray(confidence_signs, dtype=int), registered_signs):
        raise ResidualGraphDeemError("confidence-sign registry mismatch")
    groups = np.asarray(group_ids, dtype=str)
    lengths = np.asarray(raw_lengths, dtype=float)
    if groups.shape != (len(X),) or lengths.shape != (len(X),):
        raise ValueError("row metadata does not align to X_raw")
    folds = assign_grouped_length_folds(groups, lengths)
    oof_logit = np.empty(len(X), dtype=float)
    oof_posterior = np.empty((len(X), 2), dtype=float)
    oof_contribution = np.empty_like(X)
    oof_residual = np.empty_like(X)
    manifests = []
    fits = []
    residualizer_records = []
    config = config or ContinuousDeemConfig()
    for fold in range(5):
        held = np.flatnonzero(folds == fold)
        donor = np.flatnonzero(folds != fold)
        if not len(held) or not len(donor):
            raise ResidualGraphDeemError(f"empty donor/held fold: {fold}")
        if set(donor).intersection(held):
            raise AssertionError("donor/held overlap")
        donor_risk, held_risk, transform = donor_risk_matrix(
            X[donor], X[held], feature_names
        )
        fit = fit_continuous_deem(donor_risk, feature_names, seed=seed, config=config)
        prediction = predict_continuous_deem(fit, held_risk)
        donor_prediction = predict_continuous_deem(fit, donor_risk)
        oof_logit[held] = prediction["logit"]
        oof_posterior[held] = prediction["posterior"]
        oof_contribution[held] = prediction["contributions"]
        held_residual, residualizer_record = donor_residualize_contributions(
            donor_prediction["contributions"], prediction["contributions"],
            donor_prediction["logit"], prediction["logit"],
            lengths[donor], lengths[held],
        )
        oof_residual[held] = held_residual
        record = {
            "mean": transform.mean,
            "scale": transform.scale,
            "constant_mask": transform.constant_mask,
        }
        manifests.append(
            FoldManifest(
                fold=fold,
                donor_indices=tuple(int(value) for value in donor),
                held_indices=tuple(int(value) for value in held),
                donor_group_sha256=_id_hash(sorted(set(groups[donor].tolist()))),
                held_group_sha256=_id_hash(sorted(set(groups[held].tolist()))),
                standardization_sha256=canonical_sha256(record),
                residualizer_sha256=canonical_sha256(residualizer_record),
            )
        )
        fits.append(fit)
        residualizer_records.append({
            "fold": int(fold), "donor_indices": donor.tolist(),
            "held_indices": held.tolist(), **residualizer_record,
        })
    return CrossFitResult(
        logit=oof_logit,
        posterior=oof_posterior,
        contributions=oof_contribution,
        residuals=oof_residual,
        folds=folds,
        fold_manifests=manifests,
        fit_results=tuple(fits),
        residualizer_records=tuple(residualizer_records),
    )


def row_id_tie_keys(row_ids: Sequence[str]) -> np.ndarray:
    values = [str(value) for value in row_ids]
    if len(values) != len(set(values)):
        raise ValueError("row IDs must be unique")
    ordered = {value: index for index, value in enumerate(sorted(values))}
    return np.asarray([ordered[value] for value in values], dtype=float)


def build_inventory_graph(
    coordinates: np.ndarray,
    feature_names: Sequence[str],
    row_ids: Sequence[str],
    *,
    k: int = 7,
    gates: np.ndarray | None = None,
) -> csr_matrix:
    X = validate_inventory(coordinates, feature_names)
    weights = metric_weights(feature_names)
    scale = np.sqrt(weights)
    if gates is not None:
        gate_values = np.asarray(gates, dtype=float)
        if gate_values.shape != (X.shape[1],) or np.any(gate_values < 0) or not np.isfinite(gate_values).all():
            raise ValueError("gates must be finite nonnegative inventory weights")
        scale = gate_values
    graph = self_safe_knn_graph(
        X * scale[None, :], k=int(k), tie_keys=row_id_tie_keys(row_ids)
    ).tocsr()
    validate_sparse_graph(graph)
    return graph


def validate_sparse_graph(graph: csr_matrix) -> None:
    W = csr_matrix(graph, dtype=float)
    if W.shape[0] != W.shape[1] or W.shape[0] < 3:
        raise ValueError("graph must be square")
    if W.nnz and (not np.isfinite(W.data).all() or np.min(W.data) < 0):
        raise ValueError("graph weights must be finite and nonnegative")
    delta = W - W.T
    error = float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
    if error > 1e-10:
        raise ValueError(f"graph symmetry error {error:.3e}")
    if np.any(np.abs(W.diagonal()) > 1e-12):
        raise ValueError("graph contains self edges")


def graph_health(graph: csr_matrix) -> dict[str, Any]:
    W = csr_matrix(graph, dtype=float)
    validate_sparse_graph(W)
    count, labels = connected_components(W, directed=False)
    sizes = np.bincount(labels, minlength=count)
    degree = np.asarray(W.sum(axis=1)).ravel()
    isolated = degree <= EPS
    largest = float(np.max(sizes) / W.shape[0])
    isolated_fraction = float(np.mean(isolated))
    return {
        "n_nodes": int(W.shape[0]),
        "n_edges": int(W.nnz // 2),
        "n_components": int(count),
        "largest_component_fraction": largest,
        "isolated_fraction": isolated_fraction,
        "degree_min": float(np.min(degree)),
        "degree_mean": float(np.mean(degree)),
        "degree_max": float(np.max(degree)),
        "healthy": bool(largest >= 0.90 and isolated_fraction <= 0.05),
    }


def unique_edge_loss(values: np.ndarray, graph: csr_matrix) -> float:
    X = np.asarray(values, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    W = sparse.triu(csr_matrix(graph), k=1).tocoo()
    differences = X[W.row] - X[W.col]
    return float(np.sum(W.data[:, None] * differences * differences))


def normalized_rayleigh(values: np.ndarray, laplacian: csr_matrix) -> float:
    X = np.asarray(values, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    centered = X - X.mean(axis=0, keepdims=True)
    numerator = float(np.sum(centered * (laplacian @ centered)))
    denominator = float(np.sum(centered * centered)) + EPS
    return numerator / denominator


def _row_normalized_two_step(graph: csr_matrix) -> csr_matrix:
    W = csr_matrix(graph, dtype=float)
    degree = np.asarray(W.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    inverse[degree > EPS] = 1.0 / degree[degree > EPS]
    transition = sparse.diags(inverse) @ W
    return (transition @ transition).tocsr()


def _torch_sparse(matrix: csr_matrix, *, dtype, device):
    import torch
    coo = sparse.coo_matrix(matrix)
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, dtype=dtype).coalesce()


def _train_external_dufs(
    target_coordinates: np.ndarray,
    reference_graph: csr_matrix,
    *,
    seed: int,
    config: DufsConfig,
) -> tuple[np.ndarray, list[float]]:
    import torch
    X = np.asarray(target_coordinates, dtype=float)
    if X.ndim != 2 or len(X) != reference_graph.shape[0] or not np.isfinite(X).all():
        raise ValueError("DUFS target coordinates and external graph disagree")
    set_determinism(seed)
    tensor = torch.as_tensor(X, dtype=torch.float64)
    transition = _torch_sparse(
        _row_normalized_two_step(reference_graph), dtype=torch.float64, device="cpu"
    )
    mu = torch.full((X.shape[1],), float(config.mu0), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([mu], lr=float(config.learning_rate))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 4_000_037)
    history = []
    for _ in range(int(config.epochs)):
        noise = torch.randn(X.shape[1], generator=generator, dtype=torch.float64)
        gates = torch.clamp(mu + float(config.gate_sigma) * noise, 0.0, 1.0)
        selected = tensor * gates[None, :]
        propagated = torch.sparse.mm(transition, selected)
        trace = -(selected * propagated).sum() / len(X)
        probability = 0.5 * (
            1.0 + torch.erf(mu / (float(config.gate_sigma) * math.sqrt(2.0)))
        )
        loss = trace / (probability.sum() + EPS)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite cross-view DUFS objective")
        history.append(float(loss.detach()))
    probability = ndtr(mu.detach().cpu().numpy() / float(config.gate_sigma))
    return np.asarray(probability, dtype=float), history


def cross_view_dufs(
    residuals: np.ndarray,
    feature_names: Sequence[str],
    folds: Sequence[int],
    row_ids: Sequence[str],
    *,
    config: DufsConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    X = validate_inventory(residuals, feature_names)
    names = tuple(str(name) for name in feature_names)
    fold_values = np.asarray(folds, dtype=int)
    if fold_values.shape != (len(X),) or len(row_ids) != len(X):
        raise ValueError("DUFS fold/row metadata mismatch")
    config = config or DufsConfig()
    groups = family_index_map(names)
    per_family_seed = {}
    histories = {}
    raw_average = np.zeros(X.shape[1], dtype=float)
    for family, indices in groups.items():
        other = tuple(index for other_family, members in groups.items() if other_family != family for index in members)
        if not other:
            raise ResidualGraphDeemError("cross-view DUFS requires another present family")
        fold_seed = np.empty((len(np.unique(fold_values)), len(config.seeds), len(indices)), dtype=float)
        for fold_position, fold in enumerate(sorted(np.unique(fold_values))):
            donor = np.flatnonzero(fold_values != fold)
            other_names = tuple(names[index] for index in other)
            reference = build_inventory_graph(
                X[donor][:, other], other_names, np.asarray(row_ids)[donor], k=config.k
            )
            for seed_position, seed in enumerate(config.seeds):
                probability, history = _train_external_dufs(
                    X[donor][:, indices], reference, seed=int(seed), config=config
                )
                fold_seed[fold_position, seed_position] = probability
                histories[f"{family}::fold{int(fold)}::seed{int(seed)}"] = history
        by_seed = fold_seed.mean(axis=0)
        per_family_seed[family] = by_seed
        raw_average[list(indices)] = by_seed.mean(axis=0)
    if float(np.max(raw_average)) <= 0.05:
        raise ResidualGraphDeemError("all cross-view DUFS gates are closed")
    gates = np.zeros_like(raw_average)
    family_mass = {}
    cosines = []
    for family, indices in groups.items():
        values = raw_average[list(indices)]
        rms = float(np.sqrt(np.mean(values * values)))
        if rms <= EPS:
            raise ResidualGraphDeemError(f"DUFS family gates closed: {family}")
        gates[list(indices)] = values / rms * math.sqrt(1.0 / (len(groups) * len(indices)))
        family_mass[family] = float(np.sum(gates[list(indices)] ** 2))
        by_seed = per_family_seed[family]
        for left in range(len(by_seed)):
            for right in range(left + 1, len(by_seed)):
                denominator = np.linalg.norm(by_seed[left]) * np.linalg.norm(by_seed[right])
                cosines.append(float(np.dot(by_seed[left], by_seed[right]) / max(denominator, EPS)))
    median_cosine = float(np.median(cosines)) if cosines else 1.0
    if median_cosine < float(config.median_cosine_min):
        raise ResidualGraphDeemError(
            f"DUFS gate stability failed: median cosine {median_cosine:.4f}"
        )
    effective = float((np.sum(gates * gates) ** 2) / (np.sum(gates ** 4) + EPS))
    return gates, {
        "raw_probabilities": raw_average,
        "per_family_seed_probabilities": per_family_seed,
        "family_mass": family_mass,
        "effective_feature_count": effective,
        "median_seed_cosine": median_cosine,
        "histories": histories,
        "target_family_excluded_from_reference": True,
    }


def present_family_laplacian(
    residuals: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[csr_matrix, tuple[str, ...], np.ndarray]:
    X = validate_inventory(residuals, feature_names)
    groups = family_index_map(feature_names)
    order = tuple(groups)
    sums = np.column_stack([X[:, indices].sum(axis=1) for indices in groups.values()])
    correlation = spearmanr(sums, axis=0).statistic
    correlation = np.atleast_2d(np.asarray(correlation, dtype=float))
    affinity = np.abs(np.nan_to_num(correlation, nan=0.0))
    np.fill_diagonal(affinity, 0.0)
    W = csr_matrix(affinity)
    return symmetric_normalized_laplacian(W), order, affinity


def fold_artifact_diagnostics(
    coordinates: np.ndarray,
    folds: Sequence[int],
    group_ids: Sequence[str],
    graph: csr_matrix,
    *,
    permutations: int = 999,
    seed: int = 20260821,
) -> dict[str, Any]:
    X = np.asarray(coordinates, dtype=float)
    fold_values = np.asarray(folds, dtype=int)
    groups = np.asarray(group_ids, dtype=str)
    if X.ndim != 2 or fold_values.shape != (len(X),) or groups.shape != (len(X),):
        raise ValueError("fold-artifact inputs disagree")
    W = sparse.triu(csr_matrix(graph), k=1).tocoo()
    edge_weight = float(np.sum(W.data))

    def same_fold_rate(labels):
        return float(np.sum(W.data * (labels[W.row] == labels[W.col])) / max(edge_weight, EPS))

    def centroid_predictability(labels):
        total = float(np.sum((X - X.mean(axis=0)) ** 2)) + EPS
        between = 0.0
        for value in np.unique(labels):
            selected = X[labels == value]
            between += len(selected) * float(np.sum((selected.mean(axis=0) - X.mean(axis=0)) ** 2))
        return between / total

    observed_edge = same_fold_rate(fold_values)
    observed_predict = centroid_predictability(fold_values)
    unique_groups = sorted(set(groups.tolist()))
    group_fold = {group: int(fold_values[np.flatnonzero(groups == group)[0]]) for group in unique_groups}
    labels = np.asarray([group_fold[group] for group in unique_groups], dtype=int)
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    null_edge = np.empty(int(permutations), dtype=float)
    null_predict = np.empty(int(permutations), dtype=float)
    for index in range(int(permutations)):
        shuffled = generator.permutation(labels)
        mapping = dict(zip(unique_groups, shuffled.tolist()))
        permuted = np.asarray([mapping[group] for group in groups], dtype=int)
        null_edge[index] = same_fold_rate(permuted)
        null_predict[index] = centroid_predictability(permuted)
    p_edge = float((1 + np.sum(null_edge >= observed_edge)) / (len(null_edge) + 1))
    p_predict = float((1 + np.sum(null_predict >= observed_predict)) / (len(null_predict) + 1))
    return {
        "same_fold_edge_rate": observed_edge,
        "same_fold_edge_p": p_edge,
        "fold_centroid_predictability": observed_predict,
        "fold_predictability_p": p_predict,
        "permutations": int(permutations),
        "healthy": bool(p_edge >= 0.05 and p_predict >= 0.05),
    }


def random_gate_control(gates: np.ndarray, feature_names: Sequence[str], *, seed: int) -> np.ndarray:
    values = np.asarray(gates, dtype=float).copy()
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    for indices in family_index_map(feature_names).values():
        values[list(indices)] = generator.permutation(values[list(indices)])
    return values


def permute_graph_nodes(graph: csr_matrix, permutation: Sequence[int]) -> csr_matrix:
    order = np.asarray(permutation, dtype=int)
    W = csr_matrix(graph)
    if sorted(order.tolist()) != list(range(W.shape[0])):
        raise ValueError("node permutation must contain every row exactly once")
    output = W[order][:, order].tocsr()
    validate_sparse_graph(output)
    return output


def ensemble_seed_scores(results: Sequence[ContinuousDeemResult]) -> tuple[np.ndarray, dict[str, Any]]:
    if len(results) != 5 or sorted(result.seed for result in results) != list(SEEDS):
        raise ResidualGraphDeemError("ensemble requires exactly healthy seeds 0..4")
    if not all(result.health.get("healthy") for result in results):
        raise ResidualGraphDeemError("ensemble contains an unhealthy seed")
    scores = np.asarray([result.score for result in results], dtype=float)
    correlations = []
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            value = spearmanr(scores[left], scores[right]).statistic
            correlations.append(abs(float(value)))
    return scores.mean(axis=0), {
        "median_abs_spearman": float(np.median(correlations)),
        "minimum_abs_spearman": float(np.min(correlations)),
        "pairwise_abs_spearman": correlations,
    }


__all__ = [
    "ARM_SPECS", "ArmSpec", "ContinuousDeemConfig", "ContinuousDeemResult",
    "CrossFitResult", "DufsConfig", "FoldManifest", "FrozenFitManifest",
    "GraphDeemConfig", "LAMBDA_GRID", "ResidualGraphDeemError", "SCHEMA_VERSION",
    "SEEDS", "Standardization", "apply_standardization", "assign_grouped_length_folds",
    "atomic_save_npz", "atomic_write_json", "build_inventory_graph", "canonical_sha256",
    "cross_view_dufs", "crossfit_continuous_deem", "donor_residualize_contributions",
    "donor_risk_matrix",
    "ensemble_seed_scores", "environment_fingerprint", "equal_family_risk_anchor", "family_index_map",
    "fit_continuous_deem", "fit_standardization", "fold_artifact_diagnostics",
    "graph_health", "jsonable", "metric_weights", "normalized_rayleigh",
    "persistent_mala", "permute_graph_nodes", "predict_continuous_deem",
    "present_family_laplacian", "random_gate_control", "row_id_tie_keys",
    "sha256_file", "symmetric_normalized_laplacian", "unique_edge_loss",
    "validate_inventory", "validate_sparse_graph", "view_members",
]
