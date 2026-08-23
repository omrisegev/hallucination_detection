"""A narrow, label-free adapter from continuous detector views to DEEM 0.2.0.

DEEM's paper and API consume categorical learner predictions.  Our inputs are
continuous hallucination/correctness scores with incomparable scales.  The two
registered adaptations are therefore:

``hard``
    Per-view empirical median split.  This is closest to the binary-classifier
    setting of Shaham et al. (2016) and Maymon et al. (2026).

``soft``
    Per-view empirical ranks interpreted as two-class pseudo-probabilities
    ``[1-r, r]``.  These are not claimed to be calibrated probabilities; the
    transform simply preserves ordering and more within-view information than
    the median split.

Both receive already risk-oriented inventory coordinates.  The package's
majority-vote permutation is retained as a diagnostic, while the frozen v1
experiment aligns the public class to an external target-free risk consensus.

The API was verified against the pinned ``deem==0.2.0`` wheel.  In particular,
``predict(return_probs=True)`` returns *unaligned* probabilities, so this adapter
first requests an aligned hard prediction to obtain the Hungarian class map and
then applies that map to the probability columns.  Omitting this step can mirror
AUROC silently.

Paper: https://arxiv.org/abs/2601.20556
Code:  https://github.com/shaham-lab/deem
"""

from dataclasses import asdict, dataclass, field
from importlib import metadata
import random

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "DEEM_PINNED_VERSION",
    "DeemConfig",
    "DeemRunResult",
    "continuous_to_deem_hard",
    "continuous_to_deem_soft",
    "hard_adapter020_config",
    "repaired_soft_adapter020_config",
    "risk_consensus_align",
    "fit_deem_score",
]


DEEM_PINNED_VERSION = "0.2.0"


def _validate_continuous(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (samples, features)")
    if X.shape[0] < 3 or X.shape[1] < 3:
        raise ValueError("DEEM experiment needs at least 3 samples and 3 features")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")
    return X


def continuous_to_deem_soft(X, *, clip=1e-3):
    """Map oriented continuous views to ``(N, 2, M)`` rank probabilities."""
    X = _validate_continuous(X)
    n, m = X.shape
    probs = np.empty_like(X, dtype=float)
    for j in range(m):
        # Average ranks make ties deterministic.  The half-offset avoids exact
        # zero/one probabilities even before the explicit numerical clip.
        probs[:, j] = (rankdata(X[:, j], method="average") - 0.5) / n
    probs = np.clip(probs, float(clip), 1.0 - float(clip))
    out = np.empty((n, 2, m), dtype=float)
    out[:, 0, :] = 1.0 - probs
    out[:, 1, :] = probs
    return out


def continuous_to_deem_hard(X):
    """Map oriented continuous views to deterministic median-split labels."""
    soft = continuous_to_deem_soft(X)
    return (soft[:, 1, :] >= 0.5).astype(np.int64)


@dataclass(frozen=True)
class DeemConfig:
    """Fully explicit DEEM configuration used by the registered experiment."""

    input_mode: str = "soft"
    use_preprocessing: bool = True
    preprocessing_layers: int = 1
    preprocessing_activation: str = "sparsemax"
    preprocessing_init: str = "identity"
    hidden_dim: int = 1
    cd_k: int = 10
    deterministic: bool = True
    learning_rate: float = 0.001
    momentum: float = 0.9
    epochs: int = 100
    batch_size: int = 1024
    sampler_steps: int = 5
    use_weighted: bool = True
    init_method: str = "mv_rand"
    device: str = "auto"
    strict_version: bool = True
    alignment: str = "majority_vote"
    anchor_tolerance: float = 1e-6
    # Amendment A1.1: what to do when the risk-consensus orientation is
    # ambiguous (|high-low| <= tolerance, i.e. a degenerate posterior).
    # "raise" preserves the historical fail-closed behavior and stays the
    # default for every caller; "identity" adopts the identity class map
    # deterministically -- orientation of a zero-signal posterior carries
    # no information in either direction, and the produced score then
    # surfaces through the health record as collapsed rather than as a
    # worker crash.  Only the deem-vs-iupcr adapter worker opts in.
    alignment_ambiguous: str = "raise"


@dataclass
class DeemRunResult:
    score: np.ndarray
    aligned_probabilities: np.ndarray
    class_map: dict
    seed: int
    package_version: str
    config: dict
    package_class_map: dict = field(default_factory=dict)
    alignment: str = "majority_vote"
    history: dict = field(default_factory=dict)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _aligned_probabilities(model, predictions):
    """Apply DEEM's unsupervised majority-vote Hungarian map to probabilities."""
    # This computes and caches class_map_.  No labels are passed.
    model.predict(predictions, align_to=predictions)
    mapping = model.get_class_mapping() or {}
    raw = np.asarray(model.predict(predictions, return_probs=True), dtype=float)
    if raw.ndim == 1:
        raw = np.column_stack([1.0 - raw, raw])
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise RuntimeError(f"expected DEEM binary probabilities (N,2), got {raw.shape}")

    aligned = np.zeros_like(raw)
    for raw_class in range(raw.shape[1]):
        aligned_class = int(mapping.get(raw_class, raw_class))
        aligned[:, aligned_class] = raw[:, raw_class]
    row_sum = aligned.sum(axis=1, keepdims=True)
    aligned = aligned / np.maximum(row_sum, 1e-12)
    return aligned, {int(k): int(v) for k, v in mapping.items()}


def hard_adapter020_config(**overrides):
    """Frozen packaged 0.2.0 hard-control configuration."""
    values = dict(input_mode="hard", learning_rate=1e-3, epochs=100,
                  use_preprocessing=True, preprocessing_layers=1,
                  preprocessing_activation="sparsemax", preprocessing_init="identity",
                  hidden_dim=1, sampler_steps=5, batch_size=1024, momentum=0.9,
                  use_weighted=True, init_method="mv_rand", alignment="risk_consensus")
    values.update(overrides)
    return DeemConfig(**values)


def repaired_soft_adapter020_config(**overrides):
    """Frozen packaged 0.2.0 repaired soft-rank control configuration."""
    values = asdict(hard_adapter020_config())
    values.update(input_mode="soft", learning_rate=1e-4)
    values.update(overrides)
    return DeemConfig(**values)


def risk_consensus_align(probabilities, X_risk, *, feature_names=None, tolerance=1e-6,
                         ambiguous="raise"):
    """Align a binary latent posterior to the external equal-family risk anchor.

    ``ambiguous`` selects the Amendment A1.1 policy for a degenerate
    posterior: "raise" (default, historical) or "identity" (deterministic
    identity class map, margin reported as the sub-tolerance difference).
    """
    raw = np.asarray(probabilities, dtype=float)
    X = _validate_continuous(X_risk)
    if raw.shape != (len(X), 2):
        raise ValueError("probabilities must have shape (N,2)")
    if feature_names is None:
        anchor = X.mean(axis=1)
    else:
        from .residual_graph_deem import equal_family_risk_anchor
        anchor = equal_family_risk_anchor(X, feature_names)
    q = raw[:, 1]
    high = float(np.sum(q * anchor) / max(np.sum(q), 1e-12))
    low = float(np.sum((1.0 - q) * anchor) / max(np.sum(1.0 - q), 1e-12))
    difference = high - low
    if abs(difference) <= float(tolerance):
        if ambiguous == "identity":
            return raw.copy(), {0: 0, 1: 1}, float(difference)
        raise ValueError("risk-consensus alignment is ambiguous")
    if difference < 0:
        return raw[:, ::-1].copy(), {0: 1, 1: 0}, float(-difference)
    return raw.copy(), {0: 0, 1: 1}, float(difference)


def fit_deem_score(X, *, seed=0, config=None, feature_names=None, verbose=False):
    """Fit DEEM without labels and return its aligned continuous class-1 score."""
    X = _validate_continuous(X)
    config = config or DeemConfig()
    if config.input_mode not in ("hard", "soft"):
        raise ValueError("DeemConfig.input_mode must be 'hard' or 'soft'")

    try:
        package_version = metadata.version("deem")
        from deem import DEEM
    except Exception as exc:
        raise RuntimeError(
            "DEEM is unavailable. Install the registered dependency with "
            "`pip install -e \".[dependency-experiment]\"`."
        ) from exc
    if config.strict_version and package_version != DEEM_PINNED_VERSION:
        raise RuntimeError(
            f"DEEM version drift: found {package_version}, registered "
            f"{DEEM_PINNED_VERSION}. Refusing to produce incomparable numbers."
        )

    predictions = (continuous_to_deem_hard(X) if config.input_mode == "hard"
                   else continuous_to_deem_soft(X))
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

    model = DEEM(
        n_classes=2,
        hidden_dim=int(config.hidden_dim),
        cd_k=int(config.cd_k),
        deterministic=bool(config.deterministic),
        learning_rate=float(config.learning_rate),
        momentum=float(config.momentum),
        epochs=int(config.epochs),
        batch_size=min(int(config.batch_size), len(X)),
        device=config.device,
        auto_hyperparameters=False,
        random_state=seed,
        use_preprocessing=bool(config.use_preprocessing),
        preprocessing_layers=int(config.preprocessing_layers),
        preprocessing_activation=config.preprocessing_activation,
        preprocessing_init=config.preprocessing_init,
        sampler_steps=int(config.sampler_steps),
        sampler_oh_mode=(config.input_mode == "soft"),
        use_weighted=bool(config.use_weighted),
        init_method=config.init_method,
    )
    try:
        model.fit(predictions, verbose=bool(verbose))
    except Exception as exc:
        setattr(exc, "deem_history", _jsonable(getattr(model, "history_", {})))
        setattr(exc, "deem_config", asdict(config))
        raise
    package_aligned, package_mapping = _aligned_probabilities(model, predictions)
    margin = float("inf")
    if config.alignment == "majority_vote":
        aligned, mapping = package_aligned, package_mapping
    elif config.alignment == "risk_consensus":
        raw = np.asarray(model.predict(predictions, return_probs=True), dtype=float)
        if raw.ndim == 1:
            raw = np.column_stack([1.0 - raw, raw])
        aligned, mapping, margin = risk_consensus_align(
            raw, X, feature_names=feature_names, tolerance=config.anchor_tolerance,
            ambiguous=config.alignment_ambiguous,
        )
    else:
        raise ValueError("alignment must be 'majority_vote' or 'risk_consensus'")
    return DeemRunResult(
        score=aligned[:, 1].copy(),
        aligned_probabilities=aligned,
        class_map=mapping,
        package_class_map=package_mapping,
        alignment=(
            "risk_consensus_identity_fallback"
            if (config.alignment == "risk_consensus"
                and abs(margin) <= float(config.anchor_tolerance))
            else config.alignment
        ),
        seed=seed,
        package_version=package_version,
        config=asdict(config),
        history=_jsonable(getattr(model, "history_", {})),
    )
