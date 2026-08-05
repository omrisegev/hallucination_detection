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

Both receive feature-relative signs from the incumbent U-PCR ``sign(rho)``
orientation before entering this module.  DEEM's label permutation is aligned
to majority vote, never to correctness labels, and the final continuous class-1
probability is returned for AUROC evaluation.

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


@dataclass
class DeemRunResult:
    score: np.ndarray
    aligned_probabilities: np.ndarray
    class_map: dict
    seed: int
    package_version: str
    config: dict
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


def fit_deem_score(X, *, seed=0, config=None, verbose=False):
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
    model.fit(predictions, verbose=bool(verbose))
    aligned, mapping = _aligned_probabilities(model, predictions)
    return DeemRunResult(
        score=aligned[:, 1].copy(),
        aligned_probabilities=aligned,
        class_map=mapping,
        seed=seed,
        package_version=package_version,
        config=asdict(config),
        history=_jsonable(getattr(model, "history_", {})),
    )
