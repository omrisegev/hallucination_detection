"""LOS-aligned direct probability fusion with selected-token and tail inputs.

This module extends the frozen v1 rank representation by two risk-oriented
coordinates that are already present in the gray-box caches:

* selected-token surprisal ``-log p(selected token)``;
* probability mass outside the retained top-K head.

It deliberately keeps the v1 K, fusion estimators, gate and readout unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .direct_probability_fusion import direct_rank_risk, logprob_matrix


# Saved log-probabilities are float32.  Across the complete frozen corpus the
# largest exp/log rounding excess above probability mass one is 3.35e-7.
TAIL_TOLERANCE = 5e-7


def selected_surprisal(value: Any, n_observations: int) -> np.ndarray:
    """Validate the cached selected-token negative log-probabilities."""

    values = np.asarray(value, dtype=float)
    if values.shape != (int(n_observations),):
        raise ValueError(
            "selected-token surprisal must align one-to-one with probability rows; "
            f"got {values.shape} for {n_observations} rows"
        )
    if not np.isfinite(values).all() or (values < -TAIL_TOLERANCE).any():
        raise ValueError("selected-token surprisal contains invalid values")
    return np.maximum(values, 0.0)


def residual_tail_mass(logprobs: np.ndarray) -> np.ndarray:
    """Return unobserved probability mass outside the supplied sorted head."""

    values = np.asarray(logprobs, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("tail mass needs a nonempty [observations, ranks] matrix")
    if not np.isfinite(values).all():
        raise ValueError("tail mass input contains nonfinite values")
    head_mass = np.exp(values).sum(axis=1)
    if (head_mass > 1.0 + TAIL_TOLERANCE).any():
        raise ValueError("saved top-K probability mass exceeds one")
    return np.clip(1.0 - head_mass, 0.0, 1.0)


def augmented_probability_risk(
    topk_value: Any,
    selected_token_surprisal: Any,
    *,
    k: int = 15,
) -> np.ndarray:
    """Build ``[rank risks, selected surprisal, residual tail]``.

    The rank columns are exactly the frozen v1 representation.  The two new
    columns are high-is-risk.  No top-K renormalization is performed.
    """

    logprobs = logprob_matrix(topk_value, k=k)
    selected = selected_surprisal(selected_token_surprisal, len(logprobs))
    tail = residual_tail_mass(logprobs)
    return np.column_stack((direct_rank_risk(logprobs), selected, tail))


def augmented_feature_names(k: int = 15) -> tuple[str, ...]:
    """Human-readable column names in exact matrix order."""

    return tuple(
        [f"rank_{index}_risk" for index in range(1, int(k) + 1)]
        + ["selected_token_surprisal", "residual_tail_mass"]
    )


__all__ = [
    "TAIL_TOLERANCE",
    "augmented_feature_names",
    "augmented_probability_risk",
    "residual_tail_mass",
    "selected_surprisal",
]
