"""Historical raw29 -> retained hist26 channels for the locked family transfer.

The source ``hist29_align.py`` reduces the float64 ``raw.npy`` arrays built
by ``fixed_application_pipelines.raw_token_feature_matrix``.  In particular,
there is NO float32 round trip here (unlike the separate bank11 cache).
The already-vendored token views and scalar helpers are text-identical to
the source historical modules; reusing them avoids importing fitting code.

These are offline full-answer views.  Prefix backfill, centered CUSUM and
interpolated STFT features do not define a causal streaming estimator.
This module does not orient or standardize channels: source signs and
answer-z are applied by the transfer scorer after the Top10 reduction.
"""
from __future__ import annotations

import numpy as np

from .external_generalization._bank11.token_feature_views import (
    BROAD_TOKEN_VIEWS,
    token_feature_views,
)


STREAM_NAMES = ("trace_length_series", *BROAD_TOKEN_VIEWS)
RETAINED_STREAM_NAMES = tuple(
    name for name in STREAM_NAMES
    if name not in {"trace_length_series", "entropy_series", "spilled_series"}
)
HIST_NAMES = tuple("hist_" + name for name in RETAINED_STREAM_NAMES)
RETAINED_INDICES = tuple(STREAM_NAMES.index(name) for name in RETAINED_STREAM_NAMES)


def token_streams(row: dict) -> np.ndarray:
    """Return all 29 historical, original-direction token streams as float64.

    ``token_entropies`` is the saved normalized top15 entropy, not the separate
    full-vocabulary entropy. Secondary handling and short-trace defaults are
    inherited unchanged from the historical extractor. Missing secondary
    telemetry therefore produces NaNs, which the caller's completeness gate
    must reject; it is never replaced with a newly invented stream.
    """
    views = token_feature_views(row)
    n = len(views[BROAD_TOKEN_VIEWS[0]])
    if n == 0:
        raise ValueError("a token trace must not be empty")
    matrix = np.column_stack([
        np.full(n, float(n)),
        *[views[name] for name in BROAD_TOKEN_VIEWS],
    ])
    if matrix.shape != (n, 29):
        raise ValueError("historical token stream schema changed")
    return np.asarray(matrix, dtype=np.float64)


def reduce_steps(matrix: np.ndarray, spans: np.ndarray) -> dict[str, np.ndarray]:
    """Historical finite-only Top10 means; empty segments stay NaN.

    The original hist29 aligner removes nonfinite values independently for each
    channel before choosing the largest min(10, count) values. Returning NaN
    for an empty segment preserves official empty-step routing in the caller.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    spans = np.asarray(spans)
    if matrix.ndim != 2 or matrix.shape[1] != 29:
        raise ValueError("historical token matrix must have 29 columns")
    if spans.ndim != 2 or spans.shape[1] != 2 or not np.issubdtype(spans.dtype, np.integer):
        raise ValueError("step spans must be an integer (steps, 2) array")
    if (spans < 0).any() or (spans[:, 1] > len(matrix)).any() or (spans[:, 0] > spans[:, 1]).any():
        raise ValueError("invalid historical step span")
    out = np.full((len(spans), len(RETAINED_INDICES)), np.nan)
    for j, (a, b) in enumerate(spans):
        for k, column in enumerate(RETAINED_INDICES):
            values = matrix[a:b, column]
            values = values[np.isfinite(values)]
            if len(values):
                count = min(10, len(values))
                out[j, k] = np.partition(values, len(values) - count)[-count:].mean()
    return {name: out[:, k] for k, name in enumerate(HIST_NAMES)}


def step_features(row: dict, spans: np.ndarray) -> dict[str, np.ndarray]:
    """Return the 26 retained hist_* raw step channels in lock order."""
    return reduce_steps(token_streams(row), spans)
