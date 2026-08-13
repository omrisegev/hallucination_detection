"""Fixed shared-feature pipelines for RAG and reasoning applications.

The application pipelines in this module share one semantic feature contract:
the 28 token-resolved streams in :mod:`token_feature_views` plus a constant
per-response trace-length stream.  Together they cover the frozen 30-feature
``mixed-v2`` response contract; CUSUM magnitude and CUSUM location are two
reductions of the same positional stream, hence 29 streams rather than 30
independent token columns.  Trace length is deliberately constant across the
tokens of one answer: it may shift answer risk, but it cannot invent a local
peak.

The two applications differ only in their observed structure:

* RAG compares the same token features under full, no-context, and available
  leave-one-chunk-out evidence conditions.
* reasoning preserves one uninterrupted token trajectory and maps its frozen
  risk stream to supplied step boundaries only after fusion.

All fitting helpers are label-free by construction.  Evaluation labels and
operating-threshold calibration live in the experiment driver, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from .repeated_measurement_reliability import FixedMixedV2Transformer
from .token_feature_views import (
    BROAD_TOKEN_VIEWS,
    TOKEN_TO_GLOBAL_FEATURES,
    token_feature_views,
)
from .upcr import upcr_fit


CONTRACT_VERSION = "shared-token-mixed-v2-applications-v1-2026-08-13"
MAX_FIT_TOKENS = 60_000
EPS = 1e-12

SHARED_TOKEN_VIEWS = ("trace_length_series",) + tuple(BROAD_TOKEN_VIEWS)
SHARED_GLOBAL_FEATURES = ("trace_length",) + tuple(
    TOKEN_TO_GLOBAL_FEATURES[name][0] for name in BROAD_TOKEN_VIEWS
)

# The exact/approximate distinction is inherited from token_feature_views.py.
APPROXIMATE_TOKEN_VIEWS = frozenset({
    "entropy_pe_series",
    "entropy_stft_high_series",
    "entropy_stft_frame_entropy",
    "entropy_rolling_tail_ratio",
    "entropy_rolling_spectral_entropy",
    "entropy_rolling_low_band_power",
    "entropy_rolling_high_band_power",
    "entropy_rolling_hl_ratio",
    "entropy_rolling_dominant_freq",
    "entropy_rolling_spectral_centroid",
    "entropy_rolling_rs_hurst",
})
EXACT_TOKEN_VIEWS = frozenset(SHARED_TOKEN_VIEWS) - APPROXIMATE_TOKEN_VIEWS

RAG_NOCTX_BLOCKS = ("full", "noctx_drop")
RAG_LOO_BLOCKS = (
    "full",
    "noctx_drop",
    "loo_max_drop",
    "loo_top2_mean_drop",
    "loo_positive_mean_drop",
    "loo_negative_std",
)


def raw_token_feature_matrix(row: Mapping[str, Any]) -> np.ndarray:
    """Return ``tokens × 29`` raw streams in the frozen shared order."""

    views = token_feature_views(dict(row))
    token_count = len(views[BROAD_TOKEN_VIEWS[0]])
    matrix = np.column_stack([
        np.full(token_count, float(token_count)),
        *[views[name] for name in BROAD_TOKEN_VIEWS],
    ])
    if matrix.shape[1] != len(SHARED_TOKEN_VIEWS):
        raise RuntimeError("shared token feature count changed")
    if matrix.shape[0] == 0:
        raise ValueError("a token trace must not be empty")
    return np.asarray(matrix, dtype=float)


def condition_trace_row(trace: Any) -> dict[str, Any]:
    """Adapt a canonical RAG condition trace to ``raw_token_feature_matrix``."""

    return {
        "token_entropies": np.asarray(trace.entropy, dtype=float),
        "token_spilled_energies": -np.asarray(trace.target_logprob, dtype=float),
        "token_logsumexp": np.asarray(trace.logsumexp, dtype=float),
        "top_k_logprobs": {
            "ids": np.asarray(trace.top_ids),
            "logprobs": np.asarray(trace.top_logprobs, dtype=float),
        },
    }


def _ordered_token_sample(
    records: Sequence[tuple[str, np.ndarray]],
    max_tokens: int = MAX_FIT_TOKENS,
) -> np.ndarray:
    """Deterministically sample a sorted collection of token matrices.

    Sorting by stable unit ID makes the sample invariant to input row order.
    Evenly spaced positions avoid a random-state dependency and keep coverage
    across the complete unlabeled fit population.
    """

    ordered = [np.asarray(values, dtype=float) for _, values in sorted(records)]
    if not ordered:
        raise ValueError("at least one token matrix is required")
    width = ordered[0].shape[1]
    if any(item.ndim != 2 or item.shape[1] != width for item in ordered):
        raise ValueError("token matrices must have one shared column count")
    joined = np.vstack(ordered)
    if len(joined) <= int(max_tokens):
        return joined
    indexes = np.linspace(0, len(joined) - 1, int(max_tokens), dtype=int)
    return joined[indexes]


def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    medians = np.nanmedian(values, axis=0)
    clean = np.where(np.isfinite(values), values, medians[None, :])
    scale = clean.std(axis=0)
    keep = np.isfinite(medians) & np.isfinite(scale) & (scale > 1e-8)
    if int(keep.sum()) < 3:
        raise ValueError("fewer than three non-degenerate fusion columns remain")
    mean = clean[:, keep].mean(axis=0)
    std = clean[:, keep].std(axis=0)
    return (clean[:, keep] - mean[None, :]) / std[None, :], keep, mean, std


def _fit_iu_confidence(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit frozen full-pool two-component IU-PCR to confidence-oriented columns."""

    standardized, keep, mean, std = _standardize_fit(values)
    F = standardized.T
    fitted = upcr_fit(
        F,
        loss="l2",
        exclusion=False,
        difficulty_gate=False,
        simple_avg_fallback=False,
        recompute_after_exclusion=False,
        g2_projection_k=1,
        scale_ratio=0.25,
        n_components=2,
        auto_components=False,
    )
    weights = np.asarray(fitted.w, dtype=float)
    anchor = standardized.mean(axis=1)
    raw = standardized @ weights
    corr = float(np.corrcoef(raw, anchor)[0, 1])
    flipped = bool(np.isfinite(corr) and corr < 0)
    if flipped:
        weights = -weights
    return weights, keep, mean, std, {
        "g2_hat": float(fitted.g2_hat),
        "projection_residual": float(fitted.proj_residual),
        "weight_norm": float(np.linalg.norm(weights)),
        "orientation_correlation": corr,
        "orientation_flipped": flipped,
        "labels_seen_during_fit": False,
        "components": 2,
        "scale_ratio": 0.25,
    }


@dataclass
class SharedTokenIUModel:
    transformer: FixedMixedV2Transformer
    weights: np.ndarray
    keep: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    diagnostics: dict

    def risk(self, raw_matrix: np.ndarray) -> np.ndarray:
        confidence = self.transformer.transform(np.asarray(raw_matrix, dtype=float))
        selected = confidence[:, self.keep]
        clean = np.where(np.isfinite(selected), selected, self.mean[None, :])
        standardized = (clean - self.mean[None, :]) / self.std[None, :]
        return -(standardized @ self.weights)


def fit_shared_token_iu(
    records: Sequence[tuple[str, np.ndarray]],
    *,
    max_fit_tokens: int = MAX_FIT_TOKENS,
) -> SharedTokenIUModel:
    """Fit the shared mixed-v2 transform and IU-PCR on unlabeled token rows."""

    raw_fit = _ordered_token_sample(records, max_tokens=max_fit_tokens)
    transformer = FixedMixedV2Transformer.fit(raw_fit, SHARED_GLOBAL_FEATURES)
    confidence = transformer.training_output
    weights, keep, mean, std, diagnostics = _fit_iu_confidence(confidence)
    diagnostics.update({
        "contract": CONTRACT_VERSION,
        "n_fit_tokens": int(len(raw_fit)),
        "n_input_streams": len(SHARED_TOKEN_VIEWS),
        "n_kept_streams": int(keep.sum()),
        "exact_streams": len(EXACT_TOKEN_VIEWS),
        "approximate_streams": len(APPROXIMATE_TOKEN_VIEWS),
    })
    return SharedTokenIUModel(transformer, weights, keep, mean, std, diagnostics)


def fit_shared_mixed_transformer(
    records: Sequence[tuple[str, np.ndarray]],
    *,
    max_fit_tokens: int = MAX_FIT_TOKENS,
) -> FixedMixedV2Transformer:
    """Fit only the shared mixed-v2 coordinate system on unlabeled tokens."""

    raw_fit = _ordered_token_sample(records, max_tokens=max_fit_tokens)
    return FixedMixedV2Transformer.fit(raw_fit, SHARED_GLOBAL_FEATURES)


def rag_evidence_matrix(
    condition_raw: Mapping[str, np.ndarray],
    transformer: FixedMixedV2Transformer,
    *,
    profile: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convert ``X[t,c,f]`` to token rows for the fixed evidence head.

    Every output column is confidence-oriented.  Positive full-minus-removed
    values mean that the removed evidence supported confidence.  ``-std`` is
    used because stable support should also point toward groundedness.
    """

    if "full" not in condition_raw or "noctx" not in condition_raw:
        raise ValueError("RAG evidence requires full and no-context conditions")
    transformed = {
        name: transformer.transform(np.asarray(values, dtype=float))
        for name, values in condition_raw.items()
    }
    full = transformed["full"]
    if any(values.shape != full.shape for values in transformed.values()):
        raise ValueError("RAG condition token grids are not aligned")
    blocks = [full, full - transformed["noctx"]]
    block_names = list(RAG_NOCTX_BLOCKS)
    if profile == "noctx":
        pass
    elif profile == "loo":
        loo_names = sorted(
            (name for name in transformed if name.startswith("loo_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if not loo_names:
            raise ValueError("the LOO profile requires at least one LOO condition")
        drops = np.stack([full - transformed[name] for name in loo_names], axis=0)
        ordered = np.sort(drops, axis=0)
        maximum = ordered[-1]
        second = ordered[-2] if len(ordered) > 1 else ordered[-1]
        positive = drops > 0.0
        positive_sum = np.where(positive, drops, 0.0).sum(axis=0)
        positive_count = positive.sum(axis=0)
        positive_mean = np.divide(
            positive_sum,
            positive_count,
            out=np.zeros_like(positive_sum),
            where=positive_count > 0,
        )
        blocks.extend([
            maximum,
            0.5 * (maximum + second),
            positive_mean,
            -drops.std(axis=0),
        ])
        block_names = list(RAG_LOO_BLOCKS)
    else:
        raise ValueError("profile must be 'noctx' or 'loo'")
    values = np.column_stack(blocks)
    names = tuple(
        f"{block}::{feature}"
        for block in block_names
        for feature in SHARED_TOKEN_VIEWS
    )
    if values.shape[1] != len(names):
        raise RuntimeError("RAG evidence column registry disagrees with matrix")
    return values, names


@dataclass
class RagEvidenceIUHead:
    profile: str
    feature_names: tuple[str, ...]
    weights: np.ndarray
    keep: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    diagnostics: dict

    def risk(self, evidence_matrix: np.ndarray) -> np.ndarray:
        selected = np.asarray(evidence_matrix, dtype=float)[:, self.keep]
        clean = np.where(np.isfinite(selected), selected, self.mean[None, :])
        standardized = (clean - self.mean[None, :]) / self.std[None, :]
        return -(standardized @ self.weights)


def fit_rag_evidence_head(
    records: Sequence[tuple[str, np.ndarray, tuple[str, ...]]],
    *,
    profile: str,
    max_fit_tokens: int = MAX_FIT_TOKENS,
) -> RagEvidenceIUHead:
    """Fit IU-PCR to already transformed evidence blocks."""

    if not records:
        raise ValueError("at least one RAG evidence record is required")
    first_names = records[0][2]
    if any(names != first_names for _, _, names in records):
        raise ValueError("RAG evidence records have different feature registries")
    fit = _ordered_token_sample(
        [(sample_id, matrix) for sample_id, matrix, _ in records],
        max_tokens=max_fit_tokens,
    )
    weights, keep, mean, std, diagnostics = _fit_iu_confidence(fit)
    diagnostics.update({
        "contract": CONTRACT_VERSION,
        "profile": profile,
        "n_fit_tokens": int(len(fit)),
        "n_input_columns": len(first_names),
        "n_kept_columns": int(keep.sum()),
        "blocks": list(RAG_NOCTX_BLOCKS if profile == "noctx" else RAG_LOO_BLOCKS),
    })
    return RagEvidenceIUHead(profile, first_names, weights, keep, mean, std, diagnostics)


def aggregate_risk(values: Sequence[float], operator: str) -> float:
    """Frozen final adapters used by the application experiment."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    if operator == "mean":
        return float(values.mean())
    if operator == "max":
        return float(values.max())
    if operator == "q90":
        return float(np.quantile(values, 0.90))
    if operator == "top20_mean":
        count = max(1, int(np.ceil(0.20 * len(values))))
        return float(np.mean(np.sort(values)[-count:]))
    raise ValueError(f"unknown risk aggregation operator: {operator}")


def contract_audit() -> dict[str, Any]:
    covered = ["trace_length"] + [
        feature for values in TOKEN_TO_GLOBAL_FEATURES.values() for feature in values
    ]
    return {
        "version": CONTRACT_VERSION,
        "token_stream_count": len(SHARED_TOKEN_VIEWS),
        "covered_global_feature_count": len(covered),
        "covered_global_features": covered,
        "exact_stream_count": len(EXACT_TOKEN_VIEWS),
        "approximate_stream_count": len(APPROXIMATE_TOKEN_VIEWS),
        "exact_streams": sorted(EXACT_TOKEN_VIEWS),
        "approximate_streams": sorted(APPROXIMATE_TOKEN_VIEWS),
        "token_to_global_features": {
            "trace_length_series": ["trace_length"],
            **{
                name: list(TOKEN_TO_GLOBAL_FEATURES[name])
                for name in BROAD_TOKEN_VIEWS
            },
        },
    }


def smoke() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for index in range(6):
        n = 80 + index
        lp = -np.sort(-rng.normal(-4.0, 0.7, size=(n, 8)), axis=1)
        rows.append({
            "token_entropies": rng.uniform(0.1, 2.0, n),
            "token_spilled_energies": rng.uniform(0.1, 5.0, n),
            "token_logsumexp": rng.uniform(1.0, 6.0, n),
            "top_k_logprobs": {"logprobs": lp},
        })
    matrices = [(str(index), raw_token_feature_matrix(row)) for index, row in enumerate(rows)]
    model = fit_shared_token_iu(matrices, max_fit_tokens=300)
    score = model.risk(matrices[0][1])
    assert score.shape == (80,) and np.isfinite(score).all()
    assert len(contract_audit()["covered_global_features"]) == 30
    base = model.transformer
    condition = {"full": matrices[0][1], "noctx": matrices[0][1] + 0.01}
    noctx, names = rag_evidence_matrix(condition, base, profile="noctx")
    assert noctx.shape == (80, 58) and len(names) == 58
    condition["loo_0"] = matrices[0][1] + 0.02
    condition["loo_1"] = matrices[0][1] - 0.01
    loo, names = rag_evidence_matrix(condition, base, profile="loo")
    assert loo.shape == (80, 174) and len(names) == 174
    print("fixed_application_pipelines.smoke: PASS")


__all__ = [
    "APPROXIMATE_TOKEN_VIEWS",
    "CONTRACT_VERSION",
    "EXACT_TOKEN_VIEWS",
    "RAG_LOO_BLOCKS",
    "RAG_NOCTX_BLOCKS",
    "SHARED_GLOBAL_FEATURES",
    "SHARED_TOKEN_VIEWS",
    "RagEvidenceIUHead",
    "SharedTokenIUModel",
    "aggregate_risk",
    "condition_trace_row",
    "contract_audit",
    "fit_rag_evidence_head",
    "fit_shared_mixed_transformer",
    "fit_shared_token_iu",
    "rag_evidence_matrix",
    "raw_token_feature_matrix",
]


if __name__ == "__main__":
    smoke()
