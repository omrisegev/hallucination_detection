"""Label-free answer-gate features and fixed readouts for the q15 finalist."""
from __future__ import annotations

import numpy as np

from .direct_probability_fusion import logprob_matrix
from .direct_probability_fusion_v2 import residual_tail_mass
from .probability_normalization_ablation import step_readout
from .renyi_alpha_sweep import escort_varentropy
from .renyi_view_fusion import EPS, head_distribution, renyi_entropy
from .uniform_multiscale_fusion import feature_bank as multiscale_feature_bank


SIGNAL_NAMES = (
    "entropy_native",
    "q15_H0lim",
    "q15_VE0",
    "q15_VE0.75",
    "q15_VE1",
    "q15_H1",
    "q15_Hinf",
    "raw_neglogp1",
    "tail15_mass",
    "tail50_mass",
    "q15_raw4_mean",
)
READOUT_NAMES = ("token_mean", "token_top10", "mean_step_top10")
METHODS = tuple(f"{signal}__{readout}" for signal in SIGNAL_NAMES for readout in READOUT_NAMES)
BASELINE = "entropy_native__token_mean"

# Cross-panel answer-level readouts. These intentionally do not depend on
# ProcessBench step boundaries, so one definition can be selected on the
# historical math cells and transferred unchanged.
TEMPORAL_READOUT_NAMES = (
    "token_mean",
    "token_top10",
    "token_q75",
    "token_q90",
    "token_q95",
    "rolling8_max",
    "region_q1_mean",
    "region_q2_mean",
    "region_q3_mean",
    "region_q4_mean",
    "position_slope",
)
TEMPORAL_METHODS = tuple(
    f"{signal}__{readout}"
    for signal in SIGNAL_NAMES
    for readout in TEMPORAL_READOUT_NAMES
)


def _top_mean(values: np.ndarray, count: int = 10) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("top mean requires a finite nonempty vector")
    keep = min(int(count), len(values))
    return float(np.partition(values, len(values) - keep)[-keep:].mean())


def token_signals(logprobs: np.ndarray, entropy: np.ndarray) -> dict[str, np.ndarray]:
    """Build the frozen finite token-signal roster without benchmark labels.

    ``entropy`` is the generation cache's native top-15-renormalized Shannon
    entropy, not full-vocabulary entropy. ``q15_H1`` independently reconstructs
    the same definition from the saved top-50 matrix as an identity control.
    """
    logprobs = np.asarray(logprobs, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)
    if logprobs.ndim != 2 or logprobs.shape[1] < 50 or len(logprobs) != len(entropy):
        raise ValueError("gate signals require aligned T-by-at-least-50 logprobs and entropy")
    if not np.isfinite(logprobs).all() or not np.isfinite(entropy).all():
        raise ValueError("gate inputs must be finite")

    q15, _ = head_distribution(logprobs, k=15)
    lp15 = logprob_matrix({"logprobs": logprobs}, k=15)
    lp50 = logprob_matrix({"logprobs": logprobs}, k=50)
    finalist = np.asarray(multiscale_feature_bank(logprobs)["bank"][:, :4], dtype=np.float64)
    if finalist.shape != (len(logprobs), 4):
        raise ValueError("invalid frozen q15 finalist token bank")

    tail15 = residual_tail_mass(lp15)
    tail50 = residual_tail_mass(lp50)
    signals = {
        "entropy_native": entropy,
        "q15_H0lim": finalist[:, 0],
        "q15_VE0": finalist[:, 1],
        "q15_VE0.75": finalist[:, 2],
        "q15_VE1": finalist[:, 3],
        "q15_H1": renyi_entropy(q15, 1.0),
        "q15_Hinf": renyi_entropy(q15, np.inf),
        "raw_neglogp1": -lp15[:, 0],
        "tail15_mass": tail15,
        "tail50_mass": tail50,
        "q15_raw4_mean": finalist.mean(axis=1),
    }
    if tuple(signals) != SIGNAL_NAMES:
        raise AssertionError("gate signal roster drift")
    if any(value.shape != (len(entropy),) or not np.isfinite(value).all() for value in signals.values()):
        raise ValueError("nonfinite gate token signal")
    if np.any(signals["tail15_mass"] + 1e-12 < signals["tail50_mass"]):
        raise ValueError("missing mass must decrease as retained support grows")
    return signals


def apply_readouts(values: np.ndarray, spans: np.ndarray) -> dict[str, float]:
    """Reduce one token signal to the three frozen whole-answer detectors."""
    values = np.asarray(values, dtype=np.float64)
    spans = np.asarray(spans, dtype=np.int64)
    if spans.ndim != 2 or spans.shape[1] != 2 or len(spans) == 0:
        raise ValueError("gate readout requires nonempty S-by-2 spans")
    if np.min(spans) < 0 or np.max(spans) > len(values):
        raise ValueError("gate span is outside the token vector")
    per_step = step_readout(values, spans)
    output = {
        "token_mean": float(values.mean()),
        "token_top10": _top_mean(values, 10),
        "mean_step_top10": float(per_step.mean()),
    }
    if tuple(output) != READOUT_NAMES or not np.isfinite(list(output.values())).all():
        raise ValueError("invalid gate readout")
    return output


def _region_mean(values: np.ndarray, quarter: int) -> float:
    """Mean in one normalized-position quarter, finite even for T < 4."""
    if quarter not in range(4):
        raise ValueError("quarter must be in {0,1,2,3}")
    n = len(values)
    positions = (np.arange(n, dtype=np.float64) + 0.5) / n
    mask = (positions >= quarter / 4.0) & (positions < (quarter + 1) / 4.0)
    if np.any(mask):
        return float(values[mask].mean())
    center = (quarter + 0.5) / 4.0
    return float(values[int(np.argmin(np.abs(positions - center)))])


def apply_temporal_readouts(values: np.ndarray) -> dict[str, float]:
    """Reduce a token signal to the frozen cross-panel temporal bank."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("temporal readouts require a finite nonempty vector")
    window = min(8, len(values))
    rolling = np.convolve(values, np.ones(window) / window, mode="valid")
    if len(values) == 1:
        slope = 0.0
    else:
        position = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
        slope = float(np.polyfit(position, values, 1)[0])
    output = {
        "token_mean": float(values.mean()),
        "token_top10": _top_mean(values, 10),
        "token_q75": float(np.quantile(values, 0.75)),
        "token_q90": float(np.quantile(values, 0.90)),
        "token_q95": float(np.quantile(values, 0.95)),
        "rolling8_max": float(rolling.max()),
        "region_q1_mean": _region_mean(values, 0),
        "region_q2_mean": _region_mean(values, 1),
        "region_q3_mean": _region_mean(values, 2),
        "region_q4_mean": _region_mean(values, 3),
        "position_slope": slope,
    }
    if tuple(output) != TEMPORAL_READOUT_NAMES or not np.isfinite(list(output.values())).all():
        raise ValueError("invalid temporal readout")
    return output


def answer_detectors(logprobs: np.ndarray, entropy: np.ndarray, spans: np.ndarray) -> dict[str, float]:
    """Return all 33 label-free answer detector candidates."""
    output = {}
    for signal, values in token_signals(logprobs, entropy).items():
        for readout, detector in apply_readouts(values, spans).items():
            output[f"{signal}__{readout}"] = detector
    if tuple(output) != METHODS:
        raise AssertionError("gate method roster drift")
    return output


def answer_temporal_detectors(logprobs: np.ndarray, entropy: np.ndarray) -> dict[str, float]:
    """Return all 121 token-only answer detector candidates."""
    output = {}
    for signal, values in token_signals(logprobs, entropy).items():
        for readout, detector in apply_temporal_readouts(values).items():
            output[f"{signal}__{readout}"] = detector
    if tuple(output) != TEMPORAL_METHODS:
        raise AssertionError("temporal gate method roster drift")
    return output


__all__ = [
    "BASELINE", "METHODS", "READOUT_NAMES", "SIGNAL_NAMES", "TEMPORAL_METHODS",
    "TEMPORAL_READOUT_NAMES", "answer_detectors", "answer_temporal_detectors",
    "apply_readouts", "apply_temporal_readouts", "token_signals",
]
