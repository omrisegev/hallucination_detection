"""Unified causal IU-PCR for tokenwise evidence, early warning, and localization.

The module deliberately separates three contracts:

* :func:`extract_base_streams` replays the existing nine primitive risk channels and
  broad-28 token views from a saved telemetry row.
* :class:`CausalFeatureBankState` is a genuinely streaming, fixed-state DSP bank.  It
  accepts one already risk-oriented base vector at a time and never reads a suffix,
  final trace length, correctness, or a first-error annotation.
* :class:`UnifiedCausalIU` freezes robust references, feature order/signs, ordinary
  two-component IU-PCR weights, an accumulator, and warning thresholds.  The final
  score is literally the last value returned by ``update``.

Labels may be used by the experiment harness to choose a roster, signs, and an
accumulator.  They are never accepted by the online update path.  The resulting method
is therefore supervised-developed and IU-PCR-fused, rather than fully label-free.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from math import exp, log, pi
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import expit, logsumexp

from .feature_utils import compute_hurst_exponent, compute_spectral_features, permutation_entropy
from .local_online_comprehensive import BROAD_RISK_SIGNS
from .multitask_trajectory import CHANNEL_NAMES, equal_positions, raw_risk_channels
from .token_feature_views import BROAD_TOKEN_VIEWS
from .upcr import upcr_fit


EPS = 1e-12
CLIP_Z = 8.0
BASE_NAMES = tuple(f"raw::{name}" for name in CHANNEL_NAMES) + tuple(
    f"broad::{name}" for name in BROAD_TOKEN_VIEWS
)
EWMA_SPANS = (4, 8, 16, 32, 64)
FAST_SLOW_PAIRS = ((4, 16), (8, 32), (16, 64))
WINDOWS = (8, 16, 32, 64)
BOCPD_HAZARDS = (1.0 / 50.0, 1.0 / 100.0)
BOCPD_MAX_RUN = 128
DEFAULT_POSITIONS_PER_TRACE = 32
IU_FIT = {
    "loss": "l2",
    "exclusion": False,
    "difficulty_gate": False,
    "simple_avg_fallback": False,
    "recompute_after_exclusion": False,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "n_components": 2,
    "auto_components": False,
}


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    centre = float(np.median(values))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return centre, scale


def _as_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if np.isfinite(output) else float("nan")


def feature_name(base: str, transform: str) -> str:
    return f"{base}::{transform}"


def parse_feature_name(name: str) -> tuple[str, str]:
    parts = str(name).split("::")
    if len(parts) < 3:
        raise ValueError(f"invalid unified causal feature name: {name}")
    return "::".join(parts[:-1]), parts[-1]


def _forbid_future_names(names: Iterable[str]) -> None:
    forbidden = ("trace_length", "final_length", "future", "lookahead", "suffix")
    offenders = [name for name in names if any(token in str(name).lower() for token in forbidden)]
    if offenders:
        raise ValueError(f"non-causal feature names are forbidden: {offenders[:5]}")


def extract_base_streams(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the frozen 9+28 causal base streams in risk orientation.

    The broad views are the existing historical reconstruction.  Some broad views are
    intentionally duplicates of primitive streams; redundancy pruning happens only
    inside a development fold, as required by the protocol.
    """

    raw = raw_risk_channels(row)
    broad = causal_broad_risk_matrix(row)
    n = len(raw[CHANNEL_NAMES[0]])
    if broad.shape != (n, len(BROAD_TOKEN_VIEWS)):
        raise RuntimeError("broad token views do not align with primitive telemetry")
    output = {f"raw::{name}": np.asarray(raw[name], dtype=float) for name in CHANNEL_NAMES}
    output.update({
        f"broad::{name}": np.asarray(broad[:, index], dtype=float)
        for index, name in enumerate(BROAD_TOKEN_VIEWS)
    })
    if tuple(output) != BASE_NAMES:
        raise RuntimeError("base feature order changed")
    return output


def _aligned_optional(values: Any, n: int) -> np.ndarray:
    if values is None:
        return np.full(n, np.nan, dtype=float)
    result = np.asarray(values, dtype=float).reshape(-1)
    if not len(result):
        return np.full(n, np.nan, dtype=float)
    if len(result) >= n:
        return result[:n].copy()
    return np.concatenate([result, np.full(n - len(result), result[-1])])


def _causal_cusum(values: np.ndarray) -> np.ndarray:
    """Online-centred absolute CUSUM; unlike the historical view, no final mean."""

    output = np.zeros(len(values), dtype=float)
    mean = 0.0
    total = 0.0
    for index, value in enumerate(np.asarray(values, dtype=float)):
        if not np.isfinite(value):
            output[index] = np.nan
            continue
        mean += (float(value) - mean) / (index + 1)
        total += float(value) - mean
        output[index] = abs(total)
    return output


def _causal_window_statistics(entropy: np.ndarray) -> dict[str, np.ndarray]:
    """Strict-prefix versions of the historical entropy-only broad views."""

    n = len(entropy)
    names = (
        "entropy_rolling_spectral_entropy",
        "entropy_rolling_low_band_power",
        "entropy_rolling_high_band_power",
        "entropy_rolling_hl_ratio",
        "entropy_rolling_dominant_freq",
        "entropy_rolling_spectral_centroid",
        "entropy_stft_high_series",
        "entropy_stft_frame_entropy",
        "entropy_rolling_tail_ratio",
        "entropy_sw_var_series",
        "entropy_pe_series",
        "entropy_rolling_rs_hurst",
    )
    output = {name: np.zeros(n, dtype=float) for name in names}
    for index in range(n):
        window = np.asarray(entropy[max(0, index - 15):index + 1], dtype=float)
        output["entropy_sw_var_series"][index] = float(np.var(window))
        output["entropy_rolling_tail_ratio"][index] = float(
            np.mean(window[-min(5, len(window)):]) / (np.mean(window) + EPS)
        )
        try:
            pe_value = float(
                permutation_entropy(window, order=3, delay=1)
            )
            output["entropy_pe_series"][index] = pe_value if np.isfinite(pe_value) else 0.0
        except (ValueError, FloatingPointError, ZeroDivisionError):
            output["entropy_pe_series"][index] = 0.0
        try:
            hurst = float(
                compute_hurst_exponent(window)
            )
            output["entropy_rolling_rs_hurst"][index] = hurst if np.isfinite(hurst) else 0.5
        except (ValueError, FloatingPointError, ZeroDivisionError):
            output["entropy_rolling_rs_hurst"][index] = 0.5
        try:
            spectral = compute_spectral_features(window) or {}
        except (ValueError, FloatingPointError, ZeroDivisionError):
            spectral = {}
        for suffix in (
            "spectral_entropy", "low_band_power", "high_band_power", "hl_ratio",
            "dominant_freq", "spectral_centroid",
        ):
            value = _as_float(spectral.get(suffix, 0.0))
            output[f"entropy_rolling_{suffix}"][index] = 0.0 if not np.isfinite(value) else value

        centred = window - np.mean(window)
        power = np.abs(np.fft.rfft(centred)) ** 2
        frequencies = np.fft.rfftfreq(len(window)) if len(window) > 1 else np.zeros(1)
        total = float(np.sum(power))
        high = float(np.sum(power[frequencies >= 0.40]))
        probability = power / (total + EPS)
        output["entropy_stft_high_series"][index] = high
        output["entropy_stft_frame_entropy"][index] = float(
            -np.sum(probability * np.log(probability + EPS))
        )
    return output


def _rolling_stat(values: np.ndarray, operator: str) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        window = np.asarray(values[max(0, index - 15):index + 1], dtype=float)
        window = window[np.isfinite(window)]
        if not len(window):
            continue
        output[index] = float(np.var(window) if operator == "var" else np.min(window))
    return output


def causal_broad_risk_matrix(row: Mapping[str, Any]) -> np.ndarray:
    """Strictly causal versions of all existing broad-28 token views.

    Historical ``token_feature_views`` attributes the first completed window back to
    earlier tokens and centres CUSUM with the completed-trace mean.  Both conventions are
    valid for offline localization but violate suffix invariance.  This method retains the
    same 28 semantic channels while defining short prefixes from the observations actually
    available at that time.
    """

    entropy = np.asarray(row.get("token_entropies", ()), dtype=float).reshape(-1)
    n = len(entropy)
    if not n:
        raise ValueError("token_entropies must contain at least one token")
    spilled = _aligned_optional(row.get("token_spilled_energies"), n)
    energy = _aligned_optional(row.get("token_logsumexp"), n)
    entropy_views = _causal_window_statistics(entropy)
    out: dict[str, np.ndarray] = {
        "entropy_series": entropy.copy(),
        **entropy_views,
        "entropy_cusum_abs_series": _causal_cusum(entropy),
        "spilled_series": spilled.copy(),
        "spilled_sw_var_series": _rolling_stat(spilled, "var"),
        "spilled_cusum_abs_series": _causal_cusum(spilled),
        "spilled_rolling_min": _rolling_stat(spilled, "min"),
        "energy_series": energy.copy(),
        "energy_rolling_min": _rolling_stat(energy, "min"),
        "energy_sw_var_series": _rolling_stat(energy, "var"),
        "energy_cusum_abs_series": _causal_cusum(energy),
    }
    raw = raw_risk_channels(row)
    # Convert the risk-oriented primitive forms back to the historical broad-view
    # convention.  BROAD_RISK_SIGNS below then applies the single canonical orientation.
    out.update({
        "top1_logprob_series": -raw["neg_top1"],
        "logprob_margin_series": -raw["neg_margin"],
        "topk_entropy_series": raw["topk_entropy"],
        "topk_varentropy_series": raw["topk_varentropy"],
        "topk_renyi2_series": raw["topk_renyi2"],
        "topk_tail_mass_series": raw["topk_tail_mass"],
    })
    matrix = np.column_stack([out[name] for name in BROAD_TOKEN_VIEWS])
    matrix = np.asarray(matrix, dtype=float) * BROAD_RISK_SIGNS[None, :]
    return matrix


def base_matrix(row: Mapping[str, Any], names: Sequence[str] = BASE_NAMES) -> np.ndarray:
    names = tuple(names)
    _forbid_future_names(names)
    # The compact finalists use only the nine primitive risk channels.  Avoid
    # reconstructing all 28 broad spectral views in that deployed/raw-only
    # path; the values are exactly the same columns extract_base_streams would
    # have returned.
    if names and all(name.startswith("raw::") for name in names):
        raw = raw_risk_channels(row)
        return np.column_stack([raw[name.removeprefix("raw::")] for name in names])
    streams = extract_base_streams(row)
    return np.column_stack([streams[name] for name in names])


@dataclass(frozen=True)
class BaseReference:
    names: tuple[str, ...]
    centres: np.ndarray
    scales: np.ndarray
    availability: np.ndarray
    positions_per_trace: int
    n_traces: int

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape[-1] != len(self.names):
            raise ValueError("base vector does not match frozen reference")
        standardized = (values - self.centres) / self.scales
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)
        return np.clip(standardized, -CLIP_Z, CLIP_Z)

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "centres": self.centres.tolist(),
            "scales": self.scales.tolist(),
            "availability": self.availability.tolist(),
            "positions_per_trace": int(self.positions_per_trace),
            "n_traces": int(self.n_traces),
        }


def fit_base_reference(
    rows: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str] = BASE_NAMES,
    positions_per_trace: int = DEFAULT_POSITIONS_PER_TRACE,
    raw_base_matrices: Sequence[np.ndarray] | None = None,
) -> BaseReference:
    """Fit robust base references with exactly equal contribution per trace."""

    names = tuple(names)
    _forbid_future_names(names)
    if not rows:
        raise ValueError("at least one trace is required")
    if raw_base_matrices is not None and len(raw_base_matrices) != len(rows):
        raise ValueError("raw base matrices do not match reference traces")
    columns: list[list[np.ndarray]] = [[] for _ in names]
    availability = np.zeros(len(names), dtype=float)
    canonical_index = {name: index for index, name in enumerate(BASE_NAMES)}
    for row_index, row in enumerate(rows):
        if raw_base_matrices is None:
            matrix = base_matrix(row, names)
        else:
            full = np.asarray(raw_base_matrices[row_index], dtype=float)
            if full.ndim != 2:
                raise ValueError("raw base cache must be a matrix")
            if full.shape[1] == len(names):
                matrix = full
            elif full.shape[1] == len(BASE_NAMES):
                matrix = full[:, [canonical_index[name] for name in names]]
            else:
                raise ValueError("raw base cache violates the frozen base schema")
        positions = equal_positions(len(matrix), positions_per_trace)
        sample = matrix[positions]
        for index in range(sample.shape[1]):
            values = sample[:, index]
            availability[index] += float(np.mean(np.isfinite(values)))
            columns[index].append(values)
    centres, scales = [], []
    for parts in columns:
        centre, scale = _robust_location_scale(np.concatenate(parts))
        centres.append(centre)
        scales.append(scale)
    return BaseReference(
        names=names,
        centres=np.asarray(centres, dtype=float),
        scales=np.asarray(scales, dtype=float),
        availability=availability / len(rows),
        positions_per_trace=int(positions_per_trace),
        n_traces=len(rows),
    )


@dataclass
class _BOCPDState:
    hazard: float
    max_run: int = BOCPD_MAX_RUN
    observation_variance: float = 1.0
    prior_mean: float = 0.0
    prior_strength: float = 1.0
    probabilities: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=float))
    means: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    strengths: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=float))

    @staticmethod
    def _log_normal(value: float, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
        return -0.5 * (np.log(2.0 * pi * variance) + (value - mean) ** 2 / variance)

    def update(self, value: float) -> float:
        value = float(value)
        pred_var = self.observation_variance * (1.0 + 1.0 / self.strengths)
        log_pred = self._log_normal(value, self.means, pred_var)
        log_previous = np.log(np.maximum(self.probabilities, EPS))
        prior_pred_var = self.observation_variance * (1.0 + 1.0 / self.prior_strength)
        log_prior_pred = float(self._log_normal(
            value, np.asarray([self.prior_mean]), np.asarray([prior_pred_var])
        )[0])

        change_log_mass = logsumexp(log_previous + log(self.hazard)) + log_prior_pred
        growth = log_previous + log(1.0 - self.hazard) + log_pred
        joint = np.concatenate([[change_log_mass], growth[: self.max_run]])
        joint -= logsumexp(joint)
        probabilities = np.exp(joint)

        new_means = np.empty(len(probabilities), dtype=float)
        new_strengths = np.empty(len(probabilities), dtype=float)
        new_strengths[0] = self.prior_strength + 1.0
        new_means[0] = (
            self.prior_strength * self.prior_mean + value
        ) / new_strengths[0]
        previous_count = len(probabilities) - 1
        old_strengths = self.strengths[:previous_count]
        new_strengths[1:] = old_strengths + 1.0
        new_means[1:] = (
            old_strengths * self.means[:previous_count] + value
        ) / new_strengths[1:]
        self.probabilities = probabilities
        self.means = new_means
        self.strengths = new_strengths
        return float(probabilities[0])


@dataclass
class _BatchedBOCPDState:
    """The same conjugate BOCPD recursion for every base channel at once."""

    n_channels: int
    hazard: float
    max_run: int = BOCPD_MAX_RUN
    probabilities: np.ndarray = field(init=False)
    means: np.ndarray = field(init=False)
    strengths: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.probabilities = np.ones((self.n_channels, 1), dtype=float)
        self.means = np.zeros((self.n_channels, 1), dtype=float)
        self.strengths = np.ones(1, dtype=float)

    @staticmethod
    def _logsumexp_axis1(values: np.ndarray) -> np.ndarray:
        maximum = np.max(values, axis=1, keepdims=True)
        return maximum[:, 0] + np.log(np.sum(np.exp(values - maximum), axis=1) + EPS)

    def update(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(self.n_channels)
        pred_var = 1.0 + 1.0 / self.strengths
        log_pred = -0.5 * (
            np.log(2.0 * pi * pred_var)[None, :]
            + (values[:, None] - self.means) ** 2 / pred_var[None, :]
        )
        log_previous = np.log(np.maximum(self.probabilities, EPS))
        prior_log_pred = -0.5 * (
            log(4.0 * pi) + (values ** 2) / 2.0
        )
        change = self._logsumexp_axis1(log_previous + log(self.hazard)) + prior_log_pred
        growth = log_previous + log(1.0 - self.hazard) + log_pred
        joint = np.column_stack([change, growth[:, : self.max_run]])
        normalizer = self._logsumexp_axis1(joint)
        probabilities = np.exp(joint - normalizer[:, None])

        previous_count = probabilities.shape[1] - 1
        old_strengths = self.strengths[:previous_count]
        new_strengths = np.r_[2.0, old_strengths + 1.0]
        new_means = np.empty_like(probabilities)
        new_means[:, 0] = values / 2.0
        new_means[:, 1:] = (
            old_strengths[None, :] * self.means[:, :previous_count] + values[:, None]
        ) / (old_strengths[None, :] + 1.0)
        self.probabilities = probabilities
        self.means = new_means
        self.strengths = new_strengths
        return probabilities[:, 0]


@dataclass
class _OneChannelDSP:
    max_window: int = max(WINDOWS)
    count: int = 0
    buffer: deque[float] = field(default_factory=lambda: deque(maxlen=max(WINDOWS)))
    ewma: dict[int, float] = field(default_factory=lambda: {span: 0.0 for span in EWMA_SPANS})
    positive_sum: float = 0.0
    positive_count: int = 0
    cusum: float = 0.0
    running_mean: float = 0.0
    page_hinkley_sum: float = 0.0
    page_hinkley_min: float = 0.0
    bocpd: dict[int, _BOCPDState] = field(default_factory=lambda: {
        50: _BOCPDState(1.0 / 50.0),
        100: _BOCPDState(1.0 / 100.0),
    })

    def update(self, value: float) -> dict[str, float]:
        value = float(value)
        self.count += 1
        self.buffer.append(value)
        output = {"level": value}
        previous_ewma = dict(self.ewma)
        for span in EWMA_SPANS:
            alpha = 2.0 / (span + 1.0)
            self.ewma[span] = (1.0 - alpha) * self.ewma[span] + alpha * value
            output[f"ewma{span}"] = self.ewma[span]
        for fast, slow in FAST_SLOW_PAIRS:
            output[f"fastminus{fast}_{slow}"] = self.ewma[fast] - self.ewma[slow]

        data = np.asarray(self.buffer, dtype=float)
        for window in WINDOWS:
            current = data[-min(window, len(data)):]
            median = float(np.median(current))
            output[f"mean{window}"] = float(np.mean(current))
            output[f"var{window}"] = float(np.var(current))
            output[f"mad{window}"] = float(np.median(np.abs(current - median)))

        output["innovation64"] = value - previous_ewma[64]
        self.positive_sum += max(value, 0.0)
        self.positive_count += int(value > 0.0)
        output["positive_area"] = self.positive_sum / self.count
        output["persistence"] = self.positive_count / self.count

        self.cusum = max(0.0, self.cusum + value - 0.25)
        output["cusum_pos_k025"] = self.cusum
        previous_mean = self.running_mean
        self.running_mean += (value - self.running_mean) / self.count
        self.page_hinkley_sum += value - previous_mean - 0.05
        self.page_hinkley_min = min(self.page_hinkley_min, self.page_hinkley_sum)
        output["page_hinkley_pos_d005"] = max(
            0.0, self.page_hinkley_sum - self.page_hinkley_min
        )
        output["bocpd_h50"] = self.bocpd[50].update(value)
        output["bocpd_h100"] = self.bocpd[100].update(value)
        return output


TRANSFORM_NAMES = (
    ("level",)
    + tuple(f"ewma{span}" for span in EWMA_SPANS)
    + tuple(f"fastminus{fast}_{slow}" for fast, slow in FAST_SLOW_PAIRS)
    + tuple(
        f"{operator}{window}"
        for window in WINDOWS
        for operator in ("mean", "var", "mad")
    )
    + (
        "innovation64",
        "positive_area",
        "persistence",
        "cusum_pos_k025",
        "page_hinkley_pos_d005",
        "bocpd_h50",
        "bocpd_h100",
    )
)


@dataclass
class CausalFeatureBankState:
    """Fixed-memory causal DSP replay over an ordered set of base streams."""

    base_names: tuple[str, ...] = BASE_NAMES
    feature_names: tuple[str, ...] = field(init=False)
    time: int = 0
    buffer: deque[np.ndarray] = field(init=False)
    ewma: dict[int, np.ndarray] = field(init=False)
    positive_sum: np.ndarray = field(init=False)
    positive_count: np.ndarray = field(init=False)
    cusum: np.ndarray = field(init=False)
    running_mean: np.ndarray = field(init=False)
    page_hinkley_sum: np.ndarray = field(init=False)
    page_hinkley_min: np.ndarray = field(init=False)
    bocpd: dict[int, _BatchedBOCPDState] = field(init=False)

    def __post_init__(self) -> None:
        self.base_names = tuple(self.base_names)
        _forbid_future_names(self.base_names)
        dimension = len(self.base_names)
        self.buffer = deque(maxlen=max(WINDOWS))
        self.ewma = {span: np.zeros(dimension, dtype=float) for span in EWMA_SPANS}
        self.positive_sum = np.zeros(dimension, dtype=float)
        self.positive_count = np.zeros(dimension, dtype=float)
        self.cusum = np.zeros(dimension, dtype=float)
        self.running_mean = np.zeros(dimension, dtype=float)
        self.page_hinkley_sum = np.zeros(dimension, dtype=float)
        self.page_hinkley_min = np.zeros(dimension, dtype=float)
        self.bocpd = {
            50: _BatchedBOCPDState(dimension, 1.0 / 50.0),
            100: _BatchedBOCPDState(dimension, 1.0 / 100.0),
        }
        self.feature_names = tuple(
            feature_name(base, transform)
            for base in self.base_names
            for transform in TRANSFORM_NAMES
        )

    def update(self, base_values: Mapping[str, Any] | Sequence[float] | np.ndarray) -> np.ndarray:
        if isinstance(base_values, Mapping):
            values = np.asarray([_as_float(base_values.get(name)) for name in self.base_names])
        else:
            values = np.asarray(base_values, dtype=float).reshape(-1)
            if len(values) != len(self.base_names):
                raise ValueError("base vector does not match feature-bank order")
        values = np.where(np.isfinite(values), values, 0.0)
        previous_ewma = {span: value.copy() for span, value in self.ewma.items()}
        self.buffer.append(values.copy())
        transforms: dict[str, np.ndarray] = {"level": values}
        for span in EWMA_SPANS:
            alpha = 2.0 / (span + 1.0)
            self.ewma[span] = (1.0 - alpha) * self.ewma[span] + alpha * values
            transforms[f"ewma{span}"] = self.ewma[span].copy()
        for fast, slow in FAST_SLOW_PAIRS:
            transforms[f"fastminus{fast}_{slow}"] = self.ewma[fast] - self.ewma[slow]
        history = np.vstack(self.buffer)
        for window in WINDOWS:
            current = history[-min(window, len(history)):]
            median = np.median(current, axis=0)
            transforms[f"mean{window}"] = np.mean(current, axis=0)
            transforms[f"var{window}"] = np.var(current, axis=0)
            transforms[f"mad{window}"] = np.median(np.abs(current - median[None, :]), axis=0)
        transforms["innovation64"] = values - previous_ewma[64]
        self.positive_sum += np.maximum(values, 0.0)
        self.positive_count += values > 0.0
        transforms["positive_area"] = self.positive_sum / (self.time + 1)
        transforms["persistence"] = self.positive_count / (self.time + 1)
        self.cusum = np.maximum(0.0, self.cusum + values - 0.25)
        transforms["cusum_pos_k025"] = self.cusum.copy()
        previous_mean = self.running_mean.copy()
        self.running_mean += (values - self.running_mean) / (self.time + 1)
        self.page_hinkley_sum += values - previous_mean - 0.05
        self.page_hinkley_min = np.minimum(self.page_hinkley_min, self.page_hinkley_sum)
        transforms["page_hinkley_pos_d005"] = np.maximum(
            0.0, self.page_hinkley_sum - self.page_hinkley_min
        )
        transforms["bocpd_h50"] = self.bocpd[50].update(values)
        transforms["bocpd_h100"] = self.bocpd[100].update(values)
        output = np.column_stack([transforms[name] for name in TRANSFORM_NAMES]).reshape(-1)
        self.time += 1
        return np.asarray(output, dtype=float)

    def update_many(self, base_values: np.ndarray, *, chunk_size: int | None = None) -> np.ndarray:
        matrix = np.asarray(base_values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.base_names):
            raise ValueError("base matrix must be T by number-of-base-streams")
        # ``chunk_size`` changes only iteration boundaries.  It is present so the
        # equivalence test exercises the same public API used by batched callers.
        size = max(1, len(matrix)) if chunk_size is None else max(1, int(chunk_size))
        rows = []
        for start in range(0, len(matrix), size):
            rows.extend(self.update(row) for row in matrix[start:start + size])
        return np.vstack(rows) if rows else np.empty((0, len(self.feature_names)))


def causal_feature_matrix(
    row: Mapping[str, Any],
    reference: BaseReference,
    *,
    feature_order: Sequence[str] | None = None,
    raw_base: np.ndarray | None = None,
) -> np.ndarray:
    """Replay the complete DSP bank; optional order changes output columns only."""

    if raw_base is None:
        raw = base_matrix(row, reference.names)
    else:
        full = np.asarray(raw_base, dtype=float)
        if full.ndim != 2:
            raise ValueError("raw base cache must be a matrix")
        if full.shape[1] == len(reference.names):
            raw = full
        elif full.shape[1] == len(BASE_NAMES):
            canonical_index = {name: index for index, name in enumerate(BASE_NAMES)}
            raw = full[:, [canonical_index[name] for name in reference.names]]
        else:
            raise ValueError("raw base cache violates the frozen base schema")
    standardized = reference.transform(raw)
    state = CausalFeatureBankState(reference.names)
    full = state.update_many(standardized)
    if feature_order is None:
        return full
    order = tuple(feature_order)
    if sorted(order) != sorted(state.feature_names) or len(order) != len(state.feature_names):
        raise ValueError("feature_order must be a full permutation")
    lookup = {name: index for index, name in enumerate(state.feature_names)}
    return full[:, [lookup[name] for name in order]]


def all_feature_names(base_names: Sequence[str] = BASE_NAMES) -> tuple[str, ...]:
    names = tuple(
        feature_name(base, transform)
        for base in base_names
        for transform in TRANSFORM_NAMES
    )
    _forbid_future_names(names)
    return names


@dataclass(frozen=True)
class AccumulatorSpec:
    kind: str
    span: int | None = None
    drift: float = 0.0

    @property
    def name(self) -> str:
        if self.kind == "identity":
            return "identity"
        if self.kind == "leaky":
            return f"leaky_s{int(self.span)}_d{self.drift:g}"
        if self.kind == "hazard":
            return "cumulative_hazard"
        raise KeyError(self.kind)

    def validate(self) -> None:
        if self.kind not in {"identity", "leaky", "hazard"}:
            raise ValueError(f"unknown accumulator {self.kind}")
        if self.kind == "leaky" and (self.span is None or int(self.span) <= 0):
            raise ValueError("leaky accumulator requires a positive span")


ACCUMULATOR_ROSTER = (
    (AccumulatorSpec("identity"),)
    + tuple(AccumulatorSpec("leaky", span, drift) for span in (8, 16, 32, 64) for drift in (0.0, 0.25, 0.5))
    + (AccumulatorSpec("hazard"),)
)


@dataclass
class AccumulatorState:
    spec: AccumulatorSpec
    risk: float = 0.0

    def update(self, evidence: float) -> tuple[float, float]:
        evidence = float(evidence)
        previous = self.risk
        if self.spec.kind == "identity":
            self.risk = evidence
            contribution = max(0.0, evidence)
        elif self.spec.kind == "leaky":
            decay = exp(-1.0 / float(self.spec.span))
            injected = evidence - float(self.spec.drift)
            self.risk = max(0.0, decay * previous + injected)
            contribution = max(0.0, injected)
        elif self.spec.kind == "hazard":
            hazard = max(0.0, 2.0 * float(expit(evidence)) - 1.0)
            contribution = (1.0 - previous) * hazard
            self.risk = previous + contribution
        else:
            raise KeyError(self.spec.kind)
        return float(self.risk), float(contribution)


@dataclass(frozen=True)
class UnifiedUpdate:
    token_index: int
    evidence: float
    risk: float
    contribution: float
    warning_5pct: bool
    warning_10pct: bool


@dataclass(frozen=True)
class UnifiedFinal:
    global_score: float
    localization_token: int
    first_alarm_token: int | None
    first_alarm_token_5pct: int | None
    first_alarm_token_10pct: int | None
    trajectory: tuple[UnifiedUpdate, ...]


@dataclass
class UnifiedCausalIUState:
    model: "UnifiedCausalIU"
    bank: CausalFeatureBankState = field(init=False)
    accumulator: AccumulatorState = field(init=False)
    trajectory: list[UnifiedUpdate] = field(default_factory=list)
    best_contribution: float = float("-inf")
    localization_token: int = 0
    first_alarm_5: int | None = None
    first_alarm_10: int | None = None

    def __post_init__(self) -> None:
        self.bank = CausalFeatureBankState(self.model.reference.names)
        self.accumulator = AccumulatorState(self.model.accumulator)

    def _ordered_base_vector(self, token_telemetry: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
        if isinstance(token_telemetry, Mapping):
            if "base_values" in token_telemetry:
                values = token_telemetry["base_values"]
                if isinstance(values, Mapping):
                    return np.asarray([_as_float(values.get(name)) for name in self.model.reference.names])
                return np.asarray(values, dtype=float).reshape(-1)
            return np.asarray([
                _as_float(token_telemetry.get(name)) for name in self.model.reference.names
            ])
        return np.asarray(token_telemetry, dtype=float).reshape(-1)

    def update(self, token_telemetry: Mapping[str, Any] | Sequence[float]) -> UnifiedUpdate:
        raw = self._ordered_base_vector(token_telemetry)
        if len(raw) != len(self.model.reference.names):
            raise ValueError("token telemetry does not match frozen base roster")
        standardized_base = self.model.reference.transform(raw)
        complete = self.bank.update(standardized_base)
        evidence = self.model._evidence_from_selected_features(
            complete[self.model.feature_indices]
        )
        return self._update_evidence(evidence)

    def _update_evidence(self, evidence: float) -> UnifiedUpdate:
        """Advance only the frozen accumulator/output state for one evidence value.

        The offline full-feature-matrix scorer enters here after applying the exact
        same feature preprocessing as the live bank.  Keeping this logic shared makes
        thresholds, localization tie-breaking, alarms, and terminal-score identity
        independent of whether causal DSP was replayed online or precomputed once.
        """

        evidence = float(evidence)
        risk, contribution = self.accumulator.update(evidence)
        token = len(self.trajectory)
        warning_5 = bool(risk > self.model.warning_threshold_5pct)
        warning_10 = bool(risk > self.model.warning_threshold_10pct)
        if warning_5 and self.first_alarm_5 is None:
            self.first_alarm_5 = token
        if warning_10 and self.first_alarm_10 is None:
            self.first_alarm_10 = token
        if contribution > self.best_contribution:
            self.best_contribution = contribution
            self.localization_token = token
        update = UnifiedUpdate(
            token_index=token,
            evidence=evidence,
            risk=risk,
            contribution=contribution,
            warning_5pct=warning_5,
            warning_10pct=warning_10,
        )
        self.trajectory.append(update)
        return update

    def update_many(
        self,
        token_telemetry: Sequence[Mapping[str, Any]] | np.ndarray,
        *,
        chunk_size: int | None = None,
    ) -> list[UnifiedUpdate]:
        size = len(token_telemetry) if chunk_size is None else max(1, int(chunk_size))
        output = []
        for start in range(0, len(token_telemetry), size):
            output.extend(self.update(value) for value in token_telemetry[start:start + size])
        return output

    def finalize(self) -> UnifiedFinal:
        if not self.trajectory:
            raise RuntimeError("cannot finalize an empty trace")
        # This assignment—not a recomputation—is the bit-identity contract.
        terminal = self.trajectory[-1].risk
        return UnifiedFinal(
            global_score=terminal,
            localization_token=int(self.localization_token),
            first_alarm_token=self.first_alarm_10,
            first_alarm_token_5pct=self.first_alarm_5,
            first_alarm_token_10pct=self.first_alarm_10,
            trajectory=tuple(self.trajectory),
        )


@dataclass(frozen=True)
class UnifiedCausalIU:
    reference: BaseReference
    feature_names: tuple[str, ...]
    feature_indices: np.ndarray
    feature_medians: np.ndarray
    feature_centres: np.ndarray
    feature_scales: np.ndarray
    feature_signs: np.ndarray
    weights: np.ndarray
    evidence_centre: float
    evidence_scale: float
    accumulator: AccumulatorSpec
    warning_threshold_5pct: float = float("inf")
    warning_threshold_10pct: float = float("inf")
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        feature_roster: Sequence[str] | None = None,
        feature_signs: Mapping[str, float] | Sequence[float] | None = None,
        accumulator: AccumulatorSpec = AccumulatorSpec("identity"),
        positions_per_trace: int = DEFAULT_POSITIONS_PER_TRACE,
        reference: BaseReference | None = None,
        feature_matrices: Sequence[np.ndarray] | None = None,
    ) -> "UnifiedCausalIU":
        """Fit and freeze the ordinary two-component IU-PCR head.

        Outcome labels are not read here.  A supervised harness may pass a roster and
        signs selected on development data; their provenance is recorded by that harness.
        """

        if not rows:
            raise ValueError("at least one fit trace is required")
        accumulator.validate()
        reference = reference or fit_base_reference(
            rows, positions_per_trace=positions_per_trace
        )
        canonical = all_feature_names(reference.names)
        roster = tuple(feature_roster) if feature_roster is not None else canonical
        _forbid_future_names(roster)
        if len(set(roster)) != len(roster) or not set(roster) <= set(canonical):
            raise ValueError("feature roster contains duplicates or unknown features")
        if len(roster) < 3:
            raise ValueError("IU-PCR requires at least three features")
        canonical_index = {name: index for index, name in enumerate(canonical)}
        indices = np.asarray([canonical_index[name] for name in roster], dtype=int)

        if feature_matrices is not None and len(feature_matrices) != len(rows):
            raise ValueError("feature matrices do not match fit traces")
        samples = []
        for row_index, row in enumerate(rows):
            full = (
                causal_feature_matrix(row, reference)
                if feature_matrices is None
                else np.asarray(feature_matrices[row_index], dtype=float)
            )
            if full.ndim != 2 or full.shape[1] != len(canonical):
                raise ValueError("precomputed feature matrix violates frozen bank schema")
            positions = equal_positions(len(full), positions_per_trace)
            samples.append(full[positions][:, indices])
        matrix = np.vstack(samples)
        medians = np.nanmedian(matrix, axis=0)
        clean = np.where(np.isfinite(matrix), matrix, medians)
        centres, scales = [], []
        for column in clean.T:
            centre, scale = _robust_location_scale(column)
            centres.append(centre)
            scales.append(scale)
        centres = np.asarray(centres, dtype=float)
        scales = np.asarray(scales, dtype=float)
        keep = np.isfinite(medians) & np.isfinite(centres) & np.isfinite(scales) & (scales > 1e-8)
        if int(keep.sum()) < 3:
            raise ValueError("fewer than three non-degenerate causal coordinates remain")
        roster = tuple(name for name, use in zip(roster, keep) if use)
        indices = indices[keep]
        medians = medians[keep]
        centres = centres[keep]
        scales = scales[keep]
        standardized = (clean[:, keep] - centres) / scales

        if feature_signs is None:
            signs = np.ones(len(roster), dtype=float)
        elif isinstance(feature_signs, Mapping):
            signs = np.asarray([float(feature_signs.get(name, 1.0)) for name in roster])
        else:
            original = np.asarray(feature_signs, dtype=float).reshape(-1)
            if len(original) != len(keep):
                raise ValueError("feature signs do not match pre-degeneracy roster")
            signs = original[keep]
        signs = np.where(signs < 0.0, -1.0, 1.0)
        oriented = standardized * signs
        fitted = upcr_fit(oriented.T, **IU_FIT)
        weights = np.asarray(fitted.w, dtype=float)
        evidence = oriented @ weights
        evidence_centre, evidence_scale = _robust_location_scale(evidence)
        return cls(
            reference=reference,
            feature_names=roster,
            feature_indices=indices,
            feature_medians=medians,
            feature_centres=centres,
            feature_scales=scales,
            feature_signs=signs,
            weights=weights,
            evidence_centre=evidence_centre,
            evidence_scale=evidence_scale,
            accumulator=accumulator,
            diagnostics={
                "labels_seen_during_fit": False,
                "upstream_roster_and_signs_may_be_supervised": True,
                "claim_label": "supervised-developed, IU-PCR-fused, causal streaming",
                "n_fit_traces": len(rows),
                "positions_per_trace": int(positions_per_trace),
                "input_features": len(keep),
                "retained_features": int(keep.sum()),
                "components": int(fitted.n_components_used),
                "loss": IU_FIT["loss"],
                "graph_or_laplacian": False,
                "scale_ratio": IU_FIT["scale_ratio"],
                "g2_hat": float(fitted.g2_hat),
                "projection_residual": float(fitted.proj_residual),
            },
        )

    @classmethod
    def fit_dufs_path(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        lambdas: Sequence[float],
        feature_roster: Sequence[str] | None = None,
        feature_signs: Mapping[str, float] | Sequence[float] | None = None,
        accumulator: AccumulatorSpec = AccumulatorSpec("identity"),
        positions_per_trace: int = DEFAULT_POSITIONS_PER_TRACE,
        reference: BaseReference | None = None,
        feature_matrices: Sequence[np.ndarray] | None = None,
        ordinary_model: "UnifiedCausalIU" | None = None,
        graph_k: int = 7,
        dufs_seeds: Sequence[int] = (11, 23, 37),
        dufs_epochs: int = 80,
    ) -> dict[float, "UnifiedCausalIU"]:
        """Fit one DUFS graph and return a leakage-free Laplacian-IU lambda path.

        Every returned model uses the exact ordinary-IU roster, preprocessing,
        orientation and two-component subspace.  DUFS sees only the standardized
        fit-partition coordinates; labels and validation rows are not accepted.
        ``lambda=0`` is checked against ordinary IU bit-for-bit but is not returned,
        because the ordinary model is the canonical zero-complexity control.
        """

        requested = tuple(dict.fromkeys(float(value) for value in lambdas))
        if not requested:
            raise ValueError("DUFS lambda path must not be empty")
        if any(not np.isfinite(value) or value <= 0.0 for value in requested):
            raise ValueError("DUFS lambdas must be finite and strictly positive")
        if int(graph_k) < 1:
            raise ValueError("graph_k must be positive")
        if int(dufs_epochs) < 1:
            raise ValueError("dufs_epochs must be positive")

        ordinary = ordinary_model or cls.fit(
            rows,
            feature_roster=feature_roster,
            feature_signs=feature_signs,
            accumulator=accumulator,
            positions_per_trace=positions_per_trace,
            reference=reference,
            feature_matrices=feature_matrices,
        )
        if ordinary.accumulator != accumulator:
            ordinary = ordinary.with_accumulator(accumulator)
        if feature_matrices is not None and len(feature_matrices) != len(rows):
            raise ValueError("feature matrices do not match fit traces")

        canonical = all_feature_names(ordinary.reference.names)
        sampled = []
        for row_index, row in enumerate(rows):
            full = (
                causal_feature_matrix(row, ordinary.reference)
                if feature_matrices is None
                else np.asarray(feature_matrices[row_index], dtype=float)
            )
            if full.ndim != 2 or full.shape[1] != len(canonical):
                raise ValueError("precomputed feature matrix violates frozen bank schema")
            positions = equal_positions(len(full), positions_per_trace)
            sampled.append(full[positions][:, ordinary.feature_indices])
        matrix = np.vstack(sampled)
        clean = np.where(np.isfinite(matrix), matrix, ordinary.feature_medians)
        oriented = (
            (clean - ordinary.feature_centres) / ordinary.feature_scales
        ) * ordinary.feature_signs
        F = oriented.T

        # Imports are intentionally lazy.  The ordinary causal runner remains usable on
        # analysis hosts without the optional PyTorch dependency unless DUFS is requested.
        from .adapted_dufs import adapted_dufs_soft_gates
        from .laplacian_upcr import build_graph_from_features, laplacian_iu_path

        gates, gate_diagnostics = adapted_dufs_soft_gates(
            F,
            seeds=tuple(int(seed) for seed in dufs_seeds),
            epochs=int(dufs_epochs),
        )
        graph = build_graph_from_features(F, gates=gates, k=int(graph_k))
        path = laplacian_iu_path(
            F,
            (0.0, *requested),
            graph=graph,
            baseline_kwargs=IU_FIT,
        )
        if not np.array_equal(path[0.0].w, ordinary.weights):
            raise AssertionError("DUFS lambda=0 is not bit-identical to ordinary IU")

        output: dict[float, UnifiedCausalIU] = {}
        for lambda_ in requested:
            fitted = path[lambda_]
            weights = np.asarray(fitted.w, dtype=float)
            evidence_centre, evidence_scale = _robust_location_scale(oriented @ weights)
            output[lambda_] = replace(
                ordinary,
                weights=weights,
                evidence_centre=evidence_centre,
                evidence_scale=evidence_scale,
                diagnostics={
                    **dict(ordinary.diagnostics),
                    "claim_label": (
                        "supervised-developed, DUFS-Laplacian-IU-PCR-fused, "
                        "causal streaming"
                    ),
                    "fusion": "dufs_laplacian_iu_pcr",
                    "graph_or_laplacian": True,
                    "graph_lambda": float(lambda_),
                    "graph_k": int(graph_k),
                    "dufs_seeds": [int(seed) for seed in dufs_seeds],
                    "dufs_epochs": int(dufs_epochs),
                    "dufs_gate_diagnostics": gate_diagnostics,
                    "laplacian_diagnostics": fitted.diagnostics,
                    "lambda_zero_exact": True,
                    "same_roster_as_ordinary": True,
                },
            )
        return output

    def start(self) -> UnifiedCausalIUState:
        return UnifiedCausalIUState(self)

    def _evidence_from_selected_features(self, selected: np.ndarray) -> float:
        """Apply the frozen per-coordinate and evidence transforms to one token."""

        selected = np.asarray(selected, dtype=float).reshape(-1)
        if len(selected) != len(self.feature_names):
            raise ValueError("selected features do not match frozen feature roster")
        clean = np.where(np.isfinite(selected), selected, self.feature_medians)
        robust = (clean - self.feature_centres) / self.feature_scales
        evidence = float((robust * self.feature_signs) @ self.weights)
        return float((evidence - self.evidence_centre) / self.evidence_scale)

    def _matrix_feature_indices(
        self,
        feature_order: Sequence[str] | None,
    ) -> np.ndarray:
        """Resolve the selected roster inside a full causal-bank column order."""

        canonical = all_feature_names(self.reference.names)
        if feature_order is None:
            return self.feature_indices
        order = tuple(str(name) for name in feature_order)
        if len(order) != len(canonical) or len(set(order)) != len(order):
            raise ValueError("feature_order must be a full permutation")
        if set(order) != set(canonical):
            raise ValueError("feature_order must be a full permutation")
        lookup = {name: index for index, name in enumerate(order)}
        return np.asarray([lookup[name] for name in self.feature_names], dtype=int)

    def evidence_from_feature_matrix(
        self,
        values: np.ndarray,
        *,
        feature_order: Sequence[str] | None = None,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """Score a precomputed complete causal DSP matrix without replaying the bank.

        ``values`` must contain all causal-bank coordinates, either in canonical order
        or in the explicitly supplied full ``feature_order`` permutation.  Rows are
        deliberately reduced one at a time with the same NumPy expression as live
        ``update`` so chunk boundaries cannot change floating-point results.
        """

        matrix = np.asarray(values, dtype=float)
        canonical = all_feature_names(self.reference.names)
        if matrix.ndim != 2 or matrix.shape[1] != len(canonical):
            raise ValueError("precomputed feature matrix violates frozen bank schema")
        indices = self._matrix_feature_indices(feature_order)
        size = len(matrix) if chunk_size is None else max(1, int(chunk_size))
        evidence = np.empty(len(matrix), dtype=float)
        for start in range(0, len(matrix), size):
            stop = min(len(matrix), start + size)
            for row_index in range(start, stop):
                evidence[row_index] = self._evidence_from_selected_features(
                    matrix[row_index, indices]
                )
        return evidence

    def score_causal_matrix(
        self,
        values: np.ndarray,
        *,
        feature_order: Sequence[str] | None = None,
        chunk_size: int | None = None,
    ) -> UnifiedFinal:
        """Return the exact live trajectory from a precomputed full causal matrix."""

        evidence = self.evidence_from_feature_matrix(
            values,
            feature_order=feature_order,
            chunk_size=chunk_size,
        )
        state = self.start()
        size = max(1, len(evidence)) if chunk_size is None else max(1, int(chunk_size))
        for start in range(0, len(evidence), size):
            for value in evidence[start:start + size]:
                state._update_evidence(float(value))
        return state.finalize()

    def score_base_matrix(self, values: np.ndarray, *, chunk_size: int | None = None) -> UnifiedFinal:
        state = self.start()
        state.update_many(np.asarray(values, dtype=float), chunk_size=chunk_size)
        return state.finalize()

    def score_row(self, row: Mapping[str, Any], *, chunk_size: int | None = None) -> UnifiedFinal:
        return self.score_base_matrix(
            base_matrix(row, self.reference.names), chunk_size=chunk_size
        )

    def with_thresholds(self, threshold_5pct: float, threshold_10pct: float) -> "UnifiedCausalIU":
        return UnifiedCausalIU(
            reference=self.reference,
            feature_names=self.feature_names,
            feature_indices=self.feature_indices,
            feature_medians=self.feature_medians,
            feature_centres=self.feature_centres,
            feature_scales=self.feature_scales,
            feature_signs=self.feature_signs,
            weights=self.weights,
            evidence_centre=self.evidence_centre,
            evidence_scale=self.evidence_scale,
            accumulator=self.accumulator,
            warning_threshold_5pct=float(threshold_5pct),
            warning_threshold_10pct=float(threshold_10pct),
            diagnostics=dict(self.diagnostics),
        )

    def with_accumulator(self, accumulator: AccumulatorSpec) -> "UnifiedCausalIU":
        accumulator.validate()
        return UnifiedCausalIU(
            reference=self.reference,
            feature_names=self.feature_names,
            feature_indices=self.feature_indices,
            feature_medians=self.feature_medians,
            feature_centres=self.feature_centres,
            feature_scales=self.feature_scales,
            feature_signs=self.feature_signs,
            weights=self.weights,
            evidence_centre=self.evidence_centre,
            evidence_scale=self.evidence_scale,
            accumulator=accumulator,
            warning_threshold_5pct=float("inf"),
            warning_threshold_10pct=float("inf"),
            diagnostics=dict(self.diagnostics),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.diagnostics.get(
                "claim_label", "supervised-developed, IU-PCR-fused, causal streaming"
            ),
            "reference": self.reference.as_dict(),
            "feature_names": list(self.feature_names),
            "feature_indices": self.feature_indices.tolist(),
            "feature_medians": self.feature_medians.tolist(),
            "feature_centres": self.feature_centres.tolist(),
            "feature_scales": self.feature_scales.tolist(),
            "feature_signs": self.feature_signs.tolist(),
            "weights": self.weights.tolist(),
            "evidence_centre": float(self.evidence_centre),
            "evidence_scale": float(self.evidence_scale),
            "accumulator": {
                "kind": self.accumulator.kind,
                "span": self.accumulator.span,
                "drift": self.accumulator.drift,
                "name": self.accumulator.name,
            },
            "warning_threshold_5pct": float(self.warning_threshold_5pct),
            "warning_threshold_10pct": float(self.warning_threshold_10pct),
            "diagnostics": dict(self.diagnostics),
        }


def calibrate_warning_thresholds(
    model: UnifiedCausalIU,
    rows: Sequence[Mapping[str, Any]],
    *,
    is_clean: Sequence[bool] | None = None,
    max_horizon: int | None = None,
) -> UnifiedCausalIU:
    """Calibrate 5%/10% ever-warning FPR on per-trace horizon maxima only."""

    clean = np.ones(len(rows), dtype=bool) if is_clean is None else np.asarray(is_clean, dtype=bool)
    if len(clean) != len(rows):
        raise ValueError("clean mask does not match calibration rows")
    maxima = []
    for row, use in zip(rows, clean):
        if not use:
            continue
        risk = np.asarray([item.risk for item in model.score_row(row).trajectory], dtype=float)
        if max_horizon is not None:
            risk = risk[: max(1, int(max_horizon))]
        maxima.append(float(np.max(risk)))
    if not maxima:
        raise ValueError("warning calibration requires at least one clean trace")
    return model.with_thresholds(
        float(np.quantile(maxima, 0.95, method="higher")),
        float(np.quantile(maxima, 0.90, method="higher")),
    )


__all__ = [
    "ACCUMULATOR_ROSTER",
    "BASE_NAMES",
    "BOCPD_HAZARDS",
    "FAST_SLOW_PAIRS",
    "TRANSFORM_NAMES",
    "WINDOWS",
    "AccumulatorSpec",
    "AccumulatorState",
    "BaseReference",
    "CausalFeatureBankState",
    "UnifiedCausalIU",
    "UnifiedCausalIUState",
    "UnifiedFinal",
    "UnifiedUpdate",
    "all_feature_names",
    "base_matrix",
    "calibrate_warning_thresholds",
    "causal_feature_matrix",
    "extract_base_streams",
    "feature_name",
    "fit_base_reference",
    "parse_feature_name",
]
