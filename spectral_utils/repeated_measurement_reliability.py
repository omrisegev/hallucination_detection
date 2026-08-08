"""Label-free repeated-measurement diagnostics for answer-level U-PCR.

The repeated measurements in this module are moving-block bootstrap views of
one saved token-telemetry trace.  They are *not* repeated LLM generations.  A
single bootstrap index is applied to every token-resolved channel so their
cross-channel alignment is preserved.

The resulting within-item covariance describes sensitivity to this particular
resampling procedure.  It must pass the diagnostics below before it may be
interpreted as nuisance covariance or used by a fusion method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, rankdata

from .dufs_liu_feature_contract import FEATURE_TRANSFORMS
from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1


EPS = 1e-12


def circular_moving_block_indices(length, block_length, rng):
    """Draw one length-preserving circular moving-block bootstrap index."""
    length = int(length)
    block_length = int(max(1, min(block_length, length)))
    starts = rng.integers(0, length, size=int(np.ceil(length / block_length)))
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % length).ravel()[:length]


def bootstrap_trace_row(row, indices):
    """Apply synchronized token indices to all telemetry used by the features."""
    indices = np.asarray(indices, dtype=int)
    output = {}
    for name in ("token_entropies", "token_spilled_energies", "token_logsumexp"):
        values = row.get(name)
        if values is not None:
            array = np.asarray(values)
            if len(array) != len(row["token_entropies"]):
                raise ValueError(f"unaligned telemetry channel: {name}")
            output[name] = array[indices]
    top_k = row.get("top_k_logprobs")
    if top_k is not None:
        output["top_k_logprobs"] = {
            name: np.asarray(values)[indices]
            for name, values in top_k.items()
        }
    return output


def _mode_percentile(values, grid_size=512, min_prominence=0.05):
    values = np.asarray(values, dtype=float)
    if len(values) < 50 or np.std(values) < EPS:
        return 0.5
    try:
        kde = gaussian_kde(values)
        grid = np.linspace(values.min(), values.max(), int(grid_size))
        density = kde(grid)
    except Exception:
        return 0.5
    if not np.isfinite(density).all() or density.max() <= 0:
        return 0.5
    peaks, properties = find_peaks(
        density, prominence=float(min_prominence) * float(density.max())
    )
    peak = (
        int(peaks[np.argmax(properties["prominences"])])
        if len(peaks) else int(np.argmax(density))
    )
    return float(np.mean(values < grid[peak]))


def _safe_scale(values):
    scale = np.asarray(values, dtype=float).copy()
    scale[scale < EPS] = 1.0
    return scale


@dataclass
class FixedMixedV2Transformer:
    """Population-fitted mixed-v2 transform reusable on bootstrap samples."""

    names: tuple[str, ...]
    raw_median: np.ndarray
    oriented_mean: np.ndarray
    oriented_std: np.ndarray
    sorted_oriented: tuple[np.ndarray, ...]
    mode_centres: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    training_output: np.ndarray

    @classmethod
    def fit(cls, raw_matrix, names):
        raw = np.asarray(raw_matrix, dtype=float)
        names = tuple(str(name) for name in names)
        if raw.ndim != 2 or raw.shape[1] != len(names):
            raise ValueError("raw_matrix and names disagree")
        unknown = sorted(set(names) - set(CONFIDENCE_FEATURE_SIGNS_V1))
        if unknown:
            raise KeyError("unregistered feature(s): " + ", ".join(unknown))
        medians = np.nanmedian(raw, axis=0)
        filled = np.where(np.isfinite(raw), raw, medians[None, :])
        signs = np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[n] for n in names])
        oriented_raw = filled * signs[None, :]
        mean = oriented_raw.mean(axis=0)
        std = _safe_scale(oriented_raw.std(axis=0))
        oriented = (oriented_raw - mean[None, :]) / std[None, :]
        transformed = oriented.copy()
        sorted_values = []
        centres = np.full(len(names), np.nan)
        for j, name in enumerate(names):
            values = oriented[:, j]
            sorted_values.append(np.sort(values))
            operation = FEATURE_TRANSFORMS.get(name, "raw")
            if operation == "squared":
                transformed[:, j] = -(values ** 2)
            elif operation == "mode":
                centres[j] = _mode_percentile(values)
                percentiles = (rankdata(values) - 0.5) / len(values)
                transformed[:, j] = -np.abs(percentiles - centres[j])
            elif operation != "raw":
                raise RuntimeError(f"unsupported mixed-v2 operation: {operation}")
        output_mean = transformed.mean(axis=0)
        output_std = _safe_scale(transformed.std(axis=0))
        training = (transformed - output_mean[None, :]) / output_std[None, :]
        return cls(
            names, medians, mean, std, tuple(sorted_values), centres,
            output_mean, output_std, training,
        )

    def transform(self, raw_matrix):
        raw = np.asarray(raw_matrix, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(self.names):
            raise ValueError("raw_matrix has the wrong shape")
        filled = np.where(np.isfinite(raw), raw, self.raw_median[None, :])
        signs = np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[n] for n in self.names])
        oriented = (
            filled * signs[None, :] - self.oriented_mean[None, :]
        ) / self.oriented_std[None, :]
        transformed = oriented.copy()
        for j, name in enumerate(self.names):
            operation = FEATURE_TRANSFORMS.get(name, "raw")
            if operation == "squared":
                transformed[:, j] = -(oriented[:, j] ** 2)
            elif operation == "mode":
                reference = self.sorted_oriented[j]
                left = np.searchsorted(reference, oriented[:, j], side="left")
                right = np.searchsorted(reference, oriented[:, j], side="right")
                percentile = (0.5 * (left + right) + 0.5) / len(reference)
                transformed[:, j] = -np.abs(percentile - self.mode_centres[j])
        return (
            transformed - self.output_mean[None, :]
        ) / self.output_std[None, :]


def covariance_components(original, replicates):
    """Estimate total, within-procedure, and raw signal covariance matrices."""
    original = np.asarray(original, dtype=float)
    replicates = np.asarray(replicates, dtype=float)
    if replicates.ndim != 3 or replicates.shape[0] != original.shape[0]:
        raise ValueError("replicates must have shape (items, repeats, features)")
    total = np.cov(original, rowvar=False, ddof=1)
    centered = replicates - replicates.mean(axis=1, keepdims=True)
    within = np.einsum("irp,irq->pq", centered, centered)
    within /= replicates.shape[0] * max(replicates.shape[1] - 1, 1)
    within = 0.5 * (within + within.T)
    signal = 0.5 * ((total - within) + (total - within).T)
    return total, within, signal


def psd_projection(matrix):
    matrix = 0.5 * (np.asarray(matrix) + np.asarray(matrix).T)
    values, vectors = eigh(matrix)
    clipped = np.maximum(values, 0.0)
    return (vectors * clipped[None, :]) @ vectors.T, values


def generalized_reliability(signal, within, max_condition=100.0):
    """Solve S_signal v=lambda S_within v after explicit PSD/ridge repair."""
    signal_psd, raw_signal_eigenvalues = psd_projection(signal)
    noise_values = eigh(within, eigvals_only=True)
    high, low = float(noise_values[-1]), float(noise_values[0])
    ridge_for_condition = max(
        0.0, (high - max_condition * low) / (max_condition - 1.0)
    )
    floor = max(float(np.trace(within)) / max(len(within), 1) * 1e-6, 1e-8)
    ridge = max(ridge_for_condition, floor)
    regularized = within + ridge * np.eye(len(within))
    values, vectors = eigh(signal_psd, regularized)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return {
        "eigenvalues": values,
        "vectors": vectors,
        "ridge": float(ridge),
        "noise_condition": float(np.linalg.cond(regularized)),
        "raw_signal_eigenvalues": raw_signal_eigenvalues,
    }


def matrix_correlation(left, right):
    indices = np.triu_indices_from(left)
    a, b = np.asarray(left)[indices], np.asarray(right)[indices]
    if np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def subspace_overlap(left_vectors, right_vectors, k=3):
    """Mean squared canonical cosine; one means identical subspaces."""
    k = int(min(k, left_vectors.shape[1], right_vectors.shape[1]))
    if k < 1:
        return float("nan")
    q_left, _ = np.linalg.qr(left_vectors[:, :k])
    q_right, _ = np.linalg.qr(right_vectors[:, :k])
    return float(np.linalg.norm(q_left.T @ q_right, ord="fro") ** 2 / k)


__all__ = [
    "FixedMixedV2Transformer",
    "bootstrap_trace_row",
    "circular_moving_block_indices",
    "covariance_components",
    "generalized_reliability",
    "matrix_correlation",
    "psd_projection",
    "subspace_overlap",
]
