"""Frozen development feature contract for the next DUFS-LIU run.

This module does not select a contract.  It only applies the mixed contract that
was selected by the exhaustive 24-cell development search recorded in
``results/dufs_liu_feature_contract_search``.  The selected score is retrospective;
the mapping must stay frozen for evaluation on new dataset/model families.

Every replacement occupies the raw parent's single column.  Parent and transform
never coexist, because deterministic duplicates distort U-PCR's covariance pairs.
"""

from types import MappingProxyType

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, rankdata

from .feature_contract import (
    CONFIDENCE_FEATURE_SIGNS_V1,
    FIXED_STABLE_EXCLUDED_V1,
    LEGACY_FEATURE_SIGNS,
)


CONTRACT_VERSION = "dufs-liu-mixed-v2-development-2026-08-07"
FEATURE_ORDER = (
    "pe_mean",
    "stft_spectral_entropy",
    "cusum_shift_idx",
    "rpdi",
)
FEATURE_TRANSFORMS = MappingProxyType({
    "pe_mean": "squared",
    "stft_spectral_entropy": "mode",
    "cusum_shift_idx": "raw",
    "rpdi": "raw",
})
EPS = 1e-12


def _zscore_columns(matrix):
    matrix = np.asarray(matrix, dtype=float)
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale < EPS] = 1.0
    return centred / scale


def _percentile_rank(values):
    values = np.asarray(values, dtype=float)
    return (rankdata(values) - 0.5) / len(values)


def _mode_percentile(values, grid_size=512, min_prominence=0.05):
    """Return the label-free KDE mode location on the percentile scale."""
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
    if len(peaks):
        peak = int(peaks[np.argmax(properties["prominences"])])
    else:
        peak = int(np.argmax(density))
    return float(np.mean(values < grid[peak]))


def dufs_liu_mixed_v2_matrix(raw_matrix, feature_names):
    """Apply the frozen mixed-v2 contract to a raw sample-by-feature matrix.

    Returns ``(matrix, names, details)``.  Missing registered views stay missing;
    every other in-scope view keeps its confidence-oriented raw value.
    """
    if set(FEATURE_ORDER) != set(FIXED_STABLE_EXCLUDED_V1):
        raise RuntimeError("the registered four-view quarantine changed")
    raw = np.asarray(raw_matrix, dtype=float)
    names = tuple(str(name) for name in feature_names)
    if raw.ndim != 2 or raw.shape[1] != len(names):
        raise ValueError("raw matrix and feature names disagree")
    if not np.isfinite(raw).all():
        raise ValueError("raw matrix contains non-finite values")
    unknown = sorted(set(names) - set(CONFIDENCE_FEATURE_SIGNS_V1))
    if unknown:
        raise KeyError("unregistered feature(s): " + ", ".join(unknown))

    signs = np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[name] for name in names], dtype=float)
    oriented = _zscore_columns(raw * signs[None, :])
    output = oriented.copy()
    details = {}
    for index, name in enumerate(names):
        choice = FEATURE_TRANSFORMS.get(name, "raw")
        values = oriented[:, index]
        if choice == "raw":
            replacement = values
            centre = None
        elif choice == "squared":
            replacement = -(values ** 2)
            centre = 0.0
        elif choice == "mode":
            centre = _mode_percentile(values)
            replacement = -np.abs(_percentile_rank(values) - centre)
        else:  # The frozen registry must never silently acquire a new operation.
            raise RuntimeError(f"unsupported frozen transform: {choice}")
        output[:, index] = replacement
        if name in FEATURE_TRANSFORMS:
            details[name] = {"transform": choice, "centre": centre}
    return _zscore_columns(output), names, details


def dufs_liu_mixed_v2_from_bundle(stored_matrix, feature_names, stored_legacy_signs):
    """Recover raw columns from the exported bundle and apply mixed-v2."""
    stored = np.asarray(stored_matrix, dtype=float)
    names = tuple(str(name) for name in feature_names)
    legacy = np.asarray(stored_legacy_signs, dtype=float)
    if stored.ndim != 2 or stored.shape[1] != len(names):
        raise ValueError("stored matrix and feature names disagree")
    if legacy.shape != (len(names),):
        raise ValueError("stored legacy signs have the wrong shape")
    expected = np.asarray([LEGACY_FEATURE_SIGNS[name] for name in names], dtype=float)
    if not np.array_equal(legacy, expected):
        raise RuntimeError("stored legacy signs disagree with the registered mapping")
    raw = stored * legacy[None, :]
    return dufs_liu_mixed_v2_matrix(raw, names)


__all__ = [
    "CONTRACT_VERSION",
    "FEATURE_ORDER",
    "FEATURE_TRANSFORMS",
    "dufs_liu_mixed_v2_matrix",
    "dufs_liu_mixed_v2_from_bundle",
]
