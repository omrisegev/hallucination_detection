"""Versioned orientation contract for hallucination-detection features.

Every sign in ``CONFIDENCE_FEATURE_SIGNS_V1`` has the same meaning:

    oriented_value = raw_value * sign
    higher oriented_value = more likely correct

The mapping is frozen from the Step-204/218 orientation audit and the subsequent
eight-family leave-one-family-out check.  It is an offline calibration artifact,
not something that may be re-estimated from labels in a deployment cell.

``LEGACY_FEATURE_SIGNS`` reproduces the historical pipeline.  In particular,
the four spilled-energy views were absent from the old table and therefore
silently received ``+1`` through ``dict.get(..., +1)``.  They are explicit here
so legacy reconstruction does not depend on that fallback.

The stable schema excludes the four views whose relationship is recurrently
non-monotone or whose direction does not transfer reliably.  A future
transformation may replace them, but a raw and transformed copy must not coexist:
deterministic duplicates bias U-PCR's pair equations.
"""

from types import MappingProxyType

import numpy as np


SCHEMA_VERSION = "confidence-orientation-v1"


# Historical signs used by FEATURE_SIGNS + REPGRID_VIEW_SIGNS.  The spilled
# entries make the previous implicit +1 fallback explicit.
LEGACY_FEATURE_SIGNS = MappingProxyType({
    "epr": -1,
    "trace_length": +1,
    "spectral_entropy": -1,
    "low_band_power": -1,
    "high_band_power": -1,
    "hl_ratio": -1,
    "dominant_freq": -1,
    "spectral_centroid": -1,
    "stft_max_high_power": -1,
    "stft_spectral_entropy": -1,
    "rpdi": -1,
    "sw_var_peak": -1,
    "pe_mean": -1,
    "hurst_exponent": +1,
    "cusum_max": -1,
    "cusum_shift_idx": +1,
    "epr_spilled": +1,
    "sw_var_peak_spilled": +1,
    "cusum_max_spilled": +1,
    "min_spilled": +1,
    "epr_energy": -1,
    "min_energy": -1,
    "sw_var_peak_energy": -1,
    "cusum_max_energy": -1,
    "mean_top1_logprob": +1,
    "logprob_margin": +1,
    "mean_logprob_entropy": -1,
    "varentropy": -1,
    "renyi_entropy_2": -1,
    "topk_tail_mass": -1,
})


# Frozen confidence-oriented schema.  The thirteen differences from the legacy
# mapping are deliberate; see FIXED_SIGN_CHANGES_V1 below.
CONFIDENCE_FEATURE_SIGNS_V1 = MappingProxyType({
    "epr": -1,
    "trace_length": -1,
    "spectral_entropy": -1,
    "low_band_power": -1,
    "high_band_power": +1,
    "hl_ratio": +1,
    "dominant_freq": +1,
    "spectral_centroid": +1,
    "stft_max_high_power": -1,
    "stft_spectral_entropy": -1,
    "rpdi": -1,
    "sw_var_peak": -1,
    "pe_mean": -1,
    "hurst_exponent": -1,
    "cusum_max": -1,
    "cusum_shift_idx": -1,
    "epr_spilled": -1,
    "sw_var_peak_spilled": -1,
    "cusum_max_spilled": -1,
    "min_spilled": -1,
    "epr_energy": +1,
    "min_energy": +1,
    "sw_var_peak_energy": -1,
    "cusum_max_energy": -1,
    "mean_top1_logprob": +1,
    "logprob_margin": +1,
    "mean_logprob_entropy": -1,
    "varentropy": -1,
    "renyi_entropy_2": -1,
    "topk_tail_mass": -1,
})


FIXED_SIGN_CHANGES_V1 = MappingProxyType({
    name: CONFIDENCE_FEATURE_SIGNS_V1[name]
    for name, old_sign in LEGACY_FEATURE_SIGNS.items()
    if CONFIDENCE_FEATURE_SIGNS_V1[name] != old_sign
})


# These raw views are quarantined, not declared useless.  They may return after
# a separately validated replacement such as a mode-centred fold.
FIXED_STABLE_EXCLUDED_V1 = frozenset({
    "pe_mean",
    "stft_spectral_entropy",
    "cusum_shift_idx",
    "rpdi",
})


def confidence_sign_vector(feature_names):
    """Return the frozen v1 sign vector, failing on an unregistered feature."""
    names = list(feature_names)
    missing = [name for name in names if name not in CONFIDENCE_FEATURE_SIGNS_V1]
    if missing:
        raise KeyError(
            "feature(s) missing from confidence orientation contract: "
            + ", ".join(sorted(set(missing)))
        )
    return np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[name] for name in names], dtype=float)


def confidence_oriented_matrix(raw_matrix, feature_names, *, stable=False):
    """Orient an ``(n_samples, n_features)`` raw matrix under the frozen schema.

    Returns ``(matrix, kept_names, signs)``.  With ``stable=True`` the four
    quarantined raw views are removed before orientation.
    """
    raw = np.asarray(raw_matrix, dtype=float)
    names = list(feature_names)
    if raw.ndim != 2 or raw.shape[1] != len(names):
        raise ValueError(
            f"raw_matrix shape {raw.shape} does not match {len(names)} feature names"
        )
    if not np.isfinite(raw).all():
        raise ValueError("raw_matrix contains non-finite values")
    keep = np.asarray([
        not stable or name not in FIXED_STABLE_EXCLUDED_V1 for name in names
    ], dtype=bool)
    kept_names = [name for name, selected in zip(names, keep) if selected]
    signs = confidence_sign_vector(kept_names)
    return raw[:, keep] * signs, kept_names, signs


def consensus_anchor(oriented_matrix):
    """Return the mean-view global anchor for an already aligned feature matrix."""
    matrix = np.asarray(oriented_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("oriented_matrix must contain at least one feature")
    anchor = matrix.mean(axis=1)
    if not np.isfinite(anchor).all() or float(anchor.std()) < 1e-12:
        raise ValueError("feature consensus is non-finite or constant")
    return anchor


__all__ = [
    "SCHEMA_VERSION",
    "LEGACY_FEATURE_SIGNS",
    "CONFIDENCE_FEATURE_SIGNS_V1",
    "FIXED_SIGN_CHANGES_V1",
    "FIXED_STABLE_EXCLUDED_V1",
    "confidence_sign_vector",
    "confidence_oriented_matrix",
    "consensus_anchor",
]
