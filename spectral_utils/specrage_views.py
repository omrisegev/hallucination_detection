"""Frozen feature-provenance views for SpecRaGE-U-PCR.

The grouping in this file is determined by the primitive trajectory from which
each feature is computed.  It is not learned from correctness labels or from
feature AUROC.  Keeping the grouping explicit prevents a seemingly innocuous
view rearrangement from becoming an unreported supervised hyperparameter.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .feature_contract import (
    CONFIDENCE_FEATURE_SIGNS_V1,
    FIXED_STABLE_EXCLUDED_V1,
    LEGACY_FEATURE_SIGNS,
)
from .dufs_liu_feature_contract import dufs_liu_mixed_v2_from_bundle


EPS = 1e-12
VIEW_SCHEMA_VERSION = "specrage-provenance-views-v1-2026-08-06"


FEATURE_TO_VIEW = {
    # Mean entropy is separated from trajectory dynamics so SpecRaGE can
    # distinguish a level shift from frequency/time-domain corruption.
    "epr": "entropy_level",
    "trace_length": "structural",
    "spectral_entropy": "entropy_dynamics",
    "low_band_power": "entropy_dynamics",
    "high_band_power": "entropy_dynamics",
    "hl_ratio": "entropy_dynamics",
    "dominant_freq": "entropy_dynamics",
    "spectral_centroid": "entropy_dynamics",
    "stft_max_high_power": "entropy_dynamics",
    "stft_spectral_entropy": "entropy_dynamics",
    "rpdi": "entropy_dynamics",
    "sw_var_peak": "entropy_dynamics",
    "pe_mean": "entropy_dynamics",
    "hurst_exponent": "entropy_dynamics",
    "cusum_max": "entropy_dynamics",
    "cusum_shift_idx": "entropy_dynamics",
    # Realized sampled-token surprisal / spilled-energy trajectory.
    "epr_spilled": "sampled_token_energy",
    "sw_var_peak_spilled": "sampled_token_energy",
    "cusum_max_spilled": "sampled_token_energy",
    "min_spilled": "sampled_token_energy",
    # Full-vocabulary log-partition trajectory.
    "epr_energy": "partition_energy",
    "min_energy": "partition_energy",
    "sw_var_peak_energy": "partition_energy",
    "cusum_max_energy": "partition_energy",
    # Top-k distribution summaries.
    "mean_top1_logprob": "topk_distribution",
    "logprob_margin": "topk_distribution",
    "mean_logprob_entropy": "topk_distribution",
    "varentropy": "topk_distribution",
    "renyi_entropy_2": "topk_distribution",
    "topk_tail_mass": "topk_distribution",
}

VIEW_ORDER = (
    "entropy_level",
    "entropy_dynamics",
    "sampled_token_energy",
    "partition_energy",
    "topk_distribution",
    "structural",
)


def zscore_columns(matrix):
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale < EPS] = 1.0
    return centered / scale


def fixed_stable_from_bundle(stored_matrix, names, stored_legacy_signs):
    """Recover raw columns from the artifact and apply fixed_stable_v1.

    ``stored_matrix`` is the bundle's historically oriented sample-by-feature
    matrix.  Multiplying by the stored +/-1 legacy signs recovers its raw
    standardized direction.  The registered confidence signs are then applied
    before unstable columns are removed.
    """
    matrix = np.asarray(stored_matrix, dtype=float)
    names = tuple(str(name) for name in names)
    legacy = np.asarray(stored_legacy_signs, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("stored matrix and feature names disagree")
    if legacy.shape != (len(names),):
        raise ValueError("stored legacy signs have the wrong shape")
    unknown = sorted(set(names) - set(FEATURE_TO_VIEW))
    if unknown:
        raise KeyError("unregistered SpecRaGE feature(s): " + ", ".join(unknown))
    expected_legacy = np.asarray([LEGACY_FEATURE_SIGNS[name] for name in names])
    if not np.array_equal(legacy, expected_legacy):
        raise RuntimeError("stored legacy signs do not match the registered contract")
    raw = matrix * legacy[None, :]
    fixed = np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[name] for name in names])
    oriented = zscore_columns(raw * fixed[None, :])
    keep = np.asarray([name not in FIXED_STABLE_EXCLUDED_V1 for name in names])
    return oriented[:, keep], tuple(name for name, flag in zip(names, keep) if flag)


def provenance_views(matrix, names, *, require_two=True):
    """Split an oriented sample-by-feature matrix into frozen named views."""
    matrix = np.asarray(matrix, dtype=float)
    names = tuple(str(name) for name in names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("matrix and feature names disagree")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    unknown = sorted(set(names) - set(FEATURE_TO_VIEW))
    if unknown:
        raise KeyError("unregistered SpecRaGE feature(s): " + ", ".join(unknown))
    output = OrderedDict()
    for view in VIEW_ORDER:
        indices = [index for index, name in enumerate(names)
                   if FEATURE_TO_VIEW[name] == view]
        if indices:
            output[view] = zscore_columns(matrix[:, indices])
    if require_two and len(output) < 2:
        raise ValueError("at least two provenance views are required")
    return output


def view_members(names):
    """Return the auditable feature membership for a concrete feature pool."""
    names = tuple(str(name) for name in names)
    unknown = sorted(set(names) - set(FEATURE_TO_VIEW))
    if unknown:
        raise KeyError("unregistered SpecRaGE feature(s): " + ", ".join(unknown))
    return {
        view: [name for name in names if FEATURE_TO_VIEW[name] == view]
        for view in VIEW_ORDER
        if any(FEATURE_TO_VIEW[name] == view for name in names)
    }


__all__ = [
    "FEATURE_TO_VIEW",
    "VIEW_ORDER",
    "VIEW_SCHEMA_VERSION",
    "fixed_stable_from_bundle",
    "dufs_liu_mixed_v2_from_bundle",
    "provenance_views",
    "view_members",
    "zscore_columns",
]
