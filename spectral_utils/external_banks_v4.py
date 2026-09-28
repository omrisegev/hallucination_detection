"""external_banks_v4: versioned raw step channels added to the frozen 48-channel bank.

No labels are accepted anywhere in this module. Every channel is a per-answer,
offline function of one answer's teacher-forced telemetry and its step spans.

Channels (step level, raw = before the scorer's per-answer z-scoring):

  realized_z           CT7 view 7 'chosen_token_z_despiked': pooled chosen-token
                       z-test per step (step_z_readouts column 2 over the step
                       sufficient statistics), answer-standardized, step 0 set to
                       the mean of the other steps, standardized again. This channel
                       is answer-standardized BY DEFINITION; its "raw" value is the
                       definitional value (a further answer-z is idempotent).
                       Source: cumulative-vote-fusion-v2 ct7_profiles_v1/profiles.npy
                       column 6 (built by frozen_locator_ct7.despiked_chosen_token_z).
  realized_drv         DERIVATIVE_CHANNELS.npz 'derivative' column 'chosen_surprisal':
                       derivative_step_readout (EMA16 -> positive first difference ->
                       mean of the 3 largest rises in the step) of the risk-oriented
                       bank11 token matrix (claude_feature_bank_v1, float64).
  ct7_ve1              Top10 step mean of the CT7 order-1 escort varentropy token
                       stream (family_external_features.ct7_streams column 3).
  hist_entropy_series  finite-only Top10 step mean of the historical entropy_series
  hist_spilled_series  and spilled_series token streams (family_hist_features).

Digit family (optional, separate mask): digit_alternative, digit_spread,
digit_alternative_innovation from digit_feature_family.token_family over the
top-K ids/logprobs with digit ids 15..24, reduced by step_family (Top2 mean of
active tokens; 'active' marks steps with at least one active token).

Two short helpers are VENDORED verbatim because their source modules are not on
this branch; the source files and hashes are recorded in VENDORED_SOURCES and the
full-source parity gate (scripts/verify_external_banks_v4_source.py) is the
acceptance test of the vendored copies.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .digit_feature_family import NAMES as DIGIT_NAMES, step_family, token_family
from .external_generalization._bank11.chosen_token_calibration import (
    SUFFICIENT, step_sufficient_stats, step_z_readouts, token_calibration,
)
from .external_generalization._bank11.claude_feature_bank_v1 import FEATURE_NAMES as BANK11_NAMES
from .external_generalization.fusion import validate_telemetry
from .family_external_features import ct7_streams
from .family_hist_features import STREAM_NAMES as HIST_STREAM_NAMES, token_streams

VERSION = "external_banks_v4"
NEW_NAMES = ("realized_z", "realized_drv", "ct7_ve1", "hist_entropy_series", "hist_spilled_series")
DIGIT_NAMES = tuple(DIGIT_NAMES)
DIGIT_IDS = tuple(range(15, 25))  # '0'..'9' for the Qwen2.5/Qwen3 BPE vocabulary (verified by the extractor)

VENDORED_SOURCES = {
    "ema, derivative_step_readout": {
        "path": ".worktrees/token-probability-fusion-v1/spectral_utils/derivative_step_channel_v1.py",
        "sha256": "f8f3da23a24f19ac7819932239ea9540f1129294f61f65482e0e5b367af97e3d"},
    "despiked_chosen_token_z": {
        "path": ".worktrees/cumulative-vote-fusion-v2/spectral_utils/frozen_locator_ct7.py",
        "sha256": "b4ac9dc41067229a2c026776e8d53e82682a62c74a219d9e0b2e196a0f5c3f63"},
    "masked_answer_standardize": {
        "path": ".worktrees/cumulative-vote-fusion-v2/spectral_utils/digitfree_broad50.py",
        "sha256": "28817e3ae252499830e1ac937965f076de8162d3530307ab409970e3d6227180"},
}

# ---- vendored verbatim: derivative_step_channel_v1.py ---------------------------------
EMA_WINDOW = 16
WORST_M = 3


def ema(values: np.ndarray, window: int = EMA_WINDOW) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim not in (1, 2):
        raise ValueError("ema expects a 1-d or 2-d token series")
    if not len(x):
        return x.copy()
    alpha = 2.0 / (window + 1.0)
    zi = ((1.0 - alpha) * x[0])
    zi = np.atleast_1d(zi)[None, :] if x.ndim == 2 else np.atleast_1d(zi)
    out, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0, zi=zi)
    return out


def derivative_step_readout(matrix: np.ndarray, spans: np.ndarray,
                            window: int = EMA_WINDOW, worst_m: int = WORST_M) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    spans = np.asarray(spans, dtype=int)
    if x.ndim != 2:
        raise ValueError("expected a [tokens, channels] matrix")
    n_steps, n_channels = len(spans), x.shape[1]
    out = np.zeros((n_steps, n_channels), dtype=float)
    if len(x) < 2:
        return out

    live = np.isfinite(x).all(axis=0) & (x.std(axis=0) > 1e-12)
    if not live.any():
        return out

    smoothed = ema(x[:, live], window)
    rises = np.diff(smoothed, axis=0, prepend=smoothed[:1])
    np.maximum(rises, 0.0, out=rises)

    idx = np.flatnonzero(live)
    for s, (a, b) in enumerate(spans):
        seg = rises[a:b]
        if not len(seg):
            continue
        k = min(worst_m, len(seg))
        out[s, idx] = np.partition(seg, len(seg) - k, axis=0)[-k:].mean(axis=0)
    return out


# ---- vendored verbatim: digitfree_broad50.masked_answer_standardize ---------------------
def masked_answer_standardize(x, available, offsets):
    """Unavailable evidence is masked, then neutral zero in standardized space."""
    out = np.zeros_like(x, dtype=float)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for j in range(x.shape[1]):
            ok = available[a:b, j]; v = x[a:b, j][ok]
            if len(v) and v.std() > 1e-12:
                out[a:b, j][ok] = (v-v.mean())/v.std()
    return out


# ---- vendored verbatim: frozen_locator_ct7.despiked_chosen_token_z ----------------------
def despiked_chosen_token_z(sufficient_stats: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Pooled chosen-token z-test per step, step 0 neutralized, answer-standardized."""
    stats = np.asarray(sufficient_stats, float)
    if stats.ndim != 2 or stats.shape[1] != len(SUFFICIENT) or stats.shape[0] != offsets[-1]:
        raise ValueError("sufficient statistics do not match the step offsets")
    z = step_z_readouts(stats)[:, 2]
    z = masked_answer_standardize(z[:, None], np.isfinite(z)[:, None], offsets)[:, 0]
    out = z.copy()
    for a, b in zip(offsets[:-1], offsets[1:]):
        if b - a > 1:
            out[a] = z[a + 1:b].mean()
    return masked_answer_standardize(out[:, None], np.isfinite(out)[:, None], offsets)[:, 0]


# ---- channel definitions -----------------------------------------------------------------
def checked_spans(row, spans=None):
    """Integer [steps, 2] spans; empty steps must be removed by the caller."""
    spans = np.asarray(row["step_token_spans"] if spans is None else spans)
    if spans.ndim != 2 or spans.shape[1] != 2 or not len(spans) or not np.issubdtype(spans.dtype, np.integer):
        raise ValueError("step spans must be a nonempty integer (steps, 2) array")
    spans = spans.astype(int)
    n = len(row["gen_token_ids"])
    if np.any(spans[:, 1] <= spans[:, 0]):
        raise ValueError("caller must remove empty steps before feature extraction")
    if np.any(spans[:, 0] < 0) or np.any(spans[:, 1] > n):
        raise ValueError("step span outside the token trace")
    return spans


def _topk(row):
    top = row["top_k_logprobs"]
    return np.asarray(top["ids"]), np.asarray(top["logprobs"], float)


def realized_z(row, spans):
    ids, lp = _topk(row)
    x, censored, diag = token_calibration(lp, ids, np.asarray(row["gen_token_ids"]),
                                          np.asarray(row["token_spilled_energies"]))
    stats = step_sufficient_stats(x, censored, diag, spans)
    return despiked_chosen_token_z(stats, np.array([0, len(spans)]))


def realized_drv(row, spans):
    bank = validate_telemetry(row)
    return derivative_step_readout(bank, spans)[:, list(BANK11_NAMES).index("chosen_surprisal")]


def ct7_ve1(row, spans):
    ct, valid = ct7_streams(row)
    out = []
    for a, b in spans:
        v = ct[a:b, 3][valid[a:b, 3]]
        k = min(10, len(v))
        out.append(np.sort(v)[-k:].mean() if k else np.nan)
    return np.array(out)


def hist_series(row, spans, names=("entropy_series", "spilled_series")):
    """Historical finite-only Top10 step means (family_hist_features.reduce_steps rule)."""
    matrix = token_streams(row)
    out = np.full((len(spans), len(names)), np.nan)
    for k, name in enumerate(names):
        column = HIST_STREAM_NAMES.index(name)
        for j, (a, b) in enumerate(spans):
            values = matrix[a:b, column]
            values = values[np.isfinite(values)]
            if len(values):
                count = min(10, len(values))
                out[j, k] = np.partition(values, len(values) - count)[-count:].mean()
    return out


def new_step_features(row, spans=None):
    """Raw [steps, 5] matrix of NEW_NAMES; no missing-value fill (callers check finiteness)."""
    spans = checked_spans(row, spans)
    hist = hist_series(row, spans)
    matrix = np.column_stack((realized_z(row, spans), realized_drv(row, spans),
                              ct7_ve1(row, spans), hist[:, 0], hist[:, 1]))
    if matrix.shape != (len(spans), len(NEW_NAMES)):
        raise ValueError("external_banks_v4 schema changed")
    return matrix, NEW_NAMES


def digit_step_features(row, spans=None):
    """Raw [steps, 3] digit values and their [steps, 3] availability mask.

    Mirrors scripts/run_digit_family_extension_v1.py: token_family(ids, logprobs,
    range(15, 25)) then step_family. Unavailable steps hold storage zero.
    """
    spans = checked_spans(row, spans)
    ids, lp = _topk(row)
    values, active, _ = token_family(ids, lp, range(DIGIT_IDS[0], DIGIT_IDS[-1] + 1))
    return step_family(values, active, spans)
