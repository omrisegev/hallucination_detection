"""Derivative step channel (token-probability line, length axis, attack 2).

The mechanism, not the recipe. A level readout carries an order-statistic prior that
grows with step length: the Top-k mean of more tokens is larger even with no signal.
A derivative readout differences the level away, so the per-step *level* cannot leak
in. Attack 1 (`length_calibrated_readout`) tried to subtract the prior out of the
level statistic and is closed -- its cost was uniform across chain length. This
attacks the same defect from the other side, by using a statistic that does not
carry the prior to begin with.

Borrowed idea, deliberately not a transplant, per the project's standing rule:

  - Their score is computed **per answer**; ours must be a **step** readout, so the
    aggregation happens inside each step.
  - Their series is Shannon entropy; ours runs on **every oriented channel** of the
    bank, entropy being only one of them.
  - Every constant here is a project choice, not theirs: the EMA span is the bank's
    own WINDOW (16) and the per-step aggregation keeps the WORST_M = 3 sharpest
    rises. Nothing is labelled with an author's name.

Orientation. Bank channels are already signed so that larger means more suspect, so
a *drop in evidence* is a *rise in risk*: we keep positive first differences.

A caveat this module does not resolve: "mean of the M largest rises in the step" is
itself an order statistic and so is not automatically length-free. Whether it is
length-free in practice is an empirical question, and the diagnostic that answers it
(`within_answer_corr_log_length`) is reported beside every result, exactly as Step 420
reported it for the level readout.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

EMA_WINDOW = 16          # the token bank's own WINDOW; not taken from any paper
WORST_M = 3              # sharpest rises kept inside each step


def ema(values: np.ndarray, window: int = EMA_WINDOW) -> np.ndarray:
    """Causal exponential moving average, span convention alpha = 2 / (window + 1).

    Accepts ``[tokens]`` or ``[tokens, channels]`` and filters down axis 0. The
    recurrence ``out[i] = a*x[i] + (1-a)*out[i-1]`` seeded with ``out[-1] = x[0]`` is a
    one-pole IIR filter, so it is evaluated by `lfilter` in C rather than by a Python
    loop: at 7M tokens by 11 channels the loop is ~77M iterations and dominates the
    whole extraction.
    """
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
    """Return the ``[steps, channels]`` derivative readout for one answer.

    ``matrix`` is the risk-oriented ``[tokens, channels]`` bank for a single answer and
    ``spans`` its ``[steps, 2]`` token spans. Per channel: EMA, first difference, keep
    the rises, then take the mean of the ``worst_m`` largest rises inside each step.
    A step with no token, or an answer too short to difference, contributes zero.
    """
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


def level_step_readout(matrix: np.ndarray, spans: np.ndarray, k: int = 10) -> np.ndarray:
    """The matching LEVEL readout: Top-k mean per step, per channel.

    Present so the derivative is always compared against a level readout built by the
    same code path over the same tokens, rather than against a number from another run.
    """
    x = np.asarray(matrix, dtype=float)
    spans = np.asarray(spans, dtype=int)
    out = np.zeros((len(spans), x.shape[1]), dtype=float)
    for s, (a, b) in enumerate(spans):
        seg = x[a:b]
        if not len(seg):
            continue
        kk = min(k, len(seg))
        out[s] = np.partition(seg, len(seg) - kk, axis=0)[-kk:].mean(axis=0)
    return out
