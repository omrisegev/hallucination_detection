"""REWIND-inspired entropy cut-point locator (Ding et al., arXiv 2610.19561).

REWIND chooses where to cut a reasoning trace before regenerating its suffix:

1. peak      -- the generated token with the highest entropy;
2. gradient  -- within the ``window`` tokens before the peak (peak included),
                the token with the largest one-step entropy increase
                ``H[t] - H[t-1]``;
3. aligned   -- the last delimiter token within the ``lookback`` tokens before
                that start point; the prefix is kept *through* the delimiter, so
                the cut (first regenerated) token is ``delimiter + 1``.  Without a
                delimiter the gradient start point is the cut.

The paper uses this rule for regeneration, never for localization (its benchmarks
have no first-error labels).  Here the cut token is mapped to the ProcessBench step
containing it and read as a predicted first error.  Deviations from the paper:
entropies are the cached top-K entropies, not full-vocabulary entropies, and the
delimiter set is ``. , \\n`` matched as substrings of byte-level BPE tokens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

REWIND_WINDOW = 5
REWIND_LOOKBACK = 10
# Byte-level BPE renders "\n" as "Ċ" (Llama-3 / Qwen vocabularies).
DELIMITER_MARKERS = (".", ",", "\n", "Ċ")


def delimiter_mask(token_strings: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [any(marker in text for marker in DELIMITER_MARKERS) for text in token_strings],
        dtype=bool,
    )


def rewind_cut_points(
    entropies: Sequence[float],
    is_delimiter: Sequence[bool],
    *,
    window: int = REWIND_WINDOW,
    lookback: int = REWIND_LOOKBACK,
) -> dict[str, int]:
    """Return the token index after each REWIND stage (-1 for an empty trace)."""

    h = np.asarray(entropies, dtype=float)
    if not len(h) or not np.isfinite(h).any():
        return {"peak": -1, "gradient": -1, "aligned": -1}
    h = np.where(np.isfinite(h), h, -np.inf)
    peak = int(np.argmax(h))

    lo = max(1, peak - window)
    if peak >= 1:
        candidates = np.arange(lo, peak + 1)
        jumps = h[candidates] - h[candidates - 1]
        start = int(candidates[int(np.argmax(jumps))])
    else:
        start = peak

    delimiters = np.asarray(is_delimiter, dtype=bool)
    aligned = start
    for index in range(start - 1, max(-1, start - 1 - lookback), -1):
        if delimiters[index]:
            aligned = index + 1
            break
    return {"peak": peak, "gradient": start, "aligned": int(aligned)}


def token_to_step(token: int, step_token_spans: Sequence[Any]) -> int:
    if token < 0:
        return -1
    spans = step_token_spans or ()
    for index, span in enumerate(spans):
        if span is not None and int(span[0]) <= int(token) < int(span[1]):
            return int(index)
    return int(len(spans) - 1) if spans else -1


def rewind_row(row: Mapping[str, Any], id_to_token) -> dict[str, Any]:
    """Detector score (mean token entropy) plus stage-wise cut tokens and steps."""

    entropies = np.asarray(row["token_entropies"], dtype=float)
    strings = [id_to_token(int(token)) or "" for token in row["gen_token_ids"]]
    if len(strings) != len(entropies):
        raise ValueError(f"{row.get('id')}: token ids and entropies differ in length")
    cuts = rewind_cut_points(entropies, delimiter_mask(strings))
    spans = row["step_token_spans"]
    return {
        "mean_entropy": float(np.nanmean(entropies)) if len(entropies) else float("nan"),
        "cut_tokens": cuts,
        "cut_steps": {stage: token_to_step(token, spans) for stage, token in cuts.items()},
    }
