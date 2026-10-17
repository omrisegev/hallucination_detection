"""Drop the first forced token from a teacher-forced telemetry entry ("mask1").

Why: with Qwen3's empty think block in the prefix (2026-10-10), the first forced token of a
teacher-forced trace still carries 10-28 nats of surprisal, because right after the block the
model expects almost exactly one opening word ("We") and most benchmark traces start with
another. That token is a property of the model meeting someone else's first word, not evidence
about any step, so Omri decided (2026-10-10) to remove it before scoring.

The rule, applied identically to every cell:
  * every per-token field loses index 0 (`PER_TOKEN_FIELDS`, plus any extra field whose
    leading length equals the trace length, which must be declared by the caller);
  * every half-open step span [lo, hi) becomes [max(lo - 1, 0), hi - 1);
  * a step that becomes empty (hi - 1 <= its new lo) is an explicit failure: `MaskError`,
    never a silent drop. A step that was ALREADY empty in the source (hi <= lo; the external
    records keep empty steps as (k, k)) stays empty: (k, k) -> (max(k-1, 0), max(k-1, 0)).
The scoring prefix is NOT recomputed: the telemetry of tokens 1..T-1 is unchanged, it simply no
longer includes token 0.
"""
from __future__ import annotations

import copy

import numpy as np

# Per-token fields of run_teacher_forced / run_prmbench_teacher_forced / run_external_telemetry.
PER_TOKEN_FIELDS = (
    "gen_token_ids", "token_entropies", "token_spilled_energies", "token_logsumexp",
    "token_entropy_full", "actual_token_logprobs", "token_offsets",
)
TOP_K_FIELDS = ("top_k_logprobs", "top_k_logprobs_raw")
SPAN_FIELDS = ("step_token_spans",)
RULE = ("mask1: drop forced token 0 from every per-token field; step spans [lo,hi) -> "
        "[max(lo-1,0), hi-1]; a step that becomes empty raises MaskError")


class MaskError(ValueError):
    """A step would become empty, or the entry is inconsistent with the rule."""


def _drop0(value, n):
    if isinstance(value, np.ndarray):
        if value.shape[:1] != (n,):
            raise MaskError(f"per-token array has leading shape {value.shape[:1]}, trace {n}")
        return value[1:].copy()
    value = list(value)
    if len(value) != n:
        raise MaskError(f"per-token list has length {len(value)}, trace {n}")
    return value[1:]


def shift_spans(spans):
    out = []
    for i, sp in enumerate(spans):
        if sp is None:  # an unmapped step stays unmapped (already flagged by align_diag)
            out.append(None)
            continue
        lo, hi = int(sp[0]), int(sp[1])
        if hi <= lo:
            # Already empty in the source (external records keep empty steps as (k, k)): it stays empty,
            # shifted with its neighbours. Only a step that BECOMES empty is a failure.
            k = max(lo - 1, 0)
            out.append((k, k) if isinstance(sp, tuple) else type(sp)((k, k)))
            continue
        nlo, nhi = max(lo - 1, 0), hi - 1
        if nhi <= nlo:
            raise MaskError(f"step {i} span ({lo}, {hi}) becomes empty after dropping token 0")
        out.append((nlo, nhi) if isinstance(sp, tuple) else type(sp)((nlo, nhi)))
    return out


def mask_first_token(entry: dict, extra_per_token=()) -> dict:
    """Return a NEW entry with forced token 0 removed (the input is not modified)."""
    if "gen_token_ids" not in entry:
        raise MaskError("entry has no gen_token_ids")
    n = len(entry["gen_token_ids"])
    if n < 2:
        raise MaskError(f"trace of length {n} cannot lose its first token")
    out = copy.deepcopy(entry)
    for key in tuple(PER_TOKEN_FIELDS) + tuple(extra_per_token):
        if key in out and out[key] is not None:
            out[key] = _drop0(out[key], n)
    for key in TOP_K_FIELDS:
        tk = out.get(key)
        if isinstance(tk, dict):
            out[key] = {k: _drop0(v, n) for k, v in tk.items()}
        elif tk is not None:
            out[key] = _drop0(tk, n)
    for key in SPAN_FIELDS:
        if key in out and out[key] is not None:
            out[key] = shift_spans(out[key])
    diag = out.get("align_diag")
    if isinstance(diag, dict) and "step_token_lengths" in diag and out.get("step_token_spans"):
        diag["step_token_lengths"] = [0 if sp is None else sp[1] - sp[0]
                                      for sp in out["step_token_spans"]]
    out["first_token_mask"] = {"rule": "mask1", "dropped_token_id": int(entry["gen_token_ids"][0])}
    return out


def mask_cache(cache: dict, extra_per_token=()):
    """Mask every entry of a {key: entry} cache. Returns (masked, failures) where failures is
    a list of {key, error}; failed entries are NOT in `masked` and must be reported."""
    masked, failures = {}, []
    for k, e in cache.items():
        try:
            masked[k] = mask_first_token(e, extra_per_token)
        except MaskError as ex:
            failures.append({"key": str(k), "error": str(ex)})
    return masked, failures
