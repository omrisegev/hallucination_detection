"""Token-level builders of CT7's seven streams (item 3, 2026-09-23).

CT7 (`frozen_locator_ct7.py`) is a step-level recipe: five digit-free bank streams read by a
Top10 step mean, a BOCPD signed residual read the same way, and a pooled chosen-token z-test per
step, all answer-standardized and averaged. Item 3 asks what happens when the seven streams are
fused at TOKEN level before any readout. This module builds the token streams from one raw
telemetry row so that the extraction script and the tests share one code path.

Columns (order = CT7 view order):
  0 H0lim, 1 ve0, 2 ve0.75, 3 ve1, 4 H0lim_prefix_innovation  digitfree_broad50.token_bank columns
                                                              [27, 28, 30, 31, 41]; token 0 of the
                                                              innovation is invalid (masked)
  5 bocpd_residual   mean over the five standardized bank streams of (z - BOCPD prior mean),
                     hazard 1/32 (aligned_context_predictors.bocpd_mean); recomputed from the
                     answer's own five streams with whole-answer mean/std
  6 chosen_std_excess  chosen_token_calibration.token_calibration column 3: the per-token
                     standardized excess surprisal (-log q(x) - H) / sqrt(VE + .01)

The BOCPD column of the frozen CT7 view was built from `temporal_context_data_v1` (Step 387 /
Step 420 recomputation). The extraction script uses that source verbatim when it is present and
falls back to `bocpd_residual_from_bank` otherwise; either way the answer-standardized Top10 of
the column must replay CT7's view 5 (gate ii of the protocol).

No labels anywhere in this module.
"""
from __future__ import annotations

import numpy as np

from .aligned_context_predictors import bocpd_mean
from .chosen_token_calibration import token_calibration
from .digitfree_broad50 import NAMES as BANK50, token_bank
from .length_calibrated_readout import step_topk_and_calibrated

STREAMS = ("H0lim", "ve0", "ve0.75", "ve1", "H0lim_prefix_innovation", "bocpd_residual", "chosen_std_excess")
BANK_COLS = [list(BANK50).index(n) for n in STREAMS[:5]]
CHOSEN_COL = 3          # std_excess_surprisal in token_calibration's [pit, pit_normal, excess, std_excess]
HAZARD = 1 / 32


def bank_streams(logp, ids, provided, surprisal) -> tuple[np.ndarray, np.ndarray]:
    """[T x 5] bank streams and their validity mask (token 0 of the innovation is invalid)."""
    x, valid, _ = token_bank(logp, ids, provided, surprisal)
    return x[:, BANK_COLS].astype(float), valid[:, BANK_COLS].astype(bool)


def bocpd_residual_from_bank(x5: np.ndarray, valid5: np.ndarray, *, hazard: float = HAZARD) -> np.ndarray:
    """Signed BOCPD residual per token from the answer's own five streams.

    z = (x - mean)/scale with the whole-answer mean and sd of each stream over its valid tokens
    (invalid entries set to 0 in z), then mean over the five streams of z - prior predictive
    mean of the reset-before-observation Gaussian BOCPD (unit variances, hazard 1/32).
    """
    x = np.asarray(x5, float); v = np.asarray(valid5, bool)
    z = np.zeros_like(x)
    for j in range(x.shape[1]):
        col = x[v[:, j], j]
        if len(col) and col.std() > 1e-12:
            z[v[:, j], j] = (col - col.mean()) / col.std()
    return (z - bocpd_mean(z, hazard=hazard)).mean(axis=1)


def bocpd_residual_temporal_recipe(logp, entropy, *, hazard: float = HAZARD) -> np.ndarray:
    """Signed BOCPD residual per token, rebuilt from the raw row by the recipe that produced
    `temporal_context_data_v1` and CT7's view 5, for use when that bundle is absent.

    Chain (scripts/run_temporal_research_baseline.py -> scripts/prepare_temporal_context_data.py
    -> scripts/run_length_calibrated_streams_v1.py::_bocpd_one): the first four oriented columns
    of `renyi_locator_feature_bank.feature_matrix(logprobs, token_entropies)` (float64), the prefix
    innovation of the first column appended (token 0 = 0, included), per-answer mean and
    max(std, 1e-8) over that float64 matrix, the matrix stored as float32 in features.npy and read
    back as float64, then z = (raw - mean)/scale and the mean over the five streams of
    z - bocpd_mean(z). This differs from `bocpd_residual_from_bank` (a different orientation
    rule, token 0 masked, no float32 round trip), which is why that route cannot replay view 5.
    """
    from .renyi_locator_feature_bank import feature_matrix
    from .temporal_research_features import prefix_innovation
    matrix = feature_matrix(np.asarray(logp, np.float64), np.asarray(entropy, np.float64))["matrix"][:, :4]
    innovation, _ = prefix_innovation(matrix[:, 0])
    augmented = np.column_stack((matrix, innovation))
    mean = augmented.mean(axis=0); scale = np.maximum(augmented.std(axis=0), 1e-8)
    raw = augmented.astype(np.float32).astype(np.float64)
    z = (raw - mean) / scale
    return (z - bocpd_mean(z, hazard=hazard)).mean(axis=1)


def chosen_stream(logp, ids, provided, surprisal) -> np.ndarray:
    """[T] per-token standardized excess surprisal (entropy-free chosen-token evidence)."""
    x, _, _ = token_calibration(logp, ids, provided, surprisal)
    return x[:, CHOSEN_COL].astype(float)


def answer_streams(logp, ids, provided, surprisal, *, bocpd: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """[T x 7] CT7 token streams and validity for one answer.

    `bocpd` may be supplied from the verbatim temporal source; otherwise it is recomputed from
    the five bank streams of this answer.
    """
    x5, v5 = bank_streams(logp, ids, provided, surprisal)
    b = bocpd_residual_from_bank(x5, v5) if bocpd is None else np.asarray(bocpd, float)
    c = chosen_stream(logp, ids, provided, surprisal)
    T = len(x5)
    if b.shape != (T,) or c.shape != (T,):
        raise ValueError("stream length mismatch")
    x = np.column_stack([x5, b, c]); valid = np.column_stack([v5, np.ones((T, 2), bool)])
    if not np.isfinite(x[valid]).all():
        raise ValueError("nonfinite token stream")
    return x, valid


def step0_token_mask(spans: np.ndarray, n_tokens: int) -> np.ndarray:
    """True for the tokens of the answer's first official step."""
    spans = np.asarray(spans, int); m = np.zeros(int(n_tokens), bool)
    if len(spans):
        a, b = spans[0]; m[a:b] = True
    return m


def despike_step0(x: np.ndarray, valid: np.ndarray, spans: np.ndarray, cols) -> np.ndarray:
    """Token analogue of CT7's step-0 rule: for the given columns, the tokens of step 0 are
    replaced by the mean of the column over the valid tokens of the other steps (answers with
    one step are left unchanged)."""
    x = np.array(x, float, copy=True); spans = np.asarray(spans, int)
    if len(spans) < 2:
        return x
    m0 = step0_token_mask(spans, len(x))
    for j in np.atleast_1d(cols):
        rest = (~m0) & valid[:, j]
        if rest.any():
            x[m0, j] = x[rest, j].mean()
    return x


def masked_answer_local(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Within-answer z-score per column over its valid tokens; invalid or constant -> 0."""
    x = np.asarray(x, float); v = np.asarray(valid, bool); z = np.zeros_like(x)
    for j in range(x.shape[1]):
        col = x[v[:, j], j]
        if len(col) and col.std() > 1e-8:
            z[v[:, j], j] = (col - col.mean()) / col.std()
    return z


def masked_step_top10(values: np.ndarray, valid: np.ndarray, spans: np.ndarray, k: int = 10) -> np.ndarray:
    """Top-k mean over the valid tokens of each step (NaN for a step with no valid token)."""
    return step_topk_and_calibrated(np.asarray(values, float), np.asarray(valid, bool), np.asarray(spans, int), k=k)[0]


def synthetic_row(rng: np.random.Generator, n_tokens: int, n_steps: int, K: int = 50, V: int = 300) -> dict:
    """One synthetic telemetry row with a top-50 payload (H0: provided tokens sampled from the
    model's own distribution), official step spans and the fields the builders read."""
    temps = np.exp(rng.uniform(np.log(.1), np.log(4.0), n_tokens))
    logits = rng.standard_normal((n_tokens, V)) * 3 / temps[:, None]
    logits -= logits.max(axis=1, keepdims=True)
    full = np.exp(logits); full /= full.sum(axis=1, keepdims=True)
    order = np.argsort(-full, axis=1)[:, :K]; top = np.take_along_axis(full, order, axis=1)
    qk = top / top.sum(axis=1, keepdims=True); c = qk.cumsum(axis=1)
    pick = (c < rng.random(n_tokens)[:, None]).sum(axis=1).clip(0, K - 1)
    provided = order[np.arange(n_tokens), pick]
    surprisal = -np.log(full[np.arange(n_tokens), provided])
    cuts = np.sort(rng.choice(np.arange(1, n_tokens), n_steps - 1, replace=False)) if n_steps > 1 else np.array([], int)
    starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [n_tokens]])
    return {"top_k_logprobs": {"logprobs": np.log(top), "ids": order}, "gen_token_ids": provided,
            "token_spilled_energies": surprisal, "step_token_spans": np.column_stack([starts, ends]),
            "token_entropies": -(qk * np.log(qk)).sum(axis=1)}


def self_test(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    row = synthetic_row(rng, 120, 6)
    p = row["top_k_logprobs"]
    x, valid = answer_streams(p["logprobs"], p["ids"], row["gen_token_ids"], row["token_spilled_energies"])
    assert x.shape == (120, 7) and valid.shape == (120, 7)
    assert not valid[0, 4] and valid[1:, 4].all() and valid[:, :4].all() and valid[:, 5:].all()
    # the bank columns are exactly token_bank's columns
    xb, vb, _ = token_bank(p["logprobs"], p["ids"], row["gen_token_ids"], row["token_spilled_energies"])
    assert np.array_equal(x[:, :5], xb[:, BANK_COLS]) and np.array_equal(valid[:, :5], vb[:, BANK_COLS])
    # the masked Top10 equals the bank's step_bank readout for every step and stream
    from .digitfree_broad50 import step_bank
    sb, avail, _ = step_bank(p["logprobs"], p["ids"], row["gen_token_ids"], row["token_spilled_energies"], row["step_token_spans"])
    for j, col in enumerate(BANK_COLS):
        t = masked_step_top10(x[:, j], valid[:, j], row["step_token_spans"])
        ok = avail[:, col]
        assert np.allclose(t[ok], sb[ok, col]) and np.isnan(t[~ok]).all()
    # despike: step-0 tokens of the chosen column equal the other steps' mean; other columns untouched
    d = despike_step0(x, valid, row["step_token_spans"], [6])
    m0 = step0_token_mask(row["step_token_spans"], 120)
    assert np.allclose(d[m0, 6], x[~m0, 6].mean()) and np.array_equal(d[:, :6], x[:, :6])
    # answer-local standardization ignores invalid tokens and leaves them at zero
    z = masked_answer_local(x, valid)
    assert z[0, 4] == 0 and abs(z[1:, 4].mean()) < 1e-9 and abs(z[1:, 4].std() - 1) < 1e-9
    # BOCPD residual: the first token's residual equals its z (prior mean zero)
    b = bocpd_residual_from_bank(x[:, :5], valid[:, :5])
    z5 = masked_answer_local(x[:, :5], valid[:, :5])
    assert np.isclose(b[0], z5[0].mean())
    return {"tokens": 120, "steps": 6}


if __name__ == "__main__":
    print("self_test:", self_test())
