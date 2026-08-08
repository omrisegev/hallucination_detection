#!/usr/bin/env python3
"""Transparent per-token locator baselines for the GL-LIU external-family confirmation.

Six obvious rules, none fitted, none label-free-fusion-based. Each is paired with the
SAME detector risk (the frozen candidate's own `answer_dufs_liu_mixed` score) when scored
by `run.py`, so the comparison isolates locator quality — does a sophisticated Laplacian
locator beat an argmax-entropy rule when both start from an identical error-presence signal?
This is the check `docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md`
asks for ("transparent token baselines... reveal whether a complex locator beats obvious
rules") and neither GL-LIU v1 nor the factorial study ever ran it.

Each function returns one predicted step index per row (or `NO_ERROR` if undecidable),
matching `scripts.gl_liu_v1.run.token_to_step`'s output contract exactly, so its output can
be evaluated with the same `two_stage_localization.evaluate_two_stage`.
"""
from __future__ import annotations

import numpy as np

NO_ERROR = -1


def _token_to_step(index, spans):
    if index is None:
        return NO_ERROR
    for step, span in enumerate(spans):
        if span is not None and span[0] <= index < span[1]:
            return int(step)
    return NO_ERROR


def max_entropy_step(row) -> int:
    """The step containing the single highest-entropy token."""
    e = row.get("token_entropies")
    if not e:
        return NO_ERROR
    return _token_to_step(int(np.argmax(e)), row["step_token_spans"])


def min_token_prob_step(row) -> int:
    """The step containing the token the model was most surprised by (max spilled energy,
    i.e. max -log p == min p)."""
    s = row.get("token_spilled_energies")
    if not s:
        return NO_ERROR
    return _token_to_step(int(np.argmax(s)), row["step_token_spans"])


def entropy_cusum_argmax_step(row) -> int:
    """The step containing the token where |cumsum(H - mean(H))| peaks — the same
    unweighted CUSUM statistic `feature_utils.compute_cusum_residuals` already carries as
    `cusum_shift_idx`."""
    e = row.get("token_entropies")
    if not e:
        return NO_ERROR
    e = np.asarray(e, dtype=float)
    cusum = np.abs(np.cumsum(e - e.mean()))
    return _token_to_step(int(np.argmax(cusum)), row["step_token_spans"])


def change_point_step(row, min_segment=2) -> int:
    """Single change-point via exhaustive binary segmentation: the split token index `t`
    maximizing the segment-size-weighted mean gap
    `sqrt(t*(n-t)/n) * |mean(H[:t]) - mean(H[t:])|`. Distinct from the CUSUM baseline above
    (which is unweighted and has no segment-size normalization) — the standard single
    change-point test statistic, not a windowed/rolling quantity."""
    e = row.get("token_entropies")
    if not e or len(e) < 2 * min_segment:
        return NO_ERROR
    e = np.asarray(e, dtype=float)
    n = len(e)
    prefix = np.concatenate([[0.0], np.cumsum(e)])
    t = np.arange(min_segment, n - min_segment + 1)
    mean_left = prefix[t] / t
    mean_right = (prefix[n] - prefix[t]) / (n - t)
    weight = np.sqrt(t * (n - t) / n)
    stat = weight * np.abs(mean_left - mean_right)
    split = int(t[np.argmax(stat)])
    return _token_to_step(split, row["step_token_spans"])


def random_step(row, rng: np.random.Generator) -> int:
    spans = row["step_token_spans"]
    if not spans:
        return NO_ERROR
    return int(rng.integers(0, len(spans)))


def last_step(row) -> int:
    spans = row["step_token_spans"]
    if not spans:
        return NO_ERROR
    return int(len(spans) - 1)


def all_locators(rows, seed=0):
    """`dict[name -> np.ndarray[int]]`, matching `token_locators`'s output shape in
    `scripts.gl_liu_v1.run` so both can be scored by the same downstream code."""
    rng = np.random.default_rng(seed)
    return {
        "baseline_max_entropy": np.asarray([max_entropy_step(r) for r in rows], dtype=int),
        "baseline_min_token_prob": np.asarray([min_token_prob_step(r) for r in rows], dtype=int),
        "baseline_entropy_cusum": np.asarray([entropy_cusum_argmax_step(r) for r in rows], dtype=int),
        "baseline_change_point": np.asarray([change_point_step(r) for r in rows], dtype=int),
        "baseline_random": np.asarray([random_step(r, rng) for r in rows], dtype=int),
        "baseline_last_step": np.asarray([last_step(r) for r in rows], dtype=int),
    }


def smoke() -> None:
    rng = np.random.default_rng(0)
    n_steps, step_len = 10, 20
    n = n_steps * step_len
    spans = [(i * step_len, (i + 1) * step_len) for i in range(n_steps)]
    bad_step = 6

    flat = rng.normal(0.4, 0.05, n).clip(0.01, None)
    burst = flat.copy()
    burst[bad_step * step_len: (bad_step + 1) * step_len] += rng.normal(1.2, 0.1, step_len)
    row = {"token_entropies": burst.tolist(),
           "token_spilled_energies": (burst * 1.1).tolist(),
           "step_token_spans": spans}

    assert max_entropy_step(row) == bad_step
    assert min_token_prob_step(row) == bad_step
    assert entropy_cusum_argmax_step(row) in (bad_step, bad_step + 1, bad_step - 1)
    assert change_point_step(row) in (bad_step, bad_step + 1, bad_step - 1)
    assert last_step(row) == n_steps - 1
    r = np.random.default_rng(1)
    picks = {random_step(row, r) for _ in range(50)}
    assert len(picks) > 1, "random_step must vary across calls"
    assert picks <= set(range(n_steps))

    empty_row = {"token_entropies": [], "token_spilled_energies": [], "step_token_spans": []}
    assert max_entropy_step(empty_row) == NO_ERROR
    assert last_step(empty_row) == NO_ERROR

    locators = all_locators([row, row])
    assert set(locators) == {
        "baseline_max_entropy", "baseline_min_token_prob", "baseline_entropy_cusum",
        "baseline_change_point", "baseline_random", "baseline_last_step",
    }
    assert all(len(v) == 2 for v in locators.values())

    print("token_baselines.smoke: PASS (7 checks, planted-burst step %d)" % bad_step)


if __name__ == "__main__":
    smoke()
