#!/usr/bin/env python
"""Cache the published C1 arm's fused series at TOKEN granularity.

Stage B saved C1's STEP readout but not the token-level fusion it was built from, and a
sequential readout needs the token series: the step series is ~8 points per answer, which
is far too short for a change-point detector to have anything to work with, while the
token series is ~713.

This is the same code path as `stage_b_2x2_v1.run_cell` for the C1 cell -- same folds,
same standardizer, same L-SML fit -- and it re-derives C1's step readout from the token
series it caches and checks it against the frozen one. If that check passes, the cached
token series IS the arm, not an approximation of it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
OUT = RES / "C1_TOKEN_SERIES.npz"
TOKEN_CAP = 60_000

from spectral_utils.claude_feature_bank_v1 import (  # noqa: E402
    fit_l_sml_weights, fit_token_standardizer, fuse_token_matrix,
)


def step_top10(values: np.ndarray, spans: np.ndarray, k: int = 10) -> np.ndarray:
    out = np.empty(len(spans))
    for i, (a, b) in enumerate(spans):
        seg = values[a:b]
        kk = min(k, len(seg))
        out[i] = np.partition(seg, len(seg) - kk)[-kk:].mean()
    return out


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)

    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(float)
        tok_off = np.asarray(z["token_offsets"], int)
        step_spans = np.asarray(z["step_spans"], int)

    n = len(records)
    mats = [tokens[tok_off[i]:tok_off[i + 1]] for i in range(n)]
    spans = [step_spans[offsets[i]:offsets[i + 1]] for i in range(n)]

    total_tokens = int(tok_off[-1])
    token_l_sml = np.full(total_tokens, np.nan, dtype=np.float32)
    token_equal = np.full(total_tokens, np.nan, dtype=np.float32)
    step_l_sml = np.full(int(offsets[-1]), np.nan)

    started = time.time()
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        std = fit_token_standardizer((mats[i] for i in train), cap=TOKEN_CAP)
        weights, _ = fit_l_sml_weights((mats[i] for i in train), std)
        for i in test:
            fused, equal = fuse_token_matrix(mats[i], std, weights=weights)
            ta, tb = int(tok_off[i]), int(tok_off[i + 1])
            token_l_sml[ta:tb] = fused
            token_equal[ta:tb] = equal
            step_l_sml[offsets[i]:offsets[i + 1]] = step_top10(fused, spans[i])
        print(f"[fold] {fold}: {len(test)} answers  ({time.time() - started:.0f}s)", flush=True)

    if not (np.isfinite(token_l_sml).all() and np.isfinite(token_equal).all()):
        raise ValueError("non-finite token fusion")

    with np.load(RES / "STAGE_B_SCORES.npz", allow_pickle=False) as z:
        frozen = np.asarray(z["C1_pooled_before__l_sml"], float)
    gap = float(np.max(np.abs(step_l_sml - frozen)))
    print(f"\nC1 step readout re-derived from the cached token series: max |diff| = {gap:.3e}")
    if gap > 1e-9:
        raise SystemExit("the cached token series does not reproduce the frozen C1 step arm")

    np.savez(OUT, token_l_sml=token_l_sml, token_equal=token_equal,
             token_offsets=tok_off, step_replay_max_abs_diff=np.asarray(gap))
    print(f"written: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, "
          f"{(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
