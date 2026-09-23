#!/usr/bin/env python
"""Extract the token bank once, and cache everything the length axis and Stage B need.

Writes two artefacts under `results/token_probability_fusion_v1/`:

  DERIVATIVE_CHANNELS.npz   LEVEL and DERIVATIVE step readouts, two [145597, 11] matrices
  TOKEN_MATRICES.npz        the risk-oriented token bank itself, [6968779, 11] float32,
                            plus the per-answer token offsets and step spans

Caching the token matrices costs 0.31 GB and buys the rest of the programme for free:
Stage B's two "fuse before the readout" cells need token-level standardisation and
fusion, which cannot be recovered from a step readout. Paying one 25-minute pass here
is cheaper than paying it again per experiment.

The raw pickles are read from the MAIN checkout: `dataset_cache` in this worktree is
deliberately a set of LFS pointers (see the worktree-smudge note), and the pickles are
identical either way.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
OUT = ROOT / "results" / "token_probability_fusion_v1" / "DERIVATIVE_CHANNELS.npz"
OUT_TOKENS = ROOT / "results" / "token_probability_fusion_v1" / "TOKEN_MATRICES.npz"

from spectral_utils.claude_feature_bank_v1 import FEATURE_NAMES, build_token_feature_matrix  # noqa: E402
from spectral_utils.derivative_step_channel_v1 import (  # noqa: E402
    EMA_WINDOW, WORST_M, derivative_step_readout, level_step_readout,
)
import scripts.run_claude_feature_bank_v1 as runner  # noqa: E402


def main() -> None:
    # The runner resolves dataset_cache against its own ROOT; point it at the real data.
    runner.ROOT = MAIN

    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], dtype=int)
    print(f"roster: {len(records)} answers, {offsets[-1]} steps", flush=True)

    total, n_ch = int(offsets[-1]), len(FEATURE_NAMES)
    level = np.zeros((total, n_ch), dtype=float)
    deriv = np.zeros((total, n_ch), dtype=float)

    token_counts = np.asarray([int(r["tokens"]) for r in records], dtype=np.int64)
    token_offsets = np.concatenate([[0], np.cumsum(token_counts)])
    tokens = np.zeros((int(token_offsets[-1]), n_ch), dtype=np.float32)
    all_spans = np.zeros((total, 2), dtype=np.int32)

    # Stream one source cell at a time and free it. Resolving every cell up front holds
    # all nine pickles -- several GB of raw top-50 logprob arrays -- and on a 16 GB box
    # shared with other sessions that pages, which costs far more than it saves.
    by_cell: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_cell.setdefault(str(record["cell"]), []).append(index)

    started, seen = time.time(), 0
    for cell, path, kind, dataset in runner.source_specs():
        indexes = by_cell.get(cell, [])
        if not indexes:
            continue
        source = runner.source_row_map(runner.load_pickle(path), kind=kind, dataset=dataset)
        for i in indexes:
            row = source[str(records[i]["row_id"])]
            a, b = int(offsets[i]), int(offsets[i + 1])
            spans = np.asarray(row["step_token_spans"], dtype=int)
            if spans.shape != (b - a, 2):
                raise ValueError(f"answer {i}: {spans.shape} spans for {b - a} steps")
            matrix = build_token_feature_matrix(row)
            ta, tb = int(token_offsets[i]), int(token_offsets[i + 1])
            if len(matrix) != tb - ta:
                raise ValueError(f"answer {i}: {len(matrix)} tokens, roster says {tb - ta}")
            tokens[ta:tb] = matrix.astype(np.float32)
            all_spans[a:b] = spans
            level[a:b] = level_step_readout(matrix, spans)
            deriv[a:b] = derivative_step_readout(matrix, spans)
        seen += len(indexes)
        del source
        gc.collect()
        rate = seen / (time.time() - started)
        print(f"[cell] {cell}: {len(indexes)} answers  ({seen}/{len(records)}, "
              f"{rate:.1f}/s, eta {(len(records) - seen) / rate / 60:.1f} min)", flush=True)

    if not (np.isfinite(level).all() and np.isfinite(deriv).all()):
        raise ValueError("non-finite step readout")
    if not np.isfinite(tokens).all():
        raise ValueError("non-finite token matrix")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, level=level, derivative=deriv,
        channels=np.asarray(FEATURE_NAMES, dtype=str),
        ema_window=np.asarray(EMA_WINDOW), worst_m=np.asarray(WORST_M),
    )
    # Uncompressed: this one is read repeatedly and compression would dominate load time.
    np.savez(OUT_TOKENS, tokens=tokens, token_offsets=token_offsets,
             step_spans=all_spans, channels=np.asarray(FEATURE_NAMES, dtype=str))
    print(f"written: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"written: {OUT_TOKENS}  ({OUT_TOKENS.stat().st_size / 1e6:.1f} MB, "
          f"{(time.time() - started) / 60:.1f} min total)")


if __name__ == "__main__":
    main()
