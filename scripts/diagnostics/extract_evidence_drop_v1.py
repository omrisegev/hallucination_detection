#!/usr/bin/env python
"""Rebuild the Mind-the-Gap evidence series exactly, at token granularity.

`Ê_i = −H(P̃)`, the negative entropy of the **renormalized top-20** distribution, then
EMA span 5, then the first difference. Every constant here is theirs — top-20, span 5,
M = 5 — used unchanged so that what follows tests their mechanism and not my constants.
The earlier derivative attempt in this worktree used EMA 16 and M = 3 over all eleven
bank channels and so never tested it.

Our bank's `q15_H1` is the same quantity at top-15, so the two must be highly but not
perfectly correlated; that correlation is computed here as the sanity check.

Saved, all aligned to the roster's own offsets:

  risk_token   [tokens]  −Δ of the EMA'd evidence. A *drop in evidence* is a *rise in
                         risk*, so this is oriented like every other channel in the bank.
  evidence     [tokens]  Ê itself, before smoothing, kept so the profile diagnostic can
                         separate "the level is informative" from "the change is".
  step_m5      [steps]   mean of the M = 5 most negative Δ inside the step
  step_worst   [steps]   the single most negative Δ inside the step

The token→step collapse is undefined in the paper (digest: "SLA's token→step aggregation
is undefined ... the dominant free parameter for Table 3 and is never given"), so both
collapses are OUR pre-registered choice and are labelled as such wherever they appear.

Reads the raw pickles from the MAIN checkout: `dataset_cache` in this worktree is
deliberately a set of LFS pointers.
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
OUT = ROOT / "results" / "token_probability_fusion_v1" / "EVIDENCE_DROP.npz"

TOP_K = 20          # theirs (Eq. 10), not our bank's 15
EMA_SPAN = 5        # theirs
WORST_M = 5         # theirs

from spectral_utils.derivative_step_channel_v1 import ema  # noqa: E402
import scripts.run_claude_feature_bank_v1 as runner  # noqa: E402


def evidence_series(row: dict, n_tokens: int) -> np.ndarray:
    """−H of the renormalized top-20, one value per token."""
    payload = row.get("top_k_logprobs") or row.get("top_k_logprobs_raw")
    if not isinstance(payload, dict):
        raise ValueError("row has no saved top-k log-probability payload")
    logprobs = np.asarray(payload["logprobs"], dtype=float)
    if logprobs.ndim != 2 or logprobs.shape[0] != n_tokens or logprobs.shape[1] < TOP_K:
        raise ValueError(f"top-k logprobs {logprobs.shape}, expected [{n_tokens}, >={TOP_K}]")
    p = np.exp(logprobs[:, :TOP_K])
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    return -entropy


def collapse(delta: np.ndarray, spans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both token→step collapses of Δ, returned risk-oriented (higher = more suspect)."""
    risk = -delta
    m5 = np.zeros(len(spans))
    worst = np.zeros(len(spans))
    for s, (a, b) in enumerate(spans):
        seg = risk[a:b]
        if not len(seg):
            continue
        k = min(WORST_M, len(seg))
        m5[s] = np.partition(seg, len(seg) - k)[-k:].mean()
        worst[s] = seg.max()
    return m5, worst


def main() -> None:
    runner.ROOT = MAIN

    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], dtype=int)
    n = len(records)
    print(f"roster: {n} answers, {offsets[-1]} steps", flush=True)

    token_counts = np.asarray([int(r["tokens"]) for r in records], dtype=np.int64)
    token_offsets = np.concatenate([[0], np.cumsum(token_counts)])
    total_tokens = int(token_offsets[-1])

    risk_token = np.zeros(total_tokens, dtype=np.float32)
    evidence = np.zeros(total_tokens, dtype=np.float32)
    step_m5 = np.zeros(int(offsets[-1]), dtype=np.float32)
    step_worst = np.zeros(int(offsets[-1]), dtype=np.float32)

    # Sanity check against the bank's own top-15 entropy, on a bounded sample of tokens.
    sample_ours, sample_theirs = [], []
    with np.load(ROOT / "results" / "token_probability_fusion_v1" / "TOKEN_MATRICES.npz",
                 allow_pickle=False, mmap_mode="r") as z:
        channels = list(np.asarray(z["channels"], dtype=str))
        h1_column = channels.index("q15_H1")
        bank_h1 = np.asarray(z["tokens"][:, h1_column], dtype=np.float32)

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
            ta, tb = int(token_offsets[i]), int(token_offsets[i + 1])
            a, b = int(offsets[i]), int(offsets[i + 1])
            spans = np.asarray(row["step_token_spans"], dtype=int)
            if spans.shape != (b - a, 2):
                raise ValueError(f"answer {i}: {spans.shape} spans for {b - a} steps")
            e = evidence_series(row, tb - ta)
            smoothed = ema(e, EMA_SPAN)
            delta = np.diff(smoothed, prepend=smoothed[:1])
            evidence[ta:tb] = e
            risk_token[ta:tb] = -delta
            step_m5[a:b], step_worst[a:b] = collapse(delta, spans)
            if i % 37 == 0:
                sample_ours.append(bank_h1[ta:tb])
                sample_theirs.append(-e)
        seen += len(indexes)
        del source
        gc.collect()
        rate = seen / (time.time() - started)
        print(f"[cell] {cell}: {len(indexes)} answers  ({seen}/{n}, {rate:.1f}/s, "
              f"eta {(n - seen) / rate / 60:.1f} min)", flush=True)

    for name, array in (("evidence", evidence), ("risk_token", risk_token),
                        ("step_m5", step_m5), ("step_worst", step_worst)):
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite {name}")

    ours = np.concatenate(sample_ours)
    theirs = np.concatenate(sample_theirs)
    correlation = float(np.corrcoef(ours, theirs)[0, 1])
    print(f"\nsanity: corr(bank q15_H1, their top-20 H) = {correlation:.6f} "
          f"over {len(ours)} tokens from {len(sample_ours)} answers")
    if not 0.90 < correlation < 0.999999:
        raise ValueError("top-20 entropy should be highly but not perfectly correlated "
                         f"with the bank's top-15 entropy; got {correlation}")

    np.savez(OUT, risk_token=risk_token, evidence=evidence, step_m5=step_m5,
             step_worst=step_worst, token_offsets=token_offsets,
             top_k=np.asarray(TOP_K), ema_span=np.asarray(EMA_SPAN),
             worst_m=np.asarray(WORST_M), h1_correlation=np.asarray(correlation))
    print(f"written: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, "
          f"{(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
