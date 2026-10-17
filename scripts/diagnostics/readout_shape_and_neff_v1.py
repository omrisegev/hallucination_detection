#!/usr/bin/env python
"""What transformation separates the cells we fail on, and can it set K?

Two gaps in the earlier readout work, both raised by Omri.

**The curve was never actually sampled.** The grid went 1, 3, 5, 10, 20, 40, 80, 160 --
between 10 and 40 only K=20 was ever measured, so "there is no maximum between them" was
never established, it was never looked for. Sampling every integer K costs nothing: one
descending sort per step yields the running mean, and therefore the whole ladder at once.

**And searching a grid is not engineering.** The better question is what distributional
property distinguishes the cells we do badly on from the ones we do well on, and whether
that property *derives* K rather than a search finding it.

The candidate here is not a guess. Fixed quantiles (K proportional to n) failed badly and
fixed counts partly worked, which says the truth is between the two -- and the quantity
that sits between them is the **effective number of independent tokens**. If tokens
inside a step are autocorrelated, the variance of a top-K mean is governed by

    n_eff = n / (1 + 2 * sum_k rho_k)

rather than by the raw token count. n_eff is label-free, measurable per cell, and is
exactly the thing a fixed count and a fixed fraction each get wrong in opposite
directions.

The target is **per cell**, not per answer, and deliberately so: the spectral screen
showed the per-answer room between two sensible widths is 0.70 pp with every descriptor
at chance, while the differences between cells are large and cell identity is available
at scoring time.

Reported: the full K curve per cell (label-using, the reference), and per-cell
distributional statistics (label-free). The question is whether any of the latter tracks
the argmax of the former across the eight cells.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
OUT = RES / "READOUT_SHAPE_AND_NEFF.json"
MAX_K = 64
MAX_LAG = 40

_b = importlib.util.spec_from_file_location("stage_b", Path(__file__).with_name("stage_b_2x2_v1.py"))
SB = importlib.util.module_from_spec(_b)
_b.loader.exec_module(SB)

from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402


def running_topk(block: np.ndarray, kmax: int) -> np.ndarray:
    """Top-K mean for K = 1..kmax (clipped at len(block)); one sort gives them all."""
    n = len(block)
    order = np.sort(block)[::-1]
    run = np.cumsum(order) / np.arange(1, n + 1)
    idx = np.minimum(np.arange(1, kmax + 1), n) - 1
    return run[idx]


def effective_n(series: np.ndarray, max_lag: int = MAX_LAG) -> float:
    """n / (1 + 2 * sum of positive autocorrelations), truncated at the first negative."""
    x = np.asarray(series, float)
    n = len(x)
    if n < 4 or x.std() < 1e-12:
        return float(n)
    x = x - x.mean()
    denom = float((x * x).sum())
    total = 0.0
    for lag in range(1, min(max_lag, n - 1) + 1):
        rho = float((x[:-lag] * x[lag:]).sum() / denom)
        if rho <= 0:                      # initial-positive-sequence truncation
            break
        total += rho
    return float(n / (1.0 + 2.0 * total))


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)
    pb = np.char.startswith(cells, "pb_")
    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(float)
        tok_off = np.asarray(z["token_offsets"], int)
        step_spans = np.asarray(z["step_spans"], int)
    n = len(records)
    spans = [step_spans[offsets[i]:offsets[i + 1]] for i in range(n)]
    mats = [tokens[tok_off[i]:tok_off[i + 1]] for i in range(n)]

    print("[fit] token-level L-SML on donor folds, once", flush=True)
    fused = [None] * n
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        std = SB.fit_token_standardizer((mats[i] for i in train), cap=SB.TOKEN_CAP)
        w, _ = SB.fit_l_sml_weights((mats[i] for i in train), std)
        for i in np.flatnonzero(folds == fold):
            fused[i], _ = SB.fuse_token_matrix(mats[i], std, weights=w)

    print("[sweep] every integer K from 1 to 64", flush=True)
    grid = np.zeros((int(offsets[-1]), MAX_K))
    for i in range(n):
        a, b = offsets[i], offsets[i + 1]
        for s, (lo, hi) in enumerate(spans[i]):
            grid[a + s] = running_topk(fused[i][lo:hi], MAX_K)
    for k in range(MAX_K):
        grid[:, k] = masked_answer_standardize(
            grid[:, k][:, None], np.isfinite(grid[:, k])[:, None], offsets)[:, 0]

    cell_names = sorted(set(cells[pb]))
    report = {"schema": "token-probability-fusion-v1-readout-shape-and-neff",
              "curve": {}, "stats": {}}

    for c in cell_names:
        idx = np.flatnonzero((cells == c) & (target >= 0))
        curve = []
        for k in range(MAX_K):
            hit = [int(np.argmax(grid[offsets[i]:offsets[i + 1], k]) == target[i]) for i in idx]
            curve.append(float(np.mean(hit)))
        report["curve"][c] = curve

        # label-free per-cell description of the fused token series inside steps
        all_idx = np.flatnonzero(cells == c)
        neff, raw, skew, kurt, ratio = [], [], [], [], []
        for i in all_idx:
            z = fused[i]
            for lo, hi in spans[i]:
                seg = z[lo:hi]
                if len(seg) < 8 or seg.std() < 1e-12:
                    continue
                e = effective_n(seg)
                neff.append(e)
                raw.append(float(len(seg)))
                ratio.append(e / len(seg))
                d = (seg - seg.mean()) / seg.std()
                skew.append(float((d ** 3).mean()))
                kurt.append(float((d ** 4).mean() - 3.0))
        report["stats"][c] = {
            "tokens_per_step": float(np.mean(raw)),
            "n_eff": float(np.mean(neff)),
            "n_eff_ratio": float(np.mean(ratio)),
            "skewness": float(np.mean(skew)),
            "excess_kurtosis": float(np.mean(kurt)),
            "best_K": int(np.argmax(report["curve"][c]) + 1),
            "sla_at_best": float(max(report["curve"][c])),
            "sla_at_10": float(report["curve"][c][9]),
        }
        print(f"  {c}: best K={report['stats'][c]['best_K']}", flush=True)

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    print()
    print("=" * 96)
    print("1. THE CURVE, SAMPLED AT EVERY INTEGER K (label-using reference)")
    print("=" * 96)
    show = [1, 3, 5, 8, 10, 12, 15, 18, 20, 24, 28, 32, 40, 48, 64]
    print(f"{'cell':22s}" + "".join(f"{k:6d}" for k in show) + f"{'argmax':>8s}")
    for c in cell_names:
        cu = report["curve"][c]
        print(f"{c:22s}" + "".join(f"{100*cu[k-1]:6.1f}" for k in show) +
              f"{report['stats'][c]['best_K']:8d}")

    print()
    print("=" * 96)
    print("2. DOES A LABEL-FREE DISTRIBUTIONAL STATISTIC TRACK THE OPTIMAL K?")
    print("=" * 96)
    print(f"{'cell':22s} {'tok/step':>9s} {'n_eff':>8s} {'n_eff/n':>8s} {'skew':>7s} "
          f"{'kurt':>7s} | {'best K':>7s} {'K/n_eff':>8s}")
    for c in cell_names:
        s = report["stats"][c]
        print(f"{c:22s} {s['tokens_per_step']:9.1f} {s['n_eff']:8.2f} {s['n_eff_ratio']:8.3f} "
              f"{s['skewness']:7.2f} {s['excess_kurtosis']:7.2f} | {s['best_K']:7d} "
              f"{s['best_K']/s['n_eff']:8.2f}")

    best = np.array([report["stats"][c]["best_K"] for c in cell_names], float)
    print()
    for name in ("tokens_per_step", "n_eff", "n_eff_ratio", "skewness", "excess_kurtosis"):
        v = np.array([report["stats"][c][name] for c in cell_names], float)
        if v.std() > 1e-12:
            r = float(np.corrcoef(v, best)[0, 1])
            rs = float(np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(best)))[0, 1])
            print(f"  corr(best K, {name:18s}) = {r:+.3f}   (rank {rs:+.3f})")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
