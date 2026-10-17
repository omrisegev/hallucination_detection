#!/usr/bin/env python
"""Evaluate the per-answer adaptive readout against fixed-K baselines.

C1 architecture throughout: token-level L-SML fusion (fitted once on donor folds, and
independent of the readout), then the step readout. Compared against:

  K=10        what we use today; must replay 35.92
  K=20 / K=40 the best fixed widths the grid found (LABEL-USING choices)
  adaptive    per-answer K* chosen by the label-free SNR criterion
  oracle-K    per-answer K chosen to maximise SLA -- a LABEL-USING ceiling on how much
              per-answer adaptation could ever be worth. Not a candidate.

The test that decides whether this is intelligent or merely another parameter: does K*,
chosen with no labels, come out LARGER on the long subsets? If the algorithm recovers
the length dependence from the data, that is the mechanism working. If K* is flat across
subsets, the adaptation is inert whatever its accuracy.

Paired source-group bootstrap intervals against K=10 and against the best fixed K.
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
OUT = RES / "ADAPTIVE_READOUT.json"
SUBS = ("gsm8k", "math", "olympiadbench", "omnimath")
SEED = 20260918
DRAWS = 10_000

_b = importlib.util.spec_from_file_location("stage_b", Path(__file__).with_name("stage_b_2x2_v1.py"))
SB = importlib.util.module_from_spec(_b)
_b.loader.exec_module(SB)

from spectral_utils.adaptive_step_readout_v1 import (  # noqa: E402
    adaptive_step_scores, ladder_for, step_scores_over_ladder,
)
from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402


def astd(v, offsets):
    return masked_answer_standardize(v[:, None], np.isfinite(v)[:, None], offsets)[:, 0]


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    groups = np.asarray([r["group_id"] for r in records], str)
    sub = np.array([c[3:-3] if c.startswith("pb_") else "prm" for c in cells])
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

    print("[fit] token-level L-SML, once", flush=True)
    fused = [None] * n
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        std = SB.fit_token_standardizer((mats[i] for i in train), cap=SB.TOKEN_CAP)
        w, _ = SB.fit_l_sml_weights((mats[i] for i in train), std)
        for i in np.flatnonzero(folds == fold):
            fused[i], _ = SB.fuse_token_matrix(mats[i], std, weights=w)
        print(f"  fold {fold} done", flush=True)

    total_steps = int(offsets[-1])
    arms = {f"K={k}": np.zeros(total_steps) for k in (10, 20, 40)}
    arms["adaptive"] = np.zeros(total_steps)
    arms["oracle_K"] = np.zeros(total_steps)
    chosen_k = np.zeros(n, int)
    oracle_k = np.zeros(n, int)

    rng = np.random.default_rng(SEED)
    print("[readout] fixed, adaptive and oracle", flush=True)
    for i in range(n):
        a, b = offsets[i], offsets[i + 1]
        sp = spans[i]
        lengths = sp[:, 1] - sp[:, 0]
        lad = ladder_for(int(lengths.max()) if len(lengths) else 1)
        grid = step_scores_over_ladder(fused[i], sp, lad)
        for k in (10, 20, 40):
            arms[f"K={k}"][a:b] = grid[:, int(np.argmin(np.abs(lad - k)))]
        scores, kstar = adaptive_step_scores(fused[i], sp, rng)
        arms["adaptive"][a:b] = scores
        chosen_k[i] = kstar
        # LABEL-USING ceiling: the K whose argmax lands on the true error, if any does
        if target[i] >= 0 and len(sp) > 1:
            hits = np.flatnonzero(np.argmax(grid, axis=0) == target[i])
            j = hits[0] if len(hits) else 0
        else:
            j = 0
        oracle_k[i] = int(lad[j])
        arms["oracle_K"][a:b] = grid[:, j]
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    scored = {k: astd(v, offsets) for k, v in arms.items()}
    peaks = {k: SB.peaks_of(v, offsets) for k, v in scored.items()}

    report = {"schema": "token-probability-fusion-v1-adaptive-readout",
              "development_only": True, "arms": {}, "chosen_k": {}}
    for k, p in peaks.items():
        per = SB.sla_gate_free(p, target, cells)
        report["arms"][k] = {
            "per_cell": {c: v["sla"] for c, v in per.items()},
            "mean_sla": float(np.mean([v["sla"] for v in per.values()])),
            "per_subset": {s: float(np.mean([v["sla"] for c, v in per.items()
                                             if c[3:-3] == s])) for s in SUBS}}
    for s in SUBS:
        m = (sub == s) & pb
        report["chosen_k"][s] = {
            "mean_K_star": float(chosen_k[m].mean()),
            "median_K_star": float(np.median(chosen_k[m])),
            "mean_tokens_per_step": float(np.mean(
                [(step_spans[offsets[i]:offsets[i + 1], 1]
                  - step_spans[offsets[i]:offsets[i + 1], 0]).mean()
                 for i in np.flatnonzero(m)])),
            "mean_oracle_K": float(oracle_k[m & (target >= 0)].mean())}

    rng2 = np.random.default_rng(SEED + 7)
    acc = {k: [] for k in arms}
    for idx in SB._group_draws(groups, pb & (target >= 0), rng2, DRAWS):
        t, c = target[idx], cells[idx]
        for k, p in peaks.items():
            per = SB.sla_gate_free(p[idx], t, c)
            acc[k].append(float(np.mean([v["sla"] for v in per.values()])))
    draws = {k: np.asarray(v) for k, v in acc.items()}
    report["intervals"] = {
        f"{k} - K=10": SB.interval(draws[k] - draws["K=10"]) for k in arms if k != "K=10"}
    report["intervals"]["adaptive - K=20"] = SB.interval(draws["adaptive"] - draws["K=20"])
    report["intervals"]["adaptive - K=40"] = SB.interval(draws["adaptive"] - draws["K=40"])

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")
    np.savez_compressed(RES / "ADAPTIVE_READOUT_SCORES.npz", chosen_k=chosen_k, **scored)

    print()
    print("=" * 96)
    print("ADAPTIVE READOUT vs FIXED K  (C1 architecture, gate-free SLA)")
    print("=" * 96)
    print(f"{'arm':14s} {'mean':>7s} " + "".join(f"{s[:9]:>11s}" for s in SUBS) +
          f" {'vs K=10':>24s}")
    for k in ("K=10", "K=20", "K=40", "adaptive", "oracle_K"):
        a = report["arms"][k]
        iv = report["intervals"].get(f"{k} - K=10")
        tail = (f"{iv['point_pp']:+6.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]"
                f"{'*' if iv['excludes_zero'] else ' '}") if iv else "  (reference)"
        print(f"{k:14s} {100*a['mean_sla']:7.2f} " +
              "".join(f"{100*a['per_subset'][s]:11.2f}" for s in SUBS) + f" {tail:>24s}")
    print()
    for name in ("adaptive - K=20", "adaptive - K=40"):
        iv = report["intervals"][name]
        print(f"  {name:20s} {iv['point_pp']:+6.2f} pp "
              f"[{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]"
              f"{'  excludes zero' if iv['excludes_zero'] else '  includes zero'}")

    print()
    print("=" * 96)
    print("DID THE ALGORITHM DISCOVER THE LENGTH DEPENDENCE ON ITS OWN?")
    print("=" * 96)
    print(f"{'subset':16s} {'tok/step':>9s} {'mean K*':>9s} {'median K*':>10s} {'oracle K':>9s}")
    for s in SUBS:
        c = report["chosen_k"][s]
        print(f"{s:16s} {c['mean_tokens_per_step']:9.1f} {c['mean_K_star']:9.2f} "
              f"{c['median_K_star']:10.1f} {c['mean_oracle_K']:9.2f}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
