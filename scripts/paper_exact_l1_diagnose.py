#!/usr/bin/env python
"""
L1 provenance diagnosis: why our uPRM Eq. 6 reconstruction under-detects errors.

The L1 pilot (Qwen2.5-14B, 30 rows/subset) reproduced the paper's control well on GSM8K
(F1 0.640 vs their 0.498) and badly on the three harder subsets (0.198 / 0.124 / 0.172 vs
0.428 / 0.294 / 0.266), with `error_acc` 0.07-0.11 against `correct_acc` 0.87-0.92 — the
judge answers "clean" almost always.

This script diagnoses **provenance**, per handoff §6: "On an outcome that differs from the
paper, preserve the result and diagnose provenance — never tune toward the published number
on evaluation labels." It therefore reports the mechanism and changes nothing.

The mechanism to test
---------------------
Eq. 6 gives, for a T-step trajectory,

    S(j)   = log p-(j) + sum_{i<j} log p+(i)      for j = 1..T
    S(T+1) = sum_{i<=T} log p+(i)

so the clean candidate beats the last error candidate exactly when `p+(T) > 0.5`, and more
generally

    S(j) - S(T+1) = log p-(j) - sum_{i>=j} log p+(i)

Every `log p+` term is <= 0, so the second term is a non-negative bonus for early j. If the
model's renormalised marker distribution is saturated (`p+ ~ 0.99` everywhere), then
`log p+ ~ -0.01` and `log p-(j) ~ -4.6`, and S(T+1) wins by a wide margin regardless of
content. The prediction becomes "clean" almost always, which is exactly the observed
error_acc/correct_acc split.

If that is what the data shows, the deficit is in the **marker surface form**, which this
project declares as its own (the paper publishes no prompt and no code), not in the scoring
arithmetic. That is a reportable finding about our reconstruction, not a bug to tune away.

Usage:
    python scripts/paper_exact_l1_diagnose.py --run $SHARED/results/paper_exact/l1_uprm_judge_pilot
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spectral_utils.paper_exact import evaluator as EV        # noqa: E402
from spectral_utils.paper_exact.shards import read_shards      # noqa: E402


def recover_marker_logprobs(scores: dict):
    """Invert Eq. 6 to recover the per-step (log p+, log p-) the scores were built from.

    `scores` is {j: S(j)} with keys 1..T+1. From the recurrence,
        cum_pos(j) = sum_{i<j} log p+(i),  cum_pos(1) = 0,  cum_pos(T+1) = S(T+1)
        log p-(j)  = S(j) - cum_pos(j)
        log p+(j)  = cum_pos(j+1) - cum_pos(j)
    and cum_pos(j+1) is recoverable only for j<T from the next candidate, so the chain is
    rebuilt forward from S(T+1) backwards. Doing it this way means the diagnosis needs no
    extra forward passes and no re-run.
    """
    ks = sorted(int(k) for k in scores)
    if len(ks) < 2:
        return None
    T = len(ks) - 1
    S = {int(k): float(v) for k, v in scores.items()}
    # cum_pos(j) for j=1..T+1: cum_pos(1)=0; cum_pos(T+1)=S(T+1). Between them, log p+(i) is
    # not individually identified by the S values alone EXCEPT through the differences
    # S(j+1)-S(j) = log p-(j+1) - log p-(j) + log p+(j). One extra assumption is needed, so
    # instead of inventing one, report the identified quantities only.
    lp_neg_minus_cum = {j: S[j] for j in ks if j <= T}          # = log p-(j) + cum_pos(j)
    return {
        "T": T,
        "S": S,
        "S_clean": S[T + 1],
        # Identified without assumptions: the total positive mass and the winning margin.
        "total_log_p_pos": S[T + 1],
        "mean_log_p_pos": S[T + 1] / max(1, T),
        "best_error_S": max((S[j] for j in ks if j <= T), default=float("-inf")),
        "clean_margin": S[T + 1] - max((S[j] for j in ks if j <= T), default=float("-inf")),
        "argmax_is_clean": max(S, key=lambda j: S[j]) == T + 1,
        "_lp_neg_plus_cum": lp_neg_minus_cum,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for rec in read_shards(args.run, verify=False):
        if not rec.get("scores"):
            continue
        d = recover_marker_logprobs(rec["scores"])
        if d is None:
            continue
        rows.append({
            "subset": rec["subset"], "label": int(rec["label"]),
            "prediction": rec["prediction"], "n_steps": len(rec["steps"]),
            **{k: v for k, v in d.items() if not k.startswith("_")},
        })
    if not rows:
        sys.exit(f"no scored rows found under {args.run}")

    print(f"[l1-diag] {len(rows)} scored rows\n")
    print(f"{'subset':<16} {'n':>4} {'mean log p+':>12} {'p+ implied':>11} "
          f"{'clean win %':>12} {'mean margin':>12}")
    per_subset = {}
    for subset in sorted({r["subset"] for r in rows}):
        sel = [r for r in rows if r["subset"] == subset]
        mlp = float(np.mean([r["mean_log_p_pos"] for r in sel]))
        clean_pct = 100.0 * float(np.mean([r["argmax_is_clean"] for r in sel]))
        margin = float(np.mean([r["clean_margin"] for r in sel
                               if np.isfinite(r["clean_margin"])]))
        per_subset[subset] = {
            "n": len(sel), "mean_log_p_pos": mlp, "implied_p_pos": float(np.exp(mlp)),
            "pct_predicted_clean": clean_pct, "mean_clean_margin": margin,
            "pct_actually_clean": 100.0 * float(np.mean([r["label"] == EV.NO_ERROR
                                                         for r in sel])),
            "mean_n_steps": float(np.mean([r["n_steps"] for r in sel])),
        }
        print(f"{subset:<16} {len(sel):>4} {mlp:>12.4f} {np.exp(mlp):>11.4f} "
              f"{clean_pct:>11.1f}% {margin:>12.3f}")

    print(f"\n{'subset':<16} {'predicted clean':>16} {'actually clean':>15}  verdict")
    for subset, st in per_subset.items():
        gap = st["pct_predicted_clean"] - st["pct_actually_clean"]
        verdict = ("saturated toward '+' — the clean candidate wins on margin, not content"
                   if st["implied_p_pos"] > 0.95 and gap > 15 else
                   "marker distribution is informative here")
        print(f"{subset:<16} {st['pct_predicted_clean']:>15.1f}% "
              f"{st['pct_actually_clean']:>14.1f}%  {verdict}")

    report = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": args.run, "n_rows": len(rows), "per_subset": per_subset,
        "mechanism": "S(T+1) - S(T) = log p+(T) - log p-(T), so a marker distribution "
                     "saturated toward '+' makes the clean candidate win by margin rather "
                     "than by content. Every log p+ term is <= 0, so S(T+1) accumulates a "
                     "near-zero penalty while every error candidate pays log p-(j).",
        "conclusion": "If implied p+ is near 1 and the predicted-clean rate far exceeds the "
                      "actual clean rate, the deficit is in the MARKER SURFACE FORM (which "
                      "this project declares as its own — the paper publishes no prompt or "
                      "code), not in the Eq. 6 arithmetic. Report it as a property of our "
                      "reconstruction of their baseline. Do NOT retune the prompt against "
                      "ProcessBench labels.",
        "not_a_fix": "Changing the marker tokens, system message, or decision rule to raise "
                     "F1 on these rows would be tuning on evaluation labels. Any such change "
                     "must be pre-registered and evaluated on rows held out from this "
                     "diagnosis.",
    }
    out = args.out or os.path.join(args.run, "L1_DIAGNOSIS.json")
    with open(out + ".tmp", "w") as f:
        json.dump(report, f, indent=2, default=float)
    os.replace(out + ".tmp", out)
    print(f"\ndiagnosis -> {out}")


if __name__ == "__main__":
    main()
