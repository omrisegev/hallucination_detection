#!/usr/bin/env python
"""Can the step readout be calibrated without labels?

`Top-K mean` with a FIXED K is not scale-free: K=10 is the top 17.6% of a GSM8K step
and the top 10.1% of an Omni-MATH step, so the same name denotes a different statistic
on different subsets. A fixed QUANTILE is scale-free by construction. Step 420 tried to
subtract the length bias out of the fixed-count statistic and failed uniformly; this asks
the prior question instead -- what should the aggregation be, and **can we choose it
without looking at labels?**

The experiment is deliberately fit-free. Everything runs in CT7's architecture -- the
per-channel step readout, answer-standardized, then an equal-weight mean of the eleven
channels -- so there is no standardizer, no L-SML, no folds, and therefore no route by
which a label could leak. Only the readout changes.

Reported side by side:

  the ceiling      gate-free SLA per subset. **Label-using.** It is the reference the
                   label-free criteria are judged against, never a selection rule.

  criterion 1      split-half decision reproducibility. Each step's tokens are split
                   odd/even, the readout is applied to each half, and we ask whether the
                   two halves choose the same step. Measures whether the decision would
                   survive a different but equivalent sample of the same answer.
  criterion 2      decisiveness: the top1-top2 margin and the effective number of
                   contested steps. Cannot stand alone -- K=1 buys large margins without
                   buying accuracy -- so it is only read next to criterion 1.
  criterion 3      length decoupling: |corr(step value, log tokens in step)| within the
                   answer. Step 420 already showed that minimising this costs accuracy,
                   so it is expected to choose badly. Included because that prediction
                   should be tested rather than assumed.

**The question is not "which K is best" -- that is a label-using answer. It is whether
any label-free criterion PICKS the K that the ceiling shows is best.** If none does,
calibration by these statistics is closed and we say so.

Fairness note on criterion 1: a half-step has n/2 tokens, so applying a fixed K to it
would compute a different statistic than on the full step (K/n doubles), which would
penalise count rules for a reason that is an artefact of the split rather than a
property of the rule. Count rules therefore use a proportionally reduced K on each half,
and quantile rules keep q. Each half then computes the SAME statistic as the whole.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
OUT = RES / "READOUT_CALIBRATION.json"

SUBS = ("gsm8k", "math", "olympiadbench", "omnimath")
N_CH = 11

from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402


def variants() -> dict[str, dict]:
    """name -> {kind, param}. `count` is a fixed K; `quantile` a fixed fraction."""
    v = {f"K={k}": {"kind": "count", "param": k} for k in (1, 3, 5, 10, 20, 40, 80, 160)}
    v |= {"K=all (step mean)": {"kind": "count", "param": 10**9}}
    v |= {f"q={q}": {"kind": "quantile", "param": q} for q in (0.05, 0.1, 0.2, 0.4)}
    v |= {"K=max(5,0.1n)": {"kind": "hybrid", "param": (5, 0.1)},
          "K=max(5,0.2n)": {"kind": "hybrid", "param": (5, 0.2)}}
    return v


def k_for(spec: dict, n: int, scale: float = 1.0) -> int:
    """Effective K for a block of n tokens; `scale` shrinks a count rule on a half-step."""
    kind, p = spec["kind"], spec["param"]
    if kind == "count":
        k = max(1, int(round(p * scale)))
    elif kind == "quantile":
        k = max(1, math.ceil(p * n))
    else:
        base, q = p
        k = max(max(1, int(round(base * scale))), math.ceil(q * n))
    return min(max(k, 1), n)


def topk_mean(block: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k largest values of each column of `block` ([tokens, channels])."""
    n = len(block)
    if k >= n:
        return block.mean(axis=0)
    return np.partition(block, n - k, axis=0)[-k:].mean(axis=0)


def readout(matrix: np.ndarray, spans: np.ndarray, spec: dict) -> tuple[np.ndarray, ...]:
    """Return (full, even-half, odd-half) step readouts, each [steps, channels]."""
    out = [np.zeros((len(spans), matrix.shape[1])) for _ in range(3)]
    for s, (a, b) in enumerate(spans):
        block = matrix[a:b]
        n = len(block)
        if n == 0:
            continue
        out[0][s] = topk_mean(block, k_for(spec, n))
        for j, half in enumerate((block[0::2], block[1::2]), start=1):
            h = half if len(half) else block
            out[j][s] = topk_mean(h, k_for(spec, len(h), scale=len(h) / n))
    return tuple(out)


def fuse(step_matrix: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    std = masked_answer_standardize(step_matrix, np.isfinite(step_matrix), offsets)
    return std.mean(axis=1)


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    sub = np.array([c[3:-3] if c.startswith("pb_") else "prm" for c in cells])
    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(float)
        tok_off = np.asarray(z["token_offsets"], int)
        step_spans = np.asarray(z["step_spans"], int)
    n_ans = len(records)
    ntok = (step_spans[:, 1] - step_spans[:, 0]).astype(float)
    log_tok = np.log(np.maximum(ntok, 1.0))

    report = {"schema": "token-probability-fusion-v1-readout-calibration",
              "architecture": "CT7-style: per-channel step readout, answer-standardized, equal fusion",
              "fit_free": True, "variants": {}}

    for name, spec in variants().items():
        print(f"[readout] {name}", flush=True)
        full = np.zeros((int(offsets[-1]), N_CH))
        h1 = np.zeros_like(full)
        h2 = np.zeros_like(full)
        for i in range(n_ans):
            m = tokens[tok_off[i]:tok_off[i + 1], :N_CH]
            sp = step_spans[offsets[i]:offsets[i + 1]]
            a, b = offsets[i], offsets[i + 1]
            full[a:b], h1[a:b], h2[a:b] = readout(m, sp, spec)

        F = fuse(full, offsets)
        A = fuse(h1, offsets)
        B = fuse(h2, offsets)

        block: dict = {"per_subset": {}}
        for s in SUBS:
            idx = np.flatnonzero(sub == s)
            err = idx[target[idx] >= 0]
            hit, repro, marg, eff, corr = [], [], [], [], []
            for i in idx:
                a, b = offsets[i], offsets[i + 1]
                v = F[a:b]
                if b - a < 2:
                    continue
                repro.append(int(np.argmax(A[a:b]) == np.argmax(B[a:b])))
                o = np.sort(v)[::-1]
                marg.append(float(o[0] - o[1]))
                p = np.exp(v - v.max())
                p /= p.sum()
                eff.append(float(np.exp(-(p * np.log(p + 1e-12)).sum())))
                y = log_tok[a:b]
                if v.std() > 0 and y.std() > 0:
                    corr.append(float(np.corrcoef(v, y)[0, 1]))
            for i in err:
                a, b = offsets[i], offsets[i + 1]
                hit.append(int(np.argmax(F[a:b]) == target[i]))
            block["per_subset"][s] = {
                "sla": float(np.mean(hit)),                       # LABEL-USING
                "reproducibility": float(np.mean(repro)),          # criterion 1
                "margin": float(np.mean(marg)),                    # criterion 2
                "effective_contested": float(np.mean(eff)),        # criterion 2
                "abs_corr_log_tokens": float(abs(np.nanmean(corr))),  # criterion 3
            }
        for key in ("sla", "reproducibility", "margin", "effective_contested",
                    "abs_corr_log_tokens"):
            block[key] = float(np.mean([block["per_subset"][s][key] for s in SUBS]))
        report["variants"][name] = block

    # ------------- does any label-free criterion pick the ceiling's choice? -------------
    names = list(report["variants"])
    def best(metric, lo=False, subset=None):
        f = (lambda n: report["variants"][n]["per_subset"][subset][metric]) if subset else \
            (lambda n: report["variants"][n][metric])
        return min(names, key=f) if lo else max(names, key=f)

    ceiling = best("sla")
    picks = {"criterion1_reproducibility": best("reproducibility"),
             "criterion2_margin": best("margin"),
             "criterion2_fewest_contested": best("effective_contested", lo=True),
             "criterion3_length_decoupled": best("abs_corr_log_tokens", lo=True)}
    report["selection"] = {
        "label_using_best": ceiling,
        "label_free_picks": picks,
        "cost_of_each_pick_pp": {k: 100 * (report["variants"][ceiling]["sla"]
                                           - report["variants"][v]["sla"])
                                 for k, v in picks.items()},
        "per_subset_ceiling": {s: best("sla", subset=s) for s in SUBS},
        "per_subset_criterion1": {s: best("reproducibility", subset=s) for s in SUBS},
    }

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    # ------------------------------------------------------------------ console
    print()
    print("=" * 108)
    print("READOUT GRID -- the ceiling is LABEL-USING; the three criteria are label-free")
    print("=" * 108)
    print(f"{'readout':16s} {'SLA mean':>9s} {'gsm8k':>7s} {'math':>7s} {'olymp':>7s} {'omni':>7s}"
          f" | {'repro':>7s} {'margin':>7s} {'contest':>8s} {'|corr|':>7s}")
    for n in names:
        b = report["variants"][n]
        ps = b["per_subset"]
        star = " <<" if n == ceiling else ""
        print(f"{n:16s} {100*b['sla']:9.2f} " +
              " ".join(f"{100*ps[s]['sla']:7.2f}" for s in SUBS) +
              f" | {100*b['reproducibility']:7.2f} {b['margin']:7.3f} "
              f"{b['effective_contested']:8.2f} {b['abs_corr_log_tokens']:7.3f}{star}")

    print()
    print("=" * 108)
    print("DOES A LABEL-FREE CRITERION FIND THE CEILING'S CHOICE?")
    print("=" * 108)
    print(f"  label-using best readout : {ceiling}  ({100*report['variants'][ceiling]['sla']:.2f})")
    for k, v in picks.items():
        cost = report["selection"]["cost_of_each_pick_pp"][k]
        verdict = "MATCHES" if v == ceiling else f"costs {cost:.2f} pp"
        print(f"  {k:32s} picks {v:16s} -> {100*report['variants'][v]['sla']:.2f}  {verdict}")
    print()
    print("  per subset, ceiling vs criterion 1:")
    for s in SUBS:
        print(f"    {s:16s} ceiling {report['selection']['per_subset_ceiling'][s]:16s}"
              f" criterion1 {report['selection']['per_subset_criterion1'][s]}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
