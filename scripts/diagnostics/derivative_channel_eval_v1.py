#!/usr/bin/env python
"""Length axis, attack 2: does a derivative step channel repair the long-chain deficit?

Attack 1 (subtracting the length prior out of the level statistic) is closed: its cost
was uniform across chain length. This attacks the same defect from the other side, with
a statistic that does not carry the prior to begin with -- see
`spectral_utils/derivative_step_channel_v1.py` for the mechanism and for which
constants are ours rather than borrowed.

Arms, all fitted on donor folds exactly as Stage B fits its cells:

  LEVEL      the eleven Top-10 channels, answer-standardized   (= Stage B's C4 bank)
  DRV        the eleven derivative channels, answer-standardized
  LEVEL+DRV  all twenty-two together
  CT7        frozen, the standing comparator; not refitted

Three things are reported, all by subset, because the mean is what hid the asymmetry:

  1. gate-free SLA per cell, and the short/long interaction against CT7;
  2. whether DRV adds anything over LEVEL (redundancy between the two attacks);
  3. the within-answer correlation of each step score with log step length -- the
     diagnostic Step 420 used, and the one that decides whether the derivative really
     is length-free or merely differently length-biased. "Mean of the M largest rises
     in a step" is itself an order statistic, so this is an open question, not an
     assumption.
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
OUT = RES / "DERIVATIVE_CHANNEL_EVAL.json"

SHORT, LONG = ("gsm8k", "math"), ("olympiadbench", "omnimath")
BOOTSTRAP_DRAWS = 10_000
SEED = 20260918

from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402

_b = importlib.util.spec_from_file_location("stage_b", Path(__file__).with_name("stage_b_2x2_v1.py"))
SB = importlib.util.module_from_spec(_b)
_b.loader.exec_module(SB)


def subset_of(cell: str) -> str:
    return cell[3:-3]


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    groups = np.asarray([r["group_id"] for r in records], str)
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)
    pb = np.char.startswith(cells, "pb_")

    with np.load(RES / "DERIVATIVE_CHANNELS.npz", allow_pickle=False) as z:
        level, deriv = z["level"].copy(), z["derivative"].copy()
    with np.load(ROOT / "results" / "chosen_token_calibration_v1" / "CT7_DEV_SCORES.npz",
                 allow_pickle=False) as z:
        ct7 = z["step_scores"].copy()

    n = len(records)
    def per_answer(mat):
        return [mat[offsets[i]:offsets[i + 1]] for i in range(n)]

    banks = {
        "LEVEL": per_answer(level),
        "DRV": per_answer(deriv),
        "LEVEL+DRV": per_answer(np.column_stack([level, deriv])),
    }
    scores = {"CT7": ct7}
    report = {"schema": "token-probability-fusion-v1-derivative-channel",
              "development_only": True, "cells": {}}

    for name, mats in banks.items():
        print(f"[bank] {name}", flush=True)
        local = [SB.answer_local(m) for m in mats]
        res = SB.run_cell(local, None, folds, offsets)
        for rule in ("equal", "l_sml"):
            scores[f"{name}__{rule}"] = res[rule]
        report["cells"][name] = {
            "weight_ipr_mean": float(np.mean([f["weight_ipr"] for f in res["fits"]])),
            "K_by_fold": [f["K"] for f in res["fits"]],
        }

    peaks = {k: SB.peaks_of(v, offsets) for k, v in scores.items()}
    report["sla_gate_free"] = {k: SB.sla_gate_free(p, target, cells) for k, p in peaks.items()}
    report["mean_sla"] = {k: float(np.mean([v["sla"] for v in per.values()]))
                          for k, per in report["sla_gate_free"].items()}

    # ---- length-bias diagnostic: within-answer corr(step score, log step length) ----
    step_len = np.diff(offsets)  # steps per answer, not token count
    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        spans = np.asarray(z["step_spans"], int)
    tokens_per_step = (spans[:, 1] - spans[:, 0]).astype(float)
    log_len = np.log(np.maximum(tokens_per_step, 1.0))
    report["within_answer_corr_log_step_length"] = {}
    for k, s in scores.items():
        vals = []
        for a, b in zip(offsets[:-1], offsets[1:]):
            if b - a > 2 and np.std(s[a:b]) > 0 and np.std(log_len[a:b]) > 0:
                vals.append(np.corrcoef(s[a:b], log_len[a:b])[0, 1])
        report["within_answer_corr_log_step_length"][k] = float(np.nanmean(vals))

    # ---- short/long interaction against CT7 and against LEVEL ----------------
    rng = np.random.default_rng(SEED)
    acc = {k: [] for k in scores}
    for i in SB._group_draws(groups, pb & (target >= 0), rng, BOOTSTRAP_DRAWS):
        t, c = target[i], cells[i]
        for k, p in peaks.items():
            per = {kk: v["sla"] for kk, v in SB.sla_gate_free(p[i], t, c).items()}
            acc[k].append((np.mean([v for kk, v in per.items() if subset_of(kk) in SHORT]),
                           np.mean([v for kk, v in per.items() if subset_of(kk) in LONG])))
    draws = {k: np.asarray(v) for k, v in acc.items()}

    report["interaction"] = {}
    for ref in ("CT7", "LEVEL__equal"):
        for k in scores:
            if k == ref:
                continue
            ds = draws[k][:, 0] - draws[ref][:, 0]
            dl = draws[k][:, 1] - draws[ref][:, 1]
            report["interaction"][f"{k} - {ref}"] = {
                "short_chain": SB.interval(ds), "long_chain": SB.interval(dl),
                "long_minus_short": SB.interval(dl - ds)}

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")
    np.savez_compressed(RES / "DERIVATIVE_ARM_SCORES.npz", **scores)

    # ------------------------------------------------------------------ console
    order = ["CT7", "LEVEL__equal", "LEVEL__l_sml", "DRV__equal", "DRV__l_sml",
             "LEVEL+DRV__equal", "LEVEL+DRV__l_sml"]
    print()
    print("=" * 104)
    print("DERIVATIVE CHANNEL -- gate-free SLA by subset")
    print("=" * 104)
    print(f"{'cell':24s}" + "".join(f"{a.replace('__','/'):>17s}" for a in order))
    for cell in sorted(report["sla_gate_free"]["CT7"]):
        print(f"{cell:24s}" + "".join(
            f"{100*report['sla_gate_free'][a][cell]['sla']:17.2f}" for a in order))
    for lab, grp in (("SHORT", SHORT), ("LONG", LONG)):
        print(f"{lab:24s}" + "".join(f"{100*np.mean([v['sla'] for k,v in report['sla_gate_free'][a].items() if subset_of(k) in grp]):17.2f}" for a in order))
    print(f"{'MEAN':24s}" + "".join(f"{100*report['mean_sla'][a]:17.2f}" for a in order))
    print()
    print("within-answer corr with log tokens-per-step (Step 420's length diagnostic):")
    for a in order:
        print(f"  {a:22s} {report['within_answer_corr_log_step_length'][a]:+.3f}")
    print()
    print("=" * 104)
    print("SHORT / LONG INTERACTION")
    print("=" * 104)
    for k, v in report["interaction"].items():
        f = lambda x: "*" if x["excludes_zero"] else " "  # noqa: E731
        print(f"{k:34s} short {v['short_chain']['point_pp']:+6.2f}{f(v['short_chain'])} "
              f"long {v['long_chain']['point_pp']:+6.2f}{f(v['long_chain'])} "
              f"long-short {v['long_minus_short']['point_pp']:+6.2f} "
              f"[{v['long_minus_short']['ci95_pp'][0]:+6.2f},{v['long_minus_short']['ci95_pp'][1]:+6.2f}]"
              f"{f(v['long_minus_short'])}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
