#!/usr/bin/env python
"""Controls for the impulse finding that the handoff amendment now rests on.

The amendment (handoff section 8, 2026-09-19) reports two things from the profile of the
step score around the true first error:

  (i)  the error step is a sharp isolated impulse, neighbours at or below zero;
  (ii) the mean AFTER the error is 0.19-0.35 SD BELOW the mean before it, i.e. "the model
       becomes more confident after it errs".

(i) needs no control: a profile of +0.40 SD at offset 0 against ~0.0 at +/-1 is what it is,
and it is measured on answer-standardized scores so it cannot be a scale artefact.

(ii) does need one, and did not get one. Both quantities are computed at the SAME index k
in the SAME answer, so any general trend of the score with position inside an answer maps
directly onto "after minus before" without any post-error mechanism existing. The project
has already measured such a trend in this readout family: the step-LENGTH prior recorded
in `project_readout_length_prior_late_bias`. So the control is mandatory, and it is cheap:

  * POSITIONAL PROFILE: mean standardized score against relative position, over the same
    erroneous answers, ignoring the target entirely.
  * WITHIN-ANSWER POSITION CONTROL: for each answer compare after(k)-before(k) at the TRUE
    k against the mean of after(j)-before(j) over every admissible j in that same answer.
    This holds the answer, its length and the statistic fixed and varies only the index, so
    what survives is the part of (ii) that the true error location actually explains.
  * LOCAL-MAXIMUM NULL: the 52.7-57.1% local-maximum rate is quoted with no null. A random
    interior index of an i.i.d. series is a local maximum 1/3 of the time, so the number to
    report is the excess over the same answer's own admissible indices.

Two arms are profiled, not one: the published C1 token L-SML arm (the 35.92 anchor) and the
eleven-channel answer-standardized equal mean that the amendment actually used (34.95).
Nothing here is fitted and no arm is changed; this is a measurement of an existing claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
OUT = RES / "IMPULSE_SHAPE_CONTROL.json"

SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
OFFSETS = tuple(range(-3, 4))
SEED = 20260919


def subset_of(cell: str) -> str:
    """pb_<subset>_<model> -> <subset>. `cell[3:-3]` is the recorded bug: it maps both
    pb_gsm8k_q4 and pb_gsm8k_q8 to gsm8k only because both suffixes are three characters,
    which is true here but silently halves the population the moment a model id is not."""
    return cell.split("_", 1)[1].rsplit("_", 1)[0] if cell.startswith("pb_") else "prm"


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    subsets = np.asarray([subset_of(c) for c in cells], str)

    with np.load(RES / "STAGE_B_SCORES.npz", allow_pickle=False) as z:
        c1 = np.asarray(z["C1_pooled_before__l_sml"], float)
    with np.load(RES / "STEP_VIEWS_12CH.npz", allow_pickle=False) as z:
        views = np.asarray(z["views"], float)
    arms = {"C1_token_l_sml": c1, "eleven_channel_equal_mean": views[:, :11].mean(1)}
    # Their statistic, profiled by the same code path: if the evidence-drop series shows a
    # different shape around the true error than our level readout does, that is what would
    # justify a different decision rule for it. If it is also an isolated impulse, it will
    # not. Both of OUR token->step collapses are profiled, since the paper defines neither.
    drop = RES / "EVIDENCE_DROP.npz"
    if drop.exists():
        with np.load(drop, allow_pickle=False) as z:
            arms["evidence_drop_mean_of_5_worst"] = np.asarray(z["step_m5"], float)
            arms["evidence_drop_single_worst"] = np.asarray(z["step_worst"], float)

    # ---- anchor: C1 must still be the 35.92 arm before any shape claim is made --------
    sla = []
    for cell in sorted(set(cells[np.char.startswith(cells, "pb_")])):
        mask = (cells == cell) & (target >= 0)
        peaks = np.asarray([int(np.argmax(c1[offsets[i]:offsets[i + 1]]))
                            for i in np.flatnonzero(mask)])
        sla.append((peaks == target[mask]).mean())
    anchor = float(100 * np.mean(sla))
    print(f"anchor: C1 mean gate-free SLA = {anchor:.2f}  (must be 35.92)")
    if abs(anchor - 35.92) > 0.005:
        raise SystemExit(f"anchor failed: {anchor:.4f}")

    rng = np.random.default_rng(SEED)
    report: dict = {"schema": "impulse-shape-control-v1", "development_only": True,
                    "anchor_c1_mean_sla": anchor, "arms": {}}

    for arm, scores in arms.items():
        per_subset = {}
        for subset in SUBSETS:
            rows = np.flatnonzero((subsets == subset) & (target >= 0))
            profile = {o: [] for o in OFFSETS}
            positional = [[] for _ in range(10)]
            true_gap, answer_gap, gap_rows = [], [], 0
            true_max, answer_max, max_rows = [], [], 0
            for i in rows:
                a, b = int(offsets[i]), int(offsets[i + 1])
                v = scores[a:b]
                std = v.std()
                v = (v - v.mean()) / std if std > 1e-12 else np.zeros_like(v)
                k, n = int(target[i]), len(v)
                for o in OFFSETS:
                    if 0 <= k + o < n:
                        profile[o].append(v[k + o])
                # positional profile: this answer's shape, target ignored entirely
                if n > 1:
                    for j in range(n):
                        positional[min(int(10 * j / n), 9)].append(v[j])
                # within-answer position control for "after minus before"
                admissible = [j for j in range(n) if 0 < j < n - 1]
                if admissible and 0 < k < n - 1:
                    gaps = [v[j + 1:].mean() - v[:j].mean() for j in admissible]
                    true_gap.append(v[k + 1:].mean() - v[:k].mean())
                    answer_gap.append(float(np.mean(gaps)))
                    gap_rows += 1
                    peaks = [bool(v[j] > v[j - 1] and v[j] > v[j + 1]) for j in admissible]
                    true_max.append(bool(v[k] > v[k - 1] and v[k] > v[k + 1]))
                    answer_max.append(float(np.mean(peaks)))
                    max_rows += 1
            per_subset[subset] = {
                "n_error": int(len(rows)),
                "profile_sd": {str(o): float(np.mean(profile[o])) for o in OFFSETS},
                "positional_decile_sd": [float(np.mean(d)) if d else None for d in positional],
                "after_minus_before": {
                    "at_true_error": float(np.mean(true_gap)),
                    "same_answer_any_index": float(np.mean(answer_gap)),
                    "excess_over_position_control":
                        float(np.mean(np.asarray(true_gap) - np.asarray(answer_gap))),
                    "n": gap_rows},
                "local_maximum_rate": {
                    "at_true_error": float(np.mean(true_max)),
                    "same_answer_any_index": float(np.mean(answer_max)),
                    "excess_over_position_control":
                        float(np.mean(np.asarray(true_max, float) - np.asarray(answer_max))),
                    "n": max_rows},
            }
        report["arms"][arm] = per_subset

    # ---- where the argmax misses land, on the anchor arm ------------------------------
    miss = {}
    for subset in SUBSETS:
        rows = np.flatnonzero((subsets == subset) & (target >= 0))
        early = late = exact = 0
        for i in rows:
            a, b = int(offsets[i]), int(offsets[i + 1])
            p = int(np.argmax(c1[a:b]))
            k = int(target[i])
            exact += p == k
            early += p < k
            late += p > k
        miss[subset] = {"exact": exact / len(rows), "early": early / len(rows),
                        "late": late / len(rows), "n": int(len(rows))}
    report["argmax_miss_direction_c1"] = miss

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    # ----------------------------------------------------------------------- console
    for arm in arms:
        print()
        print("=" * 100)
        print(f"{arm}: profile of the step score around the TRUE first error (answer SD)")
        print("=" * 100)
        print(f"{'subset':16s}" + "".join(f"{o:>8d}" for o in OFFSETS))
        for subset in SUBSETS:
            row = report["arms"][arm][subset]["profile_sd"]
            print(f"{subset:16s}" + "".join(f"{row[str(o)]:8.3f}" for o in OFFSETS))

        print()
        print(f"{arm}: POSITIONAL PROFILE -- the same answers, target ignored (decile of position)")
        print(f"{'subset':16s}" + "".join(f"{d:>7d}" for d in range(10)))
        for subset in SUBSETS:
            row = report["arms"][arm][subset]["positional_decile_sd"]
            print(f"{subset:16s}" + "".join(f"{v:7.3f}" for v in row))

        print()
        print(f"{arm}: 'after minus before' AGAINST ITS POSITION CONTROL")
        print(f"{'subset':16s} {'at true k':>10s} {'any k, same answer':>20s} {'excess':>10s} {'n':>7s}")
        for subset in SUBSETS:
            g = report["arms"][arm][subset]["after_minus_before"]
            print(f"{subset:16s} {g['at_true_error']:10.3f} {g['same_answer_any_index']:20.3f} "
                  f"{g['excess_over_position_control']:+10.3f} {g['n']:7d}")

        print()
        print(f"{arm}: LOCAL-MAXIMUM RATE against its own null")
        print(f"{'subset':16s} {'at true k':>10s} {'any k, same answer':>20s} {'excess':>10s}")
        for subset in SUBSETS:
            m = report["arms"][arm][subset]["local_maximum_rate"]
            print(f"{subset:16s} {100*m['at_true_error']:10.1f} "
                  f"{100*m['same_answer_any_index']:20.1f} "
                  f"{100*m['excess_over_position_control']:+10.1f}")

    print()
    print("=" * 100)
    print("C1 argmax: where the misses land")
    print("=" * 100)
    print(f"{'subset':16s} {'exact':>8s} {'early':>8s} {'late':>8s} {'late-early':>11s}")
    for subset in SUBSETS:
        m = miss[subset]
        print(f"{subset:16s} {100*m['exact']:8.2f} {100*m['early']:8.2f} {100*m['late']:8.2f} "
              f"{100*(m['late']-m['early']):+11.2f}")
    print()
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
