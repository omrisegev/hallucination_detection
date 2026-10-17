"""The step-level baseline of the eleven-channel bank - the arm depth actually has to beat.

    python scripts/run_step_level_bank_baseline_v1.py

WHY THIS EXISTS RATHER THAN REUSING 35.92
-----------------------------------------
Step 422's published 35.92 gate-free SLA is a TOKEN-level fusion: the standardizer and the
L-SML weights are fitted on token matrices, and only the fused token score is reduced to steps
with the Top-10 readout. The white-box depth field is only available to us at STEP level, so a
depth arm fused at step level cannot be compared to 35.92 without confounding "depth helps"
with "fitting scope changed". This script supplies the matched control: the SAME eleven
channels, the SAME source-group folds, the SAME metric, fused at STEP level. 35.92 and 32.59
are printed beside it as context, never as the contrast.

WHAT IS FIXED HERE, AND WHY EACH IS NOT A FREE CHOICE
-----------------------------------------------------
* Standardization is ANSWER-LOCAL (`lsml_gate_locator_research.answer_standardize`), the
  canonical step-level convention. The Step-422 token arm used a pooled standardizer fitted on
  donor folds; the handoff flags that as a declared deviation. Answer-local is also what the
  project's answer-only mandate asks for, and it is what every other step-level arm uses.
* Orientation: the eleven channels arrive already multiplied by their risk signs, so higher is
  riskier before anything is fitted. The fusion's own sign is then fixed by `_orient` against
  channel 0 (`q15_H1`, the entropy anchor) - the project's standing anchor rule. No label
  touches the orientation. This is the discipline Stage 1a lacked, where the headline depended
  on a sign chosen by looking at outcomes.
* Cross-fitting is by SOURCE GROUP, five outer folds, from the canonical fold release. Weights
  are fitted on the four donor folds' steps and applied to the held-out fold. PRMBench holds
  several perturbed copies of one source question and ProcessBench repeats questions under
  different answer ids, so folding by answer would put the same question on both sides.
* Effective views (`effective_rank`, the participation ratio of the correlation spectrum) is
  reported beside every fused arm, per the project's reporting contract.

ARMS
----
`equal`       unweighted mean of the standardized channels. The mandatory simple control.
`continuous`  CONT L-SML with `small_m_guard=True` - the maintained configuration. Note this
              worktree is off master precisely because the guard is ABSENT from the two
              feature branches, where the same call would silently be a different estimator.
`joint`       Joint grouping + two-stage readout, on L-SML's own discovered partition. Included
              because the Step-399 method card shows Joint's advantage GROWS with roster
              redundancy (-0.26 at 8 diverse streams, +4.31 at 24), and eleven channels with a
              cancelling pair sit between those.

An honest expectation, recorded before the run: the bank is effectively TEN, not eleven -
`energy_level` and `energy_innovation` form a two-member unidentified L-SML group that cancels
to fifteen digits in every fold. A two-member group is exactly the regime where L-SML is
undetermined, so watch the group structure in the output, not only the SLA.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_utils.gate_free_sla import (  # noqa: E402
    argmax_predictions,
    chance_by_cell,
    first_error_step,
    group_codes,
    load_folds,
    load_roster,
    paired_intervals,
    pb_mean,
    shared_group_weights,
    sla_by_cell,
)
from spectral_utils.lsml_gate_locator_research import (  # noqa: E402
    FusionRecipe,
    answer_standardize,
    effective_rank,
    fit_fusion_weights,
)

TREES = Path(__file__).resolve().parents[2]
EVALUATION = TREES.parent / "results" / "localization_full_benchmark_v3" / "evaluation"
FOLDS = TREES.parent / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"
TPF = TREES / "token-probability-fusion-v1" / "results"
BANK = TPF / "token_probability_fusion_v1" / "DERIVATIVE_CHANNELS.npz"
CT7 = TPF / "chosen_token_calibration_v1" / "CT7_DEV_SCORES.npz"
TOKEN_SPANS = TPF / "token_probability_fusion_v1" / "TOKEN_MATRICES.npz"
OUT = Path(__file__).resolve().parents[1] / "results" / "step_level_bank_baseline_v1"

DRAWS = 10000
SEED = 20260919
#: Step 422, token-level fusion of the same bank. Context for scope, not a contrast.
TOKEN_LEVEL_CONTEXT = {"token CONT L-SML": 0.3592, "token equal mean": 0.3259}


def fold_fit_score(values, offsets, folds, recipe, *, seed):
    """Fit `recipe` on each fold's donor answers, score the held-out ones. Never sees labels."""
    scores = np.full(values.shape[0], np.nan, dtype=np.float64)
    records = []
    for fold in np.unique(folds):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train_rows = np.concatenate([np.arange(offsets[a], offsets[a + 1]) for a in train])
        test_rows = np.concatenate([np.arange(offsets[a], offsets[a + 1]) for a in test])
        weight, meta = fit_fusion_weights(values[train_rows], recipe, seed=seed)
        scores[test_rows] = values[test_rows] @ weight
        records.append({
            "fold": int(fold), "train_answers": int(len(train)),
            "test_answers": int(len(test)), "train_steps": int(len(train_rows)),
            "weights": [float(w) for w in weight],
            "weight_ipr": float(1.0 / np.sum((np.abs(weight) / np.abs(weight).sum()) ** 2)),
            "meta": {k: v for k, v in meta.items()
                     if isinstance(v, (str, int, float, bool, list))},
        })
        print(f"  [{recipe.name}] fold {fold}: train {len(train)}a/{len(train_rows)}s "
              f"-> test {len(test)}a", flush=True)
    if not np.isfinite(scores).all():
        raise SystemExit(f"{recipe.name}: {int((~np.isfinite(scores)).sum())} unscored steps")
    return scores, records


def main() -> int:
    t0 = time.time()
    for path in (EVALUATION / "JOINED.json", FOLDS, BANK):
        if not path.exists():
            print(f"missing input: {path}")
            return 2
    OUT.mkdir(parents=True, exist_ok=True)

    print("[load] roster, folds, eleven-channel step bank")
    roster = load_roster(EVALUATION)
    offsets, cells = roster["offsets"], roster["cells"]
    truth = first_error_step(roster["labels"], offsets, roster["target"], cells)
    codes, ng = group_codes(roster["group_id"])
    folds = load_folds(FOLDS, roster["group_id"])
    with np.load(BANK, allow_pickle=False) as z:
        raw = z["level"].astype(np.float64)
        channels = list(z["channels"].astype(str))
    if raw.shape != (int(offsets[-1]), 11):
        raise SystemExit(f"bank is {raw.shape}, expected ({int(offsets[-1])}, 11)")
    err = truth >= 0
    pb_panels = [p for p in sorted(set(cells.tolist())) if p.startswith("pb_")]
    chance = chance_by_cell(truth, offsets, cells)
    print(f"  {len(truth)} answers | {int(err.sum())} erroneous | {raw.shape[0]} steps | "
          f"{ng} groups | {len(np.unique(folds))} folds")
    print(f"  channels: {', '.join(channels)}")

    values = answer_standardize(raw, offsets)
    print(f"[views] effective views over the standardized bank: "
          f"{effective_rank(values):.3f} of {values.shape[1]}")

    weights_matrix = shared_group_weights(ng, DRAWS, SEED + 1)
    recipes = [
        FusionRecipe(name="equal", members=tuple(channels), mode="equal", anchor=0),
        FusionRecipe(name="continuous", members=tuple(channels), mode="continuous", anchor=0),
    ]

    arms, fits = {}, {}
    for recipe in recipes:
        print(f"[fit] {recipe.name}")
        scores, records = fold_fit_score(values, offsets, folds, recipe, seed=SEED)
        arms[recipe.name] = scores
        fits[recipe.name] = records

    # Joint needs a partition. Use the one CONT L-SML itself discovered on the full bank, which
    # is label-free, and only if it satisfies Joint's identifiability precondition (K >= 3 with
    # every group >= 3). Skipping loudly beats fabricating a partition to satisfy the API.
    _, cont_meta = fit_fusion_weights(
        values, FusionRecipe(name="probe", members=tuple(channels),
                             mode="continuous", anchor=0), seed=SEED)
    discovered = np.asarray(cont_meta["groups"], dtype=np.int64)
    sizes = np.bincount(discovered)
    print(f"[groups] CONT L-SML discovered K={cont_meta['K']} with sizes {sizes.tolist()}")
    if len(sizes) >= 3 and sizes.min() >= 3:
        recipe = FusionRecipe(name="joint", members=tuple(channels), mode="joint",
                              groups=tuple(discovered.tolist()), anchor=0)
        print("[fit] joint")
        scores, records = fold_fit_score(values, offsets, folds, recipe, seed=SEED)
        arms["joint"], fits["joint"] = scores, records
    else:
        print("[fit] joint SKIPPED - the discovered partition violates K>=3 with all groups>=3, "
              "which is Joint's identifiability precondition, not a tunable")

    if CT7.exists():
        arms["ct7_locator"] = np.load(CT7, allow_pickle=False)["step_scores"].astype(np.float64)
    else:
        print(f"  WARNING: CT7 absent at {CT7} - the production locator row is missing")

    # Two length/position controls, because every step readout inherits both priors and a
    # depth or fusion arm that merely reproduces them is not a finding. Step length alone is
    # documented at ~29.7 SLA against 16.58 chance.
    if TOKEN_SPANS.exists():
        with np.load(TOKEN_SPANS, allow_pickle=False) as z:
            spans = z["step_spans"].astype(np.int64)
        if len(spans) != raw.shape[0]:
            raise SystemExit(f"step_spans has {len(spans)} rows, expected {raw.shape[0]}")
        arms["step_length"] = (spans[:, 1] - spans[:, 0]).astype(np.float64)
    else:
        print(f"  WARNING: {TOKEN_SPANS} absent - the step-length control is missing")
    counts = np.diff(offsets)
    arms["position"] = np.concatenate([np.arange(n) for n in counts]).astype(np.float64)

    print(f"[score] gate-free SLA, {DRAWS} shared paired draws")
    results = {k: sla_by_cell(argmax_predictions(v, offsets), truth, codes, cells, ng,
                              weights=weights_matrix) for k, v in arms.items()}
    contrasts = {k: paired_intervals(results[k], results["equal"])
                 for k in results if k != "equal"}

    order = [k for k in ("position", "step_length", "ct7_locator", "equal", "continuous",
                         "joint") if k in results]
    print("\n=== gate-free SLA, tolerance 0, erroneous answers only ===")
    print("ProcessBench and PRMBench are never averaged together.\n")
    print(f"{'cell':<22}{'chance':>9}" + "".join(f"{k:>14}" for k in order))
    for panel in pb_panels + [p for p in sorted(set(cells.tolist())) if p not in pb_panels]:
        print(f"{panel:<22}{100 * chance[panel]:>9.2f}"
              + "".join(f"{100 * results[k][panel]['point']:>14.2f}" for k in order))
    print(f"{'mean over 8 PB cells':<22}"
          f"{100 * float(np.mean([chance[p] for p in pb_panels])):>9.2f}"
          + "".join(f"{100 * pb_mean(results[k]):>14.2f}" for k in order))

    print("\n=== token-level context (DIFFERENT fitting scope - not a contrast) ===")
    for name, value in TOKEN_LEVEL_CONTEXT.items():
        print(f"  {name:<24}{100 * value:>8.2f}")

    print("\n=== vs the equal-weight control, paired over source groups ===")
    print(f"{'arm':<16}{'delta':>9}{'low':>9}{'high':>9}   verdict")
    for name in order:
        if name == "equal":
            continue
        e = contrasts[name]["POOLED"]
        pb_delta = float(np.mean([contrasts[name][p]["point"] for p in pb_panels]))
        wins = sum(1 for p in pb_panels if contrasts[name][p]["low"] > 0)
        loses = sum(1 for p in pb_panels if contrasts[name][p]["high"] < 0)
        print(f"{name:<16}{100 * pb_delta:>+9.2f}{'':>9}{'':>9}   "
              f"beats equal in {wins}/8 PB cells, loses in {loses}/8")

    payload = {
        "population": {"answers": int(len(truth)), "erroneous": int(err.sum()),
                       "steps": int(raw.shape[0]), "groups": int(ng)},
        "channels": channels,
        "effective_views_bank": float(effective_rank(values)),
        "chance_by_cell": chance,
        "sla_by_cell": {k: {p: {kk: vv for kk, vv in e.items() if kk != "draws"}
                            for p, e in v.items()} for k, v in results.items()},
        "pb_mean": {k: pb_mean(v) for k, v in results.items()},
        "token_level_context": TOKEN_LEVEL_CONTEXT,
        "vs_equal": contrasts,
        "fits": fits,
        "draws": DRAWS, "seconds": round(time.time() - t0, 1),
    }
    (OUT / "RESULTS.json").write_text(json.dumps(payload, indent=1, default=float),
                                      encoding="utf-8")
    np.savez_compressed(OUT / "STEP_SCORES.npz", **{k: v for k, v in arms.items()})
    print(f"\n{time.time() - t0:.0f}s   wrote {OUT / 'RESULTS.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
