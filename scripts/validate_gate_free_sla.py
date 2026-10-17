"""Validate the gate-free SLA engine against a REPORTED number, not a fixture.

    python scripts/validate_gate_free_sla.py

Step 422 published, on this exact population and protocol, a gate-free mean Step-level
Localization Accuracy of 35.92 for token-level CONT L-SML over the eleven-channel bank and
32.59 for its equal-weight control. Those runs saved their out-of-fold step scores, so the
engine here can be checked by replaying them: same scores in, the published per-cell numbers
out. Anything less than exact agreement means the engine is measuring something else, and every
later depth number computed with it would inherit that.

This is the check that has to pass before any new arm is built on top, because a synthetic
fixture cannot catch a disagreement about the POPULATION - which answers count as erroneous,
how ties are broken, whether PRMBench is folded in.
"""
from __future__ import annotations

import json
import sys
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
    pb_mean,
    sla_by_cell,
)

TREES = Path(__file__).resolve().parents[2]
EVALUATION = TREES.parent / "results" / "localization_full_benchmark_v3" / "evaluation"
FOLDS = TREES.parent / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"
STEP422 = TREES / "token-probability-fusion-v1" / "results" / "claude_feature_bank_token_lsml_v1"

#: what Step 422 reported, mean over the eight ProcessBench cells
PUBLISHED = {"l_sml": 0.3592, "equal": 0.3259}
ARM_NAMES = {"l_sml": "token_l_sml", "equal": "token_equal_mean"}


def main() -> int:
    for path in (EVALUATION / "JOINED.json", FOLDS, STEP422 / "OOF_SCORES.npz",
                 STEP422 / "GATE_ISOLATION.json"):
        if not path.exists():
            print(f"missing input: {path}")
            return 2

    roster = load_roster(EVALUATION)
    offsets, cells = roster["offsets"], roster["cells"]
    truth = first_error_step(roster["labels"], offsets, roster["target"], cells)
    codes, ng = group_codes(roster["group_id"])
    folds = load_folds(FOLDS, roster["group_id"])
    reference = json.loads((STEP422 / "GATE_ISOLATION.json").read_text(
        encoding="utf-8"))["sla_gate_free"]
    scores = np.load(STEP422 / "OOF_SCORES.npz", allow_pickle=False)

    pb_panels = [p for p in sorted(set(cells.tolist())) if p.startswith("pb_")]
    chance = chance_by_cell(truth, offsets, cells)

    print(f"population   {len(truth)} answers | {int((truth >= 0).sum())} erroneous | "
          f"{ng} source groups | {len(np.unique(folds))} outer folds")
    print(f"chance       {100 * float(np.mean([chance[p] for p in pb_panels])):.2f} "
          f"mean over the 8 ProcessBench cells\n")

    ok = True
    print(f"{'arm':22}{'mine':>9}{'published':>11}{'max cell diff':>15}   verdict")
    for key, published in PUBLISHED.items():
        got = sla_by_cell(argmax_predictions(scores[key], offsets), truth, codes, cells, ng)
        ref = reference[ARM_NAMES[key]]
        cell_diff = max(abs(got[p]["point"] - ref[p]["sla"]) for p in pb_panels)
        pop_diff = [(p, got[p]["n"], ref[p]["n_error"])
                    for p in pb_panels if got[p]["n"] != ref[p]["n_error"]]
        mine = pb_mean(got)
        passed = cell_diff < 1e-12 and not pop_diff and abs(mine - published) < 5e-5
        ok &= passed
        print(f"{ARM_NAMES[key]:22}{100 * mine:>9.2f}{100 * published:>11.2f}"
              f"{cell_diff:>15.2e}   {'PASS' if passed else 'FAIL'}")
        if pop_diff:
            print(f"   POPULATION MISMATCH (cell, mine, theirs): {pop_diff}")

    # The fold release is the other half of the contract: a later arm cross-fitted on
    # different folds is not comparable to these numbers, however close its mean looks.
    sizes = {int(f): int((folds == f).sum()) for f in np.unique(folds)}
    print(f"\nouter folds  {sizes}")
    print(f"             every source group in exactly one fold, "
          f"{len(set(roster['group_id'].tolist()))} groups covered")

    print("\nVALIDATED - the engine reproduces the published numbers exactly."
          if ok else "\nFAILED - do not build on this engine until it agrees.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
