#!/usr/bin/env python3
"""Complete the preregistered DEEM step the probe chain skipped.

SPEC_SOLVER_MECHANISM_STUDY.md §5.4 says: "The winner is then run on all 5 registered seeds."
The probe did not do that.  Step 3 of the tie-breaker (which is what runs seeds 1-4) is skipped
when only one configuration survives step 1 -- correct for *selection*, but it meant decision 1
("soft repair succeeded") was computed from 3 fits at seed 0 rather than the 15 (3 cells x 5
seeds) the 90%-completion rule is written against.

This script runs exactly that missing step and recomputes decision 1 on the full set.  It
selects nothing: the configuration comes from the probe's own summary.json.  AUROC is not
consulted.

Usage:
    python scripts/deem_winner_validation.py --data-dir local_cache
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_dependency_fusion_experiment import (                             # noqa: E402
    DEEM_BASE, dataset_family, derive_oriented_matrix, load_cells,
)
from deem_soft_collapse_probe import (                                     # noqa: E402
    COMPLETION_MIN, SEEDS, cross_seed_stability, health, one_fit, write_csv,
)

OUT = os.path.join(REPO, "results", "deem_probe")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO, "local_cache"))
    parser.add_argument("--out-dir", default=OUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(os.path.join(args.out_dir, "summary.json"), encoding="utf-8") as handle:
        probe = json.load(handle)
    winner = probe["decision_soft_deem"]["winner"]
    pilot = probe["pilot_cells"]
    print(f"validating the probe's winner on all {len(SEEDS)} registered seeds: "
          f"lr={winner['learning_rate']:g}, epochs={winner['epochs']}")
    print(f"pilot cells: {', '.join(pilot)}")

    started = time.time()
    cells = load_cells(os.path.abspath(args.data_dir))
    cfg = replace(DEEM_BASE, input_mode="soft", use_preprocessing=True, device=args.device,
                  learning_rate=float(winner["learning_rate"]), epochs=int(winner["epochs"]))
    artifact_dir = os.path.join(args.out_dir, "winner_validation_artifacts")

    rows = []
    for ck in pilot:
        F, _, _ = derive_oriented_matrix(cells[ck])
        for seed in SEEDS:
            rec = one_fit(F, ck, "winner_validation", seed, cfg, artifact_dir)
            rows.append(rec)
            print(f"  {ck:28s} seed={seed} sd={rec['score_sd']:.3e} {rec['health']}", flush=True)

    healthy = [r for r in rows if r["health"] == "healthy"]
    completion = len(healthy) / max(len(rows), 1)
    by_cell = {}
    for ck in pilot:
        by_cell[ck] = cross_seed_stability([r for r in rows if r["cell"] == ck])

    decision = {
        "winner": winner, "n_fits": len(rows), "n_healthy": len(healthy),
        "completion_rate": completion, "required": COMPLETION_MIN,
        "repair_succeeded": bool(completion >= COMPLETION_MIN),
        "cross_seed_stability_by_cell": by_cell,
        "median_score_sd": float(np.nanmedian([r["score_sd"] for r in rows])),
        "supersedes": "the probe's decision 1, which was computed from 3 seed-0 fits",
        "verdict": ("soft DEEM repaired on all 5 seeds — run its predefined evaluation, "
                    "independently of hard DEEM" if completion >= COMPLETION_MIN else
                    "abandon soft DEEM — the winner does not reach 90% healthy completion "
                    "across seeds"),
    }

    write_csv(os.path.join(args.out_dir, "winner_validation.csv"), rows)
    with open(os.path.join(args.out_dir, "winner_validation.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"status": "SECONDARY diagnostic; completes SPEC §5.4",
                   "decision_soft_deem_final": decision,
                   "runtime_seconds": time.time() - started}, handle, indent=2, sort_keys=True)

    print(f"\ncompletion {len(healthy)}/{len(rows)} = {completion:.2f} "
          f"(required {COMPLETION_MIN:.2f})")
    for ck, s in by_cell.items():
        print(f"  cross-seed |Spearman| median  {ck:28s} {s:.4f}")
    print(f"\nDECISION 1 (final): {decision['verdict']}")
    print(f"wrote {args.out_dir}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
