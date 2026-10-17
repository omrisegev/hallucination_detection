#!/usr/bin/env python3
"""Repair only P3F evaluation after a prediction-flip field-name mismatch.

The DUFS/LIU score freeze is immutable.  This script verifies its hashes and
rebuilds the label-side evaluation without fitting or changing any score.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase3_compact_fusion as p3  # noqa: E402
from scripts.reasoning_localization import run_phase3_context_dufs_family as run  # noqa: E402


def main() -> None:
    manifest_path = run.OUTPUT / "score_freeze/SCORE_FREEZE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    verified = run._verified(manifest)
    labels = p1._load_pb_labels(p1.DEFAULT_RELEASE.resolve())
    evaluator = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    h0 = c1.evaluate_arm(run.H0, run._rows(verified, labels, "h0_combined"), evaluator)
    arms = {run.H0: h0}
    for variant in run.VARIANTS:
        arms[variant] = p3._rerank(
            variant, h0,
            run._rows(verified, labels, f"{variant.lower()}_local"),
            evaluator,
        )
    abstain = {
        (row["cell_id"], row["row_id"]): int(row["prediction_step"]) == -1
        for row in h0["decisions"]
    }
    mismatches = {
        arm: sum(
            (int(row["prediction_step"]) == -1)
            != abstain[(row["cell_id"], row["row_id"])]
            for row in arms[arm]["decisions"]
        )
        for arm in run.VARIANTS
    }
    if any(mismatches.values()):
        raise run.ContextDUFSError(f"H0 abstention alias failed: {mismatches}")

    pairs = [*run.PRIMARY, (run.F3, run.F0)]
    contrasts = [
        run.p3e._contrast(left, right, metric, arms, (left, right) in run.PRIMARY)
        for left, right in pairs for metric in p1.PB_METRICS
    ]
    evaluation_root = run.OUTPUT / "evaluation"
    evaluation_root.mkdir(exist_ok=True)
    panels = [row for arm in arms.values() for row in arm["panels"]]
    run._write_csv(evaluation_root / "PROCESSBENCH_PANELS.csv", panels)
    run._write_csv(evaluation_root / "PROCESSBENCH_BY_CELL.csv", [row for arm in arms.values() for row in arm["by_cell"]])
    run._write_csv(evaluation_root / "PAIRWISE_CONTRASTS.csv", contrasts)

    parent = {(row["cell_id"], row["row_id"]): row for row in arms[run.F0]["decisions"]}
    flips = []
    for variant in (run.F1, run.F2, run.F3):
        for row in arms[variant]["decisions"]:
            base = parent[(row["cell_id"], row["row_id"])]
            if int(row["prediction_step"]) != int(base["prediction_step"]):
                flips.append({
                    "variant_id": variant,
                    "cell_id": row["cell_id"],
                    "row_id": row["row_id"],
                    "parent_prediction_step": base["prediction_step"],
                    "candidate_prediction_step": row["prediction_step"],
                    "first_error": row["true_first_error"],
                })
    if flips:
        run._write_csv(evaluation_root / "PREDICTION_FLIPS.csv", flips)

    primary = [
        row for row in contrasts
        if row["metric_id"] == "macro_f1"
        and (row["left_variant_id"], row["right_variant_id"]) in run.PRIMARY
    ]
    primary_map = {(row["left_variant_id"], row["right_variant_id"]): row for row in primary}
    hard_valid = max(manifest["alias_max_errors"].values()) <= run.ALIAS_TOLERANCE and not any(mismatches.values())
    topk_eligible = hard_valid and any(
        primary_map[(variant, run.F0)]["delta"] > 0
        and primary_map[(variant, run.F0)]["ci_high"] >= run.HARM
        and primary_map[(variant, run.F0)]["worst_unit_delta"] >= -.020
        for variant in (run.F1, run.F2)
    )
    context_supported = (
        primary_map[(run.F2, run.F1)]["ci_low"] > 0
        and primary_map[(run.F2, run.F3)]["ci_low"] > 0
    )
    summary = {
        "schema": "reasoning-localization-p3f-evaluation-v1",
        "status": "COMPLETE",
        "experiment_id": run.EXPERIMENT,
        "primary_contrasts": primary,
        "alias_max_errors": manifest["alias_max_errors"],
        "abstention_mismatches": mismatches,
        "context_mechanism_supported": context_supported,
        "topk_secondary_control_eligible": topk_eligible,
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "evaluation_repair": "prediction decision field true_first_error replaces nonexistent first_error; no score changed",
    }
    summary["payload_sha256"] = c1.payload_sha(summary)
    atomic_write_json(evaluation_root / "SUMMARY.json", summary)
    run._plot(evaluation_root / "P3F_RESULTS.svg", panels, contrasts)
    repair = {
        "schema": "reasoning-localization-p3f-evaluation-repair-v1",
        "status": "COMPLETE",
        "reason": "Original evaluation expected decision field first_error; evaluator emits true_first_error.",
        "score_freeze_manifest_sha256": sha256_file(manifest_path),
        "frozen_runner_sha256": manifest["runner_sha256"],
        "repair_script_sha256": sha256_file(Path(__file__).resolve()),
        "scores_changed": False,
        "labels_refit": False,
    }
    atomic_write_json(run.OUTPUT / "EVALUATION_REPAIR.json", repair)
    atomic_write_json(run.OUTPUT / "RUN_COMPLETE.json", {"status": "COMPLETE_WITH_EVALUATION_REPAIR", "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
