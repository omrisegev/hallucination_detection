#!/usr/bin/env python3
"""Freeze the completed Stage-A reducer conclusion and development parent."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)
from scripts.reasoning_localization import run_phase2_reducer as base  # noqa: E402


SELECTED = "P2R_A_TOPK10"
ATOMIC_VARIANTS = (
    "C1_ENT_SW16",
    "C2_ENT_SWADAPT",
    "C3_ENT_CCUSUM",
    "C4_ENT_SAMPLED",
    "C5_ENT_ENERGY",
    "C6_DSP12",
    "C7_EDIS_ONSET",
    "C8_SELF_INNOV",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    target = base.PHASE_ROOT / "P2R_STAGE_A_SELECTION.json"
    if target.exists():
        raise FileExistsError("refusing to overwrite frozen Stage-A selection")

    candidate_ids = base.STAGE_A_VARIANTS[1:]
    for variant_id in candidate_ids:
        run_path = base.PHASE_ROOT / variant_id.lower() / "RUN_MANIFEST.json"
        if not run_path.is_file():
            raise RuntimeError(f"Stage-A variant has not run: {variant_id}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("status") not in {"COMPLETE", "HARD_FAIL"}:
            raise RuntimeError(f"Stage-A variant is not terminal: {variant_id}")

    metric_rows = read_csv(base.PROGRAM_ROOT / "METRICS_LONG.csv")
    scores = {
        row["variant_id"]: float(row["value"])
        for row in metric_rows
        if row["experiment_id"] == "P2_REDUCER_STUDY"
        and row["cell_id"] == "aggregate"
        and row["metric_id"] == "macro_f1"
    }
    rankable_pool = [
        base.REFERENCE,
        *[
            variant_id for variant_id in candidate_ids
            if variant_id not in base.EXPLORATORY_STAGE_A_VARIANTS
        ],
    ]
    if max(rankable_pool, key=scores.__getitem__) != SELECTED:
        raise RuntimeError("top-ten is not the raw-best preregistered Stage-A row")

    contrast_path = base.PHASE_ROOT / "reducer_interim/P2R_CONTRASTS.csv"
    contrast = next(
        row for row in read_csv(contrast_path)
        if row["left_variant_id"] == SELECTED and row["metric_id"] == "macro_f1"
    )
    gate_rows = [
        row for row in read_csv(base.PROGRAM_ROOT / "GATES_LONG.csv")
        if row["variant_id"] == SELECTED
    ]
    failed = [row["gate_id"] for row in gate_rows if row["passed"] == "false"]
    if failed != ["P2R_SIMULTANEOUS_CI_BENEFIT"]:
        raise RuntimeError(f"unexpected top-ten failed-gate roster: {failed}")

    artifact = {
        "schema": "reasoning-localization-phase2-reducer-stage-a-selection-v1",
        "status": "FROZEN_AFTER_STAGE_A",
        "confirmatory_reference": base.REFERENCE,
        "development_parent": SELECTED,
        "selection_basis": "raw-best preregistered Stage-A macro F1 after the closed eleven-contrast family; continuation is diagnostic, not promotion",
        "selection_opened": True,
        "promotion_status": "NO_PROMOTION",
        "fresh_confirmation_required": True,
        "macro_f1": scores[SELECTED],
        "paired_delta_macro_f1": float(contrast["delta"]),
        "simultaneous_ci": [float(contrast["ci_low"]), float(contrast["ci_high"])],
        "practical_benefit_bound": 0.005,
        "failed_promotion_gates": failed,
        "downstream_comparators": [base.REFERENCE, SELECTED],
        "posthoc_amendments_excluded_from_selection": list(
            base.EXPLORATORY_STAGE_A_VARIANTS
        ),
        "source_hashes": {
            "protocol": sha256_file(
                REPO / "docs/experiments/REASONING_LOCALIZATION_03662_ANCHOR_V1.md"
            ),
            "metrics": sha256_file(base.PROGRAM_ROOT / "METRICS_LONG.csv"),
            "contrasts": sha256_file(contrast_path),
            "gates": sha256_file(base.PROGRAM_ROOT / "GATES_LONG.csv"),
            "top_ten_run": sha256_file(
                base.PHASE_ROOT / SELECTED.lower() / "RUN_MANIFEST.json"
            ),
        },
    }
    atomic_write_json(target, artifact)

    experiment_path = base.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment = next(
        row for row in experiments["experiments"]
        if row["experiment_id"] == "P2_REDUCER_STUDY"
    )
    experiment.update({
        "stage_a_execution_status": "COMPLETE",
        "stage_a_confirmatory_reference": base.REFERENCE,
        "stage_a_development_parent": SELECTED,
        "stage_a_decision": "NO_PROMOTION",
        "stage_a_selection_opened": True,
        "stage_a_fresh_confirmation_required": True,
        "stage_a_selection_artifact": target.resolve().relative_to(REPO).as_posix(),
    })
    atomic_write_json(experiment_path, experiments)

    variant_path = base.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variant_path.read_text(encoding="utf-8"))
    for row in variants["variants"]:
        if row["variant_id"] in ATOMIC_VARIANTS:
            row["step_reducer"] = (
                "frozen P2R_A_TOPK10 selection-opened development parent; "
                "P2R_A_TOPK5_REFERENCE remains a required comparator"
            )
    atomic_write_json(variant_path, variants)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
