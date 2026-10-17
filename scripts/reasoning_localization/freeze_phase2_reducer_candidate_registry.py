#!/usr/bin/env python3
"""Freeze the next sequential non-reference Phase-2R Stage-A execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as base  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer_candidate as candidate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=candidate.CANDIDATES)
    args = parser.parse_args()
    variant_id = args.variant
    target = base.PHASE_ROOT / f"{variant_id}_EXECUTION_REGISTRY.json"
    output = base.PHASE_ROOT / variant_id.lower()
    if target.exists() or output.exists():
        raise FileExistsError(f"refusing to overwrite an existing execution: {variant_id}")

    variant_path = base.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variant_path.read_text(encoding="utf-8"))["variants"]
    by_id = {row["variant_id"]: row for row in variants}
    position = base.STAGE_A_VARIANTS.index(variant_id)
    for previous in base.STAGE_A_VARIANTS[:position]:
        if by_id[previous]["execution_status"] not in {"COMPLETE", "HARD_FAIL"}:
            raise RuntimeError(f"prior Stage-A row has not been discussed/completed: {previous}")
    if by_id[variant_id]["execution_status"] != "PLANNED":
        raise RuntimeError(f"candidate is not PLANNED: {variant_id}")
    for later in base.STAGE_A_VARIANTS[position + 1:]:
        if by_id[later]["execution_status"] != "PLANNED":
            raise RuntimeError(f"later Stage-A row opened out of order: {later}")

    reference_registry = base.PHASE_ROOT / f"{base.REFERENCE}_EXECUTION_REGISTRY.json"
    reference_run = candidate.REFERENCE_ROOT / "RUN_MANIFEST.json"
    reference_score = candidate.REFERENCE_ROOT / "score_freeze/SCORE_FREEZE_MANIFEST.json"
    reference_eval = candidate.REFERENCE_ROOT / "evaluation/EVALUATION_MANIFEST.json"
    reference_decisions = candidate.REFERENCE_ROOT / "evaluation/PROCESSBENCH_DECISIONS.csv"
    reference_cells = candidate.REFERENCE_ROOT / "evaluation/PROCESSBENCH_BY_CELL.csv"
    reference_samples = candidate.REFERENCE_ROOT / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz"
    runner = Path(candidate.__file__).resolve()
    score_builder = Path(base.__file__).resolve()
    release = base.DEFAULT_RELEASE.resolve()
    sources = [
        ("protocol", REPO / "docs/experiments/REASONING_LOCALIZATION_03662_ANCHOR_V1.md"),
        ("reference_execution_registry", reference_registry),
        ("reference_run_manifest", reference_run),
        ("reference_score_freeze_manifest", reference_score),
        ("reference_evaluation_manifest", reference_eval),
        ("reference_thresholds", candidate.THRESHOLD_PATH),
        ("reference_decisions", reference_decisions),
        ("reference_by_cell", reference_cells),
        ("reference_bootstrap", reference_samples),
        ("prepared_input_manifest", release / "build_A/localization/inputs/MANIFEST.json"),
        ("processbench_labels", release / "build_A/localization/evaluation/localization_decisions.csv"),
        ("localization_contract", REPO / "spectral_utils/reconstruction_benchmark/localization_contract.py"),
        ("localization_evaluation", REPO / "spectral_utils/reconstruction_benchmark/localization_evaluation.py"),
        ("fixed_application_pipelines", REPO / "spectral_utils/fixed_application_pipelines.py"),
        ("phase1_runner", REPO / "scripts/reasoning_localization/run_phase1_baseline.py"),
        ("score_builder", score_builder),
    ]
    registry = {
        "schema": "reasoning-localization-phase2-reducer-candidate-execution-registry-v1",
        "status": "FROZEN_BEFORE_RUN",
        "phase": "P2",
        "experiment_id": "P2_REDUCER_STUDY",
        "variant_id": variant_id,
        "analysis_tier": (
            "POST_HOC_EXPLORATORY"
            if variant_id in base.EXPLORATORY_STAGE_A_VARIANTS
            else "PREREGISTERED_DEVELOPMENT"
        ),
        "promotion_eligible": variant_id not in base.EXPLORATORY_STAGE_A_VARIANTS,
        "stage_a_position": position,
        "stage_a_order": list(base.STAGE_A_VARIANTS),
        "release_root": str(release),
        "processbench_cells": list(base.PB_CELLS),
        "population_id": "current_common_eight_qwen",
        "reference_variant": base.REFERENCE,
        "reference_threshold_path": str(candidate.THRESHOLD_PATH),
        "reference_threshold_sha256": sha256_file(candidate.THRESHOLD_PATH),
        "candidate_rethresholding_allowed": False,
        "bootstrap_draws": base.BOOTSTRAP_DRAWS,
        "bootstrap_seed": base.BOOTSTRAP_SEED,
        "practical_benefit_bound": 0.005,
        "practical_harm_bound": -0.005,
        "promotion_worst_cell_bound": -0.020,
        "hard_worst_cell_bound": candidate.HARD_WORST_CELL_BOUND,
        "component_regression_bound": -0.010,
        "minimum_nonnegative_cells": 6,
        "multiplicity_contract": (
            "Preregistered Stage-A contrasts retain their closed eleven-test Bonferroni family; "
            "post-hoc tail-fraction amendments form a separate descriptive Bonferroni family and "
            "cannot promote on the opened ProcessBench development population"
        ),
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "score_builder_path": str(score_builder),
        "score_builder_sha256": sha256_file(score_builder),
        "frozen_registry_hashes": {
            name: sha256_file(base.PROGRAM_ROOT / name)
            for name in (
                "METHOD_REGISTRY.json", "VARIANT_REGISTRY.json",
                "EXPERIMENT_REGISTRY.json",
            )
        },
        "frozen_sources": [
            {"role": role, "path": str(path), "sha256": sha256_file(path)}
            for role, path in sources
        ],
    }
    atomic_write_json(target, registry)
    print(json.dumps({
        "status": "FROZEN_BEFORE_RUN",
        "variant_id": variant_id,
        "registry": str(target),
        "registry_sha256": sha256_file(target),
        "runner_sha256": registry["runner_sha256"],
        "reference_threshold_sha256": registry["reference_threshold_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
