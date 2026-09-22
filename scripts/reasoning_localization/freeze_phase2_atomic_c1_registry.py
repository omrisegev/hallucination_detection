#!/usr/bin/env python3
"""Freeze the exact C1 execution contract before opening ProcessBench results."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402


def main() -> None:
    if c1.REGISTRY_PATH.exists() or c1.OUTPUT_ROOT.exists():
        raise FileExistsError("refusing to overwrite an opened C1 execution")

    variant_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variant_path.read_text(encoding="utf-8"))["variants"]
    by_id = {row["variant_id"]: row for row in variants}
    for variant_id in (c1.REFERENCE, c1.CANDIDATE):
        if variant_id not in by_id or by_id[variant_id]["execution_status"] != "PLANNED":
            raise RuntimeError(f"{variant_id} is not registered and PLANNED")
    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))["experiments"]
    experiment = next(row for row in experiments if row["experiment_id"] == "P2_ATOMIC")
    if experiment["execution_status"] != "PLANNED" or experiment.get("opened_variants") != []:
        raise RuntimeError("P2 atomic roster was opened before C1 registry freeze")

    release = p1.DEFAULT_RELEASE.resolve()
    runner = Path(c1.__file__).resolve()
    selection = p1.PROGRAM_ROOT / "phase_2/P2R_STAGE_A_SELECTION.json"
    top10_run = c1.P2R_TOP10_ROOT / "RUN_MANIFEST.json"
    top10_freeze = c1.P2R_TOP10_ROOT / "score_freeze/SCORE_FREEZE_MANIFEST.json"
    top5_run = c1.P1_TOP5_ROOT / "RUN_MANIFEST.json"
    sources = [
        ("protocol", REPO / "docs/experiments/REASONING_LOCALIZATION_03662_ANCHOR_V1.md"),
        ("method_registry", p1.PROGRAM_ROOT / "METHOD_REGISTRY.json"),
        ("variant_registry", variant_path),
        ("experiment_registry", experiments_path),
        ("stage_a_selection", selection),
        ("stage_a_top10_run", top10_run),
        ("stage_a_top10_score_freeze", top10_freeze),
        ("phase1_top5_run", top5_run),
        ("phase1_top5_cells", c1.P1_TOP5_ROOT / "evaluation/PROCESSBENCH_BY_CELL.csv"),
        ("phase1_top5_bootstrap", c1.P1_TOP5_ROOT / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz"),
        ("prepared_input_manifest", release / "build_A/localization/inputs/MANIFEST.json"),
        ("processbench_labels", release / "build_A/localization/evaluation/localization_decisions.csv"),
        ("localization_contract", REPO / "spectral_utils/reconstruction_benchmark/localization_contract.py"),
        ("localization_evaluation", REPO / "spectral_utils/reconstruction_benchmark/localization_evaluation.py"),
        ("fixed_application_pipelines", REPO / "spectral_utils/fixed_application_pipelines.py"),
        ("phase1_runner", REPO / "scripts/reasoning_localization/run_phase1_baseline.py"),
        ("reducer_runner", REPO / "scripts/reasoning_localization/run_phase2_reducer.py"),
        ("atomic_runner", runner),
        ("atomic_tests", REPO / "scripts/test_reasoning_localization_phase2_atomic.py"),
    ]
    registry = {
        "schema": "reasoning-localization-phase2-atomic-c1-execution-registry-v1",
        "status": "FROZEN_BEFORE_RUN",
        "phase": "P2",
        "experiment_id": "P2_ATOMIC",
        "candidate": c1.CANDIDATE,
        "atomic_reference": c1.REFERENCE,
        "confirmatory_reference": "R1_ENTROPY_TOP5",
        "release_root": str(release),
        "processbench_cells": list(p2r.PB_CELLS),
        "population_id": "current_common_eight_qwen",
        "window": c1.WINDOW,
        "variance_ddof": 0,
        "fusion_weight": c1.FUSION_WEIGHT,
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "primary_comparison_family_size": c1.PRIMARY_COMPARISON_FAMILY,
        "comparators": [c1.REFERENCE, "R1_ENTROPY_TOP5"],
        "practical_benefit_bound": c1.BENEFIT,
        "practical_harm_bound": c1.HARM,
        "component_regression_bound": c1.COMPONENT_BOUND,
        "promotion_worst_cell_bound": c1.PROMOTION_WORST_CELL_BOUND,
        "hard_worst_cell_bound": c1.HARD_WORST_CELL_BOUND,
        "score_contract": {
            "entropy": "negative mixed-v2 entropy confidence coordinate",
            "swvar": "per-response reset; available-prefix population variance over at most 16 tokens",
            "reducer": "top-ten mean independently per channel",
            "fusion": "equal mean of within-cell empirical midranks",
            "response_detector": "geometric mean with equal_feature_mean response rank",
            "threshold": "independent deterministic grouped five-fold cross-fit per arm after score freeze",
        },
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "registry_path": str(c1.REGISTRY_PATH.resolve()),
        "frozen_sources": [
            {"role": role, "path": str(path), "sha256": sha256_file(path)}
            for role, path in sources
        ],
    }
    c1.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(c1.REGISTRY_PATH, registry)
    print(json.dumps({
        "status": registry["status"],
        "candidate": c1.CANDIDATE,
        "registry": str(c1.REGISTRY_PATH),
        "registry_sha256": sha256_file(c1.REGISTRY_PATH),
        "runner_sha256": registry["runner_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
