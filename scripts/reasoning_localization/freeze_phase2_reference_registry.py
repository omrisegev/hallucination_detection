#!/usr/bin/env python3
"""Freeze only the first authorized Phase-2R reference execution."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization.run_phase2_reducer import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DEFAULT_RELEASE,
    PB_CELLS,
    PHASE_ROOT,
    P1_REFERENCE_ROOT,
    PROGRAM_ROOT,
    REFERENCE,
    STAGE_A_VARIANTS,
)


def main() -> None:
    if PHASE_ROOT.exists():
        raise FileExistsError(f"refusing to mutate existing Phase-2 root: {PHASE_ROOT}")
    PHASE_ROOT.mkdir(parents=True, exist_ok=False)
    runner = REPO / "scripts/reasoning_localization/run_phase2_reducer.py"
    release = DEFAULT_RELEASE.resolve()
    sources = [
        ("protocol", REPO / "docs/experiments/REASONING_LOCALIZATION_03662_ANCHOR_V1.md"),
        ("phase1_run_manifest", P1_REFERENCE_ROOT / "RUN_MANIFEST.json"),
        ("phase1_score_freeze_manifest", P1_REFERENCE_ROOT / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
        ("phase1_evaluation_manifest", P1_REFERENCE_ROOT / "evaluation/EVALUATION_MANIFEST.json"),
        ("phase1_decisions", P1_REFERENCE_ROOT / "evaluation/PROCESSBENCH_DECISIONS.csv"),
        ("phase1_by_cell", P1_REFERENCE_ROOT / "evaluation/PROCESSBENCH_BY_CELL.csv"),
        ("phase1_panels", P1_REFERENCE_ROOT / "evaluation/PROCESSBENCH_PANELS.csv"),
        ("phase1_bootstrap", P1_REFERENCE_ROOT / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz"),
        ("prepared_input_manifest", release / "build_A/localization/inputs/MANIFEST.json"),
        ("processbench_labels", release / "build_A/localization/evaluation/localization_decisions.csv"),
        ("localization_contract", REPO / "spectral_utils/reconstruction_benchmark/localization_contract.py"),
        ("localization_evaluation", REPO / "spectral_utils/reconstruction_benchmark/localization_evaluation.py"),
        ("fixed_application_pipelines", REPO / "spectral_utils/fixed_application_pipelines.py"),
        ("phase1_runner", REPO / "scripts/reasoning_localization/run_phase1_baseline.py"),
    ]
    registry = {
        "schema": "reasoning-localization-phase2-reducer-execution-registry-v1",
        "status": "FROZEN_BEFORE_RUN",
        "phase": "P2",
        "experiment_id": "P2_REDUCER_STUDY",
        "variant_id": REFERENCE,
        "stage_a_order": list(STAGE_A_VARIANTS),
        "release_root": str(release),
        "processbench_cells": list(PB_CELLS),
        "population_id": "current_common_eight_qwen",
        "score_parent": "R1_ENTROPY_TOP5",
        "score_alias_tolerance": 1e-12,
        "threshold_contract": "reconstruct exact R1 model-fold ledgers once; later reducers may not rethreshold",
        "length_strata_contract": "model-fold calibration-only erroneous-row token-length tertiles using numpy linear quantiles",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "frozen_registry_hashes": {
            name: sha256_file(PROGRAM_ROOT / name)
            for name in ("METHOD_REGISTRY.json", "VARIANT_REGISTRY.json", "EXPERIMENT_REGISTRY.json")
        },
        "frozen_sources": [
            {"role": role, "path": str(path), "sha256": sha256_file(path)}
            for role, path in sources
        ],
    }
    target = PHASE_ROOT / f"{REFERENCE}_EXECUTION_REGISTRY.json"
    atomic_write_json(target, registry)
    print(f"frozen {REFERENCE}: {target}")


if __name__ == "__main__":
    main()
