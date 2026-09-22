#!/usr/bin/env python3
"""Create the immutable-before-run execution registries for Phase 1."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization.run_phase1_baseline import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DEFAULT_RELEASE,
    PB_CELLS,
    PRM_CELL,
    PROGRAM_ROOT,
    VARIANTS,
)


def main() -> None:
    phase_root = PROGRAM_ROOT / "phase_1"
    phase_root.mkdir(parents=True, exist_ok=True)
    runner = REPO / "scripts/reasoning_localization/run_phase1_baseline.py"
    release = DEFAULT_RELEASE.resolve()
    sources = [
        ("protocol", REPO / "docs/experiments/REASONING_LOCALIZATION_03662_ANCHOR_V1.md"),
        ("prepared_input_manifest", release / "build_A/localization/inputs/MANIFEST.json"),
        ("evaluation_manifest", release / "build_A/localization/evaluation/MANIFEST.json"),
        ("ab_verification", release / "localization/AB_VERIFICATION.json"),
        ("processbench_labels", release / "build_A/localization/evaluation/localization_decisions.csv"),
        ("prmbench_labels", release / "build_A/localization/evaluation/prmbench_steps.npz"),
        ("localization_contract", REPO / "spectral_utils/reconstruction_benchmark/localization_contract.py"),
        ("localization_evaluation", REPO / "spectral_utils/reconstruction_benchmark/localization_evaluation.py"),
        ("token_local_fusion", REPO / "spectral_utils/token_local_fusion.py"),
        ("fixed_application_pipelines", REPO / "spectral_utils/fixed_application_pipelines.py"),
        ("mindgap_adaptation", REPO / "scripts/gl_liu_v1/localization/evidence_drop.py"),
    ]
    frozen_sources = [
        {"role": role, "path": str(path), "sha256": sha256_file(path)}
        for role, path in sources
    ]
    for variant_id in VARIANTS:
        target = phase_root / f"{variant_id}_EXECUTION_REGISTRY.json"
        output = phase_root / variant_id.lower()
        if target.exists() or output.exists():
            raise FileExistsError(f"refusing to overwrite frozen or executed Phase-1 state: {target}")
        payload = {
            "schema": "reasoning-localization-phase1-execution-registry-v1",
            "status": "FROZEN_BEFORE_RUN",
            "phase": "P1",
            "experiment_id": "P1_BASELINES",
            "variant_id": variant_id,
            "variant_order": list(VARIANTS),
            "release_root": str(release),
            "processbench_cells": list(PB_CELLS),
            "prmbench_cell": PRM_CELL,
            "common_population_contract": "exact build_A prepared rows, row order, step spans, source groups, and model-specific five-fold threshold evaluator",
            "score_contract": "empirical-midrank local step x equal_feature_mean response detector, geometric-mean fusion",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "processbench_practical_benefit_delta": 0.005,
            "processbench_practical_harm_delta": -0.005,
            "directional_comparator": "R3_IU29",
            "frozen_registry_hashes": {
                name: sha256_file(PROGRAM_ROOT / name)
                for name in ("METHOD_REGISTRY.json", "VARIANT_REGISTRY.json", "EXPERIMENT_REGISTRY.json")
            },
            "runner_path": str(runner),
            "runner_sha256": sha256_file(runner),
            "frozen_sources": frozen_sources,
        }
        atomic_write_json(target, payload)
        print(f"frozen {variant_id}: {target}")


if __name__ == "__main__":
    main()
