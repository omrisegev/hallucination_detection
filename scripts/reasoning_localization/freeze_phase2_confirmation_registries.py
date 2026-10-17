#!/usr/bin/env python3
"""Freeze both Llama confirmation registries before either result is opened."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_confirmation as runner  # noqa: E402


def main() -> None:
    if any(runner.registry_path(variant).exists() or runner.output_root(variant).exists() for variant in runner.VARIANTS):
        raise FileExistsError("refusing to overwrite an opened confirmation registry or output")
    variants_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    variants = {row["variant_id"]: row for row in json.loads(variants_path.read_text())["variants"]}
    for variant in runner.VARIANTS:
        if variants[variant]["execution_status"] != "PLANNED" or variants[variant]["evidence_status"] != "TRANSFER":
            raise RuntimeError(f"{variant} is not registered and unopened")
    release = p1.DEFAULT_RELEASE.resolve()
    sources = [
        ("confirmation_protocol", REPO / "docs/experiments/REASONING_LOCALIZATION_03662_LLAMA_CONFIRMATION_V1.md"),
        ("anchor_protocol", REPO / "docs/experiments/REASONING_LOCALIZATION_03662_ANCHOR_V1.md"),
        ("variant_registry", variants_path), ("experiment_registry", experiments_path),
        ("phase2_snapshot_manifest", p1.PROGRAM_ROOT / "snapshots/phase_2/REPORT_MANIFEST.json"),
        ("phase2_snapshot_report", p1.PROGRAM_ROOT / "snapshots/phase_2/REPORT.html"),
        ("phase1_top5_freeze", p1.PROGRAM_ROOT / "phase_1/r1_entropy_top5/score_freeze/SCORE_FREEZE_MANIFEST.json"),
        ("phase1_family6_freeze", p1.PROGRAM_ROOT / "phase_1/r2_family6_top5_current/score_freeze/SCORE_FREEZE_MANIFEST.json"),
        ("phase1_iu29_freeze", p1.PROGRAM_ROOT / "phase_1/r3_iu29/score_freeze/SCORE_FREEZE_MANIFEST.json"),
        ("c7_score_freeze", p1.PROGRAM_ROOT / "phase_2/atomic/c7_edis_onset/score_freeze/SCORE_FREEZE_MANIFEST.json"),
        ("c8_score_freeze", p1.PROGRAM_ROOT / "phase_2/atomic/c8_self_innov/score_freeze/SCORE_FREEZE_MANIFEST.json"),
        ("prepared_input_manifest", release / "build_A/localization/inputs/MANIFEST.json"),
        ("processbench_labels", release / "build_A/localization/evaluation/localization_decisions.csv"),
        ("localization_evaluation", REPO / "spectral_utils/reconstruction_benchmark/localization_evaluation.py"),
        ("phase1_runner", REPO / "scripts/reasoning_localization/run_phase1_baseline.py"),
        ("atomic_runner", REPO / "scripts/reasoning_localization/run_phase2_atomic_remaining.py"),
        ("confirmation_runner", Path(runner.__file__).resolve()),
    ]
    frozen = [{"role":role, "path":str(path.resolve()), "sha256":sha256_file(path)} for role,path in sources]
    runner.ROOT.mkdir(parents=True, exist_ok=True)
    for variant in runner.VARIANTS:
        registry = {
            "schema":"reasoning-localization-p2-confirmation-execution-v1",
            "status":"FROZEN_BEFORE_RUN", "phase":"P2C", "experiment_id":"P2_CONFIRMATION_LLAMA4",
            "variant_id":variant, "source_variant_id":runner.SOURCE_VARIANT[variant],
            "release_root":str(release), "llama_cells":list(runner.LLAMA_CELLS),
            "population_id":"current_llama4_scorer_transfer", "primary_family_size":runner.PRIMARY_FAMILY_SIZE,
            "bootstrap_draws":p1.BOOTSTRAP_DRAWS, "bootstrap_seed":p1.BOOTSTRAP_SEED,
            "comparators":[runner.TOP10, runner.TOP5, runner.FAMILY6] + ([runner.IU29_PARENT] if variant == runner.VARIANTS[1] else []),
            "practical_benefit_bound":runner.BENEFIT, "practical_harm_bound":runner.HARM,
            "promotion_worst_cell_bound":runner.PROMOTION_WORST_CELL,
            "hard_worst_cell_bound":runner.HARD_WORST_CELL, "component_regression_bound":runner.COMPONENT_BOUND,
            "evidence_boundary":"TRANSFER, not FRESH_CONFIRMATION",
            "fusion_forbidden":True, "score_contract":variants[variant]["exact_contract"],
            "runner_path":str(Path(runner.__file__).resolve()), "runner_sha256":sha256_file(Path(runner.__file__).resolve()),
            "registry_path":str(runner.registry_path(variant).resolve()), "frozen_sources":frozen,
        }
        atomic_write_json(runner.registry_path(variant), registry)
    print(json.dumps({variant:sha256_file(runner.registry_path(variant)) for variant in runner.VARIANTS}, indent=2))


if __name__ == "__main__":
    main()
