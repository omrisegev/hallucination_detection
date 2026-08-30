#!/usr/bin/env python3
"""Freeze the exact C2 execution inputs before opening results."""

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
from scripts.reasoning_localization import run_phase2_atomic_c2 as c2  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402


def main() -> None:
    if c2.REGISTRY_PATH.exists() or c2.OUTPUT_ROOT.exists():
        raise FileExistsError("refusing to overwrite opened C2")
    variant_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = {row["variant_id"]: row for row in json.loads(variant_path.read_text())["variants"]}
    if variants[c1.CANDIDATE]["execution_status"] != "HARD_FAIL":
        raise RuntimeError("C1 terminal gate is not registered")
    if variants[c2.CANDIDATE]["execution_status"] != "PLANNED" or "c2_contract" not in variants[c2.CANDIDATE]:
        raise RuntimeError("C2 exact contract is not registered")
    experiment_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiment = next(row for row in json.loads(experiment_path.read_text())["experiments"] if row["experiment_id"] == "P2_ATOMIC")
    if experiment.get("opened_variants") != [c1.CANDIDATE] or experiment.get("next_variant") != c2.CANDIDATE:
        raise RuntimeError("atomic sequence is not ready for C2")
    release = p1.DEFAULT_RELEASE.resolve(); runner = Path(c2.__file__).resolve()
    sources = [
        ("protocol", REPO / "docs/experiments/REASONING_LOCALIZATION_03662_ANCHOR_V1.md"),
        ("method_registry", p1.PROGRAM_ROOT / "METHOD_REGISTRY.json"),
        ("variant_registry", variant_path), ("experiment_registry", experiment_path),
        ("c1_run", c1.OUTPUT_ROOT / "RUN_MANIFEST.json"),
        ("c1_registry", c1.REGISTRY_PATH),
        ("stage_a_selection", p1.PROGRAM_ROOT / "phase_2/P2R_STAGE_A_SELECTION.json"),
        ("stage_a_top10_score_freeze", c1.P2R_TOP10_ROOT / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
        ("phase1_top5_cells", c1.P1_TOP5_ROOT / "evaluation/PROCESSBENCH_BY_CELL.csv"),
        ("phase1_top5_bootstrap", c1.P1_TOP5_ROOT / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz"),
        ("prepared_input_manifest", release / "build_A/localization/inputs/MANIFEST.json"),
        ("processbench_labels", release / "build_A/localization/evaluation/localization_decisions.csv"),
        ("localization_contract", REPO / "spectral_utils/reconstruction_benchmark/localization_contract.py"),
        ("localization_evaluation", REPO / "spectral_utils/reconstruction_benchmark/localization_evaluation.py"),
        ("fixed_application_pipelines", REPO / "spectral_utils/fixed_application_pipelines.py"),
        ("phase1_runner", REPO / "scripts/reasoning_localization/run_phase1_baseline.py"),
        ("c1_runner", Path(c1.__file__).resolve()), ("c2_runner", runner),
        ("atomic_tests", REPO / "scripts/test_reasoning_localization_phase2_atomic.py"),
    ]
    registry = {
        "schema":"reasoning-localization-phase2-atomic-c2-execution-registry-v1","status":"FROZEN_BEFORE_RUN",
        "phase":"P2","experiment_id":"P2_ATOMIC","candidate":c2.CANDIDATE,"atomic_reference":c2.REFERENCE,
        "confirmatory_reference":"R1_ENTROPY_TOP5","release_root":str(release),"processbench_cells":list(p2r.PB_CELLS),
        "population_id":"current_common_eight_qwen","fraction":c2.FRACTION,"min_window":c2.MIN_WINDOW,
        "max_window":c2.MAX_WINDOW,"variance_ddof":0,"fusion_weight":c1.FUSION_WEIGHT,
        "bootstrap_draws":p1.BOOTSTRAP_DRAWS,"bootstrap_seed":p1.BOOTSTRAP_SEED,
        "primary_comparison_family_size":c2.PRIMARY_COMPARISON_FAMILY,
        "comparators":[c2.REFERENCE,"R1_ENTROPY_TOP5"],"practical_benefit_bound":c1.BENEFIT,
        "practical_harm_bound":c1.HARM,"component_regression_bound":c1.COMPONENT_BOUND,
        "promotion_worst_cell_bound":c1.PROMOTION_WORST_CELL_BOUND,"hard_worst_cell_bound":c1.HARD_WORST_CELL_BOUND,
        "score_contract":variants[c2.CANDIDATE]["c2_contract"],"runner_path":str(runner),
        "runner_sha256":sha256_file(runner),"registry_path":str(c2.REGISTRY_PATH.resolve()),
        "frozen_sources":[{"role":role,"path":str(path),"sha256":sha256_file(path)} for role,path in sources],
    }
    c2.REGISTRY_PATH.parent.mkdir(parents=True,exist_ok=True); atomic_write_json(c2.REGISTRY_PATH,registry)
    print(json.dumps({"status":registry["status"],"candidate":c2.CANDIDATE,"registry_sha256":sha256_file(c2.REGISTRY_PATH),"runner_sha256":registry["runner_sha256"]},indent=2))


if __name__ == "__main__": main()
