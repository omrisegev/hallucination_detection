#!/usr/bin/env python3
"""Freeze one C3--C8 execution registry against the common batch runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_remaining as runner  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402


DEFAULT_ELIGIBILITY = {
    "C3_ENT_CCUSUM": False, "C4_ENT_SAMPLED": True, "C5_ENT_ENERGY": True,
    "C7_EDIS_ONSET": False, "C8_SELF_INNOV": False,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=runner.VARIANTS, required=True)
    parser.add_argument("--promotion-eligible", choices=("true", "false"))
    parser.add_argument("--eligibility-reason")
    args = parser.parse_args()
    variant = args.variant
    path = runner.registry_path(variant)
    if path.exists() or runner.output_root(variant).exists():
        raise FileExistsError(f"refusing to overwrite opened {variant}")
    variants_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = {row["variant_id"]: row for row in json.loads(variants_path.read_text())["variants"]}
    if variants[variant]["execution_status"] != "PLANNED" or "exact_contract" not in variants[variant]:
        raise RuntimeError(f"{variant} exact contract is not registered and unopened")
    if variant == "C6_DSP12":
        if args.promotion_eligible is None or not args.eligibility_reason:
            raise RuntimeError("C6 requires an explicit frozen parent-eligibility decision")
        eligible = args.promotion_eligible == "true"
        reason = args.eligibility_reason
    else:
        if args.promotion_eligible is not None:
            raise RuntimeError("only C6 accepts an explicit eligibility override")
        eligible = DEFAULT_ELIGIBILITY[variant]
        reason = ({"C3_ENT_CCUSUM":"diagnostic amendment after exact-parent hard failure",
                   "C4_ENT_SAMPLED":"independent preregistered atomic source",
                   "C5_ENT_ENERGY":"independent preregistered atomic source",
                   "C7_EDIS_ONSET":"exploratory adaptation cannot auto-promote",
                   "C8_SELF_INNOV":"diagnostic prior with uncertain ProcessBench and negative PRMBench evidence"})[variant]
    release = p1.DEFAULT_RELEASE.resolve()
    sources = [
        ("protocol", REPO / "docs/experiments/REASONING_LOCALIZATION_03662_ANCHOR_V1.md"),
        ("method_registry", p1.PROGRAM_ROOT / "METHOD_REGISTRY.json"),
        ("variant_registry", variants_path),
        ("experiment_registry", p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"),
        ("stage_a_selection", p1.PROGRAM_ROOT / "phase_2/P2R_STAGE_A_SELECTION.json"),
        ("stage_a_top10_freeze", c1.P2R_TOP10_ROOT / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
        ("phase1_top5_cells", c1.P1_TOP5_ROOT / "evaluation/PROCESSBENCH_BY_CELL.csv"),
        ("phase1_top5_bootstrap", c1.P1_TOP5_ROOT / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz"),
        ("prepared_input_manifest", release / "build_A/localization/inputs/MANIFEST.json"),
        ("processbench_labels", release / "build_A/localization/evaluation/localization_decisions.csv"),
        ("localization_contract", REPO / "spectral_utils/reconstruction_benchmark/localization_contract.py"),
        ("localization_evaluation", REPO / "spectral_utils/reconstruction_benchmark/localization_evaluation.py"),
        ("fixed_application_pipelines", REPO / "spectral_utils/fixed_application_pipelines.py"),
        ("token_local_fusion", REPO / "spectral_utils/token_local_fusion.py"),
        ("upcr", REPO / "spectral_utils/upcr.py"),
        ("phase1_runner", REPO / "scripts/reasoning_localization/run_phase1_baseline.py"),
        ("c1_runner", Path(c1.__file__).resolve()),
        ("batch_runner", Path(runner.__file__).resolve()),
    ]
    if variant == "C8_SELF_INNOV":
        sources.extend([
            ("phase1_r3_registry", p1.PROGRAM_ROOT / "phase_1/R3_IU29_EXECUTION_REGISTRY.json"),
            ("phase1_r3_freeze", p1.PROGRAM_ROOT / "phase_1/r3_iu29/score_freeze/SCORE_FREEZE_MANIFEST.json"),
        ])
    if variant == "C6_DSP12":
        for parent in ("C4_ENT_SAMPLED", "C5_ENT_ENERGY"):
            sources.append((f"{parent.lower()}_run", runner.output_root(parent) / "RUN_MANIFEST.json"))
    registry = {
        "schema":"reasoning-localization-phase2-atomic-remaining-execution-registry-v1",
        "status":"FROZEN_BEFORE_RUN","phase":"P2","experiment_id":"P2_ATOMIC",
        "candidate":variant,"atomic_reference":runner.REFERENCE,"confirmatory_reference":"R1_ENTROPY_TOP5",
        "release_root":str(release),"processbench_cells":list(p2r.PB_CELLS),
        "population_id":"current_common_eight_qwen","topk":runner.TOPK,
        "bootstrap_draws":p1.BOOTSTRAP_DRAWS,"bootstrap_seed":p1.BOOTSTRAP_SEED,
        "primary_comparison_family_size":runner.PRIMARY_COMPARISON_FAMILY,
        "comparators":[runner.REFERENCE,"R1_ENTROPY_TOP5"],
        "promotion_eligible":eligible,"eligibility_reason":reason,
        "practical_benefit_bound":c1.BENEFIT,"practical_harm_bound":c1.HARM,
        "component_regression_bound":c1.COMPONENT_BOUND,
        "promotion_worst_cell_bound":c1.PROMOTION_WORST_CELL_BOUND,
        "hard_worst_cell_bound":c1.HARD_WORST_CELL_BOUND,
        "score_contract":variants[variant]["exact_contract"],
        "runner_path":str(Path(runner.__file__).resolve()),
        "runner_sha256":sha256_file(Path(runner.__file__).resolve()),
        "registry_path":str(path.resolve()),
        "frozen_sources":[{"role":role,"path":str(source),"sha256":sha256_file(source)} for role,source in sources],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, registry)
    print(json.dumps({"status":"FROZEN_BEFORE_RUN","candidate":variant,"promotion_eligible":eligible,
                      "registry_sha256":sha256_file(path),"runner_sha256":registry["runner_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
