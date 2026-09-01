#!/usr/bin/env python3
"""Register the first bounded Phase-3 compact outer-fusion experiment."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402

EXPERIMENT = "P3_COMPACT_OUTER_FUSION"
PARENT = "P3A_H2_EQUAL_OUTER_REFERENCE"
CANDIDATE = "P3B_H2_OUTER_IU"
FAMILY_SIZE = 3


def main() -> None:
    variants_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variants_path.read_text())
    already_registered = {row["variant_id"] for row in variants["variants"]} & {PARENT, CANDIDATE}
    if already_registered not in (set(), {PARENT, CANDIDATE}):
        raise RuntimeError(f"partial Phase-3 variant registration: {sorted(already_registered)}")
    source = next(row for row in variants["variants"] if row["variant_id"] == "P2D_H2_CLEAN_C7")
    shared = {
        "phase": "P3",
        "execution_status": "PLANNED",
        "decision_status": "PENDING",
        "evidence_status": "DEVELOPMENT",
        "statistical_status": "NOT_EVALUATED",
        "task_ids": ["processbench_first_error"],
        "step_reducer": "mean of largest min(10, |I_s|) token risks",
        "detector": "frozen H0 response detector and five-fold threshold evaluator",
        "supervision": "target-free score fit; labels imported only after score freeze",
        "limitations": "Development-open Qwen population; fresh confirmation is required for promotion.",
    }
    parent = deepcopy(source)
    parent.update(shared)
    parent.update({
        "variant_id": PARENT, "display_name": "P3A H2 equal outer-family reference",
        "display_order": 170, "parent_variant_ids": ["P2D_H2_CLEAN_C7"],
        "role": "exact_fusion_parent", "rankable": False,
        "fusion": "equal mean of four frozen H2 family token-risk curves",
        "novelty": "Exact parent instantiation; no method novelty.",
        "signals": ["entropy level", "entropy dynamics plus C7", "partition energy without level", "top-k distribution"],
        "failure_hypothesis": "Not applicable; this arm must alias H2 exactly.",
    })
    candidate = deepcopy(parent)
    candidate.update({
        "variant_id": CANDIDATE, "display_name": "P3B H2 ordinary outer IU-PCR",
        "display_order": 171, "parent_variant_ids": [PARENT], "role": "compact_outer_fusion",
        "rankable": True, "fusion": "ordinary two-component IU-PCR over four frozen H2 family confidence streams",
        "novelty": "Changes only the outer family weighting while retaining the H2 streams, top-ten reducer, and H0 detector.",
        "failure_hypothesis": "Four-family covariance is too small or dependent for IU-PCR and equal weighting is more stable.",
    })
    if not already_registered:
        variants["variants"].extend([parent, candidate])
        atomic_write_json(variants_path, variants)

    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text())
    existing_experiment = next((row for row in experiments["experiments"] if row["experiment_id"] == EXPERIMENT), None)
    main_p3 = next(row for row in experiments["experiments"] if row["experiment_id"] == "P3_FUSION")
    main_p3["prerequisite"] = "Phase-2 complete; user-approved PROMISING_UNCONFIRMED H0 detector/top-ten development lane"
    main_p3["promotion_gates"][0] = "Only Phase-2 survivors or explicitly user-approved PROMISING_UNCONFIRMED development parents"
    main_p3["candidate_branch_contract"]["status"] = "development amendment active; first ordinary-IU arm registered separately"
    experiment_row = {
        "experiment_id": EXPERIMENT, "display_name": "H2 compact outer-family IU-PCR",
        "phase": "P3", "execution_status": "PLANNED",
        "question": "Does ordinary IU-PCR improve the outer weighting of the four frozen H2 family curves?",
        "prerequisite": "User-authorized development amendment carrying H0 detector and top-ten as PROMISING_UNCONFIRMED",
        "population_ids": ["current_common_eight_qwen"], "task_ids": ["processbench_first_error"],
        "variant_order": [PARENT, CANDIDATE], "registered_comparators": [PARENT, "P2C_F6_TOP10_REFERENCE"],
        "primary_metrics": ["paired_delta_macro_f1"],
        "bootstrap": "20,000 paired whole-source-question draws; Bonferroni family size 3 reserved across Phase-3 fusion mechanisms",
        "multiplicity_family_size": FAMILY_SIZE,
        "promotion_gates": [
            "candidate minus exact parent point delta >= +0.003",
            "multiplicity-valid CI lower > +0.003",
            "candidate also improves H0 point estimate",
            "worst-cell delta >= -0.020 and exact/clean deltas >= -0.010",
            "score freeze precedes label import and parent alias <= 1e-12",
        ],
        "next_variant": CANDIDATE,
        "report_sections": ["p3_parent_fusion", "p3_complexity"],
    }
    if existing_experiment is None:
        experiments["experiments"].append(experiment_row)
        atomic_write_json(experiments_path, experiments)
    elif existing_experiment != experiment_row:
        raise RuntimeError("existing Phase-3 compact experiment differs from frozen registration")

    registry = {
        "schema": "reasoning-localization-p3-compact-fusion-execution-v1",
        "status": "FROZEN_BEFORE_RUN", "experiment_id": EXPERIMENT,
        "variant_id": CANDIDATE, "parent_variant_id": PARENT,
        "release_root": str(p1.DEFAULT_RELEASE.resolve()),
        "cells": list(p2r.PB_CELLS), "family_roster": [
            "entropy_level", "entropy_dynamics_plus_C7", "partition_energy_without_energy_series", "topk_distribution"
        ],
        "outer_fusion": "ordinary IU-PCR; IU_CONFIG; two components",
        "detector": "p1.combine_with_common_detector unchanged",
        "step_reducer": "topk_step_mean k=10",
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "multiplicity_family_size": FAMILY_SIZE,
        "labels_seen_during_fit": False, "targets_accessed_during_fit": False,
        "protocol_path": str((REPO / "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_COMPACT_FUSION_V1.md").resolve()),
        "protocol_sha256": sha256_file(REPO / "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_COMPACT_FUSION_V1.md"),
        "runner_path": str((REPO / "scripts/reasoning_localization/run_phase3_compact_fusion.py").resolve()),
        "runner_sha256": sha256_file(REPO / "scripts/reasoning_localization/run_phase3_compact_fusion.py"),
        "frozen_registry_hashes": {
            "EXPERIMENT_REGISTRY.json": sha256_file(experiments_path),
            "VARIANT_REGISTRY.json": sha256_file(variants_path),
        },
    }
    target = p1.PROGRAM_ROOT / "phase_3/compact_outer_iu/P3B_H2_OUTER_IU_EXECUTION_REGISTRY.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, registry)
    print(json.dumps({"status": "REGISTERED_BEFORE_RESULTS", "experiment": EXPERIMENT, "registry": str(target)}, indent=2))


if __name__ == "__main__":
    main()
