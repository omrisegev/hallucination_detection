#!/usr/bin/env python3
"""Invalidate detector-changing v1 and freeze the detector-preserving v2 registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization.register_phase3_compact_fusion import CANDIDATE, EXPERIMENT, FAMILY_SIZE, PARENT  # noqa: E402

ROOT = p1.PROGRAM_ROOT / "phase_3/compact_outer_iu"


def main() -> None:
    invalid = ROOT / CANDIDATE.lower()
    if not (invalid / "RUN_COMPLETE.json").exists():
        raise RuntimeError("v1 artifact missing")
    atomic_write_json(invalid / "INVALIDATION.json", {
        "schema": "reasoning-localization-p3-compact-invalidation-v1", "status": "HARD_FAIL",
        "reason": "v1 recomputed the clean/error decision from each candidate curve instead of copying the frozen H0 abstention decision",
        "rankable": False, "labels_were_opened": True, "replacement": f"{CANDIDATE.lower()}_v2",
        "v1_run_sha256": sha256_file(invalid / "RUN_COMPLETE.json")})
    protocol = REPO / "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_COMPACT_FUSION_V1.md"
    runner = REPO / "scripts/reasoning_localization/run_phase3_compact_fusion.py"
    registry = {
        "schema": "reasoning-localization-p3-compact-fusion-execution-v2", "status": "FROZEN_BEFORE_RUN",
        "experiment_id": EXPERIMENT, "variant_id": CANDIDATE, "parent_variant_id": PARENT,
        "release_root": str(p1.DEFAULT_RELEASE.resolve()), "cells": list(p2r.PB_CELLS),
        "family_roster": ["entropy_level", "entropy_dynamics_plus_C7", "partition_energy_without_energy_series", "topk_distribution"],
        "outer_fusion": "ordinary IU-PCR; IU_CONFIG; two components", "step_reducer": "topk_step_mean k=10",
        "detector": "freeze H0 combined score, evaluate H0 once, copy its abstention decision, rerank only non-abstentions",
        "parent_alias": "P3A local step scores must match frozen P2D H2 local scores <= 1e-12",
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "multiplicity_family_size": FAMILY_SIZE, "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False, "protocol_path": str(protocol.resolve()),
        "protocol_sha256": sha256_file(protocol), "runner_path": str(runner.resolve()),
        "runner_sha256": sha256_file(runner), "supersedes": "P3B_H2_OUTER_IU_EXECUTION_REGISTRY.json",
        "v1_invalidation_sha256": sha256_file(invalid / "INVALIDATION.json")}
    target = ROOT / "P3B_H2_OUTER_IU_EXECUTION_REGISTRY_AMENDMENT_V2.json"
    atomic_write_json(target, registry)
    print(json.dumps({"status": "V2_FROZEN_BEFORE_RUN", "registry": str(target)}, indent=2))


if __name__ == "__main__":
    main()
