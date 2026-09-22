#!/usr/bin/env python3
"""Freeze all Phase-2C execution registries before the first score opens."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_conditional as runner  # noqa: E402


def main() -> None:
    runner.ROOT.mkdir(parents=True, exist_ok=True)
    sources = [
        ("protocol", REPO / "docs/experiments/REASONING_LOCALIZATION_03662_CONDITIONAL_ABLATION_V1.md"),
        ("variant_registry", p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"),
        ("experiment_registry", p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"),
        ("token_fusion", REPO / "spectral_utils/token_local_fusion.py"),
        ("atomic_transforms", REPO / "scripts/reasoning_localization/run_phase2_atomic_remaining.py"),
    ]
    frozen_sources = [{"role": role, "path": str(path.resolve()), "sha256": sha256_file(path)} for role, path in sources]
    for variant in runner.VARIANTS:
        path = runner.registry_path(variant)
        if path.exists():
            raise FileExistsError(path)
        payload = {
            "schema": "reasoning-localization-p2c-execution-v1", "status": "FROZEN_BEFORE_RUN",
            "experiment_id": "P2_CONDITIONAL_ABLATION", "variant_id": variant,
            "variant_order": list(runner.VARIANTS), "primary_family_size": runner.PRIMARY_FAMILY_SIZE,
            "cells": list(runner.p2r.PB_CELLS), "release_root": str(p1.DEFAULT_RELEASE.resolve()),
            "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED,
            "runner_path": str(Path(runner.__file__).resolve()), "runner_sha256": sha256_file(Path(runner.__file__).resolve()),
            "frozen_sources": frozen_sources,
            "label_boundary": "all 14 score formulations and registries frozen before the first Phase-2C ProcessBench label opens",
        }
        atomic_write_json(path, payload)
    print(json.dumps({"status": "FROZEN_BEFORE_RESULTS", "registries": len(runner.VARIANTS), "runner_sha256": sha256_file(Path(runner.__file__).resolve())}, indent=2))


if __name__ == "__main__":
    main()
