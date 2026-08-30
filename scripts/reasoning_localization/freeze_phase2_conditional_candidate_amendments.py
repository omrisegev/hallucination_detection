#!/usr/bin/env python3
"""Freeze a pre-candidate amendment adding required secondary contrasts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase2_conditional as runner  # noqa: E402


def main() -> None:
    runner_sha = sha256_file(Path(runner.__file__).resolve())
    created = []
    for variant in runner.VARIANTS[1:]:
        original = runner.ROOT / f"{variant}_EXECUTION_REGISTRY.json"
        amended = runner.ROOT / f"{variant}_EXECUTION_REGISTRY_AMENDMENT_V2.json"
        if amended.exists():
            raise FileExistsError(amended)
        payload = json.loads(original.read_text())
        payload.update({
            "schema": "reasoning-localization-p2c-execution-v1",
            "runner_sha256": runner_sha,
            "runner_path": str(Path(runner.__file__).resolve()),
            "amendment_id": "P2C_PRE_CANDIDATE_SECONDARY_CONTRASTS_V2",
            "amends_registry_path": str(original.resolve()),
            "amends_registry_sha256": sha256_file(original),
            "amendment_timing": "after exact-parent reconstruction but before any candidate score or label opened",
            "required_secondary_metrics": ["first_error_exact", "within_one", "clean_abstention_accuracy"],
            "reason": "The V1 runner recorded macro-F1 only; V2 adds preregistered gate metrics without changing any score formulation.",
        })
        atomic_write_json(amended, payload)
        created.append(str(amended))
    print(json.dumps({"status": "FROZEN_BEFORE_FIRST_CANDIDATE", "runner_sha256": runner_sha, "registries": len(created)}, indent=2))


if __name__ == "__main__":
    main()
