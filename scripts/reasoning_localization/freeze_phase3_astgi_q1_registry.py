#!/usr/bin/env python3
"""Freeze the analytic ASTGI-Q1 point-query execution contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402

ROOT = p1.PROGRAM_ROOT / "phase_3/astgi_query_heads"
REGISTRY = ROOT / "P3T_Q1_EXECUTION_REGISTRY_AMENDMENT_V2.json"
EXPERIMENT = "P3_ASTGI_QUERY_HEADS"
VARIANT = "P3T_Q1_POINT_QUERY"
PARENT = "P3A_H2_EQUAL_OUTER_REFERENCE"
CELLS = (
    "processbench_gsm8k_qwen3_4b",
    "processbench_math_qwen3_4b",
    "processbench_olympiadbench_qwen3_4b",
    "processbench_omnimath_qwen3_4b",
    "processbench_gsm8k_qwen3_8b",
    "processbench_math_qwen3_8b",
    "processbench_olympiadbench_qwen3_8b",
    "processbench_omnimath_qwen3_8b",
)


def main() -> None:
    if REGISTRY.exists():
        raise FileExistsError(f"refusing to overwrite frozen registry: {REGISTRY}")
    ROOT.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "reasoning-localization-p3tq1-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": EXPERIMENT,
        "variant_id": VARIANT,
        "parent_variant_id": PARENT,
        "release_root": str(p1.DEFAULT_RELEASE.resolve()),
        "processbench_cells": list(CELLS),
        "population_id": "current_common_eight_qwen",
        "metric": "official_macro_f1",
        "bootstrap_draws": 20000,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "multiplicity_family_size": 3,
        "benefit_delta": 0.003,
        "harm_delta": -0.005,
        "reducer": "top10 mean; same frozen reducer as H2",
        "query": {
            "family_order": [
                "entropy_level",
                "entropy_dynamics_plus_C7",
                "partition_energy_without_energy_series",
                "topk_distribution",
            ],
            "q_onset": [0.2, 0.4, 0.2, 0.2],
            "temperature": 1.0,
            "boundary_gamma": 0.05,
            "formula": "a_t=softmax(z_t/temperature+q_onset); r_t=a_t dot z_t+gamma*(1-u_t)",
            "permutation": "reverse q_onset",
        },
        "controls": [
            "P3T_Q1_MEAN_ALIAS",
            "P3T_Q1_QUERY_PERMUTED",
            "P3T_Q1_NO_BOUNDARY",
        ],
        "primary_contrasts": [
            [VARIANT, PARENT],
            [VARIANT, "P3T_Q1_QUERY_PERMUTED"],
            [VARIANT, "P3T_Q1_NO_BOUNDARY"],
        ],
        "label_access": "labels imported only after score-freeze manifest and hashes verify",
        "causal_validity": "completed-trace point pooling; not early-certified",
        "runner_sha256": sha256_file(Path(__file__).with_name("run_phase3_astgi_q1.py")),
        "protocol": "docs/experiments/REASONING_LOCALIZATION_03662_ASTGI_Q1_EXECUTION_V1.md",
        "supersedes": "P3T_Q1_EXECUTION_REGISTRY.json",
        "amendment_reason": "The prepared input contains intentionally masked tokens outside scored step spans; Q1 assigns them neutral position u=1, while the unchanged step reducer consumes only explicit spans.",
    }
    atomic_write_json(REGISTRY, row)
    print(json.dumps({"status": "FROZEN", "registry": str(REGISTRY), "runner_hash_pending": True}, indent=2))


if __name__ == "__main__":
    main()
