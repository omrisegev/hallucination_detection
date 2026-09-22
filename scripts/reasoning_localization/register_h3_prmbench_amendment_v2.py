#!/usr/bin/env python3
"""Record the V1 pre-label alias failure and freeze the corrected V2 control."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_h3_prmbench_diagnostic as v1  # noqa: E402
from scripts.reasoning_localization import run_h3_prmbench_diagnostic_v2 as v2  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402


FAILURE = v1.ROOT / "PRELABEL_HARD_FAIL.json"
PROTOCOL = REPO / "docs/experiments/REASONING_LOCALIZATION_03662_H3_PRMBENCH_DIAGNOSTIC_AMENDMENT_V2.md"


def record_failure() -> None:
    if FAILURE.exists():
        raise FileExistsError(FAILURE)
    if not (v1.ROOT / "score_freeze").is_dir():
        raise RuntimeError("missing V1 pre-label attempt directory")
    if any((v1.ROOT / "score_freeze").iterdir()):
        raise RuntimeError("V1 unexpectedly wrote frozen scores")
    atomic_write_json(
        FAILURE,
        {
            "schema": "reasoning-localization-h3-prmbench-prelabel-failure-v1",
            "status": "HARD_FAIL",
            "stage": "PRELABEL_PARENT_ALIAS",
            "experiment_id": v1.EXPERIMENT,
            "observed_max_abs_error": 0.23125777605843761,
            "required_max_abs_error": 1e-12,
            "cause": "V1 incorrectly compared a top-ten H0 candidate to the Phase-1 R2 top-five artifact.",
            "scientific_result_opened": False,
            "prmbench_label_artifact_loaded": False,
            "score_freeze_manifest_written": False,
            "variant_execution_effect": "none; evaluated variants remain PLANNED under V2",
            "v1_execution_registry": str(v1.REGISTRY.relative_to(REPO)),
            "v1_execution_registry_sha256": sha256_file(v1.REGISTRY),
            "v1_runner": str(Path(v1.__file__).resolve().relative_to(REPO)),
            "v1_runner_sha256": sha256_file(Path(v1.__file__).resolve()),
            "resolution": "Use a non-rankable top-five control for exact Phase-1 alias; evaluate the unchanged top-ten roster only after that control passes.",
        },
    )


def update_experiment() -> None:
    path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    payload = json.loads(path.read_text())
    matches = [
        row for row in payload["experiments"] if row["experiment_id"] == v1.EXPERIMENT
    ]
    if len(matches) != 1:
        raise RuntimeError("missing PRMBench diagnostic experiment")
    matches[0].update(
        {
            "execution_status": "PLANNED",
            "active_execution_contract": "AMENDMENT_V2",
            "prelabel_attempts": [
                {
                    "attempt": "V1",
                    "status": "HARD_FAIL",
                    "failure_artifact": str(FAILURE.relative_to(REPO)),
                    "failure_artifact_sha256": sha256_file(FAILURE),
                    "labels_opened": False,
                    "scientific_result_opened": False,
                }
            ],
            "verdict": "PENDING_UNDER_PRELABEL_AMENDMENT_V2",
        }
    )
    atomic_write_json(path, payload)


def freeze_v2() -> None:
    if v2.REGISTRY.exists() or v2.ROOT.exists():
        raise FileExistsError("V2 registry or output already exists")
    v1_contract = json.loads(v1.REGISTRY.read_text())
    payload = {
        **v1_contract,
        "schema": "reasoning-localization-h3-prmbench-diagnostic-execution-v2",
        "status": "FROZEN_BEFORE_RUN",
        "runner": str(Path(v2.__file__).resolve().relative_to(REPO)),
        "runner_sha256": sha256_file(Path(v2.__file__).resolve()),
        "protocol": str(PROTOCOL.relative_to(REPO)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "supersedes_execution_registry": str(v1.REGISTRY.relative_to(REPO)),
        "supersedes_execution_registry_sha256": sha256_file(v1.REGISTRY),
        "superseded_v1_failure": str(FAILURE.relative_to(REPO)),
        "superseded_v1_failure_sha256": sha256_file(FAILURE),
        "parent_alias_control": {
            "control": "H0 family6 top-five plus common response detector",
            "source": v1_contract["phase1_h0_score_artifact"],
            "max_abs_error": 1e-12,
            "rankable": False,
        },
        "evaluated_h0_reducer": "top-ten",
    }
    atomic_write_json(v2.REGISTRY, payload)


def main() -> None:
    record_failure()
    update_experiment()
    freeze_v2()
    print(
        json.dumps(
            {
                "v1_failure_sha256": sha256_file(FAILURE),
                "v2_registry_sha256": sha256_file(v2.REGISTRY),
                "status": "FROZEN_BEFORE_RUN",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
