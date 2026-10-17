#!/usr/bin/env python3
"""Freeze the execution registry for the historical H3 head-to-head."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_h3_historical_headtohead as run  # noqa: E402


PROTOCOL = REPO / "docs/experiments/REASONING_LOCALIZATION_03662_H3_HISTORICAL_HEADTOHEAD_V1.md"
EXTERNAL_REPO = Path("/Users/osegev/Desktop/hallucination_detection/.worktrees/reconstruction-science-run-v1")
IDENTITY_KEY = EXTERNAL_REPO / "results/reconstruction_benchmark_v1/private_control/2026-08-24_external_final_answer_v3_opaque/external_final_answer/external-id-v2.key"
S1_REGISTRY = run.PROGRAM / "phase_0/P0_S1_EXECUTION_REGISTRY.json"


def source(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": str(path if path.is_absolute() else path.relative_to(REPO)), "sha256": sha256_file(path)}


def main() -> None:
    if run.REGISTRY.exists():
        raise FileExistsError(run.REGISTRY)
    s1 = json.loads(S1_REGISTRY.read_text(encoding="utf-8"))
    raw = [row for row in s1["raw_sources"] if row["model"] == "qwen3_8b"]
    raw.sort(key=lambda row: row["family"])
    score_sources = []
    for model, family in run.CELLS:
        path = run.candidate_source(model, family)
        score_sources.append(source(path, f"frozen_h3_scores:{model}:{family}"))
    raw_sources = [
        {"role": f"historical_raw_units:{row['family']}", "path": row["path"], "sha256": row["sha256"]}
        for row in raw
    ]
    sources = [
        source(PROTOCOL, "frozen_protocol"),
        source(run.S0_ROWS, "historical_per_question"),
        source(run.S0_VERIFY, "historical_replay_verification"),
        source(run.PROGRAM / "phase_0/p0_s0_historical_replay/P0_S0_POPULATION.json", "historical_population"),
        source(run.PROGRAM / "phase_2/diagnostic/h3_reliability_fusion_v1/SCORE_FREEZE_MANIFEST.json", "qwen_h3_score_manifest"),
        source(run.PROGRAM / "phase_2/transfer/h3_llama4/SCORE_FREEZE_MANIFEST.json", "llama_h3_score_manifest"),
        source(EXTERNAL_REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json", "external_identity_registry"),
        source(EXTERNAL_REPO / "configs/reconstruction_benchmark_v1/populations.json", "population_registry"),
        source(IDENTITY_KEY, "controller_identity_key"),
        *raw_sources,
        *score_sources,
    ]
    payload = {
        "schema": "reasoning-localization-h3-historical-headtohead-execution-v1",
        "status": "FROZEN_BEFORE_RUN", "experiment_id": run.EXPERIMENT,
        "protocol": str(PROTOCOL.relative_to(REPO)), "protocol_sha256": sha256_file(PROTOCOL),
        "runner": str(Path(run.__file__).resolve().relative_to(REPO)), "runner_sha256": sha256_file(Path(run.__file__).resolve()),
        "external_repo": str(EXTERNAL_REPO),
        "external_registry": "configs/reconstruction_benchmark_v1/external_final_answer.json",
        "population_registry": "configs/reconstruction_benchmark_v1/populations.json",
        "identity_key": str(IDENTITY_KEY), "identity_key_sha256": sha256_file(IDENTITY_KEY),
        "raw_unit_sources": [{"family": row["family"], "path": row["path"], "sha256": row["sha256"]} for row in raw],
        "cells": [f"processbench_{family}_{model}" for model, family in run.CELLS],
        "systems": list(run.END_TO_END),
        "bootstrap_draws": run.BOOTSTRAP_DRAWS, "bootstrap_seed": run.SEED,
        "bootstrap_unit": "source question linked across scorer copies within family",
        "primary_contrast": "H3_EQUAL_C8 - HISTORICAL_FINALIST",
        "inferential_null": 0.0, "practical_context_bound": 0.003,
        "expected": {
            "entropy_macro_f1": 0.3614213583669282,
            "finalist_macro_f1": 0.3662328341717007,
            "population_sha256": "d12d651cad9bec326686c2c83070644d22ca058ed57e942f683452050e757a05",
            "n_cells": 8, "n_scorer_rows": 1270, "n_groups": 635,
        },
        "frozen_candidate_rule": "Import exact score artifacts; fit only one H0 threshold per cell on historical calibration labels; H2/H3 copy H0 abstention.",
        "audit_labels_opened_before_score_freeze_in_this_run": False,
        "fresh_confirmation": False, "evidence_status": "RETROSPECTIVE",
        "sources": sources,
    }
    run.REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run.REGISTRY, payload)
    print(json.dumps({"registry": str(run.REGISTRY), "sha256": sha256_file(run.REGISTRY), "runner_sha256": payload["runner_sha256"], "protocol_sha256": payload["protocol_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
