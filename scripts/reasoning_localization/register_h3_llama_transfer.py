#!/usr/bin/env python3
"""Freeze the H3-equal four-Llama scorer-transfer contract before execution."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_h3_llama_transfer as run  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402


PHASE1_DECISIONS = (
    p1.PROGRAM_ROOT
    / "phase_1/r2_family6_top5_current/evaluation/PROCESSBENCH_DECISIONS.csv"
)
AUDIT = p1.PROGRAM_ROOT / "phase_2/transfer/FRESH_PROCESSBENCH_POPULATION_AUDIT.json"
RELEASE = Path(
    "/Users/osegev/Desktop/hallucination_detection/.worktrees/"
    "reconstruction-science-run-v1/results/reconstruction_benchmark_v1/"
    "releases/2026-08-24_localization_v1"
)


def population_audit() -> dict[str, object]:
    groups: dict[str, set[str]] = {model: set() for model in p1.MODELS}
    with PHASE1_DECISIONS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[row["model_id"]].add(row["group_id"])
    sizes = {model: len(values) for model, values in groups.items()}
    overlap = {}
    models = list(p1.MODELS)
    for i, left in enumerate(models):
        for right in models[i + 1 :]:
            overlap[f"{left}__{right}"] = {
                "intersection": len(groups[left] & groups[right]),
                "left_only": len(groups[left] - groups[right]),
                "right_only": len(groups[right] - groups[left]),
            }
    payload = {
        "schema": "reasoning-localization-fresh-processbench-audit-v1",
        "status": "COMPLETE",
        "source": str(PHASE1_DECISIONS.relative_to(REPO)),
        "source_sha256": sha256_file(PHASE1_DECISIONS),
        "group_counts": sizes,
        "pairwise_overlap": overlap,
        "verdict": "NO_FRESH_LOCAL_PROCESSBENCH_POPULATION",
        "interpretation": (
            "Qwen3-4B, Qwen3-8B, and Llama-3.1-8B are scorer copies of the exact "
            "same 3,400 source questions. Llama evaluation is scorer-family transfer "
            "on previously opened questions, never fresh confirmation."
        ),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(AUDIT, payload)
    return payload


def register_variants() -> None:
    path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    registry = json.loads(path.read_text())
    ids = {row["variant_id"] for row in registry["variants"]}
    if ids & set(run.ARMS):
        raise RuntimeError("H3 Llama transfer variants already registered")
    common = {
        "phase": "P2E",
        "method_id": "fusion_selection",
        "role": "scorer_family_transfer",
        "execution_status": "PLANNED",
        "decision_status": "PENDING",
        "statistical_status": "NOT_EVALUATED",
        "evidence_status": "TRANSFER",
        "rankable": True,
        "task_ids": ["processbench_first_error"],
        "access_tier": "gray_box_single_pass",
        "step_reducer": "top-ten mean over the frozen token-risk curve",
        "detector": "one grouped Llama cross-fitted H0 threshold; candidate arms copy H0 abstention",
        "supervision": "target-free scores; labels only after freeze for the previously registered evaluator",
        "causal_validity": "completed-trace localization; not an early-detection claim",
        "limitations": (
            "The 3,400 questions and their labels were already opened in Phase 1. "
            "This can test scorer-family transfer but cannot confirm or promote H3."
        ),
    }
    registry["variants"].extend(
        [
            {
                **common,
                "variant_id": run.H0,
                "display_name": "H0 family6/top-ten — Llama transfer reference",
                "display_order": 147,
                "parent_variant_ids": ["P2C_F6_TOP10_REFERENCE"],
                "signals": ["exact five-family H0 token curve", "frozen complete-answer detector"],
                "transforms": ["equal family mean", "top-ten step reducer"],
                "fusion": "same H0 local/global geometric-rank combination",
                "novelty": "No method novelty; exact scorer-family transfer reference.",
                "failure_hypothesis": "The Qwen development behavior does not transfer to Llama scorer traces.",
                "prior_evidence": "Exact Qwen H0 score alias is required before Llama labels open.",
            },
            {
                **common,
                "variant_id": run.H2,
                "display_name": "H2 compact family edits — Llama transfer",
                "display_order": 148,
                "parent_variant_ids": [run.H0, "P2D_H2_CLEAN_C7"],
                "signals": [
                    "H0 without sampled-token-energy family",
                    "partition family without energy_series",
                    "C7 inserted inside entropy dynamics",
                ],
                "transforms": ["fixed family/view removal", "frozen C7 onset insertion"],
                "fusion": "equal family mean; H0 abstention copied exactly",
                "novelty": "Tests the exact H2 cleanup across scorer family without retuning.",
                "failure_hypothesis": "The favorable Qwen point estimates were scorer-specific.",
                "prior_evidence": "H2 gained +0.00983 F1 on Qwen but its simultaneous interval crossed zero.",
            },
            {
                **common,
                "variant_id": run.H3,
                "display_name": "H3 equal C8 reranker — Llama transfer",
                "display_order": 149,
                "parent_variant_ids": [run.H2, "P2D_H3_EQUAL_C8_RERANK"],
                "signals": ["H2 step ranks", "C8 self-innovation step ranks"],
                "transforms": ["within-response rank", "fixed equal 50/50 reranking"],
                "fusion": "0.5 H2 rank + 0.5 C8 rank on H0 non-abstentions only",
                "novelty": "Tests the frozen role separation across scorer family: H0 detects, H3 localizes.",
                "failure_hypothesis": "C8 complementarity does not survive a new scorer family.",
                "prior_evidence": (
                    "Qwen development delta +0.01239 with simultaneous CI above zero, "
                    "but below the +0.003 practical lower-bound requirement."
                ),
            },
        ]
    )
    atomic_write_json(path, registry)


def register_experiment(audit: dict[str, object]) -> None:
    path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    registry = json.loads(path.read_text())
    if any(row["experiment_id"] == run.EXPERIMENT for row in registry["experiments"]):
        raise RuntimeError("H3 Llama transfer experiment already registered")
    registry["experiments"].append(
        {
            "experiment_id": run.EXPERIMENT,
            "display_name": "Frozen H3-equal Llama scorer-family transfer",
            "phase": "P2E",
            "execution_status": "PLANNED",
            "question": "Does the frozen H3 role-separated ranking effect transfer from Qwen to Llama scorer traces on the same source questions?",
            "prerequisite": "Exact Qwen H0/H2/H3 score alias at 1e-12 and score freeze before Llama label import.",
            "population_ids": ["current_llama4_scorer_transfer"],
            "task_ids": ["processbench_first_error"],
            "primary_metrics": ["paired_delta_macro_f1"],
            "registered_comparators": [run.H0, run.H2],
            "promotion_gates": [
                "Qwen H0/H2/H3 max absolute score error <= 1e-12",
                "H2/H3 abstention decisions alias H0 exactly",
                "20,000 paired whole-question draws with three-contrast Bonferroni intervals",
                "No promotion because these questions and labels were already opened in Phase 1",
                "Fresh ProcessBench questions remain mandatory for Phase-3 eligibility",
            ],
            "report_sections": ["p2e_llama_absolute", "p2e_llama_forest"],
            "variant_order": list(run.ARMS),
            "bootstrap": "20,000 paired whole-source-question draws; Bonferroni simultaneous macro-F1 intervals across three contrasts",
            "evidence_boundary": audit["interpretation"],
            "fresh_confirmation": False,
            "verdict": "PENDING",
        }
    )
    atomic_write_json(path, registry)


def freeze_execution_registry() -> None:
    input_manifest = RELEASE / "build_A/localization/inputs/MANIFEST.json"
    source_manifest = run.SOURCE_H3 / "SCORE_FREEZE_MANIFEST.json"
    payload = {
        "schema": "reasoning-localization-h3-llama-transfer-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": run.EXPERIMENT,
        "runner": str(Path(run.__file__).resolve().relative_to(REPO)),
        "runner_sha256": sha256_file(Path(run.__file__).resolve()),
        "protocol": "docs/experiments/REASONING_LOCALIZATION_03662_H3_LLAMA_TRANSFER_V1.md",
        "protocol_sha256": sha256_file(REPO / "docs/experiments/REASONING_LOCALIZATION_03662_H3_LLAMA_TRANSFER_V1.md"),
        "arms": list(run.ARMS),
        "cells": list(run.CELLS),
        "release_root": str(RELEASE),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha256_file(input_manifest),
        "source_h3_manifest": str(source_manifest.relative_to(REPO)),
        "source_h3_manifest_sha256": sha256_file(source_manifest),
        "population_audit": str(AUDIT.relative_to(REPO)),
        "population_audit_sha256": sha256_file(AUDIT),
        "bootstrap": {
            "draws": p1.BOOTSTRAP_DRAWS,
            "seed": p1.BOOTSTRAP_SEED,
            "unit": "whole source question",
            "primary_family_size": run.PRIMARY_FAMILY,
        },
        "score_alias_tolerance": 1e-12,
        "labels_seen_before_score_freeze": False,
        "fresh_confirmation": False,
    }
    atomic_write_json(run.REGISTRY, payload)


def main() -> None:
    if run.ROOT.exists() or run.REGISTRY.exists():
        raise FileExistsError("H3 Llama transfer was already frozen or run")
    audit = population_audit()
    register_variants()
    register_experiment(audit)
    freeze_execution_registry()
    print(
        json.dumps(
            {
                "experiment": run.EXPERIMENT,
                "registry_sha256": sha256_file(run.REGISTRY),
                "population_audit_sha256": sha256_file(AUDIT),
                "status": "FROZEN_BEFORE_RUN",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
