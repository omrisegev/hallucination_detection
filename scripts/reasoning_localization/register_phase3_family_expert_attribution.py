#!/usr/bin/env python3
"""Register the bounded Phase-3 one-family-at-a-time IU ladder."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402

PROGRAM = REPO / "results/reasoning_localization_03662_v1"
METHOD = "hierarchical_family_expert_attribution"
EXPERIMENT = "P3_HIER_FAMILY_ATTRIBUTION"
VARIANTS = (
    "P3E0_H2_XFIT_EQUAL_REFERENCE",
    "P3E1_DYNAMICS_IU_ONLY",
    "P3E2_PARTITION_IU_ONLY",
    "P3E3_TOPK_IU_ONLY",
    "P3E4_ALL_MULTI_IU_CONTROL",
)


def upsert(rows: list[dict], key: str, row: dict) -> None:
    hits = [index for index, current in enumerate(rows) if current.get(key) == row[key]]
    if len(hits) > 1:
        raise RuntimeError(f"duplicate {key}={row[key]}")
    if hits:
        rows[hits[0]] = row
    else:
        rows.append(row)


def main() -> None:
    methods_path = PROGRAM / "METHOD_REGISTRY.json"
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    upsert(methods["methods"], "method_id", {
        "method_id": METHOD,
        "display_name": "One-family-at-a-time IU expert attribution",
        "problem": "Identify which H2 provenance family, if any, benefits from learned within-family IU compression.",
        "plain_summary": "Changes equal compression to ordinary IU in exactly one multi-view family while every other family and the outer mean remain fixed.",
        "input_operation_output": "frozen H2 member views -> donor-cross-fitted family equal/IU expert -> equal outer family mean -> token risk",
        "novelty": "Factorizes the inconclusive all-family P3C intervention into three atomic family-expert contrasts.",
        "assumptions": ["Within-family views share a target but may differ in label-free reliability.", "A useful IU family expert is stable under grouped donor folds."],
        "limitations": ["Opened development population.", "Tests ordinary IU only; STG, DUFS, B3 and L-SML remain gated follow-ons."],
        "references": ["docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_FAMILY_EXPERT_ATTRIBUTION_V1.md", "P3C_H2_INNER_IU_EQUAL_OUTER"],
    })
    atomic_write_json(methods_path, methods)

    variants_path = PROGRAM / "VARIANT_REGISTRY.json"
    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    parent = deepcopy(next(row for row in variants["variants"] if row["variant_id"] == "P3A_H2_EQUAL_OUTER_REFERENCE"))
    definitions = [
        (VARIANTS[0], "Cross-fitted equal-family H2 control", 177, [], "matched_crossfit_parent", False, "Equal within every family; establishes the donor-only parent."),
        (VARIANTS[1], "IU only inside entropy dynamics", 178, [VARIANTS[0]], "single_family_candidate", True, "Changes only dynamics+C7 compression."),
        (VARIANTS[2], "IU only inside partition energy", 179, [VARIANTS[0]], "single_family_candidate", True, "Changes only partition-energy compression."),
        (VARIANTS[3], "IU only inside top-k distribution", 180, [VARIANTS[0]], "single_family_candidate", True, "Changes only top-k compression."),
        (VARIANTS[4], "IU inside all multi-view families", 181, [VARIANTS[1], VARIANTS[2], VARIANTS[3]], "closure_control", False, "Reconstructs the joint intervention under the same cross-fit contract."),
    ]
    for variant_id, name, order, parents, role, rankable, novelty in definitions:
        row = deepcopy(parent)
        row.update({
            "variant_id": variant_id,
            "display_name": name,
            "display_order": order,
            "phase": "P3",
            "method_id": METHOD,
            "parent_variant_ids": parents or ["P3A_H2_EQUAL_OUTER_REFERENCE"],
            "role": role,
            "rankable": rankable,
            "execution_status": "PLANNED",
            "decision_status": "PENDING",
            "evidence_status": "DEVELOPMENT",
            "statistical_status": "NOT_EVALUATED",
            "fusion": novelty,
            "novelty": novelty,
            "supervision": "five-fold donor-only fit; all score trees frozen before label import",
            "limitations": "Opened Qwen-eight development population; no fresh confirmation or PRMBench transfer from this registration.",
        })
        upsert(variants["variants"], "variant_id", row)
    atomic_write_json(variants_path, variants)

    experiments_path = PROGRAM / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    upsert(experiments["experiments"], "experiment_id", {
        "experiment_id": EXPERIMENT,
        "display_name": "Atomic H2 family-expert attribution",
        "phase": "P3",
        "execution_status": "PLANNED",
        "question": "Which individual H2 multi-view family benefits from ordinary IU compression under a matched donor-cross-fit contract?",
        "prerequisite": "P3C aggregate inner-IU result is inconclusive and cannot attribute family contributions.",
        "population_ids": ["current_common_eight_qwen"],
        "task_ids": ["processbench_first_error"],
        "variant_order": list(VARIANTS),
        "registered_comparators": [VARIANTS[0], "P3A_H2_EQUAL_OUTER_REFERENCE", "P3C_H2_INNER_IU_EQUAL_OUTER"],
        "primary_metrics": ["paired_delta_macro_f1"],
        "bootstrap": "20,000 paired whole-source-question draws; Bonferroni across four E1-E4 minus E0 macro-F1 contrasts",
        "promotion_gates": ["point delta >= +0.003", "simultaneous CI lower > +0.003", "H0 abstention mismatch = 0", "exact delta >= -0.010", "worst cell >= -0.020", "finite stable donor fits"],
        "next_variant": VARIANTS[0],
        "report_sections": ["p3_parent_fusion", "p3_complexity"],
        "protocol": "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_FAMILY_EXPERT_ATTRIBUTION_V1.md",
    })
    fusion = next(row for row in experiments["experiments"] if row["experiment_id"] == "P3_FUSION")
    fusion["next_variant"] = VARIANTS[0]
    atomic_write_json(experiments_path, experiments)
    print(json.dumps({"status": "PLANNED", "experiment": EXPERIMENT, "variants": list(VARIANTS)}, indent=2))


if __name__ == "__main__":
    main()
