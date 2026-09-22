#!/usr/bin/env python3
"""Register the single Phase-3 top-k family-local DUFS control."""

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
METHOD = "topk_local_dufs_control"
EXPERIMENT = "P3_TOPK_LOCAL_DUFS_CONTROL"
VARIANTS = ("P3K0_TOPK_IU_PARENT", "P3K1_TOPK_LOCAL_DUFS_LIU")


def upsert(rows: list[dict], key: str, row: dict) -> None:
    hits = [i for i, current in enumerate(rows) if current.get(key) == row[key]]
    if len(hits) > 1:
        raise RuntimeError(f"duplicate {key}={row[key]}")
    if hits:
        rows[hits[0]] = row
    else:
        rows.append(row)


def main() -> None:
    methods_path = PROGRAM / "METHOD_REGISTRY.json"
    methods = json.loads(methods_path.read_text())
    upsert(methods["methods"], "method_id", {
        "method_id": METHOD,
        "display_name": "Top-k family-local DUFS control",
        "problem": "Test whether DUFS geometry adds value to the only other P3E family with a positive ordinary-IU point estimate.",
        "plain_summary": "Runs DUFS and LIU only on the six top-k views while all other family and decision components remain fixed.",
        "input_operation_output": "six donor top-k views -> DUFS gates/sample graph -> top-k-only LIU -> unchanged equal outer fusion -> token risk",
        "novelty": "A single premise-gated cross-family control after contextual DUFS failed in dynamics.",
        "assumptions": ["Top-k views may contain label-free local geometry not used by ordinary IU."],
        "limitations": ["Opened development population.", "One family-local control; no contextual or method-factorial expansion."],
        "references": [
            "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_TOPK_DUFS_CONTROL_V1.md",
            "P3E3_TOPK_IU_ONLY",
            "P3F1_DYNAMICS_LOCAL_DUFS_LIU",
        ],
    })
    atomic_write_json(methods_path, methods)

    variants_path = PROGRAM / "VARIANT_REGISTRY.json"
    variants = json.loads(variants_path.read_text())
    parent = deepcopy(next(row for row in variants["variants"] if row["variant_id"] == "P3E3_TOPK_IU_ONLY"))
    definitions = (
        (VARIANTS[0], "Top-k IU matched parent", 186, ["P3E3_TOPK_IU_ONLY"], "matched_parent", False, "Exact P3E3 ordinary-IU top-k parent."),
        (VARIANTS[1], "Top-k family-local DUFS-LIU", 187, [VARIANTS[0]], "secondary_control", True, "DUFS gates, graph and LIU use only the six top-k views."),
    )
    for variant_id, name, order, parents, role, rankable, novelty in definitions:
        row = deepcopy(parent)
        row.update({
            "variant_id": variant_id, "display_name": name, "display_order": order,
            "phase": "P3", "method_id": METHOD, "parent_variant_ids": parents,
            "role": role, "rankable": rankable, "execution_status": "PLANNED",
            "decision_status": "PENDING", "evidence_status": "DEVELOPMENT",
            "statistical_status": "NOT_EVALUATED", "fusion": novelty, "novelty": novelty,
            "supervision": "five-fold donor-only DUFS/LIU; held responses projection-only; scores frozen before labels",
            "limitations": "Opened Qwen-eight development population; conditional secondary control only.",
        })
        upsert(variants["variants"], "variant_id", row)
    atomic_write_json(variants_path, variants)

    experiments_path = PROGRAM / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text())
    upsert(experiments["experiments"], "experiment_id", {
        "experiment_id": EXPERIMENT,
        "display_name": "Top-k family-local DUFS secondary control",
        "phase": "P3", "execution_status": "PLANNED",
        "question": "Does family-local DUFS improve the six-view top-k IU expert?",
        "prerequisite": "P3F top-k secondary-control eligibility gate passed without contextual-mechanism support.",
        "population_ids": ["current_common_eight_qwen"], "task_ids": ["processbench_first_error"],
        "variant_order": list(VARIANTS), "registered_comparators": [VARIANTS[0], "P3E3_TOPK_IU_ONLY"],
        "primary_metrics": ["paired_delta_macro_f1"],
        "bootstrap": "20,000 paired whole-source-question draws; one frozen primary contrast",
        "promotion_gates": ["point delta >= +0.003", "CI lower > +0.003", "lambda-zero and P3E3 aliases <= 1e-12", "H0 abstention mismatch = 0", "exact delta >= -0.010", "worst cell >= -0.020"],
        "next_variant": VARIANTS[0], "report_sections": ["p3_parent_fusion", "p3_complexity"],
        "protocol": "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_TOPK_DUFS_CONTROL_V1.md",
    })
    next(row for row in experiments["experiments"] if row["experiment_id"] == "P3_FUSION")["next_variant"] = VARIANTS[0]
    atomic_write_json(experiments_path, experiments)
    print(json.dumps({"status": "PLANNED", "experiment": EXPERIMENT, "variants": list(VARIANTS)}, indent=2))


if __name__ == "__main__":
    main()
