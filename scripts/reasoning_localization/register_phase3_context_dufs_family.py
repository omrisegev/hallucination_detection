#!/usr/bin/env python3
"""Register the bounded Phase-3 context-conditioned DUFS family ladder."""

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
METHOD = "context_dufs_family_expert"
EXPERIMENT = "P3_CONTEXT_DUFS_FAMILY"
VARIANTS = (
    "P3F0_DYNAMICS_IU_PARENT",
    "P3F1_DYNAMICS_LOCAL_DUFS_LIU",
    "P3F2_DYNAMICS_CONTEXT_DUFS_LIU",
    "P3F3_DYNAMICS_CONTEXT_PERM_CONTROL",
)


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
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    upsert(methods["methods"], "method_id", {
        "method_id": METHOD,
        "display_name": "Context-conditioned DUFS family expert",
        "problem": "Test whether all compact H2 views can define useful donor geometry for one surviving family without entering that family's output weights.",
        "plain_summary": "Learns a DUFS sample graph from either dynamics alone or all H2 views, then applies Laplacian IU only to the dynamics coordinates.",
        "input_operation_output": "donor H2 views -> DUFS gates/sample graph -> dynamics-only LIU solve -> unchanged equal outer family fusion -> token risk",
        "novelty": "Separates the coordinates that define label-free sample geometry from the coordinates allowed to receive family-expert weights.",
        "assumptions": [
            "Outside-family H2 views contain label-free neighbourhood information relevant to dynamics reliability.",
            "Aligned outside-family context beats a within-response circular-shift control.",
        ],
        "limitations": [
            "Opened ProcessBench development population.",
            "A context-conditioned expert is not a pure one-family intervention because other families affect its graph.",
        ],
        "references": [
            "docs/methods/dufs_liu.md",
            "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_CONTEXT_DUFS_FAMILY_V1.md",
            "P3E1_DYNAMICS_IU_ONLY",
        ],
    })
    atomic_write_json(methods_path, methods)

    variants_path = PROGRAM / "VARIANT_REGISTRY.json"
    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    parent = deepcopy(next(row for row in variants["variants"] if row["variant_id"] == "P3E1_DYNAMICS_IU_ONLY"))
    definitions = (
        (VARIANTS[0], "Dynamics IU matched parent", 182, ["P3E1_DYNAMICS_IU_ONLY"], "matched_parent", False,
         "Exact ordinary-IU dynamics parent under the same score-freeze contract."),
        (VARIANTS[1], "Dynamics-local DUFS-LIU", 183, [VARIANTS[0]], "family_local_candidate", True,
         "DUFS gates, graph, and LIU all use only dynamics views."),
        (VARIANTS[2], "All-H2 context DUFS for dynamics", 184, [VARIANTS[1]], "context_candidate", True,
         "All compact H2 views define the graph; only dynamics receives LIU weights."),
        (VARIANTS[3], "Permuted-context DUFS control", 185, [VARIANTS[2]], "negative_control", False,
         "Non-dynamics context is circularly shifted within donor responses before DUFS."),
    )
    for variant_id, name, order, parents, role, rankable, novelty in definitions:
        row = deepcopy(parent)
        row.update({
            "variant_id": variant_id,
            "display_name": name,
            "display_order": order,
            "phase": "P3",
            "method_id": METHOD,
            "parent_variant_ids": parents,
            "role": role,
            "rankable": rankable,
            "execution_status": "PLANNED",
            "decision_status": "PENDING",
            "evidence_status": "DEVELOPMENT",
            "statistical_status": "NOT_EVALUATED",
            "fusion": novelty,
            "novelty": novelty,
            "supervision": "five-fold donor-only DUFS/LIU; held responses projection-only; scores frozen before label import",
            "limitations": "Opened Qwen-eight development population; no labels select gates, graphs, lambda, signs, or variants.",
        })
        upsert(variants["variants"], "variant_id", row)
    atomic_write_json(variants_path, variants)

    experiments_path = PROGRAM / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    upsert(experiments["experiments"], "experiment_id", {
        "experiment_id": EXPERIMENT,
        "display_name": "Dynamics family-local versus all-H2 context DUFS",
        "phase": "P3",
        "execution_status": "PLANNED",
        "question": "Does DUFS help the dynamics expert, and does aligned outside-family H2 context add value beyond dynamics-only geometry?",
        "prerequisite": "P3E1 dynamics IU is promising unconfirmed and passes the bounded method-specific eligibility checks.",
        "population_ids": ["current_common_eight_qwen"],
        "task_ids": ["processbench_first_error"],
        "variant_order": list(VARIANTS),
        "registered_comparators": [VARIANTS[0], VARIANTS[1], VARIANTS[3], "P3E1_DYNAMICS_IU_ONLY"],
        "primary_metrics": ["paired_delta_macro_f1"],
        "bootstrap": "20,000 paired whole-source-question draws; Bonferroni across four frozen macro-F1 contrasts",
        "promotion_gates": [
            "F2-F0 point delta >= +0.003",
            "F2-F0 simultaneous CI lower > +0.003",
            "F2 beats F1 and F3 with simultaneous CI lower > 0 for a contextual-mechanism claim",
            "lambda-zero parent aliases <= 1e-12",
            "H0 abstention mismatch = 0",
            "exact delta >= -0.010",
            "worst cell >= -0.020",
        ],
        "next_variant": VARIANTS[0],
        "report_sections": ["p3_parent_fusion", "p3_complexity"],
        "protocol": "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_CONTEXT_DUFS_FAMILY_V1.md",
    })
    next(row for row in experiments["experiments"] if row["experiment_id"] == "P3_FUSION")["next_variant"] = VARIANTS[0]
    atomic_write_json(experiments_path, experiments)
    print(json.dumps({"status": "PLANNED", "experiment": EXPERIMENT, "variants": list(VARIANTS)}, indent=2))


if __name__ == "__main__":
    main()
