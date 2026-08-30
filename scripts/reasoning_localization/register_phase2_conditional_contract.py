#!/usr/bin/env python3
"""Register the corrected, bounded Phase-2 conditional-contribution roster."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402


PARENT = "P2C_F6_TOP10_REFERENCE"
VARIANTS = (
    PARENT,
    "P2C_F6_MINUS_ENTROPY_LEVEL",
    "P2C_F6_MINUS_ENTROPY_DYNAMICS",
    "P2C_F6_MINUS_SAMPLED_ENERGY",
    "P2C_F6_MINUS_PARTITION_ENERGY",
    "P2C_F6_MINUS_TOPK_DISTRIBUTION",
    "P2C_F6_PLUS_STRUCTURAL_CONTROL",
    "P2C_F6_MINUS_ENTROPY_SWVAR16_VIEW",
    "P2C_F6_MINUS_ENTROPY_CUSUM_VIEW",
    "P2C_F6_MINUS_SAMPLED_LEVEL_VIEW",
    "P2C_F6_MINUS_PARTITION_LEVEL_VIEW",
    "P2C_F6_SWAP_C1_SWVAR16",
    "P2C_F6_PLUS_C7_EDIS_VIEW",
    "P2C_F6_PLUS_C8_OUTER_EXPERT",
)


def main() -> None:
    variants_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(variants_path.read_text())
    rows = payload["variants"]
    by_id = {row["variant_id"]: row for row in rows}
    if any(item in by_id for item in VARIANTS):
        raise RuntimeError("Phase-2 conditional roster already registered")
    source = by_id["R2_FAMILY6_TOP5_CURRENT"]
    descriptions = {
        PARENT: ("Exact current five-family R2 representation with top-ten reducer", "conditional_reference"),
        "P2C_F6_MINUS_ENTROPY_LEVEL": ("Remove the entropy-level family", "family_leave_one_out"),
        "P2C_F6_MINUS_ENTROPY_DYNAMICS": ("Remove the entropy-dynamics family", "family_leave_one_out"),
        "P2C_F6_MINUS_SAMPLED_ENERGY": ("Remove the sampled-token-energy family", "family_leave_one_out"),
        "P2C_F6_MINUS_PARTITION_ENERGY": ("Remove the partition-energy family", "family_leave_one_out"),
        "P2C_F6_MINUS_TOPK_DISTRIBUTION": ("Remove the top-k-distribution family", "family_leave_one_out"),
        "P2C_F6_PLUS_STRUCTURAL_CONTROL": ("Add the zero-weight structural context stream as a sixth equal family", "negative_control"),
        "P2C_F6_MINUS_ENTROPY_SWVAR16_VIEW": ("Remove entropy_sw_var_series inside entropy dynamics", "view_leave_one_out"),
        "P2C_F6_MINUS_ENTROPY_CUSUM_VIEW": ("Remove entropy_cusum_abs_series inside entropy dynamics", "view_leave_one_out"),
        "P2C_F6_MINUS_SAMPLED_LEVEL_VIEW": ("Remove spilled_series inside sampled-token energy", "view_leave_one_out"),
        "P2C_F6_MINUS_PARTITION_LEVEL_VIEW": ("Remove energy_series inside partition energy", "view_leave_one_out"),
        "P2C_F6_SWAP_C1_SWVAR16": ("Replace historical SWVar member with exact C1 available-prefix SWVar16", "formulation_swap"),
        "P2C_F6_PLUS_C7_EDIS_VIEW": ("Insert frozen C7 onset inside entropy dynamics", "conditional_insertion"),
        "P2C_F6_PLUS_C8_OUTER_EXPERT": ("Fuse frozen C8 localizer with the complete five-family parent as an outer expert", "conditional_outer_expert"),
    }
    for order, variant_id in enumerate(VARIANTS, start=130):
        row = deepcopy(source)
        novelty, role = descriptions[variant_id]
        row.update({
            "variant_id": variant_id,
            "display_name": novelty,
            "display_order": order,
            "phase": "P2C",
            "parent_variant_ids": ["R2_FAMILY6_TOP5_CURRENT"] if variant_id == PARENT else [PARENT],
            "role": role,
            "execution_status": "PLANNED",
            "decision_status": "PENDING",
            "evidence_status": "DEVELOPMENT",
            "statistical_status": "NOT_EVALUATED",
            "rankable": variant_id != PARENT,
            "step_reducer": "mean of largest min(10, |I_s|) token risks",
            "novelty": novelty,
            "limitations": "Current ProcessBench population is development-open; fresh confirmation is required for a universal claim.",
            "task_ids": ["processbench_first_error"],
        })
        if variant_id == "P2C_F6_PLUS_C7_EDIS_VIEW":
            row["signals"] = ["five-family parent", "frozen C7 EDIS onset placed inside entropy_dynamics"]
        if variant_id == "P2C_F6_PLUS_C8_OUTER_EXPERT":
            row["signals"] = ["five-family parent token-risk curve", "frozen C8 IU29-plus-self-innovation token-risk curve"]
        rows.append(row)
    atomic_write_json(variants_path, payload)

    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text())
    if any(row["experiment_id"] == "P2_CONDITIONAL_ABLATION" for row in experiments["experiments"]):
        raise RuntimeError("Phase-2 conditional experiment already registered")
    experiments["experiments"].append({
        "experiment_id": "P2_CONDITIONAL_ABLATION",
        "display_name": "Conditional contribution and insertion study",
        "phase": "P2C",
        "execution_status": "PLANNED",
        "question": "Which frozen families, views, or uncertain atomic signals contribute inside the exact current five-family R2 representation?",
        "prerequisite": "C1--C8 atomic screen and Llama transfer complete; corrected executable R2 parent frozen before results.",
        "population_ids": ["current_common_eight_qwen"],
        "task_ids": ["processbench_first_error"],
        "variant_order": list(VARIANTS),
        "registered_comparators": [PARENT],
        "primary_metrics": ["paired_delta_macro_f1"],
        "bootstrap": "20,000 paired whole-source-question draws; Bonferroni simultaneous interval across 13 candidate contrasts",
        "multiplicity_family_size": 13,
        "promotion_gates": [
            "conditional contribution or candidate delta >= +0.003 with simultaneous CI lower > +0.003",
            "at least six of eight cells nonnegative",
            "worst-cell delta >= -0.020 and no hard delta below -0.030",
            "exact-error and clean-abstention deltas each >= -0.010",
            "all provenance, population, alias, and label-firewall checks pass",
        ],
        "structural_correction": "Current R2 averages five non-structural families; structural is a zero-weight retained context stream and is tested only as an insertion control.",
        "c7_c8_boundary": "C7 is inserted only inside entropy_dynamics; C8 is one outer expert and its 58 coordinates cannot be post-selected.",
        "next_variant": PARENT,
        "report_sections": ["p2c_family_diagram", "p2c_delta_forest", "p2c_family_cell_heatmap", "p2c_exact_clean"],
    })
    atomic_write_json(experiments_path, experiments)
    print(json.dumps({"status": "REGISTERED_BEFORE_RESULTS", "variants": list(VARIANTS)}, indent=2))


if __name__ == "__main__":
    main()
