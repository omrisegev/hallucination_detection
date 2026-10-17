#!/usr/bin/env python3
"""Register the atomic top-ten reference and exact C1 SWVar16 contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402
from scripts.reasoning_localization.run_phase1_baseline import PROGRAM_ROOT  # noqa: E402


REFERENCE = "P2A_TOPK10_REFERENCE"
CANDIDATE = "C1_ENT_SW16"


def main() -> None:
    selection_path = PROGRAM_ROOT / "phase_2/P2R_STAGE_A_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "FROZEN_AFTER_STAGE_A"
        or selection.get("development_parent") != "P2R_A_TOPK10"
    ):
        raise RuntimeError("Stage-A top-ten parent is not frozen")

    variant_path = PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(variant_path.read_text(encoding="utf-8"))
    by_id = {row["variant_id"]: row for row in payload["variants"]}
    if REFERENCE in by_id:
        raise RuntimeError("atomic C1 contract is already registered")
    payload["variants"].append({
        "access_tier": "gray_box_single_pass",
        "causal_validity": "token curve causal; completed-step top-ten readout",
        "decision_status": "PENDING",
        "detector": "equal_feature_mean response detector with per-arm grouped five-fold threshold",
        "display_name": "P2A Entropy / step top-ten atomic reference",
        "display_order": 54,
        "evidence_status": "DEVELOPMENT",
        "execution_status": "PLANNED",
        "failure_hypothesis": "The reducer-study gain was specific to the frozen top-five operating threshold.",
        "fusion": "none",
        "limitations": "Selection-opened reducer parent; this calibration reference is not fresh confirmation.",
        "method_id": "entropy_level",
        "novelty": "Re-evaluates the frozen top-ten local score under the per-arm atomic threshold contract and must alias its reducer-study step scores exactly.",
        "parent_variant_ids": ["P2R_A_TOPK10"],
        "phase": "P2",
        "prior_evidence": "Top-ten was the raw-best preregistered Stage-A reducer but missed the practical-benefit CI gate.",
        "rankable": True,
        "role": "atomic_reference",
        "signals": ["token entropy"],
        "statistical_status": "NOT_EVALUATED",
        "step_reducer": "mean of largest min(10, |I_s|) entropy risks",
        "supervision": "score label-free; labels enter only grouped held-fold threshold calibration and evaluation after complete score freeze",
        "task_ids": ["processbench_first_error"],
        "transforms": ["level"],
        "variant_id": REFERENCE,
    })
    c1 = by_id[CANDIDATE]
    if c1["execution_status"] != "PLANNED":
        raise RuntimeError("C1 has already opened")
    c1.update({
        "parent_variant_ids": [REFERENCE],
        "detector": "equal_feature_mean response detector with per-arm grouped five-fold threshold",
        "fusion": "equal mean of within-cell entropy-step and SWVar-step empirical midranks",
        "step_reducer": "apply frozen top-ten separately to entropy and causal SWVar16; then equal step-rank fusion",
        "supervision": "all scores and fusion frozen label-free; labels enter only grouped held-fold threshold calibration and evaluation",
        "causal_validity": "SWVar token transform is prefix-safe and reset per response; localization reducer waits for the completed step",
        "limitations": "Window 16 is historically motivated and frozen, not selected on ProcessBench; step-rank fusion uses the existing within-cell transductive normalization contract.",
        "c1_contract": {
            "entropy_input": "negative mixed-v2 entropy confidence coordinate; affine-equivalent to raw entropy risk",
            "swvar": "population variance ddof=0 over tokens max(0,t-15)..t",
            "warmup": "available prefix; one-token variance equals zero",
            "reset": "every response",
            "channel_reducer": "mean of largest min(10, step_length) values",
            "fusion": "0.5 * empirical_midrank(entropy_step) + 0.5 * empirical_midrank(swvar_step), within cell",
            "response_combination": "geometric mean with equal_feature_mean response empirical midrank",
            "threshold": "separate deterministic grouped five-fold cross-fit per arm after score freeze",
            "comparators": [REFERENCE, "R1_ENTROPY_TOP5"],
            "suffix_invariance": "exact deterministic prefix replay audit",
        },
    })
    atomic_write_json(variant_path, payload)

    experiment_path = PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment = next(
        row for row in experiments["experiments"]
        if row["experiment_id"] == "P2_ATOMIC"
    )
    if experiment["execution_status"] != "PLANNED":
        raise RuntimeError("P2 atomic experiment already opened")
    experiment.update({
        "prerequisite": "P2R Stage-A selection artifact frozen; top-ten is the development parent and top-five remains confirmatory comparator",
        "atomic_reference": REFERENCE,
        "confirmatory_reference": "R1_ENTROPY_TOP5",
        "selected_reducer": "P2R_A_TOPK10",
        "variant_order": [
            "C1_ENT_SW16", "C2_ENT_SWADAPT", "C3_ENT_CCUSUM",
            "C4_ENT_SAMPLED", "C5_ENT_ENERGY", "C6_DSP12",
            "C7_EDIS_ONSET", "C8_SELF_INNOV",
        ],
        "opened_variants": [],
        "opened_primary_comparisons": 0,
        "atomic_contract": {
            "score_freeze": "all reference and candidate scores complete before labels open",
            "threshold": "per-arm deterministic grouped five-fold cross-fit after score freeze; no score parameter refit",
            "reducer": "top-ten fixed for C1-C8 except C7's explicitly registered event onset",
            "comparators": [REFERENCE, "R1_ENTROPY_TOP5"],
            "multiplicity": "Bonferroni across every opened candidate-by-required-comparator primary contrast",
            "transfer": "ProcessBench first; PRMBench remains unopened until a ProcessBench survivor freezes",
        },
        "registered_comparators": [REFERENCE, "R1_ENTROPY_TOP5"],
        "task_ids": ["processbench_first_error"],
        "population_ids": ["current_common_eight_qwen"],
    })
    atomic_write_json(experiment_path, experiments)
    print(json.dumps({
        "status": "REGISTERED_BEFORE_C1",
        "reference": REFERENCE,
        "candidate": CANDIDATE,
    }, indent=2))


if __name__ == "__main__":
    main()
