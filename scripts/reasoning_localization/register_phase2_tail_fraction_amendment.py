#!/usr/bin/env python3
"""Register the bounded post-hoc top-10% and top-5% reducer amendment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as base  # noqa: E402


ROWS = (
    {
        "variant_id": "P2R_A_TOPQ10_EXPLORATORY",
        "display_name": "P2R-A Top-10%-mean exploratory",
        "display_order": 52,
        "step_reducer": "R_s = mean of largest max(1, ceil(0.10 |I_s|)) token risks",
        "novelty": "A bounded post-hoc diagnostic that turns the nominal q90 upper tail into an actual multi-token mean.",
        "failure_hypothesis": "Ten percent may still over-average long steps or under-average short steps relative to fixed top-ten.",
    },
    {
        "variant_id": "P2R_A_TOPQ05_EXPLORATORY",
        "display_name": "P2R-A Top-5%-mean exploratory",
        "display_order": 53,
        "step_reducer": "R_s = mean of largest max(1, ceil(0.05 |I_s|)) token risks",
        "novelty": "A smaller persistence-aware tail mean between sparse peak reducers and the post-hoc top-10%-mean diagnostic.",
        "failure_hypothesis": "Five percent may collapse toward max/top-three behavior on typical steps.",
    },
)


def main() -> None:
    variant_path = base.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(variant_path.read_text(encoding="utf-8"))
    existing = {row["variant_id"] for row in payload["variants"]}
    registered = existing.intersection(row["variant_id"] for row in ROWS)
    if registered and registered != {row["variant_id"] for row in ROWS}:
        raise RuntimeError("partial tail-fraction amendment registration")
    for spec in ROWS:
        if spec["variant_id"] in existing:
            row = next(
                row for row in payload["variants"]
                if row["variant_id"] == spec["variant_id"]
            )
            if row["execution_status"] != "PLANNED":
                raise RuntimeError("refusing to rewrite an opened amendment")
            row["statistical_status"] = "NOT_EVALUATED"
            continue
        payload["variants"].append({
            "access_tier": "same as frozen Phase-1 reference",
            "causal_validity": "completed-step readout",
            "decision_status": "PENDING",
            "detector": "frozen Phase-1 detector, score, and threshold",
            "display_name": spec["display_name"],
            "display_order": spec["display_order"],
            "evidence_status": "DEVELOPMENT",
            "execution_status": "PLANNED",
            "failure_hypothesis": spec["failure_hypothesis"],
            "fusion": "upstream representation and family fusion frozen",
            "limitations": "Requested after ProcessBench reducer labels were opened; descriptive here and ineligible for promotion without fresh confirmation.",
            "method_id": "step_scoring_reducer",
            "novelty": spec["novelty"],
            "parent_variant_ids": ["P2R_A_TOPK5_REFERENCE"],
            "phase": "P2",
            "prior_evidence": "Post-hoc mechanism follow-up motivated by fixed top-eight/top-ten gains and the failure of single q90/q75 order statistics.",
            "rankable": False,
            "reducer_stage": "A_IDENTITY_AGGREGATION_POSTHOC_AMENDMENT",
            "role": "exploratory_reducer_candidate",
            "signals": ["frozen Phase-1 reference token-risk curve"],
            "statistical_status": "NOT_EVALUATED",
            "step_reducer": spec["step_reducer"],
            "supervision": "no refit or rethresholding; labels only for frozen evaluation; promotion forbidden on current opened population",
            "task_ids": ["processbench_first_error"],
            "transforms": ["identity"],
            "variant_id": spec["variant_id"],
        })
    atomic_write_json(variant_path, payload)

    experiment_path = base.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment = next(
        row for row in experiments["experiments"]
        if row["experiment_id"] == "P2_REDUCER_STUDY"
    )
    order = experiment["reducer_contract"]["stage_a_order"]
    if order[-2:] != list(base.EXPLORATORY_STAGE_A_VARIANTS):
        if order[-1] != "P2R_A_MEDIAN":
            raise RuntimeError("unexpected pre-amendment Stage-A roster")
        order.extend(list(base.EXPLORATORY_STAGE_A_VARIANTS))
    experiment["reducer_contract"]["posthoc_amendment"] = {
        "variants": list(base.EXPLORATORY_STAGE_A_VARIANTS),
        "reason": "User-requested distinction between a single q90/q75 order statistic and a mean over the upper 10% or 5% of token risks.",
        "inference": "Separate descriptive two-contrast Bonferroni family; current ProcessBench data cannot promote either amendment.",
        "fresh_confirmation_required": True,
    }
    atomic_write_json(experiment_path, experiments)
    print(json.dumps({
        "status": "REGISTERED_POST_HOC_EXPLORATORY",
        "variants": list(base.EXPLORATORY_STAGE_A_VARIANTS),
    }, indent=2))


if __name__ == "__main__":
    main()
