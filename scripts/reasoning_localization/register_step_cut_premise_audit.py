#!/usr/bin/env python3
"""Register the development-only STEP-CUT premise verdict.

This script records provenance and closes the planned within-answer temporal
graph arm without inventing a Phase-3 score. It intentionally leaves the
separate feature/family-support arm planned.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "results" / "reasoning_localization_03662_v1"
EXPERIMENTS = PROGRAM / "EXPERIMENT_REGISTRY.json"
VARIANTS = PROGRAM / "VARIANT_REGISTRY.json"

ARTIFACT = (
    "/Users/osegev/Documents/Codex/2026-08-29/"
    "referenced-chatgpt-conversation-this-is-an/outputs/"
    "step_cut_exploratory_v1/STEP_CUT_EXPLORATORY.html"
)
ARTIFACT_SHA256 = (
    "b40b7791b55c8d3d2405b15eba5b603564dfae3c921f15deb74dbfbd68cf1bd2"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def exactly_one(rows: list[dict], key: str, value: str) -> dict:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {key}={value}, found {len(matches)}")
    return matches[0]


def main() -> None:
    experiments = load(EXPERIMENTS)
    variants = load(VARIANTS)

    experiment = exactly_one(
        experiments["experiments"], "experiment_id", "P3_STG_GRAPH_TRANSFER"
    )
    experiment.update(
        {
            "branch_status": "FEATURE_SUPPORT_PLANNED__TEMPORAL_GRAPH_CLOSED",
            "temporal_graph_premise_status": "NOT_PASSED",
            "temporal_graph_variant_status": "NOT_RUN_BY_GATE",
            "combined_feature_time_graph_status": "NOT_RUN_BY_GATE",
            "feature_support_status": "PLANNED_SEPARATE_PREMISE",
            "chain_only_status": "DIAGNOSTIC_ONLY_NEW_PROTOCOL_REQUIRED",
            "step_cut_premise_audit": {
                "artifact": ARTIFACT,
                "artifact_sha256": ARTIFACT_SHA256,
                "evidence_status": "DEVELOPMENT",
                "label_boundary": (
                    "scores frozen before label import, but population labels were "
                    "already opened elsewhere"
                ),
                "method": (
                    "five token-local family axes; top-ten within step; temporal "
                    "chain plus mutual 2NN; donor-only bandwidth, scaling and null "
                    "centering; negative-log-conductance boundary score"
                ),
                "qwen_late_error_hit1_vs_length_matched_chance": {
                    "delta": 0.02328,
                    "ci_low": 0.00445,
                    "ci_high": 0.04230,
                },
                "full_graph_vs_chain_only_hit1": {
                    "delta": -0.02329,
                    "ci_low": -0.04004,
                    "ci_high": -0.00689,
                },
                "full_graph_vs_step_permutation_hit1": {
                    "delta": -0.02230,
                    "ci_low": -0.04419,
                    "ci_high": -0.00024,
                },
                "full_graph_vs_random_edges_hit1": {
                    "delta": -0.02297,
                    "ci_low": -0.03541,
                    "ci_high": -0.01043,
                },
                "full_graph_mrr_vs_chance": {
                    "delta": -0.01475,
                    "ci_low": -0.02652,
                    "ci_high": -0.00268,
                },
                "qwen_entropy_graph_fusion_vs_entropy_top10_hit1": {
                    "delta": -0.03342,
                    "ci_low": -0.05499,
                    "ci_high": -0.01309,
                },
                "llama_entropy_graph_fusion_vs_entropy_top10_hit1": {
                    "delta": -0.02793,
                    "ci_low": -0.05136,
                    "ci_high": -0.00493,
                },
                "verdict": (
                    "apparent lift over chance is topology/position bias; graph "
                    "content premise not passed"
                ),
            },
        }
    )

    temporal = exactly_one(
        variants["variants"], "variant_id", "P3G_T1_TEMPORAL_GRAPH"
    )
    temporal.update(
        {
            "execution_status": "NOT_RUN_BY_GATE",
            "decision_status": "NO_PROMOTION",
            "evidence_status": "DEVELOPMENT",
            "statistical_status": "DESCRIPTIVE",
            "rankable": False,
            "causal_validity": "not applicable; branch closed before main execution",
            "prior_evidence": (
                "Development-only STEP-CUT screen: full graph lost to chain-only, "
                "step-permuted features and random edges with grouped-bootstrap "
                "intervals excluding zero; entropy-plus-graph fusion caused "
                "directional Hit@1 harm on Qwen and Llama."
            ),
            "limitations": (
                "The exact Phase-3 arm was not run and receives no numeric score. "
                "The exploratory population was already opened, so the premise "
                "audit is development evidence only."
            ),
            "failure_hypothesis": (
                "Graph conductance primarily captures topology or position rather "
                "than error-localizing feature content."
            ),
        }
    )

    feature = exactly_one(
        variants["variants"], "variant_id", "P3G_F1_STG_FEATURE_SUPPORT"
    )
    feature["limitations"] = (
        "Remains planned under a separate feature-axis premise. The failed "
        "within-answer STEP-CUT graph cannot be used to justify, tune, or combine "
        "this arm; opening still requires an eligible compact survivor roster."
    )
    feature["prior_evidence"] = (
        "Corrected final-answer STG-SU evidence supports only a narrow sparse-"
        "support premise. STEP-CUT closes the temporal graph axis but does not "
        "evaluate sparse support among feature/family blocks."
    )

    write(EXPERIMENTS, experiments)
    write(VARIANTS, variants)


if __name__ == "__main__":
    main()
