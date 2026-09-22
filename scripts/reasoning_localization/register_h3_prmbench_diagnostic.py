#!/usr/bin/env python3
"""Audit local PB inventory and freeze the H2/H3 PRMBench diagnostic."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from scripts.reasoning_localization import run_h3_prmbench_diagnostic as run  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402


SOURCE_REPO = Path("/Users/osegev/Desktop/hallucination_detection")
RELEASE = p1.DEFAULT_RELEASE
AUDIT = p1.PROGRAM_ROOT / "phase_2/transfer/FRESH_PROCESSBENCH_INVENTORY_AUDIT_V2.json"
BOOTSTRAP_SEED = 2026083102


def id_roster(directory: Path, prefix: str) -> tuple[list[str], dict[str, int]]:
    ids, counts = [], {}
    for family in p1.FAMILIES:
        path = directory / f"{prefix}{family}.pkl"
        with path.open("rb") as handle:
            data = pickle.load(handle)
        family_ids = sorted(str(row["id"]) for row in data.values())
        if len(family_ids) != len(set(family_ids)):
            raise RuntimeError(f"duplicate source IDs in {path}")
        counts[family] = len(family_ids)
        ids.extend(family_ids)
    return sorted(ids), counts


def inventory_audit() -> dict[str, Any]:
    trace_root = SOURCE_REPO / "dataset_cache/repgrid"
    rosters: dict[str, list[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    rosters["pb_llama31_8b"], counts["pb_llama31_8b"] = id_roster(
        trace_root / "pb_llama31_8b", "processbench_"
    )
    trace_assets = {}
    for model in (
        "pb_qwen3_4b",
        "pb_qwen3_8b",
        "pb_llama31_8b",
        "pb_qwen3_4b_pilot",
        "pb_qwen3_8b_pilot",
    ):
        trace_assets[model] = {}
        for family in p1.FAMILIES:
            path = trace_root / model / f"processbench_{family}.pkl"
            prefix = path.read_bytes()[:128]
            trace_assets[model][family] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "local_state": (
                    "GIT_LFS_POINTER_NOT_MATERIALIZED"
                    if prefix.startswith(b"version https://git-lfs")
                    else "MATERIALIZED"
                ),
            }
    competitor_specs = {
        "pb_prm_qwen25math7b_full": (
            SOURCE_REPO / "dataset_cache/four_localization/pb_prm_qwen25math7b_full",
            "pb_prm_",
        ),
        "pb_critic_qwen72b_full": (
            SOURCE_REPO / "dataset_cache/four_localization/pb_critic_qwen72b_full",
            "pb_critic_",
        ),
    }
    for name, (directory, prefix) in competitor_specs.items():
        rosters[name], counts[name] = id_roster(directory, prefix)

    canonical = set(rosters["pb_llama31_8b"])
    full_names = (
        "pb_prm_qwen25math7b_full",
        "pb_critic_qwen72b_full",
    )
    overlap = {
        name: {
            "intersection": len(canonical & set(rosters[name])),
            "canonical_only": len(canonical - set(rosters[name])),
            "candidate_only": len(set(rosters[name]) - canonical),
        }
        for name in full_names
    }
    prior_audit_path = (
        p1.PROGRAM_ROOT
        / "phase_2/transfer/FRESH_PROCESSBENCH_POPULATION_AUDIT.json"
    )
    prior_audit = json.loads(prior_audit_path.read_text())
    if len(canonical) != 3400 or any(item["candidate_only"] for item in overlap.values()):
        raise RuntimeError("unexpected ProcessBench source inventory")
    if prior_audit.get("verdict") != "NO_FRESH_LOCAL_PROCESSBENCH_POPULATION":
        raise RuntimeError("prior opaque-group population audit changed")
    payload = {
        "schema": "reasoning-localization-fresh-processbench-inventory-audit-v2",
        "status": "COMPLETE",
        "canonical_source_count": len(canonical),
        "canonical_source_roster_sha256": sha256_bytes(canonical_json_bytes(sorted(canonical))),
        "family_counts": counts,
        "full_roster_overlap": overlap,
        "trace_assets": trace_assets,
        "prior_opaque_group_audit": {
            "path": str(prior_audit_path.relative_to(REPO)),
            "sha256": sha256_file(prior_audit_path),
            "group_counts": prior_audit["group_counts"],
            "pairwise_overlap": prior_audit["pairwise_overlap"],
            "verdict": prior_audit["verdict"],
        },
        "eligible_trace_models": ["qwen3_4b", "qwen3_8b", "llama31_8b"],
        "ineligible_same-row_assets": {
            "qwen25math7b_prm": "supervised competitor predictions; no H2/H3 primitive trace streams",
            "qwen72b_critic": "entropy belongs to generated critique text, not original reasoning-trace tokens",
        },
        "localization_input_manifests": [
            str(RELEASE / "build_A/localization/inputs/MANIFEST.json"),
            str(RELEASE / "build_B/localization/inputs/MANIFEST.json"),
        ],
        "verdict": "NO_FRESH_LOCAL_PROCESSBENCH_SOURCE_IDS",
        "interpretation": (
            "Every full local ProcessBench asset resolves to the same 3,400 official source IDs; "
            "the sealed release independently proves exact opaque-group overlap across all three "
            "eligible scorer models. Additional PRM/critic assets are access- and signal-contract "
            "mismatches rather than fresh H2/H3 confirmation populations."
        ),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(AUDIT, payload)
    return payload


def register_variants() -> None:
    path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(path.read_text())
    if {row["variant_id"] for row in payload["variants"]} & set(run.ARMS):
        raise RuntimeError("PRMBench diagnostic variants already registered")
    common = {
        "phase": "P2F",
        "method_id": "fusion_selection",
        "role": "prmbench_cross_task_diagnostic",
        "execution_status": "PLANNED",
        "decision_status": "PENDING",
        "statistical_status": "NOT_EVALUATED",
        "evidence_status": "TRANSFER",
        "rankable": True,
        "task_ids": ["prmbench_step_error"],
        "access_tier": "gray_box_single_pass",
        "step_reducer": "mean of largest min(10, |I_s|) token risks",
        "detector": "unchanged Phase-1 common response detector; no abstention threshold",
        "supervision": "target-free score fit; frozen ProcessBench-derived method transferred without PRMBench tuning",
        "causal_validity": "completed-response every-step ranking; not an early-detection claim",
        "limitations": (
            "PRMBench labels were opened in Phase 1 and the H2/H3 roster is outcome-selected. "
            "This diagnostic cannot promote a method or satisfy Phase 4."
        ),
    }
    payload["variants"].extend(
        [
            {
                **common,
                "variant_id": run.H0,
                "display_name": "H0 family6/top-ten — PRMBench diagnostic",
                "display_order": 154,
                "parent_variant_ids": ["P2E_H0_FAMILY6_TOP10_LLAMA4", "R2_FAMILY6_TOP5_CURRENT"],
                "signals": ["exact five-family H0 token curve", "common response detector"],
                "transforms": ["equal family mean", "top-ten step reducer", "geometric response/local rank"],
                "fusion": "same H0 local/global combination as Phase 1",
                "novelty": "Exact cross-task diagnostic reference; no method novelty.",
                "failure_hypothesis": "The compact H0 scorer does not rank dense PRMBench step errors.",
                "prior_evidence": "Phase-1 R2 PRMBench AUROC 0.589982 under top-five; exact top-ten H0 has not been evaluated.",
            },
            {
                **common,
                "variant_id": run.H2,
                "display_name": "H2 compact cleanup — PRMBench diagnostic",
                "display_order": 155,
                "parent_variant_ids": [run.H0, "P2E_H2_CLEAN_C7_LLAMA4"],
                "signals": ["H0 without sampled-token energy", "partition without energy_series", "C7 inside entropy dynamics"],
                "transforms": ["fixed family/view removal", "frozen C7 insertion", "same response detector"],
                "fusion": "equal family mean with unchanged response detector",
                "novelty": "Tests the frozen ProcessBench H2 cleanup on dense every-step ranking.",
                "failure_hypothesis": "H2's first-error point gains do not transfer to all-error-step ranking.",
                "prior_evidence": "H2 is raw-best on Llama ProcessBench transfer but remains unconfirmed.",
            },
            {
                **common,
                "variant_id": run.H3,
                "display_name": "H3 equal C8 reranker — PRMBench diagnostic",
                "display_order": 156,
                "parent_variant_ids": [run.H2, "P2E_H3_EQUAL_C8_RERANK_LLAMA4"],
                "signals": ["H2 step ranks", "C8 self-innovation step ranks"],
                "transforms": ["within-response rank", "fixed equal 50/50 reranking", "same response detector"],
                "fusion": "0.5 H2 rank plus 0.5 C8 rank, then the common response detector",
                "novelty": "Tests whether C8's within-one behavior transfers to dense step-error ranking.",
                "failure_hypothesis": "C8 improves near-miss location but not global step ranking.",
                "prior_evidence": "H3 improves Llama within-one but does not beat H2 in primary ProcessBench F1.",
            },
        ]
    )
    atomic_write_json(path, payload)


def register_experiment(audit: dict[str, Any]) -> None:
    path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    payload = json.loads(path.read_text())
    if any(row["experiment_id"] == run.EXPERIMENT for row in payload["experiments"]):
        raise RuntimeError("PRMBench diagnostic experiment already registered")
    payload["experiments"].append(
        {
            "experiment_id": run.EXPERIMENT,
            "display_name": "Frozen H2/H3 PRMBench mechanism diagnostic",
            "phase": "P2F",
            "execution_status": "PLANNED",
            "question": "Does the frozen ProcessBench H0→H2→H3 scoring ladder transfer to PRMBench every-step error ranking without PRMBench tuning?",
            "prerequisite": "Exact Phase-1 H0 PRMBench score alias and exact Qwen ProcessBench H0/H2/H3 aliases before label import.",
            "population_ids": ["prmbench_qwen3_response_v1"],
            "task_ids": ["prmbench_step_error"],
            "primary_metrics": ["paired_delta_auroc"],
            "registered_comparators": [run.H0, run.H2],
            "promotion_gates": [
                "Diagnostic only: cannot waive Phase-3/Phase-4 prerequisites",
                "H0 aliases Phase-1 PRMBench combined scores at <=1e-12",
                "All scores freeze before this run opens PRMBench labels",
                "20,000 paired source_idx grouped draws and three-contrast simultaneous intervals",
                "H3 must beat both H0 and H2 for a positive incremental premise",
                "ProcessBench and PRMBench metrics remain separate",
            ],
            "report_sections": ["p2f_prm_absolute", "p2f_prm_forest", "p2f_prm_family"],
            "variant_order": list(run.ARMS),
            "bootstrap": "20,000 paired whole-source_idx grouped draws; Bonferroni simultaneous intervals across three contrasts per metric",
            "evidence_boundary": "Frozen cross-task transfer on historically opened PRMBench labels; no promotion.",
            "fresh_confirmation": False,
            "inventory_audit_verdict": audit["verdict"],
            "verdict": "PENDING",
        }
    )
    atomic_write_json(path, payload)


def freeze_execution() -> None:
    protocol = REPO / "docs/experiments/REASONING_LOCALIZATION_03662_H3_PRMBENCH_DIAGNOSTIC_V1.md"
    input_manifest = RELEASE / "build_A/localization/inputs/MANIFEST.json"
    label_path = RELEASE / "build_A/localization/evaluation/prmbench_steps.npz"
    phase1_h0 = (
        p1.PROGRAM_ROOT
        / "phase_1/r2_family6_top5_current/score_freeze/cells"
        / p1.PRM_CELL
        / "scores.npz"
    )
    payload = {
        "schema": "reasoning-localization-h3-prmbench-diagnostic-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": run.EXPERIMENT,
        "runner": str(Path(run.__file__).resolve().relative_to(REPO)),
        "runner_sha256": sha256_file(Path(run.__file__).resolve()),
        "protocol": str(protocol.relative_to(REPO)),
        "protocol_sha256": sha256_file(protocol),
        "arms": list(run.ARMS),
        "cell_id": p1.PRM_CELL,
        "release_root": str(RELEASE),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": sha256_file(input_manifest),
        "label_artifact": str(label_path),
        "label_artifact_sha256": sha256_file(label_path),
        "phase1_h0_score_artifact": str(phase1_h0.relative_to(REPO)),
        "phase1_h0_score_sha256": sha256_file(phase1_h0),
        "inventory_audit": str(AUDIT.relative_to(REPO)),
        "inventory_audit_sha256": sha256_file(AUDIT),
        "bootstrap": {"draws": 20_000, "seed": BOOTSTRAP_SEED, "unit": "source_idx", "primary_family_size": run.PRIMARY_FAMILY},
        "practical_bounds": {"benefit": run.BENEFIT_BOUND, "harm": run.HARM_BOUND},
        "expected": {"responses": 6208, "steps": 83280, "error_families": 9, "single_class_families": ["multi_solutions"]},
        "labels_seen_before_score_freeze_in_this_run": False,
        "historically_opened_population": True,
        "phase4_promotion": False,
    }
    atomic_write_json(run.REGISTRY, payload)


def main() -> None:
    if run.ROOT.exists() or run.REGISTRY.exists():
        raise FileExistsError("PRMBench diagnostic was already frozen or run")
    audit = inventory_audit()
    register_variants()
    register_experiment(audit)
    freeze_execution()
    print(
        json.dumps(
            {
                "experiment": run.EXPERIMENT,
                "inventory_verdict": audit["verdict"],
                "registry_sha256": sha256_file(run.REGISTRY),
                "status": "FROZEN_BEFORE_RUN",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
