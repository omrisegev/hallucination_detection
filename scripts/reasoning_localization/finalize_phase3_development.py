#!/usr/bin/env python3
"""Audit and freeze the opened-population Phase-3 development record.

This finalizer never recomputes scores or bootstrap intervals.  It verifies the
existing Steps 340--345 evidence, closes the remaining gated templates, and
binds every Phase-3 artifact into one immutable hash inventory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase3_dynamics_stg_su as p3s  # noqa: E402
from spectral_utils.token_local_fusion import SU_CONFIG  # noqa: E402


REPORT = REPO / "results/reasoning_localization_03662_v1"
PHASE3 = REPORT / "phase_3"
EXPERIMENT_REGISTRY = REPORT / "EXPERIMENT_REGISTRY.json"
VARIANT_REGISTRY = REPORT / "VARIANT_REGISTRY.json"
FREEZE_MANIFEST = PHASE3 / "PHASE3_DEVELOPMENT_FREEZE.json"
BASE_COMMIT = "35a7a3a3dd43698d5878761a4c5e9595e9e59752"
VERDICT = "PHASE3_DEVELOPMENT_CLOSED__NO_PROMOTION"
TOLERANCE = 1e-15

UNEXECUTED_VARIANTS = {
    "P3_EQUAL_FAMILY_OUTER_REFERENCE",
    "P3_HIER_FAMILY_EXPERTS",
    "P3T_T0_FROZEN_PARENT",
    "P3T_T1_DSP_FIRST",
    "P3T_T2_CAUSAL_TEMPORAL",
    "P3T_T3_TWO_AXIS_LOWRANK",
    "P3G_T0_PARENT",
    "P3G_F1_STG_FEATURE_SUPPORT",
    "P3T_Q2_LEARNED_COORD",
    "P3T_Q3_CAUSAL_NEIGHBOR",
    "P3T_Q4_ONE_LAYER",
}

SUMMARY_PATHS = {
    "p3b": PHASE3 / "compact_outer_iu/p3b_h2_outer_iu_v2/evaluation/SUMMARY.json",
    "p3c": PHASE3 / "hier_inner_iu/p3c_h2_inner_iu_equal_outer/evaluation/SUMMARY.json",
    "p3d": PHASE3 / "deployed_upcr_prune_refit/p3d_compact_view_ladder_v1/evaluation/SUMMARY.json",
    "p3e": PHASE3 / "family_expert_attribution/p3e_family_expert_attribution_v1/evaluation/SUMMARY.json",
    "p3f": PHASE3 / "context_dufs_family/p3f_context_dufs_family_v1/evaluation/SUMMARY.json",
    "p3k": PHASE3 / "topk_dufs_control/p3k_topk_local_dufs_v1/evaluation/SUMMARY.json",
    "p3s": PHASE3 / "dynamics_stg_su/p3s_dynamics_stg_su_v1/evaluation/SUMMARY.json",
    "q1": PHASE3 / "astgi_query_heads/p3t_q1_point_query_v1/evaluation/SUMMARY.json",
}

PANEL_PATHS = {
    "p3b": SUMMARY_PATHS["p3b"].with_name("PROCESSBENCH_PANELS.csv"),
    "p3c": SUMMARY_PATHS["p3c"].with_name("PANELS.csv"),
    "p3d": SUMMARY_PATHS["p3d"].with_name("PROCESSBENCH_PANELS.csv"),
    "p3e": SUMMARY_PATHS["p3e"].with_name("PROCESSBENCH_PANELS.csv"),
    "p3f": SUMMARY_PATHS["p3f"].with_name("PROCESSBENCH_PANELS.csv"),
    "p3k": SUMMARY_PATHS["p3k"].with_name("PROCESSBENCH_PANELS.csv"),
    "p3s": SUMMARY_PATHS["p3s"].with_name("PROCESSBENCH_PANELS.csv"),
    "q1": SUMMARY_PATHS["q1"].with_name("PROCESSBENCH_PANELS.csv"),
}

ACTIVE_RUNNER_CONTRACTS = (
    (
        "scripts/reasoning_localization/run_phase3_astgi_q1.py",
        "results/reasoning_localization_03662_v1/phase_3/astgi_query_heads/P3T_Q1_EXECUTION_REGISTRY_AMENDMENT_V2.json",
        "results/reasoning_localization_03662_v1/phase_3/astgi_query_heads/p3t_q1_point_query_v1/score_freeze/SCORE_FREEZE_MANIFEST.json",
    ),
    (
        "scripts/reasoning_localization/run_phase3_compact_fusion.py",
        "results/reasoning_localization_03662_v1/phase_3/compact_outer_iu/P3B_H2_OUTER_IU_EXECUTION_REGISTRY_AMENDMENT_V2.json",
        "results/reasoning_localization_03662_v1/phase_3/compact_outer_iu/p3b_h2_outer_iu_v2/score_freeze/SCORE_FREEZE_MANIFEST.json",
    ),
    (
        "scripts/reasoning_localization/run_phase3_context_dufs_family.py",
        "results/reasoning_localization_03662_v1/phase_3/context_dufs_family/P3F_EXECUTION_REGISTRY.json",
        "results/reasoning_localization_03662_v1/phase_3/context_dufs_family/p3f_context_dufs_family_v1/score_freeze/SCORE_FREEZE_MANIFEST.json",
    ),
    (
        "scripts/reasoning_localization/run_phase3_deployed_upcr_prune_refit.py",
        "results/reasoning_localization_03662_v1/phase_3/deployed_upcr_prune_refit/P3D_COMPACT_VIEW_EXECUTION_REGISTRY.json",
        "results/reasoning_localization_03662_v1/phase_3/deployed_upcr_prune_refit/p3d_compact_view_ladder_v1/score_freeze/SCORE_FREEZE_MANIFEST.json",
    ),
    (
        "scripts/reasoning_localization/run_phase3_dynamics_stg_su.py",
        "results/reasoning_localization_03662_v1/phase_3/dynamics_stg_su/P3S_EXECUTION_REGISTRY_AMENDMENT_V2.json",
        "results/reasoning_localization_03662_v1/phase_3/dynamics_stg_su/p3s_dynamics_stg_su_v1/score_freeze/SCORE_FREEZE_MANIFEST.json",
    ),
    (
        "scripts/reasoning_localization/run_phase3_family_expert_attribution.py",
        "results/reasoning_localization_03662_v1/phase_3/family_expert_attribution/P3E_EXECUTION_REGISTRY.json",
        "results/reasoning_localization_03662_v1/phase_3/family_expert_attribution/p3e_family_expert_attribution_v1/score_freeze/SCORE_FREEZE_MANIFEST.json",
    ),
    (
        "scripts/reasoning_localization/run_phase3_inner_iu.py",
        "results/reasoning_localization_03662_v1/phase_3/hier_inner_iu/P3C_H2_INNER_IU_EQUAL_OUTER_EXECUTION_REGISTRY.json",
        None,
    ),
    (
        "scripts/reasoning_localization/run_phase3_topk_dufs_control.py",
        "results/reasoning_localization_03662_v1/phase_3/topk_dufs_control/P3K_EXECUTION_REGISTRY.json",
        "results/reasoning_localization_03662_v1/phase_3/topk_dufs_control/p3k_topk_local_dufs_v1/score_freeze/SCORE_FREEZE_MANIFEST.json",
    ),
)

FROZEN_PROTOCOL_CONTRACTS = (
    (
        "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_COMPACT_FUSION_V1.P3B."
        "frozen-c5b59c9e513eae4a8d3f68a3dfde8591ea71f8ec4ed0cd013cc10d8da16fa79a.md",
        (
            "results/reasoning_localization_03662_v1/phase_3/compact_outer_iu/"
            "P3B_H2_OUTER_IU_EXECUTION_REGISTRY.json",
            "results/reasoning_localization_03662_v1/phase_3/compact_outer_iu/"
            "P3B_H2_OUTER_IU_EXECUTION_REGISTRY_AMENDMENT_V2.json",
        ),
    ),
    (
        "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_COMPACT_FUSION_V1.P3C."
        "frozen-cf87a9c523b456eb480828879679fd2b8c7a3d8b7cc9d39317ce055f25a57a66.md",
        (
            "results/reasoning_localization_03662_v1/phase_3/hier_inner_iu/"
            "P3C_H2_INNER_IU_EQUAL_OUTER_EXECUTION_REGISTRY.json",
        ),
    ),
)


class Phase3FreezeError(RuntimeError):
    """Raised when the opened development evidence drifts."""


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase3FreezeError(f"missing required artifact: {path.relative_to(REPO)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        if isinstance(observed, bytes) and isinstance(expected, bytes):
            observed_digest = hashlib.sha256(observed).hexdigest()
            expected_digest = hashlib.sha256(expected).hexdigest()
            raise Phase3FreezeError(
                f"{label} drift: expected SHA256 {expected_digest}, observed {observed_digest}"
            )
        raise Phase3FreezeError(f"{label} drift: expected {expected!r}, observed {observed!r}")


def _assert_close(observed: Any, expected: float, label: str) -> None:
    value = float(observed)
    if not math.isfinite(value) or not math.isfinite(expected):
        raise Phase3FreezeError(f"{label} is non-finite: expected {expected!r}, observed {value!r}")
    if abs(value - expected) > TOLERANCE:
        raise Phase3FreezeError(f"{label} drift: expected {expected!r}, observed {value!r}")


def _all_zero(values: Any, label: str) -> None:
    items: Iterable[Any] = values.values() if isinstance(values, Mapping) else (values,)
    if any(float(value) != 0.0 for value in items):
        raise Phase3FreezeError(f"{label} is not an exact zero alias: {values!r}")


def _contrast(summary: Mapping[str, Any], left: str, right: str, metric: str = "macro_f1") -> Mapping[str, Any]:
    candidates = list(summary.get("contrasts", []))
    if not candidates:
        candidates.extend(summary.get("primary_contrasts", []))
    if not candidates and "primary_contrast" in summary:
        candidates.append(summary["primary_contrast"])
    matches = [
        row for row in candidates
        if row.get("left_variant_id") == left
        and row.get("right_variant_id") == right
        and row.get("metric_id") == metric
    ]
    if len(matches) != 1:
        raise Phase3FreezeError(f"expected one contrast {left} vs {right} ({metric}), found {len(matches)}")
    return matches[0]


def _panel_value(panel: Path, arm: str, metric: str = "official_macro_f1") -> float:
    with panel.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("arm_id") == arm and row.get("metric_id") == metric]
    if len(matches) != 1:
        raise Phase3FreezeError(f"expected one panel row {arm}/{metric} in {panel.relative_to(REPO)}")
    return float(matches[0]["value"])


def _assert_contrast(
    summary: Mapping[str, Any], left: str, right: str, expected: tuple[float, float, float], metric: str = "macro_f1"
) -> None:
    row = _contrast(summary, left, right, metric)
    for field, value in zip(("delta", "ci_low", "ci_high"), expected):
        _assert_close(row[field], value, f"{left} vs {right} {metric} {field}")


def audit_evidence() -> None:
    summaries = {name: _load(path) for name, path in SUMMARY_PATHS.items()}
    for name, summary in summaries.items():
        _assert_equal(summary.get("status"), "COMPLETE", f"{name} summary status")
        if "abstention_mismatches" in summary:
            _all_zero(summary["abstention_mismatches"], f"{name} H0 abstention mismatches")
        if "alias_max_errors" in summary:
            _all_zero(summary["alias_max_errors"], f"{name} parent/lambda-zero aliases")

    for frozen_relative, registry_relatives in FROZEN_PROTOCOL_CONTRACTS:
        frozen_path = REPO / frozen_relative
        frozen_sha = sha256_file(frozen_path)
        filename_sha = frozen_path.name.rsplit(".frozen-", 1)[1].removesuffix(".md")
        _assert_equal(filename_sha, frozen_sha, f"{frozen_relative} content address")
        for registry_relative in registry_relatives:
            registry = _load(REPO / registry_relative)
            _assert_equal(registry.get("protocol_sha256"), frozen_sha, f"{registry_relative} protocol SHA")

    for runner_relative, registry_relative, score_manifest_relative in ACTIVE_RUNNER_CONTRACTS:
        runner_path = REPO / runner_relative
        registry_path = REPO / registry_relative
        registry = _load(registry_path)
        runner_sha = sha256_file(runner_path)
        _assert_equal(registry.get("runner_sha256"), runner_sha, f"{registry_relative} runner SHA")
        if score_manifest_relative is not None:
            score_manifest = _load(REPO / score_manifest_relative)
            _assert_equal(score_manifest.get("runner_sha256"), runner_sha, f"{score_manifest_relative} runner SHA")
            _assert_equal(
                score_manifest.get("execution_registry_sha256"),
                sha256_file(registry_path),
                f"{score_manifest_relative} execution registry SHA",
            )

    invalid_registry = _load(PHASE3 / "compact_outer_iu/P3B_H2_OUTER_IU_EXECUTION_REGISTRY.json")
    invalid_manifest = _load(PHASE3 / "compact_outer_iu/p3b_h2_outer_iu/score_freeze/SCORE_FREEZE_MANIFEST.json")
    _assert_equal(invalid_manifest.get("runner_sha256"), invalid_registry.get("runner_sha256"), "invalidated P3B runner lineage")
    _assert_equal(
        invalid_manifest.get("execution_registry_sha256"),
        sha256_file(PHASE3 / "compact_outer_iu/P3B_H2_OUTER_IU_EXECUTION_REGISTRY.json"),
        "invalidated P3B registry lineage",
    )
    _assert_equal(_load(PHASE3 / "compact_outer_iu/p3b_h2_outer_iu/INVALIDATION.json").get("status"), "HARD_FAIL", "P3B invalidation status")

    expected_scores = {
        ("p3b", "P3A_H2_EQUAL_OUTER_REFERENCE"): 0.36409025375978243,
        ("p3b", "P3B_H2_OUTER_IU"): 0.35002108508301566,
        ("p3c", "P3C_H2_INNER_IU_EQUAL_OUTER"): 0.3601667200655385,
        ("p3d", "P3D0_H2_VIEW_FULLPOOL_IU"): 0.3542404565286024,
        ("p3d", "P3D1_H2_VIEW_DEPLOYED_UPCR"): 0.35673991131969185,
        ("p3e", "P3E0_H2_XFIT_EQUAL_REFERENCE"): 0.3642840904432074,
        ("p3e", "P3E1_DYNAMICS_IU_ONLY"): 0.36687633790587715,
        ("p3e", "P3E3_TOPK_IU_ONLY"): 0.3656029652144618,
        ("p3f", "P3F1_DYNAMICS_LOCAL_DUFS_LIU"): 0.3670445265151818,
        ("p3f", "P3F2_DYNAMICS_CONTEXT_DUFS_LIU"): 0.3668706387879822,
        ("p3k", "P3K1_TOPK_LOCAL_DUFS_LIU"): 0.3655383001537459,
        ("p3s", "P3S0_DYNAMICS_IU_PARENT"): 0.36687633790587715,
        ("p3s", "P3S1_DYNAMICS_CANONICAL_SU"): 0.3670927587811277,
        ("p3s", "P3S2_DYNAMICS_STG_SU"): 0.3667618682882361,
        ("q1", "P3T_Q1_POINT_QUERY"): 0.3545837259937046,
    }
    for (panel_id, arm), expected in expected_scores.items():
        _assert_close(_panel_value(PANEL_PATHS[panel_id], arm), expected, f"{arm} macro-F1")

    _assert_contrast(summaries["p3b"], "P3B_H2_OUTER_IU", "P3A_H2_EQUAL_OUTER_REFERENCE", (-0.014069168676766775, -0.02392633928916833, -0.004635347487107384))
    _assert_contrast(summaries["p3c"], "P3C_H2_INNER_IU_EQUAL_OUTER", "P3A_H2_EQUAL_OUTER_REFERENCE", (-0.003923533694243919, -0.012613730862857738, 0.004538820561836029))
    _assert_contrast(summaries["p3d"], "P3D1_H2_VIEW_DEPLOYED_UPCR", "P3D0_H2_VIEW_FULLPOOL_IU", (0.0024994547910894283, -0.008682760918677849, 0.013780672657681344))
    _assert_contrast(summaries["p3e"], "P3E1_DYNAMICS_IU_ONLY", "P3E0_H2_XFIT_EQUAL_REFERENCE", (0.002592247462669728, -0.0018388870156779019, 0.00719432039789189))
    _assert_equal((_contrast(summaries["p3e"], "P3E1_DYNAMICS_IU_ONLY", "P3E0_H2_XFIT_EQUAL_REFERENCE")["wins"], _contrast(summaries["p3e"], "P3E1_DYNAMICS_IU_ONLY", "P3E0_H2_XFIT_EQUAL_REFERENCE")["ties"], _contrast(summaries["p3e"], "P3E1_DYNAMICS_IU_ONLY", "P3E0_H2_XFIT_EQUAL_REFERENCE")["losses"]), (6, 0, 2), "dynamics-IU W/T/L")
    _assert_contrast(summaries["p3f"], "P3F1_DYNAMICS_LOCAL_DUFS_LIU", "P3F0_DYNAMICS_IU_PARENT", (0.00016818860930467583, -0.001466411568946157, 0.0018181909401465037))
    _assert_contrast(summaries["p3f"], "P3F2_DYNAMICS_CONTEXT_DUFS_LIU", "P3F0_DYNAMICS_IU_PARENT", (-5.699117894941708e-06, -0.0015765461162222476, 0.0015862347764893454))
    _assert_contrast(summaries["p3k"], "P3K1_TOPK_LOCAL_DUFS_LIU", "P3K0_TOPK_IU_PARENT", (-6.466506071589606e-05, -0.0016566882118517312, 0.0016234711424794577))
    _assert_contrast(summaries["p3s"], "P3S2_DYNAMICS_STG_SU", "P3S0_DYNAMICS_IU_PARENT", (-0.00011446961764105534, -0.002963579189727652, 0.00269680951930603))
    _assert_contrast(summaries["p3s"], "P3S2_DYNAMICS_STG_SU", "P3S3_DYNAMICS_STG_PERMUTED_SUPPORT", (0.0008842181476292899, -0.002316114196056174, 0.004155605306047398))
    _assert_contrast(summaries["p3s"], "P3S2_DYNAMICS_STG_SU", "P3S4_DYNAMICS_RANDOM_SUPPORT_CONTROL", (-0.00028835734484056186, -0.0028961822694179677, 0.0021804921718379224))
    _assert_equal(summaries["p3s"]["support_mechanism_supported"], False, "STG support mechanism")
    _assert_close(summaries["p3s"]["p3e_parent_alias_max_error"], 0.0, "P3S parent alias")
    _assert_contrast(summaries["q1"], "P3T_Q1_POINT_QUERY", "P3A_H2_EQUAL_OUTER_REFERENCE", (-0.009506527766077855, -0.019618110172464103, 0.0005159166856906234))
    _assert_contrast(summaries["q1"], "P3T_Q1_POINT_QUERY", "P3A_H2_EQUAL_OUTER_REFERENCE", (-0.010644413615866655, -0.02012253158501525, -0.0012103736113640276), "first_error_exact")

    roster = _load(PHASE3 / "deployed_upcr_prune_refit/P3D_COMPACT_VIEW_EXECUTION_REGISTRY.json")
    _assert_equal(roster["labels_opened"], False, "H2 registry labels_opened")
    _assert_equal(roster["n_member_views"], 24, "H2 member count")
    _assert_equal(Counter(roster["member_families"]), Counter({"entropy_level": 1, "entropy_dynamics": 14, "partition_energy": 3, "topk_distribution": 6}), "H2 family roster 1/14/3/6")
    names = set(roster["member_names"])
    for excluded in ("energy_series", "trace_length", "sampled_energy"):
        if excluded in names:
            raise Phase3FreezeError(f"excluded H2 member unexpectedly present: {excluded}")

    stg_registry = _load(PHASE3 / "dynamics_stg_su/P3S_EXECUTION_REGISTRY_AMENDMENT_V2.json")
    expected_stg = {
        "minimum_fold_fraction": p3s.MIN_FOLD_FRACTION,
        "probability_threshold": p3s.PROBABILITY_THRESHOLD,
        "feature_permutation_seed": p3s.FEATURE_PERMUTATION_SEED,
        "random_support_seeds": list(p3s.RANDOM_SUPPORT_SEEDS),
        "runner_sha256": sha256_file(Path(p3s.__file__).resolve()),
        "token_fusion_sha256": sha256_file(REPO / "spectral_utils/token_local_fusion.py"),
        "su_config": dict(SU_CONFIG),
    }
    for key, expected in expected_stg.items():
        _assert_equal(stg_registry.get(key), expected, f"P3S V2 {key}")

    fair_report = (REPO / "results/fair_paper_exact_comparisons_v1/REPORT.md").read_text(encoding="utf-8")
    if "| 3400 | 0.326141 |" not in fair_report:
        raise Phase3FreezeError("canonical 3,400-row ProcessBench record drift")
    with (REPORT / "METRICS_LONG.csv").open(newline="", encoding="utf-8") as handle:
        historical = next(csv.DictReader(handle))
    _assert_equal(historical["variant_id"], "H_STAGE4_FINALIST", "historical audit-anchor variant")
    _assert_close(historical["value"], 0.3662328341717007, "historical Stage-4 audit anchor")
    _assert_equal(historical["status"], "CONTEXT_ONLY", "historical anchor status")


def _project_registries() -> tuple[dict[str, Any], dict[str, Any]]:
    variants = _load(VARIANT_REGISTRY)
    experiments = _load(EXPERIMENT_REGISTRY)

    p3_rows = [row for row in variants["variants"] if row.get("phase") == "P3"]
    _assert_equal(len(p3_rows), 38, "Phase-3 variant count")
    ids = {row["variant_id"] for row in p3_rows}
    if not UNEXECUTED_VARIANTS < ids:
        raise Phase3FreezeError(f"Phase-3 gated roster drift: missing {sorted(UNEXECUTED_VARIANTS - ids)}")
    for row in p3_rows:
        if row["variant_id"] in UNEXECUTED_VARIANTS:
            row.update(
                execution_status="NOT_RUN_BY_GATE",
                decision_status="NO_PROMOTION",
                statistical_status="NOT_EVALUATED",
                rankable=False,
                limitations=(
                    "NOT_RUN_BY_GATE: Phase 3 closed on the opened population without activating this template; "
                    "this is not an evaluated scientific failure."
                ),
            )

    experiment_map = {row["experiment_id"]: row for row in experiments["experiments"]}
    phase3_experiments = [row for row in experiments["experiments"] if row.get("phase") == "P3"]
    _assert_equal(len(phase3_experiments), 11, "Phase-3 experiment count")
    for row in phase3_experiments:
        row["execution_status"] = "COMPLETE"
        row["next_variant"] = None
        row["phase3_closure"] = VERDICT
        row["evidence_boundary"] = (
            "Opened-population retrospective development diagnostics only. Later confidence families were "
            "registered adaptively after earlier outcomes and do not form one joint confirmatory family."
        )
    fusion = experiment_map["P3_FUSION"]
    fusion["verdict"] = VERDICT
    fusion["result_artifact"] = "results/reasoning_localization_03662_v1/phase_3/PHASE3_DEVELOPMENT_FREEZE.json"
    fusion["result_summary"] = (
        "Steps 340-345 preserved; no Phase-3 method promoted. H2 and dynamics-IU remain opened-development "
        "evidence only; two-block, Q2-Q4, PRMBench transfer, and early detection were not run by gate."
    )
    fusion["candidate_branch_contract"]["status"] = (
        "closed at Step 346; unexecuted hierarchy/two-block variants are NOT_RUN_BY_GATE"
    )
    stg = experiment_map["P3_STG_GRAPH_TRANSFER"]
    stg["branch_status"] = "CLOSED__UNEXECUTED_VARIANTS_NOT_RUN_BY_GATE"
    stg["feature_support_status"] = "NOT_RUN_BY_GATE"
    stg["combined_feature_time_graph_status"] = "NOT_RUN_BY_GATE"
    for experiment_id in ("P4_PRMBENCH_TRANSFER", "P5_EARLY_TRANSFER"):
        row = experiment_map[experiment_id]
        row["execution_status"] = "BLOCKED"
        row["next_variant"] = None
        row["block_reason"] = (
            "No Phase-3 ProcessBench survivor or independent fresh-question/population confirmation opened transfer."
        )
    return variants, experiments


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _phase3_files() -> list[dict[str, Any]]:
    rows = []
    relative_paths = subprocess.check_output(
        [
            "git", "ls-files", "-z", "--cached", "--others",
            "--exclude-standard", "--", PHASE3.relative_to(REPO).as_posix(),
        ],
        cwd=REPO,
    ).decode("utf-8").split("\0")[:-1]
    for relative in sorted(relative_paths):
        path = REPO / relative
        if path == FREEZE_MANIFEST:
            continue
        rows.append({
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return rows


def _manifest_payload() -> dict[str, Any]:
    files = _phase3_files()
    tree_digest = hashlib.sha256(_canonical_bytes({"files": files})).hexdigest()
    return {
        "schema": "reasoning-localization-phase3-development-freeze-v1",
        "snapshot_label": "phase_3",
        "source_commit": BASE_COMMIT,
        "verdict": VERDICT,
        "evidence_boundary": "opened-population retrospective development diagnostics; no promotion or fresh confirmation",
        "not_run_by_gate": sorted(UNEXECUTED_VARIANTS),
        "excluded_scopes": ["two-block", "Q2-Q4", "PRMBench transfer", "early detection", "new bootstrap or scoring"],
        "file_count": len(files),
        "phase3_tree_sha256": tree_digest,
        "files": files,
    }


def _verify_manifest() -> dict[str, Any]:
    frozen = _load(FREEZE_MANIFEST)
    expected = _manifest_payload()
    _assert_equal(frozen, expected, "Phase-3 immutable artifact inventory")
    return frozen


def finalize(*, check: bool) -> dict[str, Any]:
    audit_evidence()
    projected_variants, projected_experiments = _project_registries()
    if check:
        _assert_equal(VARIANT_REGISTRY.read_bytes(), _canonical_bytes(projected_variants), "final Variant Registry bytes")
        _assert_equal(EXPERIMENT_REGISTRY.read_bytes(), _canonical_bytes(projected_experiments), "final Experiment Registry bytes")
        manifest = _verify_manifest()
    else:
        if FREEZE_MANIFEST.exists():
            _verify_manifest()
        atomic_write_json(VARIANT_REGISTRY, projected_variants)
        atomic_write_json(EXPERIMENT_REGISTRY, projected_experiments)
        if FREEZE_MANIFEST.exists():
            manifest = _verify_manifest()
        else:
            manifest = _manifest_payload()
            atomic_write_json(FREEZE_MANIFEST, manifest)
            manifest = _verify_manifest()
    return {
        "status": "CHECKED" if check else "FINALIZED",
        "verdict": VERDICT,
        "phase3_variants": 38,
        "phase3_files": manifest["file_count"],
        "phase3_tree_sha256": manifest["phase3_tree_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the final state without writing")
    args = parser.parse_args()
    try:
        print(json.dumps(finalize(check=args.check), sort_keys=True))
        return 0
    except Phase3FreezeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
