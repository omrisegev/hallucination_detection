#!/usr/bin/env python3
"""Apply the score-free Phase-1 task-specialist decision amendment."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402


ROOT = REPO / "results/reasoning_localization_03662_v1"
VARIANT = "R2_FAMILY6_TOP5_CURRENT"
INCUMBENT = "R3_IU29"


def main() -> None:
    source = ROOT / "phase_1/final/P1_CONTRASTS.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pb = next(
        row for row in rows
        if row["left_variant_id"] == VARIANT and row["right_variant_id"] == INCUMBENT
        and row["task_id"] == "processbench_first_error" and row["metric_id"] == "macro_f1"
    )
    prm = next(
        row for row in rows
        if row["left_variant_id"] == VARIANT and row["right_variant_id"] == INCUMBENT
        and row["task_id"] == "prmbench_step_error" and row["metric_id"] == "auroc"
    )
    if float(pb["ci_low"]) <= 0.005 or float(prm["ci_high"]) >= 0.0:
        raise RuntimeError("recorded contrasts do not satisfy the specialist amendment contract")

    variant_path = ROOT / "VARIANT_REGISTRY.json"
    registry = json.loads(variant_path.read_text(encoding="utf-8"))
    variant = next(row for row in registry["variants"] if row["variant_id"] == VARIANT)
    old_decision = variant["decision_status"]
    variant["decision_status"] = "PROCESSBENCH_SPECIALIST"
    atomic_write_json(variant_path, registry)

    claims_path = ROOT / "CLAIMS.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims["claims"] = [
        row for row in claims["claims"] if row["claim_id"] != "CLAIM_P1_R2_TASK_CONFLICT"
    ]
    claims["claims"].append({
        "claim_id": "CLAIM_P1_R2_TASK_CONFLICT",
        "text": "R2_FAMILY6_TOP5_CURRENT is a ProcessBench specialist: it improves first-error macro F1 over IU29 but lowers PRMBench AUROC.",
        "verdict": "PB_SPECIALIST",
        "task_scope": "Phase-1 Qwen-eight ProcessBench development panel and separate PRMBench error-response panel.",
        "evidence_refs": ["PLOT_P1_DELTA_FOREST", "PLOT_P1_PRM_FOREST", "TABLE_CONTRASTS"],
        "worst_case_behavior": "PRMBench AUROC delta is negative with a familywise interval below zero; the method is not a task-general promotion.",
        "claim_boundary": f"ProcessBench delta {float(pb['delta']):+.6f} and PRMBench AUROC delta {float(prm['delta']):+.6f} use separate estimators and are never averaged.",
        "fresh_confirmation_required": True,
    })
    atomic_write_json(claims_path, claims)

    amendment_root = ROOT / "phase_1/amendments/task_specialist_semantics"
    amendment_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(amendment_root / "AMENDMENT.json", {
        "schema": "reasoning-localization-phase1-reporting-amendment-v1",
        "status": "COMPLETE",
        "score_or_metric_changed": False,
        "variant_id": VARIANT,
        "old_decision_status": old_decision,
        "new_decision_status": "PROCESSBENCH_SPECIALIST",
        "processbench_delta": float(pb["delta"]),
        "processbench_familywise_ci": [float(pb["ci_low"]), float(pb["ci_high"])],
        "prmbench_auroc_delta": float(prm["delta"]),
        "prmbench_familywise_ci": [float(prm["ci_low"]), float(prm["ci_high"])],
        "source_artifact": str(source.relative_to(REPO)),
        "source_sha256": sha256_file(source),
        "reason": "The preregistered no-task-average semantics require an explicit specialist label when ProcessBench improves and PRMBench degrades.",
    })
    subprocess.run(
        [sys.executable, str(REPO / "scripts/reasoning_localization/build_reasoning_localization_report.py")],
        cwd=REPO,
        check=True,
    )
    print(json.dumps({
        "variant_id": VARIANT,
        "old": old_decision,
        "new": "PROCESSBENCH_SPECIALIST",
    }, indent=2))


if __name__ == "__main__":
    main()
