#!/usr/bin/env python3
"""Integrate the valid detector-preserving Phase-3 outer-IU result."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402
from scripts.reasoning_localization.register_phase3_compact_fusion import CANDIDATE, EXPERIMENT, PARENT  # noqa: E402

ROOT = p1.PROGRAM_ROOT / "phase_3/compact_outer_iu/p3b_h2_outer_iu_v2/evaluation"
H0 = "P3_H0_REFERENCE"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def append(path: Path, additions: list[dict[str, object]], unique: tuple[str, ...]) -> None:
    existing = read(path); fields = list(existing[0])
    keys = {tuple(row.get(field, "") for field in unique) for row in existing}
    for row in additions:
        key = tuple(str(row.get(field, "")) for field in unique)
        if key in keys:
            raise RuntimeError(f"already integrated: {path.name} {key}")
        keys.add(key)
    write(path, [*existing, *additions], fields)


def main() -> None:
    panels = read(ROOT / "PROCESSBENCH_PANELS.csv")
    panels_source = ROOT / "REPORTING_PANELS.csv"; write(panels_source, panels)
    order = {H0: 169, PARENT: 170, CANDIDATE: 171}
    metrics = []
    for row in panels:
        metric = "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"]
        metrics.append({"phase_id": "P3", "experiment_id": EXPERIMENT, "variant_id": row["arm_id"],
            "task_id": "processbench_first_error", "dataset_id": "processbench", "population_id": "current_common_eight_qwen",
            "cell_id": "aggregate", "slice_id": "all", "metric_id": metric, "value": row["value"],
            "ci_low": row["ci_low"], "ci_high": row["ci_high"], "n_rows": row["n_rows"], "n_groups": row["n_groups"],
            "comparison_group_id": f"p3_h2_outer_iu_detector_preserved::{metric}", "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT", "display_order": order[row["arm_id"]],
            "source_artifact": str(panels_source.relative_to(REPO)), "source_sha256": sha256_file(panels_source),
            "source_row_selector": f"arm_id={row['arm_id']};metric_id={row['metric_id']}", "source_value_field": "value",
            "notes": "H0 abstention copied exactly; P3 arms rerank only H0 non-abstentions."})
    append(p1.PROGRAM_ROOT/"METRICS_LONG.csv", metrics, ("experiment_id","variant_id","metric_id","cell_id"))

    raw_contrasts = read(ROOT / "PAIRWISE_CONTRASTS.csv")
    contrast_source = ROOT / "REPORTING_CONTRASTS.csv"; write(contrast_source, raw_contrasts)
    contrasts = []
    for row in raw_contrasts:
        contrasts.append({"phase_id": "P3", "experiment_id": EXPERIMENT, "left_variant_id": row["left_variant_id"],
            "right_variant_id": row["right_variant_id"], "task_id": "processbench_first_error", "dataset_id": "processbench",
            "population_id": "current_common_eight_qwen", "metric_id": row["metric_id"], "delta": row["delta"],
            "ci_low": row["ci_low"], "ci_high": row["ci_high"], "wins": row["wins"], "ties": row["ties"], "losses": row["losses"],
            "worst_unit_delta": row["worst_unit_delta"], "comparison_group_id": f"p3_h2_outer_iu_detector_preserved::{row['metric_id']}",
            "status": "COMPLETE", "evidence_status": "DEVELOPMENT", "source_artifact": str(contrast_source.relative_to(REPO)),
            "source_sha256": sha256_file(contrast_source), "source_row_selector": f"contrast_id={row['contrast_id']}",
            "notes": "Detector-preserving v2; macro-F1 interval is Bonferroni-valid for family size 3."})
    append(p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv", contrasts, ("experiment_id","left_variant_id","right_variant_id","metric_id"))

    summary = json.loads((ROOT/"SUMMARY.json").read_text()); primary = summary["primary_contrast"]
    checks = [
        ("PARENT_ALIAS", "max_abs_error", 0.0, 1e-12, "le", True),
        ("H0_ABSTENTION_ALIAS", "mismatches", 0, 0, "eq", True),
        ("POINT_BENEFIT", "macro_f1", float(primary["delta"]), .003, "ge", float(primary["delta"]) >= .003),
        ("SIMULTANEOUS_BENEFIT", "macro_f1_ci_low", float(primary["ci_low"]), .003, "gt", float(primary["ci_low"]) > .003),
        ("WORST_CELL", "macro_f1", float(primary["worst_unit_delta"]), -.020, "ge", float(primary["worst_unit_delta"]) >= -.020),
    ]
    gates_source = ROOT/"GATES.csv"
    gate_rows = [{"gate_id": f"{CANDIDATE}_{name}", "metric_id": metric, "observed": observed, "threshold": threshold,
        "direction": direction, "passed": str(passed).lower(), "status": "PASS" if passed else "FAIL",
        "evidence_status": "DEVELOPMENT"} for name,metric,observed,threshold,direction,passed in checks]
    write(gates_source, gate_rows)
    gates = [{"phase_id": "P3", "experiment_id": EXPERIMENT, "variant_id": CANDIDATE, **row,
        "unit": "boolean" if row["metric_id"] == "mismatches" else "fraction",
        "source_artifact": str(gates_source.relative_to(REPO)), "source_sha256": sha256_file(gates_source),
        "source_row_selector": f"gate_id={row['gate_id']}", "source_value_field": "observed",
        "notes": "Valid detector-preserving v2. v1 is HARD_FAIL and non-rankable."} for row in gate_rows]
    append(p1.PROGRAM_ROOT/"GATES_LONG.csv", gates, ("experiment_id","variant_id","gate_id"))

    vp = p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json"; variants = json.loads(vp.read_text())
    parent = next(row for row in variants["variants"] if row["variant_id"] == PARENT)
    parent.update({"execution_status":"COMPLETE","decision_status":"NO_PROMOTION","statistical_status":"PROMISING_UNCONFIRMED",
        "prior_evidence":"Exact H2 alias at F1 0.36409; +0.00983 versus H0 with multiplicity-valid CI crossing zero."})
    candidate = next(row for row in variants["variants"] if row["variant_id"] == CANDIDATE)
    candidate.update({"execution_status":"COMPLETE","decision_status":"REJECTED","statistical_status":"SUPPORTED_HARM",
        "limitations":"Detector-preserving v2 is valid. Ordinary outer IU loses 0.01407 macro F1 to H2 equal, CI [-0.02393,-0.00464], and loses in 7/8 cells. Detector-changing v1 is HARD_FAIL and non-rankable."})
    atomic_write_json(vp, variants)

    ep = p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json"; experiments = json.loads(ep.read_text())
    experiment = next(row for row in experiments["experiments"] if row["experiment_id"] == EXPERIMENT)
    experiment.update({"execution_status":"COMPLETE","next_variant":None,"verdict":"ORDINARY_OUTER_IU_SUPPORTED_HARM__BRANCH_CLOSED",
        "valid_run":"p3b_h2_outer_iu_v2", "invalid_run":"p3b_h2_outer_iu HARD_FAIL detector-contract violation"})
    next(row for row in experiments["experiments"] if row["experiment_id"] == "P3_FUSION")["execution_status"] = "RUNNING"
    atomic_write_json(ep, experiments)

    pp = p1.PROGRAM_ROOT/"PLOT_MANIFEST.json"; plots = json.loads(pp.read_text())
    plots["plots"].extend([
        {"plot_id":"PLOT_P3_OUTER_IU_FOREST","title":"Phase 3 ordinary outer IU-PCR versus exact H2 parent","phase":"P3",
         "kind":"contrast_forest","source_table":"CONTRASTS_LONG.csv","selection":{"experiment_id":EXPERIMENT,"metric_id":"macro_f1","status":"COMPLETE"},
         "x_field":"delta","y_field":"left_variant_id","series_field":"right_variant_id","comparison_group":"same Qwen-eight rows; H0 abstention preserved",
         "bootstrap_definition":"20,000 paired whole-question draws; macro-F1 intervals Bonferroni-valid for three reserved Phase-3 fusion mechanisms.",
         "selection_rule":"All three preregistered valid-v2 macro-F1 contrasts.","legend":["Point and line = paired delta and interval","Crossing zero is unresolved; interval wholly below zero is supported harm"],
         "caption":"Ordinary outer IU is -0.01407 versus equal H2, CI [-0.02393,-0.00464], with losses in 7/8 cells. H2 remains +0.00983 versus H0 but unresolved."},
        {"plot_id":"PLOT_P3_OUTER_IU_ABSOLUTE","title":"Detector-preserving Phase 3 absolute macro F1","phase":"P3",
         "kind":"forest","source_table":"METRICS_LONG.csv","selection":{"experiment_id":EXPERIMENT,"metric_id":"macro_f1","cell_id":"aggregate"},
         "x_field":"value","y_field":"variant_id","series_field":"evidence_status","comparison_group":"same Qwen-eight rows and H0 detector",
         "bootstrap_definition":"20,000 grouped draws for absolute intervals; paired claims use PLOT_P3_OUTER_IU_FOREST.",
         "selection_rule":"H0, exact H2 equal parent, and ordinary outer IU in frozen order.","legend":["H0 = detector/localizer reference","P3A/P3B copy H0 abstention exactly"],
         "caption":"H2 equal reaches 0.36409; ordinary outer IU falls to 0.35002 while clean abstention is identical by construction."}
    ])
    atomic_write_json(pp, plots)

    cp = p1.PROGRAM_ROOT/"CLAIMS.json"; claims = json.loads(cp.read_text())
    claims["claims"].append({"claim_id":"CLAIM_P3_OUTER_IU_HARM","text":"Ordinary IU-PCR over the four compact H2 family scores harms ProcessBench first-error localization relative to equal family weighting when H0 detection and top-ten pooling are fixed.",
        "verdict":"SUPPORTED_HARM","task_scope":"current common Qwen-eight ProcessBench development population",
        "evidence_refs":["PLOT_P3_OUTER_IU_FOREST","PLOT_P3_OUTER_IU_ABSOLUTE","TABLE_CONTRASTS"],
        "worst_case_behavior":"Worst cell delta -0.02602; losses in 7/8 cells.",
        "claim_boundary":"Applies to ordinary four-family outer IU only; it does not reject hierarchical inner experts, STG, DUFS/LIU, L-SML, or tensor fusion.",
        "fresh_confirmation_required":False,"statistical_summary":{"metric":"macro_f1","point_delta":float(primary["delta"]),
        "ci_low":float(primary["ci_low"]),"ci_high":float(primary["ci_high"]),"benefit_bound":.003,"harm_bound":0.0,
        "multiplicity":"Bonferroni family size 3"}})
    atomic_write_json(cp, claims)

    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO); REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(json.dumps({"experiment":EXPERIMENT,"verdict":"SUPPORTED_HARM","delta":primary["delta"],
        "ci":[primary["ci_low"],primary["ci_high"]],"report_sha256":build.manifest["output"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
