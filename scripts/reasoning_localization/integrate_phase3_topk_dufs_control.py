#!/usr/bin/env python3
"""Integrate the completed P3K top-k local DUFS control."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase3_topk_dufs_control as run  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402

K0, K1 = run.VARIANTS
EVAL = run.OUTPUT / "evaluation"
H0_REPORT = "P2C_F6_TOP10_REFERENCE"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    columns = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n"); writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in columns} for row in rows])


def append(path: Path, additions: list[dict[str, Any]], unique: tuple[str, ...]) -> None:
    rows = read(path); fields = list(rows[0]); keys = {tuple(row.get(f, "") for f in unique) for row in rows}
    for row in additions:
        key = tuple(str(row.get(f, "")) for f in unique)
        if key in keys: raise RuntimeError(f"duplicate {key}")
        keys.add(key)
    write(path, [*rows, *additions], fields)


def upsert(rows: list[dict], key: str, row: dict) -> None:
    hits = [i for i, current in enumerate(rows) if current.get(key) == row[key]]
    if len(hits) > 1: raise RuntimeError(f"duplicate {key}")
    if hits: rows[hits[0]] = row
    else: rows.append(row)


def main() -> None:
    panels_path = EVAL / "PROCESSBENCH_PANELS.csv"; panels = read(panels_path)
    orders = {run.H0: 169, K0: 186, K1: 187}
    metrics = []
    for row in panels:
        metric = "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"]
        metrics.append({
            "phase_id":"P3","experiment_id":run.EXPERIMENT,"variant_id":H0_REPORT if row["arm_id"]==run.H0 else row["arm_id"],
            "task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen","cell_id":"aggregate","slice_id":"all",
            "metric_id":metric,"value":row["value"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"n_rows":row["n_rows"],"n_groups":row["n_groups"],
            "comparison_group_id":f"p3k_topk_dufs::{metric}","status":"COMPLETE","evidence_status":"DEVELOPMENT","display_order":orders[row["arm_id"]],"axis_value":"",
            "source_artifact":str(panels_path.relative_to(REPO)),"source_sha256":sha256_file(panels_path),"source_row_selector":f"arm_id={row['arm_id']};metric_id={row['metric_id']}","source_value_field":"value",
            "notes":"Five-fold donor-only top-k DUFS/LIU; other families, H0 and top-ten fixed.",
        })
    append(p1.PROGRAM_ROOT/"METRICS_LONG.csv",metrics,("experiment_id","variant_id","metric_id","cell_id"))

    contrasts_path=EVAL/"PAIRWISE_CONTRASTS.csv"; raw=read(contrasts_path); contrasts=[]
    for row in raw:
        contrasts.append({
            "phase_id":"P3","experiment_id":run.EXPERIMENT,"left_variant_id":row["left_variant_id"],"right_variant_id":row["right_variant_id"],
            "task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen","metric_id":row["metric_id"],
            "delta":row["delta"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"p_adjusted":"","wins":row["wins"],"ties":row["ties"],"losses":row["losses"],"worst_unit_delta":row["worst_unit_delta"],
            "comparison_group_id":f"p3k_topk_dufs::{row['metric_id']}","status":"COMPLETE","evidence_status":"DEVELOPMENT",
            "source_artifact":str(contrasts_path.relative_to(REPO)),"source_sha256":sha256_file(contrasts_path),"source_row_selector":f"left_variant_id={row['left_variant_id']};right_variant_id={row['right_variant_id']};metric_id={row['metric_id']}",
            "notes":f"{row['statistical_status']}; worst={row['worst_unit_id']}; single frozen contrast.",
        })
    append(p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv",contrasts,("experiment_id","left_variant_id","right_variant_id","metric_id"))

    summary=json.loads((EVAL/"SUMMARY.json").read_text()); macro=next(row for row in raw if row["metric_id"]=="macro_f1"); exact=next(row for row in raw if row["metric_id"]=="first_error_exact")
    checks=[
        ("POINT","macro_f1",macro["delta"],run.BENEFIT,"ge"),("CI","macro_f1_ci_low",macro["ci_low"],run.BENEFIT,"gt"),
        ("EXACT","first_error_exact",exact["delta"],-.010,"ge"),("WORST","worst_cell_delta",macro["worst_unit_delta"],-.020,"ge"),
        ("ALIAS_PARENT","max_abs_error",summary["alias_max_errors"]["p3e_parent"],run.ALIAS_TOLERANCE,"le"),
        ("ALIAS_ZERO","max_abs_error",summary["alias_max_errors"]["lambda_zero"],run.ALIAS_TOLERANCE,"le"),
        ("ABSTENTION_ALIAS","mismatches",max(summary["abstention_mismatches"].values()),0.0,"eq"),
    ]
    gate_rows=[]
    for suffix,metric,observed,threshold,direction in checks:
        value=float(observed); passed=value>=threshold if direction=="ge" else value>threshold if direction=="gt" else value<=threshold if direction=="le" else value==threshold
        gate_rows.append({"gate_id":f"P3K_{suffix}","variant_id":K1,"metric_id":metric,"observed":observed,"threshold":threshold,"direction":direction,"passed":str(passed).lower(),"unit":"fraction","status":"PASS" if passed else "FAIL","evidence_status":"DEVELOPMENT"})
    gate_source=EVAL/"REPORTING_GATES.csv"; write(gate_source,gate_rows)
    gates=[{"phase_id":"P3","experiment_id":run.EXPERIMENT,**row,"source_artifact":str(gate_source.relative_to(REPO)),"source_sha256":sha256_file(gate_source),"source_row_selector":f"gate_id={row['gate_id']}","source_value_field":"observed","notes":"Failed improvement gate means no promotion, not generic rejection."} for row in gate_rows]
    append(p1.PROGRAM_ROOT/"GATES_LONG.csv",gates,("experiment_id","variant_id","gate_id"))

    panel_values={row["arm_id"]:float(row["value"]) for row in panels if row["metric_id"]=="official_macro_f1"}
    variants_path=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json"; registry=json.loads(variants_path.read_text())
    statuses={K0:("NO_PROMOTION","INCONCLUSIVE",f"Exact P3E3 parent alias; F1={panel_values[K0]:.6f}."),K1:("NO_PROMOTION",macro["statistical_status"],f"F1={panel_values[K1]:.6f}; delta {float(macro['delta']):+.6f} [{float(macro['ci_low']):+.6f},{float(macro['ci_high']):+.6f}].")}
    for variant,(decision,statistical,limitations) in statuses.items():
        row=next(item for item in registry["variants"] if item["variant_id"]==variant); row.update({"execution_status":"COMPLETE","decision_status":decision,"statistical_status":statistical,"limitations":limitations})
    atomic_write_json(variants_path,registry)

    experiments_path=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json"; experiments=json.loads(experiments_path.read_text()); experiment=next(row for row in experiments["experiments"] if row["experiment_id"]==run.EXPERIMENT)
    experiment.update({"execution_status":"COMPLETE","next_variant":None,"verdict":"NO_PROMOTION"}); next(row for row in experiments["experiments"] if row["experiment_id"]=="P3_FUSION")["next_variant"]=None; atomic_write_json(experiments_path,experiments)

    claims_path=p1.PROGRAM_ROOT/"CLAIMS.json"; claims=json.loads(claims_path.read_text()); upsert(claims["claims"],"claim_id",{
        "claim_id":"CLAIM_P3K_TOPK_LOCAL_DUFS","text":"Family-local DUFS-LIU was tested as a single secondary control inside the six-view top-k family against its exact ordinary-IU parent.",
        "verdict":"PROMISING_UNCONFIRMED" if float(macro["delta"])>0 and float(macro["ci_low"])<=0 else ("SUPPORTED_HARM" if float(macro["ci_high"])<run.HARM else "INCONCLUSIVE"),
        "task_scope":"Current common eight-Qwen ProcessBench first-error development population; five-fold donor cross-fit.",
        "evidence_refs":["PLOT_P3K_TOPK_DUFS",f"CONTRAST:{K1}:{K0}","TABLE_GATES"],"fresh_confirmation_required":True,
        "statistical_summary":{"metric":"macro_f1","point_delta":float(macro["delta"]),"ci_low":float(macro["ci_low"]),"ci_high":float(macro["ci_high"]),"benefit_bound":run.BENEFIT,"harm_bound":run.HARM,"bound_basis":"Frozen P3K practical bounds.","multiplicity":"Single frozen primary contrast."},
        "worst_case_behavior":f"Worst-cell delta {float(macro['worst_unit_delta']):+.6f}; W/T/L {macro['wins']}/{macro['ties']}/{macro['losses']}.",
        "claim_boundary":"Opened development evidence; no contextual graph, other fusion method, PRMBench transfer, or fresh confirmation is implied."}); atomic_write_json(claims_path,claims)

    plots_path=p1.PROGRAM_ROOT/"PLOT_MANIFEST.json"; plots=json.loads(plots_path.read_text()); upsert(plots["plots"],"plot_id",{
        "plot_id":"PLOT_P3K_TOPK_DUFS","title":"Top-k family-local DUFS control","phase":"P3","kind":"contrast_forest","source_table":"CONTRASTS_LONG.csv","selection":{"experiment_id":run.EXPERIMENT,"metric_id":"macro_f1","status":"COMPLETE"},
        "x_field":"delta","y_field":"left_variant_id","series_field":"right_variant_id","comparison_group":"same Qwen-eight rows; top-k-only donor DUFS; H0 and top-ten fixed","bootstrap_definition":"20,000 paired whole-question draws; one frozen contrast.","selection_rule":"The only registered K1-minus-K0 contrast.","legend":["K0 ordinary IU","K1 family-local DUFS-LIU"],"caption":f"Top-k DUFS minus IU {float(macro['delta']):+.5f} [{float(macro['ci_low']):+.5f},{float(macro['ci_high']):+.5f}]."}); atomic_write_json(plots_path,plots)
    build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO); REPORTING.write_build(p1.PROGRAM_ROOT,build); print(json.dumps({"status":"INTEGRATED","report_sha256":build.manifest["output"]["sha256"]},indent=2))


if __name__=="__main__": main()
