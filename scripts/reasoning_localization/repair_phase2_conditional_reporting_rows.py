#!/usr/bin/env python3
"""Normalize the first Phase-2C rows to the strict reporting source contract."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import sha256_file  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization import run_phase2_conditional as runner  # noqa:E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa:E402

EXP="P2_CONDITIONAL_ABLATION"

def read(path):
    with path.open(newline="") as h: return list(csv.DictReader(h))
def write(path, rows, fields=None):
    if fields is None: fields=list(rows[0])
    with path.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows([{f:r.get(f,"") for f in fields} for r in rows])

def main():
    metric_master=p1.PROGRAM_ROOT/"METRICS_LONG.csv"; contrast_master=p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv"; gate_master=p1.PROGRAM_ROOT/"GATES_LONG.csv"
    metric_fields=list(read(metric_master)[0]); contrast_fields=list(read(contrast_master)[0]); gate_fields=list(read(gate_master)[0])
    metrics=[r for r in read(metric_master) if r["experiment_id"]!=EXP]
    contrasts=[r for r in read(contrast_master) if r["experiment_id"]!=EXP]
    gates=[r for r in read(gate_master) if r["experiment_id"]!=EXP]
    for order,variant in ((130,runner.PARENT),(131,"P2C_F6_MINUS_ENTROPY_LEVEL")):
        er=runner.output_root(variant)/"evaluation"; source=er/"REPORTING_METRICS.csv"
        unique=[{**r,"status":"COMPLETE","evidence_status":"DEVELOPMENT"} for r in read(er/"PROCESSBENCH_PANELS.csv") if r["arm_id"]==variant]
        # The parent self-comparison emitted its panel twice; retain one row per metric.
        unique=list({r["metric_id"]:r for r in unique}.values()); write(source,unique)
        for r in unique:
            metric="macro_f1" if r["metric_id"]=="official_macro_f1" else r["metric_id"]
            metrics.append({"phase_id":"P2C","experiment_id":EXP,"variant_id":variant,"task_id":"processbench_first_error","dataset_id":"processbench",
                "population_id":"current_common_eight_qwen","cell_id":"aggregate","slice_id":"all","metric_id":metric,"value":r["value"],"ci_low":r["ci_low"],"ci_high":r["ci_high"],
                "n_rows":r["n_rows"],"n_groups":r["n_groups"],"comparison_group_id":f"p2c_qwen8_five_family_top10::{metric}","status":"COMPLETE","evidence_status":"DEVELOPMENT",
                "display_order":order,"source_artifact":str(source.relative_to(REPO)),"source_sha256":sha256_file(source),
                "source_row_selector":f"arm_id={variant};metric_id={r['metric_id']}","source_value_field":"value","notes":"Five-family top-ten conditional study."})
    variant="P2C_F6_MINUS_ENTROPY_LEVEL"; er=runner.output_root(variant)/"evaluation"; csource=er/"REPORTING_CONTRASTS.csv"
    crows=[{**r,"status":"COMPLETE","evidence_status":"DEVELOPMENT"} for r in read(er/"PAIRWISE_CONTRASTS.csv")];write(csource,crows)
    for r in crows:
        metric=r["metric_id"]
        contrasts.append({"phase_id":"P2C","experiment_id":EXP,"left_variant_id":variant,"right_variant_id":runner.PARENT,"task_id":"processbench_first_error",
            "dataset_id":"processbench","population_id":"current_common_eight_qwen","metric_id":metric,"delta":r["candidate_minus_parent_delta"],"ci_low":r["ci_low"],"ci_high":r["ci_high"],
            "wins":r["wins"],"ties":r["ties"],"losses":r["losses"],"worst_unit_delta":r["worst_unit_delta"],"comparison_group_id":f"p2c_qwen8_five_family_top10::{metric}",
            "status":"COMPLETE","evidence_status":"DEVELOPMENT","source_artifact":str(csource.relative_to(REPO)),"source_sha256":sha256_file(csource),
            "source_row_selector":f"metric_id={metric}","notes":"Candidate-minus-parent; reverse signs for leave-one-out contribution."})
    gsource=er/"GATES.csv"; grows=[]
    for r in read(gsource): grows.append({**r,"status":"PASS" if r["passed"]=="true" else "FAIL","evidence_status":"DEVELOPMENT"})
    write(gsource,grows)
    for r in grows:
        gates.append({"phase_id":"P2C","experiment_id":EXP,"variant_id":variant,**r,"unit":"boolean" if r["metric_id"]=="all_registered_gates" else "fraction",
            "source_artifact":str(gsource.relative_to(REPO)),"source_sha256":sha256_file(gsource),"source_row_selector":f"gate_id={r['gate_id']}","source_value_field":"observed",
            "notes":"Aggregate contribution is supported, but clean abstention blocks full conditional promotion."})
    write(metric_master,metrics,metric_fields);write(contrast_master,contrasts,contrast_fields);write(gate_master,gates,gate_fields)
    build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO);REPORTING.write_build(p1.PROGRAM_ROOT,build)
    print(json.dumps({"status":"REPAIRED","report_sha256":build.manifest["output"]["sha256"]},indent=2))
if __name__=="__main__":main()
