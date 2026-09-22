#!/usr/bin/env python3
"""Integrate C2 and recompute the four opened atomic primary contrasts."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_bytes, atomic_write_json, load_npz_no_pickle, sha256_file,
)
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c2 as c2  # noqa: E402


CANDIDATES = (c1.CANDIDATE, c2.CANDIDATE)
OUTPUTS = {c1.CANDIDATE: c1.OUTPUT_ROOT, c2.CANDIDATE: c2.OUTPUT_ROOT}
FAMILY_SIZE = 4


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[Mapping[str, object]]) -> None:
    handle = StringIO(newline=""); writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows); atomic_write_bytes(path, handle.getvalue().encode())


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def add_c2_metrics() -> None:
    output = c2.OUTPUT_ROOT
    _, panels = read_csv(output / "evaluation/PROCESSBENCH_PANELS.csv")
    _, cells = read_csv(output / "evaluation/PROCESSBENCH_BY_CELL.csv")
    _, strata = read_csv(output / "evaluation/STEP_LENGTH_STRATA.csv")
    _, flips = read_csv(output / "evaluation/PREDICTION_FLIP_SUMMARY.csv")
    _, raw_contrasts = read_csv(output / "evaluation/PAIRWISE_CONTRASTS.csv")
    rows: list[dict[str, object]] = []
    for row in panels:
        if row["arm_id"] != c2.CANDIDATE: continue
        rows.append({"source_id":f"panel::{row['metric_id']}","population_id":row["population_id"],"cell_id":"aggregate","slice_id":"all",
                     "metric_id":"macro_f1" if row["metric_id"]=="official_macro_f1" else row["metric_id"],"value":row["value"],
                     "ci_low":row["ci_low"],"ci_high":row["ci_high"],"n_rows":row["n_rows"],"n_groups":row["n_groups"],
                     "notes":"C2 absolute metric under its grouped five-fold atomic threshold"})
    for row in cells:
        if row["arm_id"] != c2.CANDIDATE: continue
        rows.append({"source_id":f"cell::{row['cell_id']}","population_id":"current_common_eight_qwen","cell_id":row["cell_id"],"slice_id":row["slice_id"],
                     "metric_id":"macro_f1","value":row["official_macro_f1"],"ci_low":"","ci_high":"","n_rows":row["n_examples"],"n_groups":row["n_examples"],
                     "notes":"descriptive per-cell C2 metric"})
    for row in strata:
        if row["level"]=="aggregate":
            rows.append({"source_id":f"length::{row['stratum']}::{row['metric_id']}","population_id":"current_common_eight_qwen_step_length",
                         "cell_id":row["stratum"],"slice_id":row["stratum"],"metric_id":row["metric_id"],"value":row["value"],"ci_low":"","ci_high":"",
                         "n_rows":row["n_error"],"n_groups":row["n_error"],"notes":"descriptive C2 calibration-frozen error-step length stratum"})
    for row in flips:
        rows.append({"source_id":f"flip::{row['cell_id']}::{row['transition']}","population_id":"current_common_eight_qwen_prediction_flips",
                     "cell_id":row["cell_id"],"slice_id":row["transition"],"metric_id":"prediction_flip_count","value":row["count"],"ci_low":"","ci_high":"",
                     "n_rows":row["count"],"n_groups":row["count"],"notes":"exact deterministic C2-versus-atomic-top-ten transition count"})
    for row in raw_contrasts:
        if row["right_variant_id"] != c2.REFERENCE: continue
        metric = {"first_error_exact":"first_error_exact_delta","clean_abstention_accuracy":"clean_abstention_delta"}.get(row["metric_id"])
        if metric:
            rows.append({"source_id":f"scatter::{metric}","population_id":"current_common_eight_qwen","cell_id":"aggregate","slice_id":"versus_atomic_top10",
                         "metric_id":metric,"value":row["delta"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"n_rows":6800,"n_groups":3400,
                         "notes":"paired component delta versus P2A_TOPK10_REFERENCE"})
    source=output/"evaluation/REPORT_METRICS.csv"; write_csv(source,list(rows[0]),rows); source_sha=sha256_file(source)
    path=p1.PROGRAM_ROOT/"METRICS_LONG.csv"; fields,existing=read_csv(path)
    existing=[r for r in existing if not (r["experiment_id"]=="P2_ATOMIC" and r["variant_id"]==c2.CANDIDATE)]
    additions=[]
    for index,row in enumerate(rows):
        additions.append({"phase_id":"P2","experiment_id":"P2_ATOMIC","variant_id":c2.CANDIDATE,"task_id":"processbench_first_error",
                          "dataset_id":"processbench","population_id":row["population_id"],"cell_id":row["cell_id"],"slice_id":row["slice_id"],
                          "metric_id":row["metric_id"],"value":row["value"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"n_rows":row["n_rows"],
                          "n_groups":row["n_groups"],"comparison_group_id":f"p2a::{row['population_id']}::{row['metric_id']}","status":"COMPLETE",
                          "evidence_status":"DEVELOPMENT","display_order":5200+index,"axis_value":"","source_artifact":relative(source),"source_sha256":source_sha,
                          "source_row_selector":f"source_id={row['source_id']}","source_value_field":"value","notes":row["notes"]})
    write_csv(path,fields,existing+additions)


def arm_data(variant: str) -> dict[str, Any]:
    root=OUTPUTS[variant]/"evaluation"; _,cells=read_csv(root/"PROCESSBENCH_BY_CELL.csv")
    cells=[r for r in cells if r["arm_id"]==variant]
    arrays=load_npz_no_pickle(root/"PROCESSBENCH_BOOTSTRAP_SAMPLES.npz")
    return {"by_cell":cells,"samples":{m:np.asarray(arrays[f"{variant}__{m}"],dtype=float) for m in p1.PB_METRICS}}


def reference_data() -> dict[str, Any]:
    root=c2.OUTPUT_ROOT/"evaluation"; _,cells=read_csv(root/"PROCESSBENCH_BY_CELL.csv"); cells=[r for r in cells if r["arm_id"]==c2.REFERENCE]
    arrays=load_npz_no_pickle(root/"PROCESSBENCH_BOOTSTRAP_SAMPLES.npz")
    return {"by_cell":cells,"samples":{m:np.asarray(arrays[f"{c2.REFERENCE}__{m}"],dtype=float) for m in p1.PB_METRICS}}


def recompute_contrasts() -> tuple[Path,list[dict[str,object]]]:
    comparators={c2.REFERENCE:reference_data(),"R1_ENTROPY_TOP5":c1.comparator_top5()}; rows=[]
    for variant in CANDIDATES:
        left=arm_data(variant)
        for comparator_id,right in comparators.items():
            right_cells={r["cell_id"]:r for r in right["by_cell"]}
            for metric in p1.PB_METRICS:
                metric_id="macro_f1" if metric=="official_macro_f1" else metric
                left_point=float(np.mean([float(r[metric]) for r in left["by_cell"]])); right_point=float(np.mean([float(right_cells[r["cell_id"]][metric]) for r in left["by_cell"]]))
                draws=left["samples"][metric]-right["samples"][metric]; q=0.025/FAMILY_SIZE if metric=="official_macro_f1" else 0.025
                cell={r["cell_id"]:float(r[metric])-float(right_cells[r["cell_id"]][metric]) for r in left["by_cell"]}
                family={f:float(np.mean([v for cid,v in cell.items() if right_cells[cid]["slice_id"]==f])) for f in p1.FAMILIES}; eps=1e-12
                rows.append({"contrast_id":f"pb::{variant}::{comparator_id}::{metric_id}","left_variant_id":variant,"right_variant_id":comparator_id,
                             "metric_id":metric_id,"source_metric_id":metric,"delta":left_point-right_point,"ci_low":float(np.quantile(draws,q)),"ci_high":float(np.quantile(draws,1-q)),
                             "wins":sum(v>eps for v in cell.values()),"ties":sum(abs(v)<=eps for v in cell.values()),"losses":sum(v<-eps for v in cell.values()),
                             "worst_unit_delta":min(cell.values()),"worst_unit_id":min(cell,key=cell.get),"family_wins":sum(v>eps for v in family.values()),
                             "family_ties":sum(abs(v)<=eps for v in family.values()),"family_losses":sum(v<-eps for v in family.values()),
                             "worst_family_delta":min(family.values()),"worst_family_id":min(family,key=family.get),
                             "multiplicity_family_size":FAMILY_SIZE if metric=="official_macro_f1" else 1,
                             "inference":"Bonferroni simultaneous percentile interval across four opened atomic primary contrasts" if metric=="official_macro_f1" else "unadjusted paired diagnostic percentile interval"})
    source=p1.PROGRAM_ROOT/"phase_2/atomic/P2A_CONTRASTS.csv"; write_csv(source,list(rows[0]),rows); source_sha=sha256_file(source)
    path=p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv"; fields,existing=read_csv(path); existing=[r for r in existing if r["experiment_id"]!="P2_ATOMIC"]
    additions=[]
    for row in rows:
        additions.append({"phase_id":"P2","experiment_id":"P2_ATOMIC","left_variant_id":row["left_variant_id"],"right_variant_id":row["right_variant_id"],
                          "task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen","metric_id":row["metric_id"],
                          "delta":row["delta"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"p_adjusted":"","wins":row["wins"],"ties":row["ties"],"losses":row["losses"],
                          "worst_unit_delta":row["worst_unit_delta"],"comparison_group_id":f"p2a::processbench::{row['metric_id']}","status":"COMPLETE","evidence_status":"DEVELOPMENT",
                          "source_artifact":relative(source),"source_sha256":source_sha,"source_row_selector":f"contrast_id={row['contrast_id']}",
                          "notes":f"{row['inference']}; worst cell {row['worst_unit_id']}; family W/T/L {row['family_wins']}/{row['family_ties']}/{row['family_losses']}; worst family {row['worst_family_id']} {float(row['worst_family_delta']):+.6f}; practical bounds +0.005/-0.005"})
    write_csv(path,fields,existing+additions); return source,rows


def integrate_gates(contrasts:list[dict[str,object]]) -> None:
    by={(r["left_variant_id"],r["right_variant_id"],r["metric_id"]):r for r in contrasts}; normalized=[]
    for variant in CANDIDATES:
        root=OUTPUTS[variant]; run=json.loads((root/"RUN_MANIFEST.json").read_text()); _,raw=read_csv(root/"evaluation/GATES.csv")
        for row in raw:
            if "PREMISE_PROMOTION" in row["gate_id"]: continue
            normalized.append({"variant_id":variant,"gate_id":row["gate_id"],"observed":row["observed"],"threshold":row["required"],"direction":"contract",
                               "passed":str(row["status"]=="PASS").lower(),"status":"COMPLETE","evidence_status":"DEVELOPMENT","notes":f"{row['detail']}; raw gate status={row['status']}"})
        all_pass=True
        for comparator in (c2.REFERENCE,"R1_ENTROPY_TOP5"):
            primary=by[(variant,comparator,"macro_f1")]; exact=by[(variant,comparator,"first_error_exact")]; clean=by[(variant,comparator,"clean_abstention_accuracy")]
            checks=[("POINT_BENEFIT",primary["delta"],f">={c1.BENEFIT}",float(primary["delta"])>=c1.BENEFIT),
                    ("CI_BENEFIT",primary["ci_low"],f">{c1.BENEFIT}",float(primary["ci_low"])>c1.BENEFIT),
                    ("NONNEGATIVE_CELLS",int(primary["wins"])+int(primary["ties"]),">=6",int(primary["wins"])+int(primary["ties"])>=6),
                    ("WORST_CELL",primary["worst_unit_delta"],f">={c1.PROMOTION_WORST_CELL_BOUND}",float(primary["worst_unit_delta"])>=c1.PROMOTION_WORST_CELL_BOUND),
                    ("EXACT_ERROR",exact["delta"],f">={c1.COMPONENT_BOUND}",float(exact["delta"])>=c1.COMPONENT_BOUND),
                    ("CLEAN_ABSTENTION",clean["delta"],f">={c1.COMPONENT_BOUND}",float(clean["delta"])>=c1.COMPONENT_BOUND)]
            for name,observed,threshold,passed in checks:
                all_pass &= passed; normalized.append({"variant_id":variant,"gate_id":f"{variant}_VS_{comparator}_{name}","observed":observed,"threshold":threshold,
                                                       "direction":"contract","passed":str(passed).lower(),"status":"COMPLETE","evidence_status":"DEVELOPMENT","notes":f"{variant} versus {comparator}; four-contrast family"})
        normalized.append({"variant_id":variant,"gate_id":f"{variant}_PREMISE_PROMOTION","observed":str(all_pass and run["status"]!="HARD_FAIL").lower(),
                           "threshold":"all promotion and hard gates pass","direction":"contract","passed":str(all_pass and run["status"]!="HARD_FAIL").lower(),
                           "status":"COMPLETE","evidence_status":"DEVELOPMENT","notes":"survivor gate after four-contrast multiplicity update"})
    normalized.append({"variant_id":c2.CANDIDATE,"gate_id":"P2A_SWVAR_FAMILY_PREMISE","observed":"C1 and C2 both HARD_FAIL","threshold":"at least one preregistered SWVar atom survives",
                       "direction":"contract","passed":"false","status":"COMPLETE","evidence_status":"DEVELOPMENT","notes":"closes the SWVar family and its Phase-2R-B template"})
    source=p1.PROGRAM_ROOT/"phase_2/atomic/P2A_GATES.csv"; write_csv(source,list(normalized[0]),normalized); source_sha=sha256_file(source)
    path=p1.PROGRAM_ROOT/"GATES_LONG.csv"; fields,existing=read_csv(path); existing=[r for r in existing if r["experiment_id"]!="P2_ATOMIC"]
    additions=[{"phase_id":"P2","experiment_id":"P2_ATOMIC","variant_id":r["variant_id"],"gate_id":r["gate_id"],"metric_id":"contract_or_promotion_gate",
                "observed":r["observed"],"threshold":r["threshold"],"direction":r["direction"],"passed":r["passed"],"unit":"contract","status":r["status"],
                "evidence_status":r["evidence_status"],"source_artifact":relative(source),"source_sha256":source_sha,"source_row_selector":f"variant_id={r['variant_id']};gate_id={r['gate_id']}",
                "source_value_field":"observed","notes":r["notes"]} for r in normalized]
    write_csv(path,fields,existing+additions)


def update_status_and_claims(contrasts:list[dict[str,object]]) -> None:
    by={(r["left_variant_id"],r["right_variant_id"],r["metric_id"]):r for r in contrasts}
    path=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json"; payload=json.loads(path.read_text())
    for row in payload["variants"]:
        if row["variant_id"]==c2.CANDIDATE: row.update({"execution_status":"HARD_FAIL","decision_status":"REJECTED","statistical_status":"HARD_FAILURE","rankable":False})
        elif row["variant_id"]=="C3_ENT_CCUSUM":
            note=" Not run because its C1 SWVar16 parent hard-failed."
            row.update({"execution_status":"NOT_RUN_BY_GATE","decision_status":"NO_PROMOTION","statistical_status":"NOT_EVALUATED","rankable":False,
                        "limitations":row["limitations"] if note.strip() in row["limitations"] else row["limitations"]+note})
    atomic_write_json(path,payload)
    path=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json"; payload=json.loads(path.read_text()); exp=next(r for r in payload["experiments"] if r["experiment_id"]=="P2_ATOMIC")
    exp.update({"opened_variants":list(CANDIDATES),"latest_variant":c2.CANDIDATE,"opened_primary_comparisons":4,"multiplicity_family_size":4,
                "swvar_family_status":"CLOSED_HARD_FAILURE","swvar_reducer_template_status":"NOT_RUN_BY_GATE","cusum_c3_status":"NOT_RUN_BY_GATE_PARENT_FAILURE","next_variant":"C4_ENT_SAMPLED"})
    atomic_write_json(path,payload)
    claims_path=p1.PROGRAM_ROOT/"CLAIMS.json"; claims=json.loads(claims_path.read_text()); claims["claims"]=[r for r in claims["claims"] if r["claim_id"] not in {"CLAIM_C1_ENT_SW16","CLAIM_C2_ENT_SWADAPT"}]
    for variant in CANDIDATES:
        p=by[(variant,c2.REFERENCE,"macro_f1")]; top5=by[(variant,"R1_ENTROPY_TOP5","macro_f1")]
        supported_harm=float(p["ci_high"])<c1.HARM
        claims["claims"].append({"claim_id":f"CLAIM_{variant}","text":f"{variant} did not pass the frozen ProcessBench atomic premise gate.","verdict":"BLOCKED",
            "task_scope":"Current eight-Qwen ProcessBench development population under the P2 atomic contract.",
            "evidence_refs":["PLOT_P2_DELTA_FOREST","PLOT_P2_ATOMIC_GATE_MATRIX","PLOT_P2_EXACT_CLEAN",f"CONTRAST:{variant}:{c2.REFERENCE}"],
            "worst_case_behavior":f"Cell W/T/L versus atomic top-ten is {p['wins']}/{p['ties']}/{p['losses']}; worst cell {p['worst_unit_id']} at {float(p['worst_unit_delta']):+.6f}.",
            "claim_boundary":f"Versus top-ten: {float(p['delta']):+.6f} [{float(p['ci_low']):+.6f}, {float(p['ci_high']):+.6f}]; versus top-five: {float(top5['delta']):+.6f} [{float(top5['ci_low']):+.6f}, {float(top5['ci_high']):+.6f}]. {'The top-ten contrast supports practical harm, and ' if supported_harm else ''}the branch decision is additionally bound by the preregistered hard robustness gate.",
            "statistical_summary":{"metric":"macro_f1","point_delta":float(p["delta"]),"ci_low":float(p["ci_low"]),"ci_high":float(p["ci_high"]),"benefit_bound":c1.BENEFIT,"harm_bound":c1.HARM,
                                     "bound_basis":"P2 atomic ProcessBench practical bounds","multiplicity":p["inference"]},"fresh_confirmation_required":False})
    atomic_write_json(claims_path,claims)


def update_plot_scope() -> None:
    path=p1.PROGRAM_ROOT/"PLOT_MANIFEST.json"; payload=json.loads(path.read_text())
    plot=next(r for r in payload["plots"] if r["plot_id"]=="PLOT_P2_ATOMIC_LENGTH_HEATMAP")
    plot["selection"].pop("variant_id",None)
    plot["caption"]="C1 fixed-window and C2 adaptive-window macro F1 by calibration-frozen short, medium, and long true-error step spans."
    plot["selection_rule"]="All completed atomic SWVar candidates with aggregate calibration-frozen step-length strata."
    plot["title"]="SWVar atomic performance by true-error step length"
    atomic_write_json(path,payload)


def main() -> None:
    run=json.loads((c2.OUTPUT_ROOT/"RUN_MANIFEST.json").read_text())
    if run.get("status") not in {"COMPLETE","HARD_FAIL"} or run["execution_registry_sha256"]!=sha256_file(c2.REGISTRY_PATH): raise RuntimeError("C2 run/registry invalid")
    add_c2_metrics(); _,contrasts=recompute_contrasts(); integrate_gates(contrasts); update_status_and_claims(contrasts); update_plot_scope()
    subprocess.run([sys.executable,str(REPO/"scripts/reasoning_localization/build_reasoning_localization_report.py")],cwd=REPO,check=True)
    primary=next(r for r in contrasts if r["left_variant_id"]==c2.CANDIDATE and r["right_variant_id"]==c2.REFERENCE and r["metric_id"]=="macro_f1")
    print(json.dumps({"variant_id":c2.CANDIDATE,"execution_status":run["status"],"macro_f1":run["summary"]["candidate_macro_f1"],"delta_vs_top10":primary["delta"],
                      "simultaneous_ci":[primary["ci_low"],primary["ci_high"]],"next_variant":"C4_ENT_SAMPLED","report_sha256":sha256_file(p1.PROGRAM_ROOT/"REPORT.html")},indent=2))


if __name__ == "__main__": main()
