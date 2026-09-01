#!/usr/bin/env python3
"""Integrate completed P3E family-expert attribution results."""

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
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402
from scripts.reasoning_localization.run_phase3_family_expert_attribution import (  # noqa: E402
    BENEFIT, E0, E1, E2, E3, E4, EXPERIMENT, FAMILY_SIZE, H0, H2, OUTPUT, VARIANTS,
)

EVAL = OUTPUT / "evaluation"
H0_REPORT = "P2C_F6_TOP10_REFERENCE"
ORDERS = {H0: 169, H2: 170, E0: 177, E1: 178, E2: 179, E3: 180, E4: 181}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    columns = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows([{field: row.get(field, "") for field in columns} for row in rows])


def append(path: Path, additions: list[dict[str, Any]], unique: tuple[str, ...]) -> None:
    rows = read(path); fields = list(rows[0]); keys = {tuple(row.get(field, "") for field in unique) for row in rows}
    for row in additions:
        key = tuple(str(row.get(field, "")) for field in unique)
        if key in keys: raise RuntimeError(f"duplicate {key}")
        keys.add(key)
    write(path, [*rows, *additions], fields)


def alias(value: str) -> str:
    return H0_REPORT if value == H0 else value


def upsert(rows: list[dict], key: str, row: dict) -> None:
    hits = [index for index, current in enumerate(rows) if current.get(key) == row[key]]
    if len(hits) > 1: raise RuntimeError(f"duplicate {key}")
    if hits: rows[hits[0]] = row
    else: rows.append(row)


def main() -> None:
    panels_path = EVAL / "PROCESSBENCH_PANELS.csv"; panels = read(panels_path)
    metrics = []
    for row in panels:
        metric = "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"]
        metrics.append({"phase_id":"P3","experiment_id":EXPERIMENT,"variant_id":alias(row["arm_id"]),"task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen","cell_id":"aggregate","slice_id":"all","metric_id":metric,"value":row["value"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"n_rows":row["n_rows"],"n_groups":row["n_groups"],"comparison_group_id":f"p3e_family::{metric}","status":"COMPLETE","evidence_status":"DEVELOPMENT","display_order":ORDERS[row["arm_id"]],"axis_value":"","source_artifact":str(panels_path.relative_to(REPO)),"source_sha256":sha256_file(panels_path),"source_row_selector":f"arm_id={row['arm_id']};metric_id={row['metric_id']}","source_value_field":"value","notes":"Five-fold donor cross-fit; one family expert changes at a time; H0 abstention and top-ten fixed."})
    append(p1.PROGRAM_ROOT/"METRICS_LONG.csv", metrics, ("experiment_id","variant_id","metric_id","cell_id"))

    contrasts_path = EVAL / "PAIRWISE_CONTRASTS.csv"; raw = read(contrasts_path); contrasts = []
    for row in raw:
        contrasts.append({"phase_id":"P3","experiment_id":EXPERIMENT,"left_variant_id":alias(row["left_variant_id"]),"right_variant_id":alias(row["right_variant_id"]),"task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen","metric_id":row["metric_id"],"delta":row["delta"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"p_adjusted":"","wins":row["wins"],"ties":row["ties"],"losses":row["losses"],"worst_unit_delta":row["worst_unit_delta"],"comparison_group_id":f"p3e_family::{row['metric_id']}","status":"COMPLETE","evidence_status":"DEVELOPMENT","source_artifact":str(contrasts_path.relative_to(REPO)),"source_sha256":sha256_file(contrasts_path),"source_row_selector":f"left_variant_id={row['left_variant_id']};right_variant_id={row['right_variant_id']};metric_id={row['metric_id']}","notes":f"{row['statistical_status']}; worst={row['worst_unit_id']}; family={row['multiplicity_family_size']}."})
    append(p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv", contrasts, ("experiment_id","left_variant_id","right_variant_id","metric_id"))

    primary = {row["left_variant_id"]: row for row in raw if row["right_variant_id"] == E0 and row["metric_id"] == "macro_f1"}
    exact = {row["left_variant_id"]: row for row in raw if row["right_variant_id"] == E0 and row["metric_id"] == "first_error_exact"}
    summary = json.loads((EVAL/"SUMMARY.json").read_text())
    gate_rows = []
    for variant in (E1,E2,E3,E4):
        checks = [("POINT", "macro_f1", primary[variant]["delta"], BENEFIT, "ge"), ("SIMULTANEOUS", "macro_f1_ci_low", primary[variant]["ci_low"], BENEFIT, "gt"), ("EXACT", "first_error_exact", exact[variant]["delta"], -.010, "ge"), ("WORST", "worst_cell_delta", primary[variant]["worst_unit_delta"], -.020, "ge"), ("ABSTENTION_ALIAS", "mismatches", summary["abstention_mismatches"][variant], 0, "eq")]
        for suffix, metric, observed, threshold, direction in checks:
            value=float(observed); passed = value >= threshold if direction=="ge" else value > threshold if direction=="gt" else value==threshold
            gate_rows.append({"gate_id":f"{variant}_{suffix}","variant_id":variant,"metric_id":metric,"observed":observed,"threshold":threshold,"direction":direction,"passed":str(passed).lower(),"unit":"fraction","status":"PASS" if passed else "FAIL","evidence_status":"DEVELOPMENT"})
    gate_source=EVAL/"REPORTING_GATES.csv"; write(gate_source,gate_rows)
    gates=[]
    for row in gate_rows:
        gates.append({"phase_id":"P3","experiment_id":EXPERIMENT,**row,"source_artifact":str(gate_source.relative_to(REPO)),"source_sha256":sha256_file(gate_source),"source_row_selector":f"gate_id={row['gate_id']}","source_value_field":"observed","notes":"Failed improvement gate is no promotion, not generic rejection."})
    append(p1.PROGRAM_ROOT/"GATES_LONG.csv",gates,("experiment_id","variant_id","gate_id"))

    variants_path=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json"; registry=json.loads(variants_path.read_text())
    statuses={E0:("NO_PROMOTION","INCONCLUSIVE","Matched cross-fit reference F1 0.364284; +0.000194 versus incumbent H2."),E1:("NO_PROMOTION","PROMISING_UNCONFIRMED","Raw-best F1 0.366876; +0.002592 vs E0, CI [-0.001839,+0.007194], 6/0/2."),E2:("NO_PROMOTION","INCONCLUSIVE","F1 0.359376; -0.004908 vs E0, CI [-0.013100,+0.003050], 3/0/5."),E3:("NO_PROMOTION","PROMISING_UNCONFIRMED","F1 0.365603; +0.001319 vs E0, CI [-0.001433,+0.004355], 4/1/3."),E4:("NO_PROMOTION","INCONCLUSIVE","F1 0.359577; -0.004708 vs E0, CI [-0.013846,+0.004211], 3/0/5.")}
    for variant,(decision,statistical,limitations) in statuses.items():
        row=next(item for item in registry["variants"] if item["variant_id"]==variant); row.update({"execution_status":"COMPLETE","decision_status":decision,"statistical_status":statistical,"limitations":limitations})
    atomic_write_json(variants_path,registry)

    experiments_path=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json"; experiments=json.loads(experiments_path.read_text()); experiment=next(row for row in experiments["experiments"] if row["experiment_id"]==EXPERIMENT); experiment.update({"execution_status":"COMPLETE","next_variant":None,"verdict":"DYNAMICS_AND_TOPK_PROMISING_UNCONFIRMED__NO_PROMOTION","development_eligible_families":[E1,E3]}); next(row for row in experiments["experiments"] if row["experiment_id"]=="P3_FUSION")["next_variant"]=None; atomic_write_json(experiments_path,experiments)

    claims_path=p1.PROGRAM_ROOT/"CLAIMS.json"; claims=json.loads(claims_path.read_text()); upsert(claims["claims"],"claim_id",{"claim_id":"CLAIM_P3E_FAMILY_EXPERT_ATTRIBUTION","text":"Ordinary IU compression is not uniformly useful across H2 families: dynamics-only and top-k-only have positive unconfirmed point estimates, while partition-only is negative and accounts for much of the joint all-family loss.","verdict":"PROMISING_UNCONFIRMED","task_scope":"Current common eight-Qwen ProcessBench first-error development population; five-fold donor cross-fit.","evidence_refs":["PLOT_P3E_FAMILY_FOREST",f"CONTRAST:{E1}:{E0}","TABLE_GATES"],"fresh_confirmation_required":True,"statistical_summary":{"metric":"macro_f1","point_delta":float(primary[E1]["delta"]),"ci_low":float(primary[E1]["ci_low"]),"ci_high":float(primary[E1]["ci_high"]),"benefit_bound":BENEFIT,"harm_bound":-BENEFIT,"bound_basis":"Frozen P3E practical bounds.","multiplicity":f"Bonferroni simultaneous across {FAMILY_SIZE} family-expert contrasts."},"worst_case_behavior":"Dynamics-only loses 2/8 cells, worst -0.004668; partition-only loses 5/8 and has exact-error point delta -0.006820.","claim_boundary":"Opened development evidence; neither positive family is promoted, and no method-specific STG/DUFS/B3 variant is yet supported."}); atomic_write_json(claims_path,claims)

    plots_path=p1.PROGRAM_ROOT/"PLOT_MANIFEST.json"; plots=json.loads(plots_path.read_text()); upsert(plots["plots"],"plot_id",{"plot_id":"PLOT_P3E_FAMILY_FOREST","title":"One-family-at-a-time IU expert attribution","phase":"P3","kind":"contrast_forest","source_table":"CONTRASTS_LONG.csv","selection":{"experiment_id":EXPERIMENT,"metric_id":"macro_f1","status":"COMPLETE"},"x_field":"delta","y_field":"left_variant_id","series_field":"right_variant_id","comparison_group":"same Qwen-eight rows; five-fold donor cross-fit; H0 and top-ten fixed","bootstrap_definition":"20,000 paired whole-question draws; Bonferroni simultaneous across four E1-E4 minus E0 contrasts.","selection_rule":"Every registered family-expert contrast plus matched incumbent comparison.","legend":["Positive CI crossing zero = promising unconfirmed","Partition IU explains much of the all-family loss"],"caption":"Dynamics-only is raw best at +0.00259, top-k-only +0.00132, partition-only -0.00491, and all-family IU -0.00471 versus matched equal compression."}); atomic_write_json(plots_path,plots)
    build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO); REPORTING.write_build(p1.PROGRAM_ROOT,build); print(json.dumps({"status":"INTEGRATED","eligible":[E1,E3],"report_sha256":build.manifest["output"]["sha256"]},indent=2))


if __name__=="__main__": main()
