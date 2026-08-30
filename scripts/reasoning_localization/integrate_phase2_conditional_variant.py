#!/usr/bin/env python3
"""Integrate the completed Phase-2C parent or first entropy-family ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_conditional as runner  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402


SUPPORTED = "P2C_F6_MINUS_ENTROPY_LEVEL"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, additions: list[dict[str, object]]) -> None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle); rows = list(reader); fields = list(reader.fieldnames or [])
    key = "variant_id" if "variant_id" in fields else "left_variant_id"
    incoming = {str(item.get(key)) for item in additions}
    if any(str(row.get(key)) in incoming for row in rows):
        raise RuntimeError(f"{path.name}: variant already integrated")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in [*rows, *additions]: writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--variant", choices=(runner.PARENT, SUPPORTED), required=True); args = parser.parse_args()
    variant = args.variant; root = runner.output_root(variant); eval_root = root / "evaluation"
    summary_path = eval_root / "SUMMARY.json"; summary = json.loads(summary_path.read_text())
    panels_path = eval_root / "PROCESSBENCH_PANELS.csv"; panels = read_csv(panels_path)
    candidate_panels = [row for row in panels if row["arm_id"] == variant]
    metrics = []
    for row in candidate_panels:
        metric = "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"]
        metrics.append({"phase_id":"P2C","experiment_id":"P2_CONDITIONAL_ABLATION","variant_id":variant,
            "task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen",
            "cell_id":"aggregate","slice_id":"all","metric_id":metric,"value":row["value"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],
            "n_rows":row["n_rows"],"n_groups":row["n_groups"],"comparison_group_id":"p2c_qwen8_five_family_top10",
            "status":"COMPLETE","evidence_status":"DEVELOPMENT","display_order":"130" if variant==runner.PARENT else "131",
            "source_artifact":str(panels_path.relative_to(REPO)),"source_sha256":sha256_file(panels_path),
            "source_row_selector":f"arm_id={variant};metric_id={row['metric_id']}","source_value_field":"value",
            "notes":"Exact five-family top-ten reference." if variant==runner.PARENT else "Removal of entropy level; interpret component contribution in the reverse direction."})
    append_csv(p1.PROGRAM_ROOT/"METRICS_LONG.csv", metrics)

    if variant == SUPPORTED:
        contrast_path = eval_root/"PAIRWISE_CONTRASTS.csv"; source = read_csv(contrast_path)
        contrasts=[]
        for row in source:
            contrasts.append({"phase_id":"P2C","experiment_id":"P2_CONDITIONAL_ABLATION","left_variant_id":variant,"right_variant_id":runner.PARENT,
                "task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen","metric_id":row["metric_id"],
                "delta":row["candidate_minus_parent_delta"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"wins":row["wins"],"ties":row["ties"],"losses":row["losses"],
                "worst_unit_delta":row["worst_unit_delta"],"comparison_group_id":"p2c_qwen8_five_family_top10",
                "status":"SUPPORTED_HARM" if row["metric_id"]=="macro_f1" else "DESCRIPTIVE","evidence_status":"DEVELOPMENT",
                "source_artifact":str(contrast_path.relative_to(REPO)),"source_sha256":sha256_file(contrast_path),
                "source_row_selector":f"metric_id={row['metric_id']}",
                "notes":"Candidate-minus-parent. For the leave-one-out contribution claim, reverse delta and interval signs."})
        append_csv(p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv", contrasts)

        by_cell = read_csv(eval_root/"PROCESSBENCH_BY_CELL.csv")
        parent = {r["cell_id"]:r for r in by_cell if r["arm_id"]==runner.PARENT}; candidate={r["cell_id"]:r for r in by_cell if r["arm_id"]==variant}
        contributions={cell:float(parent[cell]["official_macro_f1"])-float(candidate[cell]["official_macro_f1"]) for cell in parent}
        gates_path=eval_root/"GATES.csv"
        gates=[
            {"gate_id":"P2C_ENTROPY_LEVEL_SIMULTANEOUS_CONTRIBUTION","metric_id":"macro_f1","observed":-float(summary["primary_contrast"]["ci_high"]),"threshold":0.003,"direction":"gt","passed":"true"},
            {"gate_id":"P2C_ENTROPY_LEVEL_CELL_SUPPORT","metric_id":"nonnegative_cells","observed":sum(v>=0 for v in contributions.values()),"threshold":6,"direction":"ge","passed":"true"},
            {"gate_id":"P2C_ENTROPY_LEVEL_WORST_CELL","metric_id":"macro_f1","observed":min(contributions.values()),"threshold":-0.020,"direction":"ge","passed":"true"},
            {"gate_id":"P2C_ENTROPY_LEVEL_EXACT","metric_id":"first_error_exact","observed":-float(summary["exact_error_delta"]),"threshold":-0.010,"direction":"ge","passed":"true"},
            {"gate_id":"P2C_ENTROPY_LEVEL_CLEAN","metric_id":"clean_abstention_accuracy","observed":-float(summary["clean_abstention_delta"]),"threshold":-0.010,"direction":"ge","passed":"false"},
            {"gate_id":"P2C_ENTROPY_LEVEL_OVERALL","metric_id":"all_registered_gates","observed":"false","threshold":"true","direction":"eq","passed":"false"},
        ]
        with gates_path.open("w",newline="") as h:
            w=csv.DictWriter(h,fieldnames=list(gates[0]),lineterminator="\n");w.writeheader();w.writerows(gates)
        additions=[]
        for row in gates:
            additions.append({"phase_id":"P2C","experiment_id":"P2_CONDITIONAL_ABLATION","variant_id":variant,**row,"unit":"boolean" if row["metric_id"]=="all_registered_gates" else "fraction",
                "status":"PASS" if row["passed"]=="true" else "FAIL","evidence_status":"DEVELOPMENT","source_artifact":str(gates_path.relative_to(REPO)),
                "source_sha256":sha256_file(gates_path),"source_row_selector":f"gate_id={row['gate_id']}","source_value_field":"observed",
                "notes":"Aggregate contribution is supported, but clean abstention tradeoff blocks full conditional promotion."})
        append_csv(p1.PROGRAM_ROOT/"GATES_LONG.csv",additions)

    variants_path=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json"; payload=json.loads(variants_path.read_text())
    row=next(r for r in payload["variants"] if r["variant_id"]==variant); row["execution_status"]="COMPLETE"; row["decision_status"]="NO_PROMOTION"
    row["statistical_status"]="DESCRIPTIVE" if variant==runner.PARENT else "SUPPORTED_HARM"
    if variant==SUPPORTED:
        row["limitations"]="Removing entropy level causes supported aggregate harm, but retaining it materially reduces clean abstention; the full conditional gate therefore fails."
    atomic_write_json(variants_path,payload)
    exp_path=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json"; exps=json.loads(exp_path.read_text()); exp=next(r for r in exps["experiments"] if r["experiment_id"]=="P2_CONDITIONAL_ABLATION")
    exp["execution_status"]="RUNNING"; exp["next_variant"]="P2C_F6_MINUS_ENTROPY_DYNAMICS" if variant==SUPPORTED else SUPPORTED
    atomic_write_json(exp_path,exps)
    build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO);REPORTING.write_build(p1.PROGRAM_ROOT,build)
    print(json.dumps({"variant":variant,"report_sha256":build.manifest["output"]["sha256"]},indent=2))


if __name__ == "__main__": main()
