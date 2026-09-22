#!/usr/bin/env python3
"""Integrate the completed C1--C8 atomic roster and close its inference family."""

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
from scripts.reasoning_localization import run_phase2_atomic_remaining as remaining  # noqa: E402


CANDIDATES = (c1.CANDIDATE, c2.CANDIDATE) + remaining.VARIANTS
ROOTS = {c1.CANDIDATE:c1.OUTPUT_ROOT, c2.CANDIDATE:c2.OUTPUT_ROOT,
         **{variant:remaining.output_root(variant) for variant in remaining.VARIANTS}}
FAMILY_SIZE = 16


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[Mapping[str, object]]) -> None:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    atomic_write_bytes(path, handle.getvalue().encode("utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def arm_data(variant: str) -> dict[str, Any]:
    root = ROOTS[variant] / "evaluation"
    _, cells = read_csv(root / "PROCESSBENCH_BY_CELL.csv")
    arrays = load_npz_no_pickle(root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz")
    return {"by_cell":[row for row in cells if row["arm_id"] == variant],
            "samples":{metric:np.asarray(arrays[f"{variant}__{metric}"], dtype=float) for metric in p1.PB_METRICS}}


def reference_data() -> dict[str, Any]:
    root = ROOTS[remaining.VARIANTS[-1]] / "evaluation"
    _, cells = read_csv(root / "PROCESSBENCH_BY_CELL.csv")
    arrays = load_npz_no_pickle(root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz")
    return {"by_cell":[row for row in cells if row["arm_id"] == remaining.REFERENCE],
            "samples":{metric:np.asarray(arrays[f"{remaining.REFERENCE}__{metric}"], dtype=float) for metric in p1.PB_METRICS}}


def make_contrast(variant: str, comparator_id: str, left: Mapping[str, Any], right: Mapping[str, Any], metric: str, *, primary: bool) -> dict[str, object]:
    right_cells = {row["cell_id"]:row for row in right["by_cell"]}
    left_point = float(np.mean([float(row[metric]) for row in left["by_cell"]]))
    right_point = float(np.mean([float(right_cells[row["cell_id"]][metric]) for row in left["by_cell"]]))
    draws = np.asarray(left["samples"][metric]) - np.asarray(right["samples"][metric])
    q = 0.025 / FAMILY_SIZE if primary and metric == "official_macro_f1" else 0.025
    cells = {row["cell_id"]:float(row[metric])-float(right_cells[row["cell_id"]][metric]) for row in left["by_cell"]}
    families = {family:float(np.mean([value for cell_id,value in cells.items() if right_cells[cell_id]["slice_id"] == family])) for family in p1.FAMILIES}
    eps = 1e-12
    metric_id = "macro_f1" if metric == "official_macro_f1" else metric
    return {"contrast_id":f"pb::{variant}::{comparator_id}::{metric_id}","left_variant_id":variant,"right_variant_id":comparator_id,
            "metric_id":metric_id,"source_metric_id":metric,"delta":left_point-right_point,
            "ci_low":float(np.quantile(draws,q)),"ci_high":float(np.quantile(draws,1-q)),
            "wins":sum(v>eps for v in cells.values()),"ties":sum(abs(v)<=eps for v in cells.values()),"losses":sum(v<-eps for v in cells.values()),
            "worst_unit_delta":min(cells.values()),"worst_unit_id":min(cells,key=cells.get),
            "family_wins":sum(v>eps for v in families.values()),"family_ties":sum(abs(v)<=eps for v in families.values()),"family_losses":sum(v<-eps for v in families.values()),
            "worst_family_delta":min(families.values()),"worst_family_id":min(families,key=families.get),
            "multiplicity_family_size":FAMILY_SIZE if primary and metric == "official_macro_f1" else 1,
            "inference":"Bonferroni simultaneous percentile interval across all 16 C1--C8 primary macro-F1 contrasts" if primary and metric == "official_macro_f1" else "unadjusted paired diagnostic percentile interval"}


def c8_parent_data() -> dict[str, Any]:
    parent = remaining.EXTRA_PARENT["C8_SELF_INNOV"]
    root = ROOTS["C8_SELF_INNOV"] / "evaluation"
    _, cells = read_csv(root / "PROCESSBENCH_BY_CELL.csv")
    arrays = load_npz_no_pickle(root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz")
    return {"by_cell":[row for row in cells if row["arm_id"] == parent],
            "samples":{metric:np.asarray(arrays[f"{parent}__{metric}"], dtype=float) for metric in p1.PB_METRICS}}


def recompute_contrasts() -> list[dict[str, object]]:
    comparators = {remaining.REFERENCE:reference_data(), "R1_ENTROPY_TOP5":c1.comparator_top5()}
    rows = []
    for variant in CANDIDATES:
        left = arm_data(variant)
        for comparator_id, right in comparators.items():
            rows.extend(make_contrast(variant, comparator_id, left, right, metric, primary=True) for metric in p1.PB_METRICS)
    parent = remaining.EXTRA_PARENT["C8_SELF_INNOV"]
    rows.extend(make_contrast("C8_SELF_INNOV", parent, arm_data("C8_SELF_INNOV"), c8_parent_data(), metric, primary=False) for metric in p1.PB_METRICS)
    source = p1.PROGRAM_ROOT / "phase_2/atomic/P2A_CONTRASTS.csv"
    write_csv(source, list(rows[0]), rows)
    source_sha = sha256_file(source)
    path = p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv"
    fields, existing = read_csv(path)
    existing = [row for row in existing if row["experiment_id"] != "P2_ATOMIC"]
    additions = []
    for row in rows:
        primary = row["right_variant_id"] in comparators
        additions.append({"phase_id":"P2","experiment_id":"P2_ATOMIC","left_variant_id":row["left_variant_id"],"right_variant_id":row["right_variant_id"],
            "task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen","metric_id":row["metric_id"],
            "delta":row["delta"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"p_adjusted":"","wins":row["wins"],"ties":row["ties"],"losses":row["losses"],
            "worst_unit_delta":row["worst_unit_delta"],"comparison_group_id":f"p2a::processbench::{row['metric_id']}::{'primary' if primary else 'parent_diagnostic'}",
            "status":"COMPLETE","evidence_status":"DEVELOPMENT","source_artifact":relative(source),"source_sha256":source_sha,
            "source_row_selector":f"contrast_id={row['contrast_id']}",
            "notes":f"{row['inference']}; worst cell {row['worst_unit_id']}; family W/T/L {row['family_wins']}/{row['family_ties']}/{row['family_losses']}; worst family {row['worst_family_id']} {float(row['worst_family_delta']):+.6f}; practical bounds +0.005/-0.005"})
    write_csv(path, fields, existing + additions)
    return rows


def integrate_metrics() -> None:
    path = p1.PROGRAM_ROOT / "METRICS_LONG.csv"
    fields, existing = read_csv(path)
    existing = [row for row in existing if row["experiment_id"] != "P2_ATOMIC"]
    additions = []
    display = 5100
    for variant in CANDIDATES:
        root = ROOTS[variant] / "evaluation"
        _, panels = read_csv(root / "PROCESSBENCH_PANELS.csv")
        _, cells = read_csv(root / "PROCESSBENCH_BY_CELL.csv")
        _, strata = read_csv(root / "STEP_LENGTH_STRATA.csv")
        _, flips = read_csv(root / "PREDICTION_FLIP_SUMMARY.csv")
        _, raw_contrasts = read_csv(root / "PAIRWISE_CONTRASTS.csv")
        rows: list[dict[str, object]] = []
        for row in panels:
            if row["arm_id"] == variant:
                rows.append({"source_id":f"panel::{row['metric_id']}","population_id":row["population_id"],"cell_id":"aggregate","slice_id":"all",
                    "metric_id":"macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"],"value":row["value"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],
                    "n_rows":row["n_rows"],"n_groups":row["n_groups"],"notes":"absolute metric under per-arm grouped five-fold atomic threshold"})
        for row in cells:
            if row["arm_id"] == variant:
                rows.append({"source_id":f"cell::{row['cell_id']}","population_id":"current_common_eight_qwen","cell_id":row["cell_id"],"slice_id":row["slice_id"],
                    "metric_id":"macro_f1","value":row["official_macro_f1"],"ci_low":"","ci_high":"","n_rows":row["n_examples"],"n_groups":row["n_examples"],"notes":"descriptive per-cell metric"})
        for row in strata:
            if row["level"] == "aggregate":
                rows.append({"source_id":f"length::{row['stratum']}::{row['metric_id']}","population_id":"current_common_eight_qwen_step_length","cell_id":row["stratum"],"slice_id":row["stratum"],
                    "metric_id":row["metric_id"],"value":row["value"],"ci_low":"","ci_high":"","n_rows":row["n_error"],"n_groups":row["n_error"],"notes":"descriptive calibration-frozen true-error step-length stratum"})
        for row in flips:
            rows.append({"source_id":f"flip::{row['cell_id']}::{row['transition']}","population_id":"current_common_eight_qwen_prediction_flips","cell_id":row["cell_id"],"slice_id":row["transition"],
                "metric_id":"prediction_flip_count","value":row["count"],"ci_low":"","ci_high":"","n_rows":row["count"],"n_groups":row["count"],"notes":"exact deterministic candidate-versus-top-ten transition count"})
        for row in raw_contrasts:
            if row["right_variant_id"] == remaining.REFERENCE:
                metric = {"first_error_exact":"first_error_exact_delta","clean_abstention_accuracy":"clean_abstention_delta"}.get(row["metric_id"])
                if metric:
                    rows.append({"source_id":f"scatter::{metric}","population_id":"current_common_eight_qwen","cell_id":"aggregate","slice_id":"versus_atomic_top10",
                        "metric_id":metric,"value":row["delta"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"n_rows":6800,"n_groups":3400,"notes":"paired component delta versus atomic top-ten"})
        source = root / "REPORT_METRICS.csv"
        write_csv(source, list(rows[0]), rows)
        source_sha = sha256_file(source)
        for row in rows:
            additions.append({"phase_id":"P2","experiment_id":"P2_ATOMIC","variant_id":variant,"task_id":"processbench_first_error","dataset_id":"processbench",
                "population_id":row["population_id"],"cell_id":row["cell_id"],"slice_id":row["slice_id"],"metric_id":row["metric_id"],"value":row["value"],
                "ci_low":row["ci_low"],"ci_high":row["ci_high"],"n_rows":row["n_rows"],"n_groups":row["n_groups"],
                "comparison_group_id":f"p2a::{row['population_id']}::{row['metric_id']}","status":"COMPLETE","evidence_status":"DEVELOPMENT","display_order":display,
                "axis_value":"","source_artifact":relative(source),"source_sha256":source_sha,"source_row_selector":f"source_id={row['source_id']}","source_value_field":"value","notes":row["notes"]})
            display += 1
    write_csv(path, fields, existing + additions)


def statistical_status(variant: str, by: Mapping[tuple[str,str,str],Mapping[str,object]]) -> str:
    top10 = by[(variant, remaining.REFERENCE, "macro_f1")]
    top5 = by[(variant, "R1_ENTROPY_TOP5", "macro_f1")]
    run = json.loads((ROOTS[variant] / "RUN_MANIFEST.json").read_text())
    if run["status"] == "HARD_FAIL":
        return "HARD_FAILURE"
    if float(top10["delta"]) > 0 and float(top5["delta"]) > 0:
        return "PROMISING_UNCONFIRMED"
    return "INCONCLUSIVE"


def integrate_gates_and_status(contrasts: list[dict[str,object]]) -> None:
    by = {(str(row["left_variant_id"]),str(row["right_variant_id"]),str(row["metric_id"])):row for row in contrasts}
    gate_rows = []
    for variant in CANDIDATES:
        run = json.loads((ROOTS[variant] / "RUN_MANIFEST.json").read_text())
        _, raw = read_csv(ROOTS[variant] / "evaluation/GATES.csv")
        for row in raw:
            if "PREMISE_PROMOTION" in row["gate_id"]:
                continue
            gate_rows.append({"variant_id":variant,"gate_id":row["gate_id"],"observed":row["observed"],"threshold":row["required"],"direction":"contract",
                "passed":str(row["status"] == "PASS").lower(),"status":"COMPLETE","evidence_status":"DEVELOPMENT","notes":f"{row['detail']}; raw status={row['status']}"})
        eligible = bool(run["summary"].get("promotion_eligible", variant in {c1.CANDIDATE,c2.CANDIDATE}))
        all_pass = eligible and run["status"] != "HARD_FAIL"
        for comparator in (remaining.REFERENCE,"R1_ENTROPY_TOP5"):
            primary = by[(variant,comparator,"macro_f1")]
            exact = by[(variant,comparator,"first_error_exact")]
            clean = by[(variant,comparator,"clean_abstention_accuracy")]
            checks = [("POINT_BENEFIT",primary["delta"],f">={c1.BENEFIT}",float(primary["delta"])>=c1.BENEFIT),
                      ("CI_PRACTICAL_BENEFIT",primary["ci_low"],f">{c1.BENEFIT}",float(primary["ci_low"])>c1.BENEFIT),
                      ("NONNEGATIVE_CELLS",int(primary["wins"])+int(primary["ties"]),">=6",int(primary["wins"])+int(primary["ties"])>=6),
                      ("WORST_CELL",primary["worst_unit_delta"],f">={c1.PROMOTION_WORST_CELL_BOUND}",float(primary["worst_unit_delta"])>=c1.PROMOTION_WORST_CELL_BOUND),
                      ("EXACT_ERROR",exact["delta"],f">={c1.COMPONENT_BOUND}",float(exact["delta"])>=c1.COMPONENT_BOUND),
                      ("CLEAN_ABSTENTION",clean["delta"],f">={c1.COMPONENT_BOUND}",float(clean["delta"])>=c1.COMPONENT_BOUND)]
            for name,observed,threshold,passed in checks:
                all_pass &= passed
                gate_rows.append({"variant_id":variant,"gate_id":f"{variant}_VS_{comparator}_{name}","observed":observed,"threshold":threshold,"direction":"contract",
                    "passed":str(passed).lower(),"status":"COMPLETE","evidence_status":"DEVELOPMENT","notes":f"closed 16-contrast family; {variant} versus {comparator}"})
        gate_rows.append({"variant_id":variant,"gate_id":f"{variant}_FINAL_PROMOTION","observed":str(all_pass).lower(),"threshold":"eligible and all hard/primary/component gates pass",
            "direction":"contract","passed":str(all_pass).lower(),"status":"COMPLETE","evidence_status":"DEVELOPMENT","notes":"final atomic decision after full multiplicity closure"})
    source = p1.PROGRAM_ROOT / "phase_2/atomic/P2A_GATES.csv"
    write_csv(source, list(gate_rows[0]), gate_rows)
    source_sha = sha256_file(source)
    path = p1.PROGRAM_ROOT / "GATES_LONG.csv"
    fields, existing = read_csv(path)
    existing = [row for row in existing if row["experiment_id"] != "P2_ATOMIC"]
    additions = [{"phase_id":"P2","experiment_id":"P2_ATOMIC","variant_id":row["variant_id"],"gate_id":row["gate_id"],"metric_id":"contract_or_promotion_gate",
        "observed":row["observed"],"threshold":row["threshold"],"direction":row["direction"],"passed":row["passed"],"unit":"contract","status":row["status"],
        "evidence_status":row["evidence_status"],"source_artifact":relative(source),"source_sha256":source_sha,
        "source_row_selector":f"variant_id={row['variant_id']};gate_id={row['gate_id']}","source_value_field":"observed","notes":row["notes"]} for row in gate_rows]
    write_csv(path, fields, existing + additions)

    variants_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(variants_path.read_text())
    if not any(row["variant_id"] == remaining.EXTRA_PARENT["C8_SELF_INNOV"] for row in payload["variants"]):
        payload["variants"].append({
            "access_tier":"gray_box_single_pass","causal_validity":"prefix-valid self lag; completed-step top-ten readout",
            "decision_status":"NO_PROMOTION","detector":"equal_feature_mean response detector with per-arm grouped five-fold threshold",
            "display_name":"C8 matched IU29 top-ten parent","display_order":28,"evidence_status":"DEVELOPMENT","execution_status":"COMPLETE",
            "failure_hypothesis":"control only","fusion":"ordinary two-component IU-PCR over the original 29 streams",
            "limitations":"Derived matched-reducer mechanism control for C8; not a separately selected candidate.","method_id":"iu29",
            "novelty":"Applies the common top-ten reducer to frozen IU29 so innovation is isolated from reducer choice.",
            "parent_variant_ids":["R3_IU29"],"phase":"P2","prior_evidence":"R3 used step-max; C8 requires a common-reducer exact parent.",
            "rankable":False,"role":"derived_parent_control","signals":["registered IU29 token risk"],"statistical_status":"DESCRIPTIVE",
            "step_reducer":"frozen top-ten","supervision":"target-free IU29 fit; labels only after score freeze","task_ids":["processbench_first_error"],
            "transforms":["registered mixed-v2 IU29"],"variant_id":remaining.EXTRA_PARENT["C8_SELF_INNOV"],
        })
    for row in payload["variants"]:
        variant = row["variant_id"]
        if variant in CANDIDATES:
            run = json.loads((ROOTS[variant] / "RUN_MANIFEST.json").read_text())
            status = statistical_status(variant, by)
            row.update({"execution_status":run["status"],"decision_status":"REJECTED" if run["status"] == "HARD_FAIL" else "NO_PROMOTION",
                        "statistical_status":status,"rankable":False})
        elif variant in {"P2R_B_POS_CUSUM_TEMPLATE","P2R_B_SWVAR_TEMPLATE","P2R_B_HIGHPASS_TEMPLATE","P2R_B_DSP_CAUSAL_TEMPLATE"}:
            row.update({"execution_status":"NOT_RUN_BY_GATE","decision_status":"NO_PROMOTION",
                        "statistical_status":"NOT_EVALUATED","rankable":False})
    atomic_write_json(variants_path, payload)

    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text())
    experiment = next(row for row in experiments["experiments"] if row["experiment_id"] == "P2_ATOMIC")
    experiment.update({"execution_status":"COMPLETE","opened_variants":list(CANDIDATES),"latest_variant":"C8_SELF_INNOV",
        "opened_primary_comparisons":16,"multiplicity_family_size":16,"next_variant":None,"survivors":[],
        "verdict":"NO_ATOMIC_PROMOTION","phase2r_b_open_transforms":[],
        "phase2r_b_reason":"SWVar, CUSUM, DSP, and self-innovation premise gates did not pass; no transform is eligible."})
    reducer_experiment = next(row for row in experiments["experiments"] if row["experiment_id"] == "P2_REDUCER_STUDY")
    reducer_experiment.update({"execution_status":"COMPLETE","stage_b_status":"NOT_RUN_BY_GATE",
        "stage_b_opened_variants":[],"stage_b_reason":"No registered atomic premise passed after the closed C1--C8 family."})
    atomic_write_json(experiments_path, experiments)


def update_claims_and_plots(contrasts: list[dict[str,object]]) -> None:
    by = {(str(row["left_variant_id"]),str(row["right_variant_id"]),str(row["metric_id"])):row for row in contrasts}
    claims_path = p1.PROGRAM_ROOT / "CLAIMS.json"
    payload = json.loads(claims_path.read_text())
    payload["claims"] = [row for row in payload["claims"] if not row["claim_id"].startswith("CLAIM_C")]
    for variant in CANDIDATES:
        top10 = by[(variant,remaining.REFERENCE,"macro_f1")]
        top5 = by[(variant,"R1_ENTROPY_TOP5","macro_f1")]
        status = statistical_status(variant, by)
        if status == "HARD_FAILURE":
            text = f"{variant} failed the frozen atomic robustness contract and is not a survivor."
            top10 = by[(variant,remaining.REFERENCE,"macro_f1")]
            verdict = "SUPPORTED_HARM" if float(top10["ci_high"]) < c1.HARM else "BLOCKED"
        elif status == "PROMISING_UNCONFIRMED":
            text = f"{variant} has a positive descriptive point estimate against both references but no supported practical improvement."
            verdict = "PROMISING_UNCONFIRMED"
        else:
            text = f"{variant} is inconclusive under the closed sixteen-contrast atomic family and is not promoted."
            verdict = "INCONCLUSIVE"
        payload["claims"].append({"claim_id":f"CLAIM_{variant}","text":text,"verdict":verdict,
            "task_scope":"Current eight-Qwen ProcessBench development population under the closed P2 atomic contract.",
            "evidence_refs":["PLOT_P2_DELTA_FOREST","PLOT_P2_ATOMIC_GATE_MATRIX","PLOT_P2_EXACT_CLEAN",f"CONTRAST:{variant}:{remaining.REFERENCE}"],
            "worst_case_behavior":f"Versus top-ten, cell W/T/L is {top10['wins']}/{top10['ties']}/{top10['losses']}; worst cell {top10['worst_unit_id']} at {float(top10['worst_unit_delta']):+.6f}.",
            "claim_boundary":f"Versus top-ten: {float(top10['delta']):+.6f} [{float(top10['ci_low']):+.6f}, {float(top10['ci_high']):+.6f}]; versus top-five: {float(top5['delta']):+.6f} [{float(top5['ci_low']):+.6f}, {float(top5['ci_high']):+.6f}]. Practical benefit requires lower bound > +0.005.",
            "statistical_summary":{"metric":"macro_f1","point_delta":float(top10["delta"]),"ci_low":float(top10["ci_low"]),"ci_high":float(top10["ci_high"]),
                "benefit_bound":c1.BENEFIT,"harm_bound":c1.HARM,"bound_basis":"P2 atomic ProcessBench practical bounds","multiplicity":top10["inference"]},
            "fresh_confirmation_required":status == "PROMISING_UNCONFIRMED"})
    atomic_write_json(claims_path, payload)

    plots_path = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    plots = json.loads(plots_path.read_text())
    updates = {
        "PLOT_P2_DELTA_FOREST":("Completed C1--C8 macro-F1 deltas against both atomic top-ten and retained top-five. All primary intervals use the closed sixteen-contrast family.","All sixteen primary macro-F1 contrasts; C8's exact-parent diagnostic is separately labeled."),
        "PLOT_P2_ATOMIC_GATE_MATRIX":("Contract, robustness, eligibility, and promotion gates for the completed C1--C8 roster.","Every emitted gate for all eight completed atomic candidates."),
        "PLOT_P2_ATOMIC_LENGTH_HEATMAP":("Completed C1--C8 macro F1 by calibration-frozen short, medium, and long true-error step spans.","All completed atomic candidates with aggregate calibration-frozen step-length strata."),
    }
    for plot in plots["plots"]:
        if plot["plot_id"] in updates:
            plot["caption"], plot["selection_rule"] = updates[plot["plot_id"]]
            if plot["plot_id"] == "PLOT_P2_ATOMIC_GATE_MATRIX":
                plot["title"] = "C1--C8 atomic premise gates"
            if plot["plot_id"] == "PLOT_P2_ATOMIC_LENGTH_HEATMAP":
                plot["title"] = "Atomic performance by true-error step length"
    atomic_write_json(plots_path, plots)


def main() -> None:
    for variant in CANDIDATES:
        run = json.loads((ROOTS[variant] / "RUN_MANIFEST.json").read_text())
        if run["status"] not in {"COMPLETE","HARD_FAIL"}:
            raise RuntimeError(f"{variant} is not terminal")
    contrasts = recompute_contrasts()
    integrate_metrics()
    integrate_gates_and_status(contrasts)
    update_claims_and_plots(contrasts)
    subprocess.run([sys.executable, str(REPO / "scripts/reasoning_localization/build_reasoning_localization_report.py")], cwd=REPO, check=True)
    summary = []
    by = {(str(row["left_variant_id"]),str(row["right_variant_id"]),str(row["metric_id"])):row for row in contrasts}
    for variant in CANDIDATES:
        panel = next(row for row in read_csv(ROOTS[variant]/"evaluation/PROCESSBENCH_PANELS.csv")[1] if row["arm_id"] == variant and row["metric_id"] == "official_macro_f1")
        contrast = by[(variant,remaining.REFERENCE,"macro_f1")]
        summary.append({"variant_id":variant,"macro_f1":float(panel["value"]),"delta_vs_top10":contrast["delta"],
                        "ci":[contrast["ci_low"],contrast["ci_high"]],"status":statistical_status(variant,by)})
    print(json.dumps({"status":"P2_ATOMIC_COMPLETE","family_size":FAMILY_SIZE,"variants":summary,
                      "report_sha256":sha256_file(p1.PROGRAM_ROOT/"REPORT.html")}, indent=2))


if __name__ == "__main__":
    main()
