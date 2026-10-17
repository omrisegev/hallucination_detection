#!/usr/bin/env python3
"""Mechanically bind completed P0-S4/S5 artifacts into the living report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "results/reasoning_localization_03662_v1"
S4 = REPORT / "phase_0/p0_s4_fivefold_split_bridge"
S5 = REPORT / "phase_0/p0_s5_population_bridges"


def read_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def write_csv(path: Path, rows):
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path):
    return path.relative_to(REPO).as_posix()


def upsert(rows, key, additions):
    ids = {row[key] for row in additions}
    return [row for row in rows if row[key] not in ids] + additions


def variant_rows():
    return [
        {
            "variant_id": "P0_S4_IU29_STEP_MAX_LOCAL_DETECTOR_FIVEFOLD",
            "display_name": "P0-S4 IU29 / local max / five-fold split",
            "method_id": "iu29", "phase": "P0", "role": "audit_bridge_state",
            "parent_variant_ids": ["P0_S3B_IU29_STEP_MAX_LOCAL_DETECTOR"],
            "signals": ["unchanged frozen LOCAL_IU29 score and locator"],
            "transforms": ["unchanged mixed-v2 IU29 token transform"],
            "detector": "unchanged maximum LOCAL_IU29 token risk",
            "step_reducer": "unchanged step maximum / token argmax mapped to step",
            "fusion": "unchanged LOCAL_IU29 two-component IU-PCR",
            "novelty": "One-factor split bridge from historical 40/20 threshold roles to deterministic five-fold source-group cross-fit on the same rows.",
            "access_tier": "gray_box_single_pass",
            "supervision": "labels enter only held-fold threshold fitting and metrics; score and locator are frozen",
            "causal_validity": "retrospective completed-trace local-only audit",
            "prior_evidence": "P0-S3B fixed the IU29 representation; S4 isolates only the threshold split.",
            "limitations": "Historical audit population; clean abstention falls materially even though aggregate macro F1 is inconclusive.",
            "failure_hypothesis": "Cross-fit thresholds shift the detector operating point toward false positives.",
            "execution_status": "COMPLETE", "statistical_status": "INCONCLUSIVE",
            "decision_status": "NO_PROMOTION", "evidence_status": "RETROSPECTIVE",
            "rankable": False, "display_order": 8,
        },
        {
            "variant_id": "P0_S5A_IU29_STEP_MAX_LOCAL_FIVEFOLD_QWEN8",
            "display_name": "P0-S5A IU29 token-only / current eight-Qwen panel",
            "method_id": "iu29", "phase": "P0", "role": "audit_population_transfer",
            "parent_variant_ids": ["P0_S4_IU29_STEP_MAX_LOCAL_DETECTOR_FIVEFOLD"],
            "signals": ["frozen token_iu29 step-only score"], "transforms": ["frozen LOCAL_IU29 token transform"],
            "detector": "maximum token-only IU29 risk under the frozen current evaluator",
            "step_reducer": "step maximum", "fusion": "token-only LOCAL_IU29",
            "novelty": "Population-regime transfer to the frozen current Qwen-3 4B/8B eight-cell panel; no new fit or inference.",
            "access_tier": "gray_box_single_pass", "supervision": "frozen label-free adapter and five-fold threshold contract",
            "causal_validity": "retrospective completed-trace localization",
            "prior_evidence": "The dual-build localization release provides a provenance-bound current-population token-only IU29 adapter.",
            "limitations": "Generated traces and scorer models differ from S4, so the cumulative score shift is descriptive and not a paired factor effect.",
            "failure_hypothesis": "Population-specific trace difficulty or scorer behavior changes the apparent absolute F1.",
            "execution_status": "COMPLETE", "statistical_status": "DESCRIPTIVE", "decision_status": "NO_PROMOTION",
            "evidence_status": "RETROSPECTIVE", "rankable": False, "display_order": 9,
        },
        {
            "variant_id": "P0_S5B_IU29_STEP_MAX_LOCAL_FIVEFOLD_FULL12",
            "display_name": "P0-S5B IU29 token-only / current full twelve-cell panel",
            "method_id": "iu29", "phase": "P0", "role": "audit_population_transfer",
            "parent_variant_ids": ["P0_S5A_IU29_STEP_MAX_LOCAL_FIVEFOLD_QWEN8"],
            "signals": ["frozen token_iu29 step-only score"], "transforms": ["frozen LOCAL_IU29 token transform"],
            "detector": "maximum token-only IU29 risk under the frozen current evaluator", "step_reducer": "step maximum",
            "fusion": "token-only LOCAL_IU29",
            "novelty": "Population-panel extension adding four Llama-3.1 cells to the frozen eight-Qwen panel.",
            "access_tier": "gray_box_single_pass", "supervision": "frozen label-free adapter and five-fold threshold contract",
            "causal_validity": "retrospective completed-trace localization",
            "prior_evidence": "The certified dual-build release contains the complete twelve-cell common-access panel.",
            "limitations": "The Llama rows are independent generated traces, so S5B-S5A is a panel-composition diagnostic rather than a row-paired treatment effect.",
            "failure_hypothesis": "A third scorer family exposes population sensitivity hidden by the Qwen-only panel.",
            "execution_status": "COMPLETE", "statistical_status": "INCONCLUSIVE", "decision_status": "NO_PROMOTION",
            "evidence_status": "RETROSPECTIVE", "rankable": False, "display_order": 10,
        },
    ]


def metric_rows(existing):
    fields = list(existing[0])
    additions = []
    def add(variant, population, n_rows, n_groups, display, axis, source, selector, value, lo="", hi="", notes=""):
        row = {k: "" for k in fields}
        metric = selector.split("metric_id=")[-1]
        row.update({"phase_id":"P0","experiment_id":"P0_BRIDGE","variant_id":variant,
                    "task_id":"processbench_first_error","dataset_id":"processbench","population_id":population,
                    "cell_id":"aggregate","slice_id":"all","metric_id":metric,
                    "value":value,"ci_low":lo,"ci_high":hi,"n_rows":str(n_rows),"n_groups":str(n_groups),
                    "comparison_group_id":f"{variant.lower()}_{metric}","status":"COMPLETE",
                    "evidence_status":"RETROSPECTIVE","display_order":str(display),"axis_value":str(axis),
                    "source_artifact":rel(source),"source_sha256":sha(source),"source_row_selector":selector,
                    "source_value_field":"value","notes":notes})
        additions.append(row)
    s4src = S4 / "P0_S4_LOCAL_AGGREGATE.csv"
    for r in read_csv(s4src):
        add(r["variant_id"], "historical_stage4_eight_cell_audit", 1270, 635, 8, 7, s4src,
            f"variant_id={r['variant_id']};metric_id={r['metric_id']}", r["value"], notes="Five-fold split bridge on unchanged historical rows and frozen IU29 score/locator.")
    flipsrc = S4 / "P0_S4_PREDICTION_FLIP_SUMMARY.csv"
    for index, r in enumerate(read_csv(flipsrc), start=65):
        row = {k: "" for k in fields}
        row.update({"phase_id":"P0","experiment_id":"P0_BRIDGE","variant_id":"P0_S4_IU29_STEP_MAX_LOCAL_DETECTOR_FIVEFOLD",
                    "task_id":"processbench_first_error","dataset_id":"processbench","population_id":"historical_stage4_eight_cell_audit",
                    "cell_id":r["flip_kind"],"slice_id":"all","metric_id":"prediction_flip_count","value":r["count"],
                    "n_rows":"1270","n_groups":"635","comparison_group_id":"p0_s4_prediction_flip_count_d12d651c",
                    "status":"COMPLETE","evidence_status":"RETROSPECTIVE","display_order":str(index),
                    "source_artifact":rel(flipsrc),"source_sha256":sha(flipsrc),"source_row_selector":f"flip_kind={r['flip_kind']}",
                    "source_value_field":"count","notes":"Deterministic S3B-to-S4 split-only prediction flips."})
        additions.append(row)
    s5src = S5 / "P0_S5_METRICS.csv"
    for variant, pop, n, order, axis in [
        ("P0_S5A_IU29_STEP_MAX_LOCAL_FIVEFOLD_QWEN8","current_common_eight_qwen",6800,9,8),
        ("P0_S5B_IU29_STEP_MAX_LOCAL_FIVEFOLD_FULL12","current_full_twelve_cell",10200,10,9),
    ]:
        for r in read_csv(s5src):
            if r["variant_id"] == variant and r["metric_id"]:
                add(variant,pop,n,n,order,axis,s5src,f"variant_id={variant};metric_id={r['metric_id']}",
                    r["value"],r["ci_low"],r["ci_high"],"Frozen current-population raw state; not paired to the historical S4 rows.")
    return [r for r in existing if r["variant_id"] not in {x["variant_id"] for x in variant_rows()}] + additions


def contrast_rows(existing):
    fields = list(existing[0]); additions=[]
    def add(src, row, population, semantics):
        out={k:"" for k in fields}; out.update({"phase_id":"P0","experiment_id":"P0_BRIDGE",
            "left_variant_id":row["left_variant_id"],"right_variant_id":row["right_variant_id"],
            "task_id":"processbench_first_error","dataset_id":"processbench","population_id":population,
            "metric_id":row["metric_id"],"delta":row["delta"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],
            "wins":row["wins"],"ties":row["ties"],"losses":row["losses"],"worst_unit_delta":row["worst_unit_delta"],
            "comparison_group_id":f"{row['left_variant_id'].lower()}_{row['metric_id']}","status":"COMPLETE",
            "evidence_status":"RETROSPECTIVE","source_artifact":rel(src),"source_sha256":sha(src),
            "source_row_selector":f"left_variant_id={row['left_variant_id']};right_variant_id={row['right_variant_id']};metric_id={row['metric_id']}",
            "notes":semantics}); additions.append(out)
    s4src=S4/"P0_S4_CONTRASTS.csv"
    for r in read_csv(s4src): add(s4src,r,"historical_stage4_eight_cell_audit","Paired adjacent split-only edge on the same 635 source-question groups.")
    s5src=S5/"P0_S5_CONTRASTS.csv"
    for r in read_csv(s5src): add(s5src,r,"current_panel_composition_qwen8_to_full12",r["contrast_semantics"])
    ids={x["variant_id"] for x in variant_rows()}
    return [r for r in existing if r["left_variant_id"] not in ids] + additions


def gate_rows(existing):
    fields=list(existing[0]); additions=[]
    for source, default_variant in [(S4/"P0_S4_GATES.csv","P0_S4_IU29_STEP_MAX_LOCAL_DETECTOR_FIVEFOLD"),(S5/"P0_S5_GATES.csv",None)]:
        for r in read_csv(source):
            variant=r.get("variant_id") or default_variant
            out={k:"" for k in fields}; out.update({"phase_id":"P0","experiment_id":"P0_BRIDGE","variant_id":variant,
                "gate_id":r["gate_id"],"metric_id":r.get("metric_id",r["gate_id"].lower()),"observed":r["observed"],
                "threshold":r["threshold"],"direction":r["direction"],"passed":r["passed"],"unit":r.get("unit","audit"),
                "status":r["status"],"evidence_status":r["evidence_status"],"source_artifact":rel(source),
                "source_sha256":sha(source),"source_row_selector":f"gate_id={r['gate_id']}","source_value_field":"observed",
                "notes":"Frozen Phase-0 provenance, access, population, or execution gate."}); additions.append(out)
    ids={x["variant_id"] for x in variant_rows()}
    return [r for r in existing if r["variant_id"] not in ids] + additions


def main():
    variants=read_json(REPORT/"VARIANT_REGISTRY.json")
    variants["variants"]=upsert(variants["variants"],"variant_id",variant_rows())
    write_json(REPORT/"VARIANT_REGISTRY.json",variants)
    write_csv(REPORT/"METRICS_LONG.csv",metric_rows(read_csv(REPORT/"METRICS_LONG.csv")))
    write_csv(REPORT/"CONTRASTS_LONG.csv",contrast_rows(read_csv(REPORT/"CONTRASTS_LONG.csv")))
    write_csv(REPORT/"GATES_LONG.csv",gate_rows(read_csv(REPORT/"GATES_LONG.csv")))

    experiments=read_json(REPORT/"EXPERIMENT_REGISTRY.json")
    p0=next(x for x in experiments["experiments"] if x["experiment_id"]=="P0_BRIDGE")
    p0["execution_status"]="COMPLETE"
    p0["bootstrap"]="S0 reproduces the historical grouped interval; S1-S4 use 20,000 paired source-question draws; S5 uses 20,000 within-cell grouped panel draws and labels nonpaired population edges explicitly"
    p0["report_sections"]=["p0_waterfall","p0_bridge_forest","p0_prediction_flips","p0_s2a_prediction_flips","p0_s2b_prediction_flips","p0_s3a_prediction_flips","p0_s3b_prediction_flips","p0_s4_prediction_flips","p0_population_transfer"]
    write_json(REPORT/"EXPERIMENT_REGISTRY.json",experiments)

    claims=read_json(REPORT/"CLAIMS.json")
    new_claims=[
      {"claim_id":"CLAIM_P0_S4_SPLIT_BRIDGE","text":"Changing only the threshold split from historical 40/20 roles to five-fold source-group cross-fit changes macro F1 by -0.00564, with a paired interval crossing zero; the aggregate direction is inconclusive, while clean abstention shows a supported component loss.","verdict":"INCONCLUSIVE","task_scope":"Historical eight-cell ProcessBench audit population only.","evidence_refs":["PLOT_P0_WATERFALL","PLOT_P0_BRIDGE_FOREST","PLOT_P0_S4_PREDICTION_FLIPS","CONTRAST:P0_S4_IU29_STEP_MAX_LOCAL_DETECTOR_FIVEFOLD:P0_S3B_IU29_STEP_MAX_LOCAL_DETECTOR","TABLE_GATES"],"worst_case_behavior":"Clean abstention delta is -0.04343 with CI [-0.07775, -0.00915]; worst scorer-cell macro delta is -0.06736.","claim_boundary":"The macro interval crossing zero is not rejection or parity. The cumulative -0.07221 displacement from 0.3662 includes all earlier factors.","statistical_summary":{"metric":"macro_f1","point_delta":-0.005639186754005907,"ci_low":-0.02022305591272321,"ci_high":0.009502877546169077,"benefit_bound":0.0,"harm_bound":0.0,"bound_basis":"Phase-0 directional audit boundary only.","multiplicity":"Single registered adjacent split edge; 20000 paired source-question draws."},"fresh_confirmation_required":True},
      {"claim_id":"CLAIM_P0_S5A_QWEN_POPULATION","text":"The frozen token-only IU29 contract scores 0.29312 macro F1 [0.27852, 0.30625] on the current eight-Qwen panel, close in raw absolute value to S4's 0.29402 but not pairwise comparable.","verdict":"DESCRIPTIVE","task_scope":"Current frozen eight-cell Qwen ProcessBench panel.","evidence_refs":["PLOT_P0_WATERFALL","PLOT_P0_POPULATION_TRANSFER","TABLE_GATES"],"worst_case_behavior":"The lowest cell is 0.26306 on OlympiadBench/Qwen-3 4B.","claim_boundary":"Different scorer models and generated traces prohibit an S4-to-S5A paired effect or prediction-flip claim; raw proximity is descriptive only.","fresh_confirmation_required":True},
      {"claim_id":"CLAIM_P0_S5B_FULL_POPULATION","text":"Extending the frozen Qwen panel with four Llama-3.1 cells gives 0.29440 macro F1 [0.28226, 0.30491]; the panel-composition delta is +0.00128 with an interval crossing zero.","verdict":"INCONCLUSIVE","task_scope":"Current frozen twelve-cell ProcessBench panel and its Qwen-only subset.","evidence_refs":["PLOT_P0_POPULATION_TRANSFER","CONTRAST:P0_S5B_IU29_STEP_MAX_LOCAL_FIVEFOLD_FULL12:P0_S5A_IU29_STEP_MAX_LOCAL_FIVEFOLD_QWEN8","TABLE_GATES"],"worst_case_behavior":"Two of four family-level composition deltas are negative; the worst is -0.00486.","claim_boundary":"The added Llama traces are independent generated responses, so this is a panel-composition diagnostic, not a row-paired causal effect or evidence of parity.","statistical_summary":{"metric":"macro_f1","point_delta":0.0012778889191231158,"ci_low":-0.006708916091292374,"ci_high":0.00920811521404516,"benefit_bound":0.0,"harm_bound":0.0,"bound_basis":"Phase-0 descriptive panel-composition boundary only.","multiplicity":"20000 within-cell source-question resamples; shared Qwen cells plus independent Llama cells."},"fresh_confirmation_required":True},
    ]
    claims["claims"]=upsert(claims["claims"],"claim_id",new_claims)
    write_json(REPORT/"CLAIMS.json",claims)

    plots=read_json(REPORT/"PLOT_MANIFEST.json")
    waterfall=next(x for x in plots["plots"] if x["plot_id"]=="PLOT_P0_WATERFALL")
    waterfall["caption"]="Raw Phase-0 states from 0.3662 through split and population bridges. S0-S4 adjacent edges support one-factor attribution; S4-S5 population edges are visually continuous but explicitly nonpaired and descriptive."
    waterfall["selection"]["variant_id"]=["R2_HISTORICAL_FAMILY6_BRIDGE","P0_S1_FAMILY6_STEP_MAX","P0_S2A_FAMILY6_STEP_MAX_DUFS_DETECTOR","P0_S2B_FAMILY6_STEP_MAX_LOCAL_DETECTOR","P0_S3A_RAW_ENTROPY_STEP_MAX_LOCAL_DETECTOR","P0_S3B_IU29_STEP_MAX_LOCAL_DETECTOR","P0_S4_IU29_STEP_MAX_LOCAL_DETECTOR_FIVEFOLD","P0_S5A_IU29_STEP_MAX_LOCAL_FIVEFOLD_QWEN8","P0_S5B_IU29_STEP_MAX_LOCAL_FIVEFOLD_FULL12"]
    waterfall["legend"]=["Bar = raw registered state","S0-S4 = one-factor common-row edges","S4-S5 = nonpaired population states; no causal attribution"]
    new_plots=[
      {"plot_id":"PLOT_P0_S4_PREDICTION_FLIPS","phase":"P0","kind":"heatmap","title":"P0-S4 split prediction-flip audit","caption":"Exact S3B-to-S4 decision transitions after changing only the threshold split.","source_table":"METRICS_LONG.csv","selection":{"experiment_id":"P0_BRIDGE","metric_id":"prediction_flip_count","variant_id":"P0_S4_IU29_STEP_MAX_LOCAL_DETECTOR_FIVEFOLD","status":"COMPLETE"},"x_field":"cell_id","y_field":"variant_id","series_field":"value","legend":["Cell = exact transition count","NO_FLIP remains visible"],"comparison_group":"p0_s4_prediction_flips","bootstrap_definition":"Not applicable; exact transitions over 1270 scorer rows.","selection_rule":"Every category emitted by the frozen S4 runner."},
      {"plot_id":"PLOT_P0_POPULATION_TRANSFER","phase":"P0","kind":"forest","title":"P0 current-population transfer states","caption":"Raw token-only IU29 macro F1 on current eight-Qwen and full twelve-cell panels. These are not paired to the historical S4 rows.","source_table":"METRICS_LONG.csv","selection":{"experiment_id":"P0_BRIDGE","metric_id":"macro_f1","variant_id":["P0_S5A_IU29_STEP_MAX_LOCAL_FIVEFOLD_QWEN8","P0_S5B_IU29_STEP_MAX_LOCAL_FIVEFOLD_FULL12"],"status":"COMPLETE"},"x_field":"value","y_field":"variant_id","series_field":"population_id","legend":["Point = equal-cell macro F1","Line = 95% grouped panel interval","Panels are separate populations"],"comparison_group":"P0 current-population raw states","bootstrap_definition":"20000 source-question resamples within each scorer cell; equal-cell panel average.","selection_rule":"Both frozen current-population token-only IU29 states; no historical-context rows."}
    ]
    plots["plots"]=upsert(plots["plots"],"plot_id",new_plots)
    write_json(REPORT/"PLOT_MANIFEST.json",plots)


if __name__=="__main__": main()
