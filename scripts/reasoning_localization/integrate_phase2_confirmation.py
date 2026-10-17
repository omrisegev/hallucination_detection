#!/usr/bin/env python3
"""Integrate the frozen C7/C8 Llama transfer and family6 complementarity audit."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_bytes, atomic_write_json, load_npz_no_pickle, sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import empirical_midrank  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_confirmation as run  # noqa: E402

EXPERIMENT = "P2_CONFIRMATION_LLAMA4"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    atomic_write_bytes(path, handle.getvalue().encode("utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def root(variant: str) -> Path:
    return run.output_root(variant)


def candidate_decisions(variant: str) -> list[dict[str, str]]:
    _, rows = read_csv(root(variant) / "evaluation/PROCESSBENCH_DECISIONS.csv")
    return [row for row in rows if row["arm_id"] == variant]


def comparator_decisions(source_variant: str, arm: str) -> list[dict[str, str]]:
    _, rows = read_csv(root(source_variant) / "evaluation/PROCESSBENCH_DECISIONS.csv")
    return [row for row in rows if row["arm_id"] == arm]


def decision_correct(row: Mapping[str, str], scope: str) -> bool | None:
    target, prediction = int(row["true_first_error"]), int(row["prediction_step"])
    if scope == "ERROR_EXACT":
        return prediction == target if target >= 0 else None
    if scope == "CLEAN_ABSTENTION":
        return prediction == -1 if target == -1 else None
    return prediction == target


def complementarity_rows() -> list[dict[str, object]]:
    c7 = candidate_decisions(run.VARIANTS[0]); c8 = candidate_decisions(run.VARIANTS[1])
    f7 = comparator_decisions(run.VARIANTS[0], run.FAMILY6)
    f8 = comparator_decisions(run.VARIANTS[1], run.FAMILY6)
    key = lambda row: (row["cell_id"], row["row_id"])
    f7_by, f8_by = {key(row):row for row in f7}, {key(row):row for row in f8}
    if f7_by != f8_by:
        raise RuntimeError("family6 decisions differ across independently frozen candidate runs")
    maps = {
        "C7_VS_FAMILY6": ({key(row):row for row in c7}, f7_by),
        "C8_VS_FAMILY6": ({key(row):row for row in c8}, f7_by),
        "C7_VS_C8": ({key(row):row for row in c7}, {key(row):row for row in c8}),
    }
    output=[]
    for comparison,(left,right) in maps.items():
        if set(left) != set(right):
            raise RuntimeError(f"decision population mismatch: {comparison}")
        for cell_id in ("aggregate", *run.LLAMA_CELLS):
            keys = sorted(left) if cell_id == "aggregate" else sorted(k for k in left if k[0] == cell_id)
            for scope in ("ALL_DECISION", "ERROR_EXACT", "CLEAN_ABSTENTION"):
                counts={name:0 for name in ("BOTH_CORRECT","LEFT_ONLY","RIGHT_ONLY","NEITHER")}
                eligible=0
                for item in keys:
                    lc,rc=decision_correct(left[item],scope),decision_correct(right[item],scope)
                    if lc is None or rc is None: continue
                    eligible += 1
                    counts["BOTH_CORRECT" if lc and rc else "LEFT_ONLY" if lc else "RIGHT_ONLY" if rc else "NEITHER"] += 1
                if not eligible: continue
                output.append({"comparison_id":comparison,"cell_id":cell_id,"scope":scope,"n":eligible,
                    **{key.lower():value for key,value in counts.items()},
                    "left_accuracy":(counts["BOTH_CORRECT"]+counts["LEFT_ONLY"])/eligible,
                    "right_accuracy":(counts["BOTH_CORRECT"]+counts["RIGHT_ONLY"])/eligible,
                    "oracle_union_accuracy":(counts["BOTH_CORRECT"]+counts["LEFT_ONLY"]+counts["RIGHT_ONLY"])/eligible,
                    "oracle_gain_vs_best":(counts["BOTH_CORRECT"]+counts["LEFT_ONLY"]+counts["RIGHT_ONLY"])/eligible - max((counts["BOTH_CORRECT"]+counts["LEFT_ONLY"])/eligible,(counts["BOTH_CORRECT"]+counts["RIGHT_ONLY"])/eligible)})
    return output


def rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    x,y=empirical_midrank(np.asarray(left,float)),empirical_midrank(np.asarray(right,float))
    if np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(np.corrcoef(x,y)[0,1])


def score_correlation_rows() -> list[dict[str, object]]:
    output=[]
    for cell_id in run.LLAMA_CELLS:
        a7=load_npz_no_pickle(root(run.VARIANTS[0])/"score_freeze/cells"/cell_id/"scores.npz")
        a8=load_npz_no_pickle(root(run.VARIANTS[1])/"score_freeze/cells"/cell_id/"scores.npz")
        if tuple(a7["row_ids"].astype(str)) != tuple(a8["row_ids"].astype(str)) or not np.array_equal(a7["segment_offsets"],a8["segment_offsets"]):
            raise RuntimeError(f"candidate score alignment mismatch: {cell_id}")
        pairs=(("C7_VS_FAMILY6",a7[f"{run.VARIANTS[0]}__combined"],a7[f"{run.FAMILY6}__combined"]),
               ("C8_VS_FAMILY6",a8[f"{run.VARIANTS[1]}__combined"],a8[f"{run.FAMILY6}__combined"]),
               ("C7_VS_C8",a7[f"{run.VARIANTS[0]}__combined"],a8[f"{run.VARIANTS[1]}__combined"]))
        for comparison,left,right in pairs:
            output.append({"comparison_id":comparison,"cell_id":cell_id,"n_steps":len(left),"spearman_rank_correlation":rank_corr(left,right)})
    for comparison in ("C7_VS_FAMILY6","C8_VS_FAMILY6","C7_VS_C8"):
        values=[row["spearman_rank_correlation"] for row in output if row["comparison_id"]==comparison]
        output.append({"comparison_id":comparison,"cell_id":"macro","n_steps":sum(int(row["n_steps"]) for row in output if row["comparison_id"]==comparison and row["cell_id"]!="macro"),"spearman_rank_correlation":float(np.mean(values))})
    return output


def collect_contrasts() -> list[dict[str, str]]:
    rows=[]
    for variant in run.VARIANTS:
        _, raw=read_csv(root(variant)/"evaluation/PAIRWISE_CONTRASTS.csv")
        rows.extend(row for row in raw if row["left_variant_id"] == variant)
    path=run.ROOT/"P2C_CONTRASTS.csv"
    write_csv(path,list(rows[0]),rows)
    return rows


def integrate_long_metrics(complementarity: list[dict[str,object]], correlations: list[dict[str,object]]) -> None:
    path=p1.PROGRAM_ROOT/"METRICS_LONG.csv"; fields,existing=read_csv(path)
    existing=[row for row in existing if row["experiment_id"] != EXPERIMENT]
    additions=[]; display=9000
    source_variants={run.VARIANTS[0]:run.VARIANTS[0],run.VARIANTS[1]:run.VARIANTS[1],run.TOP10:run.VARIANTS[1],run.TOP5:run.VARIANTS[1],run.FAMILY6:run.VARIANTS[1],run.IU29_PARENT:run.VARIANTS[1]}
    for arm,source_variant in source_variants.items():
        _,panels=read_csv(root(source_variant)/"evaluation/PROCESSBENCH_PANELS.csv")
        _,cells=read_csv(root(source_variant)/"evaluation/PROCESSBENCH_BY_CELL.csv")
        source=root(source_variant)/"evaluation/PROCESSBENCH_PANELS.csv"; source_sha=sha256_file(source)
        for row in panels:
            if row["arm_id"] != arm: continue
            additions.append({"phase_id":"P2C","experiment_id":EXPERIMENT,"variant_id":arm,"task_id":"processbench_first_error","dataset_id":"processbench",
                "population_id":"current_llama4_scorer_transfer","cell_id":"aggregate","slice_id":"all","metric_id":"macro_f1" if row["metric_id"]=="official_macro_f1" else row["metric_id"],
                "value":row["value"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"n_rows":row["n_rows"],"n_groups":row["n_groups"],
                "comparison_group_id":f"p2c::llama4::{row['metric_id']}","status":"COMPLETE","evidence_status":"TRANSFER","display_order":display,"axis_value":"",
                "source_artifact":relative(source),"source_sha256":source_sha,"source_row_selector":f"arm_id={arm};metric_id={row['metric_id']}","source_value_field":"value",
                "notes":"Frozen scorer-family transfer; source questions and labels were previously opened in Phase 1."}); display+=1
        cell_source=root(source_variant)/"evaluation/PROCESSBENCH_BY_CELL.csv"; cell_sha=sha256_file(cell_source)
        for row in cells:
            if row["arm_id"] != arm: continue
            additions.append({"phase_id":"P2C","experiment_id":EXPERIMENT,"variant_id":arm,"task_id":"processbench_first_error","dataset_id":"processbench",
                "population_id":"current_llama4_scorer_transfer","cell_id":row["cell_id"],"slice_id":row["slice_id"],"metric_id":"macro_f1","value":row["official_macro_f1"],
                "ci_low":"","ci_high":"","n_rows":row["n_examples"],"n_groups":row["n_examples"],"comparison_group_id":"p2c::llama4::cell_macro_f1",
                "status":"COMPLETE","evidence_status":"TRANSFER","display_order":display,"axis_value":"","source_artifact":relative(cell_source),"source_sha256":cell_sha,
                "source_row_selector":f"arm_id={arm};cell_id={row['cell_id']}","source_value_field":"official_macro_f1","notes":"Descriptive per-cell scorer-family transfer metric."}); display+=1
    comp_path=run.ROOT/"P2C_COMPLEMENTARITY.csv"; comp_sha=sha256_file(comp_path)
    for row in complementarity:
        if row["cell_id"] != "aggregate": continue
        metric_variant={"C7_VS_FAMILY6":run.VARIANTS[0],"C8_VS_FAMILY6":run.VARIANTS[1],"C7_VS_C8":run.VARIANTS[0]}[row["comparison_id"]]
        additions.append({"phase_id":"P2C","experiment_id":EXPERIMENT,"variant_id":metric_variant,"task_id":"processbench_first_error","dataset_id":"processbench",
            "population_id":"current_llama4_scorer_transfer_complementarity","cell_id":f"{row['comparison_id']}::{row['scope']}","slice_id":row["scope"],"metric_id":"oracle_union_accuracy",
            "value":row["oracle_union_accuracy"],"ci_low":"","ci_high":"","n_rows":row["n"],"n_groups":row["n"],"comparison_group_id":"p2c::llama4::descriptive_oracle_union",
            "status":"COMPLETE","evidence_status":"TRANSFER","display_order":display,"axis_value":"","source_artifact":relative(comp_path),"source_sha256":comp_sha,
            "source_row_selector":f"comparison_id={row['comparison_id']};cell_id=aggregate;scope={row['scope']}","source_value_field":"oracle_union_accuracy",
            "notes":f"Inaccessible descriptive oracle; gain over best arm={float(row['oracle_gain_vs_best']):+.6f}; never an implementable router result."}); display+=1
    corr_path=run.ROOT/"P2C_SCORE_CORRELATIONS.csv"; corr_sha=sha256_file(corr_path)
    for row in correlations:
        if row["cell_id"] != "macro": continue
        metric_variant={"C7_VS_FAMILY6":run.VARIANTS[0],"C8_VS_FAMILY6":run.VARIANTS[1],"C7_VS_C8":run.VARIANTS[0]}[row["comparison_id"]]
        additions.append({"phase_id":"P2C","experiment_id":EXPERIMENT,"variant_id":metric_variant,"task_id":"processbench_first_error","dataset_id":"processbench",
            "population_id":"current_llama4_scorer_transfer_step_scores","cell_id":row["comparison_id"],"slice_id":"all","metric_id":"step_score_rank_correlation","value":row["spearman_rank_correlation"],
            "ci_low":"","ci_high":"","n_rows":row["n_steps"],"n_groups":4,"comparison_group_id":"p2c::llama4::step_rank_correlation","status":"COMPLETE","evidence_status":"TRANSFER",
            "display_order":display,"axis_value":"","source_artifact":relative(corr_path),"source_sha256":corr_sha,"source_row_selector":f"comparison_id={row['comparison_id']};cell_id=macro",
            "source_value_field":"spearman_rank_correlation","notes":"Macro mean of per-cell Spearman correlations over aligned combined step scores."}); display+=1
    write_csv(path,fields,existing+additions)


def integrate_contrasts(rows: list[dict[str,str]]) -> None:
    path=p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv"; fields,existing=read_csv(path)
    existing=[row for row in existing if row["experiment_id"] != EXPERIMENT]
    source=run.ROOT/"P2C_CONTRASTS.csv"; source_sha=sha256_file(source); additions=[]
    for row in rows:
        primary=row["right_variant_id"] in (run.TOP10,run.TOP5) and row["metric_id"]=="macro_f1"
        additions.append({"phase_id":"P2C","experiment_id":EXPERIMENT,"left_variant_id":row["left_variant_id"],"right_variant_id":row["right_variant_id"],
            "task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_llama4_scorer_transfer","metric_id":row["metric_id"],
            "delta":row["delta"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"p_adjusted":"","wins":row["wins"],"ties":row["ties"],"losses":row["losses"],
            "worst_unit_delta":row["worst_unit_delta"],"comparison_group_id":f"p2c::llama4::{'primary' if primary else 'diagnostic'}::{row['metric_id']}","status":"COMPLETE",
            "evidence_status":"TRANSFER","source_artifact":relative(source),"source_sha256":source_sha,"source_row_selector":f"contrast_id={row['contrast_id']}",
            "notes":f"{row['inference']}; practical bounds +0.005/-0.005; source questions previously opened in Phase 1."})
    write_csv(path,fields,existing+additions)


def integrate_gates() -> None:
    raw_rows=[]
    for variant in run.VARIANTS:
        _,rows=read_csv(root(variant)/"evaluation/GATES.csv")
        raw_rows.extend({**row,"variant_id":variant} for row in rows)
    all_rows=[{"variant_id":row["variant_id"],"gate_id":row["gate_id"],"observed":row["observed"],"threshold":row["required"],
               "direction":"contract","passed":str(row["status"]=="PASS").lower(),"status":"COMPLETE","evidence_status":"TRANSFER",
               "notes":f"{row['detail']}; raw status={row['status']}"} for row in raw_rows]
    source=run.ROOT/"P2C_GATES.csv"; write_csv(source,list(all_rows[0]),all_rows); source_sha=sha256_file(source)
    path=p1.PROGRAM_ROOT/"GATES_LONG.csv"; fields,existing=read_csv(path); existing=[row for row in existing if row["experiment_id"] != EXPERIMENT]
    additions=[{"phase_id":"P2C","experiment_id":EXPERIMENT,"variant_id":row["variant_id"],"gate_id":row["gate_id"],"metric_id":"contract_or_promotion_gate",
        "observed":row["observed"],"threshold":row["threshold"],"direction":row["direction"],"passed":row["passed"],"unit":"contract","status":row["status"],
        "evidence_status":row["evidence_status"],"source_artifact":relative(source),"source_sha256":source_sha,"source_row_selector":f"variant_id={row['variant_id']};gate_id={row['gate_id']}",
        "source_value_field":"observed","notes":row["notes"]} for row in all_rows]
    write_csv(path,fields,existing+additions)


def update_registries_claims_plots(rows: list[dict[str,str]], complementarity: list[dict[str,object]], correlations: list[dict[str,object]]) -> None:
    by={(row["left_variant_id"],row["right_variant_id"],row["metric_id"]):row for row in rows}
    variants_path=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json"; variants=json.loads(variants_path.read_text())
    for row in variants["variants"]:
        vid=row["variant_id"]
        if vid in (run.TOP10,run.TOP5,run.FAMILY6,run.IU29_PARENT):
            row.update({"execution_status":"COMPLETE","decision_status":"NO_PROMOTION","statistical_status":"DESCRIPTIVE","rankable":True})
        elif vid in run.VARIANTS:
            primary=[by[(vid,comp,"macro_f1")] for comp in (run.TOP10,run.TOP5)]
            if any(float(item["ci_high"]) < run.HARM for item in primary): status="SUPPORTED_HARM"
            elif all(float(item["delta"]) > 0 for item in primary): status="PROMISING_UNCONFIRMED"
            else: status="INCONCLUSIVE"
            row.update({"execution_status":"COMPLETE","decision_status":"NO_PROMOTION","statistical_status":status,"rankable":False,
                        "transfer_verdict":"failed dual-reference promotion; family6 complementarity remains descriptive only"})
    atomic_write_json(variants_path,variants)

    experiments_path=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json"; experiments=json.loads(experiments_path.read_text())
    exp=next(row for row in experiments["experiments"] if row["experiment_id"]==EXPERIMENT)
    exp.update({"execution_status":"COMPLETE","prerequisite":"Completed C1--C8 Qwen screen with C7/C8 labeled PROMISING_UNCONFIRMED and both execution registries frozen before either Llama result.",
                "opened_variants":list(run.VARIANTS),"survivors":[],"verdict":"NO_TRANSFER_PROMOTION",
                "next_variant":None,"fusion_status":"NOT_RUN_BY_GATE",
                "fusion_reason":"Neither C7 nor C8 beat both frozen top-ten and top-five references; descriptive family6 complementarity cannot waive the preregistered gate."})
    atomic_write_json(experiments_path,experiments)

    claims_path=p1.PROGRAM_ROOT/"CLAIMS.json"; claims=json.loads(claims_path.read_text())
    claims["claims"]=[row for row in claims["claims"] if not row["claim_id"].startswith("CLAIM_P2C_")]
    for vid in run.VARIANTS:
        t10,t5=by[(vid,run.TOP10,"macro_f1")],by[(vid,run.TOP5,"macro_f1")]
        family=by[(vid,run.FAMILY6,"macro_f1")]
        claims["claims"].append({"claim_id":f"CLAIM_P2C_{vid}","verdict":"INCONCLUSIVE",
            "text":f"{vid} transfers above top-five and family6 on the four Llama cells but does not improve the stronger top-ten reference, so it is not promoted.",
            "task_scope":"Four-cell Llama scorer-family ProcessBench transfer panel; same source questions were previously opened in Phase 1.",
            "evidence_refs":["PLOT_P2C_LLAMA_FOREST","PLOT_P2C_CELL_HEATMAP","PLOT_P2C_COMPLEMENTARITY",f"CONTRAST:{vid}:{run.TOP10}"],
            "worst_case_behavior":f"Versus top-ten, W/T/L={t10['wins']}/{t10['ties']}/{t10['losses']}; worst cell delta {float(t10['worst_unit_delta']):+.6f}.",
            "claim_boundary":f"Top-ten delta {float(t10['delta']):+.6f} [{float(t10['ci_low']):+.6f}, {float(t10['ci_high']):+.6f}]; top-five delta {float(t5['delta']):+.6f} [{float(t5['ci_low']):+.6f}, {float(t5['ci_high']):+.6f}]; family6 comparison {float(family['delta']):+.6f} [{float(family['ci_low']):+.6f}, {float(family['ci_high']):+.6f}] is a mechanism diagnostic.",
            "statistical_summary":{"metric":"macro_f1","point_delta":float(t10["delta"]),"ci_low":float(t10["ci_low"]),"ci_high":float(t10["ci_high"]),"benefit_bound":run.BENEFIT,"harm_bound":run.HARM,"bound_basis":"P2C transfer contract","multiplicity":t10["inference"]},
            "fresh_confirmation_required":True})
    comp=next(row for row in complementarity if row["comparison_id"]=="C8_VS_FAMILY6" and row["cell_id"]=="aggregate" and row["scope"]=="ALL_DECISION")
    corr=next(row for row in correlations if row["comparison_id"]=="C8_VS_FAMILY6" and row["cell_id"]=="macro")
    claims["claims"].append({"claim_id":"CLAIM_P2C_COMPLEMENTARITY","verdict":"DESCRIPTIVE",
        "text":"C7/C8 and family6 make partially distinct correct decisions, but the oracle union is an inaccessible ceiling and is not evidence that a deployable fusion will work.",
        "task_scope":"Four-cell Llama scorer-family ProcessBench transfer panel.","evidence_refs":["PLOT_P2C_COMPLEMENTARITY","PLOT_P2C_CELL_HEATMAP"],
        "worst_case_behavior":"No learned router or fused score was evaluated; task-label-selected oracle routing is forbidden.",
        "claim_boundary":f"For C8 versus family6, all-decision oracle union={float(comp['oracle_union_accuracy']):.6f}, gain over best arm={float(comp['oracle_gain_vs_best']):+.6f}, and macro step-score rank correlation={float(corr['spearman_rank_correlation']):.6f}.",
        "statistical_summary":{"metric":"descriptive_oracle_union","point_delta":float(comp["oracle_gain_vs_best"]),"ci_low":float(comp["oracle_gain_vs_best"]),"ci_high":float(comp["oracle_gain_vs_best"]),"benefit_bound":0.0,"harm_bound":0.0,"bound_basis":"Exact descriptive value repeated as a zero-width display interval; not inferential.","multiplicity":"none"},
        "fresh_confirmation_required":True})
    atomic_write_json(claims_path,claims)

    plots_path=p1.PROGRAM_ROOT/"PLOT_MANIFEST.json"; plots=json.loads(plots_path.read_text())
    plots["plots"]=[row for row in plots["plots"] if not row["plot_id"].startswith("PLOT_P2C_")]
    plots["plots"].extend([
        {"plot_id":"PLOT_P2C_LLAMA_FOREST","title":"C7/C8 transfer deltas on the Llama scorer panel","phase":"P2C","kind":"contrast_forest","source_table":"CONTRASTS_LONG.csv",
         "selection":{"experiment_id":EXPERIMENT,"metric_id":"macro_f1","status":"COMPLETE"},"x_field":"delta","y_field":"left_variant_id","series_field":"right_variant_id",
         "comparison_group":"frozen four-cell Llama scorer-family transfer","bootstrap_definition":"20,000 paired source-question grouped draws; primary intervals are Bonferroni-simultaneous across four C7/C8 by required-reference contrasts.",
         "selection_rule":"Both candidates against top-ten, top-five, family6, and C8's exact IU29 parent; diagnostics are explicitly separated from required references.",
         "legend":["Point and line = paired delta and interval","Required references = top-ten and top-five","family6 and IU29 = mechanism diagnostics"],
         "caption":"Neither candidate beats the stronger top-ten reference; positive family6 deltas motivate only descriptive complementarity analysis."},
        {"plot_id":"PLOT_P2C_CELL_HEATMAP","title":"Llama transfer macro F1 by scorer cell","phase":"P2C","kind":"heatmap","source_table":"METRICS_LONG.csv",
         "selection":{"experiment_id":EXPERIMENT,"metric_id":"macro_f1","population_id":"current_llama4_scorer_transfer","status":"COMPLETE"},"x_field":"cell_id","y_field":"variant_id","series_field":"value",
         "comparison_group":"same four Llama cells and evaluator","bootstrap_definition":"Per-cell values are descriptive; primary uncertainty is in the paired forest.",
         "selection_rule":"All frozen C7/C8 candidates and registered Llama comparators on the common cell panel.","legend":["Cell color = macro F1","Rows = candidates and comparators"],
         "caption":"Cell-level behavior exposes the one-win/three-loss pattern versus top-ten that the aggregate alone can hide."},
        {"plot_id":"PLOT_P2C_COMPLEMENTARITY","title":"Candidate-family6 decision complementarity ceilings","phase":"P2C","kind":"heatmap","source_table":"METRICS_LONG.csv",
         "selection":{"experiment_id":EXPERIMENT,"metric_id":"oracle_union_accuracy","population_id":"current_llama4_scorer_transfer_complementarity","status":"COMPLETE"},"x_field":"cell_id","y_field":"variant_id","series_field":"value",
         "comparison_group":"deterministic paired decision overlap on the same Llama rows","bootstrap_definition":"Descriptive exact counts; no inferential fusion claim.",
         "selection_rule":"All-decision, exact-error, and clean-abstention oracle unions for C7-family6, C8-family6, and C7-C8.",
         "legend":["Value = fraction correct by either arm","Oracle union is inaccessible and label-dependent"],
         "caption":"Distinct correct decisions establish room for a future preregistered fusion hypothesis, not evidence that a learned or equal-weight fusion will realize the ceiling."},
    ])
    atomic_write_json(plots_path,plots)


def main() -> None:
    for variant in run.VARIANTS:
        manifest=json.loads((root(variant)/"RUN_MANIFEST.json").read_text())
        if manifest["status"] not in {"COMPLETE","HARD_FAIL"}: raise RuntimeError(f"{variant} is not terminal")
    complementarity=complementarity_rows(); correlations=score_correlation_rows()
    comp_path=run.ROOT/"P2C_COMPLEMENTARITY.csv"; write_csv(comp_path,list(complementarity[0]),complementarity)
    corr_path=run.ROOT/"P2C_SCORE_CORRELATIONS.csv"; write_csv(corr_path,list(correlations[0]),correlations)
    contrasts=collect_contrasts(); integrate_long_metrics(complementarity,correlations); integrate_contrasts(contrasts); integrate_gates()
    update_registries_claims_plots(contrasts,complementarity,correlations)
    summary={"schema":"reasoning-localization-p2-confirmation-summary-v1","status":"COMPLETE","experiment_id":EXPERIMENT,
             "survivors":[],"verdict":"NO_TRANSFER_PROMOTION","fusion_status":"NOT_RUN_BY_GATE",
             "artifacts":[{"path":relative(path),"sha256":sha256_file(path)} for path in (comp_path,corr_path,run.ROOT/"P2C_CONTRASTS.csv",run.ROOT/"P2C_GATES.csv")]}
    summary["payload_sha256"]=sha256_file(comp_path); atomic_write_json(run.ROOT/"SUMMARY.json",summary)
    subprocess.run([sys.executable,str(REPO/"scripts/reasoning_localization/build_reasoning_localization_report.py")],cwd=REPO,check=True)
    print(json.dumps({**summary,"report_sha256":sha256_file(p1.PROGRAM_ROOT/"REPORT.html")},indent=2))


if __name__ == "__main__": main()
