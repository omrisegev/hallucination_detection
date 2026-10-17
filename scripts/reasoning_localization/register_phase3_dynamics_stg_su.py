#!/usr/bin/env python3
"""Register the bounded dynamics-only STG-SU ladder."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402

PROGRAM = REPO / "results/reasoning_localization_03662_v1"
METHOD = "dynamics_stg_su_family_expert"
EXPERIMENT = "P3_DYNAMICS_STG_SU"
VARIANTS = (
    "P3S0_DYNAMICS_IU_PARENT",
    "P3S1_DYNAMICS_CANONICAL_SU",
    "P3S2_DYNAMICS_STG_SU",
    "P3S3_DYNAMICS_STG_PERMUTED_SUPPORT",
    "P3S4_DYNAMICS_RANDOM_SUPPORT_CONTROL",
)


def upsert(rows: list[dict], key: str, row: dict) -> None:
    hits = [i for i, current in enumerate(rows) if current.get(key) == row[key]]
    if len(hits) > 1: raise RuntimeError(f"duplicate {key}={row[key]}")
    if hits: rows[hits[0]] = row
    else: rows.append(row)


def main() -> None:
    methods_path=PROGRAM/"METHOD_REGISTRY.json"; methods=json.loads(methods_path.read_text())
    upsert(methods["methods"],"method_id",{
        "method_id":METHOD,"display_name":"Dynamics STG-SU family expert",
        "problem":"Test whether fold-stable sparse correlated-error support improves the strongest dynamics-IU family expert.",
        "plain_summary":"Uses nested donor-only stochastic gates to select recurring off-diagonal dynamics pairs, then refits the unchanged SU-PCR predictor.",
        "input_operation_output":"fourteen donor dynamics views -> nested STG support -> fixed-support SU-PCR -> unchanged equal outer family fusion -> token risk",
        "novelty":"Transfers the corrected STG support-extractor idea to one eligible localization family with exact canonical, permutation and random-support controls.",
        "assumptions":["Correlated errors among dynamics views are sparse and recur across donor response folds.","Learned pair identity matters beyond support cardinality."],
        "limitations":["Opened development population.","Nested STG is computationally heavier than IU and does not have a scalar zero-strength path."],
        "references":["docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_DYNAMICS_STG_SU_V1.md","docs/experiments/REASONING_LOCALIZATION_03662_STG_GRAPH_TRANSFER_V1.md","spectral_utils/token_local_fusion.py"],
    }); atomic_write_json(methods_path,methods)

    variants_path=PROGRAM/"VARIANT_REGISTRY.json"; variants=json.loads(variants_path.read_text()); parent=deepcopy(next(r for r in variants["variants"] if r["variant_id"]=="P3E1_DYNAMICS_IU_ONLY"))
    specs=(
        (VARIANTS[0],"Dynamics IU matched parent",188,["P3E1_DYNAMICS_IU_ONLY"],"matched_parent",False,"Exact P3E1 parent."),
        (VARIANTS[1],"Dynamics canonical SU-PCR",189,[VARIANTS[0]],"canonical_method_control",True,"Canonical threshold-based sparse support and SU refit."),
        (VARIANTS[2],"Dynamics STG-SU-PCR",190,[VARIANTS[1]],"stg_candidate",True,"Four-of-five fold-stable stochastic-gate support replaces canonical support extraction."),
        (VARIANTS[3],"Dynamics STG permuted-support control",191,[VARIANTS[2]],"negative_control",False,"Deterministic feature-label permutation of the frozen STG support."),
        (VARIANTS[4],"Dynamics random-support control",192,[VARIANTS[2]],"negative_control",False,"Mean of twenty cardinality-matched random-support SU refits."),
    )
    for vid,name,order,parents,role,rankable,novelty in specs:
        row=deepcopy(parent); row.update({"variant_id":vid,"display_name":name,"display_order":order,"phase":"P3","method_id":METHOD,"parent_variant_ids":parents,"role":role,"rankable":rankable,"execution_status":"PLANNED","decision_status":"PENDING","evidence_status":"DEVELOPMENT","statistical_status":"NOT_EVALUATED","fusion":novelty,"novelty":novelty,"supervision":"five outer donor folds with nested five-fold STG covariance validation; scores frozen before labels","limitations":"Opened Qwen-eight development population; only entropy_dynamics changes."}); upsert(variants["variants"],"variant_id",row)
    atomic_write_json(variants_path,variants)

    experiments_path=PROGRAM/"EXPERIMENT_REGISTRY.json"; experiments=json.loads(experiments_path.read_text())
    upsert(experiments["experiments"],"experiment_id",{
        "experiment_id":EXPERIMENT,"display_name":"Dynamics canonical SU versus fold-stable STG-SU","phase":"P3","execution_status":"PLANNED",
        "question":"Does learned stable sparse error support improve the dynamics family expert beyond IU, canonical SU and matched support controls?",
        "prerequisite":"P3E dynamics IU is promising unconfirmed; P3F/P3K show no DUFS value and do not hard-fail the family.","population_ids":["current_common_eight_qwen"],"task_ids":["processbench_first_error"],"variant_order":list(VARIANTS),
        "registered_comparators":[VARIANTS[0],VARIANTS[1],VARIANTS[3],VARIANTS[4],"P3E1_DYNAMICS_IU_ONLY"],"primary_metrics":["paired_delta_macro_f1"],
        "bootstrap":"20,000 paired whole-source-question draws; Bonferroni across five frozen macro-F1 contrasts",
        "promotion_gates":["S2-S0 point delta >= +0.003","S2-S0 simultaneous CI lower > +0.003","S2 beats S3 and S4 with simultaneous CI lower > 0 for a support-mechanism claim","P3E parent alias <= 1e-12","H0 abstention mismatch = 0","exact delta >= -0.010","worst cell >= -0.020","all supports theorem-valid and converged"],
        "next_variant":VARIANTS[0],"report_sections":["p3_parent_fusion","p3_complexity"],"protocol":"docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_DYNAMICS_STG_SU_V1.md",
    })
    next(r for r in experiments["experiments"] if r["experiment_id"]=="P3_FUSION")["next_variant"]=VARIANTS[0]
    atomic_write_json(experiments_path,experiments); print(json.dumps({"status":"PLANNED","experiment":EXPERIMENT,"variants":list(VARIANTS)},indent=2))


if __name__=="__main__": main()
