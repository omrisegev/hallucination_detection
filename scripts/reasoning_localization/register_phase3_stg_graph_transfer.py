#!/usr/bin/env python3
"""Register the evidence-bounded STG-SU and temporal-graph Phase-3 branch."""
from __future__ import annotations
import json,sys
from copy import deepcopy
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402

IDS=("H_STG_SU_FINAL_CONTEXT","P3G_T0_PARENT","P3G_F1_STG_FEATURE_SUPPORT","P3G_T1_TEMPORAL_GRAPH")
def main():
 vp=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json";d=json.loads(vp.read_text());rows=d["variants"];by={r["variant_id"]:r for r in rows}
 if any(x in by for x in IDS):raise RuntimeError("STG graph branch already registered")
 context=deepcopy(by["H_TRANSFORM_HIERARCHY"]);context.update({"variant_id":IDS[0],"display_name":"Historical corrected STG-SU-PCR final-answer detector","display_order":150,
  "phase":"CONTEXT","execution_status":"CONTEXT_ONLY","decision_status":"NO_PROMOTION","evidence_status":"CONTEXT_ONLY","statistical_status":"DESCRIPTIVE","rankable":False,
  "role":"historical_context","method_id":"fusion_selection","signals":["fold-stable learned sparse covariance support","canonical SU-PCR predictor"],
  "fusion":"STG support extractor followed by unchanged SU-PCR","novelty":"Replaces hard-threshold SU support with stochastic-gate fold-stable support.",
  "prior_evidence":"Corrected final-answer 24-cell equal-family AUROC 0.742875; near IU/DUFS-LIU parity but no supported advantage over IU or matched random support.",
  "limitations":"Final-answer detection only; historically open; source files untracked and absent from side-worktree HEAD 66abed7; early orientation-inverted report excluded."})
 base=deepcopy(by["P3T_T0_FROZEN_PARENT"])
 planned=[]
 specs=((IDS[1],"STG/graph exact compact parent","Exact frozen Phase-2C survivor roster and reducer; zero-strength alias target."),
        (IDS[2],"STG sparse feature/family support","Learn fold-stable sparse support or Laplacian only among eligible feature/family blocks."),
        (IDS[3],"Donor-only within-answer temporal graph","Apply one frozen masked token/bin/step graph operator before the same reducer."))
 for i,(vid,name,novelty) in enumerate(specs,151):
  r=deepcopy(base);r.update({"variant_id":vid,"display_name":name,"display_order":i,"phase":"P3","execution_status":"PLANNED","decision_status":"PENDING","evidence_status":"DEVELOPMENT",
   "statistical_status":"NOT_EVALUATED","rankable":vid!=IDS[1],"role":"graph_parent" if vid==IDS[1] else "survivor_gated_graph_candidate","parent_variant_ids":[] if vid==IDS[1] else [IDS[1]],
   "novelty":novelty,"limitations":"May open only after a compact Phase-2C survivor set freezes; no localization result exists yet.","task_ids":["processbench_first_error"]})
  planned.append(r)
 rows.extend([context,*planned]);atomic_write_json(vp,d)
 ep=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json";e=json.loads(ep.read_text())
 if any(r["experiment_id"]=="P3_STG_GRAPH_TRANSFER" for r in e["experiments"]):raise RuntimeError("experiment already registered")
 e["experiments"].append({"experiment_id":"P3_STG_GRAPH_TRANSFER","display_name":"Survivor-gated STG feature and temporal graph transfer","phase":"P3","execution_status":"PLANNED",
  "question":"Can fold-stable sparse support across surviving feature blocks, or a donor-only within-answer temporal graph, improve first-error localization beyond its exact compact parent?",
  "prerequisite":"Phase-2C compact survivor roster and reducer frozen; branch cannot rescue excluded signals post hoc.","population_ids":["current_common_eight_qwen"],"task_ids":["processbench_first_error"],
  "variant_order":list(IDS[1:]),"registered_comparators":[IDS[1],"matched equal-family plus ordinary IU","strongest compact Phase-2 parent"],"primary_metrics":["paired_delta_macro_f1"],
  "bootstrap":"20,000 whole-question grouped paired draws; multiplicity family frozen before either candidate opens",
  "promotion_gates":["exact zero-strength parent alias","donor/calibration-only graph fit with grouped-fold stable support","cardinality-matched random graph/support control","time and feature permutation controls",
   "CI-supported improvement over exact parent and strongest compact reference","no material worst-cell, exact-error, or clean-abstention regression","PRMBench transfer remains separate"],
  "early_boundary":"prefix-only, suffix-invariant edges/operators; no future tokens or whole-answer normalization","report_sections":["p3_stg_graph_lineage","p3_stg_parent_forest","p3_stg_controls"]})
 atomic_write_json(ep,e);print(json.dumps({"status":"REGISTERED_PLANNED_ONLY","variants":list(IDS)},indent=2))
if __name__=="__main__":main()
