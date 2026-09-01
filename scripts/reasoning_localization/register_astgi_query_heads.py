#!/usr/bin/env python3
"""Register the ASTGI-inspired task-query ladder without running it."""
from __future__ import annotations
import json,sys
from copy import deepcopy
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
EXP="P3_ASTGI_QUERY_HEADS"
ORDER=("P3T_Q1_POINT_QUERY","P3T_Q2_LEARNED_COORD","P3T_Q3_CAUSAL_NEIGHBOR","P3T_Q4_ONE_LAYER")
PARENTS={"D0":"P2C_F6_TOP10_REFERENCE","O0":"P3A_H2_EQUAL_OUTER_REFERENCE","S0":"P2F_H3_EQUAL_C8_RERANK_PRM"}

def main():
 mp=p1.PROGRAM_ROOT/"METHOD_REGISTRY.json";methods=json.loads(mp.read_text())
 if any(x["method_id"]=="astgi_query_heads" for x in methods["methods"]):raise RuntimeError("ASTGI method already registered")
 methods["methods"].append({"method_id":"astgi_query_heads","display_name":"ASTGI-inspired task-query trajectory heads",
  "problem":"Preserve one compact token/family representation while separating response detection, first-error onset, and persistent state-error ranking.",
  "plain_summary":"Adds a bounded query-conditioned point-pooling ladder before any learned neighborhood or one-layer propagation; each task retains its own frozen parent and evaluator.",
  "input_operation_output":"compact H2 family observations plus named C8 state stream -> task query pooling -> optional donor-learned coordinates -> optional causal neighborhood -> optional one propagation layer -> separate onset/state outputs",
  "novelty":"Adapts query-specific aggregation to the observed ProcessBench/PRMBench task conflict without reopening the failed conductance graph or forcing one scalar score across roles.",
  "assumptions":["Onset and persistent state error require different aggregation queries over a shared compact representation.","A task-blind donor objective can define stable time/family coordinates.","Any useful neighborhood survives random, temporal, and permutation controls."],
  "limitations":["Design proposal only; not paper-exact ASTGI reproduction.","Exact ASTGI paper citation and component-fidelity map are not yet registered.","Current ProcessBench and PRMBench populations are development-open and cannot provide fresh confirmation."],
  "references":["docs/experiments/REASONING_LOCALIZATION_03662_ASTGI_QUERY_HEADS_V1.md","results/reasoning_localization_03662_v1/phase_2/transfer/h3_prmbench_v2/evaluation/SUMMARY.json","results/reasoning_localization_03662_v1/phase_4/h3_historical_headtohead_v1/evaluation/SUMMARY.json"]})
 atomic_write_json(mp,methods)

 vp=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json";variants=json.loads(vp.read_text());existing={x["variant_id"] for x in variants["variants"]}
 if existing.intersection(ORDER):raise RuntimeError("ASTGI variants already registered")
 template=next(x for x in variants["variants"] if x["variant_id"]=="P3T_T0_FROZEN_PARENT")
 definitions={
  "P3T_Q1_POINT_QUERY":("Point-query pooling without graph",[PARENTS["O0"],PARENTS["S0"]],"masked query-conditioned pooling over fixed compact observations; no graph","Tests task-specific onset/state aggregation before learned coordinates or edges.","Query pooling collapses to mean pooling or encodes step length rather than error semantics."),
  "P3T_Q2_LEARNED_COORD":("Donor-learned family-time coordinates",["P3T_Q1_POINT_QUERY"],"one frozen task-blind coordinate learner before the unchanged Q1 queries","Tests whether donor-only family-by-time coordinates improve query pooling.","Self-supervised reconstruction structure is unrelated to localization."),
  "P3T_Q3_CAUSAL_NEIGHBOR":("Adaptive causal neighborhood",["P3T_Q2_LEARNED_COORD"],"one frozen causal KNN relation-aware aggregation; no propagation depth","Tests learned causal relations against chain, time-only, random, and permutation controls.","Topology/position bias reproduces the STEP-CUT failure."),
  "P3T_Q4_ONE_LAYER":("Single-layer query-conditioned propagation",["P3T_Q3_CAUSAL_NEIGHBOR"],"one residual relation-aware message-passing layer; no depth sweep","Tests whether one propagation step adds value beyond neighborhood selection.","Message passing oversmooths the onset or amplifies scorer/length geometry."),
 }
 for offset,vid in enumerate(ORDER,173):
  name,parents,fusion,novelty,failure=definitions[vid];row=deepcopy(template);row.update({"variant_id":vid,"display_name":name,"display_order":offset,
   "phase":"P3","method_id":"astgi_query_heads","parent_variant_ids":parents,"role":"task_query_trajectory_template","execution_status":"PLANNED","decision_status":"PENDING","evidence_status":"DEVELOPMENT","statistical_status":"NOT_EVALUATED","rankable":False,
   "signals":["H2 compact family observations","C7 onset member","C8 state expert"],"transforms":[fusion],"fusion":fusion,"novelty":novelty,"failure_hypothesis":failure,
   "detector":"D0 H0 abstention frozen exactly","step_reducer":"O0 and S0 task-specific reducers remain frozen at Q1; later changes require a separate factor contrast",
   "supervision":"donor/calibration-only representation and query fit; task labels evaluation-only","task_ids":["processbench_first_error","prmbench_step_error"],
   "causal_validity":"completed-trace design; Q3 edges are past-only, but Phase-5 transfer still requires full prefix/suffix-invariance audit",
   "limitations":"Design template only; exact executable function, objective, dimensions, seed, K/control roster, and noninferiority margins must be frozen before scores."})
  variants["variants"].append(row)
 t3=next(x for x in variants["variants"] if x["variant_id"]=="P3T_T3_TWO_AXIS_LOWRANK")
 if "P3T_Q4_ONE_LAYER" not in t3["parent_variant_ids"]:t3["parent_variant_ids"].append("P3T_Q4_ONE_LAYER")
 t3["limitations"]="May run only from a surviving original temporal parent or Q-ladder parent; task labels cannot select the route, rank, transform, or weights."
 atomic_write_json(vp,variants)

 ep=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json";experiments=json.loads(ep.read_text())
 if any(x["experiment_id"]==EXP for x in experiments["experiments"]):raise RuntimeError("ASTGI experiment already registered")
 tensor=next(x for x in experiments["experiments"] if x["experiment_id"]=="P3_TRAJECTORY_TENSOR")
 tensor["trajectory_tensor_contract"]["ordered_ladder"]=["P3T_T0_FROZEN_PARENT","P3T_Q1_POINT_QUERY","P3T_Q2_LEARNED_COORD","P3T_Q3_CAUSAL_NEIGHBOR","P3T_Q4_ONE_LAYER","P3T_T3_TWO_AXIS_LOWRANK"]
 tensor["trajectory_tensor_contract"]["astgi_amendment"]="Q1-Q4 are a task-query subladder; original T1/T2 remain separate alternatives and cannot be crossed with Q variants."
 experiments["experiments"].append({"experiment_id":EXP,"display_name":"ASTGI-inspired separated task-query ladder","phase":"P3","execution_status":"PLANNED",
  "question":"Can query-conditioned aggregation preserve a shared compact representation while separately improving ProcessBench onset and PRMBench state-error ranking without changing H0 detection?",
  "prerequisite":"H0/H2/H3 task-role parents frozen; exact query function or donor-only objective and controls frozen in a new execution registry",
  "population_ids":["current_common_eight_qwen","prmbench_error_responses"],"task_ids":["processbench_first_error","prmbench_step_error"],"variant_order":list(ORDER),
  "registered_comparators":[PARENTS["D0"],PARENTS["O0"],PARENTS["S0"]],"primary_metrics":["paired_delta_macro_f1","paired_delta_auroc","paired_delta_auprc"],
  "bootstrap":"20,000 whole-question grouped ProcessBench draws and separate grouped PRMBench response draws; no cross-task aggregate",
  "parent_map":PARENTS,"promotion_gates":["D0 abstention mismatch equals zero","onset head passes preregistered noninferiority-or-improvement gate versus O0","state head separately passes preregistered noninferiority-or-improvement gate versus S0","no task metric selects query parameters, coordinates, K, neighborhood, checkpoint, or route","Q1 beats mean/query-permutation/boundary-removal controls before Q2","Q2 exact zero-coordinate strength aliases Q1","Q3 beats Q2 and chain/time-only/random/time-permutation/feature-permutation controls","Q4 has one layer only and exact zero-strength Q3 alias","ProcessBench and PRMBench are never averaged","current opened populations can yield development evidence only; fresh confirmation remains required"],
  "branch_contract":{"paper_fidelity":"ASTGI-inspired adaptation; exact citation/component map pending","compact_roster":"H2 families plus named external C8 state stream only","hierarchy_boundary":"no simultaneous hierarchical-family fusion change in Q1-Q4","graph_boundary":"does not reopen failed STEP-CUT conductance graph","early_boundary":"onset only; prefix-safe relative time, past-only edges, no total-answer fraction, suffix invariance required"},
  "next_variant":"P3T_Q1_POINT_QUERY","report_sections":["p3_tensor_pipeline","p3_parent_fusion"]})
 atomic_write_json(ep,experiments)
 print(json.dumps({"status":"REGISTERED_DESIGN_ONLY","experiment":EXP,"variants":list(ORDER)},indent=2))
if __name__=="__main__":main()
