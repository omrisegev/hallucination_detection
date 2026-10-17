#!/usr/bin/env python3
"""Register matched inner-family IU with an equal outer mean."""

from __future__ import annotations
import json, sys
from copy import deepcopy
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json,sha256_file  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa:E402

EXP="P3_HIER_INNER_IU"; PARENT="P3A_H2_EQUAL_OUTER_REFERENCE"; CANDIDATE="P3C_H2_INNER_IU_EQUAL_OUTER"; FAMILY_SIZE=3

def main():
 vp=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json";v=json.loads(vp.read_text())
 if any(r["variant_id"]==CANDIDATE for r in v["variants"]):raise RuntimeError("already registered")
 parent=next(r for r in v["variants"] if r["variant_id"]==PARENT);row=deepcopy(parent)
 row.update({"variant_id":CANDIDATE,"display_name":"P3C inner-family IU with equal outer mean","display_order":172,
  "parent_variant_ids":[PARENT],"role":"hierarchical_inner_family_fusion","execution_status":"PLANNED","decision_status":"PENDING",
  "statistical_status":"NOT_EVALUATED","rankable":True,
  "fusion":"entropy singleton passthrough; ordinary IU separately inside dynamics+C7, partition-minus-level, and top-k; equal outer mean",
  "novelty":"Isolates learned within-family compression while retaining the successful equal outer family weighting.",
  "failure_hypothesis":"Within-family covariance violates IU assumptions or suppresses localized members.",
  "limitations":"One common IU flavour is fixed before results; this is not per-family outcome selection."})
 v["variants"].append(row);atomic_write_json(vp,v)
 ep=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json";e=json.loads(ep.read_text())
 e["experiments"].append({"experiment_id":EXP,"display_name":"Matched hierarchical inner-family IU","phase":"P3","execution_status":"PLANNED",
  "question":"Does ordinary IU inside each eligible H2 family beat equal within-family means when the outer mean, H0 detector, and top-ten reducer are fixed?",
  "prerequisite":"P3B outer IU supported-harm verdict; family-expert branch remains separately eligible","population_ids":["current_common_eight_qwen"],
  "task_ids":["processbench_first_error"],"variant_order":[PARENT,CANDIDATE],"registered_comparators":[PARENT,"P2C_F6_TOP10_REFERENCE"],
  "primary_metrics":["paired_delta_macro_f1"],"bootstrap":"20,000 paired whole-question draws; same reserved Phase-3 family size 3",
  "multiplicity_family_size":FAMILY_SIZE,"promotion_gates":["parent local-score alias <=1e-12","H0 abstention mismatch=0",
   "delta vs parent >=+0.003 with CI lower >+0.003","improves H0 point estimate","worst cell >=-0.020 and exact delta >=-0.010"],
  "next_variant":CANDIDATE,"report_sections":["p3_parent_fusion","p3_complexity"]});atomic_write_json(ep,e)
 protocol=REPO/"docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_COMPACT_FUSION_V1.md";runner=REPO/"scripts/reasoning_localization/run_phase3_inner_iu.py"
 reg={"schema":"reasoning-localization-p3-inner-iu-execution-v1","status":"FROZEN_BEFORE_RUN","experiment_id":EXP,
  "variant_id":CANDIDATE,"parent_variant_id":PARENT,"release_root":str(p1.DEFAULT_RELEASE.resolve()),"cells":list(p2r.PB_CELLS),
  "inner_roster":{"entropy_level":"passthrough","entropy_dynamics_plus_C7":"ordinary IU","partition_without_energy_series":"ordinary IU","topk_distribution":"ordinary IU"},
  "outer_fusion":"equal mean","detector":"copy frozen H0 abstention; rerank non-abstentions only","step_reducer":"top10",
  "multiplicity_family_size":FAMILY_SIZE,"labels_seen_during_fit":False,"protocol_sha256":sha256_file(protocol),"runner_sha256":sha256_file(runner)}
 target=p1.PROGRAM_ROOT/"phase_3/hier_inner_iu/P3C_H2_INNER_IU_EQUAL_OUTER_EXECUTION_REGISTRY.json";target.parent.mkdir(parents=True,exist_ok=True);atomic_write_json(target,reg)
 print(json.dumps({"status":"REGISTERED_BEFORE_RESULTS","registry":str(target)},indent=2))
if __name__=="__main__":main()
