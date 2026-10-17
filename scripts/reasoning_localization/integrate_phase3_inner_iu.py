#!/usr/bin/env python3
"""Integrate the matched inner-family IU result."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json,sha256_file  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa:E402
from scripts.reasoning_localization.register_phase3_inner_iu import EXP,PARENT,CANDIDATE  # noqa:E402
ROOT=p1.PROGRAM_ROOT/"phase_3/hier_inner_iu"/CANDIDATE.lower()/"evaluation";H0RAW="P3_H0_REFERENCE";H0="P2C_F6_TOP10_REFERENCE"
def read(p):
 with p.open(newline="") as h:return list(csv.DictReader(h))
def write(p,rows,fields=None):
 fields=fields or list(rows[0])
 with p.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows([{f:r.get(f,"") for f in fields} for r in rows])
def append(p,adds,unique):
 old=read(p);fields=list(old[0]);keys={tuple(r.get(f,"") for f in unique) for r in old}
 for r in adds:
  k=tuple(str(r.get(f,"")) for f in unique)
  if k in keys:raise RuntimeError(f"duplicate {k}")
  keys.add(k)
 write(p,[*old,*adds],fields)
def alias(x):return H0 if x==H0RAW else x
def main():
 panels=read(ROOT/"PANELS.csv");src=ROOT/"REPORTING_PANELS.csv";write(src,panels);orders={H0RAW:169,PARENT:170,CANDIDATE:172};m=[]
 for r in panels:
  metric="macro_f1" if r["metric_id"]=="official_macro_f1" else r["metric_id"]
  m.append({"phase_id":"P3","experiment_id":EXP,"variant_id":alias(r["arm_id"]),"task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen","cell_id":"aggregate","slice_id":"all","metric_id":metric,"value":r["value"],"ci_low":r["ci_low"],"ci_high":r["ci_high"],"n_rows":r["n_rows"],"n_groups":r["n_groups"],"comparison_group_id":f"p3_inner_iu::{metric}","status":"COMPLETE","evidence_status":"DEVELOPMENT","display_order":orders[r["arm_id"]],"source_artifact":str(src.relative_to(REPO)),"source_sha256":sha256_file(src),"source_row_selector":f"arm_id={r['arm_id']};metric_id={r['metric_id']}","source_value_field":"value","notes":"H0 abstention copied; candidate reranks non-abstentions."})
 append(p1.PROGRAM_ROOT/"METRICS_LONG.csv",m,("experiment_id","variant_id","metric_id","cell_id"))
 raw=read(ROOT/"CONTRASTS.csv");cs=ROOT/"REPORTING_CONTRASTS.csv";write(cs,raw);c=[]
 for r in raw:c.append({"phase_id":"P3","experiment_id":EXP,"left_variant_id":alias(r["left_variant_id"]),"right_variant_id":alias(r["right_variant_id"]),"task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen","metric_id":r["metric_id"],"delta":r["delta"],"ci_low":r["ci_low"],"ci_high":r["ci_high"],"wins":r["wins"],"ties":r["ties"],"losses":r["losses"],"worst_unit_delta":r["worst_unit_delta"],"comparison_group_id":f"p3_inner_iu::{r['metric_id']}","status":"COMPLETE","evidence_status":"DEVELOPMENT","source_artifact":str(cs.relative_to(REPO)),"source_sha256":sha256_file(cs),"source_row_selector":f"left={r['left_variant_id']};right={r['right_variant_id']};metric={r['metric_id']}","notes":"Detector-preserving inner-family IU."})
 append(p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv",c,("experiment_id","left_variant_id","right_variant_id","metric_id"))
 s=json.loads((ROOT/"SUMMARY.json").read_text());p=s["primary_contrast"];gsrc=ROOT/"GATES.csv";g=[{"gate_id":f"{CANDIDATE}_PARENT_ALIAS","metric_id":"max_abs_error","observed":0,"threshold":1e-12,"direction":"le","passed":"true","status":"PASS","evidence_status":"DEVELOPMENT"},{"gate_id":f"{CANDIDATE}_ABSTENTION_ALIAS","metric_id":"mismatches","observed":0,"threshold":0,"direction":"eq","passed":"true","status":"PASS","evidence_status":"DEVELOPMENT"},{"gate_id":f"{CANDIDATE}_BENEFIT","metric_id":"macro_f1","observed":p["ci_low"],"threshold":.003,"direction":"gt","passed":"false","status":"FAIL","evidence_status":"DEVELOPMENT"}];write(gsrc,g);gm=[]
 for r in g:gm.append({"phase_id":"P3","experiment_id":EXP,"variant_id":CANDIDATE,**r,"unit":"fraction","source_artifact":str(gsrc.relative_to(REPO)),"source_sha256":sha256_file(gsrc),"source_row_selector":f"gate_id={r['gate_id']}","source_value_field":"observed","notes":"CI crossing zero is inconclusive, not rejection."})
 append(p1.PROGRAM_ROOT/"GATES_LONG.csv",gm,("experiment_id","variant_id","gate_id"))
 vp=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json";v=json.loads(vp.read_text());row=next(x for x in v["variants"] if x["variant_id"]==CANDIDATE);row.update({"execution_status":"COMPLETE","decision_status":"NO_PROMOTION","statistical_status":"INCONCLUSIVE","limitations":"Delta -0.00392 versus H2 equal, CI [-0.01261,+0.00454]; 3/0/5 and worst cell -0.01568. Not rejected because the interval crosses zero."});atomic_write_json(vp,v)
 ep=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json";e=json.loads(ep.read_text());x=next(x for x in e["experiments"] if x["experiment_id"]==EXP);x.update({"execution_status":"COMPLETE","next_variant":None,"verdict":"INCONCLUSIVE__NO_PROMOTION__NOT_REJECTED"});atomic_write_json(ep,e)
 pp=p1.PROGRAM_ROOT/"PLOT_MANIFEST.json";plots=json.loads(pp.read_text());plots["plots"].append({"plot_id":"PLOT_P3_INNER_IU_FOREST","title":"Inner-family IU versus equal within-family compression","phase":"P3","kind":"contrast_forest","source_table":"CONTRASTS_LONG.csv","selection":{"experiment_id":EXP,"metric_id":"macro_f1","status":"COMPLETE"},"x_field":"delta","y_field":"left_variant_id","series_field":"right_variant_id","comparison_group":"same Qwen-eight rows, H0 abstention and equal outer mean","bootstrap_definition":"20,000 whole-question paired draws; macro-F1 interval uses reserved Phase-3 family size 3.","selection_rule":"Candidate versus exact H2 parent and H0.","legend":["Interval crossing zero = inconclusive, not rejection","Outer mean and detector are unchanged"],"caption":"Uniform inner-family IU is -0.00392 versus H2 equal, CI [-0.01261,+0.00454]. It neither improves nor shows supported material harm."});atomic_write_json(pp,plots)
 build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO);REPORTING.write_build(p1.PROGRAM_ROOT,build);print(json.dumps({"verdict":"INCONCLUSIVE","delta":p["delta"],"ci":[p["ci_low"],p["ci_high"]],"report_sha256":build.manifest["output"]["sha256"]},indent=2))
if __name__=="__main__":main()
