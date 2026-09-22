#!/usr/bin/env python3
"""Integrate a completed post-parent Phase-2C conditional candidate."""

from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization import run_phase2_conditional as runner  # noqa:E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa:E402

EXP="P2_CONDITIONAL_ABLATION"
NEXT={
 "P2C_F6_MINUS_ENTROPY_DYNAMICS":"P2C_F6_MINUS_SAMPLED_ENERGY",
 "P2C_F6_MINUS_SAMPLED_ENERGY":"P2C_F6_MINUS_PARTITION_ENERGY",
 "P2C_F6_MINUS_PARTITION_ENERGY":"P2C_F6_MINUS_TOPK_DISTRIBUTION",
 "P2C_F6_MINUS_TOPK_DISTRIBUTION":"P2C_F6_PLUS_STRUCTURAL_CONTROL",
 "P2C_F6_PLUS_STRUCTURAL_CONTROL":"P2C_F6_SWAP_C1_SWVAR16",
 "P2C_F6_MINUS_ENTROPY_SWVAR16_VIEW":"P2C_F6_MINUS_ENTROPY_CUSUM_VIEW",
 "P2C_F6_MINUS_ENTROPY_CUSUM_VIEW":"P2C_F6_MINUS_SAMPLED_LEVEL_VIEW",
 "P2C_F6_MINUS_SAMPLED_LEVEL_VIEW":"P2C_F6_MINUS_PARTITION_LEVEL_VIEW",
 "P2C_F6_MINUS_PARTITION_LEVEL_VIEW":"P2C_F6_SWAP_C1_SWVAR16",
 "P2C_F6_SWAP_C1_SWVAR16":"P2C_F6_PLUS_C7_EDIS_VIEW",
 "P2C_F6_PLUS_C7_EDIS_VIEW":"P2C_F6_PLUS_C8_OUTER_EXPERT",
 "P2C_F6_PLUS_C8_OUTER_EXPERT":None,
}
LOO=set(runner.FAMILY_REMOVE)|set(runner.VIEW_REMOVE)

def read(path):
 with path.open(newline="") as h:return list(csv.DictReader(h))
def write(path,rows,fields=None):
 if fields is None:fields=list(rows[0])
 with path.open("w",newline="") as h:
  w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows([{f:r.get(f,"") for f in fields} for r in rows])
def append(path,rows,key):
 old=read(path);fields=list(old[0])
 if any(r.get(key)==rows[0].get(key) for r in old):raise RuntimeError(f"already integrated: {rows[0].get(key)}")
 write(path,[*old,*rows],fields)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--variant",choices=tuple(NEXT),required=True);a=ap.parse_args();v=a.variant
 er=runner.output_root(v)/"evaluation";summary=json.loads((er/"SUMMARY.json").read_text());is_loo=v in LOO
 panels=[{**r,"status":"COMPLETE","evidence_status":"DEVELOPMENT"} for r in read(er/"PROCESSBENCH_PANELS.csv") if r["arm_id"]==v]
 panels=list({r["metric_id"]:r for r in panels}.values());msource=er/"REPORTING_METRICS.csv";write(msource,panels)
 order=next(r["display_order"] for r in json.loads((p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json").read_text())["variants"] if r["variant_id"]==v)
 mrows=[]
 for r in panels:
  metric="macro_f1" if r["metric_id"]=="official_macro_f1" else r["metric_id"]
  mrows.append({"phase_id":"P2C","experiment_id":EXP,"variant_id":v,"task_id":"processbench_first_error","dataset_id":"processbench","population_id":"current_common_eight_qwen",
   "cell_id":"aggregate","slice_id":"all","metric_id":metric,"value":r["value"],"ci_low":r["ci_low"],"ci_high":r["ci_high"],"n_rows":r["n_rows"],"n_groups":r["n_groups"],
   "comparison_group_id":f"p2c_qwen8_five_family_top10::{metric}","status":"COMPLETE","evidence_status":"DEVELOPMENT","display_order":order,
   "source_artifact":str(msource.relative_to(REPO)),"source_sha256":sha256_file(msource),"source_row_selector":f"arm_id={v};metric_id={r['metric_id']}","source_value_field":"value",
   "notes":"Leave-one-out contribution uses reversed signs." if is_loo else "Candidate-minus-parent insertion or swap."})
 append(p1.PROGRAM_ROOT/"METRICS_LONG.csv",mrows,"variant_id")
 cs=[{**r,"status":"COMPLETE","evidence_status":"DEVELOPMENT"} for r in read(er/"PAIRWISE_CONTRASTS.csv")];csource=er/"REPORTING_CONTRASTS.csv";write(csource,cs)
 crows=[]
 for r in cs:
  metric=r["metric_id"]
  crows.append({"phase_id":"P2C","experiment_id":EXP,"left_variant_id":v,"right_variant_id":runner.PARENT,"task_id":"processbench_first_error","dataset_id":"processbench",
   "population_id":"current_common_eight_qwen","metric_id":metric,"delta":r["candidate_minus_parent_delta"],"ci_low":r["ci_low"],"ci_high":r["ci_high"],"wins":r["wins"],"ties":r["ties"],"losses":r["losses"],
   "worst_unit_delta":r["worst_unit_delta"],"comparison_group_id":f"p2c_qwen8_five_family_top10::{metric}","status":"COMPLETE","evidence_status":"DEVELOPMENT",
   "source_artifact":str(csource.relative_to(REPO)),"source_sha256":sha256_file(csource),"source_row_selector":f"metric_id={metric}",
   "notes":"Candidate-minus-parent; reverse signs for leave-one-out contribution." if is_loo else "Candidate-minus-parent."})
 append(p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv",crows,"left_variant_id")
 by=read(er/"PROCESSBENCH_BY_CELL.csv");parent={r["cell_id"]:r for r in by if r["arm_id"]==runner.PARENT};cand={r["cell_id"]:r for r in by if r["arm_id"]==v}
 cell_delta={c:float(cand[c]["official_macro_f1"])-float(parent[c]["official_macro_f1"]) for c in parent}
 pc=summary["primary_contrast"]
 if is_loo:
  lower=-float(pc["ci_high"]);point=-float(pc["candidate_minus_parent_delta"]);support=sum(x<=1e-12 for x in cell_delta.values());worst=min(-x for x in cell_delta.values());exact=-float(summary["exact_error_delta"]);clean=-float(summary["clean_abstention_delta"])
 else:
  lower=float(pc["ci_low"]);point=float(pc["candidate_minus_parent_delta"]);support=sum(x>=-1e-12 for x in cell_delta.values());worst=min(cell_delta.values());exact=float(summary["exact_error_delta"]);clean=float(summary["clean_abstention_delta"])
 checks=[("SIMULTANEOUS_CONTRIBUTION","macro_f1",lower,.003,"gt",lower>.003),("POINT_BENEFIT","macro_f1",point,.003,"ge",point>=.003),("CELL_SUPPORT","nonnegative_cells",support,6,"ge",support>=6),
         ("WORST_CELL","macro_f1",worst,-.020,"ge",worst>=-.020),("EXACT","first_error_exact",exact,-.010,"ge",exact>=-.010),("CLEAN","clean_abstention_accuracy",clean,-.010,"ge",clean>=-.010)]
 passed=all(x[-1] for x in checks);grows=[]
 for name,metric,obs,thr,direction,ok in checks:
  grows.append({"gate_id":f"{v}_{name}","metric_id":metric,"observed":obs,"threshold":thr,"direction":direction,"passed":str(ok).lower(),"status":"PASS" if ok else "FAIL","evidence_status":"DEVELOPMENT"})
 grows.append({"gate_id":f"{v}_OVERALL","metric_id":"all_registered_gates","observed":str(passed).lower(),"threshold":"true","direction":"eq","passed":str(passed).lower(),"status":"PASS" if passed else "FAIL","evidence_status":"DEVELOPMENT"})
 gsource=er/"GATES.csv";write(gsource,grows);gmaster=[]
 for r in grows:gmaster.append({"phase_id":"P2C","experiment_id":EXP,"variant_id":v,**r,"unit":"boolean" if r["metric_id"]=="all_registered_gates" else "fraction","source_artifact":str(gsource.relative_to(REPO)),"source_sha256":sha256_file(gsource),"source_row_selector":f"gate_id={r['gate_id']}","source_value_field":"observed","notes":"Frozen Phase-2C conditional gate."})
 append(p1.PROGRAM_ROOT/"GATES_LONG.csv",gmaster,"variant_id")
 vp=p1.PROGRAM_ROOT/"VARIANT_REGISTRY.json";payload=json.loads(vp.read_text());row=next(r for r in payload["variants"] if r["variant_id"]==v);row["execution_status"]="COMPLETE";row["decision_status"]="PROMOTED" if passed else "NO_PROMOTION"
 row["statistical_status"]="SUPPORTED_IMPROVEMENT" if passed else ("PROMISING_UNCONFIRMED" if point>0 else "INCONCLUSIVE")
 atomic_write_json(vp,payload);ep=p1.PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json";ex=json.loads(ep.read_text());e=next(r for r in ex["experiments"] if r["experiment_id"]==EXP);e["next_variant"]=NEXT[v];e["execution_status"]="RUNNING" if NEXT[v] else "COMPLETE";atomic_write_json(ep,ex)
 build=REPORTING.prepare_build(p1.PROGRAM_ROOT,REPO);REPORTING.write_build(p1.PROGRAM_ROOT,build);print(json.dumps({"variant":v,"conditional_point":point,"conditional_ci_low":lower,"overall_gate":passed,"next":NEXT[v],"report_sha256":build.manifest["output"]["sha256"]},indent=2))
if __name__=="__main__":main()
