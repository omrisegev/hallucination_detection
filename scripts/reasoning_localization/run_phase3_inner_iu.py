#!/usr/bin/env python3
"""Freeze and evaluate IU experts inside the three multi-view H2 families."""
from __future__ import annotations
import csv,importlib,json,sys,time
from pathlib import Path
from typing import Any,Mapping
import numpy as np
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json,atomic_write_npz,load_npz_no_pickle,sha256_file  # noqa:E402
from spectral_utils.reconstruction_benchmark.localization_contract import load_prepared_localization_cell,validate_fit_manifest  # noqa:E402
from spectral_utils.token_local_fusion import IU_CONFIG,fit_local_equal_family,prepare_localization_cell  # noqa:E402
from spectral_utils.upcr import upcr_fit  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa:E402
from scripts.reasoning_localization import run_phase2_atomic_remaining as atomic  # noqa:E402
from scripts.reasoning_localization import run_phase2_conditional as cond  # noqa:E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa:E402
from scripts.reasoning_localization import run_phase3_compact_fusion as p3  # noqa:E402
from scripts.reasoning_localization.register_phase3_inner_iu import EXP,PARENT,CANDIDATE,FAMILY_SIZE  # noqa:E402
ROOT=p1.PROGRAM_ROOT/"phase_3/hier_inner_iu";OUT=ROOT/CANDIDATE.lower();REG=ROOT/"P3C_H2_INNER_IU_EQUAL_OUTER_EXECUTION_REGISTRY.json"
SOURCE_H2=p1.PROGRAM_ROOT/"phase_2/diagnostic/h3_reliability_fusion_v1/score_freeze/cells"

def fit_inner(confidence:np.ndarray,fit_idx:np.ndarray)->tuple[np.ndarray,dict[str,Any]]:
 fit=confidence[fit_idx];model=upcr_fit(fit.T,**dict(IU_CONFIG));w=np.asarray(model.w).copy();anchor=fit.mean(axis=1);corr=float(np.corrcoef(fit@w,anchor)[0,1])
 if np.isfinite(corr) and corr<0:w*=-1
 return -(confidence@w),{"weights":w.tolist(),"anchor_correlation":corr,"g2_hat":float(model.g2_hat)}

def curves(cell:Any):
 prep=prepare_localization_cell(cell);full=prep.standardized_slice(0,len(prep.values));names=list(prep.kept_stream_names);families=list(prep.kept_family_names)
 level=cond._family_risk(prep,"entropy_level");entropy=atomic.primitive_risks(cell)["entropy"];onset=atomic.response_map(entropy,cell.token_offsets,atomic.edis_onset);c7=cond._standardized_risk(onset,prep.fit_indices)
 idx=[i for i,f in enumerate(families) if f=="entropy_dynamics"];dyn,dd=fit_inner(np.column_stack([full[:,idx],-c7]),prep.fit_indices)
 idx=[i for i,(n,f) in enumerate(zip(names,families)) if f=="partition_energy" and n!="energy_series"];part,pd=fit_inner(full[:,idx],prep.fit_indices)
 idx=[i for i,f in enumerate(families) if f=="topk_distribution"];top,td=fit_inner(full[:,idx],prep.fit_indices)
 parent=np.mean(p3._h2_family_matrix(cell)[0],axis=1);candidate=np.mean([level,dyn,part,top],axis=0)
 return prep,parent,candidate,{"dynamics":dd,"partition":pd,"topk":td}

def writecsv(path,rows):
 fields=list(rows[0]);
 with path.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)

def freeze(release):
 if OUT.exists():raise FileExistsError(OUT)
 reg=json.loads(REG.read_text());
 if reg.get("runner_sha256")!=sha256_file(Path(__file__).resolve()) or reg.get("status")!="FROZEN_BEFORE_RUN":raise RuntimeError("registry mismatch")
 root=OUT/"score_freeze";root.mkdir(parents=True);inp=release/"build_A/localization/inputs";man=validate_fit_manifest(inp/"MANIFEST.json",input_root=inp);by={r["cell_id"]:r for r in man["cells"]};records=[];alias=0.
 for pos,cid in enumerate(p2r.PB_CELLS,1):
  src=by[cid];cell=load_prepared_localization_cell(inp/src["artifact_path"],src);prep,parent,cand,diag=curves(cell);h0=fit_local_equal_family(prep).token_risk
  pl=p1.topk_step_mean(parent,cell.segment_starts,cell.segment_ends,k=10);cl=p1.topk_step_mean(cand,cell.segment_starts,cell.segment_ends,k=10);hl=p1.topk_step_mean(h0,cell.segment_starts,cell.segment_ends,k=10)
  alias=max(alias,float(np.max(np.abs(pl-load_npz_no_pickle(SOURCE_H2/cid/"scores.npz")["h2_local"]))))
  arr={"row_ids":np.asarray(cell.row_ids,dtype="<U80"),"segment_offsets":np.asarray(cell.segment_offsets,dtype="<i8"),"segment_lengths":np.asarray(cell.segment_ends-cell.segment_starts,dtype="<i8"),"h0_combined":p1.combine_with_common_detector(cell,hl),"parent_local":pl,"candidate_local":cl}
  target=root/"cells"/cid;target.mkdir(parents=True);sha=atomic_write_npz(target/"scores.npz",arr);rec={"cell_id":cid,"model_id":cell.model_id,"slice_id":cell.slice_id,"score_sha256":sha,"labels_seen":False,"diagnostics":diag};atomic_write_json(target/"RECORD.json",rec);records.append({"cell_id":cid,"score_sha256":sha,"record_sha256":sha256_file(target/"RECORD.json")});print(f"score-freeze {CANDIDATE}: {cid} ({pos}/8)",flush=True)
 if alias>1e-12:raise RuntimeError(f"parent alias {alias}")
 out={"status":"COMPLETE","parent_alias_max_abs_error":alias,"records":records,"registry_sha256":sha256_file(REG)};atomic_write_json(root/"SCORE_FREEZE_MANIFEST.json",out);return out

def verified(man):
 out={}
 for r in man["records"]:
  root=OUT/"score_freeze/cells"/r["cell_id"]
  if sha256_file(root/"scores.npz")!=r["score_sha256"]:raise RuntimeError("score hash")
  out[r["cell_id"]]={"record":json.loads((root/"RECORD.json").read_text()),"arrays":load_npz_no_pickle(root/"scores.npz")}
 return out

def rows(v,labels,key):
 out={m:[] for m in p1.QWEN_MODELS}
 for cid in p2r.PB_CELLS:
  rec,a=v[cid]["record"],v[cid]["arrays"];off=a["segment_offsets"];lens=a["segment_lengths"]
  for i,rid in enumerate(a["row_ids"].astype(str)):
   lo,hi=map(int,off[i:i+2]);gid,fe=labels[cid][rid];out[rec["model_id"]].append({"row_id":rid,"group_id":gid,"slice_id":rec["slice_id"],"cell_id":cid,"model_id":rec["model_id"],"first_error":fe,"step_scores":a[key][lo:hi].tolist(),"step_lengths":lens[lo:hi].tolist()})
 return out

def evaluate(release,man):
 v=verified(man);labels=p1._load_pb_labels(release);ev=importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation");h0=c1.evaluate_arm("P3_H0_REFERENCE",rows(v,labels,"h0_combined"),ev);arms={"P3_H0_REFERENCE":h0};arms[PARENT]=p3._rerank(PARENT,h0,rows(v,labels,"parent_local"),ev);arms[CANDIDATE]=p3._rerank(CANDIDATE,h0,rows(v,labels,"candidate_local"),ev)
 contrasts=[]
 for l,r in ((CANDIDATE,PARENT),(CANDIDATE,"P3_H0_REFERENCE")):
  lp={x["metric_id"]:x for x in arms[l]["panels"]};rp={x["metric_id"]:x for x in arms[r]["panels"]};rc={x["cell_id"]:x for x in arms[r]["by_cell"]}
  for metric in p1.PB_METRICS:
   draws=arms[l]["samples"][metric]-arms[r]["samples"][metric];q=.025/FAMILY_SIZE if metric=="official_macro_f1" else .025;cells={x["cell_id"]:float(x[metric])-float(rc[x["cell_id"]][metric]) for x in arms[l]["by_cell"]}
   contrasts.append({"left_variant_id":l,"right_variant_id":r,"metric_id":"macro_f1" if metric=="official_macro_f1" else metric,"delta":float(lp[metric]["value"]-rp[metric]["value"]),"ci_low":float(np.quantile(draws,q)),"ci_high":float(np.quantile(draws,1-q)),"wins":sum(x>1e-12 for x in cells.values()),"ties":sum(abs(x)<=1e-12 for x in cells.values()),"losses":sum(x<-1e-12 for x in cells.values()),"worst_unit_delta":min(cells.values())})
 er=OUT/"evaluation";er.mkdir();writecsv(er/"PANELS.csv",[x for a in arms.values() for x in a["panels"]]);writecsv(er/"BY_CELL.csv",[x for a in arms.values() for x in a["by_cell"]]);writecsv(er/"CONTRASTS.csv",contrasts);primary=next(x for x in contrasts if x["left_variant_id"]==CANDIDATE and x["right_variant_id"]==PARENT and x["metric_id"]=="macro_f1");s={"status":"COMPLETE","primary_contrast":primary,"abstention_mismatches":0};atomic_write_json(er/"SUMMARY.json",s);print(json.dumps(s,indent=2));return s

def main():
 release=p1.DEFAULT_RELEASE.resolve();start=time.perf_counter();man=freeze(release);s=evaluate(release,man);atomic_write_json(OUT/"RUN_COMPLETE.json",{"status":"COMPLETE","elapsed":time.perf_counter()-start,"summary":s})
if __name__=="__main__":main()
