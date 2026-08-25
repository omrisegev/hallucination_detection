#!/usr/bin/env python3
"""Group-OOF supervised logistic-regression ceiling on CIW-DEEM inputs."""
from __future__ import annotations
from collections import defaultdict
import importlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from spectral_utils.deem_b3_contract_ablation import prepare_arm
from spectral_utils.deem_b3_unsupervised_input_gate import build_gate_map
from spectral_utils.residual_graph_deem import atomic_write_json,canonical_sha256,sha256_file
from spectral_utils.residual_graph_deem_data import load_registry,load_target_free_bundle

def ciw_transform(X,gate_map):
 prediction=X@gate_map.prediction_matrix.T; innovation=(X-prediction)/gate_map.innovation_scale; gate=.5*gate_map.reliability; mask=np.zeros_like(X); mask[:,gate_map.core_indices]=gate[gate_map.core_indices]; return (1-mask)*X+mask*innovation

def oof_lr(X,y,folds,class_weight):
 score=np.full(len(X),np.nan); coefs=[]
 for fold in sorted(set(folds.tolist())):
  held=folds==fold; donor=~held
  if len(np.unique(y[donor]))<2 or len(np.unique(y[held]))<2: raise ValueError("class missing from grouped fold")
  scaler=StandardScaler().fit(X[donor]); model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=5000,class_weight=class_weight,random_state=0).fit(scaler.transform(X[donor]),y[donor]); score[held]=model.predict_proba(scaler.transform(X[held]))[:,1]; coefs.append(model.coef_[0])
 if not np.isfinite(score).all(): raise ValueError("incomplete OOF predictions")
 return score,float(np.mean([np.linalg.norm(value) for value in coefs]))

def stratified_group_folds(y,group_ids):
 groups=np.asarray(group_ids,dtype=str); y=np.asarray(y,dtype=int)
 for n_splits in range(5,1,-1):
  splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=20260825)
  folds=np.full(len(y),-1,dtype=int)
  for fold,(_,held) in enumerate(splitter.split(np.zeros((len(y),1)),y,groups)): folds[held]=fold
  if np.all(folds>=0) and all(len(np.unique(y[folds==fold]))==2 and len(np.unique(y[folds!=fold]))==2 for fold in range(n_splits)): return folds,n_splits
 raise ValueError("unable to construct class-valid grouped folds")

def main():
 bundle_dir=ROOT/"local_cache/deem_b3_moe_v1/bundles"; sidecar_dir=ROOT/"local_cache/deem_b3_moe_v1/label_sidecars"; out_dir=ROOT/"local_cache/deem_b3_moe_v1/ciw_supervised_lr_v1"; out_dir.mkdir(parents=True,exist_ok=True); registry=load_registry(ROOT/"configs/residual_graph_deem_24cell_v1_registry.json"); prepared_cells={}
 for row in registry["cells"]:
  cell=str(row["cell_id"]); bundle=load_target_free_bundle(bundle_dir/f"{cell}.npz"); prepared=prepare_arm(bundle.X_raw,bundle.feature_names,"D1_TRANSFORM_ONLY"); gate_map=build_gate_map(prepared,bundle.group_ids,bundle.raw_trace_length,"M1_ROOK_STATIC_R2"); prepared_cells[cell]=(bundle,prepared.X_risk,ciw_transform(prepared.X_risk,gate_map))
 pre={"schema":"ciw_supervised_lr_pre_label_v1","cells":sorted(prepared_cells),"representations":["D1_INPUT","CIW_INPUT"],"folds":5,"C":1.0,"source_sha256":sha256_file(Path(__file__)),"labels_loaded":False}; pre["content_sha256"]=canonical_sha256(pre); atomic_write_json(out_dir/"PRE_LABEL_FREEZE.json",pre)
 labels=importlib.import_module("spectral_utils.residual_graph_deem_labels"); rows=[]
 for cell,(bundle,d1,ciw) in prepared_cells.items():
  y=labels.join_labels_by_id(bundle,labels.load_label_sidecar(sidecar_dir/f"{cell}.npz")).astype(int)
  folds,n_splits=stratified_group_folds(y,bundle.group_ids)
  for representation,X in (("D1_INPUT",d1),("CIW_INPUT",ciw)):
   for weighting,class_weight in (("ordinary",None),("balanced","balanced")):
    score,coef_norm=oof_lr(X,y,folds,class_weight); rows.append({"cell_id":cell,"dataset_family":bundle.dataset_family,"representation":representation,"class_weight":weighting,"n_rows":len(y),"positive_rate":float(np.mean(y)),"n_splits":n_splits,"auroc":float(roc_auc_score(y,score)),"auprc":float(average_precision_score(y,score)),"mean_fold_coef_norm":coef_norm})
 summary={}
 for representation in ("D1_INPUT","CIW_INPUT"):
  for weighting in ("ordinary","balanced"):
   selected=[row for row in rows if row["representation"]==representation and row["class_weight"]==weighting]; by_family=defaultdict(list); by_family_ap=defaultdict(list)
   for row in selected: by_family[row["dataset_family"]].append(row["auroc"]); by_family_ap[row["dataset_family"]].append(row["auprc"])
   summary[representation+"__"+weighting]={"equal_family_auroc":float(np.mean([np.mean(value) for value in by_family.values()])),"cell_macro_auroc":float(np.mean([row["auroc"] for row in selected])),"equal_family_auprc":float(np.mean([np.mean(value) for value in by_family_ap.values()])),"cell_macro_auprc":float(np.mean([row["auprc"] for row in selected])),"family_auroc":{name:float(np.mean(value)) for name,value in by_family.items()}}
 for weighting in ("ordinary","balanced"):
  summary["CIW_MINUS_D1__"+weighting]={metric:summary["CIW_INPUT__"+weighting][metric]-summary["D1_INPUT__"+weighting][metric] for metric in ("equal_family_auroc","cell_macro_auroc","equal_family_auprc","cell_macro_auprc")}
 summary["UNSUPERVISED_CIW_REFERENCE"]={"equal_family_auroc":0.7492330051057238,"cell_macro_auroc":0.7820255514493354,"equal_family_auprc":0.7791317276773182,"cell_macro_auprc":0.7517170841581265}; atomic_write_json(out_dir/"SUMMARY.json",summary); atomic_write_json(out_dir/"PER_CELL.json",rows); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
