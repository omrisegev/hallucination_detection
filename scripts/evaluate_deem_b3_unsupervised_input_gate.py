#!/usr/bin/env python3
"""Evaluate the frozen seed-0 unsupervised input-gate screen."""
from __future__ import annotations
from collections import defaultdict
import importlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from spectral_utils.deem_b3_unsupervised_input_gate import ARMS
from spectral_utils.residual_graph_deem import atomic_write_json,canonical_sha256,sha256_file
from spectral_utils.residual_graph_deem_data import load_registry,load_target_free_bundle
FIXED="L0_FIXED_ROOK_A25"
def main():
 run_dir=ROOT/"local_cache/deem_b3_moe_v1/unsupervised_input_gate_seed0"; base_dir=ROOT/"local_cache/deem_b3_moe_v1/contract_ablation_seed0"; fixed_dir=ROOT/"local_cache/deem_b3_moe_v1/crossed_innovation_blend_seed0"; bundle_dir=ROOT/"local_cache/deem_b3_moe_v1/bundles"; sidecar_dir=ROOT/"local_cache/deem_b3_moe_v1/label_sidecars"; out_dir=ROOT/"local_cache/deem_b3_moe_v1/unsupervised_input_gate_seed0_eval"; out_dir.mkdir(parents=True,exist_ok=True)
 registry=load_registry(ROOT/"configs/residual_graph_deem_24cell_v1_registry.json"); definition=json.loads((run_dir/"RUN_DEFINITION.json").read_text()); freeze=json.loads((run_dir/"SCORE_FREEZE.json").read_text())
 for value in (definition,freeze): copy=dict(value); expected=copy.pop("content_sha256"); assert canonical_sha256(copy)==expected
 assert definition["source_hashes"]["runner"]==sha256_file(ROOT/"scripts/run_deem_b3_unsupervised_input_gate.py"); assert definition["source_hashes"]["core"]==sha256_file(ROOT/"spectral_utils/deem_b3_unsupervised_input_gate.py"); assert len(freeze["records"])==72 and not freeze["labels_accessed_during_fit"]
 records={(row["cell_id"],row["arm"]):row for row in freeze["records"]}; bundles={}; scores={}; diagnostics=[]
 for row in registry["cells"]:
  cell=str(row["cell_id"]); bundle=load_target_free_bundle(bundle_dir/f"{cell}.npz"); bundles[cell]=bundle; scores[cell]={}
  with np.load(base_dir/"fits"/cell/"D1_TRANSFORM_ONLY__seed0.npz",allow_pickle=False) as data: scores[cell]["D1_BASE"]=np.asarray(data["score"])
  with np.load(fixed_dir/"fits"/cell/"L1_ROOK_BLEND_A25__seed0.npz",allow_pickle=False) as data: assert tuple(str(x) for x in data["row_id"].tolist())==bundle.row_ids; scores[cell][FIXED]=np.asarray(data["score"])
  for arm in ARMS:
   record=records[(cell,arm)]; path=run_dir/record["npz"]; assert sha256_file(path)==record["npz_sha256"]; meta=json.loads((run_dir/record["json"]).read_text()); copy=dict(meta); expected=copy.pop("content_sha256"); assert canonical_sha256(copy)==expected
   with np.load(path,allow_pickle=False) as data: assert tuple(str(x) for x in data["row_id"].tolist())==bundle.row_ids; scores[cell][arm]=np.asarray(data["score"])
   diagnostics.append({"cell_id":cell,"arm":arm,**meta["health"]})
 pre={"schema":"unsupervised_input_gate_pre_label","all_72_fits_verified":True,"labels_imported":False}; pre["content_sha256"]=canonical_sha256(pre); atomic_write_json(out_dir/"PRE_LABEL_FREEZE.json",pre); labels=importlib.import_module("spectral_utils.residual_graph_deem_labels")
 metrics=[]
 for cell,bundle in bundles.items():
  y=labels.join_labels_by_id(bundle,labels.load_label_sidecar(sidecar_dir/f"{cell}.npz"))
  for arm in ("D1_BASE",FIXED)+ARMS: metrics.append({"cell_id":cell,"dataset_family":bundle.dataset_family,"arm":arm,"auroc":float(roc_auc_score(y,scores[cell][arm])),"auprc":float(average_precision_score(y,scores[cell][arm]))})
 lookup={(row["cell_id"],row["arm"]):row for row in metrics}; summary={}
 for arm in ARMS:
  family=defaultdict(list); values=[]; ap=[]; versus_fixed=[]
  for cell,bundle in bundles.items():
   delta=lookup[(cell,arm)]["auroc"]-lookup[(cell,"D1_BASE")]["auroc"]; family[bundle.dataset_family].append(delta); values.append(delta); ap.append(lookup[(cell,arm)]["auprc"]-lookup[(cell,"D1_BASE")]["auprc"]); versus_fixed.append(lookup[(cell,arm)]["auroc"]-lookup[(cell,FIXED)]["auroc"])
  family_delta={name:float(np.mean(group)) for name,group in family.items()}; health=[row for row in diagnostics if row["arm"]==arm]; summary[arm]={"equal_family_delta_vs_D1":float(np.mean(list(family_delta.values()))),"cell_macro_delta_vs_D1":float(np.mean(values)),"cell_macro_auprc_delta_vs_D1":float(np.mean(ap)),"cell_macro_delta_vs_fixed_A25":float(np.mean(versus_fixed)),"family_delta":family_delta,"wins_ties_losses":[int(sum(x>.0005 for x in values)),int(sum(abs(x)<=.0005 for x in values)),int(sum(x<-.0005 for x in values))],"worst_cell":float(min(values)),"mean_gate":float(np.mean([row["mean_gate"] for row in health])),"gate_sd":float(np.mean([row["gate_sd"] for row in health])),"mean_oof_r2":float(np.mean([row["mean_oof_r2"] for row in health]))}
 atomic_write_json(out_dir/"SUMMARY.json",summary); atomic_write_json(out_dir/"PER_CELL_METRICS.json",metrics); atomic_write_json(out_dir/"DIAGNOSTICS.json",diagnostics); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
