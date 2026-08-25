#!/usr/bin/env python3
"""Evaluate the five-seed conservative rook input blend finalist."""
from __future__ import annotations
from collections import defaultdict
import importlib,itertools,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from spectral_utils.residual_graph_deem import atomic_write_json,canonical_sha256,sha256_file
from spectral_utils.residual_graph_deem_data import load_registry,load_target_free_bundle
ARM="L1_ROOK_BLEND_A25"; SEEDS=(0,1,2,3,4)
def main():
 run_dir=ROOT/"local_cache/deem_b3_moe_v1/crossed_innovation_blend_full"; base_dir=ROOT/"local_cache/deem_b3_moe_v1/input_adapter_adam_full"; frozen_dir=ROOT/"local_cache/deem_b3_moe_v1/b3_frozen"; bundle_dir=ROOT/"local_cache/deem_b3_moe_v1/bundles"; sidecar_dir=ROOT/"local_cache/deem_b3_moe_v1/label_sidecars"; out_dir=ROOT/"local_cache/deem_b3_moe_v1/crossed_innovation_blend_full_eval"; out_dir.mkdir(parents=True,exist_ok=True)
 registry=load_registry(ROOT/"configs/residual_graph_deem_24cell_v1_registry.json"); definition=json.loads((run_dir/"RUN_DEFINITION.json").read_text()); freeze=json.loads((run_dir/"SCORE_FREEZE.json").read_text())
 for value in (definition,freeze): copy=dict(value); expected=copy.pop("content_sha256"); assert canonical_sha256(copy)==expected
 assert definition["source_hashes"]["runner"]==sha256_file(ROOT/"scripts/run_deem_b3_crossed_innovation_blend_full.py"); assert len(freeze["records"])==120 and not freeze["labels_accessed_during_fit"]
 records={(row["cell_id"],row["seed"]):row for row in freeze["records"]}; bundles={}; scores={}; health=[]
 for row in registry["cells"]:
  cell=str(row["cell_id"]); bundle=load_target_free_bundle(bundle_dir/f"{cell}.npz"); bundles[cell]=bundle; scores[cell]={}; frozen=[]; base=[]; candidate=[]
  for seed in SEEDS:
   with np.load(frozen_dir/"fits"/cell/f"B3__seed{seed}.npz",allow_pickle=False) as data: frozen.append(np.asarray(data["score"]))
   with np.load(base_dir/"fits"/cell/f"D1_BASE__seed{seed}.npz",allow_pickle=False) as data: assert tuple(str(x) for x in data["row_id"].tolist())==bundle.row_ids; base.append(np.asarray(data["score"]))
   record=records[(cell,seed)]; path=run_dir/record["npz"]; assert sha256_file(path)==record["npz_sha256"]; meta=json.loads((run_dir/record["json"]).read_text()); copy=dict(meta); expected=copy.pop("content_sha256"); assert canonical_sha256(copy)==expected
   with np.load(path,allow_pickle=False) as data: assert tuple(str(x) for x in data["row_id"].tolist())==bundle.row_ids; candidate.append(np.asarray(data["score"]))
   health.append({"cell_id":cell,"seed":seed,**meta["health"]})
  scores[cell]["B3_FROZEN"]=np.mean(np.stack(frozen),axis=0); scores[cell]["D1_BASE"]=np.mean(np.stack(base),axis=0); scores[cell][ARM]=np.mean(np.stack(candidate),axis=0)
 pre={"schema":"crossed_innovation_blend_full_pre_label","all_120_fits_verified":True,"labels_imported":False}; pre["content_sha256"]=canonical_sha256(pre); atomic_write_json(out_dir/"PRE_LABEL_FREEZE.json",pre); labels=importlib.import_module("spectral_utils.residual_graph_deem_labels")
 metrics=[]
 for cell,bundle in bundles.items():
  y=labels.join_labels_by_id(bundle,labels.load_label_sidecar(sidecar_dir/f"{cell}.npz"))
  for arm in ("B3_FROZEN","D1_BASE",ARM): metrics.append({"cell_id":cell,"dataset_family":bundle.dataset_family,"arm":arm,"auroc":float(roc_auc_score(y,scores[cell][arm])),"auprc":float(average_precision_score(y,scores[cell][arm]))})
 lookup={(row["cell_id"],row["arm"]):row for row in metrics}; summary={}
 for candidate,reference in ((ARM,"D1_BASE"),(ARM,"B3_FROZEN")):
  family=defaultdict(list); cell_values=[]; ap_values=[]
  for cell,bundle in bundles.items():
   delta=lookup[(cell,candidate)]["auroc"]-lookup[(cell,reference)]["auroc"]; family[bundle.dataset_family].append(delta); cell_values.append(delta); ap_values.append(lookup[(cell,candidate)]["auprc"]-lookup[(cell,reference)]["auprc"])
  family_delta={name:float(np.mean(values)) for name,values in family.items()}; arr=np.asarray(list(family_delta.values())); observed=float(arr.mean()); null=[float(np.mean(arr*np.asarray(signs))) for signs in itertools.product((-1.,1.),repeat=len(arr))]
  summary[candidate+"_vs_"+reference]={"equal_family_auroc_delta":observed,"cell_macro_auroc_delta":float(np.mean(cell_values)),"cell_macro_auprc_delta":float(np.mean(ap_values)),"family_delta":family_delta,"exact_signflip_one_sided_p":float(np.mean(np.asarray(null)>=observed-1e-15)),"wins_ties_losses":[int(sum(x>.0005 for x in cell_values)),int(sum(abs(x)<=.0005 for x in cell_values)),int(sum(x<-.0005 for x in cell_values))],"worst_cell":float(min(cell_values))}
 atomic_write_json(out_dir/"SUMMARY.json",summary); atomic_write_json(out_dir/"PER_CELL_METRICS.json",metrics); atomic_write_json(out_dir/"HEALTH.json",health); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
