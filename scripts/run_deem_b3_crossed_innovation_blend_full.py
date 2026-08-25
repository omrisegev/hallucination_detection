#!/usr/bin/env python3
"""Five-seed target-free finalist for the conservative rook input blend."""
from __future__ import annotations
import argparse,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from spectral_utils.deem_b3_contract_ablation import prepare_arm
from spectral_utils.deem_b3_crossed_innovation_blend import BLEND,fit_blended_innovation
from spectral_utils.residual_graph_deem import atomic_save_npz,atomic_write_json,canonical_sha256,sha256_file
from spectral_utils.residual_graph_deem_data import load_registry,load_target_free_bundle
SCHEMA="deem_b3_crossed_innovation_blend_full_2026_08_25"; ARM="L1_ROOK_BLEND_A25"; SEEDS=(0,1,2,3,4)
def fit_cell(payload):
 cell,bundle_dir_raw,out_dir_raw,definition_hash=payload; bundle=load_target_free_bundle(Path(bundle_dir_raw)/f"{cell}.npz"); prepared=prepare_arm(bundle.X_raw,bundle.feature_names,"D1_TRANSFORM_ONLY"); records=[]
 for seed in SEEDS:
  result,innovation=fit_blended_innovation(prepared,ARM,seed=seed)
  if not result.health["healthy"]: raise ValueError(f"unhealthy {cell} {seed}")
  fit_dir=Path(out_dir_raw)/"fits"/cell; fit_dir.mkdir(parents=True,exist_ok=True); path=fit_dir/f"{ARM}__seed{seed}.npz"
  digest=atomic_save_npz(path,schema=np.asarray(SCHEMA),cell_id=np.asarray(cell),arm=np.asarray(ARM),seed=np.asarray(seed),score=result.score,logit=result.logit,contributions=result.contributions,aligned_bias=np.asarray(result.aligned_bias),orientation=np.asarray(result.orientation),feature_names=np.asarray(prepared.feature_names,dtype=str),row_id=np.asarray(bundle.row_ids,dtype=str),source_bundle_sha256=np.asarray(bundle.bundle_sha256),run_definition_sha256=np.asarray(definition_hash),innovation_matrix=innovation.matrix,innovation_output_scale=innovation.output_scale)
  meta={"schema":SCHEMA+"_fit","cell_id":cell,"dataset_family":bundle.dataset_family,"arm":ARM,"seed":seed,"npz_sha256":digest,"health":result.health,"source_bundle_sha256":bundle.bundle_sha256,"run_definition_sha256":definition_hash,"labels_accessed_during_fit":False,"target_module_imported_during_fit":False}; meta["content_sha256"]=canonical_sha256(meta); meta_path=path.with_suffix(".json"); atomic_write_json(meta_path,meta)
  records.append({"cell_id":cell,"arm":ARM,"seed":seed,"npz":path.relative_to(out_dir_raw).as_posix(),"npz_sha256":digest,"json":meta_path.relative_to(out_dir_raw).as_posix(),"json_sha256":sha256_file(meta_path),"healthy":True})
 return records
def main():
 parser=argparse.ArgumentParser(); parser.add_argument("--registry",type=Path,default=ROOT/"configs/residual_graph_deem_24cell_v1_registry.json"); parser.add_argument("--bundle-dir",type=Path,default=ROOT/"local_cache/deem_b3_moe_v1/bundles"); parser.add_argument("--out-dir",type=Path,default=ROOT/"local_cache/deem_b3_moe_v1/crossed_innovation_blend_full"); parser.add_argument("--workers",type=int,default=4); args=parser.parse_args(); registry=load_registry(args.registry); cells=[str(row["cell_id"]) for row in registry["cells"]]; args.out_dir.mkdir(parents=True,exist_ok=True)
 definition={"schema":SCHEMA+"_definition","cells":cells,"arms":[ARM],"seeds":list(SEEDS),"blend":BLEND,"source_hashes":{"runner":sha256_file(Path(__file__)),"core":sha256_file(ROOT/"spectral_utils/deem_b3_crossed_innovation_blend.py"),"innovation":sha256_file(ROOT/"spectral_utils/deem_b3_crossed_innovation.py")},"registry_content_sha256":registry["registry_content_sha256"],"labels_accessed_during_fit":False,"target_module_imported_during_fit":False}; definition["content_sha256"]=canonical_sha256(definition); atomic_write_json(args.out_dir/"RUN_DEFINITION.json",definition)
 payloads=[(cell,str(args.bundle_dir),str(args.out_dir),definition["content_sha256"]) for cell in cells]; records=[]
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures={pool.submit(fit_cell,payload):payload[0] for payload in payloads}
  for future in as_completed(futures): records.extend(future.result()); print("completed",futures[future],flush=True)
 records.sort(key=lambda row:(row["cell_id"],row["seed"])); assert len(records)==120; freeze={"schema":SCHEMA+"_freeze","run_definition_sha256":definition["content_sha256"],"records":records,"all_healthy":True,"labels_accessed_during_fit":False,"target_module_imported_during_fit":False}; freeze["content_sha256"]=canonical_sha256(freeze); atomic_write_json(args.out_dir/"SCORE_FREEZE.json",freeze); print("frozen",len(records),"fits")
if __name__=="__main__": main()
