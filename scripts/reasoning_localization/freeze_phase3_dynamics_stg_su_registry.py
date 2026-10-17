#!/usr/bin/env python3
"""Freeze the concrete dynamics STG-SU execution contract."""
from __future__ import annotations
import json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from spectral_utils.reconstruction_benchmark.io import atomic_write_json,sha256_file  # noqa:E402
from spectral_utils.reconstruction_benchmark.localization_contract import load_prepared_localization_cell,validate_fit_manifest  # noqa:E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa:E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa:E402
from scripts.reasoning_localization import run_phase3_deployed_upcr_prune_refit as p3d  # noqa:E402
from scripts.reasoning_localization import run_phase3_dynamics_stg_su as run  # noqa:E402

def main():
 if run.REGISTRY.exists():raise FileExistsError(run.REGISTRY)
 release=p1.DEFAULT_RELEASE.resolve();root=release/"build_A/localization/inputs";manifest=validate_fit_manifest(root/"MANIFEST.json",input_root=root);by={str(r["cell_id"]):r for r in manifest["cells"]};names=families=None;cells=[]
 for cell_id in p2r.PB_CELLS:
  source=by[cell_id];path=root/source["artifact_path"];cell=load_prepared_localization_cell(path,source);_,_,cn,cf=p3d._member_matrix(cell);names=cn if names is None else names;families=cf if families is None else families
  if cn!=names or cf!=families:raise RuntimeError("member roster drift")
  cells.append({"cell_id":cell_id,"input_sha256":sha256_file(path),"n_rows":len(cell.row_ids)})
 registry={"schema":"reasoning-localization-p3s-execution-v1","status":"FROZEN_BEFORE_RUN","experiment_id":run.EXPERIMENT,"variant_order":list(run.VARIANTS),"release_root":str(release),"cells":cells,"member_names":list(names),"member_families":list(families),"family_counts":{f:list(families).count(f) for f in sorted(set(families))},"fit_contract":"five outer grouped donor folds; nested five-fold STG covariance selection; held outer responses projection-only","su_config":dict(run.SU_CONFIG),"stg_seeds":[11,23,37],"stg_epochs":120,"stg_penalties":[.1,1.,3.,4.,5.],"probability_threshold":run.PROBABILITY_THRESHOLD,"minimum_fold_fraction":run.MIN_FOLD_FRACTION,"feature_permutation_seed":run.FEATURE_PERMUTATION_SEED,"random_support_seeds":list(run.RANDOM_SUPPORT_SEEDS),"primary_contrasts":[list(p) for p in run.PRIMARY],"multiplicity_family_size":run.FAMILY_SIZE,"practical_benefit":run.BENEFIT,"practical_harm":run.HARM,"alias_tolerance":run.ALIAS_TOLERANCE,"bootstrap_draws":p1.BOOTSTRAP_DRAWS,"bootstrap_seed":p1.BOOTSTRAP_SEED,"runner_sha256":sha256_file(Path(run.__file__).resolve()),"token_fusion_sha256":sha256_file(REPO/"spectral_utils/token_local_fusion.py"),"labels_opened":False,"supersedes":"P3S_EXECUTION_REGISTRY.json","amendment_reason":"Canonical SU is a diagnostic control on the 14-view dynamics family and may exceed the sufficient sparse-support theorem; theorem validity remains mandatory for STG, permutation and random-support refits.","canonical_control_theorem_exception":True}
 run.ROOT.mkdir(parents=True,exist_ok=True);atomic_write_json(run.REGISTRY,registry);print(json.dumps({"status":registry["status"],"family_counts":registry["family_counts"],"runner_sha256":registry["runner_sha256"],"token_fusion_sha256":registry["token_fusion_sha256"]},indent=2))
if __name__=="__main__":main()
