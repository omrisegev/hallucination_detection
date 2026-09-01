#!/usr/bin/env python3
"""Freeze and evaluate dynamics-only canonical SU and STG-SU controls."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from spectral_utils.dependency_fusion import sparse_upcr_fit  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import atomic_write_json, atomic_write_npz, load_npz_no_pickle, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import load_prepared_localization_cell, validate_fit_manifest  # noqa: E402
from spectral_utils.token_local_fusion import SU_CONFIG, TokenFusionPreparation, learn_stg_sparse_support  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization import run_phase3_compact_fusion as p3  # noqa: E402
from scripts.reasoning_localization import run_phase3_deployed_upcr_prune_refit as p3d  # noqa: E402
from scripts.reasoning_localization import run_phase3_family_expert_attribution as p3e  # noqa: E402
from scripts.reasoning_localization.register_phase3_dynamics_stg_su import EXPERIMENT, VARIANTS  # noqa: E402

S0,S1,S2,S3,S4=VARIANTS
H0="P3_H0_REFERENCE"
ROOT=p1.PROGRAM_ROOT/"phase_3/dynamics_stg_su"
OUTPUT=ROOT/"p3s_dynamics_stg_su_v1"
# V2 is an explicit pre-label amendment to the original frozen registry.  The
# canonical SU control may legitimately select more sparse pairs than the
# sufficient SU theorem allows on a 14-view family; it remains a diagnostic
# control, while the STG candidate and its support controls retain the theorem
# validity gate.
REGISTRY=ROOT/"P3S_EXECUTION_REGISTRY_AMENDMENT_V2.json"
SOURCE_P3E=p3e.OUTPUT/"score_freeze/cells"
PRIMARY=((S1,S0),(S2,S0),(S2,S1),(S2,S3),(S2,S4))
FAMILY_SIZE=len(PRIMARY)
BENEFIT=0.003
HARM=-0.003
ALIAS_TOLERANCE=1e-12
MIN_FOLD_FRACTION=0.80
PROBABILITY_THRESHOLD=0.75
FEATURE_PERMUTATION_SEED=2026083102
RANDOM_SUPPORT_SEEDS=tuple(range(2026083111,2026083131))


class DynamicsSTGError(RuntimeError): pass


def _seed(base:int,cell_id:str,fold:int)->int:
    return int(hashlib.sha256(f"{base}|{cell_id}|{fold}".encode()).hexdigest()[:16],16)


def _nested_preparation(donor:np.ndarray,donor_indices:np.ndarray,owner:np.ndarray,row_ids:list[str]|tuple[str,...],names:tuple[str,...],cell_id:str,outer_fold:int)->TokenFusionPreparation:
    donor_owners=owner[donor_indices]
    unique=np.unique(donor_owners)
    chunks=[np.flatnonzero(donor_owners==row) for row in unique]
    order=np.concatenate(chunks)
    values=np.asarray(donor[order],dtype=np.float64)
    lengths=np.asarray([len(chunk) for chunk in chunks],dtype=np.int64)
    offsets=np.concatenate(([0],np.cumsum(lengths)))
    nested_ids=tuple(str(row_ids[int(row)]) for row in unique)
    folds=np.asarray([int(hashlib.sha256(f"{cell_id}|{outer_fold}|{rid}".encode()).hexdigest()[:16],16)%5 for rid in nested_ids],dtype=np.int64)
    if set(folds.tolist())!=set(range(5)): raise DynamicsSTGError("nested STG folds are incomplete")
    fit_rows=np.repeat(np.arange(len(unique),dtype=np.int64),lengths)
    m=values.shape[1]
    return TokenFusionPreparation(
        values=values,token_offsets=offsets,row_ids=nested_ids,
        fit_indices=np.arange(len(values),dtype=np.int64),fit_row_indices=fit_rows,row_folds=folds,
        medians=np.zeros(m),keep=np.ones(m,dtype=bool),mean=np.zeros(m),std=np.ones(m),standardized_fit=values,
        stream_names=names,kept_stream_names=names,kept_family_names=tuple("entropy_dynamics" for _ in names),
        diagnostics={"outer_fold":outer_fold,"labels_seen_during_fit":False},
    )


def _fit_su(donor:np.ndarray,held:np.ndarray,fixed_support:np.ndarray|None=None)->tuple[np.ndarray,dict[str,Any]]:
    fitted=sparse_upcr_fit(donor.T,**dict(SU_CONFIG),fixed_support=fixed_support)
    weights=np.asarray(fitted.w_pcr,dtype=float).copy(); anchor=donor.mean(axis=1); corr=float(np.corrcoef(donor@weights,anchor)[0,1]); flipped=bool(np.isfinite(corr) and corr<0)
    if flipped: weights*=-1
    d=fitted.decomposition
    return -(held@weights),{
        "weights":weights.tolist(),"anchor_correlation":corr,"orientation_flipped":flipped,
        "g2_hat":float(fitted.g2_hat),"projection_residual":float(fitted.projection_residual),
        "decomposition_converged":bool(d.converged),"decomposition_iterations":int(d.n_iter),
        "sparse_fraction":float(d.sparse_fraction),"sparse_pairs":int(d.meta["nnz_pairs"]),
        "theorem_support_ok":bool(d.theorem_support_ok),"relative_residual":float(d.relative_residual),
        "fixed_support":bool(d.meta.get("fixed_support",False)),
    }


def _random_support(m:int,k:int,seed:int)->np.ndarray:
    iu=np.triu_indices(m,1); support=np.zeros((m,m),dtype=bool)
    if k:
        selected=np.random.default_rng(seed).choice(len(iu[0]),size=k,replace=False)
        support[iu[0][selected],iu[1][selected]]=True; support|=support.T
    return support


def _support_stability(supports:list[np.ndarray])->dict[str,Any]:
    vals=[]
    for a,b in combinations(supports,2):
        union=np.logical_or(a,b).sum(); vals.append(float(np.logical_and(a,b).sum()/union) if union else 1.0)
    return {"mean_pairwise_jaccard":float(np.mean(vals)),"min_pairwise_jaccard":float(np.min(vals)),"fold_pair_counts":[int(np.triu(s,1).sum()) for s in supports]}


def _load_registry(release:Path)->dict[str,Any]:
    row=json.loads(REGISTRY.read_text()); required={"schema":"reasoning-localization-p3s-execution-v1","status":"FROZEN_BEFORE_RUN","experiment_id":EXPERIMENT,"variant_order":list(VARIANTS),"primary_contrasts":[list(p) for p in PRIMARY],"multiplicity_family_size":FAMILY_SIZE,"runner_sha256":sha256_file(Path(__file__).resolve()),"token_fusion_sha256":sha256_file(REPO/"spectral_utils/token_local_fusion.py"),"minimum_fold_fraction":MIN_FOLD_FRACTION,"probability_threshold":PROBABILITY_THRESHOLD,"feature_permutation_seed":FEATURE_PERMUTATION_SEED,"random_support_seeds":list(RANDOM_SUPPORT_SEEDS)}
    for key,value in required.items():
        if row.get(key)!=value: raise DynamicsSTGError(f"execution registry mismatch: {key}")
    if Path(row["release_root"]).resolve()!=release.resolve(): raise DynamicsSTGError("release mismatch")
    return row


def freeze(release:Path,registry:Mapping[str,Any])->dict[str,Any]:
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    score_root=OUTPUT/"score_freeze"; score_root.mkdir(parents=True)
    input_root=release/"build_A/localization/inputs"; manifest=validate_fit_manifest(input_root/"MANIFEST.json",input_root=input_root); by_cell={str(r["cell_id"]):r for r in manifest["cells"]}
    records=[]; max_parent_alias=0.0; global_min_jaccard=1.0
    for position,cell_id in enumerate(p2r.PB_CELLS,1):
        source=by_cell[cell_id]; input_path=input_root/source["artifact_path"]; cell=load_prepared_localization_cell(input_path,source); prep,raw,names,families=p3d._member_matrix(cell)
        if list(names)!=registry["member_names"] or list(families)!=registry["member_families"]: raise DynamicsSTGError(f"roster drift in {cell_id}")
        indices={family:np.asarray([i for i,v in enumerate(families) if v==family],dtype=np.int64) for family in ("entropy_level","entropy_dynamics","partition_energy","topk_distribution")}; dyn=indices["entropy_dynamics"]; dyn_names=tuple(names[i] for i in dyn)
        owner=np.repeat(np.arange(len(cell.row_ids)),np.diff(np.asarray(cell.token_offsets))); token_scores={v:np.full(len(raw),np.nan) for v in VARIANTS}; folds_diag=[]; outer_supports=[]
        for fold in range(5):
            held_rows=np.flatnonzero(np.asarray(prep.row_folds)==fold); held_indices=np.flatnonzero(np.isin(owner,held_rows)); fit_folds=np.asarray(prep.row_folds)[np.asarray(prep.fit_row_indices)]; donor_indices=np.asarray(prep.fit_indices)[fit_folds!=fold]
            donor,held,scale=p3d._fold_standardize(raw,donor_indices,held_indices); donor_dyn,held_dyn=donor[:,dyn],held[:,dyn]
            parent_dyn,parent_diag=p3e._fit_iu(donor_dyn,held_dyn); canonical_dyn,canonical_diag=_fit_su(donor_dyn,held_dyn)
            nested=_nested_preparation(donor_dyn,donor_indices,owner,cell.row_ids,dyn_names,cell_id,fold)
            support_result=learn_stg_sparse_support(nested,probability_threshold=PROBABILITY_THRESHOLD,minimum_fold_fraction=MIN_FOLD_FRACTION); support=np.asarray(support_result.support,dtype=bool); outer_supports.append(support.copy())
            stg_dyn,stg_diag=_fit_su(donor_dyn,held_dyn,support)
            permutation=np.random.default_rng(_seed(FEATURE_PERMUTATION_SEED,cell_id,fold)).permutation(len(dyn)); perm_support=support[np.ix_(permutation,permutation)]; perm_dyn,perm_diag=_fit_su(donor_dyn,held_dyn,perm_support)
            k=int(np.triu(support,1).sum()); random_scores=[]; random_diags=[]
            for base_seed in RANDOM_SUPPORT_SEEDS:
                rs=_random_support(len(dyn),k,_seed(base_seed,cell_id,fold)); risk,diag=_fit_su(donor_dyn,held_dyn,rs); random_scores.append(risk); random_diags.append(diag)
            random_dyn=np.mean(np.vstack(random_scores),axis=0)
            equal={family:-held[:,idx].mean(axis=1) for family,idx in indices.items()}; shared=[equal["entropy_level"],equal["partition_energy"],equal["topk_distribution"]]
            for variant,dyn_score in ((S0,parent_dyn),(S1,canonical_dyn),(S2,stg_dyn),(S3,perm_dyn),(S4,random_dyn)):
                token_scores[variant][held_indices]=np.mean([shared[0],dyn_score,shared[1],shared[2]],axis=0)
            # Canonical SU is a method control, not the proposed support
            # learner.  On this 14-view family its threshold support can exceed
            # the paper's sufficient sparse-support bound; record that fact but
            # do not silently reinterpret it as a STG failure.  The candidate,
            # permutation and cardinality-matched random refits must all remain
            # theorem-valid and converged.
            candidate_support_diagnostics = [stg_diag, perm_diag, *random_diags]
            if any(
                not diagnostic["theorem_support_ok"]
                or not diagnostic["decomposition_converged"]
                for diagnostic in candidate_support_diagnostics
            ) or not canonical_diag["decomposition_converged"]:
                raise DynamicsSTGError(
                    f"support/refit validity failed in {cell_id} fold {fold}"
                )
            selected_pairs=[[dyn_names[i],dyn_names[j]] for i,j in zip(*np.where(np.triu(support,1)))]
            folds_diag.append({"fold":fold,"scale":scale,"parent":parent_diag,"canonical":canonical_diag,"stg":stg_diag,"stg_support":dict(support_result.diagnostics),"selected_pairs":selected_pairs,"permuted":perm_diag,"random_support_count":len(random_diags),"random_sparse_pairs":[d["sparse_pairs"] for d in random_diags]})
        stability=_support_stability(outer_supports); global_min_jaccard=min(global_min_jaccard,stability["mean_pairwise_jaccard"])
        if any(not np.isfinite(score).all() for score in token_scores.values()): raise DynamicsSTGError(f"incomplete scores in {cell_id}")
        source_arrays=load_npz_no_pickle(SOURCE_P3E/cell_id/"scores.npz"); arrays={"row_ids":np.asarray(cell.row_ids,dtype="<U80"),"segment_offsets":np.asarray(cell.segment_offsets,dtype="<i8"),"segment_lengths":np.asarray(cell.segment_ends-cell.segment_starts,dtype="<i8"),"h0_combined":source_arrays["h0_combined"]}
        for variant,score in token_scores.items(): arrays[f"{variant.lower()}_local"]=p1.topk_step_mean(score,cell.segment_starts,cell.segment_ends,k=10)
        alias=float(np.max(np.abs(arrays[f"{S0.lower()}_local"]-source_arrays["p3e1_dynamics_iu_only_local"]))); max_parent_alias=max(max_parent_alias,alias)
        if alias>ALIAS_TOLERANCE: raise DynamicsSTGError(f"P3E1 alias failed in {cell_id}: {alias}")
        target=score_root/"cells"/cell_id; target.mkdir(parents=True); score_sha=atomic_write_npz(target/"scores.npz",arrays)
        record={"schema":"reasoning-localization-p3s-cell-v1","experiment_id":EXPERIMENT,"cell_id":cell_id,"model_id":str(cell.model_id),"slice_id":str(cell.slice_id),"population_id":str(cell.population_id),"n_rows":len(cell.row_ids),"member_names":list(names),"member_families":list(families),"p3e_parent_alias_max_error":alias,"support_stability":stability,"fold_diagnostics":folds_diag,"labels_seen_during_fit":False,"targets_accessed_during_fit":False,"score_sha256":score_sha,"prepared_input_sha256":sha256_file(input_path)}; record["payload_sha256"]=c1.payload_sha(record); atomic_write_json(target/"RECORD.json",record)
        records.append({"cell_id":cell_id,"record_path":f"cells/{cell_id}/RECORD.json","record_sha256":sha256_file(target/"RECORD.json"),"score_sha256":score_sha}); print(f"score-freeze P3S0-P3S4: {cell_id} ({position}/8)",flush=True)
    result={"schema":"reasoning-localization-p3s-score-freeze-v1","status":"COMPLETE","experiment_id":EXPERIMENT,"variant_ids":list(VARIANTS),"records":records,"p3e_parent_alias_max_error":max_parent_alias,"minimum_cell_mean_support_jaccard":global_min_jaccard,"labels_seen_during_fit":False,"execution_registry_sha256":sha256_file(REGISTRY),"runner_sha256":sha256_file(Path(__file__).resolve()),"token_fusion_sha256":sha256_file(REPO/"spectral_utils/token_local_fusion.py")}; result["payload_sha256"]=c1.payload_sha(result); atomic_write_json(score_root/"SCORE_FREEZE_MANIFEST.json",result); return result


def _verified(manifest:Mapping[str,Any])->dict[str,Any]:
    out={}
    for item in manifest["records"]:
        rp=OUTPUT/"score_freeze"/item["record_path"]; sp=rp.parent/"scores.npz"
        if sha256_file(rp)!=item["record_sha256"] or sha256_file(sp)!=item["score_sha256"]: raise DynamicsSTGError("score-freeze hash mismatch")
        out[item["cell_id"]]={"record":json.loads(rp.read_text()),"arrays":load_npz_no_pickle(sp)}
    return out


def _rows(verified:Mapping[str,Any],labels:Mapping[str,Any],key:str)->dict[str,list[dict[str,Any]]]:
    out={m:[] for m in p1.QWEN_MODELS}
    for cell_id in p2r.PB_CELLS:
        record,arrays=verified[cell_id]["record"],verified[cell_id]["arrays"]
        for index,row_id in enumerate(arrays["row_ids"].astype(str)):
            lo,hi=map(int,arrays["segment_offsets"][index:index+2]); group_id,first_error=labels[cell_id][row_id]; out[record["model_id"]].append({"row_id":row_id,"group_id":group_id,"slice_id":record["slice_id"],"cell_id":cell_id,"model_id":record["model_id"],"first_error":first_error,"step_scores":arrays[key][lo:hi].tolist(),"step_lengths":arrays["segment_lengths"][lo:hi].tolist()})
    return out


def _status(delta:float,lo:float,hi:float)->str:
    if lo>BENEFIT:return "SUPPORTED_IMPROVEMENT"
    if hi<HARM:return "SUPPORTED_HARM"
    if delta>0 and lo<=0:return "PROMISING_UNCONFIRMED"
    return "INCONCLUSIVE"


def _contrast(left:str,right:str,metric:str,arms:Mapping[str,Any],simultaneous:bool)->dict[str,Any]:
    lp={r["metric_id"]:r for r in arms[left]["panels"]}[metric]; rp={r["metric_id"]:r for r in arms[right]["panels"]}[metric]; draws=np.asarray(arms[left]["samples"][metric])-np.asarray(arms[right]["samples"][metric]); q=.025/FAMILY_SIZE if simultaneous and metric=="official_macro_f1" else .025
    lc={r["cell_id"]:r for r in arms[left]["by_cell"]}; rc={r["cell_id"]:r for r in arms[right]["by_cell"]}; cells={cell:float(lc[cell][metric])-float(rc[cell][metric]) for cell in lc}; delta=float(lp["value"]-rp["value"]); lo,hi=float(np.quantile(draws,q)),float(np.quantile(draws,1-q))
    return {"contrast_id":f"pb::{left}::{right}::{metric}","left_variant_id":left,"right_variant_id":right,"metric_id":"macro_f1" if metric=="official_macro_f1" else metric,"delta":delta,"ci_low":lo,"ci_high":hi,"statistical_status":_status(delta,lo,hi),"wins":sum(v>1e-12 for v in cells.values()),"ties":sum(abs(v)<=1e-12 for v in cells.values()),"losses":sum(v<-1e-12 for v in cells.values()),"worst_unit_delta":min(cells.values()),"worst_unit_id":min(cells,key=cells.get),"multiplicity_family_size":FAMILY_SIZE if simultaneous and metric=="official_macro_f1" else 1}


def _write_csv(path:Path,rows:list[dict[str,Any]])->None:
    with path.open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def evaluate(release:Path,manifest:Mapping[str,Any])->dict[str,Any]:
    verified=_verified(manifest); labels=p1._load_pb_labels(release); evaluator=importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation"); h0=c1.evaluate_arm(H0,_rows(verified,labels,"h0_combined"),evaluator); arms={H0:h0}
    for variant in VARIANTS: arms[variant]=p3._rerank(variant,h0,_rows(verified,labels,f"{variant.lower()}_local"),evaluator)
    abstain={(r["cell_id"],r["row_id"]):int(r["prediction_step"])==-1 for r in h0["decisions"]}; mismatches={arm:sum((int(r["prediction_step"])==-1)!=abstain[(r["cell_id"],r["row_id"])] for r in arms[arm]["decisions"]) for arm in VARIANTS}
    if any(mismatches.values()): raise DynamicsSTGError(f"H0 abstention alias failed: {mismatches}")
    pairs=[*PRIMARY,(S3,S0),(S4,S0)]; contrasts=[_contrast(left,right,metric,arms,(left,right) in PRIMARY) for left,right in pairs for metric in p1.PB_METRICS]; root=OUTPUT/"evaluation"; root.mkdir(); panels=[r for arm in arms.values() for r in arm["panels"]]
    _write_csv(root/"PROCESSBENCH_PANELS.csv",panels); _write_csv(root/"PROCESSBENCH_BY_CELL.csv",[r for arm in arms.values() for r in arm["by_cell"]]); _write_csv(root/"PAIRWISE_CONTRASTS.csv",contrasts)
    parent={(r["cell_id"],r["row_id"]):r for r in arms[S0]["decisions"]}; flips=[]
    for variant in (S1,S2,S3,S4):
        for r in arms[variant]["decisions"]:
            b=parent[(r["cell_id"],r["row_id"])]
            if int(r["prediction_step"])!=int(b["prediction_step"]): flips.append({"variant_id":variant,"cell_id":r["cell_id"],"row_id":r["row_id"],"parent_prediction_step":b["prediction_step"],"candidate_prediction_step":r["prediction_step"],"first_error":r["true_first_error"]})
    if flips:_write_csv(root/"PREDICTION_FLIPS.csv",flips)
    primary=[r for r in contrasts if r["metric_id"]=="macro_f1" and (r["left_variant_id"],r["right_variant_id"]) in PRIMARY]; pm={(r["left_variant_id"],r["right_variant_id"]):r for r in primary}; mechanism=pm[(S2,S3)]["ci_low"]>0 and pm[(S2,S4)]["ci_low"]>0
    summary={"schema":"reasoning-localization-p3s-evaluation-v1","status":"COMPLETE","experiment_id":EXPERIMENT,"primary_contrasts":primary,"p3e_parent_alias_max_error":manifest["p3e_parent_alias_max_error"],"minimum_cell_mean_support_jaccard":manifest["minimum_cell_mean_support_jaccard"],"abstention_mismatches":mismatches,"support_mechanism_supported":mechanism,"bootstrap_draws":p1.BOOTSTRAP_DRAWS,"bootstrap_seed":p1.BOOTSTRAP_SEED}; summary["payload_sha256"]=c1.payload_sha(summary); atomic_write_json(root/"SUMMARY.json",summary)
    vals={r["arm_id"]:float(r["value"]) for r in panels if r["metric_id"]=="official_macro_f1" and r["arm_id"] in VARIANTS}; lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="560" viewBox="0 0 1080 560">','<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:system-ui;fill:#172033}.t{font-size:22px;font-weight:700}.b{font-size:13px;font-weight:600}.l{font-size:13px}</style>','<text x="25" y="35" class="t">Dynamics IU, canonical SU and STG-SU</text>']
    for i,v in enumerate(VARIANTS):lines.extend([f'<text x="25" y="{85+38*i}" class="b">{v}</text>',f'<text x="690" y="{85+38*i}" class="b">{vals[v]:.6f}</text>'])
    lines.append('<text x="25" y="315" class="t">Frozen primary paired contrasts</text>')
    for i,r in enumerate(primary):lines.extend([f'<text x="25" y="{355+34*i}" class="l">{r["left_variant_id"]} − {r["right_variant_id"]}</text>',f'<text x="650" y="{355+34*i}" class="b">{r["delta"]:+.5f} [{r["ci_low"]:+.5f},{r["ci_high"]:+.5f}]</text>'])
    lines.append('</svg>'); (root/"P3S_RESULTS.svg").write_text("\n".join(lines)+"\n"); return summary


def main()->None:
    started=time.perf_counter(); release=p1.DEFAULT_RELEASE.resolve(); registry=_load_registry(release); frozen=freeze(release,registry); summary=evaluate(release,frozen); atomic_write_json(OUTPUT/"RUN_COMPLETE.json",{"status":"COMPLETE","elapsed_seconds":time.perf_counter()-started,"summary":summary}); print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
