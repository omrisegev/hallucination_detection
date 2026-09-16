#!/usr/bin/env python3
"""Bounded A-D Soft Joint experiment on one automatically grouped L12 roster."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lsml_gate_locator_research_v1 import (
    load_inputs, merge_step_bank, score_locator,
)
from spectral_utils.joint_lsml import (
    covariance_matrix, discover_loao_consensus_groups, fit_joint_lsml,
    hierarchical_joint_weights,
)
from spectral_utils.lsml_gate_locator_research import (
    FusionRecipe, answer_standardize, fit_fusion_weights, fit_joint_mask,
    global_loading_weights, normalized_residual_affinity, weight_diagnostics,
)


L12 = (
    "step::digit.disagreement::top1",
    "step::digit.disagreement::top2",
    "step::digit.token_clock_innovation::top1",
    "step::q15.VE0.75.prefix_mean_innovation::top10",
    "step::q15.H0lim.prefix_mean_innovation::top10",
    "step::renyi_escort.a0.25::top10",
    "step::renyi_escort.a8::top10",
    "step::direct_probability.rank_1_risk::top10",
    "step::direct_probability.rank_3_risk::top10",
    "step::direct_probability.rank_9_risk::top10",
    "step::direct_probability.rank_10_risk::top8",
    "step::step395.logtail15::top10",
    "step::step395.logtail50::top10",
    "step::step395.mass_above::top10",
)


def rows_for(offsets, answers):
    return np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in answers])


def orient_normalize(values, weight, anchor=0):
    w=np.asarray(weight,float).copy();score=np.asarray(values)@w
    from scipy.stats import spearmanr
    rho=float(spearmanr(score,np.asarray(values)[:,anchor]).statistic)
    if np.isfinite(rho) and rho<0:w=-w;rho=-rho
    w/=max(float(np.abs(w).sum()),1e-12)
    return w,rho


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,)):return float(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    return value


def main():
    data=load_inputs();raw,names=merge_step_bank(data);index={n:i for i,n in enumerate(names)}
    missing=[n for n in L12 if n not in index]
    if missing:raise RuntimeError(f"missing L12 streams: {missing}")
    x=answer_standardize(raw[:,[index[n] for n in L12]],data["offsets"])
    folds=np.asarray(data["folds"]);offsets=np.asarray(data["offsets"])
    source=np.asarray(data["groups"],str);_,source_id=np.unique(source,return_inverse=True)
    step_owner=np.repeat(source_id,np.diff(offsets))
    outputs={name:np.empty(len(x)) for name in (
        "continuous","hard_global","hard_hierarchical","soft_coassignment","affinity_no_k")}
    audit=[]
    for outer in range(5):
        train_answers=np.flatnonzero(folds!=outer);test_answers=np.flatnonzero(folds==outer)
        tr=rows_for(offsets,train_answers);te=rows_for(offsets,test_answers);train=x[tr]
        recipe=FusionRecipe("continuous",L12,"continuous",anchor=0)
        cw,cmeta=fit_fusion_weights(train,recipe,seed=397000+outer)
        outputs["continuous"][te]=x[te]@cw
        print(f"fold {outer}: discovering automatic groups",flush=True)
        discovery=discover_loao_consensus_groups(
            train,step_owner[tr],k_range=(3,4),seed=397100+outer,
            minimum_group_size=3,minimum_held_admissible_fraction=.95,
            pairwise_diagnostic_cap=4096,
        )
        if discovery["status"]!="SELECTED":
            raise RuntimeError(f"L12 auto partition blocked in fold {outer}: {discovery['status']}")
        labels=np.asarray(discovery["labels"],int);cov=covariance_matrix(train)
        hard=fit_joint_lsml(cov,labels,anchor_index=0,seed=397200+outer,starts=5)
        hw,h_rho=global_loading_weights(train,hard,0)
        outputs["hard_global"][te]=x[te]@hw
        _,hier_raw,hier_meta=hierarchical_joint_weights(train,labels,hard.global_loading,anchor_index=0,small_m_guard=True)
        hier,hier_rho=orient_normalize(train,hier_raw,0)
        outputs["hard_hierarchical"][te]=x[te]@hier
        soft_mask=np.asarray(discovery["mean_loao_coassignment"],float);np.fill_diagonal(soft_mask,0.)
        soft=fit_joint_mask(cov,soft_mask,anchor_index=0,seed=397300+outer,starts=5)
        sw,srho=global_loading_weights(train,soft,0)
        outputs["soft_coassignment"][te]=x[te]@sw
        affinity=normalized_residual_affinity(cov)
        afit=fit_joint_mask(cov,affinity,anchor_index=0,seed=397400+outer,starts=5)
        aw,arho=global_loading_weights(train,afit,0)
        outputs["affinity_no_k"][te]=x[te]@aw
        audit.append({
            "outer":outer,"K":discovery["K"],"group_sizes":discovery["group_sizes"],
            "median_ari":discovery["median_ari"],"minimum_ari":discovery["minimum_ari"],
            "continuous":{"weights":cw,"meta":cmeta,**weight_diagnostics(cw)},
            "hard":{"weights_global":hw,"weights_hierarchical":hier,
                    "multistart":hard.multistart_audit["status"],"converged":hard.converged,
                    "misfit":hard.relative_offdiag_misfit,"jacobian":hard.jacobian_audit,
                    "global_anchor_rho":h_rho,"hierarchical_anchor_rho":hier_rho,"hierarchical_meta":hier_meta},
            "soft":{"weights":sw,"multistart":soft.multistart_audit["status"],"converged":soft.converged,
                    "misfit":soft.relative_offdiag_misfit,"jacobian":soft.jacobian_audit,"anchor_rho":srho,
                    "mask_min":float(soft_mask.min()),"mask_max":float(soft_mask.max()),
                    "mask_mean":float(soft_mask[~np.eye(len(soft_mask),dtype=bool)].mean())},
            "affinity":{"weights":aw,"multistart":afit.multistart_audit["status"],"converged":afit.converged,
                        "misfit":afit.relative_offdiag_misfit,"jacobian":afit.jacobian_audit,"anchor_rho":arho,
                        "mask_mean":float(affinity[~np.eye(len(affinity),dtype=bool)].mean())},
        })
        print(f"fold {outer}: K={discovery['K']} sizes={discovery['group_sizes']} done",flush=True)
    metrics={name:{k:v for k,v in score_locator(score,data["current_gate"],data).items()
                   if k in ("pb","within","within_n","pb_cells")}
             for name,score in outputs.items()}
    incumbent=score_locator(data["digit025_scores"],data["current_gate"],data)
    result={"roster":L12,"incumbent":{"pb":incumbent["pb"],"within":incumbent["within"]},
            "metrics":metrics,"audit":audit,"labels_used_for_fit":False,"development_only":True}
    root=ROOT/"results/soft_joint_auto_v1";root.mkdir(parents=True,exist_ok=True)
    (root/"RUN.json").write_text(json.dumps(clean(result),indent=2,sort_keys=True)+"\n")
    np.savez_compressed(root/"OOF_SCORES.npz",**outputs)
    print(json.dumps(clean({"incumbent":result["incumbent"],"metrics":metrics,
        "folds":[{"outer":a["outer"],"K":a["K"],"sizes":a["group_sizes"],
                  "hard":a["hard"]["multistart"],"soft":a["soft"]["multistart"],
                  "affinity":a["affinity"]["multistart"]} for a in audit]}),indent=2))


if __name__=="__main__":main()
