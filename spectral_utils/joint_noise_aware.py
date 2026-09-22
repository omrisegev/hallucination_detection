"""Reusable training-only composition of frozen Steps404-405 Joint selection.

Inputs are already standardized within each answer. No correctness labels are
accepted. Source identities define calibration blocks; inner folds define group
discovery. Failed models require an explicit fallback choice at scoring time.
"""
import numpy as np
from .joint_sparse_membership import (exact_aliases,alias_coordinates,source_grams,
    calibrate_penalty,sparse_membership)
from .joint_block_balanced import discover_sourcefold_groups
from .joint_feature_selection import checked_fit
from .joint_lsml import covariance_matrix
from .joint_group_reliability import group_reliability_weights
from .joint_sparse_refinement import refine_joint_membership
from .lsml_gate_locator_research import _orient


def learn_nonnull_structure(train,rowfold,grams,nrows,*,seed,calibration_seed,notify=None):
    def emit(status,**detail):
        if notify:notify(status,**detail)
    active=np.arange(train.shape[1]);rounds=[]
    for round_index in range(3):
        sub=train[:,active];cov=covariance_matrix(sub)
        discovery=discover_sourcefold_groups(sub,rowfold,seed=seed+100)
        if discovery['status']!='SELECTED':raise ValueError('NO_ADMISSIBLE_SPARSE_STRUCTURE')
        penalty,calibration=calibrate_penalty(grams[:,active][:,:,active],nrows,cov,
            np.asarray(discovery['labels']),seed=calibration_seed+100*round_index)
        emit('SPARSE_FIT',round=round_index,features=len(active),penalty=penalty)
        local,audit=sparse_membership(cov,np.asarray(discovery['labels']),penalty=penalty,
            seed=seed+200+round_index)
        selected=active[local]
        rounds.append(dict(round=round_index,active_before=active,active_after=selected,
            discovery=discovery,calibration=calibration,sparse=audit))
        emit('MEMBERSHIP',round=round_index,kept=len(selected),converged=audit['converged_starts'])
        if len(selected)<6:raise ValueError('TOO_FEW_ACTIVE_MEASUREMENTS')
        if np.array_equal(active,selected):break
        active=selected
    sub=train[:,active];cov=covariance_matrix(sub)
    discovery=discover_sourcefold_groups(sub,rowfold,seed=seed+100)
    candidates=sorted([c for c in discovery['candidates'] if c['valid']],
        key=lambda c:(-c['median_ari'],-c['mean_ari'],-c['minimum_ari'],c['K']))
    attempts=[]
    for candidate in candidates:
        try:checked=checked_fit(cov,np.asarray(candidate['labels']),seed=seed+200)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            attempts.append(dict(K=candidate['K'],valid=False,reason=str(exc)));continue
        j=checked.joint;local,readout=group_reliability_weights(j.model_covariance,j.global_loading,candidate['labels'])
        w=np.zeros(train.shape[1]);w[active]=local;attempts.append(dict(K=candidate['K'],valid=True))
        return dict(valid=True,active=active,unoriented_weights=w,rounds=rounds,attempts=attempts,
            final_discovery=discovery,final_labels=candidate['labels'],readout=readout,
            observed_covariance=cov,model_covariance=j.model_covariance,global_loading=j.global_loading,
            group_loading=j.group_loading,diagonal_audit=j.diagonal_audit,
            converged_starts=j.converged_starts,multistart=j.multistart_audit,jacobian=j.jacobian_audit)
    return dict(valid=False,rounds=rounds,active=active,attempts=attempts,failure='NO_VALID_DEBIASED_STRUCTURE')


def fit_noise_aware_joint(values,offsets,source_ids,inner_folds,*,anchor_index,seed=399170,notify=None):
    x=np.asarray(values,float);offsets=np.asarray(offsets,int);inner_folds=np.asarray(inner_folds)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('FINITE_MATRIX_REQUIRED')
    if offsets[0]!=0 or offsets[-1]!=len(x) or len(offsets)!=len(inner_folds)+1:
        raise ValueError('ANSWER_OFFSETS_MISMATCH')
    if len(source_ids)!=len(inner_folds) or np.any(np.diff(offsets)<1):raise ValueError('SOURCE_ROSTER_MISMATCH')
    if not 0<=anchor_index<x.shape[1]:raise ValueError('ANCHOR_OUT_OF_RANGE')
    aliases=exact_aliases(x);z=alias_coordinates(x,aliases)
    model=dict(valid=False,aliases=aliases,anchor_index=int(anchor_index),input_width=x.shape[1],seed=seed)
    try:
        gram,sizes,_=source_grams(z,offsets,source_ids,inner_folds)
        membership=learn_nonnull_structure(z,np.repeat(inner_folds,np.diff(offsets)),gram,int(sizes.sum()),
            seed=seed,calibration_seed=seed+5000,notify=notify)
        model['membership']=membership
        if not membership['valid']:
            model['failure']=membership['failure'];return model
        initial=np.asarray(membership['active']);part=np.asarray(membership['final_labels'])
        def trace(step,kept,retention):
            if notify and step%5==0:notify('REFINEMENT',removed=step,kept=kept,retention=retention)
        fit=refine_joint_membership(z[:,initial],part,seed=seed+200,notify=trace)
        w=np.zeros(z.shape[1]);w[initial]=fit['unoriented_weights']
        extended=np.column_stack((z,x[:,anchor_index]))
        w,orientation=_orient(extended,np.r_[w,0.],z.shape[1]);w=w[:-1]
        raw=np.zeros(x.shape[1])
        for j,ids in enumerate(aliases):raw[ids]=w[j]/len(ids)
        active=initial[np.asarray(fit['active'])]
        model.update(valid=True,refinement=fit,initial_active=initial,active=active,
            canonical_weights=w,expanded_weights=raw,orientation=orientation,
            active_original=[i for j in active for i in aliases[j]])
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        model['failure']=f'{type(exc).__name__}: {exc}'
    return model


def score_noise_aware_joint(model,values,*,allow_anchor_fallback=False):
    x=np.asarray(values,float)
    if x.ndim!=2 or x.shape[1]!=model['input_width'] or not np.isfinite(x).all():raise ValueError('SCORE_INPUT_MISMATCH')
    if not model['valid']:
        if allow_anchor_fallback:return x[:,model['anchor_index']].copy()
        raise ValueError('INVALID_JOINT_MODEL: '+model['failure'])
    return alias_coordinates(x,model['aliases'])@np.asarray(model['canonical_weights'])
