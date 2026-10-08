"""Context views support existing answer-fitted IU/Joint fusion; no target API.

EMA mechanism follows unified_causal_iu, with first-observation initialization
instead of its zero-reference startup. Historical fitted/label-selected state
is never imported. Smoothing does not create independent observations.
"""
from __future__ import annotations

import hashlib
import numpy as np
from .answer_localization_v2 import (
    PRIMITIVES,STREAM_NAMES,JOINT_SEED,moment_plan,moment_matrix,prepare_local,mixture_readout)
from .adapted_dufs import adapted_dufs_soft_gates
from .joint_lsml import covariance_matrix,discover_loao_consensus_groups,fit_joint_lsml,regularized_joint_map_weights
from .laplacian_upcr import IU_FIT_DEFAULTS,build_graph_from_features,permute_graph
from .short_cycle_localization import scaled_oriented_weight
from .upcr import upcr_fit
from .window_localization import windows_to_tokens

CORE_NAMES=('equal','iu','joint0','graph010','graph_perm')
JOINT_NAMES=CORE_NAMES[2:]
PARENT_METHODS=dict(zip(CORE_NAMES,('equal','iu','joint_lambda0','joint_graph010','joint_graph_permuted')))
ARMS=tuple(b+'__'+c for b in ('moment','context') for c in CORE_NAMES)+tuple(
    b+'__'+c for b in ('moment_allk','context_allk') for c in JOINT_NAMES)+('entropy_parent',)
REP='moments27_local8'
LEGACY_K=(3,4,6,8)


def prefix_ema(values,span):
    x=np.asarray(values,float)
    if x.ndim!=2 or not len(x) or span<1 or not np.isfinite(x).all():
        raise ValueError('Invalid EMA input')
    alpha=2./(span+1.); out=np.empty_like(x);out[0]=x[0]
    for i in range(1,len(x)):out[i]=(1-alpha)*out[i-1]+alpha*x[i]
    return out


def context_matrix(raw,plan):
    raw=np.asarray(raw,float)
    if raw.shape!=(plan.token_count,len(STREAM_NAMES)):raise ValueError('RAW_SCHEMA_MISMATCH')
    x=raw[:,[STREAM_NAMES.index(s) for s in PRIMITIVES]]
    fast,slow=prefix_ema(x,8),prefix_ema(x,32)
    # Keep the parent's reduction order for a bit-identical level-column anchor.
    values=np.asarray([np.column_stack((x[a:b].mean(axis=0),fast[a:b].mean(axis=0),
        slow[a:b].mean(axis=0))).ravel() for a,b in zip(plan.starts,plan.ends)])
    names=[s+'__'+op for s in PRIMITIVES for op in ('level','ema8','ema32')]
    return values,names


def expanded_k(active_p):
    if active_p<0 or int(active_p)!=active_p:raise ValueError('Invalid feature count')
    return tuple(range(3,int(active_p)//3+1))


def joint_valid(fit):
    jac=fit.jacobian_audit
    return bool(fit.converged and fit.multistart_audit['status']=='PASS' and jac['full_global_rank']
        and np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8)


def score_context_bank(raw,parent_arrays,parent_metadata,identity):
    plan=moment_plan(len(raw),8); indices=plan.fit_indices
    parent=parent_metadata['report']; old_shared=parent['representations'][REP]['shared']
    original_iu=parent['methods'][REP+'__iu']
    assert original_iu.get('valid') and original_iu.get('readout_valid')
    common_open=original_iu['readout']['prediction']!=-1
    arrays={'step_starts':parent_arrays['step_starts'],'step_ends':parent_arrays['step_ends'],
        'window_starts':plan.starts,'window_ends':plan.ends,'fit_indices':indices}
    methods={};diagnostics={'banks':{},'groupings':{},'joint_fits':{},'common_iu_gate_open':common_open}

    def admit(arm,risk,detail,old=None):
        risk=np.asarray(risk,float);token=windows_to_tokens(plan,risk)
        steps=np.asarray([token[a:b].max() for a,b in zip(arrays['step_starts'],arrays['step_ends'])])
        if not np.isfinite(steps).all():raise ValueError('NONFINITE_STEP_SCORE')
        arrays[arm+'__window']=risk;arrays[arm+'__risk']=steps
        peak=int(np.argmax(steps));detail.update(valid=True,fixed_iu_valid=True,peak=peak,
            fixed_iu_prediction=peak if common_open else -1)
        try:
            gate=dict(old['readout']) if old is not None else mixture_readout(risk[indices],steps)
            detail.update(decision_valid=True,gate=gate,prediction=peak if gate['prediction']!=-1 else -1)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            detail.update(decision_valid=False,readout_error=str(exc),prediction=None)
        methods[arm]=detail

    def fail(arm,reason,detail=None):
        methods[arm]={'valid':False,'decision_valid':False,'fixed_iu_valid':False,'reason':reason,**(detail or {})}

    for core,source in {**{'moment__'+c:REP+'__'+m for c,m in PARENT_METHODS.items()},
                        'entropy_parent':'entropy_mean_w8'}.items():
        old=parent['methods'][source]
        if old.get('valid'):
            admit(core,parent_arrays[source+'__window'],{'parent_replay':True,'source':source},old=old if old.get('readout_valid') else None)
            np.testing.assert_array_equal(arrays[core+'__risk'],parent_arrays[source+'__step'])
        else:fail(core,'PARENT_INVALID',{'original':old})

    seed=int(hashlib.sha256((identity+'/'+REP).encode()).hexdigest()[:8],16)
    permutation=np.random.default_rng(seed).permutation(len(indices))
    if 'gates' in old_shared:
        assert seed==old_shared['gates']['seed'],'PARENT_PERMUTATION_IDENTITY_DRIFT'
        np.testing.assert_array_equal(permutation,old_shared['gates']['permutation'])
    diagnostics['graph_permutation']=permutation;diagnostics['graph_permutation_seed']=seed
    for bank in ('moment','context'):
        if bank=='moment':
            values=parent_arrays[REP+'__features'];names=[s+'__'+op for s in PRIMITIVES for op in ('level','sd','slope')]
        else:values,names=context_matrix(raw,plan)
        arrays[bank+'__features']=values
        try:
            z,anchor,normal=prepare_local(values,names,indices);fit=z[indices]
            arrays[bank+'__normalized']=z
            diagnostics['banks'][bank]={'status':'OK','normalization':normal,'names':names}
            if bank=='moment':
                for key in ('mean','sd','feature_signs'):np.testing.assert_allclose(normal[key],old_shared[key],atol=1e-12)
            else:
                for core in ('equal','iu'):
                    try:
                        if core=='equal':w=np.ones(fit.shape[1])/fit.shape[1];detail={}
                        else:
                            iu=upcr_fit(fit.T,**dict(IU_FIT_DEFAULTS))
                            if iu.abstained:raise ValueError('IU_ABSTAINED')
                            w=iu.w;detail={'g2_hat':iu.g2_hat}
                        w,boundary=scaled_oriented_weight(w,fit,anchor)
                        arrays[bank+'__'+core+'__weights']=w
                        admit(bank+'__'+core,-z@w,{**detail,**boundary,'parent_replay':False})
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:fail(bank+'__'+core,str(exc))
            graph=None;gates=None;fitted={}
            for roster in (('allk',) if bank=='moment' else ('legacy','allk')):
                family=bank if roster=='legacy' else bank+'_allk'
                try:
                    krange=LEGACY_K if roster=='legacy' else expanded_k(fit.shape[1])
                    if not krange:raise ValueError('TOO_FEW_FEATURES_FOR_THREE_GROUPS')
                    blocks=np.minimum(3,np.arange(len(fit))*4//len(fit))
                    grouping=discover_loao_consensus_groups(fit,blocks,k_range=krange,seed=JOINT_SEED,
                        minimum_group_size=3,minimum_held_admissible_fraction=.95,use_minimum_ari_tiebreak=True)
                    diagnostics['groupings'][family]={'candidate_k':krange,**grouping}
                    if grouping['status']!='SELECTED':raise ValueError('BLOCKED_NO_ADMISSIBLE_PARTITION')
                    key=tuple(int(k) for k in grouping['labels'])
                    same_parent=(bank=='moment' and 'joint' in old_shared and
                        np.array_equal(grouping['labels'],old_shared['joint']['groups']))
                    reused=key in fitted
                    if not reused:
                        fitted[key]=fit_joint_lsml(covariance_matrix(fit),grouping['labels'],anchor_index=anchor,
                            seed=JOINT_SEED,starts=5,max_sweeps=5000)
                    joint=fitted[key];jac=joint.jacobian_audit
                    diagnostics['joint_fits'][family]={'reused_identical_partition':reused,'same_parent_partition':same_parent,'groups':grouping['labels'],
                        'converged':joint.converged,'multistart':joint.multistart_audit,'jacobian':jac,
                        'diagonal':joint.diagonal_audit,'relative_offdiag_misfit':joint.relative_offdiag_misfit,
                        'valid':joint_valid(joint)}
                    arrays[family+'__model_covariance']=joint.model_covariance
                    arrays[family+'__global_loading']=joint.global_loading
                    if not joint_valid(joint):raise ValueError('JOINT_FIT_INVALID')
                    w,detail=regularized_joint_map_weights(fit,joint.model_covariance,joint.global_loading,
                        mode='liu',lam=0.,target_condition=1000.)
                    w,boundary=scaled_oriented_weight(w,fit,anchor);arrays[family+'__joint0__weights']=w
                    admit(family+'__joint0',-z@w,{'inverse':detail,**boundary,'parent_replay':False})
                    if graph is None:
                        if bank=='moment' and 'gates' in old_shared:
                            gates=np.asarray(old_shared['gates']['values']);gate_detail={'source':'exact_parent_same_answer'}
                        else:gates,gate_detail=adapted_dufs_soft_gates(fit.T,seeds=(0,1,2),epochs=120)
                        graph=build_graph_from_features(fit.T,gates=gates,k=7)
                        arrays[bank+'__gates']=gates
                        diagnostics['banks'][bank]['gates']=gate_detail
                    for core,g in (('graph010',graph),('graph_perm',permute_graph(graph,permutation))):
                        w,detail=regularized_joint_map_weights(fit,joint.model_covariance,joint.global_loading,
                            mode='liu',lam=.1,gates=gates,graph=g,graph_k=7,target_condition=1000.)
                        w,boundary=scaled_oriented_weight(w,fit,anchor);arrays[family+'__'+core+'__weights']=w
                        admit(family+'__'+core,-z@w,{'inverse':detail,**boundary,'parent_replay':False})
                    if same_parent:
                        for core in JOINT_NAMES:
                            if methods['moment__'+core]['valid']:
                                np.testing.assert_allclose(arrays[family+'__'+core+'__window'],
                                    arrays['moment__'+core+'__window'],atol=1e-10,rtol=1e-10)
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    for core in JOINT_NAMES:
                        if family+'__'+core not in methods:fail(family+'__'+core,str(exc))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            diagnostics['banks'][bank]={'status':'UNAVAILABLE','reason':str(exc)}
            for arm in ARMS:
                if arm.startswith(bank) and arm not in methods:fail(arm,str(exc))
    assert set(methods)==set(ARMS)
    return arrays,methods,diagnostics
