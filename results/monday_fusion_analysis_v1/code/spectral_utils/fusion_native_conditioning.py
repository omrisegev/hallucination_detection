"""Condition the original Joint inverse while freezing its groups and routes.

The original run did not save its fitted covariance. Reproduce that fit on
the frozen same-answer matrix/partition, and require condition-1000 weight,
window-score and native-decision replay before emitting any new candidate.
"""
from collections import Counter
from copy import deepcopy
import numpy as np
from .answer_localization_v2 import JOINT_SEED, moment_plan, prepare_local, mixture_readout
from .fusion_pair_quality import ARMS as PARENT_ARMS
from .joint_lsml import covariance_matrix, fit_joint_lsml, regularized_joint_map_weights
from .short_cycle_localization import scaled_oriented_weight
from .window_localization import windows_to_tokens

CONDITIONS=(30,100,300)
FAMILIES=('moment','context','single','dual')
NEW_ARMS=tuple(f'{family}__cond{condition}' for family in FAMILIES for condition in CONDITIONS)
ARMS=PARENT_ARMS+NEW_ARMS


def condition_head(z,indices,anchor,covariance,loading,condition):
    if condition not in (*CONDITIONS,1000):raise ValueError('UNREGISTERED_CONDITION')
    fit=z[indices]
    w,info=regularized_joint_map_weights(fit,covariance,loading,mode='liu',lam=0.,target_condition=condition)
    w,boundary=scaled_oriented_weight(w,fit,anchor)
    risk=-z@w
    if not np.isfinite(risk).all():raise ValueError('NONFINITE_RISK')
    return risk,{'inverse':info,**boundary,'standardized_weights':w}


def compose_fixed_routes(arrays,methods,routing):
    """Use original eligibility only. A failed new head/readout stays failed."""
    out,meta={},{}
    for policy in ('single','dual'):
        route=routing['routes'][policy]
        if route not in ('moment_joint','context_joint','moment_iu'):raise ValueError('UNKNOWN_ORIGINAL_ROUTE')
        for condition in CONDITIONS:
            arm=f'{policy}__cond{condition}'
            source='moment__iu' if route=='moment_iu' else f'{route.removesuffix("_joint")}__cond{condition}'
            detail=deepcopy(methods[source]);detail.update(source_arm=source,route=route)
            if detail['valid']:
                for suffix in ('window','risk'):out[arm+'__'+suffix]=np.asarray(arrays[source+'__'+suffix]).copy()
            elif detail['decision_valid'] or detail['fixed_iu_valid']:raise ValueError('INVALID_FIT_HAS_DECISION')
            meta[arm]=detail
    return out,meta


def score_conditioning(parent_arrays,parent_metadata,original_metadata,token_count):
    """No labels or cross-answer quantities are accepted by this scorer."""
    arrays={k:np.asarray(v).copy() for k,v in parent_arrays.items()}
    methods=deepcopy(parent_metadata['methods']);assert set(methods)==set(PARENT_ARMS)
    routing=deepcopy(original_metadata['routing'])
    plan=moment_plan(token_count,8)
    for k,v in (('window_starts',plan.starts),('window_ends',plan.ends),('fit_indices',plan.fit_indices)):
        np.testing.assert_array_equal(arrays[k],v)
    ss,ee=arrays['step_starts'],arrays['step_ends']
    if ss.ndim!=1 or ss.shape!=ee.shape or not len(ss) or np.any(ss<0) or np.any(ee>token_count) or np.any(ee<=ss):
        raise ValueError('INVALID_OFFICIAL_SPANS')
    diagnostics={'banks':{},'labels_accessed':False,'fit_scope':'original_same_answer_partition_replay'}
    reference=methods['moment__iu']

    def readout(risk):
        token=windows_to_tokens(plan,risk);steps=np.array([token[a:b].max() for a,b in zip(ss,ee)])
        d={'valid':True,'decision_valid':False,'fixed_iu_valid':bool(reference['valid'] and reference['decision_valid']),
           'prediction':None,'fixed_iu_prediction':None,'peak':int(np.argmax(steps))}
        try:
            gate=mixture_readout(risk[plan.fit_indices],steps)
            d.update(gate=gate,decision_valid=True,prediction=d['peak'] if gate['prediction']!=-1 else -1)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:d['readout_error']=str(exc)
        if d['fixed_iu_valid']:d['fixed_iu_prediction']=d['peak'] if reference['prediction']!=-1 else -1
        return steps,d

    for bank in ('moment','context'):
        original=original_metadata['methods'][bank+'__joint0']
        assert methods[bank+'__joint0']==original
        if not original['valid']:
            diagnostics['banks'][bank]={'status':'ORIGINAL_INVALID_PRESERVED','reason':original.get('reason',original.get('status'))}
            for condition in CONDITIONS:
                arm=f'{bank}__cond{condition}'
                methods[arm]={'valid':False,'decision_valid':False,'fixed_iu_valid':False,'prediction':None,
                    'fixed_iu_prediction':None,'source_arm':arm,'route':'pure_original_joint','reason':'ORIGINAL_JOINT_INVALID'}
            continue
        bd=original_metadata['diagnostics']['banks'][bank];shared=bd['shared']
        z,anchor,normal=prepare_local(arrays[bank+'__features'],bd['names'],plan.fit_indices)
        for key in ('mean','sd','feature_signs'):np.testing.assert_allclose(normal[key],shared[key],atol=1e-12,rtol=1e-12)
        assert normal['active_features']==shared['active_features']
        groups=np.asarray(shared['joint']['groups']);assert min(Counter(groups).values())>=3
        joint=fit_joint_lsml(covariance_matrix(z[plan.fit_indices]),groups,anchor_index=anchor,
            seed=JOINT_SEED,starts=5,max_sweeps=5000)
        jac=joint.jacobian_audit
        if not (joint.converged and joint.multistart_audit['status']=='PASS' and jac['full_global_rank']
                and np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8):raise ValueError('ORIGINAL_FIT_REPLAY_FAILED')
        np.testing.assert_allclose(joint.relative_offdiag_misfit,shared['joint']['relative_offdiag_misfit'],atol=1e-12,rtol=1e-10)
        baseline,base_detail=condition_head(z,plan.fit_indices,anchor,joint.model_covariance,joint.global_loading,1000)
        np.testing.assert_allclose(base_detail['standardized_weights'],original['standardized_weights'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(baseline,arrays[bank+'__joint0__window'],atol=1e-10,rtol=1e-10)
        base_steps,base_readout=readout(baseline)
        np.testing.assert_allclose(base_steps,arrays[bank+'__joint0__risk'],atol=1e-10,rtol=1e-10)
        for key in ('decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):assert base_readout[key]==original[key],key
        if original['decision_valid']:np.testing.assert_allclose(base_readout['gate']['bic'],original['gate']['bic'],atol=1e-9,rtol=1e-10)
        diagnostics['banks'][bank]={'status':'ORIGINAL_FIT_REPLAY_PASS','normalization':normal,'groups':groups,
            'jacobian':jac,'diagonal':joint.diagonal_audit,'multistart':joint.multistart_audit,
            'relative_offdiag_misfit':joint.relative_offdiag_misfit,'condition1000':base_detail,
            'maximum_parent_risk_difference':float(np.max(np.abs(baseline-arrays[bank+'__joint0__window'])))}
        for suffix,value in (('covariance',joint.model_covariance),('v',joint.global_loading),('u',joint.group_loading)):
            arrays['original_'+bank+'__'+suffix]=value
        for condition in CONDITIONS:
            arm=f'{bank}__cond{condition}'
            risk,detail=condition_head(z,plan.fit_indices,anchor,joint.model_covariance,joint.global_loading,condition)
            steps,decision=readout(risk)
            methods[arm]={**detail,**decision,'source_arm':arm,'route':'pure_original_joint'}
            arrays[arm+'__window']=risk;arrays[arm+'__risk']=steps
    extra,meta=compose_fixed_routes(arrays,methods,routing);arrays.update(extra);methods.update(meta)
    assert set(methods)==set(ARMS)
    return arrays,methods,routing,diagnostics
