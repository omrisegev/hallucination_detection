"""Graph/conditioning interaction on saved original Joint fits and fixed routes.

The graph-smoothed equal control replaces the Joint covariance/loading by
identity/uniform loading under the SAME trace-matched graph mechanism. It is
a diagnostic adaptation, not a paper-exact method or a replacement localizer.
"""
from copy import deepcopy
import hashlib
import numpy as np
from .answer_localization_v2 import moment_plan,prepare_local,mixture_readout
from .fusion_native_conditioning import ARMS as PARENT_ARMS,CONDITIONS,FAMILIES
from .fusion_context_bank import REP
from .joint_lsml import regularized_joint_map_weights
from .adapted_dufs import adapted_dufs_soft_gates
from .laplacian_upcr import build_graph_from_features,permute_graph,graph_diagnostics
from .short_cycle_localization import scaled_oriented_weight
from .window_localization import windows_to_tokens

GRAPHS=('graph010','graph_perm')
NEW_ARMS=tuple(f'{family}__cond{k}_{g}' for family in FAMILIES for k in CONDITIONS for g in GRAPHS)+tuple(
    f'{family}__equal_{g}' for family in FAMILIES for g in GRAPHS)
ARMS=PARENT_ARMS+NEW_ARMS


def graph_head(z,indices,anchor,covariance,loading,graph,condition,lam=.1):
    if condition not in (*CONDITIONS,1000) or lam not in (0.,.1):raise ValueError('UNREGISTERED_GRAPH_CONDITION')
    fit=z[indices]
    w,info=regularized_joint_map_weights(fit,covariance,loading,mode='liu',lam=lam,graph=graph,graph_k=7,target_condition=condition)
    w,boundary=scaled_oriented_weight(w,fit,anchor);risk=-z@w
    if not np.isfinite(risk).all():raise ValueError('NONFINITE_RISK')
    return risk,{'inverse':info,**boundary,'standardized_weights':w}


def compose_routes(arrays,methods,routing):
    out,meta={},{}
    for family in ('single','dual'):
        route=routing['routes'][family]
        if route not in ('moment_joint','context_joint','moment_iu'):raise ValueError('UNKNOWN_ORIGINAL_ROUTE')
        bank='context' if route=='context_joint' else 'moment'
        for suffix in [f'cond{k}_{g}' for k in CONDITIONS for g in GRAPHS]+['equal_'+g for g in GRAPHS]:
            arm=family+'__'+suffix
            source=bank+'__'+suffix if suffix.startswith('equal_') or route!='moment_iu' else 'moment__iu'
            d=deepcopy(methods[source]);d.update(source_arm=source,route=route)
            if d['valid']:
                for ending in ('window','risk'):out[arm+'__'+ending]=np.asarray(arrays[source+'__'+ending]).copy()
            elif d['decision_valid'] or d['fixed_iu_valid']:raise ValueError('INVALID_FIT_HAS_DECISION')
            meta[arm]=d
    return out,meta


def score_graph_conditioning(parent_arrays,parent_metadata,original_metadata,token_count):
    arrays={k:np.asarray(v).copy() for k,v in parent_arrays.items()};methods=deepcopy(parent_metadata['methods'])
    assert set(methods)==set(PARENT_ARMS)
    routing=deepcopy(original_metadata['routing']);assert parent_metadata['routing']==routing
    plan=moment_plan(token_count,8)
    for key,value in [('window_starts',plan.starts),('window_ends',plan.ends),('fit_indices',plan.fit_indices)]:np.testing.assert_array_equal(arrays[key],value)
    ss,ee=arrays['step_starts'],arrays['step_ends']
    if ss.ndim!=1 or ss.shape!=ee.shape or not len(ss) or np.any(ss<0) or np.any(ee>token_count) or np.any(ee<=ss):raise ValueError('INVALID_OFFICIAL_SPANS')
    reference=methods['moment__iu'];diagnostics={'banks':{},'labels_accessed':False,'joint_refits':0}

    def readout(risk):
        token=windows_to_tokens(plan,risk);steps=np.array([token[a:b].max() for a,b in zip(ss,ee)])
        detail={'valid':True,'decision_valid':False,'fixed_iu_valid':bool(reference['valid'] and reference['decision_valid']),
            'prediction':None,'fixed_iu_prediction':None,'peak':int(np.argmax(steps))}
        try:
            gate=mixture_readout(risk[plan.fit_indices],steps)
            detail.update(gate=gate,decision_valid=True,prediction=detail['peak'] if gate['prediction']!=-1 else -1)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:detail['readout_error']=str(exc)
        if detail['fixed_iu_valid']:detail['fixed_iu_prediction']=detail['peak'] if reference['prediction']!=-1 else -1
        return steps,detail

    def admit(arm,risk,detail):
        steps,decision=readout(risk);methods[arm]={**detail,**decision,'source_arm':arm,'route':'pure_fixed_bank'}
        arrays[arm+'__window']=risk;arrays[arm+'__risk']=steps

    for bank in ('moment','context'):
        original=original_metadata['diagnostics']['banks'][bank];sh=original['shared']
        z,anchor,normal=prepare_local(arrays[bank+'__features'],original['names'],plan.fit_indices);fit=z[plan.fit_indices]
        for key in ('mean','sd','feature_signs'):np.testing.assert_allclose(normal[key],sh[key],atol=1e-12,rtol=1e-12)
        assert normal['active_features']==sh['active_features']
        old_gate=sh.get('gates')
        if old_gate:gates=np.asarray(old_gate['values']);gate_meta={'source':'exact_original_same_answer_gates'}
        else:
            gates,info=adapted_dufs_soft_gates(fit.T,seeds=(0,1,2),epochs=120)
            gate_meta={'source':'same_recipe_computed_for_equal_control','diagnostics':info}
        graph=build_graph_from_features(fit.T,gates=gates,k=7)
        namespace='localization-cached-v1-20260907'
        identity=namespace+'/'+original_metadata['cell']+'/'+original_metadata['row_id']+'/'+REP
        seed=int(hashlib.sha256(identity.encode()).hexdigest()[:8],16);permutation=np.random.default_rng(seed).permutation(len(fit))
        if old_gate:
            assert seed==old_gate['seed'];np.testing.assert_array_equal(permutation,old_gate['permutation'])
        graphs={'graph010':graph,'graph_perm':permute_graph(graph,permutation)}
        bd={'normalization':normal,'gates':{'values':gates,**gate_meta},'seed':seed,'permutation':permutation,
            'graph':graph_diagnostics(graph),'original_joint_valid':bool(methods[bank+'__joint0']['valid'])}
        diagnostics['banks'][bank]=bd;arrays[bank+'__graph_gates']=gates
        # This identity/uniform control is exact equal fusion when lambda=0.
        p=fit.shape[1];identity_cov=np.eye(p);uniform=np.ones(p)/p
        equal0,eqd=graph_head(z,plan.fit_indices,anchor,identity_cov,uniform,graph,1000,0.)
        np.testing.assert_allclose(equal0,arrays[bank+'__equal__window'],atol=1e-10,rtol=1e-10)
        bd['equal_zero_replay_maximum_difference']=float(np.max(np.abs(equal0-arrays[bank+'__equal__window'])))
        for kind,gr in graphs.items():
            risk,detail=graph_head(z,plan.fit_indices,anchor,identity_cov,uniform,gr,1000)
            # I + .1*R has condition <= 1 + .1*P when R is PSD and trace P.
            # Thus caps 30/100/300/1000 give the same control (P<=27).
            assert p<=27 and detail['inverse']['condition_after']<=1+.1*p+1e-8
            admit(bank+'__equal_'+kind,risk,{**detail,'control':'identity_covariance_uniform_loading_same_graph'})
        if not methods[bank+'__joint0']['valid']:
            for condition in CONDITIONS:
                for kind in GRAPHS:
                    arm=f'{bank}__cond{condition}_{kind}'
                    methods[arm]={'valid':False,'decision_valid':False,'fixed_iu_valid':False,'prediction':None,'fixed_iu_prediction':None,
                        'reason':'ORIGINAL_JOINT_INVALID','source_arm':arm,'route':'pure_fixed_bank'}
            continue
        cv=arrays['original_'+bank+'__covariance'];v=arrays['original_'+bank+'__v']
        for kind,gr in graphs.items():
            baseline,base_detail=graph_head(z,plan.fit_indices,anchor,cv,v,gr,1000)
            old=methods[bank+'__'+kind]
            np.testing.assert_allclose(base_detail['standardized_weights'],old['standardized_weights'],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(baseline,arrays[bank+'__'+kind+'__window'],atol=1e-10,rtol=1e-10)
            steps,bdc=readout(baseline)
            np.testing.assert_allclose(steps,arrays[bank+'__'+kind+'__risk'],atol=1e-10,rtol=1e-10)
            for key in ('decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):assert bdc[key]==old[key]
            if old['decision_valid']:np.testing.assert_allclose(bdc['gate']['bic'],old['gate']['bic'],atol=1e-9,rtol=1e-10)
            bd[kind+'_condition1000_replay_maximum_difference']=float(np.max(np.abs(baseline-arrays[bank+'__'+kind+'__window'])))
            for condition in CONDITIONS:
                risk,detail=graph_head(z,plan.fit_indices,anchor,cv,v,gr,condition)
                admit(f'{bank}__cond{condition}_{kind}',risk,detail)
    extra,meta=compose_routes(arrays,methods,routing);arrays.update(extra);methods.update(meta);assert set(methods)==set(ARMS)
    return arrays,methods,routing,diagnostics
