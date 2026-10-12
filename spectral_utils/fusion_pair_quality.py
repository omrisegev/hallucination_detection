"""Checked pair-group Joint localization with exact frozen fusion anchors."""
from copy import deepcopy
import hashlib
import numpy as np
from .answer_localization_v2 import (JOINT_SEED, MIN_WINDOWS, moment_plan, moment_matrix,
                                   prepare_local, mixture_readout)
from .fusion_context_bank import context_matrix, REP
from .fusion_replication import ARMS as PARENT_ARMS
from .fusion_explicit_fallback import choose_route
from .joint_lsml import covariance_matrix, discover_loao_consensus_groups, regularized_joint_map_weights
from .joint_pair_jacobian import fit_joint_pairs_checked
from .adapted_dufs import adapted_dufs_soft_gates
from .laplacian_upcr import build_graph_from_features, graph_diagnostics, permute_graph
from .short_cycle_localization import scaled_oriented_weight
from .window_localization import windows_to_tokens

CORES=('joint0','graph010','graph_perm')
PURE_ARMS=tuple('pair_'+bank+'__'+core for bank in ('moment','context') for core in CORES)
ROUTED_ARMS=tuple('pair_'+policy+'__'+core for policy in ('single','dual') for core in CORES)+('pair_dual__equal','pair_dual__iu')
ARMS=PARENT_ARMS+PURE_ARMS+ROUTED_ARMS


def compose_pair_routes(arrays,methods):
    """Use fit eligibility only; preserve the selected source's readout failure."""
    flags={}
    for bank in ('moment','context'):
        valid=[methods['pair_'+bank+'__'+core]['valid'] for core in CORES]
        if len(set(valid))!=1:raise ValueError('JOINT_FAMILY_ELIGIBILITY_MISMATCH')
        flags[bank]=bool(valid[0])
    routes={policy:choose_route(flags['moment'],flags['context'],policy) for policy in ('single','dual')}
    out,meta={},{}
    for arm in ROUTED_ARMS:
        policy,core=arm.removeprefix('pair_').split('__');route=routes[policy]
        bank='context' if route=='context_joint' else 'moment'
        if core in ('equal','iu'):source=bank+'__'+core
        elif route=='moment_iu':source='moment__iu'
        else:source='pair_'+bank+'__'+core
        detail=deepcopy(methods[source]);detail.update(source_arm=source,route=route)
        if detail['valid']:
            for suffix in ('window','risk'):out[arm+'__'+suffix]=np.asarray(arrays[source+'__'+suffix]).copy()
        elif detail['decision_valid'] or detail['fixed_iu_valid']:raise ValueError('INVALID_FIT_HAS_VALID_DECISION')
        meta[arm]=detail
    return out,meta,{'eligibility':flags,'routes':routes}


def score_pair_banks(raw,starts,ends,identity,parent_arrays,parent_metadata):
    """Refit the checked native Joint extension; reuse unchanged parent scores."""
    raw=np.asarray(raw,float);starts=np.asarray(starts,int);ends=np.asarray(ends,int)
    if starts.ndim!=1 or starts.shape!=ends.shape or not len(starts) or np.any(starts<0) or np.any(ends>len(raw)) or np.any(ends<=starts):
        raise ValueError('INVALID_OFFICIAL_SPANS')
    plan=moment_plan(len(raw),8)
    if len(plan.fit_indices)<MIN_WINDOWS:raise ValueError('TOO_FEW_FIT_WINDOWS')
    arrays={k:np.asarray(v).copy() for k,v in parent_arrays.items()}
    methods=deepcopy(parent_metadata['methods']);assert set(methods)==set(PARENT_ARMS)
    for key,value in [('window_starts',plan.starts),('window_ends',plan.ends),('fit_indices',plan.fit_indices),('step_starts',starts),('step_ends',ends)]:
        np.testing.assert_array_equal(arrays[key],value)
    diagnostics={'banks':{},'labels_accessed':False}
    reference=methods['moment__iu']

    def reject(bank,reason,detail=None):
        for core in CORES:
            arm='pair_'+bank+'__'+core
            methods[arm]={'valid':False,'decision_valid':False,'fixed_iu_valid':False,
                'prediction':None,'fixed_iu_prediction':None,'reason':reason,'failure_detail':detail or {},
                'source_arm':arm,'route':'pure_pair_joint'}

    def admit(bank,core,weight,z,fit,anchor,detail):
        arm='pair_'+bank+'__'+core
        weight,boundary=scaled_oriented_weight(weight,fit,anchor);risk=-z@weight
        if not np.isfinite(risk).all():raise ValueError('NONFINITE_RISK')
        token=windows_to_tokens(plan,risk);steps=np.array([token[a:b].max() for a,b in zip(starts,ends)])
        meta={**detail,**boundary,'valid':True,'status':'OK','standardized_weights':weight,
            'decision_valid':False,'fixed_iu_valid':False,'prediction':None,'fixed_iu_prediction':None,
            'peak':int(np.argmax(steps)),'source_arm':arm,'route':'pure_pair_joint'}
        try:
            gate=mixture_readout(risk[plan.fit_indices],steps)
            meta.update(gate=gate,decision_valid=True,prediction=meta['peak'] if gate['prediction']!=-1 else -1)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:meta['readout_error']=str(exc)
        meta['fixed_iu_valid']=bool(reference['valid'] and reference['decision_valid'])
        if meta['fixed_iu_valid']:meta['fixed_iu_prediction']=meta['peak'] if reference['prediction']!=-1 else -1
        arrays[arm+'__window']=risk;arrays[arm+'__risk']=steps;methods[arm]=meta

    for bank,builder in [('moment',moment_matrix),('context',context_matrix)]:
        values,names=builder(raw,plan);np.testing.assert_allclose(values,parent_arrays[bank+'__features'],atol=1e-12,rtol=1e-12)
        z,anchor,normal=prepare_local(values,names,plan.fit_indices);fit=z[plan.fit_indices]
        grouping=discover_loao_consensus_groups(fit,np.minimum(3,np.arange(len(fit))*4//len(fit)),
            k_range=(3,4,6,8),seed=JOINT_SEED,minimum_group_size=2,
            minimum_held_admissible_fraction=.95,use_minimum_ari_tiebreak=True)
        bd={'normalization':normal,'grouping':{k:grouping.get(k) for k in ('status','K','group_sizes','labels','median_ari')}}
        diagnostics['banks'][bank]=bd
        if grouping['status']!='SELECTED':reject(bank,'BLOCKED_NO_ADMISSIBLE_PARTITION');continue
        try:wrapper=fit_joint_pairs_checked(covariance_matrix(fit),grouping['labels'],anchor_index=anchor)
        except ValueError as exc:
            if not str(exc).startswith('PAIR_'):raise
            bd.update(status=str(exc),failure_detail=getattr(exc,'detail',{}));reject(bank,str(exc),getattr(exc,'detail',{}));continue
        joint=wrapper.joint;jac=joint.jacobian_audit
        valid=bool(joint.converged and joint.multistart_audit['status']=='PASS' and jac['full_global_rank']
            and np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8)
        bd.update(status='OK' if valid else 'FIT_DIAGNOSTIC_ONLY',pairs=wrapper.pair_audit,
            native_map_audit=wrapper.native_map_audit,converged=joint.converged,multistart=joint.multistart_audit,
            jacobian=jac,diagonal=joint.diagonal_audit,relative_offdiag_misfit=joint.relative_offdiag_misfit)
        arrays['pair_'+bank+'__covariance']=joint.model_covariance;arrays['pair_'+bank+'__v']=joint.global_loading
        arrays['pair_'+bank+'__u']=joint.group_loading
        if not valid:reject(bank,'FIT_DIAGNOSTIC_ONLY');continue
        old_gates=parent_metadata['diagnostics']['banks'][bank]['shared'].get('gates')
        if old_gates:
            gates=np.asarray(old_gates['values']);gate_detail={'source':'exact_same_answer_parent_gates','diagnostics':old_gates['diagnostics']}
        else:
            gates,gate_detail=adapted_dufs_soft_gates(fit.T,seeds=(0,1,2),epochs=120)
            gate_detail={'source':'same_fixed_recipe_newly_computed','diagnostics':gate_detail}
        graph=build_graph_from_features(fit.T,gates=gates,k=7)
        seed=int(hashlib.sha256((identity+'/'+REP).encode()).hexdigest()[:8],16)
        permutation=np.random.default_rng(seed).permutation(len(fit))
        bd['gates']={'values':gates,'diagnostics':gate_detail,'seed':seed,'permutation':permutation,'graph':graph_diagnostics(graph)}
        for core,lam,gr in [('joint0',0.,graph),('graph010',.1,graph),('graph_perm',.1,permute_graph(graph,permutation))]:
            w,info=regularized_joint_map_weights(fit,joint.model_covariance,joint.global_loading,
                mode='liu',lam=lam,gates=gates,graph=gr,graph_k=7,target_condition=1000.)
            admit(bank,core,w,z,fit,anchor,{'inverse':info})
    new_arrays,new_methods,routes=compose_pair_routes(arrays,methods)
    arrays.update(new_arrays);methods.update(new_methods);assert set(methods)==set(ARMS)
    return arrays,methods,routes,diagnostics
