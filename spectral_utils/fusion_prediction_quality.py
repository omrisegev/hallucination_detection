"""Add same-answer prediction views to IU/Joint, retaining fixed bank routing.

The prediction view is an input component. Fusion and the existing step/no-error
readout still produce the method's output. No target enters this module.
"""
from copy import deepcopy
import hashlib
import numpy as np
from .answer_localization_v2 import prepare_local, moment_plan, mixture_readout, JOINT_SEED
from .joint_lsml import covariance_matrix, discover_loao_consensus_groups, fit_joint_lsml
from .upcr import upcr_fit
from .laplacian_upcr import IU_FIT_DEFAULTS, build_graph_from_features, permute_graph, graph_diagnostics
from .adapted_dufs import adapted_dufs_soft_gates
from .short_cycle_localization import scaled_oriented_weight
from .fusion_graph_conditioning import graph_head, ARMS as PARENT_ARMS
from .window_localization import windows_to_tokens

KINDS = ('ar1', 'last', 'ema32')
CORES = ('equal', 'iu', 'joint0', 'graph010', 'graph_perm', 'equal_graph010', 'equal_graph_perm')
JOINT_CORES = ('joint0', 'graph010', 'graph_perm')
NEW_ARMS = tuple(f'{kind}__{core}' for kind in KINDS for core in CORES)
ARMS = PARENT_ARMS + NEW_ARMS
CONDITION = 100


def fixed_bank(original_routing):
    route = original_routing['routes']['dual']
    if route not in ('moment_joint', 'context_joint', 'moment_iu'):
        raise ValueError('UNKNOWN_ORIGINAL_ROUTE')
    return 'context' if route == 'context_joint' else 'moment'


def fit_bank(values, names, fit_indices, identity):
    z, anchor, shared = prepare_local(values, names, fit_indices)
    fit = z[fit_indices]; p = fit.shape[1]
    arrays = {'z': z}; risks = {}; methods = {}

    def admit(core, w, detail):
        w, boundary = scaled_oriented_weight(w, fit, anchor)
        risk = -z @ w
        if not np.isfinite(risk).all(): raise ValueError('NONFINITE_RISK')
        methods[core] = {**detail, **boundary, 'valid': True, 'standardized_weights': w}
        risks[core] = risk

    admit('equal', np.ones(p)/p, {})
    try:
        iu = upcr_fit(fit.T, **dict(IU_FIT_DEFAULTS))
        if iu.abstained: raise ValueError('IU_ABSTAINED')
        admit('iu', iu.w, {'g2_hat': iu.g2_hat, 'abstained': iu.abstained})
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        methods['iu'] = {'valid': False, 'reason': str(exc)}
    joint_valid = False
    try:
        blocks = np.minimum(3, np.arange(len(fit))*4//len(fit))
        grouping = discover_loao_consensus_groups(fit, blocks, k_range=(3,4,6,8), seed=JOINT_SEED,
            minimum_group_size=3, minimum_held_admissible_fraction=.95, use_minimum_ari_tiebreak=True)
        shared['grouping'] = {k: grouping.get(k) for k in ('status','K','group_sizes','median_ari','candidates')}
        if grouping['status'] != 'SELECTED': raise ValueError('BLOCKED_NO_ADMISSIBLE_PARTITION')
        joint = fit_joint_lsml(covariance_matrix(fit), grouping['labels'], anchor_index=anchor,
                               seed=JOINT_SEED, starts=5, max_sweeps=5000)
        jac = joint.jacobian_audit
        joint_valid = bool(joint.converged and joint.multistart_audit['status']=='PASS'
            and jac['full_global_rank'] and np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8)
        shared['joint'] = {'groups': grouping['labels'], 'converged': joint.converged,
            'multistart': joint.multistart_audit, 'jacobian': jac, 'diagonal': joint.diagonal_audit,
            'relative_offdiag_misfit': joint.relative_offdiag_misfit}
        for k,v in [('covariance',joint.model_covariance),('v',joint.global_loading),('u',joint.group_loading)]: arrays[k]=v
        if not joint_valid: raise ValueError('JOINT_FIT_GUARD_FAILED')
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        shared['joint_failure'] = str(exc)
    shared['joint_valid'] = joint_valid
    # The same feature graph is constructed for Joint and simple controls,
    # even when the Joint covariance fit fails.
    graphs = {}
    try:
        gates, detail = adapted_dufs_soft_gates(fit.T, seeds=(0,1,2), epochs=120)
        graph = build_graph_from_features(fit.T, gates=gates, k=7)
        seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)
        permutation = np.random.default_rng(seed).permutation(len(fit))
        graphs = {'graph010':graph, 'graph_perm':permute_graph(graph,permutation)}
        shared['gates'] = {'values':gates, 'diagnostics':detail, 'graph':graph_diagnostics(graph),
                           'permutation':permutation, 'seed':seed}
        arrays['gates'] = gates
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        shared['graph_failure'] = str(exc)
    for core in (*JOINT_CORES, 'equal_graph010','equal_graph_perm'):
        is_joint = core in JOINT_CORES
        if is_joint and not joint_valid:
            methods[core]={'valid':False,'reason':shared['joint_failure']}; continue
        graph_kind = core.removeprefix('equal_')
        if core!='joint0' and graph_kind not in graphs:
            methods[core]={'valid':False,'reason':shared['graph_failure']}; continue
        try:
            cv, loading = (arrays['covariance'],arrays['v']) if is_joint else (np.eye(p),np.ones(p)/p)
            # Zero graph does not require a successfully constructed graph.
            risk, detail = graph_head(z,fit_indices,anchor,cv,loading,graphs.get(graph_kind),CONDITION,
                                     lam=0. if core=='joint0' else .1)
            methods[core]={**detail,'valid':True}; risks[core]=risk
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            methods[core]={'valid':False,'reason':str(exc)}
    return arrays, risks, methods, shared


def score_augmented(audit_arrays, audit_meta, original_meta, token_count, identity):
    """Only failed Joint FITS fall back; numerical/readout failures stay failures."""
    bank = fixed_bank(original_meta['routing']); plan = moment_plan(token_count,8)
    for key,value in [('window_starts',plan.starts),('window_ends',plan.ends),('fit_indices',plan.fit_indices)]:
        np.testing.assert_array_equal(audit_arrays[key],value)
    ss = np.asarray(original_meta['official_step_starts'],int)
    ee = np.asarray(original_meta['official_step_ends'],int)
    if ss.ndim!=1 or ss.shape!=ee.shape or not len(ss) or np.any(ss<0) or np.any(ee>token_count) or np.any(ee<=ss):
        raise ValueError('INVALID_OFFICIAL_SPANS')
    out={'window_starts':plan.starts,'window_ends':plan.ends,'fit_indices':plan.fit_indices,'step_starts':ss,'step_ends':ee}
    methods={}; diagnostics={'bank':bank,'variants':{},'labels_accessed':False,
        'original_joint_valid':bool(original_meta['methods'][bank+'__joint0']['valid'])}
    reference = original_meta['methods']['moment__iu']

    def readout(risk, detail):
        d=deepcopy(detail); d.update(decision_valid=False,fixed_iu_valid=False,prediction=None,fixed_iu_prediction=None)
        if not d['valid']: return None,d
        token=windows_to_tokens(plan,risk); step=np.array([token[a:b].max() for a,b in zip(ss,ee)])
        if not np.isfinite(step).all(): raise ValueError('NONFINITE_STEP_RISK')
        d['peak']=int(np.argmax(step))
        try:
            gate=mixture_readout(risk[plan.fit_indices],step)
            d.update(gate=gate,decision_valid=True,prediction=d['peak'] if gate['prediction']!=-1 else -1)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc: d['readout_error']=str(exc)
        d['fixed_iu_valid']=bool(reference['valid'] and reference['decision_valid'])
        if d['fixed_iu_valid']:d['fixed_iu_prediction']=d['peak'] if reference['prediction']!=-1 else -1
        return step,d

    for kind in KINDS:
        values=audit_arrays[bank+'__'+kind+'__features']; names=audit_meta['diagnostics']['banks'][bank+'__'+kind]['names']
        out[kind+'__features']=np.asarray(values).copy()
        try:
            fitted,risks,raw_methods,shared=fit_bank(values,names,plan.fit_indices,identity)
            for k,v in fitted.items():out[kind+'__'+k]=v
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            fitted={};risks={};raw_methods={c:{'valid':False,'reason':str(exc)} for c in CORES}
            shared={'joint_valid':False,'joint_failure':str(exc),'preparation_failure':str(exc)}
        diagnostics['variants'][kind]=shared
        # Read each actual map once, before composing the declared fallback.
        prepared={}
        for core in CORES:
            d=raw_methods[core]; step,detail=readout(risks.get(core),d)
            prepared[core]=detail
            if d['valid']:
                out[kind+'__native_'+core+'__window']=risks[core]
                out[kind+'__native_'+core+'__risk']=step
        for core in CORES:
            fallback=core in JOINT_CORES and not shared['joint_valid']
            source='iu' if fallback else core
            arm=kind+'__'+core; d=deepcopy(prepared[source])
            d.update(source_arm=kind+'__native_'+source,bank=bank,joint_fit_valid=shared['joint_valid'],
                     fallback_to_augmented_iu=fallback,route='fixed_original_bank')
            if fallback:d['joint_failure']=shared['joint_failure']
            methods[arm]=d
            if d['valid']:
                for ending in ('window','risk'):out[arm+'__'+ending]=out[kind+'__native_'+source+'__'+ending].copy()
    assert set(methods)==set(NEW_ARMS)
    return out,methods,diagnostics
