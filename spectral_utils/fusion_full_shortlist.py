"""Full pass2a: frozen routed conditioning/graph and trajectory candidates.

Reuse original feature matrices, groups, gates and routes. Recover Joint
factors through a fixed-partition fit and require original-map replay.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import numpy as np
from .answer_localization_v2 import JOINT_SEED, moment_plan, prepare_local
from .joint_lsml import covariance_matrix, fit_joint_lsml
from .fusion_native_conditioning import condition_head
from .fusion_graph_conditioning import graph_head
from .adapted_dufs import adapted_dufs_soft_gates
from .laplacian_upcr import build_graph_from_features, permute_graph
from .fusion_token_gap import apply_readout
from .fusion_trajectory_imm import SOURCES, PAIRS, observation_model, chronology, hold_curve

CORE_ARMS = ('dual__cond100', 'dual__cond100_graph010', 'dual__cond100_graph_perm',
             'dual__equal_graph010', 'dual__equal_graph_perm')
TRAJECTORY_ARMS = tuple(f'traj_{family}__{kind}' for family in PAIRS for kind in
                       (('mean', 'gls', 'hold', 'imm') if family == 'iu_joint_graph' else ('mean', 'gls')))
NEW_ARMS = CORE_ARMS + TRAJECTORY_ARMS


def invalid(reason, source):
    return dict(valid=False, decision_valid=False, fixed_iu_valid=False,
                prediction=None, fixed_iu_prediction=None, reason=reason, source_arm=source)


def score_full_shortlist(original_arrays, original_meta, identity):
    arrays = {}; methods = {}; diagnostics = dict(labels_used=False, new_partition_searches=0,
                                                 reused_dufs=False, joint_refits=0)
    n = original_meta['tokens']
    if n < 64:
        return arrays, {arm:invalid('TOO_FEW_FIT_WINDOWS',arm) for arm in NEW_ARMS}, diagnostics
    plan = moment_plan(n, 8)
    for key, expected in [('window_starts',plan.starts),('window_ends',plan.ends),('fit_indices',plan.fit_indices)]:
        np.testing.assert_array_equal(original_arrays[key],expected)
    ss, ee = original_arrays['step_starts'], original_arrays['step_ends']
    assert ss.shape == ee.shape and len(ss) == original_meta['steps']
    assert np.all(ss >= 0) and np.all(ee <= n) and np.all(ee > ss)
    reference = original_meta['methods']['moment__iu']
    route = original_meta['routing']['routes']['dual']
    assert route in ('moment_joint','context_joint','moment_iu')
    bank = 'context' if route == 'context_joint' else 'moment'
    original = original_meta['diagnostics']['banks'][bank]
    z, anchor, normal = prepare_local(original_arrays[bank+'__features'], original['names'], plan.fit_indices)
    shared = original['shared']; fit = z[plan.fit_indices]
    for key in ('mean','sd','feature_signs'):
        np.testing.assert_allclose(normal[key], shared[key], atol=1e-12, rtol=1e-12)
    assert normal['active_features'] == shared['active_features']
    arrays.update(z=z, fit_indices=plan.fit_indices, window_starts=plan.starts, window_ends=plan.ends,
                  step_starts=ss, step_ends=ee)
    diagnostics.update(bank=bank, route=route, normalization=normal)

    def replay(risk, info, source):
        old = original_meta['methods'][source]
        np.testing.assert_allclose(info['standardized_weights'], old['standardized_weights'], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(risk, original_arrays[source+'__window'], atol=1e-10, rtol=1e-10)
        step, detail = apply_readout(risk, dict(valid=True), plan, ss, ee, reference)
        np.testing.assert_allclose(step, original_arrays[source+'__risk'], atol=1e-10, rtol=1e-10)
        for key in ('decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):
            assert detail[key] == old[key], (original_meta['uid'],source,key)
        if old['decision_valid']:
            np.testing.assert_allclose(detail['gate']['bic'],old['gate']['bic'],atol=1e-9,rtol=1e-10)
        diagnostics.setdefault('replayed_original_maps',[]).append(source)

    def admit(arm, risk, info, source):
        step, detail = apply_readout(risk, dict(info,valid=True), plan, ss, ee, reference)
        detail.update(source_arm=source,route=route)
        methods[arm] = detail
        if detail['valid']:
            arrays[arm+'__window'] = risk; arrays[arm+'__risk'] = step

    # Recover only the original routed Joint partition, never discover a new one.
    joint = None
    if route != 'moment_iu':
        assert original_meta['methods'][bank+'__joint0']['valid']
        groups = np.asarray(shared['joint']['groups']); assert min(Counter(groups).values()) >= 3
        joint = fit_joint_lsml(covariance_matrix(fit), groups, anchor_index=anchor,
                              seed=JOINT_SEED, starts=5, max_sweeps=5000)
        jac = joint.jacobian_audit
        assert joint.converged and joint.multistart_audit['status']=='PASS'
        assert jac['full_global_rank'] and np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8
        np.testing.assert_allclose(joint.relative_offdiag_misfit,shared['joint']['relative_offdiag_misfit'],atol=1e-12,rtol=1e-10)
        diagnostics.update(joint_refits=1,groups=groups,joint_jacobian=jac,joint_multistart=joint.multistart_audit)
        arrays.update(covariance=joint.model_covariance,v=joint.global_loading,u=joint.group_loading)
        old_risk, old_info = condition_head(z,plan.fit_indices,anchor,joint.model_covariance,joint.global_loading,1000)
        replay(old_risk,old_info,bank+'__joint0')
        risk, info = condition_head(z,plan.fit_indices,anchor,joint.model_covariance,joint.global_loading,100)
        admit('dual__cond100',risk,info,bank+'__cond100')
    else:
        for arm in CORE_ARMS[:3]:
            methods[arm] = deepcopy(reference)
            methods[arm].update(source_arm='moment__iu',route=route)
            if reference['valid']:
                for ending in ('window','risk'): arrays[arm+'__'+ending] = original_arrays['moment__iu__'+ending].copy()

    old_gate = shared.get('gates')
    if old_gate:
        gates = np.asarray(old_gate['values']); diagnostics['reused_dufs']=True
    else:
        gates, details = adapted_dufs_soft_gates(fit.T,seeds=(0,1,2),epochs=120)
        diagnostics['computed_equal_control_dufs']=details
    graph = build_graph_from_features(fit.T,gates=gates,k=7)
    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)
    permutation = np.random.default_rng(seed).permutation(len(fit))
    if old_gate:
        assert seed == old_gate['seed']; np.testing.assert_array_equal(permutation,old_gate['permutation'])
    arrays['gates']=gates;diagnostics.update(permutation=permutation,graph_seed=seed)
    graphs = {'graph010':graph,'graph_perm':permute_graph(graph,permutation)}
    p = fit.shape[1]
    equal0, equal_info = graph_head(z,plan.fit_indices,anchor,np.eye(p),np.ones(p)/p,graph,1000,0.)
    np.testing.assert_allclose(equal0,original_arrays[bank+'__equal__window'],atol=1e-10,rtol=1e-10)
    for kind, gr in graphs.items():
        risk, info = graph_head(z,plan.fit_indices,anchor,np.eye(p),np.ones(p)/p,gr,1000)
        assert p <= 27 and info['inverse']['condition_after'] <= 1+.1*p+1e-8
        admit('dual__equal_'+kind,risk,info,bank+'__equal_'+kind)
        if joint is not None:
            baseline, details = graph_head(z,plan.fit_indices,anchor,joint.model_covariance,joint.global_loading,gr,1000)
            replay(baseline,details,bank+'__'+kind)
            risk, info = graph_head(z,plan.fit_indices,anchor,joint.model_covariance,joint.global_loading,gr,100)
            admit('dual__cond100_'+kind,risk,info,bank+'__cond100_'+kind)

    source_methods = {**original_meta['methods'],**methods}
    source_arrays = {**original_arrays,**arrays}
    for family, components in PAIRS.items():
        expected = ('mean','gls','hold','imm') if family=='iu_joint_graph' else ('mean','gls')
        detail = dict(sources=[SOURCES[k] for k in components])
        try:
            if not all(source_methods[s]['valid'] for s in detail['sources']):raise ValueError('REQUIRED_SOURCE_UNAVAILABLE')
            y = np.column_stack([source_arrays[s+'__window'] for s in detail['sources']])
            gls, mean, model = observation_model(y,plan.fit_indices); detail.update(model)
            curves = dict(mean=mean,gls=gls)
            if family=='iu_joint_graph':
                curves['hold'] = hold_curve(gls[plan.fit_indices],plan.fit_indices,len(gls))
                try:
                    level, states, info = chronology(gls,model['effective_variance'],plan.fit_indices,identity,False)
                    curves['imm']=hold_curve(level,plan.fit_indices,len(gls));detail['imm']=info
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:detail['imm_failure']=str(exc)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            curves={};detail['reason']=str(exc)
        diagnostics.setdefault('trajectory',{})[family]=detail
        for mode in expected:
            arm=f'traj_{family}__{mode}'
            if mode in curves:admit(arm,curves[mode],dict(readout=mode),'+'.join(detail['sources']))
            else:methods[arm]=invalid(detail.get(mode+'_failure',detail.get('reason','READOUT_UNAVAILABLE')),arm)
    assert set(methods)==set(NEW_ARMS)
    return arrays,methods,diagnostics
