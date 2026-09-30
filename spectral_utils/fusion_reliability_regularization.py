"""Answer-local sensitivity regularization of existing fusion weight heads.

Block perturbations define a robustness diagnostic, not certified noise or
LOCA bursts. Joint uses its native covariance head; IU/equal use an explicitly
labelled correction of their fitted weights. No correctness targets enter.
"""
from __future__ import annotations
import time
import numpy as np
from .answer_localization_v2 import (moment_plan, moment_matrix, prepare_local,
    mixture_readout, JOINT_SEED, json_safe)
from .fusion_window_sampling import NAMES, REP, perturb_windows, seed_for
from .adapted_dufs import adapted_dufs_soft_gates
from .joint_lsml import covariance_matrix, fit_joint_lsml, regularized_joint_map_weights
from .dependency_fusion import regularized_covariance_weights
from .laplacian_upcr import build_graph_from_features, symmetric_normalized_laplacian
from .short_cycle_localization import scaled_oriented_weight
from .window_localization import windows_to_tokens

CORES = ('equal', 'iu', 'joint')
FAMILIES = ('isotropic', 'dufs_graph', 'block_diag', 'block_diag_permuted', 'block_full')
LAMBDAS = (0., .1, 1., 10.)
PARENTS = {'equal_parent': REP+'__equal', 'iu_parent': REP+'__iu',
    'joint_parent': REP+'__joint_lambda0', 'joint_graph010_parent': REP+'__joint_graph010',
    'joint_graph_permuted_parent': REP+'__joint_graph_permuted', 'entropy_parent': 'entropy_mean_w8'}
ARMS = tuple(PARENTS)+tuple(c+'__'+f for c in CORES for f in FAMILIES)+('joint_graph_fixed1', 'joint_graph_fixed10')


def second_moment(deltas):
    values = np.asarray(deltas, float)
    if values.ndim < 2 or not np.isfinite(values).all(): raise ValueError('INVALID_PERTURBATIONS')
    flat = values.reshape(-1, values.shape[-1])
    if len(flat) == 0: raise ValueError('EMPTY_PERTURBATIONS')
    return (flat.T@flat)/len(flat)


def trace_match(penalty, matrix):
    penalty = np.asarray(penalty, float); total = float(np.trace(penalty))
    if penalty.shape != matrix.shape or not np.isfinite(penalty).all(): raise ValueError('INVALID_PENALTY')
    if total <= 1e-12: raise ValueError('DEGENERATE_PENALTY')
    return (penalty+penalty.T)*(.5*np.trace(matrix)/total)


def choose_lambda(losses):
    losses = np.asarray(losses, float)
    good = np.isfinite(losses)
    if not np.any(good) or not good[0]: raise ValueError('BASELINE_OR_GRID_INVALID')
    tolerance = .01*max(float(losses[0]), 1e-12)
    return int(np.flatnonzero(good & (losses <= np.min(losses[good])+tolerance))[0])


def penalty_grid(matrix, rhs, penalty, fit, anchor, validation):
    scaled = trace_match(penalty, matrix)
    weights, losses, records = [], [], []
    for lam in LAMBDAS:
        try:
            w, inverse = regularized_covariance_weights(matrix+lam*scaled, rhs, target_condition=1000.)
            w, boundary = scaled_oriented_weight(w, fit, anchor)
            loss = float(w@validation@w)
            if not np.isfinite(loss) or loss < -1e-10: raise ValueError('INVALID_STABILITY_LOSS')
            weights.append(w); losses.append(max(0., loss))
            records.append({'valid': True, 'inverse': inverse, 'boundary': boundary})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            weights.append(np.full(len(rhs), np.nan)); losses.append(np.inf)
            records.append({'valid': False, 'reason': str(exc)})
    selected = choose_lambda(losses)
    return np.asarray(weights), np.asarray(losses), selected, records, scaled


def perturbation_moments(raw, plan, original_values, normalization, identity):
    columns = [list(NAMES).index(s) for s in normalization['active_features']]
    fit_indices = plan.fit_indices
    original = original_values[fit_indices][:, columns]
    moments = []
    for replicate in range(24):
        changed = perturb_windows(raw, identity+'/reliability-v1', replicate)
        features, _ = moment_matrix(changed, plan)
        delta = (features[fit_indices][:, columns]-original)/normalization['sd']
        delta *= normalization['feature_signs']
        moments.append(second_moment(delta))
    return np.mean(moments[:16], axis=0), np.mean(moments[16:], axis=0), np.asarray(moments)


def score_reliability(parent_arrays, parent_meta, raw, identity):
    started = time.monotonic(); plan = moment_plan(len(raw), 8); indices = plan.fit_indices
    values = parent_arrays[REP+'__features']
    z, anchor, normalization = prepare_local(values, list(NAMES), indices)
    fit = z[indices]; p = fit.shape[1]
    original_shared = parent_meta['report']['representations'][REP]['shared']
    np.testing.assert_allclose(normalization['mean'], original_shared['mean'], atol=1e-12)
    arrays = {'step_starts': parent_arrays['step_starts'], 'step_ends': parent_arrays['step_ends'],
              'normalization_fit': fit}
    methods = {}; diagnostics = {'normalization': normalization, 'families': {}, 'replay_errors': {}}

    def save_risk(arm, risk, parent_core, detail, replay=False):
        parent = parent_meta['report']['methods'][parent_core]
        token = windows_to_tokens(plan, risk)
        steps = np.array([np.max(token[a:b]) for a,b in zip(arrays['step_starts'], arrays['step_ends'])])
        arrays[arm+'__window'], arrays[arm+'__risk'] = risk, steps
        try:
            gate = dict(parent['readout']) if replay else mixture_readout(risk[indices], steps)
            detail.update(valid=True, parent_replay=replay, gate_readout=gate,
                prediction=int(np.argmax(steps)) if gate['prediction'] != -1 else -1,
                fixed_parent_gate_prediction=(int(np.argmax(steps)) if parent['readout']['prediction'] != -1 else -1)
                    if parent.get('valid') and parent.get('readout_valid') else None)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            detail.update(valid=False, reason=str(exc))
        methods[arm] = detail

    for arm, parent_core in PARENTS.items():
        old = parent_meta['report']['methods'][parent_core]
        if old.get('valid') and old.get('readout_valid'):
            save_risk(arm, parent_arrays[parent_core+'__window'], parent_core, {'source': parent_core}, replay=True)
        else:
            methods[arm] = {'valid': False, 'reason': 'PARENT_UNAVAILABLE', 'original': old}
    q_train, q_validation, per_replicate = perturbation_moments(raw, plan, values, normalization, identity)
    arrays.update(Q_train=q_train, Q_validation=q_validation, Q_per_replicate=per_replicate)
    permutation = np.random.default_rng(seed_for(identity+'/reliability-diagonal-permutation')).permutation(p)
    diagonal = np.diag(q_train)
    penalties = {'isotropic': np.eye(p), 'block_diag': np.diag(diagonal),
                 'block_diag_permuted': np.diag(diagonal[permutation]), 'block_full': q_train}
    diagnostics['diagonal_permutation'] = permutation
    try:
        if 'gates' in original_shared:
            gates = np.asarray(original_shared['gates']['values'])
            diagnostics['gates_source'] = 'exact_parent'
        else:
            gates, gate_detail = adapted_dufs_soft_gates(fit.T, seeds=(0,1,2), epochs=120)
            diagnostics['gates_source'] = 'same_answer_same_parent_recipe'
            diagnostics['gate_fit'] = gate_detail
        graph = build_graph_from_features(fit.T, gates=gates, k=7)
        laplacian = symmetric_normalized_laplacian(graph)
        roughness = np.asarray(fit.T@(laplacian@fit))/len(fit)
        penalties['dufs_graph'] = (roughness+roughness.T)/2
        arrays['feature_gates'] = gates
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        diagnostics['graph_error'] = str(exc)
    matrices = {'equal': np.eye(p), 'iu': np.eye(p)}
    targets = {}; parent_for = {'equal': PARENTS['equal_parent'], 'iu': PARENTS['iu_parent'], 'joint': PARENTS['joint_parent']}
    for core in ('equal', 'iu'):
        old = parent_meta['report']['methods'][parent_for[core]]
        if old.get('valid'): targets[core] = np.asarray(old['standardized_weights'])
    if methods['joint_parent']['valid']:
        joint = fit_joint_lsml(covariance_matrix(fit), original_shared['joint']['groups'],
                              anchor_index=anchor, seed=JOINT_SEED, starts=5, max_sweeps=5000)
        jac = joint.jacobian_audit
        valid = bool(joint.converged and joint.multistart_audit['status']=='PASS' and jac['full_global_rank']
                     and np.isfinite(jac['condition_number']) and jac['condition_number'] <= 1e8)
        if not valid: raise RuntimeError('JOINT_REPLAY_VALIDITY_DRIFT')
        matrices['joint'], targets['joint'] = joint.model_covariance, joint.global_loading
        arrays['joint_model_covariance'], arrays['joint_global_loading'] = joint.model_covariance, joint.global_loading
        diagnostics['joint'] = {'converged': joint.converged, 'multistart': joint.multistart_audit,
            'jacobian': jac, 'relative_offdiag_misfit': joint.relative_offdiag_misfit}
        w, _ = regularized_joint_map_weights(fit, joint.model_covariance, joint.global_loading,
                                           mode='liu', lam=.1, gates=gates, graph=graph, target_condition=1000.)
        w, _ = scaled_oriented_weight(w, fit, anchor)
        difference = float(np.max(np.abs(-(z@w)-parent_arrays[PARENTS['joint_graph010_parent']+'__window'])))
        if difference > 1e-10: raise RuntimeError('JOINT_GRAPH_REPLAY_DRIFT')
        diagnostics['replay_errors']['joint_graph010'] = difference
    for family, penalty in penalties.items(): arrays[family+'__penalty'] = penalty
    for core in CORES:
        for family in FAMILIES:
            arm = core+'__'+family
            if core not in targets or family not in penalties:
                methods[arm] = {'valid': False, 'reason': 'CORE_OR_PENALTY_UNAVAILABLE'}; continue
            try:
                weights, losses, selected, grid, scaled = penalty_grid(matrices[core], targets[core],
                    penalties[family], fit, anchor, q_validation)
                parent_core = parent_for[core]
                zero_error = float(np.max(np.abs(-(z@weights[0])-parent_arrays[parent_core+'__window'])))
                if zero_error > 1e-10: raise RuntimeError('LAMBDA_ZERO_REPLAY_DRIFT')
                arrays[arm+'__weights_grid'] = weights
                arrays[arm+'__penalty_scaled'] = scaled
                detail = {'selected_index': selected, 'selected_lambda': LAMBDAS[selected],
                    'validation_losses': losses, 'lambda_zero_error': zero_error,
                    'weight_map': 'native_joint_model_inverse' if core=='joint' else 'correction_of_parent_weights',
                    'grid': grid}
                diagnostics['families'][arm] = detail
                save_risk(arm, -(z@weights[selected]), parent_core, dict(detail))
                if core=='joint' and family=='dufs_graph':
                    for index, name in ((2,'joint_graph_fixed1'), (3,'joint_graph_fixed10')):
                        if grid[index]['valid']:
                            save_risk(name, -(z@weights[index]), parent_core,
                                      {'selected_index':index,'selected_lambda':LAMBDAS[index], 'grid_source':arm})
                        else: methods[name] = {'valid':False, 'reason':'GRID_MEMBER_INVALID'}
            except (ValueError, np.linalg.LinAlgError) as exc:
                methods[arm] = {'valid':False, 'reason':str(exc)}
    for arm in ARMS: methods.setdefault(arm, {'valid':False, 'reason':'CORE_OR_GRID_UNAVAILABLE'})
    diagnostics['seconds'] = time.monotonic()-started
    return arrays, json_safe(methods), json_safe(diagnostics)
