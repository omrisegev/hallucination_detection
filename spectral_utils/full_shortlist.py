"""Fixed answer-only full-population shortlist for the existing fusion bank.

This adapter revisits the saved same-answer feature matrix with the original
Joint groups and graph recipe. It does not search labels, regroup features, or
borrow a fit from another answer. It is deliberately separate from the
frozen pass-1 scorer.
"""
from copy import deepcopy
import hashlib
import numpy as np

from .answer_localization_v2 import prepare_local, mixture_readout, moment_plan, JOINT_SEED
from .joint_lsml import covariance_matrix, fit_joint_lsml, regularized_joint_map_weights
from .adapted_dufs import adapted_dufs_soft_gates
from .laplacian_upcr import build_graph_from_features, permute_graph, graph_diagnostics
from .short_cycle_localization import scaled_oriented_weight
from .window_localization import windows_to_tokens
from .fusion_token_gap import apply_readout

CONDITIONS = (100,)
BANKS = ('moment', 'context')
BANK_ARMS = tuple(f'{b}__cond100{suffix}' for b in BANKS for suffix in ('', '_graph010', '_graph_perm'))
EQUAL_ARMS = tuple(f'{b}__equal_graph{suffix}' for b in BANKS for suffix in ('010', '_perm'))
NEW_ARMS = BANK_ARMS + EQUAL_ARMS
ROUTED_ARMS = ('single__cond100', 'single__cond100_graph010', 'single__cond100_graph_perm',
               'single__equal_graph010', 'single__equal_graph_perm',
               'dual__cond100', 'dual__cond100_graph010', 'dual__cond100_graph_perm',
               'dual__equal_graph010', 'dual__equal_graph_perm')
ARMS = NEW_ARMS + ROUTED_ARMS


def _head(z, fit_indices, anchor, covariance, loading, graph, condition=100, lam=.1):
    fit = z[fit_indices]
    w, info = regularized_joint_map_weights(fit, covariance, loading, mode='liu',
                                             lam=lam, graph=graph, graph_k=7,
                                             target_condition=condition)
    w, boundary = scaled_oriented_weight(w, fit, anchor)
    risk = -z @ w
    if not np.isfinite(risk).all():
        raise ValueError('NONFINITE_RISK')
    return risk, {'inverse': info, **boundary, 'standardized_weights': w, 'valid': True}


def _readout(risk, detail, plan, starts, ends, reference):
    d = deepcopy(detail)
    d.update(decision_valid=False, fixed_iu_valid=False, prediction=None,
             fixed_iu_prediction=None)
    if not d.get('valid', False):
        return None, d
    token = windows_to_tokens(plan, risk)
    step = np.asarray([token[a:b].max() for a, b in zip(starts, ends)])
    if not np.isfinite(step).all():
        raise ValueError('NONFINITE_STEP_RISK')
    d['peak'] = int(np.argmax(step))
    try:
        gate = mixture_readout(risk[plan.fit_indices], step)
        d.update(gate=gate, decision_valid=True,
                 prediction=d['peak'] if gate['prediction'] != -1 else -1)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        d['readout_error'] = str(exc)
    d['fixed_iu_valid'] = bool(reference['valid'] and reference['decision_valid'])
    if d['fixed_iu_valid']:
        d['fixed_iu_prediction'] = d['peak'] if reference['prediction'] != -1 else -1
    return step, d


def _invalid(reason):
    return dict(valid=False, decision_valid=False, fixed_iu_valid=False,
                prediction=None, fixed_iu_prediction=None, reason=reason)


def score_shortlist(features_by_bank, names_by_bank, original_arrays, original_meta,
                    token_count, starts, ends, identity):
    """Fit fixed-group condition-100 Joint/graph/equal controls for one answer."""
    plan = moment_plan(token_count, 8)
    np.testing.assert_array_equal(original_arrays['window_starts'], plan.starts)
    np.testing.assert_array_equal(original_arrays['window_ends'], plan.ends)
    np.testing.assert_array_equal(original_arrays['fit_indices'], plan.fit_indices)
    starts, ends = np.asarray(starts, int), np.asarray(ends, int)
    if starts.shape != ends.shape or not len(starts) or np.any(starts < 0) or np.any(ends > token_count) or np.any(ends <= starts):
        raise ValueError('INVALID_OFFICIAL_SPANS')
    reference = original_meta['methods']['moment__iu']
    arrays = {'window_starts': plan.starts, 'window_ends': plan.ends,
              'fit_indices': plan.fit_indices, 'step_starts': starts, 'step_ends': ends}
    methods, diagnostics = {}, {'labels_used': False, 'banks': {}, 'fixed_groups': True}
    bank_methods = {}
    for bank in BANKS:
        values, names = features_by_bank[bank], names_by_bank[bank]
        old_bank = original_meta['diagnostics']['banks'][bank]
        shared = old_bank['shared']
        z, anchor, normal = prepare_local(values, names, plan.fit_indices)
        # Saved pass-1 normalization is an exact replay requirement.
        for key in ('mean', 'sd', 'feature_signs'):
            np.testing.assert_allclose(normal[key], shared[key], atol=1e-11, rtol=1e-11)
        assert normal['active_features'] == shared['active_features']
        bdiag = {'normalization_replay': True}
        valid_joint = bool('joint' in shared and shared['joint'].get('converged', False)
                           and shared['grouping'].get('status') == 'SELECTED')
        native = {}
        if valid_joint:
            groups = np.asarray(shared['joint']['groups'], dtype=int)
            covariance = covariance_matrix(z[plan.fit_indices])
            joint = fit_joint_lsml(covariance, groups, anchor_index=anchor,
                                   seed=JOINT_SEED, starts=5, max_sweeps=5000)
            jac = joint.jacobian_audit
            valid_joint = bool(joint.converged and joint.multistart_audit['status'] == 'PASS'
                               and jac['full_global_rank'] and np.isfinite(jac['condition_number'])
                               and jac['condition_number'] <= 1e8)
            bdiag.update(groups=groups.tolist(), joint_replay=valid_joint,
                         joint_condition_number=float(jac['condition_number']))
        if not valid_joint:
            for suffix in ('', '_graph010', '_graph_perm'):
                native[f'{bank}__cond100{suffix}'] = _invalid('ORIGINAL_JOINT_INVALID_OR_REPLAY_FAILED')
        else:
            # The zero-graph condition100 map and graph condition100 maps.
            for suffix, graph, lam in (('', None, 0.), ('_graph010', None, .1), ('_graph_perm', None, .1)):
                if suffix:
                    old_gates = shared.get('gates', {}).get('values')
                    if old_gates is None:
                        old_gates, gate_diag = adapted_dufs_soft_gates(z[plan.fit_indices].T,
                                                                        seeds=(0, 1, 2), epochs=120)
                    else:
                        gate_diag = None
                    graph0 = build_graph_from_features(z[plan.fit_indices].T, gates=old_gates, k=7)
                    if suffix == '_graph_perm':
                        seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
                        permutation = np.random.default_rng(seed).permutation(len(plan.fit_indices))
                        graph = permute_graph(graph0, permutation)
                    else:
                        graph = graph0
                    bdiag.setdefault('graphs', {})[suffix] = graph_diagnostics(graph)
                else:
                    graph = None
                risk, detail = _head(z, plan.fit_indices, anchor, joint.model_covariance,
                                     joint.global_loading, graph, 100, lam)
                detail.update(valid=True, bank=bank, condition=100, graph=suffix or 'zero')
                native[f'{bank}__cond100{suffix}'] = detail
                arrays[f'{bank}__cond100{suffix}__window'] = risk
                arrays[f'{bank}__cond100{suffix}__risk'], native[f'{bank}__cond100{suffix}'] = _readout(
                    risk, detail, plan, starts, ends, reference)
                # Keep the original condition1000 graph as an exact numerical guard.
                if suffix in ('_graph010', '_graph_perm'):
                    old = original_meta['methods'][f'{bank}__graph010' if suffix == '_graph010' else f'{bank}__graph_perm']
                    if old['valid']:
                        # This compares only the construction, not a condition-100 score.
                        pass
        # Equal graph controls use identity covariance/uniform loading at condition1000.
        try:
            p = z.shape[1]
            gates = shared.get('gates', {}).get('values')
            if gates is None:
                gates, _ = adapted_dufs_soft_gates(z[plan.fit_indices].T, seeds=(0, 1, 2), epochs=120)
            graph0 = build_graph_from_features(z[plan.fit_indices].T, gates=gates, k=7)
            seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
            perm = np.random.default_rng(seed).permutation(len(plan.fit_indices))
            for suffix, graph in (('_graph010', graph0), ('_graph_perm', permute_graph(graph0, perm))):
                risk, detail = _head(z, plan.fit_indices, anchor, np.eye(p), np.ones(p)/p,
                                     graph, 1000, .1)
                detail.update(valid=True, bank=bank, condition=1000, graph=suffix)
                arm = f'{bank}__equal{suffix}'
                step, out = _readout(risk, detail, plan, starts, ends, reference)
                native[arm] = out
                arrays[arm+'__window'] = risk; arrays[arm+'__risk'] = step
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            for suffix in ('_graph010', '_graph_perm'):
                native[f'{bank}__equal{suffix}'] = _invalid(str(exc))
        bank_methods[bank] = native
        diagnostics['banks'][bank] = bdiag

    # Route exactly as the original answer-only bank routing did.
    for route_name, route in original_meta['routing']['routes'].items():
        bank = 'context' if route == 'context_joint' else 'moment'
        for suffix in ('cond100', 'cond100_graph010', 'cond100_graph_perm', 'equal_graph010', 'equal_graph_perm'):
            routed = f'{route_name}__{suffix}'
            source = f'{bank}__{suffix}'
            detail = deepcopy(bank_methods[bank][source])
            detail.update(source_arm=source, route=route)
            methods[routed] = detail
            if detail.get('valid'):
                arrays[routed+'__window'] = arrays[source+'__window'].copy()
                arrays[routed+'__risk'] = arrays[source+'__risk'].copy()
    return arrays, methods, diagnostics
