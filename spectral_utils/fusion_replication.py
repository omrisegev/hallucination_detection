"""Fixed window-fusion recipes for a source-group-disjoint replication.

Only the two already evaluated banks and the declared fallback policies are
scored. No fitting quantities or correctness labels are shared across answers.
"""
from copy import deepcopy
import numpy as np
from .answer_localization_v2 import (
    fit_local, mixture_readout, moment_plan, moment_matrix, MIN_WINDOWS)
from .fusion_context_bank import context_matrix, CORE_NAMES, PARENT_METHODS, REP
from .fusion_explicit_fallback import choose_route
from .window_localization import windows_to_tokens

BASE_ARMS = tuple(bank + '__' + core for bank in ('moment', 'context') for core in CORE_NAMES) + ('entropy_parent',)
NEW_ARMS = tuple(policy + '__' + core for policy in ('single', 'dual') for core in CORE_NAMES[2:]) + ('dual__equal', 'dual__iu')
ARMS = BASE_ARMS + NEW_ARMS


def score_fixed_banks(raw, starts, ends, identity):
    raw = np.asarray(raw, float); starts = np.asarray(starts, int); ends = np.asarray(ends, int)
    if starts.shape != ends.shape or starts.ndim != 1 or not len(starts) or np.any(starts < 0) or np.any(ends > len(raw)) or np.any(ends <= starts):
        raise ValueError('INVALID_OFFICIAL_SPANS')
    plan = moment_plan(len(raw), 8)
    if len(plan.fit_indices) < MIN_WINDOWS: raise ValueError('TOO_FEW_FIT_WINDOWS')
    arrays = {'step_starts': starts, 'step_ends': ends, 'window_starts': plan.starts,
              'window_ends': plan.ends, 'fit_indices': plan.fit_indices}
    methods = {}; diagnostics = {'banks': {}, 'labels_accessed': False}

    def admit(arm, risk, detail):
        detail = deepcopy(detail)
        detail.update(decision_valid=False, fixed_iu_valid=False, prediction=None, fixed_iu_prediction=None)
        if detail.get('valid', False):
            token = windows_to_tokens(plan, risk)
            steps = np.asarray([token[a:b].max() for a, b in zip(starts, ends)])
            if not np.isfinite(steps).all(): raise ValueError('NONFINITE_STEP_SCORE')
            arrays[arm + '__window'] = np.asarray(risk, float); arrays[arm + '__risk'] = steps
            detail['peak'] = int(np.argmax(steps))
            try:
                gate = mixture_readout(risk[plan.fit_indices], steps)
                detail.update(gate=gate, decision_valid=True,
                              prediction=detail['peak'] if gate['prediction'] != -1 else -1)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                detail['readout_error'] = str(exc)
        detail.setdefault('valid', False); methods[arm] = detail

    for bank, build in (('moment', moment_matrix), ('context', context_matrix)):
        try:
            values, names = build(raw, plan); arrays[bank + '__features'] = values
            risk, meta, shared = fit_local(values, names, plan, identity + '/' + REP)
            diagnostics['banks'][bank] = {'status': 'SCORED', 'names': names, 'shared': shared}
            for core, source in PARENT_METHODS.items():
                admit(bank + '__' + core, risk.get(source), meta.get(source, {'valid': False, 'reason': 'SOURCE_MISSING'}))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            diagnostics['banks'][bank] = {'status': 'FAILED', 'reason': str(exc)}
            for core in CORE_NAMES:
                methods.setdefault(bank + '__' + core, {'valid': False, 'decision_valid': False,
                    'fixed_iu_valid': False, 'prediction': None, 'fixed_iu_prediction': None, 'reason': str(exc)})
    entropy = np.asarray([raw[a:b, 1].mean() for a, b in zip(plan.starts, plan.ends)])
    fit = entropy[plan.fit_indices]
    if np.isfinite(entropy).all() and fit.std() > 1e-12:
        admit('entropy_parent', (entropy - fit.mean()) / fit.std(), {'valid': True})
    else: admit('entropy_parent', None, {'valid': False, 'reason': 'ENTROPY_UNAVAILABLE'})
    reference = methods['moment__iu']; common_valid = reference['valid'] and reference['decision_valid']
    for detail in methods.values():
        detail['fixed_iu_valid'] = bool(detail['valid'] and common_valid)
        if detail['fixed_iu_valid']:
            detail['fixed_iu_prediction'] = detail['peak'] if reference['prediction'] != -1 else -1
    eligibility = {}
    for bank in ('moment', 'context'):
        flags = [bool(methods[bank + '__' + c]['valid']) for c in CORE_NAMES[2:]]
        if len(set(flags)) != 1: raise ValueError('JOINT_FAMILY_ELIGIBILITY_MISMATCH')
        eligibility[bank] = flags[0]
    routes = {p: choose_route(eligibility['moment'], eligibility['context'], p) for p in ('single', 'dual')}
    for arm in BASE_ARMS: methods[arm].update(source_arm=arm, route='unchanged_reference')
    for arm in NEW_ARMS:
        policy, core = arm.split('__'); route = routes[policy]
        bank = 'context' if route == 'context_joint' else 'moment'
        source = bank + '__' + (('iu' if route == 'moment_iu' else core) if core in CORE_NAMES[2:] else core)
        methods[arm] = {**deepcopy(methods[source]), 'source_arm': source, 'route': route}
        if methods[arm]['valid']:
            for suffix in ('window', 'risk'): arrays[arm + '__' + suffix] = arrays[source + '__' + suffix].copy()
    return arrays, methods, {'eligibility': eligibility, 'routes': routes}, diagnostics
