"""Declared fit-validity routing among existing answer-local fusion scores.

No correctness targets, population statistics, or answer IDs enter this API.
An error-free prediction is a valid prediction, not a fallback trigger.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

CORES = ('joint0', 'graph010', 'graph_perm')
PARENT_ARMS = tuple(b + '__' + c for b in ('moment', 'context')
                    for c in ('equal', 'iu') + CORES) + tuple(
    b + '__' + c for b in ('moment_allk', 'context_allk') for c in CORES) + ('entropy_parent',)
NEW_ARMS = tuple(p + '__' + c for p in ('single', 'dual') for c in CORES) + (
    'dual__equal', 'dual__iu')
ARMS = PARENT_ARMS + NEW_ARMS


def choose_route(moment_valid: bool, context_valid: bool, policy: str) -> str:
    """Return a bank/estimator route from fit eligibility only."""
    if policy not in ('single', 'dual'):
        raise ValueError('UNKNOWN_ROUTING_POLICY')
    if not isinstance(moment_valid, (bool, np.bool_)) or not isinstance(context_valid, (bool, np.bool_)):
        raise TypeError('FIT_ELIGIBILITY_MUST_BE_BOOLEAN')
    if moment_valid:
        return 'moment_joint'
    if policy == 'dual' and context_valid:
        return 'context_joint'
    return 'moment_iu'


def compose_fallback(parent_arrays, parent_methods):
    """Copy immutable source predictions using one shared routing decision.

    The returned metadata names the route and source for every composite.
    Source decision failures remain failures even if another gate succeeds.
    """
    if set(parent_methods) != set(PARENT_ARMS):
        raise ValueError('PARENT_ROSTER_DRIFT')
    eligibility = {}
    for bank in ('moment', 'context'):
        flags = [parent_methods[bank + '__' + core]['valid'] for core in CORES]
        if any(type(flag) is not bool for flag in flags) or len(set(flags)) != 1:
            raise ValueError('JOINT_FAMILY_ELIGIBILITY_MISMATCH')
        eligibility[bank] = flags[0]
    routes = {p: choose_route(eligibility['moment'], eligibility['context'], p)
              for p in ('single', 'dual')}
    arrays = {}
    methods = {}

    def inherit(arm, source, route):
        detail = deepcopy(parent_methods[source])
        detail.update(source_arm=source, route=route)
        if detail['valid']:
            for suffix in ('window', 'risk'):
                score = np.asarray(parent_arrays[source + '__' + suffix], dtype=float)
                if score.ndim != 1 or not score.size or not np.isfinite(score).all():
                    raise ValueError('INVALID_SOURCE_SCORE')
                arrays[arm + '__' + suffix] = score.copy()
        elif detail['decision_valid'] or detail['fixed_iu_valid']:
            raise ValueError('INVALID_FIT_HAS_VALID_DECISION')
        methods[arm] = detail

    for arm in PARENT_ARMS:
        inherit(arm, arm, 'unchanged_reference')
    for policy in ('single', 'dual'):
        route = routes[policy]
        for core in CORES:
            source = 'moment__iu' if route == 'moment_iu' else route.split('_')[0] + '__' + core
            inherit(policy + '__' + core, source, route)
    for core in ('equal', 'iu'):
        bank = 'context' if routes['dual'] == 'context_joint' else 'moment'
        inherit('dual__' + core, bank + '__' + core, routes['dual'])
    return arrays, methods, {'eligibility': eligibility, 'routes': routes}
