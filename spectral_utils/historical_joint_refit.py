"""Historical Joint/L-SML graph roster, with the exact v2 fitting entry point.

These are full-population pooled-training comparison arms. They are not the
primary answer-only method and introduce no new graph tuning grid.
"""
import numpy as np

REQUESTED = (
    'internal_cont', 'internal_joint',
    'internal_joint_gate050', 'internal_joint_gate100',
    'internal_joint_liu010', 'internal_joint_liu050',
    'internal_joint_diag010', 'internal_joint_diag050',
    'internal_joint_modelinv_lam0',
)
ARMS = REQUESTED + ('permctl_graph_internal_joint_liu010',)
SEED = 20260905


def fold_seed(outer, inner=None):
    return SEED + 100*int(outer) + (0 if inner is None else 10*int(inner)+1)


def fit(preparation, reference, *, cell, outer, inner=None):
    seed = fold_seed(outer, inner)
    output = reference['joint_lsml_v2_localization'].fit_v2_arms(
        preparation, seed=seed, cell_key=cell,
        domain='prmbench' if cell.startswith('prm') else 'processbench',
        rows=REQUESTED, include_iu=False,
    )
    weights = {arm: np.asarray(output['weights'][arm]) for arm in ARMS if arm in output['weights']}
    metadata = {arm: output['row_meta'][arm] for arm in ARMS if arm in weights}
    failures = {arm: output['failures'][arm] for arm in ARMS if arm in output['failures']}
    assert set(weights).isdisjoint(failures)
    assert set(weights) | set(failures) == set(ARMS)
    audit = dict(seed=seed, internal_grouping_status=output['internal_grouping_status'],
        internal_K=output['internal_K'], fallback_events=output['fallback_events'],
        gate_diagnostics=output['gate_diagnostics'], labels_accessed=output['labels_accessed'],
        unused_returned_controls=sorted(set(output['weights'])-set(ARMS)),
        structural_scope='Exact historical admission/fallback behavior; no new Jacobian admissibility rule')
    assert not audit['labels_accessed']
    return weights, metadata, failures, audit
