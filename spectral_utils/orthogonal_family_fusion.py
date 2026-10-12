"""Fixed common-plus-Helmert family representation and label-free weighting.

The fitting API deliberately accepts no labels, outcomes, or metric callbacks.
Orthogonal coordinates are not presumed statistically independent.
"""
import hashlib
import numpy as np
from spectral_utils.fusion_utils import lsml_continuous
from spectral_utils.lsml_gate_locator_research import answer_standardize
from spectral_utils.monday_fusion_core import between_weights
import er_stage_a as SA
import er_stage_b as SB
import ds_group_weights as GW


def helmert_basis(m):
    if m < 1:
        raise ValueError('Empty family')
    q = np.zeros((m, m))
    q[:, 0] = 1. / np.sqrt(m)
    for k in range(1, m):
        scale = np.sqrt(k * (k + 1))
        q[:k, k] = 1. / scale
        q[k, k] = -k / scale
    np.testing.assert_allclose(q.T @ q, np.eye(m), atol=1e-12, rtol=0)
    return q


def representation(x, groups, names, off):
    groups = GW.canon(groups)
    m = x.shape[1]
    basis = np.zeros((m, m))
    new_groups, column_names, block_info, common = [], [], [], []
    pos = 0
    for h in range(int(groups.max()) + 1):
        idx = sorted(np.flatnonzero(groups == h), key=lambda j: names[j])
        q = helmert_basis(len(idx))
        basis[np.ix_(idx, np.arange(pos, pos + len(idx)))] = q
        new_groups.extend([h] * len(idx))
        column_names.extend([f'group{h}_common'] + [f'group{h}_contrast{k}' for k in range(1, len(idx))])
        block_info.append(dict(group=h, source_indices=idx, ordered_features=[names[j] for j in idx],
                               columns=list(range(pos, pos + len(idx))), basis=q.tolist()))
        common.append(pos)
        pos += len(idx)
    np.testing.assert_allclose(basis.T @ basis, np.eye(m), atol=1e-12, rtol=0)
    projected = x @ basis
    error = float(np.max(np.abs(projected @ basis.T - x)))
    if error > 1e-10:
        raise AssertionError(f'Noninvertible representation: {error}')
    values = answer_standardize(projected, off)
    common_scores = answer_standardize(projected[:, common], off)
    return values, np.asarray(new_groups), common_scores, dict(basis=basis.tolist(), blocks=block_info,
                                                              column_names=column_names, common_columns=common,
                                                              reconstruction_error=error)


def projected_marks(values, off, column_names, namespace):
    key = np.empty(values.shape)
    for j, name in enumerate(column_names):
        seed = int.from_bytes(hashlib.sha256((namespace + '|' + name).encode()).digest()[:8], 'little')
        key[:, j] = np.random.default_rng(seed).random(len(values))
    marks = SB.random_tie_marks(values, off, .2, key)
    for a, b in zip(off[:-1], off[1:]):
        expected = max(1, int(np.ceil(.2 * (b - a))))
        if not np.all((marks[a:b] > 0).sum(axis=0) == expected):
            raise AssertionError('Projected TAIL quota mismatch')
    return marks


def within_weights(method, values, marks, groups):
    groups = GW.canon(groups)
    count = int(groups.max()) + 1
    diagnostics = []
    if method == 'EQ':
        return None, dict(groups=[dict(group=h, size=int((groups == h).sum()), source='equal_control') for h in range(count)])
    if method == 'HEM':
        estimate = GW.hem_fit(marks, groups)
        weights = estimate['within']
        for h, w in enumerate(weights):
            size = int((groups == h).sum())
            source = 'equal_small_group' if size < 3 else 'equal_all_zero' if w is None else 'HEM'
            diagnostics.append(dict(group=h, size=size, source=source))
        return weights, dict(groups=diagnostics, estimate=estimate)
    if method == 'SML':
        _, meta = lsml_continuous(*values.T, groups=groups, compute_score_matrix=False, small_m_guard=True)
        weights = [np.asarray(w, float) for _, w in meta['group_weights']]
        for h, w in enumerate(weights):
            size = int((groups == h).sum())
            source = 'singleton' if size == 1 else 'sd_equal_size3_guard' if size == 3 else 'SML'
            diagnostics.append(dict(group=h, size=size, source=source, negative_weights=int((w < 0).sum())))
        return weights, dict(groups=diagnostics, small_m_flags=meta['small_m_flags'], small_m_guarded=meta['small_m_guarded'])
    weights = []
    for h in range(count):
        idx = np.flatnonzero(groups == h)
        if len(idx) < 3:
            weights.append(None)
            diagnostics.append(dict(group=h, size=len(idx), source='equal_small_group'))
            continue
        estimate = SA.tensor_mom_estimate(marks[:, idx], None) if method == 'MOM' else SA.em_estimate(marks[:, idx], 'ds')
        w = SB.mle_weights(estimate['psi'], estimate['eta'])
        if not np.isfinite(w).all():
            raise ArithmeticError('Nonfinite within weights')
        weights.append(w if w.sum() > 0 else None)
        diagnostics.append(dict(group=h, size=len(idx), source=method if w.sum() > 0 else 'equal_all_zero',
                                estimate=estimate, zero_weights=int((w == 0).sum())))
    return weights, dict(groups=diagnostics)


def fit_cell(x, marks, groups, names, off, group_key, anchor_index, namespace):
    """No labels accepted; fit all predetermined representations/methods."""
    groups = GW.canon(groups)
    rotated, rotated_groups, common_scores, basis_info = representation(x, groups, names, off)
    rotated_marks = projected_marks(rotated, off, basis_info['column_names'], namespace)
    anchor_score = x[:, anchor_index]
    anchor_group = int(groups[anchor_index])
    scores, records, failures = {}, [], []
    group_outputs = {}
    for rep, values, votes, assignment in [('RAW', x, marks, groups), ('ORTH', rotated, rotated_marks, rotated_groups)]:
        for method in ['EQ', 'SML', 'MOM', 'DS', 'HEM']:
            try:
                weights, details = within_weights(method, values, votes, assignment)
                z = GW.group_matrix(values, off, assignment, weights, answer_standardize)
                if not np.isfinite(z).all():
                    raise ArithmeticError('Nonfinite group scores')
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                for between in ['EQ', 'MOM']:
                    name = f'{rep}_{method}_{between}'
                    failures.append(dict(method=name, stage='within', reason=repr(error)))
                    scores[name] = None
                continue
            group_outputs[f'{rep}_{method}'] = z
            for h, detail in enumerate(details['groups']):
                idx = np.flatnonzero(assignment == h)
                w = np.ones(len(idx)) if weights is None or weights[h] is None else np.asarray(weights[h], float)
                w = w / np.abs(w).sum()
                detail.update(weights=w.tolist(), negative_weight_share=float(np.abs(w[w < 0]).sum()))
                if rep == 'ORTH':
                    detail.update(common_weight=float(w[0]), contrast_weight_share=float(np.abs(w[1:]).sum()))
            record = dict(representation=rep, within=method, details=details, between={})
            for between in ['EQ', 'MOM']:
                name = f'{rep}_{method}_{between}'
                try:
                    group_votes = SB.random_tie_marks(z, off, .2, group_key)
                    w, estimate = between_weights(between, z, group_votes, anchor_score, anchor_group)
                    score = GW.weighted_group_score(z, w)
                    if not np.isfinite(score).all():
                        raise ArithmeticError('Nonfinite fused score')
                    scores[name] = score
                    record['between'][between] = dict(weights=w, estimate=estimate)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                    scores[name] = None
                    failures.append(dict(method=name, stage='between', reason=repr(error)))
            records.append(record)
    for between in ['EQ', 'MOM']:
        name = 'COMMON_' + between
        try:
            votes = SB.random_tie_marks(common_scores, off, .2, group_key)
            w, estimate = between_weights(between, common_scores, votes, anchor_score, anchor_group)
            scores[name] = GW.weighted_group_score(common_scores, w)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            scores[name] = None
            failures.append(dict(method=name, stage='common_control', reason=repr(error)))
    diagnostics = dict(basis=basis_info, records=records, failures=failures,
                       n_rows=len(x), n_channels=x.shape[1], n_groups=len(np.unique(groups)),
                       anchor_index=anchor_index, anchor_group=anchor_group)
    return scores, diagnostics, group_outputs
