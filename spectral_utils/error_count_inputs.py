"""Donor-fitted vote inputs for the error-count experiment.

Canonical within-group weights belong to answer-standardized coordinates. The
absolute representation transfers them through a fixed donor within-answer SD
bridge, preserving offsets between answers. It is a registered representation
change, not an assertion that absolute and local scores are equivalent.

All fitting inputs are unlabeled. An activity mask has the full feature shape;
ordinary features are active everywhere and inactive digit values remain zero.

The scale bridge uses compensated float64 pairs (high value plus residual).
This preserves tiny within-answer variation when a much larger donor mean is
subtracted. ``np.longdouble`` has float64 precision on the target macOS host;
using its name alone would not repair that cancellation. Donor parameters stay
JSON float64 values; both coordinate components are recomputed on application.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12
_TOL = 1e-9
_QUANTILE = .8


def _two_sum(a, b):
    """Error-free float64 addition, including opposite-sign cancellation."""
    high = a + b
    bridge = high - a
    low = (a - (high - bridge)) + (b - bridge)
    return high, low


def _two_product(a, b):
    """Dekker product and its residual; no hardware FMA dependency."""
    high = a * b
    split_a, split_b = 134217729. * a, 134217729. * b  # 2**27 + 1 for binary64.
    ah, bh = split_a - (split_a - a), split_b - (split_b - b)
    al, bl = a - ah, b - bh
    low = ((ah * bh - high) + ah * bl + al * bh) + al * bl
    return high, low


def _pair_add(a, b):
    high, low = _two_sum(a[0], b[0])
    tail, tail_error = _two_sum(a[1], b[1])
    high, low = _two_sum(high, low + tail)
    return _two_sum(high, low + tail_error)


def _pair_multiply(a, b):
    high, low = _two_product(a[0], b[0])
    low += a[0] * b[1] + a[1] * b[0] + a[1] * b[1]
    return _two_sum(high, low)


def _pair_divide(a, b):
    first = a[0] / b[0]
    result = (first, np.zeros_like(first))
    # Two residual corrections retain the precision of a float64 pair.
    for _ in range(2):
        product = _pair_multiply(result, b)
        residual = _pair_add(a, (-product[0], -product[1]))
        correction = (residual[0] + residual[1]) / b[0]
        result = _pair_add(result, (correction, np.zeros_like(correction)))
    return result


def _pair_sum_columns(pair):
    high = np.zeros(pair[0].shape[0])
    low = np.zeros_like(high)
    for j in range(pair[0].shape[1]):
        high, low = _pair_add((high, low), (pair[0][:, j], pair[1][:, j]))
    return high, low


def _pair_group_scores(pair, groups):
    scores = []
    for g in groups:
        columns, weights = g['columns'], np.asarray(g['weights'])
        contribution = _pair_multiply((pair[0][:, columns], pair[1][:, columns]),
                                      (weights, np.zeros_like(weights)))
        high, low = _pair_sum_columns(contribution)
        scores.append(high + low)  # One final rounding at the public score interface.
    return np.column_stack(scores)


def _inputs(raw, offsets, active, V=None):
    x = np.asarray(raw, dtype=float)
    mask = np.asarray(active)
    off = np.asarray(offsets)
    if x.ndim != 2 or not all(x.shape):
        raise ValueError('raw must be a nonempty steps-by-features matrix')
    if not np.isfinite(x).all():
        raise ValueError('Non-finite raw input')
    if mask.shape != x.shape or mask.dtype != np.bool_:
        raise ValueError('active must be a bool matrix matching raw')
    if (off.ndim != 1 or len(off) < 2 or off.dtype.kind not in 'iu' or
            off[0] != 0 or off[-1] != len(x) or np.any(off[1:] <= off[:-1])):
        raise ValueError('offsets must delimit nonempty complete answers')
    off = off.astype(np.int64, copy=False)
    v = None if V is None else np.asarray(V, dtype=float)
    if v is not None and (v.shape != x.shape or not np.isfinite(v).all()):
        raise ValueError('V must be finite and match raw')
    return x, off, mask, v


def _answer_moments(x, off, mask):
    """Active-only population moments and pooled within-answer squared error."""
    means = np.zeros((len(off) - 1, x.shape[1]))
    scales = np.zeros_like(means)
    within_sse = np.zeros(x.shape[1])
    counts = np.zeros_like(means, dtype=np.int64)
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        block, use = x[a:b], mask[a:b]
        count = use.sum(0)
        mean = np.divide(np.where(use, block, 0.).sum(0), count,
                         out=np.zeros(x.shape[1]), where=count > 0)
        centered = np.where(use, block - mean, 0.)
        sse = np.square(centered).sum(0)
        sd = np.sqrt(np.divide(sse, count, out=np.zeros_like(sse), where=count > 0))
        means[i], scales[i], counts[i] = mean, sd, count
        within_sse += sse
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or not np.isfinite(within_sse).all():
        raise ValueError('Non-finite answer moments')
    return means, scales, within_sse, counts


def _standardize(x, off, mask, means, scales):
    out = np.zeros_like(x)
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        np.divide(x[a:b] - means[i], scales[i], out=out[a:b],
                  where=mask[a:b] & (scales[i] > _EPS))
    return out


def answer_standardize(raw, offsets, active):
    """Canonical active-only answer z-scores; constants/inactive values are zero."""
    x, off, mask, _ = _inputs(raw, offsets, active)
    means, scales, _, _ = _answer_moments(x, off, mask)
    return _standardize(x, off, mask, means, scales)


def _groups(group_weights, n_features):
    if not group_weights:
        raise ValueError('No groups provided')
    groups, seen = [], []
    for entry in group_weights:
        columns = np.asarray(entry['columns'])
        weights = np.asarray(entry['weights'], dtype=float)
        if (columns.ndim != 1 or not len(columns) or columns.dtype.kind not in 'iu' or
                np.any(columns < 0) or np.any(columns >= n_features)):
            raise ValueError('Group columns must be nonempty valid integer indices')
        if weights.shape != columns.shape or not np.isfinite(weights).all() or not np.any(weights != 0):
            raise ValueError('Group weights must be finite, matching and nonzero')
        groups.append({'columns': columns.astype(int).tolist(), 'weights': weights.tolist()})
        seen.extend(columns.tolist())
    if sorted(seen) != list(range(n_features)):
        raise ValueError('Groups must be a disjoint complete feature partition')
    return groups


def _group_scores(values, groups):
    return np.column_stack([values[:, g['columns']] @ np.asarray(g['weights']) for g in groups])


def _absolute_coordinates(x, mask, mu, r):
    use = mask & (r > _EPS)
    centered = _two_sum(x, -mu)
    safe_scale = np.where(r > _EPS, r, 1.)  # Only unused coordinates may have zero scale.
    pair = _pair_divide(centered, (safe_scale, np.zeros_like(safe_scale)))
    pair = (np.where(use, pair[0], 0.), np.where(use, pair[1], 0.))
    if not np.isfinite(pair[0]).all() or not np.isfinite(pair[1]).all():
        raise ValueError('Non-finite compensated bridge coordinates')
    return pair


def _local_scores(v, off, groups):
    virtual = _group_scores(v, groups)
    return answer_standardize(virtual, off, np.ones(virtual.shape, dtype=bool))


def _max_error(left, right, name):
    error = float(np.max(np.abs(left - right)))
    if not np.isfinite(error) or error > _TOL:
        raise RuntimeError(f'{name} reconstruction error {error} exceeds {_TOL}')
    return error


def _bridge_checks(x, v, mask, off, groups, mu, r, raw_group_weights, cross_weights):
    means, scales, _, counts = _answer_moments(x, off, mask)
    canonical = _standardize(x, off, mask, means, scales)
    coordinates = _absolute_coordinates(x, mask, mu, r)
    inverse = np.zeros_like(x)
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        # The original answer-specific mean/SD appear only in this audit, not
        # in absolute inference. Inactive and constant coordinates remain zero.
        restored = _pair_multiply((coordinates[0][a:b], coordinates[1][a:b]),
                                  (r, np.zeros_like(r)))
        mean_difference = _two_sum(mu, -means[i])
        numerator = _pair_add(restored, mean_difference)
        safe_sd = np.where(scales[i] > _EPS, scales[i], 1.)
        recovered = _pair_divide(numerator, (safe_sd, np.zeros_like(safe_sd)))
        inverse[a:b] = np.where(mask[a:b] & (scales[i] > _EPS), recovered[0] + recovered[1], 0.)
    original_groups = _group_scores(v, groups)
    recovered_groups = _group_scores(inverse, groups)
    direct = []
    for g in raw_group_weights:
        columns = g['columns']
        product = _pair_multiply((x[:, columns], np.zeros_like(x[:, columns])),
                                 (np.asarray(g['coefficients']), np.asarray(g['coefficient_residual'])))
        contribution = _pair_add(product, (np.asarray(g['active_intercept']), np.asarray(g['active_intercept_residual'])))
        use = mask[:, columns]
        high, low = _pair_sum_columns((np.where(use, contribution[0], 0.), np.where(use, contribution[1], 0.)))
        direct.append(high + low)
    direct = np.column_stack(direct)
    diagnostic = {
        'canonical_V_max_abs': _max_error(v, canonical, 'Canonical V'),
        'inverse_V_max_abs': _max_error(v, inverse, 'Inverse V'),
        'virtual_group_max_abs': _max_error(original_groups, recovered_groups, 'Virtual groups'),
        'masked_affine_max_abs': _max_error(_pair_group_scores(coordinates, groups), direct, 'Masked affine'),
        'cross_fused_max_abs': None,
        'coordinate_precision': 'compensated_float64_high_plus_residual',
        'longdouble_epsilon': float(np.finfo(np.longdouble).eps),
        'zero_answer_scale_counts': (scales <= _EPS).sum(0).tolist(),
        'no_active_answer_counts': (counts == 0).sum(0).tolist(),
    }
    if cross_weights is not None:
        cross = np.asarray(cross_weights, dtype=float)
        if cross.shape != (len(groups),) or not np.isfinite(cross).all():
            raise ValueError('cross_weights must be finite and match group order')
        diagnostic['cross_fused_max_abs'] = _max_error(original_groups @ cross,
                                                      recovered_groups @ cross, 'Cross-fused score')
    return diagnostic


def fit_vote_representation(raw, V, active, offsets, group_weights, cross_weights=None):
    """Fit the one fixed scale bridge and q80 thresholds on donor answers only.

    The canonical fuser is fitted by the caller. Its ordered within weights are
    used unchanged. Optional cross weights validate reconstruction only.
    """
    x, off, mask, v = _inputs(raw, offsets, active, V)
    groups = _groups(group_weights, x.shape[1])
    means, scales, sse, counts = _answer_moments(x, off, mask)
    n_active = counts.sum(0)
    mu = np.divide(np.where(mask, x, 0.).sum(0), n_active,
                   out=np.zeros(x.shape[1]), where=n_active > 0)
    r = np.sqrt(np.divide(sse, n_active, out=np.zeros_like(sse), where=n_active > 0))
    required = np.zeros(x.shape[1], dtype=bool)
    for g in groups:
        required[np.asarray(g['columns'])[np.asarray(g['weights']) != 0]] = True
    missing = np.flatnonzero(required & (r <= _EPS))
    if len(missing):
        raise ValueError(f'Required features have zero donor within-answer scale: {missing.tolist()}')
    if not np.isfinite(mu).all() or not np.isfinite(r).all():
        raise ValueError('Non-finite donor feature normalization')
    raw_groups = []
    for g in groups:
        cols, w = np.asarray(g['columns']), np.asarray(g['weights'])
        safe_r = np.where(r[cols] > _EPS, r[cols], 1.)
        beta = _pair_divide((w, np.zeros_like(w)), (safe_r, np.zeros_like(safe_r)))
        beta = (np.where(r[cols] > _EPS, beta[0], 0.), np.where(r[cols] > _EPS, beta[1], 0.))
        intercept = _pair_multiply(beta, (-mu[cols], np.zeros_like(mu[cols])))
        raw_groups.append({'columns': cols.tolist(), 'coefficients': beta[0].tolist(),
                           'coefficient_residual': beta[1].tolist(),
                           'active_intercept': intercept[0].tolist(),
                           'active_intercept_residual': intercept[1].tolist()})
    diagnostics = _bridge_checks(x, v, mask, off, groups, mu, r, raw_groups, cross_weights)
    absolute = _pair_group_scores(_absolute_coordinates(x, mask, mu, r), groups)
    amean, ascale = absolute.mean(0), absolute.std(0)
    local = _local_scores(v, off, groups)
    lscale = local.std(0)
    if np.any(ascale <= _EPS) or np.any(lscale <= _EPS):
        raise ValueError('Constant donor group score; vote model is undefined')
    absolute = (absolute - amean) / ascale
    if not np.isfinite(absolute).all() or not np.isfinite(local).all():
        raise ValueError('Non-finite donor group scores')
    diagnostics.update({'n_steps': len(x), 'n_answers': len(off) - 1,
                        'n_active': n_active.tolist(),
                        'unused_zero_scale_columns': np.flatnonzero(~required & (r <= _EPS)).tolist(),
                        'local_group_population_sd': lscale.tolist()})
    model = {
        'version': 2, 'n_features': x.shape[1], 'quantile': _QUANTILE,
        'quantile_method': 'linear', 'boundary': 'strictly_greater',
        'activity_rule': 'inactive_zero', 'group_weights': groups,
        'cross_weights': None if cross_weights is None else np.asarray(cross_weights, dtype=float).tolist(),
        'mu': mu.tolist(), 'r': r.tolist(), 'raw_group_weights': raw_groups,
        'absolute': {'mean': amean.tolist(), 'scale': ascale.tolist(),
                     'threshold': np.quantile(absolute, _QUANTILE, axis=0, method='linear').tolist()},
        'local': {'threshold': np.quantile(local, _QUANTILE, axis=0, method='linear').tolist()},
        'diagnostics': diagnostics,
    }
    return model


def apply_vote_representation(model, raw, V, active, offsets, kind='absolute'):
    """Apply frozen donor normalization/thresholds; return scores, votes, diagnostics."""
    if kind not in ('absolute', 'local'):
        raise ValueError(f'Unknown vote representation {kind!r}')
    x, off, mask, v = _inputs(raw, offsets, active, V)
    if model['version'] != 2 or model['n_features'] != x.shape[1]:
        raise ValueError('Vote model version or feature count mismatch')
    groups = _groups(model['group_weights'], x.shape[1])
    mu, r = np.asarray(model['mu'], float), np.asarray(model['r'], float)
    if (mu.shape != (x.shape[1],) or r.shape != mu.shape or
            not np.isfinite(mu).all() or not np.isfinite(r).all() or np.any(r < 0)):
        raise ValueError('Invalid saved feature normalization')
    diagnostics = _bridge_checks(x, v, mask, off, groups, mu, r,
                                 model['raw_group_weights'], model['cross_weights'])
    if kind == 'absolute':
        mean = np.asarray(model['absolute']['mean'], float)
        scale = np.asarray(model['absolute']['scale'], float)
        if (mean.shape != (len(groups),) or scale.shape != mean.shape or
                not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= _EPS)):
            raise ValueError('Invalid saved group normalization')
        scores = (_pair_group_scores(_absolute_coordinates(x, mask, mu, r), groups) - mean) / scale
    else:
        scores = _local_scores(v, off, groups)
    threshold = np.asarray(model[kind]['threshold'], float)
    if threshold.shape != (len(groups),) or not np.isfinite(threshold).all() or not np.isfinite(scores).all():
        raise ValueError('Non-finite or mismatched scores/thresholds')
    votes = np.where(scores > threshold, 1, -1).astype(np.int8)
    positive = votes > 0
    answer_positive = np.add.reduceat(positive.sum(1), off[:-1])
    diagnostics.update({'kind': kind, 'n_steps': len(x), 'n_answers': len(off) - 1,
                        'positive_fraction': positive.mean(0).tolist(),
                        'threshold_tie_counts': (scores == threshold).sum(0).tolist(),
                        'zero_positive_vote_answers': int(np.sum(answer_positive == 0)),
                        'all_positive_vote_answers': int(np.sum(answer_positive == np.diff(off) * len(groups)))})
    return scores, votes, diagnostics
