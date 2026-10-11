"""Portable port of the frozen F15_tailtie_lsml recipe (not the EM experiment).

Input is a finite SOURCE-DEFINITION step-feature matrix, before orientation.
This module does not reconstruct missing token features or certify their parity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .external_generalization.fusion import numerical_backend, lsml_continuous

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT/'results/family_tail_transfer_v1/TRANSFER_LOCK_V1.json'
LOCK_SHA256 = '65b35336fcc2f66b7843ec040d3bdafbbaa03bb44ae5f61f1d00335abfaea5cf'
EPS = 1e-12
FAMILY_METHODS = ('F15_tailtie_lsml', 'F15_equal', 'F15_cov_lsml')


def load_lock():
    raw = LOCK_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCK_SHA256:
        raise ValueError('frozen transfer lock changed; use a new version')
    return json.loads(raw)


def checked_offsets(offsets, rows):
    off = np.asarray(offsets)
    if off.ndim != 1 or off.dtype.kind not in 'iu' or len(off) < 2 or off[0] != 0 or off[-1] != rows or np.any(np.diff(off) <= 0):
        raise ValueError('expected nonempty answer slices and integer offsets')
    return off


def answer_standardize(values, offsets, *, final=False):
    x = np.asarray(values, dtype=np.float64)
    off = checked_offsets(offsets, len(x))
    if x.ndim not in (1, 2) or not np.isfinite(x).all():
        raise ValueError('features must be finite; upstream extraction owns missing values')
    out = np.zeros_like(x)
    for a, b in zip(off[:-1], off[1:]):
        block = x[a:b]
        sd = block.std(axis=0)
        out[a:b] = np.divide(block-block.mean(axis=0), sd, out=np.zeros_like(block),
                             where=sd > (1e-8 if final else EPS))
    return out


def build_representations(step_features, names, offsets, *, include_all=False):
    """Named feature alignment; family means are taken AFTER channel answer-z/sign.

    For the three family arms only28 channels are needed. Full nine-arm scoring
    requires48 channels. Original directions are expected; do not pre-flip inputs.
    Empty external steps must be excluded and restored by the benchmark adapter.
    """
    lock = load_lock(); recipe = lock['recipe']
    x = np.asarray(step_features, dtype=np.float64)
    names = list(names)
    if x.ndim != 2 or x.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('one unique feature name per column required')
    needed = recipe['channels_48'] if include_all else recipe['channels_28']
    missing = sorted(set(needed)-set(names))
    if missing:
        raise ValueError('missing locked channels: '+', '.join(missing))
    z = answer_standardize(x[:, [names.index(c) for c in needed]], offsets)
    raw = dict(zip(needed, z.T))
    oriented = {c: raw[c]*recipe['source_signs'][c] for c in needed}
    families = recipe['families_15']
    f = np.column_stack([np.column_stack([oriented[c] for c in members]).mean(axis=1)
                         for members in families.values()])
    reps = {'F15': (answer_standardize(f, offsets), list(families)),
            'K28': (np.column_stack([oriented[c] for c in recipe['channels_28']]), recipe['channels_28'])}
    if include_all:
        reps['A48o'] = (np.column_stack([oriented[c] for c in needed]), needed)
        reps['B11'] = (np.column_stack([raw[c] for c in recipe['bank11']]), recipe['bank11'])
    return reps


def tail_marks(families, offsets, *, centred=True):
    """Exact upstream top-ceil(20% n) quota with fractional boundary ties.

    Constant columns become zero AFTER centering. No subsequent pooled z-score.
    These are rank/tail indicators, not strictly binary when boundary ties occur.
    """
    v = np.asarray(families, dtype=np.float64)
    off = checked_offsets(offsets, len(v))
    if v.ndim != 2 or not np.isfinite(v).all():
        raise ValueError('finite family matrix required')
    result = np.zeros_like(v)
    for a, b in zip(off[:-1], off[1:]):
        block = v[a:b]; n = b-a; k = max(1, int(np.ceil(.2*n)))
        marks = np.zeros_like(block)
        for j in range(block.shape[1]):
            col = block[:, j]; threshold = np.sort(col)[::-1][k-1]
            greater, equal = col > threshold, col == threshold
            marks[greater, j] = 1.
            marks[equal, j] = (k-greater.sum())/equal.sum()
        result[a:b] = marks-marks.mean(axis=0) if centred else marks
    return result


def fit_family_tail(families, offsets, fit_answer_mask):
    """SOURCE replay/refitting API; external evaluation uses frozen weights instead.

    Import the same numerical backend as the lock, without another worktree or
    absolute Windows imports. Labels are not accepted. No hidden equal fallback.
    """
    v = np.asarray(families, dtype=np.float64)
    off = checked_offsets(offsets, len(v))
    mask = np.asarray(fit_answer_mask)
    if v.shape[1] != 15 or mask.dtype.kind != 'b' or mask.shape != (len(off)-1,):
        raise ValueError('expected fifteen families and a Boolean answer mask')
    rows = np.repeat(mask, np.diff(off))
    x, orient = tail_marks(v, off)[rows], v[rows]
    if len(x) < 3*x.shape[1] or orient[:, 0].std() <= EPS:
        raise ValueError('insufficient rows or inactive entropy orientation anchor')
    numerical_backend.NUMERICAL_FAILURES.clear()
    _, meta = lsml_continuous(*x.T, compute_score_matrix=False, small_m_guard=True)
    if numerical_backend.NUMERICAL_FAILURES or not np.isfinite(meta['residual']):
        raise ValueError('numerical estimator failure: '+repr(numerical_backend.NUMERICAL_FAILURES))
    w = np.zeros(x.shape[1])
    for cross, (idx, within) in zip(meta['cross_weights'], meta['group_weights']):
        w[np.asarray(idx, int)] = np.asarray(within)*cross
    rho = float(spearmanr(orient@w, orient[:, 0]).statistic)
    if not np.isfinite(w).all() or abs(w).sum() <= EPS or not np.isfinite(rho):
        raise ValueError('invalid weights or undefined orientation')
    if rho < 0:
        w *= -1
    w /= abs(w).sum()
    g = np.asarray(meta['c'], int)
    return {'weights': w.tolist(), 'groups': g.tolist(), 'K': len(np.unique(g)),
            'anchor_spearman': abs(rho), 'anchor_flipped': rho < 0,
            'residual': float(meta['residual'])}


def score_locked(step_features, names, offsets, *, methods=FAMILY_METHODS):
    """Frozen source weights AND source thresholds; never fits on target data.

    Returns high-is-risk scores and valid-step predictions (1=correct,0=error).
    CT7 is supplied by the existing external CT7 pipeline, not this feature API.
    """
    lock = load_lock()
    methods = tuple(methods)
    if not methods or any(m not in lock['deployment'] or m == 'ct7' for m in methods):
        raise ValueError('unknown/empty method roster or CT7 requested from feature API')
    include_all = any(m.startswith(('A48o_', 'B11_')) for m in methods)
    reps = build_representations(step_features, names, offsets, include_all=include_all)
    result = {}
    for method in methods:
        matrix, columns = reps[method.split('_')[0]]
        record = lock['deployment'][method]
        w = np.array([record['weights'][c] for c in columns])
        values = answer_standardize(matrix@w, offsets, final=True)
        tau = record['q80_threshold_fold4']
        result[method] = {'scores': values, 'pred_valid': (values < tau).astype(np.int8),
                          'threshold': tau, 'lock_sha256': LOCK_SHA256}
    return result
