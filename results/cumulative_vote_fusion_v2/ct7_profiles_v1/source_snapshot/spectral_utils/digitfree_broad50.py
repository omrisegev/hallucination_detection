"""Fixed broad bank; feature extraction never accepts correctness labels."""
import numpy as np
from scipy.special import rel_entr
from .renyi_alpha_sweep import sweep_matrix

SHAPES = ('a0.25', 'a0.5', 'H1', 'a2', 'a4', 'Hinf', 'H0lim',
          've0', 've0.5', 've0.75', 've1', 've2', 've4')
NAMES = tuple(f'rank_{i}_risk' for i in range(1, 16)) + (
    'surprisal', 'top1_loggap', 'censored_rank50', 'mass_above', 'tail15', 'tail50'
) + SHAPES + ('top2_ratio',) + tuple(f'{n}_prefix_innovation' for n in SHAPES) + (
    'top15_turnover', 'top50_truncated_js')
ANCHOR = NAMES.index('H1')
assert len(NAMES) == 50


def adjacent_views(ids, logp):
    """Align identities, not ranks; the initial observation has no predecessor."""
    ids = np.asarray(ids); logp = np.asarray(logp, float)
    n = len(ids); turnover = np.zeros(n); js = np.zeros(n)
    q = np.exp(logp - np.max(logp, axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    # Small per-answer batches avoid a token x 50 x 50 full-population tensor.
    for a in range(1, n, 256):
        b = min(n, a + 256)
        match = ids[a:b, :, None] == ids[a-1:b-1, None, :]
        turnover[a:b] = 1 - match[:, :15, :15].sum(axis=(1, 2)) / 15
        prev_on_curr = np.einsum('tij,tj->ti', match, q[a-1:b-1])
        curr_on_prev = np.einsum('tij,ti->tj', match, q[a:b])
        js[a:b] = .5 * (rel_entr(q[a:b], .5*(q[a:b]+prev_on_curr)).sum(axis=1)
            + rel_entr(q[a-1:b-1], .5*(q[a-1:b-1]+curr_on_prev)).sum(axis=1)) / np.log(2)
    return turnover, np.clip(js, 0, 1)


def token_bank(logp, ids, provided, surprisal):
    lp = np.asarray(logp, float); ids = np.asarray(ids)
    provided = np.asarray(provided); s = np.asarray(surprisal, float)
    if lp.shape != ids.shape or lp.ndim != 2 or lp.shape[1] != 50:
        raise ValueError('requires aligned top50 probabilities and token IDs')
    if provided.shape != (len(lp),) or s.shape != provided.shape or not np.isfinite(lp).all():
        raise ValueError('invalid token alignment')
    if np.any(np.diff(lp, axis=1) > 1e-5):
        raise ValueError('top50 must be rank ordered')
    if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError('duplicate candidate IDs')
    p = np.exp(lp); ranks = p[:, :15].copy(); ranks[:, 0] = 1-ranks[:, 0]
    hits = ids == provided[:, None]; rank = np.where(hits.any(axis=1), hits.argmax(axis=1), 50)
    above = (p * (np.arange(50)[None, :] < rank[:, None])).sum(axis=1)
    views, names = sweep_matrix(lp[:, :15]); shape = views[:, [names.index(n) for n in SHAPES]].copy()
    anchor = shape[:, SHAPES.index('ve1')]
    signs = np.ones(13)
    for j, name in enumerate(SHAPES):
        if name.startswith('ve') and np.dot(shape[:, j]-shape[:, j].mean(), anchor-anchor.mean()) < 0:
            shape[:, j] *= -1; signs[j] = -1
    innovation = np.zeros_like(shape)
    if len(lp) > 1:
        innovation[1:] = shape[1:] - np.cumsum(shape, axis=0)[:-1]/np.arange(1, len(lp))[:, None]
    turnover, js = adjacent_views(ids, lp)
    x = np.column_stack((ranks, s, lp[:, 0]+s, rank, above,
        np.maximum(0, 1-p[:, :15].sum(axis=1)), np.maximum(0, 1-p.sum(axis=1)),
        shape, np.exp(lp[:, 1]-lp[:, 0]), innovation, turnover, js))
    valid = np.ones(x.shape, bool); valid[0, 35:] = False
    if not np.isfinite(x).all(): raise ValueError('nonfinite feature')
    return x, valid, signs


def step_bank(logp, ids, provided, surprisal, spans):
    x, valid, signs = token_bank(logp, ids, provided, surprisal)
    out = np.full((len(spans), 50), np.nan); available = np.zeros(out.shape, bool)
    for i, (a, b) in enumerate(np.asarray(spans, int)):
        if not 0 <= a < b <= len(x): raise ValueError('invalid span')
        for j in range(50):
            values = x[a:b, j][valid[a:b, j]]
            if len(values):
                k = min(10, len(values)); out[i, j] = np.partition(values, len(values)-k)[-k:].mean()
                available[i, j] = True
    return out, available, signs


def masked_answer_standardize(x, available, offsets):
    """Unavailable evidence is masked, then neutral zero in standardized space."""
    out = np.zeros_like(x, dtype=float)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for j in range(x.shape[1]):
            ok = available[a:b, j]; v = x[a:b, j][ok]
            if len(v) and v.std() > 1e-12:
                out[a:b, j][ok] = (v-v.mean())/v.std()
    return out
