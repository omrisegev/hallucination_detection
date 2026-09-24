"""Reconstruct the frozen family-tail source representation from raw telemetry.

No labels, fitting or target calibration. Historical storage precision is part
of the representation: bank11 tokens are float32; CT7/historical streams are
float64. Every channel must pass full-source parity before external scoring.
"""
import numpy as np
from scipy.signal import lfilter

from .external_generalization.fusion import validate_telemetry, top10
from .external_generalization.ct7 import bocpd_mean
from .external_generalization._bank11.chosen_token_calibration import token_calibration
from .family_tail_transfer import load_lock


def ct7_streams(row):
    """Seven original token streams; no step-0 despiking of chosen surprisal."""
    lp = np.asarray(row['top_k_logprobs']['logprobs'], float)
    p = np.exp(lp[:, :15]); q = p/(p.sum(axis=1, keepdims=True)+1e-12)
    s = -np.log(q+1e-12)
    cols = [np.log(q+1e-12).mean(axis=1)]
    for alpha in (0., .75, 1.):
        w = np.full_like(q, 1/q.shape[1]) if alpha == 0 else q**alpha
        if alpha:
            w /= w.sum(axis=1, keepdims=True)
        m = (w*s).sum(axis=1)
        cols.append((w*s*s).sum(axis=1)-m*m)
    raw = np.column_stack(cols)
    anchor = raw[:, -1]-raw[:, -1].mean()
    bank = raw.copy()
    for j in (1, 2, 3):
        if np.dot(bank[:, j]-bank[:, j].mean(), anchor) < 0:
            bank[:, j] *= -1
    innovation = np.zeros(len(lp))
    innovation[1:] = bank[1:, 0]-np.cumsum(bank[:, 0])[:-1]/np.arange(1, len(lp))
    bank = np.column_stack((bank, innovation))
    temporal = raw.copy()
    for j in range(4):
        if np.dot(temporal[:, j]-temporal[:, j].mean(), anchor) < 0:
            temporal[:, j] *= -1
    innov = np.zeros(len(lp))
    innov[1:] = temporal[1:, 0]-np.cumsum(temporal[:, 0])[:-1]/np.arange(1, len(lp))
    augmented = np.column_stack((temporal, innov))
    z = (augmented.astype(np.float32).astype(float)-augmented.mean(0))/np.maximum(augmented.std(0), 1e-8)
    residual = (z-bocpd_mean(z)).mean(1)
    chosen, _, _ = token_calibration(lp, np.asarray(row['top_k_logprobs']['ids']),
                                     np.asarray(row['gen_token_ids']), np.asarray(row['token_spilled_energies']))
    x = np.column_stack((bank, residual, chosen[:, 3]))
    valid = np.ones(x.shape, bool); valid[0, 4] = False
    return x, valid


def nonhistorical_step_features(row, spans):
    bank = validate_telemetry(row)
    names = load_lock()['recipe']['bank11']
    result = dict(zip(names, top10(bank, spans).T))
    ct, valid = ct7_streams(row)
    cnames = ('H0lim', 've0', 've0.75', 've1', 'H0lim_prefix_innovation', 'bocpd_residual', 'chosen_std_excess')
    for j, name in enumerate(cnames):
        if name == 've1':
            continue
        values = []
        for a, b in spans:
            v = ct[a:b, j][valid[a:b, j]]
            k = min(10, len(v))
            values.append(np.sort(v)[-k:].mean() if k else np.nan)
        result['ct7_'+name] = np.array(values)
    h = bank[:, 0].astype(float)
    lo, hi = np.percentile(h, [25, 75]); scale = (hi-lo)/1.349
    if scale <= 1e-8:
        scale = h.std() if h.std() > 1e-8 else 1.
    h = (h-np.median(h))/scale
    means = np.array([h[a:b].mean() for a, b in spans])
    result['H1_first_token'] = h[np.asarray(spans)[:, 0]]
    slopes = []
    for a, b in spans:
        v = h[a:b]; t = np.arange(len(v))-(len(v)-1)/2.
        slopes.append(t@(v-v.mean())/(t@t) if len(v)>1 else 0.)
    result['H1_slope'] = np.array(slopes)
    result['H1_jump'] = np.r_[means[0], np.diff(means)]
    result['H1_frac_above_z'] = np.array([(h[a:b]>=1.).mean() for a, b in spans])
    lp = np.asarray(row['top_k_logprobs']['logprobs'], float)
    p = np.exp(lp[:, :20]); p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    evidence = (p*np.log(np.maximum(p, 1e-12))).sum(1)
    alpha = 2./6.
    smooth, _ = lfilter([alpha], [1., -(1.-alpha)], evidence, zi=np.atleast_1d((1.-alpha)*evidence[0]))
    risk = (-np.diff(smooth, prepend=smooth[:1])).astype(np.float32).astype(float)
    result['evidence_drop_risk'] = top10(risk, spans)
    return result


def step_features(row, spans=None):
    from .family_hist_features import step_features as historical_step_features
    spans = np.asarray(row['step_token_spans'] if spans is None else spans, int)
    if not len(spans) or np.any(spans[:, 1] <= spans[:, 0]):
        raise ValueError('caller must remove empty steps before feature extraction')
    result = nonhistorical_step_features(row, spans)
    result.update(historical_step_features(row, spans))
    names = load_lock()['recipe']['channels_48']
    matrix = np.column_stack([result[name] for name in names])
    # Exact source pool handling: missing step values -> answer column mean;
    # wholly missing column -> zeros, which stays zero after answer-z.
    if np.isinf(matrix).any():
        raise ValueError('infinite feature values')
    for j in range(matrix.shape[1]):
        good = np.isfinite(matrix[:, j])
        matrix[~good, j] = matrix[good, j].mean() if good.any() else 0.
    return matrix, names
