"""Answer-fitted supporting readouts of frozen fusion scores; no label API.

BOCPD uses reset-before-observation semantics and updates the fresh segment
with x_t. This is a declared adaptation, not the original paper's r_t=0
after-observation convention. IMM mixes continuous-state distributions.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .latent_state_localizer import (fit_upcr_initialized_hmm, model_to_dict,
                                    posterior_entry_curve, posterior_high_risk_curve)

CORES = tuple('moments27_local8__' + method for method in
              ('equal', 'iu', 'joint_lambda0', 'joint_graph010', 'joint_graph_permuted')) + ('entropy_mean_w8',)
READOUTS = ('parent_first', 'parent_peak', 'hold_peak', 'hmm_entry',
            'kalman_level', 'imm_level', 'bocpd_rise')


def finite_sequence(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite scalar sequence')
    return x


def normal_logpdf(x, mean, variance):
    return -.5 * (np.log(2 * np.pi * variance) + (x - mean) ** 2 / variance)


def hold_steps(values, width, tokens, starts, ends):
    x = finite_sequence(values)
    starts, ends = np.asarray(starts, int), np.asarray(ends, int)
    if width < 1 or len(x) != tokens // width or starts.shape != ends.shape:
        raise ValueError('Invalid full-window geometry')
    if not len(starts) or np.any(starts < 0) or np.any(ends > tokens) or np.any(ends <= starts):
        raise ValueError('Invalid official spans')
    expanded = np.repeat(x, width)
    expanded = np.pad(expanded, (0, tokens - len(expanded)), constant_values=x[-1])
    return np.asarray([expanded[a:b].max() for a, b in zip(starts, ends)])


def noise_variance(x):
    x = finite_sequence(x)
    estimate = (np.median(np.abs(np.diff(x))) / (.67448975 * np.sqrt(2))) ** 2 if len(x) > 1 else 1.
    return float(np.clip(estimate, .05, 1.))


def imm_filter(values, observation_variance, process_variances, transition=None):
    """Scalar random-walk IMM; transition[i,j] means source i -> destination j."""
    x = finite_sequence(values)
    q = np.asarray(process_variances, dtype=float)
    k = len(q)
    if k < 1 or observation_variance <= 0 or np.any(q < 0):
        raise ValueError('Invalid noise parameters')
    if transition is None:
        transition = np.ones((1, 1)) if k == 1 else np.array([[.95, .05], [.05, .95]])
    transition = np.asarray(transition, float)
    if transition.shape != (k, k) or np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1.):
        raise ValueError('Transition must be row stochastic')
    means, variances, modes = np.zeros(k), np.ones(k), np.ones(k)/k
    levels, covariance, mode_history, likelihoods = [], [], [], []
    for value in x:
        predicted_modes = modes @ transition
        if np.any(predicted_modes <= 0):
            raise ValueError('Unreachable mode')
        mixing = modes[:, None] * transition / predicted_modes[None, :]
        mixed_mean = means @ mixing
        mixed_var = np.sum(mixing * (variances[:, None] + (means[:, None]-mixed_mean[None, :])**2), axis=0)
        predicted_var = mixed_var + q
        innovation_var = predicted_var + observation_variance
        log_likelihood = normal_logpdf(value, mixed_mean, innovation_var)
        gain = predicted_var / innovation_var
        means = mixed_mean + gain * (value - mixed_mean)
        # Scalar Joseph form, preserving covariance under rounding.
        variances = (1-gain)**2 * predicted_var + gain**2 * observation_variance
        log_modes = np.log(predicted_modes) + log_likelihood
        evidence = float(logsumexp(log_modes))
        modes = np.exp(log_modes - evidence)
        level = float(modes @ means)
        levels.append(level)
        covariance.append(float(modes @ (variances + (means-level)**2)))
        mode_history.append(modes.copy())
        likelihoods.append(evidence)
    return {'level': np.asarray(levels), 'variance': np.asarray(covariance),
            'mode_probability': np.asarray(mode_history), 'log_predictive': np.asarray(likelihoods)}


def bocpd_filter(values, hazard=1/32, observation_variance=1., prior_mean=0., prior_variance=1.):
    """Exact untruncated Gaussian product partitions, reset before current datum."""
    x = finite_sequence(values)
    if not 0 < hazard < 1 or min(observation_variance, prior_variance) <= 0:
        raise ValueError('Invalid BOCPD parameters')
    posterior = np.ones(1)
    means, variances = np.array([prior_mean]), np.array([prior_variance])
    levels, p_reset, rise, logs, distributions = [], [], [], [], []
    for t, value in enumerate(x):
        continuation_mean = float(posterior @ means)
        continuation_var = float(posterior @ (variances + (means-continuation_mean)**2)) + observation_variance
        if t == 0:
            weights, candidate_mean, candidate_var = np.ones(1), means, variances
        else:
            weights = np.r_[hazard, (1-hazard)*posterior]
            candidate_mean = np.r_[prior_mean, means]
            candidate_var = np.r_[prior_variance, variances]
        with np.errstate(divide='ignore'):
            log_mass = np.log(weights) + normal_logpdf(value, candidate_mean, candidate_var+observation_variance)
        log_evidence = float(logsumexp(log_mass))
        posterior = np.exp(log_mass-log_evidence)
        gain = candidate_var/(candidate_var+observation_variance)
        means = candidate_mean + gain*(value-candidate_mean)
        variances = (1-gain)*candidate_var
        reset = float(posterior[0]) if t else hazard
        levels.append(float(posterior @ means))
        p_reset.append(reset)
        rise.append(reset * max(float((value-continuation_mean)/np.sqrt(continuation_var)), 0.))
        logs.append(log_evidence)
        distributions.append(posterior.copy())
    return {'level': np.asarray(levels), 'reset_probability': np.asarray(p_reset),
            'rise': np.asarray(rise), 'log_predictive': np.asarray(logs),
            'partition_posterior': distributions}


def temporal_curves(values):
    x = finite_sequence(values)
    r = noise_variance(x)
    single = imm_filter(x, r, [.01*r])
    imm = imm_filter(x, r, [.01*r, r])
    bocpd = bocpd_filter(x)
    result = {
        'hold_peak': {'valid': True, 'risk': x, 'onset': x, 'detail': {}},
        'kalman_level': {'valid': True, 'risk': single['level'], 'onset': single['level'], 'detail': {'R': r}, 'diagnostic': single},
        'imm_level': {'valid': True, 'risk': imm['level'], 'onset': imm['level'], 'detail': {'R': r}, 'diagnostic': imm},
        'bocpd_rise': {'valid': True, 'risk': bocpd['level'], 'onset': bocpd['rise'], 'detail': {'hazard': 1/32},
                       'diagnostic': {k:v for k,v in bocpd.items() if k != 'partition_posterior'}},
    }
    fit = fit_upcr_initialized_hmm([x], kind='reversible', max_iter=120)
    model = fit.selected
    detail = {'selected': model_to_dict(model), 'diagnostics': fit.diagnostics,
              'fallback_reason': fit.fallback_reason, 'candidates': [model_to_dict(c) for c in fit.candidates]}
    if model is None or model.n_iter_used >= 120:
        result['hmm_entry'] = {'valid': False, 'detail': detail, 'reason': fit.fallback_reason or 'ITERATION_CAP'}
    else:
        result['hmm_entry'] = {'valid': True, 'risk': posterior_high_risk_curve(model, x),
                               'onset': posterior_entry_curve(model, x), 'detail': detail}
    return result


def score_readouts(arrays, parent_metadata):
    """Only saved telemetry scores and fitting metadata enter; no targets."""
    starts, ends = arrays['step_starts'], arrays['step_ends']
    tokens = parent_metadata['tokens']
    output = {'step_starts': starts, 'step_ends': ends}
    details = {}
    fit_indices = arrays['moments27_local8__fit_indices']
    for core in CORES:
        parent = parent_metadata['report']['methods'][core]
        valid = parent.get('valid', False) and parent.get('readout_valid', False)
        gate = parent.get('readout', {}).get('prediction', -1) != -1
        if not valid:
            for readout in READOUTS:
                details[core+'@@'+readout] = {'valid': False, 'reason': 'PARENT_UNAVAILABLE', 'gate': gate}
            continue
        raw_steps = arrays[core+'__step']
        for readout in ('parent_first', 'parent_peak'):
            name = core+'@@'+readout
            prediction = parent['readout']['prediction'] if readout == 'parent_first' else (int(np.argmax(raw_steps)) if gate else -1)
            output[name+'__risk'] = raw_steps
            output[name+'__onset'] = raw_steps
            details[name] = {'valid': True, 'prediction': prediction, 'gate': gate}
        x = arrays[core+'__window'][fit_indices]
        curves = temporal_curves(x)
        for readout, curve in curves.items():
            name = core+'@@'+readout
            detail = {'valid': curve['valid'], 'gate': gate, **curve['detail']}
            if curve['valid']:
                risk = hold_steps(curve['risk'], 8, tokens, starts, ends)
                onset = hold_steps(curve['onset'], 8, tokens, starts, ends)
                output[name+'__risk'], output[name+'__onset'] = risk, onset
                output[name+'__window_risk'], output[name+'__window_onset'] = curve['risk'], curve['onset']
                for key, value in curve.get('diagnostic', {}).items():
                    output[name+'__'+key] = value
                detail['prediction'] = int(np.argmax(onset)) if gate else -1
            else:
                detail['reason'] = curve['reason']
            details[name] = detail
    return output, details
