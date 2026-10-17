"""Core uncertainty estimation metrics."""

import numpy as np


def calculate_ema_numpy(data, span=5):
    """Calculate Exponential Moving Average (EMA) smoothing."""
    values = np.array(data)
    if len(values) == 0:
        return np.array([])
    alpha = 2 / (span + 1)
    ema_values = np.zeros_like(values)
    ema_values[0] = values[0]
    for t in range(1, len(values)):
        ema_values[t] = alpha * values[t] + (1 - alpha) * ema_values[t-1]
    return ema_values


def get_logprob_curve(logprobs_seq):
    """
    Extract token-level log probabilities from vLLM output.
    Takes the max logprob at each step (greedy decoding confidence).
    """
    if not logprobs_seq:
        return []
    curve = []
    for step_data in logprobs_seq:
        if step_data is None:
            continue
        step_logprob = max(obj.logprob for obj in step_data.values())
        curve.append(step_logprob)
    return curve


def calculate_running_mean(curve):
    """
    Calculate cumulative moving average (running mean).
    Physical meaning: at step t, what is the average confidence of the sequence so far.
    """
    data = np.array(curve)
    if len(data) == 0:
        return np.array([])

    cumsum = np.cumsum(data)
    indices = np.arange(1, len(data) + 1)

    return cumsum / indices


def calculate_risk_baseline_mean(raw_curve):
    """
    LN_E (Length-Normalized Entropy) baseline method.

    Logic:
    1. No EMA smoothing.
    2. No Drop detection.
    3. Directly compute arithmetic mean of logprobs (equivalent to geometric mean of probabilities).
    4. Negate (Risk definition: lower score means higher risk).
    """
    curve = np.array(raw_curve)
    if len(curve) == 0:
        return 0.0

    avg_val = np.mean(curve)
    return -float(avg_val)


def calculate_risk_with_running_mean_drop(raw_curve, ema_span=5, drop_k=5):
    """
    LN_E_drop method: Running Mean + EMA + Drop detection.

    Logic:
    1. Extract logprobs.
    2. Compute running mean (transform average score into a time-varying curve).
    3. Apply EMA smoothing to this average curve.
    4. Find maximum drop in the curve -> indicates newly generated token significantly lowered quality.
    """
    curve = np.array(raw_curve)
    if len(curve) < 2:
        return 0.0

    mean_curve = calculate_running_mean(curve)
    smooth_curve = calculate_ema_numpy(mean_curve, span=ema_span)
    diffs = np.diff(smooth_curve)
    if len(diffs) == 0:
        return 0.0

    drops = diffs[diffs < 0]
    if len(drops) == 0:
        return 0.0

    sorted_drops = np.sort(drops)
    worst_drops = sorted_drops[:drop_k]

    risk = -float(np.mean(worst_drops))
    if np.isnan(risk) or np.isinf(risk):
        risk = 0.0
    return risk
