"""Post-evaluation geometry diagnostics; never used to fit a localizer."""
import numpy as np


def character_alignment(steps, offsets):
    """Independent character-overlap construction on one encoded answer."""
    positions = []
    cursor = 0
    for i, text in enumerate(steps):
        if i:
            cursor += 2
        positions.append((cursor, cursor + len(text)))
        cursor += len(text)
    result = []
    for lo, hi in positions:
        indices = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo and b > a]
        result.append([indices[0], indices[-1] + 1] if indices else None)
    return positions, result


def direct_token_projection(starts, ends, values, token_count):
    starts, ends, values = np.asarray(starts), np.asarray(ends), np.asarray(values, float)
    if starts.shape != ends.shape or values.shape != starts.shape or values.ndim != 1:
        raise ValueError('ONE_SCORE_PER_WINDOW_REQUIRED')
    if not np.isfinite(values).all() or np.any(starts < 0) or np.any(ends > token_count) or np.any(ends <= starts):
        raise ValueError('INVALID_WINDOWS')
    total = np.zeros(token_count)
    count = np.zeros(token_count, dtype=int)
    for a, b, value in zip(starts, ends, values):
        total[a:b] += value
        count[a:b] += 1
    if not (count > 0).all():
        raise ValueError('UNSCORED_TOKEN_GAP')
    return total / count


def peak_geometry(starts, ends, values, step_starts, step_ends, token_count, saved_steps):
    """Keep stored argmax; identify ties and their shared measurement support."""
    starts, ends = np.asarray(starts), np.asarray(ends)
    ss, ee, stored = np.asarray(step_starts), np.asarray(step_ends), np.asarray(saved_steps, float)
    if ss.shape != ee.shape or ss.shape != stored.shape or ss.ndim != 1 or not len(ss):
        raise ValueError('MALFORMED_STEPS')
    if np.any(ss < 0) or np.any(ee > token_count) or np.any(ee <= ss):
        raise ValueError('INVALID_STEP_SPANS')
    token = direct_token_projection(starts, ends, values, token_count)
    replay = np.array([token[a:b].max() for a, b in zip(ss, ee)])
    np.testing.assert_allclose(stored, replay, rtol=1e-12, atol=1e-12)
    peak = int(np.argmax(stored))
    tol = 1e-12 * max(1., abs(float(stored[peak])))
    exact = np.flatnonzero(stored == stored[peak]).tolist()
    near = np.flatnonzero(stored[peak] - stored <= tol).tolist()
    support_steps = {}
    peak_tokens = []
    for i in near:
        ids = np.arange(ss[i], ee[i])[np.abs(token[ss[i]:ee[i]] - stored[peak]) <= tol]
        peak_tokens.extend(ids.tolist())
        for t in ids:
            signature = tuple(np.flatnonzero((starts <= t) & (ends > t)).tolist())
            support_steps.setdefault(signature, set()).add(i)
    plateaus = [{'windows': list(k), 'steps': sorted(v)} for k, v in sorted(support_steps.items()) if len(v) > 1]
    ordered = np.sort(stored)
    margin = float(ordered[-1] - ordered[-2]) if len(stored) > 1 else None
    overlaps = [[i for i, (a, b) in enumerate(zip(ss, ee)) if a < hi and b > lo] for lo, hi in zip(starts, ends)]
    return {'peak': peak, 'step_scores': stored.tolist(), 'exact_top_steps': exact, 'numerical_top_steps': near,
            'top_two_margin': margin, 'margin_le_0_1': margin is not None and margin <= .1,
            'shared_peak_plateaus': plateaus, 'peak_token_indices': sorted(set(peak_tokens)),
            'window_counts_per_step': [sum(i in v for v in overlaps) for i in range(len(ss))],
            'cross_step_windows': [i for i, v in enumerate(overlaps) if len(v) > 1],
            'step_lengths': (ee - ss).tolist(), 'max_replay_discrepancy': float(np.max(np.abs(replay-stored)))}


def pb_outcome(target, prediction, peak, numerical_top_steps):
    """A disjoint native-decision taxonomy; separate raw-peak diagnostics."""
    clean = target == -1
    if clean:
        category = 'clean_correct' if prediction == -1 else 'clean_false_alarm'
    elif prediction == target:
        category = 'error_exact'
    elif prediction == -1:
        category = 'error_gate_closed'
    else:
        category = 'error_wrong_step'
    return {'category': category, 'correct': prediction == target,
            'raw_peak_exact': None if clean else peak == target,
            'raw_peak_before': None if clean else peak < target,
            'raw_peak_after': None if clean else peak > target,
            'target_in_top_tie': None if clean else target in numerical_top_steps,
            'exact_peak_hidden': not clean and peak == target and prediction == -1}
