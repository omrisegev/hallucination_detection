"""Descriptive pilot/full decomposition of frozen localization decisions.

No fitting, thresholds, selection or new candidate scores. Every population
keeps invalid decisions as failures. Cross-population gaps are descriptive;
they are not paired treatment effects or estimates of their causal source.
"""
import numpy as np


def harmonic(clean_accuracy, error_accuracy):
    if clean_accuracy is None or error_accuracy is None:
        return None
    return (2 * clean_accuracy * error_accuracy / (clean_accuracy + error_accuracy)
            if clean_accuracy + error_accuracy else 0.0)


def outcome_summary(target, prediction, peak, score_valid, decision_valid):
    target, prediction, peak = (np.asarray(x) for x in (target, prediction, peak))
    score_valid, decision_valid = (np.asarray(x, bool) for x in (score_valid, decision_valid))
    assert all(x.shape == target.shape for x in (prediction, peak, score_valid, decision_valid))
    assert np.all(target >= -1) and not np.any(decision_valid & ~score_valid)
    assert np.all(~decision_valid | (prediction == -1) | (prediction == peak))
    clean, error = target == -1, target >= 0
    success = decision_valid & (prediction == target)
    peak_hit = error & score_valid & (peak == target)
    suppressed = peak_hit & decision_valid & (prediction == -1)
    peak_without_decision = peak_hit & ~decision_valid
    assert int(peak_hit.sum()) == int(success[error].sum() + suppressed.sum() + peak_without_decision.sum())
    ca = float(success[clean].mean()) if clean.any() else None
    ea = float(success[error].mean()) if error.any() else None
    return dict(answers=len(target), clean=int(clean.sum()), erroneous=int(error.sum()),
                valid_decisions=int(decision_valid.sum()), clean_correct=int(success[clean].sum()),
                error_correct=int(success[error].sum()), peak_correct=int(peak_hit.sum()),
                correct_peak_suppressed=int(suppressed.sum()),
                correct_peak_invalid_decision=int(peak_without_decision.sum()),
                clean_accuracy=ca, error_exact_accuracy=ea,
                raw_peak_accuracy=float(peak_hit.sum() / error.sum()) if error.any() else None,
                f1=harmonic(ca, ea))


def pb_panel(records, arrays, method_index, mask, scorer='q8'):
    cells = np.asarray([r['cell'] for r in records])
    chosen = sorted(c for c in set(cells) if c.startswith('pb_') and (scorer == 'all' or c.endswith(scorer)))
    result = {}
    for cell in chosen:
        ix = np.asarray(mask, bool) & (cells == cell)
        result[cell] = outcome_summary(arrays['target'][ix], arrays['predictions'][ix, method_index],
                                      arrays['peaks'][ix, method_index], arrays['valid'][ix, method_index],
                                      arrays['decision'][ix, method_index])
    values = [x['f1'] for x in result.values()]
    return dict(cells=result, macro_f1=float(np.mean(values)) if values and None not in values else None)


def support_summary(records, mask, scorer='q8'):
    rows = [r for r, keep in zip(records, mask) if keep and r['cell'].startswith('pb_') and r['cell'].endswith(scorer)]
    lengths = np.asarray([r['tokens'] for r in rows])
    return dict(answers=len(rows), token_bins={
        'below64': int((lengths < 64).sum()), '64_255': int(((lengths >= 64) & (lengths < 256)).sum()),
        '256_1023': int(((lengths >= 256) & (lengths < 1024)).sum()),
        '1024_2048': int(((lengths >= 1024) & (lengths <= 2048)).sum()),
        'above2048': int((lengths > 2048).sum())})
