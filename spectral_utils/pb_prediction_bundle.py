"""Complete PB reporting from a single final decision (no fitting or labels in gates)."""
from __future__ import annotations

import numpy as np

from .historical_fusion_evaluation import pb_metrics


def full_gate_from_pb_data(data, q=.33):
    """Expand finite PB-only midranks to the full frozen answer roster."""
    mask = np.asarray(data["pb"], bool)
    ranks = np.asarray(data["tail_top10"], float)
    if ranks.shape != (int(mask.sum()),) or not np.isfinite(ranks).all():
        raise ValueError("PB calibration scores are missing or misaligned")
    opened = np.zeros(len(mask), bool)
    opened[mask] = ranks >= q
    return opened


def prediction_bundle(target, cells, peak, score_valid, gate_open, gate_valid=None):
    """Return all PB metrics and predictions; invalid decisions never count clean.

    Inputs are answer-aligned, including optional non-PB rows. ``gate_open``
    must be computed without correctness labels before calling this evaluator.
    Raw-location counts deliberately precede gating; final counts use validity.
    """
    target = np.asarray(target)
    cells = np.asarray(cells, dtype=str)
    peak = np.asarray(peak)
    score_valid = np.asarray(score_valid, dtype=bool)
    gate_open = np.asarray(gate_open, dtype=bool)
    gate_valid = np.ones(len(target), bool) if gate_valid is None else np.asarray(gate_valid, bool)
    if any(x.shape != target.shape for x in (cells, peak, score_valid, gate_open, gate_valid)) or target.ndim != 1:
        raise ValueError("PB inputs must be aligned one-dimensional arrays")
    if np.any(score_valid & (peak < 0)):
        raise ValueError("valid locator must have a nonnegative step index")
    pb = np.char.startswith(cells, "pb_")
    if np.any(pb & (target < -1)):
        raise ValueError("invalid PB target")
    decision_valid = score_valid & gate_valid
    prediction = np.where(decision_valid & gate_open, peak, -1)
    result = pb_metrics(target[pb], prediction[pb], decision_valid[pb], cells[pb])
    error, clean = pb & (target >= 0), pb & (target == -1)
    raw_hit = error & score_valid & (peak == target)
    exact = error & decision_valid & (prediction == target)
    suppressed = raw_hit & decision_valid & ~gate_open
    invalidated = raw_hit & ~decision_valid
    n_error, n_clean = int(error.sum()), int(clean.sum())
    rate = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    metrics = dict(
        pb_all8=result["macros"]["all"], pb_q4=result["macros"]["q4"], pb_q8=result["macros"]["q8"],
        pb_cells=result["cells"], pb_clean_accuracy=rate(np.sum(clean & decision_valid & (prediction == -1)), n_clean),
        pb_error_exact_accuracy=rate(exact.sum(), n_error), pb_raw_exact=rate(raw_hit.sum(), n_error),
        pb_exact_count=int(raw_hit.sum()), pb_final_exact_count=int(exact.sum()),
        pb_correct_peaks_suppressed=int(suppressed.sum()), pb_correct_peaks_invalidated=int(invalidated.sum()),
        pb_early=int(np.sum(error & score_valid & (peak < target))),
        pb_late=int(np.sum(error & score_valid & (peak > target))),
        pb_error_count=n_error, pb_clean_count=n_clean, pb_invalid=int(np.sum(pb & ~decision_valid)),
        pb_score_invalid=int(np.sum(pb & ~score_valid)),
        pb_clean_success_count=int(np.sum(clean & decision_valid & (prediction == -1))),
        pb_gate_open_count=int(np.sum(pb & gate_valid & gate_open)),
    )
    if metrics["pb_exact_count"] != metrics["pb_final_exact_count"] + metrics["pb_correct_peaks_suppressed"] + metrics["pb_correct_peaks_invalidated"]:
        raise AssertionError("raw/final/suppressed/invalidated peak counts disagree")
    if n_error != metrics["pb_exact_count"] + metrics["pb_early"] + metrics["pb_late"] + int(np.sum(error & ~score_valid)):
        raise AssertionError("raw localization counts disagree")
    for panel, key in (("all", "pb_all8"), ("q4", "pb_q4"), ("q8", "pb_q8")):
        values = [v["f1"] for c, v in metrics["pb_cells"].items() if panel == "all" or c.endswith(panel)]
        if values and all(v is not None for v in values):
            np.testing.assert_allclose(metrics[key], np.mean(values), atol=1e-14, rtol=0)
    return metrics, prediction, decision_valid
