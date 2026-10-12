"""Fixed-threshold no-error gate on raw answer-level telemetry, applied to frozen peaks.

Motivation (2026-09-08): the answer-only fusion pipeline z-scores every feature
inside the answer, so its fused step risks carry no cross-answer scale. On the
full ProcessBench population the answer-level max/mean of those risks separates
erroneous from clean answers at AUC ~0.48 (chance). Raw, un-normalized
telemetry summaries of the same answer (mean token entropy, max 8-token window
mean entropy, top-k tail mass) separate them at AUC 0.74-0.78 in every cell.

This module therefore evaluates the simplest possible repair: keep the frozen
answer-only peak (argmax fused step risk) and replace the per-answer GMM/BIC
mixture gate by ``detector >= threshold`` where ``detector`` is one raw
answer-level summary and ``threshold`` is a single constant. The constant may
be chosen (a) with labels on training source-group folds (nested, mirrors the
historical comparator protocol), (b) without labels as a quantile of the
training-fold detector distribution, or (c) on one dataset family and
transferred to another. None of these touch the fusion weights or the peaks.

Everything here consumes frozen benchmark outputs only; no new fits.
"""
from __future__ import annotations

import numpy as np

from .historical_fusion_evaluation import pb_metrics

PB_MACROS = ("q4", "q8", "all")

# Raw token telemetry column contract of results/localization_full_benchmark_v3/inputs/*/raw.npy.
# Copied from answer_localization_v2.STREAM_NAMES (that module imports the frozen
# short-cycle joint_lsml code, which is not importable from this checkout).
STREAM_NAMES = (
    "trace_length_series", "entropy_series", "entropy_rolling_spectral_entropy",
    "entropy_rolling_low_band_power", "entropy_rolling_high_band_power",
    "entropy_rolling_hl_ratio", "entropy_rolling_dominant_freq",
    "entropy_rolling_spectral_centroid", "entropy_stft_high_series",
    "entropy_stft_frame_entropy", "entropy_rolling_tail_ratio",
    "entropy_sw_var_series", "entropy_pe_series", "entropy_rolling_rs_hurst",
    "entropy_cusum_abs_series", "spilled_series", "spilled_sw_var_series",
    "spilled_cusum_abs_series", "spilled_rolling_min", "energy_series",
    "energy_rolling_min", "energy_sw_var_series", "energy_cusum_abs_series",
    "top1_logprob_series", "logprob_margin_series", "topk_entropy_series",
    "topk_varentropy_series", "topk_renyi2_series", "topk_tail_mass_series",
)


def rolling_max_mean(x: np.ndarray, width: int = 8) -> float:
    x = np.asarray(x, float)
    if len(x) < width:
        return float(x.mean())
    c = np.cumsum(np.r_[0.0, x])
    return float(((c[width:] - c[:-width]) / width).max())


DETECTORS = {
    # name: (stream, summary). Higher detector value => more likely erroneous.
    "entropy_mean": ("entropy_series", "mean"),
    "entropy_w8max": ("entropy_series", "w8max"),
    "topk_tail_mass_mean": ("topk_tail_mass_series", "mean"),
    "topk_varentropy_w8max": ("topk_varentropy_series", "w8max"),
}


def summarize(x: np.ndarray, how: str) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    if how == "mean":
        return float(x.mean())
    if how == "w8max":
        return rolling_max_mean(x, 8)
    if how == "p90":
        return float(np.percentile(x, 90))
    raise ValueError(how)


def gate_predictions(detector, threshold, peak, valid):
    """Predict the frozen peak when detector >= threshold, else no error (-1).

    Rows without a valid frozen peak or a finite detector stay invalid
    (prediction -1 with valid False), so they count as failures downstream.
    """
    detector = np.asarray(detector, float)
    peak = np.asarray(peak)
    valid = np.asarray(valid, bool) & np.isfinite(detector)
    prediction = np.where(valid & (detector >= threshold), peak, -1)
    return prediction, valid


def macro_all(target, prediction, valid, cells):
    return pb_metrics(target, prediction, valid, cells)["macros"]["all"]


def calibrate_threshold(detector, peak, valid, target, cells, grid_points=99):
    """Label-using threshold: maximize eight-cell macro F1 on the given rows."""
    detector = np.asarray(detector, float)
    eligible = np.asarray(valid, bool) & np.isfinite(detector)
    grid = np.quantile(detector[eligible], np.linspace(0.01, 0.99, grid_points))
    best, best_t = -np.inf, None
    curve = []
    for t in grid:
        pred, v = gate_predictions(detector, t, peak, valid)
        value = macro_all(target, pred, v, cells)
        value = -np.inf if value is None else value
        curve.append(value)
        if value > best:
            best, best_t = value, float(t)
    return dict(threshold=best_t, training_macro=float(best), grid=grid.tolist(),
                curve=[float(c) for c in curve], labels_used=True)


def quantile_threshold(detector, valid, q):
    """Label-free threshold: the q-quantile of the training-row detector."""
    detector = np.asarray(detector, float)
    eligible = np.asarray(valid, bool) & np.isfinite(detector)
    return float(np.quantile(detector[eligible], q))


def nested_gate(detector, peak, valid, target, cells, outer_fold, rule, **kw):
    """Apply a threshold rule fold by fold: choose on training folds, apply to the held-out fold.

    rule: 'labels' -> calibrate_threshold on training rows;
          'quantile' -> quantile_threshold(q=kw['q']) on training rows.
    """
    outer_fold = np.asarray(outer_fold)
    prediction = np.full(len(detector), -1, dtype=int)
    valid_out = np.zeros(len(detector), bool)
    thresholds = {}
    for k in sorted(set(outer_fold[outer_fold >= 0].tolist())):
        train = outer_fold != k
        test = outer_fold == k
        if rule == "labels":
            t = calibrate_threshold(detector[train], peak[train], valid[train], target[train], cells[train])["threshold"]
        elif rule == "quantile":
            t = quantile_threshold(detector[train], valid[train], kw["q"])
        else:
            raise ValueError(rule)
        p, v = gate_predictions(detector[test], t, peak[test], valid[test])
        prediction[test], valid_out[test] = p, v
        thresholds[int(k)] = float(t)
    return prediction, valid_out, thresholds


def oracle_gate(target, peak, valid):
    """Label-using upper bound: perfect error/no-error knowledge with the same frozen peaks."""
    target = np.asarray(target)
    valid = np.asarray(valid, bool)
    prediction = np.where(valid & (target >= 0), np.asarray(peak), -1)
    return prediction, valid


def paired_group_bootstrap(target, cells, groups, valid_a, pred_a, valid_b, pred_b,
                           draws=1000, seed=2026090707):
    """Paired source-group bootstrap of macro-all F1 difference (a minus b)."""
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        w = counts[inverse].astype(float)
        a = pb_metrics(target, pred_a, valid_a, cells, weights=w)["macros"]["all"]
        b = pb_metrics(target, pred_b, valid_b, cells, weights=w)["macros"]["all"]
        if a is None or b is None:
            continue
        diffs.append(a - b)
    diffs = np.asarray(diffs)
    return dict(draws=int(len(diffs)), mean=float(diffs.mean()),
                ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])


def error_modes(target, prediction, peak, valid, cells):
    """Descriptive counts per PB cell: raw peak hits, suppressed hits, false alarms."""
    target, prediction, peak = (np.asarray(x) for x in (target, prediction, peak))
    valid = np.asarray(valid, bool)
    out = {}
    for cell in sorted(set(cells)):
        if not str(cell).startswith("pb_"):
            continue
        m = np.asarray(cells) == cell
        clean, err = m & (target == -1), m & (target >= 0)
        hit = valid & (prediction == target)
        peak_hit = err & valid & (peak == target)
        out[cell] = dict(
            erroneous=int(err.sum()), clean=int(clean.sum()),
            raw_peak_exact=int(peak_hit.sum()),
            final_exact=int(hit[err].sum()),
            exact_peak_suppressed=int((peak_hit & (prediction == -1)).sum()),
            error_called_clean=int((err & valid & (prediction == -1)).sum()),
            clean_correct=int(hit[clean].sum()),
            clean_false_alarm=int((clean & valid & (prediction != -1)).sum()),
        )
    return out


def self_test():
    t = np.array([-1, -1, 0, 1, 2])
    pk = np.array([0, 1, 0, 1, 0])
    d = np.array([0.1, 0.9, 0.8, 0.7, np.nan])
    p, v = gate_predictions(d, 0.75, pk, np.ones(5, bool))
    assert p.tolist() == [-1, 1, 0, -1, -1] and v.tolist() == [True, True, True, True, False]
    p, v = oracle_gate(t, pk, np.ones(5, bool))
    assert p.tolist() == [-1, -1, 0, 1, 0]
    cells = np.array(["pb_x_q8"] * 5)
    assert abs(macro_all(t, p, v, cells) - (2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))) < 1e-12
    return True
