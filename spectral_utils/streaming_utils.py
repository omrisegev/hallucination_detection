"""
Streaming / prefix evaluation of spectral hallucination detection.

Step 148 pilot: compute the spectral feature suite on growing prefixes of the
entropy trace H(n), fuse with continuous L-SML, and measure how detection
quality evolves as the trace unfolds ("how early can we know?").  Also
provides causal per-token monitor scores and the DeepConf-style
windowed-confidence baseline (arXiv:2508.15260) computed from the same trace.

Conventions:
    - All *scores* returned here are oriented so HIGHER = more likely CORRECT
      (confidence), matching the package-wide feature-sign convention.
    - Causal trajectories are arrays aligned to token index t (0-based) where
      entry t uses only e[:t+1].
"""
import numpy as np

from .feature_utils import extract_all_features


# Sign orientation for the 16 H(n) features: +1 = higher feature value →
# more likely correct.  Mirrors scripts/run_upcr_comparison.py (Step 141+).
FEATURE_SIGNS = {
    "epr": -1,
    "trace_length": 1,
    "spectral_entropy": -1,
    "low_band_power": -1,
    "high_band_power": -1,
    "hl_ratio": -1,
    "dominant_freq": -1,
    "spectral_centroid": -1,
    "stft_max_high_power": -1,
    "stft_spectral_entropy": -1,
    "rpdi": -1,
    "sw_var_peak": -1,
    "pe_mean": -1,
    "hurst_exponent": 1,
    "cusum_max": -1,
    "cusum_shift_idx": 1,
}


# ---------------------------------------------------------------------------
# Raw-cache loading
# ---------------------------------------------------------------------------

def iter_entropy_traces(cache_obj):
    """
    Yield (ents: np.ndarray, label: int) from any known raw-cache schema.

    Handles:
        - list of per-sample dicts with 'token_entropies'/'main_entropies'
          + 'label'/'correct'
        - list of per-sample dicts with 'traces' + 'corrects' (K>1 sampling
          caches: each of the K generations yields its own trace/label pair)
        - dict keyed by integer sample index over such dicts (Phase-1/2
          inference caches)
        - dict wrapping such a list under 'results' / 'samples' / 'data'
    """
    if isinstance(cache_obj, dict):
        keys = list(cache_obj)
        if keys and all(isinstance(k, (int, np.integer)) for k in keys):
            cache_obj = [cache_obj[k] for k in sorted(keys)]
        else:
            for key in ("results", "samples", "data"):
                if isinstance(cache_obj.get(key), list):
                    cache_obj = cache_obj[key]
                    break
            else:
                raise ValueError(
                    f"Unrecognised cache schema: dict with keys {keys[:8]}"
                )

    for s in cache_obj:
        if not isinstance(s, dict):
            continue
        ents = s.get("token_entropies")
        if ents is None:
            ents = s.get("main_entropies")
        label = s.get("label")
        if label is None:
            label = s.get("correct")
        if ents is not None and label is not None:
            ents = np.asarray(ents, dtype=float)
            if len(ents) > 0:
                yield ents, int(bool(label))
        elif s.get("traces") is not None and s.get("corrects") is not None:
            for tr, c in zip(s["traces"], s["corrects"]):
                ents = np.asarray(tr, dtype=float)
                if len(ents) > 0:
                    yield ents, int(bool(c))


# ---------------------------------------------------------------------------
# Prefix features
# ---------------------------------------------------------------------------

def prefix_features(ents, n: int) -> dict | None:
    """
    Full 16-feature extraction on the first n tokens of the trace.

    n is clipped to the trace length (a trace that ended before n is simply
    complete — that is the honest online situation).  Returns None when the
    prefix is too short for spectral analysis (< 8 tokens), same contract as
    extract_all_features.
    """
    e = np.asarray(ents, dtype=float)
    return extract_all_features(e[: max(1, int(n))])


def prefix_feature_matrix(traces, n: int, feat_names: list):
    """
    Prefix features for a cohort of traces at one token budget.

    Args:
        traces:     list of 1-D entropy arrays.
        n:          token budget (each trace contributes h[:min(n, len)]).
        feat_names: feature keys to collect.

    Returns:
        (feats_dict, valid_mask) where feats_dict maps name -> np.ndarray over
        the valid traces only, and valid_mask is a bool array over the input
        order (False = prefix too short for feature extraction).
    """
    rows, valid = [], []
    for tr in traces:
        fd = prefix_features(tr, n)
        valid.append(fd is not None)
        if fd is not None:
            rows.append([fd[f] for f in feat_names])
    valid = np.asarray(valid, dtype=bool)
    if not rows:
        return {f: np.array([]) for f in feat_names}, valid
    mat = np.asarray(rows, dtype=float)
    return {f: mat[:, i] for i, f in enumerate(feat_names)}, valid


# ---------------------------------------------------------------------------
# DeepConf-style windowed-confidence baseline
# ---------------------------------------------------------------------------

def deepconf_lowest_group_conf(ents, window: int = 64) -> float:
    """
    DeepConf "lowest group confidence" analog on the entropy trace.

    Token confidence proxy = -H(t); group confidence = mean over a sliding
    window (stride 1); trace score = minimum group confidence.  Higher =
    more confident = more likely correct.  Traces shorter than the window
    fall back to the whole-trace mean.
    """
    e = np.asarray(ents, dtype=float)
    if len(e) < window:
        return float(-e.mean())
    c = np.concatenate([[0.0], np.cumsum(e)])
    win_means = (c[window:] - c[:-window]) / window
    return float(-win_means.max())


def deepconf_tail_conf(ents, window: int = 64) -> float:
    """Mean confidence of the final `window` tokens (DeepConf tail statistic)."""
    e = np.asarray(ents, dtype=float)
    return float(-e[-window:].mean())


# ---------------------------------------------------------------------------
# Causal per-token monitor trajectories
# ---------------------------------------------------------------------------

def causal_trajectories(ents, window: int = 64, sw_window: int = 16) -> dict:
    """
    Per-token causal risk trajectories (entry t uses only e[:t+1]).

    All trajectories are RISK-oriented: higher = more hallucination-suspect.
    An online monitor flags a trace the first time a trajectory crosses a
    threshold.

    Returns dict of np.ndarray, each of length len(ents):
        run_mean_ent   — running mean entropy
        run_max_ent    — running max entropy
        cusum          — streaming CUSUM: running max of |s_t|,
                         s_t = s_{t-1} + (e_t - running_mean_t)
        sw_var_sofar   — running max of trailing-window variance (sw_window)
        neg_group_conf — running max of trailing-window mean entropy (window),
                         i.e. the negative of DeepConf's
                         lowest-group-confidence-so-far
    """
    e = np.asarray(ents, dtype=float)
    n = len(e)
    t_idx = np.arange(1, n + 1, dtype=float)

    csum = np.cumsum(e)
    run_mean = csum / t_idx
    run_max = np.maximum.accumulate(e)

    # streaming CUSUM against the running mean
    s = np.empty(n)
    acc = 0.0
    for t in range(n):
        acc += e[t] - run_mean[t]
        s[t] = acc
    cusum = np.maximum.accumulate(np.abs(s))

    # trailing-window mean and variance (partial windows at the start)
    c1 = np.concatenate([[0.0], csum])
    c2 = np.concatenate([[0.0], np.cumsum(e ** 2)])
    starts_w = np.maximum(0, np.arange(n) - window + 1)
    lens_w = np.arange(n) + 1 - starts_w
    trail_mean = (c1[np.arange(n) + 1] - c1[starts_w]) / lens_w
    starts_v = np.maximum(0, np.arange(n) - sw_window + 1)
    lens_v = np.arange(n) + 1 - starts_v
    m_v = (c1[np.arange(n) + 1] - c1[starts_v]) / lens_v
    trail_var = np.maximum(
        0.0, (c2[np.arange(n) + 1] - c2[starts_v]) / lens_v - m_v ** 2
    )

    return {
        "run_mean_ent": run_mean,
        "run_max_ent": run_max,
        "cusum": cusum,
        "sw_var_sofar": np.maximum.accumulate(trail_var),
        "neg_group_conf": np.maximum.accumulate(trail_mean),
    }


# ---------------------------------------------------------------------------
# Earliness / online-monitor metrics (label-free protocol: final-answer labels)
# ---------------------------------------------------------------------------

def earliness_index(budgets, aurocs, full_auroc: float, frac: float = 0.95):
    """
    Smallest budget whose AUROC reaches frac * full_auroc.

    Returns (budget, auroc_at_budget) or (None, None) if never reached.
    """
    target = frac * full_auroc
    for b, a in zip(budgets, aurocs):
        if a is not None and not np.isnan(a) and a >= target:
            return b, a
    return None, None


def online_flag_curve(risk_trajs, labels, n_thresholds: int = 60):
    """
    Threshold sweep for an online monitor over per-token risk trajectories.

    A trace is flagged at the first token where its risk trajectory crosses
    the threshold.  Labels follow the package convention (1 = correct answer,
    0 = hallucination), so detections are flags on label-0 traces and false
    alarms are flags on label-1 traces.

    Args:
        risk_trajs: list of 1-D risk arrays (higher = more suspect).
        labels:     array-like of {0,1} final-answer correctness.
        n_thresholds: number of thresholds swept over the pooled quantile range.

    Returns list of dicts, one per threshold:
        thr, false_alarm_rate, detection_rate,
        median_flag_token / median_flag_frac (over detected hallucinations),
        frac_tokens_saved (tokens after the flag on flagged wrong traces,
        as a fraction of ALL wrong-trace tokens — the early-exit value).
    """
    labels = np.asarray(labels, dtype=int)
    peak = np.array([tr.max() for tr in risk_trajs])
    lens = np.array([len(tr) for tr in risk_trajs], dtype=float)
    thrs = np.quantile(peak, np.linspace(0.02, 0.98, n_thresholds))

    wrong = labels == 0
    total_wrong_tokens = lens[wrong].sum()
    out = []
    for thr in np.unique(thrs):
        flag_tok = np.full(len(risk_trajs), -1)
        for i, tr in enumerate(risk_trajs):
            hits = np.nonzero(tr >= thr)[0]
            if len(hits):
                flag_tok[i] = hits[0]
        flagged = flag_tok >= 0
        fa = float(flagged[~wrong].mean()) if (~wrong).any() else np.nan
        det = float(flagged[wrong].mean()) if wrong.any() else np.nan
        det_toks = flag_tok[wrong & flagged]
        det_lens = lens[wrong & flagged]
        saved = float((det_lens - det_toks).sum() / (total_wrong_tokens + 1e-12))
        out.append({
            "thr": float(thr),
            "false_alarm_rate": fa,
            "detection_rate": det,
            "median_flag_token": float(np.median(det_toks)) if len(det_toks) else np.nan,
            "median_flag_frac": float(np.median(det_toks / det_lens)) if len(det_toks) else np.nan,
            "frac_tokens_saved": saved,
        })
    return out
