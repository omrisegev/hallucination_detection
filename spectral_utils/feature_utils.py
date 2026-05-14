"""
Spectral feature extraction from token-level entropy traces H(n).

All 17 features used across Phase 4 / 5 / 6 / 7 / C:
    epr, trace_length,
    spectral_entropy, low_band_power, high_band_power, hl_ratio,
    dominant_freq, spectral_centroid,
    stft_max_high_power, stft_spectral_entropy,
    rpdi, sw_var_peak,
    pe_min, pe_mean, hurst_exponent,
    cusum_max, cusum_shift_idx,
    segment_by_citations
"""
import re
import math
import numpy as np
from scipy.signal import stft as scipy_stft


FEAT_NAMES = [
    "epr", "trace_length",
    "spectral_entropy", "low_band_power", "high_band_power",
    "hl_ratio", "dominant_freq", "spectral_centroid",
    "stft_max_high_power", "stft_spectral_entropy",
    "rpdi", "sw_var_peak",
    "pe_min", "pe_mean", "hurst_exponent",
    "cusum_max", "cusum_shift_idx",
]


def compute_spectral_features(ents, min_len: int = 8) -> dict | None:
    """
    Global FFT-based spectral features of the mean-centered entropy trace.

    Returns None if the trace is shorter than min_len.
    """
    e = np.array(ents, dtype=float)
    N = len(e)
    if N < min_len:
        return None

    e_ac     = e - e.mean()
    fft_vals = np.fft.rfft(e_ac)
    psd      = np.abs(fft_vals) ** 2
    freqs    = np.fft.rfftfreq(N)
    psd_sum  = psd.sum() + 1e-12
    psd_norm = psd / psd_sum

    spec_ent = -np.sum(psd_norm * np.log(psd_norm + 1e-12))

    low_mask   = (freqs > 0.0) & (freqs <= 0.10)
    high_mask  = (freqs >= 0.40) & (freqs <= 0.50)
    low_power  = psd[low_mask].sum()  / psd_sum
    high_power = psd[high_mask].sum() / psd_sum
    hl_ratio   = high_power / (low_power + 1e-12)

    ac_mask  = freqs > 0
    dom_freq = float(freqs[ac_mask][np.argmax(psd[ac_mask])]) if ac_mask.sum() > 0 else 0.0
    centroid = (
        float(np.sum(freqs[ac_mask] * psd_norm[ac_mask]) /
              (psd_norm[ac_mask].sum() + 1e-12))
        if ac_mask.sum() > 0 else 0.0
    )

    return {
        "spectral_entropy": float(spec_ent),
        "low_band_power":   float(low_power),
        "high_band_power":  float(high_power),
        "hl_ratio":         float(hl_ratio),
        "dominant_freq":    dom_freq,
        "spectral_centroid": centroid,
    }


def compute_stft_features(ents, nperseg: int = 16, noverlap: int = 8,
                          min_len: int = 32) -> dict:
    """
    Short-Time Fourier Transform features (time-local spectral structure).

    Returns zero-valued features if the trace is too short for STFT.
    """
    e = np.array(ents, dtype=float)
    if len(e) < min_len:
        return {"stft_max_high_power": 0.0, "stft_spectral_entropy": 0.0}

    e_ac = e - e.mean()
    f, _t, Zxx = scipy_stft(e_ac, nperseg=nperseg, noverlap=noverlap)
    psd = np.abs(Zxx) ** 2

    high_mask = f >= 0.40
    if high_mask.sum() > 0 and psd.shape[1] > 0:
        high_frac      = psd[high_mask].sum(0) / (psd.sum(0) + 1e-12)
        max_local_high = float(high_frac.max())
    else:
        max_local_high = 0.0

    psd_n     = psd / (psd.sum(0, keepdims=True) + 1e-12)
    frame_ent = -np.sum(psd_n * np.log(psd_n + 1e-12), axis=0)
    stft_ent  = float(frame_ent.mean()) if len(frame_ent) > 0 else 0.0

    return {"stft_max_high_power": max_local_high, "stft_spectral_entropy": stft_ent}


def compute_time_domain(ents, tail_frac: float = 0.20,
                        sw_window: int = 16, sw_step: int = 1) -> dict:
    """
    Time-domain features: tail ratio (rpdi) and sliding-window variance peak.

    Args:
        tail_frac:  Fraction of the trace used for the tail mean in rpdi.
        sw_window:  Sliding window size for sw_var_peak.
        sw_step:    Stride for the sliding window (1 = token-by-token).
    """
    e = np.array(ents, dtype=float)
    W    = max(1, int(len(e) * tail_frac))
    rpdi = float(e[-W:].mean() / (e.mean() + 1e-12))

    if len(e) >= sw_window:
        sw_vars     = [np.var(e[i : i + sw_window])
                       for i in range(0, len(e) - sw_window + 1, sw_step)]
        sw_var_peak = float(np.max(sw_vars))
    else:
        sw_var_peak = float(np.var(e))

    return {"rpdi": rpdi, "sw_var_peak": sw_var_peak}


def sw_var_peak_with_window(ents, sw_window: int, sw_step: int = 1) -> float:
    """
    Compute sw_var_peak for a specific window size (used in window ablation).

    Separating this from compute_time_domain allows re-computation for each
    candidate window without re-running the full feature extraction.
    """
    e = np.array(ents, dtype=float)
    if len(e) >= sw_window:
        sw_vars = [np.var(e[i : i + sw_window])
                   for i in range(0, len(e) - sw_window + 1, sw_step)]
        return float(np.max(sw_vars))
    return float(np.var(e))


def sw_var_peak_adaptive(ents, fraction: float = 0.10,
                         min_w: int = 3, max_w: int = 32,
                         sw_step: int = 1) -> float:
    """
    sw_var_peak with window proportional to trace length.

    Window = clip(int(len * fraction), min_w, max_w).
    For math traces (~1000 tokens) at fraction=0.10 → w≈32 (capped).
    For QA traces (~100 tokens) → w≈10, capturing local bursts without
    over-smoothing the way a fixed w=16 would.

    Args:
        fraction: target fraction of trace length to use as window.
        min_w:    minimum window (prevents w=1 on very short traces).
        max_w:    maximum window (prevents over-smoothing on long traces).
        sw_step:  sliding stride (1 = token-by-token).
    """
    e = np.array(ents, dtype=float)
    w = max(min_w, min(max_w, int(len(e) * fraction)))
    return sw_var_peak_with_window(ents, w, sw_step)


def permutation_entropy(ents, order=3, delay=1):
    """
    Calculate the Permutation Entropy of a 1D array.
    """
    x = np.array(ents)
    n = len(x)
    if n < order + (order - 1) * delay:
        return 0.0

    # Extract overlapping windows
    indices = np.arange(n - (order - 1) * delay)
    windows = np.array([x[indices + i * delay] for i in range(order)]).T

    # Find permutations
    perms = np.argsort(windows, axis=1)

    # Convert permutations to unique rows to count
    _, counts = np.unique(perms, axis=0, return_counts=True)
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log2(probs))
    # Normalize by log2(factorial(order))
    return max(0.0, float(pe / np.log2(math.factorial(order))))


def compute_permutation_entropy(ents, order=3, delay=1, window_size=10) -> dict:
    """
    Compute Sliding-Window Permutation Entropy.
    Returns min and mean PE.
    """
    e = np.array(ents)
    if len(e) < window_size:
        # Fallback to single PE of the whole trace
        pe = permutation_entropy(e, order, delay)
        return {"pe_min": float(pe), "pe_mean": float(pe)}

    pes = []
    for i in range(len(e) - window_size + 1):
        pes.append(permutation_entropy(e[i : i + window_size], order, delay))

    return {"pe_min": float(np.min(pes)), "pe_mean": float(np.mean(pes))}


def compute_hurst_exponent(ents) -> float:
    """
    Estimate Hurst Exponent using Rescaled Range (R/S) analysis.
    """
    x = np.array(ents)
    n = len(x)
    if n < 8:
        return 0.5  # Default to random walk for very short traces

    # We'll use a few scales: n, n/2, n/4... down to 8
    max_k = int(np.log2(n / 8))
    scales = [n // (2**i) for i in range(max_k + 1)]
    scales = sorted(list(set(scales)))  # Ensure unique and sorted

    rs_values = []
    for s in scales:
        num_chunks = n // s
        rs_chunks = []
        for i in range(num_chunks):
            chunk = x[i*s : (i+1)*s]
            if len(chunk) < 2:
                continue
            mean_adj = chunk - np.mean(chunk)
            cum_sum = np.cumsum(mean_adj)
            r = np.max(cum_sum) - np.min(cum_sum)
            s_dev = np.std(chunk) + 1e-12
            rs_chunks.append(r / s_dev)
        if rs_chunks:
            # Avoid log(0) if the signal is constant
            mean_rs = np.mean(rs_chunks)
            if mean_rs > 0:
                rs_values.append(mean_rs)
            else:
                # Remove this scale if it has no variation
                scales.remove(s)

    if len(rs_values) < 2:
        if not rs_values:
            return 0.0  # Constant signal
        # Single point estimation: R/S is roughly n^H
        return float(np.log(rs_values[0] + 1e-12) / np.log(n))

    # Linear fit of log(R/S) vs log(scales)
    coeffs = np.polyfit(np.log(scales), np.log(rs_values), 1)
    return float(coeffs[0])


def compute_cusum_residuals(ents) -> dict:
    """
    Compute CUSUM residuals to detect regime shifts.
    Returns max CUSUM and shift index.
    """
    e = np.array(ents)
    if len(e) == 0:
        return {"cusum_max": 0.0, "cusum_shift_idx": 0.0}

    # Mean-centered residuals
    residuals = e - np.mean(e)
    cusum = np.cumsum(residuals)

    abs_cusum = np.abs(cusum)
    max_idx = np.argmax(abs_cusum)
    cusum_max = abs_cusum[max_idx]
    shift_idx = max_idx / len(e)

    return {"cusum_max": float(cusum_max), "cusum_shift_idx": float(shift_idx)}


def extract_all_features(ents) -> dict | None:
    """
    Extract all 17 spectral features from a single entropy trace.

    Returns None if the trace is too short for reliable spectral analysis.
    Uses the default sw_window=16, sw_step=1. For window ablation, call
    sw_var_peak_with_window() and override the 'sw_var_peak' key.
    """
    e      = np.array(ents, dtype=float)
    result = {"epr": float(e.mean()), "trace_length": float(len(e))}

    gf = compute_spectral_features(ents)
    if gf is None:
        return None
    result.update(gf)
    result.update(compute_stft_features(ents))
    result.update(compute_time_domain(ents))

    # Advanced features (Phase C)
    result.update(compute_permutation_entropy(ents))
    result.update({"hurst_exponent": compute_hurst_exponent(ents)})
    result.update(compute_cusum_residuals(ents))

    return result


def segment_by_citations(text: str, token_offsets: list) -> list:
    """
    Segment generated text into statement spans ending in citation markers [N] or [N, M].

    Each segment is mapped to its token index range using the provided char-level offsets.
    Returns list of {text, token_start, token_end, citation_ids}.

    Notes:
        token_offsets comes from generate_full's re-tokenization of full_text.
        Its length may differ from token_entropies by 1-2 tokens; callers should
        align lengths before slicing entropies with token_start/token_end.
    """
    cite_pattern = re.compile(r"\[\d+(?:[\s,\-]*\d+)*\]")
    segments  = []
    last_end  = 0

    for match in cite_pattern.finditer(text):
        start_char, end_char = match.span()
        seg_text = text[last_end:end_char].strip()

        t_start, t_end = -1, -1
        for i, (ts, te) in enumerate(token_offsets):
            if t_start == -1 and te > last_end:
                t_start = i
            if te <= end_char:
                t_end = i

        ids = [int(x) for x in re.findall(r"\d+", match.group())]

        if t_start != -1 and t_end != -1:
            segments.append({
                "text":        seg_text,
                "token_start": t_start,
                "token_end":   t_end + 1,
                "citation_ids": ids,
            })

        last_end = end_char

    return segments
