"""
Spectral feature extraction from token-level entropy traces H(n).

All 12 features used across Phase 4 / 5 / 6 / 7:
    epr, trace_length,
    spectral_entropy, low_band_power, high_band_power, hl_ratio,
    dominant_freq, spectral_centroid,
    stft_max_high_power, stft_spectral_entropy,
    rpdi, sw_var_peak
"""
import numpy as np
from scipy.signal import stft as scipy_stft


FEAT_NAMES = [
    "epr", "trace_length",
    "spectral_entropy", "low_band_power", "high_band_power",
    "hl_ratio", "dominant_freq", "spectral_centroid",
    "stft_max_high_power", "stft_spectral_entropy",
    "rpdi", "sw_var_peak",
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


def extract_all_features(ents) -> dict | None:
    """
    Extract all 12 spectral features from a single entropy trace.

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
    return result
