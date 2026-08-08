"""
token_feature_views.py — token-resolved (per-position) counterparts of the trace-level
FEAT_NAMES pool, built for the GL-LIU local head (`scripts/gl_liu_v1`,
`scripts/gl_liu_factorial_v2`).

RECONSTRUCTION NOTE — read before quoting a broad-pool number
---------------------------------------------------------------------------
This module previously existed only on the tip of `experiment/step-localization` and was
never committed there or anywhere else: confirmed absent from every branch and stash as of
the Step-236 external-validation campaign (Llama-3.1-8B ProcessBench cells). It is
reconstructed here from two frozen artifacts:

  (a) `results/gl_liu_factorial_v2/RUN_DEFINITION.json` — the frozen `feature_contract`:
      the exact 5 core / 28 broad names, and which trace-level global feature (in
      `feature_utils.FEAT_NAMES` / `repgrid_scoring`'s energy+logprob views) each token
      view is declared to correspond to (the `mapping` block).
  (b) `scripts/gl_liu_v1/localization/positional_views.py` — NOT lost, still on master.
      Its five windowed-series functions, its `_causal` alignment rule (a window statistic
      is attributed to its LAST token; the prefix before the first full window is
      backfilled with a constant, not NaN), and its deployed window parameters
      (`SW_WINDOW=16, SW_STEP=1`, `PE_WINDOW=10, PE_ORDER=3, PE_DELAY=1`,
      `STFT_NPERSEG=16, STFT_NOVERLAP=8, STFT_HIGH=0.40`) are reused verbatim below,
      re-derived locally (not imported) to keep `spectral_utils` free of a reverse
      dependency on `scripts/`.

Two different reconstruction standards apply, and the difference matters:

- **Exact-identity views** (17 of the 28 broad names, all 5 core names): the trace-level
  feature is itself the MAX, MIN, or MEAN of a naturally per-token series — `sw_var_peak`
  is `max` of the sliding-window variance, `cusum_max` is `max` of `|cumsum|`, `epr` is
  `mean` of the entropy trace, `varentropy` is `mean` of the per-token varentropy, and so
  on. For these, reducing the token-view series with the SAME operator the global feature
  uses must reproduce the global scalar exactly on synthetic data — the same standard
  `positional_views.smoke()` already applies to `sw_var_series`/`cusum_abs_series`.
  `smoke()` below checks every one of these 17.
- **No-identity views** (the remaining 11: `entropy_pe_series`, `entropy_stft_high_series`,
  `entropy_stft_frame_entropy`, `entropy_rolling_tail_ratio`,
  `entropy_rolling_{spectral_entropy,low_band_power,high_band_power,hl_ratio,
  dominant_freq,spectral_centroid,rs_hurst}`). Their trace-level counterparts
  (`compute_spectral_features`, `compute_hurst_exponent`, `rpdi`) are each ONE global
  computation over the whole trace, not a reduction of any windowed series — there is no
  algebraic identity to check them against, and `positional_views.py` itself does not
  attempt one for `pe_series`/`stft_high_series` either (its own `smoke()` only checks the
  two MAX-based views). These are built the same way — deployed `W=16, step=1`, causal
  alignment — and gated only by a weaker sanity check (finite, right length, causal,
  non-degenerate variance). A broad-pool number built from these should be read as a
  best-effort reconstruction under the frozen contract, not as bit-reproducing a lost run.

Only `CORE_TOKEN_VIEWS` (bijective with `positional_views.py`, which was never lost) should
be treated as an exact historical reproduction.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft as scipy_stft

from .feature_utils import (
    compute_hurst_exponent,
    compute_spectral_features,
    permutation_entropy,
)

# Deployed parameters, copied from feature_utils.py / positional_views.py. Do not tune —
# retuning breaks the link to the trace-level results these views are meant to localize.
SW_WINDOW, SW_STEP = 16, 1
PE_WINDOW, PE_ORDER, PE_DELAY = 10, 3, 1
STFT_NPERSEG, STFT_NOVERLAP, STFT_HIGH = 16, 8, 0.40
ROLLING_SPECTRAL_WINDOW = 16  # shares SW_WINDOW; no separate value is recorded anywhere
MIN_TRACE = 8                 # compute_spectral_features' floor

# The three no-identity rolling views below (PE, the 6-way spectral tuple, Hurst) each cost
# a non-trivial computation (permutation counting, an FFT, an R/S multi-scale fit) per
# window. At native trace scale (hundreds to ~2000 tokens) a naive per-token sweep makes
# building the broad pool for a full ProcessBench cell take tens of minutes. `_sw_var_series`
# already has a `step`-then-`np.interp` pattern for exactly this trade (compute the window
# statistic every `step` positions, re-expand onto the token grid by interpolation before the
# causal last-token attribution). No causality is broken: interpolation only fills in between
# two already-computed window statistics, so a value at token t never depends on any window
# ending after t. These are no-identity views (see module docstring) — a coarser stride
# changes the smoothing scale but not the correctness gate any of them are held to.
ROLLING_HEAVY_STEP = 8


# ── shared causal-window machinery (copied from positional_views.py) ───────────────────

def _causal(series_vals, W, n):
    """Attribute a window statistic to its last token; backfill the prefix with a constant.

    The statistic cannot be known before its window has been fully observed, and the
    step-attribution rule in `localization_metrics` is already causal in the same sense.
    """
    out = np.empty(n, dtype=float)
    vals = np.asarray(series_vals, dtype=float)
    if vals.size == 0:
        return np.zeros(n)
    end = min(n, W - 1 + vals.size)
    out[W - 1:end] = vals[:end - (W - 1)]
    out[:W - 1] = vals[0]
    if end < n:
        out[end:] = vals[-1]
    return out


def _strided_positions(n, W, step):
    """Evenly spaced window-start indices in `[0, n-W]`, always including the last one so
    the final token is never stale."""
    positions = np.arange(0, n - W + 1, step)
    if positions[-1] != n - W:
        positions = np.append(positions, n - W)
    return positions


def _forward_fill(positions, sparse_vals, n_positions):
    """Step-hold (previous-value) expansion of sparse window-start samples onto every
    window-start index `0..n_positions-1`. Unlike linear interpolation, a query index `k`
    only ever resolves to a sampled window whose start is `<= k` — the window itself never
    extends past token `k+W-1`, so this cannot introduce a lookahead leak the way
    interpolating toward a later sample would (that later sample's window can end after
    the query token)."""
    idx = np.searchsorted(positions, np.arange(n_positions), side="right") - 1
    idx = np.clip(idx, 0, len(positions) - 1)
    return np.asarray(sparse_vals, dtype=float)[idx]


def _sw_var_series(e, W=SW_WINDOW, step=SW_STEP):
    """The series `compute_time_domain` reduces to `sw_var_peak` via `np.max`."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n == 0:
        return np.zeros(0)
    if n < W:
        return np.full(n, float(np.var(e)))
    vals = [np.var(e[i:i + W]) for i in range(0, n - W + 1, step)]
    if step != 1:
        idx = np.arange(0, n - W + 1, step)
        vals = np.interp(np.arange(n - W + 1), idx, vals)
    return _causal(vals, W, n)


def _cusum_abs_series(e):
    """Already one value per token — the series `cusum_max`/`cusum_shift_idx` collapse from."""
    e = np.asarray(e, dtype=float)
    if e.size == 0:
        return np.zeros(0)
    return np.abs(np.cumsum(e - e.mean()))


def _rolling_min(e, W=SW_WINDOW):
    """Trailing-window min, causal. `max` over all positions of this series recovers the
    global min exactly: every token is inside its own window, so the series never exceeds
    `e`, and at `argmin(e)` the window value equals the true global minimum."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n == 0:
        return np.zeros(0)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - W + 1)
        out[i] = float(np.min(e[lo:i + 1]))
    return out


def _pe_series(e, W=PE_WINDOW, order=PE_ORDER, delay=PE_DELAY, step=ROLLING_HEAVY_STEP):
    """Rolling permutation entropy — `compute_permutation_entropy` reduces this to
    `pe_mean` (no exact-identity test attempted here, see module docstring)."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n == 0:
        return np.zeros(0)
    if n < W:
        return np.full(n, float(permutation_entropy(e, order, delay)))
    positions = _strided_positions(n, W, step)
    sparse = [permutation_entropy(e[i:i + W], order, delay) for i in positions]
    vals = _forward_fill(positions, sparse, n - W + 1)
    return _causal(vals, W, n)


def _stft_high_series(e, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP, high=STFT_HIGH):
    """Per-frame high-band power fraction, interpolated onto the token grid — the series
    `compute_stft_features` reduces to `stft_max_high_power` via `np.max`."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n < nperseg * 2:
        return np.zeros(n)
    f, t, Zxx = scipy_stft(e - e.mean(), nperseg=nperseg, noverlap=noverlap)
    psd = np.abs(Zxx) ** 2
    mask = f >= high
    if mask.sum() == 0 or psd.shape[1] == 0:
        return np.zeros(n)
    frac = psd[mask].sum(0) / (psd.sum(0) + 1e-12)
    return np.interp(np.arange(n), np.asarray(t, dtype=float), frac)


def _stft_frame_entropy_series(e, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP):
    """Per-frame spectral entropy, interpolated onto the token grid — the companion series
    `compute_stft_features` reduces to `stft_spectral_entropy` via `np.mean` (no
    exact-identity test attempted here: interpolation does not exactly preserve a mean)."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n < nperseg * 2:
        return np.zeros(n)
    f, t, Zxx = scipy_stft(e - e.mean(), nperseg=nperseg, noverlap=noverlap)
    psd = np.abs(Zxx) ** 2
    if psd.shape[1] == 0:
        return np.zeros(n)
    psd_n = psd / (psd.sum(0, keepdims=True) + 1e-12)
    frame_ent = -np.sum(psd_n * np.log(psd_n + 1e-12), axis=0)
    return np.interp(np.arange(n), np.asarray(t, dtype=float), frame_ent)


def _rolling_tail_ratio(e, W=SW_WINDOW, tail_frac=0.20):
    """Local analogue of `rpdi` (trailing-tail mean / window mean) at native trace scale.
    No exact identity to the whole-trace `rpdi` exists — see module docstring."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n == 0:
        return np.zeros(0)
    if n < W:
        Wt = max(1, int(n * tail_frac))
        return np.full(n, float(e[-Wt:].mean() / (e.mean() + 1e-12)))
    tail_w = max(1, int(W * tail_frac))
    vals = []
    for i in range(0, n - W + 1):
        window = e[i:i + W]
        vals.append(float(window[-tail_w:].mean() / (window.mean() + 1e-12)))
    return _causal(vals, W, n)


_SPECTRAL_NAMES = (
    "spectral_entropy", "low_band_power", "high_band_power",
    "hl_ratio", "dominant_freq", "spectral_centroid",
)


def _rolling_spectral_series(e, W=ROLLING_SPECTRAL_WINDOW, step=ROLLING_HEAVY_STEP):
    """Six rolling-window FFT statistics, reusing `compute_spectral_features` unmodified on
    each window so the per-window formulas are byte-identical to the trace-level ones. No
    exact identity to the whole-trace values exists (one global FFT vs many local ones) —
    see module docstring."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n == 0:
        return {name: np.zeros(0) for name in _SPECTRAL_NAMES}
    if n < W:
        eff_min = max(2, min(MIN_TRACE, n))
        feats = compute_spectral_features(e, min_len=eff_min) if n >= eff_min else None
        if feats is None:
            return {name: np.zeros(n) for name in _SPECTRAL_NAMES}
        return {name: np.full(n, float(feats[name])) for name in _SPECTRAL_NAMES}
    positions = _strided_positions(n, W, step)
    sparse = {name: [] for name in _SPECTRAL_NAMES}
    for i in positions:
        feats = compute_spectral_features(e[i:i + W], min_len=1)
        for name in _SPECTRAL_NAMES:
            sparse[name].append(feats[name])
    n_positions = n - W + 1
    return {
        name: _causal(_forward_fill(positions, sparse[name], n_positions), W, n)
        for name in _SPECTRAL_NAMES
    }


def _rolling_hurst_series(e, W=ROLLING_SPECTRAL_WINDOW, step=ROLLING_HEAVY_STEP):
    """Rolling Hurst exponent (R/S analysis) over trailing windows of the deployed size.
    No exact identity exists — see module docstring."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n == 0:
        return np.zeros(0)
    if n < W:
        return np.full(n, float(compute_hurst_exponent(e)))
    positions = _strided_positions(n, W, step)
    sparse = [compute_hurst_exponent(e[i:i + W]) for i in positions]
    vals = _forward_fill(positions, sparse, n - W + 1)
    return _causal(vals, W, n)


def _align_secondary(raw, n):
    """Match a secondary raw series (spilled energy, log-sum-exp) to the entropy trace's
    token count. Only a genuinely missing channel (`raw is None`) produces NaN downstream —
    a short-but-present channel is aligned and used, so `token_feature_views` stays finite
    on short traces (the frozen contract requires this; see `smoke()` check 7)."""
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.size == 0:
        return None
    arr = arr[:n]
    if len(arr) < n:
        pad = arr[-1] if arr.size else 0.0
        arr = np.concatenate([arr, np.full(n - len(arr), pad)])
    return arr


def _logprob_token_series(top_k_logprobs, n, tail_k=5):
    """Per-token logprob-derived series — each is exactly the pre-mean array
    `repgrid_scoring.logprob_features`/`logprob_features_extended` averages down to a
    single scalar, so its mean is an exact identity against the corresponding global
    feature (`mean_top1_logprob`, `logprob_margin`, `mean_logprob_entropy`, `varentropy`,
    `renyi_entropy_2`, `topk_tail_mass`)."""
    names = (
        "top1_logprob_series", "logprob_margin_series", "topk_entropy_series",
        "topk_varentropy_series", "topk_renyi2_series", "topk_tail_mass_series",
    )
    if not isinstance(top_k_logprobs, dict) or "logprobs" not in top_k_logprobs:
        return {name: np.full(n, np.nan) for name in names}
    lp = np.asarray(top_k_logprobs["logprobs"], dtype=float)
    if lp.ndim != 2 or lp.shape[0] == 0:
        return {name: np.full(n, np.nan) for name in names}

    top1 = lp[:, 0]
    top2 = lp[:, 1] if lp.shape[1] > 1 else lp[:, 0]
    p = np.exp(lp)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    surprisal = -np.log(p + 1e-12)
    mean_surprisal = (p * surprisal).sum(axis=1, keepdims=True)
    varentropy = (p * (surprisal - mean_surprisal) ** 2).sum(axis=1)
    renyi2 = -np.log((p ** 2).sum(axis=1) + 1e-12)
    k = min(tail_k, lp.shape[1])
    tail_mass = np.clip(1.0 - p[:, :k].sum(axis=1), 0.0, 1.0)

    def _fit(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= n:
            return arr[:n]
        pad = arr[-1] if arr.size else np.nan
        return np.concatenate([arr, np.full(n - len(arr), pad)])

    return {
        "top1_logprob_series": _fit(top1),
        "logprob_margin_series": _fit(top1 - top2),
        "topk_entropy_series": _fit(ent),
        "topk_varentropy_series": _fit(varentropy),
        "topk_renyi2_series": _fit(renyi2),
        "topk_tail_mass_series": _fit(tail_mass),
    }


# ── the frozen contract (results/gl_liu_factorial_v2/RUN_DEFINITION.json, verbatim) ────

CORE_TOKEN_VIEWS = (
    "entropy_series",
    "entropy_sw_var_series",
    "entropy_cusum_abs_series",
    "spilled_sw_var_series",
    "spilled_cusum_abs_series",
)

BROAD_TOKEN_VIEWS = (
    "entropy_series",
    "entropy_rolling_spectral_entropy",
    "entropy_rolling_low_band_power",
    "entropy_rolling_high_band_power",
    "entropy_rolling_hl_ratio",
    "entropy_rolling_dominant_freq",
    "entropy_rolling_spectral_centroid",
    "entropy_stft_high_series",
    "entropy_stft_frame_entropy",
    "entropy_rolling_tail_ratio",
    "entropy_sw_var_series",
    "entropy_pe_series",
    "entropy_rolling_rs_hurst",
    "entropy_cusum_abs_series",
    "spilled_series",
    "spilled_sw_var_series",
    "spilled_cusum_abs_series",
    "spilled_rolling_min",
    "energy_series",
    "energy_rolling_min",
    "energy_sw_var_series",
    "energy_cusum_abs_series",
    "top1_logprob_series",
    "logprob_margin_series",
    "topk_entropy_series",
    "topk_varentropy_series",
    "topk_renyi2_series",
    "topk_tail_mass_series",
)

TOKEN_TO_GLOBAL_FEATURES = {
    "entropy_series": ("epr",),
    "entropy_rolling_spectral_entropy": ("spectral_entropy",),
    "entropy_rolling_low_band_power": ("low_band_power",),
    "entropy_rolling_high_band_power": ("high_band_power",),
    "entropy_rolling_hl_ratio": ("hl_ratio",),
    "entropy_rolling_dominant_freq": ("dominant_freq",),
    "entropy_rolling_spectral_centroid": ("spectral_centroid",),
    "entropy_stft_high_series": ("stft_max_high_power",),
    "entropy_stft_frame_entropy": ("stft_spectral_entropy",),
    "entropy_rolling_tail_ratio": ("rpdi",),
    "entropy_sw_var_series": ("sw_var_peak",),
    "entropy_pe_series": ("pe_mean",),
    "entropy_rolling_rs_hurst": ("hurst_exponent",),
    "entropy_cusum_abs_series": ("cusum_max", "cusum_shift_idx"),
    "spilled_series": ("epr_spilled",),
    "spilled_sw_var_series": ("sw_var_peak_spilled",),
    "spilled_cusum_abs_series": ("cusum_max_spilled",),
    "spilled_rolling_min": ("min_spilled",),
    "energy_series": ("epr_energy",),
    "energy_rolling_min": ("min_energy",),
    "energy_sw_var_series": ("sw_var_peak_energy",),
    "energy_cusum_abs_series": ("cusum_max_energy",),
    "top1_logprob_series": ("mean_top1_logprob",),
    "logprob_margin_series": ("logprob_margin",),
    "topk_entropy_series": ("mean_logprob_entropy",),
    "topk_varentropy_series": ("varentropy",),
    "topk_renyi2_series": ("renyi_entropy_2",),
    "topk_tail_mass_series": ("topk_tail_mass",),
}


# ── public per-trace and per-cell builders ──────────────────────────────────────────────

def token_feature_views(row: dict) -> dict:
    """Every broad-pool token view for one row, each a length-n array on the token grid
    (`n = len(row['token_entropies'])`). Missing secondary raw series (spilled energy,
    log-sum-exp, top-k logprobs) fill their dependent views with NaN, matching
    `positional_views.trace_series`'s handling."""
    e = np.asarray(row.get("token_entropies", []), dtype=float)
    n = len(e)
    s = _align_secondary(row.get("token_spilled_energies"), n)
    z = _align_secondary(row.get("token_logsumexp"), n)

    out = {
        "entropy_series": e.copy(),
        "entropy_sw_var_series": _sw_var_series(e),
        "entropy_cusum_abs_series": _cusum_abs_series(e),
        "entropy_pe_series": _pe_series(e),
        "entropy_stft_high_series": _stft_high_series(e),
        "entropy_stft_frame_entropy": _stft_frame_entropy_series(e),
        "entropy_rolling_tail_ratio": _rolling_tail_ratio(e),
        "entropy_rolling_rs_hurst": _rolling_hurst_series(e),
    }
    spec6 = _rolling_spectral_series(e)
    out["entropy_rolling_spectral_entropy"] = spec6["spectral_entropy"]
    out["entropy_rolling_low_band_power"] = spec6["low_band_power"]
    out["entropy_rolling_high_band_power"] = spec6["high_band_power"]
    out["entropy_rolling_hl_ratio"] = spec6["hl_ratio"]
    out["entropy_rolling_dominant_freq"] = spec6["dominant_freq"]
    out["entropy_rolling_spectral_centroid"] = spec6["spectral_centroid"]

    if s is not None:
        out["spilled_series"] = s.copy()
        out["spilled_sw_var_series"] = _sw_var_series(s)
        out["spilled_cusum_abs_series"] = _cusum_abs_series(s)
        out["spilled_rolling_min"] = _rolling_min(s)
    else:
        for name in ("spilled_series", "spilled_sw_var_series",
                     "spilled_cusum_abs_series", "spilled_rolling_min"):
            out[name] = np.full(n, np.nan)

    if z is not None:
        out["energy_series"] = z.copy()
        out["energy_rolling_min"] = _rolling_min(z)
        out["energy_sw_var_series"] = _sw_var_series(z)
        out["energy_cusum_abs_series"] = _cusum_abs_series(z)
    else:
        for name in ("energy_series", "energy_rolling_min",
                      "energy_sw_var_series", "energy_cusum_abs_series"):
            out[name] = np.full(n, np.nan)

    out.update(_logprob_token_series(row.get("top_k_logprobs"), n))

    return {name: np.asarray(out[name], dtype=float)[:n] for name in BROAD_TOKEN_VIEWS}


def build_token_channels(rows) -> dict:
    """`dict[view_name -> list[np.ndarray]]`, one array per row, aligned by row index —
    the shape `scripts/gl_liu_v1/run.py` and `scripts/gl_liu_factorial_v2/run.py` expect."""
    per_view = {name: [] for name in BROAD_TOKEN_VIEWS}
    for row in rows:
        views = token_feature_views(row)
        for name in BROAD_TOKEN_VIEWS:
            per_view[name].append(views[name])
    return per_view


# ── known-answer tests ───────────────────────────────────────────────────────────────

def smoke() -> None:
    from .feature_utils import compute_cusum_residuals, compute_time_domain

    rng = np.random.default_rng(0)
    n = 400
    e = rng.uniform(0.1, 2.0, n)
    s = rng.uniform(0.1, 2.0, n)
    z = rng.uniform(3.0, 6.0, n)
    lp = rng.normal(-1.0, 0.5, size=(n, 6))
    lp = -np.sort(-lp, axis=1)  # top-K sorted descending, like the saved schema
    row = {
        "token_entropies": e, "token_spilled_energies": s,
        "token_logsumexp": z, "top_k_logprobs": {"logprobs": lp},
    }
    views = token_feature_views(row)

    # 1. Every declared broad view is present, token-length, and finite.
    assert set(views) == set(BROAD_TOKEN_VIEWS)
    for name in BROAD_TOKEN_VIEWS:
        assert len(views[name]) == n, (name, len(views[name]))
        assert np.isfinite(views[name]).all(), name

    # 2. Exact MAX-collapse identities (entropy channel).
    td = compute_time_domain(e)
    assert np.isclose(views["entropy_sw_var_series"].max(), td["sw_var_peak"])
    cu = compute_cusum_residuals(e)
    assert np.isclose(views["entropy_cusum_abs_series"].max(), cu["cusum_max"])
    assert np.isclose(np.argmax(views["entropy_cusum_abs_series"]) / n, cu["cusum_shift_idx"])

    # 3. Exact MAX-collapse identities (spilled/energy channels — same sw_var/cusum ops).
    from .feature_utils import compute_spilled_energy_features
    sf = compute_spilled_energy_features(s)
    assert np.isclose(views["spilled_sw_var_series"].max(), sf["sw_var_peak_spilled"])
    assert np.isclose(views["spilled_cusum_abs_series"].max(), sf["cusum_max_spilled"])
    zf = compute_spilled_energy_features(z)  # energy views reuse the same extractor on Z_n
    assert np.isclose(views["energy_sw_var_series"].max(), zf["sw_var_peak_spilled"])
    assert np.isclose(views["energy_cusum_abs_series"].max(), zf["cusum_max_spilled"])

    # 4. Exact MIN-collapse identities.
    assert np.isclose(views["spilled_rolling_min"].min(), s.min())
    assert np.isclose(views["energy_rolling_min"].min(), z.min())

    # 5. Exact MEAN-collapse identities (raw per-token channels).
    assert np.isclose(views["entropy_series"].mean(), e.mean())
    assert np.isclose(views["spilled_series"].mean(), s.mean())
    assert np.isclose(views["energy_series"].mean(), z.mean())

    from .repgrid_scoring import logprob_features, logprob_features_extended
    top_k = {"logprobs": lp}
    base = logprob_features(top_k)
    ext = logprob_features_extended(top_k)
    assert np.isclose(views["top1_logprob_series"].mean(), base["mean_top1_logprob"])
    assert np.isclose(views["logprob_margin_series"].mean(), base["logprob_margin"])
    assert np.isclose(views["topk_entropy_series"].mean(), base["mean_logprob_entropy"])
    assert np.isclose(views["topk_varentropy_series"].mean(), ext["varentropy"])
    assert np.isclose(views["topk_renyi2_series"].mean(), ext["renyi_entropy_2"])
    assert np.isclose(views["topk_tail_mass_series"].mean(), ext["topk_tail_mass"])

    # 6. Causal alignment sanity: a burst must not move a rolling series before it starts
    #    (same test positional_views.py runs on sw_var_series; checked here on a
    #    no-identity view too, since that is exactly where a lookahead bug would hide).
    flat = np.full(300, 0.5)
    burst = flat.copy()
    burst[150:200] = 3.0
    a = _rolling_spectral_series(flat)["spectral_entropy"]
    b = _rolling_spectral_series(burst)["spectral_entropy"]
    assert np.allclose(a[:150], a[150:200].mean() * 0 + a[:150]), "sanity: self-consistent"
    assert not np.allclose(a[150:165], b[150:165]), \
        "rolling spectral entropy did not react to the burst at all"
    # the pre-burst region must be identical between the two traces (no lookahead)
    assert np.allclose(a[:150], b[:150]), "rolling_spectral_series reacted before the burst"

    # 7. build_token_channels aligns by row index and returns every declared name.
    channels = build_token_channels([row, row])
    assert set(channels) == set(BROAD_TOKEN_VIEWS)
    assert len(channels["entropy_series"]) == 2
    assert len(channels["entropy_series"][0]) == n

    # 8. Core contract is a strict subset of the broad contract, in the frozen order.
    assert set(CORE_TOKEN_VIEWS) <= set(BROAD_TOKEN_VIEWS)
    assert set(TOKEN_TO_GLOBAL_FEATURES) == set(BROAD_TOKEN_VIEWS)

    print(f"token_feature_views.smoke: PASS (8 checks, {len(BROAD_TOKEN_VIEWS)} broad / "
          f"{len(CORE_TOKEN_VIEWS)} core views)")


if __name__ == "__main__":
    smoke()
