"""
positional_views.py — localize by recovering the per-token series the feature extractors already
compute and then throw away.

THE DESIGN ERROR THIS REPLACES
------------------------------
The first localization arm sliced the entropy trace into steps (or 32-token windows) and re-ran
`extract_all_features` on each slice. That is the wrong move, and Omri caught it: these features
are *trace-scale* statistics. `sw_var_peak` is the peak of a sliding-window variance; on a
25-token step with `sw_window=16` it has ten window positions, and on a 32-token window it has
seventeen. It stops being a peak statistic and becomes noise. The historical results that make
`sw_var_peak` worth using at all were earned at full trace length.

THE INFORMATION WAS ALREADY THERE
---------------------------------
Every one of these features builds a positional series over the WHOLE trace and then collapses it:

  feature_utils.py:120-124  compute_time_domain      sw_vars   -> np.max(...)        argmax dropped
  feature_utils.py:203-206  compute_permutation_...  pes       -> min, mean          series dropped
  feature_utils.py:269-274  compute_cusum_residuals  abs_cusum -> max, argmax/len    KEPT as
                                                                  `cusum_shift_idx` — the one
                                                                  positional view already in the pool
  feature_utils.py:89-102   compute_stft_features    high_frac -> max                per-frame dropped

So the localizer does not need a new signal or a smaller window. It needs the argmax that is
computed on line 141 and discarded on line 141. This module recomputes those series **at native
trace scale, with the deployed window parameters**, and returns them per token.

WINDOW PARAMETERS ARE COPIED, NOT CHOSEN
----------------------------------------
`sw_window=16` and `sw_step=1` are `compute_time_domain`'s defaults; PE `window_size=10, order=3`
and STFT `nperseg=16, noverlap=8, f>=0.40` are the deployed values. Re-tuning any of them would
break the link to the trace-level results that motivate this arm at all.

=============================================================================================
PRE-REGISTERED CHOICES — fixed here BEFORE any number was computed
=============================================================================================
1. ALIGNMENT. A window statistic over `e[i:i+W]` is attributed to its LAST token, `i+W-1`. The
   statistic cannot be known before its window has been observed, and the step-attribution rule
   in `localization_metrics` is already causal in the same sense.
2. THE PREFIX. Tokens before the first full window carry the first computed value (a constant),
   not NaN. A constant prefix has zero variance and zero first-difference, so it can neither
   fire nor be imputed away — it is the honest "no evidence yet" state.
3. ORIENTATION IS DERIVED, NOT DECLARED. Polarity comes from `sign(rho-hat)` inside U-PCR, as in
   the deployed arm. `POSITIONAL_SIGNS` below exists only to mirror `prepare_cell`'s hand-sign
   step, and it is undone before rho sees the data — exactly as `upcr_rho_oriented` does.
4. STEP RISK is the MAX of the fused per-token score inside the step. The trace-level statistic
   these views come from is itself a max (`sw_var_peak`, `cusum_max`), so the max is the reading
   consistent with the feature's own definition. Mean is available as a sensitivity.
"""
import os
import sys

import numpy as np
from scipy.signal import stft as scipy_stft

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spectral_utils.feature_utils import permutation_entropy   # noqa: E402
from spectral_utils.streaming_utils import anchor_orient       # noqa: E402
from spectral_utils.upcr import upcr_fit                       # noqa: E402

# Deployed parameters, copied from feature_utils. Do not tune.
SW_WINDOW, SW_STEP = 16, 1
PE_WINDOW, PE_ORDER, PE_DELAY = 10, 3, 1
STFT_NPERSEG, STFT_NOVERLAP, STFT_HIGH = 16, 8, 0.40

# exp06's fitted U-PCR configuration, same as the deployed arm.
FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

# Hand signs mirroring the trace-level views these series collapse into. Higher = more likely
# CORRECT, matching the package convention: a big local variance, a big CUSUM excursion or a
# burst of high-frequency power all indicate trouble, hence -1.
POSITIONAL_SIGNS = {
    "sw_var_series": -1,          # collapses to sw_var_peak      (ALL_SIGNS -1)
    "sw_var_spilled_series": -1,  # collapses to sw_var_peak_spilled
    "cusum_abs_series": -1,       # collapses to cusum_max        (ALL_SIGNS -1)
    "cusum_abs_spilled_series": -1,
    "pe_series": -1,              # collapses to pe_mean          (ALL_SIGNS -1)
    "stft_high_series": -1,       # collapses to stft_max_high_power (ALL_SIGNS -1)
    "entropy_series": -1,         # collapses to epr              (ALL_SIGNS -1) — the anchor
}
POSITIONAL_VIEWS = list(POSITIONAL_SIGNS)
ANCHOR_VIEW = "entropy_series"    # the per-token analogue of `epr`, the deployed anchor

MIN_TRACE = 8                     # compute_spectral_features' floor


def _causal(series_vals, W, n):
    """Attribute a window statistic to its last token and backfill the prefix (choices 1, 2)."""
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


def sw_var_series(e, W=SW_WINDOW, step=SW_STEP):
    """The series `compute_time_domain` reduces to `sw_var_peak` via np.max (feature_utils:120)."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n < W:
        return np.full(n, float(np.var(e)) if n else 0.0)
    vals = [np.var(e[i:i + W]) for i in range(0, n - W + 1, step)]
    if step != 1:                      # re-expand a strided series onto the token grid
        idx = np.arange(0, n - W + 1, step)
        vals = np.interp(np.arange(n - W + 1), idx, vals)
    return _causal(vals, W, n)


def cusum_abs_series(e):
    """The series `compute_cusum_residuals` reduces to `cusum_max` + `cusum_shift_idx`.

    Already one value per token, so no alignment is needed — this is the view whose argmax the
    pool ALREADY carries as `cusum_shift_idx`, which is why it is the natural first localizer.
    """
    e = np.asarray(e, dtype=float)
    if e.size == 0:
        return np.zeros(0)
    return np.abs(np.cumsum(e - e.mean()))


def pe_series(e, W=PE_WINDOW, order=PE_ORDER, delay=PE_DELAY):
    """The rolling permutation entropy `compute_permutation_entropy` reduces to min/mean."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n < W:
        return np.full(n, float(permutation_entropy(e, order, delay)) if n else 0.0)
    vals = [permutation_entropy(e[i:i + W], order, delay) for i in range(n - W + 1)]
    return _causal(vals, W, n)


def stft_high_series(e, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP, high=STFT_HIGH):
    """Per-FRAME high-band power fraction — the series `compute_stft_features` reduces to a max.

    Frames are placed at their own sample times and interpolated onto the token grid, so the
    result stays token-indexed like every other view here.
    """
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


def trace_series(entropies, spilled=None) -> dict:
    """Every positional view for one trace, each a length-n array on the token grid."""
    e = np.asarray(entropies, dtype=float)
    n = len(e)
    out = {
        "entropy_series": e.copy(),
        "sw_var_series": sw_var_series(e),
        "cusum_abs_series": cusum_abs_series(e),
        "pe_series": pe_series(e),
        "stft_high_series": stft_high_series(e),
    }
    if spilled is not None and len(spilled) >= MIN_TRACE:
        s = np.asarray(spilled, dtype=float)[:n]
        if len(s) < n:                 # documented 1-2 token skew between the saved series
            s = np.concatenate([s, np.full(n - len(s), s[-1] if s.size else 0.0)])
        out["sw_var_spilled_series"] = sw_var_series(s)
        out["cusum_abs_spilled_series"] = cusum_abs_series(s)
    else:
        out["sw_var_spilled_series"] = np.full(n, np.nan)
        out["cusum_abs_spilled_series"] = np.full(n, np.nan)
    return {k: np.asarray(v, dtype=float)[:n] for k, v in out.items()}


# ── the label-free fit, over tokens pooled across traces ─────────────────────

def fit_positional_arm(rows, max_tokens=200_000, seed=0, min_views=3):
    """U-PCR over the pooled per-token series. Returns a `our_arm.UpcrArm`, or None.

    The fit population is every token of every trace. This is the same PROCEDURE as the deployed
    arm — mirror `prepare_cell`'s impute/hand-sign/z-score, undo the hand signs so `sign(rho-hat)`
    derives polarity itself, fit, orient against the anchor — applied to a new view set. It is
    not a claim to reproduce a recorded roster number; there is no recorded number for a
    token-level population.

    Labels are never involved at any point, including the guard: unlike `prepare_cell` there is
    no single-class check to satisfy, because nothing here is ever scored against a label.
    """
    from our_arm import UpcrArm

    rng = np.random.default_rng(seed)
    per_view = {v: [] for v in POSITIONAL_VIEWS}
    total = 0
    for i in rng.permutation(len(rows)):
        r = rows[int(i)]
        e = r.get("token_entropies")
        if e is None or len(e) < MIN_TRACE:
            continue
        s = trace_series(e, r.get("token_spilled_energies"))
        for v in POSITIONAL_VIEWS:
            per_view[v].append(s[v])
        total += len(e)
        if total >= max_tokens:
            break
    if total == 0:
        return None
    fd = {v: np.concatenate(per_view[v]) for v in POSITIONAL_VIEWS}

    # `prepare_cell`'s column construction, reproduced for a pool it does not know about.
    pool, hand, impute, mu, sd, cols = [], [], [], [], [], []
    for v in POSITIONAL_VIEWS:
        x = fd[v]
        bad = ~np.isfinite(x)
        if bad.all():
            continue
        med = float(np.median(x[~bad]))
        x = np.where(bad, med, x)
        h = float(POSITIONAL_SIGNS.get(v, +1))
        z = x * h
        m, s_ = float(z.mean()), float(z.std())
        if s_ < 1e-8:
            continue
        pool.append(v)
        hand.append(h)
        impute.append(med)
        mu.append(m)
        sd.append(s_)
        cols.append((z - m) / s_)
    if len(pool) < min_views:
        return None

    V = np.column_stack(cols)
    hand = np.asarray(hand)
    V_un = V * hand
    derived = np.sign(upcr_fit(V_un.T, **FIT).rho_hat_full)
    derived[derived == 0] = 1.0
    F = (V_un * derived).T
    res = upcr_fit(F, **FIT)

    anchor_col = pool.index(ANCHOR_VIEW) if ANCHOR_VIEW in pool else 0
    score, flipped = anchor_orient(res.w @ F, V[:, anchor_col])

    return UpcrArm(
        pool=pool, hand=hand, impute=np.asarray(impute), mu=np.asarray(mu),
        sd=np.asarray(sd), derived=np.asarray(derived, dtype=float),
        w=np.asarray(res.w, dtype=float), keep=np.asarray(res.keep, dtype=bool),
        flipped=bool(flipped), anchor_name=pool[anchor_col],
        score=np.asarray(score, dtype=float), n=total,
    )


def token_risk(row, arm) -> np.ndarray:
    """Per-token risk for one trace: higher = more suspect. The frozen fit, replayed."""
    e = row.get("token_entropies")
    if e is None or len(e) < MIN_TRACE:
        return np.zeros(0)
    s = trace_series(e, row.get("token_spilled_energies"))
    return -np.asarray(arm.apply(s), dtype=float)


def step_risk(row, arm, aggregation="max") -> np.ndarray:
    """Per-step risk = max (choice 4) of the per-token risk inside the step."""
    r = token_risk(row, arm)
    spans = row["step_token_spans"]
    out = np.full(len(spans), np.nan)
    if r.size == 0:
        return out
    reduce = {"max": np.max, "mean": np.mean}[aggregation]
    for i, span in enumerate(spans):
        if span is None:
            continue
        seg = r[span[0]:min(span[1], r.size)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(reduce(seg))
    return out


# ── known-answer tests ───────────────────────────────────────────────────────

def smoke() -> None:
    rng = np.random.default_rng(0)

    # 1. Every series is token-length and finite, and matches the trace-level statistic it
    #    collapses into. This is the whole claim of the module: no new signal, just the argmax.
    from spectral_utils.feature_utils import (compute_cusum_residuals, compute_time_domain)
    e = rng.uniform(0.1, 2.0, 400)
    s = trace_series(e, spilled=rng.uniform(0.1, 2.0, 400))
    for v in POSITIONAL_VIEWS:
        assert len(s[v]) == 400, (v, len(s[v]))
        assert np.isfinite(s[v]).all(), v
    td = compute_time_domain(e)
    assert np.isclose(s["sw_var_series"].max(), td["sw_var_peak"]), \
        (s["sw_var_series"].max(), td["sw_var_peak"])
    cu = compute_cusum_residuals(e)
    assert np.isclose(s["cusum_abs_series"].max(), cu["cusum_max"]), \
        (s["cusum_abs_series"].max(), cu["cusum_max"])
    # and the argmax IS the position the pool already carries, up to the /len normalisation
    assert np.isclose(np.argmax(s["cusum_abs_series"]) / 400, cu["cusum_shift_idx"]), \
        (np.argmax(s["cusum_abs_series"]) / 400, cu["cusum_shift_idx"])

    # 2. Causal alignment: a burst must not move the series BEFORE it starts.
    flat = np.full(300, 0.5)
    burst = flat.copy()
    burst[150:200] = 3.0
    a, b = sw_var_series(flat), sw_var_series(burst)
    assert np.allclose(a[:150], b[:150]), "sw_var_series reacted before the burst began"
    assert b[150:200].max() > 10 * a.max() + 1e-9, "sw_var_series did not react to the burst"

    # 3. THE LOCALIZATION TEST. A planted burst in a known step, at REAL trace scale
    #    (12 steps x 45 tokens = 540 tokens, the regime these features were built for).
    n_steps, L = 12, 45
    n = n_steps * L
    spans = [(i * L, (i + 1) * L) for i in range(n_steps)]
    bad = 7
    rows = []
    for k in range(60):
        ent = rng.normal(0.40, 0.06, n).clip(0.01, None)
        has = k % 2 == 0
        if has:
            ent[bad * L:(bad + 1) * L] += rng.normal(0.9, 0.15, L).clip(0, None)
        rows.append({"token_entropies": ent,
                     "token_spilled_energies": ent * rng.uniform(0.9, 1.1, n),
                     "step_token_spans": spans, "label": bad if has else -1})

    arm = fit_positional_arm(rows)
    assert arm is not None and len(arm.pool) >= 3, arm
    hits = [int(np.nanargmax(step_risk(r, arm))) for r in rows if r["label"] >= 0]
    acc = float(np.mean([h == bad for h in hits]))
    assert acc >= 0.9, f"planted the burst in step {bad}; peak risk hit it {acc:.0%} of the time"

    # 4. Clean traces must not concentrate on the planted step — otherwise check 3 is measuring
    #    the fit, not the signal.
    clean = [int(np.nanargmax(step_risk(r, arm))) for r in rows if r["label"] < 0]
    frac = float(np.mean([c == bad for c in clean]))
    assert frac < 0.4, f"clean traces also peak at step {bad} {frac:.0%} of the time"

    # 5. The arm is label-free by construction: fitting on a shuffled copy of the same traces
    #    must give the same weights, because nothing about order or labels enters.
    arm2 = fit_positional_arm(list(reversed(rows)))
    assert np.allclose(np.sort(np.abs(arm.w)), np.sort(np.abs(arm2.w)), atol=1e-6), \
        "the fit depends on row order — something label-like is leaking in"

    print(f"positional_views.smoke: PASS (5 checks)  [pool={len(arm.pool)} kept={arm.n_kept} "
          f"anchor={arm.anchor_name}; planted-step hit {acc:.0%}, clean false-peak {frac:.0%}]")


if __name__ == "__main__":
    smoke()
