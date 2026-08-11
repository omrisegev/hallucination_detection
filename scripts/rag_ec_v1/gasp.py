#!/usr/bin/env python
"""
gasp.py — faithful reproduction of GASP-threshold (Bouke, 2026, arXiv:2607.04223), the
"Grounding-Aware Sensitivity by Perturbation" detector, on top of this project's own
teacher-forced conditional-rescore cache (`cluster/run_conditional_rescore.py`).

FIDELITY DECLARATION (tailor-not-transplant convention, `feedback_tailor_not_transplant`)
--------------------------------------------------------------------------------------
This reproduces GASP-threshold — the paper's own reported DEFAULT detector throughout
(Section 5.5: "training-free ... we report it as the default detector throughout"), NOT
GASP-trained or GASP+base (their supervised LightGBM variants), which need labeled training
data and are out of scope for a label-free comparison.

**Response-level score (comparison level 2 — same protocol, our own sampling):**
Implements Eqs. (8)-(11) of the paper exactly, aggregated "over the whole answer" per the
paper's own stated response-level rule (Section 5.4: "the per-token values are averaged
over a sentence for span-level features and over the whole answer for response-level
features"). Scorer (Qwen2.5-1.5B-Instruct) matches one of the paper's own three tested
scorers exactly. Deviations from the paper's own evaluation, disclosed:
  - Paper: 400 responses (200/200 class-balanced), Summarization + Data2txt only, 200-token
    answer cap. Us: the full RAGTruth test split (~2,700 responses), all three task types
    including QA (which the paper excluded as "not long enough for stable estimation"),
    no answer-length cap.
  - Paper: three scorers, we use one (Qwen2.5-1.5B).

**Token-level curve (NOT a reproduction — a disclosed extension):**
The paper's finest unit is a text-split SENTENCE; it has no token-level curve. Our local
head needs one (to score against `span_token_spans`), so `token_gasp_curve` evaluates the
same four signals pointwise per token, with the leave-one-out aggregations changed from the
paper's "mean-over-span-tokens THEN max-over-chunks" to "max-over-chunks AT EACH TOKEN" —
the natural token-level generalization, analogous to how every other token-curve locator in
this project works (`scripts/gl_liu_v1/run.py:token_to_step`). This is registered as a
GASP-INSPIRED extension, not a claim about the paper's own local-head accuracy.

FEATURES (Eqs. 8-11)
--------------------------------------------------------------------------------------
Let dnll_noctx(t) = NLL_noctx(t) - NLL_full(t) = log p(t|full) - log p(t|noctx) (our
`token_spilled_energies` IS the forced-token NLL, so this difference already has exactly
the paper's Eq. (8)/(10) sign — no flip needed). Then, aggregating over token set S:
  gap    = mean_{t in S} dnll_noctx(t)                                          Eq. (8)
  jsd0   = mean_{t in S} JSD(p_full(t) || p_noctx(t))                           Eq. (9)
  drop   = max_k [ mean_{t in S} dnll_loo_k(t) ]                                Eq. (10)
  jsdloo = max_k [ mean_{t in S} JSD(p_full(t) || p_loo_k(t)) ]                 Eq. (11)
All four are LARGER for grounded spans (removing real support hurts more). The paper's
GASP-threshold hallucination score is the negated sum of the four standardized features
(Section 5.5) — higher score = more suspicious = LESS grounded.
"""
from __future__ import annotations

import numpy as np

from spectral_utils.evidence_contrast import js_divergence

GASP_FEATURE_NAMES = ("gasp_gap", "gasp_jsd0", "gasp_drop", "gasp_jsdloo")


def _loo_conditions(records_by_condition: dict) -> list:
    return sorted(
        (c for c in records_by_condition if c.startswith("loo_")),
        key=lambda c: int(c[len("loo_"):]),
    )


def _token_jsd(full_rec: dict, other_rec: dict, T: int):
    """Per-token JSD(p_full || p_other), EXACT when the cell carries it, approximate otherwise.

    `cluster/run_gasp_exact.py` computes Eqs. (9)/(11)'s divergence over the full vocabulary
    online during the forward pass and stores the resulting per-token scalar as
    `token_jsd_vs_full`. The preregistered conditional-rescore cells predate that and only have
    top-50 log-probs, so they fall back to `evidence_contrast.js_divergence`'s top-50 + shared-
    tail approximation.

    Both paths are kept deliberately: the exact cells still save `top_k_logprobs`, so running
    this function with and without the exact array measures how much the approximation costs
    instead of assuming it is negligible. Returns `None` when neither source is available.
    """
    exact = other_rec.get("token_jsd_vs_full")
    if exact is not None and len(exact):
        return np.asarray(exact, dtype=float)[:T]

    lp_full, lp_other = full_rec.get("top_k_logprobs"), other_rec.get("top_k_logprobs")
    if not lp_full or not lp_other:
        return None
    return js_divergence(
        np.asarray(lp_full["ids"])[:T], np.asarray(lp_full["logprobs"])[:T],
        np.asarray(lp_other["ids"])[:T], np.asarray(lp_other["logprobs"])[:T],
    )


def jsd_source(records_by_condition: dict) -> str:
    """`"full_vocabulary_exact"` or `"top50_tail_approximation"` — recorded beside every score
    so a panel never silently mixes the two."""
    for condition, record in records_by_condition.items():
        if condition == "full":
            continue
        if record.get("token_jsd_vs_full") is not None:
            return "full_vocabulary_exact"
    return "top50_tail_approximation"


def response_gasp_features(records_by_condition: dict) -> dict | None:
    """Eqs. (8)-(11), aggregated over the WHOLE response. Returns `None` if the response
    has neither a `noctx` nor any `loo_*` pass (nothing to compute a sensitivity from)."""
    full = records_by_condition.get("full")
    if full is None:
        return None
    T = len(full["token_entropies"])
    spill_full = np.asarray(full["token_spilled_energies"], dtype=float)[:T]
    lp_full = full.get("top_k_logprobs")

    gap = jsd0 = None
    noctx = records_by_condition.get("noctx")
    if noctx is not None:
        spill_noctx = np.asarray(noctx["token_spilled_energies"], dtype=float)[:T]
        gap = float(np.mean(spill_noctx - spill_full))
        djs = _token_jsd(full, noctx, T)
        if djs is not None:
            jsd0 = float(np.mean(djs))

    loo = _loo_conditions(records_by_condition)
    drop = jsdloo = None
    if loo:
        drop_per_k, jsdloo_per_k = [], []
        for c in loo:
            rec = records_by_condition[c]
            spill_k = np.asarray(rec["token_spilled_energies"], dtype=float)[:T]
            drop_per_k.append(float(np.mean(spill_k - spill_full)))
            djs_k = _token_jsd(full, rec, T)
            if djs_k is not None:
                jsdloo_per_k.append(float(np.mean(djs_k)))
        drop = max(drop_per_k) if drop_per_k else None
        jsdloo = max(jsdloo_per_k) if jsdloo_per_k else None

    if gap is None and drop is None:
        return None
    return {"gasp_gap": gap, "gasp_jsd0": jsd0, "gasp_drop": drop, "gasp_jsdloo": jsdloo}


def token_gasp_curve(records_by_condition: dict) -> dict | None:
    """Disclosed extension (see module docstring): the same four signals evaluated
    pointwise per token, with the leave-one-out aggregations taken as a per-token max
    over chunks rather than the paper's mean-then-max. Returns per-token arrays."""
    full = records_by_condition.get("full")
    if full is None:
        return None
    T = len(full["token_entropies"])
    spill_full = np.asarray(full["token_spilled_energies"], dtype=float)[:T]
    lp_full = full.get("top_k_logprobs")

    curves = {}
    noctx = records_by_condition.get("noctx")
    if noctx is not None:
        spill_noctx = np.asarray(noctx["token_spilled_energies"], dtype=float)[:T]
        curves["gasp_gap"] = spill_noctx - spill_full
        djs = _token_jsd(full, noctx, T)
        if djs is not None:
            curves["gasp_jsd0"] = djs

    loo = _loo_conditions(records_by_condition)
    if loo:
        drop_stack, jsdloo_stack = [], []
        for c in loo:
            rec = records_by_condition[c]
            spill_k = np.asarray(rec["token_spilled_energies"], dtype=float)[:T]
            drop_stack.append(spill_k - spill_full)
            djs_k = _token_jsd(full, rec, T)
            if djs_k is not None:
                jsdloo_stack.append(djs_k)
        if drop_stack:
            curves["gasp_drop"] = np.stack(drop_stack).max(axis=0)
        if jsdloo_stack:
            curves["gasp_jsdloo"] = np.stack(jsdloo_stack).max(axis=0)

    return curves or None


def gasp_threshold_scores(feature_rows: list) -> np.ndarray:
    """Section 5.5, GASP-threshold: standardize each of the 4 features across the
    population (unsupervised — no labels), then hallucination_score = -(sum of z-scores).
    Missing features (e.g. a response with no `noctx` pass) are mean-imputed (z=0) so one
    missing signal doesn't zero out the other three."""
    cols = {}
    for name in GASP_FEATURE_NAMES:
        raw = np.asarray([row.get(name) if row is not None else None
                          for row in feature_rows], dtype=float)
        finite = np.isfinite(raw)
        z = np.zeros(len(raw), dtype=float)
        if finite.sum() >= 2 and raw[finite].std() > 1e-12:
            mu, sd = raw[finite].mean(), raw[finite].std()
            z[finite] = (raw[finite] - mu) / sd
        cols[name] = z
    sensitivity = sum(cols[name] for name in GASP_FEATURE_NAMES)
    return -sensitivity


def gasp_threshold_curve(curve_rows: list, population_stats: dict | None = None) -> tuple:
    """Token-level counterpart of `gasp_threshold_scores`. `population_stats`, if given,
    is `{name: (mean, std)}` fit once over a population and frozen (so per-response curves
    don't each use their own local statistics, which would make a single-token response
    trivially self-normalize to zero). If omitted, stats are fit from `curve_rows` itself
    (offline/dev use) and also returned for reuse."""
    if population_stats is None:
        pooled = {name: [] for name in GASP_FEATURE_NAMES}
        for curves in curve_rows:
            if curves is None:
                continue
            for name in GASP_FEATURE_NAMES:
                if name in curves:
                    pooled[name].append(np.asarray(curves[name], dtype=float))
        population_stats = {}
        for name in GASP_FEATURE_NAMES:
            if pooled[name]:
                allvals = np.concatenate(pooled[name])
                allvals = allvals[np.isfinite(allvals)]
                if allvals.size >= 2 and allvals.std() > 1e-12:
                    population_stats[name] = (float(allvals.mean()), float(allvals.std()))

    out = []
    for curves in curve_rows:
        if curves is None:
            out.append(None)
            continue
        T = len(next(iter(curves.values())))
        sensitivity = np.zeros(T, dtype=float)
        for name in GASP_FEATURE_NAMES:
            if name in curves and name in population_stats:
                mu, sd = population_stats[name]
                sensitivity = sensitivity + (np.asarray(curves[name], dtype=float) - mu) / sd
        out.append(-sensitivity)
    return out, population_stats


# ── known-answer tests ───────────────────────────────────────────────────────────────

def smoke() -> None:
    rng = np.random.default_rng(0)
    T = 20

    def _lp(shift=0.0):
        ids = np.tile(np.arange(5), (T, 1))
        logits = -np.abs(rng.normal(0, 1, size=(T, 5))) - shift
        logits = np.sort(logits, axis=1)[:, ::-1]
        return {"ids": ids, "logprobs": logits}

    base_spill = rng.uniform(0.5, 1.5, T)
    full = {"token_entropies": base_spill.tolist(), "token_spilled_energies": base_spill.tolist(),
            "top_k_logprobs": _lp(0.0)}
    # grounded response: removing context/chunks hurts (spilled energy goes UP)
    noctx_g = {"token_entropies": (base_spill + 0.5).tolist(),
               "token_spilled_energies": (base_spill + 0.5).tolist(), "top_k_logprobs": _lp(0.4)}
    loo0_g = {"token_spilled_energies": (base_spill + 0.9).tolist(), "top_k_logprobs": _lp(0.6)}
    loo1_g = {"token_spilled_energies": (base_spill + 0.1).tolist(), "top_k_logprobs": _lp(0.05)}
    grounded = {"full": full, "noctx": noctx_g, "loo_0": loo0_g, "loo_1": loo1_g}

    # hallucinated response: removing context/chunks barely changes anything
    noctx_h = {"token_entropies": (base_spill + 0.01).tolist(),
               "token_spilled_energies": (base_spill + 0.01).tolist(), "top_k_logprobs": _lp(0.0)}
    loo0_h = {"token_spilled_energies": (base_spill + 0.02).tolist(), "top_k_logprobs": _lp(0.0)}
    loo1_h = {"token_spilled_energies": (base_spill - 0.01).tolist(), "top_k_logprobs": _lp(0.0)}
    hallucinated = {"full": full, "noctx": noctx_h, "loo_0": loo0_h, "loo_1": loo1_h}

    fg = response_gasp_features(grounded)
    fh = response_gasp_features(hallucinated)

    # 1. All four features are larger (more sensitive) for the grounded response.
    for name in GASP_FEATURE_NAMES:
        assert fg[name] > fh[name], f"{name}: grounded={fg[name]} hallucinated={fh[name]}"

    # 2. drop() picks the MEAN-then-MAX chunk (loo_0, mean delta 0.9), not a per-token max
    #    that would blend in loo_1's smaller effect — verifies the Eq. (10) aggregation order.
    assert abs(fg["gasp_drop"] - 0.9) < 1e-9

    # 3. GASP-threshold: grounded scores LOWER (more suspicious = higher score, so a
    #    grounded/high-sensitivity response must score lower than a hallucinated one).
    scores = gasp_threshold_scores([fg, fh])
    assert scores[0] < scores[1], "grounded response must score BELOW the hallucinated one"

    # 4. A response with only `full` (no interventions) yields None, not a crash.
    assert response_gasp_features({"full": full}) is None
    assert token_gasp_curve({"full": full}) is None

    # 5. Token curve: shape matches T, and the per-token max-aggregation order differs
    #    from the response-level mean-then-max by construction (sanity, not equality).
    curve_g = token_gasp_curve(grounded)
    assert curve_g["gasp_drop"].shape == (T,)
    assert curve_g["gasp_gap"].shape == (T,)

    # 6. Missing-feature imputation: a response with only noctx (no loo) still scores,
    #    using only gap/jsd0 (drop/jsdloo mean-imputed to z=0, not crashing to NaN).
    solo_noctx = {"full": full, "noctx": noctx_g}
    f_solo = response_gasp_features(solo_noctx)
    assert f_solo["gasp_drop"] is None and f_solo["gasp_gap"] is not None
    scores2 = gasp_threshold_scores([fg, fh, f_solo])
    assert np.isfinite(scores2).all()

    print("gasp.smoke: PASS (6 checks)")


if __name__ == "__main__":
    smoke()
