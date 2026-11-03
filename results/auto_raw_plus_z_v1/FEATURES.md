# The 34 CAT34 inputs (PRMBench, PRM-backbone telemetry)

**Which 17 channels.** AUTO's 17 channels are the PRM view plus the 16 N16 channels. N16 is the 11 bank11
channels, then `realized_z`, `realized_drv`, then the three digit channels. That order is
`NESTED['N16']` of `../prm_boosting_bank_sweep_v1/lsml_bank_sweep.py`. CAT34 uses columns 0-16 = Z17
(answer-z, exactly the current AUTO_N16 inputs) and columns 17-33 = the same 17 channels without the answer
z-score. In `INPUTS_<prm>.npz` they are `X34[:, j]`, with names `names34[j]`. The untransformed values are in
`raw17` (same column order as `raw17_names`).

**Telemetry source.** The PRM model's own forward pass over the PRMBench answer. Teacher-forced, thinkfix
mask1: the forced first token is dropped. Per PRM, this is `raw_features/<prm>/FEATURES.npz` (cell
`prmbench_qwen3_8b`, 6,969 answers, 94,203 steps; built by job 283832 and similar, see its MANIFEST.json). Each
channel is first a token stream. It is then reduced to one value per step: the bank11 channels take the mean
of the 10 largest token values in the step (`top10`), and the digit channels the mean of the 2 largest active
tokens. Missing step values are filled with the answer's column mean. Code:
`spectral_utils/claude_feature_bank_v1.py`, `family_external_features.py`, `external_banks_v4.py`,
`digit_feature_family.py`.

**Orientation.** The stored FEATURES.npz values are already risk-oriented. The bank builder multiplies by the
declared sign below, so higher means the step is meant to be more suspect. No further sign is applied; Z17 and
CAT34 use the values as stored. The rank-normal transform is increasing, so `raw:c` has the same orientation
as `z:c`. The fused score is flipped if it correlates negatively with `z:PRM`. "Observed within-AUC" is the
mean over answers of each column's step AUC against the error label, for Qwen-7B / Qwen-72B / PURE / RLHFlow,
for orientation only (from the `z:` columns of the bundle; never used for fitting). Below 0.5 means that
channel points the other way on PRMBench. `energy_innovation` and `top50_js` do; this is known (HISTORY Steps
438-439) and was left as it is, the same as in AUTO_N16.

## Transforms

- **`z:` columns (0-16), answer-z.** For answer a and channel c: `z = (x - mean_a(x)) / sd_a(x)`, with
  population sd, and 0 if `sd_a <= 1e-12` (`answer_standardize`).
  - Digit channels use `masked_std`: mean and sd over the answer's steps where the digit channel is active
    (`digit_active`), and 0 on inactive steps.
  - `z:PRM` is the answer-z of `log((1-r)/r)`, where r is the PRM's step reward (validity probability),
    clipped to [1e-6, 1-1e-6].
- **`raw:` columns (17-33), global rank -> normal.** `g(x) = Phi^-1((rank(x) - 0.5) / n)`, ranked over all n
  steps of all 6,969 answers of that PRM (average ranks for ties). Digit channels are ranked over their
  active steps only and are 0 on inactive steps. `raw:PRM` is g(log((1-r)/r)).
  - The transform keeps the differences between answers and puts all channels on one N(0,1) scale. Literal
    raw values would let the channel with the largest scale dominate the covariance that L-SML uses.

## Table

| col (z / raw) | name | source key in FEATURES.npz | what it measures (token level, then step reduction) | declared sign | observed within-AUC (7B / 72B / PURE / RLHFlow) |
|---|---|---|---|---|---|
| 0 / 17 | PRM | (telemetry reward, not in FEATURES.npz) | PRM step risk `log((1-r)/r)`, r = PRM step reward | + | 0.801 / 0.795 / 0.763 / 0.662 |
| 1 / 18 | q15_H1 | `q15_H1` | Shannon entropy of the saved top-15 probabilities after renormalizing them; top10 step mean | + | 0.765 / 0.794 / 0.707 / 0.704 |
| 2 / 19 | q15_VE1 | `q15_VE1` | Probability-weighted variance of -log q on the same renormalized top-15 (varentropy) | + | 0.775 / 0.733 / 0.694 / 0.679 |
| 3 / 20 | chosen_surprisal | `chosen_surprisal` | -log p(written token) | + | 0.750 / 0.798 / 0.769 / 0.762 |
| 4 / 21 | logprob_margin | `logprob_margin` | top-1 minus top-2 log-probability | - (stored negated) | 0.744 / 0.753 / 0.694 / 0.688 |
| 5 / 22 | true_tail50 | `true_tail50` | 1 - probability mass of the saved top-50 | + | 0.755 / 0.770 / 0.704 / 0.691 |
| 6 / 23 | energy_level | `energy_level` | full-vocabulary log-sum-exp Z_t | - (stored negated) | 0.799 / 0.754 / 0.656 / 0.690 |
| 7 / 24 | energy_innovation | `energy_innovation` | Z_t minus the mean of Z over the earlier prefix (token 0 = 0) | + | 0.389 / 0.444 / 0.450 / 0.386 |
| 8 / 25 | top15_turnover | `top15_turnover` | 1 - overlap of the top-15 token sets of adjacent tokens / 15 | + | 0.588 / 0.581 / 0.621 / 0.617 |
| 9 / 26 | top50_js | `top50_js` | Jensen-Shannon distance between adjacent top-50 distributions (identity-aligned) | + | 0.414 / 0.395 / 0.498 / 0.507 |
| 10 / 27 | dominant_freq16 | `dominant_freq16` | dominant non-DC frequency of the causal trailing 16-token entropy window | - (stored negated) | 0.589 / 0.590 / 0.567 / 0.578 |
| 11 / 28 | bocpd_p0 | `bocpd_p0` | Gaussian BOCPD change-point probability, hazard 1/32 | + | 0.641 / 0.638 / 0.648 / 0.677 |
| 12 / 29 | realized_z | `realized_z` | CT7 pooled chosen-token z-test per step, de-spiked; answer-standardized by definition (so raw is close to z, corr 0.98) | + | 0.572 / 0.635 / 0.748 / 0.641 |
| 13 / 30 | realized_drv | `realized_drv` | EMA16 of chosen_surprisal, positive first difference, mean of the 3 largest rises in the step | + | 0.736 / 0.784 / 0.780 / 0.726 |
| 14 / 31 | digit_alternative | `digit_alternative` | probability of the second-most-likely digit at digit positions (digit ids 15..24); top-2 mean of active tokens | + | 0.713 / 0.685 / 0.699 / 0.678 |
| 15 / 32 | digit_spread | `digit_spread` | entropy of the probability mass on digit tokens, scaled by that mass, in log10 units | + | 0.736 / 0.720 / 0.723 / 0.687 |
| 16 / 33 | digit_alternative_innovation | `digit_alternative_innovation` | digit_alternative minus its earlier-prefix mean | + | 0.706 / 0.677 / 0.689 / 0.659 |

Digit-channel note: Omri decided on 2026-09-17 to exclude digit features from future proposed methods, then
kept them on 2026-09-30. They are in N16, so they are in AUTO_N16 and here. N13 (the 11 bank11 channels plus
`realized_z` and `realized_drv`) is the digit-free subset.
