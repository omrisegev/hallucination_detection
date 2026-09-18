# Claude feature bank — token-level L-SML v1

Status: implementation of the handoff protocol; development-only until the
full run completes and is reviewed.  This experiment is designed to answer a
narrow question: does a manually oriented bank containing the newly proposed
energy, true-tail, dynamics, dominant-frequency, and BOCPD views improve over
the simple mean when the fusion is performed at the token level?

## Fixed algorithm

There is one teacher-forced response per roster row.  We do not sample several
answers.  Raw telemetry is already available at each generated token.  A
feature may use a trailing causal window of 16 tokens, but its value at token
`t` never uses a future token.

The locator is separate from the gate:

1. Build an oriented risk matrix `X[t, j]` for every token in the response.
2. Fit pooled token means and standard deviations on donor source folds only,
   capped at 60,000 deterministic tokens.
3. On each held-out source fold, standardize the same token matrix and produce
   two token-risk streams: continuous L-SML and a simple equal-weight mean.
4. For each ProcessBench step, score the step by the mean of its ten largest
   token risks (or all tokens when the step is shorter than ten).  The locator
   selects the step with the largest score.
5. Apply the fixed LOCO-5 answer gate.  A closed gate emits the clean-answer
   decision; an open gate emits the selected step.  ProcessBench macro-F1 and
   within-answer PRMBench AUROC are reported separately.

The L-SML fit receives no correctness labels.  The only sign operation after
L-SML is a deterministic coefficient gauge to resolve its global `+/-`
ambiguity: the coefficient sum is made positive, falling back to a positive
first nonzero coefficient.  This is not a feature anchor and is not learned
from data labels.

## Locator feature definitions

The following eleven channels are the bank, in the exact order used by the
runner.  The listed sign is multiplied into the raw value before
standardization, so every oriented channel has the convention “larger means
more hallucination risk.”

| Feature | Raw calculation | Risk sign |
|---|---|---:|
| `q15_H1` | Shannon entropy of the saved top-15 probabilities after renormalizing those 15 entries | `+1` |
| `q15_VE1` | Probability-weighted variance of `-log(q)` on the same renormalized top-15 distribution | `+1` |
| `chosen_surprisal` | Saved spilled energy `-log p(sampled token)` at the current token | `+1` |
| `logprob_margin` | Saved top-1 log-probability minus top-2 log-probability | `-1` |
| `true_tail50` | `1 - sum(exp(saved top-50 logprobs))`; the saved logprobs are already normalized, and no `Z` subtraction is performed | `+1` |
| `energy_level` | Raw full-vocabulary log-sum-exp `Z_t` | `-1` |
| `energy_innovation` | `Z_t` minus the mean of `Z_0...Z_(t-1)`; token 0 is zero by convention | `+1` |
| `top15_turnover` | `1 - |Top15_t ∩ Top15_(t-1)| / 15`, matched by token identity | `+1` |
| `top50_js` | Jensen-Shannon distance between adjacent top-50 distributions, aligned by token identity and renormalized within the saved top-50 support | `+1` |
| `dominant_freq16` | Dominant non-DC frequency of the causal trailing 16-token entropy window | `-1` |
| `bocpd_p0` | Gaussian BOCPD change-point probability using hazard `1/32`; `p0` from the previous update is assigned to token `t`, and token 0 is zero | `+1` |

`true_tail50` is intentionally different from the old `tail50` feature that
measures mass inside a saved support after a separate normalization.  The
implementation uses the normalized saved logprobs directly.  The bank does
not contain `digit`, `digit_disagreement`, or any digit-derived channel.

## Gate: fixed LOCO-5

LOCO-5 is not part of the token fusion.  It is an answer-level gate made from
the following five scalar features:

| Feature | Raw calculation | Risk sign |
|---|---|---:|
| `cusum_max` | Maximum absolute cumulative sum of mean-centered answer entropy | `+1` |
| `logprob_margin` | Answer mean of top-1 minus top-2 logprob | `-1` |
| `min_energy` | Minimum answer `Z_t` | `-1` |
| `spectral_entropy` | Global FFT spectral entropy of the mean-centered answer entropy | `+1` |
| `topk_tail_mass` | Answer mean mass outside the saved top-5 after renormalizing the saved top-50 support | `+1` |

Within each ProcessBench cell, each oriented scalar is converted to a
tie-aware midrank in `[0, 1]`.  The gate score is the simple mean of the five
midranks and opens at `>= 0.33`.  No correctness label enters this calculation.
The subset itself was historically selected using labels, so this complete
LOCO-5 arm is explicitly `development_only`; it must not be presented as a
label-free confirmation result.

## Comparisons and interpretation

The primary comparison is `token_l_sml` versus `token_equal_mean` under the
same LOCO-5 gate, same folds, same step adapter, and same roster.  We do not
discard a channel because its standalone score is weak: the question is the
fused result and its cross-fold stability.  We also preserve the separation
between ProcessBench macro-F1, ProcessBench SLA/localization counts, and
within-answer PRMBench AUROC.

The old length-calibrated readout is not used.  This does not assert that
length carries no evidence; it only keeps the readout contract fixed after the
negative Step 420 ablation.  Length is not a feature in this bank.
