# Fitting scope, and sw_var as a twelfth channel — both negative

Runs `scripts/diagnostics/fitting_scope_and_swvar_v1.py`; output
`results/token_probability_fusion_v1/FITTING_SCOPE_AND_SWVAR.json`. Development-only.

Both questions are run in the **C1 architecture** — token-level L-SML fusion, then the
Top-10 mean inside each step — because Stage B showed that is the only order in which
L-SML beats equal weighting at all. The target throughout is OlympiadBench and
Omni-MATH.

**Anchor.** `pooled_all` in this script *is* Stage B's C1, and the script refuses to
print anything unless it replays to 1e-9. It does: 35.9237, |diff| 0.0000 pp. An earlier
version of this run failed that check at 35.7983, which turned out to be an aggregation
bug — see the note at the end.

## Part 1 — does a narrower fitting scope help?

The standardizer and the L-SML weights are fitted inside a scope rather than over
everything. Source folds are preserved *inside* every scope, so narrowing the scope
changes who donates, never whether the split is source-disjoint.

| arm | mean | SHORT | LONG | LONG vs pooled |
|---|---|---|---|---|
| **pooled_all / L-SML** | **35.92** | **41.10** | **30.74** | reference |
| per_model / L-SML | 35.49 | 40.88 | 30.09 | −0.65 [−1.36, +0.06] |
| per_model_pb_only / L-SML | 35.17 | 40.43 | 29.92 | −0.82 [−1.87, +0.21] |
| per_cell / L-SML | 34.32 | 39.25 | 29.39 | **−1.34 [−2.51, −0.17]** |
| pooled_all / equal | 32.59 | 36.52 | 28.66 | −2.08 [−3.59, −0.60] |

**Narrowing the fitting scope does not help on the long subsets — it runs from neutral
to actively harmful.** Fitting each model on its own costs 0.65 pp on the long group
with an interval that just covers zero. Fitting each cell on its own costs 1.34 pp and
the interval excludes zero. Per subset, neither scope improves a single one of the four
long cells.

This is what the per-cell covariance screen predicted: model identity moves the marginal
correlation by 0.040–0.045, at or barely above the within-cell split-half baseline, so
there was very little for a per-model fit to fit differently. The screen measured the
covariance; this measures the accuracy; they agree.

**And the PRMBench hypothesis fails.** The session report named "exclude PRMBench from
the ProcessBench fit" as the cheapest remaining gain, on the evidence that the pooled
standardizer sits 0.040 from PRMBench and 0.084–0.106 from every ProcessBench cell.
Tested directly, it is **−0.82 pp on the long group, interval covering zero**. Being the
nearest thing to the pooled fit does not make PRMBench the reason the fit is weak. That
recommendation is withdrawn.

Why a wider fit wins is worth stating plainly: L-SML is estimating a covariance over
eleven channels, and more donors estimate it better. The gain from more data exceeds
whatever the scopes were supposed to buy by matching the scoring distribution.

## Part 2 — sw_var as a twelfth channel

`sw_var_peak` is the project's most robust historical single signal (Phase 3/4): a
16-token sliding window over the entropy series, the variance inside each window, and
the **max** over windows. Verified against `feature_utils.sw_var_peak_with_window`.

The max is an answer-level readout and localization needs a step-level one, so the
adaptation keeps the **rolling variance as a token series** and lets the existing Top-10
step readout aggregate it. This is a strict generalization, not an approximation:
**the maximum of the series reproduces the original `sw_var_peak` to 2.7e-14.** Two
declared changes — the window is causal (trailing) to match the bank's contract, and the
max is replaced by the step readout. Orientation is the declared prior from its original
use, not fitted.

| arm | mean | SHORT | LONG | LONG vs 11-channel pooled |
|---|---|---|---|---|
| bank12 pooled / L-SML | 34.11 | 39.22 | 29.00 | **−1.74 [−2.80, −0.70]** |
| bank12 per_model / L-SML | 34.25 | 39.70 | 28.80 | **−1.94 [−2.97, −0.91]** |
| bank12 per_cell / L-SML | 33.88 | 38.95 | 28.81 | **−1.92 [−3.22, −0.65]** |

**Adding sw_var hurts, and it hurts on the long subsets specifically** — every interval
excludes zero. Per cell it drops OlympiadBench-4B from 30.56 to 29.20 and Omni-MATH-4B
from 30.96 to 28.99.

### Why — and it is not that the feature is worthless

Standalone gate-free SLA of every channel, as an answer-standardized step readout:

| channel | SLA | | channel | SLA |
|---|---|---|---|---|
| q15_VE1 | 34.83 | | bocpd_p0 | 27.37 |
| logprob_margin | 34.36 | | **sw_var16** | **25.99** |
| q15_H1 | 33.64 | | energy_innovation | 17.82 |
| true_tail50 | 29.52 | | chosen_surprisal | 17.56 |
| energy_level | 28.88 | | dominant_freq16 | 17.40 |
| | | | top15_turnover | 17.12 |
| | | | top50_js | 16.52 |
| | | | *(chance)* | *16.58* |

sw_var carries **real signal** — 25.99 against a 16.58 chance floor, seventh of twelve,
ahead of five channels already in the bank. It fails for a different reason:

**It is largely a transform of a channel the bank already has.** Its correlation with
`q15_H1` — the entropy series it is computed from — is **+0.730**, and it sits at
0.52–0.58 with four more. It adds a twelfth column that is mostly a restatement of the
first, which dilutes the fusion without adding a direction.

The honest framing of the historical success: sw_var_peak earned its record on
**answer-level hallucination detection**, where the max-over-windows *was* the readout
and it competed against other answer-level scalars. Here the task is step localization,
the readout is a Top-10 mean inside a step, and the entropy it is built from is already
a channel. Different task, different readout, and the signal it carries is already
represented.

### A separate observation from that table

**Five of the eleven channels are at or barely above chance on their own**: `top50_js`
16.52, `top15_turnover` 17.12, `dominant_freq16` 17.40, `chosen_surprisal` 17.56,
`energy_innovation` 17.82, against a 16.58 floor. `chosen_surprisal` — the channel whose
sign flip drove much of the Stage B analysis — is individually indistinguishable from
chance. Weak channels can still earn their place through fusion, so this is not by
itself an argument for removing them, but it does say the bank is carrying far less
independent evidence than eleven columns suggest, which is consistent with the
conditional participation ratio of 4.57 against a shuffled null of 3.29.

## Note on the aggregation bug

The first run of this script reported `pooled_all` at 35.7983 and failed the anchor.
`group_means` keyed its per-subset dictionary by `cell[3:-3]`, which maps both
`pb_gsm8k_q4` and `pb_gsm8k_q8` to `gsm8k` and silently keeps only the second — so every
aggregate and every bootstrap interval was computed on Qwen3-8B alone. 35.7975 is exactly
the mean of the four q8 cells. This is the same bug already fixed once this session in
the length-axis script, written a second time in a new function. The anchor assertion is
now hard (1e-9, stop rather than print), and the per-subset table was never affected
because it reads the per-cell values directly.
