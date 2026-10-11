# What the LR diagnostic establishes

All ten banks, five outer source-group folds and their distinct inner maps
finished. All 50 partitions pass group-disjointness, input SHA256 checks and
saved affine test-score replay (maximum absolute difference **0.0**). Native
all-population L-SML/IU-PCR exports separately replay in **20/20** cases within
**6.34e-14**, after matching the export's answer-z representation.

One arm is explicitly undefined: N11 L-SML on the LR-selected half-bank chose
K=2 in outer fold 4. Its 4,809-answer AUC is a **partial-population diagnostic**,
not a full-panel result. No fallback was substituted.

## The gap grows with the bank, but depends on the localization metric

At N16, new source-group OOF within-answer AUC is **0.801739 LR**, **0.778901
L-SML** and **0.780467 IU-PCR**. Top1-on-error reverses this ordering: **0.578293
LR**, **0.598012 L-SML**, **0.595195 IU-PCR**. The corresponding paired top1
differences against LR are +0.019718 [0.009576, 0.029573] and +0.016901
[0.006809, 0.026381].

At N56, LR rises to **0.821129 AUC / 0.630157 top1**, while L-SML has
**0.770411 / 0.570008** and IU-PCR **0.764067 / 0.591715**. This supports the
requested observation that expanding the pool exposes a larger supervised
gap. It does not make N16 LR better on every localization metric.

These are new OOF references, not certified replays of the historical LR
folds. They are internal development-data validation, not an untouched
external benchmark.

## Selection and weighting both matter; LR selection is not directly portable

All crossover masks have the same size, and each selector is trained again
inside every validation partition. At N56, LR on its own half-mask reaches
**0.819491**, versus **0.805976** on the L-SML half-mask and **0.798368** on the
IU-PCR half-mask. LR thus benefits from a different choice of channels.

However, with that *same LR mask*, L-SML scores **0.758991** and IU-PCR
**0.759766**. The matched-mask LR weighting advantages are **+0.060500**
[0.055150, 0.065650] and **+0.059725** [0.054393, 0.065047]. Merely transferring
the LR mask cannot recover the gap. L-SML is actually worse on the LR mask
than on its own half-mask by -0.005985 [-0.008148, -0.003868]; the IU-PCR mask
difference is small and uncertain.

At N16, transferring the LR mask improves L-SML AUC by +0.010252 but does not
clearly improve IU-PCR (-0.001164, interval crosses zero). The pattern changes
as the bank grows. The factorial differences and interaction intervals are
in `CROSSOVER_FACTORS.csv`; they are metric differences, not a causal/additive
attribution of the full-bank gap.

No new label-free selector is promoted. The next scientifically justified
question is how to recover useful orientation/contrasts and weight them,
rather than assuming deletion of low-weight channels alone solves the gap.

## Large coefficients can be cancelling contrasts

In N16, LR assigns approximately +0.438 / -0.440 of normalized signed weight
to `digit_alternative` / `digit_alternative_innovation`. Their input
correlation is **0.9998205**, and they account for about **88% of absolute
coefficient mass**. This is a near-collinear contrast; it is not evidence
that digits supply 88% of predictive information.

Fixed-C family-omission probes give a more useful conditional check. At N16,
removing uncertainty levels lowers AUC by **0.015748** and surprise channels
by **0.015575**, versus **0.009997** for digits. At N56, uncertainty levels
matter most (-0.015519), followed by variability/drift (-0.007765), digits
(-0.006609) and surprise (-0.004764); rhythm omission is close to zero.
Correlated channels remain substitutable, so these omissions are not causal
importance either.

This also explains why a naive coefficient threshold is hazardous. The
training-safe R1 threshold gives **0.782561 N16 / 0.771485 N56**; the
contrast-aware R2 gives **0.801650 / 0.820933**, close to full LR. Historical
globally selected restricted results remain labelled as leaky; these new
rules use fresh training-only C=1 selectors.

All intervals are exploratory, pointwise, source-group bootstrap intervals
conditional on fitted predictions and choices. They do not certify all
examined contrasts simultaneously, causal feature effects or a new champion.
