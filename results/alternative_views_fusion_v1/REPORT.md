# Step395 — corrected alternative views and fusion

Complete on 16 September 2026: all13,769 answers,145,597 steps,6,968,779 tokens.
28 new methods plus30 frozen references; no new model training or gate change.

The seven views are provided-token surprisal, censored rank, probability mass
above the provided token, top1/provided-token log-probability gap, log tail15,
log tail50, and digit disagreement. Raw tail15/50 are additional controls.
The tails were independently recomputed from saved log probabilities without
subtracting logsumexp again. Both digit and innovation5 reference scores match.

| Method | PB % | within-AUC | PRMScore |
|---|---:|---:|---:|
| innovation5 |39.8314|.760293|.638830|
| +digit .25 |41.3300|.776036|.649780|
| Previous TCN+digit sum, secondary |42.0781|.774945|.652284|
| +mass above |40.0411|.762159|.643460|
| +logtail15 |38.8583|.758550|.641375|
| +logtail50 |38.5761|.759324|.642407|
| +raw tail15 |37.0738|.754336|.643348|
| +raw tail50 |36.5538|.754394|.643881|
| new7 equal |39.5741|.765523|.648901|
| new7 family equal |40.1790|.770944|.651520|
| new7 IU |39.0974|.763182|.647318|
| new7 diagonal shrinkage + IU |39.0973|.763214|.647342|
| new7 block shrinkage + IU |39.0789|.762915|.647187|
| augmented12 equal |39.8702|.764164|.646543|
| augmented12 family equal |40.3862|.768851|.648635|
| augmented12 IU |39.5951|.761803|.644356|
| augmented12 diagonal shrinkage + IU |39.5951|.761803|.644334|
| augmented12 block shrinkage + IU |39.5951|.761858|.644305|

Every new auxiliary uses the same .25 standardized correction to innovation5
ONCE. TCN+digit sum is a previous two-correction reference, not an amplitude
matched control. The new bank is not Step393's direct-bank replacement.
Full58-row results and all metrics are in METRICS.csv/JSON and REPORT.html.

## What this establishes

Eight primary pairs x two endpoints,10,000 paired source-group bootstrap draws,
CI99.6875%. IU reduces within relative to equal in both banks:
- new7: -.002341, CI[-.003365,-.001355]; PB-.4767pp, CI[-1.1549,+.1816].
- augmented12: -.002362, CI[-.003358,-.001394]; PB-.2751pp, CI[-.9398,+.3229].

Family equal improves within relative to equal:
- new7:+.005421, CI[+.003985,+.007049].
- augmented12:+.004687, CI[+.003432,+.006018].
Both PB intervals include zero. Neither family-equal bank exceeds digit025 on
PB/within point estimates; new7 family equal does offer a PRMScore tradeoff.
No positive primary IU/shrinkage gain. New7 block shrinkage slightly reduces
within relative to IU, CI[-.000673,-.000013]. Do not generalize to all shrinkage
estimators or identify confidence intervals containing zero with equivalence.

Median diagonal/block shrinkage alpha: new7 .13148/.05909;
augmented12 .04664/.02120. All fits native, zero equal fallbacks.
Thus the near-identical quality is not an alpha=0/fallback implementation artifact.
Shrinking to identity also preserves covariance eigenvectors, which limits the
changes possible under the retained two-PC solve; the block target is a separate
control that can change directions.

Among10,282 variable-digit answers, digit's IU coefficient is negative in2.88%
of new7 and20.23% of augmented12, versus almost100% in the previous6-view bank.
Median digit share of total absolute coefficients is2.44% and0.55%, respectively.
The previous sign-failure explanation cannot simply be reused: this bank mostly
retains the sign but assigns the useful sparse view little relative weight.
This is a weight diagnostic, not an identified causal explanation of performance.

## Complementary errors, not uncorrelated input features

On the same4,442 PB error answers, correlations of raw peak-failure events:
- surprisal/gap:.981; logtail15/logtail50:.864;
- logtail15/innovation5:.665; digit/innovation5:.103.

Digit alone has794 raw hits where innovation5 misses, but loses989 base raw hits.
Its additive correction gains310/loses239 final hits versus base. This is useful
operational complementarity, not evidence that latent U-PCR errors are independent.
The full report includes pairwise joint failures, phi, unique hits, cell/position
strata, and answer-balanced PRMB ranking losses on the same step pairs.

Among the historical707 open common misses, standalone digit hits145 (143 with
positive digit evidence, from Step393's tie audit); mass-above43; rank31;
surprisal27; gap29; logtail15/50 zero/four. Those standalone successes are not
all retained after bounded correction: digit025 recovers28, new7 family equal5,
augmented12 family equal2. No global information-impossibility conclusion follows.

## Decision and next insertion points

Retain digit025, digit1 and TCN+digit as development candidates/Pareto references;
do not replace them with these broader learned banks. The tails remain useful
gate evidence historically, but do not help as local corrections in this recipe.

The [pipeline map](../../docs/reviews/fusion_insertion_map_2026-09-16.md) separates
raw auxiliary fusion, predictor-residual fusion, background fusion BEFORE
innovation, the gate, first-error decoding and final-answer correctness.
It records earlier leaders, tested failures and proposed comparisons.
The next bounded questions are digit evidence conditional on tail at the gate,
and fusion of forecasts of the same observable feature before forming a residual.
Forecast-error covariance can be estimated without correctness labels; better
forecasting still needs a localization test. Neither follow-up was launched here.

## Integrity and limits

- Source hashes; full-token tail-formula check; provided surprisal agrees exactly
  with stored log probability wherever the token appears in top50.
- All-answer base/digit replay; canonical IU checks on up to3 answers per source
  cell for each of six fitted heads, maximum weight discrepancy0.
- Independent PB/within computation for all58 methods; all30 reference rows exact.
- Five numerical tests pass, including alpha0 identity, PSD, censored rank,
  constants and exclusion of labels from fit APIs.
- All new heads answer-local, with other-source-fold PRMScore quantiles; neural
  references retain their prior nested calibration. Fixed gate is transductive.
- Scoring152.18s, evaluation72.82s. Old flow queue unchanged.
- Current population and feature screening use development labels. Primary
  correction covers only this predefined family, not historical adaptive search.
  Secondary95% intervals do not validate a selected winner externally.
- Large raw data/checkpoints and score arrays stay local with recorded hashes;
  code, compact metrics, figures, error ledger and reports are the commit payload.
