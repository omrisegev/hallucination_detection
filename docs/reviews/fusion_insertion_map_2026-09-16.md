# Where fusion can enter the current algorithm

This map accompanies Step395, the full-population alternative-view experiment.
Its criterion is **complementary errors with respect to the same target**, not
merely low correlation between input features. All reported benchmark results
remain development results. Proposed insertions below are not completed experiments.

The current path is:

`cached token evidence -> historical background/predictor -> innovation/residual
-> per-stream step Top10 -> auxiliary fusion -> correction to innovation5
-> peak selection + separate answer gate -> evaluation`.

The historical background and the final auxiliary fusion are different operations.
U-PCR is a covariance-based, unsupervised estimator of weights; shrinkage stabilizes
its covariance input. Neither automatically identifies which signal measures
correctness. IU here is the project's canonical two-PC U-PCR implementation,
`upcr_fit_covariance` with `IU_FIT_DEFAULTS`, numerically checked against `fast_iu`.

| Insertion | Shared target and evidence | Current evidence / leading reference | A concrete U-PCR or shrinkage experiment | What would count as useful diversity? |
|---|---|---|---|---|
| Token/step auxiliary evidence | Local error risk: distribution shape; supplied-token surprisal/rank/gap; true missing mass; digit disagreement | Step393: innovation5+digit .25 =41.3300% PB/.776036 within; TCN+digit sum=42.0781%/.774945, secondary. Step395 tests seven new views and the augmented12 bank with equal, family-equal, IU, diagonal/block shrinkage. | The current experiment: centered token covariance within answer; identical per-stream Top10 and .25 correction across heads. | Different missed first-error locations and different PRMB pairwise ranking errors. Same source probabilities or low raw correlations establish neither independence nor redundancy of those errors. |
| Predictors / residual ensemble | Predict the same present feature vector from past tokens; compare signed residual evidence | Step388 TCN40.9718%/.761592, ridge40.8472%/.761620, BOCPD40.3676%/.763223. Step389 examined all16 subsets of five predictors at sizes3/4/5 with equal/IU. Best PB IU triple41.0378% adds only six net hits versus TCN; no demonstrated primary IU gain. | Reuse aligned predictions to estimate covariance of forecast errors, possibly shrink per-feature or provenance-group covariance. Compare forecast fusion with the already tested residual fusion, holding correction amplitude/readout fixed. | Errors of forecasts for the SAME observed feature component, measured on held-out source groups. Forecast error can be measured without correctness labels, but a better forecast need not improve localization. Avoid adding the common base once per predictor. |
| Historical background before innovation | Estimate the expected current value of each feature from its own past | Prefix mean is the successful simple background in innovation5:39.8314%/.760293. Mean16, noreset and BOCPD are available; ridge/TCN predict related quantities. Position-profile controls preserve part of the innovation advantage. | For each feature separately, fuse estimates of its background, then subtract ONCE: r(t)=x(t)-sum_j w_j(t) mu_j(t). Compare frozen weights/equal and learned weights; train/validate with blocked or source-excluded next-token prediction. | Distinct prediction errors under smooth drift, abrupt changes and stable context. All candidate backgrounds must estimate the same feature in the same units and use permitted history. This is different from combining residual scalars after Top10, and has not been shown to improve our task. |
| Answer gate | Is there an annotated error anywhere? Tail mass, digit disagreement count/rate, and peak prominence might provide different evidence | Current tail15 whole-answer Top10, within-cell percentile>=.33. Step365's earlier q=.3 contract preferred tail15 token mean36.8818%; that is NOT the current gate or a matched current comparison. Digit count AUC was reported by Claude as diagnostic only. | Freeze the locator. Compare tail alone, an equal combination of genuinely different gate evidence, and IU with shrinkage under the SAME source/cell calibration and operating rule. Report clean false positives, error false negatives and end-to-end PB together. | One view catches wrong-but-confident answers that another misses without systematically flagging clean answers. Tail15/tail50 are nested; digit count/rate are related. Do not duplicate these to satisfy a minimum number of views. Current Step395 does not fit or change this gate. |
| Final first-error decision / readout | Which candidate step is the FIRST error? Peak strength, temporal context, digit disagreement, sustained/end evidence | Step392 three-view equal39.6301%/.761989; IU39.0007%/.755534. Equal context+end39.5281%/.765898 is a within tradeoff. Earliest-near-max (.25), early VE peaks and contiguous-window readouts were already explored. | Fuse aligned evidence about each step, then compare the frozen argmax to a prespecified first-error decoder. Keep the gate and input evidence identical. Do not confuse a decoder change with a learned-weight gain. | One view localizes onset while another fires on a later consequence; evaluate on first-error targets, not assumed labels for every later PB step. Lower feature correlation or oracle unions are not bounds on possible fusion. |
| Correctness of the final textual answer | Whether the final answer itself is correct | This is a different target from PB first-error localization and PRMB local ranking. historical24 is retrospective transfer, not untouched confirmation. | Requires its own aligned answer-level evidence and evaluation contract. Do not simply transfer step-local IU weights and call it a validated final-answer detector. | Error independence must be measured relative to final-answer correctness; a flawed intermediate step and an incorrect final answer are not interchangeable. |

## Why the new bank is not an independence proof

Step395 completed on all13,769 answers. New7 equal/IU are39.5741/.765523
versus39.0974/.763182; augmented12 equal/IU are39.8702/.764164 versus
39.5951/.761803. IU reduces within under the corrected primary intervals in
both banks; no positive primary IU/shrinkage gain. Family equal improves within
over column equal in both banks, but its best points40.1790/.770944 (new7) and
40.3862/.768851 (augmented12) remain below digit02541.3300/.776036 on PB/within.
New7 family equal has a PRMScore tradeoff (.651520 versus digit025 .649780),
so this is not a claim of dominance on every metric. TCN+digit sum remains the
stronger previous point candidate42.0781/.774945/.652284, with a different total
correction amplitude and secondary status.

The correctly recomputed tails do not rescue this localization recipe:
base+logtail15=38.8583/.758550; base+logtail50=38.5761/.759324;
base+rawtail15=37.0738/.754336; base+rawtail50=36.5538/.754394.
The best non-digit singleton correction is mass-above40.0411/.762159, a small
development tradeoff relative to base, with PB CI including zero.

Raw localization-error phi is .981 for surprisal/gap, .864 for the two logtails,
.665 for logtail15/base, and .103 for digit/base. Digit has794 unique raw hits
versus base but also loses989 base raw hits. This supports complementarity of
operational mistakes, not independence or standalone superiority. The current
digit025 correction gains310 and loses239 final PB hits versus base. The
population and these diagnostics were exposed to development selection.

Surprisal, gap, rank and mass-above all depend on the provided token and the
scorer's distribution. Tail15 and tail50 share most of their probability mass.
Digit disagreement uses token identity and a task-specific event, but remains
derived from the same forward pass. These are candidate sources of complementary
errors, not six or seven independent experts by construction.

For PB, Step395 measures the event "this view's raw peak misses the first error"
on the same4,442 error answers, before the shared gate. For PRMB it measures
misordering of the same positive/negative step pairs, with half loss for ties
and equal weight per answer. Cell and first-error-position strata help expose
shared difficulty. These diagnostics use development labels and do not enter
weight fitting. They do not directly test the latent continuous residual
assumption f_i-y in U-PCR, especially with uncalibrated heterogeneous scores.

Thus useful complementary views can improve a fixed positive combination while
native IU still subtracts the helpful view. Conversely shrinkage can improve IU
relative to an unstable fit without beating the useful simple combination.
Both comparisons must be reported.

## Decision after Step395

Keep the successful digit and TCN+digit correction references. Do not add a broad
new-view bank or another covariance-only weight sweep as the next default step.
The existing features offer actual complementarity, but the tested moment-based
estimator does not turn it into a better detector.

Two bounded future questions are justified at different insertion points:

1. **Gate:** does digit disagreement add clean/error discrimination conditional
   on tail mass, and does that improve end-to-end PB with the locator frozen?
   Start with that matched diagnostic, then prespecify any equal/IU comparison.
   The localization-error phi measured here is not evidence about gate errors.
   A third candidate such as peak prominence must earn its place; do not split
   tail15/tail50 or count/rate into nominally independent experts.
2. **Background forecasting:** combine the existing forecasts for the same
   observable feature, then form a residual. Here forecast errors against the
   held-out observed feature are available without correctness labels. Shrinking
   their error covariance has a direct estimation target and avoids pretending
   that latent correctness reliability is identified from feature covariance.
   Compare this with U-PCR on the forecasts and equal weights using the same
   exclusions/readout. Better self-supervised forecast MSE still must translate
   into localization gains. This differs from Step389's fusion of scalar signed
   residuals after the predictors have been defined.

Neither proposal is a new result, and neither requires restarting the flow queue
or assuming that every component should use U-PCR. External confirmation of the
retained candidates remains necessary before claiming a general improvement.

## Relationship to FM, DiFlo and DOT

FM and DiFlo are alternative conditional models. DOT is a flow-derived readout,
not a third independent predictor. They can supply forecast/residual or trajectory
evidence at the predictor/auxiliary insertions after complete aligned evaluation.
Their current partial-fold results establish feasibility, not full-benchmark
quality or independent errors. The old queue remains paused; Step395 does not
resume it. More complex model families are not justified solely by adding another
column to an IU input bank.

## Sources within the project

- [Step395 frozen experiment](../experiments/ALTERNATIVE_VIEWS_FUSION_20260915.md)
- [Step395 full results](../../results/alternative_views_fusion_v1/REPORT.html)
- [Digit replay and direct fusion](../../results/digit_fusion_v1/REPORT.html)
- [Predictor combinations](../../results/predictor_subset_iu_v1/REPORT.html)
- [Step-evidence fusion](../../results/step_evidence_fusion_v1/REPORT.html)
- [Earlier gate feature/readout selection](../../results/gate_feature_readout_selection_v1/REPORT.md)
- [Tail normalization audit](claude_tail_normalization_audit_2026-09-15.md)
