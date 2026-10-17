# Red team: position_prior_v1 run_20260929 (protocol 7082b1e7d + A1 a9c448be8 + A2 90d2345b7; run 0e32cb04c)

- **A** recomputation with a fresh script (no import of the runner or the module): all 56 bank-arm within-AUCs, 5,000-draw
  source-group bootstrap (707 groups; 20 seeds for the permuted-control margin), the structural identity prior term =
  logit(pi_b)/a per fold, ProcessBench macro-8 and pooled hits.
- **B** coverage: all arms/folds/answers; 40 fold cells, 12 length-tercile cells, 32 error-type cells, early/middle/late first
  error, an exact pair decomposition by position bin (148,384 error-correct pairs), the B23 failure, a dose-matched straight-line
  check, ProcessBench per cell (32) and argmax moves.
- **C** nulls and math: within-answer shuffle and same-length swap (200 permutations), clustered z; a post-hoc dose check
  S + lambda (PRIOR - S); the shape of pi_b vs the true rate and the mark density; what the slope a estimates (refit of all 40
  fits; one-bin posteriors; true-class a); double counting; the model's own posterior (DSPOST).

Scripts in the session scratchpad (`rt_prior_A*.py`, `rt_prior_B1-B6*.py`, `rt_prior_C*.py`). The author spot-checked the dose
curve independently (B16 GRP: lambda 0 / 1 / 2 / 4 = 0.7907 / 0.8056 / 0.8097 / 0.7964; B54: 0.7628 / 0.7765 / 0.7878 / 0.7968).
Everything that uses labels beyond the evaluation is post hoc and descriptive.

| CLAIM | VERDICT | EVIDENCE |
|---|---|---|
| 1. The grouped prior version beats the grouped position channel (Step 460) on all 4 banks: +0.0069 / +0.0037 / +0.0065 / +0.0065 within-AUC, Bonferroni above 0; the permuted control is not above GRP; adopted by the frozen rule (GRP_PRIOR 0.8056 / 0.7697 / 0.7853 / 0.7765) | **numbers confirmed; adoption formally valid, with a borderline control** | **A**: all values within 0.00005; own Bonferroni lower bounds 0.0051 / 0.0025 / 0.0049 / 0.0050. The B35 permuted control GRP_PRIOR_PERM - GRP is +0.00039 [-0.00002, +0.00081]; over 20 seeds the lower bound is -0.00004 to -0.00001: condition (b) holds by about 2e-5. **B**: 20/20 folds, 12/12 length terciles, PRMScore up 4/4 pooled and 19/20 folds. **C**: shuffle p = 0/200, clustered z 8.1-11.8 |
| 2. The plain-average prior version is not adopted: +0.0043 / -0.0006 / +0.0144 / +0.0030 (fails on B23) | **confirmed** | **A**: B23 Bonferroni lower bound -0.0016 (20 seeds -0.00166 to -0.00153). **B**: B23 negative in 5/5 folds; its estimated weight is only 1.14x the channel weight 1/(p-1) and its prior is the least linear (flat over bins 0-5). PRMScore of the plain-average prior version rises over the channel version on 1/4 banks only |
| 3. The gain comes from step order and is not an artefact | **confirmed; entirely positional** | **C**: order-permuted arms -0.0002 to +0.0004 (8/8, abs z <= 1.81); same-length swap gain >= observed in 16/16 positive contrasts. **B**: against GRP the gain is 100% cross-bin pairs (same-bin contribution exactly 0) |
| 4. The model recovers "later is riskier" without an assumed direction (pi_b last > first in 40/40 fits) | **confirmed as agreement, weakened as error rate** | **B/C**: 40/40. But pi_b follows where the channels' marks agree: it rises 1.9-2.8x where the true rate rises about 8x (0.022 -> 0.229 by bin 8), and it keeps rising in the last bin where the true rate falls (0.234 -> 0.177; pi_9 > pi_8 in 35/40). The last-bin mismatch itself costs <= 0.0003; the costly errors are the scale and the missing early rise (a true-rate-shaped prior at the same a would add +0.008 to +0.017) |
| 5. The prior's advantage over the channel comes from its estimated shape | **refuted: it is the size of the weight** | **B**: a straight line in the step index with the same implied weight c matches GRP_PRIOR / BASE_PRIOR within 0.0011 on 8/8 bank-families |
| 6. The weight is principled: estimated from the model, neither double counting nor far from what works (protocol principle) | **refuted as "the right weight"; confirmed only as "larger than the channel-count weight"** | **C** (post hoc, label-using): the best lambda for within-AUC is 1.9 / 4.3 / 2.1 / 5.1 (grouped) and 2.0 / 4.0 / 2.3 / 4.5 (plain average); per-fold best 1.5-5.75 in 40/40 folds; lambda chosen on four folds beats lambda = 1 on the fifth by +0.0039 to +0.0210 (8/8). The model UNDER-weights position, so harmful double counting is not what limits it. Cause: the slope a is inflated 2.05-3.61x relative to the true-class slope (8/8): the EM posteriors are nearly hard (42-100% of fit rows below 0.02 or above 0.98), i.e. the latent class behaves like a threshold on the same channels that make up S, so content looks more separating than it is. Posteriors with a constant prevalence change a by only -3.8% to +2.4%. In SD units of S the best weight is nearly flat across banks (0.39-0.61), the model's falls 2.2x from B16 to B54 (1/(p-1) falls 2.7x): most of the bank-size dependence remains. The model's own posterior (DSPOST) under-weights position 8-24x (naive-Bayes content overconfident) |
| 7. ProcessBench cost | **confirmed, and the task conflict is total** | **A/B**: GRP_PRIOR below GRP on 4/4 banks (macro -2.82 / -1.04 / -1.47 / -0.03 points; B54's -0.03 is noise), below GRP_POS on 3/4 (+0.55 on B54); 26/32 cells lost vs GRP, 21/32 vs GRP_POS; 2,435 of 2,448 moved argmaxes moved later. **C**: ProcessBench macro-8 is best at lambda = 0 (6/8) or 0.25 (2/8); the PRMBench-best lambda costs 3-6 points |
| 8. Who pays | **disclosed** | **B**: answers whose first error is in the first third (1,195 of 6,030) lose under GRP_PRIOR vs GRP_POS on 4/4 banks (-0.0054 to -0.0084) and vs GRP on 4/4 (-0.016 to -0.030); late-error answers gain +0.025 to +0.042 vs GRP. 70% of answers unchanged vs GRP_POS (median 0); confidence + counterfactual + deception supply 67-103% of the gain; domain_inconsistency loses vs GRP on 4/4 banks |
| 9. 5 vs 10 vs 20 bins within 0.001 | **weakened** | **B**: each within 0.001 of 10 bins, but 20 bins beats 10 on 4/4 banks and 5 vs 20 differs by 0.00185 on B35 (5/5 folds); no bin sensitivity for the adopted grouped version |

## Decision under the frozen rule (A1, A2)
The grouped prior version (GRP_PRIOR) replaces the grouped position channel of Step 460; the plain average keeps the Step 460
position channel (BASE_POS), since its prior version fails on B23. The red-team reading: the adopted change is a LARGER position
weight, estimated label-free, not a better-shaped prior; it is still 2-5x below what PRMBench rewards because the label-free
latent class is a consensus of the same channels (slope a inflated), and it costs ProcessBench, which prefers no position weight.

## What the author's reading missed
- The protocol called the weight "estimated from the model" as if that made it the right one; the model's currency is inflated by
  the same circularity the declared limit 2 named, and a dose check was not in the protocol.
- The 10-bin shape was presented as the mechanism; a straight line with the same weight does as well.
- The permuted-control condition was treated as easily met; on B35 it passes by about 2e-5.
