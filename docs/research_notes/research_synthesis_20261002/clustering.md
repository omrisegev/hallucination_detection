# lsml_merge_step_v1: does one added clustering step make L-SML worth its weights?

Omri, 2026-09-28: "it makes no sense that the clustering does not work for us; it has to be improved" - without replacing L-SML,
"only maybe add another step"; and use the Dawid-Skene estimates better (drop channels with estimated balanced accuracy in
[0.45, 0.55]). This run tests both on five banks.

- Protocol `PROTOCOL.json`, frozen at 6665e28ca (before any score). Runner `scripts/experiments/lsml_merge_step_run.py`, pre-run
  review fixes a8970f3a4 (no blocker; re-smoke identical). Run `run_20260928`, outputs 577ab40e4: COMPLETE, 0 failed fits.
- Replays: every er_generality_v1 arm reproduced exactly (0.0); on B13 the automatic merge equals the stage-B2 manual level merge in
  5/5 folds and its scores are identical. Red team of three agents: `run_20260928/RED_TEAM.md` (numbers confirmed, reading weakened).
- The merge rule was designed after a label-free diagnosis on these same banks: development evidence, not confirmation.

## The added step
After L-SML finds its groups (or after the binary-mark partition), merge two groups when rho = lambda_2(union) / min(lambda_1(A),
lambda_1(B)) < 0.5, smallest first, never below 3 groups; then L-SML's own weights on the merged partition. Correction to the
diagnosis: rho is NOT size-free. For two halves of one factor rho = (1-r)/(1+(m-1)r), m = size of the smaller half, so larger halves
merge at weaker correlation; only the independence reference (rho >= 1) does not depend on size.

## Results (PRMBench within-answer AUC, 6,030 answers; DS filter = keep pi_hat > 0.5)

| Arm | B13 | B16 (B13 + 3 digit) | B20 | B32 | B51 |
|---|---:|---:|---:|---:|---:|
| DS filter, plain average | **0.7802** | **0.7918** | 0.7540 | 0.7475 | **0.7553** |
| DS filter, L-SML as is | 0.7751 | 0.7838 | **0.7556** | 0.7185 | 0.7499 |
| DS filter, L-SML partition + step + L-SML weights | 0.7731 | 0.7823 | 0.7553 | 0.7185 | 0.7499 |
| DS filter, binary partition + step + L-SML weights | 0.7788 | 0.7849 | 0.7455 | 0.7236 | 0.7477 |
| DS filter, binary partition + step, equal group weights | 0.7783 | 0.7898 | 0.7439 | 0.7445 | 0.7304 |
| Band rule without flipping, average | 0.7782 | 0.7911 | 0.7524 | **0.7563** | 0.7494 |
| Band rule with flipping, average | 0.7782 | 0.7911 | 0.7522 | 0.7406 | 0.7464 |
| Position-adjusted: plain average | 0.7622 | 0.7684 | 0.7466 | 0.7349 | 0.7464 |
| Position-adjusted: binary partition + step + L-SML | 0.7567 | 0.7630 | 0.7488 | 0.7409 | 0.7523 |

PRMScore follows the same order (plain average 0.6565 / 0.6618 / 0.6358 / 0.6286 / 0.6378). References: fam421 0.7801 / 0.6573,
CT7 0.7724 / 0.6458. ProcessBench SLA (beside): plain average 0.3750 / 0.4126 / 0.3629 / 0.3475 / 0.3505; CT7 0.3989.

## Answers

1. **The step does what it was built for on the partition.** On the raw DS-filtered partitions it unified the level family in every
   fold where it had been split (binary: B13 5/5, B16 3/5, B20 4/5, B51 5/5; L-SML's own: B13 5/5, B16 3/5, B20 1/5), merged nothing
   else, and never fired on B32. On position-adjusted and band-rule partitions it is less clean (6 merges of a non-level group at rho
   0.48-0.49994; level left split 3 times).
2. **It does not make L-SML beat averaging (central question, frozen rule: not established; not even "at least as good").** Binary
   partition + step + L-SML minus plain average: B13 -0.0014 (n.s.), B16 -0.0069, B20 -0.0085, B32 -0.0239, B51 -0.0076
   (Bonferroni below 0). But: the B13, B20 and B51 losses are fully positional and B32's about 2/3 (same-length swap null); on
   the position-adjusted banks L-SML with the step beats averaging on B20 (+0.0022), B32 (+0.0060), B51 (+0.0058) and loses on
   B13/B16 (-0.0055, -0.0054). On B32 the step never fired: that loss is the binary partition plus L-SML weights.
3. **The step does not improve L-SML on its own partition** (B13 -0.0021, B16 -0.0015, n.s.; inactive elsewhere).
4. **Where is the limit?** On B13 the merged partition has 3 groups, so L-SML's own 3-unit guard sets the cross-group and two
   within-group weights equal: "L-SML" there is an equal-per-group score. The partition alone (equal group weights) is below plain
   averaging on every bank; SML weights on top hurt on B32/B16 and rescue on B51. In stage B2 even label-fitted weights on the merged
   B13 partition reached only 0.7823 (+0.002 over averaging): on B13 there is almost no room for weights at all.
5. **The band rule does not beat the DS filter** (frozen rule: not established; loss on B51 -0.0089). Dropping the band helps only on
   B32 (+0.0088, 5/5 folds). Flipping hurts on raw B32 (-0.0157, entirely positional: it helps +0.0195 on the position-adjusted
   B32) and on B51 (-0.0030, fragile; dilution by pro-positional flipped channels).
6. **Omri's question - does SML give reversed channels a negative weight?** Rarely: 6 of 19 bank x channel cases. The sign inside a
   group follows the channel's correlation with its group mates (leading eigenvector), not its accuracy.
7. **Post hoc, outside the frozen family: the three digit features.** B16 vs B13 plain average +0.0116 [0.0091, 0.0142], about 82%
   positional (swap null +0.0095); after position adjustment +0.0063 [0.0037, 0.0089]; removing the three digit channels by
   within-answer permutation removes the gain. ProcessBench 0.3750 -> 0.4126 (all 8 cells positive; significant on gsm8k q4 and MATH
   q4/q8 only). The digit features also pull the DS estimates (estimated prevalence 0.41 vs 0.28 on B13). Needs a frozen test and
   Omri's confirmation that they are outside the 2026-09-17 digit exclusion.

## Predictions (frozen)
P1 mostly held (the step fired exactly where level was split; "every fold" failed only where level was not split to begin with; one
L-SML-partition merge on B20); P2 held exactly; P3 C1 failed, C2 held; P4 held (all five points negative); P5 failed (flipping hurts
on B51); P6 held but is mechanical (between-group dependence falls equally for clean, error and pooled correlations).

## Bottom line
The clustering is repaired in the sense intended (the level family is one group), but repairing it does not give L-SML an advantage
over plain averaging on the raw banks. The losses are largely positional; after removing position, L-SML with the step is ahead
on the three larger banks. Hypothesis, not tested: position acts as a shared factor that L-SML's weights follow (it violates the
conditional-independence assumption), so the next lever would be position, not the partition. The position-adjusted banks score
below the raw plain average, so removing position from the scores is not itself the answer.

## Addendum 2026-09-28: direct check of the L-SML assumption (REPORT_HE.html, report_data.py)

Class-conditional correlations (fold-0 partitions, 55,623 PRMBench fit-fold steps, 7,869 errors; labels for diagnosis only):
between groups of the merged binary partition the mean |r| (clean and error steps averaged) is 0.31 (B13), 0.23 (B16), 0.30 (B20),
0.14 (B32), 0.19 (B51), against 0.40 / 0.32 / 0.36 / 0.18 / 0.29 for random partitions of the same sizes (every found partition beats
all 2,000 random ones); worst between-group pairs 0.72-0.86 (B13/B16: chosen_surprisal with q15_VE1/q15_H1; B20:
ct7_H0lim_prefix_innovation with ct7_ve0 and the level channels). Dependence is higher on error steps than on clean steps in every
bank. On top-20% marks it is lower (0.08-0.21) but not zero. So the clustering finds real structure but the conditional-independence
assumption does not hold. Removing the position profile does NOT lower the between-group dependence (it rises slightly: B13 0.31 ->
0.33), which contradicts the hypothesis in the bottom line above that position is the shared factor violating the assumption; why
L-SML + step leads on the position-adjusted banks remains unexplained. The digit features always form their own group.
