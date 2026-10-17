# algorithm_decisions_v1: deciding the open components of the label-free algorithm

Omri, 2026-09-28: decide filter, clustering, use of the Dawid-Skene estimates within and between groups, within/between weights,
binary vs continuous; add the digit features to every bank; "neutralize position and use L-SML"; compare with Mind the Gap; end
with ONE candidate for the new benchmarks and an algorithm with clear stages for the advisors.

- Protocol `PROTOCOL.json`, frozen at a70dfc591 (before any score). Runner `scripts/experiments/algorithm_decisions_run.py` with
  pre-run review fixes and a parallel fit/assemble split (ec3ec5e42; split smoke = sequential smoke to 2.2e-13). Helper
  `scripts/experiments/ds_group_weights.py` (8 tests). Run `run_20260928`, outputs 7ec9205b2: COMPLETE, 0 failed fits,
  331 arms x 5 folds. Replays exact: lsml_merge_step_v1 arms 8.9e-16; stage-B2 B_sml__merge and B_oracle__merge 0.0 in 5/5 folds.
- Red team: `run_20260928/RED_TEAM.md` (numbers confirmed; two claims weakened; the proposed DSM mechanism refuted).
- Development evidence: the selection used PRMBench outcomes; the untouched test of the frozen candidate is the new benchmarks.

## Grid
8 banks (13/20/32/51 channels, each without and with digit_alternative, digit_spread, digit_alternative_innovation) x position
{P0 raw, P1 learn without position then score the original channels, P2 learn and score without position} x within-group
{EQ mean, SML, HEM latent-group EM} x between-group {EQ, SML, DSM = MLE weights from Dawid-Skene on group-score marks, HEM},
after the decided stages (DS filter pi_hat > 0.5; binary-mark partition + absorption merge). Baseline: DS filter then plain average.

## Results (PRMBench within-answer AUC, 6,030 answers)

| Arm (P0) | 13 | 13+d | 20 | 20+d | 32 | 32+d | 51 | 51+d | mean 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **DS filter, plain average (candidate)** | 0.7802 | 0.7918 | 0.7540 | 0.7691 | 0.7475 | 0.7622 | 0.7553 | 0.7646 | **0.7656** |
| hem within, DS-estimate between (best mean) | 0.7827 | 0.7923 | 0.7526 | 0.7594 | 0.7620 | 0.7723 | 0.7479 | 0.7670 | 0.7670 |
| mean within, DS-estimate between (= B2 B_sml__merge on 13) | 0.7818 | 0.7907 | 0.7527 | 0.7612 | 0.7604 | 0.7713 | 0.7420 | 0.7628 | 0.7654 |
| mean within, equal between (partition alone) | 0.7783 | 0.7898 | 0.7439 | 0.7589 | 0.7445 | 0.7607 | 0.7304 | 0.7462 | 0.7566 |
| L-SML with merge | 0.7788 | 0.7849 | 0.7455 | 0.7519 | 0.7236 | 0.7370 | 0.7477 | 0.7564 | 0.7532 |
| label-using ceiling (true group accuracies) | 0.7823 | 0.7949 | 0.7508 | 0.7664 | 0.7590 | 0.7730 | 0.7445 | 0.7630 | 0.7667 |

References: fam421 0.7801, CT7 0.7724. PRMScore and ProcessBench follow the same order; the highest ProcessBench macro-8 is the
13+digits plain average, 0.4126 (Mind the Gap published 0.3926, our reproduction 0.3722, CT7 0.3989).

## Decisions
1. **Filter: DS, keep pi_hat > 0.5** (decided before; digits leave its decisions unchanged).
2. **Position: do not neutralize (P0).** Learning without position changes the plain average by at most +0.0015 and loses on
   32/32+d (content losses); full removal loses on every bank (it discards positional signal that is real: step index alone 0.66).
3. **Fusion: the frozen rule selects the plain average.** No variant avoided a significant loss on all 8 banks, so no clustering or
   group weighting enters the candidate. Closest: hem within + DS-estimate between (mean +0.0014) - broad content gains on 32/32+d
   (+0.0145 / +0.0100; content after the position null +0.0171 / +0.0146), thin gains on 13 and 51+d (concentrated in 1% of
   answers), broad losses on 20+d (-0.0097) and 51 (-0.0074).
4. **Estimates:** as a filter, yes. As between-group weights, DS on group marks beats equal group weights on content on 6/8 banks
   and matches the label-using ceiling where it helps (32/32+d), but the partition itself costs more than the weights recover on most
   banks. Within groups, hem weights do not help consistently. DS overestimates prevalence (0.28-0.41 on channels, 0.19-0.26 on
   groups, vs 0.14) and stretches the weight ratios, mainly toward the level group (top weight in 40/40 bank-folds).
5. **L-SML:** below the plain average on all 8 banks, raw and learned without position (7/8 and 8/8 significant).
6. **Binary vs continuous:** binary marks for estimates (filter, partition, DS/hem), continuous values for what is combined -
   unchanged; no configuration of the combination beats the plain average of the continuous values.
7. **Digits:** raise the plain average on every bank (+0.0093 to +0.0152; ProcessBench +0.022 to +0.038); content beyond position is
   significant on 20, 32, 51 (+0.0048 to +0.0070) but not on 13 (+0.0022). Scope under the 2026-09-17 exclusion: Omri's decision.

## Predictions (frozen)
P1 held (13: EQ_DSM = B2 0.7818 exactly; +0.0016, significant at 95% though predicted not); P2 held (DSM above EQ between 8/8,
HEM 6/8); P3 held (candidate = plain average); P4 mostly (5/6 pool banks); P5 failed on 20 (hem prevalence not closer); P6 held;
P7 partly (level group most influential by estimate shift 40/40, by prevalence not on 32+d/51; toward the truth 5/8 banks).

## For the new benchmarks
Frozen candidate: DS filter (pi_hat > 0.5) then plain average of the surviving channels, on a bank chosen before the test (with
digits if Omri keeps them). Optional, clearly labelled as bank-dependent: hem within + DS-estimate between weights.
