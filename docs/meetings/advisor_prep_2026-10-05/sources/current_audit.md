# Red team: per_dataset_fit_v1 run_20260929 (protocol 355db88f5 + A1 42decb9c4 + A2 55e79c4b2; run ba0ef880d)

Pre-run review (independent subagent, before the full run): GO WITH FIXES, no blocker; all fixes applied as amendment A2.

- **A** recomputation with a fresh script: every PRMBench within-AUC and ProcessBench macro-8 / pooled value, 5,000-draw
  bootstrap (707 PRMBench and 1,979 ProcessBench source groups; q4/q8 twins resampled together), a forward structural check
  of all first-error (FE) arms at all 145,597 steps, bitwise equalities.
- **B** coverage: 36/36 cell-bank fits, 64 ProcessBench cells per comparison, PRMBench folds and length terciles, the true
  first-error distribution by position (descriptive), survivor-set recovery and a sign-flip decomposition on B35, q4/q8 twins.
- **C** nulls and math: every input of the fit traced; q_t rebuilt exactly (16/16); slope shrinkage and alternative readouts
  (post hoc); start-of-answer excess in clean answers; cross-fit slopes and dose; within-answer shuffles and sign-flip nulls.

| CLAIM | VERDICT | EVIDENCE |
|---|---|---|
| 1. Fitting per model per dataset, transductively, is label-free | **confirmed, with caveats** | **C**: no label, target or fold quantity enters any fit (all helpers traced); the PRMScore threshold is a label-free quantile. Caveats: the recipe (banks, 20% marks, 10 bins, 0.5 filter, anchor) was developed on PRMBench labels, so PRMBench numbers are development evidence; smoke outcomes were seen on 4/8 ProcessBench cells (A2); all four banks carry the digit features |
| 2. Per-dataset fitting costs nothing on PRMBench | **confirmed** | **A/B**: the plain average is bitwise identical to the pooled fit (its only fitted object, the DS survivor set, comes out the same; 20/20 bank-folds), so half of primary A is a zero-width contrast. Grouped prior version: -0.0003 / +0.0002 / -0.0009 / +0.0042 (B16/B23/B35/B54; shuffle z -0.8 / 0.3 / -1.3 / 3.4). B54 +0.0042 holds in 5/5 folds and 3/3 terciles and comes from the grouped content (the per-dataset partition without ProcessBench rows: K 8 vs 7, flatter weights), not from the prior; it breaks the frozen P1 band (0.003) in the favourable direction |
| 3. On ProcessBench the per-cell DS filter drops the position channel (32/32) so the position channel no longer costs ProcessBench | **confirmed narrowly** | **A/B**: POS dropped in 32/32 ProcessBench cell-banks, kept 4/4 on PRMBench; the plain average with the channel equals the one without it in 31/32 (pb_gsm8k_q4 B54: the joint DS fit also dropped one bank channel). But the grouped position channel still costs ProcessBench in 21/32 cells, and the drop is most likely a side effect of the reversed latent class (claim 4), not a recognition that position is irrelevant. The per-cell filter barely filters on ProcessBench (keeps 23/23, 31-35/35, 48-54/54) |
| 4. The per-cell position prior on ProcessBench tracks the start of the answer and makes the prior arms collapse | **confirmed on the macro, narrowed in scope** | **B/C**: the prior peaks at the first bin in 28/32 grouped and 20/32 plain-average ProcessBench fits (rises in 0/32 and 1/32; P2 refuted), while the true first-error rate is LOWER at the start (0.119 in bin 0 vs 0.17-0.19 in bins 1-3). The start excess is an artefact: step 0 of CLEAN answers scores +0.54 to +0.91 answer-SD, as high as when step 0 is the first error. The collapse is the 17/32 grouped cell-banks where the first-bin prior exceeds 0.9: the method then predicts step 0 in a median 95% of answers (truth 7-19%), median cost -23 points; elsewhere -2.9. GSM8K does not collapse (0/8); the plain-average prior collapses only on B35 |
| 5. The first-error readout fails because the posteriors are too sharp and the prior says the first step is the error class (my mechanism) | **gap confirmed; mechanism refuted** | **A/B/C**: FE is 9.9-20.4 points below the content argmax on 8/8 bank-families (Bonferroni intervals far below 0; 64/64 cells), also without any prior (FE_x0: 7.0-11.3 below, 50-72% of calls at step 0 vs 12.2% true). Posteriors are NOT near-hard (0.7-12.4% of steps above 0.95); shrinking the slope makes FE worse 8/8; the cross-fitted (smaller) slope makes it worse on 4/4 (P3 refuted). Real mechanism: the product q_t prod(1 - q_s) with a high per-step hazard (latent prevalence 0.3-0.5; survival to step 3 only 0.02-0.24) and within-answer standardized content (every answer has high-q steps), so the earliest high-q step wins. The plain argmax is already right: no simple label-free readout beats it consistently (best alternatives within +1.0 point on 4/8) |
| 6. Per-dataset content argmax on ProcessBench is about equal to the pooled one (B35 sign flips the exception) | **weakened** | **B**: macro changes BASE +0.05 / +0.02 / -0.84 / -0.51 points, GRP -1.28 / +0.29 / +0.72 / -0.96; per-cell swings -5.9 to +2.9; the q4/q8 twins (same answers) agree on the sign of the change in only 9-10/16 pairs: noise level. B35's sign flips cost -1.31 points net, 55% from one cell; several flips rest on correlations of 0.01-0.05 and differ between twins |
| 7. Cross-fitted slopes are smaller and help the plain-average prior version on PRMBench | **numbers confirmed; not adopted** | **A/C**: smaller than the model slope in 34/36 cell-banks (median ratio 0.57; P4's "every cell" fails on two). PRMBench within-AUC vs the non-cross-fit prior version +0.0042 (0.8089, the highest so far) / -0.0068 / +0.0117 (0.7972) / +0.0014: fails on B23 (the halves' slopes 0.75 vs 3.05 re-weight the content itself). The gain is a dose effect (1.4-2.2x more position weight); it reaches the post-hoc dose optimum on B16 and B35, not B54. The same extra weight costs ProcessBench 0.8-3.4 points |

## Decisions under the frozen rules
- Per-dataset contract: adopted by Omri's decision; it costs nothing on PRMBench.
- First-error readout: not adopted for either family (8/8 Bonferroni intervals far below 0). The argmax of the per-dataset content
  score stays the first-error readout.
- Cross-fit: not adopted (fails on B23 for the plain-average prior version; worse in the first-error readout on 4/4).

## What the author's reading missed
- The FE mechanism was written from the protocol's own expectation (sharp posteriors) without testing it; the red team's slope
  shrinkage refuted it. The product readout needs a label-free check of its implied first-error distribution (share of step-0
  calls) before it is run.
- "Position dropped in 32/32" was stated as "no ProcessBench cost from position"; it holds for the plain average only (31/32),
  not for the grouped channel.
- A per-dataset latent class can latch onto a positional telemetry artefact (the start-of-answer excess present in clean
  answers); a position prior then amplifies it.
