# Red team: position_channel_v1 run_20260929 (protocol e15764273 + amendment A1 at 106ec28ca; run a6cc87f78)

- **A** recomputation with a fresh script that does not import the runner: within-answer AUC from the raw labels (6,030 eligible
  answers), 5,000-draw source-group bootstrap (707 groups, own seed), POS rebuilt from the offsets, algebra identity per fold.
- **B** coverage: all 30 arms x 5 folds x 145,597 rows; 40 fold cells, 24 length-tercile cells, 64 error-type cells, early- vs
  late-error answers, ProcessBench per cell (32 per variant), pooled ProcessBench win/loss, concentration.
- **C** nulls and math: within-answer shuffle and same-length swap (200 permutations), source-clustered z; Dawid-Skene re-fits on all
  5 folds with POS, POS flipped (first steps marked) and random mark channels (10 seeds on fold 0, one per fold); permuted and flipped
  POS inside the score; the weight algebra; a post-hoc label-using dose check (descriptive only); the orientation diagnostic per benchmark.

Scripts in the session scratchpad (`redteam_A.py`, `redteam_B1-3.py`, `redteam_C_ds.py`, `redteam_C_scores.py`, `redteam_C_spear.py`).

| CLAIM | VERDICT | EVIDENCE |
|---|---|---|
| 1. Adding POS raises PRMBench within-answer AUC on all 4 digit banks, Bonferroni: BASE_POS-BASE +0.0084 / +0.0059 / +0.0090 / +0.0042, GRP_POS-GRP +0.0081 / +0.0048 / +0.0074 / +0.0073 (B16/B23/B35/B54) | **numbers confirmed; reading: a positional prior that helps late errors and hurts early ones** | **A**: all 16 arm values exact to 6 decimals; own bootstrap Bonferroni lower bounds within 0.0001 of the claim, 8/8 above 0. **B**: above 0 in 40/40 fold cells and 24/24 length-tercile cells. **C**: within-shuffle z 9-13, source-clustered z 10.9-16.5; order-permuted POS -0.0003 to -0.0010; flipped POS -0.0043 to -0.0100; same-length swap null above the observed gain on 8/8 (no content beyond position, by construction, amendment A1). **B**: late-error answers (4,267) +0.0070 to +0.0178; early-error answers (1,294) lose on 8/8 (-0.0042 to -0.0181; B35 GRP_POS -0.0264); `domain_inconsistency` loses on 8/8 contrasts, `circular` on 4/8 |
| 2. Every position variant is above the step index alone on all 4 banks (A1 clause b) | **confirmed** | **A**: POS_ALONE 0.661731 recomputed; runner 95% intervals +0.095 to +0.147, all above 0 (8/8). **A/C**: POS rebuilt from offsets equals the stored array exactly |
| 3. PRMScore rises on 8/8; the ProcessBench cost of BASE_POS is small (B16 -0.0087, B23 +0.0002, B35 -0.0003, B54 -0.0031), larger for GRP_POS (-0.0058 to -0.0125) | **weakened** | **B**: PRMScore recomputed from the metadata `error_steps`, 8/8 up (+0.0016 to +0.0038), 37/40 folds (the 3 declines are GRP_POS). ProcessBench macro-8 exact, but pooled over 4,442 answers all 8 variants lose (-0.0020 to -0.0149; answers lost > answers won on 8/8, e.g. 106 vs 60); BASE_POS loses 20/32 cells, GRP_POS 25/32; B23's +0.0002 is a macro artefact (pooled -0.0020). Every moved argmax moved later (316/316 on B16 BASE_POS); earliest third of first errors -0.039 (B16 BASE_POS) / -0.059 (B16 GRP_POS) |
| 4. The DS filter keeps POS in 20/20 bank-folds (pi_hat 0.554-0.583): label-free support for the channel | **"kept" is not evidence; the direction is** | **C**: random mark channels are also kept (pi_hat 0.5004-0.5056; kept 59/60), so pi_hat > 0.5 admits noise. But POS flipped to the first steps is dropped 20/20 (pi_hat 0.426-0.473) while POS is kept 20/20 (0.554-0.583); fitted together 0.555-0.586 vs 0.423-0.473; label-using truth on fold-0 fit steps POS 0.557, flipped 0.403, random 0.504. The DS latent class has prevalence 0.28-0.41 vs a true error rate 0.141: it is not the error class |
| 5. The logged label-free orientation diagnostic (Spearman POS vs BASE on fit rows) is +0.03 / -0.03 / -0.07 / -0.12, i.e. contradicts the prior on 3 of 4 banks | **refuted (design error in the diagnostic)** | **C**, verified independently by the author (same values): the fit rows include ProcessBench steps (35% of fold-0 fit steps) while DS and the evaluation use PRMBench only. PRMBench fit rows +0.154 / +0.090 / +0.036 / +0.007 (4/4 positive); ProcessBench fit rows -0.20 / -0.25 / -0.25 / -0.34. Agents A and B had repeated the pooled figure as "the label-free sign contradicts the prior"; that reading is withdrawn |
| 6. Protocol principle: "the label-free machinery decides whether and how much position counts" | **refuted for BASE_POS, weakened for GRP_POS** | **A/C**: BASE_POS = ((p-1) BASE + POS)/p on 20/20 bank-folds (error <= 8.9e-16), so the position weight relative to BASE is 1/(p-1) = 0.071 / 0.048 / 0.033 / 0.022, set by bank size; pi_hat only gates, and nothing in the pipeline can flip the sign. **C** post-hoc label-using dose (descriptive, not a selection): best c about 0.24-0.29 (within-AUC 0.7913-0.8096); BASE_POS realises 47% / 27% / 27% / 15% of that positional gain. GRP_POS weight share 0.042-0.092 comes from the DS estimates but is similarly small (4-13x below the post-hoc best) and is estimated against the inflated latent class |
| 7. The gain is not carried by a few answers (top-1% share 23-42%, trimmed mean positive 8/8) | **confirmed, with the shape** | **B/C**: median answer delta exactly 0 on 8/8; 58-81% of answers unchanged; improvers : decliners about 2 : 1 (B16 BASE_POS 1,349 vs 686) |
| 8. The running-mean channel (CUM) does not help (BASE_CUM -0.0018 to +0.0018; GRP_CUM lowers PRMScore on 4/4) | **runner only, not red-teamed** | CONTRASTS.csv |

## Decision under the frozen rule (amendment A1)
Both BASE_POS and GRP_POS meet (a) Bonferroni above the method without position and (b) above the step index alone, on 4/4 banks, so
the rule adopts them as position variants. The red-team reading qualifies what was adopted: a positional prior (later = riskier)
whose direction the DS fit does separate without labels (claim 4), whose weight is not estimated in the plain average (claim 6), and
which trades early errors and first-error localization (ProcessBench) for late errors.

## What the author's reading missed
- The orientation diagnostic was computed on rows that mix the two benchmarks; on the benchmark the method is fitted on, it agrees
  with the prior. The runner should have computed it on the PRMBench fit rows that DS uses.
- A keep decision (pi_hat > 0.5) needs a random-channel floor and a flipped control before it is read as support.
- The protocol's principle sentence overstated what the machinery decides: in a plain average an added channel's weight is 1/(p-1),
  a function of bank size.
- "Small ProcessBench cost" hid pooled losses on 8/8 and cell losses on 20/32.
