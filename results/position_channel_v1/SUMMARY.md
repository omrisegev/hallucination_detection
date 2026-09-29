# position_channel_v1: step position as an additional channel

Omri, 2026-09-29: find a principled way to integrate step position as an additional channel (after the external check, Step 459,
showed that the step index alone beats every method on Hard2Verify and Socratic). No report yet: results are collected first.

- Protocol `PROTOCOL.json` frozen at e15764273; amendment A1 (106ec28ca) after the fold-0 smoke (feasibility only) and the pre-run
  review, before the full run: the content-share clause of the adoption rule could not be met by any position channel by construction
  (under the same-length swap the base falls to its positional AUC, so adding position gains more under the null); replaced by "above
  the step index alone on all four banks". Run `run_20260929` (a6cc87f78): COMPLETE, BASE and GRP replay algorithm_decisions_v1 to
  8.9e-16, every row written once, 410 s. Red team `run_20260929/RED_TEAM.md`.
- Principle (declared prior): error propagation, later steps are riskier. POS = per-answer z-score of the step index. BASE_POS: one
  Dawid-Skene fit on the bank marks plus the POS marks, plain average of the survivors. GRP_POS: POS as its own group, DS-estimate MLE
  weight. CUM (running mean of the answer's own BASE score) as the data-driven alternative.

## Results (PRMBench within-answer AUC; 6,030 answers; Bonferroni-8 interval of the gain)

| Bank | without position | + POS (plain average) | gain [Bonferroni] | grouped | grouped + POS | gain [Bonferroni] |
|---|---:|---:|---|---:|---:|---|
| 16 (13+d) | 0.7918 | **0.8002** | +0.0084 [0.0066, 0.0101] | 0.7907 | 0.7988 | +0.0081 [0.0061, 0.0101] |
| 23 (20+d) | 0.7691 | 0.7751 | +0.0059 [0.0044, 0.0074] | 0.7612 | 0.7660 | +0.0048 [0.0034, 0.0062] |
| 35 (32+d) | 0.7622 | 0.7712 | +0.0090 [0.0075, 0.0104] | 0.7713 | 0.7787 | +0.0074 [0.0052, 0.0096] |
| 54 (51+d) | 0.7646 | 0.7688 | +0.0042 [0.0032, 0.0052] | 0.7628 | 0.7701 | +0.0073 [0.0057, 0.0090] |

References: step index alone 0.6617, ct7 0.7724, fam421 0.7801. PRMScore rises on 8/8 (+0.0016 to +0.0038; best B16 grouped + POS
0.6667). ProcessBench first-error accuracy (macro-8) falls: plain average + POS -0.0087 / +0.0002 / -0.0003 / -0.0031, grouped + POS
-0.0125 / -0.0088 / -0.0060 / -0.0058; pooled over answers all 8 lose. CUM: no gain (-0.0018 to +0.0018; grouped CUM lowers PRMScore
on 4/4).

## Reading
- By the frozen rule (A1) both variants are adopted: above the method without position and above position alone on 4/4 banks.
- The gain is a positional prior, not content: the same-length swap null exceeds it on 8/8 (as expected by construction), the whole
  gain comes from answers whose error is late (4,267 answers, +0.007 to +0.018), and answers with an early error lose on 8/8
  (1,294 answers). ProcessBench, which asks for the first error, loses for the same reason.
- What is label-free: the direction. The DS fit keeps POS (pi_hat 0.554-0.583) and drops POS flipped to the first steps
  (0.426-0.473) in 20/20 bank-folds; the correlation of POS with the base score on PRMBench fit rows is positive on 4/4 banks.
  ("Kept" alone is not evidence: random channels get pi_hat 0.500-0.506 and pass 59/60.)
- What is not: the weight. In the plain average the position weight is exactly 1/(p-1) of the base (0.071 / 0.048 / 0.033 / 0.022),
  a function of bank size, so the gain shrinks as the bank grows. Post hoc (labels, descriptive only) about 4-13x more weight would
  reach 0.79-0.81; the plain average realises 15-47% of that. The grouped weight comes from the DS estimates but is similarly small.
- Open: a principled, label-free estimate of HOW MUCH position should count; the protocol's external check is deferred until that
  is decided, since a variant whose weight changes would be checked twice.
