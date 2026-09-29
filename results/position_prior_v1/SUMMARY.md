# position_prior_v1: step position as a prior inside the Dawid-Skene model

Omri, 2026-09-29: run the proposed next step after position_channel_v1 (Step 460), where the step index added as a channel was
adopted but its weight was 1/(p-1) of the base, a function of bank size. Here the Dawid-Skene latent-class model itself estimates
how much position counts. No report yet: results are collected first.

- Protocol `PROTOCOL.json` frozen at 7082b1e7d; amendment A1 (a9c448be8, after the fold-0 smoke): the one-bin check tests the
  optimum (relative log-likelihood rise <= 1e-6), not parameter slack of the existing fit; amendment A2 (90d2345b7, after the
  independent pre-run review, verdict GO): declared limit that pi_b measures where the channels agree, wording, pooled
  ProcessBench in the decision. Run `run_20260929` (0e32cb04c): COMPLETE, four replays (Step 457 and Step 460) at 8.9e-16, 430 s.
  Red team `run_20260929/RED_TEAM.md`.
- Model (`scripts/experiments/position_prior_ds.py`, 7 tests): the prevalence of the latent error class depends on the relative
  position bin (10 bins), P(Y=1 | bin b) = pi_b, estimated by EM without labels, no direction or shape assumed. Score = content
  S + logit(pi_b)/a, with a the latent-class slope of S (log-odds per unit of S).

## Results (PRMBench within-answer AUC, 6,030 answers)

| Bank | grouped | + position channel (Step 460) | + position prior | prior - channel [Bonferroni] | plain average + channel | + prior | prior - channel |
|---|---:|---:|---:|---|---:|---:|---|
| 16 (13+d) | 0.7907 | 0.7988 | **0.8056** | +0.0069 [0.0051, 0.0087] | 0.8002 | 0.8045 | +0.0043 |
| 23 (20+d) | 0.7612 | 0.7660 | 0.7697 | +0.0037 [0.0025, 0.0050] | 0.7751 | 0.7744 | -0.0006 (n.s.) |
| 35 (32+d) | 0.7713 | 0.7787 | 0.7853 | +0.0065 [0.0049, 0.0082] | 0.7712 | 0.7856 | +0.0144 |
| 54 (51+d) | 0.7628 | 0.7701 | 0.7765 | +0.0065 [0.0050, 0.0080] | 0.7688 | 0.7717 | +0.0030 |

References: step index alone 0.6617, ct7 0.7724, fam421 0.7801. PRMScore of the grouped prior version 0.6686 / 0.6460 / 0.6560 /
0.6525 (above the channel version on 4/4). ProcessBench first-error accuracy (macro-8) of the grouped prior version falls vs the
version without position by 2.8 / 1.0 / 1.5 / 0.0 points and vs the channel version on 3/4 banks. The order-permuted control adds
nothing (-0.0002 to +0.0004).

## Reading
- By the frozen rule the grouped prior version replaces the grouped position channel (4/4 banks, permuted control passes; on B35
  by about 2e-5). The plain average keeps the Step 460 channel (the prior version fails on B23, where its weight is only 1.14x
  the channel's).
- What the prior adds over the channel is a LARGER position weight: a straight line in the step index with the same weight does as
  well (within 0.0011 on 8/8). The model's 10-bin shape follows where the channels' marks agree, not the error rate (it keeps
  rising in the last bin, where errors fall, and misses the steep early rise).
- The label-free weight is still too small: post hoc, 2-5x more weight is best on PRMBench (in 40/40 folds). The reason is that
  the label-free latent class is a consensus of the same channels (posteriors nearly hard), so content looks 2-3.6x more
  separating than it is and position gets too little. In units of the content score's spread the best weight is nearly the same
  on every bank (0.39-0.61), the model's is not.
- The task conflict is total: ProcessBench (first error) is best with no position weight at all; the step-validity benchmark
  wants a large one. Early-error answers lose on every bank; late-error answers gain.
- Open: a label-free slope that does not share the content channels' circularity (e.g. latent class from one half of the channels,
  slope measured on the other half), and whether one position weight can serve both tasks at all. External check still deferred.
