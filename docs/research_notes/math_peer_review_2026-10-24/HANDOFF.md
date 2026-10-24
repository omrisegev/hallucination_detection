# Handoff: math peer review of PRM boosting (2026-10-24)

Review: `MATH_PEER_REVIEW_HE.md` (Hebrew, English terms). Base: `claude/prm-boosting` @ 976845d5.
No jobs were run. The only computation is the Gaussian toy model in section 3.

## Tasks in priority order (all CPU; labels only for evaluation and diagnostics)

| # | Task | Fixes |
|---|---|---|
| 1 | Write the improvement condition AUC(w) = Phi(w'd / sqrt(2 w' Sigma_y w)). Build a label-free gate (d from rank-1 loadings, Sigma_y from the residual). Test if the gate predicts the sign of CHAIN - PRM on 20 cells (PRMBench, Socratic, Hard2Verify, ProcessBench x 5 PRMs). | W3, W8, W1 |
| 2 | Corrected weight u_i / (1 - u_i^2), with loadings from `_rank1_masked(..., scale='complete')`. | W4, W5 |
| 3 | Nuisance diagnostics: corr(fused latent, position/length); share of the off-diagonal covariance that remains within class. | W1, W2 |
| 4 | Bootstrap with a full refit (SML, K, partition, sign); partition stability (ARI, K variance). | W7, W11 |
| 5 | Posterior from tensor-estimated prevalence; Bayes rule / expected count; calibration (ECE) vs Q80, GAP, K2. | W10 |
| 6 | Split-half inductive fit on PRMBench (split by source group); pre-register one held-out benchmark. | W12 |
| 7 | Compare global-z, answer-z and within-answer rank; results by answer length; keep the PRM on its absolute scale. | W6 |
| 8 | Make non-negative SML with abstain-on-negative the default; document how the N16 channel orientation was chosen. | W9 |

Branch rule: cut new work from `claude/prm-boosting`, merge back only with Omri's OK.
