# Pre-declared label-free count estimators (written 2026-10-21 after the parity gate, before any estimator was run)

Arms (ranking fixed, only the per-answer count k_a changes; flags = top-k_a steps by the arm's own score):
- BOOST: CHAIN_N16 on Qwen2.5-Math-PRM-7B (common.py refit, parity exact).
- TELE: thinkfix_mask1 N16 LSML_CONT and ALL_EQ_ORIENTED (bank_N16.npz, parity exact).

Reference rules (not estimators): Q80 (current: global 80th percentile of answer-z scores); ORACLE (true count);
fixed k = 1, 2, 3 (k = 2 is PRMBench's known construction prior, Steps 460 / count_structure_check_v1);
global fractions Q70 / Q75 / Q85 / Q90 (a label-free rule each; picking the best one with labels is a reference only);
PRM native threshold (reward <= 0.5, the PRM's own ranking).

Estimators (label-free in fitting AND in rule choice; nothing below is tuned after seeing results):
- E1 GAP: largest gap in the answer's sorted scores, k = argmax_{k=1..ceil(n/2)} (z_(k) - z_(k+1)); n = 1 -> k = 1.
- E2 ESD: one-sided generalized ESD outlier count (Rosner 1983), alpha = 0.05, r_max = floor(n/2),
  a test step i is run only while n - i >= 3; count may be 0.
- E3 LEVEL: count = number of the answer's steps whose GLOBAL-level score exceeds its global 80th percentile
  (TELE: bank G = mean of DS-survivor raw channels standardized once over all steps; BOOST: raw prm_risk).
  Replicates Step 459 R2 (old template) on mask1 / on the PRM.
- E4 TFRAC: global fraction = p_hat (continuous third-moment prevalence, below); flag answer-z > global (1 - p_hat) quantile.
- E5 TCONST: constant k = max(1, round(p_hat * mean n)) for every answer (assumes count independent of length).
- E6 NNR1: v_pool = rank1_offdiag_fit(pooled within-answer covariance of the oriented answer-z channels);
  per answer c_a = least-squares scale of the answer's channel correlation matrix on v_pool v_pool^T (off-diagonal only);
  p_a(1 - p_a) = clip(c_a * p_hat(1 - p_hat), 0, .25); p_a = minority root; k_a = round(n_a p_a). n_a < 4 -> k = 1.
- E7 PRMNAT (BOOST only): count = number of steps with reward <= 0.5 (the PRM's native rule), placed by the CHAIN ranking.
- E8 ANSPOIS: answer-level evidence e_a = answer mean of the E3 global-level score; answers ranked by e_a get counts
  by matching rank quantiles to Poisson(lambda = p_hat * mean n), floored at 1.

p_hat (continuous Jaffe-style third moment): channels oriented by the sign of their pooled correlation with the arm's
score; R = pooled covariance, v = rank1_offdiag_fit(R); T_ijk = mean(x_i x_j x_k) over distinct i<j<k;
gamma = sum T v_i v_j v_k / sum (v_i v_j v_k)^2 (Bernoulli skewness, (1-2p)/sqrt(p(1-p))); p_hat = (1 - gamma/sqrt(gamma^2+4))/2,
minority class taken as the error class. TELE channels: the 16 mask1 N16 channels; BOOST: 16 PRM-backbone N16 channels + z(prm_risk).
All fits pool all 6,969 answers (controls included, no metadata used).

Metrics: official pooled PRMScore on 6,211 noncontrols; count MAE; zero-vs-nonzero accuracy; Spearman(pred, true count);
per-category PRMScore for the best estimator. Labels are read only after the counts are fixed.
