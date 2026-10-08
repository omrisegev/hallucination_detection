# LR zero-weight analysis (PRMBench, 10 nested banks) — summary

Population: PRMBench, 6,969 answers, 75,038 steps, same 5 source-group folds as monday_lr_reference_v1.
Replay: N16 out-of-fold within-answer AUC 0.8223662 (target 0.822366); all 10 banks' cached fold models replay identically.

## Near-zero rule
The pre-registered rule (|w| < 5% of max |w|) fails on the digit banks: the max is the collinear digit pair (|w| ≈ 9–12),
so it marks 12–47 channels, including energy_level and realized_drv. Second rule: 5% of the largest |w| outside collinear contrast pairs.

## Findings
1. Near-zero channels: 1–2 in the 11–25 channel banks, 6–10 in the 40–56 channel banks. Mostly rhythm (F5), plus true_tail50 (F1)
   and top15_turnover (F3); chosen_surprisal (F2) only in the large banks. No digit channel. 17/44 near-zero entries change sign across folds.
   N16: true_tail50 (−0.020), top15_turnover (+0.017).
2. Dropping them costs nothing: every bank's change is within ±0.0004 AUC, all paired 95% CIs (1,000 source-group draws) include 0.
   Dropping the bottom quartile: only N11 loses (−0.0012). The pre-registered 5%-of-max rule loses 0.005–0.100 (N14 keeps only the digit pair).
3. The zero set is not unique. L1 paths keep energy_level + realized_drv first (AUC .8028); N56: ~11 channels reach .8274, 20 reach .8308, full .8355.
   Exact copies (q15_H1 = hist_entropy_series, q15_VE1 = ct7_ve1, chosen_surprisal = hist_spilled_series) split weight; removing one moves it to the other.
   Weight depends on the bank: chosen_surprisal is top-5 in N11/N13/N14 but near zero once realized_drv and copies are present.
4. Digit pair (corr 0.9998): N16 +9.30 / −9.38, sum −0.08 (fold SD 0.02), each alone fold SD 0.55; removing either collapses the other to ≈ −0.13
   at AUC cost 0.001–0.002. Large opposite weights are a collinearity effect, not two signals. Same pattern without digits on N53/N56 (hist_topk_entropy vs q15_H1).
5. Stable signal: energy_level + realized_drv (plus digit_spread in digit banks).

## Deviations
L1 solver liblinear on C = 1e-5..0.1 (saga too slow), recorded in PROTOCOL.json before results. N53/N56 fixed-C refits differ by up to 1.4 in weight at equal AUC
(undetermined collinear directions). Three bank jobs reran alone after out-of-memory, unchanged code.

Files: PROTOCOL.json, run.py, GATE.json, ZERO_WEIGHTS.csv, L1_PATH.csv, PAIRS.csv, RESTRICTED.csv, AGGREGATE_CHECKS.json,
figures/W1_weights_N16_N56.png (.svg). Summary text written by the main session from the analysis agent's report.
