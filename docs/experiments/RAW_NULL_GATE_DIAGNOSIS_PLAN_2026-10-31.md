# Raw-null gate: diagnosis plan, one assumption at a time (proposal, 2026-10-31)

**Status:** this is a proposal. Nothing in it has run, and it needs Omri's OK before any stage starts.

**Goal:** find which assumption of the raw-null gate fails, and where. Each stage below isolates one link of the
chain:

score → null → P(err) → worst step → answer decision.

**Inputs:** the existing cells and loaders, and the v1b `labelfree/` arrays (per-step and per-answer P(err)). The
work is CPU only, uses no new inference, and changes no frozen file.

**Labels:** labels are used **only to diagnose**. Every number that uses labels is labelled as an oracle or
diagnostic. It is never a candidate method or a threshold for one. All cells are development data, already inspected.

**Bug fixes:** the bug fixes (v1b `fix`, and `fix_mono` reported apart) come first. See
`RAW_NULL_GATE_V1B_BUGFIX_2026-10-31.md`.

| Stage | Assumption tested | What we compute (per cell, per arm) | What a failure means |
|---|---|---|---|
| D1 Signal | Raw scores rank answers: answers with an error score higher | Answer-level AUC of every gate statistic (raw fused at the worst step, raw answer max, G2 worst P(err), G3 posterior, each group's answer max). Also the oracle best operating point: closed_clean at kept_error = .85 | Low AUC: the problem is the signal, not the gate. High AUC with failing gates: the problem is downstream (D2-D6) |
| D2 Null recovery ("the centre is correct steps") | Central matching recovers the true distribution of correct steps | The true null from labels: the mean, sd, skewness and kurtosis of correct-step scores, and the true share of correct steps. Compare with the fitted μ₀, σ₀ and π₀. Measure the share of error steps inside the central .25-.75 band | Fitted σ₀ ≫ true sd with many error steps in the centre: contamination (errors are not a tail). Fitted ≈ true but a non-Gaussian shape: the Gaussian-null assumption fails |
| D3 Step separation | Error steps form a right tail | Step-level AUC and standardized shift (Cohen's d) of error versus correct steps, per channel, per group score and for the fused score. Compare with the synthetic d at which v1 works (selftest d = 2.3-3.0) | Small d (for example < 1): the two-groups model cannot separate errors at step level, whatever the null |
| D4 Answer nuisance | One null fits all answers (the pooled raw score) | Split the variance of correct-step scores into between-answer and within-answer parts. Measure the correlation of an answer's mean with its length and difficulty | A large between-answer share means the pooled null is wide because answers differ, not because of errors. That points to an answer-level or hierarchical null |
| D5 Channel shape | Each group score is continuous and close to Gaussian at its centre | Per channel: the number of distinct values, the mass at the mode, and whether the null fit is ok. Repeat D2 per group | Discrete or spiky groups (bocpd_p0, dominant_freq16, turnover, digit_spread) give a meaningless null and should not enter G2/G3 as they are |
| D6 Conditional independence (G2's product) | Groups are independent given the step label | Residual correlation between group scores within correct steps and within error steps. Also the within-answer correlation | Strong conditional correlation means the product of LRs double-counts evidence. That explains G2's over-calling |
| D7 Worst-step choice | The answer-z argmax is the right step to test | Answer-level AUC of the tested value under different worst-step rules: the answer-z argmax (v1), the raw argmax, and the P(err) argmax | If the raw argmax is much better, the v1 choice is a design error |
| D8 Threshold placement | P(err) is calibrated, so a .5 cut is meaningful | Reliability curves of step and answer P(err) against the truth, with the fitted null and with an oracle null (the D2 true null plugged in) | Calibrated with the oracle null and miscalibrated with the fitted one: the fault is the null (D2). Miscalibrated even with the oracle null: the model (Gaussian null, independence) is wrong |

**Suggested order:** D1 → D2 and D3 → D4 → D5 → D6 → D7 → D8.

**Why this order:**
- **D1 first.** It tells whether any downstream fix can help at all.
- **D2 and D3 next.** They test the zero assumption, which both reviews name as the main cause.
- **D4-D6 after that.** They explain why that assumption fails, and whether a different null (per answer,
  hierarchical, or a better-shaped null) is worth building.
- **D7-D8 last.** They locate the remaining loss in the decision rule.

**Outputs:** one machine-readable table per stage (cells × arms), one short HTML report, and a "which assumption
breaks where" summary per benchmark. Only a stage that finds a fixable cause leads to a new pre-registered variant.
That variant is then compared with v1, `fix` and GAP_ONLY under the same pass rule, and its confirmation must be on data
not used in this diagnosis.
