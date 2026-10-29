# ds_gate_v1: hierarchical Dawid-Skene "no-error" gate on GAP votes. E1 condition check: FAILED, stopped as frozen

**Question.** Can a label-free hierarchical Dawid-Skene model on binary GAP votes give calibrated P(answer has an
error), so that a gate can mark answers clean (k = 0) before GAP? Protocol: `PROTOCOL.json`, with amendments A1–A6
made after the independent code review (`REVIEW.md`) and before E1 touched real data.

**Code checks.**
- The model recovers its parameters on data simulated from itself (`test_synthetic.py`, 2 seeds, monotone likelihood).
- The E-step matches brute-force enumeration (review).
- No label enters the fit.

**E1 result.** `E1_RESULTS.csv`, `E1_VERDICT.json`. 16 cells (4 PRMs × PRMBench, ProcessBench, CRV-Arith, CRV-GSM8K);
all valid (K = 3–7), with no likelihood drops and no degenerate fits.
- **D1 prevalence fails everywhere:** the estimated share of answers with an error is 0.88–0.996 in every cell,
  whatever the truth (PRMBench 0.87, ProcessBench 0.65, CRV-GSM8K 0.21, CRV-Arith 0.08).
- **D3 calibration fails:** ECE is 0.32–0.33 on ProcessBench and 0.77–0.86 on CRV.
- **D4 gate fails:** at the ρ̂ threshold it closes only 4–10% of clean ProcessBench answers, and it already closes
  10–12% of PRMBench answers that do have an error.
- **D5 count fails:** sum q tracks the true count no better than the GAP count. On CRV it is often negative.

**Why.** GAP votes are relative: every group flags a few steps in every answer, clean ones included. In a clean
answer the groups still agree on the same "most unusual" steps, through shared position/backbone structure and
because GAP picks extremes. The model therefore reads clean answers as erroneous: ψ ≫ φ0 is fitted, but the
coincidence it relies on is not error-specific. This repeats, in a model that was built to absorb forced votes, the
earlier finding that per-answer relative votes cannot identify whether an answer contains any error.

**What would be needed (not run; needs a new protocol and approval).** Absolute votes that keep each answer's level:
- raw, not answer-standardized, cleaned channels thresholded on pooled quantiles;
- the PRM's native flag (reward < 0.5), which is absolute.

That is Omri's raw-signal idea. E2 (decision rules and comparisons) is not run, per the frozen protocol.
