# Negative result — two-step flag count (prm-boost-two-step-count-v1)

**Hypothesis (one sentence):** calling an answer clean when the PRM's native rule flags none of its steps, and
otherwise flagging with GAP on CHAIN_N16, improves step-level decisions on sets with clean answers without hurting
the official PRMBench pool.

**Date / step:** 2026-10-24, Step 470 [PRM math rigor]. Pre-registration committed first: `88b05f0b`,
`TWO_STEP_PREREG.json`.

**Benchmark / population:**
- PRMBench, all 6,969 answers (934 clean), and the official 6,211-answer pool. Labels: `prmbench_meta.json`.
- Hard2Verify, 177-answer common set (41 clean).
- Paired bootstrap, 2,000 draws: by source group on PRMBench, by answer on Hard2Verify.

## Arms compared
All numbers are from `stage3/TWO_STEP_RESULTS.csv`, ranking CHAIN_N16, ranges over the 5 PRMs. ProcessBench was
not part of the pre-registration.

| arm | what it is | PRMB official PRMScore | PB macro-F1 | vs reference |
|---|---|---|---|---|
| reference (incumbent) | GAP on CHAIN_N16 | 66.25–71.09 | not run | — |
| simple control | the same two-step rule on the PRM's own probability | 55.20–68.15 | not run | also loses to its own GAP: −0.86 to −3.89 |
| candidate | native "clean" decision, then GAP on CHAIN_N16 | 60.66–68.59 | not run | −2.50 to −5.60, every CI below 0 |

**Win/loss record:** candidate vs reference, 0–5 on the PRMBench official pool and 0–5 on all PRMBench answers
including controls (−0.76 to −4.35, every CI below 0).
- On Hard2Verify the point estimate is below 0 for 5/5 PRMs: −0.12 to −33.1. RLHFlow's CI [−0.54, +0.12]
  includes 0; the other four exclude it.
- Paired CIs on the primary endpoints include 0: no.
- Verdict: **fail** (`stage3/TWO_STEP_VERDICT.json`).

## Three plausible reasons it failed
1. **The native rule calls many erroneous answers clean** (`stage3/TWO_STEP_RESULTS.csv`, `erroneous_called_clean`).
   - PRMBench: 23–39%.
   - Hard2Verify: 2–83%; Math-Shepherd 83%, Qwen-7B 59%.
   - Each such answer loses every correct flag GAP would have made. Stage 1c had measured only the clean-answer
     side (.41–.83 called clean on PRMBench). That is the quantity this pre-registration disclosed as unmeasured.
2. **Clean answers are a minority, so the asymmetry is unfavourable.** They are 934 / 6,969 on PRMBench and
   41 / 177 on Hard2Verify. Recovering their false flags cannot pay for the lost true flags in the larger
   erroneous set.
3. **The PRMs' native thresholds are miscalibrated in the "everything is fine" direction.** This is PRISM's
   "overcredit". It matches `stage1c/PRM_COUNT.csv`: on Socratic and Hard2Verify the Qwen PRMs' expected count is
   1–2 errors below the truth.

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open

What is closed: the PRM's native threshold as the answer-level decision. What stays open: an answer-level
"clean" decision with a better-calibrated score. The clean-vs-erroneous AUC of the PRM sum is .82–.84 for the Qwen
PRMs on PRMBench (`stage1c/PRM_COUNT.csv`), so the information exists. What this result shows is that the native
threshold sits in the wrong place.

## What would reopen it
An answer-level score and threshold whose rate of calling erroneous answers clean is low, for example below 5%.
Two possible sources:
- raw-level telemetry (not answer-z standardized) exported from AIRCC;
- a threshold set on a labeled calibration split with a guarantee on that error rate (Laufer-Goldshtein risk
  control). That changes the access level.

**Source files:**
- Results: `stage3/TWO_STEP_RESULTS.csv`, `stage3/TWO_STEP_VERDICT.json`, `TWO_STEP_PREREG.json`,
  `stage1c/PRM_COUNT.csv`.
- Command: `COLLECTION_DATA=/Users/osegev/Desktop/hd_collection_data python stage3_two_step_count.py stage3`.
- n_checked / n_total: 10 / 10 cells (5 PRMBench, 5 Hard2Verify), all answers in each population.
