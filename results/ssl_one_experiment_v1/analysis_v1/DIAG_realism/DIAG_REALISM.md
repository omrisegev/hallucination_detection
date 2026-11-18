# Why the synthetic errors fail the realism check (dev only, 2026-11-18)

Omri chose "Diagnose the synthetic errors" after the realism stop. This file gives the findings and a proposal.
Script: `results/ssl_one_experiment_v1/diag_realism.py` (commit 626fe59). Outputs: `analysis_v1/DIAG_realism/` (commit 8fb640d).
Data: PRMBench dev (3,499 V2 steps k, 6,548 real wrong steps, 41,045 real clean steps); ProcessBench dev answers (V1 only, because PB V2 did not run). The test side is not used.
Check: the script gives the same realism numbers as the analysis run (step .8209, answer V2 .7232, V1 .806).

## 1. Main finding

**The synthetic error is the same signal as a real error, but about 1.7 times stronger.**

- Every channel moves in the same direction for real and synthetic errors (pattern agreement .84).
- The size of the move is larger for the synthetic step. Median over the channels: 1.67 times the real-error move.
- The classifier separates them by this size, not by one special channel.

| Channel (raw top10 readout) | Real wrong − clean (clean SDs) | Synthetic − clean (clean SDs) | Ratio |
|---|---|---|---|
| q15_H1 (entropy) | +0.88 | +1.66 | 1.9 |
| chosen_surprisal | +0.87 | +1.69 | 1.9 |
| energy_level | +0.75 | +1.46 | 2.0 |
| true_tail50 | +0.88 | +1.47 | 1.7 |
| realized_drv | +0.81 | +1.46 | 1.8 |
| digit_spread | +0.78 | +1.65 | 2.1 |
| realized_z | +0.25 | +1.18 | 4.8 |
| energy_innovation | −0.29 | −0.95 | 3.3 |

Interpretation: a step from a different problem is off-topic. The model is much more surprised by it than by a real reasoning error, which is usually on-topic and subtle.

## 2. What is NOT the cause

| Check | Result | Meaning |
|---|---|---|
| Leave one channel out | AUROC stays .81-.82 for every channel | No single channel causes the gap |
| Token channels only / step channels only | .81 / .75 | Both channel families show the gap |
| Raw values instead of within-answer z | .87 | The gap is in the absolute size |
| Position, length and step count only | .61 | These explain a part, not most |
| Donor matching (length ratio, position distance, relax level) | P(synthetic) .65-.67 in every stratum | Better donor matching will not fix it |
| Answer step count only (answer level) | .50 PRMBench, .48 PB | Not a length artifact |
| Raw values of step k−1 | AUROC .500 on all channels | Correct: the prefix before k is unchanged |

## 3. Other observations

- **Spill-over:** the step after the foreign step is also disturbed (raw AUROC vs the parent's step k+1 up to .77 for true_tail50 and .75 for entropy). The model stays confused for one more step.
- **Parent category (PRMBench):** a foreign step looks least synthetic inside "circular" (.56), "domain_inconsistency" (.58) and "missing_condition" (.60) answers, and most synthetic inside "multi_solutions" (.79), "correct" (.74) and "redundancy" (.72) answers.
- **Real errors that look most like the synthetic ones:** domain_inconsistency (.40) and redundancy (.40). Least: deception (.27) and missing_condition (.29). These two are content errors with little telemetry trace.
- **Answer level:** V2 answers are about 1.4 times stronger than real wrong answers (Simes feature 1.75 times). V1 (permuted steps) is stronger still: PRMBench AUROC .81, ProcessBench .90.
- **Consequence for π̂:** BBE assumes that the known positives look like the unknown positives. BBE divides the share of pool answers above a high cut by the share of known positives above it. Positives that are too strong put too many of them above the cut, so π̂ would be too low. The realism stop protected the gate from this bias.

## 4. Proposal: synthetic errors with a realistic dose

Requirement from the data: the synthetic error must move entropy and surprisal by about 0.9 clean SDs (as real errors do), not 1.7. Donor matching does not help, so the error itself must be smaller and on-topic.

| Option | What it is | Expected dose | Known wrong? | Cost |
|---|---|---|---|---|
| A. Numeric edit (recommended) | In step k, change one number (for example the last number of the step: ±1, a digit swap, or ×10). Later steps keep the old number. Steps without a number are not used. | Small: only a few tokens change, on-topic | Yes, almost always | No generation. Text edit on CPU, then the same telemetry pass |
| B. On-topic donor | Step k comes from another answer to the same question (PB: another generator; PRMBench: a sibling version), at a similar position, with different text | Medium | Not always: the donor step can be a correct alternative | Same as A |
| C. Model-written subtle error | An LLM rewrites step k with one subtle error (as PRMBench itself was built) | Closest to real | Not always: needs a check | Extra generation, about 1-2 GPU hours |

Plan for A (and B if chosen):

1. Build the variant for the dev answers only of both benchmarks (no labels used).
2. One GPU pass on AIRCC (dev half only: about 6-8 minutes per benchmark and variant on B200).
3. Run the same realism check (C2ST ≤ .65, agreement ≥ .5). Nothing else.
4. Only if it passes: build the test-side variant, rerun the analysis on dev, then score test once.

For ProcessBench, this new variant replaces the V2 rerun: V2 failed on PRMBench and PB V1 fails at .90, so a V2 rerun would most likely fail too.
