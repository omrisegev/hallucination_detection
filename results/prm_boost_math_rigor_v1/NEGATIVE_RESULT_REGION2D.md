# Negative result — 2-D clean region in (S, number of steps), calibrated track (prm-boost-clean-region-2d-v1)

**Hypothesis (one sentence):** making the certified clean threshold depend on answer length (one τ per length
tercile) lets more PRMs gain over GAP on all PRMBench answers, and lets the rule transfer to Hard2Verify.

**Date / step:** 2026-10-25, Step 473 [PRM math rigor]. Pre-registration committed first: `f57b4931`.

**Benchmark / population:** the same as Step 471.
- PRMBench, all 6,969 answers: 50/50 source-group splits, 10 seeds, seed-0 paired bootstrap.
- Hard2Verify, 177 answers: transfer of the seed-0 rule.
- Access: calibrated.

## Arms compared
All numbers are from `stage6/REGION2D_RESULTS.json`, seed 0, α = 0.05, PRMBench test answers. "Gain" is the
difference from GAP.

| arm | what it is | PRMB gain, all answers | PB macro-F1 | vs reference |
|---|---|---|---|---|
| reference (incumbent) | GAP on CHAIN_N16 | 0 | not run | — |
| simple control | 1-D τ on S (Step 471) | Qwen-7B +0.43, 72B +0.52, others −0.35 to −0.06 | not run | partial |
| candidate (primary) | per-length-tercile τ on S | Qwen-7B +0.50 [+0.31, +0.70], 72B +0.48 [+0.32, +0.66], PURE −0.12, RLHFlow −0.13, MS +0.00 | not run | **partial** |
| secondary | 1-D τ on S / n | Qwen-7B +0.56, 72B +0.64, PURE +0.22 [+0.02, +0.44], RLHFlow −0.26, MS −0.10 | not run | not the verdict |

**Win/loss record:**
- **Guarantee:** 5/5 PRMs (≥ 9/10 seeds each).
- **All-answer gain with CI > 0:** 2/5 (the Qwen PRMs). 3 were needed.
- **Official-pool non-inferiority:** 5/5, up from 4/5 for the 1-D rule. The range is −0.46 to +0.13.
- **Hard2Verify transfer:** about 0 everywhere. Clean-recall is 0 to .02.
- **Verdict: partial** (`stage6/REGION2D_VERDICT.json`).

The secondary variant would have met the pass thresholds: 3/5 gains, with PURE +0.22 [+0.02, +0.44]; non-inferior
4/5. It was declared secondary, so this is not a pass. It is a lead for a new pre-registration, on data the project
has already seen.

## Three plausible reasons it failed
1. **Length was not the main limit on PRMBench.** Clean-recall of the weak PRMs barely moved (PURE .158, RLHFlow
   .143, MS .099), so their PRM sum separates clean answers too weakly whatever the boundary's shape. The break-even
   was about .3.
2. **The transfer failure is a level shift between benchmarks, not a length effect.** Even per-step S/n calls
   essentially nothing clean on Hard2Verify. The PRMs' probabilities are higher on Hard2Verify answers, so a
   PRMBench τ sits below them. That calls for calibration on the target benchmark, which needs labels there.
3. **Stratifying costs calibration data.** Each tercile certifies on a third of the erroneous answers, which
   widens the Hoeffding–Bentkus margin. The realized false-clean rate fell to .025–.035, below the .05 budget,
   leaving clean-recall unused.

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open

The length-stratified boundary adds no gain for the weak PRMs. It does make the official pool non-inferior for all
5 PRMs. A per-step normalized score looked better in a secondary row.

## What would reopen it
Either of two things:
- an answer score that separates clean answers better for the weak PRMs (raw-level telemetry, an AIRCC export);
- calibration on the target benchmark when it is large enough, for transfer.

A learned region (CoCoNet) only makes sense on top of a score with more separation.

**Source files:**
- Results: `stage6/REGION2D_RESULTS.json`, `stage6/REGION2D_VERDICT.json`, `CLEAN_REGION_2D_PREREG.json`.
- Code: `stage6_clean_region_2d.py`.
- Command: `COLLECTION_DATA=/Users/osegev/Desktop/hd_collection_data python stage6_clean_region_2d.py stage6`.
- n_checked / n_total: 2 variants × 5 / 5 PRMs × 10 seeds, plus 5 / 5 Hard2Verify transfers.
