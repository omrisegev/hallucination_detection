# Negative result — risk-controlled clean-answer threshold, calibrated track (prm-boost-calibrated-clean-v1)

**Hypothesis (one sentence):** with τ certified on labeled PRMBench calibration groups to keep the false-clean rate
≤ 5%, calling answers with S = Σ(1 − r) < τ clean (no flags), and flagging the rest with GAP on CHAIN_N16, improves
step decisions on all PRMBench answers without hurting the official pool.

**Date / step:** 2026-10-25, Step 471 [PRM math rigor]. Pre-registration committed first: `ca070f05`.

**Benchmark / population:**
- PRMBench, all 6,969 answers including controls.
  - Split: 50/50 by source group, 10 seeds.
  - Uncertainty: seed-0 paired bootstrap over test groups, 2,000 draws.
- Hard2Verify, 177 answers: transfer of the seed-0 τ.
- Access: **calibrated**. Labels are used on the calibration split.

## Arms compared
All numbers are from `stage4/CALIBRATED_RESULTS.json`, α = 0.05, seed 0, PRMBench test answers. "Gain" is the
difference from GAP. "False-clean" is the share of erroneous test answers called clean; "clean-recall" is the
share of truly clean test answers called clean.

| arm | what it is | PRMB gain, all answers | PB macro-F1 | vs reference |
|---|---|---|---|---|
| reference (incumbent) | GAP on CHAIN_N16 | 0 | not run | — |
| simple control | label-free native-threshold two-step (Step 470) | −0.76 to −4.35 (full data) | not run | lost everywhere |
| candidate | certified τ, then GAP | Qwen-7B +0.43 [+0.26, +0.62]; Qwen-72B +0.52 [+0.33, +0.71]; PURE −0.06; RLHFlow −0.35; MS −0.12 | not run | see the record below |

**Win/loss record:**
- **All answers incl. controls:** 2 PRMs gain with CI > 0 (the Qwen PRMs), 3 lose or tie.
- **Official non-control pool:** −0.02 (7B), −0.16 (72B), −0.45 (PURE), −0.32 (MS), −0.59 (RLHFlow). Four of five
  are within the −0.5 margin.
- **The guarantee held for 5/5 PRMs** (realized test false-clean ≤ .05 in ≥ 9/10 seeds; realized .024–.041 at
  seed 0).
- **Clean-recall:** .34 for the Qwen PRMs, .10–.16 for the others.
- **Hard2Verify transfer:** the PRMBench τ calls almost nothing clean (false-clean 0–.01), so the deltas are 0 to
  −0.14.
- **Verdict: partial** (`stage4/CALIBRATED_VERDICT.json`). Endpoint 1 holds 5/5, endpoint 2 holds 2/5 (3 needed),
  endpoint 3 holds 4/5.

## Three plausible reasons it failed
1. **The guarantee holds, but clean-recall is high enough to pay only for the Qwen PRMs.** At α = .05 they reach
   .34, the others .10–.16. The design's break-even was about .3 (`CALIBRATED_CLEAN_THRESHOLD_DESIGN.md` §3). The
   result matches that prediction: only the PRMs above it gain.
2. **Every erroneous answer called clean costs its GAP true flags.** A 3–4% false-clean rate on the large
   erroneous majority still costs the official pool up to −0.59 (RLHFlow), where clean answers are not scored at
   all.
3. **The answer score is not transferable.** S = Σ(1 − r) grows with answer length and with each PRM's
   calibration. Hard2Verify answers are longer, so the PRMBench τ falls below almost every Hard2Verify S, and the
   rule never fires there. A per-step-normalized or length-adjusted score would be a different, new variant.

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open

The mechanism works as designed: the false-clean rate is certified, and it pays where the PRM separates clean
answers well (the Qwen PRMs, +0.43 and +0.52 on all answers). It does not generalize to the weaker PRMs, nor
across benchmarks with this score.

## What would reopen it
A better-separating answer score that is comparable across answer lengths. For example, raw-level telemetry
(an AIRCC export) combined with S per step, or a length-normalized S, under a new pre-registration with
calibration on the target benchmark when it is large enough.

**Source files:**
- Results: `stage4/CALIBRATED_RESULTS.json`, `stage4/CALIBRATED_VERDICT.json`, `CALIBRATED_CLEAN_PREREG.json`.
- Code: `stage4_calibrated_clean.py`, `spectral_utils/risk_control.py`.
- Command: `COLLECTION_DATA=/Users/osegev/Desktop/hd_collection_data python stage4_calibrated_clean.py stage4`.
- n_checked / n_total: 5 / 5 PRMs × 10 seeds × 2 α on PRMBench, plus 5 / 5 Hard2Verify transfers.
