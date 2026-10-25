# Pre-registration: two-step flag count, "clean or not, then how many" (2026-10-24)

**This file and its JSON (`results/prm_boost_math_rigor_v1/TWO_STEP_PREREG.json`) are committed before the
evaluation runs.** ID `prm-boost-two-step-count-v1`. Requested by Omri, 2026-10-24.

## Question

GAP flags at least one step in every answer, so it can never say "this answer is clean". Does a first step that
decides clean vs erroneous at the answer level improve step-level decisions on sets with clean answers, without
hurting the official PRMBench pool?

## Rule (label-free, one variant)

1. **Step 1, the answer decision.** An answer is called clean iff the PRM flags none of its scored steps with its
   own native threshold (`native_error_flag` / `native_valid == 0`, i.e. reward ≤ .5, the authors' rule).
2. **Step 2, the count.** For answers not called clean, flag the steps chosen by GAP on CHAIN_N16, unchanged from
   the incumbent. Clean answers get no flags.

The ranking stays CHAIN_N16, so within-answer AUC is unchanged. Only the flags change.

## Why this variant, and what was seen before

- **The threshold is the PRM's own native rule.** It is not tuned: any learned threshold on S = Σ(1−r) would need
  a level that the PRMs do not calibrate (`stage1c/PRM_COUNT.csv`).
- **Disclosure.** Stage 1c (commit `0d42a6b8`) already measured the native rule's clean-answer zero rate on these
  data: .79–.83 for the Qwen PRMs on PRMBench, .44–.93 on Hard2Verify. It did not measure how often the rule calls
  an erroneous answer clean, or any step-level metric of this two-step rule.
- **Telemetry is not in v1.** The exported channels are answer-z standardized and carry no answer-level
  information, so a telemetry answer score needs raw-level channels from AIRCC. That is deferred to a v2 with its
  own pre-registration.

## Evaluation (labels only here)

| Benchmark | Population | Metric | Comparison |
|---|---|---|---|
| PRMBench, all 6,969 answers incl. controls | every step | pooled step macro-F1 ×100 (the PRMScore formula on all steps) | two-step − GAP, both on CHAIN_N16 |
| PRMBench, official pool (6,211 non-control) | official | PRMScore | two-step − GAP (non-inferiority) |
| Hard2Verify, 177-answer common set | official (empty steps as in the collection) | official metric ×100 | two-step − GAP |

- **Bootstrap.** Paired, 2,000 draws. PRMBench is resampled by source group and Hard2Verify by answer, with the
  collection's seeds (`evaluate_collection.boot_weights`).
- **Also reported:**
  - share of truly clean answers called clean;
  - share of erroneous answers called clean;
  - the same two-step rule on the PRM's own ranking (PRM prob, GAP).

## Success criteria (fixed now)

- **(a) PRMBench all-answer.** Two-step − GAP has a 95% CI above 0 in ≥ 3 of 5 PRMs.
- **(b) PRMBench official.** Point difference ≥ −0.5 PRMScore in ≥ 4 of 5 PRMs (non-inferiority).
- **(c) Hard2Verify.** Point difference > 0 in ≥ 3 of 5 PRMs. The CIs are reported, but with 177 answers they are
  expected to be wide.

**Verdict.**
- **Pass:** (a), (b) and (c) all hold.
- **Partial:** (b) holds plus exactly one of (a) and (c).
- **Fail:** anything else.

A fail or a partial is written with `/negative-result`. No threshold or rule changes after the results.

Development data, inspected many times. This is not untouched confirmation.
