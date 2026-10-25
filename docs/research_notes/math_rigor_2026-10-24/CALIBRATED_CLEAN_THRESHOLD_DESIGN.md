# Design: a risk-controlled "clean answer" threshold (calibrated track) — 2026-10-25

**Status.** This is a design only: nothing has been run and nothing is pre-registered yet. It follows the failed
label-free two-step rule (`results/prm_boost_math_rigor_v1/NEGATIVE_RESULT_TWO_STEP.md`, Step 470). It is a
**separate track with a different access level**: labels are used on a declared calibration split. Its results
are never mixed with the label-free method. This is the same access level as CROP, and it uses
Laufer-Goldshtein's risk-control tools (`ADVISOR_TOOLS.md`).

## 1. Why a guarantee, and on what

- **What went wrong.** The two-step rule failed because the PRM's native rule calls 23–39% of erroneous PRMBench
  answers clean. Each such answer loses every correct flag GAP would have given it.
- **What helps.** A clean decision helps only if it is rarely wrong in that direction, because clean answers are
  a minority: 934 of 6,969 on PRMBench, 41 of 177 on Hard2Verify.
- **The risk to control:**

      R(τ) = P( answer called clean | answer has ≥ 1 error ),   called clean ⇔ s(answer) < τ

  τ should be as large as possible (to catch more clean answers) while R(τ) stays ≤ α.

## 2. The rule

- **Answer score s.** Fixed before any calibration.
  - Primary: S = Σ(1 − r), the PRM's expected error count. In Stage 1c its clean-vs-erroneous AUC is .82–.84 for
    the Qwen PRMs on PRMBench (`stage1c/PRM_COUNT.csv`). That number was seen on the same data, so the choice is
    disclosed, not confirmed.
  - Secondary, reported, no claim: max(1 − r), the most suspicious step.
  - Telemetry enters only in a later version, once raw-level channels are exported from AIRCC. The current exports
    are answer-z standardized and carry no answer-level information.
- **Threshold.** Learn-then-Test (Angelopoulos et al. 2021) over a decreasing grid τ_1 > τ_2 > …, tested in
  **fixed sequence** from the most conservative τ.
  - For each τ, H_τ: R(τ) > α. The p-value is Hoeffding–Bentkus on the empirical false-clean rate over the
    calibration split's erroneous answers.
  - The output is the largest τ whose hypothesis and all earlier ones are rejected at level δ.
  - If none is rejected, the rule never calls an answer clean, which is plain GAP. The fallback is built in.
  - Guarantee: with probability ≥ 1 − δ over the calibration draw, R(τ̂) ≤ α. This holds for answers exchangeable
    with the calibration set.
- **Flags.**
  - Answers called clean get no flags.
  - The rest get GAP on CHAIN_N16, unchanged.
- **Constants, fixed now.**
  - α ∈ {0.02, 0.05}, both reported. 0.05 is primary.
  - δ = 0.10.
  - The τ grid: 200 quantiles of s on the calibration split.

## 3. Splits and feasibility

| Population | Calibration | Test | Can it be certified? |
|---|---|---|---|
| PRMBench, all 6,969 answers incl. controls | 50% of **source groups** (no source question on both sides), stratified by PRM-free answer length, seed fixed in the pre-registration | the other 50% | Yes. About 3,000 erroneous calibration answers give a Hoeffding margin √(ln(1/δ)/(2n)) ≈ 0.020. To certify α = .05 the empirical false-clean rate must be about ≤ .03 |
| Hard2Verify, 177 | — | all 177, using τ̂ from PRMBench | **No.** It has about 136 erroneous answers; even all of them as calibration would give a margin of about .09. This is a transfer test: exchangeability is broken, and the realized false-clean rate is reported as is (CROP's own cross-domain test reached 14.8% against a 5% target) |
| Socratic | — | optional transfer report | Only 6 clean answers, so the clean decision has almost no room to help. Reported only for the violation rate |

- **Repetitions.** The split is repeated with 10 seeds, as CROP does. Reported: the mean and spread of the
  realized test R, how often R > α, the clean-recall on the test split, and the PRMScore deltas.
- **Expected outcome, from the AUC alone (not computed).** An AUC of .82–.84 does not ensure that a low
  false-clean rate leaves useful clean-recall. The break-even (§4) can be negative even when the guarantee holds.
  A "never clean" outcome (plain GAP) is an acceptable, honest result.

## 4. Evaluation endpoints (test split only)

1. **Guarantee check.** Realized test false-clean rate ≤ α in ≥ 90% of the 10 seeds (δ = .10).
2. **Usefulness.** Pooled step macro-F1 on all PRMBench test answers including controls (PRMScore formula) minus
   plain GAP on the same answers, paired bootstrap by source group. The official non-control pool is reported for
   non-inferiority with margin −0.5.
3. **Clean-recall:** the share of truly clean test answers called clean.
4. **Transfer.** Hard2Verify official metric delta and realized false-clean rate, with τ̂ from PRMBench.

**Break-even, written in advance.** Calling a truly clean answer clean saves its GAP false flags (at least 1 per
answer). Calling an erroneous answer clean costs its GAP true flags. The rule improves step-F1 only if:

    (clean-recall × #clean × mean false flags saved)  >  (R × #erroneous × mean true flags lost)

On PRMBench #clean / #erroneous ≈ 0.18. So at α = .05 the rule needs clean-recall of about ≥ 0.3 to pay, assuming
similar flags per answer. This sets expectations; it is not a selection criterion.

## 5. Extension with more than one risk (later, separate)

Pareto Testing (Laufer-Goldshtein et al., ICLR 2023) can control two risks at once:
- the false-clean rate ≤ α;
- the per-answer F1 loss relative to GAP ≤ ε, a bounded loss on [0, 1].

It can also choose among the answer scores (S, max(1 − r), later telemetry) inside one tested family, with the
PRM-only and GAP-only configurations included as fallbacks. This is the natural place for the design-grid
correction of review weakness W12. It needs its own pre-registration.

## 6. What this track can and cannot claim

- **Can claim:** a clean-answer decision with a finite-sample guarantee on false-clean calls for answers like the
  calibration set, with GAP as the fallback, and a measured gain or loss.
- **Cannot claim:** label-free operation, a guarantee on Hard2Verify or other shifted sets, or anything about
  untouched data. PRMBench is development data.

## 7. Implementation plan (once approved)

1. `spectral_utils/risk_control.py`:
   - Hoeffding–Bentkus p-value;
   - fixed-sequence LTT over a monotone grid;
   - unit tests on synthetic data, checking that the realized violation rate is ≤ δ over many draws.
2. `results/prm_boost_math_rigor_v1/stage4_calibrated_clean.py`:
   - source-group splits;
   - τ̂ per PRM and α;
   - the endpoints above.
   - It reuses `evaluate_collection.counts_by_answer` and `official_from_counts`, and `common.py`.
3. Pre-registration `CALIBRATED_CLEAN_PREREG.md` + JSON, committed before the first run, with success criteria:
   - endpoint 1 holds for ≥ 4 of 5 PRMs;
   - endpoint 2 is positive with CI > 0 for ≥ 3 of 5;
   - the official-pool non-inferiority holds for ≥ 4 of 5.
4. Cost: CPU only, about 1 day.
