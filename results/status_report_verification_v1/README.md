# Verification of the PRM Boosting status report (sent 21-23 Oct 2026)

Page checked: `PRM_Boosting_Status_2.html` (upload). Read-only check, done 2026-10-29: no inference, no edits to the page.
Every row is in [VERIFY.csv](VERIFY.csv).

**Verdict: no wrong numbers.** Every number in the embedded data and every prose number matches its source to the
precision shown, except one that cannot be traced to a stored output. A few wordings need a caveat before reuse,
and the newer AUTO results change two narrative points.

## Counts
- 1,446 numbers and claims checked: 1,445 PASS, 0 FAIL, 1 UNVERIFIABLE.
  - 1,341 values from the embedded JS (`PB`, `ABL` with every `ctr`/`rule` CI, `EXT`, `CAT`, `prmScale`, `n`).
    They were matched against `STATUS_MAIN.csv`, `STATUS_CONTRASTS.csv`, `ABLATION_GAP.csv`, `PRMBENCH_CATEGORIES.csv`,
    `STATUS_META.json`, and `prm_standalone_telemetry_v1/tables/STANDALONE_{MAIN,CONTRASTS}.csv`.
  - 105 prose and page-computed numbers. They were recomputed from the per-PRM `ABL5_RESULTS`, `PRM_WEIGHT_SHARE.csv`,
    `CROP_C1_MAIN.csv` and section-3 `RESULTS.csv`, plus the raw PRMBench labels (`prmbench_meta.json`) for the
    6,969 / 6,211 / 6,035 / 97% counts.
- 13 page-computed differences (category Δ and the "flag 2 / true count − GAP" rows) are off by 0.01. The page
  subtracts already-rounded values. This is rounding only and is marked PASS.
- Cross-check: `prm_sub5_all_benchmarks_v1/SUB5_ALL.csv` (2026-10-28) reproduces the PRM, TELE, MEAN and CHAIN_N16
  values on PRMBench, Socratic and Hard2Verify for the 4 PRMs it covers.

## UNVERIFIABLE
| where | page | source | note |
|---|---|---|---|
| Section "A sign rule inverted five external fits" | covariance −0.176 = −0.186 + 0.010 | Only the text of `sml_weight_estimator_v1/README.md` §1 | The cited output `COV_DECOMPOSITION.txt` is not in the repo, and the number was not recomputed (that needs the Socratic gold labels and the score npz). The arithmetic is consistent. It was computed on the pre-digit-fix Socratic `bank_v2`; the N13 arms did not change with the fix. |

## Caveats for the advisors (correct numbers, wording to qualify)
1. **CROP line ("LR keeps 25.1–25.4% vs 23.2–23.3% for L-SML")**: this is **Qwen-7B only**, at α = .05 on the C1
   subset. It compares two LRs (Telemetry13 and Telemetry16 + PRM) with CHAIN_N16 transductive and inductive. The page
   does not say Qwen-7B.
   - On the other PRMs the LR margin is larger: RLHFlow 22.1–22.4 vs 16.3–16.4.
   - Label-free LSML_CONT keeps 25.3 (Qwen-7B) and 26.2 (Qwen-72B), as much as or more than the Telemetry13 LR.
2. **"Step length alone 0.619"**: this holds for the three Qwen-tokenizer PRMs. RLHFlow is 0.616 and Math-Shepherd 0.623.
3. **"GAP adds +2.0 to +2.3 to L-SML but only +0.45 to +2.93 to the PRM"**: on RLHFlow (+2.93 vs +2.33) and
   Math-Shepherd (+2.50 vs +2.02), GAP helps the PRM *more* than it helps L-SML. "Only" holds for the three Qwen-family PRMs.
4. **"In every case the PRM got 0–21%"**: one placement (AUTO signed, PURE) has a negative share, −3%.
5. **"Standalone telemetry beats the PRM on 3 of 5"**: for PURE this is +0.0075 within-AUC, with a CI lower bound of exactly 0.000.
6. **Headline box, external sets**: the two largest Socratic losses (Math-Shepherd −16.1, Qwen-72B −14.3) are not named
   there. They are explained further down the page.
7. **"PRM alone" under Q80, flag 2 and true count**: these reuse the PRM scale chosen under GAP, not the best scale for
   each rule. The effect is at most 0.08 points on PRMBench.
8. **Sign-flip count**: `sml_weight_estimator_v1/README.md` §1 still says "3 of 20". The page's 5 of 20 (3 cells) follows
   the corrected collection README and `STATUS_META.json`.

## What newer results change
Sources: `prm_boost_math_rigor_v1/stage8/AUTO_FORCEDK_RESULTS.json` (and `AUTO_FORCEDK_VERDICT.json`), and
`bracha_lr_transfer_v1/SUMMARY.md`.

- **AUTO_N16 instead of CHAIN_N16 on PRMBench gives smaller gains.**
  - AUTO is below CHAIN for all 5 PRMs: −0.33 to −1.61 PRMScore and −0.010 to −0.032 within-AUC. The RLHFlow official
    CI is the only one that includes 0. The stage-8 verdict itself reads "CHAIN preferred".
  - AUTO − PRM (same rule, point): +1.86 / +0.68 / +2.83 / +6.53 / +4.57, against the page's +2.39 / +1.60 / +3.27 /
    +6.86 / +6.18. Within-AUC is +0.023 to +0.102, against the page's +0.037 to +0.112.
  - The Qwen-72B gain (+0.68) is the material change. No AUTO − PRM CI is stored, so "every interval above zero" is
    unchecked for AUTO.
- **The PRM-weight narrative is CHAIN-specific.**
  - AUTO gives the PRM only 0.5–7.0% of the weight on PRMBench (CHAIN: 13–24%). So "PRMBench telemetry is strong, so the
    PRM gets 13–24%" does not hold for AUTO.
  - A supervised LR trained on PRM800K gives the Qwen PRMs 3–5%. Under-weighting is therefore not specific to
    label-free covariance weighting, which weakens the page's "The reason" framing.
- **The external pattern is unchanged under AUTO.**
  - Socratic: Qwen-72B −14.29 → −11.31; Qwen-7B −3.21 → −4.62; PURE −5.59 → −7.13; RLHFlow +6.67 → +7.28;
    Math-Shepherd unchanged.
  - Hard2Verify: Qwen-72B +7.85 → +10.51; RLHFlow +27.95 → +30.38; PURE −10.79 → −12.06.
- **"Supervised LR is still better" needs its scope.** It holds for the in-domain CROP-protocol LR. An LR trained on
  PRM800K and transferred is *below the PRM alone* on PRMBench for all 4 PRMs (67.1 / 68.0 / 64.9 / 56.4 vs
  68.0 / 69.5 / 66.2 / 59.4) and below AUTO by 2.2–9.6 points (`bracha_lr_transfer_v1/SUMMARY.md`).

## Reproduce
The check scripts are not part of this folder. Every row of `VERIFY.csv` names its source file. The JS values were
taken from the page's `<script>` block and compared at the precision the page shows.
