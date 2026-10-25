# Negative result — AUTO-F: AUTO's partition with block-CCA/Fisher weights (prm-boost-auto-f-v1)

**Hypothesis (one sentence):** keeping AUTO's label-free partition and replacing the L-SML eigenvector weights by
block-vs-rest CCA composites and one-factor Fisher weights improves AUTO and is not inferior to CHAIN on PRMBench.

**Date / step:** 2026-10-25, Step 476 [PRM math rigor]. Pre-registration committed first: `1dba7eb7`.

**Benchmark / population:**
- PRMBench, all 6,969 answers, 5 PRMs. Socratic (2,995) and Hard2Verify (177 common answers) are reported.
- Access: label-free fitting. The LDA/QDA diagnostic uses labels.

## Arms compared
Source: `stage9/AUTO_F_RESULTS.json`, PRMBench within-AUC / GAP PRMScore.

| arm | what it is | Qwen-7B | Qwen-72B | PURE | RLHFlow | MS |
|---|---|---|---|---|---|---|
| reference (incumbent) | CHAIN_N16 | .838 / 70.40 | .843 / 71.09 | .816 / 69.51 | .775 / 66.25 | .778 / 66.39 |
| simple control | AUTO_N16 | .824 / 69.87 | .828 / 70.18 | .799 / 69.06 | .765 / 65.92 | .746 / 64.78 |
| candidate | AUTO-F | .782 / 67.21 | .797 / 67.44 | .732 / 63.82 | .711 / 61.41 | .677 / 59.43 |
| diagnostic (labels, 5-fold CV) | LDA / QDA | .845 / .829 | .859 / .849 | .835 / .819 | .806 / .799 | .808 / .797 |

**Win/loss record:**
- **E1 (AUTO-F > AUTO): 0/5.** The within-AUC Δ is −.031 to −.069, every CI below 0.
- **E2 (non-inferior to CHAIN): 0/5.** The within-AUC Δ is −.046 to −.102, and the PRMScore Δ is −3.2 to −7.0.
- **Verdict: fail.**
- **E3 (the PRM share is flatter in K than CHAIN's): 5/5,** but trivially. The share stays near 0 (.001–.07) at every
  K, so flatness here is not a fix for W8.
- **Diagnostic.** QDA − LDA ≥ .01 in 0/5 PRMBench PRMs, and QDA is below LDA in 14/15 cells. The class-dependent
  (Ising, quadratic) direction is not supported.
- **Reported without a criterion.** Hard2Verify, where AUTO-F beats CHAIN in 4/5 (+.047 to +.065, small n).
  Socratic is mixed and unstable: PURE −.218, Qwen-72B +.050.

## Three plausible reasons it failed
1. **Without an anchor, the common factor is the nuisance, not the error.**
   - Stage 1 showed that 82–100% of the covariance between views is structure within the class.
   - Block-vs-rest CCA maximizes the correlation *across blocks*, and FA picks the factor that dominates the
     covariance. Both lock onto that shared telemetry "confidence state" (η).
   - The Fisher weights are consistent for the wrong factor.
   - CHAIN's PRM group, with its anchor, is what pins the latent to the error.
2. **The PRM's coefficient inside its composite is small.** In AUTO the PRM shares a group of 3–6 channels. The
   group's CCA direction against the rest follows the channels whose noise is shared with the rest, which is
   telemetry. The PRM's least-squares share falls to .001–.07.
3. **Equal-ish weights protect against misspecification.** This is the forecast-combination puzzle. AUTO's shrinking
   eigenvector is less harmed by the nuisance factor than a "consistent" estimator of the wrong factor.

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open

Fisher/CCA weighting without an anchor fails here. The supervised linear headroom remains: LDA is above CHAIN by
+.007 to +.031. This result reframes CHAIN's own-group PRM: it is the identifying anchor, not an independence
assumption.

## What would reopen it
A partition-free estimator that targets the factor defined by the PRM, rather than the dominant factor. One example
is CCA of the telemetry against the PRM (the PRM as one block). Its open problem is the PRM-vs-telemetry scalar,
which is not identified in one pass. A second option is a nuisance-removal step that strips η before the factor
analysis.

**Source files:**
- Results: `stage9/AUTO_F_RESULTS.json`, `stage9/AUTO_F_VERDICT.json`.
- Code: `stage9_auto_f.py`. Pre-registration: `AUTO_F_PREREG.json`.
- Command: `COLLECTION_DATA=<data> python stage9_auto_f.py stage9`.
- n_checked / n_total: 15 / 15 cells; 3 arms + 7 forced K × 3 arms + LDA/QDA per cell.
