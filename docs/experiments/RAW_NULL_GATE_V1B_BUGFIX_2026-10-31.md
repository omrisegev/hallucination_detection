# raw_null_gate_v1b: bug-fix release of the raw-null gate (2026-10-31)

**Base:** `docs/experiments/RAW_NULL_GATE_PREREG_2026-10-30.md` (v1). Everything in v1 stays the same, except the
items below. v1 code, outputs and hashes stay frozen in `results/raw_null_gate_v1/`.

**Source of the items:** the two post-run reviews `results/raw_null_gate_v1/REVIEW_A_PC_2026-10-31.md` and
`REVIEW_B_CLOUD_2026-10-31.md`. Omri asked (2026-10-31) to fix the bugs that have a clear, known fix, prove
that the fixes work, and only then plan a diagnosis of the design. v1b is a bug-fix release. It is not a new method,
and nothing in it is tuned on labels.

**What v1b does NOT change** (these are design limits and go to the diagnosis plan): the zero assumption, the
choice of the worst step, the cuts .5 and .05, the channel groups, discrete channels, G1b.

## Fixes

| # | v1 behaviour | v1b behaviour | Why this is a bug |
|---|---|---|---|
| F1 | G2: LR_g = (π₀/(1 − π₀))(1/lfdr_g − 1) with the **shared median** π₀ | LR_g = (π₀_g/(1 − π₀_g))(1/lfdr_g − 1) with the **group's own** π₀_g. The prior odds term keeps the shared π₀ | f1_g/f0_g = (π₀_g/(1 − π₀_g))(1 − lfdr_g)/lfdr_g follows from lfdr_g = π₀_g f0_g/f_g. With the shared π₀, a single group does not give back its own 1 − lfdr |
| F2 | 3.3 accepts every fit with b₂ < 0, also with π₀_raw > 1 | **Dropped from v1b, unchanged from v1.** The fit only records sd(x) as a diagnostic | Not a clear fix. On pure N(0, 1) data with no errors, central matching gives π₀_raw ≈ 1.02 (99th pct 1.03–1.07) and σ₀/sd ≈ 1.03 (max 1.04–1.11), n = 9k–145k, 60 runs each. A strict "π₀_raw > 1" or "σ₀ ≥ sd" rule would reject every error-free fit. Identification goes to the diagnosis plan |
| F3 | P(err \| z) = 1 − lfdr(z) above μ₀, which can fall again in the right tail | Above μ₀, P(err) is made non-decreasing in z: sort the scored values above μ₀ by z and take the running maximum of 1 − lfdr (ties get the same value). G2 uses lfdr_mono = 1 − P_mono in LR_g | 3.3 step 5 is one-sided because a higher risk score must not be read as more correct. A falling right tail contradicts that |
| F4 | G3 needs ≥ 3 groups with an identified null | G3 needs ≥ 3 groups whose answer votes are **not constant** (at least one +1 and one −1). Constant voters are dropped before DS-EM | A constant voter carries no information. DS-EM with constant voters can flip labels (v1: all groups vote "clean", prevalence .95) |
| F5 | PRMBench no-harm counts every cell | It counts only cells where the gate is valid (`ok`) | An invalid gate closes nothing, so its metric equals GAP_ONLY and passes trivially |
| F6 | ECE ranks P(err) as is | ECE ranks round(P(err), 9). Ties are broken by index (stable), as before | Float differences of 1e-13 between two correct implementations moved steps across bin edges |
| F7 | The verifier does not compare P(err) arrays or VERDICT.json | The v1b verifier also compares p_err_step (G1, G2), p_err_worst (G1, G2), p_answer (G1b, G3) to 1e-6, and recomputes VERDICT.json | A shared bug in these arrays would go undetected |

Every other number (bins 120, degree 7, quantiles, clips, cuts, Šidák, DS-EM init and anchor, the pass rule) is
unchanged from v1.

**F3 is reported separately.** F1 and F4–F7 are bug fixes. F3 is a correction from known practice (a monotone
lfdr), not a pure bug fix, so its effect is shown apart:
- variant `fix` = F1, F4, F5, F6, F7 (the primary v1b);
- variant `fix_mono` = the same plus F3.

Both variants run on the same cells. Outputs go to `results/raw_null_gate_v1b/<variant>/`.

**Output contract (both variants).** RESULTS.csv, FITS.json, VERDICT.json and CHECK.json keep the v1 schema. Each
`labelfree/<arm>_<mode>_<bench>_<prm>.npz` holds z_fused, raw_fused, f_gap, clean_<G> for G in G1, G1b, G2, G3, and
p_err_step_G1, p_err_step_G2, p_err_worst_G1, p_err_worst_G2, p_answer_G1b, p_answer_G3 (all NaN when that gate is
invalid). `labelfree/` stays on the PC (`.gitignore` excludes `*.npz` under results).

## Proof that the fixes work (required before any interpretation)

1. **Unit tests**, one or more per fix (`tests/test_raw_null_gate_v1b.py`):
   - F1: with one group, G2 P(err) = P_mono of that group to 1e-9.
   - F2 (characterization only): a pure N(0, 1) score gives π₀_raw > 1 and is accepted, in v1 and v1b alike.
   - F3: P(err) is non-decreasing in z above μ₀.
   - F4: constant voters are dropped; with < 3 varying voters G3 is invalid.
   - F5: an invalid cell never counts as no-harm.
   - F6: ECE is unchanged under a 1e-12 perturbation of P(err).
2. **Real-data rerun** of v1b on the PC (selftest, check, run, verify), with the new independent verifier.
3. **Fix audit on real data** (`FIX_AUDIT.json`), label-free: counts of fits with π₀_raw > 1.08 (above the
   pure-Gaussian 99th percentile at the smallest n), reported only; count of non-monotone P(err) pairs above μ₀ (must be 0); G3 cells using a constant voter (must be 0);
   the v1 to v1b change in each of these counts.

The outcome of the pass rule in v1b is reported as it comes. A pass in v1b would still be development evidence on
already inspected benchmarks, not confirmation.
