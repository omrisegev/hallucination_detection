# raw_null_gate_v1b: bug-fix release, run record (2026-10-31)

**Spec:** `docs/experiments/RAW_NULL_GATE_V1B_BUGFIX_2026-10-31.md`.

**Inputs:** the bugs come from the two v1 reviews (`results/raw_null_gate_v1/REVIEW_*`). The v1 code and results are
unchanged.

**Environment:** the Windows PC, Python 3.13 / numpy 2.2.4 / scipy 1.17.1 / sklearn 1.8.0. Data:
`C:\Users\omris\TAU\hd_collection_data`.

**Code:**
- `spectral_utils/raw_null_gate_v1b.py`;
- `run_raw_null_gate_v1b.py`;
- `verify_raw_null_gate_v1b.py`. A separate agent wrote it from the specs only, without reading the module or the
  runner.

The hashes are in `PROTOCOL.json`.

## What was fixed, and what was not

| # | Item | Status |
|---|---|---|
| F1 | G2 LR uses each group's own π₀ (v1: the shared median) | Fixed (both variants) |
| F2 | Reject a null with π₀_raw > 1 or σ₀ ≥ sd | **Dropped.** See below |
| F3 | Monotone P(err) above μ₀ | Known-practice correction, variant `fix_mono` only |
| F4 | G3 drops constant voters; needs ≥ 3 varying voters | Fixed (both variants) |
| F5 | PRMBench no-harm counts only valid gates | Fixed (both variants) |
| F6 | ECE ranks round(P(err), 9) | Fixed (both variants) |
| F7 | The verifier compares the P(err)/p_answer arrays and VERDICT.json | Fixed |
| - | v1 README: swapped runner/verifier columns; false claim that `labelfree/` is committed | Corrected in `results/raw_null_gate_v1/README.md` |

**Why F2 was dropped.** On pure N(0, 1) data with no errors (60 runs per n, n = 9k-145k), central matching gives:
- π₀_raw with median about 1.02, a 99th percentile of 1.03-1.07, and a maximum of 1.08;
- σ₀/sd of about 1.03, with a maximum of 1.04-1.11.

So the strict rule would reject every error-free fit. A threshold for "not identified" is a calibration choice, not
a clear fix. It goes to the diagnosis plan (`docs/experiments/RAW_NULL_GATE_DIAGNOSIS_PLAN_2026-10-31.md`, D2).

## Proof that the fixes act

**Unit tests:** `tests/test_raw_null_gate_v1b.py`, 9 tests, all pass. Each test shows v1 next to v1b on the same
input:
- F1: one group gives back its own 1 - lfdr, and data with no errors is over-called less.
- F3: P(err) is non-decreasing above μ₀.
- F4: a constant voter is dropped.
- F5: an invalid cell counts as 4/4 no-harm in v1 and 0/4 in v1b.
- F6: under 1e-13 noise, v1's ECE changes and v1b's does not.
- F2: a characterization test, showing why the rule was dropped.

**Synthetic verifier:** `fix` passes 1,868 comparisons and `fix_mono` 1,895, with 0 failed.

**Real data, label-free audit (`FIX_AUDIT.json`):**

| Count | v1 | fix | fix_mono |
|---|---|---|---|
| G3 valid cells | 55 | 36 | 34 |
| G3 valid cells that use a constant voter | 36 | 0 | 0 |
| Constant voters dropped | 0 | 61 | 64 |
| G1 decreasing P(err) pairs above μ₀ | (not saved in v1) | 206,047 | 0 |
| Steps whose P(err) F3 changes: G1 / G2 | - | - | 494,803 / 1,101,157 |
| SUB5 sa G3 PRMBench no-harm | 3/4 (all 3 invalid) | 0/4 | 0/4 |
| G1 fits ok with π₀_raw > 1.08 (reported only) | 11 / 68 | 11 / 68 | 11 / 68 |
| Group fits ok with π₀_raw > 1.08 (reported only) | 111 / 244 | 111 / 244 | 111 / 244 |

## Run

**Both variants:** check `CHECK PASSED 68 ok, 12 missing`, then `RUN DONE`.

**Verify:** `REAL FAIL`, 3 of 8,393 (`fix`) and 3 of 8,383 (`fix_mono`).
- **Which comparisons:** all 6 are `npz:p_answer_G1b` on TELE16 standalone PRMBench (pure_prm, qwen_prm_72b,
  rlhflow_prm). The maximum difference is 4.5e-5 to 8.0e-5.
- **Not a fix:** G1b is unchanged from v1, and the v1 verifier never compared this array.
- **Likely cause, ties:** these 3 cells are exactly the ones in which GAP-flagged steps have values exactly equal to
  values in G1b's reference set: 152-296 such steps, apparently duplicated steps across PRMBench variants. A ProcessBench
  control cell has 0 such steps and matches. With exact ties, float-level differences flip the count of reference
  values ≥ the worst step by a few, and that moves p by up to about 8e-5.
- **Decisions match:** `clean_G1b` matches exactly in all cells, and the recomputed VERDICT matches. This is the same
  tie mechanism as the v1 ece_step mismatches. It has not been fixed (the code is frozen); the known fix is to round
  before comparing, as in F6.

## Result

**Pass rule:** 0 of 20 entries pass, in both `fix` and `fix_mono` (as in v1).

**Medians over the valid cells,** closed_clean / kept_error:

| Gate | Benchmark | v1 | fix | fix_mono |
|---|---|---|---|---|
| G2 | ProcessBench | .33 / .68 (20) | .18 / .81 (20) | .15 / .94 (20) |
| G2 | CRV arith | .36 / .94 (20) | .24 / .95 (20) | .18 / .98 (20) |
| G2 | CRV GSM8K | .03 / .97 (20) | .03 / .96 (20) | .00 / 1.00 (20) |
| G2 | PRMBench | .61 / .72 (8) | .41 / .80 (8) | .04 / .95 (8) |
| G3 | ProcessBench | .93 / .26 (16) | .98 / .13 (6) | .98 / .13 (6) |
| G3 | CRV arith | .43 / .91 (18) | .43 / .92 (16) | .43 / .92 (16) |
| G3 | CRV GSM8K | .94 / .45 (16) | .81 / .56 (10) | .90 / .48 (8) |
| G3 | PRMBench | .59 / .63 (5) | .78 / .39 (4) | .78 / .39 (4) |

G1 and G1b are unchanged by `fix`. F3 changes G1's P(err) on 494,803 steps, but only 51 of 180,132 G1 answer
decisions, all in one cell. The per-cell results are in RESULTS.csv.

**Reading:**
- **The bugs did not cause the failure.** Fixing them moves each gate's operating point but does not make any entry
  pass.
- **Remaining causes:** the remaining causes are design assumptions. The next step is the diagnosis plan
  (`docs/experiments/RAW_NULL_GATE_DIAGNOSIS_PLAN_2026-10-31.md`), which is not started and needs Omri's OK.

## Files

- **Per variant** (`fix/`, `fix_mono/`): CHECK.json, RESULTS.csv, FITS.json, VERDICT.json, VERIFY.json, and the
  check, run and verify logs.
- **Top level:** FIX_AUDIT.json and audit.log.
- **Not committed:** `labelfree/*.npz`, with per-step and per-answer P(err). `.gitignore` excludes them, so they stay
  on the PC.
