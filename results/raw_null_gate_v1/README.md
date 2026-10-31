# raw_null_gate_v1: run record

## What ran

- Code: the frozen commit `fd9ab513`, without changes.
- Machine: the Windows PC.
- Software: Python 3.13, numpy 2.2.4, scipy 1.17.1, sklearn 1.8.0.
- Data: `C:\Users\omris\TAU\hd_collection_data`.
- Procedure: `docs/handoffs/RAW_NULL_GATE_RUNBOOK_2026-10-30.md`.

## Steps

| Step | Command | Last line |
|---|---|---|
| A | `run_raw_null_gate.py selftest` | `SELFTEST PASSED` |
| B | `run_raw_null_gate.py check` | `CHECK PASSED 68 ok, 12 missing` (the 12 expected PRMBench backbone cells; 0 FAIL) |
| C | `run_raw_null_gate.py run` | `RUN DONE` |
| D | `verify_raw_null_gate.py real` | `REAL FAIL (7471 comparisons over 68 cells, 3 failed)` |

## Verdict

All 20 entries have the status `complete` and `passes=false`. Each cell shows `n_good / n` from `VERDICT.json`.

| Entry (arm, mode, gate) | ProcessBench | CRV arith | CRV GSM8K | PRMBench no-harm | Passes |
|---|---|---|---|---|---|
| AUTO, prm, G1 | 0/4 | 2/4 | 0/4 | 0/0 | false |
| AUTO, prm, G1b | 0/4 | 0/4 | 0/4 | 0/0 | false |
| AUTO, prm, G2 | 0/4 | 2/4 | 0/4 | 0/0 | false |
| AUTO, prm, G3 | 0/4 | 3/4 | 0/4 | 0/0 | false |
| TELE16, prm, G1 | 0/4 | 2/4 | 0/4 | 0/0 | false |
| TELE16, prm, G1b | 0/4 | 0/4 | 0/4 | 0/0 | false |
| TELE16, prm, G2 | 0/4 | 2/4 | 0/4 | 0/0 | false |
| TELE16, prm, G3 | 0/4 | 3/4 | 0/4 | 0/0 | false |
| TELE16, sa, G1 | 0/4 | 1/4 | 0/4 | 0/4 | false |
| TELE16, sa, G1b | 0/4 | 0/4 | 0/4 | 0/4 | false |
| TELE16, sa, G2 | 0/4 | 0/4 | 0/4 | 2/4 | false |
| TELE16, sa, G3 | 0/4 | 2/4 | 0/4 | 0/4 | false |
| SUB5, prm, G1 | 0/4 | 1/4 | 0/4 | 0/0 | false |
| SUB5, prm, G1b | 0/4 | 0/4 | 0/4 | 0/0 | false |
| SUB5, prm, G2 | 0/4 | 1/4 | 0/4 | 0/0 | false |
| SUB5, prm, G3 | 0/4 | 1/4 | 1/4 | 0/0 | false |
| SUB5, sa, G1 | 0/4 | 0/4 | 0/4 | 0/4 | false |
| SUB5, sa, G1b | 0/4 | 0/4 | 0/4 | 0/4 | false |
| SUB5, sa, G2 | 0/4 | 2/4 | 0/4 | 1/4 | false |
| SUB5, sa, G3 | 0/4 | 2/4 | 0/4 | 3/4 | false |

## Deviation from the runbook

The runbook says: after a verify FAIL, commit only `CHECK.json` and the logs. Omri gave explicit approval on
2026-10-31 to commit the full results, so `RESULTS.csv`, `FITS.json` and `VERDICT.json` are committed. `labelfree/*.npz`
is NOT committed: `.gitignore` excludes `results/**/*.npz`, so these files stay on the PC.
An earlier commit (`aeef7a28`) has the logs and `VERIFY.json`.

## The 3 verify mismatches

All 3 mismatches are `G2.ece_step` on SUB5 standalone cells. The tolerance is 1e-6.

| Cell | Runner | Verifier | Difference |
|---|---|---|---|
| SUB5, standalone, PRMBench, qwen_prm_7b | 0.136894 | 0.139077 | 0.0022 |
| SUB5, standalone, PRMBench, rlhflow_prm | 0.129785 | 0.129713 | 0.00007 |
| SUB5, standalone, ProcessBench, rlhflow_prm | 0.086129 | 0.085404 | 0.0007 |

All other fields of these 3 cells match. `ece_step` is a secondary reading and is not part of the pass rule.
Thus the verdict does not depend on it.

**Hypothesis.** Many steps share exactly the G2 baseline P(err), where all likelihood ratios are 1. The two
independent implementations can have float-level differences. These differences change the order of tied values,
which moves steps between the equal-count rank bins. The hypothesis is not proven.

**Evidence (read-only check, 2026-10-31).** The check recomputed only the label-free G2 `p_err_step` for the 3
cells and counted the ties in the evaluated steps. It did not change files and it did not read label values.

| Cell | Evaluated steps | Distinct values | Steps with the most common value | Decile cut points inside an exact tie |
|---|---|---|---|---|
| PRMBench, qwen_prm_7b | 83,371 | 37,211 | 24,249 (29.1%) | 5 of 9 |
| PRMBench, rlhflow_prm | 83,371 | 38,778 | 25,216 (30.2%) | 5 of 9 |
| ProcessBench, rlhflow_prm | 16,548 | 9,662 | 6,026 (36.4%) | 5 of 9 |

Thus the bin membership of these steps depends on how ties are ordered. This agrees with the hypothesis. It does
not show that ties are the only cause.

## Independent reviews (2026-10-31)

Two independent read-only reviews, after the run:
- `REVIEW_A_PC_2026-10-31.md` (reviewer on the Windows PC, with access to the saved arrays).
- `REVIEW_B_CLOUD_2026-10-31.md` (reviewer without real data; output files, code, synthetic probes).

Both agree: the FAIL is real for the frozen gates, but its main cause is an empirical null that is not
identified (raw pi0 > 1 in about half of all fits) and threshold placement, not absence of signal.
Review B also found: a validity bug in the PRMBench no-harm count (`run_raw_null_gate.py:285`, does not
change the outcome); the G2 formula uses the shared median pi0 inside each group's LR; and two errors in
this README (the runner/verifier columns of the mismatch table are swapped; `labelfree/*.npz` is NOT
committed, it is excluded by `.gitignore` and stays on the PC). Both were corrected above on 2026-10-31 (Runner = RESULTS.csv, Verifier = VERIFY.json "mine").
