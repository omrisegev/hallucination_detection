# RUNBOOK: raw-null clean-answer gate (for the agent on the Windows machine)

**Read this whole file before running anything. Follow it literally.** If any step fails, STOP and report the exact
error text. Do not try to fix code. Do not change thresholds. Do not skip a step.

## 0. What you are doing, in one paragraph

You will run a frozen, pre-registered CPU experiment. It tests whether a label-free rule can tell that an answer to a
math problem contains no error, so that no step is flagged in it.
- **The design is fixed in:** `docs/experiments/RAW_NULL_GATE_PREREG_2026-10-30.md`.
- **The sources of the ideas are in:** `docs/research_notes/RAW_NULL_GATE_LITERATURE_2026-10-30.md`.
- **Your job:** run four commands, check that each one passes, commit the outputs, and push.
- **Not your job:** interpreting results, or changing the method.

## 1. Rules (hard)

1. **Do NOT edit these files:**
   - `spectral_utils/raw_null_gate.py`
   - `results/raw_null_gate_v1/run_raw_null_gate.py`
   - `results/raw_null_gate_v1/verify_raw_null_gate.py`
   - `results/raw_null_gate_v1/PROTOCOL.json`

   The runner checks their SHA-256 hashes and stops if any of them changed. If it stops for that reason, restore the
   files with `git checkout -- <file>`. Do not "update the hash".
2. **Do NOT edit any file under these directories:** `results/ds_gate_*`, `results/bracha_lr_transfer_v1`,
   `results/prm_boost_*`, `spectral_utils/`.
3. **Do NOT open or print label files** to "inspect" them. Labels are read only inside the runner, after the
   label-free stage.
4. **`missing` cells are expected.** PRMBench backbone ('prm') raw banks exist only on AIRCC, so 12 cells (AUTO,
   TELE16 and SUB5, all 'prm', on prmbench) will be `missing`. That is NOT a failure. Any status that starts with
   `FAIL` IS a failure.
5. **Use the Python environment that ran `results/ds_gate_v1b_fixed/run_e1.py` before.** It has numpy, scipy,
   sklearn and pytest. Run every command from the repository root.

## 2. Setup

```bash
git fetch origin
git checkout claude/raw-null-gate-2026-10-30
git pull --ff-only
git log --oneline -1
```

**Expected:** the last commit message mentions "raw-null gate". **The working tree must be clean,** meaning
`git status` shows no modified tracked files.

## 3. The four commands, in order

**Step A, self-test (no data needed, about 1 minute):**
```bash
python results/raw_null_gate_v1/run_raw_null_gate.py selftest
```
- **Pass:** the last line is `SELFTEST PASSED`. Any other ending is a STOP.

**Step B, data and parity check (a few minutes):**
```bash
python results/raw_null_gate_v1/run_raw_null_gate.py check
```
- **Pass:** the last line is `CHECK PASSED <n> ok, <m> missing`. With all expected data present this reads
  `CHECK PASSED 68 ok, 12 missing`. It writes `results/raw_null_gate_v1/CHECK.json`.
- **What `check` already does:** it runs the whole label-free stage on every cell. A crash therefore appears here,
  not in Step C.
- **Missing files:** a missing file that is not one of the 12 expected PRMBench backbone cells is reported as
  `FAIL unexpected missing file`. That usually means the data path on this machine differs. STOP and report it; do
  not edit paths in code.
- **If it prints `CHECK FAILED`:** STOP. Report every line that starts with `FAIL`, together with its `reason`.

**Step C, the run (one pass over all ok cells; expect tens of minutes):**
```bash
python results/raw_null_gate_v1/run_raw_null_gate.py run > results/raw_null_gate_v1/run.log 2>&1
```
- **Pass:** the last line of `run.log` is `RUN DONE`.
- **It writes:** `RESULTS.csv`, `FITS.json`, `VERDICT.json` and `labelfree/*.npz`.

**Step D, independent verification:**
```bash
python results/raw_null_gate_v1/verify_raw_null_gate.py real
```
- **Pass:** the last line starts with `REAL PASS`, and it writes `VERIFY.json`.
- **Exit codes:** 0 means PASS, 1 means FAIL, 2 means it refused because an input file is missing or unreadable.
- **If it says FAIL or refuses:** STOP. Do NOT rerun Step C with changes. Report the failing rows from
  `VERIFY.json`, or the refusal message.

## 3a. Tested before hand-off (on the Mac, 2026-10-30)

These were run on the frozen code:
- `selftest`: PASSED.
- `verify_raw_null_gate.py synthetic`: PASS, 1,374 comparisons.
- **A full mock run with fake loaders:** check 68 ok and 12 missing, `RUN DONE`, and `REAL PASS` with 7,459
  comparisons and 0 failed.

So if something fails on your machine, the cause is the data or the environment there, not the logic. Report it;
do not fix it.

## 4. Commit and push (only if A–D all passed)

```bash
git add results/raw_null_gate_v1/CHECK.json results/raw_null_gate_v1/RESULTS.csv results/raw_null_gate_v1/FITS.json results/raw_null_gate_v1/VERDICT.json results/raw_null_gate_v1/VERIFY.json results/raw_null_gate_v1/run.log results/raw_null_gate_v1/labelfree
git commit -m "raw_null_gate_v1: run on the Windows machine (selftest, check, run, verify all passed)"
git push origin claude/raw-null-gate-2026-10-30
```

**If something stopped,** commit only `CHECK.json` and any log files, with a message saying exactly where it
stopped, and push them.

## 5. What to report back (copy, do not interpret)

1. The last line of each of the four commands.
2. From `CHECK.json`: the counts of `ok`, `missing` and `FAIL`.
3. The full content of `VERDICT.json`.
4. The overall line of `VERIFY.json`.
5. The commit hash you pushed.

Do not add conclusions about whether the method works. The analysis is done afterwards, on the other machine.
