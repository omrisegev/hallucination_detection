# Running the Step-227 studies

This is the execution handoff for Claude on the computer that contains the experiment data. All
study code is already implemented. Do not redesign the experiments, edit the registered runner, or
choose settings after looking at AUROC.

## Current state: do not start the real-data studies

The full source-matched synthetic admission benchmark has already run on 40 independent train/test
draws in each of four known-truth worlds. Its decision is **`STOP_AND_REVISE`**, not pass. Therefore
Claude must let the already-running registered sweep finish normally, but must **not** start any new
Step-227 computation on the real cache.

The failed gates are:

- relative sign(rho) polarity recovery;
- SDSF versus SU-PCR mean improvement in the planted sparse-dependency world;
- the paired confidence interval excluding zero;
- recovery of a meaningful fraction of the oracle-available gain.

Inspect the committed evidence, without rerunning or changing it:

```bash
python -c "import json; d=json.load(open('results/synthetic_dependency_fusion/summary.json')); print(d['admission']['decision']); [print(g['pass'], g['gate'], g['observed']) for g in d['admission']['gates']]"
```

Expected first line: `STOP_AND_REVISE`. `scripts/run_step227_studies.py` verifies that the result is a
full run, checks its SHA-256 against the current synthetic script, and exits before reading real data
when admission is not `PROCEED_TO_REAL_DATA`. There is intentionally no bypass.

The code below is the future runbook after a revised method has passed a separately preregistered,
disjoint-seed confirmation. It is retained so execution on the data computer is mechanical once that
scientific gate is genuinely satisfied.

The orchestrator runs, in order:

1. verifies the source-matched synthetic admission result;
2. the solver-mechanism smoke test;
3. the residual-identifiability smoke test;
4. the complete solver-mechanism study;
5. the complete residual-identifiability study;
6. the DEEM collapse diagnosis, label-free repair pilot, and—only if the pilot passes—the frozen
   repaired-soft evaluation.

The scientific decisions and thresholds are fixed in `SPEC_SOLVER_MECHANISM_STUDY.md`.

## Non-negotiable safeguards

- Work on `master`.
- Do not edit these four hash-pinned files:

  - `scripts/run_dependency_fusion_experiment.py`
  - `spectral_utils/dependency_fusion.py`
  - `spectral_utils/deem_adapter.py`
  - `spectral_utils/upcr.py`

- Do not interrupt, restart, or compete with the registered dependency-fusion sweep.
- Do not stage, commit, or push anything. Omri will review and commit the generated artifacts.
- Do not edit `HISTORY.md` or `PROGRESS.md`.
- Do not select a κ, DEEM configuration, threshold, or reported subset using labels. The scripts
  enforce the preregistered choices.

## 1. Update and inspect the checkout

From the repository root:

```bash
git switch master
git pull --ff-only origin master
git status --short --branch
```

Stop if `git status` shows an unexpected local modification, especially in a hash-pinned file. Do
not reset or discard someone else's work.

The expected source commit at the time of this handoff is the commit containing
`RUN_STEP227_STUDIES.md`. A later fast-forward is acceptable only if Omri says it is.

## 2. Prepare Python

Use the same environment that ran the registered sweep when possible. Otherwise create a clean
environment and install the repository plus the pinned DEEM extra:

```bash
python -m venv .venv-step227
source .venv-step227/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dependency-experiment]"
```

On Windows PowerShell, activation is:

```powershell
.venv-step227\Scripts\Activate.ps1
```

Verify the pinned package and compile the new scripts:

```bash
python -c "from importlib.metadata import version; assert version('deem') == '0.2.0'; print('deem', version('deem'))"
python -m py_compile scripts/synthetic_dependency_fusion_validation.py scripts/run_step227_studies.py scripts/solver_mechanism_study.py scripts/residual_identifiability_study.py scripts/deem_soft_collapse_probe.py
```

The data directory passed below must contain the same inputs used by the registered sweep, normally
`derived_views.pkl` and `trace_cells.pkl`. Use an absolute path if it is outside the repository.

## 3. Confirm the registered sweep state

If the registered sweep is still running, identify its PID without signalling it. On POSIX:

```bash
ps -ef | grep '[r]un_dependency_fusion_experiment.py'
```

On Windows PowerShell:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_dependency_fusion_experiment.py' } | Select-Object ProcessId, CommandLine
```

The orchestrator accepts that PID and waits, but only after the synthetic admission gate passes. It
requires all three conditions before starting any numeric Step-227 work:

- the PID no longer exists;
- `results/dependency_fusion_study/records.jsonl` has been quiet for at least 15 minutes;
- `summary.json` is readable and newer than `records.jsonl`.

The PID check is cross-platform. A process-query failure blocks execution. `--skip-wait` skips only
polling; it does **not** bypass these gates.

## 4. Run everything—but only after synthetic admission passes

Do not execute the commands in this section while the committed decision is `STOP_AND_REVISE`.

If the registered sweep is still running or its former PID is known:

```bash
python scripts/run_step227_studies.py \
  --pid <REGISTERED_SWEEP_PID> \
  --data-dir /absolute/path/to/data \
  --device auto
```

If the sweep is already known to have completed and the PID is gone:

```bash
python scripts/run_step227_studies.py \
  --pid <FORMER_SWEEP_PID> \
  --skip-wait \
  --data-dir /absolute/path/to/data \
  --device auto
```

If the old PID was not recorded, omit `--pid`; the fresh-checkpoint gates must still pass. Use
`--device cpu`, `--device cuda`, or `--device mps` only when the environment requires an explicit
choice.

Run this in a persistent terminal session. The residual study performs 48,000 decompositions and
automatically uses up to four cell workers only when its timing probe projects more than eight
single-core hours. The DEEM stage may be substantially longer if a healthy configuration reaches the
all-cell evaluation.

Every subprocess log is written under:

```text
results/_step227_logs/
```

A nonzero exit stops the chain. Do not work around a failed gate. Report the command, terminal tail,
and corresponding log to Omri.

## 5. Expected gates and outputs

The solver study must print:

- GOOD_6 = `0.7733442`;
- both committed factorial corners reproduced within `1e-9` AUROC;
- the reconstructed ridge vector within `1e-10` relative error.

The residual study requires exactly 1,000 successful draws for each of two nulls on each cell. Any
failed draw or incomplete split-half repetition stops the study.

The DEEM probe first runs a two-epoch adapter-equivalence check. Its maximum score difference must be
at most `1e-10`. It then saves aligned probabilities, scores, epoch diagnostics, and last-finite
checkpoints for collapsed/failed runs.

Expected output directories:

```text
results/synthetic_dependency_fusion/
results/solver_mechanism/
results/residual_identifiability/
results/deem_probe/
results/_step227_logs/
```

Important files include:

- `results/synthetic_dependency_fusion/REPORT.md`
- `results/synthetic_dependency_fusion/summary.json`
- `results/synthetic_dependency_fusion/replicates.csv`
- `results/solver_mechanism/heldout_repetitions.csv`
- `results/residual_identifiability/family_tests.csv`
- `results/residual_identifiability/null_draws_summary.csv`
- `results/deem_probe/grid.csv`
- `results/deem_probe/evaluation_per_cell.csv` when the repair pilot passes
- each study's `summary.json`

## 6. Handoff after completion

Run:

```bash
git status --short
git diff -- scripts/run_dependency_fusion_experiment.py spectral_utils/dependency_fusion.py spectral_utils/deem_adapter.py spectral_utils/upcr.py
```

The second command must print nothing. Do not stage or commit. Send Omri:

- the complete terminal outcome;
- `results/_step227_logs/`;
- the three study result directories;
- `git status --short`;
- any failed gate or incomplete output, without rerunning it under changed settings.
