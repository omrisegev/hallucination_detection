---
description: Mandatory pre-submit gate for any cluster/GPU job — worktree and sync state, disk and TMPDIR locally and on AIRCC, CPU smoke of the preset, and the label-sanity gate on any named cache. Prints a PASS/FAIL table; nothing may be sbatch'ed without a PASS from this session. (Not related to scripts/preflight_*.py, which are statistical replay checks.)
---

Run every step. Do not skip a step because it "obviously passes". Report each row as
PASS / FAIL / SKIPPED(reason). Stop and report on the first FAIL; do not submit.

Argument: `$ARGUMENTS` = preset id(s) from `cluster/presets.py`, optionally followed by a
cache path (`cache/repgrid/<cell>` or a raw pkl) to gate. Ask if missing.

## 1. Tree state (read-only)
```
git worktree list
git status --porcelain | head -20
cat SYNC_COMMIT.json 2>/dev/null
```
- FAIL if the working tree is dirty and the job is a full run (dirty = smoke/pilot only).
- Name the branch that will be synced. If it is not the branch Omri named this session, FAIL.

## 2. Local resources
```
df -h .
echo "TMPDIR=$TMPDIR"
```
- FAIL if free space on the repo drive < 10 GB.
- FAIL if TMPDIR is unset or points at a volume with < 10 GB (set it explicitly and re-run).

## 3. Cluster resources (two probes; report observations, never "VPN down")
```
ssh -o ConnectTimeout=10 -o BatchMode=yes aircc 'echo probe1'
```
If it fails, retry once with `ConnectTimeout=30`. If both fail, report "two probes timed
out" and FAIL this row. Otherwise:
```
ssh aircc 'export SLURM_CONF_SERVER=controller-primary; S=/shared/cycle2_tau_averbuch_prj/omrisegev1; df -h $S | tail -1; echo TMPDIR=$TMPDIR; squeue -u omrisegev1 -h | wc -l'
```
- FAIL if `$SHARED` has < 50 GB free.
- Note the number of queued/running jobs; `cluster/sync_code.sh` refuses to sync while any exist.

## 4. CPU smoke of the preset (the part that catches 4 of 6 pilot bugs offline)
```
PYTHONPATH=. python scripts/smoke_preset.py <preset_id> [<preset_id> ...]
```
- FAIL on non-zero exit. Quote the failing check verbatim.
- For a new driver or a modified `run_inference.py`, also run the driver end to end on
  CPU with `--n-samples 5 --max-new 32` (or the driver's smallest flags) into a scratch
  dir, so serialization, asserts and the JSON dump all execute once. FAIL on any exception.

## 5. Label-sanity gate on any named cache
```
PYTHONPATH=. python scripts/inspect_cell.py <cache path>
```
- Exit 1 = DEGENERATE (too few minority rows, accuracy outside [0.05, 0.98], or >10 %
  of traces pinned at max_new). FAIL. Do not pass `--allow-degenerate` without Omri's
  explicit reason.

## 6. Verdict
Print:
```
| check | result | detail |
| tree | PASS/FAIL | branch, dirty count |
| local disk / TMPDIR | ... | ... |
| cluster disk / TMPDIR | ... | ... |
| smoke | ... | ... |
| label sanity | ... / SKIPPED | ... |
PREFLIGHT: PASS | FAIL
```
Only after `PREFLIGHT: PASS` may `/aircc-submit` proceed, in this same session.
