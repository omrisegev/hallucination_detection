---
description: Check AIRCC job status — squeue/sacct for the user's jobs plus a log tail, with a one-line interpretation. Accepts an optional job id; without one, shows all of omrisegev1's jobs.
---

Report cluster job status. `$SHARED` = `/shared/cycle2_tau_averbuch_prj/omrisegev1`.

## Step 1 — Connectivity pre-check

`ssh -o ConnectTimeout=5 aircc 'echo ok'`. On failure: **check TAU VPN**, stop.

## Step 2 — Queue + accounting

No job id given:
```bash
ssh aircc 'squeue -u omrisegev1 -o "%.10i %.9P %.20j %.2t %.10M %.6D %R"'
```

With job id `<ID>` (also covers finished jobs that left the queue):
```bash
ssh aircc 'sacct -j <ID> --format=JobID,JobName%20,State,Elapsed,ExitCode,NodeList'
```

## Step 3 — Log tail

```bash
ssh aircc 'tail -n 30 $SHARED/logs/*_<ID>.out 2>/dev/null || tail -n 30 $SHARED/logs/spectral_infer_<ID>.out'
```

Look for:
- `pyxis: importing docker image: ...` / `pyxis: imported docker image: ...` — normal Pyxis
  container import (~8 min on first job per node; subsequent jobs are instant)
- `[driver] GPU: NVIDIA B200 capability=(10, 0)` — container started, GPU visible
- Driver progress lines (`Loaded N problems`, `=== T=... ===`, per-problem lines)
- `PREEMPTED — checkpoint saved` — preempted cleanly; Slurm will requeue
- `ALL TEMPS COMPLETE` — job finished successfully

## Step 4 — Interpret

| Observation | Verdict |
|---|---|
| `PD` + Reason Priority/Resources | Queued — normal, just waiting |
| `R` + `pyxis: importing...` in log | STARTING — container import in progress (~8 min first time on this node) |
| `R` + `[driver] GPU:` + progress lines | RUNNING, healthy |
| `REQUEUED`/restarted + `PREEMPTED` line earlier in log | Preempted and auto-resumed — verify the resume banner (`N/M problems already complete`) appears after restart |
| `F` / non-zero ExitCode | FAILED — quote the last error lines; common causes: OOM, bad partition/QoS, missing HF_TOKEN for gated model, pip install failure |
| exit code 125 + BPF/cgroup error in log | Pyxis enroot failure (rare; try requeue or different node) |
| `CD` + `ALL TEMPS COMPLETE` | DONE — suggest `/aircc-fetch` |

**Note**: rootless Docker is NOT available on power-gpu nodes (daemon failed since 2026-07-01).
All jobs use Pyxis. If you see a Docker-related error, it means an old sbatch template was
used — update to the Pyxis templates in `cluster/`.

Print exactly one bold verdict line. For repeated polling, spawn the **cluster-ops**
agent instead of looping here.
