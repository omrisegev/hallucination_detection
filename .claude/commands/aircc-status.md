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
ssh aircc 'tail -n 30 $SHARED/logs/*_<ID>.out'
```

Look for the driver's progress lines (`[T=..] problem i/N cand k/K`), a
`PREEMPTED — checkpoint saved` line, or a final `ALL TEMPS COMPLETE`.

## Step 4 — Interpret

| Observation | Verdict |
|---|---|
| `PD` + Reason Priority/Resources | Queued — normal, just waiting |
| `R` + fresh progress lines in log | RUNNING, healthy |
| `REQUEUED`/restarted + `PREEMPTED` line earlier in log | Preempted and auto-resumed — no action needed; verify the resume banner (`N/M problems already complete`) appears after restart |
| `F` / non-zero ExitCode | FAILED — quote the last error lines; common causes: docker daemon (rerun happens on requeue), OOM, bad partition/QoS |
| `CD` + `ALL TEMPS COMPLETE` | DONE — suggest `/aircc-fetch` |

Print exactly one bold verdict line. For repeated polling, spawn the **cluster-ops**
agent instead of looping here.
