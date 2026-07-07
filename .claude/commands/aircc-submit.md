---
description: Submit an inference job to the AIRCC cluster — syncs the working tree, runs sbatch with the right partition/QoS, and reports the job id. Accepts DATASET (gsm8k|math500|amc23|aime24 for math; QA datasets require run_inference.py extension), TEMPS, K, N, MAX_NEW, OUT; asks for missing parameters.
---

Submit a `cluster/run_inference.py` job to AIRCC. `$SHARED` below means
`/shared/cycle2_tau_averbuch_prj/omrisegev1`.

## Step 1 — Connectivity pre-check

`ssh -o ConnectTimeout=5 aircc 'echo ok'`. On failure: tell the user **check TAU VPN**, and stop.

## Step 2 — Collect parameters

Required: `DATASET`. Currently supported by `run_inference.py`: `{gsm8k, math500, amc23, aime24}`.
For QA datasets (TriviaQA, NQ-Open, HotpotQA, etc.) the driver needs extending — see the
Thesis Replication Grid plan. Ask if not given.

Defaults (EDIS protocol, ask only if the user wants to deviate): `TEMPS=0.2,0.6,1.0`,
`K=8`, `N=30` (aime24; use 40 for amc23, 50–100 for math500/gsm8k), `MAX_NEW=1024`,
`OUT=$SHARED/results/edis_<DATASET>`.

Read `OWNER_PARTITION` / `OWNER_QOS` from `cluster/aircc.env`. If the file is missing,
run `/aircc-setup` Step 4 first (or `ssh aircc sdata` and ask the user).

Known values (confirmed 2026-07-06): partition=`power-gpu`, QoS=`owner_880`.

**Gated models** (LLaMA-2 family, LLaMA-3): the sbatch must export `HF_TOKEN` before
`load_model`. Add `--export=ALL,HF_TOKEN=<token>` to the sbatch call, or set it in the
sbatch template body. Without it, `from_pretrained` will silently fall back to a non-gated
model or fail with a 401.

For the **smoke test** variant (user says "smoke" or "sandbox test"): submit
`cluster/smoke_test.sbatch` instead — no parameters, sandbox partition is hardcoded.

## Step 3 — Sync code

`bash cluster/sync_code.sh` — tars the working tree (minus .git/pkl/ipynb/pdf/cache/results)
to `$SHARED/code`. This is push-independent by design; do NOT try `git push` instead.

## Step 4 — Submit

```bash
ssh aircc "cd $SHARED/code && sbatch -p $OWNER_PARTITION --qos=$OWNER_QOS \
    cluster/submit_inference.sbatch --dataset <DATASET> --temps <TEMPS> \
    --k <K> --n-samples <N> --max-new <MAX_NEW> --out <OUT>"
```

Parse `Submitted batch job <ID>` from the output. If sbatch rejects the partition/QoS,
show the error and re-run `ssh aircc sdata` to re-discover.

## Step 5 — Confirm and hand off

Run `ssh aircc 'squeue -j <ID> -o "%.10i %.9P %.20j %.2t %.10M %R"'` and show the row.
Report: job id, state (PD = queued is normal), log path
`$SHARED/logs/spectral_infer_<ID>.out`, expected runtime (AIME24 3 temps ≈ 1.5–3 h).

**First job on any node** will print `pyxis: importing docker image: ...` and take ~8 min
before the Python driver starts — this is the Pyxis enroot container import. It's a one-time
cost per node; subsequent jobs on that node reuse the named cache (`ngc_pytorch_2501`)
instantly. This is normal and not an error.

Tell the user to check progress with `/aircc-status <ID>`, or spawn the **cluster-ops**
agent for hands-off monitoring. Jobs survive preemption automatically (checkpoint+requeue) —
no babysitting needed.
