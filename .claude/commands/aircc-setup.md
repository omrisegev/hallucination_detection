---
description: One-time AIRCC cluster bootstrap — guides the interactive first login (user types passwords), then automates ssh config, connectivity check, partition/QoS discovery, /shared directory setup, and image+model prefetch. Run once before any cluster work; safe to re-run (idempotent).
---

Bootstrap AIRCC cluster access. Steps 1–2 are **USER-MANUAL** (interactive password
prompts — Claude cannot type them); everything after is automated.

## Step 0 — Check what's already done

Run `ssh -o ConnectTimeout=5 -o BatchMode=yes aircc 'echo ok'` (Bash).
- Prints `ok` → key + config already work. Skip to Step 4.
- Fails → check whether `~/.ssh/id_ed25519_aircc` exists. If yes, skip to Step 3. If no, continue.

## Step 1 — USER-MANUAL: first login + password change

Tell the user to do this in their **own terminal** (not through Claude), with **TAU VPN connected**:

```
ssh omrisegev1@slurm-login.iucc.ac.il
```

- Enter the temporary password from the AIRCC welcome email (it must be typed twice before the new one).
- Set a new password.

## Step 2 — USER-MANUAL: generate + register SSH key, download it

Still in the user's own terminal:

1. From the cluster's interactive menu, select **option 2 (Generate & register SSH key)**.
2. Log out, then in a **local PowerShell** window download the private key:

```powershell
scp omrisegev1@slurm-login.iucc.ac.il:/shared/home/users/omrisegev1/.ssh/id_ed25519 $env:USERPROFILE\.ssh\id_ed25519_aircc
```

(If `~/.ssh` doesn't exist locally, create it first: `mkdir $env:USERPROFILE\.ssh`.)

Wait for the user to confirm both steps are done before continuing.

## Step 3 — Write the ssh config (automated)

Append to `~/.ssh/config` (create if missing) — check first that a `Host aircc` block doesn't already exist:

```
Host aircc
    HostName slurm-login.iucc.ac.il
    User omrisegev1
    IdentityFile ~/.ssh/id_ed25519_aircc
    ServerAliveInterval 60
```

Verify: `ssh -o ConnectTimeout=5 aircc 'echo ok; hostname'`. If it fails, tell the user: **check that TAU VPN is connected** — that is the cause ~90% of the time.

## Step 4 — Discover account partition/QoS

Run `ssh aircc 'sdata'` and `ssh aircc 'sinfo -o "%P %a %l %D"'`. From the output, identify:
- The **owner** partition + QoS (the welcome email says queue type = owner)
- Confirm the sandbox QoS is `sandbox_owner_880` (used by `cluster/smoke_test.sbatch`)

Write the discovered values to `cluster/aircc.env` (create it):

```bash
OWNER_PARTITION=<discovered>
OWNER_QOS=<discovered>
```

`/aircc-submit` reads this file. If `sdata` output is ambiguous, show it to the user and ask.

## Step 5 — Cluster directory setup + code sync + prefetch

```bash
ssh aircc 'bash -s' < cluster/setup_cluster.sh
bash cluster/sync_code.sh
ssh aircc 'cd /shared/cycle2_tau_averbuch_prj/omrisegev1/code && sbatch cluster/prefetch.sbatch'
```

The prefetch job pulls the ~10 GB NGC image and downloads Qwen2.5-Math-1.5B-Instruct to
`$SHARED/hf_cache` (login node can't do this — no docker/GPU). Check it with `/aircc-status`.

## Step 6 — Report

Print a summary: connectivity ✓/✗, partition/QoS recorded, dirs created, prefetch job id.
Then point to the verification ladder: sandbox smoke test → owner-queue smoke → real job
(see `cluster/README.md`).
