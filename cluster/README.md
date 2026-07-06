# AIRCC cluster — quick reference

National AIRCC cluster (Nebius, Cycle #2). Login `omrisegev1@slurm-login.iucc.ac.il`
(**TAU VPN required**), Slurm scheduler, rootless Docker, 8× NVIDIA **B200** (sm_100),
5760 GPU-h, 10 TB. Local ssh alias: `aircc` (set up by `/aircc-setup`).

## Non-negotiables

- **Always work under** `$SHARED = /shared/cycle2_tau_averbuch_prj/omrisegev1` — never `$HOME`.
- **B200 needs the NGC image** `nvcr.io/nvidia/pytorch:25.01-py3`. Stock PyTorch images
  fail with "CUDA error: no kernel image is available". Never pip-upgrade torch inside it.
- **Preemption**: lower-tier jobs get SIGTERM, then 15 min later SIGKILL, then auto-requeue.
  `submit_inference.sbatch` + `run_inference.py` handle this end-to-end (checkpoint + resume);
  keep that pattern for any new job type.
- The login node runs no GPU/docker work — image pulls and model downloads happen in a
  sandbox job (`prefetch.sbatch`).

## Directory layout on /shared

```
$SHARED/code/       # synced working tree (tar-over-ssh, see sync_code.sh)
$SHARED/hf_cache/   # HF_HOME — models + datasets
$SHARED/results/    # job outputs (raw_*.pkl)
$SHARED/logs/       # Slurm logs (%x_%j.out) + container exit codes
$SHARED/pip_cache/  # PIP_CACHE_DIR — kills per-job pip cold-start
```

## Cheat sheet

```bash
# one-time
ssh aircc 'bash -s' < cluster/setup_cluster.sh
bash cluster/sync_code.sh                          # local tree -> $SHARED/code
ssh aircc "cd $SHARED/code && sbatch cluster/prefetch.sbatch"

# sandbox smoke test (2 GPU / 2 h debug partition)
ssh aircc "cd $SHARED/code && sbatch cluster/smoke_test.sbatch"

# real job — partition/QoS from `ssh aircc sdata`, passed at submit time
ssh aircc "cd $SHARED/code && sbatch -p <partition> --qos=<qos> \
    cluster/submit_inference.sbatch --dataset aime24 --temps 0.2,0.6,1.0 \
    --k 8 --n-samples 30 --max-new 1024 --out $SHARED/results/edis_aime24"

# monitor
ssh aircc 'squeue -u omrisegev1'
ssh aircc "sacct -j <id> --format=JobID,State,Elapsed,ExitCode,NodeList"
ssh aircc "tail -n 30 $SHARED/logs/spectral_infer_<id>.out"

# fetch
scp "aircc:$SHARED/results/edis_aime24/raw_*.pkl" cache/edis_aime24/

# deliberate preemption test (verifies the checkpoint/resume chain)
ssh aircc 'scancel --signal=TERM <id>'   # expect "PREEMPTED — checkpoint saved" in log
```

## Output schema

`raw_{dataset}_T{temp}.pkl` = `{idx: {question, gold_row, candidates: [K × {full_text,
token_entropies, token_spilled_energies, token_offsets, top_k_logprobs, gen_token_ids,
label}]}}` — the CLAUDE.md rich-save schema; anything derivable offline, including
re-grading from `full_text`.

## Slurm state cheat codes

| State | Meaning |
|---|---|
| `PD` | pending — check `squeue` REASON column (Priority/Resources = normal wait) |
| `R` | running |
| `RQ`/requeued | was preempted; driver resumes from checkpoint automatically |
| `CG` | completing |
| `F` / non-zero ExitCode | failed — read the log tail |
