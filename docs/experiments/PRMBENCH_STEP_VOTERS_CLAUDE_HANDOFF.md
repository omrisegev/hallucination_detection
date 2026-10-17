# PRMBench Step Voters — Claude Session Coordination & Handoff

**Branch**: `claude/prm-voters-processbench-pilot`  
**Date**: 2026-10-17  
**Isolated Cluster Dir**: `/shared/cycle2_tau_averbuch_prj/omrisegev1/code_prm_voters_v1`  
**Cluster Results**: `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prmbench_voters_v1`  
**Track B Results (Already Finished)**: `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_gen_v1`  
**Sync Commit**: code identical to `246ebb4cc8cc44c69ff255637a952f135ee7371f` (verified by file hashes 2026-10-17; the hash written in the cluster `SYNC_COMMIT.json` does not exist)

---

## 1. What Has Been Completed

### A. Code & Cluster Environment Fix
- The sbatch script `cluster/submit_prmbench_step_voters.sbatch` was updated to support the compute nodes' offline environment.
- In `nvcr.io/nvidia/pytorch:25.01-py3`, packages are installed from the staged wheelhouse `/shared/cycle2_tau_averbuch_prj/omrisegev1/wheelhouse/tf_thinkfix_20261010` using:
  ```bash
  WHEELS=$SHARED/wheelhouse/tf_thinkfix_20261010
  python -m pip install -q --user --no-deps "$WHEELS"/*.whl
  ```
  This resolves `transformers 4.57.6`, `accelerate 1.15.0`, `datasets 5.0.1`, `pyarrow 25.0.1`, `scikit-learn 1.9.1`, and dependencies without attempting external network calls or failing on `hf-xet`.
- Clean commit `6afa897af` created and pushed/synced to `code_prm_voters_v1`.

### B. Pilot Gate Execution (N=30, 35 meta rows) — ALL PASSED
All three pilots were submitted and executed on NVIDIA B200 GPUs (`power-gpu`, `owner_940`):
1. **Job 286481** (`uprm_judge`, `Qwen/Qwen3-8B`):
   - 35 rows scored in 1.8s.
   - `n_failed == 0`, `n_unscored == 0`, `n_truncated == 0`, zero recovery errors.
2. **Job 286483** (`uprm_judge`, `Qwen/Qwen2.5-14B-Instruct`):
   - 35 rows scored in 2.0s.
   - `n_failed == 0`, `n_unscored == 0`, `n_truncated == 0`, zero recovery errors.
3. **Job 286484** (`critic`, `Qwen/Qwen2.5-72B-Instruct`):
   - 35 rows generated in 12m50s (~10–15s per critique).
   - `n_failed == 0`, `n_unscored == 0`, `n_truncated == 0`.
- All manifests are saved in `results/prmbench_voters_pilot_v1/` and on the cluster at `$SHARED/results/prmbench_voters_v1/*_pilot/`.
- Pilot pkl files are cached locally at `cache/prmbench_voters_v1/`.

### C. Track B Status Verification
- Contrary to initial assumptions in the original handoff doc, Track B PRMs (`qwen_prm_72b`, `rlhflow`, `pure`, `mathshepherd`) were **already executed by Codex on Oct 13–14**.
- Complete 1.1 GB telemetry files exist at:
  - `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_gen_v1/full_qwen_prm_72b/prmbench_telemetry.pkl`
  - `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_gen_v1/full_rlhflow_prm/prmbench_telemetry.pkl`
  - `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_gen_v1/full_pure_prm/prmbench_telemetry.pkl`
  - `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_gen_v1/full_math_shepherd/prmbench_telemetry.pkl`
  - Existing PRM-7B baseline: `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_v1/full_prm/prmbench_telemetry.pkl`
- **DO NOT re-run Track B.** Only Track A step voters (the 2 judges and the sharded 72B critic) need full runs.

---

## 2. Immediate Next Steps for Claude

### Step 1: Submit Full Runs on `aircc`
*(Ensure human approval has been given for the full run GPU chains)*

SSH to `aircc` with `export SLURM_CONF_SERVER=controller-primary`:
```bash
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1
ACC="-A cycle3_tau_averbuch_prj -p power-gpu --qos=owner_940"
OUT=$SH/results/prmbench_voters_v1
V=cluster/submit_prmbench_step_voters.sbatch

chain() { local n=$1; shift; local opts=(); while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
  local prev="" dep=(); for _ in $(seq "$n"); do
    dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:$prev)
    prev=$(sbatch --parsable $ACC "${opts[@]}" "${dep[@]}" "$@"); echo "  link $prev"; done; }

cd $SH/code_prm_voters_v1

# 1. Judges (each a linear chain of 2 links)
chain 2 -- $V --kind uprm_judge --model Qwen/Qwen3-8B             --out $OUT/uprm_qwen3_8b_full
chain 2 -- $V --kind uprm_judge --model Qwen/Qwen2.5-14B-Instruct --out $OUT/uprm_qwen25_14b_full

# 2. Critic 72B (6 parallel shards, each a linear chain of 3 links)
for i in 0 1 2 3 4 5; do
  chain 3 -- $V --kind critic --model Qwen/Qwen2.5-72B-Instruct --n-shards 6 --shard $i --out $OUT/critic_qwen72b_full
done
```

### Step 2: Sync Results to Drive / Local
When full runs complete, copy results to Google Drive and local cache:
```bash
nohup $SH/bin/rclone copy $OUT gdrive:hallucination_detection/cluster_results/prmbench_voters_v1 \
  --transfers 4 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v > $SH/logs/upload_prmbench_voters.log 2>&1 &
```
Pull the pkl files to local `cache/prmbench_voters_v1/`.

### Step 3: Run Analysis
Run `scripts/run_prm_voters_prmbench.py` on CPU:
- Baseline check: verify PRM-7B reproduces official PRMScore of **0.6546**.
- Evaluate MoM error-count estimation and continuous L-SML fusion on PRMBench using all voters vs 4 PRMs alone.
- Write report to `results/prm_voters_prmbench_v1/README.md`.
- Update `PROGRESS.md` and `HISTORY.md`.


## 3. Status 2026-10-17 (Claude)
Full runs submitted with Omri's approval: jobs 287340-287361 (j8b 287340-41, j14b 287342-43, critic shards 0-5 as linear 3-link chains 287344-287361). Track B directory names corrected above.
