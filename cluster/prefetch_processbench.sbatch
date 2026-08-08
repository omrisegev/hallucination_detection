#!/bin/bash
# Prefetch Qwen/ProcessBench (4 configs, 3,400 rows) into the SHARED HF datasets cache.
#
# Why a dedicated job rather than letting the inference job download it: `cluster/cpu_job.sbatch`
# does not export HF_HOME, so anything it downloads lands in the container's ephemeral cache and
# is gone when the job ends — and `cluster/prefetch.sbatch` only handles model snapshots. The
# GPU inference job must find the dataset already on shared storage; a 4-config download racing
# 8 concurrent jobs is exactly the kind of shared-state corruption that killed the first wave.
#
# ProcessBench ships four configs and `load_dataset` needs each by name, so all four are pulled
# here explicitly. Total is a few MB — this job is seconds of work, minutes of queue.
#
# Submit:
#   sbatch -p power-gpu --qos=owner_880 cluster/prefetch_processbench.sbatch
#
#SBATCH --job-name=prefetch_pb
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code
# No --container-name, for the same enroot-rootfs race reason as submit_inference.sbatch.
# --gpus=1 is requested only because power-gpu is the one partition that is up; nothing here
# touches the GPU.

set -uo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export HF_HOME=$SHARED/hf_cache
export HF_DATASETS_CACHE=$SHARED/hf_cache/datasets
export PIP_CACHE_DIR=$SHARED/pip_cache
export PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID}
mkdir -p "$PYTHONUSERBASE"

echo "[sbatch] job $SLURM_JOB_ID on $(hostname)"

pip install -q --user -r cluster/requirements.txt

# The loader is imported from the package rather than calling `load_dataset` inline, so this
# job exercises the SAME code path the inference driver will use. A config name that works here
# is a config name that works there; a prefetch that "succeeds" against a different call is not
# evidence of anything.
exec python - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from spectral_utils.processbench import SUBSETS, load_processbench

total = 0
for subset in SUBSETS:
    rows = load_processbench(subset)
    n_err = sum(1 for r in rows if r["label"] >= 0)
    steps = [len(r["steps"]) for r in rows]
    total += len(rows)
    print(f"[prefetch] {subset:16s} {len(rows):5d} rows | {n_err:5d} with an annotated error "
          f"step | steps/row min {min(steps)} median {sorted(steps)[len(steps)//2]} "
          f"max {max(steps)}", flush=True)

print(f"[prefetch] TOTAL {total} rows across {len(SUBSETS)} configs")
assert total == 3400, f"expected ProcessBench's 3,400 rows, got {total} — wrong split or config"
print("[prefetch] done")
PYEOF
