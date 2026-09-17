#!/bin/bash
# AIRCC per-layer lens capture for the LOCALIZATION rows (ProcessBench / PRMBench),
# teacher-forced, one forward pass per provided chain. Runs
# cluster/run_localization_layer_views.py inside the B200 NGC container. Preemption-safe:
# --signal=B:TERM@900 delivers SIGTERM to the driver, which finishes the current batch,
# writes per-row npz files (already atomic) and exits 85; a clean exit 85 is NOT requeued,
# so chain resume links LINEARLY with --dependency=afterany:<prev> (cluster/README.md).
#
# Pilot (gate order: local smoke -> N=30 pilot -> full):
#   sbatch -p power-gpu --qos=owner_880 cluster/submit_localization_layer_views.sbatch \
#       --task processbench --subset gsm8k --model Qwen/Qwen3-8B --n-samples 30 \
#       --telemetry $R/pb_qwen3_8b/gsm8k.pkl --out $R/pb_layer_views_qwen3_8b_pilot/gsm8k
# Full:
#   sbatch -p power-gpu --qos=owner_880 cluster/submit_localization_layer_views.sbatch \
#       --task prmbench --model Qwen/Qwen3-8B --telemetry $R/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl \
#       --out $R/prmbench_layer_views_qwen3_8b
#
#SBATCH --job-name=loc_layer_views
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code
# No --container-name on purpose (named containers race on shared enroot rootfs creation).

set -uo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export HF_HOME=$SHARED/hf_cache
export HF_DATASETS_CACHE=$SHARED/hf_cache/datasets
export PIP_CACHE_DIR=$SHARED/pip_cache
export PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID}
mkdir -p "$PYTHONUSERBASE"
# Same convention as submit_backfill.sbatch: the token line is filled from the live
# (gitignored) copy on the cluster; Qwen3 weights are already in $HF_HOME so the token is
# only needed if a download is triggered.
if [ -f "$SHARED/hf_token" ]; then export HF_TOKEN="$(cat "$SHARED/hf_token")"; fi

echo "[sbatch] localization layer views job $SLURM_JOB_ID on $(hostname), args: $*"

pip install -q --user -r cluster/requirements.txt

# exec replaces bash with python so SIGTERM from Slurm reaches the driver directly.
exec python cluster/run_localization_layer_views.py "$@"
