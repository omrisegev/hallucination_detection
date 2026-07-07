#!/bin/bash
# Prefetch: pull NGC image via Pyxis (creating the named container cache) and
# download model weights to shared storage. Run once before the first inference job.
#
# Submit: sbatch cluster/prefetch.sbatch
#
#SBATCH --job-name=prefetch
#SBATCH --partition=sandbox
#SBATCH --qos=sandbox_owner_880
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code
#SBATCH --container-name=ngc_pytorch_2501

set -uo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
MODEL_ID=${1:-Qwen/Qwen2.5-Math-1.5B-Instruct}

pip install -q huggingface_hub

python - <<PYEOF
import os
from huggingface_hub import snapshot_download

shared = "$SHARED"
model_id = "$MODEL_ID"
flat_dir = os.path.join(shared, "hf_cache", "flat", model_id.replace("/", "__"))

if os.path.exists(os.path.join(flat_dir, "config.json")):
    print(f"model already at {flat_dir}; skipping download")
else:
    os.makedirs(flat_dir, exist_ok=True)
    out = snapshot_download(repo_id=model_id, local_dir=flat_dir)
    print(f"model at {out}")

print("[prefetch] done")
PYEOF
