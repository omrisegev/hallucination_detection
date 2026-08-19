#!/bin/bash
# Sealed A6-S0b chain on AIRCC: prepare -> run-pythia -> run-analysis -> verify.
#
# Why the sealed chain runs here and not on the Windows machine: the frozen
# S0a artifacts and the S0b source closure are LF/forward-slash trees; a
# Windows checkout CRLF-converts sources and keys manifests with backslashes,
# breaking hash-exact verification. Code therefore reaches the cluster as a
# git bundle cloned into $SHARED/code_a6_s0b (never tar-of-working-tree, which
# on Windows carries CRLF bytes, and never a bare export, which breaks the
# recorded `git rev-parse HEAD`). prepare authenticates the sealed S0a tree
# against its own recorded manifest chain, pinned to the Step-268 SHA-256
# constants.
#
# Stage-aware and idempotent: every stage is guarded by its terminal artifact,
# so a preempted/requeued or chained job re-enters at the first unfinished
# stage (run-pythia checkpoints per prompt; bootstrap/control checkpoints
# resume; prepare is skipped once the boundary exists).
#
# The pip userbase is PERSISTENT and chain-specific (not per-job): the frozen
# boundary records exact package versions at prepare, and load_and_verify
# fails closed if any version differs at run/verify. Links run strictly
# linearly (afterany), so the shared userbase is never written concurrently.
#
# Submit as a linear chain (one wall per link, per the chaining rule):
#   cd $SHARED/code_a6_s0b
#   J1=$(sbatch -p power-gpu --qos=owner_880 cluster/a6_s0b_chain.sbatch | awk '{print $4}')
#   J2=$(sbatch --dependency=afterany:$J1 -p power-gpu --qos=owner_880 cluster/a6_s0b_chain.sbatch | awk '{print $4}')
#   J3=$(sbatch --dependency=afterany:$J2 -p power-gpu --qos=owner_880 cluster/a6_s0b_chain.sbatch | awk '{print $4}')
#
#SBATCH --job-name=a6s0b
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code_a6_s0b
# --gpus=1 despite a CPU-only workload: power-gpu is the only up partition and
# the Pyxis container path is proven under it (see cpu_job.sbatch).

set -euo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export PIP_CACHE_DIR=$SHARED/pip_cache
export PYTHONUSERBASE=$SHARED/pyuserbase_a6s0b
mkdir -p "$PYTHONUSERBASE"

OUT=results/automatic_group_free_phase_a6_s0b_v1
PYTHIA=local_cache/a6_s0b_pythia_c4fc8d5

echo "[a6s0b] job $SLURM_JOB_ID on $(hostname), HEAD $(git rev-parse HEAD)"

pip install -q --user -r cluster/requirements.txt

if [ ! -f "$PYTHIA/model.safetensors" ]; then
    echo "[a6s0b] stage 0: download + authenticate Pythia"
    python scripts/download_a6_s0b_pythia.py --out "$PYTHIA"
fi

if [ ! -f "$OUT/A6_S0B_BOUNDARY.json" ]; then
    echo "[a6s0b] stage 1: prepare (manifest-chain authentication of sealed S0a + Pythia)"
    python scripts/automatic_group_free_phase_a6_s0b.py prepare \
        --out "$OUT" \
        --pythia-source "$PYTHIA" \
        --pythia-official-tree "$PYTHIA/PYTHIA_OFFICIAL_TREE.json"
fi

if [ ! -f "$OUT/PYTHIA_NLL_COMPLETE.json" ]; then
    echo "[a6s0b] stage 2: run-pythia (14,400 prompt NLLs, per-prompt checkpoints)"
    python scripts/automatic_group_free_phase_a6_s0b.py run-pythia --out "$OUT"
fi

if [ ! -f "$OUT/S0B_COMPLETE.json" ] && [ ! -f "$OUT/S0B_CLOSED.json" ]; then
    echo "[a6s0b] stage 3: run-analysis (shortcut audit + matching freeze)"
    python scripts/automatic_group_free_phase_a6_s0b.py run-analysis --out "$OUT"
fi

# The verify-success marker lives OUTSIDE $OUT: the sealed output root has a
# fail-closed path allowlist and must contain only registered artifacts.
HEAD_SHA=$(git rev-parse HEAD)
VERIFY_MARK=$SHARED/logs/a6s0b_verify_ok_${HEAD_SHA}
if [ ! -f "$VERIFY_MARK" ]; then
    echo "[a6s0b] stage 4: authoritative verify (full no-write replay)"
    python scripts/automatic_group_free_phase_a6_s0b.py verify --out "$OUT"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$VERIFY_MARK"
else
    echo "[a6s0b] verify already recorded at $VERIFY_MARK; nothing to do"
fi

echo "[a6s0b] chain complete"
