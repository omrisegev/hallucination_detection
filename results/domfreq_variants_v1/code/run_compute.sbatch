#!/bin/bash
# domfreq_variants_v1 stage 1 (CPU only): per-step variant cache for 4 PRMs x {PRMBench, Socratic}. No labels read.
#SBATCH --job-name=dfv_compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/results/domfreq_variants_v1/logs/%x_%j.out
set -uo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/pyenv_native_312_20261011/activate.sh
export MPLBACKEND=Agg
R=/shared/cycle2_tau_averbuch_prj/omrisegev1/results; D=$R/domfreq_variants_v1; O=$R/prm_boost_gen_v1; E=$R/prm_boost_external_v1
cd $D
py() { python3 -B $D/code/compute_variants.py --workers 16 --out $D/cache "$@" || exit 3; }
py --bench prmbench --tag qwen_prm_7b --tele $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl --feat $R/prm_boost_v1/bank_prm/FEATURES.npz
for a in qwen_prm_72b pure_prm rlhflow_prm; do
  py --bench prmbench --tag $a --tele $O/full_$a/prmbench_telemetry.pkl --feat $O/bank_$a/FEATURES.npz
done
py --bench socratic --tag qwen_prm_7b --tele $E/full_qwen_prm_socratic/external_telemetry.pkl --feat $E/full_qwen_prm_socratic/bank_v2/FEATURES.npz
for a in qwen_prm_72b pure_prm rlhflow_prm; do
  py --bench socratic --tag $a --tele $E/full_${a}_socratic/external_telemetry.pkl --feat $E/full_${a}_socratic/bank_v2/FEATURES.npz
done
echo ALL_DONE
