#!/bin/bash
# CPU job: Z17 (parity with AUTO_N16) and CAT34 for the 4 PRMs. Inputs are the bank-sweep inputs (AIRCC 288319).
# Copy auto_raw_plus_z.py next to this file on AIRCC, then: sbatch -A cycle3_tau_averbuch_prj -p cpu run.sbatch
#SBATCH --job-name=auto_raw_z
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/auto_raw_plus_z_v1/%x_%j.out
set -uo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/pyenv_native_312_20261011/activate.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; SW=$O/bank_sweep_v1
D=/shared/cycle2_tau_averbuch_prj/omrisegev1/auto_raw_plus_z_v1; X=$D/out
cd $D
python3 -B $D/auto_raw_plus_z.py qwen_prm_7b $R/prm_boost_v1/bank_prm/FEATURES.npz $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl $SW/SWEEP_RESULTS_qwen_prm_7b.csv $X || exit 3
for a in qwen_prm_72b pure_prm rlhflow_prm; do
  python3 -B $D/auto_raw_plus_z.py $a $O/bank_$a/FEATURES.npz $O/full_$a/prmbench_telemetry.pkl $SW/SWEEP_RESULTS_$a.csv $X || exit 3
done
