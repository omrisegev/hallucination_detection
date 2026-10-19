#!/bin/bash
#SBATCH -J prmboost_abl5
#SBATCH -p cpu
#SBATCH -A owner_940
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o prmboost_abl5_%j.out
set -uo pipefail
# Same inputs and mirror as ../prm_boosting_bank_sweep_v1/run_sweep.sh (job 288319).
# Before submitting, copy ablation5.py into $M.
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; M=$S/eval_mirror_prm_gen_v1
test -f $M/ablation5.py || { echo "missing $M/ablation5.py"; exit 2; }
cd $M; X=$O/ablation5_v1
python3 -B $M/ablation5.py qwen_prm_7b $R/prm_boost_v1/bank_prm/FEATURES.npz $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl $X
for a in qwen_prm_72b pure_prm rlhflow_prm math_shepherd; do
  python3 -B $M/ablation5.py $a $O/bank_$a/FEATURES.npz $O/full_$a/prmbench_telemetry.pkl $X
done
