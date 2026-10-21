#!/bin/bash
# Collection plan section 3 (approved by Omri 2026-10-21), CPU. New output dirs only.
#  3b Math-Shepherd bank sweep (N14/S6 were stale) and controls (N16 rows stale) on the digit-fixed bank
#  3a Step 467 weight-estimator bank study, PRMBench only, on the thinkfix_mask1 source bank
#SBATCH -J pbc_section3
#SBATCH -p cpu
#SBATCH -A cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o pbc_section3_%j.out
set -euo pipefail
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; M=$S/eval_mirror_prm_gen_v1
J=$R/prm_boost_collection_v1/section3
set +u; . $S/pyenv_native_312_20261011/activate.sh; set -u
OUT3B1=$O/bank_sweep_digitfix_v1; OUT3B2=$O/controls_digitfix_v1; OUT3A=$R/prm_boost_collection_v1/section3_v1/score_banks_thinkfix_mask1
for d in $OUT3B1 $OUT3B2 $OUT3A; do test ! -e $d || { echo "refusing: $d exists"; exit 2; }; done
cd $M
echo "[3b] bank sweep, math_shepherd, digit-fixed bank"
python3 -B $M/lsml_bank_sweep.py math_shepherd $O/bank_math_shepherd_digitfix/FEATURES.npz $O/full_math_shepherd/prmbench_telemetry.pkl $OUT3B1
echo "[3b] controls, math_shepherd, digit-fixed bank"
python3 -B $M/lsml_controls.py math_shepherd $O/bank_math_shepherd_digitfix/FEATURES.npz $O/full_math_shepherd/prmbench_telemetry.pkl $OUT3B2
echo "[3a] weight-estimator bank study, PRMBench only, thinkfix_mask1"
mkdir -p $(dirname $OUT3A)
python3 -B $J/score_banks_prmbench.py $M $R/thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz \
  $R/prmbench_qwen3_8b_telemetry_thinkfix/prmbench_telemetry.pkl $OUT3A \
  $M/.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv
md5sum $OUT3B1/* $OUT3B2/* $OUT3A/* > $R/prm_boost_collection_v1/section3_v1/MD5.txt
echo SECTION3_DONE
