#!/bin/bash
# Stage 1.2: PRMBench sign-safe variant (|v| SML weights, no anchor flip). Diagnostic, post-registration.
#SBATCH -J pbc_b_signabs
#SBATCH -p cpu
#SBATCH -A cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o pbc_b_%j.out
set -euo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_collection_v1/common.sh
E=$O/ablation5_signabs_v1; test ! -e $E || { echo "refusing: $E exists"; exit 2; }
python3 -B $SS/run_prmbench.py $M abs qwen_prm_7b $R/prm_boost_v1/bank_prm/FEATURES.npz $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl $E
for a in $PRMS_PB; do
  python3 -B $SS/run_prmbench.py $M abs $a $O/bank_$a/FEATURES.npz $O/full_$a/prmbench_telemetry.pkl $E
done
python3 -B $SS/run_prmbench.py $M abs math_shepherd $O/bank_math_shepherd_digitfix/FEATURES.npz $O/full_math_shepherd/prmbench_telemetry.pkl $E
md5sum $E/* > $J/MD5_ablation5_signabs_v1.txt
echo JOB_B_DONE
