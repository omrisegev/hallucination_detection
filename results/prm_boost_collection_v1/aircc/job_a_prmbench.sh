#!/bin/bash
# Stage 1.0 + 1.1 (plan docs/experiments/PRM_BOOSTING_COLLECTION_PLAN_2026-10-21.md). CPU, label-free scoring.
#SBATCH -J pbc_a_prmbench
#SBATCH -p cpu
#SBATCH -A cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o pbc_a_%j.out
set -euo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_collection_v1/common.sh
E=$O/ablation5_export_v1; test ! -e $E || { echo "refusing: $E exists"; exit 2; }
echo "[1.0] parity: qwen_prm_7b with export must be byte-identical to ablation5_v1"
python3 -B $M/ablation5.py qwen_prm_7b $R/prm_boost_v1/bank_prm/FEATURES.npz $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl $E
cmp $E/ABL5_RESULTS_qwen_prm_7b.csv $O/ablation5_v1/ABL5_RESULTS_qwen_prm_7b.csv || { echo "PARITY FAIL"; exit 3; }; echo "PARITY_1.0 PASS qwen_prm_7b"
echo "[1.1] PRMBench export, remaining PRMs"
for a in $PRMS_PB; do
  python3 -B $M/ablation5.py $a $O/bank_$a/FEATURES.npz $O/full_$a/prmbench_telemetry.pkl $E
  cmp $E/ABL5_RESULTS_$a.csv $O/ablation5_v1/ABL5_RESULTS_$a.csv || { echo "PARITY FAIL"; exit 3; }; echo "PARITY_1.1 PASS $a"
done
python3 -B $M/ablation5.py math_shepherd $O/bank_math_shepherd_digitfix/FEATURES.npz $O/full_math_shepherd/prmbench_telemetry.pkl $E
cmp $E/ABL5_RESULTS_math_shepherd.csv $O/ablation5_digitfix_v1/ABL5_RESULTS_math_shepherd.csv || { echo "PARITY FAIL"; exit 3; }; echo "PARITY_1.1 PASS math_shepherd(digitfix)"
for f in BOOTSTRAP DIAG; do for a in qwen_prm_7b $PRMS_PB; do
  ext=csv; [ $f = DIAG ] && ext=json
  cmp -s $E/ABL5_${f}_$a.$ext $O/ablation5_v1/ABL5_${f}_$a.$ext && echo "INFO $f $a identical" || echo "INFO $f $a DIFFERS"
done; done
md5sum $E/* > $J/MD5_ablation5_export_v1.txt
echo JOB_A_DONE
