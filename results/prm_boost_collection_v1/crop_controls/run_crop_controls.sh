#!/bin/bash
# CROP controls C1-C7, 5 PRMs sequentially (CPU). Usage: sbatch run_crop_controls.sh  (CROP_SMOKE=1 for the preflight smoke)
#SBATCH -J pbc_crop_ctl
#SBATCH -p cpu
#SBATCH -A cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o pbc_crop_ctl_%j.out
set -euo pipefail
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; J=$R/prm_boost_collection_v1/crop_controls
set +u; . $S/pyenv_native_312_20261011/activate.sh; set -u
EXP=$O/ablation5_export_v1
JL=$S/hf_cache/hub/datasets--hitsmy--PRMBench_Preview/snapshots/5cc7683d0ae5797f84d7aeac0607966f277c39e1/prmbench_preview.jsonl
BASE=${OUT:-$R/prm_boost_collection_v1/crop_controls_v3}
cd $S/eval_mirror_prm_gen_v1
PRMS=${PRMS:-"qwen_prm_7b qwen_prm_72b pure_prm rlhflow_prm math_shepherd"}
for a in $PRMS; do
  OUT=$BASE/$a; test ! -e $OUT || { echo "refusing: $OUT exists"; exit 2; }; mkdir -p $OUT
  case $a in
    qwen_prm_7b) F=$R/prm_boost_v1/bank_prm/FEATURES.npz; T=$R/prm_boost_v1/full_prm/prmbench_telemetry.pkl;;
    math_shepherd) F=$O/bank_math_shepherd_digitfix/FEATURES.npz; T=$O/full_math_shepherd/prmbench_telemetry.pkl;;
    *) F=$O/bank_$a/FEATURES.npz; T=$O/full_$a/prmbench_telemetry.pkl;;
  esac
  python3 -B $J/crop_controls.py $a $F $T $EXP $JL $OUT
  md5sum $OUT/* > $OUT/MD5.txt
done
echo CROP_CONTROLS_DONE
