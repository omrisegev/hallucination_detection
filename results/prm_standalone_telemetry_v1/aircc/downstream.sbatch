#!/bin/bash
# CPU, label-free scoring. After all five full_<prm> runs are COMPLETE (manifest status, 0 failed rows).
#   1. P1 parity: rebuild one BACKBONE bank with this exact command and compare it with the frozen one (must be exact).
#   2. Per PRM: standalone bank (build_bank.py, the standalone tokenizer's digit ids), then ablation5 with the
#      standalone FEATURES and the PRM's own backbone pkl (rows + rewards), exported as ABL5_*_<prm>__standalone.
#      Only the telemetry-only arms (MEAN_*, TELE_*) are reported (no fusion with the PRM, Omri 2026-10-22).
#SBATCH -J sa_downstream
#SBATCH -p cpu
#SBATCH -A cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --mem=96G
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -o /shared/cycle2_tau_averbuch_prj/omrisegev1/logs/sa_downstream_%j.out
set -euo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_standalone_telemetry_v1/code/common.sh
set +u; . $S/pyenv_native_312_20261011/activate.sh; set -u
python3 -c 'import numpy, scipy, sklearn; print("PYENV", numpy.__version__, scipy.__version__)'
K=$W/code                                   # build_bank.py, digit_token_ids.py, common.sh, parity_npz.py (HANDOFF step 2)
BASE=$R/thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz; MASK=$S/code_prm_boost_v1/mask_fixed
E=$W/abl5; test ! -e $E || { echo "refusing: $E exists"; exit 2; }
export ABL5_EXPORT_SCORES=1 DIGIT_TOKEN_IDS_PY=$K/digit_token_ids.py

echo "[P1] parity: rebuild the pure_prm BACKBONE bank, compare with the frozen $O/bank_pure_prm"
TPURE=$(snapshot Qwen/Qwen2.5-Math-7B)/tokenizer.json
(cd $C && python3 -B $K/build_bank.py --raw $O/full_pure_prm/prmbench_telemetry.pkl --base $BASE --mask-code $MASK \
   --tokenizer $TPURE --out $W/parity_bank_pure_prm --workers 16)
python3 -B $K/parity_npz.py $W/parity_bank_pure_prm/FEATURES.npz $O/bank_pure_prm/FEATURES.npz --require values offsets row_ids \
   --out $W/PARITY_P1_bank_pure_prm.json || { echo "P1 PARITY FAIL: the bank command differs from the frozen banks"; exit 3; }
echo "P1 PASS"

for p in $PRMS_SA; do
  m=${SA_MODEL[$p]}; raw=$W/full_$p/prmbench_telemetry.pkl
  python3 -c "import json,sys; m=json.load(open('$W/full_$p/manifest.json')); sys.exit(0 if m['status']=='COMPLETE' and m['n_failed_rows']==0 else 1)" \
    || { echo "$p: full run not COMPLETE with 0 failures; stopping"; exit 4; }
  echo "[bank] $p from $m"
  (cd $C && python3 -B $K/build_bank.py --raw $raw --base $BASE --mask-code $MASK \
     --tokenizer "$(snapshot "$m")/tokenizer.json" --out $W/bank_$p --workers 16)
  echo "[ablation5] $p"
  (cd $M && python3 -B $M/ablation5.py ${p}__standalone $W/bank_$p/FEATURES.npz "${PRM_PKL[$p]}" $E)
done
md5sum $E/* > $W/MD5_abl5.txt
echo SA_DOWNSTREAM_DONE
