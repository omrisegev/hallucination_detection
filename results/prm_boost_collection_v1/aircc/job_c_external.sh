#!/bin/bash
# Stage 1.3 + 1.4: Socratic and Hard2Verify, all arms + sign-safe, 5 PRMs; Math-Shepherd Hard2Verify bank rebuilt
# with Mistral-7B-v0.1 digit ids. Parity on the overlapping arms is checked at the end (exit 1 on any required miss).
#SBATCH -J pbc_c_external
#SBATCH -p cpu
#SBATCH -A cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --mem=96G
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -o pbc_c_%j.out
set -euo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_collection_v1/common.sh
for d in $X/collect_socratic_v1 $X/collect_socratic_signabs_v1 $X/collect_h2v_v1 $X/collect_h2v_signabs_v1 $X/full_math_shepherd_hard2verify/bank_v3; do
  test ! -e $d || { echo "refusing: $d exists"; exit 2; }; done
mkdir -p $X/collect_socratic_v1 $X/collect_socratic_signabs_v1
echo "[1.3] Socratic"
for a in qwen_prm qwen_prm_72b pure_prm rlhflow_prm math_shepherd; do
  bk=bank_v2; [ $a = math_shepherd ] && bk=bank_v3
  python3 -B $M/results/prm_boost_ablation5_v1/ablation5_socratic.py $X/full_${a}_socratic/$bk $X/collect_socratic_v1/SCORES_$a.npz
  python3 -B $SS/run_socratic.py $M abs $X/full_${a}_socratic/$bk $X/collect_socratic_signabs_v1/SCORES_$a.npz
done
python3 -B $M/results/sml_weight_estimator_v1/socratic_chain_flip/diag_chain_socratic.py --repo $M \
  --bank $X/full_math_shepherd_socratic/bank_v3 --out $X/collect_socratic_v1/DIAG_ms_v3.json
echo "[1.4] Hard2Verify: rebuild Math-Shepherd bank_v3 (Mistral digit ids)"
(cd $C && python3 -B scripts/prm_boost_build_bank_external_v2.py --raw $X/full_math_shepherd_hard2verify/external_telemetry.pkl \
  --mask-code $S/code_thinkfix_rescore_v1 --names-from $R/prm_boost_v1/bank_prm/FEATURES.npz --tokenizer $TMS \
  --out $X/full_math_shepherd_hard2verify/bank_v3 --workers 16)
mkdir -p $X/collect_h2v_v1 $X/collect_h2v_signabs_v1
for a in qwen_prm qwen_prm_72b pure_prm rlhflow_prm math_shepherd; do
  bk=bank_v2; [ $a = math_shepherd ] && bk=bank_v3
  python3 -B $M/results/prm_boost_ablation5_v1/ablation5_socratic.py $X/full_${a}_hard2verify/$bk $X/collect_h2v_v1/SCORES_$a.npz
  python3 -B $SS/run_socratic.py $M abs $X/full_${a}_hard2verify/$bk $X/collect_h2v_signabs_v1/SCORES_$a.npz
done
echo "[parity] overlapping arms vs frozen scores_socratic / scores_hard2verify (4 unchanged-bank PRMs gate; Math-Shepherd reported)"
P=$J/parity; mkdir -p $P; fail=0
REQS="score__PRM score__HALF_N16 score__PRMG_N16"; REQH="$REQS score__CHAIN_N16"; AL="score__PRMG_N16=score__LSML_N16_PRMG"
for a in qwen_prm qwen_prm_72b pure_prm rlhflow_prm; do
  python3 -B $J/parity_npz.py $X/collect_socratic_v1/SCORES_$a.npz $X/scores_socratic/SCORES_$a.npz --require $REQS --alias $AL --out $P/socratic_$a.json || fail=1
  python3 -B $J/parity_npz.py $X/collect_h2v_v1/SCORES_$a.npz $X/scores_hard2verify/SCORES_$a.npz --require $REQH --alias $AL --out $P/h2v_$a.json || fail=1
done
for b in socratic hard2verify; do s=socratic; [ $b = hard2verify ] && s=h2v
  python3 -B $J/parity_npz.py $X/collect_${s}_v1/SCORES_math_shepherd.npz $X/scores_$b/SCORES_math_shepherd.npz --alias $AL --out $P/${s}_math_shepherd_INFO.json || true
done
md5sum $X/collect_socratic_v1/* $X/collect_socratic_signabs_v1/* $X/collect_h2v_v1/* $X/collect_h2v_signabs_v1/* $X/full_math_shepherd_hard2verify/bank_v3/* > $J/MD5_external.txt
[ $fail = 0 ] && echo JOB_C_DONE PARITY_EXTERNAL PASS || { echo PARITY_EXTERNAL FAIL; exit 1; }
