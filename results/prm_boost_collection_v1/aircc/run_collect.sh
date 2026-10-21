#!/bin/bash
# DRAFT, written on the Mac without AIRCC access. Check every path, the account/QoS and the tokenizer path during
# /preflight before submitting. Plan: docs/experiments/PRM_BOOSTING_COLLECTION_PLAN_2026-10-21.md, stage 1.
# CPU only, label-free scoring, every output in a NEW directory. Split into linearly chained jobs if it exceeds a wall.
#SBATCH -J prmboost_collect
#SBATCH -p cpu
#SBATCH -A cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --mem=96G
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -o prmboost_collect_%j.out
set -euo pipefail
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; M=$S/eval_mirror_prm_gen_v1
X=$R/prm_boost_external_v1; C=$S/code_thinkfix_bank_553de79; H=$S/hf_cache/hub
TMS=$H/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da/tokenizer.json
SS=$M/results/sml_weight_estimator_v1/socratic_sign      # nonneg_sml.py, run_prmbench.py, run_socratic.py copied here
export ABL5_EXPORT_SCORES=1
cd $M

echo "[1.0] parity: qwen_prm_7b with export must equal ablation5_v1"
python3 -B $M/ablation5.py qwen_prm_7b $R/prm_boost_v1/bank_prm/FEATURES.npz $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl $O/ablation5_export_v1
cmp $O/ablation5_export_v1/ABL5_RESULTS_qwen_prm_7b.csv $O/ablation5_v1/ABL5_RESULTS_qwen_prm_7b.csv

echo "[1.1] PRMBench export, remaining 4 PRMs (math_shepherd = digit-fixed bank)"
for a in qwen_prm_72b pure_prm rlhflow_prm; do
  python3 -B $M/ablation5.py $a $O/bank_$a/FEATURES.npz $O/full_$a/prmbench_telemetry.pkl $O/ablation5_export_v1
  cmp $O/ablation5_export_v1/ABL5_RESULTS_$a.csv $O/ablation5_v1/ABL5_RESULTS_$a.csv
done
python3 -B $M/ablation5.py math_shepherd $O/bank_math_shepherd_digitfix/FEATURES.npz $O/full_math_shepherd/prmbench_telemetry.pkl $O/ablation5_export_v1
cmp $O/ablation5_export_v1/ABL5_RESULTS_math_shepherd.csv $O/ablation5_digitfix_v1/ABL5_RESULTS_math_shepherd.csv

echo "[1.2] PRMBench sign-safe variant (|v|, no anchor flip), diagnostic"
python3 -B $SS/run_prmbench.py $M abs qwen_prm_7b $R/prm_boost_v1/bank_prm/FEATURES.npz $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl $O/ablation5_signabs_v1
for a in qwen_prm_72b pure_prm rlhflow_prm; do
  python3 -B $SS/run_prmbench.py $M abs $a $O/bank_$a/FEATURES.npz $O/full_$a/prmbench_telemetry.pkl $O/ablation5_signabs_v1
done
python3 -B $SS/run_prmbench.py $M abs math_shepherd $O/bank_math_shepherd_digitfix/FEATURES.npz $O/full_math_shepherd/prmbench_telemetry.pkl $O/ablation5_signabs_v1

echo "[1.3] Socratic, all arms + sign-safe + sign diagnosis"
mkdir -p $X/collect_socratic_v1 $X/collect_socratic_signabs_v1
for a in qwen_prm qwen_prm_72b pure_prm rlhflow_prm math_shepherd; do
  bk=bank_v2; [ $a = math_shepherd ] && bk=bank_v3
  python3 -B $M/results/prm_boost_ablation5_v1/ablation5_socratic.py $X/full_${a}_socratic/$bk $X/collect_socratic_v1/SCORES_$a.npz
  python3 -B $SS/run_socratic.py $M abs $X/full_${a}_socratic/$bk $X/collect_socratic_signabs_v1/SCORES_$a.npz
done
python3 -B $M/results/sml_weight_estimator_v1/socratic_chain_flip/diag_chain_socratic.py --repo $M \
  --bank $X/full_math_shepherd_socratic/bank_v3 --out $X/collect_socratic_v1/DIAG_ms_v3.json

echo "[1.4] Hard2Verify: rebuild Math-Shepherd with Mistral digit ids, then all arms"
(cd $C && python3 -B scripts/prm_boost_build_bank_external_v2.py --raw $X/full_math_shepherd_hard2verify/external_telemetry.pkl \
  --mask-code $S/code_thinkfix_rescore_v1 --names-from $R/prm_boost_v1/bank_prm/FEATURES.npz --tokenizer $TMS \
  --out $X/full_math_shepherd_hard2verify/bank_v3 --workers 16)
mkdir -p $X/collect_h2v_v1 $X/collect_h2v_signabs_v1
for a in qwen_prm qwen_prm_72b pure_prm rlhflow_prm math_shepherd; do
  bk=bank_v2; [ $a = math_shepherd ] && bk=bank_v3
  python3 -B $M/results/prm_boost_ablation5_v1/ablation5_socratic.py $X/full_${a}_hard2verify/$bk $X/collect_h2v_v1/SCORES_$a.npz
  python3 -B $SS/run_socratic.py $M abs $X/full_${a}_hard2verify/$bk $X/collect_h2v_signabs_v1/SCORES_$a.npz
done
echo ALL_DONE
