#!/bin/bash
set -uo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/pyenv_native_312_20261011/activate.sh
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; O=$S/results/prm_boost_external_v1; M=$S/eval_mirror_prm_gen_v1
mkdir -p $O/scores_hard2verify
for a in qwen_prm pure_prm rlhflow_prm math_shepherd qwen_prm_72b; do
  d=$O/full_${a}_hard2verify
  echo "=== Building bank_v2 for $a ==="
  (cd $S/code_thinkfix_bank_553de79 && python3 -B scripts/prm_boost_build_bank_external.py --raw $d/external_telemetry.pkl --mask-code $S/code_thinkfix_rescore_v1 --names-from $S/results/prm_boost_v1/bank_prm/FEATURES.npz --out $d/bank_v2 --workers 8) || exit 1
  echo "=== Scoring for $a with CHAIN and fusions ==="
  (cd $M && python3 -B $M/prm_boost_score_hard2verify.py $d/bank_v2 $O/scores_hard2verify/SCORES_${a}.npz) || exit 1
done
echo "=== All Hard2Verify downstream processing complete ==="
