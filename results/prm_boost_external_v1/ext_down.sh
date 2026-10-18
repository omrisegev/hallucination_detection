set -uo pipefail
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; O=$S/results/prm_boost_external_v1; M=$S/eval_mirror_prm_gen_v1
for a in qwen_prm pure_prm rlhflow_prm math_shepherd qwen_prm_72b; do
  d=$O/full_${a}_socratic
  (cd $S/code_thinkfix_bank_553de79 && python3 -B scripts/prm_boost_build_bank_external.py --raw $d/external_telemetry.pkl --mask-code $S/code_thinkfix_rescore_v1 --names-from $S/results/prm_boost_v1/bank_prm/FEATURES.npz --out $d/bank_v2 --workers 8) || exit 1
  (cd $M && python3 -B $M/prm_boost_score_external.py $d/bank_v2 $O/scores_socratic/SCORES_${a}.npz) || exit 1
done
