set -uo pipefail
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; M=$S/eval_mirror_prm_gen_v1
cd $M; X=$O/controls_v1
python3 -B $M/lsml_controls.py qwen_prm_7b $R/prm_boost_v1/bank_prm/FEATURES.npz $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl $X
for a in qwen_prm_72b pure_prm rlhflow_prm math_shepherd; do
  python3 -B $M/lsml_controls.py $a $O/bank_$a/FEATURES.npz $O/full_$a/prmbench_telemetry.pkl $X
done
