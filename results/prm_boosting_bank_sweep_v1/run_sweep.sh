set -uo pipefail
# CPU job, same inputs and mirror as results/prm_boosting_controls_v1/run_controls.sh (AIRCC 287777).
# Before submitting, copy lsml_bank_sweep.py into $M (under sbatch, $0 is a spooled copy, so no relative paths here).
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; M=$S/eval_mirror_prm_gen_v1
test -f $M/lsml_bank_sweep.py || { echo "missing $M/lsml_bank_sweep.py"; exit 2; }
cd $M; X=$O/bank_sweep_v1
python3 -B $M/lsml_bank_sweep.py qwen_prm_7b $R/prm_boost_v1/bank_prm/FEATURES.npz $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl $X
for a in qwen_prm_72b pure_prm rlhflow_prm math_shepherd; do
  python3 -B $M/lsml_bank_sweep.py $a $O/bank_$a/FEATURES.npz $O/full_$a/prmbench_telemetry.pkl $X
done
