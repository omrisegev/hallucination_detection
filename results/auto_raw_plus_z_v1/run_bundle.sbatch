#!/bin/bash
# CPU job: rerun the refactored CAT34 script (must reproduce job 304515) and export the Drive bundle.
# sbatch -A cycle4_tau_averbuch_spot_prj -p cpu run_bundle.sbatch
#SBATCH --job-name=auto_raw_z_bundle
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/auto_raw_plus_z_v1/%x_%j.out
set -uo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/pyenv_native_312_20261011/activate.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; SW=$O/bank_sweep_v1
D=$S/auto_raw_plus_z_v1; BU=$D/bundle
SE=$S/eval_mirror_prm_gen_v1/.worktrees/readout-quickest-detection-v1/results/step_evidence_v1
cd $D
feat() { [ "$1" = qwen_prm_7b ] && echo $R/prm_boost_v1/bank_prm || echo $O/bank_$1; }
tele() { [ "$1" = qwen_prm_7b ] && echo $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl || echo $O/full_$1/prmbench_telemetry.pkl; }
for a in qwen_prm_7b qwen_prm_72b pure_prm rlhflow_prm; do
  python3 -B $D/auto_raw_plus_z.py $a $(feat $a)/FEATURES.npz $(tele $a) $SW/SWEEP_RESULTS_$a.csv $D/out_v2 || exit 3
  python3 -B $D/export_bundle.py $a $(feat $a)/FEATURES.npz $(tele $a) $BU/inputs || exit 4
  mkdir -p $BU/raw_features/$a
  cp -p $(feat $a)/FEATURES.npz $(feat $a)/MANIFEST.json $BU/raw_features/$a/
done
mkdir -p $BU/labels; cp -p $SE/OOF_STEP_SCORES.npz $SE/OOF_ANSWERS.csv $BU/labels/
for f in RESULTS BOOTSTRAP DIAG; do
  for a in qwen_prm_7b qwen_prm_72b pure_prm rlhflow_prm; do
    cmp -s <(cut -d, -f1- $D/out/${f}_$a.csv) <(cut -d, -f1- $D/out_v2/${f}_$a.csv) && echo "same $f $a" || echo "DIFF $f $a"
  done
done
