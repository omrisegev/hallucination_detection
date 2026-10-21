#!/bin/bash
# CROP controls v3 (has-error = non-empty error_steps) for 5 PRMs; one linear afterok chain. Section 3 already done (290578).
set -euo pipefail; export SLURM_CONF_SERVER=controller-primary
C=/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_collection_v1
cd $C/crop_controls; prev=""; ids=""
for a in qwen_prm_7b qwen_prm_72b pure_prm rlhflow_prm math_shepherd; do
  dep=""; [ -n "$prev" ] && dep="--dependency=afterok:$prev"
  prev=$(sbatch --parsable $dep -J pbc_crop3_$a -t 03:00:00 --export=ALL,PRMS=$a run_crop_controls.sh); ids="$ids $a=$prev"
done
echo "JOBS$ids" | tee -a $C/crop_controls/JOB_IDS.txt
