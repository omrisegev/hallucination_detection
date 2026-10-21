#!/bin/bash
# One job per PRM, linear afterok chain; each PRM writes crop_controls_v1/<prm>/ (new dirs only).
set -euo pipefail; export SLURM_CONF_SERVER=controller-primary
cd /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_collection_v1/crop_controls
prev=""; ids=""
for a in qwen_prm_7b qwen_prm_72b pure_prm rlhflow_prm math_shepherd; do
  dep=""; [ -n "$prev" ] && dep="--dependency=afterok:$prev"
  prev=$(sbatch --parsable $dep -J pbc_crop_$a -t 03:00:00 --export=ALL,PRMS=$a run_crop_controls.sh); ids="$ids $a=$prev"
done
echo "JOBS$ids" | tee -a JOB_IDS.txt
