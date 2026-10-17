set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1
SB="sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940"
EV='results/lsml_external_generalization_v1/evaluation'
IN='scratch/external_generalization_private/inputs'
M='C:\Users\omris\TAU\hallucination_detection'
for pair in thinkfix: thinkfix_mask1:280814; do
  V=${pair%%:*}; D=${pair##*:}; WD=$SH/thinkfix_ext_$V; cd $WD
  DEP=""; [ -n "$D" ] && DEP="--dependency=afterok:$D"
  A=$($SB $DEP --cpus-per-task=16 --mem=64G --time=06:00:00 --job-name=ext_score_$V downstream.sbatch \
      scripts/run_external_local_cpu.py --workers 16 2>/dev/null)
  B=$($SB --dependency=afterok:$A --cpus-per-task=4 --mem=32G --time=06:00:00 --job-name=ext_eval_$V downstream.sbatch \
      scripts/evaluate_external_locked_scores.py --root "$WD/$M/$EV" --inputs "$WD/$M/$IN" --draws 100000 2>/dev/null)
  echo "$V score=$A eval=$B"
done
