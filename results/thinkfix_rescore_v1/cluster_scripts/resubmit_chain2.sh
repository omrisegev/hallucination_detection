set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; M='C:\Users\omris\TAU\hallucination_detection'
SSLS='.worktrees/ssl-pseudolabel-residual-v1/scripts/extract_external_banks_v4.py'
SB="sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940"
for V in thinkfix thinkfix_mask1; do
  R="$SH/thinkfix_ext_$V/$M"; L="$R/.worktrees/ssl-pseudolabel-residual-v1/results/family_tail_transfer_v1"
  mkdir -p "$L"; cp $SH/ssl_TRANSFER_LOCK_V1.json "$L/TRANSFER_LOCK_V1.json"
  cd $SH/thinkfix_ext_$V
  D2=$($SB --cpus-per-task=16 --mem=64G --time=06:00:00 --job-name=ext2_banks_$V downstream.sbatch $SSLS external --workers 16 2>/dev/null)
  E=$($SB --dependency=afterok:$D2 --cpus-per-task=8 --mem=64G --time=08:00:00 --job-name=ext2_panel_$V downstream.sbatch results/monday_external_panel_v1/score.py 2>/dev/null)
  S8="$SB --cpus-per-task=8 --mem=64G --time=08:00:00"
  G1=$($S8 --dependency=afterok:$E --job-name=ext2_cont_score_$V downstream.sbatch results/monday_cont_baselines_v1/score_ext.py 2>/dev/null)
  H1=$($S8 --dependency=afterok:$G1 --job-name=ext2_nested_score_$V downstream.sbatch results/monday_nested_banks_v2/score_ext.py 2>/dev/null)
  F=$($S8 --dependency=afterok:$H1 --job-name=ext2_panel_eval_$V downstream.sbatch results/monday_external_panel_v1/evaluate.py 2>/dev/null)
  G2=$($S8 --dependency=afterok:$F --job-name=ext2_cont_eval_$V downstream.sbatch results/monday_cont_baselines_v1/evaluate_ext.py 2>/dev/null)
  H2=$($S8 --dependency=afterok:$G2 --job-name=ext2_nested_eval_$V downstream.sbatch results/monday_nested_banks_v2/evaluate_ext.py 2>/dev/null)
  H3=$($S8 --dependency=afterok:$H2 --job-name=ext2_nested_oriented_$V downstream.sbatch results/monday_nested_banks_v2/external_oriented.py 2>/dev/null)
  echo "$V banks=$D2 panel_score=$E cont_score=$G1 nested_score=$H1 panel_eval=$F cont_eval=$G2 nested_eval=$H2 oriented=$H3"
done
rm $SH/ssl_TRANSFER_LOCK_V1.json
