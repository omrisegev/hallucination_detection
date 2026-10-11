set -e
# Label isolation: all label-free scoring/sealing (panel score, cont score_ext, nested score_ext) finishes before any
# evaluator that opens labels (panel evaluate, cont evaluate_ext, nested evaluate_ext, external_oriented).
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1
SB="sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 --cpus-per-task=8 --mem=64G --time=08:00:00"
for pair in thinkfix:280827 thinkfix_mask1:280831; do
  V=${pair%%:*}; E=${pair##*:}; cd $SH/thinkfix_ext_$V
  G1=$($SB --dependency=afterok:$E --job-name=ext2_cont_score_$V downstream.sbatch results/monday_cont_baselines_v1/score_ext.py 2>/dev/null)
  H1=$($SB --dependency=afterok:$G1 --job-name=ext2_nested_score_$V downstream.sbatch results/monday_nested_banks_v2/score_ext.py 2>/dev/null)
  F=$($SB --dependency=afterok:$H1 --job-name=ext2_panel_eval_$V downstream.sbatch results/monday_external_panel_v1/evaluate.py 2>/dev/null)
  G2=$($SB --dependency=afterok:$F --job-name=ext2_cont_eval_$V downstream.sbatch results/monday_cont_baselines_v1/evaluate_ext.py 2>/dev/null)
  H2=$($SB --dependency=afterok:$G2 --job-name=ext2_nested_eval_$V downstream.sbatch results/monday_nested_banks_v2/evaluate_ext.py 2>/dev/null)
  H3=$($SB --dependency=afterok:$H2 --job-name=ext2_nested_oriented_$V downstream.sbatch results/monday_nested_banks_v2/external_oriented.py 2>/dev/null)
  echo "$V cont_score=$G1 nested_score=$H1 panel_eval=$F cont_eval=$G2 nested_eval=$H2 oriented=$H3"
done
