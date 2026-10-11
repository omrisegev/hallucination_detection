set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; M='C:\Users\omris\TAU\hallucination_detection'
SB="sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940"
SSLS='.worktrees/ssl-pseudolabel-residual-v1/scripts/extract_external_banks_v4.py'
for V in thinkfix thinkfix_mask1; do
  WD=$SH/thinkfix_ext_$V; R="$WD/$M"
  tar xzf $SH/e2b_$V.tgz -C "$R"; rm $SH/e2b_$V.tgz
  chmod -R go-rwx "$R/scratch"
  python3 $SH/hash_family_freeze.py "$R"
  cd $WD
  C=$($SB --cpus-per-task=16 --mem=64G --time=06:00:00 --job-name=ext2_family_$V downstream.sbatch scripts/run_family_external.py --workers 16 2>/dev/null)
  D1=$($SB --dependency=afterok:$C --cpus-per-task=8 --mem=32G --time=02:00:00 --job-name=ext2_tok_$V downstream.sbatch $SSLS tokenizers --workers 8 2>/dev/null)
  D2=$($SB --dependency=afterok:$D1 --cpus-per-task=16 --mem=64G --time=06:00:00 --job-name=ext2_banks_$V downstream.sbatch $SSLS external --workers 16 2>/dev/null)
  E=$($SB --dependency=afterok:$D2 --cpus-per-task=8 --mem=64G --time=08:00:00 --job-name=ext2_panel_$V downstream.sbatch results/monday_external_panel_v1/score.py 2>/dev/null)
  echo "$V family=$C tokenizers=$D1 banks=$D2 panel_score=$E"
done
rm $SH/hash_family_freeze.py
