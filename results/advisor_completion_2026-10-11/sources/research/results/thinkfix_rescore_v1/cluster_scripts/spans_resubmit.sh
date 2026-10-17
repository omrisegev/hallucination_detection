set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$SH/results; BK=$SH/code_thinkfix_bank_553de79; OUT=$R/thinkfix_rescore_v1
WD=$SH/thinkfix_downstream
scancel 280527 || true
cd $OUT && ln -sfn STEP_TOKEN_SPANS_OLD.npz STEP_TOKEN_SPANS_OLD.npz 2>/dev/null || true
SB="sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940"
cd $BK
A=$($SB --mem=48G --cpus-per-task=2 --time=01:00:00 --job-name=spans_thinkfix cluster/native_job.sbatch \
  scripts/step_token_lengths.py --variant thinkfix --results $R --out $OUT/STEP_TOKEN_SPANS_THINKFIX.npz)
B=$($SB --mem=48G --cpus-per-task=2 --time=01:00:00 --job-name=spans_mask1 cluster/native_job.sbatch \
  scripts/step_token_lengths.py --variant thinkfix_mask1 --results $R --out $OUT/STEP_TOKEN_SPANS_THINKFIX_MASK1.npz)
C=$($SB --dependency=afterok:$A:$B --mem=8G --cpus-per-task=1 --time=00:10:00 --job-name=spans_check cluster/native_job.sbatch \
  scripts/check_step_spans.py $OUT)
# mask1 downstream: its OWN shifted spans; waits for its bank (280519) and the span check
mkdir -p $WD/banks/thinkfix_mask1/source
ln -sfn $OUT/source_bank_thinkfix_mask1/FEATURES.npz $WD/banks/thinkfix_mask1/source/FEATURES.npz
ln -sfn $OUT/source_bank_thinkfix_mask1/MANIFEST.json $WD/banks/thinkfix_mask1/MANIFEST.json
cd $WD
D=$($SB --dependency=afterok:280519:$C --job-name=rescore_thinkfix_mask1 downstream.sbatch \
  results/thinkfix_rescore_v1/run_prmbench_rescore.py --variant thinkfix_mask1 --bank $WD/banks/thinkfix_mask1 --root $OUT \
  --spans $OUT/STEP_TOKEN_SPANS_THINKFIX_MASK1.npz --stage both)
echo "spans_thinkfix=$A spans_mask1=$B check=$C rescore_mask1=$D"
