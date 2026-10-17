set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$SH/results
MK=$SH/code_thinkfix_rescore_v1; BK=$SH/code_thinkfix_bank_553de79; OUT=$R/thinkfix_rescore_v1
for d in source_bank_old_rebuild source_bank_thinkfix source_bank_thinkfix_mask1; do
  [ -d $OUT/$d ] && rmdir $OUT/$d 2>/dev/null && echo "removed empty $d" || true
done
for d in prmbench_qwen3_8b_telemetry_thinkfix_mask1 pb_qwen3_8b_thinkfix_mask1 pb_qwen3_4b_thinkfix_mask1; do
  test ! -e $R/$d || { echo "EXISTS $R/$d"; ls -la $R/$d; exit 1; }
done
SB="sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940"
cd $MK
J_MASK=$($SB --mem=64G --cpus-per-task=2 --job-name=thinkfix_mask1 cluster/native_job.sbatch scripts/build_mask1_cells.py --results $R)
cd $BK
J_SP=$($SB --mem=48G --cpus-per-task=2 --time=01:00:00 --job-name=step_spans cluster/native_job.sbatch \
  scripts/step_token_lengths.py --results $R --out $OUT/STEP_TOKEN_SPANS_OLD.npz)
J_OLD=$($SB --mem=96G --cpus-per-task=16 --job-name=bank_old_parity cluster/native_job.sbatch scripts/rebuild_source_bank_v4.py \
  --variant old --results $R --out $OUT/source_bank_old_rebuild --parity $BK/reference/FEATURES.npz --workers 16)
J_TF=$($SB --mem=96G --cpus-per-task=16 --job-name=bank_thinkfix cluster/native_job.sbatch scripts/rebuild_source_bank_v4.py \
  --variant thinkfix --results $R --out $OUT/source_bank_thinkfix --workers 16)
J_M1=$($SB --mem=96G --cpus-per-task=16 --dependency=afterok:$J_MASK --job-name=bank_thinkfix_mask1 cluster/native_job.sbatch \
  scripts/rebuild_source_bank_v4.py --variant thinkfix_mask1 --results $R --out $OUT/source_bank_thinkfix_mask1 --workers 16)
echo "mask=$J_MASK spans=$J_SP old=$J_OLD thinkfix=$J_TF mask1=$J_M1"
