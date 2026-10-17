set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; X=$SH/results/lsml_external_generalization_v1
C1=$SH/code_thinkfix_rescore_v1; C2=$SH/code_thinkfix_rescore_v2
test ! -e $C2; cp -a $C1 $C2
tar xzf $SH/fix_src.tgz -C $C2; rm $SH/fix_src.tgz
echo '{"source": "code_thinkfix_rescore_v1 + first_token_mask fix for already-empty steps (2026-10-11)"}' > $C2/SYNC_COMMIT.json
D=$X/socratic_qwen3_8b_full_thinkfix_153589e_mask1
test ! -e ${D}_superseded_emptystep
mv $D ${D}_superseded_emptystep
cd $C2
T=$(sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 --cpus-per-task=1 --mem=4G --time=00:10:00 --job-name=masktest2 cluster/native_job.sbatch tests/test_first_token_mask.py 2>/dev/null)
B=$(sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 --dependency=afterok:$T --cpus-per-task=2 --mem=32G --time=02:00:00 --job-name=socratic_mask1_v2 cluster/native_job.sbatch scripts/rebuild_socratic_mask1.py $X 2>/dev/null)
echo "test=$T rebuild=$B"
