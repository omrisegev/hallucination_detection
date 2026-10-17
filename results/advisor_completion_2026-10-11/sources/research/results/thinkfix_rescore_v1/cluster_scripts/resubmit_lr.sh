set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; WD=$SH/thinkfix_downstream; OUT=$SH/results/thinkfix_rescore_v1
M='C:\Users\omris\TAU\hallucination_detection'
cp $SH/run_lr_rescore.py "$WD/$M/results/thinkfix_rescore_v1/run_lr_rescore.py"; rm $SH/run_lr_rescore.py
cd $WD
for V in old thinkfix thinkfix_mask1; do
  j=$(sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 --job-name=lr_$V \
      downstream.sbatch results/thinkfix_rescore_v1/run_lr_rescore.py --variant $V --bank $WD/banks/$V --root $OUT 2>/dev/null)
  echo "lr_$V=$j"
done
# Drive copy of the old/thinkfix PRMScore inputs (not in git)
RC="$SH/bin/rclone --config $SH/.config/rclone/rclone.conf"
nohup bash -c "for v in old thinkfix; do $RC copy $OUT/\$v/prmscore_inputs gdrive:hallucination_detection/cluster_results/thinkfix_rescore_v1/\$v/prmscore_inputs --transfers 4 -v; done; echo PRMSCORE_UPLOAD_DONE" > $SH/logs/upload_prmscore_inputs_20261011.log 2>&1 &
echo upload_pid=$!
tail -3 $SH/logs/upload_thinkfix_20261010.log
