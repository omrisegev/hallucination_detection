set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; WD=$SH/thinkfix_downstream; OUT=$SH/results/thinkfix_rescore_v1
cd $WD
for pair in old:280525 thinkfix:280526 thinkfix_mask1:280533; do
  V=${pair%%:*}; D=${pair##*:}
  j=$(sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 --dependency=afterok:$D --job-name=lr_$V \
      downstream.sbatch results/thinkfix_rescore_v1/run_lr_rescore.py --variant $V --bank $WD/banks/$V --root $OUT 2>/dev/null)
  echo "lr_$V=$j"
done
