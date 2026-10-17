set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; WD=$SH/thinkfix_downstream; M='C:\Users\omris\TAU\hallucination_detection'
cd $WD
j=$(sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 --cpus-per-task=1 --mem=8G --time=00:15:00 --job-name=compare \
  ${DEP:+--dependency=afterany:$DEP} downstream.sbatch results/thinkfix_rescore_v1/compare.py --frozen-root "$WD/$M" --root $SH/results/thinkfix_rescore_v1 2>/dev/null)
echo $j
sleep ${WAIT:-60}
N=30 bash $SH/logs.sh $j
