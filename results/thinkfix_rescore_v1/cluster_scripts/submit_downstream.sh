set -e
# usage: bash submit_downstream.sh <variant> [dependency jobid]
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$SH/results; WD=$SH/thinkfix_downstream; OUT=$R/thinkfix_rescore_v1
V=$1; DEP=${2:-}
case $V in old) B=source_bank_old_rebuild;; thinkfix) B=source_bank_thinkfix;; thinkfix_mask1) B=source_bank_thinkfix_mask1;; *) exit 2;; esac
mkdir -p $WD/banks/$V/source
ln -sfn $OUT/$B/FEATURES.npz $WD/banks/$V/source/FEATURES.npz
ln -sfn $OUT/$B/MANIFEST.json $WD/banks/$V/MANIFEST.json
test ! -e $OUT/$V/nested_banks
cd $WD
D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"
sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 $D --job-name=rescore_$V downstream.sbatch \
  results/thinkfix_rescore_v1/run_prmbench_rescore.py --variant $V --bank $WD/banks/$V --root $OUT \
  --spans $OUT/STEP_TOKEN_SPANS_OLD.npz --stage both
