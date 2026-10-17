set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; M='C:\Users\omris\TAU\hallucination_detection'
WD=$SH/thinkfix_step4; test ! -e $WD
mkdir -p "$WD/$M"
# reuse the chain-2 mirror's code (main spectral_utils, ssl spectral_utils) by copying, then add the step-4 bundle
cp -a "$SH/thinkfix_ext_thinkfix/$M/spectral_utils" "$WD/$M/"
mkdir -p "$WD/$M/.worktrees/ssl-pseudolabel-residual-v1"
cp -a "$SH/thinkfix_ext_thinkfix/$M/.worktrees/ssl-pseudolabel-residual-v1/spectral_utils" "$WD/$M/.worktrees/ssl-pseudolabel-residual-v1/"
tar xzf $SH/s4.tgz -C "$WD/$M"; rm $SH/s4.tgz
cp $SH/thinkfix_downstream/downstream.sbatch $WD/
cd $WD
R="$WD/$M"
j=$(sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 --cpus-per-task=4 --mem=32G --time=04:00:00 --job-name=step4_pbref \
  --export=ALL,HD_MAIN_CHECKOUT="$R" downstream.sbatch results/thinkfix_rescore_v1/step4_pb_reference.py \
  "$R/.worktrees/self-generated-step-labels-v1" $SH/results/thinkfix_rescore_v1 \
  "$R/.worktrees/ssl-pseudolabel-residual-v1/results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz" $SH/results/thinkfix_rescore_v1/step4_pb_reference 2>/dev/null)
echo step4=$j
