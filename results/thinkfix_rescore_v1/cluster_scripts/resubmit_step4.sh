set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; M='C:\Users\omris\TAU\hallucination_detection'; WD=$SH/thinkfix_step4; R="$WD/$M"
mkdir -p "$R/.worktrees/cumulative-vote-fusion-v2/spectral_utils"
mv $SH/cvf_fusion_utils.py "$R/.worktrees/cumulative-vote-fusion-v2/spectral_utils/fusion_utils.py"
OUTD=$SH/results/thinkfix_rescore_v1/step4_pb_reference
[ -d $OUTD ] && rmdir $OUTD 2>/dev/null || true
cd $WD
j=$(sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 --cpus-per-task=4 --mem=32G --time=04:00:00 --job-name=step4_pbref \
  --export=ALL,HD_MAIN_CHECKOUT="$R" downstream.sbatch results/thinkfix_rescore_v1/step4_pb_reference.py \
  "$R/.worktrees/self-generated-step-labels-v1" $SH/results/thinkfix_rescore_v1 \
  "$R/.worktrees/ssl-pseudolabel-residual-v1/results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz" $OUTD 2>/dev/null)
echo step4=$j
