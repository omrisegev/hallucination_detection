set -e
export SLURM_CONF_SERVER=controller-primary
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; M='C:\Users\omris\TAU\hallucination_detection'
EV=results/lsml_external_generalization_v1/evaluation; IN=scratch/external_generalization_private/inputs
scancel 280818 || true
for V in thinkfix thinkfix_mask1; do
  R="$SH/thinkfix_ext_$V/$M"
  grep -q '"A2"' "$R/$EV/ANALYSIS_FREEZE.json" || python3 $SH/amend_eval.py "$R"
  echo "$V A2: $(grep -c '"A2"' "$R/$EV/ANALYSIS_FREEZE.json") $(grep -c 'thinkfix A2' "$R/scripts/evaluate_external_locked_scores.py")"
done
SB="sbatch --parsable -p cpu -A cycle3_tau_averbuch_prj --qos=owner_940 --cpus-per-task=4 --mem=32G --time=06:00:00"
cd $SH/thinkfix_ext_thinkfix
a=$($SB --job-name=ext_eval_thinkfix downstream.sbatch scripts/evaluate_external_locked_scores.py --root "$SH/thinkfix_ext_thinkfix/$M/$EV" --inputs "$SH/thinkfix_ext_thinkfix/$M/$IN" --draws 100000 2>/dev/null)
cd $SH/thinkfix_ext_thinkfix_mask1
b=$($SB --dependency=afterok:280817 --job-name=ext_eval_thinkfix_mask1 downstream.sbatch scripts/evaluate_external_locked_scores.py --root "$SH/thinkfix_ext_thinkfix_mask1/$M/$EV" --inputs "$SH/thinkfix_ext_thinkfix_mask1/$M/$IN" --draws 100000 2>/dev/null)
echo "eval_thinkfix=$a eval_mask1=$b"
