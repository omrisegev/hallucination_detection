set -e
# Two external working copies: thinkfix_mask1 (primary) and thinkfix (sensitivity). Records are linked, not copied.
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; X=$SH/results/lsml_external_generalization_v1
M='C:\Users\omris\TAU\hallucination_detection'
for V in thinkfix_mask1 thinkfix; do
  WD=$SH/thinkfix_ext_$V
  test ! -e $WD
  mkdir -p "$WD/$M"; tar xzf $SH/ext_mirror.tgz -C "$WD/$M"
  chmod -R go-rwx "$WD/$M/scratch"
  A="$WD/$M/scratch/external_generalization_private/evaluation_archives"
  for c in hard2verify_qwen3_8b socratic_qwen3_8b; do
    if [ $V = thinkfix_mask1 ]; then src=$X/${c}_full_thinkfix_153589e_mask1; pre=SOURCE_; rec=$src/records
    else src=$X/${c}_full_thinkfix_153589e; pre=; rec=$src/records; fi
    mkdir -p "$A/$c"; ln -s $rec "$A/$c/records"
    for f in TOKENIZATION.json ALIGNMENT_GATE.json TIMING.json; do ln -s $src/$pre$f "$A/$c/$f"; done
    echo "$V $c: $(ls $rec | grep -c record.json) records"
  done
  cp $SH/thinkfix_downstream/downstream.sbatch $WD/downstream.sbatch
done
rm $SH/ext_mirror.tgz
