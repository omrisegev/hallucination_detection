set -e
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; M='C:\Users\omris\TAU\hallucination_detection'
T=/tmp/extpack_$$; mkdir -p $T
for V in thinkfix thinkfix_mask1; do
  E="$SH/thinkfix_ext_$V/$M/results/lsml_external_generalization_v1/evaluation"
  mkdir -p $T/$V/lsml_locked
  for f in METRICS.json CONTRASTS.json DISJOINT_CONTRASTS.json EVALUATION_PROVENANCE.json ALL_CELLS_SEALED.json ANALYSIS_FREEZE.json CPU_EXECUTION.json; do cp "$E/$f" $T/$V/lsml_locked/; done
  for c in hard2verify_qwen3_8b socratic_qwen3_8b; do mkdir -p $T/$V/lsml_locked/$c; cp "$E/$c/SEAL.json" "$E/$c/METRICS.json" $T/$V/lsml_locked/$c/; done
  cp "$SH/thinkfix_ext_$V/$M/scripts/"*THINKFIX_DIFF.txt $T/$V/lsml_locked/
done
cd $T && tar czf $SH/ext_metrics.tgz . && rm -rf $T
ls -la $SH/ext_metrics.tgz
