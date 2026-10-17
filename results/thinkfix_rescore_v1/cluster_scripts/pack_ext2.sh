set -e
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; M='C:\Users\omris\TAU\hallucination_detection'
T=/tmp/ext2pack_$$; mkdir -p $T
for V in thinkfix thinkfix_mask1; do
  R="$SH/thinkfix_ext_$V/$M"; cd "$R"
  find results/monday_external_panel_v1 results/monday_cont_baselines_v1 results/monday_nested_banks_v2 results/family_tail_external_v1 \
       .worktrees/ssl-pseudolabel-residual-v1/results/external_banks_v4 \
       -maxdepth 3 -type f \( -name '*.csv' -o -name '*.json' -o -name '*THINKFIX_DIFF.txt' \) \
       ! -path '*/shard_000/*' ! -path '*/source/*' ! -path '*/source_parity/*' ! -path '*/source_full_v1/*' ! -name 'FIT_DIAGNOSTICS.json' > /tmp/l_$$
  mkdir -p $T/$V; tar cf - -T /tmp/l_$$ | tar xf - -C $T/$V
  wc -l < /tmp/l_$$
done
cd $T && tar czf $SH/ext2_out.tgz . && rm -rf $T
ls -la $SH/ext2_out.tgz
