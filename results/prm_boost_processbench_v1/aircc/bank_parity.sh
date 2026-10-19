#!/bin/bash
# Section 8 step 5 of PREREGISTRATION_HE.md (CPU, label-free): for the PRMs whose digit ids are 15..24, the v2 bank
# builder (prm_boost_build_bank_external_v2.py --tokenizer) must give exactly the old builder's bank. Two cells each:
#   socratic_<a>  v2 on the Socratic raw telemetry  vs  the existing Socratic bank_v2 (old builder, ext_down.sh)
#   pb_<a>        old and v2 builders, both on this run's ProcessBench raw telemetry (exists after HANDOFF step 6)
# It first checks the digit ids of all 5 tokenizers. All new banks go to $O/parity/ (the builders refuse an existing
# dir; a finished one is reused). Writes $O/parity/PARITY_SUMMARY.json; pb_down.sh (RUN=full) refuses to run unless
# every socratic_/pb_ pair is exact.
#   bash bank_parity.sh            (or as a CPU job: bash submit_parity.sh)
set -uo pipefail; source "$(dirname "$0")/env.sh"
X=$S/results/prm_boost_external_v1; P=$O/parity; mkdir -p $P
[ "$(sha256sum $OLD_BUILDER | cut -d' ' -f1)" = "$OLD_BUILDER_SHA256" ] || { echo "FAIL: $OLD_BUILDER is not the frozen builder" >&2; exit 1; }
install_or_cmp $O/code/build_bank_external_v2.repo.py $NEW_BUILDER || exit 1
install_or_cmp $O/code/digit_token_ids.repo.py $BANK_CODE/scripts/digit_token_ids.py || exit 1
# Section 8 step 1: every backbone's tokenizer must give the expected '0'..'9' ids (Mistral's from job 288932).
for a in $ADAPTERS; do check_digits $a "$(tok_for $a)" || exit 1; done
build() {   # <builder script> <raw pkl> <out dir> [--tokenizer <tokenizer.json>]
  [ -e $3/FEATURES.npz ] && { echo "reuse $3"; return 0; }
  (cd $BANK_CODE && python3 -B scripts/$1 --raw $2 --mask-code $MASK_CODE --names-from $NAMES_FROM --out $3 \
     --workers ${SLURM_CPUS_PER_TASK:-8} "${@:4}")
}
pairs=()
for a in ${ONLY:-$PARITY_ADAPTERS}; do
  tk=$(tok_for $a)
  s=$X/full_${a}_socratic
  build prm_boost_build_bank_external_v2.py $s/external_telemetry.pkl $P/socratic_${a}_v2 --tokenizer "$tk" || exit 1
  pairs+=(socratic_$a $s/bank_v2 $P/socratic_${a}_v2)
  r=$O/full_${a}/external_telemetry.pkl
  if [ -e $r ]; then
    build prm_boost_build_bank_external.py $r $P/pb_${a}_old || exit 1
    build prm_boost_build_bank_external_v2.py $r $P/pb_${a}_v2 --tokenizer "$tk" || exit 1
    pairs+=(pb_$a $P/pb_${a}_old $P/pb_${a}_v2)
  else
    echo "NOTE: $r not there yet (HANDOFF step 6): pb_$a not checked; pb_down.sh RUN=full needs it"
  fi
done
python3 -B $O/code/bank_parity.py $P/PARITY_SUMMARY.json "${pairs[@]}"
