#!/bin/bash
# Omri 2026-10-19 (option 1): the Socratic reference is the bank rebuilt today with the OLD frozen builder, because the
# existing Socratic bank_v2 is not bit-reproducible by the old builder either (diag job 289294: 3.26e-9). Builds the
# remaining old-builder Socratic references, re-runs bank_parity.py with them, keeps the failed summary, then pb_down.
set -uo pipefail; source /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/env.sh
X=$S/results/prm_boost_external_v1; D=$O/parity_diag; P=$O/parity
for a in pure_prm rlhflow_prm qwen_prm_72b; do
  [ -e $D/socratic_${a}_old/FEATURES.npz ] || (cd $BANK_CODE && python3 -B scripts/prm_boost_build_bank_external.py \
     --raw $X/full_${a}_socratic/external_telemetry.pkl --mask-code $MASK_CODE --names-from $NAMES_FROM \
     --out $D/socratic_${a}_old --workers 8) &
done; wait
[ -e $P/PARITY_SUMMARY_existing_socratic_ref_FAIL.json ] || mv $P/PARITY_SUMMARY.json $P/PARITY_SUMMARY_existing_socratic_ref_FAIL.json
pairs=()
for a in $PARITY_ADAPTERS; do
  pairs+=(socratic_$a $D/socratic_${a}_old $P/socratic_${a}_v2 pb_$a $P/pb_${a}_old $P/pb_${a}_v2)
done
python3 -B $O/code/bank_parity.py $P/PARITY_SUMMARY.json "${pairs[@]}" || exit 1
bash $O/code/pb_down.sh
