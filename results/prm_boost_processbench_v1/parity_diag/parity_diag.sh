#!/bin/bash
# Diagnostic only: is the Socratic parity gap caused by the v2 builder, or by rebuilding at all?
# Rebuild the Socratic qwen_prm bank with the OLD frozen builder now, compare to the existing bank_v2 and to v2.
set -uo pipefail; source /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/env.sh
X=$S/results/prm_boost_external_v1; D=$O/parity_diag; mkdir -p $D
s=$X/full_qwen_prm_socratic
[ -e $D/socratic_qwen_prm_old/FEATURES.npz ] || (cd $BANK_CODE && python3 -B scripts/prm_boost_build_bank_external.py --raw $s/external_telemetry.pkl \
   --mask-code $MASK_CODE --names-from $NAMES_FROM --out $D/socratic_qwen_prm_old --workers ${SLURM_CPUS_PER_TASK:-8})
python3 -B $O/code/bank_parity.py $D/DIAG_SUMMARY.json \
  old_now_vs_existing $s/bank_v2 $D/socratic_qwen_prm_old \
  old_now_vs_v2 $D/socratic_qwen_prm_old $O/parity/socratic_qwen_prm_v2
