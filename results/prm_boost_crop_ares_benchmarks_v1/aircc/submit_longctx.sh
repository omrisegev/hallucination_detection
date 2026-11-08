#!/bin/bash
# DeltaBench long-context adaptation (2026-11-08): re-run the three 4,096-token PRMs (and their standalone bases) on ALL
# 1,236 DeltaBench answers with dynamic NTK RoPE x3 (12,288 tokens). Answers that fit 4,096 take the original code path,
# so the prm-mode rows of the 508 already-scored answers must reproduce exactly (merge_longctx.py checks this).
#   bash submit_longctx.sh smoke     12 rows (the 12 longest overflow answers + 4 in-range), 1 h, both modes, qwen_prm only
#   bash submit_longctx.sh full      all rows, both modes, the three adapters
# Needs a /preflight PASS in the same session. Code dir: $S/code_longctx_v1 (code_standalone_v1 + this branch's driver).
set -euo pipefail; source "$(dirname "$0")/env.sh"
phase=${1:?smoke or full}; shift || true
export TEL_CODE=$S/code_longctx_v1; FACTOR=${FACTOR:-3}
[ -s $TEL_CODE/spectral_utils/rope_extension.py ] || { echo "missing $TEL_CODE" >&2; exit 2; }
SB=$TEL_CODE/cluster/submit_prm_telemetry_offline.sbatch; mkdir -p $S/logs
LIST=${LIST:-"qwen_prm pure_prm qwen_prm_72b"}; [ $phase = smoke ] && LIST=${LIST_SMOKE:-qwen_prm}
ans=$DATA/deltabench/answers.json; [ $phase = smoke ] && ans=$DATA/deltabench/answers_longctx_smoke.json
test -s $ans || { echo "missing $ans" >&2; exit 2; }
submit() {   # name wall out args...
  local name=$1 wall=$2 out=$3; shift 3
  if [ -n "$(squeue -h -u $USER -n $name 2>/dev/null)" ]; then echo "REFUSING: $name is queued or running" >&2; return 1; fi
  sbatch --parsable -A $ACCT -p $PART --qos=$QOS --time=$wall --job-name=$name \
    --export=ALL,CODE_DIR=$TEL_CODE $SB "$@" --out $out
}
for a in $LIST; do
  w=08:00:00; [ $a = qwen_prm_72b ] && w=16:00:00; [ $phase = smoke ] && w=01:00:00
  j=$(submit lc_${phase:0:1}p_$a $w $O/${phase}_prm_${a}_deltabench_ropedyn$FACTOR --mode prm --adapter $a \
      --answers $ans --rope-dynamic-factor $FACTOR); echo "prm $a $j"
  snapshot "${SA_MODEL[$a]}" >/dev/null
  j=$(submit lc_${phase:0:1}s_$a $w $O/${phase}_sa_${a}_deltabench_ropedyn$FACTOR --mode instruct --model "${SA_MODEL[$a]}" \
      --prompt-style ${SA_STYLE[$a]} --max-batch 1 --max-batch-tokens 16384 --answers $ans --rope-dynamic-factor $FACTOR)
  echo "sa $a $j"
done
