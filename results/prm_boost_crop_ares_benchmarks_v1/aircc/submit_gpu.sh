#!/bin/bash
# GPU telemetry jobs for the missing benchmarks. One independent job per (mode, PRM, benchmark): outputs never overlap.
#   bash submit_gpu.sh parity                   PRMBench first 40 rows, both modes, --compare-to the frozen pkls
#   bash submit_gpu.sh timing [prm|sa] [prm ...]  12 deterministic rows per benchmark, 1 h wall
#   bash submit_gpu.sh full   [prm|sa] [prm ...]  all rows; WALL_7B / WALL_72B from the timing. Re-running the same
#                                                 command resumes an exit-85 job (the driver refuses a mismatched resume).
# prm = PRM rewards + backbone telemetry (--mode prm) on NEW_BENCHES.
# sa  = the PRM's standalone base model (--mode instruct) on NEW_BENCHES + SA_ONLY_BENCHES.
# Needs a /preflight PASS in the same session. Prints "<mode> <prm> <bench> <job id>".
set -euo pipefail; source "$(dirname "$0")/env.sh"
phase=${1:?parity, timing or full}; shift || true
[ -d "$TEL_CODE" ] || { echo "missing TEL_CODE $TEL_CODE" >&2; exit 2; }
mkdir -p $O $S/logs
SB=$TEL_CODE/cluster/submit_prm_telemetry_offline.sbatch

submit() {   # name wall out args...
  local name=$1 wall=$2 out=$3; shift 3
  if [ -e "$out/manifest.json" ] && [ "$phase" != full ]; then echo "REFUSING: $out already has a manifest" >&2; return 1; fi
  if [ -n "$(squeue -h -u $USER -n $name 2>/dev/null)" ]; then echo "REFUSING: $name is already queued or running" >&2; return 1; fi
  sbatch --parsable -A $ACCT -p $PART --qos=$QOS --time=$wall --job-name=$name \
    --export=ALL,CODE_DIR=$TEL_CODE $SB "$@" --out $out
}

if [ "$phase" = parity ]; then
  for a in ${*:-$ADAPTERS}; do
    test -s "${PRM_PKL[$a]}"; test -s "${SA_PKL[$a]}"; snapshot "${SA_MODEL[$a]}" >/dev/null
    j=$(submit cab_par_prm_$a 01:00:00 $O/parity_prm_$a --mode prm --adapter $a --meta-pkl "${PRM_PKL[$a]}" \
        --n-samples 40 --compare-to "${PRM_PKL[$a]}"); echo "prm $a prmbench40 $j"
    j=$(submit cab_par_sa_$a 01:00:00 $O/parity_sa_$a --mode instruct --model "${SA_MODEL[$a]}" \
        --prompt-style ${SA_STYLE[$a]} --max-batch-tokens ${SA_MAXTOK[$a]} --meta-pkl "${SA_PKL[$a]}" \
        --n-samples 40 --compare-to "${SA_PKL[$a]}"); echo "sa $a prmbench40 $j"
  done
  exit 0
fi

mode=${1:?prm or sa}; shift; list=${*:-$ADAPTERS}
case $phase in timing) ;; full) : "${WALL_7B:?set WALL_7B from timing}"; : "${WALL_72B:?set WALL_72B from timing}";; *) echo "bad phase" >&2; exit 2;; esac
benches=$NEW_BENCHES; [ "$mode" = sa ] && benches="$NEW_BENCHES $SA_ONLY_BENCHES"
for a in $list; do
  for b in $benches; do
    ans=${ANSWERS[$b]}; extra=(); wall=01:00:00
    if [ "$phase" = timing ]; then
      if [ -e "${ans%.json}_timing12.json" ]; then ans=${ans%.json}_timing12.json; else extra=(--n-samples 12); fi
    elif [ $a = qwen_prm_72b ]; then wall=$WALL_72B; else wall=$WALL_7B; fi
    test -s "$ans" || { echo "missing $ans" >&2; exit 2; }
    out=$O/${phase}_${mode}_${a}_${b}
    if [ "$mode" = prm ]; then
      j=$(submit cab_${phase:0:1}p_${a}_${b} $wall $out --mode prm --adapter $a --answers "$ans" "${extra[@]}")
    else
      snapshot "${SA_MODEL[$a]}" >/dev/null
      j=$(submit cab_${phase:0:1}s_${a}_${b} $wall $out --mode instruct --model "${SA_MODEL[$a]}" \
          --prompt-style ${SA_STYLE[$a]} --max-batch-tokens ${SA_MAXTOK[$a]} --answers "$ans" "${extra[@]}")
    fi
    echo "$mode $a $b $j"
  done
done
