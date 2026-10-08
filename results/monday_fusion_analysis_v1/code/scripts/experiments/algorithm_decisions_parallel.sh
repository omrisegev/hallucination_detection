#!/usr/bin/env bash
# algorithm_decisions_v1: fit phase in 4 parallel processes (2 banks each, balanced by the fold-0 smoke run times), then the
# assemble phase.  Usage: bash scripts/experiments/algorithm_decisions_parallel.sh RUN_ID   (extra env such as ER_FOLDS passes through)
set -u
RUN_ID=${1:?run id}
cd "$(dirname "$0")/../.."
LOG=results/algorithm_decisions_v1/${RUN_ID}_parallel.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
pids=()
for BANKSET in B54,B13 B51,B16 B35,B20 B32,B23; do
  ER_PHASE=fit ER_BANKS=$BANKSET python scripts/experiments/algorithm_decisions_run.py "$RUN_ID" > "results/algorithm_decisions_v1/${RUN_ID}_fit_${BANKSET/,/_}.log" 2>&1 &
  pids+=($!); sleep 20
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "fit phase exit: $fail" >> "$LOG"
if [ "$fail" -ne 0 ]; then echo "FIT FAILED" >> "$LOG"; exit 1; fi
ER_PHASE=assemble python scripts/experiments/algorithm_decisions_run.py "$RUN_ID" >> "$LOG" 2>&1
echo "assemble exit: $?" >> "$LOG"
