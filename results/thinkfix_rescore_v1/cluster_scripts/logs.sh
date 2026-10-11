#!/bin/bash
# usage: bash logs.sh <jobid> [lines]
export SLURM_CONF_SERVER=controller-primary
L=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs
for j in "$@"; do
  case "$j" in *[!0-9]*) continue;; esac
  f=$(ls $L/*_$j.out 2>/dev/null | head -1)
  st=$(sacct -j $j -X -n --format=State%12,Elapsed 2>/dev/null | head -1)
  echo "== $j $st $f"
  [ -n "$f" ] && grep -v -E 'Retrying|WARNING|DEPRECATION|PATH|no-warn-script|^\s*$|pyxis' "$f" | tail -${N:-8} | cut -c1-400
done
