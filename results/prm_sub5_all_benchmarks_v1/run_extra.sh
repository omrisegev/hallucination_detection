#!/bin/bash
# PRMG_N16 / EQ_N16 / HALF_N16 for every cell (stored arms, no refit), one process per cell, then re-assemble.
cd "$(dirname "$0")"
python -B sub5_all.py list | tr -d '\r' | while read m b p; do
  [ -e "parts_extra/${m}_${b}_${p}.json" ] && continue
  python -B sub5_all.py extra $m $b $p || echo "JOB FAILED $m $b $p"
done
python -B sub5_all.py assemble
