#!/bin/bash
# One process per cell (bounds memory on a 16 GB machine), then merge. Skips cells whose part file exists.
cd "$(dirname "$0")"
python -B sub5_all.py gates || exit 3
python -B sub5_all.py list | tr -d '\r' | while read m b p; do
  [ -e "parts/${m}_${b}_${p}.json" ] && continue
  python -B sub5_all.py job $m $b $p || echo "JOB FAILED $m $b $p"
done
python -B sub5_all.py assemble
