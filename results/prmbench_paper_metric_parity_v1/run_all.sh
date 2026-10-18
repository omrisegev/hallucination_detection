#!/bin/sh
# CPU only, existing local data only (about 30 s). Run from this directory.
set -e
PY=../../.venv/bin/python
$PY common.py --force      # rebuild ARM_SCORES.npz and PARITY.csv (CHAIN refit + parity vs bank sweep)
$PY prism_metrics.py
$PY crop_metrics.py
$PY slider_metrics.py
$PY aggregation_gap.py
$PY build_table.py
$PY plot_papers.py
