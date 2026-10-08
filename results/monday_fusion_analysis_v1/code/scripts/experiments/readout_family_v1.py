#!/usr/bin/env python
"""Readout-family experiment (Step 429): CPU-only, resumable, on the frozen token matrices.

Same stages as run_cumulative_vote_v2.py.  `prepare` rebuilds the seven frozen readouts and
refuses to continue unless their sha256 matches the reference run, then adds the extended
readouts; `report` writes the tables and ANCHOR_PARITY.json (no HTML: the interpretation of
this run is written in HISTORY.md, not generated).
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from cvf_v2.data import config,Dataset,prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'configs/readout_family_v1.json'))
    p.add_argument('--stage',choices=['prepare','spectral','em','inner','report','all'],default='all')
    p.add_argument('--task',choices=['pb_q4','pb_q8','prm','all'],default='all');p.add_argument('--fold',type=int)
    args=p.parse_args();d=Dataset(config(args.config))
    if not d.extended:raise SystemExit('config has no readouts_ext; use run_cumulative_vote_v2.py for the frozen protocol')
    if args.stage in ['prepare','all']:prepare(d)
    if args.stage in ['spectral','em','inner','all']:
        from cvf_v2.runner import run
        for stage in (['spectral','em','inner'] if args.stage=='all' else [args.stage]):run(d,stage,args.task,args.fold)
    if args.stage in ['report','all']:
        from cvf_v2.report import report
        report(d,html=False)

if __name__=='__main__':main()
