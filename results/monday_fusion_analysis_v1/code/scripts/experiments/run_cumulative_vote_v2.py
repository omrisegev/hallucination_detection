#!/usr/bin/env python
"""CPU-only, resumable full-population experiment. Run --stage prepare first."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import argparse
from cvf_v2.data import config,Dataset,prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['prepare','spectral','em','inner','report','render','all'],default='all')
    p.add_argument('--task',choices=['pb_q4','pb_q8','prm','all'],default='all');p.add_argument('--fold',type=int)
    args=p.parse_args();d=Dataset(config(args.config))
    if args.stage in ['prepare','all']:prepare(d)
    if args.stage in ['spectral','em','inner','all']:
        from cvf_v2.runner import run
        for stage in (['spectral','em','inner'] if args.stage=='all' else [args.stage]):run(d,stage,args.task,args.fold)
    if args.stage in ['report','all']:
        from cvf_v2.report import report
        report(d)
    if args.stage=='render':
        from cvf_v2.report import render
        render(d)

if __name__=='__main__':main()
