"""Run the already registered suites in short resumable stages, with review."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_literature_completion import PROGRAM,SUITES,base


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--suites',nargs='+',choices=SUITES,default=list(SUITES))
    args=p.parse_args();PROGRAM.mkdir(parents=True,exist_ok=True)
    def state(status,suite,stage,**extra):
        base.atomic_json(PROGRAM/'PROGRAM_STATE.json',dict(status=status,suite=suite,stage=stage,pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(),registered_suites=list(SUITES),**extra))
    for suite in args.suites:
        existing=PROGRAM/suite/'RUN_STATE.json'
        if existing.exists() and json.loads(existing.read_text()).get('status')=='COMPLETE':
            print('[program] already complete',suite,flush=True);continue
        tasks=[('smoke','run_rbm_literature_completion.py',['--smoke','--workers',str(args.workers)]),
               ('smoke_review','review_rbm_literature_completion_v2.py',['--smoke']),
               ('full','run_rbm_literature_completion.py',['--workers',str(args.workers)]),
               ('full_review','review_rbm_literature_completion_v2.py',[])]
        if existing.exists() and json.loads(existing.read_text()).get('status')=='SCORED_AWAITING_REVIEW':
            # Completed scores/metrics are immutable; resume at the failed audit.
            tasks=tasks[-1:]
        for stage,script,options in tasks:
            state('RUNNING',suite,stage)
            print('[program]',suite,stage,datetime.now(timezone.utc).isoformat(),flush=True)
            try:
                subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--source-root',str(args.source_root),
                                '--suite',suite,*options],check=True)
                if stage=='smoke':
                    smoke=json.loads((PROGRAM/suite/'SMOKE.json').read_text())
                    if smoke['status']!='PASS':raise RuntimeError('smoke has explicit failures; inspect before full run')
            except BaseException as error:
                state('FAILED',suite,stage,error=str(error));raise
        subprocess.run([sys.executable,str(ROOT/'scripts/build_rbm_literature_summary.py')],check=True)
    state('REGISTERED_SUITES_FINISHED',None,'summary',
          note='DUFS and corrected literature audit have separate completion checks; this is not overall goal completion.')


if __name__=='__main__':main()
