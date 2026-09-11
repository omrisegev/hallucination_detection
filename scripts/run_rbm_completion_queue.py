"""Dependency-aware scheduling of the frozen suites; no scientific code changes.

At most one suite until the original DUFS pipeline completes, then two suites.
Each scoring child keeps the registered two workers. A currently running child
can be adopted after its old waiting controller is retired, without stopping
that child. Windows process creation times protect against recycled PIDs.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_literature_completion import PROGRAM,SUITES,base
ORDER=('variance','capacity','temporal','stability','depth')
DEPENDENCIES={'stability':('capacity',),'depth':('capacity',)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def next_stage(directory):
    state=read(directory/'RUN_STATE.json')
    if state.get('status')=='COMPLETE' and read(directory/'RESULT_REVIEW.json').get('status')=='PASS':
        return None
    if state.get('status')=='SCORED_AWAITING_REVIEW':return 'full_review'
    if read(directory/'SMOKE.json').get('status')!='PASS':return 'smoke'
    if read(directory/'SMOKE_REVIEW.json').get('status')!='PASS':return 'smoke_review'
    return 'full'


def ready_suites(stages,active,completed):
    return [s for s in ORDER if stages[s] is not None and s not in active
            and all(d in completed for d in DEPENDENCIES.get(s,()))]


class ExistingProcess:
    """A read-only process handle, with optional creation-time identity check."""
    def __init__(self,pid,created=None):
        self.pid=int(pid);k=ctypes.WinDLL('kernel32',use_last_error=True);self.k=k
        k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
        k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        k.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        k.CloseHandle.argtypes=[wintypes.HANDLE]
        self.handle=k.OpenProcess(0x1000,False,self.pid)
        if not self.handle:raise ProcessLookupError(f'Cannot open pid {pid}: {ctypes.get_last_error()}')
        a,b,c,d=(wintypes.FILETIME() for _ in range(4))
        if not k.GetProcessTimes(self.handle,*[ctypes.byref(v) for v in (a,b,c,d)]):
            self.close();raise OSError('GetProcessTimes failed')
        self.created=(a.dwHighDateTime<<32)|a.dwLowDateTime
        if created is not None and self.created!=int(created):
            self.close();raise ProcessLookupError('PID creation time differs from saved child')
    def poll(self):
        code=wintypes.DWORD()
        if not self.k.GetExitCodeProcess(self.handle,ctypes.byref(code)):
            raise OSError('GetExitCodeProcess failed; do not infer termination')
        return None if code.value==259 else int(code.value)
    def close(self):
        if self.handle:self.k.CloseHandle(self.handle);self.handle=None


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--adopt-file',type=Path);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();source=args.source_root.resolve();PROGRAM.mkdir(parents=True,exist_ok=True)
    if os.name!='nt':raise RuntimeError('Windows process adoption only')
    import msvcrt
    lock=(PROGRAM/'QUEUE.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    jobs={};failures={};saved=read(PROGRAM/'QUEUE_STATE.json')
    adoption=read(args.adopt_file).get('children',[]) if args.adopt_file else saved.get('active',[])
    for item in adoption:
        if next_stage(PROGRAM/item['suite']) is None:continue
        proc=ExistingProcess(item['pid'],item['created'])
        jobs[item['suite']]={**item,'process':proc,'adopted':True}
    def persist(status):
        active=[{k:v for k,v in j.items() if k not in ('process','stream')} for j in jobs.values()]
        state=dict(status=status,pid=os.getpid(),active=active,failures=failures,workers_per_suite=args.workers,
            stages={s:next_stage(PROGRAM/s) for s in SUITES},updated_utc=datetime.now(timezone.utc).isoformat(),
            note='Live child handles checked by scheduler; scientific scripts and scoring settings unchanged.')
        base.atomic_json(PROGRAM/'QUEUE_STATE.json',state)
        base.atomic_json(PROGRAM/'PROGRAM_STATE.json',state)
    def launch(suite,stage):
        script='review_rbm_literature_completion_v2.py' if stage.endswith('review') else 'run_rbm_literature_completion.py'
        cmd=[sys.executable,str(ROOT/'scripts'/script),'--source-root',str(source),'--suite',suite]
        if stage.startswith('smoke'):cmd+=['--smoke']
        if not stage.endswith('review'):cmd+=['--workers',str(args.workers)]
        log=PROGRAM/suite/('queue_'+stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
        log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('wb')
        child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
        proc=ExistingProcess(child.pid)
        jobs[suite]=dict(suite=suite,stage=stage,pid=child.pid,created=proc.created,
            process=proc,stream=stream,log=str(log),adopted=False)
        print('[queue launch]',suite,stage,child.pid,flush=True)
    persist('RUNNING')
    while True:
        for suite,job in list(jobs.items()):
            code=job['process'].poll()
            if code is None:continue
            job['process'].close()
            if 'stream' in job:job['stream'].close()
            del jobs[suite]
            print('[queue exit]',suite,job['stage'],code,flush=True)
            if code!=0:failures[suite]=dict(stage=job['stage'],exit_code=code,log=job.get('log'))
            else:
                expected={'smoke':'smoke_review','smoke_review':'full','full':'full_review','full_review':None}
                if next_stage(PROGRAM/suite)!=expected[job['stage']]:
                    failures[suite]=dict(stage=job['stage'],error='Child exited without the required output state',log=job.get('log'))
                elif job['stage']=='full_review':
                    subprocess.run([sys.executable,str(ROOT/'scripts/build_rbm_literature_summary.py')],check=True)
        stages={s:next_stage(PROGRAM/s) for s in SUITES};completed={s for s in SUITES if stages[s] is None}
        dufs=source/'.worktrees/dufs-moment-selection-v1/results/dufs_moment_selection_v1'
        dufs_done=(read(dufs/'PIPELINE_STATE.json').get('status')=='COMPLETE'
            and read(dufs/'RESULT_REVIEW.json').get('status')=='PASS'
            and read(dufs/'STATE_REVIEW.json').get('status')=='PASS')
        slots=2 if dufs_done else 1
        if not failures:
            for suite in ready_suites(stages,jobs,completed)[:max(0,slots-len(jobs))]:
                launch(suite,stages[suite])
        persist('FAILED' if failures else 'RUNNING')
        if failures and not jobs:raise RuntimeError(f'Registered stage failed: {failures}')
        if len(completed)==len(SUITES) and not jobs:
            persist('REGISTERED_SUITES_FINISHED');break
        if not jobs and not failures:
            raise RuntimeError('No eligible stage; inspect dependency state')
        time.sleep(10)
    lock.close()


if __name__=='__main__':main()
