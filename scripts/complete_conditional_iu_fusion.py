"""Sequential smoke -> reviewed full run; resume only normal invocation caps.

No change to recipes, source results, or competing processes. This supervisor
holds its own lock, stops on a failure, and never assumes an old PID is alive.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/conditional_iu_fusion_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf8')) if path.exists() else {}


def emit(value):
    path=OUT/'PROGRAM_STATE.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf8');temp.replace(path)


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--family',choices=['position','graph_local','graph_tv'],required=True);args=p.parse_args()
    OUT=ROOT/'results/conditional_iu_fusion_v1'/args.family
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'PROGRAM.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    active = dict(process=None, folder=None)
    stopping = dict(signal=None)
    def scheduler_stop(signum):
        # Linux Slurm only. Stop the owned child before releasing its lock.
        # SQLite commits already preserve completed arms; an unfinished fit
        # is recomputed. Never clear a lock belonging to another process.
        process, folder = active['process'], active['folder']
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
            child_lock=folder/'RUN.lock'
            if child_lock.exists():
                if int(child_lock.read_text())!=process.pid:
                    raise RuntimeError('refuse to release an unowned child lock')
                child_lock.unlink()
        emit(dict(status='CHECKPOINTED_SCHEDULER_STOP',pid=os.getpid(),signal=signum,
                  note='Completed SQLite transactions retained; interrupted arm refits'))
        if lock.exists() and int(lock.read_text())==os.getpid():lock.unlink()
        # A clean early exit is not auto-requeued by Slurm. Wait for the
        # scheduler's kill/requeue. Wall-time expiry still requires resubmission.
        while True: signal.pause()
    # A signal handler must not call Popen.wait(): it may interrupt another
    # wait while that non-reentrant waitpid lock is held. Defer cleanup to
    # the normal control loop, outside all subprocess locks.
    if sys.platform=='linux':
        signal.signal(signal.SIGTERM,lambda signum, frame:stopping.update(signal=signum))
    try:
        for phase,complete in [('smoke','SMOKE_COMPLETE'),('full','COMPLETE_REVIEWED')]:
            folder=OUT/'smoke' if phase=='smoke' else OUT
            while True:
                if stopping['signal'] is not None:scheduler_stop(stopping['signal'])
                if (folder/'RUN.lock').exists():
                    raise RuntimeError('An existing run lock needs inspection: '+str(folder/'RUN.lock'))
                # Always invoke the driver once to verify its frozen manifest,
                # even when a checkpoint says complete.
                emit(dict(status='RUNNING_'+phase.upper(),pid=os.getpid(),phase=phase))
                with (OUT/(phase+'.stdout.log')).open('a',encoding='utf8') as stdout, (OUT/(phase+'.stderr.log')).open('a',encoding='utf8') as stderr:
                    process=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/run_conditional_iu_fusion.py'),
                                              '--source-root',str(args.source_root),'--family',args.family,'--phase',phase],
                                             cwd=ROOT,stdout=stdout,stderr=stderr,
                                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    active.update(process=process,folder=folder)
                    emit(dict(status='RUNNING_'+phase.upper(),pid=os.getpid(),child_pid=process.pid,phase=phase))
                    while True:
                        if stopping['signal'] is not None:scheduler_stop(stopping['signal'])
                        code=process.poll()
                        if code is not None:break
                        time.sleep(.25)
                state=read(folder/'RUN_STATE.json')
                if code!=0:raise RuntimeError(f'{phase} failed, exit {code}: {state}')
                if state.get('status')=='CHECKPOINTED_INVOCATION_CAP':
                    emit(dict(status='RESUMING_NORMAL_CAP',phase=phase,pid=os.getpid()))
                    time.sleep(1);continue
                if state.get('status')!=complete:raise RuntimeError(f'Unexpected terminal {phase}: {state}')
                review=read(folder/('SMOKE_REVIEW.json' if phase=='smoke' else 'RESULT_REVIEW.json'))
                if review.get('status')!='PASS':raise RuntimeError(f'{phase} did not pass its review')
                emit(dict(status=complete,pid=os.getpid(),phase=phase,review=review))
                break
        emit(dict(status='COMPLETE_REVIEWED',pid=os.getpid(),next='Present findings; no further family starts'))
    except Exception as exc:
        emit(dict(status='STOPPED_REQUIRES_REVIEW',pid=os.getpid(),reason=str(exc)));raise
    finally:
        if lock.exists() and int(lock.read_text())==os.getpid():lock.unlink()


if __name__=='__main__':main()
