"""Run the existing neural matrix sequentially, with resume and explicit failure states.

Default scope is seed 0, both frozen banks, three methods and all outer/pair
exclusions: 90 jobs. No training hyperparameters change. A STOP_AFTER_JOB file
in the queue directory prevents launching the next job without discarding work.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_temporal_neural_job import jobs,key
from scripts.train_temporal_context_model import atomic_json


class QueueLock:
    """OS-owned exclusive lock: process exit releases it, including a crash."""
    def __init__(self,path):self.path=Path(path);self.file=None
    def __enter__(self):
        self.file=self.path.open('a+b');self.file.seek(0,os.SEEK_END)
        if self.file.tell()==0:self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();self.file=None
            raise RuntimeError('another process already owns this neural queue')
        return self
    def __exit__(self,*_):
        self.file.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
        self.file.close()


def load(path):return json.loads(Path(path).read_text())


def matrix_for_seeds(seeds):
    # Preserve the registered experiment definitions; prioritize the innovation
    # bank within each exclusion. Finish all outer folds before nested fits.
    selected=[(i,j) for i,j in enumerate(jobs()) if j['seed'] in seeds]
    order={'diflo':0,'fm':1,'tcn':2}
    return sorted(selected,key=lambda row:(row[1]['seed'],len(row[1]['excluded']),row[1]['excluded'],
                                          row[1]['bank']!='innovation5',order[row[1]['method']]))


def completed_job(dest,job,data_hash):
    path=dest/'scoring/RUN_STATE.json'
    if not path.exists() or load(path).get('status')!='SCORED':return False
    state=load(path);manifest=load(dest/'scoring/MANIFEST.json');trained=manifest['model_manifest']
    if manifest.get('smoke') or state['answers']!=state['expected']:raise ValueError('incomplete completed-job marker')
    for name in ('method','bank','seed'):
        if trained[name]!=job[name]:raise ValueError('completed job configuration differs: '+name)
    if trained['excluded_folds']!=list(job['excluded']) or trained['data_manifest_sha256']!=data_hash:
        raise ValueError('completed job source/data exclusion differs')
    if not (dest/'scoring/STEP_SCORES.npz').is_file():raise ValueError('completed job has no frozen step scores')
    return True


def run(a):
    selected=matrix_for_seeds(a.seeds);a.out.mkdir(parents=True,exist_ok=True)
    data_hash=hashlib.sha256((a.data/'MANIFEST.json').read_bytes()).hexdigest()
    specification=dict(schema='temporal-neural-sequential-queue-v1',seeds=a.seeds,device=a.device,
        model_root=str(a.models),data_manifest_sha256=data_hash,source_root=str(a.source_root),
        jobs=[dict(index=i,key=key(j),**j) for i,j in selected],max_updates_per_fit=50000,
        parallel_workers=1,evaluation=str(a.evaluation),quality_scope='Full population; seed-0 results do not complete the three-seed program.')
    manifest_path=a.out/'MANIFEST.json'
    if manifest_path.exists() and load(manifest_path)!=json.loads(json.dumps(specification)):raise ValueError('queue manifest changed')
    atomic_json(manifest_path,specification)
    current=None;stopping=False;completed=[];started=time.time()
    def request_stop(*_):
        nonlocal stopping
        stopping=True
        if current is not None and current.poll() is None:current.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,request_stop)
    def state(status,**extra):
        atomic_json(a.out/'RUN_STATE.json',dict(status=status,pid=os.getpid(),completed=len(completed),
            expected=len(selected),completed_keys=completed,elapsed_seconds=time.time()-started,**extra))
    try:
        for index,job in selected:
            dest=a.models/key(job)
            if completed_job(dest,job,data_hash):
                completed.append(key(job));state('RUNNING',skipped_completed=key(job));continue
            if stopping or (a.out/'STOP_AFTER_JOB').exists():
                state('PAUSED_BETWEEN_JOBS');return
            dest.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-B','-X','utf8',str(ROOT/'scripts/run_temporal_neural_job.py'),
                '--job-index',str(index),'--data',str(a.data),'--baseline',str(a.baseline),'--out',str(a.models),'--device',a.device]
            if (dest/'RUN_STATE.json').exists() and load(dest/'RUN_STATE.json').get('status')=='TRAINED':
                command.append('--score-after-existing-training')
            print('[queue-start]',index,key(job),flush=True)
            with (dest/'QUEUE_JOB.log').open('a',encoding='utf8') as log:
                current=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                state('RUNNING',active_index=index,active_key=key(job),child_pid=current.pid,log=str(dest/'QUEUE_JOB.log'))
                code=current.wait()
            if stopping:state('STOPPED_CHECKPOINTED',active_key=key(job),child_exit=code);return
            if code:raise RuntimeError(f'job {index} {key(job)} exited {code}; inspect its QUEUE_JOB.log')
            if not completed_job(dest,job,data_hash):raise RuntimeError('child returned without complete scores: '+key(job))
            completed.append(key(job));state('RUNNING',last_completed=key(job));print('[queue-done]',key(job),flush=True)
        if (a.out/'STOP_AFTER_JOB').exists() or stopping:state('PAUSED_BEFORE_EVALUATION');return
        command=[sys.executable,'-B','-X','utf8',str(ROOT/'scripts/evaluate_temporal_context_models.py'),
            '--source-root',str(a.source_root),'--models',str(a.models),'--data',str(a.data),
            '--baseline',str(a.baseline),'--out',str(a.evaluation),'--seeds',','.join(map(str,a.seeds))]
        with (a.out/'EVALUATION.log').open('a',encoding='utf8') as log:
            current=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state('EVALUATING',child_pid=current.pid);code=current.wait()
        if stopping:state('STOPPED_DURING_EVALUATION',child_exit=code);return
        if code:raise RuntimeError(f'evaluation exited {code}; inspect EVALUATION.log')
        if load(a.evaluation/'RUN_STATE.json')['status']!='COMPLETE':raise RuntimeError('evaluation lacks completion marker')
        state('COMPLETE',full_program_complete=False)
    except BaseException as error:
        state('FAILED',error=f'{type(error).__name__}: {error}');raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'results/temporal_context_data_v1')
    p.add_argument('--baseline',type=Path,default=ROOT/'results/temporal_research_baseline_v1')
    p.add_argument('--models',type=Path,default=ROOT/'results/temporal_context_models_v1')
    p.add_argument('--out',type=Path,default=ROOT/'results/temporal_neural_queue_seed0_v1')
    p.add_argument('--evaluation',type=Path,default=ROOT/'results/temporal_context_evaluation_seed0_v1')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--seeds',default='0')
    a=p.parse_args();a.seeds=sorted(set(int(x) for x in a.seeds.split(',')))
    if not a.seeds or any(x not in (0,1,2) for x in a.seeds):raise ValueError('only registered seeds 0,1,2 are permitted')
    a.out.mkdir(parents=True,exist_ok=True)
    with QueueLock(a.out/'QUEUE.lock'):run(a)


if __name__=='__main__':main()
