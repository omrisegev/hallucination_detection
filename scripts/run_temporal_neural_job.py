"""One bounded training/scoring job from the frozen neural experiment matrix."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_temporal_context_model import atomic_json


def jobs():
    result=[]
    for seed in (0,1,2):
        for excluded in [(h,) for h in range(5)]+list(combinations(range(5),2)):
            for bank in ('original4','innovation5'):
                for method in ('diflo','fm','tcn'):
                    result.append(dict(method=method,bank=bank,seed=seed,excluded=excluded))
    return result


def key(job):
    return f"{job['method']}__{job['bank']}__seed{job['seed']}__exclude"+'_'.join(map(str,job['excluded']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job-index',type=int)
    p.add_argument('--write-matrix',type=Path);p.add_argument('--data',type=Path,default=ROOT/'results/temporal_context_data_v1')
    p.add_argument('--baseline',type=Path,default=ROOT/'results/temporal_research_baseline_v1')
    p.add_argument('--out',type=Path,default=ROOT/'results/temporal_context_models_v1')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--score-after-existing-training',action='store_true')
    a=p.parse_args();matrix=jobs()
    if a.write_matrix:
        atomic_json(a.write_matrix,dict(jobs=[dict(index=i,key=key(j),**j) for i,j in enumerate(matrix)],
            maximum_fits=len(matrix),note='Each job has <=50000 updates with unlabeled validation early stopping. No automatic submission.'))
        return
    if a.job_index is None or not 0<=a.job_index<len(matrix):raise ValueError('job-index outside frozen matrix')
    job=matrix[a.job_index];dest=a.out/key(job);dest.mkdir(parents=True,exist_ok=True)
    child=None;stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop)
    def execute(script,args):
        nonlocal child
        child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(ROOT/'scripts'/script),*map(str,args)])
        code=child.wait()
        if stopping:raise SystemExit(75)
        if code:raise subprocess.CalledProcessError(code,script)
    if a.score_after_existing_training:
        # A detached local watcher; never starts a second writer for a live fit.
        while True:
            status=json.loads((dest/'RUN_STATE.json').read_text())['status'] if (dest/'RUN_STATE.json').exists() else 'STARTING'
            if status=='TRAINED':break
            if status in ('FAILED','STOPPED_CHECKPOINTED','SMOKE_COMPLETE'):raise RuntimeError('existing fit ended without a full trained model: '+status)
            if stopping:return
            time.sleep(30)
    else:
        execute('train_temporal_context_model.py',['--data',a.data,'--out',dest,'--method',job['method'],
            '--bank',job['bank'],'--seed',job['seed'],'--excluded-folds',','.join(map(str,job['excluded'])),
            '--steps','50000','--batch-size','256','--device',a.device])
    if json.loads((dest/'RUN_STATE.json').read_text())['status']!='TRAINED':raise RuntimeError('training did not finish')
    execute('score_temporal_context_model.py',['--model',dest,'--data',a.data,'--baseline',a.baseline,'--out',dest/'scoring','--device',a.device])


if __name__=='__main__':main()
