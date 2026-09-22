"""Complete only innovation5 TCN seed0 fits, using the immutable old job code."""
from pathlib import Path
import sys,json,time,subprocess,os,hashlib
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_temporal_neural_job import jobs,key
from scripts.run_temporal_neural_queue import QueueLock,completed_job
from scripts.run_aligned_context_predictors import sha

OUT=ROOT/'results/tcn_aligned_predictor_seed0_v1'
MODELS=ROOT/'results/temporal_context_models_v1'
DATA=ROOT/'results/temporal_context_data_v1'


def write(path,obj):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n');temp.replace(path)


def execute(index,job):
    dest=MODELS/key(job);dest.mkdir(exist_ok=True)
    args=[sys.executable,'-B','-X','utf8',str(ROOT/'scripts/run_temporal_neural_job.py'),
        '--job-index',str(index),'--data',str(DATA),'--out',str(MODELS),'--device','cpu']
    if (dest/'RUN_STATE.json').exists() and json.loads((dest/'RUN_STATE.json').read_text()).get('status')=='TRAINED':
        args.append('--score-after-existing-training')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (dest/'ALIGNED_TCN_JOB.log').open('a',encoding='utf8') as log:
        child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
        code=child.wait()
    if code:raise RuntimeError(f'{key(job)} exited{code}; inspect ALIGNED_TCN_JOB.log')
    return key(job)


def run():
    selected=[(i,j) for i,j in enumerate(jobs()) if j['method']=='tcn' and j['bank']=='innovation5' and j['seed']==0]
    data_hash=sha(DATA/'MANIFEST.json');started=time.perf_counter();done=[];active={};failure=None
    sources=[Path(__file__),ROOT/'docs/experiments/TCN_ALIGNED_PREDICTOR_20260915.md',
        ROOT/'scripts/train_temporal_context_model.py',ROOT/'scripts/score_temporal_context_model.py',
        ROOT/'scripts/run_temporal_neural_job.py',ROOT/'spectral_utils/temporal_context_models.py',ROOT/'spectral_utils/context_training.py']
    manifest=dict(schema='tcn-aligned-predictor-seed0-v1',jobs=[dict(index=i,key=key(j),**j) for i,j in selected],
        max_workers=3,device='cpu',data_manifest_sha256=data_hash,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        old_flow_queue_state_sha256=sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json'),
        seed_scope='seed0 only; not the full old three-seed neural program')
    manifest=json.loads(json.dumps(manifest));path=OUT/'MANIFEST.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Study manifest changed')
    write(path,manifest)
    pending=[]
    for index,job in selected:
        dest=MODELS/key(job)
        if completed_job(dest,job,data_hash):
            old=json.loads((dest/'MANIFEST.json').read_text())
            for p,expected in old['code_sha256'].items():
                if sha(ROOT/p)!=expected:raise ValueError('Reused model code differs')
            done.append(key(job))
        else:pending.append((index,job))
    write(OUT/'REUSE.json',dict(reused=done,initial_artifacts={str((MODELS/k/f).relative_to(ROOT)):sha(MODELS/k/f)
        for k in done for f in ('MANIFEST.json','BEST.pt','TRAINING.json','scoring/MANIFEST.json','scoring/STEP_SCORES.npz')}))
    def status(value):
        write(OUT/'RUN_STATE.json',dict(status=value,pid=os.getpid(),completed=len(done),expected=15,
            completed_keys=done,active_keys=list(active.values()),seconds=time.perf_counter()-started,
            failure=failure,correctness_labels_used=False))
    # Also acquire the old queue's lock to prevent a second writer to any
    # shared TCN job. Do not change its state or remove STOP_AFTER_JOB.
    with QueueLock(ROOT/'results/temporal_neural_queue_seed0_v1/QUEUE.lock'):
        with ThreadPoolExecutor(max_workers=3) as pool:
            while pending or active:
                while pending and len(active)<3 and failure is None:
                    index,job=pending.pop(0);active[pool.submit(execute,index,job)]=key(job)
                    print('[tcn-start]',key(job),flush=True)
                status('RUNNING' if failure is None else 'DRAINING_AFTER_FAILURE')
                finished,_=wait(active,timeout=20,return_when=FIRST_COMPLETED)
                for future in finished:
                    name=active.pop(future)
                    try:
                        future.result()
                        job=next(j for _,j in selected if key(j)==name)
                        if not completed_job(MODELS/name,job,data_hash):raise RuntimeError('Incomplete scored job')
                        done.append(name);print('[tcn-done]',name,len(done),'/15',flush=True)
                    except BaseException as e:failure=repr(e)
                if failure and not active:break
    status('FAILED' if failure else 'SCORED_PENDING_EVALUATION')
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with QueueLock(OUT/'QUEUE.lock'):run()
