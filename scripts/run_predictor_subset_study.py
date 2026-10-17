"""Frozen32-arm predictor subset study, source-excluded models and answer-local IU."""
from pathlib import Path
import sys,json,io,sqlite3,time,subprocess,os,argparse,hashlib
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_context_models import TelemetryTCN,RidgePredictor
from spectral_utils.aligned_context_predictors import bocpd_mean,noreset_mean,past_mean
from spectral_utils.predictor_subset_fusion import PREDICTORS,SUBSETS,METHODS,fuse,fast_iu
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from scripts.run_aligned_context_predictors import ridge_prediction
from scripts.run_temporal_neural_queue import QueueLock

OUT=ROOT/'results/predictor_subset_iu_v1'
DATA=ROOT/'results/temporal_context_data_v1'
MODELS=ROOT/'results/temporal_context_models_v1'
JOBS=tuple((i,) for i in range(5))+tuple(combinations(range(5),2))

def key(excluded):return 'exclude'+'_'.join(map(str,excluded))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def write(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n');tmp.replace(p)

def load_models(bundle,excluded):
    stem='innovation5__exclude_'+'_'.join(map(str,excluded))
    rp=ROOT/'results/temporal_linear_context_v1'/(stem+'_model.npz')
    spec=json.loads(rp.with_suffix('.json').read_text())
    tp=MODELS/('tcn__innovation5__seed0__'+key(excluded))
    ts=json.loads((tp/'MANIFEST.json').read_text())
    expected=bundle.split(excluded)
    for s in (spec,ts):
        for name,ids in zip(('training','validation','held'),expected):
            if set(s[name+'_groups'])!={bundle.metadata[i]['group_id'] for i in ids}:raise ValueError('Source group mismatch')
    if ts['excluded_folds']!=list(excluded) or ts['smoke']:raise ValueError('TCN model contract mismatch')
    with np.load(rp) as f:ridge=RidgePredictor(f['coefficient'],f['mean'],f['scale'])
    model=TelemetryTCN(5);state=torch.load(tp/'BEST.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);model.eval()
    with np.load(ROOT/'results/temporal_linear_context_v1'/(stem+'.npz')) as f:
        rr=f['real__signed_residual_0.25']
    with np.load(tp/'scoring/STEP_SCORES.npz') as f:tr=f['real__signed_residual_0.25']
    provenance=dict(ridge_sha256=sha(rp),ridge_manifest_sha256=sha(rp.with_suffix('.json')),
        tcn_sha256=sha(tp/'BEST.pt'),tcn_manifest_sha256=sha(tp/'MANIFEST.json'))
    return ridge,model,tp,rr,tr,provenance

def tcn_predict(bundle,i,model):
    n=bundle.length[i];out=[]
    with torch.no_grad():
        for a in range(0,n,512):
            positions=np.arange(a,min(a+512,n));batch=bundle.batch(np.full(len(positions),i),positions,with_target=False)
            mu,_=model(batch['history'],batch['mask'],batch['position']);out.append(mu.numpy())
    return np.concatenate(out)

def worker(index):
    torch.set_num_threads(1);began=time.perf_counter();excluded=JOBS[index]
    folder=OUT/key(excluded);folder.mkdir(exist_ok=True)
    donepath=folder/'RUN_STATE.json'
    if donepath.exists() and json.loads(donepath.read_text()).get('status')=='SCORED':return
    bundle=FeatureBundle(DATA,'innovation5');meta=bundle.metadata
    ridge,tcn,tp,ridge_ref,tcn_ref,provenance=load_models(bundle,excluded)
    outer=len(excluded)==1
    ids=[i for i,m in enumerate(meta) if m['fold'] in excluded and (outer or not m['cell'].startswith('pb_'))]
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:base=f['steps__append_innovation__H0lim']
    with np.load(ROOT/'results/tcn_aligned_predictor_seed0_v1/SCORES_FROZEN.npz') as f:
        references={n:f['tcn__real' if n=='tcn' else n] for n in PREDICTORS}
    cache=np.load(OUT/'RESIDUALS.npy',mmap_mode='r+' if outer else 'r')
    tcndb=sqlite3.connect('file:'+str(tp/'scoring/SCORES.sqlite').replace('\\','/')+'?mode=ro',uri=True) if outer else None
    db=sqlite3.connect(folder/'ANSWERS.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS answers(idx INTEGER PRIMARY KEY, scores BLOB, diagnostic TEXT)');db.commit()
    done={r[0] for r in db.execute('SELECT idx FROM answers')}
    if not done<=set(ids):raise ValueError('Checkpoint roster mismatch')
    previous=json.loads(donepath.read_text()) if donepath.exists() else {}
    seconds_before=previous.get('seconds',0)
    manifest=dict(excluded=list(excluded),answers=len(ids),provenance=provenance,
        source_manifest_sha256=sha(OUT/'MANIFEST.json'),correctness_labels_used=False)
    mp=folder/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Job manifest drift')
    write(mp,manifest)
    def state(status,**extra):
        write(donepath,dict(status=status,answers=len(done),expected=len(ids),
            seconds=seconds_before+time.perf_counter()-began,**extra))
    try:
        for i in ids:
            if i in done:continue
            m=meta[i];a=m['offset'];n=m['tokens'];sl=slice(m['step_start'],m['step_stop'])
            raw=np.asarray(bundle.features[a:a+n],float)[:,bundle.columns]
            z=(raw-bundle.mean[i])/bundle.scale[i]
            if outer:
                record=tcndb.execute('SELECT tokens FROM answers WHERE idx=?',(i,)).fetchone()
                if record is None:raise ValueError('Missing saved outer TCN prediction')
                with np.load(io.BytesIO(record[0])) as f:mu=f['real__mean']
                r=np.column_stack(((z-ridge_prediction(z,ridge)).mean(1),(z-mu).mean(1),
                    (z-bocpd_mean(z)).mean(1),(z-noreset_mean(z)).mean(1),(z-past_mean(z)).mean(1)))
                cache[a:a+n]=r
            else:
                r=np.array(cache[a:a+n],copy=True)
                if not np.isfinite(r).all():raise ValueError('Outer residual cache incomplete')
                r[:,0]=(z-ridge_prediction(z,ridge)).mean(1)
                r[:,1]=(z-tcn_predict(bundle,i,tcn)).mean(1)
            spans=np.asarray(bundle.spans[sl])-a
            scores,diag,single=fuse(r,spans,base[sl])
            deltas=[]
            for j,name in enumerate(PREDICTORS):
                expected=(ridge_ref if name=='ridge' else tcn_ref if name=='tcn' else references[name])[sl]
                np.testing.assert_allclose(single[:,j],expected,atol=2e-10,rtol=0)
                deltas.append(float(np.max(np.abs(single[:,j]-expected))))
            diag['singleton_max_delta']=max(deltas);diag['uid']=m['uid']
            diag['canonical_weight_delta']=None
            if outer and ids.index(i)<10:
                Z=(r-r.mean(0))/r.std(0);C=Z.T@Z/len(Z);delta=0.
                for subset in SUBSETS:
                    c=C[np.ix_(subset,subset)];w,_=fast_iu(c);ref=upcr_fit_covariance(c,**IU_FIT_DEFAULTS)
                    np.testing.assert_allclose(w,ref.w,rtol=2e-7,atol=2e-7)
                    delta=max(delta,float(np.max(np.abs(w-ref.w))))
                diag['canonical_weight_delta']=delta
            buf=io.BytesIO();np.savez_compressed(buf,scores=scores)
            db.execute('INSERT INTO answers VALUES (?,?,?)',(i,buf.getvalue(),json.dumps(diag,allow_nan=False)))
            done.add(i)
            if len(done)%100==0:
                if outer:cache.flush()
                db.commit();state('SCORING');print(key(excluded),len(done),len(ids),flush=True)
        if outer:cache.flush()
        db.commit();flat=np.full((len(base),len(METHODS)),np.nan)
        audit=dict(status='PASS',answers=len(ids),singleton_max_delta=0.,canonical_weight_delta=0.)
        for i,blob,diag in db.execute('SELECT idx,scores,diagnostic FROM answers ORDER BY idx'):
            m=meta[i]
            with np.load(io.BytesIO(blob)) as f:flat[m['step_start']:m['step_stop']]=f['scores']
            d=json.loads(diag);audit['singleton_max_delta']=max(audit['singleton_max_delta'],d['singleton_max_delta'])
            audit['canonical_weight_delta']=max(audit['canonical_weight_delta'],d['canonical_weight_delta'] or 0)
        if len(done)!=len(ids):raise ValueError('Missing answers')
        np.savez_compressed(folder/'STEP_SCORES.npz',scores=flat)
        audit.update(scores_sha256=sha(folder/'STEP_SCORES.npz'),sqlite_sha256=sha(folder/'ANSWERS.sqlite'))
        write(folder/'AUDIT.json',audit);state('SCORED')
    except BaseException as error:
        if outer:cache.flush()
        db.commit();state('FAILED',error=repr(error));raise
    finally:
        db.close()
        if tcndb:tcndb.close()

def execute(index):
    folder=OUT/key(JOBS[index]);folder.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (folder/'RUN.log').open('a',encoding='utf8') as f:
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--job',str(index)],stdout=f,stderr=subprocess.STDOUT,env=env)
    if result.returncode:raise RuntimeError('Scoring failed: '+str(folder/'RUN.log'))
    return key(JOBS[index])

def run():
    OUT.mkdir(exist_ok=True);bundle=FeatureBundle(DATA,'innovation5')
    from scripts.evaluate_tcn_aligned_study import verify_inputs
    verify_inputs(bundle)
    if len(bundle.metadata)!=13769 or int(bundle.manifest['steps'])!=145597:raise ValueError('Full roster required')
    sources=[Path(__file__),ROOT/'spectral_utils/predictor_subset_fusion.py',
        ROOT/'docs/experiments/PREDICTOR_SUBSET_IU_20260915.md',
        ROOT/'spectral_utils/upcr.py',ROOT/'spectral_utils/laplacian_upcr.py',
        ROOT/'spectral_utils/context_training.py',ROOT/'spectral_utils/temporal_context_models.py',
        ROOT/'spectral_utils/aligned_context_predictors.py',ROOT/'scripts/run_aligned_context_predictors.py']
    manifest=dict(schema='predictor-subset-iu-v1',jobs=JOBS,predictors=PREDICTORS,methods=METHODS,
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        data_manifest_sha256=sha(DATA/'MANIFEST.json'),
        reference_audit_sha256=sha(ROOT/'results/tcn_aligned_predictor_seed0_v1/AUDIT.json'),
        old_flow_state_sha256=sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json'),
        answers=len(bundle.metadata),tokens=int(bundle.length.sum()),correctness_labels_used=False)
    manifest=json.loads(json.dumps(manifest));mp=OUT/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Study manifest drift')
    statepath=OUT/'RUN_STATE.json'
    if statepath.exists() and json.loads(statepath.read_text()).get('status') in ('SCORED_PENDING_EVALUATION','COMPLETE_REVIEWED'):
        print('Already fully scored; preserving completed artifacts.');return
    write(mp,manifest)
    cp=OUT/'RESIDUALS.npy'
    if not cp.exists():
        cache=np.lib.format.open_memmap(cp,mode='w+',dtype=np.float64,shape=(int(bundle.length.sum()),5))
        cache[:]=np.nan;cache.flush();del cache
    previous=json.loads(statepath.read_text()) if statepath.exists() else {}
    seconds_before=previous.get('seconds',0);began=time.perf_counter();completed=[];active={}
    def state(status,**extra):
        write(statepath,dict(status=status,completed=len(completed),expected=15,completed_keys=completed,
            active_keys=list(active.values()),seconds=seconds_before+time.perf_counter()-began,**extra))
    try:
        with QueueLock(OUT/'QUEUE.lock'),QueueLock(ROOT/'results/temporal_neural_queue_seed0_v1/QUEUE.lock'):
            for indices in (range(5),range(5,15)):
                pending=list(indices)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    while pending or active:
                        while pending and len(active)<3:
                            index=pending.pop(0);p=OUT/key(JOBS[index])/'RUN_STATE.json'
                            if p.exists() and json.loads(p.read_text()).get('status')=='SCORED':
                                completed.append(key(JOBS[index]));continue
                            active[pool.submit(execute,index)]=key(JOBS[index])
                        state('SCORING')
                        if not active:continue
                        finished,_=wait(active,timeout=20,return_when=FIRST_COMPLETED)
                        for future in finished:
                            name=active.pop(future);future.result();completed.append(name)
                            print('[subset-done]',name,len(completed),'/15',flush=True)
        state('SCORED_PENDING_EVALUATION')
    except BaseException as error:
        state('FAILED',error=repr(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=int);args=p.parse_args()
    with threadpool_limits(limits=1):
        run() if args.job is None else worker(args.job)
