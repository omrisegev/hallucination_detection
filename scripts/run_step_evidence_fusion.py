"""Full-population cached study; fitting stage never opens correctness labels."""
from pathlib import Path
import sys,json,sqlite3,io,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_research_features import prefix_innovation
from spectral_utils.step_evidence_fusion import METHODS,shape_views,score_evidence,standardize
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from scripts.run_predictor_subset_study import sha,write,JOBS,key

OUT=ROOT/'results/step_evidence_fusion_v1'
DATA=ROOT/'results/temporal_context_data_v1'
REF=ROOT/'results/tcn_aligned_predictor_seed0_v1'
MODELS=ROOT/'results/temporal_context_models_v1'


def connect(path):
    return sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True)


def extract(bundle,baseline):
    cp=OUT/'SHAPES.sqlite';db=sqlite3.connect(cp)
    db.execute('CREATE TABLE IF NOT EXISTS answers(idx INTEGER PRIMARY KEY, payload BLOB, delta REAL)')
    done={r[0] for r in db.execute('select idx from answers')}
    src=connect(ROOT/'results/temporal_research_baseline_v1/CHECKPOINT.sqlite')
    for i,m in enumerate(bundle.metadata):
        if i in done:continue
        row=src.execute('select payload from answers where idx=?',(i,)).fetchone()
        if row is None:raise ValueError('Missing frozen answer')
        with np.load(io.BytesIO(row[0]),allow_pickle=False) as f:
            raw=np.asarray(f['features'],float);spans=f['spans']
        sl=slice(m['step_start'],m['step_stop'])
        np.testing.assert_array_equal(spans,np.asarray(bundle.spans[sl])-m['offset'])
        if len(raw)!=m['tokens']:raise ValueError('Token mismatch')
        x=np.column_stack((raw[:,:4],prefix_innovation(raw[:,0])[0]))
        b,r,s=shape_views(x,spans,m['uid'])
        delta=float(np.max(np.abs(b-baseline[sl])))
        np.testing.assert_allclose(b,baseline[sl],atol=1e-10,rtol=0)
        # Independent scalar extraction on deterministic audit answers.
        if i%137==0:
            for j,(a,c) in enumerate(spans):
                v=x[a:c];k=min(10,len(v))
                e=np.mean([max(v[t:t+k,f].mean() for t in range(len(v)-k+1)) for f in range(5)])
                np.testing.assert_allclose(r[j],[v[-min(4,len(v)):].mean(),e],atol=1e-10,rtol=0)
        buf=io.BytesIO();np.savez_compressed(buf,real=r,shuffled=s)
        db.execute('insert into answers values(?,?,?)',(i,buf.getvalue(),delta));done.add(i)
        if len(done)%500==0:
            db.commit();write(OUT/'RUN_STATE.json',dict(status='EXTRACTING',answers=len(done),expected=len(bundle.metadata)))
            print('[shapes]',len(done),flush=True)
    db.commit();src.close()
    if len(done)!=len(bundle.metadata):raise ValueError('Extra/missing checkpoint rows')
    shape=np.empty((len(baseline),2));shuffled=np.empty_like(shape)
    for i,blob in db.execute('select idx,payload from answers'):
        m=bundle.metadata[i];sl=slice(m['step_start'],m['step_stop'])
        with np.load(io.BytesIO(blob),allow_pickle=False) as f:shape[sl]=f['real'];shuffled[sl]=f['shuffled']
    delta=db.execute('select max(delta) from answers').fetchone()[0];db.close()
    np.savez_compressed(OUT/'SHAPES.npz',real=shape,shuffled=shuffled)
    return shape,shuffled,delta


def run():
    began=time.perf_counter();OUT.mkdir(exist_ok=True);bundle=FeatureBundle(DATA,'innovation5')
    paths=[Path(__file__),ROOT/'spectral_utils/step_evidence_fusion.py',
           ROOT/'spectral_utils/predictor_subset_fusion.py',ROOT/'spectral_utils/upcr.py',
           ROOT/'docs/experiments/STEP_EVIDENCE_FUSION_20260915.md',
           DATA/'MANIFEST.json',DATA/'METADATA.json',DATA/'step_spans.npy',
           ROOT/'results/temporal_research_baseline_v1/CHECKPOINT.sqlite',
           ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz',
           REF/'SCORES_FROZEN.npz',REF/'AUDIT.json']
    provenance={}
    for e in JOBS:
        folder=MODELS/('tcn__innovation5__seed0__'+key(e));sp=folder/'MANIFEST.json'
        spec=json.loads(sp.read_text())
        if spec['excluded_folds']!=list(e) or spec['smoke']:raise ValueError('TCN contract drift')
        for k,ids in zip(('training','validation','held'),bundle.split(e)):
            if set(spec[k+'_groups'])!={bundle.metadata[i]['group_id'] for i in ids}:raise ValueError('Group split drift')
        p=folder/'scoring/STEP_SCORES.npz'
        provenance[key(e)]=dict(manifest_sha256=sha(sp),scores_sha256=sha(p))
    manifest=dict(schema='step-evidence-fusion-v1',methods=METHODS,
        hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        predictor_scores=provenance,labels_in_fit=False,
        flow_state_sha256=sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json'))
    manifest=json.loads(json.dumps(manifest));mp=OUT/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Frozen manifest changed')
    write(mp,manifest)
    with np.load(REF/'SCORES_FROZEN.npz') as f:baseline=f['innovation5'];tcn=f['tcn__real']
    shape,shuffled,base_delta=extract(bundle,baseline)
    audit_rows=[];maxcontext=0.;maxcanonical=0.;maxreplay=0.
    for e in JOBS:
        out=OUT/(key(e)+'.npz');dp=OUT/(key(e)+'_AUDIT.json')
        if dp.exists():
            a=json.loads(dp.read_text())
            if sha(out)!=a['scores_sha256']:raise ValueError('Changed scored job')
            continue
        with np.load(MODELS/('tcn__innovation5__seed0__'+key(e))/'scoring/STEP_SCORES.npz') as f:context=f['real__signed_residual_0.25']
        ids=[i for i,m in enumerate(bundle.metadata) if m['fold'] in e and (len(e)==1 or not m['cell'].startswith('pb_'))]
        result=np.full((len(baseline),len(METHODS)),np.nan);diagnostics=[]
        for j,i in enumerate(ids):
            m=bundle.metadata[i];sl=slice(m['step_start'],m['step_stop'])
            b=baseline[sl];c=context[sl]
            s,d=score_evidence(b,c,shape[sl],shuffled[sl])
            np.testing.assert_allclose(s[:,0],c,atol=2e-10,rtol=0)
            maxcontext=max(maxcontext,float(np.max(np.abs(s[:,0]-c))))
            if len(e)==1:np.testing.assert_allclose(c,tcn[sl],atol=1e-12,rtol=0)
            # Independent weighted sum and standardization for every output.
            a=np.column_stack((c-b,shape[sl]));ss=np.std(a,axis=0)
            z=np.column_stack([(a[:,k]-a[:,k].mean())/(ss[k] if ss[k]>1e-12 else 1.) for k in range(3)])
            q=np.column_stack((c-b,shuffled[sl]));qs=q.std(0);zz=(q-q.mean(0))/np.where(qs>1e-12,qs,1.)
            aux=[z[:,0],z[:,1],z[:,2],sum(z[:,k]/3 for k in range(3)),
                 sum(z[:,k]*d['real']['weights'][k] for k in range(3)),
                 (z[:,0]+z[:,1])/2,(z[:,0]+z[:,2])/2,
                 sum(zz[:,k]/3 for k in range(3)),sum(zz[:,k]*d['shuffled']['weights'][k] for k in range(3))]
            expected=np.column_stack([b+.25*b.std()*(v-v.mean())/(v.std() if v.std()>1e-12 else 1.) for v in aux])
            np.testing.assert_allclose(s,expected,atol=2e-10,rtol=0)
            maxreplay=max(maxreplay,float(np.max(np.abs(s-expected))))
            if j<10 and d['real']['native']:
                C=z.T@z/len(z);ref=upcr_fit_covariance(C,**IU_FIT_DEFAULTS).w
                if ref@C@np.ones(3)<0:ref=-ref
                np.testing.assert_allclose(d['real']['weights'],ref,atol=2e-7,rtol=2e-7)
                maxcanonical=max(maxcanonical,float(np.max(np.abs(np.asarray(d['real']['weights'])-ref))))
            result[sl]=s;d.update(idx=i,uid=m['uid'],steps=len(b));diagnostics.append(d)
        np.savez_compressed(out,scores=result,ids=ids)
        write(OUT/(key(e)+'_DIAGNOSTICS.json'),diagnostics)
        write(dp,dict(status='PASS',answers=len(ids),scores_sha256=sha(out),context_max_delta=maxcontext,
            canonical_max_delta=maxcanonical,independent_score_max_delta=maxreplay))
        write(OUT/'RUN_STATE.json',dict(status='SCORING',job=key(e),answers=len(ids),elapsed_seconds=time.perf_counter()-began))
        print('[evidence]',key(e),len(ids),round(time.perf_counter()-began,1),flush=True)
    write(OUT/'SCORING_AUDIT.json',dict(status='PASS',answers=len(bundle.metadata),steps=len(baseline),
        base_max_delta=base_delta,jobs={key(e):json.loads((OUT/(key(e)+'_AUDIT.json')).read_text()) for e in JOBS},
        flow_state_unchanged=manifest['flow_state_sha256']==sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json')))
    write(OUT/'RUN_STATE.json',dict(status='SCORED_PENDING_EVALUATION',answers=len(bundle.metadata),elapsed_seconds=time.perf_counter()-began))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
