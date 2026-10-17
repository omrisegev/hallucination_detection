"""Saved-parameter replay plus separate metric arithmetic for each full suite."""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from scipy.special import expit,logsumexp
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_rbm_literature_completion as run
from spectral_utils import rbm_literature_completion as model


def independent_markov(ll,A,prior,starts):
    """Separate log joint-vector implementation, versus production log odds."""
    n=len(ll);emission=np.column_stack((np.zeros(n),ll));logA=np.log(A)
    logprior=np.array([-np.logaddexp(0.,prior),-np.logaddexp(0.,-prior)])
    f=np.empty((n,2));b=np.zeros((n,2))
    for t in range(n):
        f[t]=emission[t]+(logprior if starts[t] else logsumexp(f[t-1][:,None]+logA,axis=0))
        f[t]-=logsumexp(f[t])
    for t in range(n-2,-1,-1):
        if not starts[t+1]:
            b[t]=logsumexp(logA+emission[t+1]+b[t+1],axis=1)
            b[t]-=logsumexp(b[t])
    return f[:,1]-f[:,0]+b[:,1]-b[:,0]


def replay_one(data,arrays,info,independent=False):
    _,uid,spans,anchor,banks=data;checks=0
    def get(key,name):return arrays[key+'::'+name]
    for key,d in info['models'].items():
        bank=int(key.split('_')[0][1:]);source=banks[bank];x=source['x'];p=x.shape[1]
        if d['type']=='rbm':
            _,w,b=model.unpack(get(key,'theta'),p,d['h']);ell=(x@w+b)*get(key,'signs')
            logit,post=model.mean_unit_scores(ell)
        elif d['type']=='mixture':
            theta=get(key,'theta');mu=theta[:2*p].reshape(2,p)
            lv=theta[2*p:-1].reshape(2 if d['separate'] else 1,p);eta=theta[-1]
            logpdf=-.5*np.sum(lv+(x[:,None,:]-mu)**2/np.exp(lv),axis=2)
            logit=d['orientation']*(eta+logpdf[:,1]-logpdf[:,0]);post=expit(logit)
            assert np.exp(lv).min()>=.05-1e-12
        elif d['type']=='stacked':
            _,w,b=model.unpack(get(key,'first'),p,4)
            hidden=expit((x@w+b)*get(key,'firstsign'));keep=get(key,'keep').astype(bool)
            z=(hidden[:,keep]-get(key,'mean')[keep])/get(key,'scale')[keep]
            _,w,b=model.unpack(get(key,'theta'),z.shape[1],1)
            logit=((z@w+b)*get(key,'signs'))[:,0];post=expit(logit)
        else:
            a,w,b=model.unpack(source['theta'],p,1);raw=(x@w+b)[:,0]
            prior=d['prior_logit'];starts=np.zeros(len(x),bool);starts[0]=True
            perm=np.arange(len(x))
            if d['mode']=='chain_step_reset':starts[np.unique(spans[:,0])]=True
            if d['mode']=='chain_shuffled':perm=np.random.default_rng(model.seed_for(uid,'token-permutation')).permutation(len(x))
            A=get(key,'transition');np.testing.assert_allclose(A.sum(axis=1),1.,atol=1e-14)
            if independent:l=independent_markov(raw[perm]-prior,A,prior,starts)
            else:l,_=model.markov_inference(raw[perm]-prior,A,prior,starts)
            logit=np.empty_like(l);logit[perm]=l;logit*=d['orientation'];post=expit(logit)
        for s,token in [('logit',logit),('posterior',post)]:
            # Python sort is independent of production numpy partition Top10.
            step=np.array([np.mean(sorted(token[a:b],reverse=True)[:10]) for a,b in spans])
            np.testing.assert_allclose(step,arrays['score::'+key+'_'+s],atol=2e-10,rtol=2e-10,err_msg=uid+':'+key+':'+s)
            checks+=1
    for key in info['failures']:
        for suffix in ('logit','posterior'):assert np.isnan(arrays['score::'+key+'_'+suffix]).all()
    return checks


def pb_harmonic(clean_accuracy,error_accuracy):
    """Return a fraction, matching stored METRICS; percent conversion is display-only."""
    total=clean_accuracy+error_accuracy
    return 2*clean_accuracy*error_accuracy/total if total else 0.


def auc_separate(y,s):
    y=np.asarray(y,bool);n1=y.sum();n0=len(y)-n1
    if not n1 or not n0:return np.nan
    return float((rankdata(s,method='average')[y].sum()-n1*(n1+1)/2)/(n1*n0))


def review_metrics(source,out,records,joined):
    metrics=json.loads((out/'METRICS.json').read_text())['metrics']
    offsets=joined['offsets'];target=joined['target'];labels=joined['labels']
    cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_')
    detector,thr=run.base.old._gate_contract(records)
    fmap=json.loads(run.base.old.FOLDS.read_text())['outer'];folds=np.array([fmap[r['group_id']] for r in records])
    rawmeta={str(r['idx']):r for r in run.base.old.load_pickle(run.base.old.PRMB_LABELS).values()}
    checked=0
    with np.load(out/'SCORES.npz') as arrays:
        for name,m in metrics.items():
            flat=arrays['steps__'+name];within=[];peaks=[];valid=[]
            for i in range(len(records)):
                s=flat[offsets[i]:offsets[i+1]];ok=bool(len(s) and np.isfinite(s).all());valid.append(ok)
                peaks.append(int(np.argmax(s)) if ok else -1)
                if not pb[i] and ok:
                    y=labels[offsets[i]:offsets[i+1]];keep=y>=0
                    a=auc_separate(y[keep]==1,s[keep])
                    if np.isfinite(a):within.append(a)
            valid=np.array(valid);peaks=np.array(peaks)
            dv=valid&np.isfinite(detector)&np.isfinite(thr)
            pred=np.where(detector>=thr,peaks,-1)
            np.testing.assert_array_equal(pred,arrays['prediction__'+name])
            np.testing.assert_array_equal(valid,arrays['valid__'+name])
            assert int(valid.sum())==m['valid_answers']
            cellvalues=[];qvalues={4:[],8:[]}
            for cell in sorted(set(cells[pb])):
                cc=(cells==cell)&(target<0);ee=(cells==cell)&(target>=0)
                ca=np.sum(cc&dv&(pred==target))/cc.sum();ea=np.sum(ee&dv&(pred==target))/ee.sum()
                f=pb_harmonic(ca,ea)
                cellvalues.append(f);qvalues[int(cell[-1])].append(f)
            np.testing.assert_allclose(np.mean(cellvalues),m['pb_all8'],atol=1e-12)
            for q in (4,8):np.testing.assert_allclose(np.mean(qvalues[q]),m['pb_q'+str(q)],atol=1e-12)
            if within:np.testing.assert_allclose(np.mean(within),m['prm_within'],atol=1e-12)
            assert len(within)==m['prm_within_n']
            for f,q in m['prmscore_thresholds'].items():
                train=np.flatnonzero(~pb&valid&(folds!=int(f)));test=np.flatnonzero(~pb&valid&(folds==int(f)))
                assert not {records[i]['group_id'] for i in train}&{records[i]['group_id'] for i in test}
                vals=np.concatenate([flat[offsets[i]:offsets[i+1]] for i in train])
                np.testing.assert_allclose(np.quantile(vals,.8),q,atol=1e-12)
            confusion=np.zeros(4,dtype=np.int64)  # valid-step TP,FP,TN,FN
            pooled_scores=[];pooled_truth=[]
            for i in np.flatnonzero(~pb&valid):
                sl=slice(offsets[i],offsets[i+1]);s=flat[sl]
                raw=rawmeta[str(records[i]['row_id'])]
                errors={int(v)-1 for v in raw['error_steps']}
                truth=np.array([j in errors for j in range(len(s))])
                np.testing.assert_array_equal(labels[sl]==1,truth)
                pooled_scores.extend(s);pooled_truth.extend(truth)
                if raw['classification']=='correct':continue
                q=m['prmscore_thresholds'][str(folds[i])];accept=s<q
                confusion += [np.sum(accept&~truth),np.sum(accept&truth),
                              np.sum(~accept&truth),np.sum(~accept&~truth)]
            if pooled_scores:
                np.testing.assert_allclose(auc_separate(pooled_truth,pooled_scores),m['prm_pooled'],atol=1e-12)
            tp,fp,tn,fn=(int(v) for v in confusion)
            def fmeasure(t,f,n):
                precision=t/(t+f) if t+f else -1
                recall=t/(t+n) if t+n else -1
                return 2*precision*recall/(precision+recall) if precision+recall else -1
            prm=.5*(fmeasure(tp,fp,fn)+fmeasure(tn,fn,fp))
            np.testing.assert_allclose(prm,m['prmscore_conditional'],atol=1e-12)
            if int(np.sum(~pb&valid))==int((~pb).sum()):
                np.testing.assert_allclose(prm,m['prmscore_q08'],atol=1e-12)
            else:assert m['prmscore_q08'] is None
            checked+=1
    return checked


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--suite',choices=run.SUITES,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    source=args.source_root.resolve();out=run.PROGRAM/args.suite
    records,joined,reference=run.load_contract(source)
    manifest=json.loads((out/('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json')).read_text())
    for path,digest in manifest['hashes'].items():assert run.base.old.sha256_file(Path(path))==digest,path
    con=sqlite3.connect((out/('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite')).as_uri()+'?mode=ro',uri=True)
    assert con.execute('pragma quick_check').fetchone()[0]=='ok'
    ids={i for i, in con.execute('select idx from answers')};src=sqlite3.connect(run.modeldb(source).as_uri()+'?mode=ro',uri=True)
    checked=0;n=0;start=time.perf_counter()
    with threadpool_limits(limits=1):
        for path in sorted(run.caches(source).glob('cache_*.npz')):
            with np.load(path) as z:cache={k:z[k] for k in z.files if k!='labels'}
            lengths=np.diff(cache['token_offsets']);order=np.argsort(lengths)
            independent={int(order[0]),int(order[len(order)//2]),int(order[int(.95*(len(order)-1))])}
            for k,i in enumerate(cache['ids']):
                i=int(i)
                if i not in ids:continue
                data=run.prepare_answer(k,cache,src,records,joined,reference)
                blob,info=con.execute('select payload,info from answers where idx=?',(i,)).fetchone();info=json.loads(info)
                assert info['uid']==records[i]['uid']
                with np.load(io.BytesIO(blob)) as z:
                    checked+=replay_one(data,{k:z[k] for k in z.files},info,k in independent)
                n+=1
            print('[review]',args.suite,n,'answers',flush=True)
        if not args.smoke:
            assert n==13769
            count=review_metrics(source,out,records,joined)
        else:count=0
    src.close();con.close()
    run.base.atomic_json(out/('SMOKE_REVIEW.json' if args.smoke else 'RESULT_REVIEW.json'),dict(status='PASS',
        answers=n,step_vectors_replayed=checked,metric_bundles=count,seconds=time.perf_counter()-start,
        reviewer_revision='v2: PB fraction/percentage check corrected; original reviewer retained',
        reviewer_sha256=run.base.old.sha256_file(Path(__file__)),
        scope='Full saved-state replay, separate Top10/PB/within-AUC arithmetic and fold-threshold checks; same-session review.'))
    if not args.smoke:
        run.base.atomic_json(out/'RUN_STATE.json',dict(status='COMPLETE',suite=args.suite,completed=n,expected=13769,review='PASS'))


if __name__=='__main__':main()
