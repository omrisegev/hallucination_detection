"""Resumable full-population, label-free RBM mechanism experiments."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base, csv_write, METRIC_KEYS
from spectral_utils.direct_probability_fusion import step_top_mean, zscore_columns
from spectral_utils import rbm_literature_completion as model

PROGRAM=ROOT/'results/rbm_literature_completion_v1'
SUITES=('variance','capacity','stability','depth','temporal')
VARIANTS={
    'variance':('variance_shared','variance_separate'),
    'capacity':('exact1','cd1','exact4','cd4'),
    'stability':('best_exact1','best_exact4'),
    'depth':('layer2_exact','layer2_cd'),
    'temporal':('chain_full','chain_step_reset','chain_shuffled')}
REFS=('rbm6__old','rbm6__logit_old','rbm12__old','rbm12__logit_old',
      'initial6__old','initial12__old','entropy__old','var15__old','var50__old',
      'var15_iu__old','var15_equal__old','shrinkage__old','diagonal__old')


def methods(suite):
    return tuple(f'b{bank}_{v}_{s}' for bank in (6,12) for v in VARIANTS[suite]
                 for s in ('logit','posterior'))


def parent(source):
    return source/'.worktrees/rbm-logit-readout-v1/results/rbm_logit_readout_v1'


def caches(source):
    return source/'.worktrees/rbm-supervision-matched-v1/results/rbm_supervision_matched_v1'


def modeldb(source):
    return source/'.worktrees/higher-moment-fusion-v1/results/higher_moment_fusion_v1/CHECKPOINT.sqlite'


def load_contract(source):
    base.old.configure_source_root(source)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text())['records']
    with np.load(base.old.BENCH/'evaluation/JOINED.npz') as z:
        joined={k:z[k] for k in z.files}
    with np.load(parent(source)/'SCORES.npz') as z:
        reference={m:z['steps__'+m] for m in REFS}
    assert len(records)==13769 and len({r['uid'] for r in records})==13769
    folds=json.loads(base.old.FOLDS.read_text())['outer']
    assert all(r['group_id'] in folds for r in records)
    return records,joined,reference


def manifest_for(source,suite):
    paths=[Path(__file__),ROOT/'spectral_utils/rbm_literature_completion.py',
        ROOT/'scripts/test_rbm_literature_completion.py',
        ROOT/'scripts/review_rbm_literature_completion.py',
        ROOT/'docs/experiments/RBM_LITERATURE_COMPLETION_V1.md',
        ROOT/'scripts/run_direct_probability_temporal.py',
        ROOT/'scripts/run_direct_probability_fusion_v2.py',
        ROOT/'spectral_utils/direct_probability_fusion.py',
        base.old.BENCH/'evaluation/JOINED.json',base.old.BENCH/'evaluation/JOINED.npz',
        base.old.FOLDS,base.old.FIXED_GATE/'DETECTORS.npz',base.old.FIXED_GATE/'METRICS.json',
        modeldb(source),parent(source)/'SCORES.npz',parent(source)/'METRICS.json',
        parent(source)/'RESULT_REVIEW.json',caches(source)/'MANIFEST.json',
        caches(source)/'ZERO_UPDATE_REVIEW.json']
    paths+=sorted(caches(source).glob('cache_*.npz'))
    assert len(list(caches(source).glob('cache_*.npz')))==9
    if suite in ('depth','stability'):
        state=json.loads((PROGRAM/'capacity/RUN_STATE.json').read_text())
        assert state['status']=='COMPLETE', 'capacity must finish and pass review first'
        paths += [PROGRAM/'capacity/CHECKPOINT.sqlite',PROGRAM/'capacity/RESULT_REVIEW.json']
    return dict(schema='rbm-literature-completion-v1',suite=suite,
        base='de237a3622c776f2cfd866b39e0b0de3ca30fe90',methods=list(methods(suite)),
        banks=[6,12],fit_scope='current answer only; no labels',bootstrap=10000,
        hashes={str(p):base.old.sha256_file(p) for p in paths})


def connect(path,manifest):
    con=sqlite3.connect(path)
    con.execute('pragma journal_mode=WAL')
    con.execute('create table if not exists manifest (id integer primary key,payload text)')
    con.execute('create table if not exists answers (idx integer primary key,payload blob,info text)')
    old=con.execute('select payload from manifest where id=1').fetchone()
    if old:
        if json.loads(old[0])!=manifest:
            con.close()
            raise AssertionError('frozen manifest changed')
    else:
        con.execute('insert into manifest values (1,?)',(base.dumps(manifest),));con.commit()
    return con


def prepare_answer(k,cache,src,records,joined,reference):
    i=int(cache['ids'][k]);uid=records[i]['uid']
    assert str(cache['uid'][k])==uid
    ta,tb=cache['token_offsets'][k:k+2];sa,sb=cache['step_offsets'][k:k+2]
    spans=cache['spans'][sa:sb]-ta
    assert spans.shape==(records[i]['steps'],2)
    assert np.all(spans[:,0]>=0) and np.all(spans[:,1]<=tb-ta)
    assert np.all(spans[:,1]>spans[:,0])
    blob,info=src.execute('select payload,info from answers where idx=?',(i,)).fetchone()
    info=json.loads(info);assert info['uid']==uid
    xfull=cache['x'][ta:tb];banks={}
    sl=slice(joined['offsets'][i],joined['offsets'][i+1])
    with np.load(io.BytesIO(blob)) as states:
        for bank,degree in ((6,3),(12,6)):
            key=f'd{degree}__rbm';d=info['diagnostics'][key];cols=np.array(d['columns'])
            x=np.ascontiguousarray(xfull[:,cols])
            assert np.isfinite(x).all() and x.shape[1]>=3
            np.testing.assert_allclose(x.mean(axis=0),0.,atol=1e-10)
            np.testing.assert_allclose(x.std(axis=0),1.,atol=1e-10)
            if bank==6:
                d12=info['diagnostics']['d6__rbm']
                np.testing.assert_array_equal(d['normalization_mean'],d12['normalization_mean'][:6])
                np.testing.assert_array_equal(d['normalization_scale'],d12['normalization_scale'][:6])
            a,w,b=states[key+'::a'],states[key+'::w'],float(states[key+'::b'])
            raw=b+x@w;o=d['orientation']
            assert o in (-1,1)
            for s,score in [('old',expit(o*raw)),('logit_old',o*raw)]:
                step=step_top_mean(score,spans[:,0],spans[:,1],10)
                np.testing.assert_allclose(step,reference[f'rbm{bank}__{s}'][sl],atol=1e-12,rtol=1e-12)
            banks[bank]=dict(x=x,theta=model.pack(a,w[:,None],[b]),orientation=o,columns=cols)
    anchor=xfull[:,np.array(info['diagnostics']['d3__rbm']['columns'])].mean(axis=1)
    return i,uid,spans,anchor,banks


def worker(task):
    suite,data,capacity=task
    i,uid,spans,anchor,banks=data
    arrays={};meta=dict(uid=uid,n_tokens=len(anchor),models={},failures={})
    def save(bank,variant,logit,posterior,state,details):
        key=f'b{bank}_{variant}'
        if not np.isfinite(logit).all() or not np.isfinite(posterior).all():
            raise ValueError('nonfinite token score')
        for name,score in [('logit',logit),('posterior',posterior)]:
            arrays['score::'+key+'_'+name]=step_top_mean(score,spans[:,0],spans[:,1],10)
        for name,value in state.items():arrays[key+'::'+name]=np.asarray(value)
        meta['models'][key]=details
    for bank,source in banks.items():
        x=source['x'];p=x.shape[1]
        a,w,b=model.unpack(source['theta'],p,1);w=w[:,0];b=float(b[0])
        for variant in VARIANTS[suite]:
            key=f'b{bank}_{variant}'
            try:
                if suite=='variance':
                    sep=variant=='variance_separate'
                    ell,theta,diag=model.variance_fit(x,a,w,b,sep)
                    post=expit(ell);o=-1 if np.std(post)>1e-12 and np.corrcoef(post,anchor)[0,1]<0 else 1
                    save(bank,variant,o*ell,expit(o*ell),dict(theta=theta),
                         dict(type='mixture',separate=sep,orientation=o,**diag))
                elif suite=='capacity':
                    h=int(variant[-1]);seed=model.seed_for(uid,f'bank{bank}:init{h}') if h==4 else 0
                    init=model.initial(p,h,seed)
                    if variant.startswith('exact'):
                        theta,diag=model.exact_fit(x,h,theta=init)
                    else:
                        theta,diag=model.cd_fit(x,h,theta=init,seed=model.seed_for(uid,f'bank{bank}:cd{h}'))
                    ell,signs=model.oriented_units(x,theta,h,anchor)
                    logit,post=model.mean_unit_scores(ell)
                    save(bank,variant,logit,post,dict(theta=theta,signs=signs),dict(type='rbm',h=h,**diag))
                elif suite=='stability':
                    h=int(variant[-1]);basekey=f'b{bank}_exact{h}'
                    theta0=capacity[0][basekey+'::theta'];obj=model.ExactRBM(x,h)
                    candidates=[theta0];stats=[dict(start=0,nll_final=obj(theta0)[0])]
                    for restart in (1,2):
                        seed=model.seed_for(uid,f'bank{bank}:h{h}:restart{restart}')
                        theta,diag=model.exact_fit(x,h,seed=seed)
                        candidates.append(theta);stats.append(dict(start=restart,**diag))
                    chosen=int(np.argmin([obj(t)[0] for t in candidates]));theta=candidates[chosen]
                    scores=[];peaks=[]
                    for t in candidates:
                        u,_=model.oriented_units(x,t,h,anchor);l,_=model.mean_unit_scores(u)
                        scores.append(l);peaks.append(int(np.argmax(step_top_mean(l,spans[:,0],spans[:,1],10))))
                    corr=np.corrcoef(scores)
                    ell,signs=model.oriented_units(x,theta,h,anchor);logit,post=model.mean_unit_scores(ell)
                    save(bank,variant,logit,post,dict(theta=theta,signs=signs,starts=np.stack(candidates)),
                         dict(type='rbm',h=h,chosen_start=chosen,starts=stats,
                              logit_correlations=[[float(v) if np.isfinite(v) else None for v in row] for row in corr],
                              peaks=peaks,unique_peaks=len(set(peaks))))
                elif suite=='depth':
                    basekey=f'b{bank}_exact4'
                    first=capacity[0][basekey+'::theta'];_,fw,fb=model.unpack(first,p,4)
                    firstsign=capacity[0][basekey+'::signs']
                    hidden=expit((x@fw+fb)*firstsign)
                    z,keep,mean,scale=zscore_columns(hidden)
                    if z.shape[1]<3:raise ValueError('fewer than three varying hidden views')
                    if variant=='layer2_exact':theta,diag=model.exact_fit(z,1)
                    else:theta,diag=model.cd_fit(z,1,seed=model.seed_for(uid,f'bank{bank}:layer2cd'))
                    ell,signs=model.oriented_units(z,theta,1,anchor);logit,post=model.mean_unit_scores(ell)
                    save(bank,variant,logit,post,dict(theta=theta,signs=signs,first=first,firstsign=firstsign,
                         keep=keep,mean=mean,scale=scale),dict(type='stacked',h=1,first_h=4,**diag))
                elif suite=='temporal':
                    raw=x@w+b;prior=float(b+a@w+.5*w@w)
                    starts=np.zeros(len(raw),bool);starts[0]=True
                    perm=np.arange(len(raw))
                    if variant=='chain_step_reset':starts[np.unique(spans[:,0])]=True
                    elif variant=='chain_shuffled':
                        perm=np.random.default_rng(model.seed_for(uid,'token-permutation')).permutation(len(raw))
                    l,A,diag=model.markov_fit(raw[perm],prior,starts)
                    restored=np.empty_like(l);restored[perm]=l;restored*=source['orientation']
                    save(bank,variant,restored,expit(restored),dict(transition=A),
                         dict(type='markov',mode=variant,prior_logit=prior,orientation=source['orientation'],**diag))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError,RuntimeError,KeyError) as e:
                meta['failures'][key]=f'{type(e).__name__}: {e}'
                for s in ('logit','posterior'):arrays['score::'+key+'_'+s]=np.full(len(spans),np.nan)
    return i,base.packed(**arrays),base.dumps(meta)


def score(source,suite,con,records,joined,reference,workers,smoke):
    done={i for i, in con.execute('select idx from answers')};start=time.perf_counter()
    src=sqlite3.connect(modeldb(source).as_uri()+'?mode=ro',uri=True)
    cap=None
    if suite in ('stability','depth'):
        cap=sqlite3.connect((PROGRAM/'capacity/CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    out=PROGRAM/suite;statepath=out/('SMOKE_STATE.json' if smoke else 'RUN_STATE.json')
    with ProcessPoolExecutor(max_workers=workers,initializer=base.worker_init) as pool:
        for path in sorted(caches(source).glob('cache_*.npz')):
            with np.load(path) as z:cache={k:z[k] for k in z.files if k!='labels'}
            ids=cache['ids'];todo=[k for k,i in enumerate(ids) if int(i) not in done]
            if smoke:
                lengths=np.diff(cache['token_offsets']);ordered=np.argsort(lengths)
                chosen={int(ordered[0]),int(ordered[len(ordered)//2]),int(ordered[int(.95*(len(ordered)-1))])}
                todo=[k for k in todo if k in chosen]
            print('[cell]',suite,path.stem,'remaining',len(todo),flush=True)
            for begin in range(0,len(todo),16):
                tasks=[]
                for k in todo[begin:begin+16]:
                    data=prepare_answer(k,cache,src,records,joined,reference);extra=None
                    if cap is not None:
                        row=cap.execute('select payload,info from answers where idx=?',(data[0],)).fetchone()
                        with np.load(io.BytesIO(row[0])) as z:states={name:z[name] for name in z.files if not name.startswith('score::')}
                        extra=(states,json.loads(row[1]))
                    tasks.append((suite,data,extra))
                for row in pool.map(worker,tasks,chunksize=1):
                    con.execute('insert into answers values (?,?,?)',row);done.add(row[0])
                con.commit()
                state=dict(status='SMOKE' if smoke else 'RUNNING',suite=suite,completed=len(done),expected=13769,
                           pid=os.getpid(),seconds=time.perf_counter()-start)
                base.atomic_json(statepath,state)
                print('[checkpoint]',suite,len(done),'13769',round(state['seconds'],1),'seconds',flush=True)
    src.close()
    if cap is not None:cap.close()


def comparisons(suite):
    pairs=[];primary=set()
    for bank in (6,12):
        main='posterior' if bank==6 else 'logit'
        for s in ('posterior','logit'):
            ref=f'rbm{bank}__'+('old' if s=='posterior' else 'logit_old')
            for v in VARIANTS[suite]:pairs.append((f'b{bank}_{v}_{s}',ref))
            if suite=='variance':a,b=f'b{bank}_variance_separate_{s}',f'b{bank}_variance_shared_{s}'
            elif suite=='capacity':
                a,b=f'b{bank}_exact4_{s}',f'b{bank}_exact1_{s}'
                pairs += [(f'b{bank}_cd{h}_{s}',f'b{bank}_exact{h}_{s}') for h in (1,4)]
            elif suite=='stability':a,b=f'b{bank}_best_exact4_{s}',f'b{bank}_exact4_{s}'
            elif suite=='depth':a,b=f'b{bank}_layer2_exact_{s}',f'b{bank}_exact4_{s}'
            else:
                a,b=f'b{bank}_chain_full_{s}',f'b{bank}_chain_shuffled_{s}'
                pairs.append((a,f'b{bank}_chain_step_reset_{s}'))
            pairs.append((a,b))
            if s==main:primary.add((a,b))
    return list(dict.fromkeys(pairs)),primary


def evaluate(source,suite,con,records,joined,reference):
    out=PROGRAM/suite;scores=dict(reference);health=[]
    for m in methods(suite):scores[m]=np.full(int(joined['offsets'][-1]),np.nan)
    if suite in ('depth','stability'):
        with np.load(PROGRAM/'capacity/SCORES.npz') as z:
            for bank in (6,12):
                for h in (1,4):
                    for s in ('posterior','logit'):
                        key=f'b{bank}_exact{h}_{s}';scores[key]=z['steps__'+key]
    for i,blob,info in con.execute('select idx,payload,info from answers order by idx'):
        d=json.loads(info);assert d['uid']==records[i]['uid'];health.append(dict(idx=i,**d))
        with np.load(io.BytesIO(blob)) as z:
            for m in methods(suite):scores[m][joined['offsets'][i]:joined['offsets'][i+1]]=z['score::'+m]
    assert len(health)==13769
    print('[metrics]',suite,len(scores),'configurations',flush=True)
    metrics,per=base.evaluate_arrays(records,joined,scores)
    old=json.loads((parent(source)/'METRICS.json').read_text())['metrics']
    for m in REFS:
        for k in METRIC_KEYS:np.testing.assert_allclose(metrics[m][k],old[m][k],atol=1e-12,rtol=0)
    pairs,primary=comparisons(suite)
    print('[bootstrap]',suite,'10000 source-group draws',flush=True)
    contrasts=base.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=primary)
    target=joined['target'];pb=np.array([r['cell'].startswith('pb_') for r in records]);error=pb&(target>=0)
    changes=[]
    for a,b in pairs:
        c=contrasts[a+'_minus_'+b];c['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
        new=error&per[a]['decision_valid']&(per[a]['prediction']==target)
        oldhit=error&per[b]['decision_valid']&(per[b]['prediction']==target)
        gain=new&~oldhit;loss=oldhit&~new
        c.update(gained=int(gain.sum()),lost=int(loss.sum()),
                 lost_early=int(np.sum(loss&(per[a]['peak']<target))),
                 lost_late=int(np.sum(loss&(per[a]['peak']>target))),
                 lost_gate=int(np.sum(loss&(per[a]['peak']==target)&(per[a]['prediction']==-1))))
        for i in np.flatnonzero(gain|loss):
            changes.append(dict(comparison=a+'_minus_'+b,uid=records[i]['uid'],cell=records[i]['cell'],
                target=int(target[i]),before=int(per[b]['prediction'][i]),after=int(per[a]['prediction'][i]),
                peak_after=int(per[a]['peak'][i]),change='gained' if gain[i] else 'lost'))
    base.atomic_json(out/'METRICS.json',dict(suite=suite,n_answers=13769,metrics=metrics,contrasts=contrasts,
        scope='Full cached development; unlabeled answer-local fusion, external fixed entropy gate.'))
    base.atomic_json(out/'FIT_HEALTH.json',health)
    np.savez_compressed(out/'SCORES.npz',**{'steps__'+m:s for m,s in scores.items()},
        **{'prediction__'+m:p['prediction'] for m,p in per.items()},
        **{'valid__'+m:p['valid'] for m,p in per.items()})
    csv_write(out/'COMPARISON.csv',[dict(method=m,**{k:v[k] for k in METRIC_KEYS},
        valid_answers=v['valid_answers'],prm_within_n=v['prm_within_n']) for m,v in metrics.items()])
    csv_write(out/'PB_CELLS.csv',[dict(method=m,cell=c,**v) for m,x in metrics.items() for c,v in x['pb_cells'].items()])
    csv_write(out/'CHANGED_SUCCESSES.csv',changes)
    base.atomic_json(out/'RUN_STATE.json',dict(status='SCORED_AWAITING_REVIEW',suite=suite,completed=13769,expected=13769))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--suite',choices=SUITES,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--smoke',action='store_true');p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args();source=args.source_root.resolve();suite=args.suite
    out=PROGRAM/suite;out.mkdir(parents=True,exist_ok=True)
    records,joined,reference=load_contract(source)
    print('[manifest]',suite,'verify cached states and evaluation contract',flush=True)
    manifest=manifest_for(source,suite);con=connect(out/('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'),manifest)
    base.atomic_json(out/('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'),manifest)
    with threadpool_limits(limits=1):
        if not args.evaluate_only:score(source,suite,con,records,joined,reference,args.workers,args.smoke)
        n=con.execute('select count(*) from answers').fetchone()[0]
        if args.smoke:
            failures=[json.loads(i)['failures'] for i, in con.execute('select info from answers')]
            base.atomic_json(out/'SMOKE.json',dict(status='PASS' if not any(failures) else 'FAIL',answers=n,
                             failures=failures,scope='mechanics and runtime only; no candidate ranking'))
        else:
            assert n==13769;evaluate(source,suite,con,records,joined,reference)
    con.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        PROGRAM.mkdir(parents=True,exist_ok=True)
        (PROGRAM/'LAST_ERROR.log').write_text(traceback.format_exc(),encoding='utf8')
        raise
