"""Matched token-Top10 coefficient updates; full grouped development test."""
import argparse,io,json,os,sqlite3,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base,csv_write,METRIC_KEYS
from scripts import run_rbm_logit_readout as boot
from spectral_utils.higher_moment_fusion import representation_order
from spectral_utils.direct_probability_fusion import zscore_columns,step_top_mean
from spectral_utils.matched_rbm_coefficient_update import MatchedObjective,Top10,fit

OUT=ROOT/'results/rbm_supervision_matched_v1'
METHODS=('unsupervised_update','supervised_update')


def parent(source):return source/'.worktrees/rbm-position-fusion-v1/results/rbm_position_fusion_v1_overlap_fix'
def modeldb(source):return source/'.worktrees/higher-moment-fusion-v1/results/higher_moment_fusion_v1/CHECKPOINT.sqlite'


def setup(source):
    base.old.configure_source_root(source)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text())['records']
    with np.load(base.old.BENCH/'evaluation/JOINED.npz') as z:joined={k:z[k] for k in z.files}
    fmap=json.loads(base.old.FOLDS.read_text())['outer'];folds=np.array([int(fmap[r['group_id']]) for r in records])
    with np.load(parent(source)/'SCORES.npz') as z:reference=z['steps__rbm12__logit_old']
    assert len(records)==13769
    return records,joined,folds,reference


def make_manifest(source):
    assert json.loads((parent(source)/'RUN_STATE.json').read_text())['status']=='COMPLETE'
    assert json.loads((parent(source)/'RESULT_REVIEW.json').read_text())['status']=='PASS'
    paths=[Path(__file__),ROOT/'spectral_utils/matched_rbm_coefficient_update.py',ROOT/'scripts/test_matched_rbm_coefficient_update.py',
        ROOT/'scripts/verify_matched_rbm_supervision.py',ROOT/'docs/experiments/RBM_SUPERVISION_MATCHED_V1.md',
        ROOT/'scripts/run_direct_probability_temporal.py',modeldb(source),parent(source)/'MANIFEST.json',parent(source)/'METRICS.json',
        parent(source)/'SCORES.npz',base.old.BENCH/'evaluation/JOINED.json',base.old.BENCH/'evaluation/JOINED.npz',base.old.FOLDS,
        source/'results/fusion_fixed_gate_v1/DETECTORS.npz',source/'results/fusion_fixed_gate_v1/METRICS.json']
    return dict(schema='matched-rbm-supervision-v1',base='f7203a8a6',methods=METHODS,
        scope='Same answer-specific frozen RBM, same correction parameterization; only training loss differs. Both use other training answers.',
        hashes={str(p):base.old.sha256_file(p) for p in paths})


def prepare(source,records,joined,reference):
    src=sqlite3.connect(modeldb(source).as_uri()+'?mode=ro',uri=True);detector,_=base.old._gate_contract(records)
    total=0;maxdiff=0.
    for cell,path,kind,dataset in base.source_specs():
        dest=OUT/f'cache_{cell}.npz'
        if dest.exists():
            with np.load(dest) as z:total+=len(z['ids'])
            continue
        indices=[i for i,r in enumerate(records) if r['cell']==cell]
        print('[cache load]',cell,flush=True);rows=base.old._source_row_map(base.old.load_pickle(path),kind=kind,dataset=dataset)
        xs=[];bases=[];spans_list=[];aa=[];ww=[];bb=[];orient=[];active=[];token_offsets=[0];step_offsets=[0];labels=[]
        for i in indices:
            r=records[i];row=rows[r['row_id']];spans=np.asarray(row['step_token_spans'],int)
            lp=np.asarray(base.old._topk_payload(row)['logprobs'],float)
            with np.load(base.old.BENCH/'scores'/f'{r["uid"]}.npz') as z:
                np.testing.assert_array_equal(spans[:,0],z['step_starts']);np.testing.assert_array_equal(spans[:,1],z['step_ends'])
            if kind=='pb':np.testing.assert_allclose(np.mean(row['token_entropies']),detector[i],atol=1e-12,rtol=0)
            blob,info=src.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone();info=json.loads(info)
            assert info['uid']==r['uid'];d=info['diagnostics']['d6__rbm']
            x=representation_order(lp,row['token_spilled_energies'],6);z,keep,mean,scale=zscore_columns(x)
            np.testing.assert_array_equal(mean,d['normalization_mean']);np.testing.assert_array_equal(scale,d['normalization_scale'])
            np.testing.assert_array_equal(np.flatnonzero(keep),d['columns'])
            with np.load(io.BytesIO(blob)) as state:
                a,w,b=state['d6__rbm::a'],state['d6__rbm::w'],float(state['d6__rbm::b'])
            o=d['orientation'];assert o in (-1,1);ell=o*(b+z@w)
            sl=slice(joined['offsets'][i],joined['offsets'][i+1]);replay=step_top_mean(ell,spans[:,0],spans[:,1],10)
            np.testing.assert_array_equal(replay,reference[sl])
            full=np.zeros((len(z),12));full[:,keep]=z;av=np.zeros(12);wv=np.zeros(12);av[keep]=a;wv[keep]=w
            xs.append(full);bases.append(ell);spans_list.append(spans+token_offsets[-1]);aa.append(av);ww.append(wv);bb.append(b);orient.append(o);active.append(keep)
            token_offsets.append(token_offsets[-1]+len(z));step_offsets.append(step_offsets[-1]+len(spans))
            if kind=='pb':
                target=int(joined['target'][i]);y=np.zeros(len(spans),int)
                if target>=0:y[target]=1;y[target+1:]=-1
            else:y=joined['labels'][sl]
            labels.append(y);total+=1
        out=dict(ids=np.array(indices),uid=np.array([records[i]['uid'] for i in indices]),x=np.concatenate(xs),base=np.concatenate(bases),
            spans=np.concatenate(spans_list),a=np.asarray(aa),w=np.asarray(ww),b=np.asarray(bb),orientation=np.asarray(orient),active=np.asarray(active),
            token_offsets=np.array(token_offsets),step_offsets=np.array(step_offsets),labels=np.concatenate(labels))
        np.savez_compressed(dest,**out)
        base.atomic_json(OUT/'RUN_STATE.json',dict(status='PREPARING',completed_answers=total,expected=13769,pid=os.getpid()))
        print('[cache saved]',cell,'answers',len(indices),'tokens',len(out['x']),flush=True)
        del rows,out,xs,bases,spans_list
    src.close();assert total==13769
    base.atomic_json(OUT/'ZERO_UPDATE_REVIEW.json',dict(status='PASS',answers=total,scope='Bit-exact original token Logit/Top10 replay while building fixed input caches.'))


def subset(cache,indices):
    offsets=[0];step_offsets=[0];tokens=[];bases=[];spans=[];ys=[]
    for i in indices:
        a,b=cache['token_offsets'][i:i+2];s,e=cache['step_offsets'][i:i+2]
        tokens.append(cache['x'][a:b]);bases.append(cache['base'][a:b]);spans.append(cache['spans'][s:e]-a+offsets[-1]);ys.append(cache['labels'][s:e])
        offsets.append(offsets[-1]+b-a);step_offsets.append(step_offsets[-1]+e-s)
    return dict(x=np.concatenate(tokens),base=np.concatenate(bases),spans=np.concatenate(spans),labels=np.concatenate(ys),
        offsets=np.array(offsets),step_owner=np.repeat(np.arange(len(indices)),np.diff(step_offsets)),
        **{k:cache[k][indices] for k in ('a','w','b','orientation','active')})


def fit_all(records,joined,folds):
    scores={m:np.full(int(joined['offsets'][-1]),np.nan) for m in METHODS};thresholds={m:{} for m in METHODS};health=[]
    for path in sorted(OUT.glob('cache_*.npz')):
        cell=path.stem[6:]
        with np.load(path) as z:cache={k:z[k] for k in z.files}
        ids=cache['ids'];top=Top10(cache['x'],cache['spans'])
        for fold in sorted(set(folds)):
            train=np.flatnonzero(folds[ids]!=fold);test=np.flatnonzero(folds[ids]==fold)
            assert not {records[i]['group_id'] for i in ids[train]}&{records[i]['group_id'] for i in ids[test]}
            dest=OUT/f'fit_{cell}_{fold}.npz';jp=dest.with_suffix('.json')
            if not dest.exists():
                objective=MatchedObjective(**subset(cache,train));result={};meta=dict(cell=cell,fold=int(fold),methods={})
                for method,fun in [('unsupervised_update',objective.unsupervised),('supervised_update',objective.supervised)]:
                    t=time.perf_counter();delta,info=fit(fun,12);info['seconds']=time.perf_counter()-t
                    result[method]=delta;meta['methods'][method]=info
                    print('[fit]',cell,int(fold),method,info['iterations'],'iterations',round(info['seconds'],1),'s',flush=True)
                result.update(train_ids=ids[train],test_ids=ids[test])
                base.atomic_json(jp,meta);np.savez_compressed(dest,**result);del objective
            meta=json.loads(jp.read_text())
            with np.load(dest) as z:
                np.testing.assert_array_equal(z['train_ids'],ids[train]);np.testing.assert_array_equal(z['test_ids'],ids[test])
                for method in METHODS:
                    delta=z[method];flat,_=top.evaluate(cache['base']+cache['x']@delta[:-1]+delta[-1])
                    for idx in test:
                        i=ids[idx];s,e=cache['step_offsets'][idx:idx+2];scores[method][joined['offsets'][i]:joined['offsets'][i+1]]=flat[s:e]
                    if not cell.startswith('pb_'):
                        cal=np.concatenate([flat[cache['step_offsets'][i]:cache['step_offsets'][i+1]] for i in train])
                        thresholds[method][str(fold)]=float(np.quantile(cal,.8))
                    health.append(dict(cell=cell,fold=int(fold),method=method,**meta['methods'][method]))
            base.atomic_json(OUT/'RUN_STATE.json',dict(status='FITTING',completed_fits=len(health),expected_fits=90,pid=os.getpid()))
        del cache,top
    assert len(health)==90 and all(np.isfinite(s).all() for s in scores.values())
    base.atomic_json(OUT/'FIT_HEALTH.json',health);base.atomic_json(OUT/'CALIBRATION.json',thresholds)
    return scores,thresholds


def expressions():
    return ({'supervised_vs_unsupervised':{'supervised_update':1,'unsupervised_update':-1},
        'supervised_vs_frozen':{'supervised_update':1,'rbm12__logit_old':-1},
        'unsupervised_vs_frozen':{'unsupervised_update':1,'rbm12__logit_old':-1}}, {'supervised_vs_unsupervised'})


def evaluate(source,records,joined,folds,reference,scores,thresholds):
    rm,rp=base.evaluate_arrays(records,joined,{'rbm12__logit_old':reference})
    old=json.loads((parent(source)/'METRICS.json').read_text())['metrics']['rbm12__logit_old']
    for k in METRIC_KEYS:np.testing.assert_allclose(rm['rbm12__logit_old'][k],old[k],atol=1e-12,rtol=0)
    metrics,per=base.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True)
    metrics.update(rm);per.update(rp);scores['rbm12__logit_old']=reference
    sf=np.repeat(folds,np.diff(joined['offsets']));pm=np.repeat([not r['cell'].startswith('pb_') for r in records],np.diff(joined['offsets']))&(joined['labels']>=0)
    fau={str(f):base.old.auc(joined['labels'][pm&(sf==f)]==1,reference[pm&(sf==f)]) for f in sorted(set(folds))}
    metrics['rbm12__logit_old'].update(prm_fold_auc=float(np.mean(list(fau.values()))),prm_fold_aucs=fau)
    boot.OUT=OUT;boot.expressions=expressions;print('[bootstrap]10000 paired source-group draws',flush=True)
    contrasts=boot.bootstrap(records,joined,metrics,per)
    base.atomic_json(OUT/'METRICS.json',dict(n_answers=len(records),metrics=metrics,contrasts=contrasts))
    np.savez_compressed(OUT/'SCORES.npz',**{'steps__'+m:s for m,s in scores.items()},**{'prediction__'+m:p['prediction'] for m,p in per.items()},**{'valid__'+m:p['valid'] for m,p in per.items()})
    csv_write(OUT/'COMPARISON.csv',[dict(method=m,**{k:v[k] for k in METRIC_KEYS},prm_fold_auc=v['prm_fold_auc'],valid_answers=v['valid_answers']) for m,v in metrics.items()])
    csv_write(OUT/'PB_CELLS.csv',[dict(method=m,cell=c,**v) for m,x in metrics.items() for c,v in x['pb_cells'].items()])
    target=joined['target'];err=np.array([r['cell'].startswith('pb_') for r in records])&(target>=0);forensics={}
    for key,e in expressions()[0].items():
        pos=next(m for m,c in e.items() if c==1);neg=next(m for m,c in e.items() if c==-1)
        gained=err&(per[pos]['prediction']==target)&(per[neg]['prediction']!=target)
        lost=err&(per[pos]['prediction']!=target)&(per[neg]['prediction']==target)
        forensics[key]=dict(gained=int(gained.sum()),lost=int(lost.sum()),lost_early=int(np.sum(lost&(per[pos]['peak']<target))),lost_late=int(np.sum(lost&(per[pos]['peak']>target))))
    base.atomic_json(OUT/'FORENSICS.json',forensics)
    base.atomic_json(OUT/'RUN_STATE.json',dict(status='SCORED_AWAITING_REVIEW',answers=len(records),fits=90))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    source=args.source_root.resolve();OUT.mkdir(parents=True,exist_ok=True);records,joined,folds,reference=setup(source)
    manifest=make_manifest(source);mp=OUT/'MANIFEST.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'source or protocol changed'
    else:base.atomic_json(mp,manifest)
    with threadpool_limits(limits=1):
        prepare(source,records,joined,reference)
        if args.prepare_only:return
        scores,thresholds=fit_all(records,joined,folds);evaluate(source,records,joined,folds,reference,scores,thresholds)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);base.atomic_json(OUT/'ERROR.json',dict(error=repr(e)));raise
