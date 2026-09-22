"""Full group-disjoint supervised diagnostic, using frozen step means."""
import argparse
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base,csv_write,METRIC_KEYS
from scripts import run_rbm_logit_readout as bootstrap_driver
from spectral_utils.supervised_position_diagnostic import MODES,design,listwise_loss,binary_loss,balanced_answer_weights,optimize

OUT=ROOT/'results/rbm_supervised_position_diagnostic_v1'
NEW=tuple('supervised_'+m for m in MODES)


def locations(source):
    return (source/'.worktrees/rbm-data-diagnostics-v1/results/rbm_data_diagnostics_v1',
            source/'.worktrees/rbm-position-fusion-v1/results/rbm_position_fusion_v1_overlap_fix')


def setup(source):
    base.old.configure_source_root(source)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text())['records']
    with np.load(base.old.BENCH/'evaluation/JOINED.npz') as z:joined={k:z[k] for k in z.files}
    folds=json.loads(base.old.FOLDS.read_text())['outer']
    outer=np.array([int(folds[r['group_id']]) for r in records])
    assert len(records)==13769 and len(set(outer))==5
    return records,joined,outer


def manifest(source):
    diag,prior=locations(source)
    for directory in (diag,prior):
        assert json.loads((directory/'RUN_STATE.json').read_text())['status']=='COMPLETE'
        assert json.loads((directory/'RESULT_REVIEW.json').read_text())['status']=='PASS'
    paths=[diag/'CHECKPOINT.sqlite',diag/'MANIFEST.json',prior/'METRICS.json',prior/'SCORES.npz',
           base.old.BENCH/'evaluation/JOINED.json',base.old.BENCH/'evaluation/JOINED.npz',base.old.FOLDS,
           Path(__file__),ROOT/'spectral_utils/supervised_position_diagnostic.py',
           ROOT/'scripts/run_direct_probability_temporal.py',ROOT/'scripts/test_supervised_position_diagnostic.py',
           ROOT/'scripts/verify_supervised_position_diagnostic.py',
           ROOT/'docs/experiments/RBM_SUPERVISED_POSITION_DIAGNOSTIC_V1.md']
    return dict(schema='supervised-position-diagnostic-v1',base='ce008fb1d',
                scope='Supervised, grouped outer-fold development; step means, not token Top10',
                hashes={str(p):base.old.sha256_file(p) for p in paths})


def extract(source,records,joined):
    path=OUT/'FEATURES.npz'
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    diag,_=locations(source);con=sqlite3.connect((diag/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    n=int(joined['offsets'][-1]);x=np.zeros((n,12));c=np.zeros(n);entropy=np.zeros(n);length=np.zeros(n,int)
    seen=set()
    for i,blob,info in con.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        info=json.loads(info);assert info['uid']==records[i]['uid'];a,b=joined['offsets'][i:i+2];ns=b-a
        with np.load(io.BytesIO(blob)) as z:
            cols=z['b12__columns'];sm=z['b12__step_means']
            assert sm.shape==(ns,len(cols)) and len(set(cols))==len(cols)
            x[a:b,cols]=sm;entropy[a:b]=z['step_entropy'];length[a:b]=z['lengths']
        c[a:b]=np.where(np.arange(ns)<(ns+1)//2,-1.,1.);seen.add(i)
        if len(seen)%2500==0:print('[features]',len(seen),flush=True)
    con.close();assert len(seen)==len(records) and np.isfinite(x).all() and (length>0).all()
    out=dict(x=x,context=c,entropy=entropy,length=length,uid=np.array([r['uid'] for r in records]))
    np.savez_compressed(path,**out);return out


def fold_inputs(records,joined,outer,features,cell,fold):
    offsets=joined['offsets'];cells=np.array([r['cell'] for r in records])
    train=np.flatnonzero((cells==cell)&(outer!=fold));test=np.flatnonzero((cells==cell)&(outer==fold))
    tg={records[i]['group_id'] for i in train};vg={records[i]['group_id'] for i in test};assert not tg&vg
    trainsteps=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in train])
    teststeps=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in test])
    mean=features['x'][trainsteps].mean(axis=0);sd=features['x'][trainsteps].std(axis=0);sd=np.where(sd>1e-10,sd,1.)
    x=(features['x']-mean)/sd
    return train,test,trainsteps,teststeps,mean,sd,x


def fit_all(records,joined,outer,features):
    offsets=joined['offsets'];all_scores={m:np.full(int(offsets[-1]),np.nan) for m in NEW}
    thresholds={m:{} for m in NEW};health=[];start=time.perf_counter()
    for cell in sorted({r['cell'] for r in records}):
        for fold in sorted(set(outer)):
            path=OUT/f'fit_{cell}_{fold}.npz';jp=path.with_suffix('.json')
            train,test,ts,vs,mean,sd,x=fold_inputs(records,joined,outer,features,cell,fold)
            if not path.exists():
                arrays=dict(mean=mean,scale=sd,train_answers=train,test_answers=test,train_steps=ts,test_steps=vs)
                meta=dict(cell=cell,fold=int(fold),train_groups=sorted({records[i]['group_id'] for i in train}),
                          test_groups=sorted({records[i]['group_id'] for i in test}),fits={})
                pb=cell.startswith('pb_')
                if pb:
                    eligible=train[joined['target'][train]>=0]
                    rows=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in eligible])
                    of=np.r_[0,np.cumsum([offsets[i+1]-offsets[i] for i in eligible])];y=joined['target'][eligible]
                    meta.update(training_error_answers=len(eligible),training_steps=len(rows),objective='listwise first-error')
                else:
                    rows=ts[joined['labels'][ts]>=0];y=(joined['labels'][rows]==1).astype(float)
                    owner=np.repeat(np.arange(len(records)),np.diff(offsets));weight=balanced_answer_weights(y,owner[rows])
                    meta.update(training_steps=len(rows),objective='balanced step BCE',positive_steps=int(y.sum()))
                for mode,name in zip(MODES,NEW):
                    d=design(x,features['context'],mode)
                    if pb:fun=lambda theta:listwise_loss(theta,d[rows],of,y);dims=d.shape[1]
                    else:fun=lambda theta:binary_loss(theta,d[rows],y,weight);dims=d.shape[1]+1
                    theta,info=optimize(fun,dims)
                    scores=d@theta if pb else d@theta[:-1]+theta[-1]
                    arrays[name+'_theta']=theta;arrays[name+'_test']=scores[vs]
                    if not pb:
                        arrays[name+'_calibration']=scores[ts];info['threshold']=float(np.quantile(scores[ts],.8))
                    meta['fits'][name]=info
                base.atomic_json(jp,meta)
                temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)
            meta=json.loads(jp.read_text())
            with np.load(path) as z:
                np.testing.assert_array_equal(z['test_steps'],vs);np.testing.assert_array_equal(z['mean'],mean);np.testing.assert_array_equal(z['scale'],sd)
                for name in NEW:
                    all_scores[name][vs]=z[name+'_test']
                    if not cell.startswith('pb_'):thresholds[name][str(fold)]=meta['fits'][name]['threshold']
                    health.append(dict(cell=cell,fold=int(fold),method=name,**meta['fits'][name]))
            base.atomic_json(OUT/'RUN_STATE.json',dict(status='FITTING',completed_fits=len(health),expected_fits=135,pid=os.getpid(),elapsed_seconds=time.perf_counter()-start))
            print('[fits]',len(health),'/135',cell,int(fold),flush=True)
    assert all(np.isfinite(s).all() for s in all_scores.values())
    base.atomic_json(OUT/'FIT_HEALTH.json',health);base.atomic_json(OUT/'CALIBRATION.json',thresholds)
    return all_scores,thresholds


def expressions():
    return ({'conditional_vs_prior':{'supervised_conditional':1,'supervised_prior':-1},
             'prior_vs_static':{'supervised_prior':1,'supervised_static':-1},
             'conditional_vs_static':{'supervised_conditional':1,'supervised_static':-1},
             'static_vs_rbm':{'supervised_static':1,'rbm12__logit_old':-1},
             'conditional_vs_rbm':{'supervised_conditional':1,'rbm12__logit_old':-1},
             'static_vs_equal_mean':{'supervised_static':1,'equal_step_mean':-1}}, {'conditional_vs_prior'})


def evaluate(source,records,joined,outer,new,thresholds,features):
    _,prior=locations(source)
    with np.load(prior/'SCORES.npz') as z:refs={k[7:]:z[k] for k in z.files if k.startswith('steps__')}
    assert len(refs)==35
    metrics,per=base.evaluate_arrays(records,joined,refs)
    original=json.loads((prior/'METRICS.json').read_text())['metrics']
    for m in refs:
        for k in METRIC_KEYS:np.testing.assert_allclose(metrics[m][k],original[m][k],atol=1e-12,rtol=0)
    new_m,new_p=base.evaluate_arrays(records,joined,new,calibration_thresholds=thresholds,fold_auc=True)
    controls={'entropy_step_mean':features['entropy'],'equal_step_mean':features['x'].mean(axis=1)}
    cm,cp=base.evaluate_arrays(records,joined,controls)
    metrics.update(new_m);metrics.update(cm);per.update(new_p);per.update(cp)
    scores={**refs,**controls,**new};stepfold=np.repeat(outer,np.diff(joined['offsets']))
    prmask=np.repeat([not r['cell'].startswith('pb_') for r in records],np.diff(joined['offsets']))&(joined['labels']>=0)
    for m,s in scores.items():
        if m in NEW:continue
        fa={str(f):base.old.auc(joined['labels'][prmask&(stepfold==f)]==1,s[prmask&(stepfold==f)]) for f in sorted(set(outer))}
        metrics[m].update(prm_fold_auc=float(np.mean(list(fa.values()))),prm_fold_aucs=fa)
    print('[bootstrap] 10000 paired source-group draws',flush=True)
    bootstrap_driver.OUT=OUT;bootstrap_driver.expressions=expressions
    contrasts=bootstrap_driver.bootstrap(records,joined,metrics,per)
    base.atomic_json(OUT/'METRICS.json',dict(n_answers=len(records),metrics=metrics,contrasts=contrasts,
        scope='Supervised grouped development diagnostic. Step means differ from RBM Top10. Conditional-on-fitted-predictions CIs.'))
    np.savez_compressed(OUT/'SCORES.npz',**{'steps__'+m:s for m,s in scores.items()},
        **{'prediction__'+m:v['prediction'] for m,v in per.items()},**{'valid__'+m:v['valid'] for m,v in per.items()})
    names={'supervised_static':'Supervised fixed fusion','supervised_prior':'Supervised fixed fusion + position prior',
           'supervised_conditional':'Supervised position-dependent fusion','entropy_step_mean':'Entropy, step mean',
           'equal_step_mean':'Equal weights, normalized step means'}
    csv_write(OUT/'COMPARISON.csv',[dict(method=m,label=names.get(m,m),fit_access='other answers and labels' if m in NEW else 'existing reference',
         **{k:v[k] for k in METRIC_KEYS},prm_fold_auc=v['prm_fold_auc'],valid_answers=v['valid_answers']) for m,v in metrics.items()])
    csv_write(OUT/'PB_CELLS.csv',[dict(method=m,cell=c,**v) for m,x in metrics.items() for c,v in x['pb_cells'].items()])
    target=joined['target'];err=np.array([r['cell'].startswith('pb_') for r in records])&(target>=0);forensics={}
    for key,e in expressions()[0].items():
        pos=next(m for m,c in e.items() if c==1);neg=next(m for m,c in e.items() if c==-1)
        ah=per[pos]['prediction']==target;bh=per[neg]['prediction']==target
        gained=err&ah&~bh;lost=err&~ah&bh
        forensics[key]=dict(gained=int(gained.sum()),lost=int(lost.sum()),
            lost_early=int(np.sum(lost&(per[pos]['peak']<target))),lost_late=int(np.sum(lost&(per[pos]['peak']>target))))
    base.atomic_json(OUT/'FORENSICS.json',forensics)
    csv_write(OUT/'ANSWER_CHOICES.csv',[dict(uid=r['uid'],cell=r['cell'],group_id=r['group_id'],target=int(target[i]),
        **{m+'_peak':int(per[m]['peak'][i]) for m in NEW},**{m+'_prediction':int(per[m]['prediction'][i]) for m in NEW}) for i,r in enumerate(records)])
    base.atomic_json(OUT/'RUN_STATE.json',dict(status='SCORED_AWAITING_REVIEW',completed_fits=135,answers=len(records)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args();source=a.source_root.resolve()
    OUT.mkdir(parents=True,exist_ok=True);records,joined,outer=setup(source)
    m=manifest(source);mp=OUT/'MANIFEST.json'
    if mp.exists():assert json.loads(mp.read_text())==m,'manifest mismatch'
    else:base.atomic_json(mp,m)
    with threadpool_limits(limits=1):
        features=extract(source,records,joined);new,thresholds=fit_all(records,joined,outer,features)
        evaluate(source,records,joined,outer,new,thresholds,features)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);base.atomic_json(OUT/'ERROR.json',dict(error=repr(e)));raise
