"""Full PB first-error versus step-BCE objective comparison; PRMB unchanged."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_matched_rbm_supervision as previous
from scripts import run_rbm_logit_readout as bootstrap_driver
from scripts.run_rbm_data_diagnostics import base,csv_write,METRIC_KEYS
from spectral_utils.matched_rbm_coefficient_update import Top10,fit
from spectral_utils.rbm_first_error_objective import FirstErrorObjective

OUT=ROOT/'results/rbm_first_error_objective_v1'
NEW='first_error_pb'


def parent(source):return source/'.worktrees/rbm-supervision-matched-v1/results/rbm_supervision_matched_v1'


def manifest(source):
    prior=parent(source)
    assert json.loads((prior/'RUN_STATE.json').read_text())['status']=='COMPLETE'
    for name in ('RESULT_REVIEW.json','MODEL_REVIEW.json','BATCHED_ZERO_REVIEW.json'):
        assert json.loads((prior/name).read_text())['status']=='PASS'
    paths=[Path(__file__),ROOT/'spectral_utils/rbm_first_error_objective.py',ROOT/'spectral_utils/matched_rbm_coefficient_update.py',
        ROOT/'scripts/test_rbm_first_error_objective.py',ROOT/'scripts/verify_rbm_first_error_objective.py',
        ROOT/'scripts/run_direct_probability_temporal.py',ROOT/'docs/experiments/RBM_FIRST_ERROR_OBJECTIVE_V1.md',
        base.old.BENCH/'evaluation/JOINED.json',base.old.BENCH/'evaluation/JOINED.npz',base.old.FOLDS,
        source/'results/fusion_fixed_gate_v1/DETECTORS.npz',source/'results/fusion_fixed_gate_v1/METRICS.json']
    paths += [prior/n for n in ('SCORES.npz','METRICS.json','CALIBRATION.json','MANIFEST.json')]
    paths += sorted(prior.glob('cache_pb_*.npz'))
    paths += [previous.parent(source)/n for n in ('SCORES.npz','METRICS.json')]
    return dict(schema='rbm-first-error-objective-v1',base='f9d984266',scope='PB objective change only; PRMB scores copied unchanged from supervised BCE',
        hashes={str(p):base.old.sha256_file(p) for p in paths})


def expressions():
    e={'first_error_vs_bce':{NEW:1,'supervised_update':-1},'first_error_vs_unsupervised':{NEW:1,'unsupervised_update':-1}}
    for m in ('var15_iu__old','var50__old','entropy__old'):e['first_error_vs_'+m]={NEW:1,m:-1}
    return e,{'first_error_vs_bce'}


def train(source,records,joined,folds,score):
    health=[];prior=parent(source)
    for path in sorted(prior.glob('cache_pb_*.npz')):
        cell=path.stem[6:]
        with np.load(path) as z:cache={k:z[k] for k in z.files}
        ids=cache['ids'];top=Top10(cache['x'],cache['spans'])
        for fold in sorted(set(folds)):
            train=np.flatnonzero(folds[ids]!=fold);test=np.flatnonzero(folds[ids]==fold)
            assert not {records[i]['group_id'] for i in ids[train]}&{records[i]['group_id'] for i in ids[test]}
            dest=OUT/f'fit_{cell}_{fold}.npz';jp=dest.with_suffix('.json')
            if not dest.exists():
                data=previous.subset(cache,train);step_offsets=np.r_[0,np.cumsum(np.bincount(data['step_owner']))]
                targets=joined['target'][ids[train]]
                objective=FirstErrorObjective(data['x'],data['base'],data['spans'],step_offsets,targets)
                started=time.perf_counter();delta,info=fit(objective,12)
                info.update(cell=cell,fold=int(fold),seconds=time.perf_counter()-started,
                    train_answers=len(train),training_error_answers=int(np.sum(targets>=0)),training_clean_answers=int(np.sum(targets<0)))
                assert delta[-1]==0.,'common intercept must remain zero'
                base.atomic_json(jp,info);np.savez_compressed(dest,delta=delta,train_ids=ids[train],test_ids=ids[test])
                del data,objective
            info=json.loads(jp.read_text())
            with np.load(dest) as z:
                np.testing.assert_array_equal(z['train_ids'],ids[train]);np.testing.assert_array_equal(z['test_ids'],ids[test]);delta=z['delta']
                flat,_=top.evaluate(cache['base']+cache['x']@delta[:-1]+delta[-1])
            for local in test:
                i=ids[local];a,b=cache['step_offsets'][local:local+2];score[joined['offsets'][i]:joined['offsets'][i+1]]=flat[a:b]
            health.append(info);base.atomic_json(OUT/'RUN_STATE.json',dict(status='FITTING',completed=len(health),expected=40,pid=os.getpid()))
            print('[fit]',len(health),'/40',cell,int(fold),info['iterations'],'iterations',round(info['seconds'],1),'s',flush=True)
    assert len(health)==40 and np.isfinite(score).all();base.atomic_json(OUT/'FIT_HEALTH.json',health)
    return score


def evaluate(source,records,joined,folds,scores):
    calibration=json.loads((parent(source)/'CALIBRATION.json').read_text());calibration[NEW]=calibration['supervised_update'].copy()
    modeled={m:scores[m] for m in (NEW,'supervised_update','unsupervised_update')}
    metrics,per=base.evaluate_arrays(records,joined,modeled,calibration_thresholds=calibration,fold_auc=True)
    controls={m:s for m,s in scores.items() if m not in modeled};cm,cp=base.evaluate_arrays(records,joined,controls);metrics.update(cm);per.update(cp)
    old=json.loads((parent(source)/'METRICS.json').read_text())['metrics'];historical=json.loads((previous.parent(source)/'METRICS.json').read_text())['metrics']
    for m in ('supervised_update','unsupervised_update'):
        for k in ('pb_all8','pb_q4','pb_q8','prm_within','prm_fold_auc','prmscore_q08'):np.testing.assert_allclose(metrics[m][k],old[m][k],atol=1e-12,rtol=0)
    for m in controls:
        for k in METRIC_KEYS:np.testing.assert_allclose(metrics[m][k],historical[m][k],atol=1e-12,rtol=0)
    prmask=np.repeat([not r['cell'].startswith('pb_') for r in records],np.diff(joined['offsets']))
    np.testing.assert_array_equal(scores[NEW][prmask],scores['supervised_update'][prmask])
    for k in ('prm_within','prm_fold_auc','prmscore_q08'):assert metrics[NEW][k]==metrics['supervised_update'][k]
    print('[bootstrap]10000 source-group paired draws',flush=True);bootstrap_driver.OUT=OUT;bootstrap_driver.expressions=expressions
    contrasts=bootstrap_driver.bootstrap(records,joined,metrics,per)
    base.atomic_json(OUT/'METRICS.json',dict(n_answers=len(records),new_pb_answers=6800,copied_prmb_answers=6969,metrics=metrics,contrasts=contrasts,
        note='PRMB statistics on the new row are unchanged inherited BCE results, not new transfer evidence.'))
    base.atomic_json(OUT/'CALIBRATION.json',calibration)
    np.savez_compressed(OUT/'SCORES.npz',**{'steps__'+m:s for m,s in scores.items()},**{'prediction__'+m:p['prediction'] for m,p in per.items()},**{'valid__'+m:p['valid'] for m,p in per.items()})
    csv_write(OUT/'COMPARISON.csv',[dict(method=m,**{k:v[k] for k in METRIC_KEYS},prm_fold_auc=v.get('prm_fold_auc'),valid_answers=v['valid_answers']) for m,v in metrics.items()])
    csv_write(OUT/'PB_CELLS.csv',[dict(method=m,cell=c,**v) for m,x in metrics.items() for c,v in x['pb_cells'].items()])
    target=joined['target'];err=np.array([r['cell'].startswith('pb_') for r in records])&(target>=0);forensics={}
    for key,e in expressions()[0].items():
        neg=next(m for m,c in e.items() if c==-1);a,b=per[NEW],per[neg]
        gained=err&(a['prediction']==target)&(b['prediction']!=target);lost=err&(a['prediction']!=target)&(b['prediction']==target)
        forensics[key]=dict(gained=int(gained.sum()),lost=int(lost.sum()),lost_early=int(np.sum(lost&(a['peak']<target))),lost_late=int(np.sum(lost&(a['peak']>target))))
    base.atomic_json(OUT/'FORENSICS.json',forensics)
    base.atomic_json(OUT/'RUN_STATE.json',dict(status='SCORED_AWAITING_REVIEW',fits=40,answers=len(records)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);args=p.parse_args();source=args.source_root.resolve()
    OUT.mkdir(parents=True,exist_ok=True);records,joined,folds,_=previous.setup(source);m=manifest(source)
    mp=OUT/'MANIFEST.json'
    if mp.exists():assert json.loads(mp.read_text())==m,'manifest mismatch'
    else:base.atomic_json(mp,m)
    with np.load(parent(source)/'SCORES.npz') as z:scores={k[7:]:z[k] for k in z.files if k.startswith('steps__')}
    with np.load(previous.parent(source)/'SCORES.npz') as z:
        for method in ('var15_iu__old','var50__old','entropy__old'):scores[method]=z['steps__'+method]
    with threadpool_limits(limits=1):
        scores[NEW]=train(source,records,joined,folds,scores['supervised_update'].copy())
        evaluate(source,records,joined,folds,scores)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);base.atomic_json(OUT/'ERROR.json',dict(error=repr(e)));raise
