"""Independent source recomputation; never reads VALIDATION_METRICS or reports."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json, pickle, hashlib, time, collections
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'source'

def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def zscore(values,eps):
    x=np.asarray(values,float); sd=x.std(axis=0)
    return np.divide(x-x.mean(axis=0),sd,out=np.zeros_like(x),where=sd>eps)

def main():
    start=time.perf_counter()
    paths={
      'predictions':SOURCE/'VALIDATION.npz','fits':SOURCE/'VALIDATION_FITS.json',
      'bundle':SOURCE/'BUNDLE.json','inputs':SOURCE/'INPUTS.json',
      'roster':ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json',
      'labels':ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.npz',
      'folds':ROOT/'results/localization_source_group_audit_v1/FOLDS_V2.json',
      'metadata':ROOT/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl',
      'level':ROOT/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz',
      'ct7':ROOT/'.worktrees/token-probability-fusion-v1/results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz',
      'fit_code':ROOT/'scripts/fit_external_source_bundle.py',
    }
    records=json.loads(paths['roster'].read_text())['records']
    with np.load(paths['labels']) as data: off=data['offsets']; error=data['labels'].astype(bool)
    with np.load(paths['predictions']) as data: saved={k:data[k] for k in data.files}
    fold_map=json.loads(paths['folds'].read_text())['outer']
    folds=np.array([fold_map[r['group_id']] for r in records]); sf=np.repeat(folds,np.diff(off))
    assert len(records)==13769 and len(error)==145597
    assert np.array_equal(saved['offsets'],off) and np.array_equal(saved['folds'],folds)
    assert np.array_equal(np.diff(off),[r['steps'] for r in records])
    with paths['metadata'].open('rb') as f: metadata={r['idx']:r for r in pickle.load(f).values()}
    prm=np.array([r['cell'].startswith('prm') for r in records])
    official=np.array([prm[i] and metadata[r['row_id']]['classification']!='correct' for i,r in enumerate(records)])
    for i in np.flatnonzero(prm):
        a,b=off[i:i+2]; m=metadata[records[i]['row_id']]
        assert m['n_steps']==b-a
        assert np.array_equal(error[a:b],[j+1 in m['error_steps'] for j in range(b-a)])
    mask=np.repeat(official,np.diff(off)); good=~error
    assert int(prm.sum())==6969 and int(official.sum())==6211 and int(mask.sum())==83371
    arms=sorted(k.removesuffix('_score') for k in saved if k.endswith('_score'))
    assert len(arms)==7
    stats={}
    for arm in arms:
        score=saved[arm+'_score']; pred=saved[arm+'_pred']
        assert score.shape==error.shape and np.isfinite(score).all()
        assert pred.shape==error.shape and set(np.unique(pred)) <= {0,1}
        p=pred.astype(bool)
        def counts(use):
            return [int(np.count_nonzero(use&good&p)),int(np.count_nonzero(use&error&p)),
                    int(np.count_nonzero(use&error&~p)),int(np.count_nonzero(use&good&~p))]
        def metrics(c):
            tp,fp,tn,fn=c
            valid_f1=2*tp/(2*tp+fp+fn);error_f1=2*tn/(2*tn+fp+fn)
            return {'valid_F1':valid_f1,'error_F1':error_f1,'official_PRMScore':(valid_f1+error_f1)/2}
        within=[]
        for i in np.flatnonzero(official):
            a,b=off[i:i+2]; y=error[a:b]; s=score[a:b]
            if y.any() and (~y).any():
                # Direct pairwise comparisons, independent of rank-statistic implementation.
                delta=s[y,None]-s[None,~y]
                within.append(float(((delta>0)+.5*(delta==0)).mean()))
        assert len(within)==6030
        c=counts(mask)
        stats[arm]={'TP_FP_TN_FN':c,**metrics(c),'within_auc':float(np.mean(within)),
                    'within_eligible_answers':len(within),'official_answers':int(official.sum()),
                    'official_steps':int(mask.sum()),'finite_full_steps':int(np.isfinite(score).sum()),
                    'by_fold':[{'fold':f,'TP_FP_TN_FN':counts(mask&(sf==f)),**metrics(counts(mask&(sf==f)))} for f in range(5)]}
    print('INDEPENDENT_METRICS',json.dumps(stats),flush=True)
    local={a:np.empty(len(error)) for a in arms if a.startswith('local_')}
    native=np.empty(len(records),bool); reasons=collections.Counter(); checkpoints=hashlib.sha256()
    files=list((SOURCE/'local_records').glob('*.json'))
    assert {p.name for p in files}=={str(i)+'.json' for i in range(len(records))}
    for i in range(len(records)):
        raw=(SOURCE/'local_records'/(str(i)+'.json')).read_bytes(); row=json.loads(raw)
        checkpoints.update(str(i).encode()+b'\0'+hashlib.sha256(raw).digest())
        assert row['index']==i
        a,b=off[i:i+2]
        for arm in local:
            arr=np.asarray(row['scores'][arm],float)
            assert arr.shape==(b-a,) and np.isfinite(arr).all()
            local[arm][a:b]=arr
        native[i]=row['diagnostics']['native']
        if not native[i]: reasons[row['diagnostics'].get('reason','UNKNOWN')]+=1
    assert np.array_equal(native,saved['native'])
    local_errors={arm:float(np.max(np.abs(v-saved[arm+'_score']))) for arm,v in local.items()}
    assert all(e==0 for e in local_errors.values())
    level=np.load(paths['level'])['level'].astype(float)
    ct7=np.load(paths['ct7'])['step_scores'].astype(float)
    for a,b in zip(off[:-1],off[1:]):
        level[a:b]=zscore(level[a:b],1e-12);ct7[a:b]=zscore(ct7[a:b],1e-8)
    assert float(np.max(np.abs(ct7-saved['ct7_score'])))==0
    fits=json.loads(paths['fits'].read_text()); bundle=json.loads(paths['bundle'].read_text())
    fit_checks=[]
    def replay(fit):
        groups=np.array(fit['groups']); unique,counts=np.unique(groups,return_counts=True)
        partition=np.array([1/(len(unique)*counts[np.flatnonzero(unique==g)[0]]) for g in groups])
        result={**local,'ct7':ct7,'frozen_lsml':level@np.array(fit['weights']),
                'frozen_equal':level.mean(1),'frozen_partition_equal':level@partition}
        for arm in ('frozen_lsml','frozen_equal','frozen_partition_equal'):
            for a,b in zip(off[:-1],off[1:]): result[arm][a:b]=zscore(result[arm][a:b],1e-8)
        return result
    for obj in fits:
        test=obj['test'];cal=obj['calibration'];train=[f for f in range(5) if f not in (test,cal)]
        assert cal==(test+1)%5 and test!=cal and len(train)==3
        train_groups={r['group_id'] for i,r in enumerate(records) if folds[i] in train}
        cal_groups={r['group_id'] for i,r in enumerate(records) if folds[i]==cal}
        test_groups={r['group_id'] for i,r in enumerate(records) if folds[i]==test}
        assert not(train_groups&cal_groups or train_groups&test_groups or cal_groups&test_groups)
        current=replay(obj['fit']); threshold_errors={}; score_errors={}; pred_mismatches={}
        for arm,v in current.items():
            threshold=float(np.quantile(v[sf==cal],.8,method='linear'))
            threshold_errors[arm]=abs(threshold-obj['thresholds'][arm])
            score_errors[arm]=float(np.max(np.abs(v[sf==test]-saved[arm+'_score'][sf==test])))
            pred_mismatches[arm]=int(np.count_nonzero((v[sf==test]<threshold)!=saved[arm+'_pred'][sf==test]))
        assert max(threshold_errors.values())<1e-12 and max(score_errors.values())<1e-12 and max(pred_mismatches.values())==0
        fit_checks.append({'test':test,'calibration':cal,'fit_folds':train,
            'fit_answers':int(np.isin(folds,train).sum()),'calibration_answers':int((folds==cal).sum()),'test_answers':int((folds==test).sum()),
            'fit_steps':int(np.isin(sf,train).sum()),'calibration_steps':int((sf==cal).sum()),'test_steps':int((sf==test).sum()),
            'source_group_disjoint':True,'threshold_errors':threshold_errors,'oof_score_errors':score_errors,'decision_mismatches':pred_mismatches})
    assert sorted(obj['test'] for obj in fits)==list(range(5))
    deployed=replay(bundle['fit']); final_errors={arm:abs(float(np.quantile(v[sf==4],.8,method='linear'))-bundle['thresholds'][arm]) for arm,v in deployed.items()}
    assert max(final_errors.values())<1e-12 and bundle['fit_folds']==[0,1,2,3] and bundle['calibration_fold']==4
    result={'status':'PASS','scope':'independent full source metric/decision/calibration replay; no new fits; no summary metrics read',
      'answers_all':len(records),'steps_all':len(error),'prmb_answers':int(prm.sum()),'official_answers':int(official.sum()),'official_steps':int(mask.sum()),
      'label_polarity':'JOINED label1=error; prediction1=valid; confusion ordered TP(valid), FP, TN(error), FN',
      'metric_definition':'arithmetic mean of global valid-step F1 and error-step F1; excludes classification=correct synthetic controls',
      'arms':stats,'local_checkpoints_checked':len(records),'local_native':int(native.sum()),'local_fallback':int((~native).sum()),
      'fallback_reasons':dict(reasons),'local_score_errors':local_errors,'local_records_indexed_sha256':checkpoints.hexdigest(),
      'validation_fit_replay':fit_checks,'deployment_threshold_errors':final_errors,
      'calibration_access':'Unlabeled pooled PB+PRMB steps; one held-out source fold for q80 calibration. Validation weights use other three folds; deploy weights use0..3 and calibration4.',
      'limits':'CPU local estimator fitting was not repeated. Checkpoint coverage and emitted scores audited for every answer; frozen weights and thresholds replayed without refit.',
      'sources':{k:{'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size} for k,p in paths.items()},'seconds':time.perf_counter()-start}
    (OUT/'INDEPENDENT_SOURCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',len(records),int(native.sum()),int(mask.sum()),flush=True)

if __name__=='__main__': main()
