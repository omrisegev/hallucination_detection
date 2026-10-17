"""Group exclusion, unlabeled prior and nested calibration firewall fixtures."""
import copy,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_conditional_iu_fusion as d


def run():
    records=[dict(uid='u'+str(i),cell='prm_fixture',group_id=g,steps=2,target=999,label='DO_NOT_READ')
             for i,g in enumerate(['a','a','b','c','d','e'])]
    folds={0:0,1:0,2:1,3:2,4:3,5:4}
    meta=d.training_metadata(records,folds)
    assert all(set(v)=={'uid','cell','group_id','fold'} for v in meta.values())
    stats={i:{k:np.broadcast_to(np.eye(12)*(i+1),(16,12,12)).copy() for k in ('real','shuffle')} for i in meta}
    prior,info=d.fit_prior(stats,meta,'prm_fixture',(4,))
    # Four groups: a's two answers together receive exactly a quarter.
    np.testing.assert_allclose([info['answer_weights'][str(i)] for i in range(5)],[.125,.125,.25,.25,.25])
    np.testing.assert_allclose(prior['real'],np.broadcast_to(np.eye(12)*3.375,(16,12,12)))
    changed=copy.deepcopy(records)
    changed[5].update(target=-123,label='changed held labels')
    changedstats=copy.deepcopy(stats);changedstats[5]['real'][:]=np.nan
    changedstats[5]['shuffle'][:]=1e20
    replay,replayinfo=d.fit_prior(changedstats,d.training_metadata(changed,folds),'prm_fixture',(4,))
    for name in prior:np.testing.assert_array_equal(prior[name],replay[name])
    assert info==replayinfo
    # Both outer and inner held groups must be absent from shared fitting.
    nested,ninfo=d.fit_prior(stats,meta,'prm_fixture',(0,4))
    assert ninfo['training_ids']==[2,3,4]
    np.testing.assert_allclose(nested['real'],np.broadcast_to(4*np.eye(12),(16,12,12)))
    labels=copy.deepcopy(meta);labels[5]['label']=1
    try:d.fit_prior(stats,labels,'prm_fixture',(4,))
    except ValueError:pass
    else:raise AssertionError('label-bearing fit metadata accepted')
    crossing=copy.deepcopy(meta);crossing[5]['group_id']='a'
    try:d.fit_prior(stats,crossing,'prm_fixture',(4,))
    except ValueError:pass
    else:raise AssertionError('source group split across training/test accepted')
    checks=['equal source-group then answer weights','held feature and label mutation leaves prior bitwise identical',
            'nested two-fold prior exclusion','fit API rejects label-bearing metadata','cross-fold source-group leak rejected']
    # Deliberately distinct outer and inner values: wrong calibration is detectable.
    methods=('baseline','position');offsets=np.arange(0,13,2)
    joined=dict(offsets=offsets)
    predictions={i:dict(baseline=np.array([i,i+1.]),position=np.array([10000+i,10001+i])) for i in range(6)}
    for i in predictions:
        for f in range(5):
            if f!=folds[i]:predictions[i]['position__inner_for_'+str(f)]=np.array([100*f+2*i,100*f+2*i+1.])
    scores={name:np.concatenate([predictions[i][name] for i in range(6)]) for name in methods}
    scores['reference__fixture']=scores['baseline']+20000
    # Helper receives joined offsets explicitly to avoid a module-global roster.
    thresholds,coverage,unavailable=d.build_calibration(predictions,scores,records,joined,folds,list(range(6)),methods)
    assert not unavailable
    for f in range(5):
        keep=[i for i in range(6) if folds[i]!=f]
        expected=np.quantile(np.concatenate([predictions[i]['position__inner_for_'+str(f)] for i in keep]),.8)
        assert thresholds['position'][str(f)]==expected
        assert thresholds['baseline'][str(f)]==np.quantile(np.concatenate([predictions[i]['baseline'] for i in keep]),.8)
        assert thresholds['reference__fixture'][str(f)]==thresholds['baseline'][str(f)]+20000
    for row in coverage:
        assert not set(row['training_groups'])&set(row['excluded_groups'])
        assert all(folds[i]!=row['outer_fold'] for i in row['training_ids'])
    changedpred=copy.deepcopy(predictions)
    for i in changedpred:
        if folds[i]!=4:changedpred[i]['position__inner_for_4'][:]=np.nan
    t,c,u=d.build_calibration(changedpred,scores,records,joined,folds,list(range(6)),methods)
    assert t['position']['4'] is None and u==[dict(method='position',outer_fold=4)]
    row=next(r for r in c if r['method']=='position' and r['outer_fold']==4)
    assert row['answers']==0 and row['expected']==5
    checks+=['PRMScore reads inner predictions, never outer fitted predictions','calibration excludes test source groups',
             'answer-local and reference calibration unchanged','all-invalid calibration explicitly unavailable with original denominator']
    d.START=time.monotonic();d.STOP_REQUESTED=True
    try:
        try:d.cap_check()
        except d.hy.InvocationCap:pass
        else:raise AssertionError('scheduler stop latch ignored')
    finally:d.STOP_REQUESTED=False
    checks.append('scheduler stop honored at checkpoint boundary')
    return dict(status='PASS',count=len(checks),checks=checks)


if __name__=='__main__':print(json.dumps(run(),indent=2))
