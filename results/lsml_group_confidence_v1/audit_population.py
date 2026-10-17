"""Independent population/coverage audit; does not read result summaries."""
from pathlib import Path
import hashlib, json, pickle
import numpy as np
import pandas as pd

ROOT = Path(r'C:\Users\omris\TAU\hallucination_detection')
OUT = ROOT / 'results/lsml_group_confidence_v1'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
def read(name): return json.loads((OUT/name).read_text(encoding='utf8'))
seal, freeze = read('SEAL.json'), read('FREEZE.json')
for name, key in [('PREDICTIONS.npz','predictions_sha256'),('CALIBRATION.npz','calibration_sha256'),('FREEZE.json','freeze_sha256'),('FITS.json','fits_sha256')]:
    assert sha(OUT/name) == seal[key], name
paths = {k: Path(v['path']) for k,v in freeze['inputs'].items()}
for k,p in paths.items(): assert sha(p) == freeze['inputs'][k]['sha256'], k
for k,h in freeze['code'].items(): assert sha(ROOT/k) == h, k
ans = pd.read_csv(paths['oof_answers'])
records = json.loads(paths['joined'].read_text())['records']
folds = json.loads(paths['folds'].read_text())['outer']
z = np.load(OUT/'PREDICTIONS.npz'); calz = np.load(OUT/'CALIBRATION.npz')
source = np.load(paths['source_scores']); base = np.load(paths['arrays']); old = np.load(paths['oof_step_scores'])
off = z['offsets']; sizes = np.diff(off)
assert len(ans) == len(records) == 13769 and off[-1] == 145597 and len(off) == 13770
assert ans.uid.is_unique and (sizes > 0).all()
for i,r in enumerate(records):
    assert (ans.uid[i],str(ans.id[i]),ans.source_group[i],ans.cell[i],ans.fold[i],sizes[i]) == (r['uid'],str(r['row_id']),r['group_id'],r['cell'],folds[r['group_id']],r['steps'])
np.testing.assert_array_equal(off,base['offsets']); np.testing.assert_array_equal(off,old['offsets'])
np.testing.assert_array_equal(z['folds'],ans.fold); np.testing.assert_array_equal(source['folds'],ans.fold)
assert ans.groupby('source_group').fold.nunique().max() == 1
np.testing.assert_array_equal(z['writes'], np.ones(145597,dtype=np.int8))
arms = freeze['arms']
for arm in arms:
    assert z[arm+'_score'].shape == z[arm+'_pred'].shape == (145597,)
    assert np.isfinite(z[arm+'_score']).all() and np.isin(z[arm+'_pred'],[0,1]).all()
fits = read('FITS.json'); assert len(fits) == 5
sf = np.repeat(ans.fold.to_numpy(),sizes)
checks=[]
for f in fits:
    k,c=f['test'],f['calibration']; tr=f['fit_folds']
    assert c==(k+1)%5 and set(tr)==set(range(5))-{k,c}
    assert not (set(ans.source_group[ans.fold.isin(tr)]) & set(ans.source_group[ans.fold.isin([k,c])]))
    masks={'train':np.isin(sf,tr),'calibration':sf==c,'test':sf==k}
    for name,mask in masks.items(): assert int(mask.sum())==f[name+'_steps']
    model=read('MODEL_'+str(k)+'.json')
    assert model['converged'] and model['rows']==f['train_steps']
    assert sha(OUT/('MODEL_'+str(k)+'.json'))==f['model_hash']
    for arm in arms:
        ca=calz[str(k)+'__'+arm]
        assert ca.shape==(f['calibration_steps'],) and np.isfinite(ca).all()
        tau=float(np.quantile(ca,.8)); assert tau==f['thresholds'][arm]
        np.testing.assert_array_equal(z[arm+'_pred'][sf==k],z[arm+'_score'][sf==k]<tau)
    checks.append({'test':k,'calibration':c,'fit_folds':tr,**{n+'_steps':int(m.sum()) for n,m in masks.items()}})
anchors={}
for arm,ref in [('bank11_lsml','frozen_lsml'),('bank11_equal','frozen_equal'),('ct7_z','ct7')]:
    diff=float(np.max(np.abs(z[arm+'_score']-source[ref+'_score'])))
    assert diff<=1e-6
    np.testing.assert_array_equal(z[arm+'_pred'],source[ref+'_pred'])
    anchors[arm]={'steps_checked':145597,'answers_checked':13769,'max_score_difference':diff,'decision_mismatches':0}
# Labels are opened only after verifying that a predictions seal exists.
labels=base['labels'].astype(bool)
np.testing.assert_array_equal(labels,old['labels'].astype(bool)); np.testing.assert_array_equal(ans.target,base['target'])
meta={r['idx']:r for r in pickle.loads(paths['metadata'].read_bytes()).values()}
prm=ans.cell.str.startswith('prm').to_numpy(); pb=~prm
for i in np.flatnonzero(prm):
    a,b=off[i:i+2]
    np.testing.assert_array_equal(labels[a:b],np.isin(np.arange(1,b-a+1),meta[ans.id[i]]['error_steps']))
noncontrol=np.array([prm[i] and meta[ans.id[i]]['classification']!='correct' for i in range(len(ans))])
eligible=np.array([prm[i] and labels[a:b].any() and (~labels[a:b]).any() for i,(a,b) in enumerate(zip(off[:-1],off[1:]))])
assert (int(prm.sum()),int(noncontrol.sum()),int(eligible.sum()),int((pb&(ans.target>=0)).sum()))==(6969,6211,6030,4442)
design=json.loads(paths['design'].read_text()); names=json.loads(paths['pool_names'].read_text()); pool=np.load(paths['pool_z'])
assert pool.shape==(145597,52) and np.isfinite(pool).all()
assert len(design['kept'])==28 and len(design['splits']['S3_M15'])==15
flat=sum(design['splits']['S3_M15'].values(),[])
assert sorted(flat)==sorted(design['kept']) and len(set(flat))==28
def az(x):
    result=np.zeros_like(x,dtype=float)
    for a,b in zip(off[:-1],off[1:]):
        v=x[a:b]; sd=v.std(axis=0)
        np.divide(v-v.mean(axis=0),sd,out=result[a:b],where=sd>1e-8)
    return result
x=az(pool[:,[names.index(n) for n in design['kept']]])*np.array([design['orientation'][n] for n in design['kept']])
# Independently compose likelihood contributions from saved parameters on every step.
replays=[]
for f in fits:
    k=f['test']; model=read('MODEL_'+str(k)+'.json')
    h=np.asarray(model['mu'])+np.asarray(model['sd'])*(x@np.asarray(f['expansion']))
    theta=np.asarray(model['theta']); a=np.asarray(model['a']); gr=np.asarray(model['groups'])
    for average in [False,True]:
        score=np.full(len(h),np.log(model['pi']/(1-model['pi'])))
        for g in range(len(a)):
            ix=np.flatnonzero(gr==g)
            if len(ix)==1:
                q=h[:,ix[0]]
                score+=q*np.log(a[g,1]/a[g,0])+(1-q)*np.log((1-a[g,1])/(1-a[g,0]))
            else:
                e=(h[:,ix]*np.log(theta[ix,1]/theta[ix,0])+(1-h[:,ix])*np.log((1-theta[ix,1])/(1-theta[ix,0]))).sum(axis=1)
                if average: e=e/len(ix)
                score+=np.logaddexp(np.log(1-a[g,1]),np.log(a[g,1])+e)-np.logaddexp(np.log(1-a[g,0]),np.log(a[g,0])+e)
        score=az(score); arm='confidence_average' if average else 'confidence_sum'
        caldiff=float(np.max(np.abs(score[sf==f['calibration']]-calz[str(k)+'__'+arm])))
        testdiff=float(np.max(np.abs(score[sf==k]-z[arm+'_score'][sf==k])))
        assert max(caldiff,testdiff)<1e-10
        replays.append({'fold':k,'arm':arm,'calibration_max_diff':caldiff,'test_max_diff':testdiff})
report={'audit':'population/coverage, no summaries read','status':'PASS','answers_checked':13769,'answers_total':13769,'steps_checked':145597,'steps_total':145597,'arms_checked':len(arms),'finite_score_checks':len(arms)*145597,'decision_checks':len(arms)*145597,'source_groups':int(ans.source_group.nunique()),'all_writes_exactly_once':True,'input_hashes_checked':len(paths),'code_hashes_checked':len(freeze['code']),'folds':checks,'same_model_candidate_calibration_replays':replays,'anchors':anchors,'labels':{'prm_answers_checked':int(prm.sum()),'prm_steps_checked':int(sizes[prm].sum()),'noncontrol_answers':int(noncontrol.sum()),'noncontrol_steps':int(sizes[noncontrol].sum()),'within_auc_answers':int(eligible.sum()),'pb_error_answers':int((pb&(ans.target>=0)).sum()),'source_labels_identical':True,'raw_one_based_prm_labels_identical':True},'limitations':['Bank orientation/filter/families were selected using development labels before this experiment; current fold separation does not make bank discovery unsupervised or unbiased.','Conditional independence is an assumption, not certified by named groups or low observed correlation.','Anchor scores replay within numerical tolerance, not asserted byte identical; decisions are exact.'],'seal_sha256':sha(OUT/'SEAL.json'),'predictions_sha256':sha(OUT/'PREDICTIONS.npz')}
(OUT/'AUDIT_POPULATION.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:report[k] for k in ['status','answers_checked','steps_checked','arms_checked','input_hashes_checked','code_hashes_checked','source_groups','anchors','labels']}))
