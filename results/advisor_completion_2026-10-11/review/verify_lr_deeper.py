"""Independent training-only selector/C-tuning and all saved affine replay."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
P=Path(__file__).resolve().parents[1];I=P/'inputs';p=np.load(I/'population.npz')
off=p['offsets'];y=p['y'];fold=p['fold'];ai=np.repeat(np.arange(len(fold)),np.diff(off));sf=fold[ai]
X=np.load(I/'bank_V_N16.npz')['V'];groups=p['group'];splits=json.loads((I/'FOLDS_V2.json').read_text())['prmbench']
m=json.loads((P/'lr/checkpoints/N16/outer_0/MODELS.json').read_text());info=m['arms']['LR__LR_HALF']
imap=splits['inner']['0'];inn=np.array([imap[g] if f!=0 else -1 for g,f in zip(groups,fold)])[ai]
losses=[];mask_matches=[]
for C in [.001,.01,.1,1.,10.,100.,1000.]:
    ll=[]
    for k in range(5):
        tr=(sf!=0)&(inn!=k);va=(sf!=0)&(inn==k)
        scale=StandardScaler().fit(X[tr]);selector=LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=20261003).fit(scale.transform(X[tr]),y[tr])
        coef=selector.coef_[0];cols=np.sort(np.lexsort((np.arange(len(coef)),-np.abs(coef)))[:8])
        assert np.array_equal(cols,info['inner_selected_columns'][str(k)])
        mask_matches.append(True)
        scale=StandardScaler().fit(X[tr][:,cols]);fit=LogisticRegression(C=C,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=20261003).fit(scale.transform(X[tr][:,cols]),y[tr])
        prediction=fit.predict_proba(scale.transform(X[va][:,cols]))
        counts=np.bincount(y[va].astype(int),minlength=2);weight=len(y[va])/(2*counts[y[va].astype(int)])
        ll.append(log_loss(y[va],prediction,sample_weight=weight))
    losses.append(float(np.mean(ll)))
assert np.max(np.abs(np.asarray(losses)-info['inner_balanced_log_loss']))<1e-9
C_grid=[.001,.01,.1,1.,10.,100.,1000.];winner=C_grid[int(np.argmin(losses))]
assert winner==info['C']
replay_error=0.;mean_error=0.;scale_error=0.;nmodels=0;failures=[];nmasks=0
for path in sorted((P/'lr/checkpoints').glob('*/*/MODELS.json')):
    m=json.loads(path.read_text());bk=m['bank'];outer=m['outer_fold'];Xb=np.load(I/f'bank_V_{bk}.npz')['V'];tr=sf!=outer;te=sf==outer
    score=np.load(path.with_name('SCORES.npz'));assert np.array_equal(score['rows'],np.flatnonzero(te))
    for arm,q in m['arms'].items():
        if 'failed' in q:
            assert arm not in score.files;failures.append(dict(bank=bk,outer=outer,arm=arm,reason=q['failed']));continue
        col=q['columns'];model=q['model'];mean=Xb[tr][:,col].mean(0);sd=Xb[tr][:,col].std(0)
        mean_error=max(mean_error,float(np.max(np.abs(mean-model['training_mean']))))
        scale_error=max(scale_error,float(np.max(np.abs(np.where(sd>0,sd,1)-model['training_scale']))))
        predicted=Xb[te][:,col]@model['weight']+model['intercept']
        replay_error=max(replay_error,float(np.max(np.abs(predicted-score[arm]))));nmodels+=1
    rank=np.asarray(m['selector_models']['LR_SELECTOR_C1']['weight_std']);want=np.sort(np.lexsort((np.arange(len(rank)),-abs(rank)))[:int(np.ceil(len(rank)/2))])
    assert np.array_equal(want,m['outer_masks']['LR_HALF']);nmasks+=1
assert replay_error<1e-9 and mean_error<1e-12 and scale_error<1e-12
assert len(failures)==1 and failures[0]['bank']=='N11' and failures[0]['arm']=='LSML__LR_HALF'
result=dict(status='PASS',n_affine_models=nmodels,n_outer_training_masks=nmasks,max_test_affine_abs=replay_error,
            max_training_mean_abs=mean_error,max_training_scale_abs=scale_error,
            independent_nested_probe='N16outer0LR_LR_HALF: 5 inner C1 selectors and35 fitted C-grid classifiers',
            inner_mask_replays=len(mask_matches),inner_loss_max_abs=float(np.max(np.abs(np.asarray(losses)-info['inner_balanced_log_loss']))),
            winning_C=winner,explicit_method_failures=failures,
            limitation='Other inner-grid fits inspected structurally, not all independently refitted')
(P/'review/LR_DEEP_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
