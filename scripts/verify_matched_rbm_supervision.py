"""Independent arithmetic replay of the completed temporal benchmark metrics."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--result-dir',type=Path,default=ROOT/'results/rbm_supervision_matched_v1')
    args=parser.parse_args();out=args.result_dir
    result=json.loads((out/'METRICS.json').read_text(encoding='utf8'))
    manifest=json.loads((out/'MANIFEST.json').read_text(encoding='utf8'))
    # Recheck small contract/code artifacts. Raw cache hashes were checked at
    # launch; do not re-read every multi-GB cache in this result-only verifier.
    for name,expected in manifest['hashes'].items():
        path=Path(name)
        if 'dataset_cache' in path.parts:continue
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    bench=args.source_root/'results/localization_full_benchmark_v3/evaluation'
    records=json.loads((bench/'JOINED.json').read_text(encoding='utf8'))['records']
    arrays=np.load(bench/'JOINED.npz');scores=np.load(out/'SCORES.npz')
    offsets,labels,target=arrays['offsets'],arrays['labels'],arrays['target']
    cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_')
    assert len(records)==13769 and result['n_answers']==13769
    folds=json.loads((args.source_root/'results/localization_source_group_audit_v1/FOLDS_V2.json').read_text(encoding='utf8'))['outer']
    outer=np.array([int(folds[r['group_id']]) for r in records])
    updated=('unsupervised_update','supervised_update');calibration={m:{} for m in updated};modelchecks=[]
    for path in sorted(out.glob('cache_*.npz')):
        cell=path.stem[6:]
        with np.load(path) as z:cache={k:z[k] for k in z.files}
        ids=cache['ids'];to=cache['token_offsets'];so=cache['step_offsets'];spans=cache['spans'];x=cache['x']
        np.testing.assert_array_equal(cache['uid'],[records[i]['uid'] for i in ids])
        for local,i in enumerate(ids):
            y=labels[offsets[i]:offsets[i+1]].copy()
            if pb[i]:
                y=np.zeros(len(y),int)
                if target[i]>=0:y[target[i]]=1;y[target[i]+1:]=-1
            np.testing.assert_array_equal(y,cache['labels'][so[local]:so[local+1]])
        for fold in sorted(set(outer)):
            fp=out/f'fit_{cell}_{fold}.npz';info=json.loads(fp.with_suffix('.json').read_text())
            train=np.flatnonzero(outer[ids]!=fold);test=np.flatnonzero(outer[ids]==fold)
            assert not {records[i]['group_id'] for i in ids[train]}&{records[i]['group_id'] for i in ids[test]}
            with np.load(fp) as fit:
                np.testing.assert_array_equal(fit['train_ids'],ids[train]);np.testing.assert_array_equal(fit['test_ids'],ids[test])
                for method in updated:
                    delta=fit[method];token=cache['base']+x@delta[:-1]+delta[-1]
                    step=np.array([np.sort(token[a:b])[-min(10,b-a):].mean() for a,b in spans])
                    for local in test:
                        i=ids[local]
                        np.testing.assert_allclose(step[so[local]:so[local+1]],scores['steps__'+method][offsets[i]:offsets[i+1]],atol=1e-11,rtol=1e-12)
                    if not cell.startswith('pb_'):
                        calibration[method][str(fold)]=float(np.quantile(np.concatenate([step[so[i]:so[i+1]] for i in train]),.8))
                    if method=='unsupervised_update':
                        losses=[]
                        for i in train:
                            xx=x[to[i]:to[i+1]];a=cache['a'][i];o=cache['orientation'][i]
                            w=cache['w'][i]+o*cache['active'][i]*delta[:-1];b=cache['b'][i]+o*delta[-1]
                            prior=b+a@w+.5*w@w
                            l0=-np.logaddexp(0,prior)-.5*np.sum((xx-a)**2,axis=1)
                            l1=-np.logaddexp(0,-prior)-.5*np.sum((xx-a-w)**2,axis=1)
                            losses.append(-np.logaddexp(l0,l1).mean())
                        loss=float(np.mean(losses)+.005*(delta@delta))
                    else:
                        ys=[];ss=[];mass=[]
                        for i in train:
                            yy=cache['labels'][so[i]:so[i+1]];known=(yy==0)|(yy==1)
                            ys.extend(yy[known]);ss.extend(step[so[i]:so[i+1]][known]);mass.extend(np.full(int(known.sum()),1/known.sum()))
                        yy=np.asarray(ys);ss=np.asarray(ss);mass=np.asarray(mass)
                        for label in (0,1):mass[yy==label]*=.5/mass[yy==label].sum()
                        loss=float(np.sum(mass*np.where(yy==1,np.logaddexp(0,-ss),np.logaddexp(0,ss)))+.005*(delta@delta))
                    np.testing.assert_allclose(loss,info['methods'][method]['final_loss'],atol=1e-10,rtol=1e-10)
                    modelchecks.append(dict(cell=cell,fold=int(fold),method=method,loss=loss))
        print('[model review]',cell,flush=True)
    assert len(modelchecks)==90
    (out/'MODEL_REVIEW.json').write_text(json.dumps(dict(status='PASS',models=90,checks=modelchecks),indent=2)+'\n',encoding='utf8')
    stepfold=np.repeat(outer,np.diff(offsets));stepprm=np.repeat(~pb,np.diff(offsets))
    gate_dir=args.source_root/'results/fusion_fixed_gate_v1'
    detector=np.load(gate_dir/'DETECTORS.npz')['entropy_mean']
    gate=json.loads((gate_dir/'METRICS.json').read_text(encoding='utf8'))
    thresholds=gate['arms']['dual__iu']['rows']['entropy_mean|quantile_0.3']['thresholds']
    gate_threshold=np.array([thresholds[str(f)] for f in outer])
    with (args.source_root/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl').open('rb') as f:
        meta={str(r['idx']):r for r in pickle.load(f).values()}
    checked={}
    for name,metric in result['metrics'].items():
        flat=scores['steps__'+name];valid=scores['valid__'+name];pred=scores['prediction__'+name]
        actual_valid=np.array([np.isfinite(flat[offsets[i]:offsets[i+1]]).all() and offsets[i]<offsets[i+1] for i in range(len(records))])
        np.testing.assert_array_equal(valid,actual_valid)
        for i in np.flatnonzero(pb & valid):
            s=flat[offsets[i]:offsets[i+1]]
            expected=int(np.argmax(s)) if detector[i]>=gate_threshold[i] else -1
            assert int(pred[i])==expected,(name,records[i]['uid'],'gate/readout mismatch')
        f1cells={}
        for cell in sorted(set(cells[pb])):
            clean=(cells==cell)&(target<0);error=(cells==cell)&(target>=0)
            hit=valid & (pred==target)
            ca=hit[clean].sum()/clean.sum();ea=hit[error].sum()/error.sum()
            f1=2*ca*ea/(ca+ea) if ca+ea else 0.
            np.testing.assert_allclose(f1,metric['pb_cells'][cell]['f1'],atol=1e-14)
            f1cells[cell]=f1
        np.testing.assert_allclose(np.mean(list(f1cells.values())),metric['pb_all8'],atol=1e-14)
        within=[];pooled_y=[];pooled_s=[]
        for i in np.flatnonzero(~pb & valid):
            s=flat[offsets[i]:offsets[i+1]];y=labels[offsets[i]:offsets[i+1]];use=y>=0
            s=s[use];y=y[use]==1;pooled_y.extend(y);pooled_s.extend(s)
            if y.any() and (~y).any():
                # Direct positive/negative pairs; no rank-based AUC code reused.
                d=s[y][:,None]-s[~y][None,:]
                within.append(float(np.mean((d>0)+.5*(d==0))))
        if within:np.testing.assert_allclose(np.mean(within),metric['prm_within'],atol=1e-14)
        assert len(within)==metric['prm_within_n']
        if pooled_y and name not in updated:np.testing.assert_allclose(roc_auc_score(pooled_y,pooled_s),metric['prm_pooled'],atol=1e-14)
        if name in updated:assert metric['prm_pooled'] is None
        fa=[]
        for fold in sorted(set(outer[~pb])):
            mask=stepprm&(stepfold==fold)&(labels>=0);v=roc_auc_score(labels[mask]==1,flat[mask]);fa.append(v)
            np.testing.assert_allclose(v,metric['prm_fold_aucs'][str(fold)],atol=1e-14)
        np.testing.assert_allclose(np.mean(fa),metric['prm_fold_auc'],atol=1e-14)
        tp=tn=fp=fn=0
        for fold in sorted(set(outer[~pb])):
            train=np.flatnonzero(~pb & valid & (outer!=fold));test=np.flatnonzero(~pb & valid & (outer==fold))
            assert not set(records[i]['group_id'] for i in train)&set(records[i]['group_id'] for i in test)
            if not len(train):continue
            q=calibration[name][str(fold)] if name in updated else float(np.quantile(np.concatenate([flat[offsets[i]:offsets[i+1]] for i in train]),.8))
            np.testing.assert_allclose(q,metric['prmscore_thresholds'][str(fold)],atol=1e-11)
            for i in test:
                row=meta[str(records[i]['row_id'])]
                if row['classification']=='correct':continue # official total excludes synthetic controls
                s=flat[offsets[i]:offsets[i+1]];accepted=s<q
                correct=~np.isin(np.arange(1,len(s)+1),row['error_steps'])
                tp+=int(np.sum(accepted & correct));fn+=int(np.sum(~accepted & correct))
                tn+=int(np.sum(~accepted & ~correct));fp+=int(np.sum(accepted & ~correct))
        if metric['prmscore_conditional'] is not None:
            positive=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else -1
            negative=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else -1
            np.testing.assert_allclose(.5*(positive+negative),metric['prmscore_conditional'],atol=1e-14)
        checked[name]=dict(valid_answers=int(valid.sum()),within_answers=len(within),pb_cells=len(f1cells))
    payload=dict(status='PASS',scope='Separate arithmetic implementation in the same Codex session; not an external independent scientist review.',
        checks=['code/contract hashes','all PB cells with full denominators','PRMB direct-pair AUC',
            'mean fold AUC via sklearn','same-model held-group q0.8 thresholds','manual PRMScore confusion counts',
            '90 model and objective replays; original Top10; preserved unknown labels'],methods=checked)
    (out/'RESULT_REVIEW.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    print('PASS:',len(checked),'methods/references; PB8, PRMB within/pooled, fold thresholds, PRMScore.')


if __name__=='__main__':main()
