"""Separate model and metric arithmetic for the supervised position diagnostic."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--result-dir',type=Path,default=ROOT/'results/rbm_supervised_position_diagnostic_v1')
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
    supervised={'supervised_static','supervised_prior','supervised_conditional'}
    # Independently rebuild the design without calling the training module.
    features=np.load(out/'FEATURES.npz');xf=features['x'];context=features['context']
    np.testing.assert_array_equal(features['uid'],[r['uid'] for r in records])
    stepfold=np.repeat(outer,np.diff(offsets));stepprm=np.repeat(~pb,np.diff(offsets))
    calibration={m:{} for m in supervised};models=0
    for path in sorted(out.glob('fit_*.npz')):
        info=json.loads(path.with_suffix('.json').read_text());cell=info['cell'];fold=info['fold']
        train=np.flatnonzero((cells==cell)&(outer!=fold));test=np.flatnonzero((cells==cell)&(outer==fold))
        assert not set(records[i]['group_id'] for i in train)&set(records[i]['group_id'] for i in test)
        ts=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in train]);vs=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in test])
        with np.load(path) as z:
            np.testing.assert_array_equal(z['train_answers'],train);np.testing.assert_array_equal(z['test_answers'],test)
            np.testing.assert_array_equal(z['train_steps'],ts);np.testing.assert_array_equal(z['test_steps'],vs)
            mean=xf[ts].mean(axis=0);sd=xf[ts].std(axis=0);sd=np.where(sd>1e-10,sd,1.)
            np.testing.assert_array_equal(z['mean'],mean);np.testing.assert_array_equal(z['scale'],sd)
            x=(xf-mean)/sd
            for method in supervised:
                theta=z[method+'_theta'];score=x@theta[:12]
                if method!='supervised_static':score+=context*theta[12]
                if method=='supervised_conditional':score+=context*(x@theta[13:25])
                if not cell.startswith('pb_'):score+=theta[-1]
                np.testing.assert_allclose(score[vs],z[method+'_test'],atol=1e-11,rtol=1e-12)
                np.testing.assert_array_equal(z[method+'_test'],scores['steps__'+method][vs])
                assert info['fits'][method]['converged']
                if not cell.startswith('pb_'):
                    np.testing.assert_allclose(score[ts],z[method+'_calibration'],atol=1e-11,rtol=1e-12)
                    q=float(np.quantile(score[ts],.8));calibration[method][str(fold)]=q
                    np.testing.assert_allclose(q,info['fits'][method]['threshold'],atol=1e-11,rtol=1e-12)
                models+=1
    assert models==135
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
        if pooled_y and name not in supervised:np.testing.assert_allclose(roc_auc_score(pooled_y,pooled_s),metric['prm_pooled'],atol=1e-14)
        if name in supervised:assert metric['prm_pooled'] is None
        foldaucs=[]
        for fold in sorted(set(outer[~pb])):
            mask=stepprm&(stepfold==fold)&(labels>=0)
            v=roc_auc_score(labels[mask]==1,flat[mask]);foldaucs.append(v)
            np.testing.assert_allclose(v,metric['prm_fold_aucs'][str(fold)],atol=1e-14)
        np.testing.assert_allclose(np.mean(foldaucs),metric['prm_fold_auc'],atol=1e-14)
        tp=tn=fp=fn=0
        for fold in sorted(set(outer[~pb])):
            train=np.flatnonzero(~pb & valid & (outer!=fold));test=np.flatnonzero(~pb & valid & (outer==fold))
            assert not set(records[i]['group_id'] for i in train)&set(records[i]['group_id'] for i in test)
            if not len(train):continue
            q=(calibration[name][str(fold)] if name in supervised else
               float(np.quantile(np.concatenate([flat[offsets[i]:offsets[i+1]] for i in train]),.8)))
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
    payload=dict(status='PASS',models_replayed=models,scope='Separate arithmetic implementation in the same Codex session; not an external independent scientist review.',
        checks=['code/contract hashes','all PB cells with full denominators','PRMB direct-pair AUC',
            'fold AUC via sklearn; no supervised pooled OOF AUC','same-model held-group q0.8 thresholds',
            '135 saved-model replay and training-only scaling','manual PRMScore confusion counts'],methods=checked)
    (out/'RESULT_REVIEW.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    print('PASS:',len(checked),'methods/references; PB8, PRMB within/pooled, fold thresholds, PRMScore.')


if __name__=='__main__':main()
