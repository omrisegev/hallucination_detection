"""Independent arithmetic replay of the completed temporal benchmark metrics."""
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
    args=parser.parse_args();out=ROOT/'results/direct_probability_temporal_v3'
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
    with (args.source_root/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl').open('rb') as f:
        meta={str(r['idx']):r for r in pickle.load(f).values()}
    checked={}
    for name,metric in result['metrics'].items():
        flat=scores['steps__'+name];valid=scores['valid__'+name];pred=scores['prediction__'+name]
        actual_valid=np.array([np.isfinite(flat[offsets[i]:offsets[i+1]]).all() and offsets[i]<offsets[i+1] for i in range(len(records))])
        np.testing.assert_array_equal(valid,actual_valid)
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
        if pooled_y:np.testing.assert_allclose(roc_auc_score(pooled_y,pooled_s),metric['prm_pooled'],atol=1e-14)
        tp=tn=fp=fn=0
        for fold in sorted(set(outer[~pb])):
            train=np.flatnonzero(~pb & valid & (outer!=fold));test=np.flatnonzero(~pb & valid & (outer==fold))
            assert not set(records[i]['group_id'] for i in train)&set(records[i]['group_id'] for i in test)
            if not len(train):continue
            q=float(np.quantile(np.concatenate([flat[offsets[i]:offsets[i+1]] for i in train]),.8))
            np.testing.assert_allclose(q,metric['prmscore_thresholds'][str(fold)],atol=1e-14)
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
            'pooled AUC via sklearn','held-group q0.8 thresholds','manual PRMScore confusion counts'],methods=checked)
    (out/'RESULT_REVIEW.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    print('PASS:',len(checked),'methods/references; PB8, PRMB within/pooled, fold thresholds, PRMScore.')


if __name__=='__main__':main()
