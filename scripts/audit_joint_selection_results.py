"""Independent pairwise-AUC/count audit of complete Joint inclusion results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_digitfree_broad50_v1 import load_data,sha
from scripts.run_lsml_gate_locator_research_v1 import dump
from spectral_utils.digitfree_broad50 import masked_answer_standardize


def metrics(score,gate,data):
    target=data['target'];cells=data['cells'].astype(str);peaks=[];aucs=[]
    for i,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
        s=score[a:b];peaks.append(int(np.argmax(s)))
        if cells[i].startswith('prmbench'):
            y=data['labels'][a:b];pos=s[y==1];neg=s[y==0]
            if len(pos) and len(neg):aucs.append(float(np.mean((pos[:,None]>neg)+.5*(pos[:,None]==neg))))
    pred=np.where(gate,peaks,-1);f1=[]
    for c in sorted(set(cells)):
        if not c.startswith('pb_'):continue
        clean=(cells==c)&(target==-1);error=(cells==c)&(target>=0)
        ca=np.mean(pred[clean]==target[clean]);ea=np.mean(pred[error]==target[error])
        f1.append(2*ca*ea/(ca+ea) if ca+ea else 0.)
    return float(np.mean(f1)),float(np.mean(aucs)),len(aucs)


def main(top8=False):
    root=ROOT/'results'/('broad50_top8_v1' if top8 else 'joint_feature_selection_bocpd_v1')
    results=json.loads((root/'RESULTS.json').read_text());manifest=root/('FIT_MANIFEST.json' if top8 else 'MANIFEST.json')
    for p,h in json.loads(manifest.read_text())['hashes'].items():assert sha(p)==h,p
    data=load_data();x=data['x'];offsets=data['offsets']
    if top8:
        raw=np.full_like(x,np.nan);available=np.zeros(x.shape,bool);done=np.zeros(len(offsets)-1,bool)
        for p in (root/'extracted').glob('*.npz'):
            with np.load(p) as z:
                values=z['values'];mask=z['available'];cursor=0
                for i in z['indexes']:
                    assert not done[i];a,b=offsets[i:i+2];n=b-a;raw[a:b]=values[cursor:cursor+n];available[a:b]=mask[cursor:cursor+n];done[i]=True;cursor+=n
        assert done.all();x=masked_answer_standardize(raw,available,offsets)
    with np.load(ROOT/'results/joint_feature_selection_bocpd_v1/INPUTS.npz') as z:
        banks={'B50':x,**{f'B51_{k}':np.column_stack((x,z[k])) for k in ('bocpd','noreset')}}
    with np.load(root/'SCORES.npz') as z:scores={k:z[k] for k in z.files}
    gate=scores.pop('gate');np.testing.assert_array_equal(gate,data['gate'])
    maxerror=0.;selection=[];rowfold=np.repeat(data['folds'],np.diff(offsets))
    for path in sorted(root.glob('B*_fold*.json')):
        bank=path.name.split('_fold')[0];d=json.loads(path.read_text());held=rowfold==d['outer']
        for arm,w in d['weights'].items():
            expected=banks[bank][held]@np.asarray(w);actual=scores[bank+'__'+arm][held]
            maxerror=max(maxerror,float(np.max(np.abs(expected-actual))));np.testing.assert_allclose(expected,actual,atol=1e-12,rtol=1e-12)
        fit=d.get('selection')
        if fit:
            a=fit['automatic'];index=fit['automatic_index'];np.testing.assert_array_equal(a['active'],fit['path'][index]['active'])
            assert a['minimum_retention']>=.95;assert min(a['group_sizes'])>=2
            if index+1<len(fit['path']):assert fit['path'][index+1]['minimum_retention']<.95
            w=np.asarray(d['weights']['joint_auto']);assert np.all(w[np.setdiff1d(np.arange(len(w)),a['active'])]==0)
            selection.append(dict(bank=bank,fold=d['outer'],kept=len(a['active']),retention=a['minimum_retention']))
    replay={}
    for name,score in scores.items():
        assert np.isfinite(score).all();pb,within,n=metrics(score,gate,data);m=results['metrics'][name]
        np.testing.assert_allclose([pb,within],[m['pb'],m['within']],atol=1e-12,rtol=0)
        assert n==m['within_n'];replay[name]=dict(pb=pb,within=within)
    np.testing.assert_allclose(replay['historical_bocpd']['pb'],.4036763497979659,atol=1e-12)
    audit=dict(status='PASS',answers=len(data['target']),steps=len(x),metrics_checked=len(replay),
        score_replay_max_error=maxerror,selection=selection,score_sha256=sha(root/'SCORES.npz'),
        checks=['manifest hashes','all fold weight maps','selected support and95%first-crossing rule',
                'full independent pair-comparison AUROC','independent PB count formula','historical BOCPD replay'])
    dump(root/'AUDIT.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--top8',action='store_true');main(p.parse_args().top8)
