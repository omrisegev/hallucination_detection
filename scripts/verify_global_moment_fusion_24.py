"""Separate score/metric arithmetic replay from saved cell artifacts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from scipy.special import expit


def rank_auc(y,s):
    y=np.asarray(y,bool);n=int(y.sum());m=len(y)-n
    if not n or not m:raise ValueError('single class')
    return float((rankdata(s,method='average')[y].sum()-n*(n+1)/2)/(n*m))


def main():
    p=argparse.ArgumentParser();p.add_argument('--result-dir',type=Path,required=True);args=p.parse_args();out=args.result_dir
    manifest=json.loads((out/'MANIFEST.json').read_text());result=json.loads((out/'METRICS.json').read_text())
    for name,digest in manifest['hashes'].items():
        if 'dataset_cache' in Path(name).parts:continue
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    checked={}
    for cell,row in result['cells'].items():
        with np.load(out/'cells'/cell/'SCORES.npz') as z:
            y=z['labels'];X=z['X'];assert len(y)==row['n']==len(X)
            np.testing.assert_array_equal(z['score__var15_raw'],X[:,1])
            for method,expected in row['auroc'].items():
                s=z['score__'+method]
                if expected is None:
                    assert not np.isfinite(s).all();continue
                np.testing.assert_allclose(rank_auc(y,s),expected,atol=1e-12,rtol=0)
                if method.startswith('ref_') or method in ('var15_raw','var50_raw'):continue
                d=row['diagnostics'][method];cols=d['columns'];mean=np.asarray(d['normalization_mean']);scale=np.asarray(d['normalization_scale'])
                data=z['C'] if method.startswith('var15_') else X
                Z=(data[:,cols]-mean[cols])/scale[cols]
                if method.startswith('var15_'):pred=Z@z[method+'::w']
                elif method in ('equal','iu'):pred=d['orientation']*(Z@z[method+'::w'])
                elif method.startswith('rbm'):
                    pred=expit(z[method+'::b']+Z@z[method+'::w']);pred=pred if d['orientation']>0 else 1-pred
                else:
                    prefix=method+'::';hidden=np.tanh(Z@z[prefix+'W::moments'].T+z[prefix+'d::moments'])
                    parts=Z*z[prefix+'w::moments']+(2/Z.shape[1])*np.tanh(hidden@z[prefix+'V::moments'].T+z[prefix+'e::moments'])
                    pred=expit(z[prefix+'b']+parts.sum(axis=1));pred=pred if d['orientation']>0 else 1-pred
                np.testing.assert_allclose(pred,s,atol=1e-10,rtol=1e-10,err_msg=cell+' '+method)
            checked[cell]=dict(n=len(y),methods=len(row['auroc']))
    assert len(checked)==24
    for m,macro in result['macro'].items():
        for name,domain in (('all24',None),('qa9','QA'),('math15','math')):
            values=[r['auroc'][m] for r in result['cells'].values() if domain is None or r['domain']==domain]
            if any(v is None for v in values):assert macro[name] is None
            else:np.testing.assert_allclose(np.mean(values),macro[name],atol=1e-12)
    (out/'RESULT_REVIEW.json').write_text(json.dumps(dict(status='PASS',scope='Separate arithmetic in same session; not external scientific review',
        checks=['artifact hashes','rank AUROC with ties','all cell/macros','normalization and fitted state replay','raw Var15 input identity'],cells=checked),indent=2)+'\n')
    print('PASS all24 cells, metric arithmetic and fitted-state replay',flush=True)


if __name__=='__main__':main()
