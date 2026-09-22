"""Post-fit density/covariance and task-change diagnostics; no new fits."""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from scipy.special import expit,logsumexp
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_rbm_literature_completion as run
from spectral_utils import rbm_literature_completion as model


def model_moments(theta,p,h):
    a,w,b=model.unpack(theta,p,h);states=model.hidden_states(h);wh=states@w.T
    logp=states@b+wh@a+.5*np.square(wh).sum(axis=1)
    pi=np.exp(logp-logsumexp(logp));mu=a+wh;mean=pi@mu
    centered=mu-mean
    return mean,np.eye(p)+(centered*pi[:,None]).T@centered


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=('variance','capacity'),required=True);args=p.parse_args()
    out=run.PROGRAM/args.suite
    assert json.loads((out/'RUN_STATE.json').read_text())['status']=='COMPLETE'
    assert json.loads((out/'RESULT_REVIEW.json').read_text())['status']=='PASS'
    source=ROOT.parents[1];records,joined,reference=run.load_contract(source)
    db=sqlite3.connect((out/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    src=sqlite3.connect(run.modeldb(source).as_uri()+'?mode=ro',uri=True)
    pred=np.load(out/'SCORES.npz');rows=[];coef=[]
    with threadpool_limits(limits=1):
        for path in sorted(run.caches(source).glob('cache_*.npz')):
            with np.load(path) as z:cache={k:z[k] for k in z.files if k!='labels'}
            for k,i in enumerate(cache['ids']):
                i=int(i);data=run.prepare_answer(k,cache,src,records,joined,reference)
                _,uid,_,_,banks=data;blob,info=db.execute('select payload,info from answers where idx=?',(i,)).fetchone();info=json.loads(info)
                with np.load(io.BytesIO(blob)) as z:
                    for key,d in info['models'].items():
                        bank=int(key.split('_')[0][1:]);x=banks[bank]['x'];p=x.shape[1];theta=z[key+'::theta']
                        empirical=x.T@x/len(x);off=~np.eye(p,dtype=bool)
                        if d['type']=='rbm':
                            mean,cov=model_moments(theta,p,d['h']);nll=model.ExactRBM(x,d['h'])(theta)[0]
                        else:
                            obj=model.TwoStateVariance(x,d['separate']);mu,lv,eta=obj.decode(theta)
                            pi=np.array([expit(-eta),expit(eta)]);mean=pi@mu;centered=mu-mean
                            cov=np.diag(pi@np.exp(lv))+(centered*pi[:,None]).T@centered
                            parts,_=obj.component_logp(theta);nll=-logsumexp(parts,axis=1).mean()
                            quadratic=.5*(np.exp(-lv[0])-np.exp(-lv[1]))*d['orientation']
                            linear=(mu[1]*np.exp(-lv[1])-mu[0]*np.exp(-lv[0]))*d['orientation']
                            intercept=(eta-.5*np.sum(lv[1]-lv[0]+mu[1]**2*np.exp(-lv[1])-mu[0]**2*np.exp(-lv[0])))*d['orientation']
                            # Independent explicit polynomial must equal mixture log odds.
                            np.testing.assert_allclose(intercept+x@linear+(x*x)@quadratic,
                                d['orientation']*(parts[:,1]-parts[:,0]),atol=1e-9,rtol=1e-10)
                            for col,a,b in zip(banks[bank]['columns'],linear,quadratic):
                                coef.append(dict(uid=uid,bank=bank,method=key,column=int(col),linear=float(a),quadratic=float(b)))
                        original_nll=model.ExactRBM(x,1)(banks[bank]['theta'])[0]
                        mode='posterior' if bank==6 else 'logit';m=key+'_'+mode
                        ref=f'rbm{bank}__'+('old' if mode=='posterior' else 'logit_old')
                        t=int(joined['target'][i]);is_pb=records[i]['cell'].startswith('pb_')
                        new=int(pred['prediction__'+m][i]);old=int(pred['prediction__'+ref][i])
                        rows.append(dict(uid=uid,group_id=records[i]['group_id'],cell=records[i]['cell'],bank=bank,method=key,
                            data_nll=float(nll),nll_gain_vs_original=float(original_nll-nll),
                            mean_rmse=float(np.sqrt(np.mean((mean-x.mean(axis=0))**2))),
                            variance_rmse=float(np.sqrt(np.mean((np.diag(cov)-np.diag(empirical))**2))),
                            covariance_relative_error=float(np.linalg.norm(cov-empirical)/np.linalg.norm(empirical)),
                            offdiagonal_relative_error=float(np.linalg.norm((cov-empirical)[off])/np.linalg.norm(empirical[off])),
                            pb_gained=bool(is_pb and new==t and old!=t and pred['valid__'+m][i]),
                            pb_lost=bool(is_pb and old==t and (new!=t or not pred['valid__'+m][i]))))
            print('[mechanisms]',args.suite,path.stem,len(rows),flush=True)
    summaries=[]
    for m in sorted({r['method'] for r in rows}):
        sub=[r for r in rows if r['method']==m]
        for field in ('data_nll','nll_gain_vs_original','variance_rmse','covariance_relative_error','offdiagonal_relative_error'):
            v=np.array([r[field] for r in sub])
            summaries.append(dict(method=m,diagnostic=field,n=len(v),median=float(np.median(v)),mean=float(v.mean()),
                mean_gained=float(np.mean([r[field] for r in sub if r['pb_gained']])) if any(r['pb_gained'] for r in sub) else None,
                mean_lost=float(np.mean([r[field] for r in sub if r['pb_lost']])) if any(r['pb_lost'] for r in sub) else None))
    run.csv_write(out/'MODEL_MECHANISMS.csv',rows);run.csv_write(out/'MODEL_MECHANISM_SUMMARY.csv',summaries)
    if coef:run.csv_write(out/'LINEAR_QUADRATIC_COEFFICIENTS.csv',coef)
    run.base.atomic_json(out/'MECHANISM_REVIEW.json',dict(status='PASS',model_rows=len(rows),coefficient_rows=len(coef),
        note='Retrospective descriptive density/covariance diagnostics; gained/lost associations are not causal effects or significance tests.'))
    src.close();db.close();pred.close()


if __name__=='__main__':main()
