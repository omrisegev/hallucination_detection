"""Explain every changed PB decision in the registered bank12 variance contrast.

No refits, alternative scores, selected thresholds or new benchmark candidates.
Labels identify gained/lost cases after the full experiment. The polynomial
decomposition explains the fitted score, not a causal effect of retraining.
"""
import csv
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_rbm_literature_completion as run
from spectral_utils import rbm_literature_completion as model


def main():
    source=ROOT.parents[1];out=run.PROGRAM/'variance'
    assert json.loads((out/'RESULT_REVIEW.json').read_text())['status']=='PASS'
    comparison='b12_variance_separate_logit_minus_b12_variance_shared_logit'
    with (out/'CHANGED_SUCCESSES.csv').open(encoding='utf8') as f:
        changed={r['uid']:r for r in csv.DictReader(f) if r['comparison']==comparison}
    contrast=json.loads((out/'METRICS.json').read_text())['contrasts'][comparison]
    assert len(changed)==contrast['gained']+contrast['lost']
    records,joined,reference=run.load_contract(source)
    db=sqlite3.connect((out/'CHECKPOINT.sqlite').as_uri()+'?mode=ro',uri=True)
    src=sqlite3.connect(run.modeldb(source).as_uri()+'?mode=ro',uri=True)
    rows=[]
    with threadpool_limits(limits=1):
        for path in sorted(run.caches(source).glob('cache_pb_*.npz')):
            with np.load(path) as z:cache={k:z[k] for k in z.files if k!='labels'}
            for k,i in enumerate(cache['ids']):
                i=int(i);uid=records[i]['uid']
                if uid not in changed:continue
                _,_,spans,anchor,banks=run.prepare_answer(k,cache,src,records,joined,reference)
                x=banks[12]['x'];p=x.shape[1];key='b12_variance_separate'
                blob,info=db.execute('select payload,info from answers where idx=?',(i,)).fetchone()
                d=json.loads(info)['models'][key]
                with np.load(io.BytesIO(blob)) as z:
                    theta=z[key+'::theta'];step=z['score::'+key+'_logit']
                obj=model.TwoStateVariance(x,True);mu,lv,eta=obj.decode(theta)
                quad=.5*(np.exp(-lv[0])-np.exp(-lv[1]))*d['orientation']
                linear=(mu[1]*np.exp(-lv[1])-mu[0]*np.exp(-lv[0]))*d['orientation']
                intercept=(eta-.5*np.sum(lv[1]-lv[0]+mu[1]**2*np.exp(-lv[1])-mu[0]**2*np.exp(-lv[0])))*d['orientation']
                L=x@linear;Q=(x*x)@quad;score=intercept+L+Q
                parts,_=obj.component_logp(theta)
                np.testing.assert_allclose(score,d['orientation']*(parts[:,1]-parts[:,0]),atol=1e-9,rtol=1e-10)
                case=changed[uid];target=int(case['target']);chosen=int(case['peak_after'])
                def summarize(j):
                    a,b=spans[j];ix=np.argsort(score[a:b],kind='stable')[-min(10,b-a):]+a
                    np.testing.assert_allclose(score[ix].mean(),step[j],atol=1e-9,rtol=1e-10)
                    return dict(score=float(score[ix].mean()),linear=float(L[ix].mean()),
                                quadratic=float(Q[ix].mean()),anchor=float(anchor[ix].mean()),
                                low_anchor_fraction=float(np.mean(anchor[ix]<np.median(anchor))))
                selected=summarize(chosen);truth=summarize(target)
                lm=selected['linear']-truth['linear'];qm=selected['quadratic']-truth['quadratic']
                np.testing.assert_allclose(lm+qm,step[chosen]-step[target],atol=2e-9,rtol=1e-10)
                post=expit(score)
                corr=lambda y:float(np.corrcoef(y,anchor)[0,1]) if np.std(y)>1e-12 else None
                rows.append(dict(uid=uid,group_id=records[i]['group_id'],cell=records[i]['cell'],
                    change=case['change'],target=target,chosen=chosen,offset=chosen-target,
                    score_margin=float(step[chosen]-step[target]),linear_margin=lm,quadratic_margin=qm,
                    quadratic_reverses_linear=(lm<0 and lm+qm>0),
                    chosen_anchor=selected['anchor'],target_anchor=truth['anchor'],
                    chosen_low_anchor_fraction=selected['low_anchor_fraction'],
                    logit_anchor_correlation=corr(score),posterior_anchor_correlation=corr(post),
                    variance_ratio_min=float(np.exp(lv[1]-lv[0]).min()),
                    variance_ratio_max=float(np.exp(lv[1]-lv[0]).max())))
            print('[variance losses]',path.stem,len(rows),flush=True)
    assert {r['uid'] for r in rows}==set(changed)
    summaries={}
    for kind in ('gained','lost'):
        sub=[r for r in rows if r['change']==kind]
        summaries[kind]=dict(n=len(sub),early=sum(r['offset']<0 for r in sub),late=sum(r['offset']>0 for r in sub),
            quadratic_reverses_linear=sum(r['quadratic_reverses_linear'] for r in sub),
            negative_logit_anchor_correlation=sum(r['logit_anchor_correlation'] is not None and r['logit_anchor_correlation']<0 for r in sub),
            majority_selected_tokens_below_anchor_median=sum(r['chosen_low_anchor_fraction']>.5 for r in sub),
            median_selected_low_anchor_fraction=float(np.median([r['chosen_low_anchor_fraction'] for r in sub])),
            median_linear_margin=float(np.median([r['linear_margin'] for r in sub])),
            median_quadratic_margin=float(np.median([r['quadratic_margin'] for r in sub])))
    run.csv_write(out/'VARIANCE_LOSS_DECOMPOSITION.csv',rows)
    run.base.atomic_json(out/'VARIANCE_LOSS_DECOMPOSITION.json',dict(status='PASS',comparison=comparison,
        changed_answers=len(rows),summary=summaries,script_sha256=run.base.old.sha256_file(Path(__file__)),
        note='All changed erroneous PB answers. Exact decomposition of fitted Logit/Top10 margins; no refit or alternate benchmark. Conditional descriptive analysis, not a label-free selection rule.'))
    src.close();db.close()


if __name__=='__main__':main()
