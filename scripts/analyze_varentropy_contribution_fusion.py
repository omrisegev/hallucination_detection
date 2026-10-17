"""Describe saved coefficients and changed predictions, without refitting."""
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.varentropy_contribution_fusion import METHODS


def main():
    out=ROOT/'results/varentropy_contribution_fusion_v1'
    metrics=json.loads((out/'METRICS.json').read_text(encoding='utf8'))
    cases=json.loads((out/'ERROR_CASES.json').read_text(encoding='utf8'))
    con=sqlite3.connect(out/'CHECKPOINT.sqlite')
    W=np.full((metrics['n_answers'],len(METHODS),50),np.nan);E=W.copy()
    B=np.full((len(W),len(METHODS)),np.nan);uid=['']*len(W);active={m:[] for m in METHODS};maxerror=0.
    for i,blob,info in con.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        info=json.loads(info);uid[i]=info['uid'];maxerror=max(maxerror,info['raw50_max_error'])
        with np.load(io.BytesIO(blob)) as a:W[i]=a['weights'];E[i]=a['effective'];B[i]=a['intercepts']
        for m,d in info['diagnostics'].items():active[m].append(d['active_columns'])
    con.close()
    if not all(uid) or len(set(uid))!=len(uid):raise ValueError('missing/duplicate coefficient rows')
    np.savez_compressed(out/'COEFFICIENTS.npz',uids=np.asarray(uid),methods=np.asarray(METHODS),
        standardized_weights=W,effective_raw_weights=E,intercepts=B)
    weights={}
    for j,m in enumerate(METHODS):
        k=int(m.split('__')[0][1:]);w=W[:,j,:k];ok=np.isfinite(w).all(axis=1);w=w[ok]
        scale=np.abs(w).sum(axis=1,keepdims=True);share=np.divide(w,scale,out=np.zeros_like(w),where=scale>0)
        a=np.asarray(active[m])
        weights[m]=dict(valid=len(w),active_min=int(a.min()) if len(a) else None,
            active_median=float(np.median(a)) if len(a) else None,
            rank_groups_abs_share={label:float(np.abs(share[:,lo:min(hi,k)]).sum(axis=1).mean())
                for label,lo,hi in [('rank1',0,1),('ranks2_5',1,5),('ranks6_15',5,15),('ranks16_50',15,50)] if lo<k},
            answers_with_negative_weight=int((w<0).any(axis=1).sum()),
            mean_negative_abs_share=float(np.maximum(-share,0).sum(axis=1).mean()))
    changes={}
    for pair,rows in cases.items():
        lost=[r for r in rows if r['change']=='lost'];gained=[r for r in rows if r['change']=='gained']
        changes[pair]=dict(gained=len(gained),lost=len(lost),lost_to_early=sum(0<=r['after']<r['target'] for r in lost),
            lost_to_late=sum(r['after']>r['target'] for r in lost),lost_to_clean=sum(r['after']==-1 for r in lost))
    result=dict(scope='Descriptive coefficients and paired errors; no new fitting or selection.',
        raw50_full_token_replay_max_error=maxerror,weights=weights,changes=changes)
    (out/'DIAGNOSTICS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
