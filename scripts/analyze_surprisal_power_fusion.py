"""Export saved polynomial coefficients and describe errors; no refitting."""
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.surprisal_power_fusion import METHODS


def main():
    out=ROOT/'results/surprisal_power_fusion_v1'
    m=json.loads((out/'METRICS.json').read_text(encoding='utf8'))
    con=sqlite3.connect(out/'CHECKPOINT.sqlite')
    W=np.full((m['n_answers'],6,48),np.nan);E=W.copy();B=np.full((len(W),6),np.nan)
    uid=['']*len(W);active={name:[] for name in METHODS};max_error=0.
    for i,blob,info in con.execute('SELECT idx,payload,info FROM answers ORDER BY idx'):
        info=json.loads(info);uid[i]=info['uid'];max_error=max(max_error,info['raw50_max_error'])
        with np.load(io.BytesIO(blob)) as a:W[i]=a['weights'];E[i]=a['effective'];B[i]=a['intercepts']
        for name,d in info['diagnostics'].items():active[name].append(d['active_columns'])
    con.close()
    if not all(uid) or len(set(uid))!=len(uid):raise ValueError('coefficient UID mismatch')
    np.savez_compressed(out/'COEFFICIENTS.npz',uids=np.asarray(uid),methods=np.asarray(METHODS),
        standardized_weights=W,effective_raw_weights=E,intercepts=B)
    summary={}
    for j,name in enumerate(METHODS):
        degree=int(name[1]);w=W[:,j,:16*degree];ok=np.isfinite(w).all(axis=1);w=w[ok]
        den=np.abs(w).sum(axis=1,keepdims=True);share=np.divide(w,den,out=np.zeros_like(w),where=den>0)
        shaped=share.reshape(len(w),degree,16);a=np.asarray(active[name])
        summary[name]=dict(valid=len(w),active_min=int(a.min()) if len(a) else None,
            active_median=float(np.median(a)) if len(a) else None,
            mean_absolute_share_by_power=np.abs(shaped).sum(axis=2).mean(axis=0).tolist() if len(w) else None,
            chosen_absolute_share_by_power=np.abs(shaped[:,:,15]).mean(axis=0).tolist() if len(w) else None,
            mean_negative_absolute_share=float(np.maximum(-share,0).sum(axis=1).mean()) if len(w) else None)
    cases=json.loads((out/'ERROR_CASES.json').read_text(encoding='utf8'));changes={}
    for name,rows in cases.items():
        lost=[r for r in rows if r['change']=='lost']
        changes[name]=dict(gained=sum(r['change']=='gained' for r in rows),lost=len(lost),
            lost_early=sum(0<=r['after']<r['target'] for r in lost),
            lost_late=sum(r['after']>r['target'] for r in lost),lost_to_clean=sum(r['after']==-1 for r in lost))
    report=dict(scope='Descriptive saved-coefficient/error analysis; not causal importance or new selection.',
        all_token_varentropy50_replay_max_error=max_error,weights=summary,changes=changes)
    (out/'DIAGNOSTICS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
