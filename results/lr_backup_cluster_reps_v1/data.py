import json, pickle, numpy as np, pandas as pd
from scipy.stats import rankdata
import os
# REPO_ROOT = repo checkout (or a staged copy of the input paths below)
U=os.environ.get('REPO_ROOT','/mnt/user-data/uploads/hallucination_detection').rstrip('/')+'/'
DIG=['digit_alternative','digit_spread','digit_alternative_innovation']
EPS=1e-12
def astd(x, off, active=None):
    out=np.zeros(x.shape)
    for a,b in zip(off[:-1],off[1:]):
        blk=x[a:b]
        if active is None:
            m=blk.mean(0); s=blk.std(0); out[a:b]=np.divide(blk-m,s,out=np.zeros_like(blk),where=s>EPS)
        else:
            for j in range(x.shape[1]):
                v=active[a:b,j]; y=blk[:,j][v]
                if len(y) and y.std()>EPS: out[a:b,j][v]=(y-y.mean())/y.std()
    return out
def load():
    F=np.load(U+'.worktrees/ssl-pseudolabel-residual-v1/results/external_banks_v4/source/FEATURES.npz')
    A=pd.read_csv(U+'scratch/backup_slide_lr_2026-10-18/ANSWERS_MIN.csv')  # uid,id,source_group,fold,cell,target from step_evidence_v1/OOF_ANSWERS.csv (branch claude/readout-quickest-detection-v1)
    J=np.load(U+'results/localization_full_benchmark_v3/evaluation/JOINED.npz')
    assert list(map(str,F['uids']))==A.uid.astype(str).tolist()
    assert np.array_equal(F['offsets'],J['offsets'])
    return F,A,J['labels'].astype(bool)
def bank(F,A,LAB,names,cell='prmbench_qwen3_8b'):
    nm=[str(x) for x in F['names']]; X=F['values'].astype(float); off=F['offsets'].astype(np.int64)
    ans=np.flatnonzero(A.cell.to_numpy()==cell)
    rows=np.concatenate([np.arange(off[i],off[i+1]) for i in ans]); o=np.r_[0,np.cumsum(np.diff(off)[ans])]
    Xs=X[rows]
    V=np.zeros((len(rows),len(names)))
    pl=[c for c in names if c not in DIG]
    if pl:
        Vp=astd(Xs[:,[nm.index(c) for c in pl]],o)
        for q,c in enumerate(pl): V[:,names.index(c)]=Vp[:,q]
    dg=[c for c in names if c in DIG]
    if dg:
        DA=F['digit_active'].astype(bool)[rows]
        Vd=astd(Xs[:,[nm.index(c) for c in DIG]],o,DA)
        for c in dg: V[:,names.index(c)]=Vd[:,DIG.index(c)]
    folds=np.repeat(A.fold.to_numpy()[ans],np.diff(o)); sg=A.source_group.to_numpy()[ans]
    return V,o,LAB[rows],folds,ans,sg
def auc(y,s):
    n1=int(y.sum()); n0=len(y)-n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum()-n1*(n1+1)/2)/(n1*n0)
def per_answer(s,o,y,nonctl):
    n=len(o)-1; a=np.full(n,np.nan); c1=np.zeros((n,4),int)
    for j in range(n):
        p,q=o[j],o[j+1]; ss=s[p:q]; yy=y[p:q]; a[j]=auc(yy,ss)
        if nonctl[j]:
            f=np.zeros(q-p,bool); f[np.lexsort((np.arange(q-p),-ss))[0]]=True
            v,g=~f,~yy; c1[j]=[(v&g).sum(),(v&~g).sum(),(~v&~g).sum(),(~v&g).sum()]
    return a,c1
def prmscore(c1):
    tp,fp,tn,fn=c1.sum(0)
    return .5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))
