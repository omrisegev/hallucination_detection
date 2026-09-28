import sys; sys.path.insert(0,'C:/Users/omris/TAU/hallucination_detection/.worktrees/depth-feature-fusion-v1')
from load import *
from spectral_utils.prmbench import prmbench_evaluate
DDOF=int(sys.argv[1]) if len(sys.argv)>1 else 0
a,off,lab,S,tau,raw,names,surv,meta,mp=load()
n=len(a); L=np.diff(off); fold=a.fold.values
isprm=~a.cell.str.startswith('pb_').values
aid=np.repeat(np.arange(n),L); sfold=fold[aid]; sprm=isprm[aid]
# full-population check of S reconstruction (sd>1e-12 else 0)
Zs=np.empty_like(raw)
for i in range(n):
    s,e=off[i],off[i+1]; x=raw[s:e]; sd=x.std(0); Zs[s:e]=np.where(sd>1e-12,(x-x.mean(0))/np.where(sd>1e-12,sd,1),0.)
print('S recon maxdiff (sd>1e-12 conv)', np.abs(Zs[:,surv].mean(1)-S).max())
# R0
zS=np.empty_like(S)
for i in range(n):
    s,e=off[i],off[i+1]; x=S[s:e]; zS[s:e]=(x-x.mean())/max(x.std(),1e-8)
F0=zS>=tau[sfold]
# R1
G=np.empty_like(S); tauG=np.zeros(5)
X=raw[:,surv]
for k in range(5):
    c=(k+1)%5; fit=(sfold!=k)&(sfold!=c)
    mu=X[fit].mean(0); sd=X[fit].std(0,ddof=DDOF)
    Gk=((X-mu)/sd).mean(1)
    tauG[k]=np.quantile(Gk[(sfold==c)&sprm],0.8)
    ev=sfold==k; G[ev]=Gk[ev]
F1=G>=tauG[sfold]
print('tauG',tauG)
# R2
F2=np.zeros_like(F0)
for i in range(n):
    s,e=off[i],off[i+1]; ni=int(F1[s:e].sum())
    if ni: o=np.argsort(-S[s:e],kind='stable')[:ni]; F2[s+o]=True
assert (np.add.reduceat(F2,off[:-1])==np.add.reduceat(F1,off[:-1])).all()
np.savez('flags.npz',F0=F0,F1=F1,F2=F2,G=G)
# official PRMScore
metas=list(meta.values()); byidx={m['idx']:m for m in metas}
ids=a.id.values
cls=np.array([byidx[ids[i]]['classification'] if isprm[i] else '' for i in range(n)])
nst_ok=all(byidx[ids[i]]['n_steps']==L[i] for i in np.where(isprm)[0]); print('n_steps match',nst_ok)
def official(F):
    preds=[{'idx':ids[i],'labels':[0 if f else 1 for f in F[off[i]:off[i+1]]]} for i in np.where(isprm)[0]]
    t=prmbench_evaluate(preds,metas)['total']; return 0.5*(t['f1']+t['negative_f1']),t
res={}
for nm,F in (('R0',F0),('R1',F1),('R2',F2)):
    ps,t=official(F); res[nm]=ps
    print(nm,'PRMScore %.16f'%ps,'f1 %.4f negf1 %.4f'%(t['f1'],t['negative_f1']))
ctl=np.where(isprm&(cls=='correct'))[0]; nonc=np.where(isprm&(cls!='correct'))[0]
print('controls',len(ctl),'noncontrol',len(nonc))
cnt=lambda F,idx: np.array([F[off[i]:off[i+1]].sum() for i in idx])
for nm,F in (('R0',F0),('R1',F1),('R2',F2)):
    fc=cnt(F,ctl); lc=L[ctl]; fn=cnt(F,nonc)
    print(nm,'ctl any-flag %.4f pooled step-rate %.4f mean-per-answer rate %.4f | noncontrol no-flag %.4f (%d)'%((fc>0).mean(),fc.sum()/lc.sum(),(fc/lc).mean(),(fn==0).mean(),(fn==0).sum()))
np.save('res_official.npy',res)
