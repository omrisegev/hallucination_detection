from load import *
a,off,lab,S,tau,raw,names,surv,meta,mp=load()
fl=np.load('flags.npz'); n=len(a); L=np.diff(off)
isprm=~a.cell.str.startswith('pb_').values
byidx={m['idx']:m for m in meta.values()}; ids=a.id.values
nonc=[i for i in np.where(isprm)[0] if byidx[ids[i]]['classification']!='correct']
def counts(F):
    C=np.zeros((len(nonc),4))
    for r,i in enumerate(nonc):
        f=F[off[i]:off[i+1]]; err=np.zeros(L[i],bool)
        for s in byidx[ids[i]]['error_steps']:
            if 1<=s<=L[i]: err[s-1]=True
        # positive = valid kept: TP correct&kept, FP error&kept, TN error&flagged, FN correct&flagged
        C[r]=[(~err&~f).sum(),(err&~f).sum(),(err&f).sum(),(~err&f).sum()]
    return C
def ps(C):
    tp,fp,tn,fn=C[...,0],C[...,1],C[...,2],C[...,3]
    f1=2*tp/(2*tp+fp+fn); nf1=2*tn/(2*tn+fn+fp); return 0.5*(f1+nf1)
C={k:counts(fl[k]) for k in ('F0','F1','F2')}
for k in C: print(k,'pooled PRMScore %.16f'%ps(C[k].sum(0)))
# label vector agreement check: OOF labels vs metadata error steps
lab_ok=all(np.array_equal(lab[off[i]:off[i+1]]==1, np.isin(np.arange(1,L[i]+1),byidx[ids[i]]['error_steps'])) for i in np.where(isprm)[0])
print('OOF labels == metadata error_steps (in range):',lab_ok)
g=a.source_group.values[nonc]; ug,gi=np.unique(g,return_inverse=True); G=len(ug); print('groups',G,'answers',len(nonc))
Gc={k:np.zeros((G,4)) for k in C}
for k in C: np.add.at(Gc[k],gi,C[k])
rng=np.random.default_rng(777); B=2000
W=np.stack([np.bincount(rng.integers(0,G,G),minlength=G) for _ in range(B)]).astype(float)
base=ps(Gc['F0'].sum(0))
for k,nm in (('F2','R2'),('F1','R1')):
    d=ps(W@Gc[k])-ps(W@Gc['F0']); pt=ps(Gc[k].sum(0))-base
    q=lambda lo:np.quantile(d,[lo,1-lo])
    print(nm,'-R0 point %+.5f  boot mean %+.5f sd %.5f  95%% %s  97.5%% %s  99%% %s  P(d<=0)=%.4f'%(pt,d.mean(),d.std(),q(.025).round(4),q(.0125).round(4),q(.005).round(4),(d<=0).mean()))
d=ps(W@Gc['F2'])-ps(W@Gc['F0'])
for lv in (0.05/4,0.05/3,0.05/2):
    print('R2 level %.4f'%(1-lv), np.quantile(d,[lv/2,1-lv/2]).round(4))
d=ps(W@Gc['F1'])-ps(W@Gc['F0'])
for lv in (0.05/5,0.05/4):
    print('R1 level %.4f'%(1-lv), np.quantile(d,[lv/2,1-lv/2]).round(4))
