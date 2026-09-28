import sys; sys.path.insert(0,'C:/Users/omris/TAU/hallucination_detection/.worktrees/depth-feature-fusion-v1')
from load import *
from spectral_utils.prmbench import prmbench_evaluate
a,off,lab,S,tau,raw,names,surv,meta,mp=load()
fl=np.load('flags.npz'); n=len(a); L=np.diff(off)
isprm=~a.cell.str.startswith('pb_').values; aid=np.repeat(np.arange(n),L); sprm=isprm[aid]
metas=list(meta.values()); ids=a.id.values
def official(F):
    preds=[{'idx':ids[i],'labels':[0 if f else 1 for f in F[off[i]:off[i+1]]]} for i in np.where(isprm)[0]]
    t=prmbench_evaluate(preds,metas)['total']; return 0.5*(t['f1']+t['negative_f1'])
for k in ('F0','F1','F2'): print(k,'PRMB flag share %.4f'%fl[k][sprm].mean())
zS=np.empty_like(S)
for i in range(n):
    s,e=off[i],off[i+1]; x=S[s:e]; zS[s:e]=(x-x.mean())/max(x.std(),1e-8)
target=fl['F2'][sprm].mean()
t=np.quantile(zS[sprm],1-target); Fm=zS>=t
print('diagnostic: answer-z rule at matched global flag share %.4f -> PRMScore %.6f, controls-with-flag n/a'%(Fm[sprm].mean(),official(Fm)))
# proportional count: n_i = round(target*L_i) top-S
Fp=np.zeros_like(Fm)
for i in range(n):
    s,e=off[i],off[i+1]; ni=int(np.floor(target*(e-s)+0.5))
    if ni: Fp[s+np.argsort(-S[s:e],kind='stable')[:ni]]=True
print('diagnostic: proportional count round(%.4f*L_i), top-S -> share %.4f PRMScore %.6f'%(target,Fp[sprm].mean(),official(Fp)))
