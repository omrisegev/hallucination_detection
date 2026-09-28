from load import *
a,off,lab,S,tau,raw,names,surv,meta,mp=load()
print(len(surv),[names[i] for i in surv]); print('nan raw',np.isnan(raw).sum(0), 'nan S',np.isnan(S).sum())
n=len(a); L=np.diff(off); print('len min',L.min(), (L==1).sum())
Z=np.empty_like(raw)
for ddof in (0,1):
  for i in range(n):
    s,e=off[i],off[i+1]; x=raw[s:e]; m=x.mean(0); sd=x.std(0,ddof=ddof) if e-s>ddof else np.zeros(x.shape[1])
    Z[s:e]=(x-m)/np.maximum(sd,1e-8)
  print(ddof, np.abs(Z[:,surv].mean(1)-S).max())
