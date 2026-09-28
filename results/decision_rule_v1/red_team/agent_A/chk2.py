from load import *
a,off,lab,S,tau,raw,names,surv,meta,mp=load()
n=len(a)
Z=np.empty_like(raw); SD=np.empty_like(raw)
for i in range(n):
    s,e=off[i],off[i+1]; x=raw[s:e]; m=x.mean(0); sd=x.std(0)
    Z[s:e]=(x-m)/np.maximum(sd,1e-8); SD[s:e]=sd
dif=np.abs(Z[:,surv].mean(1)-S)
bad=np.where(dif>1e-9)[0]; print(len(bad))
ai=np.searchsorted(off,bad,side='right')-1; ua=np.unique(ai); print('answers',len(ua))
for i in ua[:5]:
    s,e=off[i],off[i+1]; print(i,a.cell[i],e-s, dif[s:e].max()); print(' sd',SD[s][surv].round(6))
    print(' Z',Z[s:e][:, surv].round(3)[:3]); print(' S',S[s:e][:3])
