from load import *
a,off,lab,S,tau,raw,names,surv,meta,mp=load()
for i in (2485,4410):
    s,e=off[i],off[i+1]; x=raw[s:e]
    np.set_printoptions(precision=17)
    print(i, x[:, [names.index('logprob_margin'), names.index('true_tail50')]], x.std(0)[[3,4]])
    for eps in (1e-8,1e-6,1e-12):
        sd=x.std(0); z=np.where(sd>eps,(x-x.mean(0))/np.where(sd>eps,sd,1),0.0)
        print(eps, np.abs(z[:,surv].mean(1)-S[s:e]).max())
