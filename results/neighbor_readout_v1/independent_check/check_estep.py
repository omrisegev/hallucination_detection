import itertools, numpy as np, sys
sys.argv=['x']
import h1_indep as H
rng=np.random.default_rng(0)
n=np.array([1,2,4,3]); off=np.concatenate([[0],np.cumsum(n)]); s=rng.normal(size=off[-1])
D=H.Data(s,off); mu=np.array([-.2,1.]); sd=np.array([.8,1.3]); rho=.6; a=-1.; b=.7
ll,gam,xi,lp=H.estep(D,mu,sd,rho,a,b)
from scipy.stats import norm
tot=0; post=[]
for j,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
    m=hi-lo; Z=0; P=np.zeros(m)
    for z in itertools.product([0,1],repeat=m):
        p=1.
        for t in range(m):
            e=1/(1+np.exp(-(a+b*(t+.5)/m)))
            prev=0 if t==0 else z[t-1]
            pz1 = e if prev==0 else rho
            p*= pz1 if z[t]==1 else 1-pz1
            p*= norm.pdf(s[lo+t],mu[z[t]],sd[z[t]])
        Z+=p; P+=p*np.array(z)
    tot+=np.log(Z); post+=list(P/Z)
post=np.array(post)
lo_=(lp[:,:,1]-lp[:,:,0])[D.M]
print(ll,tot, np.abs(1/(1+np.exp(-lo_))-post).max(), np.abs(gam[D.M][:,1]-post).max())
