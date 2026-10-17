"""Same saved RBM and coefficient correction, two alternative learning losses."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

RIDGE=.01


class Top10:
    def __init__(self,x,spans):
        self.x=x;self.spans=np.asarray(spans,int);self.groups=[]
        lengths=np.diff(self.spans,axis=1).ravel()
        if np.any(lengths<=0) or np.any(self.spans<0) or np.any(self.spans[:,1]>len(x)):
            raise ValueError('invalid step spans')
        buckets=np.ceil(np.log2(lengths)).astype(int)
        for bucket in np.unique(buckets):
            steps=np.flatnonzero(buckets==bucket);width=int(lengths[steps].max())
            relative=np.arange(width)[None,:];valid=relative<lengths[steps,None]
            idx=np.minimum(self.spans[steps,0,None]+relative,len(x)-1)
            self.groups.append((steps,idx,valid,np.minimum(10,lengths[steps])))

    def evaluate(self,token,derivative=False):
        values=np.empty(len(self.spans));jac=np.zeros((len(values),self.x.shape[1]+1)) if derivative else None
        for steps,idx,valid,k in self.groups:
            s=np.where(valid,token[idx],-np.inf)
            # Stable descending order also defines the subgradient at exact ties.
            top=np.argsort(-s,axis=1,kind='stable')[:,:min(10,s.shape[1])]
            chosen=np.take_along_axis(idx,top,axis=1);use=np.arange(top.shape[1])[None,:]<k[:,None]
            values[steps]=np.where(use,token[chosen],0.).sum(axis=1)/k
            if derivative:
                jac[steps,:-1]=(self.x[chosen]*use[:,:,None]).sum(axis=1)/k[:,None]
                jac[steps,-1]=1.
        return values,jac


class MatchedObjective:
    def __init__(self,x,offsets,a,w,b,orientation,active,base,spans,labels,step_owner):
        self.x=x;self.offsets=offsets;self.a=a;self.w=w;self.b=b;self.orientation=orientation
        self.active=active;self.base=base;self.n=len(a);self.top=Top10(x,spans)
        self.owner=np.repeat(np.arange(self.n),np.diff(offsets));self.o=orientation[self.owner]
        self.mass=1./(np.diff(offsets)[self.owner]*self.n)
        self.visible=.5*np.sum(self.mass[:,None]*(x-a[self.owner])**2)
        known=(labels==0)|(labels==1);self.y=np.where(known,labels,0)
        counts=np.bincount(step_owner[known],minlength=self.n)
        self.label_mass=np.zeros(len(labels));self.label_mass[known]=1./counts[step_owner[known]]
        for label in (0,1):
            mask=known&(labels==label);total=self.label_mass[mask].sum()
            if total<=0:raise ValueError('missing supervised training class')
            self.label_mass[mask]*=.5/total

    def unsupervised(self,delta):
        dw,db=delta[:-1],delta[-1]
        raww=self.w+self.orientation[:,None]*self.active*dw
        rawb=self.b+self.orientation*db
        s=rawb+np.sum(self.a*raww,axis=1)+.5*np.sum(raww**2,axis=1)
        raw=self.o*(self.base+self.x@dw+db)
        loss=self.visible-np.sum(self.mass*np.logaddexp(0,raw))+np.logaddexp(0,s).mean()
        prior=expit(s);post=expit(raw);r=self.mass*self.o*post
        gw=np.mean((prior*self.orientation)[:,None]*(self.a+raww)*self.active,axis=0)-self.x.T@r
        gb=np.mean(prior*self.orientation)-r.sum()
        return float(loss+.5*RIDGE*(delta@delta)),np.r_[gw,gb]+RIDGE*delta

    def supervised(self,delta):
        score,jac=self.top.evaluate(self.base+self.x@delta[:-1]+delta[-1],True)
        loss=np.sum(self.label_mass*(np.logaddexp(0,score)-self.y*score))
        grad=jac.T@(self.label_mass*(expit(score)-self.y))
        return float(loss+.5*RIDGE*(delta@delta)),grad+RIDGE*delta


def fit(fun,p):
    initial=np.zeros(p+1);start=fun(initial)[0]
    result=minimize(fun,initial,jac=True,method='L-BFGS-B',
        options=dict(maxiter=200,ftol=1e-10,gtol=1e-6,maxls=40))
    end,grad=fun(result.x)
    if not np.isfinite(result.x).all() or not np.isfinite(end) or end>start+1e-8:
        raise RuntimeError('nonfinite or worsening fit')
    return result.x,dict(converged=bool(result.success),message=str(result.message),iterations=int(result.nit),
        initial_loss=float(start),final_loss=float(end),gradient_max=float(np.max(np.abs(grad))),
        delta_norm=float(np.linalg.norm(result.x)))
