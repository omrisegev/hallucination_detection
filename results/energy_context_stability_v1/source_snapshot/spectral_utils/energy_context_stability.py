"""Target-label-free context moments and fusion stability diagnostics."""
import numpy as np
from scipy.spatial import cKDTree
from .cca_iu_isolation import iu_moments, simplex_qp

FLOOR=1e-6


def history_landmarks(x, count=16, lag=16):
    x=np.asarray(x,float);n=len(x)
    if n<=lag: return np.array([],int),np.empty((0,x.shape[1])),np.empty((0,x.shape[1]))
    positions=np.unique(np.linspace(lag,n-1,min(count,n-lag),dtype=int))
    windows=x[positions[:,None]-np.arange(lag,0,-1)[None,:]]
    return positions,windows.mean(axis=1),(windows**2).mean(axis=1)


def group_weights(groups,answers=None):
    if answers is not None:
        names,first,inverse,counts=np.unique(answers,return_index=True,return_inverse=True,return_counts=True)
        answer_groups=np.asarray(groups)[first]
        _,group_inverse,group_counts=np.unique(answer_groups,return_inverse=True,return_counts=True)
        return 1./counts[inverse]/group_counts[group_inverse[inverse]]/len(group_counts)
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    return 1./counts[inverse]/len(counts)


def moments(x,w):
    w=np.asarray(w,float);w=w/w.sum();mu=w@x;z=x-mu
    return mu,(z*w[:,None]).T@z


def position_basis(position,length):
    centers=(np.arange(16)+.5)/16
    p=np.clip(position,centers[0],centers[-1]);right=np.clip(np.searchsorted(centers,p),1,15);left=right-1
    fraction=(p-centers[left])/(centers[right]-centers[left]);b=np.zeros((len(p),65));b[:,0]=1
    strata=np.searchsorted([256,512,1024],length,side='left')
    b[np.arange(len(p)),1+16*strata+left]=1-fraction
    b[np.arange(len(p)),1+16*strata+right]=fraction
    return b


class ContextFit:
    def fit(self,x,hm,hs,position,length,groups,answers=None):
        weights=group_weights(groups,answers)
        self.mean,C=moments(x,weights);self.sd=np.sqrt(np.maximum(np.diag(C),1e-12))
        z=(x-self.mean)/self.sd
        _,self.C=moments(z,weights);self.C+=FLOOR*np.eye(x.shape[1])
        self.var_y=.25*np.trace(self.C)/len(self.sd)
        energy=self.energy(hm,hs)
        design=position_basis(position,length)
        ridge=.01*np.eye(design.shape[1]);ridge[0,0]=0
        self.profile=np.linalg.solve(design.T@(weights[:,None]*design)+ridge,design.T@(weights[:,None]*energy))
        residual=energy-design@self.profile
        self.rmean,rcov=moments(residual,weights);self.rsd=np.sqrt(np.maximum(np.diag(rcov),1e-12))
        pos=np.column_stack([position,np.log1p(length)])
        self.pmean,pcov=moments(pos,weights);self.psd=np.sqrt(np.maximum(np.diag(pcov),1e-12))
        return self

    def energy(self,hm,hs):
        e=(hs-2*hm*self.mean+self.mean**2)/self.sd**2
        return np.log1p(np.maximum(e,0.))

    def transform(self,hm,hs,position,length):
        p=(np.column_stack([position,np.log1p(length)])-self.pmean)/self.psd/np.sqrt(2)
        e=self.energy(hm,hs)-position_basis(position,length)@self.profile
        r=(e-self.rmean)/self.rsd/np.sqrt(len(self.sd))
        return p,np.column_stack([p,r])

    def as_dict(self):
        return {k:v.tolist() if isinstance(v,np.ndarray) else float(v) for k,v in vars(self).items()}


class GroupNeighbors:
    def __init__(self,coords,groups):
        self.coords=np.asarray(coords);self.names,self.groups=np.unique(groups,return_inverse=True)
        self.rows=[np.flatnonzero(self.groups==i) for i in range(len(self.names))]
        self.tree=cKDTree(self.coords)

    def nearest(self,query,k=64):
        if len(self.names)<k: raise ValueError('Insufficient distinct reference groups')
        count=min(len(self.coords),2*k)
        while True:
            d,idx=self.tree.query(query,k=count,workers=1)
            chosen=[];distance=[];good=True
            for dd,ii in zip(d,idx):
                _,first=np.unique(self.groups[ii],return_index=True);first=np.sort(first)
                if len(first)<k: good=False;break
                chosen.append(ii[first[:k]]);distance.append(dd[first[:k]])
            if good:break
            if count==len(self.coords):raise ValueError('Could not retrieve distinct groups')
            count=min(2*count,len(self.coords))
        chosen=np.array(chosen);distance=np.array(distance);width=distance[:,-1]
        ratio=np.divide(distance,width[:,None],out=np.zeros_like(distance),where=width[:,None]>1e-12)
        kernel=np.exp(-.5*ratio**2);kernel/=kernel.sum(axis=1,keepdims=True)
        return chosen,kernel,distance[:,-1]

    def random(self,n,rng,k=64):
        gs=np.array([rng.choice(len(self.names),k,replace=False) for _ in range(n)])
        rows=np.array([[rng.choice(self.rows[g]) for g in row] for row in gs])
        return rows


def conditional_moments(x,kernel,global_C):
    mu=np.einsum('nk,nkm->nm',kernel,x)
    z=x-mu[:,None,:]
    C=np.einsum('nk,nki,nkj->nij',kernel,z,z)
    # Global mean is zero in the training coordinates. Fixed borrowing .5.
    return .5*mu,.5*C+.5*global_C+.5*FLOOR*np.eye(x.shape[-1])


def heads(C,sd,var_y):
    C=np.asarray(C)
    if C.ndim==2:C=C[None]
    rho,g2,residual=iu_moments(C,var_y)
    values,U=np.linalg.eigh(C);u=U[:,:,-2:];v=values[:,-2:]
    native=np.einsum('nik,nk->ni',u,np.einsum('nik,ni->nk',u,rho)/(v+1e-12))
    B=sd/np.mean(sd);Q=C*B[None,:,None]*B[None,None,:];r=rho*B
    w=simplex_qp(Q,r);qp=.75/len(sd)+.25*w
    # Explicit heuristic groups, not GroupFS: H0lim,VE0,innovation vs VE075,VE1.
    h=np.array([1/3,1/3,0.,0.,1/3]);ve=np.array([0.,0.,.5,.5,0.]);d=h-ve
    beta=np.clip((r@d-np.einsum('i,nij,j->n',d,Q,ve))/np.einsum('i,nij,j->n',d,Q,d),0,1)
    group=.75/5+.25*(ve+beta[:,None]*d)
    return dict(C=C,rho=rho,g2=g2,additive_residual=residual,native_a=native,
                qp_w=qp,qp_a=qp*B,group_w=group,group_a=group*B,beta=beta,
                condition=values[:,-1]/np.maximum(values[:,0],1e-15))


def gaussian_nll(target,mu,C):
    e=target-mu;sign,ld=np.linalg.slogdet(C)
    if np.any(sign<=0):raise ValueError('Non-positive covariance')
    return .5*(ld+np.einsum('ni,ni->n',e,np.linalg.solve(C,e[...,None])[...,0])+C.shape[-1]*np.log(2*np.pi))


def unit(x):
    x=x.reshape(len(x),-1)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-15)


def bootstrap_metrics(local_x,kernel,fit,rng,repeats=64):
    """Conditional group bootstrap; each neighbor row is a distinct source."""
    n,k,m=local_x.shape
    out={key:[] for key in ('C','rho','native_a','qp_a','group_a')}
    for row in range(n):
        draws=rng.integers(0,k,size=(repeats,k))
        x=local_x[row][draws];w=kernel[row][draws];w/=w.sum(axis=1,keepdims=True)
        _,C=conditional_moments(x,w,fit.C);result=heads(C,fit.sd,fit.var_y)
        for key in out:out[key].append(result[key])
    return {k:np.array(v) for k,v in out.items()}


def signal_noise(point,boot):
    # Ratio describes sampled anchors conditional on neighborhood/global fit.
    p=unit(point);b=unit(boot.reshape((-1,)+boot.shape[2:])).reshape(len(boot),boot.shape[1],-1)
    between=float(np.sum(np.var(p,axis=0,ddof=1)))
    noise=float(np.mean(np.sum(np.var(b,axis=1,ddof=1),axis=1)))
    return dict(between=between,bootstrap_noise=noise,ratio=between/max(noise,1e-20))
