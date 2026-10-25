# Toy: identification of the error factor by SML without an anchor (standalone telemetry). Gaussian, numpy only.
# Case D Fisher column ignores the eta-y correlation; read only SML and mean there.
import numpy as np

rng=np.random.default_rng(0)
N=200000; p=0.3; m=8
def auc(y,s):
    r=np.empty(len(s)); r[np.argsort(s)]=np.arange(1,len(s)+1); n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def sml(X):
    C=np.corrcoef(X.T); np.fill_diagonal(C,0)
    w,V=np.linalg.eigh(C); v=V[:,-1]
    if v.sum()<0: v=-v
    return v
def run(d, lam_scale, lam_dir, rho_eta_y=0.0, blocks=None):
    y=(rng.random(N)<p).astype(int)
    eta=rng.standard_normal(N)
    if rho_eta_y: eta = rho_eta_y*(y-p)/np.sqrt(p*(1-p)) + np.sqrt(1-rho_eta_y**2)*eta
    lam=lam_scale*lam_dir
    X=np.outer(y,d)+np.outer(eta,lam)+rng.standard_normal((N,m))
    X=(X-X.mean(0))/X.std(0)
    v=sml(X)
    Sig=np.eye(m)+np.outer(lam,lam)
    wf=np.linalg.solve(Sig,d)
    u_dir=d/np.linalg.norm(d); l_dir=lam/np.linalg.norm(lam) if lam_scale>0 else u_dir
    return dict(cos_u=abs(v@ (X.std(0)**0*u_dir))/np.linalg.norm(v), cos_l=abs(v@l_dir)/np.linalg.norm(v),
                auc_sml=auc(y,X@v), auc_mean=auc(y,X.mean(1)), auc_fisher=auc(y,X@wf), auc_best1=max(auc(y,X[:,k]) for k in range(m)))
d=np.full(m,0.5)
print("A: homogeneous, no nuisance"); print(run(d,0,np.ones(m)))
# nuisance orthogonal-ish to u: half channels +, half -
ldir=np.r_[np.ones(4),-np.ones(4)]/np.sqrt(8)
print("B: sweep nuisance strength, direction orthogonal to signal")
for s in [0,0.5,1.0,1.5,2.0,3.0]:
    r=run(d,s,ldir); print(s, {k:round(v,3) for k,v in r.items()})
# nuisance mixed direction (not orthogonal)
ldir2=np.r_[np.ones(6),-0.5*np.ones(2)]; ldir2/=np.linalg.norm(ldir2)
print("C: nuisance partly aligned with signal")
for s in [0,1,2,3]:
    r=run(d,s,ldir2); print(s, {k:round(v,3) for k,v in r.items()})
print("D: nuisance correlated with y (position-like), rho=0.4, orth direction")
for s in [1,2,3]:
    r=run(d*0.4,s,ldir,rho_eta_y=0.4); print(s, {k:round(v,3) for k,v in r.items()})
