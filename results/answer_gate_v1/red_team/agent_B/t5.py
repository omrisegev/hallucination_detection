exec(open('load.py').read())
import sys; sys.path.insert(0,W)
from spectral_utils.upcr import upcr_fit
FIT=dict(loss='l2',exclusion=True,difficulty_gate=False,simple_avg_fallback=True,recompute_after_exclusion=True,g2_projection_k=1,scale_ratio=0.25)
out=[]
for cell in sorted(set(cells)):
    cm=cells==cell
    for k in range(5):
        c=(k+1)%5; fa=cm&~np.isin(fold,[k,c])
        Xf=X[fa]; mu=Xf.mean(0); sd=Xf.std(0); keep=sd>1e-5
        Z=(X[fa][:,keep]-mu[keep])/sd[keep]; Ff=Z.T
        probe=upcr_fit(Ff,**FIT); pol=np.sign(probe.rho_hat_full); pol[pol==0]=1
        res=upcr_fit(Ff*pol[:,None],**FIT)
        # compare to FIT_LOG views kept
        lg=[r for r in FL if r['cell']==cell and r['fold']==k and r['detector']=='D1_upcr_full'][0]
        out.append(dict(cell=cell,fold=k,used_simple_average=bool(res.used_simple_average),g2_at_ceiling=bool(res.g2_at_ceiling),kept=int(res.keep.sum()),log_kept=lg['extra']['views_kept_by_upcr']))
T=pd.DataFrame(out); print(T.groupby('cell')[['used_simple_average','g2_at_ceiling']].sum().to_string()); print('kept matches log:',(T.kept==T.log_kept).all(), 'n fits',len(T))
