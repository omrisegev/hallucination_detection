"""Diagnostic (labels used only after the label-free pooled fit): does the pooled leading eigvec point at the
label (error-minus-correct) direction, and how big is the label 'spike' vs the nuisance top eigenvalue? PRMBench."""
import sys; sys.argv=['x']
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('run', '/private/tmp/claude-501/-Users-osegev-Desktop-hallucination-detection/98d49308-5ca3-48f7-a452-63da0bb7d2a1/scratchpad/answer_local/run.py')
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
ce='prmbench_qwen3_8b'; ans=np.flatnonzero(R.cells==ce)
fits={}
for bk in R.BANKS:
    bank=R.pop.cell_bank(bk,ans); V=np.asarray(bank['V'],float); off=bank['off']; lev=R.pop.level[bank['rows']]
    sg=np.array([(np.sign(np.corrcoef(V[:,j],lev)[0,1]) or 1.) if V[:,j].std()>0 else 1. for j in range(V.shape[1])])
    Vo=V*sg; Rp=np.nan_to_num(np.corrcoef(Vo.T)); np.fill_diagonal(Rp,1.); ev,U=np.linalg.eigh(Rp)
    fits[bk]=(Vo,off,Rp,ev,U)
_,flags=R.read_labels()
for bk,(Vo,off,Rp,ev,U) in fits.items():
    ds=[];ps=[]
    for j,i in enumerate(ans):
        y=flags[int(i)]
        if y.all() or not y.any(): continue
        X=Vo[off[j]:off[j+1]]; ds.append(X[y].mean(0)-X[~y].mean(0)); ps.append(y.mean())
    d=np.mean(ds,0); p=np.mean(ps); m=len(d)
    lam_label=np.mean([q*(1-q) for q in ps])*d@d
    cos_top=abs(U[:,-1]@d)/np.linalg.norm(d); cos2=abs(U[:,-2]@d)/np.linalg.norm(d)
    # fraction of label direction energy in top-3 nuisance eigvecs
    e3=((U[:,-3:].T@d)**2).sum()/(d@d)
    # oracle: AUC-free check of how equal weights align with d
    cos_eq=abs(np.ones(m)@d)/np.sqrt(m)/np.linalg.norm(d)
    print(f"{bk} m={m} top_eig={ev[-1]:.2f} 2nd={ev[-2]:.2f} label_spike p(1-p)|d|^2={lam_label:.2f} "
          f"cos(lead_eigvec,d)={cos_top:.3f} cos(2nd,d)={cos2:.3f} top3_energy={e3:.3f} cos(equal,d)={cos_eq:.3f} "
          f"BBP sqrt(m/n) n=8:{np.sqrt(m/8):.2f} n=13:{np.sqrt(m/13):.2f} n=33:{np.sqrt(m/33):.2f}")
for bk,(Vo,off,Rp,ev,U) in fits.items():
    w=np.abs(U[:,-1]); print(bk,'cos(equal, pooled w)=%.3f'%(w.sum()/np.sqrt(len(w))/np.linalg.norm(w)), 'w/mean min..max %.2f..%.2f'%((w/w.mean()).min(),(w/w.mean()).max()))
