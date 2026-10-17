exec(open('load.py').read())
from scipy.stats import spearmanr
style=pd.read_pickle('style.pkl')
P=np.flatnonzero(prm)
ws=[sum(ch.isspace() for x in raw[ids[i]]['steps'] for ch in x)/max(1,sum(len(x) for x in raw[ids[i]]['steps'])) for i in P]
style['ws_share']=ws
print('median whitespace char share:',style.groupby('grp').ws_share.median().round(3).to_dict())
# step-count matched AUROC (reproduce the run's length-matched definition)
def cmp(pos,neg): return ((pos[:,None]>neg[None,:])+0.5*(pos[:,None]==neg[None,:]))
bins=np.quantile(ns[ms],[0,.2,.4,.6,.8,1]); eb=np.clip(np.searchsorted(bins,ns[err_nc],'right')-1,0,4); mb=np.clip(np.searchsorted(bins,ns[ms],'right')-1,0,4)
lm_w=np.array([(mb==b).mean()/max((eb==b).mean(),1e-12) for b in eb])
cb=np.clip(np.searchsorted(bins,ns[control],'right')-1,0,4); lc_w=np.array([(mb==b).mean()/max((cb==b).mean(),1e-12) for b in cb])
for d in ['D1_upcr_full','D2_lsml_cont_good5','D5_epr']:
    v=D['A_'+d].astype(float); M=cmp(v[err_nc],v[ms]); Mc=cmp(v[control],v[ms])
    print(d,'step-matched AUROC err vs ms=%.4f ; SAME matching ctrl vs ms=%.4f'%((lm_w[:,None]*M).sum()/(lm_w.sum()*M.shape[1]),(lc_w[:,None]*Mc).sum()/(lc_w.sum()*Mc.shape[1])))
# answer-level detectors vs error share and length among erroneous PRMBench answers
nerr=np.bincount(aid,weights=labels.astype(float),minlength=n); share=nerr/ns
for d in ['D1_upcr_full','D2_lsml_cont_good5','D5_epr','D6b_length_anchored']:
    v=D['A_'+d].astype(float)
    print(d,'among 6035 erroneous: rho(A, n_steps)=%.3f rho(A, error share)=%.3f rho(A, n_err)=%.3f'%(spearmanr(v[err_nc],ns[err_nc])[0],spearmanr(v[err_nc],share[err_nc])[0],spearmanr(v[err_nc],nerr[err_nc])[0]))
# fold / group integrity for the PB population
print('PB answers',pb.sum(),'cells',len(set(cells[pb])),'PB source groups',len(set(groups[pb])),'PRMB groups',len(set(groups[prm])))
print('folds per cell answers:',pd.crosstab(cells,fold).to_string())
