exec(open('load.py').read())
from scipy.stats import spearmanr
nerr=np.bincount(aid,weights=labels.astype(float),minlength=n)
share=nerr/ns; tl=X[:,names.index('trace_length')]
m=err_nc
print('PRMBench erroneous answers:',m.sum())
print('n_err per answer distribution:',pd.Series(nerr[m].astype(int)).value_counts().sort_index().to_dict())
print('spearman(n_err, n_steps)=%.3f  spearman(share, n_steps)=%.3f  spearman(share, trace_length)=%.3f'%(spearmanr(nerr[m],ns[m])[0],spearmanr(share[m],ns[m])[0],spearmanr(share[m],tl[m])[0]))
q=pd.qcut(ns[m],5,duplicates='drop')
df=pd.DataFrame({'q':q,'ns':ns[m],'nerr':nerr[m],'share':share[m]})
print(df.groupby('q',observed=True).agg(n=('ns','size'),mean_steps=('ns','mean'),mean_nerr=('nerr','mean'),mean_share=('share','mean')).round(3).to_string())
qt=pd.qcut(tl[m],5)
df2=pd.DataFrame({'q':qt,'tl':tl[m],'nerr':nerr[m],'share':share[m],'ns':ns[m]})
print(df2.groupby('q',observed=True).agg(n=('tl','size'),mean_tokens=('tl','mean'),mean_steps=('ns','mean'),mean_nerr=('nerr','mean'),mean_share=('share','mean')).round(3).to_string())
rows=[]
for c in sorted(np.unique(cls[m])):
    mm=m&(cls==c); rows.append(dict(cls=c,n=int(mm.sum()),mean_nerr=nerr[mm].mean(),med_steps=np.median(ns[mm]),mean_share=share[mm].mean(),rho_share_steps=spearmanr(share[mm],ns[mm])[0],rho_nerr_steps=spearmanr(nerr[mm],ns[mm])[0]))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# pooled step-level error rate by answer length (the quantity PRMScore sees): errors / steps within bins, over ALL noncontrol answers (ms included, 0 errors)
nc=noncontrol
qa=pd.qcut(ns[nc],5,duplicates='drop')
d3=pd.DataFrame({'q':qa,'ns':ns[nc],'nerr':nerr[nc]}).groupby('q',observed=True).agg(n=('ns','size'),steps=('ns','sum'),errs=('nerr','sum'))
d3['step_error_rate']=d3.errs/d3.steps; print('noncontrol (6211) step-level error rate by step-count quintile:'); print(d3.round(4).to_string())
