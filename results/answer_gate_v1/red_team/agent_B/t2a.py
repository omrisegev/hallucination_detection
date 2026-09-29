exec(open('load.py').read())
from scipy.stats import spearmanr
fl=pd.DataFrame([r for r in FL if r['detector'] in ('D6b_length_anchored','D2_lsml_cont_good5','D3_lsml_full','D1_upcr_full')])
piv=fl[fl.detector=='D6b_length_anchored'].pivot(index='cell',columns='fold',values='flipped').astype(int)
print('D6b flipped (1 = -length, shorter = more suspicious):'); print(piv.to_string()); print('total flipped',piv.values.sum(),'/',piv.size)
print('D3 flips:',fl[(fl.detector=='D3_lsml_full')&fl.flipped][['cell','fold']].values.tolist())
print('D2 K/degenerate:',pd.Series([str(r['extra']) for r in FL if r['detector']=='D2_lsml_cont_good5']).value_counts().to_dict())
print('D1 extras:',pd.Series([str(r['extra']) for r in FL if r['detector']=='D1_upcr_full']).value_counts().head(20).to_dict())
print('views_kept:',pd.Series([r['views_kept'] for r in FL]).value_counts().to_dict(), ' imputed:',pd.Series([r['imputed_fit_values'] for r in FL]).value_counts().to_dict())
# correlation of epr and length per cell (all answers, label-free) -> explains flips
for c in sorted(set(cells)):
    m=cells==c; r=spearmanr(X[m,names.index('epr')],X[m,names.index('trace_length')])[0]
    print(c, 'spearman(epr, trace_length)=%.3f'%r, 'n=',m.sum())
# D6b actual direction in PRMBench: check A_D6b = -A_D6?
m=prm; print('PRMBench A_D6b == -A_D6 :',np.allclose(D['A_D6b_length_anchored'][m],-D['A_D6_length'][m]))
