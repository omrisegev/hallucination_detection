exec(open('load.py').read())
key=np.load('key.npy',allow_pickle=True)
tps=X[:,names.index('trace_length')]/ns
print('tps quantiles ms',np.quantile(tps[ms],[0,.1,.25,.5,.75,.9,1]).round(1))
print('tps quantiles err',np.quantile(tps[err_nc],[0,.1,.25,.5,.75,.9,.95,.99,1]).round(1))
print('share of erroneous with tps >= ms 10th pct',(tps[err_nc]>=np.quantile(tps[ms],.1)).mean().round(4),' >= ms median',(tps[err_nc]>=np.median(tps[ms])).mean().round(4), 'n=',(tps[err_nc]>=np.median(tps[ms])).sum())
def cmp(pos,neg): return ((pos[:,None]>neg[None,:])+0.5*(pos[:,None]==neg[None,:]))
bins=np.quantile(tps[ms],[0,.2,.4,.6,.8,1])
eb=np.clip(np.searchsorted(bins,tps[err_nc],'right')-1,0,4); inr=(tps[err_nc]>=bins[0])&(tps[err_nc]<=bins[-1])
mb=np.clip(np.searchsorted(bins,tps[ms],'right')-1,0,4)
print('erroneous inside ms tps range per bin', [int(((eb==b)&inr).sum()) for b in range(5)])
for d in ['D1_upcr_full','D2_lsml_cont_good5','D5_epr']:
    v=D['A_'+d].astype(float); M=cmp(v[err_nc],v[ms])
    # within-bin AUROC averaged over bins (erroneous restricted to ms tps range)
    per=[]
    for b in range(5):
        e=(eb==b)&inr; m=mb==b
        if e.sum() and m.sum(): per.append((b,int(e.sum()),int(m.sum()),round(float(cmp(v[err_nc][e],v[ms][m]).mean()),3)))
    print(d,'tps-bin AUROC err vs ms:',per)
    # controls too
    cb=np.clip(np.searchsorted(bins,tps[control],'right')-1,0,4); cin=(tps[control]>=bins[0])&(tps[control]<=bins[-1])
    per=[]
    for b in range(5):
        e=(cb==b)&cin; m=mb==b
        if e.sum() and m.sum(): per.append((b,int(e.sum()),int(m.sum()),round(float(cmp(v[control][e],v[ms][m]).mean()),3)))
    print(d,'tps-bin AUROC ctrl vs ms:',per)
# within-question err vs control for all questions
for d in ['D1_upcr_full','D2_lsml_cont_good5','D5_epr']:
    v=D['A_'+d].astype(float); res={}
    for pfx in ['all','prm_train_p1','prm_test_p1','prm_test_p2']:
        a=[]
        ks=set(key[control]) if pfx=='all' else {k for k in set(key[control]) if k.rsplit('_',1)[0]==pfx}
        idx_by={}
        for i in np.flatnonzero(err_nc|control): idx_by.setdefault(key[i],[]).append(i)
        for k in ks:
            ii=np.array(idx_by.get(k,[])); ci=ii[control[ii]]; ei=ii[err_nc[ii]]
            if len(ci) and len(ei): a.append(float(cmp(v[ei],v[ci]).mean()))
        res[pfx]=(round(np.mean(a),4),len(a))
    print(d,'within-question AUROC err vs ctrl (mean, n_questions):',res)
