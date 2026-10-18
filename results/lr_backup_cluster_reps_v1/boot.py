import pickle,numpy as np,json,sys
P=pickle.load(open(sys.argv[1],'rb')); sc=P['scores']; sg=P['sg']
g,inv=np.unique(sg,return_inverse=True); rng=np.random.default_rng(20261018); D=2000
W=rng.multinomial(len(g),np.full(len(g),1/len(g)),size=D).astype(float)
def gs(x):
    o=np.zeros(len(g)); np.add.at(o,inv,x); return o
def mean_draws(a):
    f=np.isfinite(a); return (W@gs(np.nan_to_num(a)))/(W@gs(f.astype(float))), np.nanmean(a)
out={}
for bk in ['N43','N56','L1_21']:
    keys=[k for (b,k) in sc if b==bk]
    if bk!='L1_21':
        base=max([k for k in keys if k.startswith('cache:') and 'LR' not in k],key=lambda k:np.nanmean(sc[(bk,k)])); lr='cache:LR_nested'
    else:
        base='ALL_EQ(surv)'; lr='LR_allX(surv)'
    bd,b0=mean_draws(sc[(bk,base)]); ld,l0=mean_draws(sc[(bk,lr)])
    res={'baseline':base,'baseline_auc':b0,'lr':lr,'lr_auc':l0}
    for k in keys:
        d,v=mean_draws(sc[(bk,k)])
        res[k]={'auc':v,'d_base':v-b0,'d_base_ci':[float(np.quantile(d-bd,.025)),float(np.quantile(d-bd,.975))],
                'd_lr':v-l0,'d_lr_ci':[float(np.quantile(d-ld,.025)),float(np.quantile(d-ld,.975))],
                'gap_closed':(v-b0)/(l0-b0)}
    out[bk]=res
    print(bk,'base',base,round(b0,4),'LR',round(l0,4))
    for k in sorted(keys,key=lambda k:-res[k]['auc']):
        r=res[k]; print(f"  {k:28s} {r['auc']:.4f} dBase {r['d_base']:+.4f} [{r['d_base_ci'][0]:+.4f},{r['d_base_ci'][1]:+.4f}] dLR {r['d_lr']:+.4f} closed {r['gap_closed']:.2f}")
json.dump(out,open(sys.argv[2],'w'),indent=1)
