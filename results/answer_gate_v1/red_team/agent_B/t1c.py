exec(open('load.py').read())
P=np.flatnonzero(prm)
key=np.full(n,'',object)
for i in P:
    r=raw[ids[i]]; c=r['cls']; key[i]=srckey(r['rawidx'], 'redundency' if c=='correct' else c)
# sanity: key consistent with source_group
kk=pd.DataFrame({'key':key[P],'sg':groups[P],'fold':fold[P]})
print('keys',kk.key.nunique(),'source_groups',kk.sg.nunique(), 'keys spanning >1 group', (kk.groupby('key').sg.nunique()>1).sum(), 'keys spanning >1 fold',(kk.groupby('key').fold.nunique()>1).sum())
mskeys=set(key[ms]); print('ms answers',ms.sum(),'distinct ms questions',len(mskeys))
pref=pd.Series([k.rsplit('_',1)[0] for k in key[P]]).value_counts(); print('key prefixes (all PRMB):',pref.to_dict())
print('ms key prefixes',pd.Series([k.rsplit('_',1)[0] for k in key[ms]]).value_counts().to_dict())
sib_err=err_nc&np.isin(key,list(mskeys)); sib_ctrl=control&np.isin(key,list(mskeys))
print('erroneous answers on ms questions',sib_err.sum(),'of',err_nc.sum(),'; controls on ms questions',sib_ctrl.sum())
print('erroneous answers on NON-ms questions', (err_nc&~np.isin(key,list(mskeys))).sum())
res={}
feats={'raw_epr':X[:,names.index('epr')],'raw_epr_spilled':X[:,names.index('epr_spilled')],'tokens_per_step':X[:,names.index('trace_length')]/ns}
for d in ['D1_upcr_full','D2_lsml_cont_good5','D5_epr']: feats['A_'+d]=D['A_'+d].astype(float)
out=[]
for f,v in feats.items():
    w_err=[];w_ctrl=[];c_err=[];ms_lower_than_all=[]; nq_err=0; nq_ctrl=0
    for k in mskeys:
        mi=np.flatnonzero(ms&(key==k)); ei=np.flatnonzero(err_nc&(key==k)); ci=np.flatnonzero(control&(key==k))
        for m in mi:
            if len(ei): w_err.append(np.mean((v[ei]>v[m])+0.5*(v[ei]==v[m]))); ms_lower_than_all.append(bool((v[ei]>v[m]).all()))
            if len(ci): w_ctrl.append(np.mean((v[ci]>v[m])+0.5*(v[ci]==v[m])))
        if len(ci) and len(ei): c_err.append(np.mean([(v[e]>v[c])+0.5*(v[e]==v[c]) for e in ei for c in ci]))
    out.append(dict(feature=f, within_q_AUC_err_vs_ms=np.mean(w_err), n_ms_with_err_sibs=len(w_err), share_ms_below_all_err_sibs=np.mean(ms_lower_than_all),
                    within_q_AUC_ctrl_vs_ms=np.mean(w_ctrl), n_ms_with_ctrl=len(w_ctrl), within_q_AUC_err_vs_ctrl=np.mean(c_err), n_q_ctrl_and_err=len(c_err)))
pd.set_option('display.width',250); print(pd.DataFrame(out).set_index('feature').round(4).to_string())
np.save('key.npy',key)
