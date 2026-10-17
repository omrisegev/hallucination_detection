exec(open('load.py').read())
import sys; sys.path.insert(0,W)
from spectral_utils.prmbench import prmbench_evaluate
SSL=r'C:\Users\omris\TAU\hallucination_detection\.worktrees\ssl-pseudolabel-residual-v1\results\expectation_realization_v1'
S=np.load(SSL+r'\run_20260927_stage_b\STEP_SCORES.npz')['S_equal'].astype(float)
S_=int(off[-1]); step_fold=fold[aid]; prm_step=prm[aid]; pb_step=pb[aid]
zS=np.empty(S_)
for i in range(n):
    s=S[off[i]:off[i+1]]; zS[off[i]:off[i+1]]=(s-s.mean())/max(s.std(),1e-8)
P=np.flatnonzero(prm); Pb=np.flatnonzero(pb)
def prmscore(fl):
    v=~fl; res=prmbench_evaluate([{'idx':ids[i],'labels':v[off[i]:off[i+1]].astype(int).tolist()} for i in P],[meta[ids[i]] for i in P])
    return 0.5*(res['total']['f1']+res['total']['negative_f1'])
PBc=sorted(set(cells[pb]))
def pbf1(fl):
    pr=np.full(n,-2)
    for i in Pb:
        fi=np.flatnonzero(fl[off[i]:off[i+1]]); pr[i]=int(fi[0]) if len(fi) else -1
    f=[]
    for c in PBc:
        m=pb&(cells==c); e=m&(target>=0); o=m&(target<0)
        ae=(pr[e]==target[e]).mean(); ac=(pr[o]==-1).mean(); f.append(0 if ae==0 and ac==0 else 2*ae*ac/(ae+ac))
    return float(np.mean(f))
def offset_from_value(val, wgt=1.0, sign=1.0):
    """label-free OFFSET rule mirroring answer_gate_run: zA = sign*(val - mu_fit)/sd_fit per cell and fold, tau = q80 of calibration-fold steps per benchmark"""
    f=np.zeros(S_,bool)
    for k in range(5):
        c=(k+1)%5; zA=np.full(n,np.nan)
        for cell in set(cells):
            cm=cells==cell; fa=cm&~np.isin(fold,[k,c]); mu=val[fa].mean(); sd=val[fa].std()
            zA[cm]=sign*(val[cm]-mu)/sd
        Dv=wgt*zA[aid]+zS
        for bm in (prm_step,pb_step):
            calm=bm&(step_fold==c); evm=bm&(step_fold==k); tau=np.quantile(Dv[calm],.8); f[evm]=Dv[evm]>=tau
    return f
def topk(counts):
    f=np.zeros(S_,bool)
    for i in range(n):
        k=int(min(counts[i],ns[i]))
        if k>0: a=off[i]; f[a+np.argsort(-S[a:off[i+1]],kind='stable')[:k]]=True
    return f
out={}
for r in ['R0_frozen','R2_allocate','R2pb_allocate','OFFSET_D1_upcr_full','OFFSET_D2_lsml_cont_good5','OFFSET_D5_epr','OFFSET_D6_length','OFFSET_D6b_length_anchored','OFFSET_D4_equal_full','OFFSET_D3_lsml_full']:
    fl=D[r].astype(bool); out[r]=(prmscore(fl),pbf1(fl),float(fl[prm_step].mean()))
tl=X[:,names.index('trace_length')]
new={'OFFSET_minus_tokens (recomputed D6b dir)':offset_from_value(tl,1,-1),'OFFSET_minus_nsteps':offset_from_value(ns.astype(float),1,-1),
     'OFFSET_minus_log_nsteps':offset_from_value(np.log(ns.astype(float)),1,-1),'OFFSET_plus_nsteps':offset_from_value(ns.astype(float),1,1)}
for k2 in (1,2,3): new[f'fixed_top{k2}_per_answer']=topk(np.full(n,k2))
new['fixed_top2_prm_only_rate_ref']=None; del new['fixed_top2_prm_only_rate_ref']
nerr=np.bincount(aid,weights=labels.astype(float),minlength=n)
new['ORACLE_count=true_n_err (labels, ceiling)']=topk(np.where(prm,nerr,0))
for k2,fl in new.items(): out[k2]=(prmscore(fl),pbf1(fl),float(fl[prm_step].mean()))
print('recomputed D6b from scratch equals DECISIONS on PRMBench:', np.array_equal(new['OFFSET_minus_tokens (recomputed D6b dir)'][prm_step],D['OFFSET_D6b_length_anchored'].astype(bool)[prm_step]))
print(pd.DataFrame(out,index=['PRMScore','PB_F1_macro8','PRMB_flag_rate']).T.round(4).to_string())
np.savez_compressed('t2_newflags.npz',**{k.split(' ')[0]:v for k,v in new.items()})
