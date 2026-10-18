import warnings, sys, json
from data import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
F,A,LAB=load()
L=pd.read_csv(U+'results/monday_lr_weights_v1/L1_PATH.csv'); L=L[(L.bank=='N56')&(L.C<=0.0022)]
L1=[c.strip() for s in L.channels_entering_at_this_C_median.dropna() for c in s.split(';')]
R56=pickle.load(open(U+'results/monday_lr_reference_v1/cache/N56.pkl','rb'))
def setup(bk):
    if bk=='L1_21':
        R=R56; names=list(L1); g56=dict(zip(R['fit_diag']['survivors'],R['fit_diag']['groups']))
        nxt=max(g56.values())+1; gr=[]
        for c in names:
            if c in g56: gr.append(g56[c])
            else: gr.append(nxt); nxt+=1
        surv=names
    else:
        R=pickle.load(open(U+f'results/monday_lr_reference_v1/cache/{bk}.pkl','rb')); names=R['names']
        surv=R['fit_diag']['survivors']; gr=R['fit_diag']['groups']
    V,o,y,folds,ans,sg=bank(F,A,LAB,names)
    X=V[:,[names.index(c) for c in surv]]
    u,gr=np.unique(gr,return_inverse=True)
    nonctl=R56['methods']['ALL_EQ']['c1'].sum(1)>0
    return dict(R=R,V=V,X=X,o=o,y=y,folds=folds,sg=sg,groups=gr,surv=list(surv),nonctl=nonctl,names=names)
def az(s,o): return astd(s[:,None] if s.ndim==1 else s,o)
def reps(S):
    X,o,g=S['X'],S['o'],S['groups']; G=g.max()+1
    out={k:[] for k in ['MEAN','PC1','PC2','PC3','DIFF']}; meta={k:[] for k in out}
    for h in range(G):
        mem=np.flatnonzero(g==h); Xg=X[:,mem]
        out['MEAN'].append(Xg.mean(1)); meta['MEAN'].append((h,len(mem)))
        C=Xg.T@Xg/len(Xg); w,v=np.linalg.eigh(C); v=v[:,::-1]; w=w[::-1]
        for k in range(min(3,len(mem))):
            vec=v[:,k]
            if k==0 and vec.sum()<0: vec=-vec
            out[f'PC{k+1}'].append(Xg@vec); meta[f'PC{k+1}'].append((h,float(w[k]/w.sum())))
        if len(mem)>=2:
            if len(mem)==2: out['DIFF'].append(Xg[:,0]-Xg[:,1]); meta['DIFF'].append((h,0))
            else:
                for q in range(len(mem)): out['DIFF'].append(Xg[:,q]-Xg.mean(1)); meta['DIFF'].append((h,q))
    return {k:(az(np.column_stack(v),o) if v else np.zeros((len(X),0))) for k,v in out.items()},meta
def orient_anchor(M,anchor):
    s=np.array([np.sign(np.corrcoef(M[:,j],anchor)[0,1]) or 1. if M[:,j].std()>0 else 1. for j in range(M.shape[1])]); return M*s,s
def orient_oracle(M,y):
    s=np.array([np.sign(np.corrcoef(M[:,j],y)[0,1]) or 1. if M[:,j].std()>0 else 1. for j in range(M.shape[1])]); return M*s,s
def sml(M,anchor):
    C=np.corrcoef(M,rowvar=False); w,v=np.linalg.eigh(C); v1=v[:,-1]; s=M@v1
    if np.corrcoef(s,anchor)[0,1]<0: v1=-v1; s=-s
    return s, v1
def lr_oof(M,y,folds,C=1.):
    oof=np.zeros(len(M))
    for k in range(5):
        tr=folds!=k; te=~tr; sc=StandardScaler().fit(M[tr])
        m=LogisticRegression(C=C,class_weight='balanced',max_iter=3000).fit(sc.transform(M[tr]),y[tr])
        oof[te]=m.decision_function(sc.transform(M[te]))
    return oof
results={}; scores={}
for bk in ['N43','N56','L1_21']:
    S=setup(bk)
    o,y,folds=S['o'],S['y'],S['folds']; R,meta=reps(S)
    if bk=='N56': A56=R['MEAN'].mean(1)
    anchor=R['MEAN'].mean(1)   # label-free consensus = MEAN_EQ
    if bk=='L1_21': anchor=A56   # the subset's own mean is not oriented; use the full-N56 label-free consensus
    sets={'MEAN':['MEAN'],'PC1':['PC1'],'PC1+PC2':['PC1','PC2'],'PC1+PC2+PC3':['PC1','PC2','PC3'],'MEAN+DIFF':['MEAN','DIFF'],'DIFF':['DIFF'],'PC2':['PC2'],'PC3':['PC3']}
    sc={}
    sc['ALL_EQ(surv)']=S['X'].mean(1)
    sc['LR_allX(surv)']=lr_oof(S['X'],y,folds)
    sc['LR_allX(bank)']=lr_oof(S['V'],y,folds)
    info={}
    for name,parts in sets.items():
        M=np.column_stack([R[p] for p in parts])
        first=[p in ('MEAN','PC1') for p in parts for _ in range(R[p].shape[1])]
        Ma,sa=orient_anchor(M,anchor)
        if bk!='L1_21':
            for j,f in enumerate(first):   # level/PC1 columns keep the bank orientation
                if f: Ma[:,j]=M[:,j]; sa[j]=1
        Mo,so=orient_oracle(M,y)
        sc[f'{name}|EQ_anchor']=Ma.mean(1)
        s_,w_=sml(Ma,anchor); sc[f'{name}|SML_anchor']=s_
        sc[f'{name}|EQ_oraclesign']=Mo.mean(1)
        sc[f'{name}|LR_oracle']=lr_oof(M,y,folds)
        info[name]={'n_cols':int(M.shape[1]),'anchor_vs_oracle_sign_agree':float((sa==so).mean()),'sml_neg_weights':int((w_<0).sum())}
    info['groups']=int(S['groups'].max()+1); info['n_surv']=len(S['surv']); info['group_sizes']=np.bincount(S['groups']).tolist()
    info['pc_var_share']={k:meta[k] for k in ['PC1','PC2','PC3']}
    res={}
    for k,s in sc.items():
        a,c1=per_answer(s,o,y,S['nonctl']); res[k]={'within_auc':float(np.nanmean(a)),'prmscore_top1':float(prmscore(c1))}
        scores[(bk,k)]=a
    if bk!='L1_21':
        for k in ['ALL_EQ','MEAN_EQ','MEAN_MOM_ga','MEAN_MOM_cc','HEM_MOM_cc','TCI_ONLY_ga']:
            st=S['R']['methods'][k]; res['cache:'+k]={'within_auc':float(np.nanmean(st['auc'])),'prmscore_top1':float(prmscore(st['c1']))}; scores[(bk,'cache:'+k)]=st['auc']
        res['cache:LR_nested']={'within_auc':float(np.nanmean(S['R']['lr']['auc'])),'prmscore_top1':float(prmscore(S['R']['lr']['c1']))}; scores[(bk,'cache:LR_nested')]=S['R']['lr']['auc']
    results[bk]={'metrics':res,'info':info}
    print(bk, json.dumps(info,default=str)[:600])
    for k,v in res.items(): print(f'  {k:28s} {v["within_auc"]:.4f} {v["prmscore_top1"]:.4f}')
    sys.stdout.flush()
json.dump(results,open('RESULTS_v2.json','w'),indent=1,default=str)
pickle.dump({'scores':scores,'sg':S['sg'],'L1':L1},open('PER_ANSWER_v2.pkl','wb'))
