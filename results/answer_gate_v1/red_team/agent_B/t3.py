exec(open('load.py').read())
from scipy.stats import spearmanr
Pb=np.flatnonzero(pb); PBc=sorted(set(cells[pb]))
rules=['R0_frozen','R2pb_allocate','OFFSET_D1_upcr_full','OFFSET_D2_lsml_cont_good5','OFFSET_D5_epr','OFFSET_D6_length']
pred={}; nfl={}
for r in rules:
    fl=D[r].astype(bool); nfl[r]=np.bincount(aid,weights=fl,minlength=n); pr=np.full(n,-2)
    for i in Pb:
        fi=np.flatnonzero(fl[off[i]:off[i+1]]); pr[i]=int(fi[0]) if len(fi) else -1
    pred[r]=pr
rows=[]
for c in PBc:
    m=pb&(cells==c); e=m&(target>=0); o=m&(target<0)
    row={'cell':c,'n_correct':int(o.sum()),'n_err':int(e.sum())}
    for r in ['R0_frozen','R2pb_allocate','OFFSET_D1_upcr_full','OFFSET_D2_lsml_cont_good5']:
        ac=(pred[r][o]==-1).mean(); ae=(pred[r][e]==target[e]).mean(); f=0 if ac==0 and ae==0 else 2*ac*ae/(ac+ae)
        sh={'R0_frozen':'R0','R2pb_allocate':'R2pb','OFFSET_D1_upcr_full':'D1','OFFSET_D2_lsml_cont_good5':'D2'}[r]
        row[f'{sh}_corrUnfl']=ac; row[f'{sh}_errAcc']=ae; row[f'{sh}_F1']=f; row[f'{sh}_errUnfl']=(pred[r][e]==-1).mean()
    rows.append(row)
T=pd.DataFrame(rows).set_index('cell'); pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(T[[c for c in T.columns if c.endswith(('corrUnfl','errAcc','F1'))]+['n_correct','n_err']].round(3).to_string())
print('macro F1:',T[[c for c in T.columns if c.endswith('_F1')]].mean().round(4).to_dict())
print('errUnfl (erroneous answers left fully unflagged):'); print(T[[c for c in T.columns if c.endswith('errUnfl')]].round(3).to_string())
# overlap of unflagged correct answers
oc=pb&(target<0); unf={r:set(np.flatnonzero(oc&(nfl[r]==0))) for r in rules}
def jac(a,b): return len(a&b)/max(len(a|b),1)
print('unflagged correct answers (all PB, n_correct=%d):'%oc.sum(),{r:len(unf[r]) for r in rules})
for a,b in [('OFFSET_D1_upcr_full','R2pb_allocate'),('OFFSET_D2_lsml_cont_good5','R2pb_allocate'),('OFFSET_D1_upcr_full','OFFSET_D2_lsml_cont_good5'),('OFFSET_D5_epr','OFFSET_D1_upcr_full'),('OFFSET_D6_length','OFFSET_D1_upcr_full'),('OFFSET_D6_length','R2pb_allocate')]:
    print(f'  {a} vs {b}: jaccard={jac(unf[a],unf[b]):.3f} intersection={len(unf[a]&unf[b])} only_a={len(unf[a]-unf[b])} only_b={len(unf[b]-unf[a])}')
# per cell jaccard D1 vs R2pb, D2 vs R2pb
for c in PBc:
    cm=set(np.flatnonzero(cells==c))
    print(c, 'J(D1,R2pb)=%.3f J(D2,R2pb)=%.3f J(D1,D2)=%.3f'%(jac(unf['OFFSET_D1_upcr_full']&cm,unf['R2pb_allocate']&cm),jac(unf['OFFSET_D2_lsml_cont_good5']&cm,unf['R2pb_allocate']&cm),jac(unf['OFFSET_D1_upcr_full']&cm,unf['OFFSET_D2_lsml_cont_good5']&cm)))
# who is unflagged: n_steps of unflagged vs flagged correct answers
for r in rules:
    u=oc&(nfl[r]==0); f=oc&(nfl[r]>0)
    print(r,'median steps unflagged correct=%.1f flagged correct=%.1f; AUROC(n_steps small->unflagged)='%(np.median(ns[u]),np.median(ns[f])), round(1-spearmanr(ns[oc],(nfl[r][oc]==0))[0],3) if u.sum() else None)
# PB answer-level: do unflagged sets track the answer-level zA? (fraction of unflagged answers with A below cell median)
for r,dn in [('OFFSET_D1_upcr_full','A_D1_upcr_full'),('OFFSET_D2_lsml_cont_good5','A_D2_lsml_cont_good5')]:
    u=pb&(nfl[r]==0); print(r,'unflagged PB answers (any label):',u.sum(),'share with A<0:',round((D[dn][u]<0).mean(),3))
# also R2pb unflagged erroneous overlap
oe=pb&(target>=0); ue={r:set(np.flatnonzero(oe&(nfl[r]==0))) for r in rules}
print('unflagged ERRONEOUS PB answers:',{r:len(ue[r]) for r in rules}, 'n_err=',oe.sum())
print('J(D1,R2pb) erroneous-unflagged=%.3f'%jac(ue['OFFSET_D1_upcr_full'],ue['R2pb_allocate']))
