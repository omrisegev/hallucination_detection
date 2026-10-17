exec(open('t2c.py').read().split("out={}")[0])
from sklearn.metrics import roc_auc_score
Pb=np.flatnonzero(pb)
oc=pb&(target<0); oe=pb&(target>=0)
def gate_stats(fl):
    nf=np.bincount(aid,weights=fl,minlength=n)
    return float((nf[oc]==0).mean()), float((nf[oe]==0).mean())
rules=['R0_frozen','R2pb_allocate','OFFSET_D1_upcr_full','OFFSET_D2_lsml_cont_good5','OFFSET_D5_epr','OFFSET_D6_length','OFFSET_D6b_length_anchored']
for r in rules:
    fl=D[r].astype(bool); a,b=gate_stats(fl); print(f'{r:32s} PB correct unflagged={a:.3f} erroneous unflagged={b:.3f}  PB F1={pbf1(fl):.4f}')
# random-offset null: zA ~ N(0,1) iid per answer per fold model, same calibration
rng=np.random.default_rng(7); res=[]
for s in range(20):
    f=np.zeros(S_,bool)
    for k in range(5):
        c=(k+1)%5; zA=rng.standard_normal(n); Dv=zA[aid]+zS
        for bm in (prm_step,pb_step):
            calm=bm&(step_fold==c); evm=bm&(step_fold==k); tau=np.quantile(Dv[calm],.8); f[evm]=Dv[evm]>=tau
    a,b=gate_stats(f); res.append((prmscore(f),pbf1(f),a,b))
res=np.array(res); print('RANDOM zA~N(0,1) OFFSET null, 20 draws: PRMScore mean %.4f sd %.4f | PB F1 mean %.4f sd %.4f | correct unflagged %.3f erroneous unflagged %.3f'%(res[:,0].mean(),res[:,0].std(),res[:,1].mean(),res[:,1].std(),res[:,2].mean(),res[:,3].mean()))
# unflagged-correct vs steps: AUROC that fewer steps -> unflagged, among correct PB answers
for r in rules:
    nf=np.bincount(aid,weights=D[r].astype(bool),minlength=n); u=(nf[oc]==0).astype(int)
    if 0<u.sum()<len(u): print(r,'AUROC(-n_steps predicts unflagged | correct PB)=%.3f'%roc_auc_score(u,-ns[oc]))
