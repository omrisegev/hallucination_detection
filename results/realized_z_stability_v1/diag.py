import numpy as np, json, collections
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from load import pb, prmb
from auc16 import within, D
G={g['uid']:g for g in json.load(open('/home/user/hallucination_detection/results/prm_boost_processbench_v1/local/processbench_gold.json'))}
F3=['chosen_surprisal','realized_z','realized_drv','q15_H1','energy_level','digit_spread']
pbd=D['PB']; pr=D['PRMB']
gen=np.array([G[u]['generator'] for u in pbd['ids']]); print(collections.Counter(gen))
# 1. PRMBench under PB protocol (steps up to first error only)
def prefix_keep(d):
    k=np.zeros(len(d['err']),bool)
    for a,b in zip(d['off'][:-1],d['off'][1:]):
        y=d['err'][a:b]; k[a:b]= np.arange(b-a)<= (np.argmax(y) if y.any() else b-a)
    return k&d['keep']
print('\n1) PRMBench, PB protocol (prefix up to first error)')
kp=prefix_keep(pr)
for f in F3:
    x=pr['V'][:,pr['names'].index(f)]
    print(f'  {f:20s} PRMB all-steps {within(pr["err"],x,pr["off"],pr["keep"])[0]:.3f}  PRMB prefix {within(pr["err"],x,pr["off"],kp)[0]:.3f} (n={within(pr["err"],x,pr["off"],kp)[1]})  PB prefix {within(pbd["err"],pbd["V"][:,pbd["names"].index(f)],pbd["off"],pbd["keep"])[0]:.3f}')
# 2. position trend in clean answers / correct steps
def pos(d):
    p=np.zeros(len(d['err'])); 
    for a,b in zip(d['off'][:-1],d['off'][1:]): p[a:b]=np.arange(b-a)/max(b-a-1,1)
    return p
print('\n2) Spearman(feature, relative position) on correct kept steps')
for f in F3:
    o=[]
    for d in (pbd,pr):
        x=d['V'][:,d['names'].index(f)]; m=d['keep']&~d['err']; o.append(spearmanr(x[m],pos(d)[m])[0])
    print(f'  {f:20s} PB {o[0]:+.3f}  PRMB {o[1]:+.3f}')
# 3. PB within by generator and subset
print('\n3) PB within-AUC by generator')
for g in sorted(set(gen)):
    m=np.repeat(gen==g,np.diff(pbd['off']))
    print(f'  {g:28s}', ' '.join(f'{f[:10]}={within(pbd["err"],pbd["V"][:,pbd["names"].index(f)],pbd["off"],pbd["keep"]&m)[0]:.3f}' for f in F3), 'n=',within(pbd["err"],pbd["V"][:,0],pbd["off"],pbd["keep"]&m)[1])
