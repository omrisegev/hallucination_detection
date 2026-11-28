import numpy as np
from scipy.stats import spearmanr
from fast import wauc, get
D=get()
F=['chosen_surprisal','realized_z','realized_drv','q15_H1','energy_level','digit_spread']
print('A) step length')
for k,d in D.items():
    y=d['err']; off=d['off']; lt=np.log(d['ntok'])
    fe=[];pre=[]
    for a,b in zip(off[:-1],off[1:]):
        yy=y[a:b]
        if yy.any() and np.argmax(yy)>0: j=a+np.argmax(yy); fe.append(lt[j]); pre.append(lt[a:j].mean())
    print(f' {k}: median ntok {np.median(d["ntok"]):.0f}; first-error log-ntok minus prefix mean {np.mean(np.array(fe)-np.array(pre)):+.3f}; within-AUC of log ntok {wauc(y,lt,off,d["keep"]):.3f}; within-AUC of position {wauc(y,d["pos"],off,d["keep"]):.3f}')
    m=d['keep']&~y
    print('   spearman(feature, log ntok) correct steps:', ', '.join(f'{f}={spearmanr(d["col"](f)[m],lt[m])[0]:+.2f}' for f in F))
print('\nB) chosen_surprisal readouts within-AUC (PB | PRMB)')
for r in D['PB']['rn']:
    v=[wauc(d['err'],d['R'][:,d['cn'].index('chosen_surprisal'),d['rn'].index(r)],d['off'],d['keep']) for d in D.values()]
    print(f'  {r:12s} {v[0]:.3f} | {v[1]:.3f}')
print('\nC) label-free residualization (OLS on all steps of the benchmark), within-AUC PB | PRMB')
def resid(x,Z):
    Z=np.column_stack([np.ones(len(x))]+Z); b=np.linalg.lstsq(Z,x,rcond=None)[0]; return x-Z@b
for f in F:
    out=[]
    for k,d in D.items():
        x=d['col'](f); lt=np.log(d['ntok'])
        out.append((wauc(d['err'],x,d['off'],d['keep']), wauc(d['err'],resid(x,[lt]),d['off'],d['keep']), wauc(d['err'],resid(x,[d['pos']]),d['off'],d['keep']), wauc(d['err'],resid(x,[lt,lt**2,d['pos'],(d['idx']==0).astype(float)]),d['off'],d['keep'])))
    print(f'  {f:18s} raw {out[0][0]:.3f}|{out[1][0]:.3f}  -len {out[0][1]:.3f}|{out[1][1]:.3f}  -pos {out[0][2]:.3f}|{out[1][2]:.3f}  -len-pos-first {out[0][3]:.3f}|{out[1][3]:.3f}')
