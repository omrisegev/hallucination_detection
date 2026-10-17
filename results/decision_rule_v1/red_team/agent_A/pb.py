from load import *
a,off,lab,S,tau,raw,names,surv,meta,mp=load()
fl=np.load('flags.npz'); n=len(a); L=np.diff(off)
cells=sorted(c for c in a.cell.unique() if c.startswith('pb_'))
tgt=a.target.values
for k,nm in (('F0','R0'),('F1','R1'),('F2','R2')):
    F=fl[k]; pred=np.array([ (np.flatnonzero(F[off[i]:off[i+1]])[0] if F[off[i]:off[i+1]].any() else -1) for i in range(n)])
    f1s=[];rows=[]
    for c in cells:
        m=(a.cell.values==c); e=m&(tgt>=0); ok=m&(tgt==-1)
        ae=(pred[e]==tgt[e]).mean(); ac=(pred[ok]==-1).mean()
        f=0 if ae+ac==0 else 2*ae*ac/(ae+ac); f1s.append(f); rows.append('%s %.3f/%.3f/%.4f'%(c,ae,ac,f))
    print(nm,'PB macro F1 %.4f'%np.mean(f1s)); print('   ',' | '.join(rows))
    assert (tgt[a.cell.str.startswith('pb_').values]<L[a.cell.str.startswith('pb_').values]).all()
