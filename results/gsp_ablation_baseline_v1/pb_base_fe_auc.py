import sys; sys.path.insert(0,'.')
import numpy as np, ablation as AB
T, N = AB.T, AB.N
for prm in AB.PRMS:
    c, X, ab = AB.load_cell('processbench', prm); Xt = X[:, 2:]
    tel = AB.Rn.c4u_t(c, Xt); s, _, _ = N.lsml(Xt, Xt.mean(1))
    lab = np.array([T.A.PB_GOLD[u]['label'] for u in c.ids])
    def fe(v):
        r=[]
        for j,(a,b) in enumerate(zip(c.off[:-1], c.off[1:])):
            k = lab[j]
            if k <= 0 or k >= b-a: continue
            r.append(np.mean((v[a+k] > v[a:a+k]) + .5*(v[a+k] == v[a:a+k])))
        return np.mean(r)
    w = N.lw(tel['psi'], tel['eta'])
    print(prm, 'FE-AUC GSP %.3f LSML %.3f mean %.3f' % (fe(tel['rank']), fe(s), fe(Xt.mean(1))), 'eta', np.round(tel['eta'],3), 'w', np.round(w,2), flush=True)
    del c
