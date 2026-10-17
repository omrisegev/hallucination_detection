"""Paired source-group bootstrap (2000 draws) for the out-of-fold small banks vs PRM and vs EQ2(PRM, N16 backbone)."""
import pickle
import numpy as np
import pandas as pd
import fuse as FU

D = FU.OUT


def main():
    z = np.load(D / 'SMALL_BANK_OOF.npz', allow_pickle=True); o = z['offsets']; ids = z['idx'].astype(str)
    cc = np.load(D / 'CHANNEL_CONDCORR_prm_backbone.npz', allow_pickle=True)
    nb = [str(x) for x in cc['names']]; prm = cc['S'][:, nb.index('prm_risk')].astype(float); y = cc['y'].astype(bool)
    az = lambda v: FU.ext.answer_standardize(np.asarray(v, float)[:, None], o)[:, 0]
    bbz = az(np.load(D / 'SCORES_prm_backbone.npz')['N16__MEAN_MOM_ga'])
    prmd = {r['idx']: r for r in pickle.load(open(FU.PRM_PKL, 'rb')).values()}
    nonc = np.array([prmd[i]['classification'] != 'correct' for i in ids])
    err = np.zeros(o[-1], bool)
    for j, i in enumerate(ids):
        for e in prmd[i]['error_steps']:
            if 1 <= e <= o[j + 1] - o[j]: err[o[j] + e - 1] = True
    native = np.concatenate([np.asarray(prmd[i]['labels']) == 0 for i in ids])
    A = pd.read_csv(FU.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig').set_index('id')
    ug, inv = np.unique(A.loc[ids, 'source_group'].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(20261013)
    W = np.vstack([np.ones(len(ug)), rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000)])
    arms = {'PRM native': (prm, native), 'EQ2(PRM,BB)': (.5 * prm + .5 * bbz, None),
            'small k=3 OOF': (z['oof_k3'], None), 'small k=6 OOF': (z['oof_k6'], None)}
    B = {}
    for k, (s, f) in arms.items():
        if f is None: zs = az(s); f = zs > np.quantile(zs, .8)
        au = np.array([FU.auc(y[o[j]:o[j + 1]], s[o[j]:o[j + 1]]) for j in range(len(ids))])
        c4 = np.zeros((len(ids), 4))
        for j in np.flatnonzero(nonc):
            yy, ff = err[o[j]:o[j + 1]], f[o[j]:o[j + 1]]
            c4[j] = [(~yy & ~ff).sum(), (yy & ~ff).sum(), (yy & ff).sum(), (~yy & ff).sum()]
        G = np.zeros((len(ug), 6)); np.add.at(G, inv, np.column_stack([np.where(np.isfinite(au), au, 0), np.isfinite(au), c4]))
        S = W @ G; tp, fp, tn, fn = S[:, 2:].T
        B[k] = (S[:, 0] / S[:, 1], 100 * (tp / (tp + .5 * (fp + fn)) + tn / (tn + .5 * (fp + fn))) / 2)
    rows = []
    for a in ('small k=3 OOF', 'small k=6 OOF', 'EQ2(PRM,BB)'):
        for b in ('PRM native', 'EQ2(PRM,BB)'):
            if a == b: continue
            r = {'contrast': f'{a} - {b}'}
            for m, nm in ((0, 'auc'), (1, 'prmscore')):
                d = B[a][m] - B[b][m]; r[nm] = f'{d[0]:+.4f} [{np.quantile(d[1:], .025):+.4f}, {np.quantile(d[1:], .975):+.4f}]'
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(D / 'SMALL_BANK_BOOTSTRAP.csv', index=False); print(df.to_string())


if __name__ == '__main__':
    main()
