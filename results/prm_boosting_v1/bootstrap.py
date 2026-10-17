"""Paired source-group bootstrap (2000 draws) of within-AUC and pooled PRMScore differences.
PRMScore here = (F1_valid + F1_error)/2 from pooled confusion counts over non-control rows, which is
the official total (prmbench_evaluate pools the same counts); its point value is checked against it.

    python bootstrap.py [tag]
"""
import pickle, sys
import numpy as np
import pandas as pd
import fuse as FU

TAG = sys.argv[1] if len(sys.argv) > 1 else 'qwen3_8b_mask1'
DRAWS, SEED = 2000, 20261013


def main():
    z = np.load(FU.OUT / f'SCORES_{TAG}.npz', allow_pickle=True); o = z['offsets']; ids = z['idx'].astype(str)
    az = lambda v: FU.ext.answer_standardize(np.asarray(v, float)[:, None], o)[:, 0]
    A = pd.read_csv(FU.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig').set_index('id')
    with np.load(FU.SE / 'OOF_STEP_SCORES.npz') as q:
        OFF = q['offsets'].astype(np.int64); LAB = q['labels'].astype(bool)
    pos = {i: k for k, i in enumerate(A.index.astype(str))}
    lab = np.concatenate([LAB[OFF[pos[i]]:OFF[pos[i] + 1]] for i in ids])
    prm = {r['idx']: r for r in pickle.load(open(FU.PRM_PKL, 'rb')).values()}
    nonc = np.array([prm[i]['classification'] != 'correct' for i in ids])
    # official labels: error_steps (1-based, out-of-range inert) define the evaluated error set
    err = np.zeros(o[-1], bool)
    for j, i in enumerate(ids):
        for e in prm[i]['error_steps']:
            if 1 <= e <= o[j + 1] - o[j]: err[o[j] + e - 1] = True
    groups = A.loc[ids, 'source_group'].to_numpy(); ug, inv = np.unique(groups, return_inverse=True)
    q80 = lambda s: (lambda zs: zs > np.quantile(zs, .8))(az(s))
    arms = {'PRM_native': (z['prm_risk'], np.concatenate([np.asarray(prm[i]['labels']) == 0 for i in ids])),
            'PRM_Q80': (z['prm_risk'], q80(z['prm_risk'])),
            'N16_MEAN_MOM_ga': (z['N16__MEAN_MOM_ga'], q80(z['N16__MEAN_MOM_ga'])),
            'N16P_MEAN_MOM_ga': (z['N16P__MEAN_MOM_ga'], q80(z['N16P__MEAN_MOM_ga']))}
    tv = .5 * (az(z['prm_risk']) + az(z['N16__MEAN_MOM_ga'])); arms['TWO_VIEW_MEAN_MOM_ga'] = (tv, q80(tv))
    stats = {}
    for k, (s, f) in arms.items():
        au = np.array([FU.auc(lab[o[j]:o[j + 1]], s[o[j]:o[j + 1]]) for j in range(len(ids))])
        c = np.zeros((len(ids), 4))
        for j in range(len(ids)):
            if not nonc[j]: continue
            y, ff = err[o[j]:o[j + 1]], f[o[j]:o[j + 1]]
            c[j] = [(~y & ~ff).sum(), (y & ~ff).sum(), (y & ff).sum(), (~y & ff).sum()]   # TP FP TN FN (valid = positive)
        stats[k] = (np.where(np.isfinite(au), au, 0), np.isfinite(au).astype(float), c)
    rng = np.random.default_rng(SEED)
    W = np.vstack([np.ones(len(ug)), rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=DRAWS)])
    def agg(k):
        a, m, c = stats[k]; G = np.zeros((len(ug), 6)); np.add.at(G, inv, np.column_stack([a, m, c]))
        S = W @ G; tp, fp, tn, fn = S[:, 2:].T
        return S[:, 0] / S[:, 1], 100 * (tp / (tp + .5 * (fp + fn)) + tn / (tn + .5 * (fp + fn))) / 2
    R = {k: agg(k) for k in arms}
    rows = [{'arm': k, 'within_auc': R[k][0][0], 'prmscore': R[k][1][0]} for k in arms]
    for a, b in [('TWO_VIEW_MEAN_MOM_ga', 'PRM_native'), ('TWO_VIEW_MEAN_MOM_ga', 'PRM_Q80'), ('TWO_VIEW_MEAN_MOM_ga', 'N16_MEAN_MOM_ga'),
                 ('N16P_MEAN_MOM_ga', 'PRM_native'), ('N16P_MEAN_MOM_ga', 'PRM_Q80'), ('N16P_MEAN_MOM_ga', 'N16_MEAN_MOM_ga'),
                 ('PRM_Q80', 'N16_MEAN_MOM_ga')]:
        for m, name in ((0, 'within_auc'), (1, 'prmscore')):
            d = R[a][m] - R[b][m]
            rows.append({'arm': f'{a} - {b}', name + '_diff': d[0], name + '_lo': np.quantile(d[1:], .025), name + '_hi': np.quantile(d[1:], .975)})
    df = pd.DataFrame(rows).groupby('arm', sort=False).first().reset_index()
    df.to_csv(FU.OUT / f'BOOTSTRAP_{TAG}.csv', index=False)
    pd.set_option('display.width', 250); print(df.round(4).to_string())
    print('source groups', len(ug), 'answers', len(ids))


if __name__ == '__main__':
    main()
