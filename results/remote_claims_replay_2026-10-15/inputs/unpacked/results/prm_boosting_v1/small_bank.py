"""PRM-dominant small bank (Omri 2026-10-13): PRM risk + k of our strongest, conditionally independent channels.

Pool: the 56 channels of the PRM-backbone bank (bb:) and of the Qwen3-8B mask1 bank (q3:), oriented
answer-z as in channel_atlas.py (orientation = sign of the full-data within-AUC; every top channel is +1).
Fusion: score = w * z(PRM) + (1 - w) * mean_j z(channel_j), w = 1/2 (PRM keeps half the weight whatever k is).

LABEL-SELECTED: channels are chosen with labels by greedy forward selection, which maximizes the training
within-AUC of the fused score (so a channel only enters if it adds information beyond PRM and the channels
already chosen, i.e. strength and conditional independence together). Honest estimate: selection runs inside
the 5 source-group folds (FOLDS_V2 'fold'), and each test fold is scored with its own training-fold selection.
Also reported: the selection on all data (what one would deploy) and fixed named sets.

    python small_bank.py
"""
import json, pickle
import numpy as np
import pandas as pd
import fuse as FU

KMAX, W_PRM = 6, .5
D = FU.OUT


def pairs(y, o):
    """All (error, valid) step pairs inside each answer, with weight 1/(#pairs of that answer)."""
    E, Vv, Wt, Aid = [], [], [], []
    for j in range(len(o) - 1):
        yy = y[o[j]:o[j + 1]]; e = np.flatnonzero(yy) + o[j]; v = np.flatnonzero(~yy) + o[j]
        if len(e) and len(v):
            ee, vv = np.meshgrid(e, v, indexing='ij'); E.append(ee.ravel()); Vv.append(vv.ravel())
            Wt.append(np.full(ee.size, 1. / ee.size)); Aid.append(np.full(ee.size, j))
    return np.concatenate(E), np.concatenate(Vv), np.concatenate(Wt), np.concatenate(Aid)


def main():
    Z = {t: np.load(D / f'CHANNEL_CONDCORR_{t}.npz', allow_pickle=True) for t in ('prm_backbone', 'qwen3_8b_mask1')}
    o = Z['prm_backbone']['offsets']; y = Z['prm_backbone']['y'].astype(bool); ids = Z['prm_backbone']['idx'].astype(str)
    assert np.array_equal(o, Z['qwen3_8b_mask1']['offsets']) and np.array_equal(y, Z['qwen3_8b_mask1']['y'])
    nb = [str(x) for x in Z['prm_backbone']['names']]
    prm = Z['prm_backbone']['S'][:, nb.index('prm_risk')].astype(float)
    names, cols = [], []
    for t, p in (('prm_backbone', 'bb:'), ('qwen3_8b_mask1', 'q3:')):
        nm = [str(x) for x in Z[t]['names']]
        for k, c in enumerate(nm):
            if c != 'prm_risk': names.append(p + c); cols.append(Z[t]['S'][:, k])
    X = np.column_stack(cols).astype(float)
    # drop exact duplicates (e.g. q15_VE1 == ct7_ve1, hist_entropy_series == q15_H1) keeping the first name
    keep, seen = [], {}
    for k in range(X.shape[1]):
        key = (names[k][:3], np.round(X[:200, k], 6).tobytes())
        if key in seen and np.allclose(X[:, k], X[:, seen[key]]): continue
        seen[key] = k; keep.append(k)
    X = X[:, keep]; names = [names[k] for k in keep]
    A = pd.read_csv(FU.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig').set_index('id')
    fold = A.loc[ids, 'fold'].to_numpy(); afold = np.repeat(fold, np.diff(o))
    E, Vv, Wt, Aid = pairs(y, o)
    def auc_on(score, amask):
        m = amask[Aid]; c = (score[E[m]] > score[Vv[m]]) + .5 * (score[E[m]] == score[Vv[m]])
        return float((Wt[m] * c).sum() / len(np.unique(Aid[m])))
    def fused(sel):
        return W_PRM * prm + (1 - W_PRM) * (X[:, sel].mean(1) if sel else 0) if sel else prm
    def greedy(amask):
        sel, path = [], []
        for _ in range(KMAX):
            best = max((auc_on(fused(sel + [k]), amask), k) for k in range(X.shape[1]) if k not in sel)
            sel.append(best[1]); path.append((names[best[1]], round(best[0], 4)))
        return sel, path
    # out-of-fold: selection on 4 folds, prefix of length k applied to the 5th
    oof = {k: np.zeros(len(prm)) for k in range(1, KMAX + 1)}; fold_paths = {}
    for f in range(5):
        tr = fold != f; sel, path = greedy(tr); fold_paths[f] = path
        te = afold == f
        for k in range(1, KMAX + 1): oof[k][te] = fused(sel[:k])[te]
        print('fold', f, path, flush=True)
    sel_all, path_all = greedy(np.ones(len(ids), bool)); print('all-data selection', path_all, flush=True)
    # evaluation (official PRMScore under ANSWER_Z_Q80 + within-AUC), same as compose.py
    prmd = {r['idx']: r for r in pickle.load(open(FU.PRM_PKL, 'rb')).values()}
    meta = [{'idx': i, 'classification': prmd[i]['classification'], 'error_steps': prmd[i]['error_steps']} for i in ids]
    az = lambda v: FU.ext.answer_standardize(np.asarray(v, float)[:, None], o)[:, 0]
    allm = np.ones(len(ids), bool)
    def evaluate(s):
        zs = az(s); f = zs > np.quantile(zs, .8)
        preds = [{'idx': i, 'labels': (~f[o[j]:o[j + 1]]).astype(int).tolist()} for j, i in enumerate(ids)]
        t = FU.prmbench_evaluate(preds, meta)['total']
        return auc_on(s, allm), 100 * (t['f1'] + t['negative_f1']) / 2
    bbz = az(np.load(D / 'SCORES_prm_backbone.npz')['N16__MEAN_MOM_ga'])
    named = {'digits+surprisal+evidence_drop (Omri)': ['bb:digit_spread', 'bb:digit_alternative', 'bb:chosen_surprisal', 'bb:evidence_drop_risk']}
    rows = [{'arm': 'PRM alone (Q80)', **dict(zip(['within_auc', 'prmscore'], evaluate(prm)))},
            {'arm': 'EQ2(PRM, N16 backbone MEAN_MOM_ga)', **dict(zip(['within_auc', 'prmscore'], evaluate(.5 * prm + .5 * bbz)))}]
    for nm, chans in named.items():
        rows.append({'arm': nm, 'channels': chans, **dict(zip(['within_auc', 'prmscore'], evaluate(fused([names.index(c) for c in chans]))))})
    for k in range(1, KMAX + 1):
        rows.append({'arm': f'greedy k={k}, out-of-fold', **dict(zip(['within_auc', 'prmscore'], evaluate(oof[k])))})
        rows.append({'arm': f'greedy k={k}, all-data selection (in-sample)', 'channels': [p[0] for p in path_all[:k]],
                     **dict(zip(['within_auc', 'prmscore'], evaluate(fused(sel_all[:k]))))})
    df = pd.DataFrame(rows); df.to_csv(D / 'SMALL_BANK.csv', index=False)
    np.savez_compressed(D / 'SMALL_BANK_OOF.npz', idx=ids, offsets=o, **{f'oof_k{k}': v for k, v in oof.items()})
    json.dump({'fold_paths': fold_paths, 'all_data_path': path_all, 'w_prm': W_PRM, 'pool_size': len(names)},
              open(D / 'SMALL_BANK_SELECTION.json', 'w'), indent=1, default=str)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
    print(df.round(4).to_string())


if __name__ == '__main__':
    main()
