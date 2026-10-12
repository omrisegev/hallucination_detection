"""Post-hoc control (written after fuse.py showed that PRM as one channel of 17 is diluted):
two-view equal fusion = answer-z(PRM risk) + answer-z(our fused N16 score), weight 1/2 each.
Label-free; the equal weight is a declared default, not fitted. Same evaluation as fuse.py.

    python two_view.py [tag]
"""
import sys
import numpy as np
import pandas as pd
import fuse as FU

TAG = sys.argv[1] if len(sys.argv) > 1 else 'qwen3_8b_mask1'


def main():
    z = np.load(FU.OUT / f'SCORES_{TAG}.npz', allow_pickle=True)
    o = z['offsets']; ids = z['idx'].astype(str)
    az = lambda v: FU.ext.answer_standardize(np.asarray(v, float)[:, None], o)[:, 0]
    A = pd.read_csv(FU.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
    with np.load(FU.SE / 'OOF_STEP_SCORES.npz') as q:
        OFF = q['offsets'].astype(np.int64); LAB = q['labels'].astype(bool)
    pos = {i: k for k, i in enumerate(A.id.astype(str))}
    lab = [LAB[OFF[pos[i]]:OFF[pos[i] + 1]] for i in ids]
    import pickle
    prm = {r['idx']: r for r in pickle.load(open(FU.PRM_PKL, 'rb')).values()}
    meta = [{'idx': i, 'classification': prm[i]['classification'], 'error_steps': prm[i]['error_steps']} for i in ids]
    rows = []
    for s in ('MEAN_MOM_ga', 'DS_FILTER_EQ', 'ALL_EQ'):
        score = .5 * (az(z['prm_risk']) + az(z[f'N16__{s}']))
        zs = az(score); flags = zs > np.quantile(zs, .8)          # ANSWER_Z_Q80
        au = np.array([FU.auc(lab[j], score[o[j]:o[j + 1]]) for j in range(len(ids))])
        preds = [{'idx': i, 'labels': (~flags[o[j]:o[j + 1]]).astype(int).tolist()} for j, i in enumerate(ids)]
        ev = FU.prmbench_evaluate(preds, meta); t = ev['total']; bc = ev['by_classification']
        r = {'bank': 'PRM+N16(2view)', 'scorer': s, 'rule': 'ANSWER_Z_Q80', 'within_auc': float(np.nanmean(au)),
             'prmscore': 100 * (t['f1'] + t['negative_f1']) / 2, 'f1_valid': t['f1'], 'f1_error': t['negative_f1'], 'flag_rate': float(flags.mean())}
        for c in sorted({m['classification'] for m in meta} - {'correct'}):
            r['prmscore__' + c] = 100 * (bc['f1'][c] + bc['negative_f1'][c]) / 2
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(FU.OUT / f'RESULTS_TWO_VIEW_{TAG}.csv', index=False)
    print(df[['bank', 'scorer', 'within_auc', 'prmscore', 'f1_valid', 'f1_error', 'flag_rate']].round(4).to_string())


if __name__ == '__main__':
    main()
