"""channel_dominance.py: which of the 16 telemetry channels dominate, standalone vs PRM backbone, on ProcessBench.

Per cell (mode in {standalone 'sa', backbone 'prm'}, PRM pair) and channel, on all 3,400 answers:
  orient     label-free sign: sign of the pooled Pearson corr of the channel with the stored TELE_N16 score.
  fusion_r   mean over answers of the within-answer Pearson corr(oriented channel, TELE_N16). Label-free.
             High = the fused score follows this channel ("dominant in the fusion").
  first_auc  answers with gold first error >= 1: AUC of the oriented channel, first-error step vs the steps
             before it (those are correct by the ProcessBench definition). Mean over answers. Uses labels.
  sla_gap    ProcessBench SLA (Mind the Gap metric) of the oriented channel ALONE under the GAP first-flag rule.
Inputs: SCORES_<sa|CHAIN>_<prm>... score files (for TELE_N16) and SCORES_<mode>_<prm>_processbench_CHANNELS.npz
(keys uids, offsets, channels__N16, channel_names__N16). Search dirs: env CHANNEL_DIRS (';'-separated), then a
recursive search under env DATA_ROOT (default ../hd_collection_data).
    python channel_dominance.py  -> CHANNELS.csv, CHANNELS_SUMMARY.md
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from evaluate_sla_standalone import gap_predict, SUBSETS, PRMS, GOLD  # noqa: E402

DATA_ROOT = Path(os.environ.get('DATA_ROOT', ROOT.parent / 'hd_collection_data'))
DIRS = [Path(p) for p in os.environ.get('CHANNEL_DIRS', '').split(';') if p]


def find(name):
    for d in DIRS:
        if (d / name).exists(): return d / name
    hits = sorted(DATA_ROOT.rglob(name)) if DATA_ROOT.exists() else []
    return hits[0] if hits else None


def score_file(mode, prm):
    return find(f'SCORES_sa_{prm}_processbench.npz') if mode == 'sa' else \
        (find(f'SCORES_CHAIN_{prm}.npz') or find(f'SCORES_prm_{prm}_processbench.npz'))


def auc(pos, neg):
    if len(neg) == 0: return np.nan
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / len(neg))


def main():
    gold = {g['uid']: g for g in json.load(open(GOLD))}
    rows, missing = [], []
    for mode in ('sa', 'prm'):
        for prm in PRMS:
            sf = score_file(mode, prm); cf = find(f'SCORES_{mode}_{prm}_processbench_CHANNELS.npz')
            if sf is None or cf is None:
                missing.append((mode, prm, str(sf), str(cf))); print('MISSING', mode, prm, sf, cf); continue
            S = np.load(sf); Z = np.load(cf)
            su = S['uids'].astype(str); sl = S['nonempty_lengths'].astype(int); tele = S['score__TELE_N16'].astype(float)
            assert len(tele) == sl.sum(), (sf, len(tele), sl.sum())   # ProcessBench: no empty steps (DATA_CHECK)
            ss = np.r_[0, np.cumsum(sl)]
            names = [str(x) for x in Z['channel_names__N16']]; M = np.asarray(Z['channels__N16'], float)
            cu = Z['uids'].astype(str); off = Z['offsets'].astype(np.int64); pos = {u: j for j, u in enumerate(cu)}
            X = np.concatenate([M[off[pos[u]]:off[pos[u] + 1]] for u in su]); assert len(X) == len(tele), (cf, len(X))
            lab = np.array([gold[u]['label'] for u in su]); sub = np.array([gold[u]['subset'] for u in su])
            for c, nm in enumerate(names):
                x = np.nan_to_num(X[:, c])
                sgn = 1.0 if np.corrcoef(x, tele)[0, 1] >= 0 else -1.0
                x = sgn * x
                rs, aucs, hit = [], [], []
                for j, (a, b) in enumerate(zip(ss[:-1], ss[1:])):
                    xa, ta = x[a:b], tele[a:b]
                    if b - a >= 3 and xa.std() > 0 and ta.std() > 0: rs.append(np.corrcoef(xa, ta)[0, 1])
                    if lab[j] >= 1: aucs.append(auc(xa[lab[j]], xa[:lab[j]]))
                    if lab[j] != -1: hit.append((j, gap_predict(xa)[0] == lab[j]))
                hj = np.array([h for _, h in hit]); hs = sub[[j for j, _ in hit]]
                rows.append(dict(mode='standalone' if mode == 'sa' else 'backbone', prm=prm, channel=nm, orient=int(sgn),
                                 fusion_r=round(float(np.nanmean(rs)), 4), first_auc=round(float(np.nanmean(aucs)), 4),
                                 sla_gap=round(float(np.mean([100 * hj[hs == s].mean() for s in SUBSETS])), 2)))
            print('done', mode, prm, flush=True)
    with open(HERE / 'CHANNELS.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # summary: mean over the 4 PRM pairs, standalone vs backbone
    L = ['# Channel dominance, standalone vs backbone (ProcessBench, mean over available PRM pairs)', '',
         f'Missing cells: {missing}', '',
         '| channel | fusion_r SA | fusion_r BB | first_auc SA | first_auc BB | SLA SA | SLA BB |', '|---|---|---|---|---|---|---|']
    names = list(dict.fromkeys(r['channel'] for r in rows))
    def m(mode, nm, k):
        v = [r[k] for r in rows if r['mode'] == mode and r['channel'] == nm]; return round(float(np.mean(v)), 3) if v else float('nan')
    for nm in sorted(names, key=lambda n: -m('standalone', n, 'fusion_r')):
        L.append(f"| {nm} | {m('standalone', nm, 'fusion_r')} | {m('backbone', nm, 'fusion_r')} | {m('standalone', nm, 'first_auc')} | "
                 f"{m('backbone', nm, 'first_auc')} | {m('standalone', nm, 'sla_gap')} | {m('backbone', nm, 'sla_gap')} |")
    (HERE / 'CHANNELS_SUMMARY.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
