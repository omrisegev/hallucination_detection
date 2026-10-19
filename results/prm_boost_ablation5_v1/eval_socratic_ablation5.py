"""Five-arm ablation copy of ../prm_boost_external_v1/eval_external.py: same metrics and bootstrap, plus
paired contrasts between arms (TELE-MEAN, CHAIN-HALF, CHAIN-EQ, PRMG-HALF, CHAIN-TELE, HALF-MEAN).

Evaluate the frozen PRM-boosting recipes on an external benchmark (local, gold labels stay local).

Inputs: SCORES_<adapter>.npz written label-free by score_external.py on AIRCC. Metrics follow
results/external_lr_prm_v1 exactly: the benchmark's official metric (Socratic: PRMScore; Hard2Verify:
Balanced F1) and within-answer AUC, via spectral_utils.external_generalization.evaluation.confusion /
within_auc and spectral_utils.family_external_metrics.summary. Rules: PRM native (its own reward
threshold, all steps) and ANSWER_Z_Q80 on the non-empty steps (empty steps predicted incorrect).
Paired bootstrap over answers (1,000 draws, seed 20261018) for every arm minus PRM native and PRM Q80.

    python eval_external.py <benchmark> <dir with SCORES_*.npz> <gold dir> <out dir>
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from spectral_utils.external_generalization.evaluation import confusion, within_auc  # noqa: E402
from spectral_utils.family_external_metrics import summary  # noqa: E402

bench, sdir, gdir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
gold = {g['uid']: g for g in json.loads((gdir / f'{bench}.json').read_text(encoding='utf8'))}


def az_q80(s, off):
    az = np.zeros_like(s)
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; az[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return az >= np.quantile(az, .8)


def per_answer(S, score_ne, pred_valid_all):
    uids = [str(u) for u in S['uids']]; lens = S['nonempty_lengths']; ne = S['nonempty'].astype(bool)
    full_s = np.full(int(lens.sum()), np.nan); full_s[np.flatnonzero(ne)] = score_ne
    starts = np.r_[0, np.cumsum(lens)]; C, A = [], []
    for i, u in enumerate(uids):
        g = gold[u]; a, b = starts[i], starts[i + 1]; inc = np.asarray(g['include'], bool)
        C.append(confusion(g['correct'], pred_valid_all[a:b], inc))
        v = within_auc(g['correct'], full_s[a:b], inc & ne[a:b]); A.append(np.nan if v is None else v)
    return np.array(C, float), np.array(A, float)


rows, boot = [], []
rng = np.random.default_rng(20261018)
for f in sorted(sdir.glob('SCORES_*.npz')):
    model = f.stem[len('SCORES_'):]
    S = dict(np.load(f)); ne = S['nonempty'].astype(bool)
    lens = S['nonempty_lengths']; starts = np.r_[0, np.cumsum(lens)]
    off = np.r_[0, np.cumsum(np.add.reduceat(ne.astype(int), starts[:-1]))]
    res = {}
    for k in [k for k in S if k.startswith('score__')]:
        arm = k[len('score__'):]; s = S[k]
        pv = np.zeros(int(lens.sum()), int); pv[np.flatnonzero(ne)] = (~az_q80(s, off)).astype(int)
        res[f'{arm}_Q80'] = per_answer(S, s, pv)
    res['PRM_native'] = per_answer(S, S['score__PRM'], S['native_valid'].astype(int))
    n = len(S['uids'])
    for name, (C, A) in res.items():
        rows.append(dict(benchmark=bench, model=model, arm=name, official_metric=summary(C, bench)['metric'],
                         within_auc=float(np.nanmean(A)), n_answers=n))
    W = rng.integers(0, n, size=(1000, n))
    def stat(C, A, idx): return summary(C[idx], bench)['metric'], float(np.nanmean(A[idx]))
    pairs = [(a, r) for a in res for r in ('PRM_native', 'PRM_Q80') if a != r]
    for b in ('N13', 'N16'):
        pairs += [(f'TELE_{b}_Q80', f'MEAN_{b}_Q80'), (f'CHAIN_{b}_Q80', f'HALF_{b}_Q80'), (f'CHAIN_{b}_Q80', f'EQ_{b}_Q80'),
                  (f'PRMG_{b}_Q80', f'HALF_{b}_Q80'), (f'CHAIN_{b}_Q80', f'TELE_{b}_Q80'), (f'HALF_{b}_Q80', f'MEAN_{b}_Q80')]
    for name, ref in pairs:
        if True:
            d = np.array([np.subtract(stat(*res[name], w), stat(*res[ref], w)) for w in W])
            p = np.subtract(stat(*res[name], np.arange(n)), stat(*res[ref], np.arange(n)))
            for j, mt in enumerate(('official_metric', 'within_auc')):
                boot.append(dict(benchmark=bench, model=model, contrast=f'{name} - {ref}', metric=mt, delta=float(p[j]),
                                 ci_low=float(np.quantile(d[:, j], .025)), ci_high=float(np.quantile(d[:, j], .975))))
import csv
for path, rr in ((out / f'RESULTS_{bench}.csv', rows), (out / f'BOOTSTRAP_{bench}.csv', boot)):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rr[0])); w.writeheader(); w.writerows(rr)
for r in rows: print(r['model'], r['arm'], round(r['official_metric'], 4), round(r['within_auc'], 4))
