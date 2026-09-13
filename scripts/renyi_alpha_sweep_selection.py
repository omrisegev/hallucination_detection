"""Selection-stability analysis for the Renyi alpha sweep (development evidence only).

Reads results/renyi_alpha_sweep_v1/{METRICS.json, SCORES.npz} and reports, for each family
(Renyi entropy H_alpha; escort varentropy VE_alpha):
  * the full curve (PB all-8, PRMB within, pooled, PRMScore) by alpha;
  * the argmax alpha per endpoint on the whole development set;
  * fold-wise stability: for each of the 5 outer source-group folds (FOLDS_V2) the within-AUC argmax alpha on
    PRMB answers in that fold, and for each PB cell the PB-F1 argmax alpha, plus how far the per-fold/per-cell
    optimum's metric is from the global optimum's (a flat region is not a sharp optimum);
  * a "held-out selection" number: choose alpha by within-AUC on the other four folds, evaluate on the held fold,
    mean over folds (cross-fitted development estimate; still development data, not confirmation).
Writes SELECTION.json and SELECTION.md.  No score is changed; no candidate is promoted.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ['RENYI_V2_ROSTER'] = 'all'
from scripts import run_renyi_view_fusion_v2 as run
from spectral_utils.renyi_alpha_sweep import ALPHA_OF, FAMILY_OF, METHODS

base = run.base
OUT = ROOT / 'results/renyi_alpha_sweep_v1'


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s, method='average')[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    source = Path(sys.argv[1]).resolve(); base.old.configure_source_root(source)
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    offsets, labels, target = joined['offsets'], joined['labels'], joined['target']
    metrics = json.loads((OUT / 'METRICS.json').read_text(encoding='utf8'))['metrics']
    fmap = json.loads(base.old.FOLDS.read_text(encoding='utf8'))['outer']; folds = np.array([fmap[r['group_id']] for r in records])
    cells = np.array([r['cell'] for r in records]); pb = np.char.startswith(cells, 'pb_')
    Z = np.load(OUT / 'SCORES.npz')
    families = {'renyi_entropy': [m for m in METHODS if FAMILY_OF[m] == 'renyi_entropy'],
                'escort_varentropy': [m for m in METHODS if FAMILY_OF[m] == 'escort_varentropy']}
    # per-answer within-AUC for every arm (PRMB mixed answers), and per-answer PB hit
    within = {}; hit = {}
    for m in METHODS:
        flat = Z['steps__' + m]; pred = Z['prediction__' + m]; valid = Z['valid__' + m]
        w = np.full(len(records), np.nan)
        for i in np.flatnonzero(~pb):
            s = flat[offsets[i]:offsets[i + 1]]; y = labels[offsets[i]:offsets[i + 1]]; keep = y >= 0
            if valid[i] and keep.any(): w[i] = auc(y[keep] == 1, s[keep])
        within[m] = w; hit[m] = valid & (pred == target)
    out = {}
    lines = ['# Renyi alpha sweep — selection stability (development evidence)', '']
    for fam, ms in families.items():
        alphas = [ALPHA_OF[m] for m in ms]
        curve = [dict(method=m, alpha=('inf' if np.isinf(ALPHA_OF[m]) else ALPHA_OF[m]), pb=100 * metrics[m]['pb_all8'],
                      within=metrics[m]['prm_within'], pooled=metrics[m]['prm_pooled'], prmscore=metrics[m]['prmscore_q08']) for m in ms]
        argmax = {k: max(curve, key=lambda c: c[k])['method'] for k in ('pb', 'within', 'pooled', 'prmscore')}
        # fold-wise within-AUC argmax (PRMB), cell-wise PB argmax
        fold_arg = {}
        for f in sorted(set(folds[~pb])):
            idx = np.flatnonzero((~pb) & (folds == f))
            means = {m: float(np.nanmean(within[m][idx])) for m in ms}
            best = max(means, key=means.get); fold_arg[int(f)] = dict(best=best, best_within=means[best], n=int(len(idx)),
                                                                    global_best_within=means[argmax['within']])
        cell_arg = {}
        for cell in sorted(set(cells[pb])):
            cc = (cells == cell)
            def f1(m):
                clean = cc & (target < 0); err = cc & (target >= 0)
                a = hit[m][clean].mean(); b = hit[m][err].mean(); return 200 * a * b / (a + b) if a + b else 0.0
            vals = {m: f1(m) for m in ms}; best = max(vals, key=vals.get)
            cell_arg[cell] = dict(best=best, best_f1=vals[best], global_best_f1=vals[argmax['pb']])
        # cross-fitted selection by within-AUC: choose on other folds, evaluate on held fold
        held = []
        for f in sorted(set(folds[~pb])):
            train = np.flatnonzero((~pb) & (folds != f)); test = np.flatnonzero((~pb) & (folds == f))
            choice = max(ms, key=lambda m: float(np.nanmean(within[m][train])))
            held.append(dict(fold=int(f), chosen=choice, held_within=float(np.nanmean(within[choice][test])),
                             oracle_within=float(max(np.nanmean(within[m][test]) for m in ms))))
        out[fam] = dict(curve=curve, argmax=argmax, fold_within_argmax=fold_arg, cell_pb_argmax=cell_arg, cross_fitted=held,
                        cross_fitted_mean_within=float(np.mean([h['held_within'] for h in held])),
                        cross_fitted_mean_oracle=float(np.mean([h['oracle_within'] for h in held])))
        lines += [f'## {fam}', '', '| alpha | PB % | within | pooled | PRMScore |', '|---|---:|---:|---:|---:|']
        lines += [f"| {c['alpha']} | {c['pb']:.2f} | {c['within']:.4f} | {c['pooled']:.4f} | {c['prmscore']:.4f} |" for c in curve]
        lines += ['', 'argmax: ' + ', '.join(f'{k}: {v}' for k, v in argmax.items()),
                  'fold-wise within argmax: ' + ', '.join(f"fold {f}: {v['best']} ({v['best_within']:.4f} vs global-best {v['global_best_within']:.4f})" for f, v in fold_arg.items()),
                  'cell-wise PB argmax: ' + ', '.join(f"{c}: {v['best']} ({v['best_f1']:.2f} vs {v['global_best_f1']:.2f})" for c, v in cell_arg.items()),
                  f"cross-fitted within (choose on 4 folds, evaluate on 1): {out[fam]['cross_fitted_mean_within']:.4f} "
                  f"(oracle per fold {out[fam]['cross_fitted_mean_oracle']:.4f}); choices: " + ', '.join(h['chosen'] for h in held), '']
    base.atomic_json(OUT / 'SELECTION.json', run.clean(out)); (OUT / 'SELECTION.md').write_text('\n'.join(lines), encoding='utf8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
