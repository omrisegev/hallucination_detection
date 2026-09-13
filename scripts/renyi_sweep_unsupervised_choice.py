"""Can a label-free statistic of the answer tell us which view to trust?  (post hoc, exploratory)

Views: VE_0, VE_0.75, VE_1 (varentropy15), H0lim, H1 (entropy).  Label-free per-answer statistics available from
the frozen contract without touching pickles: number of steps, number of tokens, the gate detector (mean token
entropy), and statistics of the saved step scores themselves (argmax position of each view, agreement between
views, spread of the step scores).  Labels are used ONLY to evaluate: on PB erroneous answers, which view hits.

Reported: (1) when VE_0 and VE_0.75 disagree on the argmax step, who is right, split by n_steps, by the relative
position of each argmax, and by which argmax is earlier; (2) hit rates of simple label-free rules on the pair:
"later argmax", "earlier argmax", "VE_0.75 if its argmax is step 0 else VE_0", "mean of per-answer z-scored step
scores", "max of z-scored step scores" (each rule uses no labels; their evaluation does); (3) PRMB: per-answer
AUC difference VE_0 − VE_0.75 versus n_steps / n_tokens / detector (Spearman), to see whether any label-free
quantity predicts which view ranks better.  Writes UNSUPERVISED_CHOICE.{md,json}.  Nothing here is a candidate.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ['RENYI_V2_ROSTER'] = 'all'
from scripts import run_renyi_view_fusion_v2 as run

base = run.base
OUT = ROOT / 'results/renyi_alpha_sweep_v1'
V = {'VE_0': 'view__ve0', 'VE_0.75': 'view__ve0.75', 'VE_1': 'view__ve1', 'H0lim': 'view__H0lim', 'H1': 'view__H1'}


def auc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if not n1 or not n0: return np.nan
    return float((rankdata(s, method='average')[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    source = Path(sys.argv[1]).resolve(); base.old.configure_source_root(source)
    records = json.loads((base.old.BENCH / 'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined = np.load(base.old.BENCH / 'evaluation/JOINED.npz', allow_pickle=False)
    offsets, labels, target = joined['offsets'], joined['labels'], joined['target']
    steps = np.diff(offsets); cells = np.array([r['cell'] for r in records]); pb = np.char.startswith(cells, 'pb_')
    tokens = np.array([r['tokens'] for r in records]); detector, thr = base.old._gate_contract(records)
    with np.load(OUT / 'SCORES.npz') as npz:
        S = {k: npz['steps__' + m] for k, m in V.items()}; valid = {k: npz['valid__' + m] for k, m in V.items()}
    out = {}; L = ['# Label-free choice between views (post hoc, exploratory)', '']
    # ---- PB: disagreement analysis and simple rules on the VE_0 / VE_0.75 pair
    E = np.flatnonzero(pb & (target >= 0) & valid['VE_0'] & valid['VE_0.75'] & np.isfinite(detector) & (detector >= thr))
    am = {k: np.array([int(np.argmax(S[k][offsets[i]:offsets[i + 1]])) for i in E]) for k in V}
    t = target[E]; n = steps[E]
    hit = {k: am[k] == t for k in V}
    dis = am['VE_0'] != am['VE_0.75']
    out['pb'] = dict(n_gated_erroneous=int(len(E)), disagree_fraction=float(dis.mean()),
                     when_disagree=dict(ve0_right=float(hit['VE_0'][dis].mean()), ve075_right=float(hit['VE_0.75'][dis].mean()),
                                        neither=float((~hit['VE_0'] & ~hit['VE_0.75'])[dis].mean())))
    L += ['## ProcessBench (gate open, erroneous, both views valid): %d answers' % len(E), '',
          f"VE_0 and VE_0.75 point at different steps on {dis.mean():.1%} of them. When they disagree: VE_0 right {hit['VE_0'][dis].mean():.3f}, "
          f"VE_0.75 right {hit['VE_0.75'][dis].mean():.3f}, neither {(~hit['VE_0'] & ~hit['VE_0.75'])[dis].mean():.3f}.", '']
    # who is right when they disagree, by which argmax is earlier
    earlier_is_ve0 = am['VE_0'] < am['VE_0.75']
    rows = []
    for name, msk in (('VE_0 points earlier', dis & earlier_is_ve0), ('VE_0.75 points earlier', dis & ~earlier_is_ve0)):
        rows.append(dict(case=name, n=int(msk.sum()), ve0_right=float(hit['VE_0'][msk].mean()), ve075_right=float(hit['VE_0.75'][msk].mean()),
                         earlier_right=float(np.where(earlier_is_ve0, hit['VE_0'], hit['VE_0.75'])[msk].mean()),
                         later_right=float(np.where(earlier_is_ve0, hit['VE_0.75'], hit['VE_0'])[msk].mean())))
    out['pb']['by_who_is_earlier'] = rows
    L += ['| disagreement case | n | VE_0 right | VE_0.75 right | earlier argmax right | later argmax right |', '|---|---:|---:|---:|---:|---:|']
    L += [f"| {r['case']} | {r['n']} | {r['ve0_right']:.3f} | {r['ve075_right']:.3f} | {r['earlier_right']:.3f} | {r['later_right']:.3f} |" for r in rows]
    # by n_steps
    L += ['', '| n_steps (disagreements only) | n | VE_0 right | VE_0.75 right |', '|---|---:|---:|---:|']; out['pb']['by_steps'] = {}
    for lo, hi, nm in ((1, 4, '1-4'), (5, 8, '5-8'), (9, 12, '9-12'), (13, 10 ** 6, '13+')):
        msk = dis & (n >= lo) & (n <= hi)
        if msk.sum(): out['pb']['by_steps'][nm] = dict(n=int(msk.sum()), ve0=float(hit['VE_0'][msk].mean()), ve075=float(hit['VE_0.75'][msk].mean()))
        if msk.sum(): L.append(f"| {nm} | {int(msk.sum())} | {hit['VE_0'][msk].mean():.3f} | {hit['VE_0.75'][msk].mean():.3f} |")
    # label-free rules, evaluated on all gated erroneous answers (hit rate) -- no labels used inside the rules
    def z(x): sd = x.std(); return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)
    rules = {}
    rules['VE_0 alone'] = am['VE_0']; rules['VE_0.75 alone'] = am['VE_0.75']
    rules['later argmax of the pair'] = np.maximum(am['VE_0'], am['VE_0.75'])
    rules['earlier argmax of the pair'] = np.minimum(am['VE_0'], am['VE_0.75'])
    rules['VE_0.75 if its argmax is step 0, else VE_0'] = np.where(am['VE_0.75'] == 0, am['VE_0.75'], am['VE_0'])
    for combo, keys in (('mean of z-scored steps: VE_0 + VE_0.75', ('VE_0', 'VE_0.75')), ('mean of z-scored steps: VE_0 + VE_0.75 + VE_1 + H0lim', ('VE_0', 'VE_0.75', 'VE_1', 'H0lim')),
                        ('mean of z-scored steps: all five', tuple(V))):
        rules[combo] = np.array([int(np.argmax(np.mean([z(S[k][offsets[i]:offsets[i + 1]]) for k in keys], axis=0))) for i in E])
    rules['max of z-scored steps: VE_0 + VE_0.75'] = np.array([int(np.argmax(np.max([z(S[k][offsets[i]:offsets[i + 1]]) for k in ('VE_0', 'VE_0.75')], axis=0))) for i in E])
    rules['oracle: either of the pair (ceiling, uses labels)'] = np.where(hit['VE_0'], am['VE_0'], am['VE_0.75'])
    out['pb']['rules_hit_rate'] = {k: float(np.mean(v == t)) for k, v in rules.items()}
    L += ['', '## Label-free combination rules on the pair (hit rate on the same answers; rules use no labels)', '', '| rule | exact-hit rate |', '|---|---:|']
    L += [f'| {k} | {v:.3f} |' for k, v in out['pb']['rules_hit_rate'].items()]
    # ---- PRMB: does any label-free quantity predict which view ranks better?
    M = np.flatnonzero(~pb); d = np.full(len(M), np.nan); A = {k: np.full(len(M), np.nan) for k in V}
    for j, i in enumerate(M):
        y = labels[offsets[i]:offsets[i + 1]]; keep = y >= 0
        if not (y[keep] == 1).any() or not (y[keep] == 0).any(): continue
        for k in V:
            s = S[k][offsets[i]:offsets[i + 1]]
            if np.isfinite(s).all(): A[k][j] = auc(y[keep] == 1, s[keep])
    ok = np.isfinite(A['VE_0']) & np.isfinite(A['VE_0.75']); diff = (A['VE_0'] - A['VE_0.75'])[ok]
    feats = {'n_steps': steps[M][ok], 'n_tokens': tokens[M][ok], 'mean token entropy (gate detector)': detector[M][ok],
             'argmax position of VE_0 (relative)': np.array([np.argmax(S['VE_0'][offsets[i]:offsets[i + 1]]) / max(steps[i] - 1, 1) for i in M])[ok],
             'argmax position of VE_0.75 (relative)': np.array([np.argmax(S['VE_0.75'][offsets[i]:offsets[i + 1]]) / max(steps[i] - 1, 1) for i in M])[ok],
             'argmax disagreement (|pos diff|)': np.array([abs(np.argmax(S['VE_0'][offsets[i]:offsets[i + 1]]) - np.argmax(S['VE_0.75'][offsets[i]:offsets[i + 1]])) / max(steps[i] - 1, 1) for i in M])[ok],
             'std of VE_0 step scores / std of VE_0.75': np.array([S['VE_0'][offsets[i]:offsets[i + 1]].std() / max(S['VE_0.75'][offsets[i]:offsets[i + 1]].std(), 1e-9) for i in M])[ok]}
    out['prmb'] = dict(n=int(ok.sum()), mean_diff=float(diff.mean()), spearman_with_auc_diff={})
    L += ['', f'## PRMBench: does a label-free answer statistic predict (AUC VE_0 − AUC VE_0.75)?  n = {int(ok.sum())}, mean diff {diff.mean():+.4f}', '',
          '| label-free statistic | Spearman with AUC diff | VE_0 better in lowest quartile | in highest quartile |', '|---|---:|---:|---:|']
    for name, x in feats.items():
        x = np.asarray(x, float); rho = spearmanr(x, diff).correlation; q1, q3 = np.quantile(x, [.25, .75])
        lo = np.mean(diff[x <= q1] > 0); hi = np.mean(diff[x >= q3] > 0)
        out['prmb']['spearman_with_auc_diff'][name] = dict(rho=float(rho), ve0_better_low_quartile=float(lo), ve0_better_high_quartile=float(hi))
        L.append(f'| {name} | {rho:+.3f} | {lo:.3f} | {hi:.3f} |')
    L += ['', 'Reading rule: a |Spearman| near 0 means the statistic cannot tell which view to trust; the quartile columns show the fraction of answers where VE_0 has the higher AUC.']
    base.atomic_json(OUT / 'UNSUPERVISED_CHOICE.json', run.clean(out)); (OUT / 'UNSUPERVISED_CHOICE.md').write_text('\n'.join(L), encoding='utf8'); print('\n'.join(L))


if __name__ == '__main__':
    main()
