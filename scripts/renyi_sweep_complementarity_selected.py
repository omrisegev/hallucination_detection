"""Position behaviour of the selected-token surprisal and other frozen streams (post hoc, descriptive).

Adds to COMPLEMENTARITY: view__sel1 (= -log p of the selected token, from the Stage-3 fast pass), the frozen
direct-probability IU (17 inputs) and the sweep leaders VE_0 / VE_0.75 / VE_1 / H0lim, on the same strata
(PB hit rate by first-error position; PRMB within-AUC by first-error position), plus pairwise overlap of sel1
with the leaders.  Writes COMPLEMENTARITY_SELECTED_TOKEN.{md,json}.
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

base = run.base
SWEEP = ROOT / 'results/renyi_alpha_sweep_v1'; FAST = ROOT / 'results/renyi_view_fusion_v2/fast_pass'
ARMS = {'sel1 (-log p selected)': (FAST, 'view__sel1'), 'direct_iu (17 inputs)': (FAST, 'direct_iu'),
        'VE_0': (SWEEP, 'view__ve0'), 'VE_0.75': (SWEEP, 'view__ve0.75'), 'VE_1 (varentropy15)': (SWEEP, 'view__ve1'),
        'H0lim': (SWEEP, 'view__H0lim'), 'H1 (entropy)': (SWEEP, 'view__H1'), 'Hinf': (SWEEP, 'view__Hinf')}


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
    Z = {}
    for d in (SWEEP, FAST):
        with np.load(d / 'SCORES.npz') as npz:
            Z[d] = {k: npz[k] for k in npz.files if any(k.endswith('__' + m) for _, m in ARMS.values())}
    E = np.flatnonzero(pb & (target >= 0)); rel = target[E] / np.maximum(steps[E] - 1, 1)
    rel_s = np.select([rel < 1 / 3, rel < 2 / 3], ['first_third', 'middle_third'], 'last_third')
    abs_s = np.select([target[E] == 0, target[E] == 1, target[E] == 2, target[E] <= 5], ['step0', 'step1', 'step2', 'step3-5'], 'step6+')
    names = list(ARMS)
    H = np.column_stack([(Z[d]['valid__' + m] & (Z[d]['prediction__' + m] == target))[E] for d, m in ARMS.values()])
    M = np.flatnonzero(~pb); P = np.full((len(M), len(names)), np.nan); first_rel = np.full(len(M), np.nan)
    for j, i in enumerate(M):
        y = labels[offsets[i]:offsets[i + 1]]; keep = y >= 0
        if not (y[keep] == 1).any() or not (y[keep] == 0).any(): continue
        first_rel[j] = np.flatnonzero(y == 1)[0] / max(len(y) - 1, 1)
        for k, (d, m) in enumerate(ARMS.values()):
            s = Z[d]['steps__' + m][offsets[i]:offsets[i + 1]]
            if np.isfinite(s).all(): P[j, k] = auc(y[keep] == 1, s[keep])
    ok = np.isfinite(first_rel) & np.isfinite(P).all(1); P = P[ok]; prel = first_rel[ok]
    prel_s = np.select([prel < 1 / 3, prel < 2 / 3], ['first_third', 'middle_third'], 'last_third')
    out = dict(scope='POST_HOC_DESCRIPTIVE', arms=names, n_pb_erroneous=int(len(E)), n_prmb_mixed=int(ok.sum()))
    out['pb_by_relative'] = {k: {n: float(H[rel_s == k, j].mean()) for j, n in enumerate(names)} for k in ('first_third', 'middle_third', 'last_third')}
    out['pb_by_absolute'] = {k: {n: float(H[abs_s == k, j].mean()) for j, n in enumerate(names)} for k in ('step0', 'step1', 'step2', 'step3-5', 'step6+')}
    out['prmb_by_relative'] = {k: {n: float(P[prel_s == k, j].mean()) for j, n in enumerate(names)} for k in ('first_third', 'middle_third', 'last_third')}
    s1 = names.index('sel1 (-log p selected)'); out['sel1_overlap'] = {}
    for j, n in enumerate(names):
        if j == s1: continue
        a, b = H[:, s1], H[:, j]
        out['sel1_overlap'][n] = dict(both=int((a & b).sum()), only_sel1=int((a & ~b).sum()), only_other=int((~a & b).sum()), union_rate=float((a | b).mean()),
                                      auc_corr=float(np.corrcoef(P[:, s1], P[:, j])[0, 1]))
    base.atomic_json(SWEEP / 'COMPLEMENTARITY_SELECTED_TOKEN.json', run.clean(out))
    L = ['# Selected-token surprisal and other frozen streams by error position (post hoc)', '', f'PB erroneous {len(E)}; PRMB mixed {int(ok.sum())}.', '',
         '## PB exact-hit rate', '', '| arm | overall | step0 | step1 | step2 | step3-5 | step6+ | first third | middle | last third |', '|---|' + '---:|' * 9]
    for j, n in enumerate(names):
        L.append(f'| {n} | {H[:, j].mean():.3f} | ' + ' | '.join(f"{out['pb_by_absolute'][k][n]:.3f}" for k in ('step0', 'step1', 'step2', 'step3-5', 'step6+')) + ' | ' +
                 ' | '.join(f"{out['pb_by_relative'][k][n]:.3f}" for k in ('first_third', 'middle_third', 'last_third')) + ' |')
    L += ['', '## PRMB within-AUC by first-error position', '', '| arm | overall | first third | middle | last third |', '|---|---:|---:|---:|---:|']
    for j, n in enumerate(names):
        L.append(f'| {n} | {P[:, j].mean():.4f} | ' + ' | '.join(f"{out['prmb_by_relative'][k][n]:.4f}" for k in ('first_third', 'middle_third', 'last_third')) + ' |')
    L += ['', '## Overlap of sel1 PB hits with the other arms', '', '| other | both | only sel1 | only other | union | per-answer AUC corr |', '|---|---:|---:|---:|---:|---:|']
    L += [f"| {n} | {d['both']} | {d['only_sel1']} | {d['only_other']} | {d['union_rate']:.3f} | {d['auc_corr']:.3f} |" for n, d in out['sel1_overlap'].items()]
    (SWEEP / 'COMPLEMENTARITY_SELECTED_TOKEN.md').write_text('\n'.join(L), encoding='utf8'); print('\n'.join(L))


if __name__ == '__main__':
    main()
