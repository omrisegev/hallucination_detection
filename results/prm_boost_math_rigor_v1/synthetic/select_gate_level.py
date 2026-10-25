"""Choose the gate's interval level on synthetic data only, for the estimator chosen by select_gate_estimator.py.

Rule, fixed before this script was run:
  - Same regimes and seeds as select_gate_estimator.py (heavy-tailed family is primary).
  - For each regime, the 'cumulant' gate's G is bootstrapped once over answers (B = 100, 100, 60 by size).
  - Candidate rules: point estimate (fuse iff G > 0), and fuse iff the lower quantile of a central interval is > 0
    at levels 0.50, 0.70, 0.80, 0.90.
  - Score: action accuracy (abstain -> PRM), averaged with equal weight over the three sizes.
  - Choose the best; ties within 0.02 go to the HIGHER level (the more conservative rule).

    python select_gate_level.py <out_dir>
"""
import json, sys, zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_gate_estimator as S  # noqa: E402
from spectral_utils.fusion_theory import anchored_loadings, covariate_basis, gate_statistic, within_answer_residualize  # noqa: E402

LEVELS = ('point', .5, .7, .8, .9)


def boot_G(Xe, off, w, B, seed):
    u, info = anchored_loadings(Xe, 0, 'cumulant')
    if u is None:
        return None, []
    G0 = gate_statistic(info['C'], u, w)['G']
    rng = np.random.default_rng(seed); st, en = off[:-1], off[1:]; Gs = []
    for _ in range(B):
        pick = rng.integers(0, len(st), len(st))
        rows = np.concatenate([np.arange(st[j], en[j]) for j in pick])
        ub, ib = anchored_loadings(Xe[rows], 0, 'cumulant')
        if ub is not None:
            Gs.append(gate_statistic(ib['C'], ub, w)['G'])
    return G0, Gs


def decide(G0, Gs, B, level):
    if G0 is None:
        return 'abstain'
    if level == 'point':
        return 'fuse' if G0 > 0 else 'prm'
    if len(Gs) < .8 * B:
        return 'abstain'
    lo, hi = np.quantile(Gs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return 'fuse' if lo > 0 else ('prm' if hi < 0 else 'abstain')


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    plan = {'hard2verify': 30, 'socratic': 30, 'prmbench': 16}; Bs = {'hard2verify': 100, 'socratic': 100, 'prmbench': 60}
    res = {}
    for family, (noise, shared) in {'heavy': ('exp', False), 'gauss': ('gauss', False)}.items():
        for size, (J, nbar) in S.SIZES.items():
            for r in range(plan[size]):
                seed = zlib.crc32(f'{family}/{size}/{r}'.encode())
                rng = np.random.default_rng(seed)
                X, y, off, pos = S.regime(rng, J, nbar, noise, shared)
                Xz = S.answer_z(X, off); w = S.chain_weights(Xz)
                truth = S.within_auc(y, Xz @ w, off) - S.within_auc(y, Xz[:, 0], off)
                Xe = within_answer_residualize(Xz, covariate_basis(pos), off)
                G0, Gs = boot_G(Xe, off, w, Bs[size], seed)
                for lev in LEVELS:
                    act = decide(G0, Gs, Bs[size], lev)
                    res.setdefault((family, lev, size), []).append(((act == 'fuse') == (truth > 0), act == 'abstain'))
            print(family, size, 'done', flush=True)
    summ = {}
    for family in ('heavy', 'gauss'):
        for lev in LEVELS:
            per = {s: {'action_accuracy': float(np.mean([a for a, _ in res[(family, lev, s)]])),
                       'abstain_rate': float(np.mean([b for _, b in res[(family, lev, s)]]))} for s in S.SIZES}
            summ[f'{family}/{lev}'] = {'per_size': per, 'mean_action_accuracy': float(np.mean([v['action_accuracy'] for v in per.values()]))}
    heavy = {lev: summ[f'heavy/{lev}']['mean_action_accuracy'] for lev in LEVELS}
    best = max(heavy.values())
    tied = [lev for lev in LEVELS if heavy[lev] >= best - .02]
    chosen = max(tied, key=lambda l: 0. if l == 'point' else l)          # higher level wins ties
    summ['_selection'] = {'rule': 'max mean action accuracy (heavy), ties within .02 -> higher level', 'tied': tied, 'chosen': chosen}
    (out / 'LEVEL_SUMMARY.json').write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ['_selection'], default=str))
    for k, v in summ.items():
        if k != '_selection':
            print(k, round(v['mean_action_accuracy'], 3), {s: (round(x['action_accuracy'], 2), round(x['abstain_rate'], 2)) for s, x in v['per_size'].items()})


if __name__ == '__main__':
    main(Path(sys.argv[1]))
