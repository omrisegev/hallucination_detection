"""Choose ONE label-free gate estimator on synthetic data only, before any real-data gate is computed (plan Stage A).

Selection rule, fixed before this script was run (also in GATE_PREREG.md):
  - Data: random regimes of the latent-class model (FRAMING_THEORY.md, model M) with answer effects, a step-position
    covariate that also loads on the PRM, a telemetry-wide latent nuisance, and heavy-tailed noise (centered
    exponential). Answer counts and lengths match Hard2Verify (~180 x 9), Socratic (~3,000 x 9) and PRMBench
    (~7,000 x 13.5).
  - Fusion being gated: the shipped CHAIN cross stage (lsml_continuous over singleton groups of answer-z views,
    anchor flip toward the PRM), exactly as in results/prm_boost_ablation5_v1/ablation5.py.
  - Truth (uses labels, simulation only): sign of mean within-answer AUC(CHAIN) - AUC(PRM).
  - Gate: fusion_theory.gate_decision on the answer-z views residualized on a cubic position basis within answers.
    'fuse' -> use CHAIN; 'prm' or 'abstain' -> use the PRM.
  - Score per method: ACTION ACCURACY = share of regimes where the chosen arm is the better one, averaged with equal
    weight over the three sizes. Primary family: heavy-tailed noise. Ties (within 0.02) are broken by the lower
    abstain rate at the PRMBench size. The 'rank' method has no bootstrap (too slow); it decides on the point
    estimate when its profile is not flat, and is reported for comparison.
  - Robustness rows (reported, not used for selection): Gaussian noise; noise shared by PRM and telemetry.

    python select_gate_estimator.py <out_dir> [--quick]
"""
import json, os, sys, time, zlib
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests'))
from lever_imports import ensure_spectral_package  # noqa: E402

ensure_spectral_package()
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402
from spectral_utils.fusion_theory import covariate_basis, gate_decision, within_answer_residualize  # noqa: E402

SIZES = {'hard2verify': (180, 9), 'socratic': (3000, 9), 'prmbench': (7000, 13)}
METHODS = ('cumulant', 'cumulant_w01', 'fa', 'cca', 'rank')


def answer_z(x, off):
    out = np.empty_like(x)
    for a, b in zip(off[:-1], off[1:]):
        v = x[a:b]; out[a:b] = (v - v.mean(0)) / np.maximum(v.std(0), 1e-8)
    return out


def within_auc(y, s, off):
    vals = []
    for a, b in zip(off[:-1], off[1:]):
        yy, ss = y[a:b], s[a:b]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 and n0:
            r = np.argsort(np.argsort(ss, kind='stable'), kind='stable') + 1
            vals.append((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(vals))


def regime(rng, J, nbar, noise, shared):
    K = int(rng.integers(4, 7)); m = K + 1
    n = np.clip(rng.poisson(nbar, J), 3, None); off = np.r_[0, np.cumsum(n)]; N = int(off[-1])
    pos = np.concatenate([(np.arange(k) + .5) / k for k in n])
    late = rng.random() < .5                                          # errors drift later in half of the regimes
    p0 = rng.uniform(.1, .2)
    prob = np.clip(p0 * (0.4 + 1.2 * pos) if late else np.full(N, p0), .01, .9)
    y = (rng.random(N) < prob).astype(float)
    D = np.r_[rng.uniform(.3, 2.), rng.uniform(.1, 1., K)]
    lam = np.r_[0., rng.uniform(0, .9, K)]                            # latent nuisance: telemetry only
    gam = rng.uniform(-.6, .6, m)                                     # position effect, PRM included
    g = rng.uniform(-1, 1, m)                                         # answer effect
    A = np.repeat(rng.standard_normal(J), n); eta = rng.standard_normal(N)
    eps = (rng.exponential(1., (N, m)) - 1.) if noise == 'exp' else rng.standard_normal((N, m))
    X = A[:, None] * g + y[:, None] * D + (pos[:, None] - .5) * gam + eta[:, None] * lam + eps
    if shared:
        sh = (rng.exponential(1., N) - 1.) if noise == 'exp' else rng.standard_normal(N)
        X += sh[:, None] * np.r_[.5, rng.uniform(0, .5, K)]
    return X, y, off, pos


def chain_weights(Xz):
    s, d = lsml_continuous(*[Xz[:, j] for j in range(Xz.shape[1])], groups=np.arange(Xz.shape[1]))
    w = np.asarray(d['cross_weights'], float)
    if np.corrcoef(np.asarray(s, float), Xz[:, 0])[0, 1] < 0:
        w = -w
    return w


def run(out, quick=False):
    out.mkdir(parents=True, exist_ok=True)
    plan = {'hard2verify': 30, 'socratic': 30, 'prmbench': 16}
    B = {'hard2verify': 100, 'socratic': 100, 'prmbench': 60}
    if quick:
        plan = {k: 3 for k in plan}; B = {k: 20 for k in B}
    rows = []
    t0 = time.time()
    for family, (noise, shared) in {'heavy': ('exp', False), 'gauss': ('gauss', False), 'shared_heavy': ('exp', True)}.items():
        for size, (J, nbar) in SIZES.items():
            for r in range(plan[size]):
                seed = zlib.crc32(f'{family}/{size}/{r}'.encode())   # deterministic across processes
                rng = np.random.default_rng(seed)
                X, y, off, pos = regime(rng, J, nbar, noise, shared)
                Xz = answer_z(X, off)
                w = chain_weights(Xz)
                truth = within_auc(y, Xz @ w, off) - within_auc(y, Xz[:, 0], off)
                Xe = within_answer_residualize(Xz, covariate_basis(pos), off)
                for meth in METHODS:
                    dec = gate_decision(Xe, off, w, method=meth, B=0 if meth == 'rank' else B[size], seed=seed)
                    action = 'fuse' if dec['decision'] == 'fuse' else 'prm'
                    rows.append({'family': family, 'size': size, 'regime': r, 'method': meth, 'truth': truth,
                                 'decision': dec['decision'], 'correct': (action == 'fuse') == (truth > 0),
                                 'G': dec.get('G')})
            print(family, size, 'done', round(time.time() - t0), 's', flush=True)
    import csv
    with open(out / 'REGIMES.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
    summ = {}
    for family in ('heavy', 'gauss', 'shared_heavy'):
        for meth in METHODS:
            per = {}
            for size in SIZES:
                rr = [x for x in rows if x['family'] == family and x['method'] == meth and x['size'] == size]
                dec = [x for x in rr if x['decision'] in ('fuse', 'prm')]
                per[size] = {'n': len(rr), 'action_accuracy': float(np.mean([x['correct'] for x in rr])),
                             'abstain_rate': float(np.mean([x['decision'] == 'abstain' for x in rr])),
                             'accuracy_when_deciding': float(np.mean([x['correct'] for x in dec])) if dec else None,
                             'always_fuse_accuracy': float(np.mean([x['truth'] > 0 for x in rr]))}
            summ[f'{family}/{meth}'] = {'per_size': per,
                                        'mean_action_accuracy': float(np.mean([per[s]['action_accuracy'] for s in SIZES]))}
    heavy = {m: summ[f'heavy/{m}'] for m in METHODS if m != 'rank'}
    best = max(heavy.values(), key=lambda v: v['mean_action_accuracy'])['mean_action_accuracy']
    tied = [m for m, v in heavy.items() if v['mean_action_accuracy'] >= best - .02]
    chosen = min(tied, key=lambda m: heavy[m]['per_size']['prmbench']['abstain_rate'])
    summ['_selection'] = {'rule': 'max mean action accuracy (heavy-tailed), ties within .02 -> lowest PRMBench-size abstain',
                          'tied': tied, 'chosen': chosen, 'quick': quick, 'seconds': round(time.time() - t0)}
    (out / 'SUMMARY.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ['_selection']), flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]), quick='--quick' in sys.argv)
