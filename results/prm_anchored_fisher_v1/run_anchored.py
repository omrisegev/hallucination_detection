"""PRM-anchored Fisher fusion (P5) on all local cells (docs/experiments/PRM_ANCHORED_FISHER_PREREG_2026-10-28.md).

    python run_anchored.py            # 15 cells -> RESULTS.json, RESULTS.csv, PARITY.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'results/prm_boost_math_rigor_v1')); sys.path.insert(0, str(ROOT / 'results/prm_view_selection_v1/stage1'))
import common as MR  # noqa: E402
import stage1 as S1  # noqa: E402
from collection_parity import h2v_common_uids  # noqa: E402
from spectral_utils import prm_anchored_fisher as PAF  # noqa: E402
from spectral_utils.fusion_theory import covariate_basis, within_answer_residualize  # noqa: E402

E = MR.E
BENCHES = ['prmbench', 'socratic', 'hard2verify']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
ARMS = [('ANCH_AUTO', 'auto', 'prm', False), ('ANCH_RANK', 'rank', 'prm', False), ('ANCH_CUM', 'cumulant', 'prm', False),
        ('ANCH_CUMW', 'cumulant_w01', 'prm', False), ('ANCH_CCA', 'cca', 'prm', False),
        ('ANCH_AUTO_G', 'auto', 'prm', True), ('ANCH_AUTO_FBC', 'auto', 'chain', False)]
B, SEED = 2000, 20261028
COMMON = h2v_common_uids()


def gapf(c, s):
    z = E.answer_z(s, c.off)
    return E.topk_flags(z, c.off, E.gap_counts(z, c.off))


def official(c, f):
    return float(E.official(c, f)[0]) * (1 if c.bench == 'prmbench' else 100)


def within(c, s):
    return float(np.nanmean(E.per_answer_auc(c.err, s, c.off, c.keep if c.bench != 'prmbench' else None)))


class Boot:
    """Paired bootstrap of the official metric from per-unit confusion counts (PRMBench: source groups; external:
    answers)."""

    def __init__(self, c):
        self.c = c
        u, self.inv = np.unique(np.asarray(c.groups).astype(str), return_inverse=True)
        self.W = np.random.default_rng(SEED).multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(float)
        self.nu = len(u); self.k = 1 if c.bench == 'prmbench' else 100

    def units(self, f):
        U = np.zeros((self.nu, 4)); np.add.at(U, self.inv, E.counts_by_answer(self.c, f).astype(float)); return U

    def diff(self, fa, fb):
        d = self.k * (E.official_from_counts(self.W @ self.units(fa), self.c.bench)
                      - E.official_from_counts(self.W @ self.units(fb), self.c.bench))
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def oracle_fisher(c, VC):
    """Labels used, in-sample: LDA direction on the cross-stage views (headroom only)."""
    m = np.ones(len(VC), bool) if c.bench == 'prmbench' else c.keep
    y = c.err[m]; X = VC[m]
    X1, X0 = X[y], X[~y]
    Sw = (np.cov(X1.T) * (len(X1) - 1) + np.cov(X0.T) * (len(X0) - 1)) / (len(X) - 2)
    return VC @ np.linalg.solve(Sw, X1.mean(0) - X0.mean(0))


def run_cell(bench, prm):
    t0 = time.time()
    c, VC, w_chain, fused_chain, part = MR.chain_inputs(bench, prm)
    VC2, _ = PAF.chain_views(c.arms[MR.prm_arm(c)], E.channels(c, 'N16'), c.off)
    view_parity = float(np.abs(VC2 - VC).max())
    if view_parity > 1e-9:
        raise SystemExit(f'VIEW PARITY FAIL {bench}/{prm}: {view_parity}')
    pos, loglen = MR.covariates(c)
    Xe = within_answer_residualize(VC, covariate_basis(pos, loglen), c.off)
    scores, diag = {'CHAIN_N16': fused_chain, 'ORACLE_FISHER': oracle_fisher(c, VC)}, {}
    for name, method, fb, gate in ARMS:
        w, inf = PAF.anchored_fisher_weights(Xe, c.off, method)
        use = w is not None and (not gate or inf.get('G', -1) > 0)
        scores[name] = VC @ w if use else (VC[:, 0] if fb == 'prm' else fused_chain)
        diag[name] = dict(fused=bool(use), route=inf.get('route', ''), reason=inf.get('reason', '' if use else 'gate G <= 0'),
                          excluded=inf.get('excluded_groups', []), prm_share=inf.get('prm_share'), G=inf.get('G'),
                          n_neg_w=inf.get('n_negative_weights'), w=None if w is None else np.round(w, 5).tolist())
    if bench == 'hard2verify':
        mask, _ = MR.eval_rows(c, bench)
        ce = E.load(bench, prm, uids_keep=COMMON)
        scores = {k: v[mask] for k, v in scores.items()}
    else:
        ce = c
    fP = gapf(ce, ce.arms[S1.prm_arm(bench, prm)])
    bt = Boot(ce)
    flags = {k: gapf(ce, v) for k, v in scores.items()}
    chain_ps = official(ce, flags['CHAIN_N16']); status = float(S1.STATUS[(bench, prm, 'CHAIN_N16')]['GAP'])
    parity = dict(cell=f'{bench}/{prm}', view_max_diff=view_parity, chain=chain_ps, status=status, diff=abs(chain_ps - status))
    if parity['diff'] > 1e-6:
        raise SystemExit(f'CHAIN PARITY FAIL {parity}')
    base = dict(bench=bench, prm=prm, K=part['K'], prm_ps=official(ce, fP), prm_within=within(ce, ce.arms[S1.prm_arm(bench, prm)]),
                chain_ps=chain_ps, chain_within=within(ce, scores['CHAIN_N16']),
                tele_ps=float(S1.STATUS[(bench, prm, 'TELE_N16')]['GAP']),
                chain_prm_share=float(abs(w_chain[0]) / np.abs(w_chain).sum()))
    rows = []
    for name in [a[0] for a in ARMS] + ['ORACLE_FISHER']:
        r = dict(base, arm=name, ps=official(ce, flags[name]), within=within(ce, scores[name]),
                 ci_vs_chain=bt.diff(flags[name], flags['CHAIN_N16']), ci_vs_prm=bt.diff(flags[name], fP), **diag.get(name, {}))
        r['d_vs_chain'] = r['ps'] - chain_ps; r['d_vs_prm'] = r['ps'] - base['prm_ps']
        rows.append(r)
    print(time.strftime('%H:%M:%S'), f'{bench}/{prm}', f'{time.time() - t0:.0f}s', 'PRM %.2f CHAIN %.2f' % (base['prm_ps'], chain_ps),
          {r['arm']: (round(r['ps'], 2), r.get('fused', ''), r.get('route', '')) for r in rows}, flush=True)
    return rows, parity


def main():
    rows, par = [], []
    for bench in BENCHES:
        for prm in PRMS:
            r, p = run_cell(bench, prm); rows += r; par.append(p)
            (HERE / 'RESULTS.json').write_text(json.dumps(rows, indent=1, default=float))
            (HERE / 'PARITY.json').write_text(json.dumps(par, indent=1))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(HERE / 'RESULTS.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
    print('done')


if __name__ == '__main__':
    main()
