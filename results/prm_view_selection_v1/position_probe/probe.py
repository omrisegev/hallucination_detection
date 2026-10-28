"""Exploratory probe (not pre-registered): remove a cubic in relative step position from the 16 telemetry channels
(pooled least squares on each cell's pool, label-free; the PRM is left untouched), then rerun stage 1's method
unchanged on a few subsets. Compares with the stage-1 representation."""
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'stage1')); sys.argv = sys.argv[:1]
import stage1 as S
E = S.E

def residual_cell(b, p):
    o = S.CellData(b, p); c = o.c
    n = np.diff(c.off); pos = np.concatenate([(np.arange(k) + .5) / k for k in n])
    B = np.column_stack([np.ones_like(pos), pos, pos ** 2, pos ** 3])
    X = o.Z * o.sign[None, :]                                 # raw answer-z (stored orientation undone)
    beta = np.linalg.lstsq(B[o.pool], X[o.pool, 1:], rcond=None)[0]
    X[:, 1:] = X[:, 1:] - B @ beta                            # channels only
    sign = np.sign([1.] + [np.corrcoef(X[o.pool, 0], X[o.pool, k])[0, 1] for k in range(1, 17)]); sign[sign == 0] = 1
    X = X * sign
    o.Z = np.column_stack([E.answer_z(X[:, k], c.off) for k in range(17)]); o.Z[:, 0] = X[:, 0]
    sd = np.array([[X[a:b2, k].std() for k in range(17)] for a, b2 in zip(c.off[:-1], c.off[1:])])
    const = np.repeat(sd < 1e-8, n, axis=0); V = np.empty_like(o.Z)
    for k in range(17):
        f = E.topk_flags(o.Z[:, k], c.off, E.gap_counts(o.Z[:, k], c.off)); f[const[:, k]] = False; V[:, k] = np.where(f, 1., -1.)
    o.V = V; o.Vp = V[o.pool]; o.R = np.cov(np.ascontiguousarray(o.Vp).T); o.sign_res = sign
    return o

NAMES = ['q15_H1','q15_VE1','chosen_surprisal','logprob_margin','true_tail50','energy_level','energy_innovation','top15_turnover',
         'top50_js','dominant_freq16','bocpd_p0','realized_z','realized_drv','digit_alternative','digit_spread','digit_alternative_innovation']
SUBS = {'top9of10': ['q15_VE1','chosen_surprisal','top15_turnover','dominant_freq16','bocpd_p0'],
        'alt8': ['q15_H1','chosen_surprisal','energy_level','top50_js','digit_spread'],
        'all16': NAMES}
rows = []
for b, p in S.CELLS:
    base = S.CellData(b, p); res = residual_cell(b, p); c = base.c
    keep = c.keep if b != 'prmbench' else None
    w = lambda s: np.nanmean(E.per_answer_auc(c.err, s, c.off, keep))
    ab_old = sum(w(base.Z[:, k]) > .5 for k in range(1, 17)); ab_new = sum(w(res.Z[:, k]) > .5 for k in range(1, 17))
    line = f'{b[:5]} {p:14s} PRM {base.ps_prm:6.2f} | flips {int((base.sign[1:]<0).sum()):2d}->{int((res.sign_res[1:]<0).sum()):2d} | ch>.5 {ab_old:2d}->{ab_new:2d}'
    for nm, sub in SUBS.items():
        Sx = [NAMES.index(x) for x in sub]; o0, o1 = base.evaluate(Sx), res.evaluate(Sx)
        g = lambda o: (o.get('ps_fused', np.nan), o.get('ps_mean', np.nan), o.get('w_prm_share', np.nan), o['K'])
        a, z = g(o0), g(o1)
        line += f' | {nm}: fused {a[0]:.2f}->{z[0]:.2f} (mean {a[1]:.2f}->{z[1]:.2f}, wPRM {a[2]:.2f}->{z[2]:.2f}, K {a[3]}->{z[3]})'
        rows.append(dict(cell=f'{b}/{p}', subset=nm, prm=base.ps_prm, fused_base=a[0], fused_res=z[0], mean_base=a[1], mean_res=z[1],
                         wprm_base=a[2], wprm_res=z[2], K_base=a[3], K_res=z[3]))
    print(line, flush=True)
E.write_csv(HERE / 'PROBE.csv', rows)
