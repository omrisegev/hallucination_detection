"""Pre-registered synthetic checks, run before any real data.

(a) No true mixture: 3,000 unimodal heavy-tailed draws, Student-t (3 df) and a skewed log-normal (standard,
    mu 0, sigma 1). Report pi_hat and whether BIC prefers 2 components.
(b) A true mixture: prevalence .65 and .20 (share in the HIGH component = "has error"), separation d = 1 and 2,
    Gaussian and t5 noise (t5 rescaled to unit variance, so d is in noise-sd units). n = 3,000.

The pre-registered row is replicate 0 (seed 20261029). Replicates 1..19 (seeds 20261029 + r) are an added
spread estimate, labelled as such; they do not replace replicate 0.
    python results/answer_mixture_gate_v1/synthetic.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mixture as MX  # noqa: E402

N, REPS = 3000, 20


def draw_a(kind, rng):
    if kind == 't3':
        return rng.standard_t(3, N)
    if kind == 'lognormal':
        return rng.lognormal(0., 1., N)
    raise ValueError(kind)


def draw_b(prev, d, noise, rng):
    y = rng.random(N) < prev
    e = rng.standard_normal(N) if noise == 'gauss' else rng.standard_t(5, N) * np.sqrt(3 / 5)
    return d * y + e, y


def main():
    rows = []
    for rep in range(REPS):
        seed = MX.SEED + rep
        for kind in ('t3', 'lognormal'):
            rng = np.random.default_rng([seed, 1, ['t3', 'lognormal'].index(kind)])
            x = draw_a(kind, rng)
            for tr in (None, 'yeo-johnson'):
                f = MX.fit_mixture(x, tr)
                rows.append(dict(check='a', replicate=rep, seed=seed, data=kind, prevalence='', d='', noise='',
                                 model='raw' if tr is None else 'yeo-johnson', pi_true='', pi_hat=f['pi_hat'],
                                 abs_err='', bic2_minus_bic1=f['bic2_minus_bic1'], bic_prefers_2=f['bic_prefers_2'],
                                 share_called_err=float((f['post_err'] >= .5).mean()), clean_closed='', err_kept='',
                                 converged=f['converged']))
        for prev in (.65, .20):
            for d in (1., 2.):
                for noise in ('gauss', 't5'):
                    rng = np.random.default_rng([seed, 2, int(prev * 100), int(d), ['gauss', 't5'].index(noise)])
                    x, y = draw_b(prev, d, noise, rng)
                    for tr in (None, 'yeo-johnson'):
                        f = MX.fit_mixture(x, tr); s = MX.score(f, y, x)
                        rows.append(dict(check='b', replicate=rep, seed=seed, data='mixture', prevalence=prev, d=d,
                                         noise=noise, model='raw' if tr is None else 'yeo-johnson',
                                         pi_true=s['pi_true'], pi_hat=f['pi_hat'], abs_err=s['abs_err'],
                                         bic2_minus_bic1=f['bic2_minus_bic1'], bic_prefers_2=f['bic_prefers_2'],
                                         share_called_err=float((f['post_err'] >= .5).mean()),
                                         clean_closed=s['clean_closed'], err_kept=s['err_kept'],
                                         converged=f['converged']))
        print('replicate', rep, 'done', flush=True)
    with open(HERE / 'SYNTHETIC.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # summary: replicate 0 (pre-registered) and the 20-replicate spread
    summ = {}
    keys = sorted({(r['check'], r['data'], str(r['prevalence']), str(r['d']), r['noise'], r['model']) for r in rows})
    for k in keys:
        rs = [r for r in rows if (r['check'], r['data'], str(r['prevalence']), str(r['d']), r['noise'], r['model']) == k]
        r0 = [r for r in rs if r['replicate'] == 0][0]; ph = np.array([r['pi_hat'] for r in rs])
        summ['|'.join(k)] = {'rep0_pi_hat': r0['pi_hat'], 'rep0_bic2_minus_bic1': r0['bic2_minus_bic1'],
                             'rep0_bic_prefers_2': r0['bic_prefers_2'], 'rep0_abs_err': r0['abs_err'],
                             'rep0_clean_closed': r0['clean_closed'], 'rep0_err_kept': r0['err_kept'],
                             'reps_pi_hat_median': float(np.median(ph)), 'reps_pi_hat_min': float(ph.min()),
                             'reps_pi_hat_max': float(ph.max()),
                             'reps_bic_prefers_2': int(sum(r['bic_prefers_2'] for r in rs)), 'reps': len(rs)}
    (HERE / 'SYNTHETIC_SUMMARY.json').write_text(json.dumps(summ, indent=1))
    for k, v in summ.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == '__main__':
    main()
