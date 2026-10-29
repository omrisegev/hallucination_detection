"""Independent synthetic checks (own code, own seeds; draws differ from SYNTHETIC.csv, compare qualitatively).
(a) unimodal: Student-t 3 df, standard log-normal; (b) mixtures: prevalence .65/.20 x d 1/2 x Gaussian / unit-var t5.
n = 3000; replicates 0..4 (verifier seed base 777). Primary model (raw) and Yeo-Johnson secondary.
    .venv/bin/python -I results/answer_mixture_gate_v1/verification/verify_synth.py
"""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

OUT = Path('/Users/osegev/Desktop/hallucination_detection/results/answer_mixture_gate_v1/verification')
N, REPS, SEED = 3000, 5, 20261029


def fit(x, yj):
    if yj: x = PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(x[:, None])[:, 0]
    X = x[:, None]
    g2 = GaussianMixture(2, covariance_type='full', n_init=10, random_state=SEED).fit(X)
    g1 = GaussianMixture(1, covariance_type='full', n_init=10, random_state=SEED).fit(X)
    k = int(np.argmax(g2.means_.ravel()))
    return float(g2.weights_[k]), float(g2.bic(X) - g1.bic(X)), g2.predict_proba(X)[:, k]


rows = []
for rep in range(REPS):
    rng = np.random.default_rng(777 + rep)
    for kind in ('t3', 'lognormal'):
        x = rng.standard_t(3, N) if kind == 't3' else np.exp(rng.standard_normal(N))
        for yj in (False, True):
            p, db, _ = fit(x, yj)
            rows.append(dict(check='a', rep=rep, data=kind, prev='', d='', noise='', model='yj' if yj else 'raw',
                             pi_hat=p, dbic=db, bic_prefers_2=db < 0, abs_err=''))
    for prev in (.65, .20):
        for d in (1., 2.):
            for noise in ('gauss', 't5'):
                y = rng.uniform(size=N) < prev
                e = rng.standard_normal(N) if noise == 'gauss' else rng.standard_t(5, N) / np.sqrt(5 / 3)
                x = d * y + e
                for yj in (False, True):
                    p, db, post = fit(x, yj)
                    rows.append(dict(check='b', rep=rep, data='mixture', prev=prev, d=d, noise=noise,
                                     model='yj' if yj else 'raw', pi_hat=p, dbic=db, bic_prefers_2=db < 0,
                                     abs_err=abs(p - y.mean())))
with open(OUT / 'VERIFY_SYNTHETIC.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

ref = json.loads((OUT.parent / 'SYNTHETIC_SUMMARY.json').read_text())
summ = {}
for r in rows:
    key = '|'.join([r['check'], r['data'], str(r['prev']), str(r['d']), r['noise'], 'raw' if r['model'] == 'raw' else 'yeo-johnson'])
    summ.setdefault(key, []).append(r)
out = {}
for key, rs in summ.items():
    ph = np.array([r['pi_hat'] for r in rs]); r0 = [r for r in rs if r['rep'] == 0][0]
    rr = ref.get(key, {})
    out[key] = {'ver_rep0_pi_hat': r0['pi_hat'], 'ver_rep0_bic_prefers_2': bool(r0['bic_prefers_2']),
                'ver_rep0_abs_err': r0['abs_err'], 'ver_pi_hat_range_5reps': [float(ph.min()), float(ph.max())],
                'ver_bic_prefers_2_of_5': int(sum(r['bic_prefers_2'] for r in rs)),
                'ref_rep0_pi_hat': rr.get('rep0_pi_hat'), 'ref_rep0_bic_prefers_2': rr.get('rep0_bic_prefers_2'),
                'ref_rep0_abs_err': rr.get('rep0_abs_err'),
                'ref_pi_hat_range_20reps': [rr.get('reps_pi_hat_min'), rr.get('reps_pi_hat_max')],
                'ref_bic_prefers_2_of_20': rr.get('reps_bic_prefers_2')}
    o = out[key]
    print(f"{key:40s} ver rep0 pi_hat={o['ver_rep0_pi_hat']:.3f} bic2={o['ver_rep0_bic_prefers_2']!s:5s} "
          f"5rep range=[{o['ver_pi_hat_range_5reps'][0]:.3f},{o['ver_pi_hat_range_5reps'][1]:.3f}] bic2 {o['ver_bic_prefers_2_of_5']}/5 | "
          f"ref rep0 {o['ref_rep0_pi_hat']:.3f} bic2={o['ref_rep0_bic_prefers_2']!s:5s} 20rep [{o['ref_pi_hat_range_20reps'][0]:.3f},"
          f"{o['ref_pi_hat_range_20reps'][1]:.3f}] bic2 {o['ref_bic_prefers_2_of_20']}/20")
(OUT / 'verify_synth_summary.json').write_text(json.dumps(out, indent=1))
