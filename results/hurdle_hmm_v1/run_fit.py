"""Label-free phase. Fits the hurdle HMM per cell and saves outputs. Reads NO label arrays.
Usage: python run_fit.py <bundle_dir> <out_dir> [primary|secondary]"""
import importlib.util, json, pathlib, sys, time, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sp = importlib.util.spec_from_file_location('hh', ROOT / 'spectral_utils' / 'hurdle_hmm.py')
H = importlib.util.module_from_spec(sp); sp.loader.exec_module(H)
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
PRIMARY = [f'AUTO_prm_{b}_{p}' for b in ('processbench', 'crv_arith', 'crv_gsm8k') for p in PRMS] + \
          [f'TELE16_sa_prmbench_{p}' for p in PRMS]
LABEL_FREE_KEYS = ['offsets', 'step_raw_fused', 'step_z_fused']


def gap_counts(s, off):  # verbatim copy of banks_robustness.py:gap_counts
    k = []
    for a, b in zip(off[:-1], off[1:]):
        m = b - a
        if m == 1: k.append(1); continue
        z = np.sort(s[a:b])[::-1]; km = min(int(np.ceil(m / 2)), m - 1)
        k.append(int(np.argmax(z[:km] - z[1:km + 1]) + 1))
    return np.array(k)


def topk_flags(s, off, k):
    f = np.zeros(len(s), bool)
    for a, (lo, hi) in enumerate(zip(off[:-1], off[1:])):
        if k[a] > 0:
            f[lo + np.argsort(-s[lo:hi], kind='stable')[:k[a]]] = True
    return f


def main(bundle, out, which):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    cells = PRIMARY if which == 'primary' else sorted(
        p.stem for p in pathlib.Path(bundle).glob('*.npz') if p.stem not in PRIMARY)
    for c in cells:
        if (out / f'{c}.npz').exists(): continue
        B = np.load(pathlib.Path(bundle) / f'{c}.npz', allow_pickle=False)
        off = B['offsets'].astype(np.int64); x = B['step_raw_fused'].astype(float); z = B['step_z_fused'].astype(float)
        t0 = time.time(); r = H.fit(x, off, q_anchor=0.30, max_fit_answers=2000, seed=0)
        ll, pA, py = H.score(x, off, r)
        k_hh = np.where(pA > .5, np.maximum(1, np.round([py[a:b].sum() for a, b in zip(off[:-1], off[1:])])), 0).astype(int)
        np.savez_compressed(out / f'{c}.npz', pA=pA, py=py, ll=ll,
                            flag_bayes=py > .5, flag_hhk=topk_flags(py, off, k_hh),
                            flag_gap=topk_flags(z, off, gap_counts(z, off)),
                            fit_json=json.dumps({**{k: float(v) for k, v in r['params'].items()}, 'mu0': r['mu0'],
                                                 's0': r['s0'], 'anchor_t': r['anchor_t'], 'nll': float(r['nll']),
                                                 'converged': r['converged'], 'n_fit': r['n_fit']}))
        print(c, json.dumps({k: round(float(v), 4) for k, v in r['params'].items()}),
              f'conv={r["converged"]} secs={time.time() - t0:.0f}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'primary')
