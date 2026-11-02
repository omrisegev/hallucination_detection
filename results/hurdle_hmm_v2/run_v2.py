"""v2 run. Per primary cell and repetition: split, calibrate on m labelled pool answers, fit, score, controls.
Reads labels ONLY of the m labelled answers (calibration). Test-half labels are read in evaluate_v2.py.
Usage: python run_v2.py <bundle_dir> <out_dir>"""
import json, pathlib, sys, time, numpy as np
from common import H, controls
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
PRIMARY = [f'AUTO_prm_{b}_{p}' for b in ('processbench', 'crv_arith', 'crv_gsm8k') for p in PRMS] + \
          [f'TELE16_sa_prmbench_{p}' for p in PRMS]
M_LAB, R = 200, 5


def main(bundle, out):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    for c in PRIMARY:
        B = np.load(pathlib.Path(bundle) / f'{c}.npz', allow_pickle=False)
        off = B['offsets'].astype(np.int64); n = np.diff(off); A = len(n)
        x = B['step_raw_fused'].astype(float); z = B['step_z_fused'].astype(float)
        for r in range(R):
            fp = out / f'{c}__r{r}.npz'
            if fp.exists(): continue
            rng = np.random.default_rng(1000 + r); perm = rng.permutation(A)
            pool, test = np.sort(perm[:A // 2]), np.sort(perm[A // 2:])
            lab = np.sort(rng.choice(pool, M_LAB, replace=False))
            # --- labels of the labelled answers only ---
            ha = B['ans_has_err_LABEL'][lab].astype(bool)
            st = np.concatenate([np.arange(off[a], off[a + 1]) for a in lab])
            st = st[B['step_keep_eval'][st].astype(bool)]
            ys = B['step_err_LABEL'][st].astype(bool)
            pi = float(np.clip(ha.mean(), 1 / (2 * M_LAB), 1 - 1 / (2 * M_LAB)))
            cs = x[st][~ys]; s_share = float(ys.mean())
            y_lab_only = np.zeros(len(x), bool); y_lab_only[st] = ys          # labels outside st are never used
            t0 = time.time()
            fit = H.fit(x, off, anchor=(float(cs.mean()), float(cs.std())), fixed_pi=pi, max_fit_answers=2000, seed=r)
            ll, pA, py = H.score(x, off, fit)
            ctl = controls(x, z, off, st, y_lab_only, pi, s_share)
            k_hh = np.where(pA > .5, np.maximum(1, np.round([py[a:b].sum() for a, b in zip(off[:-1], off[1:])])), 0).astype(int)
            from common import topk_flags
            th = fit['theta']
            at_bound = bool(np.any(np.isclose(th[1:4], -8) | np.isclose(th[1:4], 8)) or np.any(np.isclose(th[4:6], -4) | np.isclose(th[4:6], 3)) or np.isclose(th[6], -6) or np.isclose(th[6], 6))
            np.savez_compressed(fp, test=test, lab=lab, pA=pA, py=py, flag_HH_BAYES=py > .5, flag_HH_K=topk_flags(py, off, k_hh),
                                **{f'flag_{k}': v for k, v in ctl.items()},
                                fit_json=json.dumps({**{k: float(v) for k, v in fit['params'].items()}, 'mu0': fit['mu0'],
                                                     's0': fit['s0'], 'pi_cal': pi, 's_share': s_share, 'nll': float(fit['nll']),
                                                     'converged': fit['converged'], 'at_bound': at_bound}))
            print(c, r, f"pi={pi:.3f} delta={fit['params']['delta']:.3f} rho={fit['params']['rho']:.3f} f={fit['params']['f']:.3f} bound={at_bound} {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
