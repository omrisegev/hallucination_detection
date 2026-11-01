"""Post hoc: CROP-style certified prefix with a LABEL-FREE threshold. lambda = q-quantile of all raw step
scores (pooled, no within-answer normalization). Keep each answer's steps until the first step with
score > lambda. risk = share of answers whose kept prefix contains a labelled error step; kept = mean share
of steps kept. Also the mode-based prevalence idea: pi0_hat = 2 * F(mode) (clean null symmetric, errors
to the right). Labels only for evaluation."""
import glob, os, sys, numpy as np, pandas as pd
from scipy.stats import gaussian_kde
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.npz'))):
    B = np.load(f); off = B['offsets']
    bench = [b for b in ('processbench', 'crv_arith', 'crv_gsm8k', 'prmbench') if b in f][0]
    s = B['step_raw_fused'].astype(float); y = B['step_err_LABEL'].astype(bool); ev = B['ans_evaluated']
    fin = np.isfinite(s); g = np.linspace(np.nanquantile(s, .01), np.nanquantile(s, .99), 512)
    mode = g[np.argmax(gaussian_kde(s[fin][::max(1, fin.sum() // 20000)])(g))]
    pi0_hat = min(1.0, 2 * (s[fin] < mode).mean())
    lab = B['step_mask'] & fin
    true_pi0 = (~y[lab]).mean()
    for q in (0.3, 0.5, 0.7):
        lam = np.quantile(s[fin], q); risk = []; kept = []
        for a in range(len(off) - 1):
            if not ev[a]: continue
            ss = s[off[a]:off[a + 1]]; yy = y[off[a]:off[a + 1]]
            over = np.where(~(ss <= lam))[0]; M = over[0] if len(over) else len(ss)
            risk.append(yy[:M].any()); kept.append(M / len(ss))
        rows.append(dict(bench=bench, cell=os.path.basename(f)[:-4], q=q, risk=np.mean(risk), kept=np.mean(kept),
                         pi0_hat_mode=pi0_hat, true_step_clean=true_pi0))
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False)
print(d.groupby(['bench', 'q'])[['risk', 'kept', 'pi0_hat_mode', 'true_step_clean']].median().round(3).to_string())
