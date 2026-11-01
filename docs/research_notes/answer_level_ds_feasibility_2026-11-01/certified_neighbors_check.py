"""Post hoc (Omri idea): a step that is NOT certified clean (raw score >= q-quantile) but whose two neighbours
ARE certified clean (< q-quantile). Is it less often an error? Steps with two neighbours only; both neighbours
and the step must be labelled (step_mask). Also the same split for the top-scored steps (above q80).
Labels only for evaluation."""
import glob, os, sys, numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.npz'))):
    B = np.load(f); off = B['offsets']
    bench = [b for b in ('processbench', 'crv_arith', 'crv_gsm8k', 'prmbench') if b in f][0]
    s = B['step_raw_fused'].astype(float); y = B['step_err_LABEL'].astype(bool); m = B['step_mask'] & np.isfinite(s)
    for q in (0.3, 0.5):
        t = np.nanquantile(s[np.isfinite(s)], q); hi = np.nanquantile(s[np.isfinite(s)], 0.8)
        cert = s < t
        acc = {}
        for a in range(len(off) - 1):
            for i in range(off[a] + 1, off[a + 1] - 1):
                if not (m[i] and m[i - 1] and m[i + 1]) or cert[i]: continue
                k = int(cert[i - 1]) + int(cert[i + 1]); top = s[i] >= hi
                for key in ((k, 'all'), (k, 'top20') if top else None):
                    if key is None: continue
                    acc.setdefault(key, []).append(y[i])
        for (k, band), v in acc.items():
            rows.append(dict(bench=bench, cell=os.path.basename(f)[:-4], q=q, band=band, n_cert_neighbours=k,
                             n=len(v), err_rate=np.mean(v)))
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False)
g = d.groupby(['bench', 'q', 'band', 'n_cert_neighbours']).agg(err=('err_rate', 'median'), n=('n', 'median')).unstack('n_cert_neighbours')
pd.set_option('display.width', 250); print(g.round(3).to_string())
