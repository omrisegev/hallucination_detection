"""Check of the other machine's final-step finding on our backbone cells (2026-11-11).
For each cell: final-step level of the 16 channels (answer-z, mean of channels oriented as TELE16 would be: sign of
pooled corr with TELE16) for correct / wrong final steps; and PRM / TELE16 / C4U quality with and without final steps."""
import numpy as np, pandas as pd
import analyze as A
rows = []
for b in ['prmbench', 'socratic', 'hard2verify', 'processbench', 'crv_arith', 'crv_gsm8k', 'prm800k', 'deltabench_ext']:
    for p in A.PRMS:
        d = A.load(b, p); e, off, k, X = d['err'], d['off'], d['keep'], d['X'].astype(float)
        last = np.zeros(len(e), bool); last[off[1:] - 1] = True
        Z = X[:, 2:]; sg = np.sign([np.corrcoef(Z[:, j], d['tele'])[0, 1] for j in range(16)])
        tel = (Z * sg).mean(1)                                   # label-free orientation from TELE16
        nl = ~last
        r = dict(bench=b, prm=p, n_steps_mean=float(np.diff(off).mean()),
                 last_level_correct=float(tel[last & k & ~e].mean()) if (last & k & ~e).any() else np.nan,
                 last_level_wrong=float(tel[last & k & e].mean()) if (last & k & e).any() else np.nan,
                 last_err_share=float(e[last & k].mean()))
        for a, s in dict(PRM=d['prm'], TELE16=d['tele'], C4U=d['c4u'], ch_mean=tel).items():
            r[f'{a}_within'] = float(np.nanmean(A.within(e, s, off, k)))
            r[f'{a}_within_nolast'] = float(np.nanmean(A.within(e, s, off, k & nl)))
            r[f'{a}_pooled'] = A.auc(e[k], s[k]); r[f'{a}_pooled_nolast'] = A.auc(e[k & nl], s[k & nl])
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv('T10_last_step.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
g = df.groupby('bench', sort=False).mean(numeric_only=True)
print(g[['last_err_share', 'last_level_correct', 'last_level_wrong']].round(2))
for m in ('within', 'pooled'):
    t = pd.DataFrame({f'{a} {s}': g[f'{a}_{m}{s}'] for a in ('PRM', 'TELE16', 'C4U') for s in ('', '_nolast')})
    t['TELE-PRM'] = g[f'TELE16_{m}'] - g[f'PRM_{m}']; t['TELE-PRM nolast'] = g[f'TELE16_{m}_nolast'] - g[f'PRM_{m}_nolast']
    t['C4U-PRM'] = g[f'C4U_{m}'] - g[f'PRM_{m}']; t['C4U-PRM nolast'] = g[f'C4U_{m}_nolast'] - g[f'PRM_{m}_nolast']
    print(m); print(t.round(3))
