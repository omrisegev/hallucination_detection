import numpy as np, pandas as pd
def q(x, a): x = x[np.isfinite(x)]; return [float(np.quantile(x, a / 2)), float(np.quantile(x, 1 - a / 2))]
rng4 = np.random.default_rng(9)
for trial in range(300):
    m = 5; mus = rng4.normal(0, 2.5, m); D = {f'R{j}': rng4.normal(mus[j], 1, 4000) for j in range(m)}
    prim = pd.DataFrame([{'contrast': f'R{j}', 'p': float(min(1, 2 * min((D[f'R{j}'] <= 0).mean(), (D[f'R{j}'] >= 0).mean())))} for j in range(m)]).sort_values('p')
    still = True; still2 = True
    for r_, (_, row) in enumerate(prim.iterrows()):
        a = .05 / (m - r_); lo, hi = q(D[row.contrast], a); rej = still and (lo > 0 or hi < 0); still = rej
        rej2 = still2 and row.p < a; still2 = rej2
        if rej != rej2: print(trial, row.contrast, 'p', row.p, 'alpha', a, 'ci', [round(lo, 4), round(hi, 4)], 'ci_rej', rej, 'p_rej', rej2)
