"""Post hoc (Omri idea 2026-11-01): instead of a threshold that flags errors, a one-sided threshold that
certifies "clean". Score = raw pooled-z fused score (no within-answer normalization).
Answer level: answer max, log(n) removed across answers. Step level: step raw score.
For the bottom q of the score (label-free cut), report purity (share clean) vs the base clean rate,
and the share of all clean units recovered. Labels only for evaluation. Medians over cells."""
import glob, os, sys, numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.npz'))):
    B = np.load(f); off = B['offsets']; n = np.diff(off).astype(float)
    bench = [b for b in ('processbench', 'crv_arith', 'crv_gsm8k', 'prmbench') if b in f][0]
    s = B['ans_raw_fused_answer_max'].astype(float); ok = np.isfinite(s)
    Dm = np.column_stack([np.ones(ok.sum()), np.log(n[ok])])
    r = np.full_like(s, np.nan); r[ok] = s[ok] - Dm @ np.linalg.lstsq(Dm, s[ok], rcond=None)[0]
    ev = B['ans_evaluated'] & ok; clean = ~B['ans_has_err_LABEL'].astype(bool)
    st = B['step_raw_fused'].astype(float); m = B['step_mask'] & np.isfinite(st); sc = ~B['step_err_LABEL'].astype(bool)
    for q in (0.05, 0.10, 0.20, 0.30, 0.50):
        t = np.quantile(r[ok], q); sel = ev & (r < t)
        ts = np.quantile(st[m], q); ss = m & (st < ts)
        rows.append(dict(bench=bench, cell=os.path.basename(f)[:-4], q=q,
            ans_base_clean=clean[ev].mean(), ans_purity=clean[sel].mean() if sel.any() else np.nan,
            ans_clean_recovered=(sel & clean).sum() / max((ev & clean).sum(), 1),
            step_base_clean=sc[m].mean(), step_purity=sc[ss].mean(),
            step_clean_recovered=(ss & sc).sum() / (m & sc).sum()))
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False)
pd.set_option('display.width', 250)
print(d.groupby(['bench', 'q'])[['ans_base_clean', 'ans_purity', 'ans_clean_recovered', 'step_base_clean', 'step_purity', 'step_clean_recovered']].median().round(3).to_string())
