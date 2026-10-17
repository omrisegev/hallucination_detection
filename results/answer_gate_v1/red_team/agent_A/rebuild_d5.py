"""Item 5: rebuild D5 (epr) from (a) ANSWER_FEATURES 'epr' and (b) raw q15_H1 tokens; compare with A_D5_epr.
Also rebuild D6 (+trace_length, fixed direction) as a second single-feature check."""
import numpy as np
from common import load_all, FEAT, ROOT

df, off, lab, d, bymeta = load_all()
cells = df.cell.values; folds = df.fold.values
f = np.load(FEAT, allow_pickle=True)
names = list(f["names"]); X = f["X"]
assert list(f["ids"]) == list(df.id.values)

# (b) raw token entropy mean per answer
TOK = ROOT + "/.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz"
z = np.load(TOK, mmap_mode="r")
ch = list(z["channels"]); to = z["token_offsets"]
H = np.asarray(z["tokens"][:, ch.index("q15_H1")], dtype=np.float64)
epr_raw = np.array([H[to[i]:to[i + 1]].mean() for i in range(len(df))])
epr_feat = X[:, names.index("epr")]
print("epr feature vs raw q15_H1 answer mean: max abs diff =", float(np.max(np.abs(epr_raw - epr_feat))))
tl_raw = np.diff(to).astype(float)
print("trace_length feature vs raw token count: max abs diff =",
      float(np.max(np.abs(tl_raw - X[:, names.index("trace_length")]))))


def zrebuild(v, ddof, cal_excluded=True):
    out = np.full(len(v), np.nan)
    for c in np.unique(cells):
        for k in range(5):
            cal = (k + 1) % 5
            if cal_excluded:
                fit = (cells == c) & (folds != k) & (folds != cal)
            else:
                fit = (cells == c) & (folds != k)
            ev = (cells == c) & (folds == k)
            mu = v[fit].mean(); sd = v[fit].std(ddof=ddof)
            out[ev] = (v[ev] - mu) / sd
    return out


for label, v in [("epr feature", epr_feat), ("epr raw tokens", epr_raw)]:
    for ddof in (0, 1):
        for cx in (True, False):
            r = zrebuild(v, ddof, cx)
            dd = np.abs(r - d["A_D5_epr"].astype(np.float64))
            print(f"D5 from {label:15s} ddof={ddof} fit={'excl k,(k+1)%5' if cx else 'excl k only   '}: "
                  f"max|diff|={dd.max():.3e} mean|diff|={dd.mean():.3e}")
for ddof in (0, 1):
    r = zrebuild(X[:, names.index("trace_length")], ddof)
    print(f"D6 (+trace_length) ddof={ddof}: max|diff| vs A_D6_length = "
          f"{np.max(np.abs(r - d['A_D6_length'].astype(np.float64))):.3e}")
