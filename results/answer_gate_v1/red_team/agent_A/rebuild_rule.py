"""Extra: rebuild R0 and OFFSET_D5 flags from the frozen step score S_equal + my own D5 rebuild,
and check that every OFFSET rule's flags are a top-n of the frozen within-answer ranking of S."""
import json
import numpy as np
from common import load_all, FEAT, ROOT, PRMCELL

ER = ROOT + "/.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1/"
df, off, lab, d, bymeta = load_all()
cells = df.cell.values; folds = df.fold.values
n_steps = np.diff(off)
S = np.load(ER + "run_20260927_stage_b/STEP_SCORES.npz")
assert np.array_equal(S["offsets"], off)
S = S["S_equal"]
thr = json.load(open(ER + "run_20260927_stage_b_thr/THRESHOLDS.json"))["S_equal"]

ans = np.repeat(np.arange(len(df)), n_steps)
step_fold = folds[ans]
is_prm_step = cells[ans] == PRMCELL


def answer_z(v, ddof=0):
    out = np.empty_like(v)
    for i in range(len(df)):
        x = v[off[i]:off[i + 1]]
        sd = x.std(ddof=ddof)
        out[off[i]:off[i + 1]] = (x - x.mean()) / sd if sd > 0 else 0.0
    return out


# is S_equal already answer-standardized?
az0 = answer_z(S, 0)
print("S_equal already answer-z (ddof0)? max|S - az(S)| =", float(np.nanmax(np.abs(S - az0))))
multi = n_steps[ans] > 1

for name, Z in [("S raw", S), ("answer-z ddof0", az0), ("answer-z ddof1", answer_z(S, 1))]:
    r0 = np.array([Z[j] >= thr[str(step_fold[j])] for j in range(len(S))])
    print(f"R0 rebuild using {name}: mismatches vs R0_frozen = {int((r0 != d['R0_frozen']).sum())}")

f = np.load(FEAT, allow_pickle=True)
epr = f["X"][:, list(f["names"]).index("epr")]


def zfold(v, c, k):
    cal = (k + 1) % 5
    fit = (cells == c) & (folds != k) & (folds != cal)
    return fit, v[fit].mean(), v[fit].std()


best = None
for zname, Z in [("answer-z ddof0", az0), ("S raw", S)]:
    flags = np.zeros(len(S), bool)
    for k in range(5):
        cal = (k + 1) % 5
        zA_step = np.full(len(S), np.nan)  # zA under fold-k model, for fold-k and cal-fold answers
        for c in np.unique(cells):
            _, mu, sd = zfold(epr, c, k)
            sel = (cells == c) & ((folds == k) | (folds == cal))
            idx = np.where(sel)[0]
            for i in idx:
                zA_step[off[i]:off[i + 1]] = (epr[i] - mu) / sd
        D = zA_step + Z
        for bench in (True, False):
            calmask = (step_fold == cal) & (is_prm_step == bench)
            tau = np.quantile(D[calmask], 0.8)
            ev = (step_fold == k) & (is_prm_step == bench)
            flags[ev] = D[ev] >= tau
    mm = int((flags != d["OFFSET_D5_epr"]).sum())
    print(f"OFFSET_D5 rebuild with {zname}: step mismatches vs saved = {mm} of {len(S)}")

# top-n consistency for every rule: flagged set equals the n highest-S steps (ties earlier-first)
for r in [k for k in d.files if k.startswith(("OFFSET", "R0", "R2"))]:
    fl = d[r]
    bad = 0
    for i in range(len(df)):
        s = S[off[i]:off[i + 1]]; f_ = fl[off[i]:off[i + 1]]
        n = int(f_.sum())
        order = np.lexsort((np.arange(len(s)), -s))  # descending S, earlier first on ties
        top = np.zeros(len(s), bool); top[order[:n]] = True
        if not np.array_equal(top, f_):
            # allow ties at the boundary
            if n and n < len(s) and np.isclose(s[order[n - 1]], s[order[n]]):
                continue
            bad += 1
    print(f"{r:32s} answers whose flags are not a top-n of S: {bad}")
