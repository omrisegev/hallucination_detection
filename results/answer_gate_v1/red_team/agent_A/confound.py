"""Extra: how much of the PRMBench 'fair' AUROC (erroneous vs multi_solutions) is length?
And does length explain why OFFSET rules lose PRMScore?"""
import numpy as np
from scipy.stats import rankdata, spearmanr
from common import load_all, PRMCELL, FEAT

df, off, lab, d, bymeta = load_all()
cells = df.cell.values; ids = df.id.values
n_steps = np.diff(off)
f = np.load(FEAT, allow_pickle=True)
tl = f["X"][:, list(f["names"]).index("trace_length")]
prm_idx = np.where(cells == PRMCELL)[0]
cls = {i: bymeta[ids[i]]["classification"] for i in prm_idx}
has_err = {i: bool(np.any(lab[off[i]:off[i + 1]] == 1)) for i in prm_idx}
err = np.array([i for i in prm_idx if cls[i] != "correct" and has_err[i]])
ms = np.array([i for i in prm_idx if cls[i] == "multi_solutions"])


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); n1 = len(pos)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg))


print("median steps: erroneous", np.median(n_steps[err]), " multi_solutions", np.median(n_steps[ms]))
print("median tokens: erroneous", np.median(tl[err]), " multi_solutions", np.median(tl[ms]))
print("AUROC(-steps) err vs ms:", round(auc(-n_steps[err], -n_steps[ms]), 4),
      " AUROC(-tokens):", round(auc(-tl[err], -tl[ms]), 4))

# step-count quintile bins of multi_solutions (amendment R5 definition)
edges = np.quantile(n_steps[ms], [0.2, 0.4, 0.6, 0.8])
be = np.searchsorted(edges, n_steps[err], side="right"); bm = np.searchsorted(edges, n_steps[ms], side="right")
for det in ["D1_upcr_full", "D2_lsml_cont_good5", "D4_equal_full", "D5_epr", "D6_length"]:
    A = d["A_" + det].astype(float)
    num = den = 0.0
    for b in range(5):
        pe, pm = A[err][be == b], A[ms][bm == b]
        if len(pe) and len(pm):
            w = len(pe) * len(pm); num += auc(pe, pm) * w; den += w
    # token-length residualized: regress A on log(tokens) within PRMBench non-control, AUROC of residual
    nc = np.concatenate([err, ms])
    X = np.c_[np.ones(len(nc)), np.log(tl[nc]), np.log(n_steps[nc])]
    beta = np.linalg.lstsq(X, A[nc], rcond=None)[0]
    res = A[nc] - X @ beta
    print(f"{det:20s} raw={auc(A[err], A[ms]):.4f}  step-quintile-matched={num / den:.4f}  "
          f"residual on log tokens+log steps={auc(res[:len(err)], res[len(err):]):.4f}  "
          f"spearman(A, steps) on PRMB non-control={spearmanr(A[nc], n_steps[nc])[0]:+.3f}")

# error fraction vs length on PRMBench erroneous answers
frac = np.array([np.mean(lab[off[i]:off[i + 1]] == 1) for i in err])
print("PRMB erroneous answers: spearman(error-step fraction, n_steps) =", round(spearmanr(frac, n_steps[err])[0], 3),
      " spearman(n error steps, n_steps) =",
      round(spearmanr([np.sum(lab[off[i]:off[i + 1]] == 1) for i in err], n_steps[err])[0], 3))
