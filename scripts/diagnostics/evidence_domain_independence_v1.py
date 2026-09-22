"""Diagnostic (labels used for measurement only, never for fitting): are the evidence families
conditionally independent given the step label, and how many effective independent signals are there?"""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\omris\TAU\hallucination_detection\.worktrees\fusion-independence-atlas-v1")
from scripts.run_digitfree_broad50_v1 import load_data
from spectral_utils.digitfree_broad50 import NAMES
from sklearn.metrics import roc_auc_score

data = load_data(); x = data["x"]; off = data["offsets"]; labels = data["labels"]; cells = data["cells"].astype(str)
with np.load(r"C:\Users\omris\TAU\hallucination_detection\.worktrees\fusion-independence-atlas-v1\results\joint_feature_selection_bocpd_v1\INPUTS.npz") as z:
    bocpd = z["bocpd"]; noreset = z["noreset"]
n_steps = np.diff(off); step_idx = np.concatenate([np.arange(n) for n in n_steps])
rel_pos = np.concatenate([np.arange(n) / max(n - 1, 1) for n in n_steps])
nsteps_col = np.repeat(n_steps, n_steps).astype(float)
names = list(NAMES) + ["bocpd", "noreset", "rel_pos", "n_steps"]
X = np.column_stack([x, bocpd, noreset, rel_pos, nsteps_col])

FAM = {
    "rank_risk": [f"rank_{i}_risk" for i in range(1, 16)],
    "provided_token": ["surprisal", "top1_loggap", "censored_rank50", "mass_above"],
    "tail": ["tail15", "tail50"],
    "entropy_shape": ["a0.25", "a0.5", "H1", "a2", "a4", "Hinf", "H0lim"],
    "varentropy": ["ve0", "ve0.5", "ve0.75", "ve1", "ve2", "ve4"],
    "top2_ratio": ["top2_ratio"],
    "shape_innov": [f"{n}_prefix_innovation" for n in ("a0.25", "a0.5", "H1", "a2", "a4", "Hinf", "H0lim")],
    "ve_innov": [f"{n}_prefix_innovation" for n in ("ve0", "ve0.5", "ve0.75", "ve1", "ve2", "ve4")],
    "dynamics": ["top15_turnover", "top50_truncated_js"],
    "bocpd": ["bocpd"], "noreset": ["noreset"],
    "position": ["rel_pos"], "n_steps": ["n_steps"],
}
idx = {f: [names.index(n) for n in m] for f, m in FAM.items()}

# PRMB labelled steps
prm = np.repeat(np.char.startswith(cells, "prmbench_"), n_steps)
valid = prm & (labels >= 0); y = labels[valid] == 1; Xv = X[valid]
print(f"PRMB labelled steps {valid.sum()}, error rate {y.mean():.3f}")

def orient_auc(v):
    a = roc_auc_score(y, v); return max(a, 1 - a), (1 if a >= .5 else -1)

# family virtual = mean of members oriented (by label AUC, diagnostic only) then standardized
virt = {}; print("\n== family: members, mean member AUC, family-mean AUC, within-family mean |corr| ==")
for f, ii in idx.items():
    cols = []; aucs = []
    for i in ii:
        a, s = orient_auc(Xv[:, i]); aucs.append(a); cols.append(s * (Xv[:, i] - Xv[:, i].mean()) / (Xv[:, i].std() + 1e-12))
    M = np.column_stack(cols); v = M.mean(1); v = (v - v.mean()) / (v.std() + 1e-12); virt[f] = v
    c = np.corrcoef(M.T) if M.shape[1] > 1 else np.ones((1, 1)); wc = np.abs(c[np.triu_indices(M.shape[1], 1)]).mean() if M.shape[1] > 1 else 1.0
    print(f"{f:15s} m={len(ii):2d} memberAUC={np.mean(aucs):.3f} familyAUC={roc_auc_score(y, v):.3f} within|corr|={wc:.2f}")

F = list(virt); V = np.column_stack([virt[f] for f in F])
def corr(M): return np.corrcoef(M.T)
def pr(c): lam = np.linalg.eigvalsh(c); lam = np.maximum(lam, 0); return lam.sum() ** 2 / (lam ** 2).sum()
marg = corr(V)
# conditional given label: center within class, pool
Vc = V.copy()
for cls in (True, False):
    m = y == cls; Vc[m] -= Vc[m].mean(0)
cond = corr(Vc)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("\nfamilies:", F)
print("\nmarginal correlation among family virtuals:\n", marg)
print("\nconditional (within-label) correlation:\n", cond)
print(f"\neffective independent signals (participation ratio): marginal {pr(marg):.2f}, conditional {pr(cond):.2f}, of {len(F)} families")
# greedy: pick a set with all pairwise conditional |rho| < thr, best AUC first
aucs = {f: roc_auc_score(y, virt[f]) for f in F}
for thr in (.3, .4, .5):
    chosen = []
    for f in sorted(F, key=lambda f: -aucs[f]):
        if all(abs(cond[F.index(f), F.index(g)]) < thr for g in chosen): chosen.append(f)
    sub = [F.index(f) for f in chosen]
    print(f"\nthr {thr}: {len(chosen)} families {chosen}; cond PR {pr(cond[np.ix_(sub, sub)]):.2f}; AUCs {[round(aucs[f],3) for f in chosen]}")
