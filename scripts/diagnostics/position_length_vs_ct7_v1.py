"""Diagnostic only (no method): can position / length serve as additional evidence for CT7?

Step level (PRMB labelled steps, labels for measurement): AUC, conditional correlation with the seven
CT7 views, effective independent views with each candidate added, and within-answer correlation with
the CT7 fused score. Answer level (ProcessBench): does answer length separate error from clean answers,
and is it already carried by the frozen gate score?
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

W = r"C:\Users\omris\TAU\hallucination_detection\.worktrees\fusion-independence-atlas-v1"
sys.path.insert(0, W)
import scripts.run_digitfree20_ladder_v1 as L
import scripts.analyze_chosen_token_step_tests_v2 as S
from spectral_utils import frozen_locator_ct7 as C
from spectral_utils.chosen_token_calibration import SUFFICIENT

data = L.load_data(); off = data["offsets"]; cells = data["cells"].astype(str)
n = np.diff(off); target = np.asarray(data["target"]); lab = data["labels"]
with np.load(W + r"\results\joint_feature_selection_bocpd_v1\INPUTS.npz") as z:
    bocpd = z["bocpd"]
suff = S.load_folder("extracted_sufficient", off, len(SUFFICIENT))
views = C.candidate_views(data["x"], bocpd, suff, off); fused = views.mean(1)

idx = np.concatenate([np.arange(k) for k in n]); nrep = np.repeat(n, n)
tok = suff[:, 0]
raw = {
    "log_step_tokens": np.log(tok),
    "relative_position": idx / np.maximum(nrep - 1, 1),
    "absolute_index": idx.astype(float),
    "steps_from_end": (nrep - 1 - idx).astype(float),
    "is_first_step": (idx == 0).astype(float),
    "is_last_step": (idx == nrep - 1).astype(float),
    "cumulative_token_fraction": None,
}
cum = np.empty(len(tok))
for a, b in zip(off[:-1], off[1:]):
    c = np.cumsum(tok[a:b]); cum[a:b] = (c - tok[a:b]) / c[-1]   # share of the answer's tokens before this step
raw["cumulative_token_fraction"] = cum
cand = {k: S.astd(v, off) for k, v in raw.items()}

prm = np.repeat(np.char.startswith(cells, "prmbench_"), n); valid = prm & (lab >= 0); y = lab[valid] == 1


def cond(M):
    Mc = M.copy()
    for cls in (True, False):
        Mc[y == cls] -= Mc[y == cls].mean(0)
    return np.corrcoef(Mc.T)


def pr(c):
    lam = np.maximum(np.linalg.eigvalsh(c), 0); return lam.sum() ** 2 / (lam ** 2).sum()


V = views[valid]
print(f"CT7 alone: effective conditional views {pr(cond(V)):.2f} of 7")
print(f"{'candidate':26s} {'AUC':>5s} {'max|cond| CT7':>13s} {'cond w/ token view':>18s} {'eff views':>9s} {'within-ans corr w/ CT7':>22s}")
for k, v in cand.items():
    vv = v[valid]; auc = roc_auc_score(y, vv)
    c = cond(np.column_stack([V, vv]))
    wc = []
    for a, b in zip(off[:-1], off[1:]):
        if b - a > 2 and v[a:b].std() > 0 and fused[a:b].std() > 0:
            wc.append(np.corrcoef(v[a:b], fused[a:b])[0, 1])
    print(f"{k:26s} {auc:5.3f} {np.abs(c[-1, :-1]).max():13.2f} {c[-1, 6]:18.2f} {pr(c):9.2f} {np.nanmean(wc):22.2f}")

# where are first errors (PB error answers), by relative position
pb = np.char.startswith(cells, "pb_"); err = pb & (target >= 0)
relt = target[err] / np.maximum(n[err] - 1, 1)
print("\nPB first-error relative position quartiles:", np.quantile(relt, [.25, .5, .75]).round(2),
      "| share at step 0:", round(float(np.mean(target[err] == 0)), 3),
      "| share at last step:", round(float(np.mean(target[err] == n[err] - 1)), 3))
# how often is the first error the longest step, versus chance
longest = np.array([int(np.argmax(tok[a:b])) for a, b in zip(off[:-1], off[1:])])
print("PB error answers: first error = longest step", round(float(np.mean(longest[err] == target[err])), 3),
      "| chance 1/n", round(float(np.mean(1 / n[err])), 3),
      "| CT7 peak = longest step", round(float(np.mean(np.array([int(np.argmax(fused[a:b])) for a, b in zip(off[:-1], off[1:])])[err] == longest[err])), 3))

# answer level: does answer length separate error from clean PB answers; is it in the gate already?
with np.load(W + r"\results\fusion_independence_atlas_v1\baseline_replay\SCORES_FROZEN.npz") as z:
    gate_raw = z["gate_raw"]
ans_tokens = np.array([tok[a:b].sum() for a, b in zip(off[:-1], off[1:])])
yb = (target[pb] >= 0)
print("\nPB answer level (error vs clean):")
for name, v in (("n_steps", n[pb].astype(float)), ("log answer tokens", np.log(ans_tokens[pb])), ("frozen gate raw score", gate_raw[pb])):
    print(f"  {name:22s} AUC {roc_auc_score(yb, v):.3f}")
print(f"  spearman(gate raw, n_steps) {float(__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(gate_raw[pb], n[pb]).statistic):.2f}")
# within cell, since cells differ in length and error rate
aucs = []
for c in sorted(set(cells[pb])):
    m = cells == c
    aucs.append((c, roc_auc_score(target[m] >= 0, n[m]), roc_auc_score(target[m] >= 0, gate_raw[m])))
for c, a1, a2 in aucs:
    print(f"  {c:22s} n_steps AUC {a1:.3f}  gate AUC {a2:.3f}")
