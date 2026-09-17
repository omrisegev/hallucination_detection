"""Does L-SML beat averaging once the effective independent signal count exceeds 3?

Four evidence domains chosen by CONDITIONAL family correlation < .3 (diagnostic measurement,
reported separately): ve_innov (varentropy prefix innovations), dynamics (top15 turnover +
top50 truncated JS), position (relative step position), n_steps (answer length in steps).
Everything below is label-free: family virtuals are equal means of z-scored members with
natural orientation, fusion weights come from the training folds only, sign anchor is the
strongest single family by training-fold covariance with the equal score (no labels).
Scored with the frozen non-digit tail15 gate on PB and ungated within-answer AUC on PRMB.
"""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\omris\TAU\hallucination_detection\.worktrees\fusion-independence-atlas-v1")
from scripts.run_digitfree_broad50_v1 import load_data
from scripts.run_lsml_gate_locator_research_v1 import score_locator
from spectral_utils.digitfree_broad50 import NAMES
from spectral_utils.lsml_gate_locator_research import FusionRecipe, fit_fusion_weights, _orient
from spectral_utils.upcr import upcr_fit
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS
from spectral_utils.fusion_utils import sml_fuse_signed

W = r"C:\Users\omris\TAU\hallucination_detection\.worktrees\fusion-independence-atlas-v1"
data = load_data(); x = data["x"]; off = data["offsets"]; gate = data["gate"]
with np.load(W + r"\results\joint_feature_selection_bocpd_v1\INPUTS.npz") as z:
    bocpd = z["bocpd"]
n_steps = np.diff(off)
rel_pos = np.concatenate([np.arange(n) / max(n - 1, 1) for n in n_steps])
nsteps_col = np.repeat(n_steps, n_steps).astype(float)
# answer-standardize the two structural channels the same way as every other stream
def astd(v):
    out = np.empty_like(v, dtype=float)
    for a, b in zip(off[:-1], off[1:]):
        s = v[a:b]; out[a:b] = (s - s.mean()) / (s.std() + 1e-12) if s.std() > 1e-12 else 0.0
    return out
names = list(NAMES)
def col(n): return x[:, names.index(n)]

FAMILIES = {
    "ve_innov": np.column_stack([col(f"{n}_prefix_innovation") for n in ("ve0", "ve0.5", "ve0.75", "ve1", "ve2", "ve4")]).mean(1),
    "dynamics": np.column_stack([col("top15_turnover"), col("top50_truncated_js")]).mean(1),
    "position": astd(rel_pos),
    "n_steps":  astd(nsteps_col),  # NOTE: constant within an answer -> exactly zero after answer-standardization
    # reference families that the diagnostic says are redundant with each other
    "entropy_shape": np.column_stack([col(n) for n in ("a0.25", "a0.5", "H1", "a2", "a4", "Hinf", "H0lim")]).mean(1),
    "rank_risk": np.column_stack([col(f"rank_{i}_risk") for i in range(1, 16)]).mean(1),
    "provided_token": np.column_stack([col(n) for n in ("surprisal", "top1_loggap", "censored_rank50", "mass_above")]).mean(1),
    "bocpd": bocpd,
}
for k in FAMILIES: FAMILIES[k] = astd(np.asarray(FAMILIES[k], float))

SETS = {
    "D3_independent": ["ve_innov", "dynamics", "position"],
    "D4_plus_bocpd": ["ve_innov", "dynamics", "position", "bocpd"],
    "D5_plus_entropy": ["ve_innov", "dynamics", "position", "bocpd", "entropy_shape"],
    "D6_plus_provided": ["ve_innov", "dynamics", "position", "bocpd", "entropy_shape", "provided_token"],
    "D7_all_minus_nsteps": [f for f in FAMILIES if f != "n_steps"],
    "D4_redundant_control": ["entropy_shape", "rank_risk", "provided_token", "ve_innov"],
}
folds = np.repeat(data["folds"], n_steps)

def pr_of(M):
    c = np.corrcoef(M.T); lam = np.maximum(np.linalg.eigvalsh(c), 0)
    return lam.sum() ** 2 / (lam ** 2).sum()

for tag, fams in SETS.items():
    V = np.column_stack([FAMILIES[f] for f in fams])
    print(f"\n=== {tag}: {fams}  (marginal PR {pr_of(V):.2f}) ===")
    res = {}
    for arm in ("equal", "lsml", "iu", "sml_flat"):
        s = np.empty(len(V))
        for outer in range(5):
            te = folds == outer; tr = V[~te]
            if arm == "equal":
                w = np.ones(V.shape[1]) / V.shape[1]
            elif arm == "sml_flat":
                _, w = sml_fuse_signed(*[tr[:, j] for j in range(tr.shape[1])], small_m_guard=True)
                w = np.asarray(w, float)
            elif arm == "iu":
                w = upcr_fit(tr.T, **dict(IU_FIT_DEFAULTS)).w
            else:
                w, _ = fit_fusion_weights(tr, FusionRecipe(tag, tuple(fams), "continuous", anchor=0), seed=7)
            # label-free orientation: agree with the equal-weight score on the training rows
            eq = tr.mean(1)
            if np.corrcoef(tr @ w, eq)[0, 1] < 0: w = -w
            s[te] = V[te] @ w
        r = score_locator(s, gate, data); res[arm] = r
        print(f"  {arm:9s} PB {100*r['pb']:.2f}  within {r['within']:.4f}")
    print(f"  lsml - equal: {100*(res['lsml']['pb']-res['equal']['pb']):+.2f}pp PB, {res['lsml']['within']-res['equal']['within']:+.4f} within")
