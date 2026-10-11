"""
LEASH — "Logit-Entropy Adaptive Stopping Heuristic for Efficient Chain-of-Thought
Reasoning" (Quamar & Areeb, arXiv:2511.04654v1, NeurIPS 2025 Efficient Reasoning workshop).

Handoff §S2. Every row from this module is `paper-specified-partial`, never a
reproduction, because four constants the algorithm cannot run without are absent from the
paper despite its claim to report concrete settings.

Published, frozen (§Decoding settings)
--------------------------------------
    k  = 8        window for the entropy slope / margin improvement
    L  = 5        vote span (consistency)
    eH = 0.005    entropy-slope slack
    dM = 0.05     margin-improvement slack
    m  = 64       minimum rationale length
    M  = 320      maximum rationale length
    rationale decoding: nucleus p=0.95, T=0.7, do_sample=True
    final answer:       greedy (T=0.0)
    EOS is disabled during rationale generation

Missing, declared by us (§Notes of the digest; verified absent from the extract)
-------------------------------------------------------------------------------
    B      logit clip band
    tau_p  pmax saturation threshold
    w      warm-up added to m in tmin = max(m + w, k + L)
    gamma  entropy-drop gate, Href - Ht >= gamma

`SENSITIVITY_GRID` pre-registers the values we sweep on pilot IDs; `CENTRAL_CHOICE` is the
single setting frozen **before** the full evaluation. Handoff §S2 is explicit that the best
post-hoc grid point may never be called the reproduction, so `LeashConfig.as_manifest()`
records which one a run used and refuses to omit the label.

The published operating point buys ~30% token reduction at ~10 accuracy points, so LEASH is
a declared sensitivity baseline, not a matched-accuracy competitor; the honest comparison is
the whole accuracy-versus-token frontier, not its single point against ours.
"""
from dataclasses import dataclass, field

import numpy as np

# ── published constants ─────────────────────────────────────────────────────────
K_WINDOW = 8
L_VOTE = 5
EPS_H = 0.005
DELTA_M = 0.05
M_MIN = 64
M_MAX = 320
RATIONALE_DECODING = {"temperature": 0.7, "top_p": 0.95, "top_k": 0}
ANSWER_DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": 0}

# ── the four values the paper does not pin ──────────────────────────────────────
#: Pre-registered sweep, run on pilot IDs only, before any full evaluation.
SENSITIVITY_GRID = {
    "B": (10.0, 30.0, 100.0),
    "tau_p": (0.90, 0.95, 0.99),
    "w": (0, 16, 32),
    "gamma": (0.05, 0.10, 0.25),
}

#: Frozen central choice. Rationale, so the choice is auditable rather than arbitrary:
#:  B=30      wide enough that clipping never binds for a calibrated LM (max |logit| is
#:            typically < 30), so the guard stays a numerical-stability guard.
#:  tau_p=0.95 the paper calls these steps "highly confident"; 0.95 is the conventional
#:            reading and sits at the centre of the swept band.
#:  w=16      smallest non-zero warm-up that lets Href (median over the first k=8 steps)
#:            settle before the gate can fire.
#:  gamma=0.10 an entropy drop of 0.1 nat below the reference is a detectable move without
#:            demanding the near-total collapse that 0.25 requires within M=320 tokens.
CENTRAL_CHOICE = {"B": 30.0, "tau_p": 0.95, "w": 16, "gamma": 0.10}


@dataclass
class LeashConfig:
    k: int = K_WINDOW
    L: int = L_VOTE
    eps_H: float = EPS_H
    delta_M: float = DELTA_M
    m: int = M_MIN
    M: int = M_MAX
    B: float = CENTRAL_CHOICE["B"]
    tau_p: float = CENTRAL_CHOICE["tau_p"]
    w: int = CENTRAL_CHOICE["w"]
    gamma: float = CENTRAL_CHOICE["gamma"]
    #: 'central' for the frozen full-evaluation setting, 'grid:<i>' for a pilot sweep point.
    setting_label: str = "central"

    @property
    def t_min(self) -> int:
        return max(self.m + self.w, self.k + self.L)

    def as_manifest(self) -> dict:
        return {
            "published": {"k": self.k, "L": self.L, "eps_H": self.eps_H,
                          "delta_M": self.delta_M, "m": self.m, "M": self.M},
            "declared_by_us": {"B": self.B, "tau_p": self.tau_p, "w": self.w,
                               "gamma": self.gamma},
            "setting_label": self.setting_label,
            "t_min": self.t_min,
            "fidelity": "paper-specified-partial",
            "note": "B, tau_p, w and gamma are not numerically specified in the paper; "
                    "they are declared here and swept on pilot IDs only. The best grid "
                    "point is never reported as the reproduction.",
        }


def clip_logits(logits, B: float):
    """z~ = clip(finite(z), -B, B), in fp32 (§Numerical stability)."""
    z = np.asarray(logits, dtype=np.float64)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(z, -float(B), float(B))


def step_signals(logits, B: float) -> dict:
    """H_t (Eq. 1), M_t (Eq. 2) and pmax(t), from one step's raw logits."""
    z = clip_logits(logits, B)
    z = z - z.max()
    lse = np.log(np.exp(z).sum())
    lp = z - lse
    p = np.exp(lp)
    order = np.argpartition(-lp, 1)[:2]
    top2 = np.sort(lp[order])[::-1]
    return {
        "H": float(-(p * lp).sum()),
        "M": float(top2[0] - top2[1]),
        "pmax": float(p.max()),
    }


class LeashStopper:
    """Algorithm 1, driven one token at a time.

    Feed `push(H, M, pmax)` after each decoded token; `push` returns True when the rule
    fires. State is explicit so a preempted run resumes on exactly the same trajectory.

    The three conditions, in the paper's own order:
      (i)   t >= tmin = max(m + w, k + L)
      (ii)  a majority of the last L **non-saturated** steps pass the plateau test
      (iii) the entropy-drop gate Href - Ht >= gamma holds, with
            Href = median(H_1..H_k)
    """

    def __init__(self, cfg: LeashConfig = None):
        self.cfg = cfg or LeashConfig()
        self.H, self.M, self.pmax, self.sat = [], [], [], []
        self.plateau = []      # per-step Pi_t, NaN on saturated steps
        self.h_ref = None
        self.fired = None

    def push(self, H: float, M: float, pmax: float) -> bool:
        c = self.cfg
        self.H.append(float(H))
        self.M.append(float(M))
        self.pmax.append(float(pmax))
        saturated = bool(pmax >= c.tau_p)
        self.sat.append(saturated)
        t = len(self.H)                      # 1-based step index

        if t == c.k:
            self.h_ref = float(np.median(self.H[:c.k]))

        # Pi_t (Eq. 6) — only defined for non-saturated steps with a full k-window behind them.
        if saturated or t <= c.k:
            self.plateau.append(np.nan)
        else:
            s_H = (self.H[-1] - self.H[-1 - c.k]) / c.k
            d_M = self.M[-1] - self.M[-1 - c.k]
            self.plateau.append(float(bool(s_H >= -c.eps_H and d_M <= c.delta_M)))

        if t < c.t_min or self.h_ref is None:
            return False
        # (iii) entropy-drop gate, checked before the vote exactly as in Algorithm 1 line 7.
        if not (self.h_ref - self.H[-1] >= c.gamma):
            return False
        # (ii) majority over the last L non-saturated steps
        idx = [i for i in range(len(self.plateau)) if not self.sat[i]
               and not np.isnan(self.plateau[i])][-c.L:]
        if not idx:
            return False
        votes = sum(self.plateau[i] for i in idx)
        if votes >= np.ceil(len(idx) / 2.0):
            self.fired = {"t": t, "votes": float(votes), "n_voting": len(idx),
                          "h_ref": self.h_ref, "H_t": self.H[-1],
                          "entropy_drop": self.h_ref - self.H[-1]}
            return True
        return False

    def diagnostics(self) -> dict:
        return {
            "n_steps": len(self.H),
            "n_saturated": int(sum(self.sat)),
            "h_ref": self.h_ref,
            "fired": self.fired,
            "config": self.cfg.as_manifest(),
        }


def grid_points(grid: dict = None) -> list:
    """Enumerate the pre-registered sensitivity grid as labelled LeashConfig objects."""
    import itertools
    grid = grid or SENSITIVITY_GRID
    keys = sorted(grid)
    out = []
    for i, combo in enumerate(itertools.product(*(grid[k] for k in keys))):
        kw = dict(zip(keys, combo))
        out.append(LeashConfig(setting_label=f"grid:{i}", **kw))
    return out
