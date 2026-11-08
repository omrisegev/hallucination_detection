"""The functions of the frozen recipe C4U-parisi-within-jaffe-cross-v1 that are NOT on the remote (2026-11-07).

Everything else the recipe calls is imported from files that are identical on origin
(checked 2026-11-07 against the fetched remote refs):
  results/prm_boost_collection_v1/evaluate_collection.py   E   (load, channels, answer_z, topk_flags, scoring)
  results/prm_boost_math_rigor_v1/common.py                K   (eval_rows, prm_arm)
  results/prm_boost_math_rigor_v1/stage8_auto_forced_k.py  S8  (fit = L-SML guarded defaults, gap_flags, point)
  spectral_utils/binary_view_estimation.py                     (rank_one_offdiag, spectral_binary)
  spectral_utils/fusion_utils.py (via S8.fit; differs from origin only in CRLF line endings)

The bodies below are copied unchanged from the local files named in each docstring. The only edit: module aliases
are imported directly instead of through the chain of experiment scripts that used to provide them, and the
PRMBench score directory is E.DATA / 'prmbench' (the same path the chain resolved).
run_c4u.py checks that this file reproduces the frozen outputs bit for bit.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault('COLLECTION_DATA', str(HERE / 'data'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'results/prm_boost_math_rigor_v1'))
import stage8_auto_forced_k as S8  # noqa: E402  (remote-backed)
import common as K  # noqa: E402  (remote-backed)
from spectral_utils.binary_view_estimation import rank_one_offdiag, spectral_binary  # noqa: E402  (remote-backed)

E = S8.E                                                                  # evaluate_collection (remote-backed)
BUNDLE = Path(os.environ['COLLECTION_DATA']) / 'auto_raw_plus_z_bundle' / 'inputs'
RECIPE_ID = 'C4U-parisi-within-jaffe-cross-v1'
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']


# ---------------------------------------------------------------- from results/rigorous_recipe_v1/rr_common.py
class Cell:
    """Everything one PRMBench cell needs, aligned and checked against E.load."""

    def __init__(self, prm):
        c = E.load('prmbench', prm)
        self.c, self.off, self.prm = c, c.off, prm
        _, self.ans_eval = K.eval_rows(c, 'prmbench')
        self.Z = E.channels(c, 'N16')                                    # stored answer-z channels
        self.zp = E.answer_z(c.arms[K.prm_arm(c)], c.off)
        A = np.load(Path(E.DATA) / 'prmbench' / f'ABL5_SCORES_{prm}.npz')
        assert np.array_equal(A['offsets'], c.off)
        self.rawp = 1. - A['reward']                                     # PRM error probability
        B = np.load(BUNDLE / f'INPUTS_{prm}.npz')
        assert np.array_equal(B['offsets'], c.off) and np.array_equal(B['step_error'].astype(bool), c.err.astype(bool))
        assert np.abs(B['X34'][:, 1:17] - self.Z).max() < 1e-5 and np.abs(B['X34'][:, 0] - self.zp).max() < 1e-5
        self.G16 = B['X34'][:, 18:34].astype(float)                     # global rank->normal of the raw channels


def build_R1(cell):
    """Stage-1 rung R1: [zp, raw p global std, answer-z of the global rank->normal channels]; raw PRM votes p > .5."""
    Zr = np.column_stack([E.answer_z(cell.G16[:, j], cell.off) for j in range(16)])
    X = np.column_stack([cell.zp, (cell.rawp - cell.rawp.mean()) / cell.rawp.std(), Zr])
    return X, {1: cell.rawp > .5}


def lsml(X, anchor):
    s, d = S8.fit(X, anchor=anchor)
    return s, np.asarray(d['c'], int), d


def votes(cell, X, s, absolute):
    """+1 = error. Column 0 is the anchor (not reoriented). `absolute` maps column -> boolean vote (fixed
    classifier, e.g. PRM p > .5); all other columns get the GAP vote after orienting by corr with s."""
    V = np.empty(X.shape)
    for j in range(X.shape[1]):
        if j in absolute:
            V[:, j] = np.where(absolute[j], 1., -1.)
            continue
        x = X[:, j]
        if j > 0 and np.corrcoef(x, s)[0, 1] < 0:
            x = -x
        V[:, j] = np.where(S8.gap_flags(cell.c, x), 1., -1.)
    return V


# ---------------------------------------------------------------- from results/rigorous_recipe_v1/within_group_sml/run.py
def lw(psi, eta):
    psi, eta = np.clip(psi, 1e-3, 1 - 1e-3), np.clip(eta, 1e-3, 1 - 1e-3)
    return .5 * (np.log(psi / (1 - psi)) + np.log(eta / (1 - eta)))


def load(bench, prm, rep):
    """-> c, ans_eval, X, ab, wk. PRMBench uses rep 'R1', Socratic 'R0'."""
    if bench == 'prmbench':
        cell = Cell(prm)
        assert rep == 'R1'
        X, ab = build_R1(cell)
        return cell.c, cell.ans_eval, X, ab, None
    c = E.load('socratic', prm); c.groups = np.array([u.split(':')[-1] for u in c.ids])
    _, ans_eval = K.eval_rows(c, 'socratic')
    Z = E.channels(c, 'N16'); zp = E.answer_z(c.arms[K.prm_arm(c)], c.off); raw = c.arms['PRMPROB']
    X = np.column_stack([zp, (raw - raw.mean()) / raw.std(), Z])
    return c, ans_eval, X, {1: raw > .5}, c.keep


# ---------------------------------------------------------------- from results/spectral_count_v1/spec_count.py
def posterior(V, psi, eta, prev):
    psi, eta = np.clip(psi, 1e-4, 1 - 1e-4), np.clip(eta, 1e-4, 1 - 1e-4)
    lo = np.full(len(V), np.log(prev / (1 - prev)))
    for j in range(V.shape[1]):
        lo += np.where(V[:, j] > 0, np.log(psi[j] / (1 - eta[j])), np.log((1 - psi[j]) / eta[j]))
    return 1 / (1 + np.exp(-lo))


# ---------------------------------------------------------------- from results/rigorous_recipe_v1/frozen_c4u/c4u.py
class RecipeFailure(RuntimeError):
    """The recipe has no fallback: fewer than 3 groups, or no group holding the PRM column."""


def c4u(c, X, ab):
    """X: n_steps x 18 (col 0 = answer-z PRM risk, col 1 = raw PRM p standardized, cols 2.. = 16 channels).
    ab: {1: raw p > .5}. Returns rank, flags, posterior and the fitted quantities."""
    off = c.off
    s, g, _ = lsml(X, X[:, 0])
    V = votes(SimpleNamespace(c=c), X, s, ab)
    sign = np.array([1.] + [1. if np.corrcoef(X[:, j], s)[0, 1] >= 0 else -1. for j in range(1, X.shape[1])])
    Xo = np.column_stack([X[:, j] if sign[j] > 0 else -X[:, j] for j in range(X.shape[1])])
    Xo = (Xo - Xo.mean(0)) / Xo.std(0)
    ag = None; VG, XG, ws, gid = [], [], [], []
    for k in np.unique(g[g >= 0]):
        idx = np.flatnonzero(g == k); wt = np.ones(len(idx))
        if len(idx) >= 3:
            wt = np.abs(rank_one_offdiag(np.cov(Xo[:, idx].T)))
        wt = wt / wt.sum()
        if 0 in idx:
            ag = len(VG)
        VG.append(np.where(V[:, idx] @ wt > 0, 1., -1.))
        xs = Xo[:, idx] @ wt
        XG.append((xs - xs.mean()) / xs.std())
        ws.append(wt); gid.append(idx)
    if len(VG) < 3 or ag is None:
        raise RecipeFailure(f'{len(VG)} groups, PRM group {ag}')
    VG, XG = np.column_stack(VG), np.column_stack(XG)
    sp = spectral_binary(VG, anchor=ag)
    P = posterior(VG, sp['psi'], sp['eta'], float(np.clip(sp['prevalence'], .01, .99)))
    rank = XG @ lw(sp['psi'], sp['eta'])
    k = np.rint(np.add.reduceat(P, off[:-1])).astype(int)
    F = E.topk_flags(rank, off, k)
    return dict(rank=rank, flags=F, posterior=P, k=k, prevalence=float(sp['prevalence']), psi=sp['psi'],
                eta=sp['eta'], groups=gid, weights=ws, anchor_group=ag, c4_sign=sign, unassigned=int((g < 0).sum()))
