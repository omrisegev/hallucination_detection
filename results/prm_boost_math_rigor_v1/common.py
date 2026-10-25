"""Shared loaders for results/prm_boost_math_rigor_v1 (Stage 1 diagnostics, Stage 2 gate).

Builds, for one (benchmark, PRM) cell, the inputs of CHAIN_N16's cross stage exactly as
results/prm_boost_ablation5_v1/ablation5.py does: zp = answer_z(PRM risk), the N16 telemetry groups found by
L-SML on the channels alone, each group score answer-z'd, and CHAIN's effective cross weights (with the anchor flip
folded in, so that fused = VC @ w). Parity with the exported CHAIN_N16 is checked by stage0_parity.py.
"""
import os, sys
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests')); sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
from lever_imports import ensure_spectral_package  # noqa: E402

ensure_spectral_package()
import evaluate_collection as E  # noqa: E402
from collection_parity import h2v_common_uids  # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous_guarded  # noqa: E402

BENCHES = ('prmbench', 'socratic', 'hard2verify')
PERM_SEED = 20261018          # ablation5.py's PERM control seed


def prm_arm(c):
    return 'PRM_Q80' if c.bench == 'prmbench' else 'PRM'


def lsml(V, groups=None, anchor=None):
    s, d = lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])], groups=groups)
    s = np.asarray(s, float)
    ref = V[:, 0] if anchor is None else anchor
    flip = np.corrcoef(s, ref)[0, 1] < 0
    return (-s if flip else s), d, flip


def relative_position(off):
    return np.concatenate([(np.arange(b - a) + .5) / (b - a) for a, b in zip(off[:-1], off[1:])])


def perm_rows(off, seed=PERM_SEED):
    """ablation5.py's PERM control: move each answer's telemetry block to another answer with the same step count."""
    n = np.diff(off); rows = np.arange(off[-1]); rng = np.random.default_rng(seed)
    for k in np.unique(n):
        js = np.flatnonzero(n == k)
        if len(js) < 2:
            continue
        tgt = js[rng.permutation(len(js))]
        for a, b in zip(js, tgt):
            rows[off[a]:off[a + 1]] = np.arange(off[b], off[b + 1])
    return rows


def chain_inputs(bench, prm, bank='N16', permute=False):
    """Cell on its full fitted answer set: (cell, VC, w_eff, fused, info). fused == exported CHAIN (stage 0)."""
    c = E.load(bench, prm)
    V = E.channels(c, bank)
    if permute:
        V = V[perm_rows(c.off)]
    zp = E.answer_z(c.arms[prm_arm(c)], c.off)
    _, dt, _ = lsml(V, anchor=V.mean(1))
    vg = np.asarray(dt['virtual_classifiers'], float)
    VC = np.column_stack([zp] + [E.answer_z(vg[:, g], c.off) for g in range(vg.shape[1])])
    fused, dc, flip = lsml(VC, groups=np.arange(VC.shape[1]))
    w = np.asarray(dc['cross_weights'], float) * (-1. if flip else 1.)
    assert np.allclose(VC @ w, fused, atol=1e-8)
    return c, VC, w, fused, {'K': int(dt['K']), 'flip': bool(flip), 'tele_groups': np.asarray(dt['c']).tolist()}


def covariates(c):
    pos = relative_position(c.off)
    loglen = c.extra.get('log_step_tokens') if getattr(c, 'extra', None) else None
    return pos, (None if loglen is None else np.asarray(loglen, float))


def eval_rows(c, bench):
    """Boolean mask over the cell's steps that enters the evaluation (Hard2Verify: the 177-answer common set)."""
    if bench != 'hard2verify':
        return np.ones(c.off[-1], bool), np.ones(len(c.ids), bool)
    keep_ans = np.isin(c.ids, list(h2v_common_uids()))
    return np.repeat(keep_ans, np.diff(c.off)), keep_ans


def within_auc(c, score, ans_mask=None):
    """Mean within-answer AUC over answers with both classes (the collection's definition, incl. external keep)."""
    keep = c.keep if c.bench != 'prmbench' else None
    wa = E.per_answer_auc(c.err, score, c.off, keep)
    if ans_mask is not None:
        wa = wa[ans_mask]
    return float(np.nanmean(wa))
