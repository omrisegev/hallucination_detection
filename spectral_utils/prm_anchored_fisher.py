"""PRM-anchored Fisher fusion, end to end (FRAMING_THEORY.md P5; docs/research_notes/math_rigor_2026-10-24/).

The idea in one paragraph. CHAIN_N16 fuses the PRM risk with telemetry group scores by L-SML, whose weights come from
agreement among the views. When the telemetry shares a nuisance (the same backbone), that agreement is read as
accuracy, and a strong PRM is under-weighted (P3, P4). Here the PRM is the anchor (a "marker indicator" in factor
analysis): each group's loading on the error is its covariance with the PRM divided by the PRM's own loading,
u_k = C_0k / u_0. The single unknown u_0 is identified from the telemetry block's rank condition (second order, needs
>= 4 groups) or from third-order cumulants. The weights are then Fisher's, w ∝ C^{-1} u, which account for the
correlation among the views instead of assuming conditional independence.

Pipeline (one answer set, no labels):
  1. answer-z every view; the PRM view is z(logit(1 - reward)) as in CHAIN_N16;
  2. telemetry groups and group scores exactly as CHAIN_N16 (L-SML Algorithm 1 on the channels alone);
  3. estimation matrix: the cross-stage views [PRM, groups], residualized within answers on a covariate basis
     (cubic relative position, plus log step length when given) — P5(f); scoring uses the views as they are;
  4. groups whose covariance with the PRM is <= 0 are excluded (weight 0): they contradict the anchor (P5 g);
  5. u from fusion_theory.anchored_loadings ('rank', 'cumulant', 'cumulant_w01' or 'cca'; 'auto' = rank when at
     least 4 groups remain, else cumulant);
  6. w = C^{-1} u on the estimation covariance; score = views @ w;
  7. if u is not identified, fall back to the PRM view (or to a caller-given score).
  Optional: the label-free gate G(w) > 0 (P5 e), point estimate.

Assumptions (stated, not guaranteed): the nuisance does not load on the PRM after covariate adjustment, and the PRM's
noise is independent of the telemetry's. Step 469 measured within-class |corr| .13–.35 between the residualized PRM
and the group scores, so the anchor condition holds only partly on the existing data.
"""
from __future__ import annotations

import numpy as np

from .fusion_theory import (anchored_loadings, covariate_basis, fisher_weights, gate_statistic,
                            within_answer_residualize)
from .fusion_utils import lsml_continuous_guarded

METHODS = ('auto', 'rank', 'cumulant', 'cumulant_w01', 'cca')


def answer_z(x, offsets):
    """Per-answer z-score, population sd floored at 1e-8 (evaluate_collection.answer_z)."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    for a, b in zip(offsets[:-1], offsets[1:]):
        v = x[a:b]
        out[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return out


def relative_position(offsets):
    return np.concatenate([(np.arange(b - a) + .5) / (b - a) for a, b in zip(offsets[:-1], offsets[1:])])


def chain_views(prm_risk, channels, offsets):
    """CHAIN_N16's cross-stage views [z(PRM), z(group 1), ..., z(group K)] and the telemetry partition.

    Mirrors results/prm_boost_math_rigor_v1/common.py chain_inputs (parity with the exported CHAIN_N16 there)."""
    V = np.asarray(channels, float)
    zp = answer_z(prm_risk, offsets)
    s, d = lsml_continuous_guarded(*[V[:, j] for j in range(V.shape[1])])
    vg = np.asarray(d['virtual_classifiers'], float)
    VC = np.column_stack([zp] + [answer_z(vg[:, g], offsets) for g in range(vg.shape[1])])
    return VC, {'K': int(d['K']), 'groups': np.asarray(d['c'], int).tolist()}


def anchored_fisher_weights(X_est, offsets, method: str = 'auto'):
    """Weights over the columns of X_est (column 0 = the anchor). Returns (w or None, info)."""
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    X = np.asarray(X_est, float)
    C = np.cov(X.T)
    keep = np.r_[0, 1 + np.flatnonzero(C[0, 1:] > 0)]
    info = {'excluded_groups': [int(k) for k in range(1, X.shape[1]) if k not in keep], 'n_used_groups': len(keep) - 1}
    if len(keep) < 3:
        info['reason'] = 'fewer than 2 groups agree with the anchor'
        return None, info
    m = method
    if method == 'auto':
        m = 'rank' if len(keep) - 1 >= 4 else 'cumulant'
    info['route'] = m
    u, li = anchored_loadings(X[:, keep], anchor=0, method=m)
    if u is None and method == 'auto' and m == 'rank':           # rank profile flat or out of range: third order
        m = 'cumulant'; info['route'] = 'cumulant (rank not identified)'
        u, li = anchored_loadings(X[:, keep], anchor=0, method=m)
    if u is None:
        info['reason'] = li.get('abstain', 'not identified')
        return None, info
    Ck = li['C']
    wk = fisher_weights(Ck, u)
    w = np.zeros(X.shape[1]); w[keep] = wk
    g = gate_statistic(Ck, u, wk, anchor=0)
    info.update(u=dict(zip(keep.tolist(), np.round(u, 6).tolist())), G=g['G'], snr_fused=g['snr_w'],
                snr_prm=g['snr_anchor'], prm_share=float(abs(w[0]) / np.abs(w).sum()),
                n_negative_weights=int((w[keep] < 0).sum()))
    return w, info


def anchored_fisher_fuse(prm_risk, channels, offsets, log_length=None, method: str = 'auto', gate: bool = False,
                         fallback=None):
    """End-to-end label-free fusion. prm_risk: per-step PRM risk (higher = more likely wrong), e.g. logit(1 - reward);
    channels: steps x P telemetry, each oriented so that higher = more likely wrong; offsets: answer boundaries.

    Returns dict(score, w, fused (bool: the anchored weights were used), views, info)."""
    VC, part = chain_views(prm_risk, channels, offsets)
    Z = covariate_basis(relative_position(offsets), log_length)
    X_est = within_answer_residualize(VC, Z, offsets)
    w, info = anchored_fisher_weights(X_est, offsets, method)
    info.update(part)
    use = w is not None and (not gate or info.get('G', -1) > 0)
    if w is not None and gate and not use:
        info['reason'] = 'gate G <= 0'
    if use:
        score = VC @ w
    else:
        score = VC[:, 0] if fallback is None else np.asarray(fallback, float)
    return {'score': score, 'w': None if w is None else w, 'fused': bool(use), 'views': VC, 'info': info}
