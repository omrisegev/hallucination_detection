"""Bug-fix release (v1b) of the raw-null clean-answer gate. Spec: docs/experiments/RAW_NULL_GATE_V1B_BUGFIX_2026-10-31.md.

v1 (spectral_utils/raw_null_gate.py) stays frozen. This module re-uses its unchanged parts and replaces only:
    F1  G2: each group's likelihood ratio uses the group's own pi0 (v1 used the shared median pi0).
    F2  dropped (see empirical_null): the fit is v1's; only its sd is recorded.
    F3  optional (monotone=True): above mu0, P(err) is the running max of 1 - lfdr over the scored values sorted by z.
        A known-practice correction, reported separately from the bug fixes.
    F4  G3 drops voters whose answer votes are constant; needs >= 3 varying voters.
(F5-F7 are in the runner and the verifier.)
"""
from __future__ import annotations

import numpy as np

from .prm_anchored_lsml import ds_em_votes
from .raw_null_gate import (P_ERR_CUT, PI0_CLIP, LR_CLIP, linear_lsml, group_scores, group_defined,  # noqa: F401
                            fused_linear, group_orientation, pooled_z, worst_step, answer_max, gate_g1b, gap_flags,
                            ds_answer_posterior, group_fit_summary, _defined)
from .raw_null_gate import empirical_null as _empirical_null_v1


def empirical_null(x, **kw):
    """v1's two-groups fit, unchanged, plus the score's sd as a diagnostic. F2 (rejecting pi0_raw > 1 or sigma0 >= sd)
    was dropped: on pure N(0, 1) data with no errors central matching gives pi0_raw ~1.02 and sigma0/sd ~1.03
    (n = 9k-145k), so that rule would reject every error-free fit. Identification goes to the diagnosis plan."""
    f = _empirical_null_v1(x, **kw)
    x = np.asarray(x, float)
    f['sd'] = float(x[np.isfinite(x)].std())
    return f


def p_error(x, fit, monotone=False):
    """One-sided P(step is wrong | score): 1 - lfdr above mu0, 0 at or below. monotone=True (F3): above mu0, the running
    max of 1 - lfdr over the values of x sorted by score (equal scores get equal values)."""
    x = np.asarray(x, float)
    if not fit['ok']:
        return np.full(len(x), np.nan)
    pe = np.where(x > fit['mu0'], 1. - fit['lfdr'](x), 0.)
    if monotone:
        up = np.flatnonzero(x > fit['mu0'])
        o = up[np.argsort(x[up], kind='stable')]
        pe[o] = np.maximum.accumulate(pe[o])
    return pe


def gate_g1(raw_fused, z_fused, off, monotone=False):
    fit = empirical_null(raw_fused)
    pe = p_error(raw_fused, fit, monotone)
    w = worst_step(z_fused, off)
    pw = pe[w]
    return dict(ok=bool(fit['ok']), reason=fit.get('reason', ''),
                clean=(pw < P_ERR_CUT) if fit['ok'] else np.zeros(len(w), bool), p_err_worst=pw, p_err_step=pe,
                fit={k: v for k, v in fit.items() if k != 'lfdr'})


def _group_null(raw_g, o_g, defined_g):
    """As v1 (u_g = o_g * pooled z over the group's defined steps, fitted on those only), with the v1b null."""
    u = np.zeros(len(raw_g))
    if defined_g.sum() < 50:
        return u, dict(ok=False, reason=f'{int(defined_g.sum())} defined steps')
    u[defined_g] = o_g * pooled_z(raw_g[defined_g])
    return u, empirical_null(u[defined_g])


def _group_perr(u, f, d, monotone):
    """P(err) of one group on its defined steps (0 elsewhere)."""
    pe = np.zeros(len(u))
    pe[d] = p_error(u[d], f, monotone)
    return pe


def gate_g2(raw_groups, orient, off, defined=None, monotone=False):
    """Product of group likelihood ratios; LR_g uses the group's own pi0 (F1). One-sided: LR = 1 at or below mu0_g and on
    undefined steps. The prior odds use the median of the identified groups' pi0, as in v1."""
    D = _defined(raw_groups, defined)
    fits, summ = [], []
    for g in np.flatnonzero(orient != 0):
        u, f = _group_null(raw_groups[:, g], orient[g], D[:, g])
        summ.append((g, f, D[:, g].mean()))
        if f['ok']:
            fits.append((g, f, u))
    if not fits:
        return dict(ok=False, reason='no group null identified', clean=np.zeros(len(off) - 1, bool),
                    groups=group_fit_summary(summ))
    pi0s = np.array([f['pi0'] for _, f, _ in fits])
    pi0 = float(np.clip(np.median(pi0s), *PI0_CLIP))
    log_odds = np.full(raw_groups.shape[0], np.log((1 - pi0) / pi0))
    for g, f, u in fits:
        l = np.maximum(1. - _group_perr(u, f, D[:, g], monotone), 1e-12)
        lr = np.clip((f['pi0'] / (1 - f['pi0'])) * (1 / l - 1), *LR_CLIP)
        log_odds += np.log(np.where((u > f['mu0']) & D[:, g], lr, 1.))
    pe = 1 / (1 + np.exp(-log_odds))
    pmax = answer_max(pe, off)
    return dict(ok=True, reason='', clean=pmax < P_ERR_CUT, p_err_step=pe, p_err_worst=pmax, pi0=pi0,
                pi0_groups=pi0s.tolist(), groups_used=[int(g) for g, _, _ in fits], groups=group_fit_summary(summ))


def gate_g3(raw_groups, orient, off, anchor_group=None, defined=None, monotone=False):
    """Answer-level DS over the group votes; constant voters are dropped (F4); needs >= 3 varying voters."""
    D = _defined(raw_groups, defined)
    votes, used, constant, summ = [], [], [], []
    for g in np.flatnonzero(orient != 0):
        u, f = _group_null(raw_groups[:, g], orient[g], D[:, g])
        summ.append((g, f, D[:, g].mean()))
        if not f['ok']:
            continue
        v = np.where(answer_max(_group_perr(u, f, D[:, g], monotone), off) >= P_ERR_CUT, 1., -1.)
        if np.all(v == v[0]):
            constant.append(int(g)); continue
        votes.append(v); used.append(int(g))
    if len(used) < 3:
        return dict(ok=False, reason=f'{len(used)} varying group votes (< 3); constant voters dropped: {constant}',
                    clean=np.zeros(len(off) - 1, bool), constant_voters=constant, groups=group_fit_summary(summ))
    Bv = np.column_stack(votes)
    anc = used.index(anchor_group) if (anchor_group is not None and anchor_group in used) else None
    fit = ds_em_votes(Bv, anchor=anc, init='majority')
    post = ds_answer_posterior(Bv, fit)
    M = np.cov(Bv.T); M = M - np.diag(np.diag(M)); v = np.linalg.eigh(M)[1][:, -1]; v = v if v.sum() >= 0 else -v
    return dict(ok=True, reason='', clean=post < P_ERR_CUT, p_answer=post, prevalence=float(fit['prevalence']),
                psi=np.asarray(fit['psi']).tolist(), eta=np.asarray(fit['eta']).tolist(), groups_used=used,
                constant_voters=constant, anchor_used=anc is not None, vote_rate=((Bv > 0).mean(0)).tolist(),
                sml_ranking=v.tolist(), groups=group_fit_summary(summ))
