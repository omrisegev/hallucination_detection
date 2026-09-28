"""algorithm_decisions_v1 helpers (label-free): the latent-group EM ('hem') of the cumulative-vote line read as group-level and
within-group estimates, and group-score construction.  Protocol: results/algorithm_decisions_v1/PROTOCOL.json.

'hem' (cvf_v2/em.py): every group g has a binary latent state a_g with P(a_g = 1 | Y = y) = t[g, y], and each member channel i
votes +1 with probability e[i, a_g].  Channels of one group may depend on each other through a_g; different groups are
conditionally independent given Y.  After orienting a_g so that a_g = 1 is the state in which the members vote +1 more often:
  group estimate      psi_g = P(a_g = 1 | error), eta_g = P(a_g = 0 | clean)
  within-group weight w_i = max(0, log(e1 (1 - e0) / (e0 (1 - e1))))   (how well channel i tracks its group's state)
A latent state with fewer than `min_latent` members is weakly identified (it trades off against the emissions): such a group
takes the mean of its members' marginal psi / eta from the same fit and equal within-group weights (declared in the protocol).
"""
from __future__ import annotations

import numpy as np

import er_stage_a as SA
import er_stage_b as SB


def canon(g) -> np.ndarray:
    return np.unique(np.asarray(g, int), return_inverse=True)[1]


def hem_fit(votes: np.ndarray, groups, seed: int = 20260927, min_latent: int = 3) -> dict:
    """votes: steps x channels in {-1, +1}; groups: one label per channel.  Returns per-group psi, eta, pi, source
    ('latent' / 'member_marginal'), latent re-orientation flags, within-group weights (list; None = equal), per-channel
    marginal psi / eta, prevalence and EM diagnostics."""
    g = canon(groups)
    x = np.asarray(votes, float)
    if x.ndim != 2 or x.shape[1] != len(g) or not np.isin(x, [-1, 1]).all():
        raise ValueError('votes must be steps x channels in {-1, +1}, one group label per channel')
    core, em = SA._cvf()
    w = np.ones(len(x)); spectral = core.fit_spectral(x, w, 'spectral')
    model = em.fit_em(x, w, 'hem', spectral, groups=g, seed=seed)
    if model.status != 'ok':
        raise ArithmeticError(f'EM status {model.status}')
    c = 1 if model.orientation > 0 else 0
    e = np.asarray(model.emissions, float); t = np.asarray(model.transition, float)
    if not np.array_equal(np.asarray(model.groups, int), g):
        raise AssertionError('EM relabelled the groups')
    psi_m = e[:, 0] * (1 - t[g, c]) + e[:, 1] * t[g, c]; fpr_m = e[:, 0] * (1 - t[g, 1 - c]) + e[:, 1] * t[g, 1 - c]
    G = int(g.max()) + 1; psi = np.empty(G); eta = np.empty(G); source, flipped, within = [], [], []
    for h in range(G):
        idx = g == h
        flip = bool(np.mean(e[idx, 1] - e[idx, 0]) < 0)
        eo = e[idx][:, ::-1] if flip else e[idx]; to = 1 - t[h] if flip else t[h]
        if idx.sum() >= min_latent:
            psi[h] = to[c]; eta[h] = 1 - to[1 - c]; source.append('latent')
            wi = np.maximum(0.0, np.log(eo[:, 1]) + np.log1p(-eo[:, 0]) - np.log(eo[:, 0]) - np.log1p(-eo[:, 1]))
            within.append(wi if wi.sum() > 0 else None)
        else:
            psi[h] = psi_m[idx].mean(); eta[h] = (1 - fpr_m[idx]).mean(); source.append('member_marginal'); within.append(None)
        flipped.append(flip)
    return {'psi': psi, 'eta': eta, 'pi': (psi + eta) / 2, 'source': source, 'latent_flipped': flipped, 'within': within,
            'sizes': np.bincount(g, minlength=G).tolist(), 'channel_psi': psi_m, 'channel_eta': 1 - fpr_m,
            'prevalence': float(model.prior if c == 1 else 1 - model.prior), 'converged': bool(model.diagnostics['converged']),
            'boundary_emissions': int(model.diagnostics['boundary_emissions'])}


def group_matrix(X: np.ndarray, off: np.ndarray, groups, within, standardize) -> np.ndarray:
    """Column g = standardize(X[:, members of g] @ w_g), w_g = within[g] normalized to sum |w| = 1, or the plain mean when
    within is None or within[g] is None (then it equals er_stage_b.group_scores)."""
    g = canon(groups); G = int(g.max()) + 1; cols = []
    for h in range(G):
        idx = np.flatnonzero(g == h); wh = None if within is None else within[h]
        if wh is None:
            cols.append(X[:, idx].mean(1))
        else:
            wh = np.asarray(wh, float)
            if wh.shape != (len(idx),) or not np.isfinite(wh).all() or np.abs(wh).sum() <= 0:
                raise ValueError(f'bad within-group weights for group {h}')
            cols.append(X[:, idx] @ (wh / np.abs(wh).sum()))
    return standardize(np.column_stack(cols), off)


def weighted_group_score(Z: np.ndarray, w: np.ndarray, signed: bool = False) -> np.ndarray:
    """Z (steps x groups) times weights normalized to sum |w| = 1.  Unsigned weights must be nonnegative; all-zero raises."""
    w = np.asarray(w, float)
    if w.shape != (Z.shape[1],) or not np.isfinite(w).all() or (not signed and (w < 0).any()):
        raise ValueError('weights must be finite, one per group (nonnegative unless signed)')
    if np.abs(w).sum() <= 0:
        raise ValueError('all group weights are 0')
    return Z @ (w / np.abs(w).sum())


def mle_group_weights(psi, eta) -> np.ndarray:
    """= er_stage_b.mle_weights: max(0, log(psi eta / ((1-psi)(1-eta)))); a group no better than chance gets 0."""
    return SB.mle_weights(psi, eta)
