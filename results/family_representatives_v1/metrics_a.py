"""Label-free structure metrics (spec section 11-A) for a step-feature matrix.

Every metric here reads only the feature matrix and the source-group fold ids
(fold ids are built from question identity, not from correctness). No label
file is opened in this module.

S1  between-group rank-1 residual: share of squared cross-group correlation mass
    left after the best rank-one fit r_ij ~ v_i v_j (groups = L-SML's own).
S1g between-group rank-1 residual among the K virtual classifiers (K >= 4).
S2  within-group rank-1 residual: same, inside each L-SML group of >= 3 members,
    pooled by squared mass.
S3  effective number of groups: 1 / sum(share_k^2), share_k = |w_k| * sd(virtual_k)
    normalized, i.e. each group's scale-aware contribution to the fused score.
S4  identifiability: K and the number of groups with fewer than 3 members.
S5  redundancy: pairs with |r| >= .95; effective rank / number of channels.
S6  sign violations: channels whose effective L-SML weight disagrees in sign with
    their loading on the bank's leading off-diagonal eigenvector, plus negative
    cross-group weights.
S7  stability over the 5 source-group folds: L-SML fitted on each fold's training
    part; mean pairwise ARI of the groupings and mean pairwise Spearman of the
    effective channel weights.
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import spectral_utils.fusion_utils as FU  # noqa: E402


def masked_rank1_share(R, mask, iters=300):
    """Share of sum_{mask} r_ij^2 not explained by v_i v_j (ALS on masked entries)."""
    Rm = np.where(mask, R, 0.0)
    tot = float((Rm ** 2).sum())
    if tot <= 0:
        return float('nan')
    w, U = eigh(Rm); v = U[:, -1] * np.sqrt(max(w[-1], 1e-9))
    for _ in range(iters):
        v = (Rm @ v) / np.maximum((mask * v[None, :] ** 2).sum(1), 1e-12)
    return float((np.where(mask, R - np.outer(v, v), 0.0) ** 2).sum() / tot)


def fit_lsml(X):
    s, meta = FU.lsml_continuous(*[X[:, j] for j in range(X.shape[1])])
    c = np.asarray(meta['c'], int)
    eff = np.zeros(X.shape[1])
    for k, (idx, w) in enumerate(meta['group_weights']):
        eff[idx] = np.asarray(w, float) * float(np.asarray(meta['cross_weights'])[k])
    return np.asarray(s, float), c, eff, meta


def effective_rank(R):
    w = np.clip(np.linalg.eigvalsh(R), 0, None); p = w / w.sum()
    return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


def structure_metrics(X, folds, names=None):
    X = np.asarray(X, float); m = X.shape[1]
    R = np.corrcoef(X, rowvar=False)
    score, c, eff, meta = fit_lsml(X)
    off = ~np.eye(m, dtype=bool)
    between = (c[:, None] != c[None, :])
    s1 = masked_rank1_share(R, between)
    num = den = 0.0
    for g in np.unique(c):
        idx = np.flatnonzero(c == g)
        if len(idx) < 3:
            continue
        sub = R[np.ix_(idx, idx)]; mk = ~np.eye(len(idx), dtype=bool)
        t = float((np.where(mk, sub, 0) ** 2).sum())
        num += masked_rank1_share(sub, mk) * t; den += t
    s2 = num / den if den > 0 else float('nan')
    cw = np.asarray(meta['cross_weights'], float)
    VC = np.asarray(meta['virtual_classifiers'], float)
    sd = VC.std(0)
    # S1g: the cross-group stage of L-SML sees only the K virtual classifiers; its model is
    # a rank-one off-diagonal correlation among them. Unlike S1 this does not depend on how
    # many channels each group holds (copies inflate S1's explained mass). K=3 fits exactly.
    Rg = np.corrcoef(VC, rowvar=False) if VC.shape[1] > 1 else np.ones((1, 1))
    s1g = masked_rank1_share(Rg, ~np.eye(len(Rg), dtype=bool)) if len(Rg) >= 4 else float('nan')
    share = np.abs(cw) * sd; share = share / share.sum()
    s3 = float(1.0 / (share ** 2).sum())
    sizes = np.bincount(c)[np.unique(c)]
    lam, U = eigh(R - np.diag(np.diag(R))); lead = U[:, -1]
    if (lead > 0).sum() < m / 2:
        lead = -lead
    s6_ch = int(((np.sign(eff) != np.sign(lead)) & (np.abs(eff) > 0)).sum())
    s6_cross = int((cw < 0).sum()) if len(cw) > 1 else 0
    pairs = int((np.abs(R[off]) >= .95).sum() // 2)
    # S7: fold stability
    groupings, weights = [], []
    for f in np.unique(folds):
        _, cf, ef, _ = fit_lsml(X[folds != f])
        groupings.append(cf)
        sg = np.sign(ef @ eff) or 1.0  # global sign is not identified; align to the full fit
        weights.append(ef * sg)
    ari = [adjusted_rand_score(a, b) for a, b in itertools.combinations(groupings, 2)]
    rho = [spearmanr(a, b)[0] for a, b in itertools.combinations(weights, 2)]
    out = dict(
        m=m, S1_between_rank1_residual=s1, S1g_group_rank1_residual=s1g, S2_within_rank1_residual=s2,
        S3_effective_groups=s3, S3_group_shares=[round(float(x), 4) for x in share],
        S4_K=int(meta['K']), S4_groups_lt3=int((sizes < 3).sum()), S4_group_sizes=[int(x) for x in sizes],
        S5_pairs_ge_095=pairs, S5_eff_rank=effective_rank(R), S5_eff_rank_ratio=effective_rank(R) / m,
        S6_channel_sign_violations=s6_ch, S6_negative_cross_weights=s6_cross,
        S7_fold_ari=float(np.mean(ari)), S7_fold_weight_spearman=float(np.nanmean(rho)),
        degenerate=bool(meta['degenerate']),
    )
    if names is not None:
        out['groups'] = {int(g): [names[i] for i in np.flatnonzero(c == g)] for g in np.unique(c)}
        out['effective_weights'] = {names[i]: round(float(eff[i]), 5) for i in range(m)}
    return out, score
