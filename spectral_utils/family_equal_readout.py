"""Partition-then-equal readouts for the seven CT7 step views (item 2, 2026-09-23).

Rule (Step 399 addendum, `docs/JOINT_LSML_METHOD_CARD_2026-09-17.md` appendix): on the Joint
partitions, "mean within group + equal weight per standardized group" reproduces the leading
arm, within-group SML eigenvectors are near-uniform, and a real cross-group eigen-solve costs
about 3 pp. The active ingredient is the partition. CT7 averages its seven views 1/7 each,
which gives the entropy-level family 5/7 of the vote; this module gives each declared family
one vote.

Three scalings of the family means, all label-free:

* ``fixed_partition_weights``   1/(K |g|) per view: the "partition only" arm, a fixed linear
                                combination with no standardization of the family means.
* ``answer_restandardize``      each family mean standardized within the answer (mean/sd,
                                constant families zeroed): a fixed rule with an answer-adaptive
                                scale. On a 2-step answer every family becomes +-1 and the fused
                                score is a majority of three signs.
* ``fold_scaled``               each family mean divided by its training-fold sd (the ladder's
                                ``ceq`` rule, `scripts/run_digitfree20_ladder_v1.py::readout_weights`);
                                pooled-donor access, declared.

``cross_eigen`` is the unguarded cross-family eigen-solve (the comparator the method card says
loses), and ``discover_partition`` is the affinity + fold-deletion stability route of the
Step 413 ladder with the minimum group size lowered to one so that a singleton family is
admissible. Nothing in this module reads labels.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .digitfree_broad50 import masked_answer_standardize
from .fusion_utils import _spectral_cluster_precomputed, sml_fuse_signed
from .joint_lsml import canonicalize_labels, covariance_matrix, residual_affinity

FAMILIES_421 = {"A_level": [0, 1, 2, 3], "B_temporal": [4, 5], "C_chosen_token": [6]}
FAMILIES_511 = {"A_level": [0, 1, 2, 3, 4], "B_temporal": [5], "C_chosen_token": [6]}


def _groups_from_families(families: dict[str, Sequence[int]], m: int) -> np.ndarray:
    labels = np.full(m, -1, int)
    for k, (_, members) in enumerate(families.items()):
        labels[np.asarray(members, int)] = k
    if (labels < 0).any():
        raise ValueError("every view must belong to exactly one family")
    return labels


def family_means(profiles: np.ndarray, groups: Sequence[int]) -> np.ndarray:
    """[steps x K] mean of the member views of each group (group ids 0..K-1)."""
    x = np.asarray(profiles, float); g = np.asarray(groups, int)
    if x.ndim != 2 or g.shape != (x.shape[1],):
        raise ValueError("profiles must be [steps x views] with one group id per view")
    return np.column_stack([x[:, g == k].mean(1) for k in range(int(g.max()) + 1)])


def fixed_partition_weights(groups: Sequence[int]) -> np.ndarray:
    """w_j = 1 / (K |g_j|): the mean of family means as one fixed linear combination."""
    g = np.asarray(groups, int); K = int(g.max()) + 1
    sizes = np.bincount(g, minlength=K)
    return 1.0 / (K * sizes[g])


def answer_restandardize(x: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, dict]:
    """Within-answer mean/sd per column (masked to finite values); constant columns -> 0.

    Returns (standardized, info) with the number of zeroed (answer, column) pairs and the
    number of 2-step answers, where the standardized value is exactly +-1.
    """
    x = np.asarray(x, float); off = np.asarray(offsets, int)
    finite = np.isfinite(x)
    z = masked_answer_standardize(x, finite, off)
    zeroed = 0
    for a, b in zip(off[:-1], off[1:]):
        for j in range(x.shape[1]):
            v = x[a:b, j][finite[a:b, j]]
            if len(v) and v.std() <= 1e-12:
                zeroed += 1
    two_step = int(np.sum(np.diff(off) == 2))
    return z, {"zeroed_family_answers": int(zeroed), "two_step_answers": two_step}


def fold_scaled(x: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide each column by its sd over the training rows (ddof=1); returns (scaled, sd)."""
    x = np.asarray(x, float)
    sd = x[np.asarray(train_rows, bool)].std(0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return x / sd, sd


def cross_eigen(virtuals: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unguarded SML eigen-solve across the family virtuals, fitted on the training rows.

    Sign gauge: the fused score is oriented to correlate positively with the equal-weight mean
    of the virtuals (label-free). Returns (scores over all rows, weights).
    """
    v = np.asarray(virtuals, float); tr = np.asarray(train_rows, bool)
    _, w = sml_fuse_signed(*[v[tr, k] for k in range(v.shape[1])], small_m_guard=False)
    w = np.asarray(w, float)
    score = v @ w
    ref = v.mean(1)
    if np.corrcoef(score[tr], ref[tr])[0, 1] < 0:
        w = -w; score = -score
    return score, w


def discover_partition(train: np.ndarray, row_folds: np.ndarray, *, k_range: Sequence[int] = (2, 3, 4),
                       min_size: int = 1, seed: int = 20260923) -> tuple[np.ndarray, dict]:
    """Residual-affinity spectral partition with block-deletion stability (ladder route).

    Tailored copy of `scripts/run_digitfree20_ladder_v1.py::partitions_affinity`: affinity is
    |C - v v^T| from the training covariance, partitions at every K are compared with the
    partitions after deleting each fold block (adjusted Rand), the smallest-size rule is
    lowered to ``min_size`` (default 1, so a singleton family is admissible), and the selected
    K maximizes median ARI, then minimum ARI, then prefers the smaller K.
    """
    from sklearn.metrics import adjusted_rand_score
    x = np.asarray(train, float); rf = np.asarray(row_folds)
    full = residual_affinity(covariance_matrix(x))[0]
    deletions = [residual_affinity(covariance_matrix(x[rf != f]))[0] for f in np.unique(rf)]
    m = x.shape[1]; out = {}
    for k in k_range:
        if k >= m:
            continue
        labels = canonicalize_labels(_spectral_cluster_precomputed(full, k, seed=seed + 1000 * k))
        parts = [canonicalize_labels(_spectral_cluster_precomputed(a, k, seed=seed + 1000 * k + j))
                 for j, a in enumerate(deletions)]
        ari = [float(adjusted_rand_score(labels, p)) for p in parts]
        sizes = [int(np.sum(labels == g)) for g in np.unique(labels)]
        out[int(k)] = {"labels": labels.tolist(), "sizes": sizes, "median_ari": float(np.median(ari)),
                       "min_ari": float(min(ari)), "admissible": bool(len(sizes) == k and min(sizes) >= min_size)}
    adm = [k for k in out if out[k]["admissible"]]
    if not adm:
        raise ValueError("no admissible partition")
    selected = sorted(adm, key=lambda k: (-out[k]["median_ari"], -out[k]["min_ari"], k))[0]
    return np.asarray(out[selected]["labels"], int), {"selected_k": int(selected), "candidates": out}


def partition_scores(profiles: np.ndarray, groups: Sequence[int], offsets: np.ndarray, *,
                     rule: str, train_rows: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Fused per-step score for one partition and one scaling rule (`raw`, `answer`, `fold`, `eigen`)."""
    g = np.asarray(groups, int)
    V = family_means(profiles, g)
    if rule == "raw":
        return np.asarray(profiles, float) @ fixed_partition_weights(g), {"weights": fixed_partition_weights(g).tolist()}
    if rule == "answer":
        z, info = answer_restandardize(V, offsets)
        return z.mean(1), info
    if train_rows is None:
        raise ValueError("fold and eigen rules need training rows")
    if rule == "fold":
        scaled, sd = fold_scaled(V, train_rows)
        return scaled.mean(1), {"family_sd_train": sd.tolist()}
    if rule == "eigen":
        score, w = cross_eigen(V, train_rows)
        return score, {"cross_weights": w.tolist()}
    raise ValueError(rule)


def self_test(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_answers, m = 300, 7
    steps = rng.integers(2, 12, n_answers); off = np.concatenate([[0], np.cumsum(steps)])
    N = int(off[-1])
    # Joint-structured plant: one shared factor in every view plus a family factor for A and B;
    # the seventh view carries the shared factor and noise only (a singleton family).
    s, fa, fb = rng.standard_normal((3, N))
    x = np.column_stack([s + .8 * fa + .3 * rng.standard_normal(N) for _ in range(4)] +
                        [s + .8 * fb + .6 * rng.standard_normal(N) for _ in range(2)] +
                        [s + rng.standard_normal(N)])
    x = masked_answer_standardize(x, np.ones(x.shape, bool), off)
    folds = np.repeat(np.arange(n_answers) % 5, steps)
    labels, info = discover_partition(x, folds, k_range=(2, 3, 4), min_size=1)
    # The residual affinity |C - vv^T| removes the shared factor and clusters the family
    # factors: A and B come out as blocks. Several K are perfectly stable, so the tie-break
    # prefers the smallest K and the singleton is absorbed rather than isolated (K = 2). That
    # is the documented behaviour of the route (method card, Step 399 addendum), not a defect.
    assert len(set(labels[:4])) == 1 and len(set(labels[4:6])) == 1 and labels[0] != labels[4], (labels, info["selected_k"])
    assert info["selected_k"] == 2 and info["candidates"][2]["median_ari"] == 1.0
    g = _groups_from_families(FAMILIES_421, m)
    # fixed weights: 1/12 x4, 1/6 x2, 1/3
    w = fixed_partition_weights(g)
    assert np.allclose(w, [1/12] * 4 + [1/6] * 2 + [1/3])
    # copies: when all seven views are copies, the raw rule equals the 1/7 mean exactly and the
    # answer rule has the same argmax and within-answer order
    c = np.repeat(x[:, :1], m, axis=1)
    raw, _ = partition_scores(c, g, off, rule="raw")
    assert np.allclose(raw, c.mean(1))
    ans, inf = partition_scores(c, g, off, rule="answer")
    for a, b in zip(off[:-1], off[1:]):
        assert np.argmax(ans[a:b]) == np.argmax(c[a:b].mean(1))
        assert np.array_equal(np.argsort(ans[a:b], kind="stable"), np.argsort(c[a:b].mean(1), kind="stable"))
    # scale invariance of the answer rule and +-1 on 2-step answers
    ans2, _ = partition_scores(3.0 * x, g, off, rule="answer")
    ans1, _ = partition_scores(x, g, off, rule="answer")
    assert np.allclose(ans1, ans2)
    two = np.flatnonzero(steps == 2)
    for i in two[:5]:
        a, b = off[i], off[i + 1]
        z, _ = answer_restandardize(family_means(x[a:b], g), np.array([0, 2]))
        # +-1 per family, or exactly 0 when two standardized members cancel (constant family)
        assert np.all(np.isclose(np.abs(z), 1.0) | np.isclose(z, 0.0))
    # fold rule: dividing by the training sd, then mean; eigen: orientation positive vs equal
    tr = folds != 0
    fs, sd = fold_scaled(family_means(x, g), tr)
    assert np.allclose(fs[tr].std(0, ddof=1), 1.0)
    es, ew = cross_eigen(family_means(x, g), tr)
    assert np.corrcoef(es[tr], family_means(x, g)[tr].mean(1))[0, 1] > 0
    return {"selected_k": info["selected_k"], "zeroed": inf["zeroed_family_answers"], "two_step": inf["two_step_answers"]}


if __name__ == "__main__":
    print("self_test:", self_test())
