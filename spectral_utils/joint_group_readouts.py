"""Within-group readouts for Joint L-SML that do not depend on the global loading v.

`hierarchical_joint_weights` (joint_lsml.py) forms each group's virtual classifier
as x_g @ v_g. When a foreign stream with a large global loading lands in a group of
complementary streams whose v entries are tiny, that stream dominates the group's
virtual classifier and the complementary streams lose their vote (Step 398:
Renyi a0.25 inside the digit group cut the digit share from .34 to .13).

Two label-free alternatives, both keeping the second stage (cross-group SML on
the virtual classifiers) unchanged:

* ``group_factor``   virtual_g = x_g @ u_g, the fitted within-group loading of the
                     Joint model (co-movement not explained by v). Sign of u_g is a
                     gauge of the model; it is fixed so that u_g . v_g >= 0, and when
                     v_g is numerically zero so that the largest |u| entry is positive.
* ``within_sml``     virtual_g = x_g @ e_g with e_g the leading eigenvector of the
                     within-group off-diagonal covariance (the same rule Continuous
                     L-SML uses inside groups). Uses no Joint loading at all; isolates
                     "Joint's partition + L-SML's readout".

No labels, no answers other than the fitting rows, no hand-supplied groups.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .fusion_utils import sml_fuse_signed
from .joint_lsml import canonicalize_labels

EPS = 1e-12
READOUTS = ("group_factor", "within_sml")


def _group_direction(x_g: np.ndarray, v_g: np.ndarray, u_g: np.ndarray, readout: str) -> tuple[np.ndarray, dict[str, Any]]:
    if readout == "group_factor":
        d = np.asarray(u_g, float).copy()
        dot = float(d @ v_g)
        if abs(dot) > EPS:
            if dot < 0:
                d = -d
        elif d[np.argmax(np.abs(d))] < 0:
            d = -d
        if float(np.abs(d).sum()) <= EPS:  # degenerate group loading: fall back to equal within group, declared
            d = np.ones_like(d) / len(d)
            return d, {"degenerate_group_loading": True}
        return d, {"degenerate_group_loading": False, "u_dot_v": dot}
    if readout == "within_sml":
        if x_g.shape[1] == 1:
            return np.ones(1), {"singleton": True}
        _, e = sml_fuse_signed(*[x_g[:, j] for j in range(x_g.shape[1])])
        return np.asarray(e, float), {"singleton": False}
    raise ValueError(readout)


def hierarchical_group_readout(values: np.ndarray, labels: Sequence[int], global_loading: Sequence[float],
                               group_loading: Sequence[float], *, readout: str,
                               small_m_guard: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-feature fusion weights: within-group direction times cross-group SML weight.

    Returns (weight, meta); orientation/normalization is left to the caller.
    """
    x = np.asarray(values, float)
    groups = canonicalize_labels(labels)
    v = np.asarray(global_loading, float); u = np.asarray(group_loading, float)
    if x.shape[1] != len(groups) or v.shape != (len(groups),) or u.shape != (len(groups),):
        raise ValueError("values/labels/loadings mismatch")
    directions, indices, virtual, notes = [], [], [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        d, note = _group_direction(x[:, idx], v[idx], u[idx], readout)
        directions.append(d); indices.append(idx); virtual.append(x[:, idx] @ d); notes.append(note)
    virtual = np.column_stack(virtual)
    _, cross = sml_fuse_signed(*[virtual[:, i] for i in range(virtual.shape[1])], small_m_guard=small_m_guard)
    weight = np.zeros(x.shape[1])
    for pos, (idx, d) in enumerate(zip(indices, directions)):
        weight[idx] = d * float(cross[pos])
    return weight, {"readout": readout, "cross_group_weights": np.asarray(cross, float),
                    "group_notes": notes, "virtual_classifier_count": int(virtual.shape[1])}


def self_test() -> bool:
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.standard_normal(n); digit = rng.standard_normal(n); fam = rng.standard_normal(n)
    # group 0: three "digit" streams sharing `digit`, plus one foreign stream that follows `signal`
    x = np.column_stack([
        digit + .3 * rng.standard_normal(n), digit + .3 * rng.standard_normal(n), digit + .3 * rng.standard_normal(n),
        signal + .3 * rng.standard_normal(n),
        signal + fam + .3 * rng.standard_normal(n), signal + fam + .3 * rng.standard_normal(n), signal + fam + .3 * rng.standard_normal(n),
        signal - fam + .3 * rng.standard_normal(n), signal - fam + .3 * rng.standard_normal(n), signal - fam + .3 * rng.standard_normal(n),
    ])
    x = (x - x.mean(0)) / x.std(0)
    labels = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    v = np.array([.05, .05, .05, .9, .8, .8, .8, .8, .8, .8])       # digit entries tiny in v
    u = np.array([.9, .9, .9, .0, .6, .6, .6, .6, .6, .6])          # digit co-movement lives in u
    for readout in READOUTS:
        w, meta = hierarchical_group_readout(x, labels, v, u, readout=readout)
        share_digit = np.abs(w[:3]).sum() / np.abs(w[:4]).sum()
        assert share_digit > .6, (readout, share_digit)   # digit streams keep the group's vote
        assert len(meta["cross_group_weights"]) == 3
    # the v-based rule, for contrast, hands group 0 to the foreign stream
    vw = v[:4]; assert np.abs(vw[:3]).sum() / np.abs(vw).sum() < .2
    return True


if __name__ == "__main__":
    print("self_test:", self_test())
