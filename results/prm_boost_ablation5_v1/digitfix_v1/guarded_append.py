

# --- appended 2026-10-19 from claude/prm-boosting 466ddc3d (spectral_utils/fusion_utils.py), the only change
# --- shipped to this mirror: gates is passed only when given, because this mirror's lsml_continuous predates it.
def lsml_continuous_guarded(*views, groups=None, tol: float = 1e-9, **kwargs):
    """lsml_continuous after dropping constant views; dropped views get weight 0.

    A constant view (sd <= tol over all samples) has zero covariance with every other view. In
    detect_dependent_groups it becomes an isolated node that still takes one of the K slots, so it can
    change how the informative views are partitioned (Math-Shepherd N16, where three digit channels
    were exactly zero: 4 telemetry groups became 5 plus a zero group). This wrapper fits L-SML on the
    non-constant views only. Without constant views the call and its outputs are exactly lsml_continuous.

    Args:
        *views, groups, **kwargs: as for lsml_continuous. ``groups`` covers all views; the dropped
            views' entries are ignored.
        tol: views with population sd <= tol are dropped.

    Returns:
        (fused_scores, meta) as lsml_continuous, with meta['c'] of length m (-1 for a dropped view),
        group_weights indices in the original view order, and meta['dropped_constant'] /
        meta['kept_views'] (original indices). Raises ValueError if fewer than 3 views remain.
    """
    X = np.column_stack(views)
    m = X.shape[1]
    sd = X.std(axis=0)
    if not np.isfinite(sd).all():
        raise ValueError("lsml_continuous_guarded: non-finite view values")
    keep = np.flatnonzero(sd > tol)
    drop = np.flatnonzero(sd <= tol)
    if len(keep) < 3:
        raise ValueError(f"lsml_continuous_guarded: only {len(keep)} non-constant views of {m}; "
                         "L-SML needs at least 3")
    if groups is not None:
        groups = np.asarray(groups, dtype=int)
        if len(groups) != m:
            raise ValueError(f"groups has length {len(groups)}, expected m={m}")
    if not len(drop):
        fused, meta = lsml_continuous(*views, groups=groups, **kwargs)
        meta = dict(meta, dropped_constant=[], kept_views=keep.tolist())
        return fused, meta
    gates = kwargs.pop('gates', None)
    if gates is not None:
        gates = np.asarray(gates, dtype=float)[keep]
    fused, meta = lsml_continuous(*[X[:, j] for j in keep],
                                  groups=None if groups is None else groups[keep],
                                  **({'gates': gates} if gates is not None else {}), **kwargs)
    c = np.full(m, -1, dtype=int)
    c[keep] = np.asarray(meta['c'], dtype=int)
    meta = dict(meta, c=c,
                group_weights=[(keep[np.asarray(idx, dtype=int)], w) for idx, w in meta['group_weights']],
                dropped_constant=drop.tolist(), kept_views=keep.tolist())
    return fused, meta
