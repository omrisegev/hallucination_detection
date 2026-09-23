"""8-token window moment bank for the window-representation measurement (item 4 / atlas B3).

The original moment-bank code (`answer_localization_v2.py`, `fusion_context_bank.py`) did not
survive on any branch; this rebuilds the small part B3 needs. Views are ``<stream>__<op>`` with
op in {level, sd, slope} over width-8 windows (the name format `shrinkage_iu.partition` parses).

Geometry reuses `window_localization.WindowPlan`, `windows_to_tokens` and
`tokens_to_official_steps` unchanged; `make_window_plan` refuses widths below 32 (an STFT
guard that a test asserts), so the plan is constructed here directly with the same fields:
dense scoring windows at the stride, plus the non-overlapping fit grid, plus a full-width window
anchored at the end, never padded.

Also here: the window-label rule (a window is labelled only when it lies inside one official
step), the effective-sample correction of the Ledoit-Wolf alpha under autocorrelated windows,
and the distance of a weight vector from equal weight. No labels are read by any builder.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .step_readouts_v1 import step_mean, step_slope, step_std
from .window_localization import WindowPlan, tokens_to_official_steps, windows_to_tokens

OPS = {"level": step_mean, "sd": step_std, "slope": step_slope}
DEFAULT_WIDTH = 8


def window_plan(token_count: int, width: int = DEFAULT_WIDTH, stride: int = 1) -> WindowPlan:
    """Dense scoring windows (stride) + non-overlapping fit grid (every `width`), width-8 legal."""
    token_count, width, stride = int(token_count), int(width), int(stride)
    if width < 2 or stride < 1 or stride > width:
        raise ValueError("width >= 2 and 1 <= stride <= width")
    if token_count < width:
        raise ValueError("TRACE_TOO_SHORT_FOR_WINDOWS")
    fit_starts = np.arange(0, token_count - width + 1, width, dtype=np.int64)
    starts = np.unique(np.concatenate((np.arange(0, token_count - width + 1, stride, dtype=np.int64), fit_starts,
                                       np.asarray([token_count - width], dtype=np.int64))))
    fit_indices = np.searchsorted(starts, fit_starts)
    return WindowPlan(token_count, width, stride, starts, starts + width, fit_indices)


def moment_bank(tokens: np.ndarray, names: Sequence[str], plan: WindowPlan,
                views: Sequence[tuple[str, str]]) -> tuple[np.ndarray, list[str]]:
    """[W x p] window values for the declared (stream, op) views; names ``<stream>__<op>``."""
    x = np.asarray(tokens, float)
    if x.ndim != 2 or x.shape[0] != plan.token_count or len(names) != x.shape[1]:
        raise ValueError("tokens must be [token_count x streams] aligned with names")
    spans = np.column_stack([plan.starts, plan.ends])
    cols, out = [], []
    cache = {}
    for stream, op in views:
        j = list(names).index(stream)
        if op not in cache:
            cache[op] = OPS[op](x, spans)
        out.append(cache[op][:, j]); cols.append(f"{stream}__{op}")
    return np.column_stack(out), cols


def window_labels(plan: WindowPlan, step_spans: np.ndarray, step_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label of the official step containing the whole window; `inside` False for straddlers."""
    spans = np.asarray(step_spans, int); lab = np.asarray(step_labels)
    y = np.zeros(len(plan.starts), lab.dtype); inside = np.zeros(len(plan.starts), bool)
    step_of = np.full(plan.token_count, -1, int)
    for s, (a, b) in enumerate(spans):
        step_of[a:b] = s
    for w, (a, b) in enumerate(zip(plan.starts, plan.ends)):
        s0, s1 = step_of[a], step_of[b - 1]
        if s0 >= 0 and s0 == s1:
            inside[w] = True; y[w] = lab[s0]
    return y, inside


def alpha_neff(alpha_lw: float, n: int, n_eff: float) -> float:
    """Ledoit-Wolf alpha scales with 1/n; with dependent windows use n_eff in its place."""
    if n_eff <= 0:
        return 1.0
    return float(min(1.0, alpha_lw * n / n_eff))


def distance_from_equal(w: np.ndarray) -> dict:
    """Cosine and L2 distance between the normalized weight vector and 1/p."""
    w = np.asarray(w, float); p = len(w)
    if np.abs(w).sum() <= 1e-12:
        return {"cosine": float("nan"), "l2": float("nan")}
    u = np.full(p, 1.0 / p)
    wn = w / np.linalg.norm(w)
    return {"cosine": float(wn @ (u / np.linalg.norm(u))), "l2": float(np.linalg.norm(w / np.abs(w).sum() - u))}


def conditional_participation_ratio(views: np.ndarray, y: np.ndarray) -> float:
    """(sum lambda)^2 / sum(lambda^2) of the within-label-centred correlation (the 1.80 statistic;
    copied from scripts/diagnostics/stage_b_2x2_v1.py without the validity argument)."""
    m = np.array(views, float, copy=True)
    for cls in (True, False):
        sel = np.asarray(y, bool) == cls
        if sel.any():
            m[sel] -= m[sel].mean(0)
    keep = m.std(0) > 1e-12
    if keep.sum() < 2:
        return float(keep.sum())
    lam = np.maximum(np.linalg.eigvalsh(np.corrcoef(m[:, keep].T)), 0.0)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def self_test(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    T = 61; x = rng.standard_normal((T, 3)); names = ["a", "b", "c"]
    plan = window_plan(T, 8, 1)
    assert plan.starts[0] == 0 and plan.ends[-1] == T and len(plan.fit_indices) == T // 8
    assert np.all(np.diff(plan.starts[plan.fit_indices]) == 8)
    V, cols = moment_bank(x, names, plan, [("a", "level"), ("a", "sd"), ("a", "slope"), ("b", "level")])
    assert cols == ["a__level", "a__sd", "a__slope", "b__level"] and V.shape == (len(plan.starts), 4)
    # literal loop
    for w, (a, b) in enumerate(zip(plan.starts, plan.ends)):
        seg = x[a:b, 0]; t = np.arange(8) - 3.5
        assert np.isclose(V[w, 0], seg.mean()) and np.isclose(V[w, 1], seg.std()) and np.isclose(V[w, 2], t @ (seg - seg.mean()) / (t @ t))
    # round trip: constant window scores -> constant tokens -> constant steps
    tok = windows_to_tokens(plan, np.full(len(plan.starts), 2.0)); assert np.allclose(tok, 2.0)
    steps = tokens_to_official_steps(tok, np.array([0, 20, 40]), np.array([20, 40, 61])); assert np.allclose(steps, 2.0)
    # window labels: only windows inside one step are labelled
    y, inside = window_labels(plan, np.array([[0, 20], [20, 40], [40, 61]]), np.array([0, 1, 0]))
    assert inside.sum() == (13 + 13 + 14) and y[inside].sum() == 13
    # participation ratio: independent columns -> p, copies -> 1; the per-channel shuffle of copies
    # restores independence (near p): the shuffle is an UPPER reference, not a floor
    n, p = 4000, 6; z = rng.standard_normal((n, p)); yy = rng.random(n) < .3
    assert conditional_participation_ratio(z, yy) > p - 0.3
    c = np.repeat(z[:, :1], p, axis=1) + 1e-3 * rng.standard_normal((n, p))
    assert conditional_participation_ratio(c, yy) < 1.01
    sh = np.column_stack([rng.permutation(c[:, j]) for j in range(p)])
    assert conditional_participation_ratio(sh, yy) > p - 0.5
    assert alpha_neff(0.2, 100, 25.0) == 0.8 and alpha_neff(0.5, 100, 25.0) == 1.0
    dq = distance_from_equal(np.ones(5)); assert np.isclose(dq["cosine"], 1.0) and np.isclose(dq["l2"], 0.0)
    return {"windows": int(len(plan.starts)), "fit_windows": int(len(plan.fit_indices))}


if __name__ == "__main__":
    print("self_test:", self_test())
