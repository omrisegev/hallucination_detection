#!/usr/bin/env python
"""Per-cell screen: is there anything for a per-cell fit to fit differently?

Amendment 2 of the Stage B pre-registration. Per-cell adaptation means fitting the
covariance and L-SML separately for each of the nine cells, label-free, using only cell
identity -- information we hold at scoring time. Before building that, this screen asks
whether the nine cells actually have different covariance structure.

**The object compared is the MARGINAL correlation matrix, not a conditional one, and
that is the point rather than a compromise.** L-SML never sees labels: what a per-cell
fit would fit is the per-cell marginal covariance of the views. If those matrices are
the same across cells, a per-cell fit has nothing different to fit and the variant
cannot help, whatever any accuracy table says.

"Nearly identical" needs a scale, so between-cell distance is judged against a
within-cell sampling baseline: each cell is split in half by SOURCE GROUP (never by
answer, which would put the same question on both sides) and the same distance computed
between its own halves. Between-cell distances at or below that baseline are noise.

Distance is the mean absolute difference over the 55 off-diagonal entries of the 11x11
correlation matrix -- in correlation units, so it reads directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
OUT = RES / "PER_CELL_COVARIANCE_SCREEN.json"

from spectral_utils.claude_feature_bank_v1 import FEATURE_NAMES  # noqa: E402
from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402

SEED = 20260918


def corr(matrix: np.ndarray) -> np.ndarray:
    keep = matrix.std(0) > 1e-12
    c = np.eye(matrix.shape[1])
    if keep.sum() >= 2:
        sub = np.corrcoef(matrix[:, keep].T)
        idx = np.flatnonzero(keep)
        c[np.ix_(idx, idx)] = sub
    return c


def offdiag_distance(a: np.ndarray, b: np.ndarray) -> float:
    iu = np.triu_indices(a.shape[0], k=1)
    return float(np.mean(np.abs(a[iu] - b[iu])))


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    groups = np.asarray([r["group_id"] for r in records], str)

    with np.load(RES / "DERIVATIVE_CHANNELS.npz", allow_pickle=False) as z:
        level = z["level"].copy()

    # Answer-standardized step views: the representation a per-cell fit would act on.
    views = masked_answer_standardize(level, np.isfinite(level), offsets)
    step_cell = np.repeat(cells, np.diff(offsets))
    step_group = np.repeat(groups, np.diff(offsets))

    names = sorted(set(cells))
    per_cell = {c: corr(views[step_cell == c]) for c in names}
    pooled = corr(views)

    rng = np.random.default_rng(SEED)
    baseline = {}
    for c in names:
        g = np.unique(step_group[step_cell == c])
        pick = rng.permutation(len(g))
        left = set(g[pick[: len(g) // 2]])
        m = step_cell == c
        is_left = np.array([x in left for x in step_group[m]])
        sub = views[m]
        baseline[c] = offdiag_distance(corr(sub[is_left]), corr(sub[~is_left]))

    between = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            between[f"{a} vs {b}"] = offdiag_distance(per_cell[a], per_cell[b])
    vs_pooled = {c: offdiag_distance(per_cell[c], pooled) for c in names}

    med_base = float(np.median(list(baseline.values())))
    med_between = float(np.median(list(between.values())))
    ratio = med_between / med_base if med_base > 0 else float("inf")

    report = {
        "schema": "token-probability-fusion-v1-per-cell-screen",
        "object": "marginal correlation of answer-standardized step views (what L-SML fits)",
        "channels": list(FEATURE_NAMES),
        "within_cell_splithalf_baseline": baseline,
        "between_cell_distance": between,
        "distance_to_pooled": vs_pooled,
        "median_within_baseline": med_base,
        "median_between": med_between,
        "ratio_between_over_within": ratio,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    print("=" * 84)
    print("PER-CELL COVARIANCE SCREEN -- mean |difference| over the 55 off-diagonal entries")
    print("=" * 84)
    print(f"{'cell':26s} {'split-half baseline':>20s} {'distance to pooled':>20s}")
    for c in names:
        print(f"{c:26s} {baseline[c]:20.4f} {vs_pooled[c]:20.4f}")
    print()
    print(f"median within-cell split-half baseline : {med_base:.4f}")
    print(f"median between-cell distance           : {med_between:.4f}")
    print(f"ratio                                  : {ratio:.2f}x")
    print()
    top = sorted(between.items(), key=lambda kv: -kv[1])[:6]
    print("largest between-cell differences:")
    for k, v in top:
        print(f"  {k:52s} {v:.4f}")
    print()
    if ratio < 1.5:
        print("READING: between-cell structure is at the level of sampling noise.")
        print("A per-cell fit has essentially nothing different to fit; do not build it.")
    else:
        print("READING: cells differ by more than sampling noise; a per-cell fit has")
        print("something to act on. This bounds nothing about whether it HELPS.")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
