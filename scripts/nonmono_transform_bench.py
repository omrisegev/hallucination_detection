#!/usr/bin/env python
"""
nonmono_transform_bench.py — do the "non-monotone" feature transforms survive an
honest measurement? (Step 217)

WHAT THIS REPLACES
------------------
`scripts/sweep_feature_transforms.py` (Gemini, commit 9770278) reported +0.54pp
from replacing/augmenting five features it called non-monotone. Review found four
defects that void that measurement, all reproduced and confirmed against the repo's
own artifacts — and building this file found a fifth, in OUR code, which is the one
that actually decides the question (see defect 5 below):

  1. `sweep_feature_transforms.py:19-26` — `safe_auc` returns the larger of
     AUROC(s) and AUROC(-s), i.e. it resolves the GLOBAL SIGN WITH THE LABELS. The
     canonical path (`inscope_bench_common.py:20`) is `anchor_orient` + raw AUROC,
     "never max(a, 1-a)". A label-resolved sign can only ever inflate.
  2. `sweep_feature_transforms.py:71` — the objective being MAXIMISED is computed on
     the wrong cells. `is_prob = 'math500' in fname or 'qa' in fname` selects 3 GPQA
     pkls (out of scope since Step 191; every feature 0.51-0.55, i.e. at chance) and
     5 duplicate copies of one MATH-500 run, while `repgrid_cells.pkl` — where 23 of
     the then-25 in-scope cells live — is classed 'Other'. ZERO in-scope cells are in
     the set being optimised.
  3. `sweep_feature_transforms.py:106` vs `:113` — the transform is chosen per feature
     by argmax on exactly the data the final number is then reported on. No held-out
     anything.
  4. `sweep_feature_transforms.py:17` — three of the five target features are not
     non-monotone by the repo's own `nonmono_gain` (`results/advisor_inscope/
     ladder_featdiag.csv`, mean over cells): `pe_mean` +0.0438 and `rpdi` +0.0132 are,
     but `dominant_freq` -0.0109, `spectral_entropy` -0.0127 and `epr_energy` -0.0130
     are all NEGATIVE — binning them is WORSE than using them monotonically. The one
     genuinely non-monotone view that is missing from the list is
     `stft_spectral_entropy` (+0.0154).

  5. OURS, found while writing this file. `gap_ladder.py:64`'s `safe_auc_raw` returns
     `max(p, 1-p)`, and `gap_ladder.py:220` applies it to EACH FOLD'S BINNED TEST
     SCORE when computing `nonmono_gain`. A bin mapping fitted on the training fold
     already carries its own direction, so there is no sign left to resolve; folding
     it is a ONE-SIDED NOISE FLOOR that can raise the binned side and never lower it,
     while the monotone baseline it is subtracted from is a single global number.
     The gain is therefore biased upward, worst exactly where the folds are least
     reliable. Measured over 682 cell x feature pairs: the inflation is never
     negative (median 0.0000, max +0.2000) and Spearman(corrected gain, inflation) =
     -0.171, p = 7e-06 — the metric credits a view MORE the closer its binned mapping
     sits to chance. `results/advisor_inscope/ladder_featdiag.csv` carries the
     inflated values, and that file is the artifact defect 4's ranking was drawn
     from, so it is load-bearing for the whole claim.

READ THIS BEFORE C1: C1'S GATE IS SUPERSEDED
---------------------------------------------
C1 gates on the PER-FEATURE MEAN of `nonmono_gain` over cells, and on that statistic
nothing survives. **That aggregation is wrong and the conclusion it produced was
withdrawn** (Omri, mid-Step-217, from the per-cell deep-dive pages). Non-monotonicity
here is CELL-SPECIFIC: a view can be +12pp non-monotone on 7 cells and flat on 17,
which averages to zero. C1 is kept because its second output — the fold-inflation
measurement in defect 5 — is real and load-bearing, but its VERDICT is not the
project's answer.

The answer comes from `scripts/nonmono_shape_test.py` (fair isotonic-vs-10-bin test,
per cell x view, against each pair's own label-permutation null) feeding
`--stage c2c3 --survivors-from-shape-test`. Summary of that chain:

  * THE NON-MONOTONICITY IS REAL. 32 of 682 cell x view pairs beat their own null,
    several at 5-7x it on cells with thousands of rows (`semenergy_triviaqa_qwen3_8b`
    / `rpdi` +0.1227, n=4392, across-seed sd 0.0019, null95 0.0168).
  * GEMINI'S LIST WAS HALF RIGHT. Recurrent: `rpdi` (7 of 24 cells), `pe_mean` (6 of
    23). NOT supported: `dominant_freq` (0 cells), `spectral_entropy` (1),
    `epr_energy` (1). MISSED by the list: `cusum_shift_idx` (6), `hurst_exponent` (3).
  * THE LABEL-FREE HANDLE DOES NOT EXIST. Marginal two-peak measures do not locate it:
    Spearman(shape_gain, KDE peak count) = +0.014, p=0.72; "KDE >= 2 peaks" flags at
    precision 0.128 against a 0.047 base rate. P(y|x) can bend without the DENSITY of
    x having two humps — the two humps visible on the deep-dive pages are the two
    CLASS-CONDITIONAL densities, a different object.
  * AND IT BUYS NOTHING IN FUSION. C2/C3 on the corrected candidate set: G3 PASSES
    (the LOCO choice is stable on 92-100% of folds, so this is not overfitting), G2
    PASSES (worst cell -1.60pp), **G1 FAILS on both arms** (-0.07pp dufs_pf, -0.04pp
    upcr). Restricted to the 11 cells where a candidate view provably beat its null it
    is still -0.13pp / -0.09pp, with more losses than wins. A single view's U-shape is
    largely redundant once 15-20 other views are in the fusion.

WHAT C1 TESTED, AND ITS (SUPERSEDED) TWO-PART ANSWER
-----------------------------------------------------
The hypothesis going in was that the non-monotonicity is an artifact of the DATA
defects: per cell, `inside_coqa_llama7b` was the overwhelming outlier (+0.0393 mean
gain, second place +0.0050), and that is the cell Step 216 rejected — 45.1% broken
answer spans, pos_rate 0.002 on the broken rows against 0.239 on the usable ones. A
mixture of two sub-populations with different feature means and a near-deterministic
label IS a U-shaped feature/label curve, which would make `|x - median(x)|` a
broken-generation detector rather than a hallucination detector.

C1 recomputes `nonmono_gain` on DEFECT-FREE sub-populations, under BOTH definitions,
and it ends Part C. The answer splits in two, and neither half is usable signal:

  (1) ON THE REJECTED CELL IT WAS REAL. `inside_coqa_llama7b` / `pe_mean` is +0.1351
      under both the folded and the corrected metric — a genuine non-monotone curve,
      on exactly the cell whose broken sub-population predicts one. Rejecting the
      cell removes it from the roster.
  (2) FOR EVERYTHING LEFT, IT IS THE METRIC (defect 5). On the surviving 24 cells the
      ranking was still led by `pe_mean` at +0.0400 folded — of which +0.0402 is the
      fold. Corrected: -0.0005. Nothing on the roster reaches the +0.01 threshold;
      the largest is `rpdi` at +0.0091.

So the data defect explains the outlier cell and the metric defect explains the
feature ranking that Gemini's five-feature list was drawn from. C2/C3 do not run.

ON `max(a, 1-a)`
----------------
It appears NOWHERE in this file as a score. `nonmono_gain` needs a monotone
baseline, and the honest baseline for "could a monotone reading of this view have
done as well" is the view under its ORACLE sign — which this file computes
explicitly (`_oracle_sign`) rather than by folding the AUROC. Both sides of the gain
are label-derived DIAGNOSTICS and neither is ever reported as a detector result:
every detector AUROC here goes through `score_cols` / `upcr_arm`, i.e.
`lsml_continuous` or U-PCR, then `anchor_orient`, then RAW AUROC.

Roster comes from `inscope_cells.INSCOPE` (24 cells since Step 216) — never from
filename matching.

Usage:
    python scripts/nonmono_transform_bench.py --stage c1
    python scripts/nonmono_transform_bench.py --stage all
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import norm, rankdata, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inscope_bench_common import load_cells, assert_good6, score_cols   # noqa: E402
from inscope_cells import INSCOPE, MATH_CELLS, CROPPED_CELLS            # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous                 # noqa: E402
from spectral_utils.streaming_utils import anchor_orient                # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                       # noqa: E402
from spectral_utils.upcr import upcr_fit                                # noqa: E402

OUT = os.path.join(REPO, "results", "nonmono_transform")

# exp06's fitted U-PCR configuration, verbatim.
FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

# C1's decision threshold, written down before the run.
C1_THRESHOLD = 0.01

# A cell counts as "pinned" (and so gets restricted to complete traces) only when a
# non-trivial share of its traces sit at the observed maximum. On an unpinned cell
# `trace_length < max` would just delete the single longest row, which is not a
# defect-free restriction, it is noise.
PINNED_MIN_FRAC = 0.01

TRANSFORMS = ("dist_median", "squared", "abs_norm_rank")
MODES = ("replace", "add")


# ── the three label-free transforms, exactly as Gemini wrote them ──────────────
def t_dist_median(x):
    return np.abs(x - np.median(x))


def t_squared(x):
    return x ** 2


def t_abs_norm_rank(x):
    pct = (rankdata(x) - 0.5) / len(x)
    return np.abs(norm.ppf(np.clip(pct, 1e-6, 1 - 1e-6)))


TRANSFORM_FN = {"dist_median": t_dist_median, "squared": t_squared,
                "abs_norm_rank": t_abs_norm_rank}


# ── diagnostics (label-derived; never reported as a detector score) ────────────
def _oracle_sign(x, y):
    """+1 if higher x means more likely positive, else -1. The declared oracle sign,
    which is what makes `nonmono_gain`'s monotone baseline a fair comparison."""
    a = roc_auc_score(y, x)
    return 1.0 if a >= 0.5 else -1.0


def nonmono_gain(x, y, n_splits=5, seed=42, folded=False):
    """Cross-fitted 4-quantile-bin positive-rate mapping, minus the same view read
    monotonically under its oracle sign.

    The binning, the fold splits, the quantile edges and the empty-bin fallback are
    line-for-line `gap_ladder.py:183-228`, so `folded=True` reproduces
    `results/advisor_inscope/ladder_featdiag.csv` — the artifact the original claim
    was argued from — EXACTLY.

    `folded` IS A DEFECT AND DEFAULTS OFF. `gap_ladder.py:64`'s `safe_auc_raw` returns
    `max(p, 1-p)`, and line 220 applies it to each fold's BINNED TEST SCORE. Folding a
    per-fold AUROC upward is not a sign resolution — a bin mapping fitted on train
    already carries its direction — it is a one-sided noise floor. On a noisy fold it
    can only move the binned side UP, while the monotone baseline is a single global
    number, so the difference is biased positive by construction, and worst exactly
    where folds are least reliable. `spilled_triviaqa_llama8b` (n=256, n_pos=6, so
    ~1 positive per test fold) reads +0.0833 folded and -0.1167 unfolded.

    Both are computed and both are reported. The C1 gate is decided on the corrected
    one, but the conclusion is stated under both so it does not rest on this choice.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2 or np.std(x) < 1e-12 or len(y) < 50:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = []
    for tr, te in skf.split(x.reshape(-1, 1), y):
        xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
        edges = np.unique(np.quantile(xtr, [0.0, 0.25, 0.5, 0.75, 1.0]))
        if len(edges) < 2:
            fold.append(0.5)
            continue
        base = float(np.mean(ytr))
        tb = np.digitize(xtr, edges[1:-1])
        rates = {b: (float(np.mean(ytr[tb == b])) if np.any(tb == b) else base)
                 for b in range(len(edges) - 1)}
        s = np.array([rates.get(b, base) for b in np.digitize(xte, edges[1:-1])])
        if len(np.unique(yte)) < 2 or np.std(s) < 1e-12:
            fold.append(0.5)
            continue
        a = float(roc_auc_score(yte, s))
        fold.append(max(a, 1.0 - a) if folded else a)
    binned = float(np.mean(fold))
    mono = float(roc_auc_score(y, _oracle_sign(x, y) * x))
    return binned - mono


# ── the defect-free sub-population (C1) ───────────────────────────────────────
def defect_free_mask(ck, cell):
    """Rows of `ck` that are free of the two known generation defects.

    math cells   -> complete traces only (`trace_length < cap`), reusing
                    `scripts/truncation_confound_audit.py`'s population logic. ITS TWO
                    CAVEATS CARRY OVER VERBATIM: (a) the complete-only sub-population
                    has a DIFFERENT PREVALENCE (truncation correlates with the label,
                    so removing truncated rows raises accuracy) and a smaller n — it is
                    a different, easier population, not a like-for-like re-scoring;
                    (b) a change here does not prove a feature is worthless, only that
                    the number on THAT cell was not attributable to the structure
                    claimed.
    QA cells     -> already defect-free: `inside_coqa_llama7b` is off the roster
                    (Step 216), `seiclr_triviaqa_opt30b` is answer-cropped, and the
                    remaining seven audit healthy (`results/answer_span/audit.csv`).
    """
    n = len(cell["labels"])
    if ck not in MATH_CELLS or "trace_length" not in cell["pool"]:
        return np.ones(n, dtype=bool), "all rows (no known defect)"
    tl = cell["V"][:, cell["pool"].index("trace_length")]
    cap = np.nanmax(tl)
    pinned = tl >= cap - 1e-9
    if pinned.mean() < PINNED_MIN_FRAC:
        return np.ones(n, dtype=bool), f"all rows (pinned {pinned.mean()*100:.2f}% < 1%)"
    return ~pinned, f"complete traces only ({pinned.mean()*100:.1f}% pinned, dropped)"


def stage_c1(cells, args):
    """THE CONFOUND GATE. Runs first and can end Part C."""
    print("\n" + "=" * 78)
    print("C1 — is the non-monotonicity an artifact of the data defects?")
    print("=" * 78)
    rows = []
    for ck in INSCOPE:
        cell = cells[ck]
        mask, why = defect_free_mask(ck, cell)
        y = cell["labels"][mask]
        n_kept, n_all = int(mask.sum()), len(mask)
        if len(np.unique(y)) < 2 or n_kept < 50:
            print(f"  {ck:<34} SKIPPED — {n_kept} rows / single class after restriction")
            continue
        for j, f in enumerate(cell["pool"]):
            rows.append(dict(
                cell=ck, feature=f, n_all=n_all, n_defect_free=n_kept, population=why,
                pos_all=round(float(cell["labels"].mean()), 4),
                pos_defect_free=round(float(y.mean()), 4),
                nonmono_gain_all=round(nonmono_gain(cell["V"][:, j],
                                                    cell["labels"]), 4),
                nonmono_gain_defect_free=round(nonmono_gain(cell["V"][mask, j], y), 4),
                folded_gain_all=round(nonmono_gain(cell["V"][:, j], cell["labels"],
                                                   folded=True), 4),
                folded_gain_defect_free=round(nonmono_gain(cell["V"][mask, j], y,
                                                           folded=True), 4)))
        print(f"  {ck:<34} n {n_all:>5} -> {n_kept:>5}  ({why})")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "c1_nonmono_gain.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {p}  ({len(rows)} rows)")

    by = {}
    for r in rows:
        by.setdefault(r["feature"], []).append(r)
    summ = []
    for f, rs in by.items():
        a = [r["nonmono_gain_all"] for r in rs if np.isfinite(r["nonmono_gain_all"])]
        d = [r["nonmono_gain_defect_free"] for r in rs
             if np.isfinite(r["nonmono_gain_defect_free"])]
        if not d:
            continue
        fa = [r["folded_gain_all"] for r in rs if np.isfinite(r["folded_gain_all"])]
        fdf = [r["folded_gain_defect_free"] for r in rs
               if np.isfinite(r["folded_gain_defect_free"])]
        summ.append(dict(feature=f, n_cells=len(d),
                         mean_all=float(np.mean(a)) if a else float("nan"),
                         mean_defect_free=float(np.mean(d)),
                         max_defect_free=float(np.max(d)),
                         folded_mean_all=float(np.mean(fa)) if fa else float("nan"),
                         folded_mean_defect_free=(float(np.mean(fdf)) if fdf
                                                  else float("nan")),
                         argmax_cell=rs[int(np.argmax(
                             [r["nonmono_gain_defect_free"] for r in rs]))]["cell"]))
    summ.sort(key=lambda r: -r["mean_defect_free"])

    print(f"\n  {'':<26}{'CORRECTED gain':>26}{'FOLDED (ladder_featdiag)':>32}")
    print(f"  {'feature':<26}{'all rows':>13}{'defect-free':>13}"
          f"{'all rows':>16}{'defect-free':>16}")
    print("  " + "-" * 84)
    for r in summ:
        flag = "  <==" if r["mean_defect_free"] >= C1_THRESHOLD else ""
        fflag = "  <== folded only" if (r["folded_mean_defect_free"] >= C1_THRESHOLD
                                        and not flag) else ""
        print(f"  {r['feature']:<26}{r['mean_all']:>+13.4f}"
              f"{r['mean_defect_free']:>+13.4f}{r['folded_mean_all']:>+16.4f}"
              f"{r['folded_mean_defect_free']:>+16.4f}{flag}{fflag}")

    p2 = os.path.join(OUT, "c1_summary.csv")
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print(f"\n  saved {p2}")

    survivors = [r["feature"] for r in summ if r["mean_defect_free"] >= C1_THRESHOLD]
    folded_surv = [r["feature"] for r in summ
                   if r["folded_mean_defect_free"] >= C1_THRESHOLD]
    # The inflation the fold buys, measured rather than argued. A one-sided floor can
    # only ADD, so this is >= 0 by construction; what it costs is a rank ordering that
    # rewards the views whose binned mapping is CLOSEST TO CHANCE.
    infl = [r["folded_gain_defect_free"] - r["nonmono_gain_defect_free"] for r in rows
            if np.isfinite(r["folded_gain_defect_free"])
            and np.isfinite(r["nonmono_gain_defect_free"])]
    corr = [r["nonmono_gain_defect_free"] for r in rows
            if np.isfinite(r["folded_gain_defect_free"])
            and np.isfinite(r["nonmono_gain_defect_free"])]
    rho, prho = spearmanr(corr, infl)
    print(f"\n  FOLD INFLATION over {len(infl)} cell x feature pairs: "
          f"min {min(infl):+.4f}, median {np.median(infl):+.4f}, max {max(infl):+.4f} "
          f"— never\n  negative, as a one-sided floor cannot be. "
          f"Spearman(corrected gain, inflation) = {rho:+.3f} (p={prho:.1e}):\n"
          f"  the folded metric credits a view MORE the worse its binned mapping "
          f"actually does.")
    print(f"\n  Under the repo's OWN (folded) definition, on these same defect-free "
          f"populations,\n  the survivor set would be: {folded_surv or 'EMPTY'}")

    print("\n" + "-" * 78)
    if survivors:
        print(f"C1 PASSES for {len(survivors)} feature(s): {survivors}")
        print("  -> Part C continues to C2/C3 on exactly these.")
    else:
        print(f"C1 FAILS — no feature reaches mean nonmono_gain >= {C1_THRESHOLD} on the "
              "defect-free\n  sub-populations of the 24-cell roster. Part C stops here, "
              "and that is the finding.")
        print("\n  THE PREMISE HAS TWO SEPARATE SOURCES, and neither is a U-shape the "
              "detector can use:")
        print("    (1) THE REJECTED CELL, where it was real. `inside_coqa_llama7b` / "
              "`pe_mean` is\n        +0.1351 under BOTH definitions — a genuine "
              "non-monotone curve. That cell is\n        45.1% broken generations with "
              "pos_rate 0.002 on the broken rows vs 0.239 on\n        the usable ones, "
              "i.e. exactly the two-sub-population mixture that MAKES a\n        U-shape. "
              "There `|x - median(x)|` is a broken-generation detector, not a\n        "
              "hallucination detector. Step 216 rejects the cell, so it leaves the roster.")
        print("    (2) THE METRIC, for everything that was left. On the surviving roster "
              "the ranking\n        was led by `pe_mean` at +0.0400 — of which +0.0402 is "
              "the per-fold\n        `max(p, 1-p)` floor in `gap_ladder.py:64,220`. "
              "Corrected, it is -0.0005.")
    print("-" * 78)
    return survivors, summ


# ── the two arms (C2/C3) ──────────────────────────────────────────────────────
def lsml_arm(V, pool, anchor, y, cols):
    cols = sorted(set(cols))
    if len(cols) < 3:
        return float("nan")
    fused, _ = lsml_continuous(*[V[:, c] for c in cols])
    s, _ = anchor_orient(np.asarray(fused, dtype=float), anchor)
    return float(roc_auc_score(y, s))


def upcr_arm(V, pool, anchor, y):
    hand = np.array([ALL_SIGNS.get(f, +1) for f in pool], dtype=float)
    V_un = V * hand
    d = np.sign(upcr_fit(V_un.T, **FIT).rho_hat_full)
    d[d == 0] = 1.0
    F = (V_un * d).T
    res = upcr_fit(F, **FIT)
    s, _ = anchor_orient(res.w @ F, anchor)
    return float(roc_auc_score(y, s)), int(res.keep.sum())


def apply_config(cell, config):
    """(V, pool) with `config` = {feature: (transform, mode)} applied.

    Columns are z-scored after transformation so the fusion sees the same scale it
    does on the canonical path. `add` mode keeps the parent view AND appends the
    transform — a deterministic function of a column already in the matrix, i.e. a
    maximally dependent pair, which is exactly what the rho >= 0.75 filter exists to
    prevent. The induced |rho| is reported per pair rather than assumed harmless.
    """
    V, pool = cell["V"], list(cell["pool"])
    cols = [V[:, j] for j in range(V.shape[1])]
    induced = {}
    for f, (tname, mode) in sorted(config.items()):
        if f not in pool:
            continue
        j = pool.index(f)
        x = V[:, j]
        t = TRANSFORM_FN[tname](x)
        sd = np.std(t)
        t = (t - np.mean(t)) / sd if sd > 1e-12 else np.zeros_like(t)
        if mode == "replace":
            cols[j] = t
        else:
            cols.append(t)
            pool.append(f"{f}__{tname}")
            induced[f"{f}__{tname}"] = float(abs(np.corrcoef(x, t)[0, 1]))
    return np.column_stack(cols), pool, induced


def score_config(cell, config, dufs_cols_names):
    """Both arms under one config. Arm A holds the DUFS-selected view set fixed
    (replaced views map to their transform, added views join the set), so it measures
    the transform's effect on FUSION. Re-running DUFS per config would be ~30s x 24
    cells x every config; that is a separate experiment and is stated as a limitation
    rather than silently skipped."""
    V, pool, induced = apply_config(cell, config)
    y, anchor = cell["labels"], cell["anchor"]
    keep = set(dufs_cols_names)
    for f, (tname, mode) in config.items():
        if mode == "add" and f in keep:
            keep.add(f"{f}__{tname}")
    cols = [i for i, f in enumerate(pool) if f in keep]
    a_pf = lsml_arm(V, pool, anchor, y, cols)
    a_up, _ = upcr_arm(V, pool, anchor, y)
    return a_pf, a_up, induced


def shape_test_survivors(min_hit):
    """C2's candidate set, taken from `nonmono_shape_test.py` rather than from C1.

    WHY THIS OVERRIDES C1 (Omri, mid-Step-217). C1 gated on the per-feature MEAN of
    `nonmono_gain` across cells and found nothing. That aggregation is wrong for a
    CELL-SPECIFIC effect: a view can be +12pp non-monotone on 7 cells and flat on 17,
    which averages to zero. The shape test measures each cell x view pair against its
    OWN label-permutation null under a fair monotone baseline (isotonic), and 32 of
    682 pairs survive — several at 5-7x their null on cells with thousands of rows.

    Requiring `min_hit` cells rather than one keeps this from being a per-cell fit:
    the transform still has to be a property of the FEATURE, chosen once, or there is
    nothing to deploy.
    """
    p = os.path.join(OUT, "shape_test.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing — run scripts/nonmono_shape_test.py first")
    hits = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["exceeds_null"] == "1":
                hits[r["feature"]] = hits.get(r["feature"], 0) + 1
    return [f for f, c in sorted(hits.items(), key=lambda kv: -kv[1]) if c >= min_hit]


def stage_c2c3(cells, survivors, args):
    print("\n" + "=" * 78)
    print("C2/C3 — leave-one-cell-out candidate selection, then the pre-registered gates")
    print("=" * 78)

    bench = list(csv.DictReader(open(
        os.path.join(REPO, "results/selector_bench/a2_groupfs__c46.csv"), newline="")))
    dufs = {r["cell"]: r["chosen"].split("|")
            for r in bench if r["variant"] == "a2.dufs_pf"}

    configs = [{}]
    for f in survivors:
        for t in TRANSFORMS:
            for m in MODES:
                configs.append({f: (t, m)})

    tbl, rho_rows = {}, []
    for ck in INSCOPE:
        cell = cells[ck]
        tbl[ck] = {}
        for cfg in configs:
            key = json.dumps(cfg, sort_keys=True)
            a_pf, a_up, induced = score_config(cell, cfg, dufs[ck])
            tbl[ck][key] = (a_pf, a_up)
            for name, r in induced.items():
                rho_rows.append(dict(cell=ck, derived=name, induced_abs_rho=round(r, 4)))
        print(f"  {ck:<34} scored {len(configs)} configs", flush=True)

    base = json.dumps({}, sort_keys=True)
    results = {}
    for arm_i, arm in enumerate(("dufs_pf", "upcr")):
        chosen_per_fold, loco = {}, {}
        for held in INSCOPE:
            train = [c for c in INSCOPE if c != held]
            cfg = {}
            for f in survivors:
                best, best_m = None, np.mean([tbl[c][base][arm_i] for c in train])
                for t in TRANSFORMS:
                    for m in MODES:
                        k = json.dumps({f: (t, m)}, sort_keys=True)
                        mm = np.mean([tbl[c][k][arm_i] for c in train])
                        if mm > best_m:
                            best_m, best = mm, (t, m)
                if best is not None:
                    cfg[f] = best
            chosen_per_fold[held] = cfg
            k = json.dumps(cfg, sort_keys=True)
            if k not in tbl[held]:
                a_pf, a_up, _ = score_config(cells[held], cfg, dufs[held])
                tbl[held][k] = (a_pf, a_up)
            loco[held] = tbl[held][k][arm_i]

        inc = {c: tbl[c][base][arm_i] for c in INSCOPE}
        d = [loco[c] - inc[c] for c in INSCOPE]
        g1_delta = float(np.mean(d))
        g1_p = float(wilcoxon(d).pvalue) if any(abs(x) > 1e-15 for x in d) else 1.0
        g1 = bool(g1_delta >= 0.005 and g1_p < 0.05)
        worst = min(d)
        g2 = bool(worst >= -0.02)
        stab = {}
        for f in survivors:
            picks = [chosen_per_fold[h].get(f, ("baseline", "-")) for h in INSCOPE]
            top = max(set(picks), key=picks.count)
            stab[f] = (top, picks.count(top) / len(picks))
        g3 = bool(all(v[1] >= 0.80 for v in stab.values())) if stab else False

        print(f"\n  ARM {arm}")
        print(f"    incumbent macro      {np.mean(list(inc.values())):.4f}")
        print(f"    LOCO macro           {np.mean(list(loco.values())):.4f}")
        print(f"    G1  >= +0.5pp, p<.05 {g1_delta*100:+.2f}pp, "
              f"{sum(x>0 for x in d)}W/{sum(x<0 for x in d)}L, p={g1_p:.4f}  "
              f"-> {'PASS' if g1 else 'FAIL'}")
        print(f"    G2  no cell < -2pp   worst {worst*100:+.2f}pp "
              f"({INSCOPE[int(np.argmin(d))]})  -> {'PASS' if g2 else 'FAIL'}")
        for f, (top, frac) in stab.items():
            print(f"    G3  {f:<22} {str(top):<24} on {frac*100:.0f}% of folds")
        print(f"    G3  >= 80% everywhere -> {'PASS' if g3 else 'FAIL'}")
        results[arm] = dict(incumbent=float(np.mean(list(inc.values()))),
                            loco=float(np.mean(list(loco.values()))),
                            g1_delta=g1_delta, g1_p=g1_p, g1=g1,
                            g2_worst=float(worst), g2=g2,
                            g3={f: [list(v[0]), v[1]] for f, v in stab.items()}, g3_pass=g3,
                            per_cell={c: [inc[c], loco[c]] for c in INSCOPE})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "c2c3_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    if rho_rows:
        with open(os.path.join(OUT, "c2_induced_rho.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rho_rows[0]))
            w.writeheader()
            w.writerows(rho_rows)
        rr = [r["induced_abs_rho"] for r in rho_rows]
        print(f"\n  induced |rho| against the parent view (Add mode): "
              f"median {np.median(rr):.3f}, max {np.max(rr):.3f} — the rho >= 0.75 "
              f"filter\n  exists to exclude exactly this kind of pair.")

    both = all(results[a]["g1"] and results[a]["g2"] and results[a]["g3_pass"]
               for a in results)
    verdict = ("ALL THREE GATES PASS ON BOTH ARMS — adopt" if both
               else "NOT ADOPTED — at least one gate fails")
    print("\n" + "-" * 78)
    print(f"C3 VERDICT: {verdict}")
    print("-" * 78)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("c1", "all", "c2c3"), default="c1")
    ap.add_argument("--survivors-from-shape-test", action="store_true",
                    help="take C2's candidate set from results/nonmono_transform/"
                         "shape_test.csv (features that beat their own label-permutation "
                         "null on >= MIN_HIT cells) instead of from C1's per-feature "
                         "mean. This is the CORRECTED path — see stage_c2c3's docstring.")
    ap.add_argument("--min-hit", type=int, default=3,
                    help="a feature must beat its null on at least this many cells")
    args = ap.parse_args()

    cells = load_cells()
    ok, macro = assert_good6(cells, verbose=True)
    if not ok:
        raise SystemExit(f"validity gate failed: GOOD_6 macro {macro:.4f}. Refusing to "
                         "measure anything off data that is not ours.")
    print(f"roster: {len(INSCOPE)} in-scope cells "
          f"(inside_coqa_llama7b rejected, {CROPPED_CELLS} answer-cropped — Step 216)")

    if args.survivors_from_shape_test:
        survivors = shape_test_survivors(args.min_hit)
        print(f"\nCANDIDATE SET from the shape test (>= {args.min_hit} cells beating "
              f"their own\nlabel-permutation null): {survivors}")
    else:
        survivors, _ = stage_c1(cells, args)
        if args.stage == "c1":
            return 0
    if not survivors:
        print("\nC2/C3 not run: no candidate survived. That is the pre-registered "
              "behaviour.")
        return 0
    stage_c2c3(cells, survivors, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
