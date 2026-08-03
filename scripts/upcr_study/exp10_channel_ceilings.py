"""
exp10_channel_ceilings.py — Phase 1 of the U-PCR clustering plan. CEILINGS ONLY.

WHAT THIS IS FOR
----------------
Before building any clustering / gating variant, price the three channels through
which U-PCR's estimation machinery can possibly reach the output. Nothing here is
adoptable; every number is either an ORACLE (uses labels, so it is an upper bound)
or a label-free sensitivity sweep. If the oracles are small, the T1 channel is
measurably closed and no variant can beat them — which is a better answer to
advisor action item 2 than Step 204's refuted variant.

THE THREE CHANNELS (upcr.py)
----------------------------
  A  which features survive     upcr.py:287-298   rho >= 0.05*var_y AND rho >= rho_max/3
  B  how v1 and v2 mix          upcr.py:336-342   w = sum_c (v_c.rho)/lambda_c * v_c
  C  the g2 scalar              upcr.py:270-277   Eq. 20 argmin

At ONE component w is a scalar multiple of v1, so rho is irrelevant there (AUROC is
scale-invariant, global sign comes from the anchor). At TWO components rho re-enters
through exactly one number, the ratio c2/c1. So every "estimate rho better" proposal
competes for B and C only — unless it also changes A.

  A claim this script set out to test and then REFUTED: that g2 escapes C into A.
  The algebra says it should — var_y and g2 enter the keep rule at upcr.py:287-293, and
  raising q lifts every rho_i by q/2 against a relative threshold rising by only q/6, so
  a larger g2 keeps weakly more views. Measured, the margin between kept and dropped rho
  is far wider than the thresholds move: across a 10.9x change in var_y the mean survivor
  count goes 20.88 -> 20.46 and most cells' masks are IDENTICAL. In practice g2 is
  confined to B/C.

THE RUNS
--------
  r7a  The var_y axis, two points on it. `Var(Y) = p(1-p)` is exact for binary Y, BUT
       only the RATIO var_y/mean(diag C) is identifiable, and moment-matching z-scored
       views to a binary Y is exactly scale_ratio = 1 (upcr.py:154-173). So p(1-p)
       against a unit diagonal is NOT an oracle — it is a one-sided scale_ratio
       perturbation into (0, 0.25], bounded above by the deployed value. Both it and
       the A1-faithful scale_ratio = 1.0 are run; r4's slice is the real test.
  r1   What does the mask throw away today? Count of dropped views + their ORACLE
       single-view AUROC (sign given by labels, so it measures signal content
       irrespective of whether sign(rho) could find it).
  r2   ORACLE MASK ceiling (channel A). Greedy forward/backward from the incumbent
       keep set, scored with labels, refit with exclusion=False. Three arms:
       in-sample (an upper bound only where the greedy CONVERGED), split-half (the
       Step-189 discipline — selection labels disjoint from scoring labels), and a
       PERMUTED-LABEL NULL on the same splits. The null is not optional: a greedy over
       30 views finds gains from noise, and without it the honest number means nothing.
  r3   ORACLE MIXING ceiling (channel B). w(theta) = cos(theta) v1 + sin(theta) v2
       over theta in [0, pi); AUROC is scale-invariant and the anchor fixes the
       global sign, so this sweeps the ENTIRE 2-D span the weight rule can reach.
       NOTE a 1-component cell still gets the full sweep (its deployed point sits at
       theta = 0), so its delta prices "add a second component", not "remix two".
       `n_components_used` is in the CSV; do not pool the two silently.
  r4   Label-free CONSTANT SWEEP: min_frac x exclude_frac x scale_ratio. A repo-wide
       grep says none of the three has ever been swept. The honest companion to r2.
  r5   Does Eq. 20's RES order masks by AUROC? PARTIAL Spearman(RES, AUROC | n_keep)
       over r4's grid, per cell. THE GATE on any differentiable-gate work. The partial
       is mandatory, not decoration — RES is computed on a block whose dimension IS
       n_keep, and n_keep is exactly what the swept constants move. Precedent: Step
       203's fit criterion went +0.223 -> -0.006 once a nuisance was removed, and the
       first two cuts of THIS run walked into the same trap twice running.
  r7b  Parameter-free var_y from the fixed point g2 = Var(yhat) = w' C w, solved
       numerically on the post-exclusion block with the mask HELD FIXED (otherwise
       the mask moves with q and the map is discontinuous).

Run:  python scripts/upcr_study/exp10_channel_ceilings.py --runs r7a
      python scripts/upcr_study/exp10_channel_ceilings.py --runs all
Out:  results/upcr_study/10_channel_ceilings/
"""
import argparse
import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from spectral_utils.upcr import upcr_fit                            # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                   # noqa: E402

# The deployed configuration, identical to exp06_orientation.FIT. Every arm below
# differs from this in exactly ONE documented way.
FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)

# THE ARM OF RECORD. `results/BENCHMARK_STANDING.md:11` registers "U-PCR + sign(rho)"
# at 0.7741, and that is exp06's `rho+anchor` arm — polarity DERIVED from sign(rho_hat)
# on unoriented features, global sign from the cell's anchor. It is NOT the
# hand-polarity arm, which scores 0.7571 (exp06 `macro_hand_anchor`). The first cut of
# this script fitted `cell["V"]` directly, which is already hand-oriented by
# prepare_cell, and so reproduced 0.75713 — the wrong arm, to 5dp. Everything here now
# runs on the derived arm, and `derived_arm_gate` asserts it.
UPCR_DERIVED_EXPECTED = 0.7741
UPCR_DERIVED_TOL = 0.002

N_SPLITS = 5            # r2 split-half repeats
GREEDY_MAX_STEPS = 60   # r2 greedy budget; 12 left 17/24 cells mid-descent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
FIT_FAILURES = []


def _all_fields(rows):
    """Union of keys, in first-seen order. `save_csv` defaults to row 0's keys with a
    raising DictWriter, so a fallback row first would crash after all the compute and a
    fallback row later would silently blank its extra columns."""
    seen = {}
    for r in rows:
        for k in r:
            seen.setdefault(k, None)
    return list(seen)


def _zsc(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def derive_cell(cell):
    """The arm of record: re-orient a cell's views by sign(rho_hat), not by hand.

    prepare_cell already applied ALL_SIGNS, and zscore(raw*s) == s*zscore(raw) for
    s = +-1, so multiplying by the hand signs UNDOES the orientation and recovers the
    raw z-scored feature (exp06_orientation.py:76-81 — getting this backwards silently
    swaps the arms). One throwaway full-pool fit then reads each view's polarity off
    sign(rho_hat_full), exactly as `upcr_pipeline_faithful(orient='derived')` does.

    The polarity is derived ONCE on the full pool and then HELD FIXED across every mask
    and every mixing angle below. That is deliberate: Part 1 is isolating channels, and
    re-deriving the orientation per subset would fold an orientation change into what is
    supposed to be a pure channel-A measurement.
    """
    V = np.asarray(cell["V"], dtype=float)
    hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
    V_raw = V * hand                                   # undo the hand orientation
    probe = upcr_fit(V_raw.T, **FIT)
    pol = np.sign(probe.rho_hat_full)
    pol[pol == 0] = 1.0
    out = dict(cell)
    out["V"] = V_raw * pol
    out["derived_polarity"] = pol
    return out


def derived_arm_gate(dcells, verbose=True):
    """Anti-regression anchor for THIS script's arm, checked by value not truthiness
    (Step 204 defect D6: `assert_good6` returns a tuple, so `bool(...)` is always True
    and two gates were silent no-ops in three files)."""
    vals = [fit_cols(c, range(c["V"].shape[1]))[0] for c in dcells.values()]
    n_bad = int(np.sum(~np.isfinite(vals)))
    macro = float(np.nanmean(vals))
    # A gate that nanmean's away its own failures is not a gate: the macro of the
    # SURVIVING cells could pass while cells silently vanished.
    ok = (abs(macro - UPCR_DERIVED_EXPECTED) <= UPCR_DERIVED_TOL) and n_bad == 0
    if verbose:
        print(f"VALIDITY: U-PCR + sign(rho) macro = {macro:.4f} "
              f"(expected {UPCR_DERIVED_EXPECTED} +/- {UPCR_DERIVED_TOL}) -> "
              f"{'PASS' if ok else 'FAIL'}"
              + (f"  [{n_bad} CELLS FAILED TO FIT]" if n_bad else ""))
    if not ok:
        raise SystemExit(
            f"DERIVED-ARM GATE FAILED: macro = {macro:.4f}, expected "
            f"{UPCR_DERIVED_EXPECTED}. This is not the arm BENCHMARK_STANDING.md "
            "registers — refusing to report ceilings against the wrong baseline.")
    return macro


def fit_cols(cell, cols, **over):
    """Fit U-PCR on a column subset and score it through the CANONICAL path
    (anchor_orient then raw AUROC, never max(a, 1-a)). Returns (auroc, result)."""
    cols = sorted(int(c) for c in cols)
    if len(cols) < 3:
        return float("nan"), None
    F = cell["V"][:, cols].T
    kw = dict(FIT)
    kw.update(over)
    try:
        res = upcr_fit(F, **kw)
    except Exception as exc:                 # never silent: a swallowed failure becomes
        FIT_FAILURES.append((len(cols), repr(exc)))   # a NaN that nanmean deletes
        return float("nan"), None
    if not np.isfinite(res.w).all() or np.allclose(res.w, 0):
        return float("nan"), res
    return S.auroc_from_score(cell, res.w @ F), res


def oracle_single_view_auroc(labels, col):
    """Signal content of one view with the sign GIVEN BY LABELS. This is the one
    place max(a, 1-a) is legitimate: the question is how much orientable signal the
    mask discarded, not what our label-free orienter would have recovered."""
    a = float(roc_auc_score(labels, col))
    return max(a, 1.0 - a)


def baseline_row(cell):
    """The deployed fit on the full pool, plus everything the other runs need."""
    m = cell["V"].shape[1]
    auc, res = fit_cols(cell, range(m))
    return auc, res


# ---------------------------------------------------------------------------
# R7a — ORACLE var_y = p(1-p)
# ---------------------------------------------------------------------------
def run_r7a(cells):
    """NOT AN ORACLE — read this before quoting the number.

    `Var(Y) = p(1-p)` is exact for a binary Y, but `upcr.py:36-39` and `moment_match`
    (upcr.py:154-173) establish that only the RATIO `var_y / mean(diag C)` is
    identifiable, and that moment-matching z-scored views to a binary Y is EXACTLY
    `scale_ratio = 1`. So for the unit-diagonal C this study uses, the A1-faithful
    `var_y` is 1.0, not p(1-p). Passing p(1-p) against a unit diagonal is a
    `scale_ratio` of p(1-p) in (0, 0.25] — bounded above by the deployed 0.25 BY
    CONSTRUCTION, i.e. a one-sided perturbation, not an oracle. The audit caught the
    mislabel. R4's `scale_ratio` slice is the honest test of this axis; the
    A1-faithful arm is run here alongside so both are on the same page.
    """
    rows = []
    for ck, cell in cells.items():
        m = cell["V"].shape[1]
        p = float(np.mean(cell["labels"]))
        var_y_oracle = p * (1.0 - p)

        auc_base, r_base = baseline_row(cell)
        auc_orc, r_orc = fit_cols(cell, range(m), var_y=var_y_oracle,
                                  scale_ratio=None)
        auc_a1, r_a1 = fit_cols(cell, range(m), scale_ratio=1.0)
        rows.append({
            "auroc_a1_faithful_ratio1": auc_a1,
            "delta_a1_pp": (auc_a1 - auc_base) * 100.0,
            "n_keep_a1": int(r_a1.keep.sum()) if r_a1 else -1,
            "cell": ck, "m": m, "n": int(cell["V"].shape[0]),
            "p": p, "var_y_oracle": var_y_oracle,
            "var_y_deployed": float(r_base.var_y) if r_base else float("nan"),
            "box_ratio_deployed_over_oracle":
                (float(r_base.var_y) / var_y_oracle) if r_base else float("nan"),
            "auroc_deployed": auc_base, "auroc_var_y_pq": auc_orc,
            "delta_pp": (auc_orc - auc_base) * 100.0,
            "n_keep_deployed": int(r_base.keep.sum()) if r_base else -1,
            "n_keep_oracle": int(r_orc.keep.sum()) if r_orc else -1,
            "g2_deployed": float(r_base.g2_hat) if r_base else float("nan"),
            "g2_oracle": float(r_orc.g2_hat) if r_orc else float("nan"),
            "g2_at_ceiling_deployed": bool(r_base.g2_at_ceiling) if r_base else None,
            "g2_at_ceiling_oracle": bool(r_orc.g2_at_ceiling) if r_orc else None,
        })
        print(f"  {S.plain_cell(ck)[:34]:34s} p={p:.3f} var_y {rows[-1]['var_y_deployed']:.4f}"
              f"->{var_y_oracle:.4f}  keep {rows[-1]['n_keep_deployed']:2d}"
              f"->{rows[-1]['n_keep_oracle']:2d}  AUROC {auc_base:.4f}->{auc_orc:.4f}"
              f"  ({rows[-1]['delta_pp']:+.2f}pp)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# R1 — what the mask throws away
# ---------------------------------------------------------------------------
def run_r1(cells):
    rows, view_rows = [], []
    for ck, cell in cells.items():
        V, y, pool = cell["V"], cell["labels"], cell["pool"]
        _, res = baseline_row(cell)
        if res is None:
            continue
        keep = res.keep
        orc = np.array([oracle_single_view_auroc(y, V[:, j])
                        for j in range(V.shape[1])])
        dropped = np.where(~keep)[0]
        kept = np.where(keep)[0]
        for j in dropped:
            view_rows.append({"cell": ck, "feature": pool[j], "col": int(j),
                              "oracle_single_auroc": float(orc[j]),
                              "rho_hat_full": float(res.rho_hat_full[j]),
                              "beats_best_kept":
                                  bool(orc[j] > orc[kept].max()) if kept.size else None})
        rows.append({
            "cell": ck, "m": int(V.shape[1]),
            "n_dropped": int(dropped.size), "n_kept": int(kept.size),
            "mean_oracle_auroc_dropped":
                float(orc[dropped].mean()) if dropped.size else float("nan"),
            "max_oracle_auroc_dropped":
                float(orc[dropped].max()) if dropped.size else float("nan"),
            "mean_oracle_auroc_kept":
                float(orc[kept].mean()) if kept.size else float("nan"),
            "max_oracle_auroc_kept":
                float(orc[kept].max()) if kept.size else float("nan"),
            "n_dropped_beating_best_kept":
                int((orc[dropped] > orc[kept].max()).sum())
                if dropped.size and kept.size else 0,
        })
        print(f"  {S.plain_cell(ck)[:34]:34s} dropped {dropped.size:2d}/{V.shape[1]:2d}"
              f"  oracle AUROC dropped mean {rows[-1]['mean_oracle_auroc_dropped']:.4f}"
              f" max {rows[-1]['max_oracle_auroc_dropped']:.4f}"
              f" | kept max {rows[-1]['max_oracle_auroc_kept']:.4f}"
              f" | {rows[-1]['n_dropped_beating_best_kept']} dropped beat best kept",
              flush=True)
    return rows, view_rows


# ---------------------------------------------------------------------------
# R2 — ORACLE MASK (channel A)
# ---------------------------------------------------------------------------
def _greedy_mask(score_fn, m, start, max_steps=GREEDY_MAX_STEPS):
    """Forward/backward greedy over column subsets. `score_fn(cols) -> float`.
    Starts from `start` (the incumbent keep set) and accepts any single add or drop
    that improves the score, until nothing does.

    Returns (cols, best, start_score, converged). `converged` is False when the step
    budget ran out mid-descent — a truncated greedy is NOT an upper bound, and the
    first cut of this script silently reported one as such on 17 of 24 cells
    (budget 12 gave +2.43pp, budget 60 gave +3.11pp). Found by the Phase-1 audit.

    `start_score` is returned so the caller can quote the mask delta against the
    greedy's OWN starting point. That matters because `upcr_fit` re-decides
    `n_components` on each sub-block (upcr.py:249-254), so the start point already
    differs from the full-pool fit on some cells and a delta taken against the
    deployed number would fold a component-count change into a channel-A figure.
    """
    cur = set(int(c) for c in start)
    if len(cur) < 3:
        return sorted(cur), float("nan"), float("nan"), True
    best = start_score = score_fn(sorted(cur))
    if not np.isfinite(best):
        return sorted(cur), float("nan"), float("nan"), True
    converged = False
    for _ in range(max_steps):
        cand_best, cand_set = best, None
        for j in range(m):                       # try every single flip
            trial = set(cur)
            if j in trial:
                if len(trial) <= 3:
                    continue
                trial.discard(j)
            else:
                trial.add(j)
            s = score_fn(sorted(trial))
            if np.isfinite(s) and s > cand_best + 1e-9:
                cand_best, cand_set = s, trial
        if cand_set is None:
            converged = True
            break
        cur, best = cand_set, cand_best
    return sorted(cur), best, start_score, converged


def run_r2(cells, dcells, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ck, cell in dcells.items():
        V, y = cell["V"], cell["labels"]
        m = V.shape[1]
        auc_base, res = baseline_row(cell)
        if res is None:
            continue
        start = list(np.where(res.keep)[0])

        # ---- in-sample oracle: an upper bound ONLY if the greedy converged ----
        cols_in, auc_in, auc_start, conv_in = _greedy_mask(
            lambda c: fit_cols(cell, c, exclusion=False)[0], m, start)

        # ---- split-half oracle: selection labels disjoint from scoring labels ----
        # Two leakage paths are closed here, not one:
        #   1. z-scoring is redone WITHIN each half, so half B never influences how
        #      half A is standardised (selector_splithalf_oracle.py:29-32).
        #   2. the sign(rho) POLARITY is re-derived on half A and then APPLIED to half
        #      B. Reusing the full-pool polarity would carry half B's own rows into the
        #      orientation of the thing being scored on half B.
        # A THIRD is closed by the audit: if half A's own fit fails there is no
        # legitimate starting mask, so the split is SKIPPED. The first cut fell back to
        # the full-cell keep set, which imports half B's rows into half A's selection.
        V0 = cells[ck]["V"]                      # hand-oriented, full-cell z-scored
        hand = np.array([ALL_SIGNS.get(f, +1) for f in cells[ck]["pool"]], dtype=float)
        halfB, halfB_base, halfB_null = [], [], []
        n_unconv = 0
        n = V.shape[0]
        for rep in range(N_SPLITS):
            idx = rng.permutation(n)
            a_idx, b_idx = idx[: n // 2], idx[n // 2:]
            if (len(np.unique(y[a_idx])) < 2) or (len(np.unique(y[b_idx])) < 2):
                continue
            raw_a = np.column_stack([_zsc(V0[a_idx, j]) for j in range(m)]) * hand
            raw_b = np.column_stack([_zsc(V0[b_idx, j]) for j in range(m)]) * hand
            try:
                pol_a = np.sign(upcr_fit(raw_a.T, **FIT).rho_hat_full)
            except Exception:
                continue
            pol_a[pol_a == 0] = 1.0
            cell_a = {"V": raw_a * pol_a,
                      "anchor": _zsc(cell["anchor"][a_idx]), "labels": y[a_idx]}
            cell_b = {"V": raw_b * pol_a,
                      "anchor": _zsc(cell["anchor"][b_idx]), "labels": y[b_idx]}
            _, res_a = fit_cols(cell_a, range(m))
            if res_a is None:
                continue                          # no legitimate start; skip the split
            start_a = list(np.where(res_a.keep)[0])
            cols_a, _, _, conv_a = _greedy_mask(
                lambda c: fit_cols(cell_a, c, exclusion=False)[0], m, start_a)
            n_unconv += (not conv_a)
            halfB.append(fit_cols(cell_b, cols_a, exclusion=False)[0])
            halfB_base.append(fit_cols(cell_b, range(m))[0])

            # PERMUTATION NULL. A forward/backward greedy over 30 views finds gains
            # even when the selection signal is pure noise, so the honest number is
            # uninterpretable without this. Same splits, same budget, same evaluation
            # on half B's TRUE labels — only half A's labels are shuffled. Step 204's
            # headline survived until a matched random control reproduced it at 0.977x.
            cell_a_null = dict(cell_a, labels=rng.permutation(cell_a["labels"]))
            cols_null, _, _, _ = _greedy_mask(
                lambda c: fit_cols(cell_a_null, c, exclusion=False)[0], m, start_a)
            halfB_null.append(fit_cols(cell_b, cols_null, exclusion=False)[0])

        def mean_or_nan(v):
            return float(np.nanmean(v)) if v else float("nan")

        sh_o, sh_b, sh_n = (mean_or_nan(halfB), mean_or_nan(halfB_base),
                            mean_or_nan(halfB_null))
        rows.append({
            "cell": ck, "m": m, "n": int(n),
            "auroc_deployed": auc_base,
            "auroc_greedy_start": auc_start,
            "auroc_oracle_mask_insample": auc_in,
            "delta_insample_pp": (auc_in - auc_base) * 100.0,
            "delta_insample_vs_start_pp": (auc_in - auc_start) * 100.0,
            "greedy_converged_insample": bool(conv_in),
            "n_cols_oracle_insample": len(cols_in),
            "n_cols_deployed": int(res.keep.sum()),
            "auroc_splithalf_oracle": sh_o,
            "auroc_splithalf_deployed": sh_b,
            "auroc_splithalf_NULL": sh_n,
            "delta_splithalf_pp": (sh_o - sh_b) * 100.0,
            "delta_splithalf_vs_null_pp": (sh_o - sh_n) * 100.0,
            "null_delta_vs_deployed_pp": (sh_n - sh_b) * 100.0,
            "n_splits_used": len(halfB),
            "n_splits_greedy_unconverged": int(n_unconv),
            "oracle_cols": "|".join(cell["pool"][c] for c in cols_in),
        })
        print(f"  {S.plain_cell(ck)[:34]:34s} in-sample {auc_in:.4f} "
              f"({rows[-1]['delta_insample_vs_start_pp']:+.2f}pp vs start"
              f"{'' if conv_in else ', NOT CONVERGED'}) | split-half {sh_o:.4f} vs "
              f"real {sh_b:.4f} ({rows[-1]['delta_splithalf_pp']:+.2f}pp) vs NULL "
              f"{sh_n:.4f} ({rows[-1]['delta_splithalf_vs_null_pp']:+.2f}pp)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# R3 — ORACLE MIXING (channel B)
# ---------------------------------------------------------------------------
def run_r3(cells, n_theta=721):
    rows = []
    for ck, cell in cells.items():
        V, y = cell["V"], cell["labels"]
        n, m = V.shape
        auc_base, res = baseline_row(cell)
        if res is None:
            continue
        keep = res.keep
        n_keep = int(keep.sum())
        F_k = V[:, keep].T
        C_k = (F_k @ F_k.T) / n

        if res.used_simple_average or n_keep < 2:
            rows.append({"cell": ck, "n_components_used": res.n_components_used,
                         "used_simple_average": bool(res.used_simple_average),
                         "auroc_deployed": auc_base,
                         "auroc_oracle_mix": auc_base, "delta_pp": 0.0,
                         "reason": "simple-average fallback: no v1/v2 mix to sweep",
                         "theta_best": float("nan"), "theta_deployed": float("nan")})
            continue

        ev, evec = eigh(C_k, subset_by_index=[n_keep - 2, n_keep - 1])
        evals, evecs = ev[::-1], evec[:, ::-1]
        v1, v2 = evecs[:, 0], evecs[:, 1]
        rho_k = res.rho_hat[keep]

        # where the deployed rule sits in the (v1, v2) plane
        c1 = float(v1 @ rho_k) / (evals[0] + 1e-12)
        c2 = (float(v2 @ rho_k) / (evals[1] + 1e-12)
              if res.n_components_used >= 2 else 0.0)
        theta_dep = float(np.arctan2(c2, c1)) % np.pi

        # AUROC is scale-invariant and anchor_orient fixes the global sign, so the
        # unit half-circle covers every direction the 2-component rule can produce.
        thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
        aucs = np.array([S.auroc_from_score(cell, (np.cos(t) * v1 + np.sin(t) * v2) @ F_k)
                         for t in thetas])
        k = int(np.nanargmax(aucs))
        auc_dep_plane = S.auroc_from_score(
            cell, (np.cos(theta_dep) * v1 + np.sin(theta_dep) * v2) @ F_k)
        rows.append({
            "cell": ck, "n_components_used": int(res.n_components_used),
            "used_simple_average": False,
            "auroc_deployed": auc_base,
            "auroc_deployed_in_plane": float(auc_dep_plane),
            "auroc_oracle_mix": float(aucs[k]),
            "delta_pp": (float(aucs[k]) - auc_base) * 100.0,
            "theta_best": float(thetas[k]), "theta_deployed": theta_dep,
            "auroc_v1_only": S.auroc_from_score(cell, v1 @ F_k),
            "lambda2_frac": float(res.lambda2_frac),
            "reason": "",
        })
        print(f"  {S.plain_cell(ck)[:34]:34s} k={res.n_components_used} deployed "
              f"{auc_base:.4f} | v1-only {rows[-1]['auroc_v1_only']:.4f} | oracle-mix "
              f"{aucs[k]:.4f} ({rows[-1]['delta_pp']:+.2f}pp)  theta "
              f"{theta_dep:.3f}->{thetas[k]:.3f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# R4 + R5 — label-free constant sweep, and whether RES orders the grid
# ---------------------------------------------------------------------------
MIN_FRACS = (0.0, 0.01, 0.05, 0.10, 0.20)
EXCLUDE_FRACS = (2.0, 3.0, 5.0, 10.0, float("inf"))
SCALE_RATIOS = (0.05, 0.10, 0.25, 0.50, 1.00)


def run_r4(cells):
    rows = []
    for ck, cell in cells.items():
        m = cell["V"].shape[1]
        auc_base, _ = baseline_row(cell)
        for mf in MIN_FRACS:
            for ef in EXCLUDE_FRACS:
                for sr in SCALE_RATIOS:
                    auc, res = fit_cols(cell, range(m), min_frac=mf,
                                        exclude_frac=ef, scale_ratio=sr)
                    rows.append({
                        "cell": ck, "min_frac": mf, "exclude_frac": ef,
                        "scale_ratio": sr, "auroc": auc,
                        "auroc_deployed": auc_base,
                        "delta_pp": (auc - auc_base) * 100.0,
                        "n_keep": int(res.keep.sum()) if res else -1,
                        "proj_residual": float(res.proj_residual) if res else float("nan"),
                        "g2_hat": float(res.g2_hat) if res else float("nan"),
                        "g2_at_ceiling": bool(res.g2_at_ceiling) if res else None,
                        "is_deployed_point": bool(mf == 0.05 and ef == 3.0
                                                  and sr == 0.25),
                    })
        best = max((r for r in rows if r["cell"] == ck),
                   key=lambda r: (r["auroc"] if np.isfinite(r["auroc"]) else -1))
        print(f"  {S.plain_cell(ck)[:34]:34s} deployed {auc_base:.4f} | best-of-grid "
              f"{best['auroc']:.4f} ({best['delta_pp']:+.2f}pp) at min_frac="
              f"{best['min_frac']} exclude_frac={best['exclude_frac']} "
              f"scale_ratio={best['scale_ratio']}", flush=True)
    return rows


def _spearman(res_v, auc_v):
    ok = np.isfinite(res_v) & np.isfinite(auc_v)
    if ok.sum() < 8 or np.std(res_v[ok]) < 1e-12 or np.std(auc_v[ok]) < 1e-12:
        return float("nan"), float("nan"), int(ok.sum())
    rho, p = spearmanr(res_v[ok], auc_v[ok])
    return float(rho), float(p), int(ok.sum())


def _partial_spearman(x, y, z):
    """Spearman(x, y) with z partialled out — rank-transform, then correlate the
    residuals after regressing each on rank(z).

    Needed because Eq. 20's residual is `||rho - v1 v1' rho|| / ||rho||` on a block whose
    DIMENSION is n_keep, and `min_frac`/`exclude_frac` move n_keep directly. Measured on
    r4's grid: Spearman(RES, n_keep) = +0.74 and Spearman(n_keep, AUROC) = +0.37, whose
    product is the whole of the +0.26 raw correlation. Controlling for scale_ratio alone
    (the first fix) just swapped one nuisance variable for another — the audit caught it.
    """
    from scipy.stats import rankdata, pearsonr
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 8:
        return float("nan")
    rx, ry, rz = (rankdata(v[ok]).astype(float) for v in (x, y, z))
    if rz.std() < 1e-12 or rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    B = np.column_stack([rz, np.ones_like(rz)])

    def resid(a):
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]

    ex, ey = resid(rx), resid(ry)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(pearsonr(ex, ey)[0])


def run_r5(r4_rows):
    """Does Eq. 20's residual order MASKS by AUROC? Eq. 20 MINIMISES the residual, so
    'orders correctly' means a NEGATIVE Spearman.

    PRIMARY is the WITHIN-scale_ratio correlation, one slice per (cell, scale_ratio).
    Correlating across the whole 125-point grid confounds the question: scale_ratio
    moves var_y, which moves both the residual's scale and the macro, and it injects a
    spurious negative component. Measured, the confound flips the sign of the headline
    (pooled -0.04 across the full grid vs +0.26 within slices). This is the Step-204 §A
    failure mode — a criterion that looks informative because it tracks a nuisance
    variable — so the confounded version is reported alongside, never alone.
    """
    rows = []
    by_cell = {}
    for r in r4_rows:
        by_cell.setdefault(r["cell"], []).append(r)
    for ck, rs in by_cell.items():
        rho_all, p_all, n_all = _spearman(
            np.array([r["proj_residual"] for r in rs], float),
            np.array([r["auroc"] for r in rs], float))
        slices = []
        for sr in sorted({r["scale_ratio"] for r in rs}):
            sub = [r for r in rs if r["scale_ratio"] == sr]
            rho, p, n = _spearman(
                np.array([r["proj_residual"] for r in sub], float),
                np.array([r["auroc"] for r in sub], float))
            if np.isfinite(rho):
                slices.append(rho)
        rho_w = float(np.mean(slices)) if slices else float("nan")
        # PRIMARY: partial out n_keep as well as scale_ratio.
        rho_p = _partial_spearman(
            np.array([r["proj_residual"] for r in rs], float),
            np.array([r["auroc"] for r in rs], float),
            np.array([r["n_keep"] for r in rs], float))
        rho_rk = _partial_spearman(
            np.array([r["proj_residual"] for r in rs], float),
            np.array([r["n_keep"] for r in rs], float),
            np.array([r["auroc"] for r in rs], float))
        rows.append({
            "cell": ck, "n_grid": n_all,
            "partial_spearman_given_n_keep": rho_p,
            "spearman_within_scale_ratio": rho_w,
            "n_slices": len(slices),
            "n_slices_ordering_correctly": int(sum(1 for s in slices if s < 0)),
            "spearman_full_grid_CONFOUNDED": rho_all, "p_full_grid": p_all,
            "partial_res_vs_nkeep_given_auroc": rho_rk,
            "orders_correctly": bool(np.isfinite(rho_p) and rho_p < 0),
        })
        print(f"  {S.plain_cell(ck)[:34]:34s} PARTIAL Spearman(RES, AUROC | n_keep) = "
              f"{rho_p:+.3f}   [within-scale_ratio {rho_w:+.3f}; raw grid {rho_all:+.3f}]"
              f"  -> {'negative' if rows[-1]['orders_correctly'] else 'not negative'}",
              flush=True)
    return rows


# ---------------------------------------------------------------------------
# R7b — parameter-free var_y from the fixed point g2 = w' C w
# ---------------------------------------------------------------------------
def run_r7b(cells, qmax=2.0, n_grid=2001):
    """Solve h(q) = w(q)' C_k w(q) - q = 0 on the post-exclusion block with the mask
    HELD FIXED at the deployed one. Holding it fixed is deliberate: the mask itself
    moves with q (upcr.py:287-293), which would make h piecewise-discontinuous and
    the root meaningless."""
    rows = []
    for ck, cell in cells.items():
        V = cell["V"]
        n, m = V.shape
        auc_base, res = baseline_row(cell)
        if res is None or res.used_simple_average:
            rows.append({"cell": ck, "n_roots_in_0_qmax": -1, "root": float("nan"),
                         "root_in_deployed_box": None, "var_y_deployed": float("nan"),
                         "auroc_deployed": auc_base, "auroc_at_root": float("nan"),
                         "delta_pp": float("nan"),
                         "note": "simple-average fallback"})
            continue
        keep = res.keep
        n_keep = int(keep.sum())
        F_k = V[:, keep].T
        C_k = (F_k @ F_k.T) / n
        k2 = min(res.n_components_used, n_keep)
        ev, evec = eigh(C_k, subset_by_index=[n_keep - k2, n_keep - 1])
        evals, evecs = ev[::-1], evec[:, ::-1]

        # rho_k(q) = rho_k(0) + q/2, and the deployed rho_k already carries g2_hat
        rho0 = res.rho_hat[keep] - 0.5 * res.g2_hat

        def w_of(q):
            rho = rho0 + 0.5 * q
            w = np.zeros(n_keep)
            for c in range(k2):
                vc = evecs[:, c]
                w += (vc @ rho) / (evals[c] + 1e-12) * vc
            return w

        qs = np.linspace(0.0, qmax, n_grid)
        h = np.array([float(w_of(q) @ C_k @ w_of(q)) - q for q in qs])
        sign_changes = np.where(np.sign(h[:-1]) * np.sign(h[1:]) < 0)[0]
        roots = []
        for i in sign_changes:                       # linear interpolation is enough
            q0, q1, h0, h1 = qs[i], qs[i + 1], h[i], h[i + 1]
            roots.append(float(q0 - h0 * (q1 - q0) / (h1 - h0)))
        root = roots[0] if roots else float("nan")

        # Score the root with the SAME held-fixed mask the root was solved under.
        # Re-fitting through `fit_cols(var_y=root)` would (a) only move the g2 GRID
        # CEILING rather than impose g2 = root, and (b) let the mask move, so it would
        # not measure the fixed point at all. h(q) is an upward parabola with h(0) >= 0,
        # so there are generally TWO roots; both are reported.
        auc_root = float("nan")
        if np.isfinite(root) and root > 0:
            auc_root = S.auroc_from_score(cell, w_of(root) @ F_k)
        rows.append({
            "cell": ck, "n_roots_in_0_qmax": len(roots),
            "root": root, "all_roots": "|".join(f"{r:.4f}" for r in roots),
            "root_in_deployed_box": bool(np.isfinite(root) and 0 < root <= res.var_y),
            "var_y_deployed": float(res.var_y),
            "var_y_oracle": float(np.mean(cell["labels"]) * (1 - np.mean(cell["labels"]))),
            "auroc_deployed": auc_base, "auroc_at_root": auc_root,
            "delta_pp": (auc_root - auc_base) * 100.0 if np.isfinite(auc_root) else float("nan"),
            "note": "",
        })
        print(f"  {S.plain_cell(ck)[:34]:34s} roots={len(roots)} first={root:.4f} "
              f"(box {res.var_y:.4f}) AUROC {auc_base:.4f}->{auc_root:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
def summarise(name, rows, base_key, arm_key):
    a = np.array([r[base_key] for r in rows], float)
    b = np.array([r[arm_key] for r in rows], float)
    s = S.wl_wilcoxon(a, b)
    print(f"\n  {name}: {S.fmt_wl(s)}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all",
                    help="comma-separated subset of r7a,r1,r2,r3,r4,r5,r7b or 'all'")
    args = ap.parse_args()
    want = ({"r7a", "r1", "r2", "r3", "r4", "r5", "r7b"}
            if args.runs == "all" else set(args.runs.split(",")))
    if "r5" in want:
        want.add("r4")                              # r5 reads r4's grid

    out = S.outdir("10_channel_ceilings")
    cells = S.load()
    S.validity_check(cells)
    dcells = {k: derive_cell(c) for k, c in cells.items()}
    macro_arm = derived_arm_gate(dcells)

    summary = {"n_cells": len(cells), "runs": sorted(want),
               "arm": "U-PCR + sign(rho) (exp06 rho+anchor), the BENCHMARK_STANDING arm",
               "arm_macro": macro_arm, "fit": {
                   k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
                   for k, v in FIT.items()}}

    if "r7a" in want:
        print("\n=== R7a — ORACLE var_y = p(1-p) ===")
        rows = run_r7a(dcells)
        S.save_csv(os.path.join(out, "r7a_oracle_var_y.csv"), rows)
        summary["r7a"] = {
            "NOTE": ("var_y=p(1-p) against a UNIT diagonal is scale_ratio in "
                     "(0, 0.25], a one-sided perturbation bounded above by the "
                     "deployed value. It is NOT an oracle. See r4's scale_ratio slice."),
            "wl_pq_vs_deployed": summarise(
                "R7a var_y = p(1-p) vs deployed", rows,
                "auroc_deployed", "auroc_var_y_pq"),
            "wl_a1_vs_deployed": summarise(
                "R7a A1-faithful scale_ratio = 1.0 vs deployed", rows,
                "auroc_deployed", "auroc_a1_faithful_ratio1"),
            "macro_deployed": float(np.nanmean([r["auroc_deployed"] for r in rows])),
            "macro_var_y_pq": float(np.nanmean([r["auroc_var_y_pq"] for r in rows])),
            "macro_a1_ratio1": float(np.nanmean(
                [r["auroc_a1_faithful_ratio1"] for r in rows])),
            "p_min": float(np.min([r["p"] for r in rows])),
            "p_median": float(np.median([r["p"] for r in rows])),
            "box_ratio_max": float(np.max([r["box_ratio_deployed_over_oracle"]
                                           for r in rows])),
            "n_g2_at_ceiling_deployed": int(sum(1 for r in rows
                                                if r["g2_at_ceiling_deployed"])),
            "n_g2_at_ceiling_oracle": int(sum(1 for r in rows
                                              if r["g2_at_ceiling_oracle"])),
            "mean_keep_deployed": float(np.mean([r["n_keep_deployed"] for r in rows])),
            "mean_keep_oracle": float(np.mean([r["n_keep_oracle"] for r in rows])),
        }

    if "r1" in want:
        print("\n=== R1 — what the exclusion mask throws away ===")
        rows, view_rows = run_r1(dcells)
        S.save_csv(os.path.join(out, "r1_mask_discards.csv"), rows)
        S.save_csv(os.path.join(out, "r1_dropped_views.csv"), view_rows)
        summary["r1"] = {
            "total_dropped": int(sum(r["n_dropped"] for r in rows)),
            "total_views": int(sum(r["m"] for r in rows)),
            "mean_oracle_auroc_dropped": float(np.nanmean(
                [r["mean_oracle_auroc_dropped"] for r in rows])),
            "mean_oracle_auroc_kept": float(np.nanmean(
                [r["mean_oracle_auroc_kept"] for r in rows])),
            "n_dropped_beating_best_kept": int(sum(
                r["n_dropped_beating_best_kept"] for r in rows)),
        }

    if "r2" in want:
        print("\n=== R2 — ORACLE MASK ceiling (channel A) ===")
        rows = run_r2(cells, dcells)
        S.save_csv(os.path.join(out, "r2_oracle_mask.csv"), rows,
                   fieldnames=_all_fields(rows))
        summary["r2"] = {
            "wl_insample_vs_start": summarise(
                "R2 in-sample greedy vs its own start (pure channel A)", rows,
                "auroc_greedy_start", "auroc_oracle_mask_insample"),
            "wl_splithalf_vs_deployed": summarise(
                "R2 split-half oracle vs deployed", rows,
                "auroc_splithalf_deployed", "auroc_splithalf_oracle"),
            "wl_splithalf_vs_NULL": summarise(
                "R2 split-half oracle vs PERMUTED-LABEL NULL  <-- the one that matters",
                rows, "auroc_splithalf_NULL", "auroc_splithalf_oracle"),
            "wl_null_vs_deployed": summarise(
                "R2 null selection vs deployed (how much the search finds from noise)",
                rows, "auroc_splithalf_deployed", "auroc_splithalf_NULL"),
            "macro_deployed": float(np.nanmean([r["auroc_deployed"] for r in rows])),
            "macro_oracle_insample": float(np.nanmean(
                [r["auroc_oracle_mask_insample"] for r in rows])),
            "macro_splithalf_deployed": float(np.nanmean(
                [r["auroc_splithalf_deployed"] for r in rows])),
            "macro_splithalf_oracle": float(np.nanmean(
                [r["auroc_splithalf_oracle"] for r in rows])),
            "macro_splithalf_NULL": float(np.nanmean(
                [r["auroc_splithalf_NULL"] for r in rows])),
            "n_cells_greedy_unconverged_insample": int(sum(
                1 for r in rows if not r["greedy_converged_insample"])),
            "n_split_greedies_unconverged": int(sum(
                r["n_splits_greedy_unconverged"] for r in rows)),
            "greedy_max_steps": GREEDY_MAX_STEPS,
        }

    if "r3" in want:
        print("\n=== R3 — ORACLE MIXING ceiling (channel B) ===")
        rows = run_r3(dcells)
        S.save_csv(os.path.join(out, "r3_oracle_mixing.csv"), rows,
                   fieldnames=_all_fields(rows))
        summary["r3"] = {
            "wl": summarise("R3 oracle v1/v2 mixing vs deployed", rows,
                            "auroc_deployed", "auroc_oracle_mix"),
            "macro_deployed": float(np.nanmean([r["auroc_deployed"] for r in rows])),
            "macro_oracle_mix": float(np.nanmean([r["auroc_oracle_mix"] for r in rows])),
            "n_two_component": int(sum(1 for r in rows
                                       if r.get("n_components_used") == 2)),
            "n_simple_average": int(sum(1 for r in rows
                                        if r.get("used_simple_average"))),
        }

    if "r4" in want:
        print("\n=== R4 — label-free constant sweep ===")
        r4_rows = run_r4(dcells)
        S.save_csv(os.path.join(out, "r4_constant_sweep.csv"), r4_rows)
        # best single GLOBAL setting (one setting for all cells — the only honest
        # label-free read of this grid; per-cell best is an oracle in disguise)
        combos = {}
        for r in r4_rows:
            combos.setdefault((r["min_frac"], r["exclude_frac"], r["scale_ratio"]),
                              []).append(r["auroc"])
        macro = {k: float(np.nanmean(v)) for k, v in combos.items()}
        best_k = max(macro, key=macro.get)
        dep_k = (0.05, 3.0, 0.25)
        summary["r4"] = {
            "deployed_macro": macro.get(dep_k, float("nan")),
            "best_global_setting": {"min_frac": best_k[0], "exclude_frac": best_k[1],
                                    "scale_ratio": best_k[2]},
            "best_global_macro": macro[best_k],
            "best_global_delta_pp": (macro[best_k] - macro.get(dep_k, np.nan)) * 100.0,
            "per_cell_oracle_macro": float(np.nanmean([
                max((r["auroc"] for r in r4_rows
                     if r["cell"] == ck and np.isfinite(r["auroc"])),
                    default=float("nan"))
                for ck in cells])),
            "n_settings": len(macro),
        }
        print(f"\n  deployed (0.05, 3.0, 0.25) macro = {summary['r4']['deployed_macro']:.4f}")
        print(f"  best GLOBAL setting {best_k} macro = {macro[best_k]:.4f} "
              f"({summary['r4']['best_global_delta_pp']:+.2f}pp)")
        print(f"  per-cell oracle over the grid = {summary['r4']['per_cell_oracle_macro']:.4f}")

        if "r5" in want:
            print("\n=== R5 — does Eq. 20's RES order the grid by AUROC? ===")
            r5_rows = run_r5(r4_rows)
            S.save_csv(os.path.join(out, "r5_res_orders_masks.csv"), r5_rows)
            def _col(k):
                return np.array([r[k] for r in r5_rows], float)

            sp_p = _col("partial_spearman_given_n_keep")
            summary["r5"] = {
                "primary": "partial_spearman_given_n_keep",
                "mean_partial_spearman": float(np.nanmean(sp_p)),
                "median_partial_spearman": float(np.nanmedian(sp_p)),
                "n_cells_negative_partial": int(np.nansum(sp_p < 0)),
                "n_cells_finite_partial": int(np.isfinite(sp_p).sum()),
                "mean_spearman_within_scale_ratio": float(
                    np.nanmean(_col("spearman_within_scale_ratio"))),
                "mean_spearman_full_grid_CONFOUNDED": float(
                    np.nanmean(_col("spearman_full_grid_CONFOUNDED"))),
                "mean_partial_res_vs_nkeep": float(
                    np.nanmean(_col("partial_res_vs_nkeep_given_auroc"))),
                "n_cells": len(r5_rows),
                "reading": (
                    "Eq. 20 MINIMISES RES, so ordering masks correctly means NEGATIVE. "
                    "The raw and within-scale_ratio numbers are both nuisance-driven: "
                    "RES is computed on a block whose dimension IS n_keep, and n_keep is "
                    "what min_frac/exclude_frac move. Only the partial is interpretable."),
            }

    if "r7b" in want:
        print("\n=== R7b — parameter-free var_y from the fixed point ===")
        rows = run_r7b(dcells)
        S.save_csv(os.path.join(out, "r7b_fixed_point.csv"), rows,
                   fieldnames=_all_fields(rows))
        summary["r7b"] = {
            "n_cells_with_root": int(sum(1 for r in rows if np.isfinite(r["root"]))),
            "n_roots_in_box": int(sum(1 for r in rows if r["root_in_deployed_box"])),
            "median_root": float(np.nanmedian([r["root"] for r in rows])),
            "wl": summarise("R7b fixed-point var_y vs deployed", rows,
                            "auroc_deployed", "auroc_at_root"),
        }

    # MERGE rather than overwrite: the runs are invoked separately (r7a is meant to
    # be run first and alone), and a plain write would silently drop the earlier
    # run's block. Resume-safe, same reason the bench CSVs are.
    spath = os.path.join(out, "summary.json")
    if os.path.exists(spath):
        import json
        try:
            with open(spath, encoding="utf-8") as f:
                prev = json.load(f)
            prev.update(summary)
            prev["runs"] = sorted(set(prev.get("runs", [])) | set(summary["runs"]))
            summary = prev
        except Exception:
            pass
    S.save_json(spath, summary)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
